import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from collections import defaultdict
import json
import os

MAX_STEPS=500

class ButterflyQuantFFN(nn.Module):
    def __init__(self, d_model, d_ff, num_butterfly_layers=None):
        super().__init__()
        self.rotation = ButterflyRotation(d_model, num_butterfly_layers)
        self.w1 = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
        self.w2 = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)

    def forward(self, x):
        x_rot = self.rotation(x)
        w1_q = BitNetQuantize.apply(self.w1)
        w2_q = BitNetQuantize.apply(self.w2)
        return F.linear(F.gelu(F.linear(x_rot, w1_q)), w2_q)


class ButterflyQuant_LM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_butterfly_layers=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)

        class BQBlock(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.attn  = nn.MultiheadAttention(d_model, max(1, d_model // 64), batch_first=True)
                self.ffn   = ButterflyQuantFFN(d_model, d_model * 4, num_butterfly_layers)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)

            def forward(self, x, mask=None, training=True):
                a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
                x = self.norm1(x + a)
                x = self.norm2(x + self.ffn(x))
                return x, 0

        self.blocks = nn.ModuleList([BQBlock(d_model) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out  = nn.Linear(d_model, vocab_size)

    def forward(self, x, training=True):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)
        T = x.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        for block in self.blocks:
            x, _ = block(x, mask=causal_mask, training=training)
        return self.out(self.norm(x)), 0


class ButterflyRotation(nn.Module):
    def __init__(self, dim: int, num_layers: int = None):
        super().__init__()
        self.dim = dim
        self.padded_dim = 2 ** math.ceil(math.log2(dim)) if dim & (dim - 1) != 0 else dim
        self.num_layers = num_layers or max(int(math.log2(self.padded_dim)), 2)
        self.angles = nn.ParameterList([
            nn.Parameter(torch.randn(self.padded_dim // 2) * 0.02)
            for _ in range(self.num_layers)
        ])
        self.needs_pad = (dim != self.padded_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch, seq_len, dim = x.shape

        if self.needs_pad:
            pad_size = self.padded_dim - dim
            x = F.pad(x, (0, pad_size))

        for layer_idx in range(self.num_layers):
            stride = 2 ** (layer_idx + 1)
            x = self._apply_butterfly_layer(x, self.angles[layer_idx], stride)

        if self.needs_pad:
            x = x[..., :dim]

        if len(original_shape) == 2:
            x = x.squeeze(1)
        return x

    def _apply_butterfly_layer(self, x: torch.Tensor, angles: torch.Tensor, stride: int) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        x_reshaped = x.reshape(batch, seq_len, dim // stride, stride)
        half_stride = stride // 2
        num_groups = dim // stride
        cos_angles = torch.cos(angles[:dim // 2]).reshape(num_groups, half_stride)
        sin_angles = torch.sin(angles[:dim // 2]).reshape(num_groups, half_stride)
        x1 = x_reshaped[..., :half_stride]
        x2 = x_reshaped[..., half_stride:]
        y1 = cos_angles * x1 - sin_angles * x2
        y2 = sin_angles * x1 + cos_angles * x2
        y = torch.cat([y1, y2], dim=-1)
        return y.reshape(batch, seq_len, dim)


class BitNetQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        gamma = weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        weight_scaled = weight / gamma
        weight_quant = weight_scaled.round().clamp(-1, 1)
        return weight_quant * gamma

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class HO_Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_butterfly_layers: int = None):
        super().__init__()
        self.phi = ButterflyRotation(d_model, num_butterfly_layers)
        self.theta = ButterflyRotation(d_ff, num_butterfly_layers)

    def forward(self, x: torch.Tensor, w_base_quant: torch.Tensor) -> torch.Tensor:
        x_rot = self.phi(x)
        y_base = F.linear(x_rot, w_base_quant)
        y = self.theta(y_base)
        return y


class HOMoE_Layer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2,
                 num_butterfly_layers: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.w_base = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
        self.experts = nn.ModuleList([
            HO_Expert(d_model, d_ff, num_butterfly_layers) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise_std = 0.1

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, dict]:
        batch, seq_len, d_model = x.shape
        w_base_quant = BitNetQuantize.apply(self.w_base)
        gate_logits = self.gate(x)
        if training:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.noise_std
        routing_weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            expert_counts[i] = (selected_experts == i).float().sum()
        expert_counts = expert_counts / expert_counts.sum().clamp(min=1)
        load_balance_loss = (expert_counts - 1.0 / self.num_experts).pow(2).mean()
        output = torch.zeros(batch, seq_len, self.d_ff, device=x.device, dtype=x.dtype)
        for i in range(self.num_experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            if not expert_mask.any():
                continue
            batch_idx, seq_idx = torch.where(expert_mask)
            x_expert = x[batch_idx, seq_idx]
            y_expert = self.experts[i](x_expert.unsqueeze(1), w_base_quant).squeeze(1)
            expert_positions = (selected_experts[batch_idx, seq_idx] == i).nonzero(as_tuple=True)[1]
            weights = routing_weights[batch_idx, seq_idx, expert_positions].unsqueeze(-1)
            output[batch_idx, seq_idx] += weights * y_expert
        return output, {'load_balance': load_balance_loss, 'expert_counts': expert_counts}


class HOMoE_TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=512, num_experts=8, top_k=2, dropout=0.1, num_butterfly_layers=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.moe = HOMoE_Layer(d_model, d_ff, num_experts, top_k, num_butterfly_layers)
        self.proj = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, training=True):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        moe_out, aux_loss = self.moe(x, training=training)
        moe_out = self.proj(moe_out)
        x = x + self.dropout(moe_out)
        x = self.norm2(x)
        return x, aux_loss


class StandardMoE_Layer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, dict]:
        batch, seq_len, d_model = x.shape
        gate_logits = self.gate(x)
        routing_weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            expert_counts[i] = (selected_experts == i).float().sum()
        expert_counts = expert_counts / expert_counts.sum().clamp(min=1)
        load_balance_loss = (expert_counts - 1.0 / self.num_experts).pow(2).mean()
        output = torch.zeros(batch, seq_len, d_model, device=x.device, dtype=x.dtype)
        for i in range(self.num_experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            if not expert_mask.any():
                continue
            batch_idx, seq_idx = torch.where(expert_mask)
            x_expert = x[batch_idx, seq_idx]
            y_expert = self.experts[i](x_expert)
            expert_positions = (selected_experts[batch_idx, seq_idx] == i).nonzero(as_tuple=True)[1]
            weights = routing_weights[batch_idx, seq_idx, expert_positions].unsqueeze(-1)
            output[batch_idx, seq_idx] += weights * y_expert
        return output, {'load_balance': load_balance_loss, 'expert_counts': expert_counts}


class StandardMoE_TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=512, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.moe = StandardMoE_Layer(d_model, d_ff, num_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, training=True):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        moe_out, aux_loss = self.moe(x, training=training)
        x = x + self.dropout(moe_out)
        x = self.norm2(x)
        return x, aux_loss


class ButterflyMoE_LM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_experts, top_k, num_butterfly_layers=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([
            HOMoE_TransformerBlock(d_model, num_heads=max(1, d_model // 64),
                                   d_ff=d_model * 4, num_experts=num_experts,
                                   top_k=top_k, num_butterfly_layers=num_butterfly_layers)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, training=True):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)
        total_aux = 0
        T = x.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)

        for block in self.blocks:
            x, aux = block(x, mask=causal_mask, training=training)
            total_aux += aux['load_balance']
        x = self.norm(x)
        return self.out(x), total_aux


class StandardMoE_LM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_experts, top_k):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([
            StandardMoE_TransformerBlock(d_model, num_heads=max(1, d_model // 64),
                                         d_ff=d_model * 4, num_experts=num_experts, top_k=top_k)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, training=True):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)
        total_aux = 0
        T = x.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        for block in self.blocks:
            x, aux = block(x, mask=causal_mask, training=training)
            total_aux += aux['load_balance']
        x = self.norm(x)
        return self.out(x), total_aux


class DenseTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,mask=None, training=True):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x, 0


class Dense_LM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([
            DenseTransformerBlock(d_model, max(1, d_model // 64), d_model * 4)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, training=True):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)
        T = x.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        for block in self.blocks:
            x, _ = block(x, mask=causal_mask, training=training)
        x = self.norm(x)
        return self.out(x), 0


class WikiTextDataset(Dataset):
    def __init__(self, split='train', seq_len=128, dataset_name='wikitext-2-raw-v1'):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len = seq_len
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        self.vocab_size = tokenizer.vocab_size

        raw = load_dataset('wikitext', dataset_name, split=split)
        texts = [t for t in raw['text'] if t.strip()]

        all_ids = []
        for text in texts:
            all_ids.extend(tokenizer.encode(text, add_special_tokens=False))

        self.data = []
        for i in range(0, len(all_ids) - seq_len, seq_len):
            chunk = torch.tensor(all_ids[i:i + seq_len + 1], dtype=torch.long)
            self.data.append(chunk)

        print(f"  [{dataset_name} / {split}] {len(self.data)} sequences, vocab={self.vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


def train_epoch(model, dataloader, optimizer, scheduler, device, load_balance_weight=0.01, max_steps=None):
    model.train()
    total_loss = 0
    for i, (inputs, targets) in enumerate(dataloader):
        if max_steps is not None and i >= max_steps:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits, aux_loss = model(inputs, training=True)
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss = ce_loss + load_balance_weight * aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += ce_loss.item()
    steps = min(max_steps, len(dataloader)) if max_steps else len(dataloader)
    return total_loss / steps

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_tokens = 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs, training=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def train_model(model, train_loader, val_loader, device, num_epochs=20,
                lr=1e-3, load_balance_weight=0.01, model_name="model", max_steps=None):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    total_steps = num_epochs * len(train_loader)

    warmup_steps = len(train_loader)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {'train_loss': [], 'val_loss': [], 'val_ppl': []}
    best_ppl = float('inf')

    patience = 3
    no_improve = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 device, load_balance_weight, max_steps=max_steps)
        val_loss, val_ppl = evaluate(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            no_improve = 0
            torch.save(model.state_dict(), f'best_{model_name}.pt')
        else:
            no_improve += 1

        if (epoch + 1) % 1 == 0:
            print(f"  [{model_name}] Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val PPL: {val_ppl:.2f}")

        if no_improve >= patience:
            print(f"  [{model_name}] Early stop at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break

    model.load_state_dict(torch.load(f'best_{model_name}.pt', map_location=device))
    print(f"  [{model_name}] Best Val PPL: {best_ppl:.2f}")
    return history, best_ppl


def measure_throughput(model, batch_size, seq_len, vocab_size, device, num_iters=100):
    model.eval()
    dummy = torch.randint(0, min(vocab_size, 50257), (batch_size, seq_len), device=device)
    for _ in range(10):
        with torch.no_grad():
            model(dummy, training=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            model(dummy, training=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    tokens_per_sec = (batch_size * seq_len * num_iters) / elapsed
    latency_ms = elapsed / num_iters * 1000
    return tokens_per_sec, latency_ms


def calculate_memory_mb(d_model, d_ff, num_experts, method='butterfly'):
    if method == 'butterfly':
        substrate_bits = d_ff * d_model * 1.58
        substrate_mb = substrate_bits / (8 * 1024 * 1024)
        phi_params = (d_model / 2) * math.log2(max(d_model, 2))
        theta_params = (d_ff / 2) * math.log2(max(d_ff, 2))
        rotation_mb = num_experts * (phi_params + theta_params) * 2 / (1024 * 1024)
        return substrate_mb + rotation_mb
    elif method == 'standard':
        return num_experts * d_ff * d_model * 4 / (1024 * 1024)
    elif method == 'qmoe':
        return num_experts * d_ff * d_model * 4 / (1024 * 1024) / 15.0
    elif method == 'moqe':
        return num_experts * d_ff * d_model * 2 / (8 * 1024 * 1024)
