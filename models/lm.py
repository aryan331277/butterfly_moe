import torch
import torch.nn as nn

from .transformer import (
    ButterflyQuantFFN,
    HOMoE_TransformerBlock,
    StandardMoE_TransformerBlock,
    StandardMoEQuant_TransformerBlock,
    DenseTransformerBlock,
)

class ButterflyQuant_LM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_butterfly_layers=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
 
        class BQBlock(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.attn  = nn.MultiheadAttention(d_model, max(1, d_model // 64),
                                                    batch_first=True)
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


class ButterflyMoE_LM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_experts, top_k,
                 num_butterfly_layers=None):
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


class StandardMoEQuant_LM(nn.Module):
    """Same shape/training recipe as StandardMoE_LM, but every expert's
    weights are run through BitNetQuantize before use."""
    def __init__(self, vocab_size, d_model, num_layers, num_experts, top_k):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([
            StandardMoEQuant_TransformerBlock(d_model, num_heads=max(1, d_model // 64),
                                               d_ff=d_model * 4, num_experts=num_experts,
                                               top_k=top_k)
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
