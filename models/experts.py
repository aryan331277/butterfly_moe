from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .butterfly import ButterflyRotation, BitNetQuantize, BitNetQuantizeNBit


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


class HOMoE_Layer_NBit(HOMoE_Layer):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2,
                 num_butterfly_layers=None, n_bits=1.58):
        super().__init__(d_model, d_ff, num_experts, top_k, num_butterfly_layers)
        self.n_bits = n_bits

    def forward(self, x, training=True):
        batch, seq_len, d_model = x.shape
        if self.n_bits == 1.58:
            w_base_quant = BitNetQuantize.apply(self.w_base)
        else:
            w_base_quant = BitNetQuantizeNBit.apply(self.w_base, self.n_bits)

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


class StandardMoE_Layer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Each expert is a full independent FFN
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


class StandardMoEQuant_Layer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        # Raw fp32 parameters per expert; quantized on every forward (STE),
        # exactly like self.w_base in HOMoE_Layer.
        self.w1 = nn.ParameterList([
            nn.Parameter(torch.randn(d_ff, d_model) * 0.02) for _ in range(num_experts)
        ])
        self.w2 = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_ff) * 0.02) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise_std = 0.1
 
    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, dict]:
        batch, seq_len, d_model = x.shape
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
 
        output = torch.zeros(batch, seq_len, d_model, device=x.device, dtype=x.dtype)
        for i in range(self.num_experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            if not expert_mask.any():
                continue
            batch_idx, seq_idx = torch.where(expert_mask)
            x_expert = x[batch_idx, seq_idx]
 
            w1_q = BitNetQuantize.apply(self.w1[i])
            w2_q = BitNetQuantize.apply(self.w2[i])
            y_expert = F.linear(F.gelu(F.linear(x_expert, w1_q)), w2_q)
 
            expert_positions = (selected_experts[batch_idx, seq_idx] == i).nonzero(as_tuple=True)[1]
            weights = routing_weights[batch_idx, seq_idx, expert_positions].unsqueeze(-1)
            output[batch_idx, seq_idx] += weights * y_expert
            
        return output, {'load_balance': load_balance_loss, 'expert_counts': expert_counts}
