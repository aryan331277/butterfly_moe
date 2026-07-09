import math
import time
import torch


def measure_throughput(model, batch_size, seq_len, vocab_size, device, num_iters=100):
    model.eval()
    dummy = torch.randint(0, min(vocab_size, 50257), (batch_size, seq_len), device=device)
    # Warmup
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
        # FP16 rotation angles per expert
        phi_params = (d_model / 2) * math.log2(max(d_model, 2))
        theta_params = (d_ff / 2) * math.log2(max(d_ff, 2))
        rotation_mb = num_experts * (phi_params + theta_params) * 2 / (1024 * 1024)
        return substrate_mb + rotation_mb
    elif method == 'standard':
        return num_experts * d_ff * d_model * 4 / (1024 * 1024)  # FP32
    elif method == 'standard_158bit':
        return num_experts * d_ff * d_model * 1.58 / (8 * 1024 * 1024)
    elif method == 'qmoe':  # ~sub-1-bit (reported 10-20x vs FP32)
        return num_experts * d_ff * d_model * 4 / (1024 * 1024) / 15.0
    elif method == 'moqe':  # 2-bit
        return num_experts * d_ff * d_model * 2 / (8 * 1024 * 1024)
