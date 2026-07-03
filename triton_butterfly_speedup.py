import math, time, torch, torch.nn as nn, torch.nn.functional as F
import triton, triton.language as tl

D_MODEL     = 512
D_FF        = 2048
NUM_EXPERTS = 16
TOP_K       = 4
NUM_RUNS    = 500
WARMUP      = 100

@triton.jit
def fused_butterfly_kernel(
        X_ptr,                  # (N, D) input
        Y_ptr,                  # (N, D) output
        ANGLES_ptr,             # (num_layers * D//2,)  precomputed flat angle table
        N,
        D          : tl.constexpr,
        NUM_LAYERS : tl.constexpr,
    ):
