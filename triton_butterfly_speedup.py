import math, time, torch, torch.nn as nn, torch.nn.functional as F
import triton, triton.language as tl

D_MODEL     = 512
D_FF        = 2048
NUM_EXPERTS = 16
TOP_K       = 4
NUM_RUNS    = 500
WARMUP      = 100

if HAS_TRITON:
    @triton.jit
    def bfly_kernel(
        inp_ptr,
        out_ptr,
        angles_ptr,
        n,
        dim: tl.constexpr,      # must be power of 2
        num_layers: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, dim)

        v = tl.load(inp_ptr + row * dim + cols)

        for layer in tl.static_range(num_layers):
            stride = 1 << (layer + 1)
            half_stride = stride >> 1
            ng = dim // stride

            grp = cols // stride
            local = cols % stride
            first_half = local < half_stride

            # swap partner within the group
            pair_idx = tl.where(first_half, cols + half_stride, cols - half_stride)

            # flat angle table layout: layer L -> angles[L*dim//2 : (L+1)*dim//2]
            angle_idx = (layer * (dim // 2)
                    + grp * half_stride
                    + tl.where(first_half, local, local - half_stride))

            cos_v = tl.cos(tl.load(angles_ptr + angle_idx))
            sin_v = tl.sin(tl.load(angles_ptr + angle_idx))

            v_pair = tl.gather(v, pair_idx, 0)

            v = tl.where(
                first_half,
                cos_v * v - sin_v * v_pair,
                sin_v * v_pair + cos_v * v,
            )

        tl.store(out_ptr + row * dim + cols, v)


    def run_bfly(x_flat, angles_flat, dim, num_layers):
        # x_flat: (N, dim), angles_flat: (num_layers * dim//2,)
        n = x_flat.shape[0]
        out = torch.empty_like(x_flat)
        bfly_kernel[(n,)](
            x_flat, out, angles_flat,
            n, dim, num_layers,
        )
        return out
