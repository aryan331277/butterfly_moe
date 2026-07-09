# ButterflyMoE: Compression-Scalable Ternary Experts via Structured Butterfly Orbits

**Paper:** [arXiv:2601.13563](https://arxiv.org/abs/2601.13563)

## Overview

ButterflyMoE rethinks how Mixture-of-Experts (MoE) layers store their parameters. Instead of treating each expert as an independent weight matrix, it represents every expert as a lightweight orthogonal rotation of a single **shared, ternary-quantized substrate**. Expert diversity comes from *how the substrate is viewed*, not from redundant storage — collapsing per-expert memory from `O(d²)` to `O(d log d)`.

```
Wi = R_out(i) · W_base · R_in(i)ᵀ
```

Where `W_base ∈ {−1, 0, +1}^(d×d)` is shared across all experts, and the rotations `R_in`, `R_out` are parameterized as Butterfly matrices requiring only `O(d log d)` learnable angles.

## Why This Matters

Standard MoE memory scales linearly with expert count — `O(N·d²)` — making large expert pools infeasible on edge and resource-constrained hardware. Even aggressive quantization methods (QMoE, MoQE) only reduce the constant factor; they don't break the linear scaling law. ButterflyMoE directly attacks the structural assumption that each expert needs its own parameter set.

## Core Strengths

### 1. Extreme, Scaling Memory Compression
- **80× compression** at 8 experts, **150× at 256 experts** — and the compression ratio *improves* as more experts are added, the opposite of every existing MoE compression method.
- At 256 experts, ButterflyMoE needs just **6.82 MB** versus **1,024 MB** for standard MoE.
- At 64 experts: **1.9 MB** vs. 256 MB for standard MoE — an order of magnitude better than QMoE, MoQE, PuzzleMoE, or Monarch-based compressors (Table 5).

### 2. Compression That Doesn't Sacrifice Utility
- At 80× compression, ButterflyMoE **outperforms an equal-memory dense baseline** (81.19 vs. 135.46 PPL), demonstrating that orbital parameterization extracts more representational value per byte rather than simply shrinking the network.
- A dedicated bitwidth ablation shows **1.58-bit ternary quantization beats both 1-bit and 2-bit** alternatives (180.67 PPL vs. 205.59 and 189.61) — the sweet spot between expressivity and gradient stability.

### 3. Expert Count Decoupled from Memory Cost
- Scaling from 8 to 16 experts costs standard MoE an extra **32 MB**; it costs ButterflyMoE only **0.21 MB**.
- Validation perplexity stays essentially flat across expert counts (128.32 → 128.38), showing experts behave as *free parameters* once the substrate is learned — enabling massive expert pools purely for routing granularity, without a memory penalty.

### 4. Solves Its Own Quantization Problem, Not Just Compresses
- Jointly training input rotations with the shared substrate **suppresses activation outliers organically** — no post-hoc clipping or calibration needed.
- Quantization MSE drops **97.2%** (0.513 → 0.0143) purely from learned rotation alignment with the ternary grid.
- Outperforms **SpinQuant**-style global rotation (143.58 vs. 132.54 PPL) while using **3.4× less memory**, and outperforms **Monarch-matrix** experts (143.58 vs. 152.72 PPL) with **1.72× less memory** — because Butterfly rotations guarantee orthogonality by construction, which block-diagonal Monarch matrices cannot.

### 5. Genuine Compute-Bound Efficiency on Edge Hardware
- A roofline analysis on ARM Cortex-A72 (representative of Raspberry Pi–class edge devices) shows ButterflyMoE achieves an arithmetic intensity of **38.1 FLOP/byte (cached)** and **8.5 FLOP/byte (cold cache)** — both above the 8.0 FLOP/byte ridge point, making it **compute-bound**.
- Standard MoE, by contrast, sits at just **1.0 FLOP/byte** — deeply memory-bound, capped at 12.5% of peak compute.
- Ternary matrix multiplication uses **only additions, no multiplications** — roughly 10× lower energy per operation.

### 6. Production-Realistic Throughput, Not Just Theoretical Savings
- A naive PyTorch implementation is 9× slower than standard MoE due to kernel-launch overhead — the paper doesn't hide this.
- A fused Triton kernel (single-launch butterfly layers held in registers, batched GEMM across active experts) recovers **8.41× speedup**, landing within **3.4% of standard MoE's throughput** (29,476 vs. 30,509 tok/s) while using **14× fewer parameters** and **4× fewer FLOPs**.

### 7. Preserved Expert Specialization Despite Shared Storage
- Pairwise cosine-similarity diversity scoring shows ButterflyMoE experts reach a diversity score of **0.87** vs. **0.912** for standard MoE — only a 5% gap despite a ~150× reduction in stored parameters, confirming experts remain functionally distinct rather than collapsing to duplicates.

### 8. Generalizes Beyond Power-of-Two Dimensions
- A zero-padding scheme extends Butterfly factorization to arbitrary model dimensions (e.g., `d = 768`), with orthogonality preserved in the active subspace and no architectural modification required — removing a common practical limitation of Butterfly/Monarch-style structured matrices.

## Summary Table

| Experts | Standard MoE (MB) | ButterflyMoE (MB) | Compression |
|---|---|---|---|
| 8   | 32.00   | 0.40 | 80.0×  |
| 16  | 64.00   | 0.61 | 104.65× |
| 64  | 256.00  | 1.85 | 138.10× |
| 256 | 1024.00 | 6.82 | 150.09× |

## Bottom Line

ButterflyMoE demonstrates that the `O(N)` memory scaling long assumed necessary for MoE architectures is an artifact of implementation, not a structural requirement. By reframing experts as orbits of a shared quantized substrate, it achieves compression ratios that *grow* with scale, retains competitive accuracy against dense and standard-MoE baselines, actively improves quantization stability through training rather than fighting it post-hoc, and validates its efficiency claims with both a hardware-grounded roofline model and real fused-kernel throughput numbers — turning MoE from memory-bound to compute-bound on edge silicon.

## Limitations (as noted by the authors)

- Experiments are conducted at small scale (4-layer, d=512, single Tesla T4 GPU) as a proof of concept; production-scale validation (64–128 experts) is future work.
- On-device energy/latency profiling is based on a theoretical roofline model, not physical edge-hardware measurement.
- Code release is planned upon acceptance, not yet available.
