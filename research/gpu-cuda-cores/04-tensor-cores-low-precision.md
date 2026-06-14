# Ada 4th-Gen Tensor Cores and Low Precision for Financial Compute

**Research date:** 2026-06-14
**Target HW:** NVIDIA RTX 3500 Ada Generation Laptop GPU (Ada Lovelace, AD104, `sm_89`), 12 GB GDDR6, 160 4th-gen Tensor Cores, CUDA 13.1.
**Scope:** What the Ada Tensor Cores actually deliver on *this* GPU, how to program them, the numerical-accuracy rules that matter for finance, and which quant workloads are GEMM-shaped enough to exploit them.

---

## 1. The single most important fact for this hardware

The RTX 3500 Ada is a *laptop/professional GeForce-class* Ada part. **On all GeForce/laptop Ada GPUs, FP16→FP32 and FP8→FP32 Tensor-Core accumulation runs at half rate.** This is a hardware product-segmentation throttle, not a software bug.

- NVIDIA's own forums confirm RTX 4090 (also AD-class GeForce) FP8 hits only ~330–340 TFLOPS vs the 660 TFLOPS whitepaper figure "due to throttling of FP8→FP32 operations similar to how FP16→FP32 operations are half-rate on GeForce cards" ([NVIDIA Dev Forums — Ada GeForce RTX 4090 FP8 cuBLASLt performance](https://forums.developer.nvidia.com/t/ada-geforce-rtx-4090-fp8-cublaslt-performance/250737)). cuBLASLt FP8 requires `CUBLAS_COMPUTE_32F`, so the throttle is unavoidable for accurate accumulation ([NVIDIA Dev Forums — FP8/FP16 accumulation on Ada RTX 4090](https://forums.developer.nvidia.com/t/fp8-fp16-accumulation-on-ada-rtx-4090/294400)).
- The RTX 3500 Ada is rated at **127 INT8 TOPS** and 15.8–23 FP32 TFLOPS depending on the 60–140 W power cap ([NanoReview — RTX 3500 Laptop Ada specs](https://nanoreview.net/en/gpu/nvidia-rtx-3500-mobile-ada)). Because FP8-dense-with-FP32-accumulate is throttled to the same lane as INT8, **realistic FP8/FP16 GEMM throughput on this card is ~115–130 TFLOPS, not the ~250–360 TFLOPS implied by datacenter L40-class Ada parts** (L40: 362 dense / 724 sparse FP8 TFLOPS — [NVIDIA L40 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/support-guide/NVIDIA-L40-Datasheet-January-2023.pdf)).

**Takeaway:** budget for ~115–130 TFLOPS of *usable* low-precision Tensor-Core throughput on this laptop, and treat any quoted "1.4 petaFLOP" Ada FP8 number ([NVIDIA Ada GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)) as marketing for sparse, FP32-accumulate-unthrottled datacenter SKUs.

---

## 2. The precision formats Ada exposes

| Format | Bits (E/M) | Dynamic range | Use in finance | Notes |
|---|---|---|---|---|
| **FP8 E4M3** | 4 exp / 3 mant | ~±448, ~2 decimal digits | Quantized indicator weights, ML feature transforms | Higher precision, narrow range; standard for forward-pass activations/weights ([Spheron — FP8 Quantization](https://www.spheron.network/blog/fp8-quantization-inference-performance-hardware-explained/)) |
| **FP8 E5M2** | 5 exp / 2 mant | ~±57344, ~1.5 digits | Wide-range gradients, rarely for values you read back | Same range as FP16 but only 2 mantissa bits ([Micikevicius et al., *FP8 Formats for Deep Learning*](https://arxiv.org/pdf/2209.05433)) |
| **FP16** | 5 / 10 | ~±65504 | Monte-Carlo paths, covariance accumulation inputs | Overflows easily on un-normalized prices |
| **BF16** | 8 / 7 | ~±3.4e38 (FP32 range) | Preferred MC/GEMM input for finance | FP32 range, only 7 mantissa bits — drop-in safe vs overflow |
| **TF32** | 8 / 10 | FP32 range | "Free" speedup for FP32-shaped code | 19-bit internal; ~3 decimal digits, transparent in cuBLAS |

Ada 4th-gen Tensor Cores support FP8 (E4M3/E5M2), FP16, BF16, TF32, INT8, INT4 plus 2:4 structured sparsity ([Ada Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)). **For finance, BF16 is almost always the right input format:** it carries the full FP32 exponent range, so raw prices (e.g. 68,000.25 BTC) and variance accumulators never overflow — the failure mode that bites naive FP16 quant code.

---

## 3. Programming model: WMMA vs `mma.sync` PTX vs CUTLASS

Three abstraction levels, in increasing order of performance and effort:

1. **WMMA C++ API** (`nvcuda::wmma`, `mma.h`). Easiest. Fixed 16×16×16 fragment tiles. **Leaves significant performance on the table:** the `sync` suffix stalls until operands are register-resident, and a naive WMMA kernel without shared-memory staging gives *no* speedup — adding shared memory is ~5× ([Sun et al., *Benchmarking GPU Tensor Cores through CUTLASS*, Applied Sciences 2023](https://www.mdpi.com/2076-3417/13/24/13022)).
2. **`mma.sync` PTX** (`mma.sync.aligned.m16n8k16...`). The real workhorse. On Ada, `m16n8k16` is the largest Tensor-Core matmul shape, with a 32-cycle instruction latency; combined with `ldmatrix` and `cp.async` it eliminates bank conflicts that WMMA cannot ([Patterson, *Implementing a fast Tensor Core matmul on the Ada Architecture*](https://www.spatters.ca/mma-matmul)). MMA-level code reportedly delivers materially higher throughput than WMMA and removes bank conflicts entirely ([Sun et al. 2023](https://www.mdpi.com/2076-3417/13/24/13022)).
3. **CUTLASS / cuBLASLt.** Don't hand-roll. For Ada FP8 use CUTLASS example `58_ada_fp8_gemm` (`mma_traits_sm89.hpp`), which fuses per-tensor scaling + bias into the epilogue ([CUTLASS `58_ada_fp8_gemm`](https://github.com/NVIDIA/cutlass/blob/main/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu); [CUTLASS GEMM API docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)). For large square GEMM, CUTLASS with tuned tiling can beat cuBLAS at N≈16,384 ([Sun et al. 2023](https://www.mdpi.com/2076-3417/13/24/13022)).

**Guidance for kimsfinance (a Rust+CUDA core):** call cuBLASLt/CUTLASS GEMM through FFI for anything matmul-shaped; only drop to hand-written `mma.sync` PTX if profiling shows a fused custom kernel (e.g. indicator-matmul + epilogue activation) is bandwidth-bound and a library call forces extra global round-trips. WMMA is rarely worth it — it is the easy on-ramp, not the destination.

---

## 4. Numerical accuracy: the rules that actually matter for finance

### Rule 1 — Always accumulate in FP32 (never trust the "fast accumulate")
The multiply happens in low precision; the **sum must be FP32**. Two distinct hazards:

- **Library "fast accumulate" modes** (FP8 GEMM `CUBLAS_*_FAST_ACCUM`, CUTLASS fast-accum) skip the periodic promotion of partial sums to higher precision — faster, but lossy. *Disable fast-accumulate for any pricing or risk number you report.* ([CUTLASS — FP8 fast accumulation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)).
- **Even "FP32 accumulate" Tensor Cores may carry fewer than 23 mantissa bits internally.** Outputs *look* like FP32 but lose low-order bits, and large-K reductions compound this ([PyTorch — *Some Matrix Multiplication Engines Are Not As Accurate As We Thought*](https://pytorch.org/blog/some-matrix-multiplication-engines-are-not-as-accurate-as-we-thought/); [Fasi et al., *Dissecting Tensor Cores via Microbenchmarks*](https://arxiv.org/pdf/2206.02874)).

### Rule 2 — Use two-stage / split-K accumulation for long reductions
For large contraction dimension K (long return histories, many MC time-steps), accumulate FP16/16 MMA results into *separate FP32 registers outside the MMA* and split K into chunks. This recovers accuracy cheaply ([Patterson, *Improving FP16/16 matmul accuracy with two-stage accumulation*](https://www.spatters.ca/two-stage-fp16-mma)).

### Rule 3 — If you need true FP32, get it back via error correction (TF32x3 / Ootomo)
TF32x3-style error-correction recovers near-FP32 accuracy from Tensor Cores while exceeding the FP32 CUDA-core peak — viable for risk numbers that need full precision but want Tensor-Core speed ([Ootomo & Yokota, *Recovering single precision accuracy from Tensor Cores*](https://arxiv.org/pdf/2203.03341)).

### Rule 4 — Mixed precision is *more* accurate than homogeneous low precision
FP16-input/FP32-accumulate beats all-FP16, and is faster — this is the validated sweet spot ([Haidar et al., SC18](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf)).

### Rule 5 — The canonical precedent: bf16 matmul + FP32 accumulate works for quant MC
Google priced financial Monte-Carlo on TPUs where "matrix multiplications in the MXU are computed in bfloat16 prior to being accumulated in single precision" and **still produced accurate estimators** vs GPUs ([Belletti et al., *Tensor Processing Units for Financial Monte Carlo*, 2019](https://research.google/pubs/pub48248/)). This is direct evidence the Ada bf16-in/FP32-accum path is sound for derivatives pricing.

---

## 5. When is FP8/INT8 acceptable? (Indicators vs pricing)

| Workload | Acceptable precision | Reasoning |
|---|---|---|
| **Technical indicators / feature transforms** (already INT8-quantized in kimsfinance's orderflow path, 0–255) | **INT8 or FP8 E4M3** fine | Signals are thresholded/bucketed; quantization error << decision granularity. INT8 is best for *uniform* feature distributions; FP8-E4M3 is better for *normal/heavy-tailed* distributions because exponent bits absorb outliers ([van Baalen et al., *FP8 vs INT8*](https://arxiv.org/pdf/2303.17951)) |
| **Covariance / correlation matrices** | **BF16 in / FP32 accum**, split-K | Inputs are normalized returns (range-safe in BF16); the X·Xᵀ reduction is the accuracy-critical part → FP32 accumulate is mandatory |
| **Monte-Carlo path generation (Heston/SV)** | **BF16 in / FP32 accum** | Validated by TPU precedent; use BF16 not FP16 to avoid variance-process overflow |
| **Option pricing / Greeks / reported P&L** | **FP32 (or TF32x3 error-corrected)** | These are the *numbers you publish*. Quantization here is a correctness/compliance risk, not a tuning knob |
| **Risk (VaR/ES) tail estimates** | **FP32** | Tail estimation amplifies small errors; do not quantize |

**The dividing line:** low precision is acceptable for anything *upstream of a discretizing decision* (a signal that becomes buy/hold/sell, a feature fed to a classifier). It is **not** acceptable for the final dollar-valued output a human or downstream system consumes.

---

## 6. Which quant workloads are GEMM-shaped (and thus Tensor-Core exploitable)?

Tensor Cores only help **dense matmul-shaped** work. The trick is recognizing the GEMM hiding inside a quant routine:

1. **Batched indicator matmuls.** Computing the same linear filter (weighted MA, polynomial/Savitzky-Golay smoothing, linear-regression channels) across thousands of instruments × windows is a batched `[windows × taps] · [taps × series]` GEMM. Use **batched GEMM** (cuBLAS `gemmBatched` / CUTLASS grouped GEMM) — designed for exactly these tall-skinny small matrices ([Abdelfattah et al., *Batched GEMM autotuning*](https://www.netlib.org/utk/people/JackDongarra/PAPERS/performance-design-and-autotuning.pdf)).
2. **Covariance / Gram matrices.** Σ = (1/n) Xᵀ X for an N-asset × T-observation return matrix is a single large GEMM (the SYRK shape). This is the highest-value Tensor-Core target in a risk engine — large, dense, BF16-friendly, FP32-accumulated. Foundational for minimum-variance/Markowitz portfolios ([Cai et al., *Precision vs Shrinkage covariance for portfolio allocation*](https://arxiv.org/abs/2305.11298)).
3. **Batched least-squares / rolling regression.** Factor models, beta estimation, rolling OLS → batched normal-equations (XᵀX, Xᵀy) GEMMs, ideal for tall-skinny batched kernels ([NAG TR1/17, *Batched Least Squares of Tall Skinny Matrices on GPU*](https://support.nag.com/doc/techrep/pdf/tr1_17.pdf)).
4. **Monte-Carlo / Heston.** Correlating Brownian increments (L · Z, where L is a Cholesky factor) and Brownian-bridge construction are matmuls; antithetic/multi-asset path bundling makes them large GEMMs ([Belletti et al. 2019](https://research.google/pubs/pub48248/)). GPU MC pricing already shows ~43× over multi-core CPU in single precision ([Jespersen, *Monte Carlo Evaluation of Financial Options using a GPU*](https://cs.au.dk/~gerth/advising/thesis/claus-jespersen.pdf)) — Tensor Cores extend this where the correlation/payoff step dominates.

**NOT GEMM-shaped** (skip Tensor Cores): elementwise indicators (EMA, ATR, RSI), sorting/ranking, event-driven backtest loops, per-tick orderflow feature extraction. These are memory- or control-bound; the project's own benchmarks already show CPU/cuDF beat naive GPU here.

---

## 7. Pitfalls checklist

- **Don't expect datacenter FP8 numbers on this laptop.** Half-rate FP32-accumulate caps you near INT8 TOPS (~127). Benchmark the actual GEMM, don't trust the whitepaper TFLOPS.
- **Don't feed raw prices to FP16.** Use BF16 (FP32 range) to avoid overflow; or pre-center/normalize.
- **Don't leave "fast accumulate" on for reported numbers.** It silently drops precision.
- **Don't assume "FP32 accumulate" = 23 mantissa bits.** Validate large-K reductions against a CPU FP64 reference; use split-K / two-stage accumulation if error exceeds tolerance.
- **Don't use WMMA and stop.** It's the easy path, not the fast one; shared-memory staging via `mma.sync`/CUTLASS is ~5× better.
- **Don't Tensor-Core-ify small or non-dense ops.** Below ~M,N,K ≈ 256, kernel-launch + quantization overhead dominates; batched GEMM amortizes this only when you have many tiles.
- **Quantize signals, never P&L.** Keep a hard precision boundary between feature pipelines (FP8/INT8 OK) and valuation/risk outputs (FP32 / TF32x3).
- **Validate per-distribution.** INT8 wins on uniform data, FP8-E4M3 on heavy-tailed financial data — measure, don't assume.

---

## Sources
- [NVIDIA Ada GPU Architecture Whitepaper (v2.02)](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [NVIDIA L40 Datasheet (Jan 2023)](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/support-guide/NVIDIA-L40-Datasheet-January-2023.pdf)
- [NVIDIA Dev Forums — Ada GeForce RTX 4090 FP8 cuBLASLt performance](https://forums.developer.nvidia.com/t/ada-geforce-rtx-4090-fp8-cublaslt-performance/250737)
- [NVIDIA Dev Forums — FP8/FP16 accumulation on Ada RTX 4090](https://forums.developer.nvidia.com/t/fp8-fp16-accumulation-on-ada-rtx-4090/294400)
- [NanoReview — RTX 3500 Laptop (Ada) specs](https://nanoreview.net/en/gpu/nvidia-rtx-3500-mobile-ada)
- [Micikevicius et al., *FP8 Formats for Deep Learning* (arXiv:2209.05433)](https://arxiv.org/pdf/2209.05433)
- [van Baalen et al., *FP8 versus INT8 for efficient deep learning inference* (arXiv:2303.17951)](https://arxiv.org/pdf/2303.17951)
- [Spheron — *What is FP8 Quantization?* (2026)](https://www.spheron.network/blog/fp8-quantization-inference-performance-hardware-explained/)
- [Sun et al., *Benchmarking GPU Tensor Cores through CUTLASS* (Applied Sciences 2023)](https://www.mdpi.com/2076-3417/13/24/13022)
- [Patterson, *Implementing a fast Tensor Core matmul on the Ada Architecture*](https://www.spatters.ca/mma-matmul)
- [Patterson, *Improving FP16/16 matmul accuracy with two-stage accumulation*](https://www.spatters.ca/two-stage-fp16-mma)
- [CUTLASS — `58_ada_fp8_gemm` example](https://github.com/NVIDIA/cutlass/blob/main/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu)
- [CUTLASS — GEMM API documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)
- [PyTorch — *Some Matrix Multiplication Engines Are Not As Accurate As We Thought*](https://pytorch.org/blog/some-matrix-multiplication-engines-are-not-as-accurate-as-we-thought/)
- [Fasi et al., *Dissecting Tensor Cores via Microbenchmarks* (arXiv:2206.02874)](https://arxiv.org/pdf/2206.02874)
- [Ootomo & Yokota, *Recovering single precision accuracy from Tensor Cores* (arXiv:2203.03341)](https://arxiv.org/pdf/2203.03341)
- [Haidar et al., *Harnessing GPU Tensor Cores for Fast FP16 Arithmetic* (SC18)](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf)
- [Belletti et al., *Tensor Processing Units for Financial Monte Carlo* (Google, 2019)](https://research.google/pubs/pub48248/)
- [Jespersen, *Monte Carlo Evaluation of Financial Options using a GPU*](https://cs.au.dk/~gerth/advising/thesis/claus-jespersen.pdf)
- [Cai et al., *Precision versus Shrinkage: Covariance Estimation for Portfolio Allocation* (arXiv:2305.11298)](https://arxiv.org/abs/2305.11298)
- [Abdelfattah et al., *Performance, Design, and Autotuning of Batched GEMM for GPUs*](https://www.netlib.org/utk/people/JackDongarra/PAPERS/performance-design-and-autotuning.pdf)
- [NAG TR1/17, *Batched Least Squares of Tall Skinny Matrices on GPU*](https://support.nag.com/doc/techrep/pdf/tr1_17.pdf)
