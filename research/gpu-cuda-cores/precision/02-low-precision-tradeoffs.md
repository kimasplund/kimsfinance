# Low-Precision GPU Compute for Finance on Ada (sm_89): Tradeoffs and a Decision Table

**Research date:** 2026-06-14
**Target HW:** NVIDIA RTX 3500 Ada (Ada Lovelace, AD104-class, `sm_89`), 12 GB GDDR6, CUDA 13.1.
**Question being answered:** Are we over-specifying numerical precision? Trading decisions are coarse (thresholds, crossovers, sign tests, discrete trades). Where can we drop below FP32 to FP16/BF16/TF32/INT8 to unlock performance, and where is it *not* safe?

**Companion doc:** `../04-tensor-cores-low-precision.md` covers the Tensor-Core/GEMM programming model and Ada's GeForce-class FP16/FP8→FP32 half-rate throttle. This doc focuses on the **format-by-format range/precision tradeoffs and the per-indicator safety map**.

---

## 1. The core framing: precision must survive the *computation*, not the *decision*

The user's intuition is correct but incomplete. A signal like `RSI > 70` or `price crosses SMA` is coarse: the final comparison tolerates errors of ~0.01–0.1 in the indicator value without changing the trade. The danger is not the decision — it is the **arithmetic that produces the value**. Two computation shapes determine whether low precision is safe:

1. **How many dependent operations feed one output value** (reduction length / recurrence depth). Naive summation error grows as O(√N) RMS for random data and O(N) worst-case ([Kahan summation, Wikipedia](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)). A 7-mantissa-bit format that is fine for a single multiply is catastrophic after 10⁶ accumulations.
2. **Whether a subtraction of near-equal large numbers occurs** (catastrophic cancellation). Subtracting nearby values can make the *relative* error of the result arbitrarily large even when inputs are nearly exact ([Catastrophic cancellation, Wikipedia](https://en.wikipedia.org/wiki/Catastrophic_cancellation)). This is exactly what naive variance (`E[x²] − E[x]²`) does — and Bollinger Bands, ATR, and z-score features all touch variance.

So the rule is: **the final compare is cheap and tolerant; the path to it is where bits die.** Low precision is safe for *short, non-cancelling, bounded* compute and unsafe for *recursive, long-cumulative, or cancellation-prone* compute.

---

## 2. The five formats: range, precision, and what they cost you

| Format | Bits (S/E/M) | Mantissa precision (ε) | Decimal digits | Max magnitude | Smallest normal |
|---|---|---|---|---|---|
| **FP64** | 1/11/52 | 2⁻⁵² ≈ 2.2e-16 | ~15–16 | 1.8e308 | 2.2e-308 |
| **FP32** | 1/8/23 | 2⁻²³ ≈ 1.2e-7 | ~7 | 3.4e38 | 1.2e-38 |
| **TF32** | 1/8/10 | 2⁻¹⁰ ≈ 9.8e-4 | ~3 | 3.4e38 (FP32 range) | 1.2e-38 |
| **BF16** | 1/8/7 | 2⁻⁷ ≈ 7.8e-3 | ~2–3 | 3.4e38 (FP32 range) | 1.2e-38 |
| **FP16** | 1/5/10 | 2⁻¹⁰ ≈ 9.8e-4 | ~3–4 | 65,504 | 6.1e-5 |
| **INT8** | quantized | range/255 per-tensor | ~2–3 (if range tight) | set by scale | set by scale |

Two facts dominate everything below ([bfloat16 range/precision, John D. Cook](https://www.johndcook.com/blog/2018/11/15/bfloat16/); [machine epsilon, Wikipedia](https://en.wikipedia.org/wiki/Machine_epsilon)):

- **FP16 has good precision (ε≈1e-3, ~3–4 digits) but a tiny range (max 65,504).** A raw BTC price of 68,000 is representable, but `price²` (4.6e9, needed for variance) **overflows FP16 to ±inf.** Volume-delta cumulative sums also blow past 65,504 quickly. FP16 is a *range* trap.
- **BF16 has full FP32 range but only ~2 digits of precision (ε≈8e-3).** It is essentially a truncated FP32 ([Cerebras: To Bfloat or not to Bfloat](https://www.cerebras.ai/blog/to-bfloat-or-not-to-bfloat-that-is-the-question)). Deep nets tolerate this because they are sensitive to exponent, not mantissa — but a price of 68,000.25 in BF16 rounds to steps of **512** (68,000 → nearest representable ≈ 68,096 or 67,584). BF16 is a *precision* trap for absolute price levels.

**The single most important number for finance:** BF16's mantissa quantum at price ~65,536 is `2⁻⁷ × 2¹⁶ = 512` ticks. At price ~256 it is 2 ticks. **BF16 absolute precision degrades linearly with the price magnitude** — fine for percent-change/normalized features, useless for an SMA you threshold-cross against the raw price.

---

## 3. Ada sm_89 throughput and memory advantages (what you actually gain)

Speedup comes from two independent levers. For most kimsfinance indicators (memory-bound, as the FP32 SMA conversion just confirmed at 1.33–1.74× scaling toward the 2× bandwidth ceiling), **memory bandwidth is the lever, not FLOPS.**

**(a) Memory/bandwidth (applies to ALL kernels, CUDA-core or Tensor-Core):**
- FP32→FP16/BF16 halves bytes moved → up to **2× on bandwidth-bound kernels** (the regime kimsfinance lives in). This is the realistic, broadly-applicable win.
- FP32→INT8 quarters bytes → up to **4×**. The orderflow path already banks this: `quantization.rs` documents 4× compression (24 B → 6 B/tick), 19 GB → 2.4 GB for 10 strategies, target accuracy <0.01% deviation.
- Halving operand width also halves register/shared-memory pressure → higher occupancy → secondary throughput gains.

**(b) Tensor-Core FLOPS (applies only to GEMM-shaped work):** On the AD102 RTX 4090 the dense rates are FP16/BF16/TF32 **82.6 TFLOPS** dense (165 with 2:4 sparsity) and INT8 **660 TOPS** dense (1321 with sparsity) ([RTX 4090 spec data, search aggregation](https://flopper.io/gpu/nvidia-geforce-rtx-4090-24gb/spec-sheet.pdf); [NVIDIA Ada GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)). **Critical caveat (see companion doc):** on GeForce/laptop Ada, FP16→FP32 and FP8→FP32 *accumulate* runs at **half rate** — a hardware product-segmentation throttle ([NVIDIA Dev Forums: Ada RTX 4090 FP8 cuBLASLt](https://forums.developer.nvidia.com/t/ada-geforce-rtx-4090-fp8-cublaslt-performance/250737)). The RTX 3500 Ada delivers ~127 INT8 TOPS and ~16–23 FP32 TFLOPS. **Tensor cores only help indicators that are matmul-shaped** (covariance matrices, multi-asset feature projections, batched MC) — they do nothing for a per-element SMA/EMA scan.

**Bottom line on speedup:** For the SMA-style indicator population, expect **~1.3–2.0× from FP32→FP16/BF16** (bandwidth) and **~3–4× from FP32→INT8** on bounded features. Do not expect tensor-core FLOPS gains unless you reshape work into GEMMs.

---

## 4. Mixed precision: compute low, ACCUMULATE in FP32 (the universal rule)

The non-negotiable pattern, identical to ML mixed-precision training: **multiply/store in FP16/BF16/INT8, but accumulate sums in FP32 (or FP64).** When you add many small values into a large running total in low precision, the small values are *swamped* and lost entirely — the same failure ML AllReduce avoids by keeping `reduce_dtype=float32` ([BFloat16 vs Float16 configurations, apxml](https://apxml.com/courses/distributed-training-pytorch-fsdp/chapter-3-mixed-precision-memory-optimization/bfloat16-vs-float16-configurations)).

This decouples the two error sources:
- **Storage/bandwidth precision** (the operands in HBM) → can be FP16/BF16/INT8, giving the 2–4× memory win.
- **Accumulator precision** (the running sum register) → stays FP32/FP64, preserving the result.

Ada Tensor Cores natively do this: FP16/BF16 inputs, FP32 accumulator. **But "FP32 accumulate" Tensor Cores may carry fewer than 23 mantissa bits internally**, and large-K reductions compound the loss ([PyTorch: Some Matrix Multiplication Engines Are Not As Accurate As We Thought](https://pytorch.org/blog/some-matrix-multiplication-engines-are-not-as-accurate-as-we-thought/)). For long reductions, split-K with explicit FP32 register accumulation (or Kahan/two-stage compensation) recovers accuracy cheaply. **Disable library "fast-accumulate" modes for any number you report** — they skip periodic promotion of partials and are lossy.

Mixed precision (low compute + FP32 accumulate) is **strictly more accurate** than homogeneous low precision and captures nearly all the performance — it is the default you want, not the exception.

---

## 5. Indicator-class safety map: where it's SAFE vs UNSAFE

### SAFE for low precision (FP16/BF16, often INT8 on normalized features)

- **Element-wise / windowed-mean indicators (SMA, WMA, VWMA, ROC, Williams %R, Stochastic %K).** Short fixed-window reductions (period 14–200). With FP32 accumulate, FP16/BF16 inputs lose <0.01% — far below the threshold granularity of `RSI>70` or `%K>80`. **The FP32 SMA conversion you measured can go to FP16-storage/FP32-accumulate for another ~1.3–2× on top.** *Caveat:* keep the **raw price** the SMA is compared against in FP32 — don't BF16 the price itself (512-tick quantum at BTC levels).
- **Normalized / bounded features (order_imbalance ∈ [0,1], z-scored velocities, RSI/MFI/CCI in [0,100]).** These are *designed* for INT8: tight known range → range/255 quantum is ~0.4% of full scale. This is exactly why the orderflow path quantizes to INT8 at <0.01% backtest deviation (`quantization.rs`). **Extend INT8 to any indicator whose output is a bounded oscillator.**
- **Crossover/sign detection inputs (MACD line = EMA_fast − EMA_slow → sign).** The *subtraction* is cancellation-prone (see unsafe), but if both EMAs are computed in FP32 and only the *sign* is consumed, the comparison is robust. Compute the EMAs carefully; the threshold is forgiving.
- **GEMM-shaped batch work (multi-asset feature projection, batched chart-feature transforms).** BF16 inputs + FP32 accumulate on Tensor Cores — the canonical ML pattern. BF16 over FP16 here because prices need FP32 range.

### UNSAFE for low precision (keep FP32, prefer FP64)

- **Recursive / IIR smoothing (EMA, Wilder's smoothing in RSI/ATR/ADX, SuperTrend, Parabolic SAR).** `EMA[i] = α·close[i] + (1−α)·EMA[i−1]` is a loop-carried recurrence (the `ema.rs` kernel comment confirms this prevents SIMD). Rounding error at each step **feeds the next step and compounds geometrically with the recurrence depth.** In BF16 (ε≈8e-3) a 50-bar EMA drifts by percent-level errors; the project correctly keeps `ema_cpu`/`ema_gpu` in **f64**. **Do not lower EMA/Wilder precision — the recurrence is the whole point of the indicator and it has no error-reset.**
- **Long cumulative sums (OBV, VWAP, Cumulative Volume Delta, Anchored VWAP).** OBV runs a multi-level prefix scan over the full series (`obv.rs`); VWAP accumulates `Σtpv / Σvolume` over an intraday session (`vwap.rs` explicitly states **"FP64 is required"** because the host cumulative sums must agree with the CPU reference). Cumulative volume delta can reach 10⁷+ — **past FP16's 65,504 range entirely (overflow to inf)** and well into BF16's coarse-quantum regime. Naive low-precision cumulation loses O(N) bits. **Keep cumulative paths in FP32-accumulate minimum; FP64 where session-length matches a CPU reference. INT8 is acceptable only for the *per-tick delta* (bounded), never the running total.**
- **Variance / standard-deviation indicators (Bollinger Bands, ATR's TR variance, any z-score denominator).** `bollinger.rs` computes `Σ(x−mean)²` and `sqrt(variance)` in **double**. Variance via `E[x²]−E[x]²` is the textbook catastrophic-cancellation case; even the two-pass `Σ(x−mean)²` form loses precision in low formats, and `x²` for raw prices **overflows FP16**. Welford's online algorithm is the numerically stable route but still needs ≥FP32 state ([Welford variance, accurate mean/variance, arXiv 2206.10662](https://arxiv.org/pdf/2206.10662)). A noisy std feeds directly into band width → false breakout signals. **Variance stays FP32/FP64; never FP16/BF16/INT8 for the accumulator.**
- **GEMM with very large contraction dimension K (long return histories, dense covariance over 10⁵+ samples).** Even FP32-accumulate Tensor Cores lose low-order bits at large K. Use split-K + FP32 (or TF32x3 error-correction) if the covariance feeds a risk number, not just a sign.

### The "free" middle: TF32

TF32 (FP32 range, 10-bit mantissa, ~3 digits) is a **drop-in replacement for FP32 in cuBLAS/cuDNN matmul** with no code change and ~no overflow risk (full FP32 range). For *matmul-shaped* finance work that currently uses FP32, TF32 gives a near-free speedup at ~3-digit accuracy — acceptable for feature transforms feeding thresholds, **not** acceptable for variance/cumulative accumulators (those aren't matmuls anyway). TF32 does nothing for element-wise CUDA-core kernels.

---

## 6. Decision Table: workload → recommended precision

| Workload class | Examples in repo | Recommended precision | Expected speedup vs FP32 | Accuracy cost | Why |
|---|---|---|---|---|---|
| Windowed mean / element-wise | SMA, WMA, VWMA, ROC | **FP16 store / FP32 accumulate** | 1.3–2× (bandwidth) | <0.01%, below threshold granularity | Short reduction, no recurrence; memory-bound |
| Bounded oscillator output | RSI, MFI, CCI, Stochastic, Williams %R (final value) | **INT8 (per-tensor scale)** | up to 4× (memory) | ~0.4% of range; <0.01% on decisions | Known tight range → quantizes cleanly |
| Normalized orderflow features | imbalance, z-scored velocity | **INT8 (already shipping)** | 4× memory (19→2.4 GB) | <0.01% backtest deviation (measured) | Designed bounded; per-feature calibration |
| Crossover / sign tests | MACD sign, price-cross-SMA | **FP32 inputs, coarse compare** | n/a (compute is cheap) | none if inputs FP32 | Subtraction cancels; keep operands FP32 |
| Recursive / IIR smoothing | EMA, Wilder (RSI/ATR/ADX), SuperTrend, PSAR | **FP32 minimum, FP64 preferred** | none — do not lower | error compounds geometrically with depth | Loop-carried; no error reset |
| Long cumulative sums | OBV, VWAP, CVD, anchored VWAP | **FP32 accumulate; FP64 if CPU-ref-matched** | none on accumulator (INT8 ok for per-tick delta) | O(N) bit loss if lowered; FP16 overflows | Sums exceed FP16 range; swamping |
| Variance / std / z-denominator | Bollinger, ATR variance | **FP32/FP64 accumulator** | none — do not lower | catastrophic cancellation; band noise → false signals | `x²` overflows FP16; cancellation |
| GEMM, small/medium K | feature projection, multi-asset transforms | **BF16 in / FP32 accumulate (Tensor Core)** or **TF32** | 1.5–2× (throttled GeForce Ada) | ~2–3 digits; fine for thresholds | BF16 keeps FP32 range for prices |
| GEMM, large K (risk numbers) | dense covariance over 10⁵ samples | **TF32 / split-K FP32 / TF32x3 error-correct** | ~1.3–1.5× with accuracy guard | near-FP32 with correction | large-K reduction loss |

**Format-selection heuristics:**
- **Never FP16 for anything that can exceed ~60,000** (raw volume sums, price², cumulative deltas) — range trap, silent overflow to inf.
- **Prefer BF16 over FP16 when range matters and precision is forgiving** (normalized/percent features in a matmul). Prefer FP16 over BF16 only for small-magnitude, precision-sensitive values that never grow.
- **INT8 only when the range is known and bounded** (oscillators, normalized features). Calibrate per-tensor.
- **Accumulate in FP32 always; FP64 only where you must match a CPU reference bit-for-bit** (the VWAP/OBV pattern already in the repo).

---

## 7. Answer to the user's question

**Yes, you are over-specifying precision on a meaningful subset — but the codebase is already mostly correct.** The orderflow INT8 path and the f64 EMA/VWAP/Bollinger choices are the *right* calls: the project intuitively split bounded features (INT8) from recursive/cumulative/variance paths (f64). The remaining upside is on the **element-wise / windowed-mean indicators (SMA, WMA, VWMA, ROC, oscillator final values)**: these are memory-bound, threshold-consumed, and safe to drop to FP16-store/FP32-accumulate (another ~1.3–2×) or INT8 for bounded oscillators (~4× memory). The places it is **never** safe — and where the f64 choices must stay — are exactly the three structural hazards: **recursive smoothing (geometric error growth), long cumulative sums (O(N) loss + FP16 overflow), and variance/std (catastrophic cancellation + FP16 overflow of x²).** The decision is coarse; the arithmetic that produces it is not, and those three classes are where the bits matter.

---

## Sources

- [NVIDIA Ada GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) — Ada 4th-gen Tensor Core formats (FP8/FP16/BF16/TF32/INT8/INT4), 2:4 sparsity.
- [RTX 4090 spec sheet aggregation](https://flopper.io/gpu/nvidia-geforce-rtx-4090-24gb/spec-sheet.pdf) — dense/sparse FP16/TF32/INT8 rates for AD102.
- [NVIDIA Dev Forums — Ada RTX 4090 FP8 cuBLASLt performance](https://forums.developer.nvidia.com/t/ada-geforce-rtx-4090-fp8-cublaslt-performance/250737) — GeForce/laptop Ada FP16/FP8→FP32 half-rate throttle.
- [bfloat16 range and precision — John D. Cook](https://www.johndcook.com/blog/2018/11/15/bfloat16/) — BF16 7-mantissa-bit / FP32-range tradeoff.
- [Machine epsilon — Wikipedia](https://en.wikipedia.org/wiki/Machine_epsilon) — FP16 ε≈9.8e-4, format precision definitions.
- [To Bfloat or not to Bfloat — Cerebras](https://www.cerebras.ai/blog/to-bfloat-or-not-to-bfloat-that-is-the-question) — BF16 as truncated FP32, exponent-vs-mantissa sensitivity.
- [BFloat16 vs Float16 configurations — apxml](https://apxml.com/courses/distributed-training-pytorch-fsdp/chapter-3-mixed-precision-memory-optimization/bfloat16-vs-float16-configurations) — swamping in low-precision accumulation, FP32 reduce_dtype rule.
- [Kahan summation algorithm — Wikipedia](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) — naive sum error O(√N) RMS / O(N) worst case; compensated summation.
- [Catastrophic cancellation — Wikipedia](https://en.wikipedia.org/wiki/Catastrophic_cancellation) — subtraction of near-equal values; ill-conditioning.
- [Accurate and consistent calculation of mean and variance — arXiv 2206.10662](https://arxiv.org/pdf/2206.10662) — Welford stability, naive variance cancellation, Kahan inadequacy for variance.
- [Some Matrix Multiplication Engines Are Not As Accurate As We Thought — PyTorch blog](https://pytorch.org/blog/some-matrix-multiplication-engines-are-not-as-accurate-as-we-thought/) — sub-23-bit internal Tensor-Core accumulation, large-K compounding.
- [Numerical Precision in ONNX and AI Inference — emmtrix](https://www.emmtrix.com/wiki/Numerical_Precision_in_ONNX_and_AI_Inference) / [FP16 precision limits — Medium](https://akbu.medium.com/floating-point-precision-and-its-limitations-cfb7247d7789) — FP16 ~3–4 decimal digits, error accumulation in extended computation.
