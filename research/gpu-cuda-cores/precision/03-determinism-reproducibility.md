# Numerical Determinism, Reproducibility, and f32 Hazards for GPU Backtesting

**Research date:** 2026-06-14
**Scope:** kimsfinance GPU pipeline (RTX 3500 Ada, sm_89, CUDA 13). Covers run-to-run non-determinism in parallel reductions/atomics, f32 range hazards, denormals/FTZ, NaN/Inf propagation, compensated summation + re-seeding for recursive indicators, and how to achieve bit-reproducibility or a defensible tolerance band.
**Companion to:** `04-tensor-cores-low-precision.md` (where lower precision is *safe*). This file answers the inverse: where precision/determinism is **NOT** safe, and why.

---

## 0. The core tension

The user's framing is correct *for the indicator values themselves*: a signal like `RSI > 70` or `price crosses SMA` is a coarse, thresholded decision, so 7 decimal digits (f32) is wildly more than the decision needs. But the question "are we over-specifying precision?" conflates two distinct properties:

1. **Accuracy** — is the number close enough to the true value? (coarse decisions → yes, f32 is plenty.)
2. **Reproducibility/determinism** — does the *same* input give the *same* number every run? (independent of accuracy, and the thing that actually breaks backtests.)

A backtest can be perfectly *accurate* (within 0.01%) and still *non-reproducible*, because a value sitting at `RSI = 70.0000001` vs `69.9999998` flips a discrete trade, which cascades into a different equity curve. The hazard is not the indicator's precision — it is the **amplification of tiny numerical noise at decision boundaries**. This is why "trading decisions are coarse" does *not* license sloppy numerics: the coarseness is exactly what makes the boundary discontinuous.

---

## 1. Why reproducible backtests matter (and what breaks them)

A backtest is a hypothesis test on a strategy. If re-running the same code on the same data and same GPU yields a different Sharpe/max-drawdown, you cannot:

- distinguish a genuine code change from numerical noise during walk-forward validation;
- A/B two strategies (the difference may be smaller than the run-to-run jitter);
- trust a "+24.52% OOS / Sharpe 1.465" result (see commit `3a437ad`) — you must be able to regenerate it bit-for-bit or within a stated tolerance.

The industry baseline for quant backtests is a **two-tier reproducibility contract**: continuous values (PnL, equity, prices) are compared with a relative tolerance (commonly `1e-6`), while **trade counts and timestamps must be byte-identical** ([SysTradeBench, arXiv 2604.04812](https://arxiv.org/pdf/2604.04812)). The continuous tolerance absorbs floating-point non-associativity; the integer-exact requirement catches the dangerous case where noise actually changed *which* trades fired.

### The non-associativity root cause

Floating-point addition is **not associative**: `(a+b)+c ≠ a+(b+c)` in general, because each `+` rounds. On a GPU, the *order* in which partial sums are combined depends on thread scheduling, the state of the memory hierarchy, and heuristic schedulers — all of which vary run to run ([Impacts of floating-point non-associativity on reproducibility, arXiv 2408.05148](https://arxiv.org/pdf/2408.05148v3)). The two ingredients of non-reproducibility are therefore **rounding error × execution order** ([Collange et al., reproducible FP atomic addition](https://annals-csis.org/Volume_5/pliks/86.pdf)).

The worst offender is **`atomicAdd` on floats**: threads arrive at the same memory address in an undefined, runtime-dependent order, so the same kernel can produce a different bit pattern on every launch — observed as up to ~1000 distinct results across repeated runs of the same reduction ([Collange et al.](https://annals-csis.org/Volume_5/pliks/86.pdf); [arXiv 2408.05148](https://arxiv.org/pdf/2408.05148v3)). This is precisely why cuDNN's `ConvolutionBackwardFilter`/`Data`, `PoolingBackward`, and `CTCLoss` are documented as **non-reproducible even on the same GPU** — they use float atomics ([NVIDIA cuBLAS/cuDNN reproducibility docs](https://docs.nvidia.com/cuda/cublas/)).

### What kimsfinance already gets right

The codebase shows good instincts here:

- `gpu/aggregation.rs` computes high/low via **order-preserving u64 encodings + `atomicMax`/`atomicMin`** (`encode_ordered_f64`/`decode_ordered_u64`). Integer atomics are **order-independent** (max/min are associative *and* commutative over the totally-ordered u64 image), so high/low are bit-reproducible. This is the right pattern.
- The test at `aggregation.rs:606` explicitly acknowledges *"atomicAdd order differs from sequential CPU summation"* and compares volume with a relative tolerance `1e-9 * max(|v|, 1.0)` rather than `==`. Volume/quote-volume use float `atomicAdd` and are therefore **not bit-reproducible** — correctly handled by tolerance, not equality.
- `kernels_backtest.cu` keeps timestamps as `int64_t` end-to-end and PnL/equity in `double`, with an explicit comment that the per-strategy loops are memory-latency bound, so Ada's 1:64 FP64 rate is not the bottleneck. This is the right call (see §5).

**Reproducibility guarantee available to us:** cuBLAS/cuDNN guarantee bit-identical results across runs *only* on the same architecture **and same SM count**, and *not* across toolkit versions ([NVIDIA docs](https://docs.nvidia.com/cuda/cublas/)). For our own kernels, the equivalent guarantee holds iff we avoid float atomics and fix the reduction tree. Pin the CUDA toolkit version in the backtest provenance record.

---

## 2. f32 range hazards vs f64 (overflow / underflow / cancellation)

f32 has a **24-bit significand (~7.2 decimal digits)** and exponent range to ~3.4×10³⁸ ([Single-precision floating-point format, Wikipedia](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)). The exponent range is huge — **overflow is rarely the f32 problem for price data**. The real hazards are **absorption** and **cancellation**:

- **Absorption in cumulative sums.** The first integer not representable in f32 is 2²⁴ = 16,777,217: `16777216 + 1 → 16777216` ([Wikipedia](https://en.wikipedia.org/wiki/Single-precision_floating-point_format); [Goldberg, "What Every Computer Scientist Should Know…"](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)). For a running **cumulative volume delta** or **CVD** (a CLAUDE.md orderflow feature, "running sum of volume deltas"), once the accumulator exceeds ~1.6×10⁷ in f32, individual tick increments below ~1 are **silently dropped**. A high-volume instrument crosses 16.7M cumulative contracts quickly, so an f32 CVD will *stall* while an f64/int64 CVD keeps counting. **CVD, cumulative PnL, and equity curves must not be f32 accumulators.**
- **Price × quantity products (quote volume / notional).** `price * qty` summed over a day can reach 10⁸–10¹¹. In f32 the absolute ULP at 10⁹ is ~64, so per-trade notionals under ~$64 vanish from the running total. f64 ULP at 10⁹ is ~1.2×10⁻⁷ — effectively exact for this range.
- **Catastrophic cancellation.** Subtracting two nearly-equal accumulated values (e.g., an indicator computed as `running_sum - lagged_running_sum`, or a variance via `E[x²] - E[x]²`) loses all the low-order bits that error has crept into: `1.0004 - 1.0 = 0.0004` with huge relative error ([arXiv 1511.06227](https://arxiv.org/pdf/1511.06227); [Goldberg](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)). Bollinger Band width and any rolling-variance indicator are exposed here. Use Welford's online variance, never the textbook `E[x²]-E[x]²` form, in *any* precision.

**Rule of thumb:** f32 is safe for **per-bar, bounded** quantities (a single bar's RSI, ATR, the SMA *of a window*, a normalized z-score). f32 is hazardous for any **monotonically growing running total** (CVD, equity, notional, cumulative volume) — these need f64 or int64.

---

## 3. Denormals and Flush-to-Zero (FTZ) — and a live finding in this build

**`build.rs` compiles every kernel with `-use_fast_math`** (confirmed at `rust/build.rs:258` and in the captured nvcc build log, formerly tracked as `rust/check_output.txt`). On sm_89, `-use_fast_math` implies **`-ftz=true`** (denormals flushed to zero), plus `-prec-div=false` and `-prec-sqrt=false` (approximate division and reciprocal-sqrt) ([NVCC Compiler Driver docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html); [CUDA Floating-Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html)).

FTZ flushes subnormal inputs/outputs (values below ~1.2×10⁻³⁸ in f32) to ±0. The trade-offs:

- **Performance:** For add/mul/fma the compiler just sets an instruction modifier — **no perf effect**. The real win is on hardware-approximated functions like `rsqrtf()`, where denormal handling is expensive; NVIDIA measured a **~20% speedup** in an n-body sim from enabling FTZ ([CUDA Pro Tip: Flush Denormals with Confidence](https://developer.nvidia.com/blog/cuda-pro-tip-flush-denormals-confidence/)). FTZ flags are no-ops on f64 and on CC < 2.0 ([CUDA Programming Guide, Floating-Point](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)).
- **Accuracy/determinism risk:** `-prec-div=false`/`-prec-sqrt=false` mean division and sqrt are **not correctly rounded** — they can differ from the CPU's IEEE result by a few ULP. For most price-scale indicators this is invisible at the decision boundary, **but it is the kind of difference that can flip a value sitting exactly on a threshold**, and it breaks GPU↔CPU bit-equality. Subnormals being flushed is harmless for price data (prices are never ~10⁻³⁸) but matters for *normalized* features near zero — e.g., a z-score denominator, or a near-flat-market volatility term, where a subnormal intermediate could flush to 0 and create a `0/0 → NaN` (see §4).

**Recommendation:** Keep `-use_fast_math` for the *rendering/throughput* path, but compile the **backtest and signal kernels with explicit `-ftz=false -prec-div=true -prec-sqrt=true -fmad=true`** so that (a) GPU division/sqrt match the CPU reference for the `0.01%` cross-check, and (b) threshold comparisons are computed on correctly-rounded values. The cost is ~0 for add/mul/fma-dominated code and only matters where you actually call `rsqrtf`/`fdividef`. This is a per-translation-unit decision; split the NVCC invocation in `build.rs`.

---

## 4. NaN / Inf propagation — the silent backtest killer

IEEE-754 makes Inf propagate through most ops (`x + Inf = Inf`), **except** cases like `finite / Inf = 0` and `Inf - Inf = NaN` ([NaN, Wikipedia](https://en.wikipedia.org/wiki/NaN)). NaN is *contagious* in arithmetic (`NaN op x = NaN`) but **treacherous in comparisons and min/max reductions**:

- Every comparison with NaN returns false, so a `price > sma` gate **silently evaluates false** when `sma` is NaN — the strategy quietly stops trading instead of erroring. EMA/SMA already seed the first `period-1` values as `NaN` (`ema.rs:123`), which is correct *if* the signal layer explicitly skips them.
- **min/max reductions are the trap.** IEEE-754-2008 `minNum`/`maxNum` ignored NaN (`min(x,NaN)=x`); IEEE-754-2019 **removed them** because they are non-associative under signaling NaN, reverting to NaN-propagating min/max ([IEEE 754-2019 minNum/maxNum removal](https://grouper.ieee.org/groups/msc/ANSI_IEEE-Std-754-2019/background/minNum_maxNum_Removal_Demotion_v3.pdf); [Agner Fog, NaN propagation](https://www.agner.org/optimize/nan_propagation.pdf)). The consequence: a parallel min/max can return a **different result depending on lane order** when a NaN is present, with only the invalid-operation status bit set as a hint — and almost nobody reads that bit ([Agner Fog](https://www.agner.org/optimize/nan_propagation.pdf)). A single NaN tick in a high/low reduction can either poison the result or be silently swallowed, **non-deterministically**.

CUDA's `fmin`/`fmax` follow the "return the non-NaN operand" convention, so a NaN can be *hidden* in a high/low reduction rather than propagated — exactly the silent-failure mode. The codebase's u64-encoded `atomicMax`/`atomicMin` for high/low sidesteps the float-min/max NaN ambiguity (NaN's u64 image sorts deterministically), which is another reason that pattern is good — but it means a NaN price would land at an *extreme* rather than being rejected. **Validate/clip NaN on ingest, not in the reduction.**

**Mitigations:**
1. **Reject NaN/Inf at the data boundary** (`download_binance_data.py` / ingest), not deep in a kernel. Count and log them; never let them reach a reduction.
2. In signal kernels, gate explicitly: `bool valid = isfinite(sma) && isfinite(rsi);` and treat invalid bars as "no signal," never as a passed/failed threshold.
3. Add a **NaN/Inf assertion** to the backtest harness: if any equity-curve point is non-finite, **hard-fail the run** rather than reporting a number.

---

## 5. Bounding error in recursive indicators (EMA, RSI Wilder, CVD)

EMA is an IIR filter with a loop-carried dependency: `EMA[i] = α·close[i] + (1-α)·EMA[i-1]` (`ema.rs:139`). RSI's Wilder smoothing and any cumulative sum are likewise recursive. Two distinct concerns:

- **Error accumulation.** Each step rounds; over millions of bars the rounding errors random-walk. For a *stable* IIR filter (`0 < α < 1`, `|1-α| < 1`) the recursion is **contractive** — old errors decay by `(1-α)` per step, so EMA error stays bounded and does **not** grow without limit; f32 EMA is generally fine for accuracy. The dangerous recursives are the **non-contractive** ones: a plain **CVD / cumulative sum has gain 1**, so errors accumulate monotonically (§2) — that is where f32 fails, not EMA.
- **Reproducibility.** Because the EMA recurrence is strictly sequential, it is **naturally deterministic** — there's no reduction order to vary. The reproducibility risk in recursive indicators comes only if you *parallelize* them (e.g., a parallel-scan EMA), which reorders the associative-looking but non-associative FP combine.

**Mitigations:**
1. **Compensated (Kahan / Gill–Møller) summation for all running totals.** Kahan keeps a running compensation term for the lost low-order bits; pairwise summation gets ~89% of naive throughput with better-than-Kahan accuracy and is the natural GPU reduction shape (reduce in a balanced binary tree per block) ([SIMDizing pairwise sums, ACM 2568070](https://dl.acm.org/doi/10.1145/2568058.2568070); [Dmitruk, vectorized Kahan/Gill–Møller, Wiley 2023](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.7763)). For CVD/equity in f32 this recovers most of the lost precision; in f64 it makes the sum essentially exact. The accuracy/throughput sweet spot for our reductions is **pairwise (tree) summation**, which is *also* what gives a deterministic, reproducible result (fixed tree → fixed order).
2. **Periodic re-seeding.** For long recursive runs, periodically **re-anchor** the recursion to an exactly-computed value: e.g., every N bars recompute the EMA seed from a fresh SMA of the trailing window (the same seeding EMA already uses at `ema.rs:127`, `sum/period`), or reset the CVD accumulator at known session boundaries (daily). This bounds worst-case drift to N bars of accumulation and matches how traders interpret these series (intraday CVD typically resets daily anyway).
3. **Accumulate in higher precision than you compare in.** Compute the *running total* in f64, store/compare the *per-bar value* in f32. This is the orderflow path's existing instinct (features quantized to INT8 *after* being computed in full precision). Never accumulate in the storage precision.

---

## 6. Bit-reproducible vs tolerance-band — choosing per quantity

You do not need *everything* bit-reproducible. Classify each quantity:

| Quantity | Target | How |
|---|---|---|
| Trade entries/exits, trade **count** | **Bit-identical** | Compute the *decision* on a deterministic value; fixed reduction tree; no float atomics; `-prec-div=true` for any division feeding a threshold |
| Timestamps, bar indices | **Bit-identical** | Keep as `int64` end-to-end (already done, `aggregation.rs:217`) |
| High / Low | **Bit-identical** | u64-encoded `atomicMax`/`atomicMin` (already done) |
| Volume, quote-volume, CVD, equity, PnL | **Tolerance `1e-6`–`1e-9` rel** | f64 accumulation + Kahan/pairwise; compare with rel-tolerance, *not* `==` |
| Indicator values (RSI/SMA/ATR/MACD) | **Tolerance `0.01%`** (existing) | f32 acceptable for the *value*; ensure the *threshold compare* is on a stable value |

**The one non-negotiable:** the *discrete* outputs (which trades fired, how many, when) must be bit-identical run-to-run and ideally GPU↔CPU-identical. Continuous P&L can ride a tolerance band. If a tiny numerical difference changes the trade set, that is not "acceptable noise" — it is a strategy sitting on a knife-edge threshold, and you should detect it (re-run with perturbed inputs; if the trade set moves, the edge is fragile).

---

## 7. Actionable checklist

**Build / compiler**
- [ ] Split NVCC compilation: keep `-use_fast_math` for render/throughput kernels; compile **signal + backtest kernels with `-ftz=false -prec-div=true -prec-sqrt=true -fmad=true`** so GPU division/sqrt match the CPU reference and threshold compares use correctly-rounded values.
- [ ] Pin and record the CUDA toolkit version in every backtest's provenance (cuBLAS/cuDNN reproducibility is *not* guaranteed across toolkit versions).

**Reductions / atomics**
- [ ] No float `atomicAdd` in any path whose result feeds a discrete decision. Use a **fixed pairwise (binary-tree) reduction** for deterministic order.
- [ ] Keep the u64-encoded `atomicMax`/`atomicMin` for high/low (good — order-independent, bit-reproducible).
- [ ] Volume/quote-volume `atomicAdd` results: continue comparing with relative tolerance, never `==` (already done); migrate to f64 + Kahan/pairwise if exactness is wanted.

**Range / precision**
- [ ] Accumulate **CVD, equity, PnL, notional in f64 (or int64)** — never f32 (absorption past 2²⁴ ≈ 1.67×10⁷).
- [ ] Compute rolling variance / Bollinger via **Welford**, never `E[x²]-E[x]²` (cancellation).
- [ ] f32 is fine for *per-bar bounded* indicator values; f16 is **not** safe for accumulators (max ≈ 6.55×10⁴) — bf16 has f32-range but only 7 mantissa bits (~2–3 digits), tolerable only for coarse normalized features, never for prices or running totals.

**NaN / Inf**
- [ ] Reject/clip NaN & Inf at ingest; count and log them. Never let them reach a reduction.
- [ ] Gate signals with explicit `isfinite()`; treat invalid bars as "no signal," not as a passed threshold.
- [ ] Hard-fail any backtest whose equity curve contains a non-finite value.

**Recursive indicators**
- [ ] Periodically re-seed EMA/RSI from a freshly computed SMA window; reset CVD at session boundaries.
- [ ] Accumulate in higher precision than you store/compare in.

**Validation**
- [ ] Adopt the two-tier contract: continuous values `rel ≤ 1e-6`; **trade counts & timestamps byte-identical**.
- [ ] Run a determinism smoke test: execute the same backtest 5× on the same GPU; assert identical trade sets and equity within tolerance.
- [ ] Run an input-perturbation test: nudge inputs by 1 ULP; if the trade set changes, flag the strategy as boundary-fragile.

---

## Sources

- [Impacts of floating-point non-associativity on reproducibility for HPC and deep learning (arXiv 2408.05148)](https://arxiv.org/html/2408.05148v3)
- [Collange et al., "Reproducible floating-point atomic addition in data-parallel environment"](https://annals-csis.org/Volume_5/pliks/86.pdf)
- [NVIDIA cuBLAS / cuDNN reproducibility documentation](https://docs.nvidia.com/cuda/cublas/)
- [NVCC Compiler Driver — fast-math / ftz / prec-div / prec-sqrt](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- [CUDA Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html)
- [CUDA Programming Guide — Floating-Point Computation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- [CUDA Pro Tip: Flush Denormals with Confidence (NVIDIA blog)](https://developer.nvidia.com/blog/cuda-pro-tip-flush-denormals-confidence/)
- [Single-precision floating-point format (Wikipedia)](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)
- [Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic"](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- [Empirical Research… Precision-specific Operation (arXiv 1511.06227)](https://arxiv.org/pdf/1511.06227)
- [Agner Fog, "Parallel floating point exception tracking and NaN propagation"](https://www.agner.org/optimize/nan_propagation.pdf)
- [IEEE 754-2019: Removal/Demotion of minNum/maxNum](https://grouper.ieee.org/groups/msc/ANSI_IEEE-Std-754-2019/background/minNum_maxNum_Removal_Demotion_v3.pdf)
- [NaN (Wikipedia)](https://en.wikipedia.org/wiki/NaN)
- [SIMDizing pairwise sums (ACM 10.1145/2568058.2568070)](https://dl.acm.org/doi/10.1145/2568058.2568070)
- [Dmitruk & Stpiczyński, vectorized Kahan/Gill–Møller summation (Wiley CCPE 2023)](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.7763)
- [SysTradeBench: drift-aware trading-system reproducibility (arXiv 2604.04812)](https://arxiv.org/pdf/2604.04812)
- [bfloat16 range and precision (John D. Cook)](https://www.johndcook.com/blog/2018/11/15/bfloat16/)
