# 00 — GPU Precision Policy (Synthesis)

**Date:** 2026-06-14 | **HW:** RTX 3500 Ada (AD104, sm_89), CUDA 13.1, 12 GB GDDR6 @ ~432 GB/s
**Inputs synthesized:** `01-accuracy-budget.md`, `02-low-precision-tradeoffs.md`, `03-determinism-reproducibility.md`, `04-indicator-precision-classes.md`, `05-decision-tolerance.md`, `06-gap-analysis.md`
**Status:** Recommended policy. This is the decision-of-record for the precision-reduction program; it supersedes the ad-hoc kernel-by-kernel conversion currently underway in `sma.rs`.

---

## Executive summary (read this first)

- **Yes, we are over-specifying precision — but only on the *front half* of the pipeline (indicators → signals), and the codebase is already mostly correct.** The orderflow INT8 path and the f64 EMA/VWAP/Bollinger choices are the *right* calls; the team already split bounded features (INT8) from recursive/cumulative/variance paths (f64) by instinct. (`02 §7`, `04` short answer, `05 §0`.)
- **The real accuracy gate is NOT "indicator value within 1e-4." It is "the same trades fire and the P&L matches within a trading-meaningful band."** The `1e-4` value test is both too strict (most of the indicator range is decision-irrelevant) and too weak (it cannot catch a near-threshold sign flip that produces a large equity difference). The repo already owns both halves of the correct gate — `signal_boundary_mismatch_ok` for signals (`orderflow_batch.rs:1383`) and the `1e-9` backtest parity for P&L (`tick_backtest_batch.rs:15`) — they just need to be chained and promoted to the *primary* acceptance test. (`05 §4`, `06 Gap 2/4`.)
- **The unlocked headroom beyond the f64→f32 win:** windowed/element-wise indicators (the largest class) can go **f16-storage / f32-accumulate for another ~1.3–2×** (bandwidth, the regime we are in); bounded oscillators can go **INT8 for ~4× memory** (the orderflow path already banks this at <0.01% measured deviation). Tensor cores only help **GEMM-shaped** work (covariance, multi-asset projections). (`02 §3`, `04 class A/D`.)
- **Three structural hazards are where the "coarse decisions tolerate coarse math" intuition is FALSE and low precision is never safe:** (1) **recursive/IIR** smoothing (EMA, Wilder RSI/ATR/ADX, SuperTrend, PSAR) — error compounds geometrically; (2) **long cumulative sums** (OBV, VWAP, CVD, equity/PnL) — O(N) bit loss and the f32 integer-exactness cliff at 2²⁴ ≈ 16.7M, plus FP16 overflows its 65,504 ceiling outright; (3) **variance/std** (Bollinger, CCI mean-dev, z-scores) — catastrophic cancellation and FP16 overflow of `x²`. (`02 §5`, `03 §2`, `04 class B/C`.)
- **FP16 has a *range* trap, BF16 has a *precision* trap.** FP16 max is 65,504 — a sum of prices, `price·volume`, or `price²` overflows to ±inf. BF16 keeps FP32 range but only ~2–3 digits, so an absolute price at BTC scale quantizes to ~512-tick buckets. **Neither is safe on raw price magnitude.** Low precision is only safe on *bounded / normalized / differenced* quantities, and FP16 on a class-A kernel additionally requires a per-window rebasing rewrite that no kernel does today. (`01 §1`, `02 §2`, `04 finding 1`.)
- **Do two things BEFORE converting another kernel:** (Gap 3) capture one `nsys`/`ncu` end-to-end profile — the sub-2× SMA result plus host-side f64↔f32 narrowing strongly suggests PCIe/host/Python boundary cost is a material fraction of wall time, so more kernel conversions may move the needle ~5%; and (Gap 2) stand up the numerical-regression CI gate — today every f32-vs-f64 test is `#[ignore]` and CI runs GPU-less, so "within tolerance" is an unverified claim. (`06 Gap 2/3`, cross-cutting rec.)
- **Reproducibility is a separate axis from accuracy and is the thing that actually breaks backtests.** A backtest can be 0.01%-accurate and still non-reproducible because a value at `RSI = 70.0000001` flips a discrete trade. Lowering accumulation precision *widens* the run-to-run gap of any non-deterministic (float-atomic) reduction. Keep equity-bound reductions f64/Kahan/fixed-order, and keep the discrete outputs (which trades, how many, when) bit-identical. (`03 §0/§1/§6`, `06 Gap 6`.)
- **Sequence:** govern first, then convert. Add the Precision policy type + backtest-equivalence gate + variance/overflow audit (Gaps 1/2/5) and run the profile (Gap 3) *before* more kernel conversions; convert the class-A f64 stragglers + the entire `persistent/` set *in parallel* once the gate exists; defer streaming/CUDA-Graph precision interactions (Gaps 7/8) to Phase 2 but design the cache key now.

---

## 1. The core answer: are we over-specifying precision?

**Where YES (over-specified — safe to reduce):**

The *indicator → signal* leg. Every strategy collapses a continuous indicator to a 2-bit decision (Buy/Sell/Hold) via a threshold or crossover — `RSI < buy_threshold` (`momentum.rs:69`), EMA golden cross `fast > slow && prev_fast <= prev_slow` (`trend.rs:74`), orderflow `if imb > 0.6 && delta > 1000.0` (`orderflow_batch.rs:883`). Demanding `1e-4` value fidelity on a quantity about to be truncated to 2 bits is over-specification *except in the thin band around each threshold* (`05 §1`). Concretely over-specified today:

- **Class-A windowed/momentum kernels still on f64**: CCI, Bollinger (the *mean* part), Donchian, Aroon, ROC, WMA, VWMA, CMF, Ichimoku, Pivot, Fibonacci, MFI typical-price/money-flow, and the parallel pre-stages of ATR/ADX/SuperTrend. Same shape as SMA/Stochastic/Williams %R, which were already converted to f32 for the measured 1.33–1.74×. These leave the documented win on the table (`04 finding 5`).
- **The entire `persistent/` batch-kernel set** — a parallel set still 100% f64 (`persistent/ema.rs`, `rsi.rs`, `atr.rs`, `macd.rs`, `obv.rs`, etc.) (`04 finding 5`).
- **Bounded oscillator *outputs*** (RSI, Stochastic, Williams %R, MFI, CCI, ROC, Aroon final values) are bounded to [0,100]/[±100]/[±1] with integer decision thresholds — ~3-digit meaningful range against a 1-digit decision. These are INT8-eligible, exactly like the orderflow features (`01 §3.1`, `02 §5 SAFE`).

**Where NO (correctly specified — do not reduce):**

The *trade → P&L accounting* leg, plus three structural arithmetic hazards.

- **Backtest P&L / equity / Sharpe / drawdown / optimizer ranking — keep f64.** Position sizing reinvests all cash each entry (`tick_engine.rs:344`), so every trade size depends on exact prior equity; the euler/optimizer *ranks parameter sets by these scalars* (`euler_search.rs:89`), so a precision-induced bias re-orders "best parameters." Hard `1e-9` parity already exists (`tick_backtest_batch.rs:15`) and must stay (`05 §3`).
- **Recursive/IIR (Class B):** EMA, Wilder RSI/ATR/ADX, SuperTrend, PSAR, Heikin-Ashi. Error compounds with feedback; contractive EMA is bounded in f32 but f16/bf16 drift flips price-cross-EMA / MACD-sign / SuperTrend-latch signals. **f32 floor, never below.** (`02 §5 UNSAFE`, `04 class B`.)
- **Long cumulative (Class C):** OBV, VWAP `ΣTPV`, CVD, OHLCV volume atomics, equity. f32 loses integer exactness above 2²⁴ ≈ 16.7M (high-volume symbols cross this in minutes); FP16 overflows 65,504 outright. **f64 or f32+Kahan/segmented-scan accumulate.** Source already mandates f64 here (`obv_optimized.rs:41`, `vwap.rs:47`, `aggregation.cu:25`) (`03 §2`, `04 class C`).
- **Variance/std (subset of Class A that touches dispersion):** Bollinger, CCI mean-dev, z-score denominators. Two-pass `Σ(x−mean)²` cancels catastrophically in f32 at price scale; `x²` overflows FP16. **f32-multiply + f64/Welford accumulate, never f16/bf16/INT8 for the accumulator.** This is a *live hazard*: a mechanical "convert Bollinger to f32" the way SMA was converted would ship a catastrophic-cancellation bug that passes a naive value test on trending data and fails on volatile data (`05 §3`, `06 Gap 5`).
- **The mis-specified accumulator already in-tree:** `fp16_mma_ptx.cu:41` accumulates in f16 (`mma.sync...f16.f16.f16.f16`) while every other tensor-core path accumulates in f32. Flag explicitly; route numeric workloads through the f32-accumulate WMMA path (`04 finding 3`).

---

## 2. Recommended precision-tier policy

Default precision is keyed to **indicator class × consumer**, derived from the union of the class tables in `04`, `02 §6`, and `05 §5`. "Expected extra speedup" is **beyond the f64→f32 win already measured** (1.33–1.74×, memory-bound).

| Tier | Default precision | Indicator class (examples) | Rationale | Extra speedup beyond f64→f32 | Tensor cores? |
|---|---|---|---|---|---|
| **T0 — keep f64** | f64 accumulate | Backtest equity/PnL/Sharpe/drawdown/optimizer rank; VWAP `ΣTPV`, OBV, CVD running total, anchored-VWAP, OHLCV volume atomics; Heston complex transcendentals; option Greeks finite-differences | Path-dependent compounding; unbounded session-long accumulation; large+small cancellation; branch-cut/cancellation sensitivity. Money must be exact to the tick. Already f64 and tested at `1e-9` (`tick_backtest_batch.rs:15`) | **none — do not reduce** | No |
| **T1 — f32 (default)** | f32 storage, f32 acc (f64 acc for any sum > ~10⁶ terms) | EMA, Wilder RSI/ATR/ADX smoothing, MACD lines, SuperTrend, PSAR, Heikin-Ashi (Class B); SMA/WMA/VWMA, ATR/ADX/SuperTrend parallel pre-stages (Class A); MACD/Bollinger-band *differences* | f32 ε (6e-8) beats the tick floor (~1e-5 rel) by 100–1000×; recursion contractive and safe; difference operands must both be ≥f32 | **~0** (already the baseline) — this is the floor, not an upside | No (sequential/element-wise) |
| **T1.5 — f16 store / f32 accumulate** | f16 in HBM, f32 register accumulate | Windowed mean / element-wise where the operand is bounded or rebased: SMA, WMA, VWMA, ROC final values; the *bounded ratio* of Stochastic/Williams %R | Halves bytes moved → ~2× on bandwidth-bound kernels (our regime). **Requires per-window rebasing** (compute on `price − window_open`) because raw prices overflow/quantize in f16 — no kernel does this today | **~1.3–2×** (bandwidth) | No (mixed-precision the universal rule) |
| **T2 — INT8 (per-tensor scale)** | INT8 output, full-precision compute upstream | Bounded oscillator *outputs* consumed only by coarse thresholds: RSI value, Stochastic %K/%D, Williams %R, MFI, CCI, Aroon, ROC | Known tight range → range/255 quantum ~0.4% of full scale; thresholds are integers. The orderflow path proves this in production at <0.01% backtest deviation (`quantization.rs`, `estimate_error` RMSE) | **~4×** (memory, 24 B→6 B/tick proven) | No |
| **T3 — INT8 (shipping)** | INT8, fast-math compute | Orderflow 6-feature path → signal generation / ML | Decision is ±1/0/1; INT8 *is* the explicit accuracy budget, signal-parity-tested (`signal_boundary_mismatch_ok`) | already realized (19 GB → 2.4 GB) | No |
| **T-GEMM** | bf16 in / **f32 accumulate**, or TF32 | Matmul-shaped: dense covariance, multi-asset feature projection, batched feature transforms | bf16 keeps FP32 range for prices; TF32 is a drop-in cuBLAS speedup at ~3 digits. **Disable library fast-accumulate** for any reported number; use split-K/TF32x3 at large K | **~1.5–2×** (throttled GeForce Ada FP16/FP8→FP32 at half rate); INT8 GEMM up to ~4× on bounded operands | **Yes (matmul only)** |

**Mixed-precision is the universal rule, not an exception:** multiply/store in f16/bf16/INT8, **accumulate in f32 (or f64)**. This decouples storage/bandwidth precision (the 2–4× win) from accumulator precision (correctness), is strictly more accurate than homogeneous low precision, and captures nearly all the performance. The `scan.rs` `AffinePrecision::F64Acc` (`scan.rs:104`) is the existing in-tree precedent — generalize it (`02 §4`).

**Placement decision rule for a new indicator:** unbounded accumulation → T0. Recursive or differenced on absolute price → T1. Windowed-mean, bounded/rebased → T1.5. Bounded output + coarse-threshold consumer → T2. Output feeds only a discrete classifier → T3. Matmul-shaped → T-GEMM.

**Hard "never" list:** absolute price levels in f16/bf16; any difference-of-large-near-equal (MACD/Bollinger) with both operands low-precision; cumulative sums in f16/bf16; PnL/equity below f64; variance accumulators below f32; `Auto` must never pick FP16 for a raw-price kernel.

---

## 2b. The "accuracy limiter" design

Replace the arbitrary `1e-4` constant with a **per-class tolerance derived from decision resolution**, and make the real gate **backtest trade/P&L equivalence**, not indicator-value match.

**(a) Configurable precision policy (Gap 1).** Define one crate-level type — `enum Precision { F64, F32, F16, Auto }` plus a `PrecisionPolicy` whose default per indicator is the tier table above. Thread it as an optional last argument (defaulting to the per-indicator safe choice) into the `*_gpu` public fns, and collapse the `sma_gpu` / `sma_gpu_f32` / `sma_gpu_f64` triplet (`sma.rs:264/304/161`) into one dispatcher. `Auto` selects from data range/call-site and is **forbidden from choosing FP16 on raw-price kernels** (`06 Gap 1/5`). Generalize the existing `AffinePrecision` rather than inventing a parallel concept.

**(b) Per-class tolerance from decision resolution (the "epsilon-of-meaning").** For each indicator-consumer pair: (1) identify the decision resolution Δ_d — for a threshold it is the threshold's least-significant digit or the empirical min gap near it; for a crossover it is the typical |gap| of the two lines at the cross; for a backtest it is one tick. (2) Set value tolerance ε_target = Δ_d / 10. (3) Choose the smallest format whose ε, *after propagation through the indicator's error-growth law* (×√n windowed, ×1/α recursive, ×n cumulative, ÷cancellation-margin for differences), stays below ε_target. RSI ε_target ≈ 0.1 RSI points → f16 clears it; absolute price Δ_d = 1 tick → f32 clears, f16 fails (`01 §5`).

**(c) The real gate — backtest equivalence (primary), value tolerance (secondary).** A precision change is acceptable iff it does not change the trades you would have taken or the money you would have made, measured end-to-end:

> **Precision-Reduction Acceptance Gate**
> 1. **Trade-set equivalence (primary):** run the candidate low-precision signals through the *unchanged f64 backtest engine*. Require identical trade **count** (exact), identical entry/exit **timestamps** (exact), and ≥99.x% identical per-tick signal decisions with every disagreement inside the threshold drift band (`signal_boundary_mismatch_ok`, `orderflow_batch.rs:1383`).
> 2. **P&L equivalence (primary):** final equity, Sharpe, max-drawdown, **and the optimizer-selected argmax parameter set** match the f64 baseline within a trading-meaningful band (a few bps on equity; argmax unchanged), with the existing `1e-9` engine parity as the inner contract.
> 3. **Value tolerance (secondary/diagnostic only):** keep `rel < 1e-4` (`sma.rs:863`) as a fast smoke test that catches gross kernel bugs — never the final word.

This is just *chaining* two tests the repo already has (`test_gpu_cpu_parity_random_signals` + the orderflow signal-parity test) into: low-precision kernel → signals → f64 backtest → compare equity/trade-set to the f64-indicator baseline (`05 §4`, `06 Gap 4`).

**(d) Determinism contract (separate axis).** Per reduction, declare bit-identical vs tolerance-band. Discrete outputs (trade entries/exits, count, timestamps, high/low) must be **bit-identical** run-to-run (fixed-tree reduction, no float atomics, `-prec-div=true` on any division feeding a threshold). Continuous P&L/volume/CVD ride a `1e-6`–`1e-9` relative band on f64+Kahan/pairwise. Add a "run twice, assert bit-identical trade set" smoke test. Note the live build finding: `build.rs` compiles *everything* with `-use_fast_math` (`build.rs:258`); split NVCC so signal+backtest kernels compile with `-ftz=false -prec-div=true -prec-sqrt=true` (≈0 cost on add/mul/fma) so GPU division/sqrt match the CPU reference (`03 §3/§6`).

---

## 3. Performance implication: where the headroom is, and where it is not

**Additional headroom beyond f64→f32 (1.33–1.74× already measured):**

- **Windowed/element-wise class → f16-store/f32-accumulate: ~1.3–2× more (bandwidth).** This is the broadest population (SMA-family + bounded oscillators) and the regime we actually live in — the FP32 SMA result scaling toward the 2× bandwidth ceiling *is* the evidence it is memory-bound. Conditional on the per-window rebasing rewrite (`02 §3`, `04 finding 1`).
- **Bounded oscillators → INT8: ~4× memory.** Proven on the orderflow path (24 B → 6 B/tick, 19 GB → 2.4 GB for 10 strategies, <0.01% deviation). Extend the *same* pattern to RSI/Stoch/Williams/MFI/CCI/ROC/Aroon outputs (`02 §3/§5`, `quantization.rs`).
- **GEMM-shaped → tensor cores: ~1.5–2× (bf16/f32-acc, throttled on GeForce Ada), up to ~4× INT8.** Only applies if work is reshaped into matmuls (covariance, multi-asset projections, batched MC). **Does nothing for per-element SMA/EMA scans** (`02 §3`). Note the GeForce/laptop-Ada half-rate throttle on FP16/FP8→FP32 accumulate.
- **Secondary:** halving operand width halves register/shared-mem pressure → higher occupancy → additional throughput on bandwidth-bound kernels.

**Where the headroom is NOT available:**

- **Recursive/IIR (Class B):** sequential, loop-carried — no SIMD, no tensor cores, and reducing below f32 corrupts the signal. **Speedup from precision here is zero.** (EMA/MACD already correctly live on CPU because single-thread GPU was 6–1,647× slower.)
- **Long cumulative (Class C):** the accumulator must stay wide (f64 or f32+Kahan); you may quantize the *output* but not the running total. No precision speedup on the accumulator (`02 §6`, `04 class C`).
- **Variance/std:** accumulator stays f32/f64 (Welford). No low-precision speedup on the dispersion accumulator (`02 §5`).
- **The boundary caveat (Gap 3):** the sub-2× SMA result + host-side f64↔f32 narrowing (`sma.rs:335/360`) + still-f64 Python boundary (`*_py.rs`) imply PCIe H2D/D2H + host conversion may be a material fraction of wall time. **If the kernel is <~50% of wall time, more kernel conversions yield ~5%** — the lever is f32-native transfer + zero-copy Python ingest, not the next kernel (`06 Gap 3`).

---

## 4. Prioritized gap list (from `06`)

| # | Gap | Severity | First step |
|---|---|---|---|
| **3** | No end-to-end profile proving the kernel (not PCIe/host/Python) is the bottleneck | **HIGH** | One `nsys` timeline + one `ncu` of the SMA f32 kernel; report H2D/kernel/D2H/host-conversion split via `MultiPhaseTimer` (`timing.rs:254`). Rule: kernel <50% wall → prioritize transfer/zero-copy over conversions |
| **2** | Numerical-regression CI gate absent (f32 tests are `#[ignore]`, CI is GPU-less) | **HIGH** | Add a required `cuda-numerical-gate` job on the existing self-hosted GPU runner: `cargo test --features gpu -- --ignored` filtered to a `precision_gate` group + a backtest `1e-9` parity job; required on PRs touching `rust/src/gpu/**` |
| **5** | No f32 overflow/cancellation/NaN audit (Bollinger two-pass variance is a live hazard; FP16 price overflow) | **HIGH** | Switch Bollinger/CCI variance to Welford or f64-in-register accumulate (`scan.rs` `F64Acc` precedent); write an overflow/NaN audit test on high-magnitude high-variance data; codify "FP16 only on pre-normalized inputs" in the policy |
| **1** | No configurable `Precision` policy threaded through calls (only local `AffinePrecision`) | **HIGH** | Crate-level `Precision{F64,F32,F16,Auto}` + `PrecisionPolicy`; collapse the `sma_gpu`/`_f32`/`_f64` fork into one dispatcher |
| **4** | Crossover/sign indicators: unhandled precision-induced tie-band flips | **MED-HIGH** | Decision-equivalence test (signal-index set identical f32-vs-f64 on near-tie data), distinct from value-tolerance; deadband/hysteresis option; route precision-critical strategies to f64 via the Gap-1 policy |
| **6** | Reduced-precision reductions threaten backtest reproducibility | **MED-HIGH** | Keep f64/Kahan/fixed-order for equity-bound reductions; add "run twice, assert bit-identical trade set" test |
| **7** | No streaming/online indicators; long-lived f32 state would drift | **MED** | Policy: streaming carries f64 state, emits f32; periodic re-seed of contractive filters; 1M-tick drift test |
| **8** | Precision × CUDA-Graph/multi-stream interaction unspecified | **MED** | Make `Precision` part of the graph/stream cache key `(indicator_set, shape, Precision)`; fix precision per captured graph; narrow host-side *before* capture |
| **9** | "Add an indicator" has no precision-aware scaffold | **MED** | `gpu_indicator!` macro / `GpuIndicator` trait that auto-wires precision-dispatched entry + host conversion + a registered precision gate by class |
| **10** | No documented accuracy contract for callers | **MED** | `docs/PRECISION_CONTRACT.md` per-indicator: precision used, guaranteed max rel error, NaN/overflow behavior, reproducibility guarantee; generate error numbers from the Gap-2 gate |

---

## 5. How this updates the phased plan

**BEFORE converting any more kernels (governance + de-risking — do these first):**

1. **Gap 3 — profile.** Capture the one `nsys`/`ncu` end-to-end split. This decides whether kernel conversion is even the right lever; it may redirect the whole effort to the transfer/Python boundary. Cheapest, highest-information action; do it first.
2. **Gap 2 — CI numerical gate.** Stand up the GPU-runner precision-gate + backtest-equivalence job and make it required on `rust/src/gpu/**`. This is what makes every "f32 is fine" claim trustworthy; without it, the next conversion can silently ship a wrong-but-fast kernel.
3. **Gap 1 — Precision policy type.** Land the crate-level `Precision`/`PrecisionPolicy` and collapse the SMA function fork. Everything else assumes it exists.
4. **Gap 2b/5 prerequisite — the backtest-equivalence gate + variance/overflow audit helper.** Promote trade-set + P&L equivalence to the primary acceptance test; add `assert_within_precision_tolerance(f64_ref, out, class)` and the overflow/NaN audit so Bollinger-class conversions can't ship a cancellation bug.

**IN PARALLEL once the gate + policy exist (the actual conversions, now safe-by-construction):**

5. Convert the **class-A f64 stragglers** (CCI mean part, Donchian, Aroon, ROC, WMA, VWMA, CMF, Ichimoku, Pivot, Fibonacci, MFI TP/money-flow, ATR/ADX/SuperTrend parallel pre-stages) and the **entire `persistent/` set** from f64→f32 — each inheriting the Gap-2 gate. Low-risk, high-yield.
6. Prototype **T1.5 (f16-store/f32-acc with rebasing)** on SMA-family and **T2 (INT8)** on the bounded oscillator outputs, each validated through the backtest-equivalence gate — the ~1.3–2× / ~4× memory upside.
7. Fix the **f16-accumulate mis-spec** (`fp16_mma_ptx.cu:41`); route numeric GEMM through the f32-accumulate WMMA path.

**DEFER to Phase 2 but design the key now:**

8. Streaming/online indicators (Gap 7) and CUDA-Graph/multi-stream precision interaction (Gap 8): keep `Precision` in the cache key and specify f64-streaming-state-now so the retrofit is cheap. Variance/Welford rewrite (Gap 5) and the `PRECISION_CONTRACT.md` (Gap 10) close out the durable program.

**One-line rule for the whole program:** the decision is coarse, the arithmetic that produces it is not — reduce precision freely *upstream of the discretizing decision* on bounded/windowed/normalized quantities, never *at or downstream of* it (recursion, cumulation, variance, P&L), and gate every change on **same trades + same money**, not on indicator-value match.

---

## Source map

- **`01-accuracy-budget.md`** — tick-size floor (~1e-5 rel), epsilon-of-meaning, per-class tolerance recipe, value-vs-decision distinction.
- **`02-low-precision-tradeoffs.md`** — five-format range/precision table, FP16 range trap / BF16 precision trap, mixed-precision rule, Ada throughput, SAFE/UNSAFE indicator map, decision table.
- **`03-determinism-reproducibility.md`** — non-associativity, float-atomic non-determinism, f32 2²⁴ cliff, `-use_fast_math`/FTZ finding, NaN/Inf propagation, two-tier reproducibility contract.
- **`04-indicator-precision-classes.md`** — per-kernel class A/B/C/D table with file:line, the f16-rebasing caveat, the `fp16_mma_ptx.cu:41` mis-spec, the five cross-cutting findings.
- **`05-decision-tolerance.md`** — strategies collapse to 2-bit decisions, orderflow INT8 as the proof point, P&L leg unsafe, the Precision-Reduction Acceptance Gate, per-stage recommendation.
- **`06-gap-analysis.md`** — the 10 gaps with severities and first steps; cross-cutting "profile + CI gate first."
