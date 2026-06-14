# GPU Precision Optimization — Gap Analysis

**Date:** 2026-06-14
**Target HW:** NVIDIA RTX 3500 Ada (AD104, sm_89), CUDA 13.1, 12 GB GDDR6 @ ~432 GB/s.
**Scope:** Read-only audit of `rust/src/gpu` (~52K LOC) against the precision-reduction effort (FP64→FP32 underway; INT8 already in the orderflow path). This document enumerates what the effort **needs but has not yet planned** — the gaps between "convert kernels to f32" and a *safe, shippable, maintainable* precision program.

**Framing the user's question.** Trading decisions are coarse: signals are thresholds and crossovers (`RSI>70`, price-crosses-SMA, `MACD` sign-flip), backtests are discrete trades. So for **most indicators feeding signal generation**, f32 (and in places INT8) is not just acceptable — it is the correct engineering choice, and the codebase already proves it knows this (`tick_aggregation.rs:25-32`, `ma_advanced.rs:34-41`, `scan.rs:48-55`, the whole `quantization.rs` INT8 path). The danger is **not** "lower precision is wrong"; it is that the effort is being executed **kernel-by-kernel as an ad-hoc rewrite** (see `sma.rs:264-361`) with no policy, no enforced accuracy contract, no CI gate, and no clear answer for the handful of places where reduced precision *is* unsafe. The gaps below are about turning a sequence of point conversions into a governed program.

---

## Where reduced precision is SAFE vs NOT SAFE (the decision the gaps must encode)

This is the spine of every gap that follows. The dividing line is **upstream vs at-or-downstream of the discretizing decision** (corroborates `04-tensor-cores-low-precision.md` §5).

| Class | Examples (file evidence) | Safe precision | Why |
|---|---|---|---|
| **Windowed sums / means** | SMA (`sma.rs`), VWMA, WMA, CMF, TP-based CCI/MFI | **f32** (in-register), INT8 *output* if it only feeds a threshold | Small windows of price-scale values; error ≪ tick size; signal is a comparison |
| **Thresholded oscillators** | RSI (`rsi.rs`), Stochastic (`stochastic.rs`), Williams %R, ROC, Aroon | **f32** | Output compared to fixed bands (70/30, 80/20); f32 has ~7 digits, bands need ~2 |
| **Crossover / sign indicators** | MACD (`macd.rs`), SMA/EMA cross, Supertrend flip | **f32** | Decision is `sign(a-b)`; only fails in a measure-zero tie band (see Gap 4) |
| **Pre-discretized features** | orderflow 6 features (`orderflow_batch.rs`, `quantization.rs`) | **INT8 (already)** | Quantization error ≪ decision granularity; mature, calibrated, RMSE-checked |
| **Variance / dispersion** | Bollinger (`bollinger.rs:56-70`), CCI mean-dev, std-dev channels | **f32 multiply, f64-or-Welford accumulate** | Two-pass `Σ(x-mean)²` cancels catastrophically in f32 at price scale (see Gap 5) |
| **Recursive IIR over long series** | EMA (`ema.rs:139`), Wilder smoothing (ATR/ADX/RSI), CVD running sum (`orderflow_batch.rs`) | **f32 OK short; f64/Kahan accumulate for long or for reported P&L** | Loop-carried error accumulates; bounded for contractive EMA, unbounded for running sums |
| **Reported dollar values** | backtest equity/P&L/Sharpe/drawdown (`tick_backtest_batch.rs:104-149`), Greeks, VaR | **f64 — do NOT reduce** | These are the numbers published; 1e-9 CPU-parity contract already exists (`:15`); compliance/correctness, not a tuning knob |

The conversion effort today (`sma.rs`) encodes this line **only in a code comment and a single `#[ignore]` test**. Everything below makes it a first-class, enforced contract.

---

## Gap 1 — No configurable Precision policy/API threaded through calls

**State of codebase.** There is **no project-wide `Precision` type**. The only precision selector that exists is a *local* enum buried in one file: `AffinePrecision { F32, F64Acc }` in `scan.rs:104-111`, threaded only through the scan's own entry points (`scan.rs:418-440`). The SMA conversion instead hard-codes the policy by **forking the function**: `sma_gpu` now unconditionally calls `sma_gpu_f32` (`sma.rs:264-271`), with `sma_gpu_f64` kept as a separate reference (`sma.rs:161`). Multiply this pattern across 27 f64 kernels and you get a combinatorial mess of `_f32`/`_f64` twins with no uniform way for a caller to request precision.

**Why it matters.** (a) A backtest that needs reproducible f64 equity and a live signal loop that wants max-throughput f32 must call *different function names*, not pass a parameter — so precision becomes a compile-time fork, not a runtime policy. (b) Without a single policy object there is nowhere to express `auto` (pick precision from data range / call site), which the hardware practically demands (BF16 vs FP16 overflow rules — `04` §2). (c) The Python boundary (`PyReadonlyArray2<f64>` in `batch_backtest_py.rs:273`, all `*_py.rs` are f64) has no way to surface the choice to users.

**Severity: HIGH** (architectural; every subsequent gap assumes it exists).

**First step.** Define one crate-level `enum Precision { F64, F32, F16, Auto }` plus a `PrecisionPolicy` (default per indicator class from the SAFE/NOT-SAFE table). Add it as an **optional last argument** (defaulting to the per-indicator safe choice) to the `*_gpu` public fns, and collapse the `sma_gpu`/`sma_gpu_f32`/`sma_gpu_f64` triplet into one dispatcher behind it. Generalize the existing `AffinePrecision` rather than inventing a parallel concept.

---

## Gap 2 — Numerical-regression CI gate is effectively absent (the f32 gates never run)

**State of codebase.** A golden f32-vs-f64 test **already exists and is well-written**: `test_sma_f32_matches_f64` (`sma.rs:847-873`) asserts `rel < 1e-4` on realistic price-scale data across periods. A 1e-9 GPU-vs-CPU backtest-equivalence test exists (`tick_backtest_batch.rs:15`). **But both are `#[ignore]` "Requires GPU"** — and the **`test.yml` CI runs on `ubuntu-latest` with no GPU** (`.github/workflows/test.yml:20`), building only the **CPU feature** (`maturin build` without `--features gpu`, `:48-51`). So every numerical accuracy test in the GPU tier is skipped in CI. Across `rust/src/gpu/*.rs` there are **51 files** carrying `#[ignore] // Requires GPU` and only **15** f32-vs-f64 / parity test fns total — none CI-enforced.

The one GPU CI workflow (`cuda-benchmark.yml`) is **performance-only**: it fails on `>10% regression` (`:18`) but asserts nothing about *numerical drift*. You can silently make an indicator wrong as long as it stays fast.

**Why it matters.** The entire premise of this effort is "lower precision is acceptable *within tolerance*." Without an enforced tolerance gate, "within tolerance" is an unverified claim that rots the moment someone converts the next kernel. This is the single highest-leverage missing safety net: a converted Bollinger (Gap 5) would pass perf CI and ship a catastrophic-cancellation bug.

**Severity: HIGH** (it is the mechanism that makes every other "f32 is fine" claim trustworthy).

**First step.** Add a `cuda-numerical-gate` job to the existing self-hosted GPU runner (the `cuda-benchmark.yml` runner already exists). Build with `--features gpu`, run `cargo test --features gpu -- --ignored` filtered to a `precision_gate` test group, and (separately) a backtest-equivalence job asserting the existing 1e-9 parity. Make it **required** on PRs touching `rust/src/gpu/**`. Standardize a `assert_within_precision_tolerance(f64_ref, f32_out, class)` helper so each new indicator inherits the gate (ties into Gap 9).

---

## Gap 3 — No end-to-end profiling proving the GPU kernel (not PCIe/host/Python) is the bottleneck

**State of codebase.** A `MultiPhaseTimer` exists that *can* separate H2D / kernel / D2H (`timing.rs:254-282`, `TimingBreakdown`), but the conversion is being justified by **kernel-vs-kernel microbenchmarks only** (`bench_sma_f64_vs_f32`, `sma.rs:877-907` times the whole `sma_gpu_*` call, not the kernel in isolation). No `nsys`/`ncu` capture exists in the repo's scripts or CI (grep for `nsys`/`ncu` hits only prose `.md` files, never a script or workflow). Critically, the f32 SMA path **narrows on the host into a fresh `Vec<f32>`** (`sma.rs:335`) and **widens back to f64 on return** (`sma.rs:360`), and the public API + Python boundary are still f64 (`*_py.rs`). So part of the "f32 win" is consumed by host-side conversion and the public side still pays f64 traffic up to the Rust boundary.

**Why it matters.** The user's own measurement (1.33–1.74x, "scales toward 2x, memory-bound") *is* the tell: if the kernel were the bottleneck, halving its ALU/DRAM would approach 2x cleanly. The sub-2x and the host-side f64↔f32 conversions strongly suggest **PCIe H2D/D2H and host conversion are a material fraction of wall time** for one-shot calls. If so, converting more kernels to f32 yields diminishing returns until the *transfer* is also f32 end-to-end and the Python array is handed over without a widening copy. You can spend weeks converting kernels and move the needle 5% because the real cost is the boundary.

**Severity: HIGH** (risks misallocating the whole effort).

**First step.** Before converting more kernels, capture **one** `nsys` timeline of a representative end-to-end indicator call (Python → Rust → GPU → Python) and one `ncu` of the SMA f32 kernel, and report the H2D/kernel/D2H/host-conversion split using the existing `MultiPhaseTimer`. Decision rule: if kernel < ~50% of wall time, prioritize **f32-native transfer + zero-copy Python ingest** over further kernel conversions.

---

## Gap 4 — Crossover / sign indicators have an unhandled tie-band under reduced precision

**State of codebase.** MACD (`macd.rs`), SMA/EMA crossovers, and Supertrend flips reduce to `sign(a - b)`. The repo already learned the *adjacent* lesson — block-boundary nondeterminism flipped `%K/%D` and CCI values (`stochastic.rs:379`, `cci.rs:319`) — but there is **no concept of a precision-induced decision flip**: when `a` and `b` are within ~1 f32 ULP at price scale (~0.004 at BTC 30k), f32 and f64 can disagree on the sign, and that flip is a *different trade*, not a rounded number.

**Why it matters.** This is the one place the "coarse decisions tolerate coarse math" intuition **breaks**: coarseness helps when the decision boundary is far from the value (RSI=45 vs band 70), but a crossover indicator's entire job is to sit *exactly on* the boundary. Near the cross, reduced precision doesn't blur a number, it toggles a discrete trade — and a backtest is the integral of those toggles. The existing f32-vs-f64 test (`sma.rs:858-870`) measures *relative value error*, which **cannot catch a sign flip** that produces a large equity difference from a tiny value difference.

**Severity: MEDIUM-HIGH** (silent, and exactly where the user's intuition is most likely to mislead).

**First step.** Add a **decision-equivalence** test (distinct from value-tolerance): for crossover/sign indicators, assert the *set of signal indices* (and their signs) is identical f32-vs-f64 on noisy near-tie data, OR that any disagreement count × position size stays under a P&L budget. Where flips are unavoidable, document a deadband / hysteresis option and route precision-critical strategies to f64 via the Gap-1 policy.

---

## Gap 5 — No f32 overflow / cancellation / NaN audit (Bollinger is a live hazard)

**State of codebase.** Bollinger computes variance as a **two-pass `Σ(close-mean)²`** in `bollinger.rs:56-70` — and it is still f64 today. A mechanical "convert to f32" (the pattern applied to SMA) would compute `diff*diff` at price scale in f32, where `close≈30000` and `diff≈5` means `diff²≈25` is summed into a register holding ~`O(period·30000²)` of catastrophic-cancellation risk: the f32 std-dev can lose most of its significant digits. The synthesis doc flags this in passing (`00-SYNTHESIS.md` opp #2 "variance cancellation in Bollinger/CCI") but there is **no audit pass** and **no overflow/NaN policy**. Separately: FP16 (a stated `f16` candidate) **overflows on raw prices** — `30000² = 9e8 < 65504²` is fine but un-normalized `Σx²` and CVD running sums (`orderflow_batch.rs`) exceed FP16's 65504 ceiling instantly (`04` §2). The codebase has NaN-injection tests for *inputs* (`sma.rs:553-572`) but no test that reduced precision doesn't *manufacture* Inf/NaN via overflow.

**Why it matters.** This is the difference between "lower precision is a bit noisier" and "lower precision silently returns `inf`/`NaN` or a 50%-wrong std-dev that widens Bollinger bands and suppresses every breakout signal." It is a correctness bug masquerading as a precision tuning, and it will pass a naive value-tolerance test on *trending* data and fail only on the volatile data that matters.

**Severity: HIGH** for Bollinger/CCI/variance + any FP16 ambition; LOW for plain windowed sums.

**First step.** Before converting any variance/dispersion or accumulating kernel: (a) switch Bollinger/CCI to **Welford** or **f64-in-register accumulation with f32 storage** (the `scan.rs` `F64Acc` precedent, `:113-120`); (b) write an overflow/NaN audit test that feeds high-magnitude + high-variance data and asserts no `inf`/`NaN` and bounded relative error; (c) codify the rule "FP16 only on pre-normalized/centered inputs, BF16 for anything carrying price magnitude" (`04` §2) in the Gap-1 policy so `Auto` never picks FP16 for a raw-price kernel.

---

## Gap 6 — Deterministic reductions for reproducible backtests are not guaranteed under the new precision

**State of codebase.** The repo has **repeatedly** fought reduction nondeterminism: `atomicAdd` ordering differs from CPU summation so volume tolerances had to be loosened (`aggregation.rs:606`, `:608`); float `atomicAdd` is used for volume in `tick_aggregation.rs:20`; cross-block races produced nondeterministic CCI/`%K`/`%D` until fixed (`cci.rs:319`, `stochastic.rs:379-380`, `:519-540`). The backtest claims a **1e-9 CPU-parity** contract (`tick_backtest_batch.rs:15`). **The risk the precision effort introduces:** lowering accumulation precision *widens* the gap between any two non-deterministic summation orders — f32 `atomicAdd` of N volumes in hardware-arbitrary order can differ run-to-run by far more than f64, and a backtest that integrates those values becomes **non-reproducible across runs of the same inputs**. There is no plan to keep reductions deterministic as precision drops.

**Why it matters.** A backtest you cannot reproduce bit-for-bit is unusable for research (you can't tell a real edge from reduction noise), for debugging (can't bisect), or for compliance. The whole value of a backtest is that the same strategy on the same data gives the same equity — reduced-precision unordered atomics quietly breaks that.

**Severity: MEDIUM-HIGH** (it is the failure mode the codebase has already been bitten by, now amplified).

**First step.** Establish a **determinism contract per reduction**: either (a) keep f64 accumulation for any value that flows into reported equity/P&L (cheap — reductions are bandwidth-bound, `scan.rs:113-118` already argues this), or (b) use a fixed-order / tree / Kahan reduction so f32 results are run-invariant. Add a "run twice, assert bit-identical" test for the backtest and aggregation paths alongside the existing CPU-parity test.

---

## Gap 7 — Streaming / online incremental indicator updates are absent (and precision interacts badly)

**State of codebase.** Every GPU indicator is **batch-only**: it uploads the full array, computes, downloads (`sma.rs:204-255` is the template). There is **no incremental/online update path** — grep for `streaming`/`incremental`/`online`/`push_candle` finds only L2-cache `AccessProperty::Streaming` (`l2_cache.rs:56`), unrelated. For a live per-bar loop (which the synthesis doc calls out as a prime CUDA-Graph target, `00-SYNTHESIS.md` §4) you currently recompute the whole indicator every new bar.

**Why it matters here specifically (the precision angle).** The naive "streaming EMA/Wilder" is `state = α·x + (1-α)·state` carried forever (`ema.rs:139`). In **f32**, a state carried across millions of live ticks accumulates rounding error without bound for non-contractive accumulators (CVD running sum, cumulative volume) and slowly for contractive ones. A batch recompute is self-correcting (fixed window); a long-lived f32 streaming state is **not**. So adding streaming and adding reduced precision *at the same time* compounds two error sources the current batch design hides. The effort needs to decide this before, not after, someone ships an f32 streaming EMA that drifts over a trading day.

**Severity: MEDIUM** (feature gap today, but a correctness trap the moment streaming + f32 meet).

**First step.** Specify streaming state precision in the Gap-1 policy: **streaming accumulators carry f64 state, emit f32** (state is O(1), so f64 is free); periodically re-seed contractive filters from a batch recompute. Add a drift test: f32-streaming-over-1M-ticks vs f64-batch must stay within tolerance.

---

## Gap 8 — Multi-stream / CUDA-Graph interaction with precision is unspecified

**State of codebase.** CUDA Graphs are disabled and multi-stream dispatch is gated off (`00-SYNTHESIS.md` §1: `batch_graphs.rs` always errors, `batch.rs:65` gated). The synthesis plan (Phase 2) revives both. **The unconsidered interaction:** a CUDA Graph captures a **fixed kernel sequence with bound buffers and bound dtypes**. If precision is a *runtime* policy (Gap 1), then F64 and F32 variants are **different graphs** — you cannot flip precision inside a captured graph; you need one captured graph per precision per shape. Likewise, mixing f32 and f64 kernels across concurrent streams means the per-stream pinned buffers and device allocations differ in width, complicating the event-gated pinned-release the synthesis doc proposes (`00-SYNTHESIS.md` opp #12).

**Why it matters.** If the precision policy and the graph/stream work are designed independently, they collide: either precision silently becomes fixed-at-capture (surprising the caller) or graph caching explodes (one graph per precision×shape×indicator-set). Better to decide the cache key now.

**Severity: MEDIUM** (only bites in Phase 2, but cheap to design for now, expensive to retrofit).

**First step.** Make **precision part of the graph/stream cache key** in the design (graph keyed by `(indicator_set, shape, Precision)`), and document that precision is fixed per captured graph. Verify the f32-narrowing conversion happens *before* capture (host-side), not inside the captured region.

---

## Gap 9 — "Add a new indicator" has no precision-aware on-ramp (the effort doesn't scale)

**State of codebase.** Adding an indicator today means: write a `const KERNEL: &str` (`sma.rs:29`), a `pub fn x_gpu`, `pub mod`/`pub use` in `mod.rs` (e.g. `:142-154`), and `#[ignore]` GPU tests. There is **no template, macro, or trait** that bakes in the precision policy — so the SMA author had to hand-roll the f32 kernel, the host narrowing/widening, *and* the f32-vs-f64 test (`sma.rs:281-361, 847-873`). The synthesis doc explicitly notes this is an **add-indicators-frequently research library** (`00-SYNTHESIS.md` §3.4) and uses that as the core argument against a megakernel. The same property makes an *unautomated* precision policy unsustainable: every new indicator is a chance to forget the tolerance test, pick the wrong accumulation width, or re-introduce a variance-cancellation bug.

**Why it matters.** A precision policy that lives in code comments and per-file discipline (today's state) decays. With ~17 indicators and growth, "remembered to add the f32-vs-f64 gate" must be structural, not cultural. This is what converts the effort from a one-time conversion into a durable property.

**Severity: MEDIUM** (compounding maintenance / silent-regression risk).

**First step.** Provide a thin **indicator scaffold**: a `gpu_indicator!` macro or a `GpuIndicator` trait that takes the kernel source(s) and an indicator *class* (from the SAFE/NOT-SAFE table), and auto-generates (a) the precision-dispatched entry point, (b) the host conversion, and (c) a `#[test]`-registered precision gate using the shared `assert_within_precision_tolerance` helper from Gap 2. New indicator = pick a class + write the kernel; the contract comes for free.

---

## Gap 10 — No documented accuracy contract (what callers can rely on)

**State of codebase.** The only place an accuracy promise is written down is the **quantization** path — and it is exemplary: per-feature calibration, documented quantize/dequantize formulas, and an `estimate_error`/RMSE method (`quantization.rs:7, 23-31, 66`). The indicators have **no equivalent**: the SMA contract is a sentence in a doc-comment ("(lenient, price-scale) tolerance is unaffected", `sma.rs:277-279`) and one `1e-4` test constant (`sma.rs:863`). Nowhere does the project state, per indicator, *what relative/absolute error a caller may see*, *which precision is used*, or *when results are bit-reproducible*.

**Why it matters.** Downstream consumers (strategy authors, the backtester, Python users via `*_py.rs`) make decisions assuming a precision they were never told. A quant comparing two strategies needs to know whether a 0.01% equity difference is signal or f32 noise. Without a published contract, the precision effort is changing the numerical meaning of every indicator silently — the worst kind of breaking change because nothing visibly breaks.

**Severity: MEDIUM** (trust/usability; also the artifact that makes Gaps 1–9 legible to users).

**First step.** Write one `docs/PRECISION_CONTRACT.md` table: per indicator (or class) → precision used, guaranteed max relative error, NaN/overflow behavior, and reproducibility guarantee — modeled on the quantization module's existing rigor. Generate the error numbers from the Gap-2 CI gate so the doc can't drift from reality.

---

## Summary of gaps (severity-ranked)

| # | Gap | Severity | One-line first step |
|---|---|---|---|
| 2 | Numerical-regression CI gate absent (f32 tests are `#[ignore]`, GPU-less CI) | **HIGH** | Add required `cargo test --features gpu --ignored` precision-gate job on the GPU runner |
| 3 | No end-to-end profile proving the kernel (not PCIe/host/Python) is the bottleneck | **HIGH** | `nsys`/`ncu` one representative call; split H2D/kernel/D2H/host-conversion before converting more |
| 5 | No f32 overflow/cancellation/NaN audit (Bollinger two-pass variance, FP16 price overflow) | **HIGH** | Welford/f64-accum for variance + overflow/NaN audit test before any variance kernel conversion |
| 1 | No configurable `Precision` policy threaded through calls (only local `AffinePrecision`) | **HIGH** | Crate-level `Precision{F64,F32,F16,Auto}` arg; collapse the `_f32`/`_f64` function forks |
| 4 | Crossover/sign indicators: unhandled precision-induced tie-band flips | **MED-HIGH** | Decision-equivalence test (signal-index set), not just value-tolerance; deadband option |
| 6 | Reduced-precision reductions threaten backtest reproducibility | **MED-HIGH** | Keep f64/Kahan/fixed-order for equity-bound reductions; "run twice, bit-identical" test |
| 7 | No streaming/online indicators; f32 long-lived state would drift | **MED** | Policy: streaming carries f64 state, emits f32; long-run drift test |
| 8 | Precision × CUDA-Graph/multi-stream interaction unspecified | **MED** | Make `Precision` part of the graph/stream cache key; fix precision per captured graph |
| 9 | "Add an indicator" has no precision-aware scaffold; policy won't scale | **MED** | `gpu_indicator!` macro/trait that auto-wires dispatch + precision gate by indicator class |
| 10 | No documented accuracy contract for callers | **MED** | `docs/PRECISION_CONTRACT.md` per-indicator error/precision/reproducibility table |

**Cross-cutting recommendation.** Do **Gap 3 (profile)** and **Gap 2 (CI gate)** *first*, before converting another kernel: the profile tells you whether kernel conversion is even the right lever (Gap 3 evidence suggests the boundary may dominate), and the gate makes every subsequent conversion safe-by-construction. Gaps 1/9/10 then make the program governed and durable; Gaps 4/5/6/7/8 are the specific places the user's "coarse decisions tolerate coarse math" intuition is *false* and must be guarded.
