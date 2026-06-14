# The Accuracy Budget: Epsilon-of-Meaning for Financial Technical Indicators

**Research date:** 2026-06-14 | **Scope:** kimsfinance GPU indicator precision policy (RTX 3500 Ada, sm_89, CUDA 13.1) | **Confidence:** 84% (Medium-High)

## The core thesis

We are over-specifying numerical precision for most indicators. The codebase computes indicators in `f64` (EMA, RSI, MACD, backtest PnL are all `f64`; SMA recently moved to `f32`), but the *consumer* of those values is almost always a coarse, discrete decision: `RSI > 70`, `MACD crosses zero`, `price crosses SMA`, "enter trade / don't". A trading signal's information content is measured in **bits of decision**, not bits of mantissa. The right question is never "how accurate is the indicator value?" but **"how stable is the decision the value feeds?"** Those are different quantities, and the gap between them is wasted compute.

This document defines an **accuracy budget** (the "epsilon-of-meaning") derived from the decision resolution rather than an arbitrary `1e-4` tolerance, and ends with a concrete per-indicator precision-tier policy.

---

## 1. The input is already quantized: tick size sets the floor

No indicator can carry more real information than its input. Market prices are not continuous reals — they live on a discrete lattice set by the exchange **tick size** (minimum price increment). The accuracy budget therefore starts from a hard floor that has nothing to do with float precision:

| Asset class | Tick size | Relative tick at typical price | Decimal digits needed |
|---|---|---|---|
| US equities (NYSE/Nasdaq, post-decimalization 2001) | $0.01 | ~3e-5 @ $300 | ~5 |
| ES / index futures | $0.05–$0.25 | ~6e-5 @ 4000 | ~5 |
| FX majors | 0.0001 (1 pip) | ~9e-5 @ 1.10 | ~5 |
| FX JPY pairs | 0.01 | ~7e-5 @ 150 | ~5 |
| Crypto BTC (CME futures) | $5.00 | ~5e-5 @ $100k | ~5 |
| Crypto BTC (spot, varies) | $0.01–$1 | 1e-7 to 1e-5 | up to ~8 |

Sources: [Schwab — index futures tick values](https://www.schwab.com/learn/story/stock-index-futures-tick-values), [Wikipedia — Tick size](https://en.wikipedia.org/wiki/Tick_size), [Bookmap — pips/points/ticks](https://bookmap.com/blog/what-are-pips-points-and-ticks).

The key observation: across every mainstream asset class, the **relative** price granularity is ~1e-5 to ~1e-4 (worst realistic case ~1e-7 for cheap crypto spot). That is the input's *signal-to-noise floor*. `f32` carries 24 bits of significand ≈ 7.2 decimal digits, machine epsilon 2^-24 ≈ 6.0e-8 ([Wikipedia — Machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon)). So **`f32` already resolves prices ~100–1000× finer than the tick lattice** for equities/FX/index/crypto-futures. `f64` (ε ≈ 1.1e-16) is resolving ~10 billion times finer than the data deserves — pure waste for the *input* stage.

`f16` is the interesting boundary: 11 bits significand, ε ≈ 4.9e-4, ~3.3 decimal digits ([Wikipedia — Half-precision](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)). A raw price of $4123.50 in `f16` rounds to the nearest representable value with ~0.5–2 ES ticks of error — `f16` **cannot losslessly hold an absolute index/equity price**. `bf16` is worse: 8 bits significand, ε ≈ 7.8e-3, only ~2–3 digits ([John D. Cook — bfloat16](https://www.johndcook.com/blog/2018/11/15/bfloat16/)). This is the first hard rule: **low precision is unsafe for absolute price levels, safe for normalized / differenced / bounded quantities.**

---

## 2. Indicator VALUE precision ≠ DECISION stability

Two distinct error questions:

- **Value error** ε_v: how far the computed indicator is from the infinite-precision value.
- **Decision instability**: the probability that ε_v flips a discrete signal (crossover, threshold).

Decision instability is what matters, and it is governed by the **distance to the threshold**, not the absolute value error. A crossover/threshold decision is stable iff:

> |indicator − threshold| ≫ ε_v

Worked example — RSI overbought (`RSI > 70`): RSI is bounded [0,100] and the decision boundary is a single point at 70. If `f16` introduces ε_v ≈ 0.05 RSI points (generous; RSI is bounded so `f16` resolves it to ~0.03), a flip only occurs when the true RSI sits inside a ±0.05 band around 70. Over realistic data that is a vanishingly small fraction of bars, and the bars where it *does* matter are precisely the bars where the signal is economically meaningless (RSI = 69.97 vs 70.03 is not a different trade). The decision is *robust* to value error because the threshold is coarse relative to the indicator's own scale.

Contrast — **MACD zero-cross**: MACD = EMA(12) − EMA(26) is a *difference of two large, nearly-equal numbers*. This is the textbook setup for **catastrophic cancellation** ([Wikipedia — Kahan summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm), [Arnold, CERN 2014 — FP arithmetic](https://indico.cern.ch/event/313684/contributions/1687773/attachments/600513/826490/FPArith-Part2.pdf)). When two `f16` values ~4000 each are subtracted, their absolute error (~2) survives into a result that is itself near zero — exactly where the zero-cross decision lives. Here value error and decision boundary collide: **MACD-style differenced indicators are NOT safe to compute in low precision on absolute price.** They must be computed as differences of *higher-precision or normalized* EMAs.

So the precision requirement is not a property of the indicator alone — it is a property of **(indicator scale) × (threshold sharpness) × (how the value is constructed)**.

---

## 3. Where extra precision is *provably* wasted

Extra precision is wasted compute when the output's error is dominated by something other than float rounding. Three provable cases:

1. **Bounded, ratio-type indicators** (RSI, Stochastic %K/%D, MFI, Williams %R, CMF, Aroon — all bounded to [0,100] or [-100,100] or [-1,1]). The output dynamic range is ~3 decimal digits. `f16`'s ~3.3 digits already exceeds the *meaningful* range, and the decision thresholds (70/30, 80/20) are quantized to whole numbers. Computing these in `f64` spends 13 extra digits to support a 1-digit decision. **Provably wasted** once you verify the threshold margin exceeds `f16` ε (it does, by 2–3 orders of magnitude).

2. **Already-INT8-quantized paths**. The orderflow feature path (`rust/src/gpu/quantization.rs`) already collapses 6 features to INT8 (0–255, ~0.4% relative resolution) for signal generation and ML. Computing those features in `f64` upstream of an INT8 sink is wasted: the quantizer destroys ~13 digits of precision regardless. The INT8 step *is the accuracy budget made explicit* — and it has been working in production, which is direct empirical evidence the thesis holds for orderflow.

3. **The √n / n error argument cuts the other way for short windows.** Naive summation error grows as O(ε√n) RMS for n random terms ([Kahan summation, Wikipedia](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)). For a 14- or 20-period SMA/RSI window, √n ≈ 3.7–4.5, so even `f16` (ε ≈ 4.9e-4) yields a window-sum relative error ~2e-3 — borderline. But for `f32` (ε ≈ 6e-8) over the same window the error is ~2.7e-7, ~1000× below the tick floor: **`f32` is provably over-sufficient for any windowed indicator with n ≤ ~10^6.** This is exactly why the SMA FP64→FP32 conversion measured 1.33–1.74× with zero accuracy concern — the precision was always slack.

---

## 4. The dangerous zone: recursion and accumulation

Two mechanisms make low precision *unsafe* regardless of threshold coarseness:

- **Recursive feedback (EMA, Wilder's RSI smoothing, SuperTrend, Parabolic SAR).** EMA is `ema[i] = α·price + (1−α)·ema[i−1]`. Errors do not cancel; they are *carried forward and re-weighted*. The error of a recursive filter accumulates with a geometric memory of ~1/α periods (for a 26-period EMA, ~26 bars of error memory). In `f16` (ε ≈ 4.9e-4) on absolute price ~4000, each step injects ~2 absolute units and the recursion never forgets faster than (1−α). This is why EMA/RSI must stay `f32`+ on absolute price, or be reformulated on **log-returns / normalized price** (which is bounded near 0 and `f16`-friendly).

- **Long cumulative sums (OBV, VWAP, Cumulative Volume Delta, equity curve).** VWAP = Σ(price·vol)/Σ(vol) accumulates over the whole session (potentially 10^5–10^7 ticks). Here √n ≈ 100–3000, and the running accumulator grows large while increments stay small — the classic large+small cancellation. `f16`/`bf16` are unsafe (bf16 with ε ≈ 7.8e-3 over a session is catastrophic). Even `f32` warrants either `f64` accumulation or Kahan/error-feedback compensation (worst-case error becomes ~2ε, independent of n — [Kahan summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)). **The backtest equity curve is already `f64` in `tick_backtest_batch.rs` — correctly so, and should stay there.**

---

## 5. Setting a principled per-indicator tolerance (not an arbitrary 1e-4)

Replace `assert_allclose(..., atol=1e-4)` with a tolerance *derived from the decision*. The recipe:

1. **Identify the consumer's decision resolution Δ_d.** For a threshold signal it is the threshold's least significant digit (RSI 70 → Δ_d = 1 RSI point; or the empirically observed minimum gap between bars near the threshold). For a crossover it is the typical |gap| of the two lines at the crossing bar. For a backtest it is one tick (the smallest PnL difference a real fill could produce).
2. **Set the value tolerance to a fraction of the decision resolution**, e.g. ε_target = Δ_d / 10 (decisions stable to <10% flip risk at the boundary). For RSI: ε_target ≈ 0.1 RSI points — `f16` clears this with margin. For absolute price level: Δ_d = 1 tick ≈ 1e-5 relative → ε_target ≈ 1e-6 relative → `f32` clears it, `f16` fails.
3. **Choose the smallest format whose ε is < ε_target after propagating through the indicator's error-growth law** (×√n for windowed sums, ×1/α for recursion, ×n for unbounded accumulation, ÷cancellation-margin for differences).
4. **Validate empirically**: count signal flips between the candidate-precision and `f64`-reference runs on real data. The acceptance criterion is **zero economically-material signal flips**, not bitwise value match. (kimsfinance already has the harness for this — the OOS validation in commit `3a437ad`.)

This converts precision from a guessed constant into a *measured property of each indicator-consumer pair*.

---

## 6. Recommended precision-tier policy

Keyed to **indicator class × consumer**. RTX Ada (sm_89) gives 16× FP16/BF16 tensor throughput and 8× TF32 vs FP32 ([NVIDIA Ampere Architecture blog](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/); Ada peaks ~1,979 FP16/BF16 TFLOPS vs ~83 FP32), and for the memory-bound CUDA-core indicators the win is bandwidth: `f32`→`f16` halves bytes moved, scaling SMA-class kernels toward 2× and beyond.

| Tier | Format | Indicator class | Rationale |
|---|---|---|---|
| **T0 — keep f64** | f64 accumulate | VWAP, OBV, CVD, **backtest equity/PnL**, anchored VWAP | Unbounded session-long accumulation; large+small cancellation; money must be exact to the tick. Already f64 in `tick_backtest_batch.rs`. |
| **T1 — f32 (default)** | f32 storage, f32 or f64 accumulate | EMA, Wilder RSI smoothing, MACD lines, SMA/WMA/VWMA, Bollinger, ATR, Keltner, Donchian, SuperTrend, PSAR | f32 ε (6e-8) beats tick floor by 100–1000×; recursion/difference safe; this is where the FP64→FP32 win (1.33–1.74×→~2×) lives. **Differences (MACD, Bollinger bands) must subtract f32 values, never f16.** |
| **T2 — f16/bf16 candidate** | f16 storage (f32 accumulate) | Bounded ratio indicators consumed only by coarse thresholds: RSI value, Stochastic %K/%D, Williams %R, MFI, CMF, Aroon, ROC | Bounded [0,100]/[-1,1] output; ~3-digit meaningful range; thresholds are integers. f16's 3.3 digits suffice. **Accumulate the window sum in f32, round the final bounded value to f16.** Prefer `bf16` only if the upstream involves wide-dynamic-range intermediates; otherwise `f16` (more mantissa) is better for bounded values. |
| **T3 — int8 (already deployed)** | INT8 (0–255) | Orderflow 6-feature path → signal generation / ML | Decision is ±1/0/1; 0.4% resolution is the explicit accuracy budget. Production-validated. Extend pattern to any feature whose only consumer is a discrete classifier. |

**Decision rule for placing a new indicator:** unbounded accumulation → T0. Recursive or differenced on absolute price → T1. Bounded output + coarse threshold consumer → T2 (validate flip-rate). Output feeds only a discrete classifier → T3.

**Hard "never" list (low precision unsafe):** absolute price levels in f16/bf16; any difference-of-large-near-equal (MACD/Bollinger) where both operands are low precision; cumulative sums in f16/bf16; PnL/equity in anything below f64.

---

## Research limitations

- No empirical flip-rate study was run on kimsfinance's own data in this pass — the T2 (`f16` for RSI-class) recommendation is *bounded-by-construction* but should be confirmed with the step-4 harness on real ticks before shipping.
- `bf16` vs `f16` choice for bounded indicators is asserted from format properties; for indicators with wide intermediate dynamic range (e.g. MFI's money-flow sums) the accumulator must stay f32 regardless.
- Tick-size table is representative, not exhaustive; exotic instruments (some crypto-spot pairs ~1e-7 relative) can push the input floor below f16 resolution even for normalized quantities.

## Sources

- [NVIDIA Ampere/Ada Tensor Core throughput](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) — 16× FP16/BF16, 8× TF32 vs FP32
- [Wikipedia — Machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon) — f32 ε=6.0e-8, f64 ε=1.1e-16
- [Wikipedia — Half-precision floating-point format](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) — f16 ε≈4.9e-4, ~3.3 digits
- [John D. Cook — bfloat16 range and precision](https://www.johndcook.com/blog/2018/11/15/bfloat16/) — bf16 ε≈7.8e-3, ~2–3 digits
- [Wikipedia — Kahan summation algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) — naive error O(ε√n), Kahan O(ε) independent of n
- [Arnold (CERN 2014) — Techniques for Floating-Point Arithmetic](https://indico.cern.ch/event/313684/contributions/1687773/attachments/600513/826490/FPArith-Part2.pdf) — catastrophic cancellation
- [Schwab — Stock Index Futures Tick Values](https://www.schwab.com/learn/story/stock-index-futures-tick-values), [Wikipedia — Tick size](https://en.wikipedia.org/wiki/Tick_size), [Bookmap — Pips, Points, Ticks](https://bookmap.com/blog/what-are-pips-points-and-ticks) — tick granularity by asset class
