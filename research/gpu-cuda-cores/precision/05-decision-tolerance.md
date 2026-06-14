# 05 — Where Precision Actually Matters: The Decision-Tolerance Gate

**Question (from the user):** Are we over-specifying numerical precision? Trading
decisions are coarse ("not infinite decimals"). Could f16/bf16/int8 be acceptable
for many indicators and unlock more performance — and where is it NOT safe?

**Short answer:** Yes for the *indicator → signal* leg, which is already a coarse
threshold/crossover collapse and the codebase already proves coarse precision is
fine there (the orderflow path runs `-use_fast_math` + INT8 and asserts only
*signal* parity, not value parity). NO for the *trade → P&L accounting* leg, which
is path-dependent compounding held to a hard `1e-9` gate. The correct accuracy
gate is therefore **"the same trades fire and P&L matches within a band,"** not
"every indicator value within 1e-4." The repo already contains both halves of this
gate; they just need to be promoted to the *primary* acceptance test for any
precision reduction.

---

## 1. Every strategy collapses a continuous indicator to a discrete decision

This is the load-bearing fact. Read the CPU strategies — none of them consume the
*magnitude* of an indicator; they all reduce it to Buy/Sell/Hold via a threshold or
a crossover. A small perturbation in the indicator value is invisible unless it
straddles the threshold.

**Thresholds (compare to a constant):**
- RSI mean-reversion: `if rsi < self.buy_threshold {Buy} else if rsi > self.sell_threshold {Sell}` — `rust/src/strategies/momentum.rs:69-75`
- RSI oversold/overbought (`<20 / >80`): `momentum.rs:178-184`
- ROC breakout (`roc > 2.0 / < -2.0`): `momentum.rs:665-671`
- CCI reversal (`cci < -100 / > 100`): `momentum.rs:774-780`
- MACD divergence (histogram sign): `momentum.rs:422-428`
- Donchian / Keltner breakout (`bar.close > upper_channel`): `rust/src/strategies/trend.rs:306-312`, `trend.rs:402-408`
- Bollinger squeeze / expansion (`bar.close > upper`, `< lower`): `rust/src/strategies/volatility.rs:86-92`, `volatility.rs:201-211`
- ATR breakout (`price_move > atr * mult`): `volatility.rs:329-335`

**Crossovers (compare two series + a sign change vs. the previous bar):**
- EMA golden/death cross: `if fast_ema > slow_ema && self.prev_fast_ema <= self.prev_slow_ema {Buy}` — `trend.rs:74-80`
- Triple-EMA stack ordering (`short > medium > long`): `trend.rs:191-197`
- MACD line/signal cross: `momentum.rs:298-304`
- Stochastic %K/%D cross in zone: `momentum.rs:542-548`

The orderflow GPU path is the same pattern, hardcoded in the kernel epilogue:
`if imb > 0.6 && delta > 1000.0 {1}` etc. — the reference mirror is at
`rust/src/gpu/orderflow_batch.rs:883-932` and the doc table at
`orderflow_batch.rs:122-140`.

**Implication:** the output of the indicator leg is a 2-bit decision
(Buy=1/Sell=-1/Hold=0). Demanding `1e-4` *value* fidelity on a quantity that is
about to be truncated to 2 bits is, by construction, over-specification — *except*
in the thin band around each threshold.

---

## 2. The orderflow INT8 path already PROVES coarse precision is acceptable

This is the existing precedent, in production, validated by tests:

1. **The kernel is compiled with full fast-math** — `prec_div=false`,
   `prec_sqrt=false`, `ftz=true`, `use_fast_math=true`
   (`rust/src/gpu/compile.rs:144-155`; comment lines `107-111` explicitly note
   "financial data rarely hits" denormals and "prioritize speed over ULP accuracy").
2. **Features are quantized to INT8 (0-255)** before they ever reach a strategy —
   `quantization_scale = 255/(max-min)`, `orderflow_batch.rs:285-296`; the kernel
   stores raw 0-255 codes (`orderflow_batch.rs:38-43`). That is ~0.4% of full
   scale of resolution, deliberately, and it is the *production* feature
   representation.
3. **The GPU↔CPU test asserts SIGNAL parity, not VALUE parity.** The crucial
   function is `signal_boundary_mismatch_ok` (`orderflow_batch.rs:1383-1400`): a
   GPU/CPU *signal* disagreement is **accepted** as long as the underlying feature
   sits inside a fast-math drift band around the threshold
   (`IMB_EPS=1e-5`, `INTENSITY_EPS=0.01`, `PVEL_EPS=5e-3`). The parity loop at
   `orderflow_batch.rs:1459-1477` literally does `if gpu_sig == cpu_sig {continue}`
   and only then checks the boundary band.

In other words, the team has *already decided and codified* that "the value drifted
but the decision is allowed to flip only near the boundary" is the correct contract
for the indicator→signal leg. The FP32 features themselves carry generous,
*decision-irrelevant* tolerances: feature 1 (a lone subtraction) is exact, but the
z-scores get `2e-3` and the cumulative sum scales with the prefix magnitude
(`orderflow_batch.rs:1434-1439`). Nobody asserts `1e-4` on those values because the
*only thing that matters downstream is the sign relative to a threshold.*

The just-completed SMA FP64→FP32 conversion is the same move one notch more
conservative: it is gated by an indicator-level `rel < 1e-4` test
(`rust/src/gpu/sma.rs:845-869`, `test_sma_f32_matches_f64`) with the justification
"SMA sums a small window of price-scale values, well within f32 range/precision"
(`sma.rs:275-278`).

---

## 3. Where precision is NOT safe: the P&L accounting leg

The backtest engine is a different animal and the codebase treats it as such. Once
a discrete trade fires, the money math is **path-dependent compounding**:

- Position sizing reinvests *all* cash every entry: `gross_position_value = cash / price`, then `cash = 0` (`rust/src/backtest/tick_engine.rs:344-354`). Every trade's size depends on the exact realized equity of all prior trades.
- P&L and equity are running f64 sums: `cash += position_value + pnl - fee - slippage_cost` (`tick_engine.rs:391`), mark-to-market at `tick_engine.rs:439`. The GPU mirror is identical (`rust/src/gpu/tick_backtest_batch.rs:667-735`).
- Trade records and metrics are all f64 (`GpuTrade` = 3×f64, `tick_backtest_batch.rs:101-110`; `BacktestResult` fields f64, `backtest/core.rs:248-315`).
- Sharpe / drawdown are sums of returns over the whole curve (`backtest/metrics.rs:90-160`), and the optimizer/euler search *ranks parameter sets by these scalars* (`backtest/euler_search.rs:11-12, 89-102`). A precision-induced bias in equity propagates into which parameter set is declared "best."

And the test bar reflects this: the GPU↔CPU backtest parity gate is **`1e-9`
relative on final equity, exact trade counts, exact timestamps, per-trade pnl
within `1e-9`** (`tick_backtest_batch.rs:13-15` doc, and the assertions at
`tick_backtest_batch.rs:1216-1222, 1370-1421`). The kernel-correctness tests
explicitly assert **no `double` is removed** from this path — the orderflow kernel
test `test_kernel_uses_no_double_precision` (`orderflow_batch.rs:1158-1168`) bans
FP64 in the *feature* kernel precisely because Ada runs FP64 at 1/64, while the
*backtest* kernel keeps f64 deliberately.

**Do not reduce precision here.** Compounding, cancellation in PnL
(`exit_value - position_value`, `tick_engine.rs:385`), and the
rank-by-tiny-scalar-difference optimizer all amplify error. f16/bf16 on equity
would silently re-order the optimizer's "best parameters."

Threshold/crossover edge cases that are *also* unsafe to coarsen further:
- **Exact-equality / strict-inequality crossovers** (`prev_fast <= prev_slow`,
  `trend.rs:74`): the decision is the *sign of a difference of two nearly-equal
  EMAs*. This is catastrophic cancellation — the one place where indicator
  precision genuinely feeds the decision. INT8 on the EMAs themselves would create
  spurious or missed crosses. (FP32 is fine — `1e-4` relative is far below typical
  EMA separation — but f16's ~`1e-3` relative is marginal and bf16's ~`1e-2` is
  not.)
- **Cumulative / prefix-sum features** (CVD, feature 5): error grows with the
  running magnitude (`orderflow_batch.rs:1414-1423`), so a low-precision accumulator
  drifts unboundedly. Keep the accumulator wide even if you quantize the *output*.

---

## 4. The REAL accuracy test (recommended gate)

The question "indicator value within 1e-4?" is the wrong gate — it is both too
strict (most of the indicator's range is decision-irrelevant) and too weak (it says
nothing about whether a near-threshold flip changed the trade set or the compounded
P&L). The right gate is **backtest-level equivalence**, and the repo already has the
two assertions; they just need to be made the *primary* acceptance criterion for any
precision change:

> **Precision-Reduction Acceptance Gate**
> For a candidate (lower-precision) indicator/feature kernel, on a representative
> historical dataset and the full strategy set:
>
> 1. **Trade-set equivalence (primary).** Run the *same* signals through the
>    *unchanged f64 backtest engine*. Require:
>    - identical trade **count** (exact),
>    - identical entry/exit **timestamps** (exact),
>    - ≥ 99.x% identical per-tick **signal** decisions; every disagreement must
>      fall in the threshold drift band (reuse `signal_boundary_mismatch_ok`,
>      `orderflow_batch.rs:1383-1400`).
> 2. **P&L equivalence (primary).** Final equity, Sharpe, max-drawdown, and
>    **optimizer-selected best parameters** must match the f64 baseline within a
>    *trading-meaningful* band (e.g. relative final-equity error ≤ a few bps and,
>    critically, **the argmax parameter set is unchanged**), using the existing
>    `1e-9` engine parity as the inner contract and a looser outer band for the
>    indicator change.
> 3. **Value tolerance (secondary, diagnostic only).** Keep the `rel < 1e-4`
>    indicator check (`sma.rs:861-863`) as a *fast smoke test*, not the acceptance
>    bar. It catches gross kernel bugs cheaply but must never be the final word.

Why this is the true gate: a precision change is *acceptable* iff it does not change
the trades you would have taken or the money you would have made — measured end to
end. That is exactly what `test_gpu_cpu_parity_random_signals`
(`tick_backtest_batch.rs:1342-1425`) and the orderflow signal-parity test
(`orderflow_batch.rs:1508-1519`) already measure in isolation; the recommendation is
to **chain them**: low-precision indicator kernel → signals → f64 backtest →
compare equity/trade-set to the f64-indicator baseline.

---

## 5. Concrete precision recommendation by stage

| Stage | Today | Safe target | Gate |
|---|---|---|---|
| Bounded oscillators feeding pure thresholds (RSI, Stoch, Williams %R, CCI, ROC) | f64 API / f32 GPU | **FP32 now; INT8/INT16 viable** for the *signal* (range is fixed & known, like orderflow features) | Trade-set + P&L equivalence (§4) |
| Moving-average **crossovers** (EMA/MACD/Triple-EMA) | f64 / f32 | **FP32** (cancellation in `fast-slow`); avoid bf16 | Trade-set equivalence, watch near-cross ticks |
| Channel/band breakouts vs. price (Bollinger/Donchian/Keltner/ATR) | f64 / f32 | **FP32** (price-scale, ample headroom — same logic as SMA `sma.rs:275-278`) | `1e-4` smoke + trade-set |
| Orderflow features → signals | **FP32 + fast-math + INT8 output** (already shipped) | already coarse; this is the proof point | `signal_boundary_mismatch_ok` (already) |
| Cumulative / prefix-sum accumulators (CVD) | f32 scan | keep **wide accumulator**, quantize only the output | prefix-magnitude-scaled tol (already, `orderflow_batch.rs:1437`) |
| **Backtest P&L / equity / Sharpe / optimizer ranking** | **f64** | **KEEP f64** — path-dependent compounding + rank-by-scalar | hard `1e-9` (already, `tick_backtest_batch.rs:1216`) |

The headline: the user's instinct is correct for the *front half* of the pipeline
(indicators→signals), and the project has already proven it there with INT8
orderflow. The discipline to preserve is keeping the *back half* (P&L accounting and
optimizer selection) in f64 and gating every precision change on **trade-set + P&L
equivalence**, not on raw indicator-value tolerance.
