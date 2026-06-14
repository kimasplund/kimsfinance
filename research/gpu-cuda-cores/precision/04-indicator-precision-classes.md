# GPU Indicator Precision Classification

**Scope:** Every GPU indicator kernel under `rust/src/gpu` (CUDA-C strings embedded in
`.rs` files, plus `.cu` files in `gpu/kernels/` and `gpu/cuda/`). Read-only analysis,
no source modified. Hardware target: RTX 3500 Ada (sm_89), where **FP64 runs at 1/64 the
FP32 rate** — so dropping FP64 -> FP32 is a ~2x ceiling for compute-bound kernels and pure
DRAM/PCIe traffic savings (~2x bytes) for memory-bound ones. `cutlass/` is vendored
third-party and excluded.

## Answer to the user's question (short version)

Yes, you are over-specifying precision in several places — but it is **not uniform**, and
the codebase already encodes the right mental model in its own comments. The safe rule:

- **Coarse, bounded, single-pass math** (class A) tolerates f16; **f32 is the safe,
  drop-in default already adopted for SMA/Stochastic/Williams %R.**
- **Anything recursive (IIR) or cumulative** (classes B/C) has *error that grows with the
  series length* — f16 there silently corrupts thresholds (RSI>70, MACD sign-flip,
  Supertrend flips). These need an **f32 floor (B)** or **f32 + compensated/segmented
  summation, or f64 (C)**.
- The signal layer is coarse, but the *inputs* to the signal layer are not: a 0.5% error
  in an EMA near a crossover changes the trade. Precision must be specified on the
  **indicator**, not the decision.

The two biggest *current* over-specifications are: (1) the **window/momentum kernels still
on f64** (CCI, Bollinger, Donchian, Aroon, ROC, WMA, VWMA, CMF, Ichimoku, Pivot,
Fibonacci, MFI typical-price) — these are class A and are leaving the documented 1.3-1.7x
on the table; and (2) the **persistent batch kernels**, which are an entire parallel set
still 100% f64.

---

## Precision classes (definitions used below)

| Class | Pattern | Error behavior | Safe minimum precision | Tensor-core? |
|-------|---------|----------------|------------------------|--------------|
| **A** | Bounded-window, non-recursive (rolling sum/mean/max/min/std over `period`) | Bounded: ~`period`-term sum, no carry across the series | **f16-eligible** (f32 = safe drop-in; f16 needs a price-rebasing/scaled trick) | No (elementwise) |
| **B** | Recursive / IIR (EMA, Wilder smoothing, Supertrend/PSAR state) | Error compounds with feedback; near a threshold a tiny drift flips a discrete signal | **f32 floor** (never f16/bf16) | No (sequential) |
| **C** | Long cumulative (OBV, CVD, VWAP cumulative, ordered prefix sums) | Catastrophic cancellation / lost low bits as the running total grows; f32 loses integer exactness above 2^24 | **f32 + compensated/segmented sum**, or **f64** for exact integer-volume parity | No (scan) |
| **D** | GEMM-shaped (Heston char-fn batch, tensor-core matmul) | Depends: matmul tolerates f16/bf16 inputs *with f32 accumulate*; transcendental/complex math (branch cuts, `exp`) does not | **f16/bf16 inputs, f32 accumulate** for matmul; **f64** for Heston complex transcendentals | **Yes** (matmul only) |

A class-A kernel is "f16-eligible" but with a caveat that recurs across this whole library:
financial prices are large-magnitude, small-relative-range (e.g. BTC ~30,000 +/- a few %).
f16 has only ~3 significant decimal digits and a max of 65,504, so raw prices either
**overflow f16** (anything >65,504, e.g. an index level or a sum of prices) or **quantize
to ~$15 buckets at 30,000**. f16 on class-A kernels therefore requires **mean-subtraction /
rebasing per window** (compute on `price - window_open`), which the repo does *not* do
today. Without rebasing, **f16 is unsafe even for class A on real price scales** — this is
the single most important range caveat in this report.

---

## Per-kernel table

### Class A — bounded-window, non-recursive (f16-eligible with rebasing; f32 = safe today)

| Kernel | file:line | Class | Safe min precision | Risks | Tensor-core eligible? |
|--------|-----------|-------|--------------------|-------|----------------------|
| `sma_kernel_f32` (production) | `sma.rs:284` | A | **f16** (with per-window rebasing); f32 already shipped | f16 raw-price quantization at 30k scale; sum of `period` prices can exceed 65,504 for period>~2 -> **f16 overflow without rebasing** | No |
| `sma_kernel` (f64 reference) | `sma.rs:33` | A | f32 | Reference path only; kept f64 for the shared-mem correctness gate (`sma.rs:785`) | No |
| `sma_kernel_shared` | `sma.rs:61` | A | f32 | Same as SMA; shared-mem variant | No |
| `stochastic_k_kernel` / `_d_kernel` | `stochastic.rs:38,79` | A | **f32** (already converted, see note `stochastic.rs:15-19`) | %K = (close-low)/(high-low)*100; window max/min are exact selections among f32-rounded inputs. f16 OK for the ratio (bounded 0-100) but **range subtraction loses bits at 30k**; rebase first | No |
| `williams_r_kernel` | `williams_r.rs:29` | A | **f32** (already converted) | Same range-subtraction caveat as Stochastic | No |
| `bollinger_bands_kernel` | `bollinger.rs:27` | A | **f32** | Computes mean AND `sum_squared_diff` (variance) in-window. **`diff = close-sma` then `diff*diff`** is fine in f32 but f16 catastrophically loses the variance (squared small deviations underflow) -> **f16 unsafe for the std band even with rebasing** | No |
| `cci_pass1_kernel` / `cci_pass2_kernel` | `cci.rs:28,70` | A | **f32** | Typical-price SMA + mean-abs-deviation; `1/(0.015*MAD)` divide amplifies a small MAD error. f32 safe; f16 risky via the MAD denominator | No |
| `donchian_kernel` | `donchian.rs:26` | A | **f32** | Pure rolling max/min (exact selections) — f16 safe for the *comparison*, but stored levels at 30k quantize to ~$15. Use f32 | No |
| `aroon_kernel` | `aroon.rs:19` | A | **f32** | Argmax/argmin of window (index math, not value math) -> precision-insensitive; f16 fine for values, but currently f64 (9 double-uses) | No |
| `roc_kernel` | `roc.rs:22` | A | **f32** | `(close[i]-close[i-n])/close[i-n]*100`; relative -> bounded. f16 marginal (3-digit), f32 ample | No |
| `wma_kernel` | `wma.rs:24` | A | **f32** | `weighted_sum += close*weight`, weight up to `period`; **`weighted_sum` ~ period^2 * price** -> for period=200, ~1.2e9 -> **far past f16's 65,504 (overflow)** and near f32's exact-int limit only for huge periods. f32 safe; **f16 unsafe (overflow)** | No |
| `vwma_kernel` | `vwma.rs:22` | A | **f32** | `Σ close*vol` and `Σ vol` over window; products `price*volume` (30k * large vol) **overflow f16 immediately**. f32 fine for typical windows | No |
| `cmf_kernel` | `cmf.rs:37` | A | **f32** | Money-flow-volume sum / volume sum over window; `mfv` products overflow f16. f32 safe | No |
| `ichimoku rolling_max/min/midpoint/span` | `ichimoku.rs:55,84,113,131` | A | **f32** | Midpoints = (max+min)/2; range subtraction at 30k loses f16 bits. f32 safe | No |
| `pivot_points_kernel` | `pivot_points.rs:65` | A | **f32** | Pivot = (H+L+C)/3 and linear band offsets; sums of 3 prices ~90k **overflow f16**. f32 safe | No |
| `fibonacci rolling_max/min + levels` | `fibonacci.rs:67,100,133` | A | **f32** | Level = high - ratio*(high-low); range subtraction loses f16 bits. f32 safe | No |
| `keltner_bands_kernel` (band offset only) | `keltner.rs:63` | A | **f32** | Only the `middle +/- mult*ATR` band-assembly is on GPU; the EMA + ATR feeding it are class B and stay on CPU (`keltner.rs:9-16`) | No |
| `calculate_typical_price_kernel` (MFI) | `mfi.rs:58` | A | **f32** | `(H+L+C)/3`; sum ~90k **overflows f16**. f32 safe. (Rolling pos/neg flow sums run on CPU per `mfi.rs:11-18`.) | No |
| `calculate_money_flow_kernel` / `separate_pos_neg_flow_kernel` (MFI) | `mfi.rs:76,91` | A | **f32** | `tp*volume` products overflow f16. f32 safe | No |
| `calculate_true_range_kernel` (ATR/Supertrend, parallel TR only) | `atr.rs:52`, `supertrend.rs:59` | A | **f32** | TR = max(H-L,|H-Cprev|,|L-Cprev|) — bounded range, single pass. **This part is class A**; the Wilder smoothing it feeds is class B (on CPU). f32 safe; f16 loses range bits at 30k | No |
| `calculate_hl_average_kernel` / `calculate_basic_bands_kernel` (Supertrend) | `supertrend.rs:85,101` | A | **f32** | (H+L)/2 and band = hl_avg +/- mult*ATR; bounded | No |
| `calculate_dm_tr_kernel` / `calculate_di_kernel` / `calculate_dx_kernel` (ADX, parallel parts) | `adx.rs:59,110,146` | A | **f32** | Directional movement + DI/DX ratios are bounded (0-100). **The Wilder smoothing of +DM/-DM/TR between these is class B.** f32 safe | No |
| `vwap_tpv_kernel` (VWAP TPV only) | `vwap.rs:50` | A→C | **f32 for the product; C applies to the cumulative stage** | TPV = ((H+L+C)/3)*volume per-tick is bounded (class A), but it **feeds an f64 CPU cumulative sum** (class C, see below). Per-element f32 fine; **f16 overflows on price*volume** | No |
| `calculate_tpv_kernel` (anchored VWAP) | `vwap_anchored.rs:69` | A→C | **f32 product, f64 cumulative** | Same as VWAP; cumulative reset at anchors limits the running-sum length, slightly relaxing class C | No |

### Class B — recursive / IIR (f32 FLOOR; f16/bf16 unsafe)

| Kernel | file:line | Class | Safe min precision | Risks | Tensor-core eligible? |
|--------|-----------|-------|--------------------|-------|----------------------|
| EMA (now CPU `ema_cpu`) | `ema.rs:102,138` | B | **f32 floor** (CPU runs f64) | `EMA[i]=α*close[i]+(1-α)*EMA[i-1]` — loop-carried feedback. f16 drift compounds over thousands of bars and **flips price-crosses-EMA / MACD-sign signals**. Single-thread GPU was 6-10x slower, so it correctly moved to CPU | No (sequential) |
| MACD (now CPU `macd_cpu`) | `macd.rs:54` | B | **f32 floor** (CPU f64) | Three chained EMAs; the histogram **sign** is the signal — accumulated IIR error in f16 would mis-time crossovers. Was "1,647x slower" on single-thread GPU; CPU is the right call | No |
| Wilder smoothing in ATR (CPU stage) | `atr.rs:34,118-127` | B | **f32 floor** (CPU f64) | `ATR[i]=((p-1)ATR[i-1]+TR[i])/p` — IIR. Feeds Supertrend/Keltner bands; f16 unsafe | No |
| Wilder smoothing in ADX (CPU stage) | `adx.rs` (smoothing between DM/DI kernels) | B | **f32 floor** | Smoothed +DM/-DM/TR are recursive; ADX>25 trend gate depends on stable smoothing | No |
| Wilder smoothing in RSI (CPU stage) | `rsi.rs:9-29` | B | **f32 floor** | avg_gain/avg_loss are IIR; **RSI>70 / RSI<30 thresholds** are exactly the coarse decisions f16 would corrupt via compounding | No |
| `WildersOp` CUB scan (fused RSI) | `rsi_fused.cu:92`, kernel `rsi_fused.rs` | B (parallelized as weighted scan) | **f64** (as written); f32 floor *if* re-derived carefully | Expresses IIR as a prefix scan with `α(1-α)^(i-k)` weights (`rsi_fused.cu:21-29`). The `(1-α)^k` weighting spans a huge dynamic range over a long series -> **needs f64 to stay parity-exact**; f32 would drift, f16 is out of the question. Currently f64 | No (scan, not GEMM) |
| `persistent_ema_kernel` | `persistent/ema.rs:53` | B | **f32 floor** (f64 today) | One-thread-per-task IIR; explicitly sequential (`persistent/ema.rs:16-17`) | No |
| `persistent_rsi_kernel` | `persistent/rsi.rs:55` | B | **f32 floor** (f64 today) | Wilder smoothing per task | No |
| `persistent_atr_kernel` | `persistent/atr.rs:57` | B | **f32 floor** (f64 today) | Wilder smoothing per task | No |
| `persistent_macd_kernel` | `persistent/macd.rs:84` | B | **f32 floor** (f64 today) | Three chained EMAs per task | No |
| `persistent_keltner_kernel` / `persistent_elder_ray_kernel` | `persistent/keltner.rs:100`, `persistent/elder_ray.rs:54` | B | **f32 floor** | Consume an EMA (sequential) computed on CPU; band assembly itself is A but the pipeline is B-gated | No |
| Supertrend trend-state (CPU stage) | `supertrend.rs:12,40-41,176-177` | B | **f32 floor** (CPU f64) | Final bands carry continuity (`keep previous unless crossed`) + trend flip — a **discrete latch**. f16 band drift would spuriously flip the latch. Inherently sequential | No |
| Parabolic SAR (now CPU) | `parabolic_sar.rs:1-17` | B | **f32 floor** (CPU f64) | SAR + accelerating EP is a sequential latch with reversal; f16 would mis-time stop-and-reverse | No |
| Heikin-Ashi (`persistent_heikin_ashi_kernel`) | `candles/heikin_ashi.rs:51` | B | **f32 floor** (f64 today) | Explicitly an IIR filter (`candles/heikin_ashi.rs:19-20`): HA-Open depends on prior HA-Open/Close. f16 candle-body drift would skew every downstream indicator | No |
| Renko / Range-bar / Volume-bar / Tick-bar builders | `candles/renko.rs`, `candles/range_bars.rs:70`, `candles/volume_bars.rs`, `candles/tick_bars.rs` | B/C | **f32 floor (price), f64 (volume accumulation)** | Sequential brick/threshold state with running volume accumulation; price thresholds tolerate f32, **accumulated volume is class C** (running sum -> f32 loses integer exactness >2^24) | No |

### Class C — long cumulative (f32 + compensated summation, or f64)

| Kernel | file:line | Class | Safe min precision | Risks | Tensor-core eligible? |
|--------|-----------|-------|--------------------|-------|----------------------|
| OBV `obv_deltas_kernel` (per-tick delta) | `obv_optimized.rs:68` | A | f32 | The +/-volume delta selection is class A; only the *scan* over it is class C | No |
| OBV `scan_blocks_kernel` / `add_block_sums_kernel` (running sum) | `obv_optimized.rs:108,159` | **C** | **f64** (as written) or **f32+Kahan** | **OVERFLOW/exactness risk flagged in-source** (`obv_optimized.rs:41-45`): OBV is a running sum of volumes; **f32 loses integer exactness above 2^24 (~16.7M)** — high-volume symbols blow past this in minutes, and OBV's *level* (not just sign) is read. f64 required for exact parity; f32+compensated is the minimum viable downgrade | No (scan) |
| `persistent_obv_kernel` | `persistent/obv.rs:53` | **C** | **f64** / f32+Kahan | Same cumulative-sum exactness risk; explicitly sequential (`persistent/obv.rs:15-22`) | No |
| VWAP CPU cumulative (`calculate_vwap_from_tpv_cpu`) | `vwap.rs:75-93` | **C** | **f64** (as written) | Two long running sums `Σ TPV`, `Σ volume`, divided. `Σ TPV` = Σ(price*volume) grows to **1e9-1e12+ over a session** -> f32 loses ~6-9 low bits -> VWAP drifts; price-vs-VWAP is the signal. Source mandates f64 (`vwap.rs:47-49`). **No f32 overflow per se (f32 max ~3.4e38) but precision loss is the killer** | No |
| Anchored VWAP cumulative | `vwap_anchored.rs` (CPU cumulative) | **C** | f64 / f32+Kahan | Anchor resets bound the sum length -> shorter runs make f32+Kahan more defensible here than for session VWAP | No |
| CVD feature `cumulative_volume_delta` (orderflow) | `kernels/orderflow_signals_batch.cu:300-304,376-449` | **C** (already f32 via segmented scan) | **f32 (block-scan) + f64 reference parity gate** | Inclusive prefix sum of (buy-sell) done as block-local scan + exclusive block prefix (`orderflow...batch.cu:30-49`). **f32 magnitude can grow large** over millions of ticks; tests compare to an **f64 running reference** (`orderflow_batch.rs:1296-1307`). The segmented two-pass scan is exactly the "compensated/segmented summation" mitigation for class C in f32 | No (scan) |
| OHLCV aggregation `aggregate_ohlcv_kernel` (volume/quote-volume atomics) | `kernels/aggregation.cu:169` | **C** | **f64** (as written) | `atomicAdd(double)` accumulates per-candle volume; **f32 atomic add of many trades loses bits past 2^24** and is also non-deterministic in order. Source keeps f64 and notes it's memory-bound (`aggregation.cu:25`) so f64 ALU cost is irrelevant | No |
| `cum_delta_partial` per-tick scan inputs | `orderflow_signals_batch.cu:376` | A→C | f32 | Delta is class A; the scan that consumes it is the class-C concern above | No |

### Class D — GEMM-shaped / heavy math (tensor-core f16/bf16 in, f32 accumulate)

| Kernel | file:line | Class | Safe min precision | Risks | Tensor-core eligible? |
|--------|-----------|-------|--------------------|-------|----------------------|
| `fp16_matmul_wmma` | `kernels/fp16_wmma.cu:19` | **D** | **f16 inputs, f32 accumulate** (already correct) | Canonical tensor-core GEMM: `fragment<accumulator,...,float>` (`fp16_wmma.cu:35`). f16 inputs + f32 accumulate is the *intended* design; the only risk is feeding **unscaled prices** (f16 overflow >65,504 / quantization at 30k) — inputs must be **normalized/standardized before the GEMM** | **Yes** |
| `fp16` MMA via PTX (`mma.sync ... f16.f16.f16.f16`) | `kernels/fp16_mma_ptx.cu:41` | **D** | **f32 accumulate strongly recommended** | **RISK FLAGGED:** this PTX path accumulates in **f16** (`...f16.f16.f16.f16`, line 41), unlike the WMMA path. f16 accumulation over a long K-dimension **loses precision fast** — fine for throughput experiments, **not safe for anything whose result is read numerically.** Prefer the f16-in/f32-accumulate WMMA path | **Yes** (but mis-specified accumulator) |
| `fp32_mma_ptx` | `kernels/fp32_mma_ptx.cu` | D | f32 (TF32 tensor-core acceptable) | Full-precision GEMM reference; TF32 (19-bit) is a safe speedup for most finance matmuls | **Yes** |
| FP8 WMMA / FP8 GEMM (CUTLASS + PTX) | `fp8_wmma.rs`, `fp8_gemm_cutlass.rs`, `kernels/fp8_*.cu` | **D** | **fp8(E4M3) in, f32 accumulate**, with per-tensor scaling | fp8 has ~2 decimal digits; only viable for **pre-scaled/normalized** operands with f32 accumulate. Research/infra path, not an indicator. Quantization kernel is "software simulation" (`kernels_fp8_wmma.cu:175-177`) | **Yes** |
| `heston_characteristic_function` | `cuda/heston/characteristic_function.cu:128` | **D (transcendental, NOT matmul)** | **f64 required** | Complex `exp`/`log`/`sqrt` with **branch-cut selection** (Gatheral 2005 stable form, `characteristic_function.cu` ~line 195+) to avoid the "Little Heston Trap." `exp(C + D*v0)` and the `(1-g e^{-dτ})` denominators are **acutely precision/branch-sensitive**; f32 mis-selects branches and corrupts calibration. **Not tensor-core eligible** (not a matmul) | No |
| Greeks (`delta/gamma/theta/vega/rho.cu`) | `cuda/greeks/*.cu` | A-like (finite-difference) | **f64** (as written) | `(price(S+ΔS)-price(S-ΔS))/(2ΔS)` — **catastrophic cancellation**: subtracting two nearly equal option prices then dividing by a tiny bump. f32 (let alone f16) **destroys the difference**; f64 is justified here. Not GEMM | No |
| Strategy P&L kernels (`covered_call/iron_condor/...`) | `cuda/strategies/*.cu` | A | f32-f64 | Payoff arithmetic; bounded but money-precision-sensitive (cents). f32 acceptable; f64 conservative | No |

---

## Cross-cutting findings / risk flags

1. **f16 + raw financial prices = silent corruption (range, not just precision).** f16 max
   is 65,504 and has ~3 significant digits. Any price level, sum of prices ((H+L+C),
   period-window sums), or price*volume product **overflows or coarse-quantizes** at real
   crypto/equity scales. f16 is only safe on class-A kernels **after per-window
   mean-subtraction/rebasing** (compute on deltas from the window's first value), which
   **no kernel does today**. This single caveat downgrades most "f16-eligible" class-A
   kernels to "f16-eligible *with a rebasing rewrite*; f32 is the safe drop-in now."

2. **f32 integer-exactness cliff at 2^24 (~16.7M) is the real cumulative hazard, not f32's
   3.4e38 range.** OBV, VWAP `ΣTPV`, OHLCV volume atomics, and CVD all accumulate volume —
   high-liquidity symbols cross 16.7M cumulative volume quickly, after which f32 silently
   drops 1-unit increments. The codebase already mandates f64 for OBV/VWAP/aggregation
   (`obv_optimized.rs:41-45`, `vwap.rs:47-49`, `aggregation.cu:25`) and uses a segmented
   f32 scan + f64 parity gate for CVD. **Do not f32 these without Kahan/segmented
   summation.**

3. **The mis-specified accumulator: `fp16_mma_ptx.cu:41` accumulates in f16**
   (`mma.sync...f16.f16.f16.f16`). Every other tensor-core path here accumulates in f32.
   This is the one place a "more performance" push could quietly ship wrong numbers — flag
   it explicitly; route numeric workloads through `fp16_matmul_wmma` (f32 accumulate).

4. **The signal layer is coarse, but precision must be specified on the indicator.** A
   crossover (price-x-SMA, MACD sign, Supertrend flip) is maximally sensitive *exactly at
   the threshold*, where the two compared quantities are nearly equal — the worst case for
   low precision. Coarse *outputs* do not justify coarse *intermediate* math for B/C/D.

5. **Where lower precision is genuinely safe and underused (the opportunity):** all
   **class-A window/momentum kernels still on f64** — CCI, Bollinger, Donchian, Aroon, ROC,
   WMA, VWMA, CMF, Ichimoku, Pivot, Fibonacci, MFI typical-price/money-flow, ADX/ATR/
   Supertrend parallel pre-stages — plus the **entire `persistent/` batch-kernel set**
   (all f64). These are the same shape as SMA/Stochastic/Williams %R, which were already
   converted to f32 for the measured 1.3-1.7x. Converting them to f32 is the
   low-risk, high-yield move; f16 on them needs the rebasing rewrite from point 1.

6. **Where lower precision is NOT safe (do not touch):** every class-B IIR (EMA/MACD/
   Wilder/Supertrend/PSAR/Heikin-Ashi) below f32; every class-C cumulative
   (OBV/VWAP/CVD/aggregation) below "f32+compensated"; Heston complex transcendentals and
   option **Greeks finite differences** below f64 (cancellation/branch-cut sensitivity).
