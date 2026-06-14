# GPU CUDA Cores — Candles / Tick / Misc Pipeline (Read-Only Analysis)

**Scope:** `rust/src/gpu/candles/*.rs` (custom bar types) plus the top-level cores
`tick_aggregation.rs`, `aggregation.rs`, `orderflow_batch.rs`, `tick_backtest_batch.rs`,
`sweep.rs`, `kernels_2d.rs`, `kernels_3d.rs`.
**Hardware target:** RTX 3500 Ada (sm_89), CUDA 13.1. Ada runs FP64 at **1/64** the FP32 rate — precision choice is the single largest lever in this set.
**Method:** static read of source; no code modified. Evidence is `file:line`.

---

## 1. Inventory

### 1a. `candles/` — custom bar generation (persistent-kernel family)

| File | LOC | What it does | Inputs→Outputs | Parallelism (as written) |
|---|---|---|---|---|
| `candles/mod.rs` | 162 | Module wiring, `TaskBatch` type aliases, re-exports `execute_batch` | — | — |
| `candles/traits.rs` | 266 | `CandleAggregator` / `Trade`/`CandleBasedAggregator` trait layer over `PersistentIndicator` | — | static dispatch |
| `candles/types.rs` | 532 | `OHLCVCandle`, `TradeData`, `TradeSide`, buffer concat helpers (SoA `[ts.., price.., vol..]`) | — | host |
| `candles/time_bars.rs` | 628 | Trades→time OHLCV via bucket = `ts/interval`. **Embedded NVRTC kernel string** | 3→5 (O,H,L,C,V) | **1 thread/task** (`time_bars.rs:227`) + known flaky-output bug (`:113-141`) |
| `candles/volume_bars.rs` | 292 | Fixed-volume bars (sequential volume accumulation) | 3→7 (+start/end time) | **1 thread/task** (`volume_bars.rs:123`) |
| `candles/tick_bars.rs` | 269 | Fixed N-trades bars | 3→6 (+count) | **grid-stride over bars** (`tick_bars.rs:132`) — the one genuinely parallel candle kernel |
| `candles/range_bars.rs` | 236 | New bar when `high-low ≥ range` | 3→5 | **1 thread/task** (`range_bars.rs:97`) |
| `candles/renko.rs` | 273 | Price-only bricks, 2×brick reversal | 2→3 | **1 thread/task** (`renko.rs:107`) |
| `candles/heikin_ashi.rs` | 219 | HA smoothing (IIR, prev-bar dependent) | 4→4 | **1 thread/task** (`heikin_ashi.rs:84`) |
| `candles/csv_loader.rs` | 659 | CSV→`TradeData` ingestion | — | host |
| `candles/batch_builder.rs` | 402 | Multi-file/multi-symbol batch assembly + `execute_batch_with_symbols` | — | host |

All candle kernels are FP64 (`grep double`: time_bars 23, heikin_ashi 13) and run through the
shared cooperative-launch driver `rust/src/gpu/persistent/mod.rs` (`launch_cooperative_kernel`, `:401/:764`),
which launches **80% of all SMs resident** (`persistent/mod.rs:330`) with `grid.sync()` between every task.

### 1b. Top-level cores

| File | LOC | What it does | Precision / layout | Optimization state |
|---|---|---|---|---|
| `tick_aggregation.rs` | 1209 | Sorted single-pass trades→OHLCV (ms feeds). Sortedness check + bucket-range reduction + atomic OHLCV | **f32 + i64 ts, SoA** (`:28-31`) | **Reference-grade.** Ordered-uint atomicMax/Min (`:599-625`), batched scalar+D2H readbacks behind one sync (`:361-369`, `:494-512`), dense-range cap + CPU fallback (`:77`,`:386-400`), cached PTX (`:760`) |
| `aggregation.rs` | 828 | Binance `Trade`→`Candle`, dense binning + segment-boundary open/close | **f64** price/qty (the `Trade` struct is f64); i64 ts | Well-optimized but FP64: ordered-u64 atomics (`:467-493`), pinned async H2D staging (`:259-281`), zero-D2H candle count (`:242-244`), segment-boundary open/close = no atomics for O/C (`:46-49`) |
| `orderflow_batch.rs` | 1569 | 6 features + 5-strategy signals + INT8 quant, fully fused | **f32, SoA** (`:1156-1168` test asserts no `double`) | **Reference-grade & already fused.** 3-pass scan (block scan → block-sum scan → fused feature/signal/quant epilogue, `:22-35`), range-group dedup so features computed once for all strategies (`:298-348`), features f32 D2H skipped on prod path (`:584-694`) |
| `tick_backtest_batch.rs` | 1442 | Path-dependent backtest, pending-order queue, per-strategy metrics | **f64** price/equity (`GpuTrade` = 3×f64, `:101-110`); i8 signals | 1 thread/strategy packed in 128-thread blocks (`:408-413`), optional equity-curve stride to avoid 8 B/tick/strategy VRAM (`:31-36`), incremental Sharpe/DD in registers. Dense `[N_strat][N_ticks]` i8 signal upload (`:359`) |
| `sweep.rs` | 960 | Parameter-sweep builder (RSI/SMA/…/MACD) + optimization metrics (Sharpe/DD/win/PF) | f64 (delegates to indicator fns) | **Mislabeled.** Doc claims "1 launch, N parameters" (`:14-20`) but `execute()` **loops parameters sequentially** (`:409-413`), one full indicator GPU call each. `SweepBatch` buffer reuse is **stubbed** (`:700-720`, all comments). Metrics computed on **host** (`:538-680`) |
| `kernels_2d.rs` | 1060 | Batch (asset×candle) RSI/SMA/Stoch + fusion (momentum, volatility) | **f64**, row-major `[n_assets,n_candles]` | RSI/Bollinger still need a **CPU Wilder/smoothing stage** (`:323-369`, `momentum_fusion` takes pre-computed `avg_gain/avg_loss`). Naive **O(period)** window scans (`:134`,`:170`,`:296-307`), `#pragma unroll 4`, **no shared memory** (`grep __shared__`: 0) |
| `kernels_3d.rs` | 1192 | Param-sweep (period×asset×candle) RSI/SMA + Sharpe reduction | **f64** | gains/losses computed once and shared across periods (`:36-40`, good), but SMA sweep re-scans **O(period)** per output with no cross-period reuse (`:141-150`). Sharpe reduction uses shared mem + tree reduction (`:174-230`, good). RSI still needs CPU Wilder per (period,asset) (`:77`) |

---

## 2. Current optimization state — evidence

### Two tiers exist in this codebase

**Tier A — modern, f32/SoA, fused, batched (the "tick" cores):**
`tick_aggregation.rs`, `orderflow_batch.rs`. These are the template the rest should follow:
- f32 price path on purpose (`tick_aggregation.rs:28-31`, `:813-817` test forbids `double`).
- Native integer `atomicMax/atomicMin` over an order-preserving uint image instead of atomicCAS retry loops (`tick_aggregation.rs:599-625`, `aggregation.rs:467-493`).
- Multiple tiny readbacks batched behind a single `synchronize()` (`tick_aggregation.rs:361-369`).
- PTX compile cached process-wide (`compile_ptx_optimized_cached`, `tick_aggregation.rs:760`, `aggregation.rs:517`).
- orderflow fuses features→signals→INT8 in one kernel and dedups identical quant ranges so 5 strategies cost ~1 strategy of feature work (`orderflow_batch.rs:298-348`, `:560-694`).

**Tier B — legacy FP64, under-parallelized, un-fused (the "candles" + 2D/3D + sweep cores):**
- **FP64 everywhere** on an Ada GPU that runs FP64 at 1/64 FP32: `kernels_2d.rs` 54 `double` refs, `kernels_3d.rs` 41, `time_bars.rs` 23, `heikin_ashi.rs` 13. This is a ~roughly-an-order-of-magnitude throughput tax on the compute-bound window scans, and **doubles** H2D/D2H transfer volume vs f32 (the exact mistake `tick_aggregation.rs:289-294` documents avoiding).
- **One thread per task** in 4 of 6 candle kernels (`time_bars.rs:227`, `volume_bars.rs:123`, `range_bars.rs:97`, `renko.rs:107`, `heikin_ashi.rs:84`) — a cooperative grid of ~hundreds of blocks is launched (`persistent/mod.rs:330,401`), and for a single-symbol task all but **one** thread idle-spin to `grid.sync()`. Effective utilization for the common 1-task case is `1 / (SMs × blocks × 256)` ≈ far below 1%.
- **`grid.sync()` between every task** (`time_bars.rs:312`, all candle kernels) serializes tasks and forces a cooperative launch even when a plain grid-stride launch would do.
- **Worst-case output over-allocation**: the persistent driver always allocates `output_size = n * num_outputs` (`persistent/mod.rs:539,576`) — i.e. one bar per input *trade* — and copies the whole thing D2H (`download_batch_results`), even though `expected_compression_ratio()` is 100 for time bars / 30 for renko / 20 for range. For a 100k-trade symbol → time bars, that is a 5×100k×8 B = 4 MB D2H where the real candles are ~5 KB.
- **Known-flaky kernel shipped**: `time_bars.rs:113-141` documents the time-bar persistent kernel "frequently produces all zeros" with pinned memory; its GPU test is `#[ignore]` as flaky (`:397`). The dense atomic path in `tick_aggregation.rs`/`aggregation.rs` already produces correct time OHLCV — the persistent time-bar kernel is redundant *and* broken.
- **`sweep.rs` is sequential despite its name** (`:409-413`); the documented 10-50× "single launch" speedup is not implemented, and `SweepBatch` is an empty shell (`:700-720`). The actual multi-period parallelism lives in `kernels_3d.rs`, which `sweep.rs` does not call.
- **Host fallback stages break the GPU pipeline**: 2D/3D RSI round-trip to CPU for Wilder smoothing (`kernels_2d.rs:323-369`, `kernels_3d.rs:77`), and `sweep.rs` computes all optimization metrics on host (`:538-680`) — D2H of every parameter's full series.

### Persistent / graph usage
- Cooperative-grid "persistent" pattern is used only by the candle kernels and is the *wrong* tool here (single-task workloads, sequential `grid.sync`).
- CUDA graphs (`cuda_graphs.rs`, `batch_graphs.rs`) exist in the indicator pipeline (`gpu/mod.rs:328-337`) but **none of the in-scope cores use them** — every core in this set is a discrete per-call launch sequence.

---

## 3. Ranked optimization opportunities

Ranked by (impact × inverse effort). Impact = expected throughput/VRAM win on sm_89.

| # | Opportunity | Impact | Effort | Evidence / rationale |
|---|---|---|---|---|
| **1** | **Convert 2D/3D batch+sweep kernels and candle kernels from FP64→FP32** (keep i64 timestamps). Prices are f32 at the source already in the tick path. | **High** | **Med** | Ada FP64 = 1/64 FP32. These kernels are arithmetic-bound window scans. Halves transfer volume too. Pattern already proven in `tick_aggregation.rs:289-294`. Risk: Bollinger std/var cancellation — mitigate with Welford or f64 accumulator in-register only. |
| **2** | **Retire the 4 single-thread candle kernels; re-parallelize.** Time bars → route to the existing dense atomic `tick_aggregation`/`aggregation` path (already correct & f32). Tick bars are already grid-stride (keep). Volume/range/renko/HA are sequential-per-series → make them grid-stride **across symbols** (one *block* per task, threads cooperate) instead of one *thread* per task, and drop the cooperative `grid.sync` for a plain launch. | **High** | **Med-High** | `time_bars.rs:227`, `volume_bars.rs:123`, `range_bars.rs:97`, `renko.rs:107`, `heikin_ashi.rs:84` all `global_tid == task_id % grid_size`. Fixes the <1% utilization *and* the documented all-zeros flaky bug (`time_bars.rs:113-141`). Highest correctness upside. |
| **3** | **Make `sweep.rs` actually batch** by dispatching to `kernels_3d.rs` (period×asset×candle) in one launch instead of the sequential per-parameter loop, and move Sharpe/DD/win/PF to the on-device reduction already present in `kernels_3d.rs:174-230`. | **High** | **Med** | `sweep.rs:409-413` loops; `:538-680` metrics on host. The 3D kernels and reduction already exist — this is wiring, not new kernels. Delivers the 10-50× the docs already promise. |
| **4** | **Size candle output buffers to `n / expected_compression_ratio` (+margin) instead of `n × num_outputs`**, and D2H only the populated prefix. | **Med** | **Low** | `persistent/mod.rs:539,576` + `download_batch_results`. Compression ratios already declared (`time_bars` 100, `renko` 30, `range` 20). ~20-100× less candle D2H and VRAM. |
| **5** | **Add a sliding-window accumulator (or shared-memory tile) to SMA/Bollinger/Stoch window scans** in `kernels_2d`/`kernels_3d` to kill the O(period) inner loop. | **Med** | **Med** | `kernels_2d.rs:134,170,296-307`, `kernels_3d.rs:141-150`. SMA/Bollinger are pure prefix-sum → O(1)/element. Stoch needs a monotonic-deque or shared tile. `grep __shared__` = 0 in kernels_2d. |
| **6** | **Eliminate the CPU Wilder round-trips** in 2D/3D RSI with an on-device sequential-scan kernel (or fold into a recurrence kernel like the indicator pipeline's `rsi_fused`). | **Med** | **Med-High** | `kernels_2d.rs:323-369`, `kernels_3d.rs:77`. Removes a full D2H+H2D per RSI batch/sweep. `gpu/mod.rs:166` already has `rsi_fused` — reuse it per (asset/period). |
| **7** | **`tick_backtest_batch`: compress the i8 signal upload** (RLE / sparse signal index) — signals are overwhelmingly HOLD=0. | **Low-Med** | **Med** | `tick_backtest_batch.rs:359` uploads dense `N_strat × N_ticks` i8. At 100M ticks × many strategies this is the dominant H2D. Kernel already grid-strides strategies. |
| **8** | **Evaluate f32 for `aggregation.rs`** to match `tick_aggregation.rs`. | **Low-Med** | **Low-Med** | `aggregation.rs` `Trade` is f64 (Binance ingest type) — the win requires touching the ingest type, hence lower priority than #1's leaf kernels. |

---

## 4. Fusion / combine potential

These cores form a natural **single tick→candle→features→signals→backtest megapipeline** that today is 4 separate launch groups with host hops between them. SoA f32 layout is the shared currency and two of the four stages already speak it.

1. **Aggregation unification (highest-value combine).** `tick_aggregation.rs` (f32, sorted, dense atomic) and the `candles/time_bars.rs` persistent kernel compute *the same thing* by different methods; `aggregation.rs` is the f64 Binance-typed twin. **Collapse to one f32 dense-atomic aggregator** and have time bars + Binance candles call it. Volume/tick/range/renko/HA stay as separate per-bar-rule kernels but should share the f32 SoA `[ts,price,vol]` input buffer (no re-upload) and the same launch driver.

2. **Aggregation → orderflow fusion.** `tick_aggregation.rs` already outputs SoA candles explicitly "for compatibility with downstream GPU kernels (orderflow analysis)" (`tick_aggregation.rs:115-117`). The aggregated `close/volume` plus buy/sell volumes feed `orderflow_batch.rs` directly. These can share one device residency — aggregate, then launch the orderflow 3-pass scan on the *same* device buffers with **no D2H/H2D** between them. Today they are independent `process_batch` calls.

3. **Orderflow → backtest fusion.** `tick_backtest_batch.rs:304-310` explicitly documents that its i8 signal stream is the integration point for `orderflow_batch::OrderflowOutput::signals`. The orderflow kernel already produces i8 signals on-device; instead of copying signals to host and re-uploading (`orderflow_batch.rs:688` D2H → `tick_backtest_batch.rs:366` H2D), pass the device pointer straight into the backtest kernel. Eliminates a full `N_strat × N_ticks` round-trip.

4. **Sweep ⊂ kernels_3d.** `sweep.rs` should *become* a thin host front-end over the `kernels_3d.rs` megakernel (period×asset×candle) + on-device Sharpe reduction (`kernels_3d.rs:174-230`), rather than a sequential per-parameter dispatcher. This is consolidation, not new code.

5. **Candle kernels share one megakernel skeleton.** Volume/range/renko/HA are all "one sequential pass per series with O,H,L,C state in registers." They differ only in the *bar-close predicate*. A single block-per-task kernel templated on the predicate (or selected by an enum arg) would replace 4 near-duplicate NVRTC strings and let them share the grid-stride-across-symbols launch and the right-sized output buffers from opportunity #4.

**Combine constraints:** fusions 2-3 require unifying precision (f32) and the launch driver across the cores; backtest currently runs f64 (`GpuTrade`), so a fully fused f32 pipeline forces a precision decision on the equity math (f32 equity over millions of marks risks drift — keep equity accumulation in f64 *in-register* while inputs stay f32, as `kernels_3d.rs:204-206` already does for Sharpe moments).
