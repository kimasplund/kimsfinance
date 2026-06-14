# GPU Technical-Indicator Cores — Inventory & Optimization Assessment

**Scope:** `rust/src/gpu/{sma,ema,ma_advanced,atr,adx,rsi,bollinger,cci,cmf,donchian,elder_ray,keltner,mfi,supertrend,ichimoku,aroon,fibonacci}.rs` + `rust/src/gpu/scan.rs`
**Target HW:** RTX 3500 Ada (sm_89), 12 GB, CUDA 13.1. **Critical hardware fact for everything below: on Ada Lovelace, FP64 throughput is 1/64 of FP32.**
**Method:** Read-only static analysis. Evidence cited as `file.rs:line`.

---

## 1. Inventory

| File | LOC | Precision | What it does | Kernel structure |
|------|-----|-----------|--------------|------------------|
| `scan.rs` | 1478 | f32 (+f64/F64Acc paths) | **Shared primitive**: 3-kernel deterministic prefix sum/max + affine linear-recurrence scan; device-resident `wilder_smooth_f32` / `ema_f32` (`scan.rs:616`, `scan.rs:656`). Graph-capturable, no host sync. | partials → aggregates → fixup, 256t×4items/tile |
| `sma.rs` | 736 | f64 | Simple MA. Naive `sum += close[idx-j]` over `period` (`sma.rs:46`). Also a `_shared` variant the docs admit gives 0-3% (`sma.rs:273`). | 1 kernel, O(n·period) |
| `ema.rs` | 752 | n/a | **Reverted to pure CPU** (`ema.rs:1-58`). `ema_gpu` is `#[deprecated]` and delegates to `ema_cpu` (`ema.rs:230`). No GPU kernel remains. | none (CPU IIR loop `ema.rs:138`) |
| `ma_advanced.rs` | 1407 | **f32** | DEMA/TEMA/KAMA (batch recurrence, 1 thread per (series,param)) + HMA (element-parallel, shared-mem halo, `ma_advanced.rs:282`). The **modern exemplar**. | batch + fused HMA |
| `atr.rs` | 422 | f64 | True Range kernel (`atr.rs:53`), then **Wilder smoothing on CPU** (3 H2D / 1 D2H / 1 sync). | 1 kernel + CPU smoothing |
| `adx.rs` | 802 | f64 | DM/TR → DI → DX kernels (`adx.rs:59/110/146`) with **4 CPU Wilder round-trips** (6 H2D / 4 D2H / 2 sync). | 3 kernels + 4× CPU smoothing |
| `rsi.rs` | 485 | f64 | gains/losses kernel + RSI kernel (`rsi.rs:51/72`), **Wilder on CPU** between them (3 H2D / 3 D2H / 2 sync). | 2 kernels + CPU smoothing |
| `bollinger.rs` | 437 | f64 | SMA + rolling stddev, **two O(period) passes over `close` per thread** (`bollinger.rs:45`, `bollinger.rs:59`). | 1 kernel, O(n·2·period) |
| `cci.rs` | 517 | f64 | Typical price + SMA (pass1), MAD + CCI (pass2). **Recomputes TP O(period)× per thread** (`cci.rs:61`). | 2 kernels, O(n·period) |
| `cmf.rs` | 620 | f64 | Money-flow-multiplier × volume, rolling sum (`cmf.rs:54`). | 1 kernel, O(n·period) |
| `donchian.rs` | 516 | f64 | Rolling max(high)/min(low) (`donchian.rs:43`). | 1 kernel, O(n·period) |
| `elder_ray.rs` | 515 | f64 | bull=high−ema, bear=low−ema. **EMA supplied by caller** (element-parallel, `elder_ray.rs:54`). | 1 kernel, O(n) |
| `keltner.rs` | 533 | f64 | bands = ema ± mult·atr. **EMA & ATR supplied by caller** (`keltner.rs:64`). | 1 kernel, O(n) |
| `mfi.rs` | 787 | f64 | TP → raw money flow → pos/neg flow → rolling sums → MFI (4 kernels, `mfi.rs:59-127`). 6 H2D / 3 D2H / 2 sync. | 4 kernels, O(n·period) |
| `supertrend.rs` | 806 | f64 | TR + HL-avg + basic bands kernels (`supertrend.rs:59/85/101`), **ATR/trend logic on CPU** (4 H2D / 3 D2H / 2 sync). | 3 kernels + CPU |
| `ichimoku.rs` | 816 | f64 | Rolling (high+low)/2 over multiple windows + displacement shifts (`ichimoku.rs:56-169`). 3 H2D / 5 D2H. | ~6 kernels, O(n·period) |
| `aroon.rs` | 541 | f64 | argmax/argmin position-in-window (`aroon.rs:49`). | 1 kernel, O(n·period) |
| `fibonacci.rs` | 708 | f64 | Rolling swing high/low + 6 retracement levels (`fibonacci.rs:68-141`). 2 H2D / 6 D2H. | ~3 kernels |

---

## 2. Current Optimization State (evidence-based)

### 2.1 Precision: pervasive f64 on an FP64-starved GPU — the single biggest waste
- **15 of 17 indicators run all device math in `double`** (`grep` of `double* __restrict__`: every file except `ema.rs` (CPU) and `ma_advanced.rs`). On sm_89 this caps those kernels at **1/64 of peak FP32 ALU throughput**.
- `ma_advanced.rs` proves the project already knows this and documents the rationale: *"Ada Lovelace executes FP64 at 1/64 of the FP32 rate, so f64 device code would be catastrophically slow… EMA-style recurrences are convex combinations (contractive), so f32 rounding does not amplify"* (`ma_advanced.rs:34-41`). The same argument applies to SMA/Bollinger/CCI/Donchian/Aroon/CMF/MFI on price-scale data, yet they remain f64.
- `scan.rs` also defaults to f32 with a documented self-healing-error justification (`scan.rs:48-55`). So **two cores already adopt f32; the other fifteen did not.**

### 2.2 Memory access: O(n·period) redundant global loads, no rolling reuse
- Every windowed indicator reloads its **entire window from global memory for every output index**. SMA (`sma.rs:46`), Bollinger (`bollinger.rs:45` + a *second* pass `bollinger.rs:59`), CCI (`cci.rs:59-62`), Donchian (`donchian.rs:43`), Aroon (`aroon.rs:49`), CMF (`cmf.rs:54`), MFI rolling sums. Total global traffic is **O(n·period)** when the math is inherently O(n) via a rolling/prefix formulation.
- Bollinger is the worst: **two** independent O(period) passes over the same `close` window per thread (mean, then variance) — `bollinger.rs:41-70`.
- CCI deliberately **recomputes typical price inline `period` times** rather than reading the stored array (`cci.rs:51-64` comment), trading a documented cross-block race for 3× redundant L2 loads per window element.
- The shared-memory escape hatch in `sma.rs` is self-described as worthless for this access pattern: *"minimal benefit (0-3%, possibly regression)"* (`sma.rs:273-278`). It is the wrong tool; a **rolling sum / prefix-sum** is the right one and already exists in `scan.rs`.

### 2.3 The flagship `scan.rs` primitive is wired into NOBODY
- `scan.rs` (1478 LOC) was purpose-built — its own module docs say so — to kill the CPU round-trips in *"RSI / ADX / MFI / MACD"* and the single-thread cumsum in *"OBV / VWAP"* (`scan.rs:5-11`), and to fix `rsi_fused.cu`'s broken non-associative CUB approach (`scan.rs:16-28`).
- **`grep -rln 'scan::'` over the whole `gpu/` dir returns only `scan.rs` itself.** No indicator imports it. The device-resident, graph-capturable `wilder_smooth_f32`/`ema_f32` are dead code outside tests.
- Meanwhile the consumers it targets still round-trip to the CPU: RSI (`rsi.rs:9-11`, 3 D2H/H2D + 2 sync), ADX (**four** CPU Wilder passes, `adx.rs:210-216`; 6 H2D/4 D2H/2 sync), MFI (`mfi.rs`, 6 H2D/3 D2H/2 sync), ATR, Supertrend.
- The intended fused path `rsi_fused.rs` is **hard-disabled**: `is_fused_available()` returns a literal `false` and the entire FFI is commented out (`rsi_fused.rs:74-79`, `rsi_fused.rs:33-69`) due to a CUDA-13 `rsqrt` exception-spec mismatch. So the only "fused" RSI is off, and the correct NVRTC replacement (`scan.rs`) is unused.

### 2.4 Launch pattern: per-call PCIe round-trips + host syncs, no graphs, no input sharing
- Every indicator follows the same per-call shape: acquire pinned buffer → H2D → launch → D2H → **`stream.synchronize()`** → return host `Array1` (e.g. `sma.rs:203-253`, `bollinger.rs:187-229`). Each call **blocks the host**.
- Multi-pass indicators sync **mid-pipeline** purely to hand data to the CPU smoother: ADX 2 syncs (`adx.rs:381`, `adx.rs:477`), RSI 2 syncs (`rsi.rs:264`, `rsi.rs:335`), MFI 2 syncs (`mfi.rs:402`, `mfi.rs:472`), Supertrend 2 syncs (`supertrend.rs:328/399`). These syncs forbid CUDA-graph capture and serialize everything.
- **Batch dispatch re-uploads shared inputs N times.** `batch.rs:489` / `batch.rs:35` call the standalone `atr_gpu`, `adx_gpu`, `cci_gpu`, … each of which independently re-copies `high`/`low`/`close` H2D. A batch over ATR+ADX+Supertrend+CCI+Ichimoku uploads `high/low/close` ~5×.
- Compile path: all 17 indicators (except CPU `ema`) go through `compile_ptx_optimized_cached` (`grep` confirms 16 hits), which is cached process-wide — so PTX compilation is **not** the hot-path problem; the transfers and syncs are.

### 2.5 Warp usage
- Single-pass kernels are perfectly coalesced on the *output* write and on the *current-index* read, but the windowed back-references (`close[idx-j]`) generate **overlapping, strided per-thread reads** that defeat coalescing for `period > 1` — they hit L1/L2 but waste bandwidth proportional to `period`. No use of warp shuffles or block-level rolling reuse except the HMA halo in `ma_advanced.rs:298-320` and the scan's `__shfl_up_sync` (`scan.rs:33-38`). The recurrence indicators in `ma_advanced.rs` run **1 active thread per warp** by design (serial-in-time, `ma_advanced.rs:93`), wasting 31/32 lanes unless the batch dimension is large.

---

## 3. Ranked Optimization Opportunities

Ranked by (Impact × inverse Effort). Impact = expected speedup/throughput on this HW.

| # | Opportunity | Impact | Effort | Evidence / rationale |
|---|-------------|--------|--------|----------------------|
| **1** | **Wire `scan.rs` into RSI/ADX/MFI/ATR/Supertrend; delete the CPU Wilder/EMA round-trips.** Replace D2H→CPU-smooth→H2D with device-resident `wilder_smooth_f32`/`ema_f32`. | **High** | **Low–Med** | The primitive already exists, is tested (`scan.rs:1355+`), graph-capturable, and CPU-parity-verified. Removes 2 syncs + 4–9 PCIe transfers per call (ADX: `adx.rs:381/477`; RSI: `rsi.rs:264/335`; MFI: `mfi.rs:402/472`). Pure integration, no new kernel math. Unblocks the dead 1478-LOC investment and the disabled `rsi_fused.rs`. |
| **2** | **Convert f64 indicator kernels to f32** (price-scale data; keep f64 only where a caller demands it). | **High** | **Med** | 15/17 cores are f64 on a 1/64-FP64 GPU (§2.1). `ma_advanced.rs:34-41` and `scan.rs:48-55` already justify f32 for exactly these recurrence/window patterns. Up to ~ALU-bound 64× headroom; realistically bandwidth-bound → ~2× from halved global traffic + large ALU win on compute-heavy ones (Bollinger, CCI, MFI). |
| **3** | **Replace O(n·period) sliding windows with rolling/prefix-sum formulations** using `inclusive_scan_f32`/`pair_sum` from `scan.rs`. Start with SMA, Bollinger, CCI, CMF, MFI. | **High** | **Med** | Turns O(n·period) global traffic into O(n) (§2.2). Bollinger does the window twice (`bollinger.rs:45,59`); CCI recomputes TP `period`× (`cci.rs:61`). Prefix-sum machinery already exists and is device-resident. Donchian/Aroon need a rolling **max** (sparse-table / monotonic deque), higher effort — defer those. |
| **4** | **Eliminate mid-pipeline host syncs → make multi-kernel indicators CUDA-graph-capturable.** Keep everything device-resident end-to-end (depends on #1). | **Med–High** | **Med** | ADX/RSI/MFI/Supertrend each `synchronize()` twice only to feed the CPU smoother (§2.4). Once #1 removes those, the whole indicator is a pure device DAG → capture once, replay per chunk via existing `cuda_graphs.rs`/`batch_graphs.rs`. |
| **5** | **Share H2D inputs across indicators in batch dispatch.** Upload `high/low/close` once per chunk; pass device buffers to indicator kernels. | **Med** | **Med** | `batch.rs:489/35` re-uploads shared OHLC per indicator. A 5-indicator batch over OHLC pays ~5× the H2D cost. Requires refactoring indicator fns to accept device slices (the `scan.rs`-style API) instead of host `Array1`. |
| **6** | **Migrate kernels from per-call `const X_KERNEL: &str` PTX modules to f32 batch kernels** following the `ma_advanced.rs` layout (`[num_series, series_len]`, 1 thread per (series,param)). | **Med** | **High** | Enables parameter sweeps & multi-series saturation on a single launch; amortizes launch/transfer. Larger rewrite touching all 17 files. |
| **7** | **Drop dead/misleading code:** `sma_gpu_shared` (admitted 0-3%, `sma.rs:273`), and either fix or remove `rsi_fused.rs` (disabled, `rsi_fused.rs:74`). | **Low** | **Low** | Reduces maintenance surface and stops shipping a 1478-LOC + disabled-FFI graveyard. Cleanup, not perf. |

---

## 4. Fusion / Combine Potential (megakernel & batch candidates)

These groups share inputs and/or window structure and are the strongest candidates to **share a megakernel or a single batched launch** (also eliminating the repeated H2D in §2.4).

### 4.1 True-Range family — share one TR kernel + one device ATR
- **ATR, ADX, Supertrend** each compute an *identical* True Range from `high/low/close`: `atr.rs:53`, `adx.rs:98-103`, `supertrend.rs:74-79` are byte-for-byte the same `max(hl, |hc|, |lc|)`. **Compute TR once, smooth once (via `scan.rs` Wilder), and feed all three.** ADX additionally needs +DM/−DM (already in the same `calculate_dm_tr_kernel`, `adx.rs:59`).
- **Keltner already consumes ATR + EMA as inputs** (`keltner.rs:64`) — it is the proof that a shared-ATR/EMA design works; extend that pattern to ATR/ADX/Supertrend.

### 4.2 EMA-consumer family — share one device EMA
- **Elder Ray (`elder_ray.rs:54`), Keltner (`keltner.rs:64`)** already take EMA as a caller-supplied input. **Bollinger middle band, MACD, DEMA/TEMA seeds, and Ichimoku-adjacent smoothing** all need EMA/SMA. A single device-resident `ema_f32`/`scan` output (§3.1) can fan out to all of them in one graph instead of each recomputing.

### 4.3 Typical-Price / windowed-SMA family — share TP + prefix sum
- **CCI (`cci.rs:44/61`) and MFI (`mfi.rs:59`)** both compute Typical Price `(h+l+c)/3` and then a windowed sum over it. **Bollinger** needs a windowed sum + sum-of-squares over `close`. All three can share a single **prefix-sum / pair-sum** pass (`inclusive_scan_pair_sum_f32`, `scan.rs:380`) producing `(Σx, Σx²)` (or `(Σtp, Σtp·v)` for MFI/CMF/VWAP), from which mean, stddev, MAD, and money-flow ratios are O(1) per index. This is the single highest-leverage fusion: it simultaneously fixes §2.2 (redundant windows) for CCI/Bollinger/CMF/MFI.

### 4.4 Rolling-extremum family — share a monotonic-window kernel
- **Donchian (`donchian.rs:43`), Aroon (`aroon.rs:49`), Fibonacci swings (`fibonacci.rs`), Ichimoku Tenkan/Kijun (`ichimoku.rs:114`)** all reduce to rolling max(high)/min(low) over a window. A shared sparse-table or monotonic-deque kernel (O(n) instead of O(n·period)) would serve all four. Higher effort than prefix-sum (max isn't invertible) but high payoff because four indicators share it.

### 4.5 Batch megakernel (the `ma_advanced.rs` model)
- The cleanest end-state: a **single fused-OHLC megakernel / graph** that uploads `high/low/close/volume` once, computes the shared intermediates (TR, TP, EMA, prefix sums, rolling extrema) once, and writes all requested indicator outputs — exactly the `[num_series, series_len]` batch layout `ma_advanced.rs` already uses (`ma_advanced.rs:25-31`). This is opportunities #2+#3+#4+#5 converged into one architecture.

---

## Top 3 (summary)

1. **Wire the already-built `scan.rs` device-resident Wilder/EMA scan into RSI/ADX/MFI/ATR/Supertrend** — it's referenced by nobody (`grep scan::` → only `scan.rs`) while those indicators still round-trip smoothing to the CPU (ADX syncs+transfers at `adx.rs:381/477`; the disabled fused path is `rsi_fused.rs:74`). High impact, low effort, zero new kernel math.
2. **Convert the 15 f64 indicator kernels to f32** — only `ma_advanced.rs` and `scan.rs` use f32, yet both already document that f64 on sm_89 runs at 1/64 throughput and that f32 is numerically safe for these patterns (`ma_advanced.rs:34-41`).
3. **Replace O(n·period) sliding windows with the existing prefix/pair-sum scans**, starting with the shared Typical-Price/windowed-SMA fusion for CCI+Bollinger+CMF+MFI (Bollinger walks the window twice at `bollinger.rs:45,59`; CCI recomputes TP `period`× at `cci.rs:61`).
