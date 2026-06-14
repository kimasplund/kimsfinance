# GPU Execution Infrastructure — Codebase Audit (batch / streams / graphs / pinned / persistent)

**Scope:** The GPU *execution plumbing* under `rust/src/gpu/` that any unified
"combined core" / megakernel would build on. Read-only analysis. Target HW:
RTX 3500 Ada (sm_89, 12 GB, 32 MB L2), CUDA 13.1, cudarc 0.17.3.

**Bottom line:** The foundation is solid on the *host-side correctness* axis
(L2 chunking with overlap-and-discard, module/PTX caching, pinned pool with
event-gated reuse, occupancy-aware cooperative launch). But almost none of the
advertised "advanced" GPU features are actually wired into a hot path:
**CUDA Graphs are entirely disabled, the async memory pool is a no-op fallback,
multi-stream dispatch is gated off, and the only real megakernel-style host
(the 4-phase `batch_backtest_genetic`) bypasses the module cache and runs every
phase serially on the default stream.** The pieces a combined core needs exist
but are disconnected.

---

## 1. Inventory of cores in scope

| File | LOC | Role | State |
|---|---:|---|---|
| `gpu/batch.rs` | 1606 | Multi-indicator batch driver: overlap-and-discard L2 chunking + per-indicator dispatch | Active, but multi-stream gated OFF |
| `gpu/batch_graphs.rs` | 202 | CUDA-Graph batch executor wrapper | **DISABLED** — `calculate_batch` always errors |
| `gpu/cuda_graphs.rs` | 732 | Per-stream graph capture/instantiate/launch (`IndicatorGraph[Builder]`) | Implemented API, **never called in prod** |
| `gpu/device.rs` | 1174 | `GpuDevice`: context, default stream, pinned pool, async allocator, **module cache**, alloc/copy helpers, occupancy query, compute-cap query | Active, central |
| `gpu/async_alloc.rs` | 535 | `AsyncAllocator` (cudaMallocAsync pool) | Pool **created but never used** — every alloc falls back to `alloc_zeros` |
| `gpu/async_transfers.rs` | 547 | `CudaEvent` RAII + `AsyncTransferExt` (htod/dtoh async + event) | Active primitives; thin usage |
| `gpu/compile.rs` | 658 | NVRTC compile (sm_89, fast-math) + SHA-256 PTX cache | Active, mature |
| `gpu/auto_select.rs` | 424 | CPU/GPU engine selector for *trade aggregation* only | Active, narrow scope |
| `gpu/streams.rs` | 577 | `StreamManager` (3 speed-classed streams) + global singleton | Active; only 1 of 3 streams used in batch |
| `gpu/persistent/mod.rs` | 1426 | Persistent cooperative-groups kernel (ROC), `TaskBatch`, buffer mgmt, occupancy launch | Active for persistent path; ROC-only real kernel |
| `gpu/persistent/pinned_memory.rs` | 1202 | `PinnedBuffer` + tiered `PinnedBufferPool` w/ event-gated reuse + RAII guard | Mature; **device pool doesn't use tiers** |
| `gpu/persistent/generic.rs` | 514 | Multi-in/multi-out generic persistent batch | Active (not deeply audited here) |
| `gpu/persistent/occupancy.rs` | 290 | `OccupancyCalculator` (cuOccupancyMaxActiveBlocks) | Active, correct |
| `gpu/persistent/traits.rs` | 102 | `PersistentIndicator` trait surface | Active |
| `gpu/mod.rs` | 784 | Module wiring + `batch_backtest_genetic` (4-phase pipeline) | Active; the de-facto "megakernel host" |

Supporting (referenced, not in primary scope): `l2_cache.rs`, `triple_buffer.rs`,
`timing.rs`, `memory_pool.rs`, `tick_backtest_batch.rs`, `orderflow_batch.rs`.

---

## 2. Current optimization state (with evidence)

### 2.1 Precision / compile flags — GOOD
- `compile.rs:127-172` `get_compile_options()`: `arch` auto-detected (`compute_89`
  default, `detect_gpu_arch` at `:80` via driver query, env override
  `KIMSFINANCE_GPU_ARCH`), `use_fast_math=true`, `ftz=true`, `prec_sqrt=false`,
  `prec_div=false`, unlimited registers. Appropriate for f64 financial data.
- PTX cache keyed by SHA-256 (`compile.rs:226-348`), `Arc<Ptx>` zero-copy
  sharing; failed compiles not cached (`:632`). Mature.
- **Precision gap for a combined core:** everything is **f64**. The indicator
  kernels, `BatchIndicatorParams`, `IndicatorResult` (`batch.rs:122-131`) are
  all `Array1<f64>`. FP8/WMMA paths exist separately (`fp8_wmma.rs`,
  `fp8_gemm_cutlass.rs`, `quantization.rs`) but are **not** integrated into the
  batch/streams infra. A megakernel targeting sm_89 throughput would want
  f32/tensor paths; none are plumbed here.

### 2.2 Memory pattern — MIXED
- **Module cache (good):** `device.rs:38,165-207` `get_or_load_function` caches
  `Arc<CudaModule>` per (device, source-hash), avoiding ~0.1–1 ms
  `cuModuleLoadData` per call. This is the right primitive.
- **But the megakernel host bypasses it:** `mod.rs:585-621` `batch_backtest_genetic`
  does `Arc::unwrap_or_clone(ptx)` + `context.load_module(ptx)` + 4×
  `load_function` on **every call** (deep-clones multi-KB PTX, pays module load
  each invocation). Same anti-pattern in `persistent/mod.rs:456-477`
  (`compile_persistent_kernel` → `Arc::unwrap_or_clone` + `load_module`).
- **alloc_zeros everywhere (extra memset pass):** `device.rs:219-223` `alloc_buffer`
  and `:694-703` `allocate_device_buffer` both `alloc_zeros` → a full
  `cudaMemsetAsync` over each buffer. `alloc_uninit` (`:253-268`) exists to skip
  it but is barely used; `batch_backtest_genetic` uses `alloc_async`/
  `allocate_device_buffer` (zeroed) for output buffers that are fully
  overwritten (`mod.rs:565-578`).
- **Per-indicator private H2D copies (no input sharing):** `batch.rs:681-685`
  doc admits indicators "upload their own input copies internally", so a batch
  of N indicators re-uploads HLC N times. Device-resident input sharing via a
  memory pool is explicitly deferred (`batch.rs:683-684`).
- **Pinned pool (good but underused):** `device.rs:73` constructs
  `PinnedBufferPool::new(16, 1_000_000)` — the **flat** constructor, **not**
  `with_default_tiers()` (`pinned_memory.rs:493`), so the device pool can't serve
  >1M-element requests from the pool and falls to one-off oversize allocs
  (`pinned_memory.rs:524-541`). The tiered design exists but isn't used by the
  central device.

### 2.3 Fusion — MINIMAL
- **No kernel fusion in the indicator batch.** `batch.rs:698-721` loops over
  indicators, each calling its own `*_gpu` function (separate H2D + kernel(s) +
  D2H + per-indicator sync). Temporal-locality chunking keeps HLC hot in L2
  (`batch.rs:629-668`) but each indicator is still its own launch set.
- **The one real fused pipeline is `batch_backtest_genetic` (`mod.rs:504-784`):**
  4 phases (indicators → signals → execution → metrics) over a flattened
  `[O,H,L,C,V]` buffer, all strategies in one grid. This is the closest existing
  "combined core" but the 4 kernels are *separate launches on the default
  stream* — fused in data layout, not in execution.
- MACD is routed to CPU (`batch.rs:219,525-533` `macd_hybrid`), an asymmetry a
  unified GPU core would need to resolve.

### 2.4 Persistent kernels / cooperative launch — PARTIAL
- `persistent/mod.rs` implements a genuine persistent kernel
  (`PERSISTENT_ROC_KERNEL`, `:243-284`) using cooperative groups + `grid.sync()`,
  launched via `launch_cooperative` (`:844-857`) with occupancy-sized grid
  (`occupancy.rs`). This is the strongest megakernel-ready building block.
- **But the only real kernel is ROC.** `PERSISTENT_ROC_KERNEL` is hardcoded; the
  `TaskBatch<I>` generic exists and `execute_batch` calls `I::compile_kernel`
  (`:973`), but `PersistentKernelManager::execute_batch` (`:377-409`) always
  compiles the ROC kernel via `compile_persistent_kernel`.
- **Param packing is unsafe/fragile:** `persistent/mod.rs:647-661` transmutes
  `I::Params` to `i32` (`transmute_copy`), explicitly noting it only works for
  i32-param indicators and "copies first 4 bytes (fast_period)" for MACD. A
  combined core cannot rely on this.
- **Wasted device construction:** `persistent/mod.rs:347-348`
  `PersistentKernelManager::new` queries SM count on `device`, then stores
  `Arc::new(GpuDevice::with_device_id(0)?)` — a **second full device** (new
  pinned pool, ~128 MB) — discarding the passed-in one.

### 2.5 CUDA Graphs — DISABLED (dead weight)
- `batch_graphs.rs:91-115` `BatchGraphExecutor::calculate_batch` validates inputs
  then **always returns `ComputationErrorStatic(GRAPH_REPLAY_DISABLED_MSG)`**.
  The module header (`:1-20`) documents why: captured graphs never wired result
  buffers, so replay recomputed everything → net-negative; and a sorted-key vs
  unsorted-key lookup guaranteed a panic.
- `cuda_graphs.rs` (732 LOC) has a *fully implemented* per-stream capture API
  (`begin_capture_stream`/`end_capture_stream`/`launch_stream`/`launch_all`,
  `:210-409`) but grep confirms **zero production callers** — it's referenced
  only by its own `#[ignore]` GPU tests. Much of the file (`:452-559`
  `optimization_guide`) is prose/constants, not code.
- Net: CUDA Graphs — the single biggest launch-overhead win for a
  many-small-kernel megakernel — are present in name only.

### 2.6 Streams — UNDERUTILIZED
- `streams.rs` builds 3 non-blocking streams (Fast/Medium/Slow) and a process
  global (`:63,168-179`). Classification logic is thorough (`:216-258`).
- **But batch uses one stream.** `batch.rs:65` `ENABLE_MULTI_STREAM_DISPATCH=false`
  and `:701-708` forces every indicator onto the **Medium** stream. The comment
  (`:56-65`) explains: the shared pinned pool isn't event-gated at the batch
  layer, so cross-stream reuse could corrupt in-flight transfers. So the 3-stream
  concurrency is paid for (creation) but not realized.
- `batch_backtest_genetic` (`mod.rs`) and `persistent` both run on
  `device.stream` (the default stream) exclusively — no overlap of H2D / compute
  / D2H.

### 2.7 Async transfers / events — PRIMITIVES OK, PIPELINE THIN
- `async_transfers.rs` provides correct `CudaEvent` RAII (`:104-319`, no-timing
  variant `:133`) and `htod/dtoh_async_pinned` (`:391-435`). These are exactly
  what an event-gated megakernel pipeline needs.
- `pinned_memory.rs` has the matching `release_with_event` + `sweep_pending`
  (`:640-655`) and host-testable `partition_completed` (`:303-316`). Good.
- **Gap:** the batch driver never uses event-gated release (that's why
  multi-stream is disabled). `triple_buffer.rs` exists but isn't part of the
  batch path.

### 2.8 Async allocator — NO-OP
- `async_alloc.rs:361-371` `alloc_async` ignores the created pool and calls
  `self.stream.alloc_zeros` — i.e. the cudaMallocAsync pool (`create_memory_pool`
  `:249-286`) is created and destroyed but **never allocates from**. Confirmed by
  the doc at `:347-359`. So `device.alloc_async` (`device.rs:295-302`) gives zero
  speedup; `alloc_stream_ordered` (`device.rs:461-470`) is an explicit
  placeholder returning `alloc_buffer`.

### 2.9 Launch pattern — SERIAL, DEFAULT-STREAM
- Megakernel host (`mod.rs:643-742`): four `launch_builder(...).launch(cfg)` calls
  back-to-back on `device.stream`, single `synchronize()` at end (`:746`). No
  graph, no events, no stream overlap. Grid dims are correct/3D where useful
  (`:628-636`), block sizes hand-tuned per phase (`:625,685,724`).

---

## 3. Ranked optimization opportunities

Ranked by (Impact × inverse-Effort). "Impact" is relative to a future combined
core / megakernel that issues many launches per call.

### HIGH impact

1. **Wire CUDA Graphs into a real replay path (capture once, replay N).**
   *Impact: High · Effort: High.* The infra (`cuda_graphs.rs`) is built but
   dead; `batch_graphs.rs` is disabled precisely because captured graphs never
   wired result buffers. For repetitive workloads (parameter sweeps, genetic
   backtests, per-bar live loops) graph replay cuts launch overhead ~50–70%
   (`cuda_graphs.rs:5-15`). The combined core is the natural place to do capture
   correctly (stable buffers, fixed kernel sequence). *Prereq:* result-buffer
   binding + stable device pointers across replays.

2. **Make `batch_backtest_genetic` use the module cache + skip redundant zeroing
   + (optionally) capture as a graph.** *Impact: High · Effort: Low–Med.*
   `mod.rs:585-621` reloads PTX/modules every call; switch to
   `device.get_or_load_function` (already exists, `device.rs:165`). Replace
   `alloc_zeros` outputs with `alloc_uninit` (`device.rs:253`) for buffers fully
   overwritten by kernels. This is the de-facto combined core today; the wins
   compound on every invocation.

3. **Enable multi-stream dispatch in `batch.rs` via event-gated pinned release.**
   *Impact: High · Effort: Med.* Flip `ENABLE_MULTI_STREAM_DISPATCH`
   (`batch.rs:65`) after switching indicator H2D staging to
   `release_with_event` (the primitive exists, `pinned_memory.rs:640`). Unlocks
   the 3 already-created streams for Fast/Med/Slow overlap (15–30% per the
   module's own claim, `streams.rs:5`).

### MEDIUM impact

4. **Share device-resident HLC inputs across the indicator batch.** *Impact: Med
   · Effort: Med–High.* `batch.rs:683-684` admits each indicator re-uploads its
   own HLC copy. Upload once per chunk, pass `CudaSlice` views to indicators.
   Requires signature changes across `*_gpu` functions — the reason it's
   deferred. A megakernel sidesteps this by design (single shared input buffer,
   like `batch_backtest_genetic`'s flattened OHLCV).

5. **Use the tiered pinned pool for the central device.** *Impact: Med · Effort:
   Low.* `device.rs:73` calls `PinnedBufferPool::new(16, 1M)`; switch to
   `with_default_tiers()` (`pinned_memory.rs:98-99,493`) so 1M–8M-element
   requests (3D sweeps, reductions) hit the pool instead of one-off oversize
   allocs (`pinned_memory.rs:524-541`).

6. **Fix `PersistentKernelManager::new` double-device allocation.** *Impact: Med
   (startup/memory) · Effort: Low.* `persistent/mod.rs:347-348` builds a second
   `GpuDevice` (new ~128 MB pinned pool) and discards the queried one. Store the
   passed device (or use `GpuDevice::global()`).

7. **Replace the unsafe `transmute_copy::<Params,i32>` param packing.** *Impact:
   Med (correctness for a unified core) · Effort: Med.* `persistent/mod.rs:647-661`
   silently truncates multi-field params (MACD). A combined core needs a typed
   param-marshalling path (struct-of-arrays per indicator).

### LOW impact

8. **Implement real `cudaMallocAsync` or delete the dead pool.** *Impact: Low
   (1.1–1.5x alloc, alloc is ~10–15% of total) · Effort: High (blocked on
   cudarc `from_raw`).* `async_alloc.rs:361-371` is a no-op; either drop the
   pool/threshold machinery or implement via raw FFI. Low priority — a megakernel
   amortizes allocation anyway.

9. **Prune dead/placeholder code for clarity.** *Impact: Low · Effort: Low.*
   `alloc_stream_ordered`/`free_stream_ordered` placeholders (`device.rs:461-503`),
   `cuda_graphs::optimization_guide` prose constants, the `.bak`/`.orig`/`.backup`
   files littering `gpu/` (e.g. `heston_pricing.rs.orig`, 20+ `*.bak`).

---

## 4. Fusion / combine potential (megakernel readiness)

**What can share a megakernel or single capture:**

- **The 4-phase backtest (`mod.rs:504-784`) is already a fused-data pipeline** —
  indicators→signals→execution→metrics over one `[O,H,L,C,V]` buffer, gridded by
  strategy. It is the strongest candidate to become *the* combined core: fuse
  phases 1–2 (indicator+signal are both per-(strategy,candle) and memory-bound),
  keep phase 3 (sequential per-strategy) and phase 4 (reduction) as cooperative
  sections with `grid.sync()` — exactly the pattern `PERSISTENT_ROC_KERNEL`
  already demonstrates (`persistent/mod.rs:250-283`). Then capture the whole
  thing as one CUDA Graph for sweep/GA replay.

- **The indicator batch (`batch.rs`) can be a persistent megakernel.** Today it's
  N independent launches sharing L2. The `persistent` infra (cooperative launch,
  occupancy sizing, grid-stride task loop) is the right host; the blocker is that
  only ROC has a real persistent kernel and param packing is unsafe. A unified
  kernel with a task-descriptor table (per task: indicator id, period(s),
  in/out pointers) plus a `switch` in the grid-stride loop would collapse N
  launches → 1. Fast indicators (ROC, Williams %R, CCI — `streams.rs:220-230`)
  are embarrassingly parallel and fuse trivially; Wilder/EMA recursives (RSI,
  ATR, MACD) need a sequential sub-pass but can coexist behind `grid.sync()`.

- **Pinned-pool + event infra is megakernel-ready** (`pinned_memory.rs` tiers +
  `release_with_event`; `async_transfers.rs` events). A combined core can run a
  triple-buffered H2D→megakernel→D2H pipeline across the 3 existing streams once
  `ENABLE_MULTI_STREAM_DISPATCH` is unblocked.

**What should stay separate:** trade aggregation (`auto_select.rs` / `aggregation.rs`)
and tick aggregation operate on raw trades, not OHLCV indicator arrays — different
input shape; keep as a pre-stage feeding the megakernel.

**Hard prerequisites before a megakernel can be hosted here:**
1. Result-buffer binding for graph capture (the documented reason graphs are
   disabled — `batch_graphs.rs:8-16`).
2. A typed param-marshalling layer to replace `transmute_copy` (`persistent/mod.rs:655`).
3. Event-gated pinned staging in the batch path to unblock multi-stream
   (`batch.rs:56-65`).
4. Module-cache adoption + `alloc_uninit` in the backtest host (`mod.rs:585`,
   `:565`) so per-call overhead doesn't dominate replay savings.

---

## Appendix — load-bearing evidence pointers

- Graphs disabled: `batch_graphs.rs:91-115` (`calculate_batch` → always errors);
  `GRAPH_REPLAY_DISABLED_MSG` `:31-34`.
- Graph API exists but unused: `cuda_graphs.rs:210-409`; zero non-test callers
  (grep across `rust/src`).
- Multi-stream gated: `batch.rs:65` (`const ENABLE_MULTI_STREAM_DISPATCH=false`),
  forced Medium stream `:701-708`.
- Async pool is a no-op: `async_alloc.rs:361-371`; `device.rs:461-470`
  (`alloc_stream_ordered` placeholder).
- Megakernel host bypasses module cache: `mod.rs:585-621`; redundant zeroing
  `mod.rs:565-578`.
- Double device alloc: `persistent/mod.rs:347-348`.
- Unsafe param packing: `persistent/mod.rs:647-661`.
- Device pinned pool not tiered: `device.rs:73` vs `pinned_memory.rs:493`.
- Real persistent/cooperative kernel (the good pattern): `persistent/mod.rs:250-283,844-857`.
- f64-only result types: `batch.rs:122-131`.
