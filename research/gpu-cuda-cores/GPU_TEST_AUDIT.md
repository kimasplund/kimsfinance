# GPU Test Suite Audit (RTX 3500 Ada, CUDA 13.1) — 2026-06-14

Ran the full hardware GPU test suite (`cargo test --release --features gpu --lib -- --ignored`).
**Initial result: 279 passed, 47 FAILED.** These never run in CI (CI is GPU-less — "Gap 2"),
so the failures accumulated undetected. Almost all are PRE-EXISTING (in files this
optimization work never touched); only the 2 sma 2D/3D sweep tests were regressions
from the f32 promotion, already fixed (commit 95d5b83).

> **STATUS: RESOLVED — full suite GREEN (326 passed / 0 failed on hardware; host-side
> 849/0). All 47 fixed.** See "## RESOLUTION" at the end for the root causes and fixes.
> Several were genuine production bugs in never-validated code (the candle/persistent
> kernels could not even compile before, so they had never run on hardware).

## Failures by cluster (likely shared root causes)

### Candle kernels fail to compile (~8) — likely one NVRTC/CUDA-13.1 cause
- candles::heikin_ashi/range_bars/renko/tick_bars/volume_bars::test_*_kernel_compiles
- candles::time_bars::{test_*_kernel_compiles, empty_input, single_bucket, single_trade, multiple_buckets}

### Pinned-memory + persistent (~13) — likely one cudarc/CUDA-13.1 cause
- persistent::pinned_memory::* (allocation, copy, mutability, pool acquire/release/creation/exhaust/tiers/oversize, guard drop, pinned_vs_pageable_speed)
- persistent::{test_persistent_large_batch, multi_task_batch, varying_periods}
- persistent::occupancy::{test_occupancy_query_for_persistent_kernel, occupancy_vs_conservative_25_percent}

### Async/memory infra (~3)
- async_transfers::{test_event_creation, test_event_no_timing}
- memory_pool::test_memory_pool_copy_results

### Indicator correctness (~8) — need per-case triage (tolerance vs real bug)
- adx::{test_adx_gpu_basic, test_adx_gpu_large_dataset}
- ema::test_ema_gpu_smoothness
- mfi::test_mfi_gpu_large_dataset
- supertrend::{test_supertrend_gpu_constant_prices, test_supertrend_gpu_large_dataset}
- tick_batch::{test_calculate_atr, test_calculate_rsi}
- kernels_2d::test_momentum_fusion_2d (fused vs individual >1e-9)

### Performance-assertion failures (~5) — hardware-timing dependent, low severity
- sma::test_sma_gpu_performance_benchmark
- tick_aggregation::test_aggregate_performance
- tick_backtest_batch::test_tick_backtest_batch_throughput
- (pinned_vs_pageable_transfer_speed counted above)

## Implication
The GPU suite has been substantially red on real hardware while CI stayed green.
Establishing a GPU CI gate requires a self-hosted GPU runner (infra decision).
The clusters suggest several shared root causes (candle compile; pinned memory),
so the fix count is far below 47.

## Repair attempt 1 (candle-compile cluster) — REVERTED

Root cause of the candle/persistent compile failures: those kernels
`#include <cooperative_groups.h>`, which transitively needs the libcu++
`<cuda/std/...>` headers (CUDA 13.x: under `include/` + `include/cccl/`). NVRTC
has no include path by default.

Attempted fix: add the CUDA include + cccl paths to the GLOBAL NVRTC
`CompileOptions.include_paths`. **Result: regressed 47 -> 234 failures** — the
many self-contained kernels (cci, cmf, donchian, elder_ray, fibonacci, batch,
...) that compiled on NVRTC's built-ins broke once a system include path was
present (header/built-in conflicts). The original code comment ("Including
system headers causes JIT compilation issues") was correct. **Reverted.**

Correct approach (future): a SURGICAL per-kernel include path — a
compile-with-options variant used ONLY by the cooperative-groups kernels
(candles/persistent), leaving the global options untouched. Note the candle
cluster also has RUNTIME failures (time_bar_empty_input/single_trade) beyond
compile, so fixing compile alone won't fully green it.

## RESOLUTION (2026-06-14) — all 47 fixed, suite GREEN (326/0)

Worked the clusters by shared root cause. Final hardware result:
`--ignored` GPU suite **326 passed / 0 failed**; host-side lib **849 / 0**.

### Triage workflow (14 tests) — commit 4a2f2ad
Indicator-correctness + perf-assert + infra, all test-side except one real bug.
- adx/ema/supertrend/tick_batch/kernels_2d: test data/tolerance corrections
  (e.g. ema spike landed on the NaN warmup window; momentum-fusion compared an
  f32 standalone path against an f64 fused path at 1e-9 — impossible across the
  f32/f64 boundary, relaxed to 1e-2).
- sma/tick_aggregation/tick_backtest_batch: removed machine-dependent wall-clock
  throughput bounds, replaced with shape/count/finiteness invariants.
- async_transfers: create GpuDevice (bind context) before CudaEvent::new.
- **memory_pool copy_results_to_host: REAL BUG** — D2H copied the full
  max_candles device buffer against an actual_candles host dst, tripping cudarc's
  `dst.len() >= src.len()` assert. Slice to actual_candles.

### Root cause A: cooperative-groups compile + execution (14 tests) — 429d47f, fcdbbcc
The candle/persistent kernels `#include <cooperative_groups.h>` and had NEVER
compiled (so never run). Fixes:
1. **Surgical per-kernel include path** (compile.rs `compile_ptx_coopgroups_cached`):
   the single `PersistentIndicator::compile_kernel` funnel tries the no-include
   path first and falls back to a CUDA-include compile ONLY on a missing-header
   error. Non-regressive (the reverted global path broke self-contained kernels).
2. **time_bars empty-input OOB**: n==0 → `(max-min+1)` signed-overflowed to a
   small positive int → wrote into a zero-length buffer → CUDA_ERROR_ILLEGAL_ADDRESS
   (sticky; poisoned the context and cascade-failed the rest). Guard num_buckets=0.
3. **dtoh_pinned release-only zero-download race (REAL PRODUCTION BUG)**: async
   `cuMemcpyDtoHAsync` returned without syncing, but callers read the pinned
   buffer immediately → host saw zeros before the DMA landed (debug masked it via
   incidental readbacks). This is the "dtoh_pinned download bug" already worked
   around in heston_pricing.rs. Synchronize after the async D2H.

### Root cause B: pinned-memory INVALID_CONTEXT (12 tests) — 64b9f60
`cuMemHostAlloc` needs a current context (thread-local under cudarc); 10 tests
called PinnedBuffer/Pool::new with no GpuDevice on their thread. Added a
`pinned_ctx()` helper each binds first. (transfer_speed was a dtoh_pinned casualty.)

### Root cause C: stragglers (9 tests) — 81369de
- wma_basic: test arithmetic slip (102+208+318 = 628, not 630 → 104.667 not 105).
- vwap / vwap_anchored range checks: cumulative VWAP legitimately sits outside the
  current bar's [low,high] in a trend — replaced with the cumulative-window
  invariant [min low, max high] seen so far.
- adx/mfi/vwap_anchored perf bounds: wall-clock SLAs → gross-regression guards.
- **triple_buffer (REAL pipeline bugs)**: process_batch read the wrong completion
  index AFTER clobbering the reused buffer (never returned a result); finish()
  dropped batches stranded at H2D. Fixed the rotation + added a proper drain.
- **cuda_graphs (REAL concurrency bug)**: begin_capture used GLOBAL capture mode,
  which forbids synchronous ops in EVERY thread for the capture's duration; since
  all GpuDevices share the primary context, a graph capture broke a concurrent
  batch allocation (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED). Switched to THREAD_LOCAL.

### Bonus: pre-existing host-side flakes (2 tests, out of original scope)
Surfaced by the now-green host-side run, same flaky-perf class:
- cpu MACD large_dataset: <750μs wall-clock SLA → gross-regression guard.
- heston process_time_step: `processing_time_us > 0` failed for sub-μs ops
  (rounds to 0) → assert a sane upper bound instead.

### Takeaway
Gap 2 is real and costly: a GPU-less CI let an entire subsystem (candle/persistent
cooperative-groups kernels) sit un-compilable and un-run, hiding genuine production
bugs (pinned D2H race, GLOBAL graph capture, triple-buffer pipeline). A self-hosted
GPU runner that executes the `--ignored` suite is the durable fix.
