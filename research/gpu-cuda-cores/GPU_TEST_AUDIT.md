# GPU Test Suite Audit (RTX 3500 Ada, CUDA 13.1) — 2026-06-14

Ran the full hardware GPU test suite (`cargo test --release --features gpu --lib -- --ignored`).
**Result: 279 passed, 47 FAILED.** These never run in CI (CI is GPU-less — "Gap 2"),
so the failures accumulated undetected. Almost all are PRE-EXISTING (in files this
optimization work never touched); only the 2 sma 2D/3D sweep tests were regressions
from the f32 promotion, already fixed (commit 95d5b83).

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
