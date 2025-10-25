# Persistent Kernels - Quick Reference Card

## Problem
**Current**: N indicator calculations = N kernel launches × ~10μs = ~100μs overhead for 10 indicators  
**Goal**: 1 kernel launch = ~10μs overhead total (90% reduction)

## Solution
**Persistent kernels** with **cooperative groups** - launch once, process multiple tasks in loop.

---

## Files Created

```
rust/
├── src/gpu/persistent.rs              (219 lines) - Infrastructure
├── benches/launch_overhead.rs         (86 lines)  - Benchmark
├── docs/
│   ├── persistent-kernels-implementation.md  - Full report
│   ├── persistent-kernel-patterns.md         - Pattern library
│   ├── PERSISTENT_KERNELS_SUMMARY.md         - Executive summary
│   └── PERSISTENT_KERNELS_QUICKREF.md        - This file
```

## Usage (Future - Phase 2)

```rust
use kimsfinance_core::gpu::{PersistentKernelManager, TaskBatch};

// Create manager
let device = GpuDevice::new()?;
let manager = PersistentKernelManager::new(&device)?;

// Create batch of tasks (e.g., RSI with different periods)
let mut batch = TaskBatch::new();
batch.add_task(close_prices.clone(), 14);  // RSI(14)
batch.add_task(close_prices.clone(), 21);  // RSI(21)
batch.add_task(close_prices.clone(), 28);  // RSI(28)

// Execute all tasks with single kernel launch
let results = manager.execute_batch(&batch)?;
// Single launch instead of 3!
```

## Expected Performance

| Batch Size | Traditional | Persistent | Speedup |
|------------|-------------|------------|---------|
| 1 task     | ~10μs       | ~10μs      | 1.0x    |
| 10 tasks   | ~100μs      | ~10μs      | 10.0x   |
| 100 tasks  | ~1ms        | ~10μs      | 100x    |

**Crossover**: Use persistent for N ≥ 5 tasks

## Status

- ✅ **Phase 1**: Infrastructure complete (compiles, ready)
- ⏳ **Phase 2**: Simple prototype (4-7 hours to implement `execute_batch()`)
- ⏳ **Phase 3**: Multi-indicator integration (8-15 hours)

## Next Step

Implement `execute_batch()` in `PersistentKernelManager`:
1. Compile persistent ROC kernel
2. Allocate GPU memory for all tasks
3. Create pointer arrays (double**)
4. Launch with cooperative groups
5. Copy results back

**Estimated time**: 4-7 hours  
**Confidence**: High (90%+)

## Benchmark

```bash
# Run launch overhead benchmark (when Phase 2 complete)
cargo bench --features gpu --bench launch_overhead

# Output will show:
# - Traditional: N × ~10μs per N launches
# - Persistent: ~10μs total (future)
# - Overhead reduction: 50-90%
```

## Key Files to Review

1. **Implementation**: `rust/src/gpu/persistent.rs`
2. **Patterns**: `rust/docs/persistent-kernel-patterns.md`
3. **Full Report**: `rust/docs/persistent-kernels-implementation.md`

## CUDA Pattern

```cuda
#include <cooperative_groups.h>

extern "C" __global__ void persistent_kernel(
    const double** inputs, double** outputs, 
    const int* sizes, int num_tasks
) {
    cg::grid_group grid = cg::this_grid();
    
    for (int task = 0; task < num_tasks; task++) {
        // Process task data
        for (int idx = global_tid; idx < sizes[task]; idx += grid_size) {
            outputs[task][idx] = compute(inputs[task][idx]);
        }
        grid.sync();  // Synchronize before next task
    }
}
```

## Trade-offs

✅ **Benefits**: 50-90% launch overhead reduction, better GPU utilization  
❌ **Costs**: More complex kernel, grid size constraints, requires modern GPU

## Requirements

- ✅ cudarc 0.17.3 (confirmed)
- ✅ CUDA 12.4 (confirmed)
- ✅ RTX 3500 Ada - Compute 8.9 (supports cooperative groups)
- ⏳ Implement `launch_cooperative()` (Phase 2)

---

**Last Updated**: 2025-10-25  
**Status**: Phase 1 Complete, Ready for Phase 2
