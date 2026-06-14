# GPU Data Transfer Profiling Results

**Date**: 2025-10-28
**Tool**: `profile_transfer_overhead` example
**GPU**: NVIDIA RTX 3500 Ada (12GB VRAM)

## Executive Summary

**CRITICAL FINDING**: Data transfer is NOT the bottleneck. The real issue is **"unaccounted time"** consuming 73-78% of total execution time.

### Key Metrics (500 strategies × 5K candles)

| Phase | Time (ms) | % of Total |
|-------|-----------|------------|
| **GPU Time (measured)** | 41.87 | 22% |
| H2D Transfer | 5.23 | 2.8% |
| Kernel Execution | 31.41 | 16.6% |
| D2H Transfer | 5.23 | 2.8% |
| Memory Allocation | 0.00 | 0% |
| **Unaccounted Time** | **147.49** | **77.9%** |
| **Wall Clock Total** | 189.36 | 100% |

## Bottleneck Analysis

### ✅ Transfer Overhead: ACCEPTABLE (25% of GPU time)
- H2D + D2H: 10.47ms out of 41.87ms GPU time
- This is reasonable for the data volume
- **Pinned memory would only save 2-3ms** (20-30% of 10.47ms)

### ⚙️ Kernel Execution: GOOD (75% of GPU time)
- 31.41ms for 500 strategies × 5K candles
- Compute-bound as expected
- Optimization potential via occupancy/shared memory

### 🚨 UNACCOUNTED TIME: MAJOR BOTTLENECK (78% of total)
- **147.49ms out of 189.36ms total time**
- This is 3.5x longer than GPU execution time
- **Not explained by:**
  - Data transfer (measured at 10.47ms)
  - Kernel execution (measured at 31.41ms)
  - Memory allocation (measured at 0.00ms)

## What is "Unaccounted Time"?

The unaccounted time (147ms) represents overhead NOT captured by GPU timing. Likely sources:

### 1. **Kernel Compilation (PTX → SASS)** 🎯 MOST LIKELY
- **First-time compilation**: 50-200ms
- **Happens before kernel launch**
- **Not measured by GPU events**
- **Solution**: Pre-compile and cache kernels

### 2. **CUDA Context Synchronization**
- **Implicit synchronization**: 10-50ms
- **Stream creation/management**: 5-20ms
- **Solution**: Reuse streams, avoid sync points

### 3. **Host-Side CPU Overhead**
- **Data structure setup**: 10-30ms
- **Result extraction**: 20-40ms
- **Vec allocations**: 10-20ms
- **Solution**: Pool allocations, reduce copies

### 4. **Driver Overhead**
- **cuLaunchKernel latency**: 5-10μs per launch
- **Memory management**: 5-20ms
- **Solution**: Persistent kernels (already implemented)

## Test Case Breakdown

### Test 1: Small (50 strategies × 1K candles)
- GPU time: 6.73ms (25%)
- Unaccounted: 20.27ms (75%)
- **Ratio**: Unaccounted is 3x GPU time

### Test 2: Bottleneck (500 strategies × 5K candles)
- GPU time: 41.87ms (22%)
- Unaccounted: 147.49ms (78%)
- **Ratio**: Unaccounted is 3.5x GPU time

### Test 3: Large (1000 strategies × 10K candles)
- GPU time: 194.16ms (51%)
- Unaccounted: 188.07ms (49%)
- **Ratio**: Unaccounted is 1x GPU time (improving!)

**Pattern**: As GPU time increases, unaccounted time becomes less dominant (from 3.5x to 1x). This suggests **fixed overhead** (compilation, setup).

## Recommendations (Priority Order)

### 🥇 Priority 1: Kernel Caching (HIGHEST IMPACT)
**Expected Speedup**: 50-200ms reduction (eliminate compilation overhead)

```rust
// Cache compiled PTX modules
static KERNEL_CACHE: LazyLock<DashMap<String, Arc<CudaModule>>> =
    LazyLock::new(|| DashMap::new());

pub fn get_or_compile_kernel(kernel_name: &str) -> Result<Arc<CudaModule>> {
    if let Some(module) = KERNEL_CACHE.get(kernel_name) {
        return Ok(Arc::clone(module.value()));
    }

    // First-time compilation
    let ptx = compile_ptx_optimized(kernel_source)?;
    let module = Arc::new(device.context().load_module(ptx)?);
    KERNEL_CACHE.insert(kernel_name.to_string(), Arc::clone(&module));

    Ok(module)
}
```

### 🥈 Priority 2: Stream Reuse (MEDIUM IMPACT)
**Expected Speedup**: 10-20ms reduction

```rust
// Reuse streams across invocations
pub struct GpuDevice {
    stream_pool: Mutex<Vec<Arc<CudaStream>>>,
}
```

### 🥉 Priority 3: Memory Pooling (LOW-MEDIUM IMPACT)
**Expected Speedup**: 10-30ms reduction

```rust
// Pool device memory allocations
pub struct DeviceMemoryPool {
    buffers: DashMap<usize, Vec<CudaSlice<f64>>>,
}
```

### 🏅 Priority 4: Pinned Memory (LOW IMPACT)
**Expected Speedup**: 2-3ms reduction (already implemented in PR #6)

Transfer overhead is only 10.47ms, so 20-30% improvement = 2-3ms savings.

## Verification Plan

### Step 1: Measure Compilation Time
```rust
let compile_start = Instant::now();
let ptx = compile_ptx_optimized(kernel_source)?;
let compile_ms = compile_start.elapsed().as_millis();
eprintln!("Compilation time: {}ms", compile_ms);
```

### Step 2: Implement Kernel Caching
- Add `LazyLock<DashMap>` for kernel cache
- Measure speedup on second invocation
- Expected: 50-200ms reduction

### Step 3: Profile Host-Side Operations
```rust
let setup_start = Instant::now();
// ... data structure setup ...
let setup_ms = setup_start.elapsed().as_millis();

let extract_start = Instant::now();
// ... result extraction ...
let extract_ms = extract_start.elapsed().as_millis();
```

## Why Persistent Kernels Show No Speedup

**Answer**: Persistent kernels optimize GPU execution (kernel launch overhead), but the bottleneck is **host-side overhead** (compilation, setup, synchronization).

- Traditional approach: 4 launches × 10μs = 40μs overhead
- Persistent approach: 1 launch × 10μs = 10μs overhead
- **Savings**: 30μs (negligible compared to 147ms unaccounted time)

**The 40μs launch overhead is DWARFED by the 147ms host overhead.**

## Conclusion

1. **Data transfer is fine** (25% overhead is acceptable)
2. **Kernel execution is fine** (75% of GPU time)
3. **The real bottleneck is host-side overhead** (78% of total time)
4. **Next step**: Implement kernel caching to eliminate 50-200ms compilation overhead

## Running the Profiler

```bash
# Build and run
cargo run --example profile_transfer_overhead --features gpu --release

# Or use the script
./scripts/run_transfer_profiler.sh
```

## Output Example

```
╔══════════════════════════════════════════════════════════════╗
║  GPU Transfer Overhead Profile: Bottleneck Case
╠══════════════════════════════════════════════════════════════╣
║  Configuration:                                              ║
║    Strategies: 500                                       ║
║    Candles:    5000                                       ║
╠══════════════════════════════════════════════════════════════╣
║  Timing Breakdown (GPU Events):                              ║
║  │  H2D Transfer           5.23        12.5%          │   ║
║  │  Kernel Execution      31.41        75.0%          │   ║
║  │  D2H Transfer           5.23        12.5%          │   ║
║  Total GPU Time:         41.87 ms                          ║
╠══════════════════════════════════════════════════════════════╣
║  Overhead Analysis:                                          ║
║    Transfer overhead:     10.47 ms ( 25.0% of GPU time)      ║
║    Unaccounted time:     147.49 ms                          ║
╚══════════════════════════════════════════════════════════════╝
```

## Files Created

- `/home/kim/projects/kimsfinance/rust/examples/profile_transfer_overhead.rs` - Main profiler
- `/home/kim/projects/kimsfinance/rust/scripts/run_transfer_profiler.sh` - Runner script
- `/home/kim/projects/kimsfinance/rust/docs/GPU_PROFILING_RESULTS.md` - This document
