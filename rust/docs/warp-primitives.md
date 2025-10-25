# Warp Primitive Optimizations for GPU Indicators

## Overview

This document describes the warp shuffle primitive optimizations implemented for GPU-accelerated indicators. Warp primitives leverage intra-warp communication to achieve 1.5-3x faster reductions compared to naive sequential loops.

## Warp Architecture

NVIDIA GPUs organize threads into **warps** of 32 threads that execute in lockstep (SIMT). Warp shuffle instructions enable direct register-to-register communication between threads in the same warp without using shared memory.

### Key Primitives

- `__shfl_down_sync()` - Shuffle data down within warp (for reductions)
- `__shfl_up_sync()` - Shuffle data up within warp (for prefix sums)
- `__shfl_xor_sync()` - Butterfly pattern shuffle
- `__syncwarp()` - Warp-level synchronization

## Reduction Patterns

### 1. Warp-Level Sum Reduction

```cuda
__device__ double warp_reduce_sum(double val) {
    // Tree reduction within warp (32 threads)
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // Thread 0 has final sum
}
```

**Performance**: O(log₂ 32) = 5 shuffle operations vs O(32) sequential additions

### 2. Block-Level Sum Reduction

```cuda
__device__ double block_reduce_sum(double val) {
    __shared__ double warp_sums[32];  // Max 32 warps per block
    
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // Step 1: Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Step 2: First thread in each warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // Step 3: First warp reduces warp sums
    if (warp_id == 0) {
        val = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0;
        val = warp_reduce_sum(val);
    }
    
    return val;  // Thread 0 has final result
}
```

**Performance**: For 256 threads = 8 warps:
- Warp reduction: 8 × 5 = 40 shuffles
- Final reduction: 5 shuffles
- Total: 45 operations vs 256 sequential additions (5.7x reduction)

### 3. Warp-Level Max/Min Reduction

```cuda
__device__ double warp_reduce_max(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        double other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmax(val, other);
    }
    return val;
}

__device__ double warp_reduce_min(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        double other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmin(val, other);
    }
    return val;
}
```

## Optimized Indicators

### 1. SMA (Simple Moving Average)

**Before** (naive):
```cuda
for (int j = 0; j < period; j++) {
    sum += close[idx - j];
}
```

**After** (warp-optimized):
```cuda
// Small window (<32): Use warp reduction
if (period <= 32) {
    double val = (threadIdx.x < period) ? close[idx - threadIdx.x] : 0.0;
    val = warp_reduce_sum(val);
    if (threadIdx.x == 0) sum = val;
}
```

**Expected Speedup**: 1.5-2x for period ≤ 32

### 2. Bollinger Bands

**Optimizations**:
- Warp reduction for SMA calculation
- Warp reduction for sum-of-squares (variance)

**Expected Speedup**: 2-2.5x for typical period=20

### 3. WMA (Weighted Moving Average)

**Challenge**: Non-uniform weights complicate warp reduction

**Solution**: Pre-load weights into thread registers, then reduce

```cuda
double weighted_val = close[idx - threadIdx.x] * weight[threadIdx.x];
weighted_val = warp_reduce_sum(weighted_val);
```

**Expected Speedup**: 1.8-2.2x for period ≤ 32

### 4. VWMA (Volume-Weighted Moving Average)

**Two simultaneous reductions**:
- Numerator: sum(close × volume)
- Denominator: sum(volume)

```cuda
double num = (threadIdx.x < period) ? close[idx - threadIdx.x] * volume[idx - threadIdx.x] : 0.0;
double den = (threadIdx.x < period) ? volume[idx - threadIdx.x] : 0.0;

num = warp_reduce_sum(num);
den = warp_reduce_sum(den);

if (threadIdx.x == 0 && den > 1e-10) {
    vwma[idx] = num / den;
}
```

**Expected Speedup**: 2-2.5x for typical period=14-20

### 5. Donchian Channels

**Dual max/min reductions**:

```cuda
double high_val = (threadIdx.x < period) ? high[idx - threadIdx.x] : -CUDART_INF;
double low_val = (threadIdx.x < period) ? low[idx - threadIdx.x] : CUDART_INF;

high_val = warp_reduce_max(high_val);
low_val = warp_reduce_min(low_val);
```

**Expected Speedup**: 2-3x for typical period=20

### 6. CCI (Commodity Channel Index)

**Two-pass with warp reductions**:
- Pass 1: SMA of typical price (warp reduction)
- Pass 2: Mean absolute deviation (warp reduction)

**Expected Speedup**: 1.8-2.2x overall

### 7. VWAP (Limited Optimization)

**Note**: VWAP uses cumulative sums (sequential dependency), so warp primitives provide limited benefit. The typical price calculation can be parallelized, but cumulative reduction remains sequential.

## Window Size Considerations

### Small Windows (<32 elements)

**Strategy**: Single warp reduction

**Performance**: 1.5-2x speedup

**Implementation**:
```cuda
if (period <= 32 && threadIdx.x == 0) {
    // Use warp reduction for entire window
}
```

### Medium Windows (32-512 elements)

**Strategy**: Multi-warp reduction with shared memory

**Performance**: 2-3x speedup

**Implementation**: Use block_reduce_sum() pattern

### Large Windows (>512 elements)

**Strategy**: Hybrid approach - warp reduction for partial sums + sequential aggregation

**Performance**: 1.5-2x speedup (memory bandwidth becomes bottleneck)

## Numerical Accuracy

Warp shuffle operations preserve full double-precision (64-bit) accuracy. Validation tests verify:

```rust
#[test]
fn test_warp_vs_naive_accuracy() {
    let epsilon = 1e-10;  // 64-bit precision tolerance
    assert!((warp_result - naive_result).abs() < epsilon);
}
```

## Implementation Status

| Indicator | Naive Loop | Warp Optimized | Expected Speedup | Status |
|-----------|------------|----------------|------------------|--------|
| SMA | ✓ | ✓ | 1.5-2x | Implemented |
| Bollinger | ✓ | ✓ | 2-2.5x | Implemented |
| WMA | ✓ | ✓ | 1.8-2.2x | Implemented |
| VWMA | ✓ | ✓ | 2-2.5x | Implemented |
| VWAP | ✓ | Partial | 1.2-1.5x | Partial (TP calc only) |
| Donchian | ✓ | ✓ | 2-3x | Implemented |
| CCI | ✓ | ✓ | 1.8-2.2x | Implemented |

## Benchmarking

Run benchmarks to validate speedup claims:

```bash
cargo bench --features gpu --bench moving_averages -- --baseline naive
cargo bench --features gpu --bench volatility_indicators -- --baseline naive
cargo bench --features gpu --bench volume_indicators -- --baseline naive
```

## References

- NVIDIA CUDA Programming Guide: Warp Shuffle Functions
- "Optimizing Parallel Reduction in CUDA" - Mark Harris (NVIDIA)
- cudarc 0.17.3 API documentation
