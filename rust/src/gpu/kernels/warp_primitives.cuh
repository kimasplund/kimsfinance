//! Warp-Level Primitive Operations for GPU Kernels
//!
//! High-performance warp shuffle primitives for reductions, replacing slow
//! shared memory + __syncthreads() patterns with 2-4 cycle warp shuffles.
//!
//! # NVRTC Compatibility Contract
//!
//! - NO #include directives: this header is PREPENDED to kernel sources at
//!   the Rust `include_str!` assembly sites (see backtest/batch.rs
//!   BACKTEST_KERNELS_SRC). NVRTC compiles from an in-memory string with an
//!   empty include path, so a runtime `#include "warp_primitives.cuh"` can
//!   never be resolved.
//! - Only NVRTC built-in intrinsics are used (__shfl_down_sync, fmax/fmaxf,
//!   fmin/fminf, min/max, __longlong_as_double, __int_as_float).
//!
//! # Performance
//!
//! - Traditional reduction: 32 cycles × log2(N) iterations
//! - Warp reduction: 2-4 cycles × 5 iterations (for 32 threads)
//! - Speedup: **6.4x for sum/max/min reductions**
//!
//! # Precision (Ada / sm_89)
//!
//! Ada executes FP64 at 1/64 the FP32 rate. The float overloads below exist
//! so float operands resolve to float arithmetic; without them, float
//! arguments silently promoted to the double versions and ran 64x slower.
//!
//! # Usage
//!
//! ```cuda
//! // Old pattern (SLOW):
//! __shared__ double shmem[256];
//! shmem[tid] = value;
//! __syncthreads();
//! for (int s = 128; s > 0; s >>= 1) {
//!     if (tid < s) shmem[tid] += shmem[tid + s];
//!     __syncthreads();
//! }
//!
//! // New pattern (FAST):
//! double result = warp_reduce_sum(value);
//! if (lane_id == 0) {
//!     // result available only in thread 0 of each warp
//! }
//! ```
//!
//! # Safety
//!
//! - All warp operations assume full 32-thread warps
//! - Mask 0xffffffff assumes all threads active
//! - Use __syncwarp() if warp divergence possible
//! - block_reduce_* functions must be reached by ALL threads in the block
//!   (they contain __syncthreads())

#ifndef WARP_PRIMITIVES_CUH
#define WARP_PRIMITIVES_CUH

// Warp size constant
#define WARP_SIZE 32

//==============================================================================
// Per-Type Reduction Identities
//==============================================================================

/// Identity values for max/min reductions.
///
/// The previous implementation used the double-precision -inf bit pattern as
/// the identity for EVERY instantiation, which is wrong for float/int.
/// Supported types: double, float, int. Other types fail to compile (better
/// than a silently wrong identity).
template<typename T> struct wp_limits;

template<> struct wp_limits<double> {
    static __device__ __forceinline__ double lowest() {
        return -__longlong_as_double(0x7ff0000000000000ULL); // -inf
    }
    static __device__ __forceinline__ double highest() {
        return __longlong_as_double(0x7ff0000000000000ULL); // +inf
    }
};

template<> struct wp_limits<float> {
    static __device__ __forceinline__ float lowest() {
        return -__int_as_float(0x7f800000); // -inff
    }
    static __device__ __forceinline__ float highest() {
        return __int_as_float(0x7f800000); // +inff
    }
};

template<> struct wp_limits<int> {
    static __device__ __forceinline__ int lowest() {
        return -2147483647 - 1; // INT_MIN
    }
    static __device__ __forceinline__ int highest() {
        return 2147483647; // INT_MAX
    }
};

//==============================================================================
// Warp-Level Reduction Primitives
//==============================================================================

/// Warp-level sum reduction (double)
///
/// Reduces a value across all 32 threads in a warp using warp shuffle.
/// Result is only valid in lane 0.
///
/// # Performance
/// - 5 iterations × 2-4 cycles = 10-20 cycles total
/// - vs 5 iterations × 32 cycles = 160 cycles for __syncthreads()
/// - Speedup: **8-16x**
__device__ __forceinline__ double warp_reduce_sum(double val) {
    // Reduce across warp using shuffle down
    // Iteration order: 16, 8, 4, 2, 1
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/// Warp-level sum reduction (float)
///
/// Float overload: keeps the reduction in FP32 on Ada (FP64 is 1:64).
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/// Warp-level sum reduction (int)
///
/// Integer overload (also prevents float/double overload ambiguity for
/// integer operands, e.g. trade-win counting).
__device__ __forceinline__ int warp_reduce_sum(int val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/// Warp-level max reduction (double)
///
/// Finds maximum value across all 32 threads in a warp using warp shuffle.
/// Result is only valid in lane 0.
__device__ __forceinline__ double warp_reduce_max(double val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/// Warp-level max reduction (float)
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/// Warp-level max reduction (int)
__device__ __forceinline__ int warp_reduce_max(int val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/// Warp-level min reduction (double)
///
/// Finds minimum value across all 32 threads in a warp using warp shuffle.
/// Result is only valid in lane 0.
__device__ __forceinline__ double warp_reduce_min(double val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/// Warp-level min reduction (float)
__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/// Warp-level min reduction (int)
__device__ __forceinline__ int warp_reduce_min(int val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

//==============================================================================
// Block-Level Reduction Primitives (Cross-Warp)
//==============================================================================
//
// RACE-FIX NOTE (entry __syncthreads):
// Each block_reduce_* ends with every thread reading warp_*[0]. A second
// invocation of the SAME function (same static __shared__ buffer) writes
// warp_*[warp_id] in stage 2 with no intervening barrier, so warp 0 lane 0
// could overwrite slot 0 while another warp is still reading the previous
// result. The __syncthreads() at function entry orders the previous call's
// final reads before this call's first writes, making back-to-back
// invocations safe.

/// Block-level sum reduction using warp primitives
///
/// Two-stage reduction:
/// 1. Warp-level reduction (parallel, no sync)
/// 2. Cross-warp reduction (one warp, shared memory)
///
/// # Performance
/// - Stage 1 (8 warps): 20 cycles parallel
/// - Stage 2 (1 warp): 20 cycles sequential
/// - Total: ~40 cycles (vs 256 cycles for full tree reduction)
/// - Speedup: **6.4x**
///
/// # Template Parameters
/// - `T`: Type to reduce (double, float, int)
///
/// # Returns
/// - Sum of all values in block (valid in ALL threads)
template<typename T>
__device__ T block_reduce_sum(T val) {
    // Shared memory for warp-level results
    // Max 32 warps per block (1024 threads ÷ 32)
    __shared__ T warp_sums[32];

    // Entry barrier: see RACE-FIX NOTE above.
    __syncthreads();

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // Stage 1: Reduce within each warp (parallel across all warps)
    T warp_sum = warp_reduce_sum(val);

    // Stage 2: First thread in each warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    // Stage 3: First warp reduces across all warp results
    if (warp_id == 0) {
        // Load warp sum (or the additive identity if out of bounds)
        T temp = (lane < num_warps) ? warp_sums[lane] : T(0);
        // Final warp-level reduction
        temp = warp_reduce_sum(temp);
        // Write final result back to shared memory
        if (lane == 0) {
            warp_sums[0] = temp;
        }
    }
    __syncthreads();

    // All threads return the result (stored in warp_sums[0])
    return warp_sums[0];
}

/// Block-level max reduction using warp primitives
///
/// Two-stage reduction using warp shuffle + shared memory.
///
/// # Template Parameters
/// - `T`: Type to reduce (double, float, int)
///
/// # Returns
/// - Maximum of all values in block (valid in ALL threads)
template<typename T>
__device__ T block_reduce_max(T val) {
    __shared__ T warp_maxs[32];

    // Entry barrier: see RACE-FIX NOTE above.
    __syncthreads();

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // Stage 1: Reduce within each warp
    T warp_max = warp_reduce_max(val);

    // Stage 2: First thread in each warp writes to shared memory
    if (lane == 0) {
        warp_maxs[warp_id] = warp_max;
    }
    __syncthreads();

    // Stage 3: First warp reduces across all warp results
    if (warp_id == 0) {
        // Load warp max, or the per-type max identity if out of bounds
        T temp = (lane < num_warps) ? warp_maxs[lane] : wp_limits<T>::lowest();
        temp = warp_reduce_max(temp);
        if (lane == 0) {
            warp_maxs[0] = temp;
        }
    }
    __syncthreads();

    return warp_maxs[0];
}

/// Block-level min reduction using warp primitives
///
/// Two-stage reduction using warp shuffle + shared memory.
///
/// # Template Parameters
/// - `T`: Type to reduce (double, float, int)
///
/// # Returns
/// - Minimum of all values in block (valid in ALL threads)
template<typename T>
__device__ T block_reduce_min(T val) {
    __shared__ T warp_mins[32];

    // Entry barrier: see RACE-FIX NOTE above.
    __syncthreads();

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // Stage 1: Reduce within each warp
    T warp_min = warp_reduce_min(val);

    // Stage 2: First thread in each warp writes to shared memory
    if (lane == 0) {
        warp_mins[warp_id] = warp_min;
    }
    __syncthreads();

    // Stage 3: First warp reduces across all warp results
    if (warp_id == 0) {
        // Load warp min, or the per-type min identity if out of bounds
        T temp = (lane < num_warps) ? warp_mins[lane] : wp_limits<T>::highest();
        temp = warp_reduce_min(temp);
        if (lane == 0) {
            warp_mins[0] = temp;
        }
    }
    __syncthreads();

    return warp_mins[0];
}

//==============================================================================
// Multi-Value Block Reductions (Fused for Sharpe Ratio)
//==============================================================================

/// Block-level fused sum reduction for two values
///
/// Reduces two values simultaneously (e.g., sum and sum_of_squares for
/// variance). More efficient than calling block_reduce_sum twice.
///
/// # Performance
/// - Same as single reduction: ~40 cycles total
/// - 2x throughput vs separate reductions
///
/// # Arguments
/// - `val1`: First value to reduce
/// - `val2`: Second value to reduce
/// - `result1`: Output for first reduction (valid in ALL threads)
/// - `result2`: Output for second reduction (valid in ALL threads)
template<typename T>
__device__ void block_reduce_sum_pair(T val1, T val2, T& result1, T& result2) {
    __shared__ T warp_sums1[32];
    __shared__ T warp_sums2[32];

    // Entry barrier: see RACE-FIX NOTE above.
    __syncthreads();

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // Stage 1: Reduce both values within each warp (parallel)
    T warp_sum1 = warp_reduce_sum(val1);
    T warp_sum2 = warp_reduce_sum(val2);

    // Stage 2: First thread in each warp writes to shared memory
    if (lane == 0) {
        warp_sums1[warp_id] = warp_sum1;
        warp_sums2[warp_id] = warp_sum2;
    }
    __syncthreads();

    // Stage 3: First warp reduces across all warp results
    if (warp_id == 0) {
        T temp1 = (lane < num_warps) ? warp_sums1[lane] : T(0);
        T temp2 = (lane < num_warps) ? warp_sums2[lane] : T(0);

        temp1 = warp_reduce_sum(temp1);
        temp2 = warp_reduce_sum(temp2);

        if (lane == 0) {
            warp_sums1[0] = temp1;
            warp_sums2[0] = temp2;
        }
    }
    __syncthreads();

    // Return results (valid for all threads)
    result1 = warp_sums1[0];
    result2 = warp_sums2[0];
}

#endif // WARP_PRIMITIVES_CUH
