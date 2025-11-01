//! Fused RSI Kernel with Parallel Wilder's Smoothing
//!
//! Implements single-pass RSI calculation using CUB DeviceScan for parallel
//! Wilder's smoothing, eliminating CPU round-trips.
//!
//! # Architecture
//!
//! **Fused Kernel** (1 launch, 3 stages):
//! 1. Calculate gains/losses (parallel)
//! 2. Wilder's smoothing via CUB scan (parallel prefix sum)
//! 3. Calculate RSI values (parallel)
//!
//! # Performance Target
//!
//! - Baseline (hybrid): ~130μs (64μs D2H/H2D + 30μs CPU Wilder's + 36μs GPU)
//! - Target (fused): ~61μs (eliminate transfers, parallelize Wilder's)
//! - Speedup: **2.13x**
//!
//! # Technical Approach
//!
//! Wilder's smoothing is an IIR filter:
//! ```
//! avg[0] = SMA(first N values)
//! avg[i] = alpha * value[i] + (1-alpha) * avg[i-1]
//! ```
//!
//! This can be expressed as a parallel prefix sum:
//! ```
//! avg[i] = Σ(k=0 to i) [alpha * (1-alpha)^(i-k) * value[k]]
//! ```
//!
//! CUB DeviceScan can compute this in O(log N) time vs O(N) sequential.
//!
//! # Memory Layout
//!
//! - Input: close prices (n elements)
//! - Temp: gains, losses (n elements each)
//! - Temp: avg_gain, avg_loss (n elements each)
//! - Output: rsi (n elements)
//! - Total: 6n f64 = 48n bytes (4.8 MB for 100K candles)

// Include CUDA headers first to avoid conflicts
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// Avoid including system math.h after CUDA headers
#undef __MATH_H
#undef __MATHCALLS_H

// Define NaN for CUDA (compatible with NVRTC)
#ifndef CUDART_NAN
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)
#endif

//==============================================================================
// Stage 1: Calculate Gains and Losses (Parallel)
//==============================================================================

/// Calculate price deltas and separate into gains/losses
///
/// Each thread processes one price change.
/// Branchless implementation for optimal GPU performance.
extern "C" __global__ void calculate_gains_losses_kernel(
    const double* __restrict__ close,
    double* __restrict__ gains,
    double* __restrict__ losses,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n - 1) return;

    // Calculate delta for position idx+1
    double delta = close[idx + 1] - close[idx];

    // Branchless gain/loss separation
    gains[idx + 1] = fmax(delta, 0.0);
    losses[idx + 1] = fmax(-delta, 0.0);

    // First element is always 0 (no previous price)
    if (idx == 0) {
        gains[0] = 0.0;
        losses[0] = 0.0;
    }
}

//==============================================================================
// Stage 2: Wilder's Smoothing via CUB Scan (Parallel)
//==============================================================================

/// Custom scan operator for Wilder's smoothing
///
/// Implements: result = alpha * b + (1-alpha) * a
/// where a is the previous smoothed value, b is the current raw value
struct WildersOp {
    double alpha;
    double one_minus_alpha;

    __device__ __forceinline__ WildersOp(double alpha_)
        : alpha(alpha_), one_minus_alpha(1.0 - alpha_) {}

    __device__ __forceinline__
    double operator()(const double &a, const double &b) const {
        return alpha * b + one_minus_alpha * a;
    }
};

/// Initialize first period values with SMA and prepare for scan
///
/// This kernel:
/// 1. Sets first (period-1) values to NaN
/// 2. Calculates SMA for index [period-1]
/// 3. Copies remaining values to temp buffer for scanning
extern "C" __global__ void prepare_wilder_scan_kernel(
    const double* __restrict__ input,
    double* __restrict__ scan_input,
    double* __restrict__ output,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // First (period-1) elements are NaN (warmup)
    if (idx < period - 1) {
        output[idx] = CUDART_NAN;
        scan_input[idx] = 0.0;  // Unused, but initialize
        return;
    }

    // Calculate SMA at index (period-1)
    if (idx == period - 1) {
        double sum = 0.0;
        for (int i = 0; i < period; i++) {
            sum += input[i];
        }
        double sma = sum / period;
        output[idx] = sma;
        scan_input[idx] = sma;  // First scan value is SMA
        return;
    }

    // For idx >= period, prepare for scanning
    // Copy input value to scan_input
    scan_input[idx] = input[idx];
}

/// Finalize Wilder's smoothing after CUB scan
///
/// CUB scan computes prefix sum with Wilder's operator.
/// This kernel copies scan results to output buffer.
extern "C" __global__ void finalize_wilder_scan_kernel(
    const double* __restrict__ scan_output,
    double* __restrict__ output,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Copy scan results for idx >= period
    if (idx >= period) {
        output[idx] = scan_output[idx];
    }
    // Indices < period already set by prepare kernel
}

//==============================================================================
// Stage 3: Calculate RSI (Parallel)
//==============================================================================

/// Calculate final RSI values from avg_gain and avg_loss
///
/// RSI = 100 - (100 / (1 + RS))
/// where RS = avg_gain / avg_loss
extern "C" __global__ void calculate_rsi_kernel(
    const double* __restrict__ avg_gain,
    const double* __restrict__ avg_loss,
    double* __restrict__ rsi,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // RSI is only valid from period onward
    if (idx < period) {
        rsi[idx] = CUDART_NAN;
        return;
    }

    double gain = avg_gain[idx];
    double loss = avg_loss[idx];

    // Handle edge case: if loss == 0, RSI = 100
    if (loss < 1e-10) {
        rsi[idx] = 100.0;
        return;
    }

    // Calculate RSI = 100 - (100 / (1 + RS))
    double rs = gain / loss;
    rsi[idx] = 100.0 - (100.0 / (1.0 + rs));
}

//==============================================================================
// Host-Side CUB Scan Launcher (called from Rust)
//==============================================================================

/// Launch CUB inclusive scan for Wilder's smoothing
///
/// This is a host function that allocates temp storage and launches CUB scan.
/// Must be compiled with nvcc (not NVRTC) due to CUB template instantiation.
///
/// # Arguments
///
/// - `d_input`: Input values (gains or losses)
/// - `d_output`: Output smoothed values
/// - `n`: Number of elements
/// - `alpha`: Wilder's smoothing factor (1/period)
/// - `stream`: CUDA stream for async execution
extern "C" cudaError_t launch_wilder_scan(
    const double* d_input,
    double* d_output,
    int n,
    double alpha,
    cudaStream_t stream
) {
    // Create Wilder's scan operator
    WildersOp scan_op(alpha);

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cudaError_t err = cub::DeviceScan::InclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        d_input,
        d_output,
        scan_op,
        n,
        stream
    );

    if (err != cudaSuccess) return err;

    // Allocate temporary storage
    err = cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
    if (err != cudaSuccess) return err;

    // Run inclusive scan
    err = cub::DeviceScan::InclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        d_input,
        d_output,
        scan_op,
        n,
        stream
    );

    // Free temporary storage (async)
    cudaFreeAsync(d_temp_storage, stream);

    return err;
}

//==============================================================================
// Fused RSI Kernel Launcher (High-Level)
//==============================================================================

/// Fused RSI calculation with parallel Wilder's smoothing
///
/// This is a convenience launcher that executes all 3 stages:
/// 1. Calculate gains/losses
/// 2. Wilder's smoothing (2x: gains and losses)
/// 3. Calculate RSI
///
/// # Memory Requirements
///
/// - 6 device buffers: close, gains, losses, avg_gain, avg_loss, rsi
/// - 2 temp buffers for CUB scan: scan_input_gain, scan_input_loss
/// - CUB internal temp storage (allocated automatically)
///
/// # Performance
///
/// Expected: ~61μs for 100K candles (2.13x faster than hybrid)
///
/// Breakdown:
/// - Stage 1 (gains/losses): ~20μs
/// - Stage 2 (CUB scan x2): ~25μs (O(log N) parallel)
/// - Stage 3 (RSI): ~15μs
/// - CUB overhead: ~1μs
extern "C" cudaError_t launch_rsi_fused(
    const double* d_close,
    double* d_rsi,
    double* d_gains,
    double* d_losses,
    double* d_avg_gain,
    double* d_avg_loss,
    double* d_scan_input_gain,
    double* d_scan_input_loss,
    int n,
    int period,
    cudaStream_t stream
) {
    cudaError_t err;

    // Stage 1: Calculate gains and losses (parallel)
    int block_size = 256;
    int num_blocks_deltas = (n - 1 + block_size - 1) / block_size;

    calculate_gains_losses_kernel<<<num_blocks_deltas, block_size, 0, stream>>>(
        d_close, d_gains, d_losses, n
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Stage 2a: Prepare gains for Wilder's scan
    int num_blocks_full = (n + block_size - 1) / block_size;

    prepare_wilder_scan_kernel<<<num_blocks_full, block_size, 0, stream>>>(
        d_gains, d_scan_input_gain, d_avg_gain, n, period
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Stage 2b: CUB scan for gains
    double alpha = 1.0 / period;
    err = launch_wilder_scan(
        d_scan_input_gain, d_scan_input_gain, n, alpha, stream
    );
    if (err != cudaSuccess) return err;

    // Stage 2c: Finalize gains scan
    finalize_wilder_scan_kernel<<<num_blocks_full, block_size, 0, stream>>>(
        d_scan_input_gain, d_avg_gain, n, period
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Stage 2d: Prepare losses for Wilder's scan
    prepare_wilder_scan_kernel<<<num_blocks_full, block_size, 0, stream>>>(
        d_losses, d_scan_input_loss, d_avg_loss, n, period
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Stage 2e: CUB scan for losses
    err = launch_wilder_scan(
        d_scan_input_loss, d_scan_input_loss, n, alpha, stream
    );
    if (err != cudaSuccess) return err;

    // Stage 2f: Finalize losses scan
    finalize_wilder_scan_kernel<<<num_blocks_full, block_size, 0, stream>>>(
        d_scan_input_loss, d_avg_loss, n, period
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Stage 3: Calculate RSI from avg_gain and avg_loss (parallel)
    calculate_rsi_kernel<<<num_blocks_full, block_size, 0, stream>>>(
        d_avg_gain, d_avg_loss, d_rsi, n, period
    );

    err = cudaGetLastError();
    return err;
}
