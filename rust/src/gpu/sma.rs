//! GPU-Accelerated SMA (Simple Moving Average)
//!
//! Provides 40-60x speedup over CPU implementation for large datasets.
//! SMA is perfectly parallelizable - each thread calculates one value independently.
//!
//! # Algorithm
//!
//! ```text
//! SMA[i] = (close[i-period+1] + close[i-period+2] + ... + close[i]) / period
//! ```
//!
//! This is an embarrassingly parallel problem - each thread computes a sum of `period`
//! consecutive values and divides by the period. No shared memory or thread synchronization
//! is needed.

use super::device::{GpuDevice, GpuError};
use super::precision::{NumericalClass, Precision};
use crate::gpu::compile::compile_ptx_optimized_cached;
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for Simple Moving Average
///
/// Algorithm: SMA[i] = sum(close[i-period+1..=i]) / period
///
/// This is an embarrassingly parallel problem - each thread operates independently
/// with no shared memory or synchronization needed. Each thread computes the sum
/// of `period` consecutive values in the close array.
const SMA_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void sma_kernel(
    const double* __restrict__ close,
    double* __restrict__ sma,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only calculate SMA for indices where we have enough history
    if (idx >= period - 1 && idx < n) {
        double sum = 0.0;

        // Sum the last `period` values
        for (int j = 0; j < period; j++) {
            sum += close[idx - j];
        }

        sma[idx] = sum / (double)period;
    } else if (idx < period - 1) {
        // Not enough history - set to NaN
        sma[idx] = CUDART_NAN;
    }
}

// Shared memory variant with bank conflict avoidance
// NOTE: Shared memory provides minimal benefit here due to low data reuse between threads.
// Each thread's window overlaps by only 1 element with adjacent threads.
// Expected improvement: 0-3% (may even regress due to shared memory overhead).
extern "C" __global__ void sma_kernel_shared(
    const double* __restrict__ close,
    double* __restrict__ sma,
    int n,
    int period
) {
    extern __shared__ double shared_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Calculate how much data this block needs
    // Need to handle edge case where first thread in block needs (period-1) elements before block_start
    int block_start = blockIdx.x * block_size;

    // The block needs data from [block_start - (period-1), block_start + block_size - 1]
    // But only if block_start - (period-1) >= 0
    int data_start = (block_start >= period - 1) ? (block_start - (period - 1)) : 0;
    int data_end = block_start + block_size - 1;
    int data_needed = (data_end - data_start + 1);

    // Cooperatively load data into shared memory
    for (int i = tid; i < data_needed && (data_start + i) < n; i += block_size) {
        shared_data[i] = close[data_start + i];
    }

    __syncthreads();

    if (idx >= period - 1 && idx < n) {
        double sum = 0.0;

        // Calculate sum from shared memory
        // Map global idx to local shared memory offset
        int local_offset = idx - data_start;

        for (int j = 0; j < period; j++) {
            sum += shared_data[local_offset - j];
        }

        sma[idx] = sum / (double)period;
    } else if (idx < period - 1) {
        sma[idx] = CUDART_NAN;
    }

    __syncthreads();
}
"#;

/// GPU-accelerated Simple Moving Average (SMA) indicator
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `close` - Close prices
/// * `period` - Window size for moving average (e.g., 5, 20, 50, 200)
/// * `stream` - Optional CUDA stream for concurrent execution (None uses device default)
///
/// # Returns
///
/// Array1<f64> containing SMA values (NaN for first `period-1` values)
///
/// # Algorithm
///
/// ```text
/// SMA[i] = (close[i-period+1] + close[i-period+2] + ... + close[i]) / period
/// ```
///
/// # Performance (Async v0.2.1)
///
/// Expected speedup: **44-67x** over CPU for n > 10,000 (~11% faster with async pinned memory)
///
/// This is one of the fastest GPU indicators due to perfect parallelism:
/// - No data dependencies between threads
/// - No shared memory requirements
/// - No thread synchronization
/// - Each thread computes independently
/// - Memory access is coalesced for optimal bandwidth
///
/// # Stream Concurrency
///
/// When a stream is provided, kernel launches execute on that stream, enabling
/// concurrent execution with other operations on different streams. This is used
/// in the batch pipeline for 4-6x speedup across Fast/Medium/Slow indicator groups.
///
/// Classification: **FAST** indicator (<5μs/candle, perfectly parallel, single kernel)
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, sma_gpu};
/// use ndarray::arr1;
///
/// let device = GpuDevice::new()?;
/// let close = arr1(&[100.0, 102.0, 104.0, 106.0, 108.0, 110.0]);
/// let sma = sma_gpu(&device, &close, 3, None)?;
///
/// // sma[2] = (100 + 102 + 104) / 3 = 102.0
/// assert!((sma[2] - 102.0).abs() < 0.01);
/// ```
pub fn sma_gpu_f64(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let n = close.len();

    // Validate inputs
    if period < 1 {
        return Err(GpuError::InvalidParameter(
            "Period must be >= 1".to_string(),
        ));
    }

    if n < period {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need >= {} points, got {}",
            period, n
        )));
    }

    // Compile PTX
    let ptx_arc = compile_ptx_optimized_cached(SMA_KERNEL).map_err(|e| {
        GpuError::CompilationError(format!("Failed to compile SMA kernel: {:?}", e))
    })?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);

    // Load module (use context, not stream)
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel function from module
    let kernel = module.load_function("sma_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
    })?;

    // Use provided stream or default to device stream
    let kernel_stream = stream.unwrap_or(&device.stream);

    // === Step 1: H2D - Asynchronously copy data to device ===
    // Acquire pinned buffer
    let mut pinned_close = device.pinned_pool.lock().acquire(n)?;
    pinned_close.as_mut_slice()[..n].copy_from_slice(close.as_slice().unwrap());

    // Allocate device buffer
    let mut d_close = device.alloc_buffer(n)?;

    // Async H2D transfer
    kernel_stream.memcpy_htod(&pinned_close.as_slice()[..n], &mut d_close)?;

    // Release pinned buffer
    device.pinned_pool.lock().release(pinned_close);

    // Allocate output buffer
    let mut d_sma = device.alloc_buffer(n)?;

    // Launch kernel using builder pattern on specified stream
    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut builder = kernel_stream.launch_builder(&kernel);
    builder.arg(&d_close);
    builder.arg(&mut d_sma);
    builder.arg(&n_i32);
    builder.arg(&period_i32);

    let config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("SMA kernel launch failed: {:?}", e)))?;
    }

    // === Step 3: D2H - Asynchronously copy results back ===
    // Acquire pinned buffer for async D2H transfer
    let mut pinned_sma = device.pinned_pool.lock().acquire(n)?;

    // Async D2H transfer
    kernel_stream.memcpy_dtoh(&d_sma, &mut pinned_sma.as_mut_slice()[..n])?;

    // Synchronize stream to ensure D2H copy is complete before CPU access
    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    // Copy to output array
    let sma_vec = pinned_sma.as_slice()[..n].to_vec();

    // Release pinned buffer
    device.pinned_pool.lock().release(pinned_sma);

    Ok(Array1::from_vec(sma_vec))
}

/// GPU-accelerated Simple Moving Average (SMA).
///
/// Computes in **FP32** internally (Ada sm_89 runs FP32 at full rate vs FP64 at
/// 1/64; measured ~1.3-1.7x faster, scaling toward ~2x with size as the kernel
/// becomes memory-bound). The public API and (price-scale) tolerance are
/// unchanged; see [`sma_gpu_f64`] for the FP64 reference path.
pub fn sma_gpu(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    sma_gpu_f32(device, close, period, stream)
}

/// SMA with an explicit [`Precision`] policy. SMA is a bounded-window indicator,
/// so `Precision::Auto` resolves to FP32 (the default [`sma_gpu`] path); pass
/// `Precision::F64` to force the exact FP64 reference.
pub fn sma_gpu_prec(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    precision: Precision,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    match precision.resolve(NumericalClass::BoundedWindow) {
        Precision::F64 => sma_gpu_f64(device, close, period, stream),
        _ => sma_gpu_f32(device, close, period, stream),
    }
}

// ============================================================================
// FP32 variant (Phase 1: FP64->FP32). Ada (sm_89) runs FP32 at full rate vs
// FP64 at 1/64, and f32 halves DRAM + PCIe traffic. SMA sums a small window of
// price-scale values, well within f32 range/precision, so the (lenient,
// price-scale) tolerance is unaffected. Public API stays f64; we convert at the
// host boundary. Benchmarked against the f64 path before promotion.
// ============================================================================

const SMA_KERNEL_F32: &str = r#"
#define CUDART_NANF __int_as_float(0x7fc00000)

extern "C" __global__ void sma_kernel_f32(
    const float* __restrict__ close,
    float* __restrict__ sma,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= period - 1 && idx < n) {
        float sum = 0.0f;
        for (int j = 0; j < period; j++) {
            sum += close[idx - j];
        }
        sma[idx] = sum / (float)period;
    } else if (idx < period - 1) {
        sma[idx] = CUDART_NANF;
    }
}
"#;

/// GPU SMA computed in **FP32** (f64 public API; conversion at the host boundary).
pub fn sma_gpu_f32(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let n = close.len();
    if period < 1 {
        return Err(GpuError::InvalidParameter("Period must be >= 1".to_string()));
    }
    if n < period {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need >= {} points, got {}",
            period, n
        )));
    }

    // Module cache: compiled + loaded once per process, reused across calls
    // (avoids the per-call PTX compile + module load).
    let kernel = device.get_or_load_function(SMA_KERNEL_F32, "sma_kernel_f32")?;
    let kernel_stream = stream.unwrap_or(&device.stream);

    // H2D: narrow to f32 host-side (halves PCIe traffic) and transfer.
    let close_f32: Vec<f32> = close.iter().map(|&x| x as f32).collect();
    let mut d_close = device.alloc_uninit::<f32>(n)?;
    kernel_stream.memcpy_htod(&close_f32[..n], &mut d_close)?;
    let mut d_sma = device.alloc_uninit::<f32>(n)?;

    let n_i32 = n as i32;
    let period_i32 = period as i32;
    let mut builder = kernel_stream.launch_builder(&kernel);
    builder.arg(&d_close);
    builder.arg(&mut d_sma);
    builder.arg(&n_i32);
    builder.arg(&period_i32);
    let config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        builder.launch(config).map_err(|e| {
            GpuError::ExecutionError(format!("SMA f32 kernel launch failed: {:?}", e))
        })?;
    }

    // D2H: copy f32 results back and widen to the f64 public output.
    let mut sma_f32 = vec![0.0f32; n];
    kernel_stream.memcpy_dtoh(&d_sma, &mut sma_f32[..])?;
    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;
    Ok(Array1::from_iter(sma_f32.into_iter().map(|x| x as f64)))
}

/// Compute SMA from a pre-uploaded device buffer into a device output buffer,
/// with **no host<->device transfer**.
///
/// This is the building block for device-resident sweeps: upload OHLCV once,
/// then run many parameter kernels reusing the on-device buffer. Profiling shows
/// the per-call H2D/D2H round-trip is ~93% of GPU time, so reusing a resident
/// buffer across a parameter sweep is ~88x faster than re-uploading per call
/// (see `bench_sma_device_resident_vs_reupload`).
///
/// `d_close` and `d_out` must both have length >= `n` (f32, on `device`).
pub fn sma_on_device(
    device: &GpuDevice,
    d_close: &CudaSlice<f32>,
    d_out: &mut CudaSlice<f32>,
    n: usize,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<(), GpuError> {
    if period < 1 {
        return Err(GpuError::InvalidParameter("Period must be >= 1".to_string()));
    }
    let kernel = device.get_or_load_function(SMA_KERNEL_F32, "sma_kernel_f32")?;
    let kernel_stream = stream.unwrap_or(&device.stream);
    let n_i32 = n as i32;
    let period_i32 = period as i32;
    let mut builder = kernel_stream.launch_builder(&kernel);
    builder.arg(d_close);
    builder.arg(d_out);
    builder.arg(&n_i32);
    builder.arg(&period_i32);
    let config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        builder.launch(config).map_err(|e| {
            GpuError::ExecutionError(format!("SMA on-device launch failed: {:?}", e))
        })?;
    }
    Ok(())
}

/// Device-resident SMA parameter sweep: upload `close` **once**, then compute
/// SMA for every period reusing the on-device input/output buffers (no per-period
/// H2D). Returns one SMA series per period.
///
/// Eliminating the redundant per-evaluation upload is the dominant lever for
/// parameter sweeps (profiling §10). The per-period D2H remains because each
/// result is consumed on the host; computing the optimization metric on-device
/// (returning only a scalar per period) is the further win.
pub fn sma_sweep_on_device(
    device: &GpuDevice,
    close: &Array1<f64>,
    periods: &[usize],
    stream: Option<&Arc<CudaStream>>,
) -> Result<Vec<Array1<f64>>, GpuError> {
    let n = close.len();
    let kernel_stream = stream.unwrap_or(&device.stream);

    // Upload close ONCE; reuse across all periods.
    let close_f32: Vec<f32> = close.iter().map(|&x| x as f32).collect();
    let mut d_close = device.alloc_uninit::<f32>(n)?;
    kernel_stream.memcpy_htod(&close_f32[..n], &mut d_close)?;
    let mut d_out = device.alloc_uninit::<f32>(n)?;
    let mut out_f32 = vec![0.0f32; n];

    let mut results = Vec::with_capacity(periods.len());
    for &period in periods {
        sma_on_device(device, &d_close, &mut d_out, n, period, Some(kernel_stream))?;
        kernel_stream.memcpy_dtoh(&d_out, &mut out_f32[..])?;
        kernel_stream.synchronize().map_err(|e| {
            GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
        })?;
        results.push(Array1::from_iter(out_f32.iter().map(|&x| x as f64)));
    }
    Ok(results)
}

/// GPU-accelerated SMA using **shared memory** optimization
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `close` - Close prices
/// * `period` - Window size for moving average
/// * `stream` - Optional CUDA stream for concurrent execution
///
/// # Returns
///
/// Array1<f64> containing SMA values (NaN for first `period-1` values)
///
/// # Performance
///
/// **WARNING**: This variant uses shared memory but provides **minimal benefit** (0-3% improvement,
/// possibly regression) because:
/// - Adjacent threads have only 1 overlapping element (out of `period` elements)
/// - Global memory access is already coalesced
/// - L1/L2 cache handles this access pattern efficiently
/// - Shared memory overhead (loading + sync) may exceed benefits
///
/// **Recommendation**: Use standard `sma_gpu()` unless profiling proves otherwise.
///
/// # Shared Memory Strategy
///
/// - Allocates `(blockDim.x + period - 1) * sizeof(f64)` bytes per block
/// - Cooperative loading: Each block loads its required window
/// - No bank conflict padding needed (sequential access pattern)
/// - 2x __syncthreads() calls per block
///
/// # Example
///
/// ```rust,ignore
/// let device = GpuDevice::new()?;
/// let close = arr1(&[100.0, 102.0, 104.0, 106.0, 108.0]);
/// let sma = sma_gpu_shared(&device, &close, 3, None)?;
/// ```
pub fn sma_gpu_shared(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let n = close.len();

    // Validate inputs
    if period < 1 {
        return Err(GpuError::InvalidParameter(
            "Period must be >= 1".to_string(),
        ));
    }

    if n < period {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need >= {} points, got {}",
            period, n
        )));
    }

    // Compile PTX
    let ptx_arc = compile_ptx_optimized_cached(SMA_KERNEL).map_err(|e| {
        GpuError::CompilationError(format!("Failed to compile SMA kernel: {:?}", e))
    })?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);

    // Load module
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get shared memory kernel function
    let kernel = module.load_function("sma_kernel_shared").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
    })?;

    // Copy data to GPU
    let d_close = device.copy_to_device(close.as_slice().unwrap())?;
    let mut d_sma = device.alloc_buffer(n)?;

    // Use provided stream or default
    let kernel_stream = stream.unwrap_or(&device.stream);

    // Calculate shared memory size
    // Each block needs data from [block_start - (period-1), block_start + block_size - 1]
    // Maximum is: block_size + (period - 1) elements
    let config = LaunchConfig::for_num_elems(n as u32);
    let block_size = config.block_dim.0 as usize; // Typically 256
    let max_data_per_block = block_size + period - 1;
    let shared_mem_bytes = (max_data_per_block * std::mem::size_of::<f64>()) as u32;

    // Launch kernel with dynamic shared memory
    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut builder = kernel_stream.launch_builder(&kernel);
    builder.arg(&d_close);
    builder.arg(&mut d_sma);
    builder.arg(&n_i32);
    builder.arg(&period_i32);

    let config_with_shared = LaunchConfig {
        grid_dim: config.grid_dim,
        block_dim: config.block_dim,
        shared_mem_bytes,
    };

    unsafe {
        builder.launch(config_with_shared).map_err(|e| {
            GpuError::ExecutionError(format!("SMA shared kernel launch failed: {:?}", e))
        })?;
    }

    // Synchronize
    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    let sma_vec = device.copy_to_host(&d_sma)?;
    Ok(Array1::from_vec(sma_vec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Simple test case with known values
        let close = arr1(&[
            100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0,
        ]);
        let sma = sma_gpu(&device, &close, 3, None).expect("SMA GPU calculation failed");

        // Verify first `period-1` values are NaN
        for i in 0..2 {
            assert!(sma[i].is_nan(), "sma[{}] should be NaN", i);
        }

        // Verify calculations
        // sma[2] = (100 + 102 + 104) / 3 = 102.0
        assert!(
            (sma[2] - 102.0).abs() < 0.01,
            "sma[2] = {}, expected 102.0",
            sma[2]
        );

        // sma[3] = (102 + 104 + 106) / 3 = 104.0
        assert!(
            (sma[3] - 104.0).abs() < 0.01,
            "sma[3] = {}, expected 104.0",
            sma[3]
        );

        // sma[4] = (104 + 106 + 108) / 3 = 106.0
        assert!(
            (sma[4] - 106.0).abs() < 0.01,
            "sma[4] = {}, expected 106.0",
            sma[4]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_gpu_exactly_period_length() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with array length exactly equal to period
        let close = arr1(&[100.0, 105.0, 110.0]);
        let sma = sma_gpu(&device, &close, 3, None).expect("SMA GPU calculation failed");

        // First 2 values should be NaN
        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());

        // sma[2] = (100 + 105 + 110) / 3 = 105.0
        assert!(
            (sma[2] - 105.0).abs() < 0.01,
            "sma[2] = {}, expected 105.0",
            sma[2]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_gpu_nan_handling() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with NaN in input (edge case)
        let close = arr1(&[100.0, f64::NAN, 110.0, 115.0, 120.0]);
        let sma = sma_gpu(&device, &close, 3, None).expect("SMA GPU calculation failed");

        // sma[2] should include NaN, resulting in NaN
        assert!(
            sma[2].is_nan(),
            "sma[2] should be NaN when input contains NaN"
        );

        // sma[4] = (110 + 115 + 120) / 3 = 115.0 (no NaN in window)
        assert!(
            (sma[4] - 115.0).abs() < 0.01,
            "sma[4] = {}, expected 115.0",
            sma[4]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with large dataset (100K candles)
        let n = 100_000;
        let close = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());

        let start = std::time::Instant::now();
        let sma = sma_gpu(&device, &close, 20, None).expect("SMA GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU SMA (n={}, period=20): {:.2}ms ({:.0} values/sec)",
            n,
            elapsed.as_secs_f64() * 1000.0,
            n as f64 / elapsed.as_secs_f64()
        );

        assert_eq!(sma.len(), n);

        // Verify first 19 values are NaN
        for i in 0..19 {
            assert!(sma[i].is_nan(), "sma[{}] should be NaN", i);
        }

        // Verify rest are computed and increasing (since input is increasing)
        for i in 20..n {
            assert!(!sma[i].is_nan(), "sma[{}] should not be NaN", i);
            assert!(
                sma[i] > sma[i - 1],
                "SMA should be increasing for increasing input"
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_gpu_different_periods() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let close = arr1(&[
            100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0,
        ]);

        // Test period=5 (common short-term SMA)
        let sma5 = sma_gpu(&device, &close, 5, None).expect("SMA GPU failed");
        // sma5[4] = (100 + 102 + 104 + 106 + 108) / 5 = 104.0
        assert!((sma5[4] - 104.0).abs() < 0.01, "sma5[4] incorrect");

        // Test period=20 would need more data, but we can test period=10
        let sma10 = sma_gpu(&device, &close, 10, None).expect("SMA GPU failed");
        // sma10[9] = (100 + 102 + 104 + 106 + 108 + 110 + 112 + 114 + 116 + 118) / 10 = 109.0
        assert!((sma10[9] - 109.0).abs() < 0.01, "sma10[9] incorrect");

        // Test period=1 (no smoothing, same as close)
        let sma1 = sma_gpu(&device, &close, 1, None).expect("SMA GPU failed");
        for i in 0..close.len() {
            assert!(
                (sma1[i] - close[i]).abs() < 0.01,
                "SMA with period=1 should equal close prices"
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_gpu_constant_prices() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with constant prices - SMA should equal the constant
        let close = arr1(&[100.0; 20]);
        let sma = sma_gpu(&device, &close, 5, None).expect("SMA GPU calculation failed");

        // All valid SMA values should be 100.0
        for i in 4..20 {
            assert!(
                (sma[i] - 100.0).abs() < 0.01,
                "SMA of constant prices should be constant"
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_gpu_declining_prices() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with declining prices
        let close = arr1(&[120.0, 115.0, 110.0, 105.0, 100.0, 95.0, 90.0]);
        let sma = sma_gpu(&device, &close, 3, None).expect("SMA GPU calculation failed");

        // sma[2] = (120 + 115 + 110) / 3 = 115.0
        assert!((sma[2] - 115.0).abs() < 0.01);

        // sma[3] = (115 + 110 + 105) / 3 = 110.0
        assert!((sma[3] - 110.0).abs() < 0.01);

        // SMA should be declining
        for i in 3..7 {
            assert!(
                sma[i] < sma[i - 1],
                "SMA should be declining for declining prices"
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_gpu_performance_benchmark() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let sizes = vec![1_000, 10_000, 100_000, 1_000_000];
        let periods = vec![5, 20, 50, 200];

        for n in sizes.iter() {
            for period in periods.iter() {
                if n < period {
                    continue;
                }

                let close = Array1::from_vec((0..*n).map(|i| 100.0 + (i as f64) * 0.001).collect());

                let start = std::time::Instant::now();
                let sma =
                    sma_gpu(&device, &close, *period, None).expect("SMA GPU calculation failed");
                let elapsed = start.elapsed();

                let throughput = *n as f64 / elapsed.as_secs_f64();
                let ns_per_candle = (elapsed.as_secs_f64() * 1_000_000_000.0) / (*n as f64);

                println!(
                    "GPU SMA (n={:7}, period={:3}): {:6.2}ms - {:12.0} values/sec - {:6.2} ns/candle",
                    n,
                    period,
                    elapsed.as_secs_f64() * 1000.0,
                    throughput,
                    ns_per_candle
                );

                // Correctness: the kernel must produce a full-length, finite result.
                // (The per-candle timing above is printed for reference but NOT asserted:
                //  absolute ns/candle thresholds are machine-dependent and flaky --
                //  dominated by kernel-launch overhead at small n and by whatever else
                //  the GPU is doing. See test_sma_gpu_shared_correctness for value checks.)
                assert_eq!(sma.len(), *n, "SMA output length must match input");
                assert!(
                    sma.iter().skip(*period - 1).all(|v| v.is_finite()),
                    "SMA values past the warm-up window must be finite"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "Period must be >= 1")]
    fn test_sma_gpu_invalid_period() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let close = arr1(&[100.0, 105.0, 110.0]);
        let _sma = sma_gpu(&device, &close, 0, None).unwrap();
    }

    #[test]
    #[should_panic(expected = "Not enough data")]
    fn test_sma_gpu_insufficient_data() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let close = arr1(&[100.0, 105.0]);
        let _sma = sma_gpu(&device, &close, 5, None).unwrap();
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_gpu_shared_correctness() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test that shared memory variant produces identical results
        let close = arr1(&[
            100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0,
        ]);

        let sma_global = sma_gpu(&device, &close, 3, None).expect("Global SMA failed");
        let sma_shared = sma_gpu_shared(&device, &close, 3, None).expect("Shared SMA failed");

        assert_eq!(sma_global.len(), sma_shared.len());

        for i in 0..close.len() {
            if sma_global[i].is_nan() {
                assert!(sma_shared[i].is_nan(), "Mismatch at index {}", i);
            } else {
                assert!(
                    (sma_global[i] - sma_shared[i]).abs() < 1e-10,
                    "Mismatch at index {}: global={}, shared={}",
                    i,
                    sma_global[i],
                    sma_shared[i]
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_gpu_shared_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let close = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());

        // Test correctness on large dataset
        // Compare the f64 global reference vs the (also f64) shared kernel: this
        // test verifies the shared-memory kernel, so both sides must be f64 for
        // the tight 1e-9 tolerance. (The public sma_gpu now computes in f32.)
        let sma_global = sma_gpu_f64(&device, &close, 20, None).expect("Global SMA failed");
        let sma_shared = sma_gpu_shared(&device, &close, 20, None).expect("Shared SMA failed");

        // Verify results match
        for i in 0..n.min(100) {
            if sma_global[i].is_nan() {
                assert!(sma_shared[i].is_nan());
            } else {
                assert!((sma_global[i] - sma_shared[i]).abs() < 1e-9);
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_gpu_shared_vs_global_performance() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let sizes = vec![10_000, 100_000];
        let periods = vec![10, 20, 50, 100, 200];

        println!("\n=== SMA: Shared Memory vs Global Memory ===");
        println!("{:-<80}", "");

        for n in sizes.iter() {
            for period in periods.iter() {
                let close = Array1::from_vec((0..*n).map(|i| 100.0 + (i as f64) * 0.001).collect());

                // Warmup
                let _ = sma_gpu(&device, &close, *period, None);
                let _ = sma_gpu_shared(&device, &close, *period, None);

                // Benchmark global memory
                let start = std::time::Instant::now();
                for _ in 0..10 {
                    let _ = sma_gpu(&device, &close, *period, None).unwrap();
                }
                let global_time = start.elapsed().as_secs_f64() / 10.0;

                // Benchmark shared memory
                let start = std::time::Instant::now();
                for _ in 0..10 {
                    let _ = sma_gpu_shared(&device, &close, *period, None).unwrap();
                }
                let shared_time = start.elapsed().as_secs_f64() / 10.0;

                let speedup = (global_time / shared_time - 1.0) * 100.0;

                println!(
                    "n={:6}, period={:3} | Global: {:6.2}ms | Shared: {:6.2}ms | Δ: {:+5.1}%",
                    n,
                    period,
                    global_time * 1000.0,
                    shared_time * 1000.0,
                    speedup
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_f32_matches_f64() {
        // Spec Phase 1 f32-vs-f64 tolerance gate, realistic price-scale data.
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let n = 50_000usize;
        let close: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 30_000.0 + 5_000.0 * ((i as f64) * 0.001).sin()));
        for &period in &[5usize, 20, 50, 200] {
            let f64_out = sma_gpu_f64(&device, &close, period, None).expect("f64 SMA failed");
            let f32_out = sma_gpu_f32(&device, &close, period, None).expect("f32 SMA failed");
            for i in 0..n {
                if f64_out[i].is_nan() {
                    assert!(f32_out[i].is_nan(), "period {} idx {}: f32 not NaN", period, i);
                    continue;
                }
                let rel = (f64_out[i] - f32_out[i]).abs() / f64_out[i].abs().max(1.0);
                assert!(
                    rel < 1e-4,
                    "period {} idx {}: f64={} f32={} rel_err={}",
                    period,
                    i,
                    f64_out[i],
                    f32_out[i],
                    rel
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU; perf benchmark
    fn bench_sma_f64_vs_f32() {
        use std::time::Instant;
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let period = 20usize;
        for &n in &[100_000usize, 1_000_000, 10_000_000] {
            let close: Array1<f64> =
                Array1::from_iter((0..n).map(|i| 30_000.0 + ((i as f64) * 0.001).sin()));
            // Warmup: compile + cache modules for both paths.
            let _ = sma_gpu_f64(&device, &close, period, None).unwrap();
            let _ = sma_gpu_f32(&device, &close, period, None).unwrap();
            let iters = 20;
            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = sma_gpu_f64(&device, &close, period, None).unwrap();
            }
            let f64_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            let t1 = Instant::now();
            for _ in 0..iters {
                let _ = sma_gpu_f32(&device, &close, period, None).unwrap();
            }
            let f32_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            println!(
                "SMA n={:>9} period={} | f64 {:7.3} ms | f32 {:7.3} ms | speedup {:.2}x",
                n,
                period,
                f64_ms,
                f32_ms,
                f64_ms / f32_ms
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU; quantifies the device-residency lever (profiling §10)
    fn bench_sma_device_resident_vs_reupload() {
        // Simulates a parameter sweep over the SAME close[]: the current path
        // re-uploads close per evaluation (H2D+kernel+D2H each time); the
        // device-resident path uploads once and reuses the on-device buffer.
        use std::time::Instant;
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let n = 1_000_000usize;
        let period = 20usize;
        let m = 50usize; // 50-parameter sweep
        let close: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 30_000.0 + ((i as f64) * 0.001).sin()));

        // Path A: re-upload per evaluation (current sweep behavior).
        let _ = sma_gpu(&device, &close, period, None).unwrap(); // warmup
        let ta = Instant::now();
        for _ in 0..m {
            let _ = sma_gpu(&device, &close, period, None).unwrap();
        }
        let reupload_ms = ta.elapsed().as_secs_f64() * 1000.0;

        // Path B: upload once, run M kernels on the resident buffer.
        let ptx = Arc::unwrap_or_clone(compile_ptx_optimized_cached(SMA_KERNEL_F32).unwrap());
        let module = device.context().load_module(ptx).unwrap();
        let kernel = module.load_function("sma_kernel_f32").unwrap();
        let stream = &device.stream;
        let close_f32: Vec<f32> = close.iter().map(|&x| x as f32).collect();
        let mut d_close = device.alloc_uninit::<f32>(n).unwrap();
        stream.memcpy_htod(&close_f32[..n], &mut d_close).unwrap();
        let mut d_out = device.alloc_uninit::<f32>(n).unwrap();
        let n_i32 = n as i32;
        let period_i32 = period as i32;
        let config = LaunchConfig::for_num_elems(n as u32);
        {
            let mut b = stream.launch_builder(&kernel);
            b.arg(&d_close);
            b.arg(&mut d_out);
            b.arg(&n_i32);
            b.arg(&period_i32);
            unsafe {
                b.launch(config).unwrap();
            }
        }
        stream.synchronize().unwrap();
        let tb = Instant::now();
        for _ in 0..m {
            let mut b = stream.launch_builder(&kernel);
            b.arg(&d_close);
            b.arg(&mut d_out);
            b.arg(&n_i32);
            b.arg(&period_i32);
            unsafe {
                b.launch(config).unwrap();
            }
        }
        stream.synchronize().unwrap();
        let resident_ms = tb.elapsed().as_secs_f64() * 1000.0;

        println!(
            "SMA sweep m={} n={}: re-upload {:.2} ms | device-resident {:.2} ms | {:.1}x",
            m,
            n,
            reupload_ms,
            resident_ms,
            reupload_ms / resident_ms
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_on_device_matches() {
        // The device-resident primitive must match the transfer-based f32 path.
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let n = 50_000usize;
        let close: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 30_000.0 + 5_000.0 * ((i as f64) * 0.001).sin()));
        let close_f32: Vec<f32> = close.iter().map(|&x| x as f32).collect();
        for &period in &[5usize, 20, 200] {
            let reference = sma_gpu_f32(&device, &close, period, None).unwrap();
            let mut d_close = device.alloc_uninit::<f32>(n).unwrap();
            device
                .stream
                .memcpy_htod(&close_f32[..n], &mut d_close)
                .unwrap();
            let mut d_out = device.alloc_uninit::<f32>(n).unwrap();
            sma_on_device(&device, &d_close, &mut d_out, n, period, None).unwrap();
            let mut out_f32 = vec![0.0f32; n];
            device.stream.memcpy_dtoh(&d_out, &mut out_f32[..]).unwrap();
            device.stream.synchronize().unwrap();
            for i in 0..n {
                if reference[i].is_nan() {
                    assert!(out_f32[i].is_nan(), "period {} idx {}: not NaN", period, i);
                } else {
                    assert!(
                        (reference[i] - out_f32[i] as f64).abs() < 1e-3,
                        "period {} idx {}: ref={} dev={}",
                        period,
                        i,
                        reference[i],
                        out_f32[i]
                    );
                }
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU; verifies correctness AND measures the sweep win
    fn bench_sma_sweep_resident_vs_naive() {
        use std::time::Instant;
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let n = 1_000_000usize;
        let periods: Vec<usize> = (5..=54).collect(); // 50-parameter sweep
        let close: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 30_000.0 + ((i as f64) * 0.001).sin()));

        // Warmup both paths.
        let _ = sma_gpu(&device, &close, 20, None).unwrap();
        let _ = sma_sweep_on_device(&device, &close, &[20], None).unwrap();

        // Naive: re-upload close per period (current sweep behavior).
        let t0 = Instant::now();
        let naive: Vec<_> = periods
            .iter()
            .map(|&p| sma_gpu(&device, &close, p, None).unwrap())
            .collect();
        let naive_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Device-resident: upload close once.
        let t1 = Instant::now();
        let resident = sma_sweep_on_device(&device, &close, &periods, None).unwrap();
        let resident_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // Correctness: resident must match the naive per-period result.
        assert_eq!(naive.len(), resident.len());
        for (a, b) in naive.iter().zip(resident.iter()) {
            for i in 0..n {
                if a[i].is_nan() {
                    assert!(b[i].is_nan());
                } else {
                    assert!((a[i] - b[i]).abs() < 1e-3, "mismatch at {}: {} vs {}", i, a[i], b[i]);
                }
            }
        }

        println!(
            "SMA sweep {} periods n={}: naive(re-upload) {:.2} ms | resident(upload-once) {:.2} ms | {:.1}x",
            periods.len(),
            n,
            naive_ms,
            resident_ms,
            naive_ms / resident_ms
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_gpu_prec() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let n = 10_000usize;
        let close: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 30_000.0 + ((i as f64) * 0.01).sin()));
        let period = 20usize;

        // F64 -> exact reference path.
        let f64v = sma_gpu_prec(&device, &close, period, Precision::F64, None).unwrap();
        let f64ref = sma_gpu_f64(&device, &close, period, None).unwrap();
        for i in 0..n {
            if f64ref[i].is_nan() {
                assert!(f64v[i].is_nan());
            } else {
                assert!((f64v[i] - f64ref[i]).abs() < 1e-12);
            }
        }

        // Auto (and F32) -> the f32 path, identical to the default sma_gpu.
        let autov = sma_gpu_prec(&device, &close, period, Precision::Auto, None).unwrap();
        let defaultv = sma_gpu(&device, &close, period, None).unwrap();
        for i in 0..n {
            if defaultv[i].is_nan() {
                assert!(autov[i].is_nan());
            } else {
                assert!((autov[i] - defaultv[i]).abs() < 1e-6);
            }
        }
    }
}
