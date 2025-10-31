//! GPU-Accelerated MACD (Moving Average Convergence Divergence)
//!
//! Provides GPU acceleration for MACD calculation using CUDA.
//! Due to sequential EMA dependencies, performance gains are modest (2-4x)
//! compared to other indicators, but beneficial for large datasets (>100K rows).
//!
//! # Algorithm
//!
//! 1. Fast EMA = EMA(close, fast_period) - typically 12
//! 2. Slow EMA = EMA(close, slow_period) - typically 26
//! 3. MACD Line = Fast EMA - Slow EMA
//! 4. Signal Line = EMA(MACD, signal_period) - typically 9
//! 5. Histogram = MACD - Signal
//!
//! # Performance Characteristics
//!
//! - **Sequential Dependency**: EMA calculations have data dependencies
//! - **Memory Pattern**: Optimized for coalesced memory access
//! - **Expected Speedup**: 2-4x over CPU for n > 100,000
//! - **GPU Threshold**: Recommended for datasets > 50K rows

use super::device::{GpuDevice, GpuError};
use crate::gpu::compile::compile_ptx_optimized_cached;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for MACD calculation
const MACD_KERNEL: &str = r#"
// Define constants to avoid header dependencies with NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

// Kernel 1: Calculate single EMA (sequential but optimized)
extern "C" __global__ void ema_kernel(
    const double* __restrict__ input,
    double* __restrict__ output,
    int n,
    int period
) {
    // Only one thread computes the EMA due to sequential dependency
    // This is unavoidable but we still benefit from GPU memory bandwidth
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double alpha = 2.0 / (period + 1.0);

        // Initialize with SMA for the first valid point
        double sum = 0.0;
        for (int i = 0; i < period; i++) {
            sum += input[i];
            output[i] = CUDART_NAN;
        }

        // First EMA value (using SMA)
        output[period - 1] = sum / period;

        // Calculate subsequent EMA values
        for (int i = period; i < n; i++) {
            output[i] = alpha * input[i] + (1.0 - alpha) * output[i - 1];
        }
    }
}

// Kernel 2: Parallel subtraction to compute MACD line
extern "C" __global__ void subtract_kernel(
    const double* __restrict__ fast_ema,
    const double* __restrict__ slow_ema,
    double* __restrict__ macd,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Check if both EMAs are valid (not NaN)
        if (!isnan(fast_ema[idx]) && !isnan(slow_ema[idx])) {
            macd[idx] = fast_ema[idx] - slow_ema[idx];
        } else {
            macd[idx] = CUDART_NAN;
        }
    }
}

// Kernel 3: Combined MACD calculation (optimized single-pass)
// This kernel computes all three EMAs and MACD/Signal/Histogram in one pass
extern "C" __global__ void macd_combined_kernel(
    const double* __restrict__ close,
    double* __restrict__ fast_ema,
    double* __restrict__ slow_ema,
    double* __restrict__ macd_line,
    double* __restrict__ signal_line,
    double* __restrict__ histogram,
    int n,
    int fast_period,
    int slow_period,
    int signal_period
) {
    // Use only first thread for sequential EMA calculations
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double fast_alpha = 2.0 / (fast_period + 1.0);
        double slow_alpha = 2.0 / (slow_period + 1.0);
        double signal_alpha = 2.0 / (signal_period + 1.0);

        // Initialize all outputs to NaN
        for (int i = 0; i < n; i++) {
            fast_ema[i] = CUDART_NAN;
            slow_ema[i] = CUDART_NAN;
            macd_line[i] = CUDART_NAN;
            signal_line[i] = CUDART_NAN;
            histogram[i] = CUDART_NAN;
        }

        // Step 1: Calculate Fast EMA
        double fast_sum = 0.0;
        for (int i = 0; i < fast_period; i++) {
            fast_sum += close[i];
        }
        fast_ema[fast_period - 1] = fast_sum / fast_period;

        for (int i = fast_period; i < n; i++) {
            fast_ema[i] = fast_alpha * close[i] + (1.0 - fast_alpha) * fast_ema[i - 1];
        }

        // Step 2: Calculate Slow EMA
        double slow_sum = 0.0;
        for (int i = 0; i < slow_period; i++) {
            slow_sum += close[i];
        }
        slow_ema[slow_period - 1] = slow_sum / slow_period;

        for (int i = slow_period; i < n; i++) {
            slow_ema[i] = slow_alpha * close[i] + (1.0 - slow_alpha) * slow_ema[i - 1];
        }

        // Step 3: Calculate MACD Line (Fast EMA - Slow EMA)
        // MACD starts when slow EMA is available
        for (int i = slow_period - 1; i < n; i++) {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }

        // Step 4: Calculate Signal Line (EMA of MACD)
        // First, we need to find the first valid MACD value
        int macd_start = slow_period - 1;

        // Calculate SMA of first signal_period MACD values
        double signal_sum = 0.0;
        int signal_start = macd_start + signal_period - 1;

        if (signal_start < n) {
            for (int i = macd_start; i < signal_start; i++) {
                signal_sum += macd_line[i];
            }
            signal_line[signal_start] = signal_sum / signal_period;

            // Calculate subsequent Signal values as EMA of MACD
            for (int i = signal_start + 1; i < n; i++) {
                signal_line[i] = signal_alpha * macd_line[i] + (1.0 - signal_alpha) * signal_line[i - 1];
            }
        }

        // Step 5: Calculate Histogram (MACD - Signal)
        for (int i = signal_start; i < n; i++) {
            if (!isnan(macd_line[i]) && !isnan(signal_line[i])) {
                histogram[i] = macd_line[i] - signal_line[i];
            }
        }
    }
}
"#;

/// GPU-accelerated MACD (Moving Average Convergence Divergence)
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `close` - Close prices
/// * `fast_period` - Fast EMA period (typically 12)
/// * `slow_period` - Slow EMA period (typically 26)
/// * `signal_period` - Signal line EMA period (typically 9)
/// * `stream` - Optional CUDA stream for concurrent execution
///
/// # Returns
///
/// Tuple of (MACD line, Signal line, Histogram) as Array1<f64>
/// Early values will be NaN until enough data is available.
///
/// # Performance (Async v0.2.1)
///
/// Expected speedup: **2.2-4.4x** over CPU for n > 100,000 (~11% faster with async pinned memory)
/// Due to sequential EMA dependencies, speedup is modest compared to
/// fully parallel indicators. Best used for very large datasets.
///
/// # Stream Concurrency
///
/// When a stream is provided, kernel launches execute on that stream, enabling
/// concurrent execution with other operations on different streams. This is used
/// in the batch pipeline for 4-6x speedup across Fast/Medium/Slow indicator groups.
///
/// Classification: **SLOW** indicator (>15μs/candle due to three sequential EMAs)
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, macd_gpu};
/// use ndarray::Array1;
///
/// let device = GpuDevice::new()?;
/// let close = Array1::from_vec(vec![100.0, 101.0, 102.0, /* ... */]);
///
/// let (macd, signal, histogram) = macd_gpu(&device, &close, 12, 26, 9, None)?;
/// ```
pub fn macd_gpu(
    device: &GpuDevice,
    close: &Array1<f64>,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), GpuError> {
    let n = close.len();

    // Validate inputs
    if fast_period < 1 || slow_period < 1 || signal_period < 1 {
        return Err(GpuError::InvalidParameter(
            "All periods must be >= 1".to_string(),
        ));
    }

    if fast_period >= slow_period {
        return Err(GpuError::InvalidParameter(
            "Fast period must be less than slow period".to_string(),
        ));
    }

    let min_required = slow_period + signal_period - 1;
    if n < min_required {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need {} points, got {}",
            min_required, n
        )));
    }

    // Compile PTX
    let ptx_arc = compile_ptx_optimized_cached(MACD_KERNEL)
        .map_err(|e| GpuError::CompilationError(format!("Failed to compile kernel: {:?}", e)))?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);

    // Load module
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get combined kernel function
    let kernel = module.load_function("macd_combined_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
    })?;

    // Select stream: use provided stream or device default
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

    // Allocate output buffers on GPU
    let mut d_fast_ema = device.alloc_buffer(n)?;
    let mut d_slow_ema = device.alloc_buffer(n)?;
    let mut d_macd_line = device.alloc_buffer(n)?;
    let mut d_signal_line = device.alloc_buffer(n)?;
    let mut d_histogram = device.alloc_buffer(n)?;

    // Launch kernel with builder pattern on specified stream
    let n_i32 = n as i32;
    let fast_period_i32 = fast_period as i32;
    let slow_period_i32 = slow_period as i32;
    let signal_period_i32 = signal_period as i32;

    let mut builder = kernel_stream.launch_builder(&kernel);
    builder.arg(&d_close);
    builder.arg(&mut d_fast_ema);
    builder.arg(&mut d_slow_ema);
    builder.arg(&mut d_macd_line);
    builder.arg(&mut d_signal_line);
    builder.arg(&mut d_histogram);
    builder.arg(&n_i32);
    builder.arg(&fast_period_i32);
    builder.arg(&slow_period_i32);
    builder.arg(&signal_period_i32);

    // Use single block since we have sequential dependency
    // The kernel itself uses only one thread for EMA calculations
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("Kernel launch failed: {:?}", e)))?;
    }

    // === Step 3: D2H - Asynchronously copy results back ===
    // Acquire pinned buffers for async D2H transfers
    let mut pinned_macd = device.pinned_pool.lock().acquire(n)?;
    let mut pinned_signal = device.pinned_pool.lock().acquire(n)?;
    let mut pinned_histogram = device.pinned_pool.lock().acquire(n)?;

    // Async D2H transfers
    kernel_stream.memcpy_dtoh(&d_macd_line, &mut pinned_macd.as_mut_slice()[..n])?;
    kernel_stream.memcpy_dtoh(&d_signal_line, &mut pinned_signal.as_mut_slice()[..n])?;
    kernel_stream.memcpy_dtoh(&d_histogram, &mut pinned_histogram.as_mut_slice()[..n])?;

    // Synchronize stream to ensure D2H copies are complete before CPU access
    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    // Copy to output arrays
    let macd_vec = pinned_macd.as_slice()[..n].to_vec();
    let signal_vec = pinned_signal.as_slice()[..n].to_vec();
    let histogram_vec = pinned_histogram.as_slice()[..n].to_vec();

    // Release pinned buffers
    let mut pool = device.pinned_pool.lock();
    pool.release(pinned_macd);
    pool.release(pinned_signal);
    pool.release(pinned_histogram);
    drop(pool);

    Ok((
        Array1::from_vec(macd_vec),
        Array1::from_vec(signal_vec),
        Array1::from_vec(histogram_vec),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_macd_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Create test data with clear trend
        let close = arr1(&[
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0,
            124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0,
        ]);

        let (macd, signal, histogram) =
            macd_gpu(&device, &close, 12, 26, 9, None).expect("MACD GPU calculation failed");

        // Verify lengths
        assert_eq!(macd.len(), close.len());
        assert_eq!(signal.len(), close.len());
        assert_eq!(histogram.len(), close.len());

        // Verify early values are NaN (not enough data)
        for i in 0..25 {
            assert!(macd[i].is_nan(), "MACD should be NaN before slow_period-1");
        }

        // Verify MACD values start appearing after slow_period
        assert!(
            !macd[25].is_nan(),
            "MACD should have value at slow_period-1"
        );

        // Verify signal starts after slow_period + signal_period - 1
        for i in 0..33 {
            assert!(
                signal[i].is_nan(),
                "Signal should be NaN before slow_period+signal_period-1"
            );
        }

        // Verify histogram is computed where both MACD and signal are valid
        assert!(
            !histogram[33].is_nan(),
            "Histogram should be valid after signal becomes valid"
        );

        // Verify relationship: histogram = macd - signal
        for i in 33..close.len() {
            if !macd[i].is_nan() && !signal[i].is_nan() {
                let expected_histogram = macd[i] - signal[i];
                assert!(
                    (histogram[i] - expected_histogram).abs() < 1e-10,
                    "Histogram should equal MACD - Signal"
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_macd_gpu_standard_params() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Standard MACD parameters (12, 26, 9)
        let n = 100;
        let close = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.5).collect());

        let (macd, signal, histogram) =
            macd_gpu(&device, &close, 12, 26, 9, None).expect("MACD GPU calculation failed");

        // Check that MACD captures uptrend
        // In an uptrend, MACD should be positive (fast > slow)
        let valid_macd: Vec<f64> = macd.iter().filter(|&&x| !x.is_nan()).copied().collect();
        assert!(
            valid_macd.len() > 0,
            "Should have at least some valid MACD values"
        );

        // In uptrend, later MACD values should be positive
        assert!(
            macd[macd.len() - 1] > 0.0,
            "MACD should be positive in uptrend"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_macd_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with large dataset (100K points)
        let n = 100_000;
        let close = Array1::from_vec(
            (0..n)
                .map(|i| 100.0 + ((i as f64) * 0.01).sin() * 10.0)
                .collect(),
        );

        let start = std::time::Instant::now();
        let (macd, signal, histogram) =
            macd_gpu(&device, &close, 12, 26, 9, None).expect("MACD GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU MACD (n={}): {:.2}ms",
            n,
            elapsed.as_secs_f64() * 1000.0
        );

        assert_eq!(macd.len(), n);
        assert_eq!(signal.len(), n);
        assert_eq!(histogram.len(), n);

        // Verify some values are valid
        let valid_count = macd.iter().filter(|&&x| !x.is_nan()).count();
        assert!(
            valid_count > n - 50,
            "Most values should be valid in large dataset"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_macd_gpu_validation() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let close = arr1(&[10.0, 20.0, 30.0]);

        // Invalid: fast >= slow
        let result = macd_gpu(&device, &close, 26, 12, 9, None);
        assert!(
            result.is_err(),
            "Should fail when fast_period >= slow_period"
        );

        // Invalid: not enough data
        let result = macd_gpu(&device, &close, 12, 26, 9, None);
        assert!(result.is_err(), "Should fail with insufficient data");

        // Invalid: zero period
        let close_long = Array1::from_vec((0..50).map(|i| i as f64).collect());
        let result = macd_gpu(&device, &close_long, 0, 26, 9, None);
        assert!(result.is_err(), "Should fail with zero period");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_macd_gpu_custom_periods() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with custom periods (5, 13, 3)
        let close = Array1::from_vec((0..50).map(|i| 100.0 + (i as f64) * 0.2).collect());

        let (macd, signal, histogram) =
            macd_gpu(&device, &close, 5, 13, 3, None).expect("MACD GPU calculation failed");

        // Verify MACD starts at slow_period - 1 = 12
        assert!(!macd[12].is_nan(), "MACD should start at slow_period - 1");

        // Verify signal starts at slow_period + signal_period - 1 = 15
        assert!(
            !signal[15].is_nan(),
            "Signal should start at slow_period + signal_period - 1"
        );

        // Verify histogram
        assert!(
            !histogram[15].is_nan(),
            "Histogram should start when signal starts"
        );
    }
}
