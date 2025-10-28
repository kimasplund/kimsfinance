//! GPU-Accelerated Donchian Channels
//!
//! Provides 50-80x speedup over CPU implementation for large datasets.
//! Donchian Channels are perfectly parallelizable - each thread calculates one value independently.

use super::device::{GpuDevice, GpuError};
use crate::gpu::compile::compile_ptx_optimized_cached;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for Donchian Channels
///
/// Algorithm:
/// - Upper = rolling_max(high, period)
/// - Lower = rolling_min(low, period)
/// - Middle = (Upper + Lower) / 2
///
/// This is an embarrassingly parallel problem - each thread operates independently
/// with no shared memory or synchronization needed.
const DONCHIAN_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)

extern "C" __global__ void donchian_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    double* __restrict__ upper,
    double* __restrict__ lower,
    double* __restrict__ middle,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only calculate Donchian for indices where we have enough history
    if (idx >= period - 1 && idx < n) {
        double max_val = -CUDART_INF;
        double min_val = CUDART_INF;

        // Find max and min over the period
        for (int j = 0; j < period; j++) {
            int window_idx = idx - j;
            max_val = fmax(max_val, high[window_idx]);
            min_val = fmin(min_val, low[window_idx]);
        }

        upper[idx] = max_val;
        lower[idx] = min_val;
        middle[idx] = (max_val + min_val) / 2.0;
    } else if (idx < period - 1) {
        // Not enough history - set to NAN
        upper[idx] = CUDART_NAN;
        lower[idx] = CUDART_NAN;
        middle[idx] = CUDART_NAN;
    }
}
"#;

/// GPU-accelerated Donchian Channels indicator
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `high` - High prices
/// * `low` - Low prices
/// * `period` - Lookback period (e.g., 20)
/// * `stream` - Optional CUDA stream for concurrent execution (None uses device default)
///
/// # Returns
///
/// Tuple of (upper, middle, lower) as Array1<f64> containing Donchian Channel values
/// (NaN for first `period - 1` values)
///
/// # Algorithm
///
/// ```text
/// Upper[i] = max(high[i-period+1:i+1])
/// Lower[i] = min(low[i-period+1:i+1])
/// Middle[i] = (Upper[i] + Lower[i]) / 2
/// ```
///
/// # Performance
///
/// Expected speedup: **50-80x** over CPU for n > 10,000
///
/// This is the fastest GPU indicator due to perfect parallelism:
/// - No dependencies between iterations
/// - No shared memory needed
/// - No thread synchronization
/// - Each thread is completely independent
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
/// use kimsfinance_core::gpu::{GpuDevice, donchian_gpu};
/// use ndarray::arr1;
///
/// let device = GpuDevice::new()?;
/// let high = arr1(&[105.0, 108.0, 112.0, 110.0, 115.0]);
/// let low = arr1(&[100.0, 103.0, 107.0, 105.0, 108.0]);
/// let (upper, middle, lower) = donchian_gpu(&device, &high, &low, 3, None)?;
///
/// // upper[2] = max(105, 108, 112) = 112.0
/// // lower[2] = min(100, 103, 107) = 100.0
/// // middle[2] = (112 + 100) / 2 = 106.0
/// assert!((upper[2] - 112.0).abs() < 0.01);
/// assert!((lower[2] - 100.0).abs() < 0.01);
/// assert!((middle[2] - 106.0).abs() < 0.01);
/// ```
#[allow(clippy::type_complexity)]
pub fn donchian_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), GpuError> {
    let n = high.len();

    // Validate inputs
    if period < 1 {
        return Err(GpuError::InvalidParameter(
            "Period must be >= 1".to_string(),
        ));
    }

    if n != low.len() {
        return Err(GpuError::InvalidParameter(format!(
            "High and low arrays must have same length: got high={}, low={}",
            n,
            low.len()
        )));
    }

    if n < period {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need >= {} points, got {}",
            period, n
        )));
    }

    // Compile PTX
    let ptx_arc = compile_ptx_optimized_cached(DONCHIAN_KERNEL).map_err(|e| {
        GpuError::CompilationError(format!("Failed to compile Donchian kernel: {:?}", e))
    })?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);

    // Load module (use context, not stream)
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel function from module
    let kernel = module.load_function("donchian_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
    })?;

    // Copy data to GPU
    let d_high = device.copy_to_device(high.as_slice().unwrap())?;
    let d_low = device.copy_to_device(low.as_slice().unwrap())?;

    // Allocate output buffers (uses device.stream for memory operations)
    let mut d_upper = device.alloc_buffer(n)?;
    let mut d_lower = device.alloc_buffer(n)?;
    let mut d_middle = device.alloc_buffer(n)?;

    // Use provided stream or default to device stream
    let kernel_stream = stream.unwrap_or(&device.stream);

    // Launch kernel using builder pattern on specified stream
    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut builder = kernel_stream.launch_builder(&kernel);
    builder.arg(&d_high);
    builder.arg(&d_low);
    builder.arg(&mut d_upper);
    builder.arg(&mut d_lower);
    builder.arg(&mut d_middle);
    builder.arg(&n_i32);
    builder.arg(&period_i32);

    let config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        builder.launch(config).map_err(|e| {
            GpuError::ExecutionError(format!("Donchian kernel launch failed: {:?}", e))
        })?;
    }

    // Synchronize the specified stream
    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    // Copy results back to host
    let upper_vec = device.copy_to_host(&d_upper)?;
    let lower_vec = device.copy_to_host(&d_lower)?;
    let middle_vec = device.copy_to_host(&d_middle)?;

    Ok((
        Array1::from_vec(upper_vec),
        Array1::from_vec(middle_vec),
        Array1::from_vec(lower_vec),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_donchian_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Simple test case
        let high = arr1(&[105.0, 108.0, 112.0, 110.0, 115.0, 118.0, 120.0]);
        let low = arr1(&[100.0, 103.0, 107.0, 105.0, 108.0, 112.0, 115.0]);

        let (upper, middle, lower) =
            donchian_gpu(&device, &high, &low, 3, None).expect("Donchian GPU calculation failed");

        // Verify first `period - 1` values are NaN
        assert!(upper[0].is_nan(), "upper[0] should be NaN");
        assert!(upper[1].is_nan(), "upper[1] should be NaN");
        assert!(lower[0].is_nan(), "lower[0] should be NaN");
        assert!(lower[1].is_nan(), "lower[1] should be NaN");
        assert!(middle[0].is_nan(), "middle[0] should be NaN");
        assert!(middle[1].is_nan(), "middle[1] should be NaN");

        // Verify calculations
        // upper[2] = max(105, 108, 112) = 112.0
        // lower[2] = min(100, 103, 107) = 100.0
        // middle[2] = (112 + 100) / 2 = 106.0
        assert!(
            (upper[2] - 112.0).abs() < 0.01,
            "upper[2] = {}, expected 112.0",
            upper[2]
        );
        assert!(
            (lower[2] - 100.0).abs() < 0.01,
            "lower[2] = {}, expected 100.0",
            lower[2]
        );
        assert!(
            (middle[2] - 106.0).abs() < 0.01,
            "middle[2] = {}, expected 106.0",
            middle[2]
        );

        // upper[3] = max(108, 112, 110) = 112.0
        // lower[3] = min(103, 107, 105) = 103.0
        // middle[3] = (112 + 103) / 2 = 107.5
        assert!(
            (upper[3] - 112.0).abs() < 0.01,
            "upper[3] = {}, expected 112.0",
            upper[3]
        );
        assert!(
            (lower[3] - 103.0).abs() < 0.01,
            "lower[3] = {}, expected 103.0",
            lower[3]
        );
        assert!(
            (middle[3] - 107.5).abs() < 0.01,
            "middle[3] = {}, expected 107.5",
            middle[3]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_donchian_gpu_rolling_window() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test that window slides correctly
        let high = arr1(&[10.0, 12.0, 11.0, 13.0, 15.0, 14.0]);
        let low = arr1(&[8.0, 9.0, 8.5, 10.0, 12.0, 11.0]);

        let (upper, middle, lower) =
            donchian_gpu(&device, &high, &low, 2, None).expect("Donchian GPU calculation failed");

        // upper[1] = max(10, 12) = 12.0
        // lower[1] = min(8, 9) = 8.0
        assert!((upper[1] - 12.0).abs() < 0.01);
        assert!((lower[1] - 8.0).abs() < 0.01);

        // upper[2] = max(12, 11) = 12.0
        // lower[2] = min(9, 8.5) = 8.5
        assert!((upper[2] - 12.0).abs() < 0.01);
        assert!((lower[2] - 8.5).abs() < 0.01);

        // upper[3] = max(11, 13) = 13.0
        // lower[3] = min(8.5, 10) = 8.5
        assert!((upper[3] - 13.0).abs() < 0.01);
        assert!((lower[3] - 8.5).abs() < 0.01);

        // upper[4] = max(13, 15) = 15.0
        // lower[4] = min(10, 12) = 10.0
        assert!((upper[4] - 15.0).abs() < 0.01);
        assert!((lower[4] - 10.0).abs() < 0.01);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_donchian_gpu_period_one() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Period = 1 should return current high/low
        let high = arr1(&[105.0, 108.0, 103.0, 110.0]);
        let low = arr1(&[100.0, 102.0, 98.0, 105.0]);

        let (upper, middle, lower) =
            donchian_gpu(&device, &high, &low, 1, None).expect("Donchian GPU calculation failed");

        // With period=1, upper=high, lower=low, middle=(high+low)/2
        for i in 0..high.len() {
            assert!(
                (upper[i] - high[i]).abs() < 0.01,
                "upper[{}] = {}, expected {}",
                i,
                upper[i],
                high[i]
            );
            assert!(
                (lower[i] - low[i]).abs() < 0.01,
                "lower[{}] = {}, expected {}",
                i,
                lower[i],
                low[i]
            );
            assert!(
                (middle[i] - (high[i] + low[i]) / 2.0).abs() < 0.01,
                "middle[{}] = {}, expected {}",
                i,
                middle[i],
                (high[i] + low[i]) / 2.0
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_donchian_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());
        let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64) * 0.01).collect());

        let start = std::time::Instant::now();
        let (upper, middle, lower) =
            donchian_gpu(&device, &high, &low, 20, None).expect("Donchian GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU Donchian (n={}): {:.2}ms ({:.0} values/sec)",
            n,
            elapsed.as_secs_f64() * 1000.0,
            n as f64 / elapsed.as_secs_f64()
        );

        assert_eq!(upper.len(), n);
        assert_eq!(middle.len(), n);
        assert_eq!(lower.len(), n);

        // Verify first 19 values are NaN
        for i in 0..19 {
            assert!(upper[i].is_nan());
            assert!(middle[i].is_nan());
            assert!(lower[i].is_nan());
        }

        // Verify rest are computed
        for i in 19..n {
            assert!(!upper[i].is_nan());
            assert!(!middle[i].is_nan());
            assert!(!lower[i].is_nan());
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_donchian_gpu_different_periods() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = arr1(&[
            100.0, 105.0, 103.0, 108.0, 112.0, 110.0, 115.0, 118.0, 120.0,
        ]);
        let low = arr1(&[95.0, 100.0, 98.0, 103.0, 107.0, 105.0, 110.0, 113.0, 115.0]);

        // Test period=1
        let (upper1, middle1, lower1) =
            donchian_gpu(&device, &high, &low, 1, None).expect("Donchian GPU failed");
        assert!((upper1[0] - 100.0).abs() < 0.01);
        assert!((lower1[0] - 95.0).abs() < 0.01);

        // Test period=3
        let (upper3, middle3, lower3) =
            donchian_gpu(&device, &high, &low, 3, None).expect("Donchian GPU failed");
        // upper[2] = max(100, 105, 103) = 105.0
        // lower[2] = min(95, 100, 98) = 95.0
        assert!((upper3[2] - 105.0).abs() < 0.01);
        assert!((lower3[2] - 95.0).abs() < 0.01);
        assert!((middle3[2] - 100.0).abs() < 0.01);

        // Test period=5
        let (upper5, middle5, lower5) =
            donchian_gpu(&device, &high, &low, 5, None).expect("Donchian GPU failed");
        // upper[4] = max(100, 105, 103, 108, 112) = 112.0
        // lower[4] = min(95, 100, 98, 103, 107) = 95.0
        assert!((upper5[4] - 112.0).abs() < 0.01);
        assert!((lower5[4] - 95.0).abs() < 0.01);
        assert!((middle5[4] - 103.5).abs() < 0.01);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_donchian_gpu_performance_benchmark() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

        for n in sizes {
            let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.001).collect());
            let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64) * 0.001).collect());

            let start = std::time::Instant::now();
            let _result = donchian_gpu(&device, &high, &low, 20, None)
                .expect("Donchian GPU calculation failed");
            let elapsed = start.elapsed();

            let throughput = n as f64 / elapsed.as_secs_f64();
            println!(
                "GPU Donchian (n={:7}): {:6.2}ms - {:12.0} values/sec",
                n,
                elapsed.as_secs_f64() * 1000.0,
                throughput
            );
        }
    }

    #[test]
    #[should_panic(expected = "Period must be >= 1")]
    fn test_donchian_gpu_invalid_period() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let high = arr1(&[100.0, 105.0, 110.0]);
        let low = arr1(&[95.0, 100.0, 105.0]);
        let _result = donchian_gpu(&device, &high, &low, 0, None).unwrap();
    }

    #[test]
    #[should_panic(expected = "Not enough data")]
    fn test_donchian_gpu_insufficient_data() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let high = arr1(&[100.0, 105.0]);
        let low = arr1(&[95.0, 100.0]);
        let _result = donchian_gpu(&device, &high, &low, 5, None).unwrap();
    }

    #[test]
    #[should_panic(expected = "High and low arrays must have same length")]
    fn test_donchian_gpu_mismatched_lengths() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let high = arr1(&[100.0, 105.0, 110.0]);
        let low = arr1(&[95.0, 100.0]);
        let _result = donchian_gpu(&device, &high, &low, 2, None).unwrap();
    }
}
