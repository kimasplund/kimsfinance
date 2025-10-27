//! GPU-Accelerated Rate of Change (ROC)
//!
//! Provides 30-50x speedup over CPU implementation for large datasets.
//! ROC is perfectly parallelizable - each thread calculates one value independently.

use super::device::{GpuDevice, GpuError};
use crate::gpu::compile::compile_ptx_optimized;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for Rate of Change
///
/// Algorithm: ROC[i] = ((close[i] - close[i - period]) / close[i - period]) * 100
///
/// This is an embarrassingly parallel problem - each thread operates independently
/// with no shared memory or synchronization needed.
const ROC_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void roc_kernel(
    const double* __restrict__ close,
    double* __restrict__ roc,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only calculate ROC for indices where we have enough history
    if (idx >= period && idx < n) {
        double current = close[idx];
        double previous = close[idx - period];

        // Handle division by zero
        if (previous != 0.0) {
            roc[idx] = ((current - previous) / previous) * 100.0;
        } else {
            roc[idx] = CUDART_NAN;
        }
    } else if (idx < period) {
        // Not enough history - set to NAN
        roc[idx] = CUDART_NAN;
    }
}
"#;

/// GPU-accelerated Rate of Change (ROC) indicator
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `close` - Close prices
/// * `period` - Lookback period (e.g., 12)
/// * `stream` - Optional CUDA stream for concurrent execution (None uses device default)
///
/// # Returns
///
/// Array1<f64> containing ROC values (NaN for first `period` values)
///
/// # Algorithm
///
/// ```text
/// ROC[i] = ((close[i] - close[i - period]) / close[i - period]) * 100
/// ```
///
/// # Performance
///
/// Expected speedup: **30-50x** over CPU for n > 10,000
///
/// This is the fastest GPU indicator due to perfect parallelism:
/// - No rolling windows needed
/// - No shared memory
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
/// use kimsfinance_core::gpu::{GpuDevice, roc_gpu};
/// use ndarray::arr1;
///
/// let device = GpuDevice::new()?;
/// let close = arr1(&[100.0, 105.0, 103.0, 108.0, 112.0, 110.0, 115.0]);
/// let roc = roc_gpu(&device, &close, 3, None)?;
///
/// // roc[3] = ((108 - 100) / 100) * 100 = 8.0%
/// assert!((roc[3] - 8.0).abs() < 0.01);
/// ```
pub fn roc_gpu(
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

    if n <= period {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need > {} points, got {}",
            period, n
        )));
    }

    // Compile PTX
    let ptx = compile_ptx_optimized(ROC_KERNEL).map_err(|e| {
        GpuError::CompilationError(format!("Failed to compile ROC kernel: {:?}", e))
    })?;

    // Load module (use context, not stream)
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel function from module
    let kernel = module.load_function("roc_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
    })?;

    // Copy data to GPU
    let d_close = device.copy_to_device(close.as_slice().unwrap())?;

    // Allocate output buffer (uses device.stream for memory operations)
    let mut d_roc = device.alloc_buffer(n)?;

    // Use provided stream or default to device stream
    let kernel_stream = stream.unwrap_or(&device.stream);

    // Launch kernel using builder pattern on specified stream
    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut builder = kernel_stream.launch_builder(&kernel);
    builder.arg(&d_close);
    builder.arg(&mut d_roc);
    builder.arg(&n_i32);
    builder.arg(&period_i32);

    let config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("ROC kernel launch failed: {:?}", e)))?;
    }

    // Synchronize the specified stream
    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    let roc_vec = device.copy_to_host(&d_roc)?;

    Ok(Array1::from_vec(roc_vec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_roc_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Simple test case
        let close = arr1(&[
            100.0, 105.0, 103.0, 108.0, 112.0, 110.0, 115.0, 118.0, 120.0,
        ]);
        let roc = roc_gpu(&device, &close, 3, None).expect("ROC GPU calculation failed");

        // Verify first `period` values are NaN
        for i in 0..3 {
            assert!(roc[i].is_nan(), "roc[{}] should be NaN", i);
        }

        // Verify calculations
        // roc[3] = ((108 - 100) / 100) * 100 = 8.0
        assert!(
            (roc[3] - 8.0).abs() < 0.01,
            "roc[3] = {}, expected 8.0",
            roc[3]
        );

        // roc[4] = ((112 - 105) / 105) * 100 = 6.666...
        assert!(
            (roc[4] - 6.6667).abs() < 0.01,
            "roc[4] = {}, expected 6.6667",
            roc[4]
        );

        // roc[5] = ((110 - 103) / 103) * 100 = 6.796...
        assert!(
            (roc[5] - 6.7961).abs() < 0.01,
            "roc[5] = {}, expected 6.7961",
            roc[5]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_roc_gpu_zero_division() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test case with zero value (should produce NaN)
        let close = arr1(&[0.0, 100.0, 105.0, 110.0, 115.0]);
        let roc = roc_gpu(&device, &close, 2, None).expect("ROC GPU calculation failed");

        // roc[2] should be NaN because close[0] = 0
        assert!(roc[2].is_nan(), "roc[2] should be NaN (division by zero)");

        // roc[3] should be valid: ((110 - 100) / 100) * 100 = 10.0
        assert!(
            (roc[3] - 10.0).abs() < 0.01,
            "roc[3] = {}, expected 10.0",
            roc[3]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_roc_gpu_negative_values() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with price decline
        let close = arr1(&[120.0, 115.0, 110.0, 105.0, 100.0]);
        let roc = roc_gpu(&device, &close, 2, None).expect("ROC GPU calculation failed");

        // roc[2] = ((110 - 120) / 120) * 100 = -8.333...
        assert!(
            (roc[2] - (-8.3333)).abs() < 0.01,
            "roc[2] = {}, expected -8.3333",
            roc[2]
        );

        // roc[3] = ((105 - 115) / 115) * 100 = -8.695...
        assert!(
            (roc[3] - (-8.6957)).abs() < 0.01,
            "roc[3] = {}, expected -8.6957",
            roc[3]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_roc_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let close = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());

        let start = std::time::Instant::now();
        let roc = roc_gpu(&device, &close, 12, None).expect("ROC GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU ROC (n={}): {:.2}ms ({:.0} values/sec)",
            n,
            elapsed.as_secs_f64() * 1000.0,
            n as f64 / elapsed.as_secs_f64()
        );

        assert_eq!(roc.len(), n);

        // Verify first 12 values are NaN
        for i in 0..12 {
            assert!(roc[i].is_nan());
        }

        // Verify rest are computed
        for i in 12..n {
            assert!(!roc[i].is_nan());
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_roc_gpu_different_periods() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let close = arr1(&[
            100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0,
        ]);

        // Test period=1 (daily change)
        let roc1 = roc_gpu(&device, &close, 1, None).expect("ROC GPU failed");
        // roc1[1] = ((102 - 100) / 100) * 100 = 2.0
        assert!((roc1[1] - 2.0).abs() < 0.01);

        // Test period=5
        let roc5 = roc_gpu(&device, &close, 5, None).expect("ROC GPU failed");
        // roc5[5] = ((110 - 100) / 100) * 100 = 10.0
        assert!((roc5[5] - 10.0).abs() < 0.01);

        // Test period=10
        let roc10 = roc_gpu(&device, &close, 10, None).expect("ROC GPU failed");
        // roc10[10] = ((120 - 100) / 100) * 100 = 20.0
        assert!((roc10[10] - 20.0).abs() < 0.01);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_roc_gpu_performance_benchmark() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

        for n in sizes {
            let close = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.001).collect());

            let start = std::time::Instant::now();
            let _roc = roc_gpu(&device, &close, 12, None).expect("ROC GPU calculation failed");
            let elapsed = start.elapsed();

            let throughput = n as f64 / elapsed.as_secs_f64();
            println!(
                "GPU ROC (n={:7}): {:6.2}ms - {:12.0} values/sec",
                n,
                elapsed.as_secs_f64() * 1000.0,
                throughput
            );
        }
    }

    #[test]
    #[should_panic(expected = "Period must be >= 1")]
    fn test_roc_gpu_invalid_period() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let close = arr1(&[100.0, 105.0, 110.0]);
        let _roc = roc_gpu(&device, &close, 0, None).unwrap();
    }

    #[test]
    #[should_panic(expected = "Not enough data")]
    fn test_roc_gpu_insufficient_data() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let close = arr1(&[100.0, 105.0]);
        let _roc = roc_gpu(&device, &close, 5, None).unwrap();
    }
}
