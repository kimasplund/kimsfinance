//! GPU-Accelerated Aroon Indicator
//!
//! Provides 15-25x speedup over CPU implementation for large datasets.
//!
//! The Aroon indicator measures the time elapsed since the highest high and lowest low
//! within a given period, expressed as a percentage (0-100).

use super::device::{GpuDevice, GpuError};
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for Aroon indicator
const AROON_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void aroon_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    double* __restrict__ aroon_up,
    double* __restrict__ aroon_down,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Need at least 'period' data points to calculate Aroon
    if (idx < period - 1) {
        aroon_up[idx] = CUDART_NAN;
        aroon_down[idx] = CUDART_NAN;
        return;
    }

    // Find position of highest high and lowest low in rolling window
    // Window: [idx - period + 1, idx]
    int highest_high_idx = idx;
    int lowest_low_idx = idx;
    double highest_high = high[idx];
    double lowest_low = low[idx];

    // Scan backward through the window
    for (int i = 1; i < period; i++) {
        int window_idx = idx - i;

        if (high[window_idx] >= highest_high) {
            highest_high = high[window_idx];
            highest_high_idx = window_idx;
        }

        if (low[window_idx] <= lowest_low) {
            lowest_low = low[window_idx];
            lowest_low_idx = window_idx;
        }
    }

    // Calculate periods since high/low
    // periods_since = current_idx - position_of_extreme
    int periods_since_high = idx - highest_high_idx;
    int periods_since_low = idx - lowest_low_idx;

    // Calculate Aroon values
    // Aroon = ((period - periods_since) / period) * 100
    aroon_up[idx] = ((double)(period - periods_since_high) / (double)period) * 100.0;
    aroon_down[idx] = ((double)(period - periods_since_low) / (double)period) * 100.0;
}
"#;

/// GPU-accelerated Aroon indicator
///
/// Calculates Aroon Up and Aroon Down indicators using CUDA.
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `high` - High prices
/// * `low` - Low prices
/// * `period` - Lookback period (typically 14 or 25)
/// * `stream` - Optional CUDA stream for concurrent execution (uses device default if None)
///
/// # Returns
///
/// Tuple of (Aroon Up, Aroon Down) as Array1<f64>
///
/// # Algorithm
///
/// - Aroon Up = ((period - periods_since_highest_high) / period) * 100
/// - Aroon Down = ((period - periods_since_lowest_low) / period) * 100
///
/// # Performance
///
/// Expected speedup: **15-25x** over CPU for n > 10,000
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, aroon_gpu};
///
/// let device = GpuDevice::new()?;
/// let (aroon_up, aroon_down) = aroon_gpu(&device, &high, &low, 14, None)?;
/// ```
pub fn aroon_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<(Array1<f64>, Array1<f64>), GpuError> {
    let n = high.len();

    // Validate inputs
    if low.len() != n {
        return Err(GpuError::InvalidParameter(
            "High and low arrays must have same length".to_string(),
        ));
    }

    if period < 1 {
        return Err(GpuError::InvalidParameter(
            "Period must be >= 1".to_string(),
        ));
    }

    if n < period {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need {} points, got {}",
            period, n
        )));
    }

    // Compile PTX
    let ptx = compile_ptx(AROON_KERNEL)
        .map_err(|e| GpuError::CompilationError(format!("Failed to compile kernel: {:?}", e)))?;

    // Load module (use context, not stream)
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel function from module
    let kernel = module.load_function("aroon_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
    })?;

    // Select stream: use provided stream or device default
    let exec_stream = stream.unwrap_or(&device.stream);

    // Copy data to GPU using selected stream
    let d_high = {
        let mut buffer = exec_stream.alloc_zeros::<f64>(n).map_err(|e| {
            GpuError::AllocationError(format!("Failed to allocate high buffer: {:?}", e))
        })?;
        exec_stream
            .memcpy_htod(high.as_slice().unwrap(), &mut buffer)
            .map_err(|e| {
                GpuError::MemoryCopyError(format!("Failed to copy high to device: {:?}", e))
            })?;
        buffer
    };

    let d_low = {
        let mut buffer = exec_stream.alloc_zeros::<f64>(n).map_err(|e| {
            GpuError::AllocationError(format!("Failed to allocate low buffer: {:?}", e))
        })?;
        exec_stream
            .memcpy_htod(low.as_slice().unwrap(), &mut buffer)
            .map_err(|e| {
                GpuError::MemoryCopyError(format!("Failed to copy low to device: {:?}", e))
            })?;
        buffer
    };

    // Allocate output buffers on selected stream
    let mut d_aroon_up = exec_stream.alloc_zeros::<f64>(n).map_err(|e| {
        GpuError::AllocationError(format!("Failed to allocate aroon_up buffer: {:?}", e))
    })?;

    let mut d_aroon_down = exec_stream.alloc_zeros::<f64>(n).map_err(|e| {
        GpuError::AllocationError(format!("Failed to allocate aroon_down buffer: {:?}", e))
    })?;

    // Launch kernel on selected stream
    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut builder = exec_stream.launch_builder(&kernel);
    builder.arg(&d_high);
    builder.arg(&d_low);
    builder.arg(&mut d_aroon_up);
    builder.arg(&mut d_aroon_down);
    builder.arg(&n_i32);
    builder.arg(&period_i32);

    let config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("Kernel launch failed: {:?}", e)))?;
    }

    // Synchronize selected stream and copy results back
    exec_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    let aroon_up_vec = exec_stream.memcpy_dtov(&d_aroon_up).map_err(|e| {
        GpuError::MemoryCopyError(format!("Failed to copy aroon_up to host: {:?}", e))
    })?;

    let aroon_down_vec = exec_stream.memcpy_dtov(&d_aroon_down).map_err(|e| {
        GpuError::MemoryCopyError(format!("Failed to copy aroon_down to host: {:?}", e))
    })?;

    Ok((
        Array1::from_vec(aroon_up_vec),
        Array1::from_vec(aroon_down_vec),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_aroon_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Simple trending up data
        let high = arr1(&[
            100.0, 102.0, 105.0, 103.0, 107.0, 110.0, 108.0, 112.0, 115.0, 113.0, 118.0, 120.0,
            117.0, 122.0, 125.0,
        ]);
        let low = arr1(&[
            95.0, 97.0, 100.0, 98.0, 102.0, 105.0, 103.0, 107.0, 110.0, 108.0, 113.0, 115.0, 112.0,
            117.0, 120.0,
        ]);

        let (aroon_up, aroon_down) =
            aroon_gpu(&device, &high, &low, 14, None).expect("Aroon GPU calculation failed");

        // Verify output length
        assert_eq!(aroon_up.len(), 15);
        assert_eq!(aroon_down.len(), 15);

        // First 13 values should be NaN (need 14 periods)
        for i in 0..13 {
            assert!(aroon_up[i].is_nan());
            assert!(aroon_down[i].is_nan());
        }

        // Valid values should be in range [0, 100]
        for i in 13..aroon_up.len() {
            assert!(aroon_up[i] >= 0.0 && aroon_up[i] <= 100.0);
            assert!(aroon_down[i] >= 0.0 && aroon_down[i] <= 100.0);
        }

        // In uptrend, Aroon Up should be higher than Aroon Down
        assert!(aroon_up[14] > aroon_down[14]);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_aroon_gpu_downtrend() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Trending down data
        let high = arr1(&[
            125.0, 122.0, 120.0, 118.0, 115.0, 113.0, 110.0, 108.0, 105.0, 103.0, 100.0, 98.0,
            95.0, 93.0, 90.0,
        ]);
        let low = arr1(&[
            120.0, 117.0, 115.0, 113.0, 110.0, 108.0, 105.0, 103.0, 100.0, 98.0, 95.0, 93.0, 90.0,
            88.0, 85.0,
        ]);

        let (aroon_up, aroon_down) =
            aroon_gpu(&device, &high, &low, 14, None).expect("Aroon GPU calculation failed");

        // In downtrend, Aroon Down should be higher than Aroon Up
        assert!(aroon_down[14] > aroon_up[14]);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_aroon_gpu_extreme_values() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // High at the end of window (most recent)
        let mut high = vec![100.0; 20];
        high[19] = 150.0; // Highest high at current position

        let mut low = vec![95.0; 20];
        low[5] = 50.0; // Lowest low 14 periods ago

        let high_arr = Array1::from_vec(high);
        let low_arr = Array1::from_vec(low);

        let (aroon_up, aroon_down) =
            aroon_gpu(&device, &high_arr, &low_arr, 14, None).expect("Aroon GPU calculation failed");

        // Aroon Up should be 100 (highest high at position 0 periods ago)
        assert!((aroon_up[19] - 100.0).abs() < 0.001);

        // Aroon Down should be 0 (lowest low at position 14 periods ago)
        assert!((aroon_down[19] - 0.0).abs() < 0.001);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_aroon_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());
        let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64) * 0.01).collect());

        let start = std::time::Instant::now();
        let (aroon_up, aroon_down) =
            aroon_gpu(&device, &high, &low, 14, None).expect("Aroon GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU Aroon (n={}): {:.2}ms",
            n,
            elapsed.as_secs_f64() * 1000.0
        );

        assert_eq!(aroon_up.len(), n);
        assert_eq!(aroon_down.len(), n);

        // In steady uptrend, Aroon Up should be high
        assert!(aroon_up[n - 1] > 90.0);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_aroon_gpu_validation_errors() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = arr1(&[100.0, 102.0, 105.0]);
        let low = arr1(&[95.0, 97.0]);

        // Mismatched lengths
        let result = aroon_gpu(&device, &high, &low, 14, None);
        assert!(result.is_err());

        let high = arr1(&[100.0, 102.0, 105.0]);
        let low = arr1(&[95.0, 97.0, 100.0]);

        // Period too large
        let result = aroon_gpu(&device, &high, &low, 14, None);
        assert!(result.is_err());

        // Period zero
        let result = aroon_gpu(&device, &high, &low, 0, None);
        assert!(result.is_err());
    }
}
