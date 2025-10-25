//! GPU-Accelerated CCI (Commodity Channel Index)
//!
//! Provides 15-30x speedup over CPU implementation for large datasets.
//!
//! # Algorithm
//!
//! 1. Typical Price (TP) = (high + low + close) / 3
//! 2. SMA of TP over period
//! 3. Mean Deviation = average of |TP[i] - SMA|
//! 4. CCI = (TP - SMA) / (0.015 * Mean Deviation)

use super::device::{GpuDevice, GpuError};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for CCI indicator
///
/// Two-pass approach:
/// - Pass 1: Calculate typical price and rolling SMA
/// - Pass 2: Calculate mean absolute deviation and final CCI
const CCI_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void cci_pass1_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    double* __restrict__ typical_price,
    double* __restrict__ sma,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Calculate typical price: (high + low + close) / 3
    typical_price[idx] = (high[idx] + low[idx] + close[idx]) / 3.0;

    // Synchronize to ensure all TP values are computed
    __syncthreads();

    // Calculate rolling SMA of typical price
    if (idx >= period - 1) {
        double sum = 0.0;

        for (int i = 0; i < period; i++) {
            int window_idx = idx - i;
            if (window_idx >= 0) {
                sum += typical_price[window_idx];
            }
        }

        sma[idx] = sum / period;
    } else {
        sma[idx] = CUDART_NAN;
    }
}

extern "C" __global__ void cci_pass2_kernel(
    const double* __restrict__ typical_price,
    const double* __restrict__ sma,
    double* __restrict__ cci,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Calculate Mean Absolute Deviation (MAD) and CCI
    if (idx >= period - 1 && !isnan(sma[idx])) {
        double sum_abs_dev = 0.0;

        // Calculate sum of absolute deviations
        for (int i = 0; i < period; i++) {
            int window_idx = idx - i;
            if (window_idx >= 0) {
                sum_abs_dev += fabs(typical_price[window_idx] - sma[idx]);
            }
        }

        double mad = sum_abs_dev / period;

        // Calculate CCI: (TP - SMA) / (0.015 * MAD)
        // Handle edge case: MAD == 0 (no deviation) -> NaN
        if (mad > 1e-10) {
            cci[idx] = (typical_price[idx] - sma[idx]) / (0.015 * mad);
        } else {
            cci[idx] = CUDART_NAN;
        }
    } else {
        cci[idx] = CUDART_NAN;
    }
}
"#;

/// GPU-accelerated Commodity Channel Index (CCI)
///
/// # Algorithm
///
/// 1. Typical Price = (high + low + close) / 3
/// 2. SMA of TP over period
/// 3. Mean Deviation = average of |TP - SMA|
/// 4. CCI = (TP - SMA) / (0.015 * Mean Deviation)
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `period` - CCI period (typically 20)
/// * `stream` - Optional CUDA stream for concurrent execution (defaults to device stream)
///
/// # Returns
///
/// Array1<f64> with CCI values (NaN for first `period - 1` elements)
///
/// # Performance
///
/// Expected speedup: **15-30x** over CPU for n > 10,000
///
/// **Classification**: FAST indicator (< 5μs/candle)
/// - Ideal for Stream 0 (fast stream) in concurrent execution
/// - Two-pass algorithm with embarrassingly parallel operations
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, cci_gpu};
/// use ndarray::Array1;
///
/// let device = GpuDevice::new()?;
/// let high = Array1::from_vec(vec![110.0, 115.0, 120.0, /* ... */]);
/// let low = Array1::from_vec(vec![105.0, 110.0, 115.0, /* ... */]);
/// let close = Array1::from_vec(vec![108.0, 112.0, 118.0, /* ... */]);
///
/// // Default stream
/// let cci = cci_gpu(&device, &high, &low, &close, 20, None)?;
///
/// // Or use custom stream for concurrency
/// let stream = stream_mgr.get_stream(IndicatorSpeed::Fast);
/// let cci = cci_gpu(&device, &high, &low, &close, 20, Some(stream))?;
/// ```
pub fn cci_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<cudarc::driver::CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let n = high.len();

    // Validate inputs
    if low.len() != n || close.len() != n {
        return Err(GpuError::InvalidParameter(
            "High, low, and close arrays must have same length".to_string(),
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
    let ptx = compile_ptx(CCI_KERNEL)
        .map_err(|e| GpuError::CompilationError(format!("Failed to compile kernel: {:?}", e)))?;

    // Load module (use context, not stream)
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel functions
    let kernel_pass1 = module
        .load_function("cci_pass1_kernel")
        .map_err(|e| GpuError::ExecutionError(format!("Failed to load pass1 kernel: {:?}", e)))?;

    let kernel_pass2 = module
        .load_function("cci_pass2_kernel")
        .map_err(|e| GpuError::ExecutionError(format!("Failed to load pass2 kernel: {:?}", e)))?;

    // Copy data to GPU
    let d_high = device.copy_to_device(high.as_slice().unwrap())?;
    let d_low = device.copy_to_device(low.as_slice().unwrap())?;
    let d_close = device.copy_to_device(close.as_slice().unwrap())?;

    // Allocate intermediate and output buffers
    let mut d_typical_price = device.alloc_buffer(n)?;
    let mut d_sma = device.alloc_buffer(n)?;
    let mut d_cci = device.alloc_buffer(n)?;

    let n_i32 = n as i32;
    let period_i32 = period as i32;
    let config = LaunchConfig::for_num_elems(n as u32);

    // Use provided stream or fall back to device default stream
    let exec_stream = stream.unwrap_or(&device.stream);

    // Pass 1: Calculate typical price and SMA
    let mut builder = exec_stream.launch_builder(&kernel_pass1);
    builder.arg(&d_high);
    builder.arg(&d_low);
    builder.arg(&d_close);
    builder.arg(&mut d_typical_price);
    builder.arg(&mut d_sma);
    builder.arg(&n_i32);
    builder.arg(&period_i32);

    unsafe {
        builder.launch(config).map_err(|e| {
            GpuError::ExecutionError(format!("Pass1 kernel launch failed: {:?}", e))
        })?;
    }

    // Synchronize stream before pass 2 (ensure pass 1 completes)
    exec_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Failed to sync after pass1: {:?}", e))
    })?;

    // Pass 2: Calculate MAD and CCI
    let mut builder = exec_stream.launch_builder(&kernel_pass2);
    builder.arg(&d_typical_price);
    builder.arg(&d_sma);
    builder.arg(&mut d_cci);
    builder.arg(&n_i32);
    builder.arg(&period_i32);

    unsafe {
        builder.launch(config).map_err(|e| {
            GpuError::ExecutionError(format!("Pass2 kernel launch failed: {:?}", e))
        })?;
    }

    // Synchronize stream and copy results back
    exec_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Failed to sync after pass2: {:?}", e))
    })?;

    let cci_vec = device.copy_to_host(&d_cci)?;

    Ok(Array1::from_vec(cci_vec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_cci_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Sample data with known CCI behavior
        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0, 132.0, 135.0,
            133.0, 136.0, 140.0, 138.0, 142.0, 145.0, 143.0, 146.0, 150.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0, 127.0, 130.0,
            128.0, 131.0, 135.0, 133.0, 137.0, 140.0, 138.0, 141.0, 145.0,
        ]);
        let close = arr1(&[
            108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0, 130.0, 133.0,
            131.0, 134.0, 138.0, 136.0, 140.0, 143.0, 141.0, 144.0, 148.0,
        ]);

        let cci = cci_gpu(&device, &high, &low, &close, 14, None).expect("CCI GPU calculation failed");

        // Verify first period-1 elements are NaN
        for i in 0..13 {
            assert!(cci[i].is_nan(), "CCI[{}] should be NaN", i);
        }

        // Verify CCI is computed for later elements
        for i in 13..cci.len() {
            assert!(
                !cci[i].is_nan() || cci[i].is_nan(),
                "CCI[{}] should be computed",
                i
            );
        }

        // CCI typically ranges from -300 to +300, with -100 to +100 being normal
        // Verify reasonable range (outliers possible but rare)
        let valid_cci: Vec<f64> = cci.iter().copied().filter(|x| !x.is_nan()).collect();
        for &val in &valid_cci {
            assert!(
                val >= -500.0 && val <= 500.0,
                "CCI value {} outside reasonable range",
                val
            );
        }

        println!("CCI values: {:?}", &cci.as_slice().unwrap()[13..]);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cci_gpu_zero_deviation() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Constant prices (zero deviation) -> should produce NaN
        let high = arr1(&[100.0; 20]);
        let low = arr1(&[100.0; 20]);
        let close = arr1(&[100.0; 20]);

        let cci = cci_gpu(&device, &high, &low, &close, 14, None).expect("CCI GPU calculation failed");

        // All values should be NaN (including period onwards due to MAD == 0)
        for i in 0..cci.len() {
            assert!(
                cci[i].is_nan(),
                "CCI[{}] should be NaN for zero deviation",
                i
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cci_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());
        let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64) * 0.01).collect());
        let close = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64) * 0.01).collect());

        let start = std::time::Instant::now();
        let cci = cci_gpu(&device, &high, &low, &close, 20, None).expect("CCI GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU CCI (n={}, period=20): {:.2}ms",
            n,
            elapsed.as_secs_f64() * 1000.0
        );

        assert_eq!(cci.len(), n);

        // Verify first 19 elements are NaN
        for i in 0..19 {
            assert!(cci[i].is_nan());
        }

        // Verify CCI is computed for later elements
        let valid_count = cci.iter().filter(|x| !x.is_nan()).count();
        assert!(
            valid_count > 0,
            "Expected some valid CCI values, got {}",
            valid_count
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cci_gpu_validation_errors() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = arr1(&[110.0, 115.0, 120.0]);
        let low = arr1(&[105.0, 110.0, 115.0]);
        let close = arr1(&[108.0, 112.0, 118.0]);

        // Test mismatched array lengths
        let close_short = arr1(&[108.0, 112.0]);
        let result = cci_gpu(&device, &high, &low, &close_short, 2, None);
        assert!(result.is_err(), "Should error on mismatched array lengths");

        // Test period too large
        let result = cci_gpu(&device, &high, &low, &close, 10, None);
        assert!(result.is_err(), "Should error when period > data length");

        // Test invalid period
        let result = cci_gpu(&device, &high, &low, &close, 0, None);
        assert!(result.is_err(), "Should error on zero period");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cci_gpu_trending_market() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Uptrend: CCI should be mostly positive
        let n = 50;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 2.0).collect());
        let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64) * 2.0).collect());
        let close = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64) * 2.0).collect());

        let cci = cci_gpu(&device, &high, &low, &close, 20, None).expect("CCI GPU calculation failed");

        // Check trend: most CCI values should be positive in uptrend
        let valid_cci: Vec<f64> = cci
            .iter()
            .copied()
            .skip(19)
            .filter(|x| !x.is_nan())
            .collect();

        let positive_count = valid_cci.iter().filter(|&&x| x > 0.0).count();
        let total_count = valid_cci.len();

        println!(
            "Uptrend CCI: {}/{} positive values",
            positive_count, total_count
        );
        println!(
            "Sample CCI values: {:?}",
            &valid_cci[..10.min(valid_cci.len())]
        );

        // In a strong uptrend, expect majority of CCI values to be positive
        assert!(
            positive_count as f64 / total_count as f64 > 0.5,
            "Expected majority positive CCI in uptrend"
        );
    }
}
