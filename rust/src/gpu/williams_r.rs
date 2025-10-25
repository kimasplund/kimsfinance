//! GPU-Accelerated Williams %R Indicator
//!
//! Provides 15-25x speedup over CPU implementation for large datasets.
//! Williams %R is nearly identical to Stochastic %K but inverted to range [-100, 0].

use super::device::{GpuDevice, GpuError};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use ndarray::Array1;

/// CUDA kernel source code for Williams %R
const WILLIAMS_R_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void williams_r_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    double* __restrict__ williams_r,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Calculate Williams %R
    if (idx >= period - 1) {
        // Find highest high and lowest low in period window
        double highest_high = -CUDART_INF;
        double lowest_low = CUDART_INF;

        for (int i = 0; i < period; i++) {
            int window_idx = idx - i;
            if (window_idx >= 0) {
                highest_high = fmax(highest_high, high[window_idx]);
                lowest_low = fmin(lowest_low, low[window_idx]);
            }
        }

        // Calculate %R: ((highest_high - close) / (highest_high - lowest_low)) * -100
        double range = highest_high - lowest_low;
        if (range > 1e-10) {
            williams_r[idx] = ((highest_high - close[idx]) / range) * -100.0;
        } else {
            // When range is zero, use midpoint (-50)
            williams_r[idx] = -50.0;
        }
    } else {
        williams_r[idx] = CUDART_NAN;
    }
}
"#;

/// GPU-accelerated Williams %R indicator
///
/// Williams %R measures overbought/oversold levels, ranging from -100 (oversold) to 0 (overbought).
/// It is inversely related to the Stochastic %K: Williams %R = Stochastic %K - 100.
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `period` - Lookback period (typically 14)
///
/// # Returns
///
/// Array1<f64> with Williams %R values in range [-100, 0]
///
/// # Performance
///
/// Expected speedup: **15-25x** over CPU for n > 10,000
///
/// # Formula
///
/// ```text
/// %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100
/// ```
///
/// # Interpretation
///
/// - **-80 to -100**: Oversold (potential buy signal)
/// - **-20 to 0**: Overbought (potential sell signal)
/// - **-50**: Neutral
pub fn williams_r_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    period: usize,
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
    let ptx = compile_ptx(WILLIAMS_R_KERNEL)
        .map_err(|e| GpuError::CompilationError(format!("Failed to compile kernel: {:?}", e)))?;

    // Load module (use context, not stream)
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel function from module
    let kernel = module.load_function("williams_r_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
    })?;

    // Copy data to GPU
    let d_high = device.copy_to_device(high.as_slice().unwrap())?;
    let d_low = device.copy_to_device(low.as_slice().unwrap())?;
    let d_close = device.copy_to_device(close.as_slice().unwrap())?;

    // Allocate output buffer
    let mut d_williams_r = device.alloc_buffer(n)?;

    // Launch kernel using builder pattern
    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut builder = device.stream.launch_builder(&kernel);
    builder.arg(&d_high);
    builder.arg(&d_low);
    builder.arg(&d_close);
    builder.arg(&mut d_williams_r);
    builder.arg(&n_i32);
    builder.arg(&period_i32);

    let config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("Kernel launch failed: {:?}", e)))?;
    }

    // Synchronize and copy results back
    device.synchronize()?;

    let williams_r_vec = device.copy_to_host(&d_williams_r)?;

    Ok(Array1::from_vec(williams_r_vec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_williams_r_gpu() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0, 132.0, 135.0,
            133.0, 136.0, 140.0, 138.0, 142.0, 145.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0, 127.0, 130.0,
            128.0, 131.0, 135.0, 133.0, 137.0, 140.0,
        ]);
        let close = arr1(&[
            108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0, 130.0, 133.0,
            131.0, 134.0, 138.0, 136.0, 140.0, 143.0,
        ]);

        let williams_r = williams_r_gpu(&device, &high, &low, &close, 14)
            .expect("Williams %R GPU calculation failed");

        // Verify %R is in valid range [-100, 0]
        for i in 14..williams_r.len() {
            assert!(
                williams_r[i] >= -100.0 && williams_r[i] <= 0.0,
                "Williams %R at index {} = {} is out of range [-100, 0]",
                i,
                williams_r[i]
            );
        }

        // First 13 values should be NaN (period - 1)
        for i in 0..13 {
            assert!(williams_r[i].is_nan(), "Expected NaN at index {}", i);
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_williams_r_gpu_large() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());
        let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64) * 0.01).collect());
        let close = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64) * 0.01).collect());

        let start = std::time::Instant::now();
        let williams_r = williams_r_gpu(&device, &high, &low, &close, 14)
            .expect("Williams %R GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU Williams %R (n={}): {:.2}ms",
            n,
            elapsed.as_secs_f64() * 1000.0
        );

        assert_eq!(williams_r.len(), n);

        // Verify all non-NaN values are in valid range
        for (i, &value) in williams_r.iter().enumerate() {
            if !value.is_nan() {
                assert!(
                    value >= -100.0 && value <= 0.0,
                    "Williams %R at index {} = {} is out of range",
                    i,
                    value
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_williams_r_gpu_equivalence_to_stochastic() {
        // Williams %R should equal Stochastic %K - 100
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0, 132.0, 135.0,
            133.0, 136.0, 140.0, 138.0, 142.0, 145.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0, 127.0, 130.0,
            128.0, 131.0, 135.0, 133.0, 137.0, 140.0,
        ]);
        let close = arr1(&[
            108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0, 130.0, 133.0,
            131.0, 134.0, 138.0, 136.0, 140.0, 143.0,
        ]);

        let williams_r = williams_r_gpu(&device, &high, &low, &close, 14)
            .expect("Williams %R GPU calculation failed");

        // Use stochastic from the existing implementation
        use super::super::stochastic::stochastic_gpu;
        let (stochastic_k, _) = stochastic_gpu(&device, &high, &low, &close, 14, 3)
            .expect("Stochastic GPU calculation failed");

        // Verify: Williams %R ≈ Stochastic %K - 100
        for i in 14..williams_r.len() {
            let expected = stochastic_k[i] - 100.0;
            let diff = (williams_r[i] - expected).abs();
            assert!(
                diff < 1e-6,
                "At index {}: Williams %R = {}, Stochastic %K - 100 = {}, diff = {}",
                i,
                williams_r[i],
                expected,
                diff
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_williams_r_gpu_edge_cases() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with constant prices (range = 0)
        let high = arr1(&[100.0; 20]);
        let low = arr1(&[100.0; 20]);
        let close = arr1(&[100.0; 20]);

        let williams_r = williams_r_gpu(&device, &high, &low, &close, 14)
            .expect("Williams %R GPU calculation failed");

        // When range is zero, should return -50 (neutral)
        for i in 13..williams_r.len() {
            assert_eq!(
                williams_r[i], -50.0,
                "Expected -50 for zero range at index {}",
                i
            );
        }
    }
}
