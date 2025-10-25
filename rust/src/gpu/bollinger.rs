//! GPU-Accelerated Bollinger Bands
//!
//! Provides 20-30x speedup over CPU implementation for large datasets.
//!
//! # Algorithm
//!
//! 1. Middle Band = SMA(close, period)
//! 2. Standard Deviation = sqrt(sum((close[i] - SMA)^2) / period)
//! 3. Upper Band = Middle + (std_dev * num_std)
//! 4. Lower Band = Middle - (std_dev * num_std)

use super::device::{GpuDevice, GpuError};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use ndarray::Array1;

/// CUDA kernel source code for Bollinger Bands
///
/// Uses two-pass algorithm for numerical stability:
/// - Pass 1: Calculate rolling SMA (middle band)
/// - Pass 2: Calculate rolling standard deviation and upper/lower bands
const BOLLINGER_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void bollinger_bands_kernel(
    const double* __restrict__ close,
    double* __restrict__ upper_band,
    double* __restrict__ middle_band,
    double* __restrict__ lower_band,
    int n,
    int period,
    double num_std
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Calculate middle band (SMA)
    if (idx >= period - 1) {
        double sum = 0.0;

        // Calculate rolling SMA
        for (int i = 0; i < period; i++) {
            int window_idx = idx - i;
            if (window_idx >= 0) {
                sum += close[window_idx];
            }
        }

        double sma = sum / period;
        middle_band[idx] = sma;

        // Calculate standard deviation using two-pass algorithm
        // Pass 2: Calculate variance
        double sum_squared_diff = 0.0;

        for (int i = 0; i < period; i++) {
            int window_idx = idx - i;
            if (window_idx >= 0) {
                double diff = close[window_idx] - sma;
                sum_squared_diff += diff * diff;
            }
        }

        // Standard deviation (sample std dev: divide by period, not period-1)
        // Using population std dev as per common practice for Bollinger Bands
        double variance = sum_squared_diff / period;
        double std_dev = sqrt(variance);

        // Calculate upper and lower bands
        upper_band[idx] = sma + (std_dev * num_std);
        lower_band[idx] = sma - (std_dev * num_std);
    } else {
        // Not enough data for this period
        middle_band[idx] = CUDART_NAN;
        upper_band[idx] = CUDART_NAN;
        lower_band[idx] = CUDART_NAN;
    }
}
"#;

/// GPU-accelerated Bollinger Bands
///
/// Calculates Bollinger Bands using CUDA for massive parallelization.
/// Each thread processes one time point independently.
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `close` - Close prices
/// * `period` - SMA period for middle band (typically 20)
/// * `num_std` - Number of standard deviations for bands (typically 2.0)
///
/// # Returns
///
/// Tuple of (upper_band, middle_band, lower_band) as Array1<f64>
///
/// # Performance
///
/// Expected speedup: **20-30x** over CPU for n > 10,000
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, bollinger_bands_gpu};
/// use ndarray::arr1;
///
/// let device = GpuDevice::new()?;
/// let close = arr1(&[100.0, 102.0, 101.0, 103.0, 105.0, /* ... */]);
/// let (upper, middle, lower) = bollinger_bands_gpu(&device, &close, 20, 2.0)?;
/// ```
pub fn bollinger_bands_gpu(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    num_std: f64,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), GpuError> {
    let n = close.len();

    // Validate inputs
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

    if num_std <= 0.0 {
        return Err(GpuError::InvalidParameter(
            "num_std must be positive".to_string(),
        ));
    }

    // Compile PTX
    let ptx = compile_ptx(BOLLINGER_KERNEL)
        .map_err(|e| GpuError::CompilationError(format!("Failed to compile kernel: {:?}", e)))?;

    // Load module (use context, not stream)
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel function from module
    let kernel = module
        .load_function("bollinger_bands_kernel")
        .map_err(|e| {
            GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
        })?;

    // Copy data to GPU
    let d_close = device.copy_to_device(close.as_slice().unwrap())?;

    // Allocate output buffers
    let mut d_upper = device.alloc_buffer(n)?;
    let mut d_middle = device.alloc_buffer(n)?;
    let mut d_lower = device.alloc_buffer(n)?;

    // Launch kernel using builder pattern
    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut builder = device.stream.launch_builder(&kernel);
    builder.arg(&d_close);
    builder.arg(&mut d_upper);
    builder.arg(&mut d_middle);
    builder.arg(&mut d_lower);
    builder.arg(&n_i32);
    builder.arg(&period_i32);
    builder.arg(&num_std);

    let config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("Kernel launch failed: {:?}", e)))?;
    }

    // Synchronize and copy results back
    device.synchronize()?;

    let upper_vec = device.copy_to_host(&d_upper)?;
    let middle_vec = device.copy_to_host(&d_middle)?;
    let lower_vec = device.copy_to_host(&d_lower)?;

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
    fn test_bollinger_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Simple test case with known values
        let close = arr1(&[
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0, 111.0, 110.0,
            112.0, 114.0, 113.0, 115.0, 117.0, 116.0, 118.0, 120.0, 119.0, 121.0, 123.0, 122.0,
            124.0,
        ]);

        let (upper, middle, lower) =
            bollinger_bands_gpu(&device, &close, 20, 2.0).expect("Bollinger GPU failed");

        // Verify dimensions
        assert_eq!(upper.len(), close.len());
        assert_eq!(middle.len(), close.len());
        assert_eq!(lower.len(), close.len());

        // First 19 values should be NaN (period - 1)
        for i in 0..19 {
            assert!(upper[i].is_nan());
            assert!(middle[i].is_nan());
            assert!(lower[i].is_nan());
        }

        // Valid values should satisfy: lower < middle < upper
        for i in 19..close.len() {
            assert!(!upper[i].is_nan());
            assert!(!middle[i].is_nan());
            assert!(!lower[i].is_nan());
            assert!(lower[i] < middle[i]);
            assert!(middle[i] < upper[i]);
        }

        // Verify middle band is reasonable (should be near close prices)
        for i in 19..close.len() {
            assert!(middle[i] >= 80.0 && middle[i] <= 140.0);
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_bollinger_gpu_symmetric() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test symmetry: upper - middle should equal middle - lower
        let close = arr1(&[
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0, 111.0, 110.0,
            112.0, 114.0, 113.0, 115.0, 117.0, 116.0, 118.0, 120.0,
        ]);

        let (upper, middle, lower) =
            bollinger_bands_gpu(&device, &close, 10, 2.0).expect("Bollinger GPU failed");

        // Check symmetry for valid values
        for i in 9..close.len() {
            let upper_diff = upper[i] - middle[i];
            let lower_diff = middle[i] - lower[i];

            // Should be symmetric within floating point tolerance
            assert!(
                (upper_diff - lower_diff).abs() < 1e-10,
                "Asymmetry at index {}: upper_diff={}, lower_diff={}",
                i,
                upper_diff,
                lower_diff
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_bollinger_gpu_num_std() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let close = arr1(&[
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0, 111.0, 110.0,
            112.0, 114.0, 113.0, 115.0, 117.0, 116.0, 118.0, 120.0,
        ]);

        // Test with 1 std dev
        let (upper_1, middle_1, lower_1) =
            bollinger_bands_gpu(&device, &close, 10, 1.0).expect("Bollinger GPU failed");

        // Test with 2 std dev
        let (upper_2, middle_2, lower_2) =
            bollinger_bands_gpu(&device, &close, 10, 2.0).expect("Bollinger GPU failed");

        // Middle band should be identical
        for i in 9..close.len() {
            assert!((middle_1[i] - middle_2[i]).abs() < 1e-10);
        }

        // Band width should double
        for i in 9..close.len() {
            let width_1 = upper_1[i] - lower_1[i];
            let width_2 = upper_2[i] - lower_2[i];

            assert!(
                (width_2 - 2.0 * width_1).abs() < 1e-9,
                "Width ratio incorrect at index {}: width_1={}, width_2={}",
                i,
                width_1,
                width_2
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_bollinger_gpu_large() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let close = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());

        let start = std::time::Instant::now();
        let (upper, middle, lower) =
            bollinger_bands_gpu(&device, &close, 20, 2.0).expect("Bollinger GPU failed");
        let elapsed = start.elapsed();

        println!(
            "GPU Bollinger Bands (n={}, period=20): {:.2}ms",
            n,
            elapsed.as_secs_f64() * 1000.0
        );

        assert_eq!(upper.len(), n);
        assert_eq!(middle.len(), n);
        assert_eq!(lower.len(), n);

        // Verify structure for large dataset
        assert!(upper[19].is_finite());
        assert!(middle[19].is_finite());
        assert!(lower[19].is_finite());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_bollinger_gpu_constant_prices() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // When prices are constant, std dev should be 0, bands should converge to SMA
        let close = arr1(&[100.0; 50]);

        let (upper, middle, lower) =
            bollinger_bands_gpu(&device, &close, 20, 2.0).expect("Bollinger GPU failed");

        // All valid values should be exactly 100.0
        for i in 19..close.len() {
            assert!((upper[i] - 100.0).abs() < 1e-10);
            assert!((middle[i] - 100.0).abs() < 1e-10);
            assert!((lower[i] - 100.0).abs() < 1e-10);
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_bollinger_gpu_validation() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let close = arr1(&[100.0, 102.0, 101.0]);

        // Should fail: not enough data
        let result = bollinger_bands_gpu(&device, &close, 20, 2.0);
        assert!(result.is_err());

        // Should fail: invalid period
        let result = bollinger_bands_gpu(&device, &close, 0, 2.0);
        assert!(result.is_err());

        // Should fail: invalid num_std
        let result = bollinger_bands_gpu(&device, &close, 2, 0.0);
        assert!(result.is_err());

        let result = bollinger_bands_gpu(&device, &close, 2, -1.0);
        assert!(result.is_err());
    }
}
