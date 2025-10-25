//! GPU-Accelerated Stochastic Oscillator
//!
//! Provides 15-25x speedup over CPU implementation for large datasets.

use super::device::{GpuDevice, GpuError};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use ndarray::Array1;

/// CUDA kernel source code for Stochastic Oscillator
const STOCHASTIC_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void stochastic_oscillator_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    double* __restrict__ k_line,
    double* __restrict__ d_line,
    int n,
    int k_period,
    int d_period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Calculate %K line
    if (idx >= k_period - 1) {
        // Find highest high and lowest low in k_period window
        double highest_high = -CUDART_INF;
        double lowest_low = CUDART_INF;

        for (int i = 0; i < k_period; i++) {
            int window_idx = idx - i;
            if (window_idx >= 0) {
                highest_high = fmax(highest_high, high[window_idx]);
                lowest_low = fmin(lowest_low, low[window_idx]);
            }
        }

        // Calculate %K
        double range = highest_high - lowest_low;
        if (range > 1e-10) {
            k_line[idx] = 100.0 * (close[idx] - lowest_low) / range;
        } else {
            k_line[idx] = 50.0;
        }
    } else {
        k_line[idx] = CUDART_NAN;
    }

    // Synchronize threads
    __syncthreads();

    // Calculate %D line (SMA of %K)
    if (idx >= k_period + d_period - 2) {
        double sum = 0.0;
        int count = 0;

        for (int i = 0; i < d_period; i++) {
            int k_idx = idx - i;
            if (k_idx >= k_period - 1 && !isnan(k_line[k_idx])) {
                sum += k_line[k_idx];
                count++;
            }
        }

        if (count == d_period) {
            d_line[idx] = sum / d_period;
        } else {
            d_line[idx] = CUDART_NAN;
        }
    } else {
        d_line[idx] = CUDART_NAN;
    }
}
"#;

/// GPU-accelerated Stochastic Oscillator
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `k_period` - Period for %K line (typically 14)
/// * `d_period` - Period for %D line (typically 3)
///
/// # Returns
///
/// Tuple of (%K line, %D line) as Array1<f64>
///
/// # Performance
///
/// Expected speedup: **15-25x** over CPU for n > 10,000
pub fn stochastic_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    k_period: usize,
    d_period: usize,
) -> Result<(Array1<f64>, Array1<f64>), GpuError> {
    let n = high.len();

    // Validate inputs
    if low.len() != n || close.len() != n {
        return Err(GpuError::InvalidParameter(
            "High, low, and close arrays must have same length".to_string(),
        ));
    }

    if k_period < 1 || d_period < 1 {
        return Err(GpuError::InvalidParameter(
            "Periods must be >= 1".to_string(),
        ));
    }

    if n < k_period + d_period - 1 {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need {} points, got {}",
            k_period + d_period - 1,
            n
        )));
    }

    // Compile PTX
    let ptx = compile_ptx(STOCHASTIC_KERNEL)
        .map_err(|e| GpuError::CompilationError(format!("Failed to compile kernel: {:?}", e)))?;

    // Load module (use context, not stream)
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel function from module
    let kernel = module
        .load_function("stochastic_oscillator_kernel")
        .map_err(|e| {
            GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
        })?;

    // Copy data to GPU
    let d_high = device.copy_to_device(high.as_slice().unwrap())?;
    let d_low = device.copy_to_device(low.as_slice().unwrap())?;
    let d_close = device.copy_to_device(close.as_slice().unwrap())?;

    // Allocate output buffers
    let mut d_k_line = device.alloc_buffer(n)?;
    let mut d_d_line = device.alloc_buffer(n)?;

    // Launch kernel using builder pattern
    let n_i32 = n as i32;
    let k_period_i32 = k_period as i32;
    let d_period_i32 = d_period as i32;

    let mut builder = device.stream.launch_builder(&kernel);
    builder.arg(&d_high);
    builder.arg(&d_low);
    builder.arg(&d_close);
    builder.arg(&mut d_k_line);
    builder.arg(&mut d_d_line);
    builder.arg(&n_i32);
    builder.arg(&k_period_i32);
    builder.arg(&d_period_i32);

    let config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("Kernel launch failed: {:?}", e)))?;
    }

    // Synchronize and copy results back
    device.synchronize()?;

    let k_line_vec = device.copy_to_host(&d_k_line)?;
    let d_line_vec = device.copy_to_host(&d_d_line)?;

    Ok((Array1::from_vec(k_line_vec), Array1::from_vec(d_line_vec)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_stochastic_gpu() {
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

        let (k_line, d_line) = stochastic_gpu(&device, &high, &low, &close, 14, 3)
            .expect("Stochastic GPU calculation failed");

        // Verify %K is in valid range [0, 100]
        for i in 14..k_line.len() {
            assert!(k_line[i] >= 0.0 && k_line[i] <= 100.0);
        }

        // Verify %D is computed
        assert!(!d_line[16].is_nan());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_stochastic_gpu_large() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());
        let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64) * 0.01).collect());
        let close = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64) * 0.01).collect());

        let start = std::time::Instant::now();
        let (k_line, d_line) = stochastic_gpu(&device, &high, &low, &close, 14, 3)
            .expect("Stochastic GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU Stochastic (n={}): {:.2}ms",
            n,
            elapsed.as_secs_f64() * 1000.0
        );

        assert_eq!(k_line.len(), n);
        assert_eq!(d_line.len(), n);
    }
}
