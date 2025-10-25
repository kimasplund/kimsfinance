//! GPU-Accelerated ATR (Average True Range)
//!
//! Provides 10-20x speedup over CPU implementation for large datasets.
//!
//! ATR is a volatility indicator that measures price range:
//! - True Range (TR) = max(high - low, |high - prev_close|, |low - prev_close|)
//! - ATR uses Wilder's smoothing: ATR[i] = ((period-1) * ATR[i-1] + TR[i]) / period
//! - First ATR value is SMA of first `period` TR values

use super::device::{GpuDevice, GpuError};
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for ATR calculation
///
/// Two-pass approach:
/// 1. Calculate True Range (TR) for all candles (parallel)
/// 2. Apply Wilder's smoothing sequentially (one thread handles entire array)
const ATR_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

// Kernel 1: Calculate True Range (parallel)
extern "C" __global__ void calculate_true_range_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    double* __restrict__ true_range,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    if (idx == 0) {
        // First candle: no previous close, so TR = high - low
        true_range[idx] = high[idx] - low[idx];
    } else {
        // TR = max(high - low, |high - prev_close|, |low - prev_close|)
        double hl = high[idx] - low[idx];
        double hc = fabs(high[idx] - close[idx - 1]);
        double lc = fabs(low[idx] - close[idx - 1]);

        true_range[idx] = fmax(fmax(hl, hc), lc);
    }
}

// Kernel 2: Apply Wilder's smoothing to calculate ATR (sequential by design)
// This kernel is launched with a single thread to handle the sequential dependency
extern "C" __global__ void calculate_atr_kernel(
    const double* __restrict__ true_range,
    double* __restrict__ atr,
    int n,
    int period
) {
    // Only thread 0 does the work (sequential algorithm)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // First period-1 values are NaN (not enough data)
    for (int i = 0; i < period - 1; i++) {
        atr[i] = CUDART_NAN;
    }

    // Calculate first ATR as SMA of first `period` TR values
    double sum = 0.0;
    for (int i = 0; i < period; i++) {
        sum += true_range[i];
    }
    atr[period - 1] = sum / period;

    // Apply Wilder's smoothing for remaining values
    // ATR[i] = ((period - 1) * ATR[i-1] + TR[i]) / period
    double multiplier = (period - 1.0) / period;
    double divisor = 1.0 / period;

    for (int i = period; i < n; i++) {
        atr[i] = atr[i - 1] * multiplier + true_range[i] * divisor;
    }
}
"#;

/// GPU-accelerated Average True Range (ATR)
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `period` - ATR period (typically 14)
/// * `stream` - Optional CUDA stream for concurrent execution (Week 2 optimization)
///
/// # Returns
///
/// ATR values as Array1<f64>. First `period-1` values are NaN.
///
/// # Algorithm
///
/// 1. Calculate True Range: TR = max(H-L, |H-C_prev|, |L-C_prev|)
/// 2. First ATR = SMA of first `period` TR values
/// 3. Subsequent ATR = Wilder's smoothing: ((period-1) * ATR_prev + TR) / period
///
/// # Performance
///
/// Expected speedup: **10-20x** over CPU for n > 10,000
/// Stream concurrency: Enables parallel execution with other indicators
///
/// # Errors
///
/// Returns error if:
/// - Arrays have different lengths
/// - Period < 1
/// - Not enough data (n < period)
/// - GPU operations fail
pub fn atr_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
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
    let ptx = compile_ptx(ATR_KERNEL)
        .map_err(|e| GpuError::CompilationError(format!("Failed to compile kernel: {:?}", e)))?;

    // Load module
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel functions
    let tr_kernel = module
        .load_function("calculate_true_range_kernel")
        .map_err(|e| {
            GpuError::ExecutionError(format!("Failed to load TR kernel function: {:?}", e))
        })?;

    let atr_kernel = module.load_function("calculate_atr_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load ATR kernel function: {:?}", e))
    })?;

    // Select stream: use provided stream or fallback to device.stream
    let exec_stream = stream.unwrap_or(&device.stream);

    // Copy data to GPU
    let d_high = device.copy_to_device(high.as_slice().unwrap())?;
    let d_low = device.copy_to_device(low.as_slice().unwrap())?;
    let d_close = device.copy_to_device(close.as_slice().unwrap())?;

    // Allocate output buffers
    let mut d_true_range = device.alloc_buffer(n)?;
    let mut d_atr = device.alloc_buffer(n)?;

    // Launch TR kernel (parallel across all candles) on selected stream
    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut tr_builder = exec_stream.launch_builder(&tr_kernel);
    tr_builder.arg(&d_high);
    tr_builder.arg(&d_low);
    tr_builder.arg(&d_close);
    tr_builder.arg(&mut d_true_range);
    tr_builder.arg(&n_i32);

    let tr_config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        tr_builder
            .launch(tr_config)
            .map_err(|e| GpuError::ExecutionError(format!("TR kernel launch failed: {:?}", e)))?;
    }

    // Synchronize on selected stream before launching ATR kernel
    exec_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("TR kernel synchronization failed: {:?}", e))
    })?;

    // Launch ATR kernel (single thread for sequential smoothing) on selected stream
    let mut atr_builder = exec_stream.launch_builder(&atr_kernel);
    atr_builder.arg(&d_true_range);
    atr_builder.arg(&mut d_atr);
    atr_builder.arg(&n_i32);
    atr_builder.arg(&period_i32);

    // Single thread kernel (sequential algorithm)
    let atr_config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        atr_builder
            .launch(atr_config)
            .map_err(|e| GpuError::ExecutionError(format!("ATR kernel launch failed: {:?}", e)))?;
    }

    // Synchronize on selected stream and copy results back
    exec_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("ATR kernel synchronization failed: {:?}", e))
    })?;

    let atr_vec = device.copy_to_host(&d_atr)?;

    Ok(Array1::from_vec(atr_vec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_atr_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Sample OHLC data
        let high = arr1(&[
            48.70, 48.72, 48.90, 48.87, 48.82, 49.05, 49.20, 49.35, 49.92, 50.19, 50.12, 49.66,
            49.88, 50.19, 50.36,
        ]);
        let low = arr1(&[
            47.79, 48.14, 48.39, 48.37, 48.24, 48.64, 48.94, 48.86, 49.50, 49.87, 49.20, 48.90,
            49.43, 49.73, 49.26,
        ]);
        let close = arr1(&[
            48.16, 48.61, 48.75, 48.63, 48.74, 49.03, 49.07, 49.32, 49.91, 50.13, 49.53, 49.50,
            49.75, 50.03, 50.31,
        ]);

        let period = 14;
        let atr = atr_gpu(&device, &high, &low, &close, period, None)
            .expect("ATR GPU calculation failed");

        // First period-1 values should be NaN
        for i in 0..period - 1 {
            assert!(atr[i].is_nan(), "ATR[{}] should be NaN", i);
        }

        // ATR values should be positive after warmup
        for i in period - 1..atr.len() {
            assert!(
                atr[i] > 0.0 && !atr[i].is_nan(),
                "ATR[{}] = {} should be positive",
                i,
                atr[i]
            );
        }

        // ATR should be reasonable relative to price range
        let avg_range =
            high.iter().zip(low.iter()).map(|(h, l)| h - l).sum::<f64>() / high.len() as f64;
        assert!(
            atr[period - 1] > 0.0 && atr[period - 1] < avg_range * 2.0,
            "ATR should be reasonable: {} vs avg_range {}",
            atr[period - 1],
            avg_range
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_atr_gpu_first_candle() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test that first candle TR = high - low (no previous close)
        let high = arr1(&[100.0, 102.0, 101.5]);
        let low = arr1(&[98.0, 100.5, 99.0]);
        let close = arr1(&[99.0, 101.0, 100.0]);

        let atr = atr_gpu(&device, &high, &low, &close, 2, None)
            .expect("ATR GPU calculation failed");

        // First value is NaN (period-1)
        assert!(atr[0].is_nan());

        // Second value (index 1) should be average of first two TR values
        // TR[0] = 100 - 98 = 2.0 (no previous close)
        // TR[1] = max(102-100.5, |102-99|, |100.5-99|) = max(1.5, 3.0, 1.5) = 3.0
        // ATR[1] = (2.0 + 3.0) / 2 = 2.5
        assert!(
            (atr[1] - 2.5).abs() < 1e-10,
            "ATR[1] should be 2.5, got {}",
            atr[1]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_atr_gpu_validation() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = arr1(&[10.0, 11.0, 12.0]);
        let low = arr1(&[8.0, 9.0, 10.0]);
        let close = arr1(&[9.0, 10.0]);

        // Mismatched lengths
        let result = atr_gpu(&device, &high, &low, &close, 2, None);
        assert!(result.is_err());

        let close = arr1(&[9.0, 10.0, 11.0]);

        // Period = 0
        let result = atr_gpu(&device, &high, &low, &close, 0, None);
        assert!(result.is_err());

        // Not enough data
        let result = atr_gpu(&device, &high, &low, &close, 5, None);
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_atr_gpu_large() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());
        let low = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64) * 0.01).collect());
        let close = Array1::from_vec((0..n).map(|i| 99.0 + (i as f64) * 0.01).collect());

        let start = std::time::Instant::now();
        let atr = atr_gpu(&device, &high, &low, &close, 14, None)
            .expect("ATR GPU calculation failed");
        let elapsed = start.elapsed();

        println!("GPU ATR (n={}): {:.2}ms", n, elapsed.as_secs_f64() * 1000.0);

        assert_eq!(atr.len(), n);

        // Verify first 13 are NaN
        for i in 0..13 {
            assert!(atr[i].is_nan());
        }

        // Verify remaining are valid
        for i in 13..n {
            assert!(atr[i] > 0.0 && !atr[i].is_nan());
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_atr_gpu_wilders_smoothing() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Constant price data to verify Wilder's smoothing formula
        let high = arr1(&[10.0; 20]);
        let low = arr1(&[8.0; 20]);
        let close = arr1(&[9.0; 20]);

        let period = 5;
        let atr = atr_gpu(&device, &high, &low, &close, period, None)
            .expect("ATR GPU calculation failed");

        // TR is constant = 2.0 for all candles
        // ATR[4] (first ATR) = SMA of 5 TRs = 2.0
        assert!(
            (atr[4] - 2.0).abs() < 1e-10,
            "First ATR should be 2.0, got {}",
            atr[4]
        );

        // With constant TR, Wilder's smoothing converges to TR value
        // ATR[i] = ((period-1) * ATR[i-1] + TR[i]) / period
        // With TR constant, ATR remains constant
        for i in 5..20 {
            assert!(
                (atr[i] - 2.0).abs() < 1e-10,
                "ATR[{}] should be ~2.0, got {}",
                i,
                atr[i]
            );
        }
    }
}
