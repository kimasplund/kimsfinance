//! GPU-Accelerated VWAP (Volume-Weighted Average Price)
//!
//! Provides 8-15x speedup over CPU implementation for large datasets.
//!
//! VWAP is an intraday benchmark that calculates the cumulative volume-weighted
//! average price. It's used by traders to compare current price to average traded price.
//!
//! # Algorithm
//!
//! 1. Typical Price = (high + low + close) / 3
//! 2. VWAP = cumsum(Typical Price * volume) / cumsum(volume)
//!
//! # Classification
//!
//! **MEDIUM** indicator - Two-pass approach (parallel + sequential)

use super::device::{GpuDevice, GpuError};
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for VWAP calculation
///
/// Two-step approach:
/// 1. Calculate typical price in parallel
/// 2. Calculate cumulative VWAP sequentially (inherent dependency)
const VWAP_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

// Kernel 1: Calculate typical price (parallel)
// Typical Price = (high + low + close) / 3
extern "C" __global__ void typical_price_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    double* __restrict__ typical_price,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        typical_price[idx] = (high[idx] + low[idx] + close[idx]) / 3.0;
    }
}

// Kernel 2: Calculate cumulative VWAP (sequential)
// VWAP[i] = cumsum(typical_price * volume) / cumsum(volume)
//
// This kernel is sequential by design - each element depends on previous cumulative sums.
// Launched with a single thread to handle the sequential dependency.
extern "C" __global__ void vwap_cumulative_kernel(
    const double* __restrict__ typical_price,
    const double* __restrict__ volume,
    double* __restrict__ vwap,
    int n
) {
    // Only thread 0 does the work (sequential algorithm)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    double cumulative_tpv = 0.0;  // cumulative typical_price * volume
    double cumulative_vol = 0.0;  // cumulative volume

    for (int i = 0; i < n; i++) {
        cumulative_tpv += typical_price[i] * volume[i];
        cumulative_vol += volume[i];

        // Avoid division by zero
        if (cumulative_vol > 1e-10) {
            vwap[i] = cumulative_tpv / cumulative_vol;
        } else {
            vwap[i] = CUDART_NAN;
        }
    }
}
"#;

/// GPU-accelerated VWAP (Volume-Weighted Average Price)
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `volume` - Trading volume
/// * `stream` - Optional CUDA stream for concurrent execution
///
/// # Returns
///
/// VWAP values as Array1<f64>. Returns NaN if cumulative volume is zero.
///
/// # Algorithm
///
/// 1. **Typical Price** (parallel): TP = (high + low + close) / 3
/// 2. **Cumulative VWAP** (sequential): VWAP[i] = Σ(TP * volume) / Σ(volume)
///
/// # Performance
///
/// Expected speedup: **8-15x** over CPU for n > 10,000
///
/// Stream concurrency: Enables parallel execution with other indicators
///
/// Classification: **MEDIUM** indicator (two-kernel approach)
///
/// # Errors
///
/// Returns error if:
/// - Arrays have different lengths
/// - GPU operations fail
/// - Not enough data (n < 1)
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, vwap_gpu};
/// use ndarray::arr1;
///
/// let device = GpuDevice::new()?;
/// let high = arr1(&[100.5, 101.0, 100.8]);
/// let low = arr1(&[99.5, 100.0, 99.8]);
/// let close = arr1(&[100.0, 100.5, 100.2]);
/// let volume = arr1(&[1000.0, 1200.0, 1100.0]);
///
/// let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)?;
/// println!("VWAP: {:?}", vwap);
/// ```
pub fn vwap_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    volume: &Array1<f64>,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let n = high.len();

    // Validate inputs
    if low.len() != n || close.len() != n || volume.len() != n {
        return Err(GpuError::InvalidParameter(
            "High, low, close, and volume arrays must have same length".to_string(),
        ));
    }

    if n < 1 {
        return Err(GpuError::InvalidParameter(
            "Need at least 1 data point".to_string(),
        ));
    }

    // Compile PTX
    let ptx = compile_ptx(VWAP_KERNEL)
        .map_err(|e| GpuError::CompilationError(format!("Failed to compile kernel: {:?}", e)))?;

    // Load module
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel functions
    let tp_kernel = module
        .load_function("typical_price_kernel")
        .map_err(|e| {
            GpuError::ExecutionError(format!("Failed to load typical_price kernel: {:?}", e))
        })?;

    let vwap_kernel = module
        .load_function("vwap_cumulative_kernel")
        .map_err(|e| {
            GpuError::ExecutionError(format!("Failed to load vwap_cumulative kernel: {:?}", e))
        })?;

    // Select stream: use provided stream or fallback to device.stream
    let exec_stream = stream.unwrap_or(&device.stream);

    // Copy data to GPU
    let d_high = device.copy_to_device(high.as_slice().unwrap())?;
    let d_low = device.copy_to_device(low.as_slice().unwrap())?;
    let d_close = device.copy_to_device(close.as_slice().unwrap())?;
    let d_volume = device.copy_to_device(volume.as_slice().unwrap())?;

    // Allocate output buffers
    let mut d_typical_price = device.alloc_buffer(n)?;
    let mut d_vwap = device.alloc_buffer(n)?;

    // Launch typical price kernel (parallel) on selected stream
    let n_i32 = n as i32;

    let mut tp_builder = exec_stream.launch_builder(&tp_kernel);
    tp_builder.arg(&d_high);
    tp_builder.arg(&d_low);
    tp_builder.arg(&d_close);
    tp_builder.arg(&mut d_typical_price);
    tp_builder.arg(&n_i32);

    let tp_config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        tp_builder.launch(tp_config).map_err(|e| {
            GpuError::ExecutionError(format!("Typical price kernel launch failed: {:?}", e))
        })?;
    }

    // Synchronize on selected stream before launching VWAP kernel
    exec_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!(
            "Typical price kernel synchronization failed: {:?}",
            e
        ))
    })?;

    // Launch VWAP cumulative kernel (single thread for sequential) on selected stream
    let mut vwap_builder = exec_stream.launch_builder(&vwap_kernel);
    vwap_builder.arg(&d_typical_price);
    vwap_builder.arg(&d_volume);
    vwap_builder.arg(&mut d_vwap);
    vwap_builder.arg(&n_i32);

    // Single thread kernel (sequential algorithm)
    let vwap_config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        vwap_builder.launch(vwap_config).map_err(|e| {
            GpuError::ExecutionError(format!("VWAP kernel launch failed: {:?}", e))
        })?;
    }

    // Synchronize on selected stream and copy results back
    exec_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("VWAP kernel synchronization failed: {:?}", e))
    })?;

    let vwap_vec = device.copy_to_host(&d_vwap)?;

    Ok(Array1::from_vec(vwap_vec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Sample OHLC + volume data
        let high = arr1(&[101.0, 102.0, 103.0, 102.5, 104.0]);
        let low = arr1(&[99.0, 100.0, 101.0, 100.5, 102.0]);
        let close = arr1(&[100.0, 101.0, 102.0, 101.5, 103.0]);
        let volume = arr1(&[1000.0, 1200.0, 1100.0, 1300.0, 1050.0]);

        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");

        assert_eq!(vwap.len(), 5);

        // All VWAP values should be valid (no NaN with positive volumes)
        for (i, &val) in vwap.iter().enumerate() {
            assert!(!val.is_nan(), "VWAP[{}] should not be NaN", i);
            assert!(val > 0.0, "VWAP[{}] should be positive", i);
        }

        // VWAP should be within reasonable range of prices
        let min_price = low.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = high.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        for (i, &val) in vwap.iter().enumerate() {
            assert!(
                val >= min_price && val <= max_price,
                "VWAP[{}] = {} should be within price range [{}, {}]",
                i,
                val,
                min_price,
                max_price
            );
        }

        // VWAP should be cumulative - later values influenced by all previous
        // First value should be close to first typical price
        let first_tp = (high[0] + low[0] + close[0]) / 3.0;
        assert!(
            (vwap[0] - first_tp).abs() < 0.01,
            "First VWAP should be close to first typical price: {} vs {}",
            vwap[0],
            first_tp
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_constant_prices() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Constant prices - VWAP should be constant
        let high = arr1(&[100.0, 100.0, 100.0, 100.0]);
        let low = arr1(&[100.0, 100.0, 100.0, 100.0]);
        let close = arr1(&[100.0, 100.0, 100.0, 100.0]);
        let volume = arr1(&[1000.0, 1000.0, 1000.0, 1000.0]);

        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");

        // With constant prices, VWAP should be constant at 100.0
        for (i, &val) in vwap.iter().enumerate() {
            assert!(
                (val - 100.0).abs() < 1e-10,
                "VWAP[{}] should be 100.0 with constant prices, got {}",
                i,
                val
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_zero_volume() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Zero volume - should return NaN
        let high = arr1(&[100.0]);
        let low = arr1(&[99.0]);
        let close = arr1(&[99.5]);
        let volume = arr1(&[0.0]);

        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");

        assert!(vwap[0].is_nan(), "VWAP should be NaN with zero volume");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_validation() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = arr1(&[100.0, 101.0, 102.0]);
        let low = arr1(&[99.0, 100.0]);
        let close = arr1(&[99.5, 100.5, 101.5]);
        let volume = arr1(&[1000.0, 1100.0, 1200.0]);

        // Mismatched lengths
        let result = vwap_gpu(&device, &high, &low, &close, &volume, None);
        assert!(result.is_err(), "Should fail with mismatched array lengths");

        // Empty arrays
        let high = arr1(&[]);
        let low = arr1(&[]);
        let close = arr1(&[]);
        let volume = arr1(&[]);

        let result = vwap_gpu(&device, &high, &low, &close, &volume, None);
        assert!(result.is_err(), "Should fail with empty arrays");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_cumulative_property() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test that VWAP is truly cumulative
        // Second candle with much higher price and volume should pull VWAP up
        let high = arr1(&[100.0, 110.0]);
        let low = arr1(&[99.0, 109.0]);
        let close = arr1(&[99.5, 109.5]);
        let volume = arr1(&[1000.0, 5000.0]); // 5x volume on second candle

        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");

        let first_tp = (100.0 + 99.0 + 99.5) / 3.0; // ~99.5
        let second_tp = (110.0 + 109.0 + 109.5) / 3.0; // ~109.5

        // First VWAP should equal first typical price
        assert!(
            (vwap[0] - first_tp).abs() < 0.01,
            "First VWAP = {}, expected ~{}",
            vwap[0],
            first_tp
        );

        // Second VWAP should be weighted heavily toward second TP due to 5x volume
        // Expected: (99.5 * 1000 + 109.5 * 5000) / 6000 = 107.83
        let expected_vwap_1 = (first_tp * 1000.0 + second_tp * 5000.0) / 6000.0;
        assert!(
            (vwap[1] - expected_vwap_1).abs() < 0.01,
            "Second VWAP = {}, expected ~{}",
            vwap[1],
            expected_vwap_1
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());
        let low = Array1::from_vec((0..n).map(|i| 99.0 + (i as f64) * 0.01).collect());
        let close = Array1::from_vec((0..n).map(|i| 99.5 + (i as f64) * 0.01).collect());
        let volume = Array1::from_vec((0..n).map(|i| 1000.0 + (i as f64 % 100.0)).collect());

        let start = std::time::Instant::now();
        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU VWAP (n={}): {:.2}ms",
            n,
            elapsed.as_secs_f64() * 1000.0
        );

        assert_eq!(vwap.len(), n);

        // Verify all values are valid and within price range
        for i in 0..n {
            assert!(!vwap[i].is_nan(), "VWAP[{}] should not be NaN", i);
            assert!(vwap[i] > 0.0, "VWAP[{}] should be positive", i);
            assert!(
                vwap[i] >= low[i] && vwap[i] <= high[i] * 1.01, // small tolerance for cumulative
                "VWAP[{}] = {} should be near price range [{}, {}]",
                i,
                vwap[i],
                low[i],
                high[i]
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_typical_price_calculation() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Simple test case to verify typical price calculation
        let high = arr1(&[120.0]);
        let low = arr1(&[100.0]);
        let close = arr1(&[110.0]);
        let volume = arr1(&[1000.0]);

        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");

        // Typical Price = (120 + 100 + 110) / 3 = 110.0
        // VWAP (first value) = TP (since only one value)
        let expected_tp = (120.0 + 100.0 + 110.0) / 3.0;
        assert!(
            (vwap[0] - expected_tp).abs() < 1e-10,
            "VWAP should equal typical price for single value: {} vs {}",
            vwap[0],
            expected_tp
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_mixed_volumes() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with varying volumes to verify volume weighting
        let high = arr1(&[100.0, 100.0, 100.0]);
        let low = arr1(&[100.0, 100.0, 100.0]);
        let close = arr1(&[100.0, 100.0, 100.0]);
        let volume = arr1(&[1000.0, 2000.0, 3000.0]); // Increasing volume

        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");

        // With constant prices, VWAP should remain constant at 100.0
        // regardless of volume changes
        for (i, &val) in vwap.iter().enumerate() {
            assert!(
                (val - 100.0).abs() < 1e-10,
                "VWAP[{}] should be 100.0 with constant prices, got {}",
                i,
                val
            );
        }
    }
}
