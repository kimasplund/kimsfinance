//! GPU-Accelerated OBV (On-Balance Volume)
//!
//! Public OBV GPU entry point. Delegates to the parallel multi-level prefix-sum
//! implementation in [`super::obv_optimized`].
//!
//! OBV is a cumulative momentum indicator that relates volume to price changes.
//!
//! # History
//!
//! The original implementation in this file ran the cumulative sum as a
//! single-thread O(n) FP64 GPU loop (measured 4.70ms for 100K candles vs ~50us
//! on CPU) and inserted a host `synchronize()` between two same-stream kernel
//! launches (which are already ordered). Both were removed: `obv_gpu` now
//! delegates to [`obv_gpu_optimized`], whose multi-level scan handles any
//! dataset size. The volume-delta kernel source is defined once, in
//! `super::obv_optimized::OBV_DELTAS_KERNEL_SRC`.

use super::device::{GpuDevice, GpuError};
use super::obv_optimized::obv_gpu_optimized;
use cudarc::driver::CudaStream;
use ndarray::Array1;
use std::sync::Arc;

/// GPU-accelerated OBV (On-Balance Volume)
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `close` - Closing prices
/// * `volume` - Trading volumes
/// * `stream` - Optional CUDA stream for concurrent execution (None uses device default)
///
/// # Returns
///
/// Array1<f64> with cumulative OBV values
///
/// # Performance
///
/// Delegates to [`obv_gpu_optimized`] (parallel multi-level prefix sum):
/// expected **40-50x** over CPU for n > 10,000, with no dataset-size cap
/// (the previous parallel scan errored above 65,536 elements).
///
/// # Stream Concurrency
///
/// When a stream is provided, kernel launches execute on that stream, enabling
/// concurrent execution with other operations on different streams. This is used
/// in the batch pipeline for 4-6x speedup across Fast/Medium/Slow indicator groups.
///
/// Classification: **FAST** indicator (fully parallel deltas + scan kernels)
///
/// # Algorithm
///
/// 1. Calculate volume deltas:
///    - If close[i] > close[i-1]: delta = +volume[i]
///    - If close[i] < close[i-1]: delta = -volume[i]
///    - If close[i] == close[i-1]: delta = 0
/// 2. Parallel prefix sum of deltas to get OBV (Hillis-Steele block scans with
///    recursive inter-block propagation)
///
/// # Errors
///
/// Returns error if:
/// - Arrays have different lengths
/// - Arrays are empty
/// - GPU operations fail
pub fn obv_gpu(
    device: &GpuDevice,
    close: &Array1<f64>,
    volume: &Array1<f64>,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    obv_gpu_optimized(device, close, volume, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test data with clear price/volume relationship
        let close = arr1(&[100.0, 102.0, 101.0, 103.0, 102.0, 105.0]);
        let volume = arr1(&[1000.0, 1500.0, 1200.0, 1800.0, 1100.0, 2000.0]);

        let result = obv_gpu(&device, &close, &volume, None).expect("OBV GPU calculation failed");

        // Verify OBV calculation:
        // idx=0: OBV = 0 (starting point)
        // idx=1: close up (102 > 100), OBV = 0 + 1500 = 1500
        // idx=2: close down (101 < 102), OBV = 1500 - 1200 = 300
        // idx=3: close up (103 > 101), OBV = 300 + 1800 = 2100
        // idx=4: close down (102 < 103), OBV = 2100 - 1100 = 1000
        // idx=5: close up (105 > 102), OBV = 1000 + 2000 = 3000
        assert_eq!(result.len(), 6);
        assert!((result[0] - 0.0).abs() < 1e-6, "Expected OBV[0] = 0");
        assert!(
            (result[1] - 1500.0).abs() < 1e-6,
            "Expected OBV[1] = 1500, got {}",
            result[1]
        );
        assert!(
            (result[2] - 300.0).abs() < 1e-6,
            "Expected OBV[2] = 300, got {}",
            result[2]
        );
        assert!(
            (result[3] - 2100.0).abs() < 1e-6,
            "Expected OBV[3] = 2100, got {}",
            result[3]
        );
        assert!(
            (result[4] - 1000.0).abs() < 1e-6,
            "Expected OBV[4] = 1000, got {}",
            result[4]
        );
        assert!(
            (result[5] - 3000.0).abs() < 1e-6,
            "Expected OBV[5] = 3000, got {}",
            result[5]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_constant_price() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Constant prices - OBV should remain at 0
        let close = arr1(&[100.0, 100.0, 100.0, 100.0, 100.0]);
        let volume = arr1(&[1000.0, 1500.0, 1200.0, 1800.0, 1100.0]);

        let result = obv_gpu(&device, &close, &volume, None).expect("OBV GPU calculation failed");

        // All OBV values should be 0 (no price change)
        for i in 0..result.len() {
            assert!(
                result[i].abs() < 1e-6,
                "Expected OBV[{}] = 0, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_monotonic_increase() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Monotonically increasing prices
        let close = arr1(&[100.0, 101.0, 102.0, 103.0, 104.0]);
        let volume = arr1(&[1000.0, 1000.0, 1000.0, 1000.0, 1000.0]);

        let result = obv_gpu(&device, &close, &volume, None).expect("OBV GPU calculation failed");

        // OBV should accumulate positively
        // OBV = [0, 1000, 2000, 3000, 4000]
        for i in 0..result.len() {
            let expected = (i as f64) * 1000.0;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "Expected OBV[{}] = {}, got {}",
                i,
                expected,
                result[i]
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_monotonic_decrease() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Monotonically decreasing prices
        let close = arr1(&[100.0, 99.0, 98.0, 97.0, 96.0]);
        let volume = arr1(&[1000.0, 1000.0, 1000.0, 1000.0, 1000.0]);

        let result = obv_gpu(&device, &close, &volume, None).expect("OBV GPU calculation failed");

        // OBV should accumulate negatively
        // OBV = [0, -1000, -2000, -3000, -4000]
        for i in 0..result.len() {
            let expected = -(i as f64) * 1000.0;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "Expected OBV[{}] = {}, got {}",
                i,
                expected,
                result[i]
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Generate large dataset with sine wave pattern.
        // 100K elements requires the multi-level scan (391 blocks -> 2 blocks -> 1).
        let n = 100_000;
        let close: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * 0.01;
                100.0 + 10.0 * (x * 0.1).sin()
            })
            .collect();
        let volume: Vec<f64> = (0..n).map(|i| 1000.0 + (i % 100) as f64 * 10.0).collect();

        let close = Array1::from_vec(close);
        let volume = Array1::from_vec(volume);

        let start = std::time::Instant::now();
        let result = obv_gpu(&device, &close, &volume, None).expect("OBV GPU calculation failed");
        let elapsed = start.elapsed();

        println!("GPU OBV (n={}): {:.2}ms", n, elapsed.as_secs_f64() * 1000.0);

        // Verify output size
        assert_eq!(result.len(), n);

        // Verify first element is 0
        assert!(result[0].abs() < 1e-6, "Expected OBV[0] = 0");

        // Verify OBV is cumulative (no NaN values)
        for i in 0..n {
            assert!(
                !result[i].is_nan(),
                "OBV should not contain NaN at index {}",
                i
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_invalid_inputs() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Mismatched array lengths
        let close = arr1(&[100.0, 101.0, 102.0]);
        let volume = arr1(&[1000.0, 1500.0]);
        let result = obv_gpu(&device, &close, &volume, None);
        assert!(result.is_err(), "Should fail with mismatched array lengths");

        // Empty arrays
        let close = arr1(&[]);
        let volume = arr1(&[]);
        let result = obv_gpu(&device, &close, &volume, None);
        assert!(result.is_err(), "Should fail with empty arrays");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_very_small_price_changes() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test floating-point tolerance with very small price changes
        let close = arr1(&[
            100.0,
            100.0 + 1e-11, // Smaller than EPSILON (1e-10) - should be treated as no change
            100.0 + 1e-9,  // Larger than EPSILON - should register as increase
            100.0,         // Back to baseline
        ]);
        let volume = arr1(&[1000.0, 1000.0, 1000.0, 1000.0]);

        let result = obv_gpu(&device, &close, &volume, None).expect("OBV GPU calculation failed");

        // idx=0: OBV = 0
        // idx=1: change < EPSILON, OBV = 0 + 0 = 0
        // idx=2: change > EPSILON, OBV = 0 + 1000 = 1000
        // idx=3: down, OBV = 1000 - 1000 = 0
        assert!((result[0] - 0.0).abs() < 1e-6, "Expected OBV[0] = 0");
        assert!(
            (result[1] - 0.0).abs() < 1e-6,
            "Expected OBV[1] = 0 (tiny change), got {}",
            result[1]
        );
        assert!(
            (result[2] - 1000.0).abs() < 1e-6,
            "Expected OBV[2] = 1000, got {}",
            result[2]
        );
        assert!(
            (result[3] - 0.0).abs() < 1e-6,
            "Expected OBV[3] = 0, got {}",
            result[3]
        );
    }
}
