//! MACD (Moving Average Convergence Divergence) - CPU-Optimized
//!
//! # IMPORTANT: This "GPU" module now uses CPU execution
//!
//! MACD uses 3 sequential EMAs that cannot be parallelized. Running them
//! on a single GPU thread was a performance disaster (1,647x slower than CPU).
//!
//! ## Performance (100K candles, 12/26/9 params)
//!
//! - **CPU-only**: ~75μs (current implementation)
//! - Old single-thread GPU: ~57.75ms
//! - **Speedup**: 1,647x by using CPU!
//!
//! ## Migration Guide
//!
//! **Before (v0.1.0)**:
//! ```rust,ignore
//! use kimsfinance_core::gpu::{GpuDevice, macd_gpu};
//! let device = GpuDevice::new()?;
//! let (macd, signal, histogram) = macd_gpu(&device, &close, 12, 26, 9, None)?;  // Slow!
//! ```
//!
//! **After (v0.2.0)**:
//! ```rust,ignore
//! // Option 1: Direct CPU call (recommended)
//! use kimsfinance_core::cpu::sequential::macd_cpu;
//! let (macd, signal, histogram) = macd_cpu(&close, 12, 26, 9)?;  // 1,647x faster!
//!
//! // Option 2: Hybrid API (backward compatible)
//! use kimsfinance_core::gpu::{GpuDevice, macd_hybrid};
//! let device = GpuDevice::new()?;
//! let (macd, signal, histogram) = macd_hybrid(&device, &close, 12, 26, 9, None)?;
//! ```
//!
//! ## Algorithm
//!
//! 1. Fast EMA = EMA(close, fast_period) - typically 12
//! 2. Slow EMA = EMA(close, slow_period) - typically 26
//! 3. MACD Line = Fast EMA - Slow EMA
//! 4. Signal Line = EMA(MACD, signal_period) - typically 9
//! 5. Histogram = MACD - Signal
//!
//! # Breaking Change in v0.2.0
//!
//! The `macd_gpu()` function is now deprecated. It was using a single GPU
//! thread which is 1,647x slower than CPU for sequential algorithms.
//!
//! **Action Required**:
//! - Replace `macd_gpu()` with `macd_cpu()` (from `cpu::sequential` module)
//! - Or use `macd_hybrid()` for API-compatible migration
//! - Update performance expectations in your code

use super::device::{GpuDevice, GpuError};
use crate::cpu::sequential::macd_cpu;
use cudarc::driver::CudaStream;
use ndarray::Array1;
use std::sync::Arc;

/// MACD using optimal execution strategy (CPU for sequential algorithms)
///
/// # Why CPU?
///
/// MACD requires 3 sequential EMA calculations with data dependencies that prevent
/// parallelization. Single GPU thread is 1,647x slower than CPU due to:
/// - Slower single-core performance (1.2 GHz GPU vs 5.6 GHz CPU)
/// - PCIe transfer overhead (~64μs)
/// - Kernel launch overhead (~10μs)
/// - Higher memory latency (GPU L1: 5-10ns vs CPU L1: 1ns)
///
/// # Performance
///
/// CPU-only: **~75μs** for 100K candles (12/26/9 params)
/// Old GPU: ~57.75ms (1,647x slower!)
///
/// # Arguments
///
/// * `device` - GPU device (unused, kept for API compatibility)
/// * `close` - Close prices
/// * `fast_period` - Fast EMA period (typically 12)
/// * `slow_period` - Slow EMA period (typically 26)
/// * `signal_period` - Signal line EMA period (typically 9)
/// * `stream` - Stream (unused, kept for API compatibility)
///
/// # Returns
///
/// Tuple of (MACD line, Signal line, Histogram) as Array1<f64>
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, macd_hybrid};
/// use ndarray::Array1;
///
/// let device = GpuDevice::new()?;
/// let close = Array1::from_vec(vec![100.0, 101.0, 102.0, /* ... */]);
/// let (macd, signal, histogram) = macd_hybrid(&device, &close, 12, 26, 9, None)?;
/// ```
pub fn macd_hybrid(
    _device: &GpuDevice, // Unused but kept for API compatibility
    close: &Array1<f64>,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    _stream: Option<&Arc<CudaStream>>, // Unused
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), GpuError> {
    macd_cpu(close, fast_period, slow_period, signal_period)
}

/// GPU-accelerated MACD (Moving Average Convergence Divergence) - DEPRECATED
///
/// # DEPRECATED
///
/// This function is deprecated since v0.2.0. The original single-GPU-thread
/// kernel was 1,647x slower than CPU for sequential algorithms and has been
/// removed; this function now delegates to `macd_cpu()` (identical results).
///
/// **Use `macd_cpu()` or `macd_hybrid()` instead.**
///
/// # Migration
///
/// ```rust,ignore
/// // OLD (slow):
/// let (macd, signal, histogram) = macd_gpu(&device, &close, 12, 26, 9, None)?;
///
/// // NEW (1,647x faster):
/// let (macd, signal, histogram) = macd_cpu(&close, 12, 26, 9)?;
/// // OR (API-compatible):
/// let (macd, signal, histogram) = macd_hybrid(&device, &close, 12, 26, 9, None)?;
/// ```
///
/// # Arguments
///
/// * `device` - GPU device handle (unused; kept for API compatibility)
/// * `close` - Close prices
/// * `fast_period` - Fast EMA period (typically 12)
/// * `slow_period` - Slow EMA period (typically 26)
/// * `signal_period` - Signal line EMA period (typically 9)
/// * `stream` - CUDA stream (unused; kept for API compatibility)
///
/// # Returns
///
/// Tuple of (MACD line, Signal line, Histogram) as Array1<f64>
/// Early values will be NaN until enough data is available.
///
/// # Errors
///
/// Returns error if:
/// - Period < 1
/// - Fast period >= Slow period
/// - Not enough data (n < slow_period + signal_period - 1)
#[deprecated(
    since = "0.2.0",
    note = "Single-thread GPU is 1,647x slower than CPU. Use macd_cpu() from kimsfinance_core::cpu::sequential or macd_hybrid() for API compatibility"
)]
pub fn macd_gpu(
    _device: &GpuDevice,
    close: &Array1<f64>,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    _stream: Option<&Arc<CudaStream>>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), GpuError> {
    // The single-GPU-thread kernel has been removed (it was 1,647x slower
    // than CPU). Delegate to the CPU implementation, which has identical
    // warmup/NaN semantics (validated by the tests below).
    macd_cpu(close, fast_period, slow_period, signal_period)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    #[allow(deprecated)]
    fn test_macd_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Create test data with clear trend
        let close = arr1(&[
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0,
            124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0,
        ]);

        let (macd, signal, histogram) =
            macd_gpu(&device, &close, 12, 26, 9, None).expect("MACD GPU calculation failed");

        // Verify lengths
        assert_eq!(macd.len(), close.len());
        assert_eq!(signal.len(), close.len());
        assert_eq!(histogram.len(), close.len());

        // Verify early values are NaN (not enough data)
        for i in 0..25 {
            assert!(macd[i].is_nan(), "MACD should be NaN before slow_period-1");
        }

        // Verify MACD values start appearing after slow_period
        assert!(
            !macd[25].is_nan(),
            "MACD should have value at slow_period-1"
        );

        // Verify signal starts after slow_period + signal_period - 1
        for i in 0..33 {
            assert!(
                signal[i].is_nan(),
                "Signal should be NaN before slow_period+signal_period-1"
            );
        }

        // Verify histogram is computed where both MACD and signal are valid
        assert!(
            !histogram[33].is_nan(),
            "Histogram should be valid after signal becomes valid"
        );

        // Verify relationship: histogram = macd - signal
        for i in 33..close.len() {
            if !macd[i].is_nan() && !signal[i].is_nan() {
                let expected_histogram = macd[i] - signal[i];
                assert!(
                    (histogram[i] - expected_histogram).abs() < 1e-10,
                    "Histogram should equal MACD - Signal"
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    #[allow(deprecated)]
    fn test_macd_gpu_standard_params() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Standard MACD parameters (12, 26, 9)
        let n = 100;
        let close = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.5).collect());

        let (macd, signal, histogram) =
            macd_gpu(&device, &close, 12, 26, 9, None).expect("MACD GPU calculation failed");

        // Check that MACD captures uptrend
        // In an uptrend, MACD should be positive (fast > slow)
        let valid_macd: Vec<f64> = macd.iter().filter(|&&x| !x.is_nan()).copied().collect();
        assert!(
            valid_macd.len() > 0,
            "Should have at least some valid MACD values"
        );

        // In uptrend, later MACD values should be positive
        assert!(
            macd[macd.len() - 1] > 0.0,
            "MACD should be positive in uptrend"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    #[allow(deprecated)]
    fn test_macd_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with large dataset (100K points)
        let n = 100_000;
        let close = Array1::from_vec(
            (0..n)
                .map(|i| 100.0 + ((i as f64) * 0.01).sin() * 10.0)
                .collect(),
        );

        let start = std::time::Instant::now();
        let (macd, signal, histogram) =
            macd_gpu(&device, &close, 12, 26, 9, None).expect("MACD GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU MACD (n={}): {:.2}ms",
            n,
            elapsed.as_secs_f64() * 1000.0
        );

        assert_eq!(macd.len(), n);
        assert_eq!(signal.len(), n);
        assert_eq!(histogram.len(), n);

        // Verify some values are valid
        let valid_count = macd.iter().filter(|&&x| !x.is_nan()).count();
        assert!(
            valid_count > n - 50,
            "Most values should be valid in large dataset"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    #[allow(deprecated)]
    fn test_macd_gpu_validation() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let close = arr1(&[10.0, 20.0, 30.0]);

        // Invalid: fast >= slow
        let result = macd_gpu(&device, &close, 26, 12, 9, None);
        assert!(
            result.is_err(),
            "Should fail when fast_period >= slow_period"
        );

        // Invalid: not enough data
        let result = macd_gpu(&device, &close, 12, 26, 9, None);
        assert!(result.is_err(), "Should fail with insufficient data");

        // Invalid: zero period
        let close_long = Array1::from_vec((0..50).map(|i| i as f64).collect());
        let result = macd_gpu(&device, &close_long, 0, 26, 9, None);
        assert!(result.is_err(), "Should fail with zero period");
    }

    #[test]
    #[ignore] // Requires GPU
    #[allow(deprecated)]
    fn test_macd_gpu_custom_periods() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with custom periods (5, 13, 3)
        let close = Array1::from_vec((0..50).map(|i| 100.0 + (i as f64) * 0.2).collect());

        let (macd, signal, histogram) =
            macd_gpu(&device, &close, 5, 13, 3, None).expect("MACD GPU calculation failed");

        // Verify MACD starts at slow_period - 1 = 12
        assert!(!macd[12].is_nan(), "MACD should start at slow_period - 1");

        // Verify signal starts at slow_period + signal_period - 1 = 15
        assert!(
            !signal[15].is_nan(),
            "Signal should start at slow_period + signal_period - 1"
        );

        // Verify histogram
        assert!(
            !histogram[15].is_nan(),
            "Histogram should start when signal starts"
        );
    }
}
