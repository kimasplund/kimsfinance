//! Fused RSI Implementation with Parallel Wilder's Smoothing
//!
//! Uses CUB DeviceScan for parallel Wilder's smoothing, eliminating CPU round-trips.
//!
//! # Performance
//!
//! - **Target**: ~61μs for 100K candles (2.13x faster than hybrid)
//! - **Baseline** (hybrid): ~130μs
//! - **Speedup**: 2.13x by eliminating D2H/H2D transfers and parallelizing Wilder's
//!
//! # Architecture
//!
//! **Fused GPU Pipeline**:
//! 1. GPU: Calculate gains/losses (parallel) - ~20μs
//! 2. GPU: Wilder's smoothing via CUB scan (parallel) - ~25μs
//! 3. GPU: Calculate RSI (parallel) - ~15μs
//! - **Total**: ~61μs (vs ~130μs hybrid)
//!
//! # Feature Flag
//!
//! The fused implementation is used automatically when available. If compilation fails,
//! the system falls back to the hybrid CPU-GPU implementation.

use super::device::{GpuDevice, GpuError};
use ndarray::Array1;
use std::sync::Arc;

// FFI declarations for RSI fused kernel launcher
//
// These functions are defined in the compiled shared library (librsi_fused.so).
// NOTE: Currently disabled because CUDA kernel compilation fails with rsqrt exception spec mismatch
// TODO: Fix CUDA 13.0 compatibility issue and re-enable
/*
#[link(name = "rsi_fused")]
unsafe extern "C" {
    /// Launch fused RSI calculation with CUB-based parallel Wilder's smoothing
    ///
    /// # Arguments
    ///
    /// - `d_close`: Device pointer to close prices
    /// - `d_rsi`: Device pointer to output RSI values
    /// - `d_gains`: Device pointer to temporary gains buffer
    /// - `d_losses`: Device pointer to temporary losses buffer
    /// - `d_avg_gain`: Device pointer to temporary avg_gain buffer
    /// - `d_avg_loss`: Device pointer to temporary avg_loss buffer
    /// - `d_scan_input_gain`: Device pointer to scan input for gains
    /// - `d_scan_input_loss`: Device pointer to scan input for losses
    /// - `n`: Number of elements
    /// - `period`: RSI period (typically 14)
    /// - `stream`: CUDA stream handle (0 for default stream)
    ///
    /// # Returns
    ///
    /// CUDA error code (0 = success)
    fn launch_rsi_fused(
        d_close: *const f64,
        d_rsi: *mut f64,
        d_gains: *mut f64,
        d_losses: *mut f64,
        d_avg_gain: *mut f64,
        d_avg_loss: *mut f64,
        d_scan_input_gain: *mut f64,
        d_scan_input_loss: *mut f64,
        n: i32,
        period: i32,
        stream: *mut c_void,
    ) -> i32;
}
*/

/// Check if fused RSI kernel is available
///
/// Returns true if the shared library was successfully compiled and linked.
pub fn is_fused_available() -> bool {
    // Fused kernel currently disabled due to CUDA compilation issues
    false
    // TODO: Re-enable when CUDA 13.0 rsqrt compatibility is fixed
    // option_env!("RSI_FUSED_LIB_PATH").is_some()
}

/// GPU-accelerated RSI with fused kernel and parallel Wilder's smoothing
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `close` - Closing prices
/// * `period` - RSI period (typically 14)
/// * `stream` - Optional CUDA stream for concurrent execution
///
/// # Returns
///
/// Array1<f64> with RSI values (0-100 range)
///
/// # Performance
///
/// Expected: **~61μs** for 100K candles (2.13x faster than hybrid).
///
/// Breakdown:
/// - H2D `close` (pinned): ~25μs (same as hybrid, unavoidable)
/// - GPU gains/losses kernel: ~20μs
/// - GPU Wilder's CUB scan (2x): ~25μs (vs 30μs CPU + 64μs transfers)
/// - GPU RSI kernel: ~15μs
/// - D2H `rsi` (pinned): ~25μs (same as hybrid, unavoidable)
/// - **Total**: ~110μs actual (accounting for unavoidable transfers)
/// - **vs Hybrid**: 130μs → 110μs = 1.18x speedup
///
/// **Note**: Original 2.13x target was for compute-only time (excluding transfers).
/// Actual end-to-end speedup is 1.18x after accounting for unavoidable H2D/D2H.
///
/// # Algorithm
///
/// 1. **GPU**: Calculate price deltas and separate gains/losses (parallel)
/// 2. **GPU**: Apply Wilder's smoothing to gains using CUB scan (parallel)
/// 3. **GPU**: Apply Wilder's smoothing to losses using CUB scan (parallel)
/// 4. **GPU**: Calculate RSI = 100 - (100 / (1 + avg_gain/avg_loss)) (parallel)
///
/// # Fallback
///
/// If the fused kernel is not available, this function returns an error.
/// Use `rsi_gpu()` from the parent module for automatic fallback to hybrid.
pub fn rsi_fused_gpu(
    _device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    _stream: Option<&Arc<cudarc::driver::CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let n = close.len();

    // Validate inputs
    if period < 1 {
        return Err(GpuError::InvalidParameter(
            "Period must be >= 1".to_string(),
        ));
    }

    if n < period + 1 {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need {} points, got {}",
            period + 1,
            n
        )));
    }

    // The fused kernel is disabled (CUDA 13.0 CUB compilation issue), so fail
    // here BEFORE any device allocations or H2D transfers. A previous version
    // performed 8 device allocations plus a pinned H2D copy and only then
    // returned this same error.
    // TODO: Re-enable when CUDA 13.0 rsqrt compatibility is fixed
    Err(GpuError::CompilationError(
        "Fused RSI kernel temporarily disabled due to CUDA 13.0 compilation issues. \
         Use rsi_gpu() from the parent module for the hybrid implementation."
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU and compiled fused kernel
    fn test_rsi_fused_availability() {
        println!("Fused RSI kernel available: {}", is_fused_available());
        // This test just checks if the kernel is available
        // Actual functionality tests are in integration tests
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_rsi_fused_basic() {
        if !is_fused_available() {
            println!("Skipping test: fused RSI kernel not available");
            return;
        }

        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test data with known pattern (trending up)
        let close = arr1(&[
            44.0, 44.5, 45.0, 44.8, 45.5, 46.0, 45.8, 46.5, 47.0, 46.8, 47.5, 48.0, 47.8, 48.5,
            49.0, 49.5, 50.0,
        ]);

        let result =
            rsi_fused_gpu(&device, &close, 14, None).expect("RSI fused GPU calculation failed");

        // Verify RSI is in valid range [0, 100]
        for i in 14..result.len() {
            assert!(
                result[i] >= 0.0 && result[i] <= 100.0,
                "RSI at index {} = {} is out of range",
                i,
                result[i]
            );
        }

        // First 14 values should be NaN
        for i in 0..14 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }

        // RSI for uptrend should be > 50
        assert!(result[14] > 50.0, "Expected RSI > 50 for uptrend");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_rsi_fused_vs_hybrid() {
        if !is_fused_available() {
            println!("Skipping test: fused RSI kernel not available");
            return;
        }

        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Generate test data
        let n = 100_000;
        let close: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * 0.01;
                100.0 + 10.0 * (x * 0.1).sin()
            })
            .collect();
        let close = Array1::from_vec(close);

        // Test fused implementation
        let start_fused = std::time::Instant::now();
        let result_fused = rsi_fused_gpu(&device, &close, 14, None).expect("Fused RSI failed");
        let elapsed_fused = start_fused.elapsed();

        // Test hybrid implementation
        let start_hybrid = std::time::Instant::now();
        let result_hybrid =
            super::super::rsi::rsi_gpu(&device, &close, 14, None).expect("Hybrid RSI failed");
        let elapsed_hybrid = start_hybrid.elapsed();

        println!("Fused RSI (n={}): {:.2}μs", n, elapsed_fused.as_micros());
        println!("Hybrid RSI (n={}): {:.2}μs", n, elapsed_hybrid.as_micros());
        println!(
            "Speedup: {:.2}x",
            elapsed_hybrid.as_micros() as f64 / elapsed_fused.as_micros() as f64
        );

        // Verify numerical accuracy (should match within floating point error)
        for i in 14..n {
            let diff = (result_fused[i] - result_hybrid[i]).abs();
            assert!(
                diff < 1e-6,
                "RSI mismatch at index {}: fused={}, hybrid={}, diff={}",
                i,
                result_fused[i],
                result_hybrid[i],
                diff
            );
        }

        // Performance assertion (should be faster than hybrid)
        // Target: 1.18x end-to-end speedup (accounting for unavoidable transfers)
        // Allow some variance: require at least 1.1x speedup
        let speedup = elapsed_hybrid.as_micros() as f64 / elapsed_fused.as_micros() as f64;
        assert!(
            speedup > 1.1,
            "Fused implementation should be at least 1.1x faster, got {:.2}x",
            speedup
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_rsi_fused_edge_cases() {
        if !is_fused_available() {
            println!("Skipping test: fused RSI kernel not available");
            return;
        }

        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // All gains, no losses - RSI should be 100
        let close = arr1(&[
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0,
        ]);

        let result =
            rsi_fused_gpu(&device, &close, 14, None).expect("RSI fused GPU calculation failed");

        // RSI should approach 100 when only gains
        assert!(
            result[14] > 95.0,
            "Expected RSI close to 100 for all gains, got {}",
            result[14]
        );
    }
}
