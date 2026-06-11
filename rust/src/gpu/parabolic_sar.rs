//! Parabolic SAR (Stop and Reverse) - CPU Implementation
//!
//! Parabolic SAR tracks trend direction and provides trailing stop levels.
//!
//! # Why CPU?
//!
//! Parabolic SAR is inherently sequential - each candle's SAR depends on:
//! 1. Previous SAR value
//! 2. Current trend state (uptrend/downtrend)
//! 3. Acceleration Factor (AF) that changes based on extreme points
//!
//! This creates a sequential dependency chain that cannot be parallelized
//! across candles. An earlier "hybrid" version of this module computed the
//! full result on CPU and then additionally performed 8 device allocations,
//! 6 H2D copies, 3 kernel launches, and a stream sync whose outputs were
//! discarded. That dead GPU pass has been removed: `parabolic_sar_gpu` is
//! now a thin wrapper over the CPU loop (strictly faster, identical output).
//!
//! # Algorithm
//!
//! Parabolic SAR formula:
//! - SAR[i] = SAR[i-1] + AF * (EP - SAR[i-1])
//! - EP = Extreme Point (highest high in uptrend, lowest low in downtrend)
//! - AF = Acceleration Factor (starts at af_start, increments by af_increment on new EP, max af_max)
//!
//! Constraints:
//! - Uptrend: SAR cannot exceed prior 2 lows
//! - Downtrend: SAR cannot be below prior 2 highs
//!
//! Reversal:
//! - Uptrend: Reversal if low <= SAR
//! - Downtrend: Reversal if high >= SAR
//! - On reversal: Switch trend, reset AF to af_start, set EP to new extreme

use super::device::{GpuDevice, GpuError};
use cudarc::driver::CudaStream;
use ndarray::Array1;
use std::sync::Arc;

/// Validate Parabolic SAR inputs (pure, host-side)
fn validate_parabolic_sar_inputs(
    n_high: usize,
    n_low: usize,
    af_start: f64,
    af_increment: f64,
    af_max: f64,
) -> Result<(), GpuError> {
    if n_high != n_low {
        return Err(GpuError::InvalidParameter(
            "High and low arrays must have same length".to_string(),
        ));
    }

    if n_high < 2 {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need at least 2 points, got {}",
            n_high
        )));
    }

    if af_start <= 0.0 || af_start > af_max {
        return Err(GpuError::InvalidParameter(format!(
            "Invalid af_start: must be 0 < af_start <= af_max (got {}, max {})",
            af_start, af_max
        )));
    }

    if af_increment <= 0.0 {
        return Err(GpuError::InvalidParameter(format!(
            "Invalid af_increment: must be > 0 (got {})",
            af_increment
        )));
    }

    Ok(())
}

/// Sequential Parabolic SAR computation (normative implementation)
///
/// Inputs must already be validated (same length, n >= 2, valid AF params).
fn parabolic_sar_cpu_impl(
    high: &Array1<f64>,
    low: &Array1<f64>,
    af_start: f64,
    af_increment: f64,
    af_max: f64,
) -> (Array1<f64>, Array1<i8>) {
    let n = high.len();

    // === Initialize state arrays ===
    let mut sar = vec![f64::NAN; n];
    let mut signal = vec![0i8; n];
    let mut is_long = vec![1i32; n]; // Start with uptrend
    let mut af = vec![af_start; n];
    let mut ep = vec![high[0]; n];
    let mut prev_sar = vec![low[0]; n];

    // Initialize first SAR value
    sar[0] = low[0];
    signal[0] = 1; // Start in uptrend

    for i in 1..n {
        // === Step 1: Calculate SAR candidate ===
        let sar_candidate = prev_sar[i - 1] + af[i - 1] * (ep[i - 1] - prev_sar[i - 1]);

        // === Step 2: Apply constraints ===
        let constrained_sar = if is_long[i - 1] == 1 {
            // Uptrend: SAR cannot exceed prior 2 lows
            if i >= 2 {
                sar_candidate.min(low[i - 1]).min(low[i - 2])
            } else {
                sar_candidate.min(low[i - 1])
            }
        } else {
            // Downtrend: SAR cannot be below prior 2 highs
            if i >= 2 {
                sar_candidate.max(high[i - 1]).max(high[i - 2])
            } else {
                sar_candidate.max(high[i - 1])
            }
        };

        sar[i] = constrained_sar;

        // === Step 3: Check for reversal ===
        let reversal = if is_long[i - 1] == 1 {
            // In uptrend: reversal if low crosses below SAR
            low[i] <= sar[i]
        } else {
            // In downtrend: reversal if high crosses above SAR
            high[i] >= sar[i]
        };

        if reversal {
            // === Step 4: Handle reversal ===
            is_long[i] = 1 - is_long[i - 1]; // Flip trend
            sar[i] = ep[i - 1]; // SAR becomes previous extreme point
            ep[i] = if is_long[i] == 1 { high[i] } else { low[i] };
            af[i] = af_start; // Reset AF
            signal[i] = if is_long[i] == 1 { 1 } else { -1 };
        } else {
            // === Step 5: Continue current trend ===
            is_long[i] = is_long[i - 1];
            signal[i] = signal[i - 1];

            // Update extreme point
            if is_long[i] == 1 {
                // Uptrend: check for new high
                if high[i] > ep[i - 1] {
                    ep[i] = high[i];
                    af[i] = (af[i - 1] + af_increment).min(af_max);
                } else {
                    ep[i] = ep[i - 1];
                    af[i] = af[i - 1];
                }
            } else {
                // Downtrend: check for new low
                if low[i] < ep[i - 1] {
                    ep[i] = low[i];
                    af[i] = (af[i - 1] + af_increment).min(af_max);
                } else {
                    ep[i] = ep[i - 1];
                    af[i] = af[i - 1];
                }
            }
        }

        // Update prev_sar for next iteration
        prev_sar[i] = sar[i];
    }

    (Array1::from_vec(sar), Array1::from_vec(signal))
}

/// Parabolic SAR (Stop and Reverse) - executes on CPU
///
/// The `device` and `stream` parameters are unused and kept only for API
/// compatibility with the other indicator entry points in this module.
///
/// # Arguments
///
/// * `device` - GPU device handle (unused; kept for API compatibility)
/// * `high` - High prices
/// * `low` - Low prices
/// * `af_start` - Initial acceleration factor (typically 0.02)
/// * `af_increment` - AF increment per new extreme (typically 0.02)
/// * `af_max` - Maximum AF value (typically 0.2)
/// * `stream` - CUDA stream (unused; kept for API compatibility)
///
/// # Returns
///
/// Tuple of (SAR values, trend signal) as (Array1<f64>, Array1<i8>)
/// - SAR values: Stop and Reverse levels
/// - Trend signal: 1 = uptrend, -1 = downtrend, 0 = initial/warmup
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, parabolic_sar_gpu};
/// use ndarray::Array1;
///
/// let device = GpuDevice::new()?;
/// let high = Array1::from_vec(vec![110.0, 115.0, 120.0, /* ... */]);
/// let low = Array1::from_vec(vec![105.0, 110.0, 115.0, /* ... */]);
///
/// let (sar, signal) = parabolic_sar_gpu(&device, &high, &low, 0.02, 0.02, 0.2, None)?;
/// ```
pub fn parabolic_sar_gpu(
    _device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    af_start: f64,
    af_increment: f64,
    af_max: f64,
    _stream: Option<&Arc<CudaStream>>,
) -> Result<(Array1<f64>, Array1<i8>), GpuError> {
    validate_parabolic_sar_inputs(high.len(), low.len(), af_start, af_increment, af_max)?;
    Ok(parabolic_sar_cpu_impl(
        high,
        low,
        af_start,
        af_increment,
        af_max,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    // ==================== CPU tests (no GPU required) ====================

    #[test]
    fn test_parabolic_sar_cpu_basic() {
        // Test data with uptrend
        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0,
        ]);

        let (sar, signal) = parabolic_sar_cpu_impl(&high, &low, 0.02, 0.02, 0.2);

        // Verify output lengths
        assert_eq!(sar.len(), high.len());
        assert_eq!(signal.len(), high.len());

        // First value should be initialized
        assert!(!sar[0].is_nan());
        assert_eq!(signal[0], 1); // Start in uptrend

        // SAR should be within reasonable range
        let overall_low = low.iter().cloned().fold(f64::INFINITY, f64::min);
        let overall_high = high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for i in 0..sar.len() {
            assert!(
                sar[i] >= overall_low - 10.0,
                "SAR {} below overall_low {} at index {}",
                sar[i],
                overall_low,
                i
            );
            assert!(
                sar[i] <= overall_high + 10.0,
                "SAR {} above overall_high {} at index {}",
                sar[i],
                overall_high,
                i
            );
        }

        // Signal should only be -1, 0, or 1
        for i in 0..signal.len() {
            assert!(
                signal[i] == -1 || signal[i] == 0 || signal[i] == 1,
                "Invalid signal at index {}: {}",
                i,
                signal[i]
            );
        }
    }

    #[test]
    fn test_parabolic_sar_cpu_reversal() {
        // Test data with clear reversal (uptrend then downtrend)
        let high = arr1(&[
            110.0, 115.0, 120.0, 125.0, 130.0, // Uptrend
            128.0, 123.0, 118.0, 113.0, 108.0, // Downtrend
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 120.0, 125.0, // Uptrend
            123.0, 118.0, 113.0, 108.0, 103.0, // Downtrend
        ]);

        let (_sar, signal) = parabolic_sar_cpu_impl(&high, &low, 0.02, 0.02, 0.2);

        // Should start in uptrend
        assert_eq!(signal[0], 1);

        // Should eventually switch to downtrend
        let has_downtrend = signal.iter().any(|&s| s == -1);
        assert!(has_downtrend, "Expected to detect downtrend reversal");
    }

    #[test]
    fn test_parabolic_sar_cpu_constant_prices() {
        // Constant prices - no trend
        let high = arr1(&[110.0; 30]);
        let low = arr1(&[100.0; 30]);

        let (sar, signal) = parabolic_sar_cpu_impl(&high, &low, 0.02, 0.02, 0.2);

        // With constant prices, should maintain initial trend
        assert_eq!(signal[0], 1);

        // SAR should be within price range
        for i in 0..sar.len() {
            assert!(
                sar[i] >= 100.0 && sar[i] <= 110.0,
                "SAR out of range at index {}: {}",
                i,
                sar[i]
            );
        }
    }

    #[test]
    fn test_parabolic_sar_cpu_large_dataset() {
        // Generate large dataset with sine wave pattern
        let n = 100_000;
        let high: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * 0.01;
                110.0 + 10.0 * (x * 0.1).sin()
            })
            .collect();
        let low: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * 0.01;
                100.0 + 10.0 * (x * 0.1).sin()
            })
            .collect();

        let high = Array1::from_vec(high);
        let low = Array1::from_vec(low);

        let (sar, signal) = parabolic_sar_cpu_impl(&high, &low, 0.02, 0.02, 0.2);

        // Verify output size
        assert_eq!(sar.len(), n);
        assert_eq!(signal.len(), n);

        // Verify all SAR values are valid (not NaN)
        for i in 0..n {
            assert!(!sar[i].is_nan(), "SAR should not be NaN at index {}", i);
        }

        // Verify oscillating data produces both uptrends and downtrends
        let has_uptrend = signal.iter().any(|&s| s == 1);
        let has_downtrend = signal.iter().any(|&s| s == -1);
        assert!(has_uptrend, "Expected to detect uptrends");
        assert!(has_downtrend, "Expected to detect downtrends");
    }

    #[test]
    fn test_parabolic_sar_input_validation() {
        // Mismatched lengths
        assert!(validate_parabolic_sar_inputs(3, 2, 0.02, 0.02, 0.2).is_err());

        // Too short dataset
        assert!(validate_parabolic_sar_inputs(1, 1, 0.02, 0.02, 0.2).is_err());

        // Invalid af_start (zero)
        assert!(validate_parabolic_sar_inputs(3, 3, 0.0, 0.02, 0.2).is_err());

        // af_start > af_max
        assert!(validate_parabolic_sar_inputs(3, 3, 0.3, 0.02, 0.2).is_err());

        // Invalid af_increment
        assert!(validate_parabolic_sar_inputs(3, 3, 0.02, 0.0, 0.2).is_err());

        // Valid inputs
        assert!(validate_parabolic_sar_inputs(3, 3, 0.02, 0.02, 0.2).is_ok());
    }

    #[test]
    fn test_parabolic_sar_cpu_af_increment() {
        // Strong uptrend to test AF increment
        let high = arr1(&[
            100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0,
        ]);
        let low = arr1(&[
            95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0,
        ]);

        let (sar, signal) = parabolic_sar_cpu_impl(&high, &low, 0.02, 0.02, 0.2);

        // Should maintain uptrend
        for i in 0..5 {
            assert_eq!(signal[i], 1, "Expected uptrend signal at index {}", i);
        }

        // SAR should stay above initial low in a consistent uptrend
        for i in 2..sar.len() {
            if signal[i] == 1 && signal[i - 1] == 1 {
                assert!(
                    sar[i] >= low[0],
                    "SAR should stay above initial low in uptrend"
                );
            }
        }
    }

    // ==================== GPU API tests (device handle only) ====================

    #[test]
    #[ignore] // Requires GPU (for device construction only; computation is CPU)
    fn test_parabolic_sar_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0,
        ]);

        let (sar, signal) = parabolic_sar_gpu(&device, &high, &low, 0.02, 0.02, 0.2, None)
            .expect("Parabolic SAR calculation failed");

        // Must match the CPU implementation exactly
        let (sar_cpu, signal_cpu) = parabolic_sar_cpu_impl(&high, &low, 0.02, 0.02, 0.2);
        for i in 0..sar.len() {
            assert_eq!(sar[i], sar_cpu[i], "SAR mismatch at index {}", i);
            assert_eq!(signal[i], signal_cpu[i], "Signal mismatch at index {}", i);
        }
    }

    #[test]
    #[ignore] // Requires GPU (for device construction only; computation is CPU)
    fn test_parabolic_sar_gpu_invalid_inputs() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Mismatched lengths
        let high = arr1(&[110.0, 115.0, 120.0]);
        let low = arr1(&[105.0, 110.0]);
        let result = parabolic_sar_gpu(&device, &high, &low, 0.02, 0.02, 0.2, None);
        assert!(result.is_err(), "Should fail with mismatched lengths");

        // Too short dataset
        let high = arr1(&[110.0]);
        let low = arr1(&[105.0]);
        let result = parabolic_sar_gpu(&device, &high, &low, 0.02, 0.02, 0.2, None);
        assert!(result.is_err(), "Should fail with insufficient data");

        // Invalid af_start
        let high = arr1(&[110.0, 115.0, 120.0]);
        let low = arr1(&[105.0, 110.0, 115.0]);
        let result = parabolic_sar_gpu(&device, &high, &low, 0.0, 0.02, 0.2, None);
        assert!(result.is_err(), "Should fail with af_start = 0");

        // af_start > af_max
        let result = parabolic_sar_gpu(&device, &high, &low, 0.3, 0.02, 0.2, None);
        assert!(result.is_err(), "Should fail with af_start > af_max");

        // Invalid af_increment
        let result = parabolic_sar_gpu(&device, &high, &low, 0.02, 0.0, 0.2, None);
        assert!(result.is_err(), "Should fail with af_increment = 0");
    }
}
