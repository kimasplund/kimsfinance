//! Test Parabolic SAR PyO3 bindings
//!
//! Validates that the calculate_parabolic_sar function matches Python expectations

use kimsfinance_core::indicators::{Indicator, ParabolicSAR};
use ndarray::arr1;

#[test]
fn test_parabolic_sar_basic() {
    // Sample data with clear uptrend
    let high = arr1(&[
        110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0,
    ]);
    let low = arr1(&[
        105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0,
    ]);

    let psar = ParabolicSAR::new(0.02, 0.02, 0.2).unwrap();
    let result = psar.calculate_hl(high.view(), low.view()).unwrap();

    // Verify output length
    assert_eq!(result.len(), high.len());

    // First value should not be NaN
    assert!(!result[0].is_nan());

    // All values should be finite
    for (i, &val) in result.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Value at index {} is not finite: {}",
            i,
            val
        );
    }

    // SAR should be within overall price range
    let overall_low = low.iter().cloned().fold(f64::INFINITY, f64::min);
    let overall_high = high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    for (i, &val) in result.iter().enumerate() {
        assert!(
            val >= overall_low - 10.0,
            "SAR {} below overall_low {} at index {}",
            val,
            overall_low,
            i
        );
        assert!(
            val <= overall_high + 10.0,
            "SAR {} above overall_high {} at index {}",
            val,
            overall_high,
            i
        );
    }
}

#[test]
fn test_parabolic_sar_custom_parameters() {
    let high = arr1(&[110.0, 115.0, 120.0, 118.0, 122.0]);
    let low = arr1(&[105.0, 110.0, 115.0, 113.0, 117.0]);

    // Test with different AF parameters
    let psar1 = ParabolicSAR::new(0.01, 0.01, 0.1).unwrap();
    let result1 = psar1.calculate_hl(high.view(), low.view()).unwrap();

    let psar2 = ParabolicSAR::new(0.03, 0.03, 0.3).unwrap();
    let result2 = psar2.calculate_hl(high.view(), low.view()).unwrap();

    // Results should differ with different parameters
    assert_eq!(result1.len(), result2.len());

    // At least some values should be different (skip first value which is initialized same)
    let differences = result1
        .iter()
        .skip(1)
        .zip(result2.iter().skip(1))
        .filter(|&(&a, &b)| (a - b).abs() > 1e-10)
        .count();

    assert!(
        differences > 0,
        "Results should differ with different parameters"
    );
}

#[test]
fn test_parabolic_sar_parameter_validation() {
    let _high = arr1(&[110.0, 115.0, 120.0]);
    let _low = arr1(&[105.0, 110.0, 115.0]);

    // Invalid af_start (negative)
    assert!(ParabolicSAR::new(-0.02, 0.02, 0.2).is_err());

    // Invalid af_start (>= af_max)
    assert!(ParabolicSAR::new(0.3, 0.02, 0.2).is_err());

    // Invalid af_increment (negative)
    assert!(ParabolicSAR::new(0.02, -0.02, 0.2).is_err());

    // Invalid af_max (<= af_start)
    assert!(ParabolicSAR::new(0.02, 0.02, 0.01).is_err());

    // Valid parameters
    assert!(ParabolicSAR::new(0.02, 0.02, 0.2).is_ok());
}

#[test]
fn test_parabolic_sar_length_validation() {
    let high = arr1(&[110.0]);
    let low = arr1(&[105.0]);

    let psar = ParabolicSAR::new(0.02, 0.02, 0.2).unwrap();

    // Insufficient data (need at least 2)
    assert!(psar.calculate_hl(high.view(), low.view()).is_err());
}

#[test]
fn test_parabolic_sar_uptrend_behavior() {
    // Clear uptrend data
    let high = arr1(&[
        101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
    ]);
    let low = arr1(&[
        100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
    ]);

    let psar = ParabolicSAR::new(0.02, 0.02, 0.2).unwrap();
    let sar = psar.calculate_hl(high.view(), low.view()).unwrap();

    // In uptrend, most SAR values should be below the lows (after initial convergence)
    let below_count = sar
        .iter()
        .skip(3) // Skip first few for convergence
        .zip(low.iter().skip(3))
        .filter(|&(&sar_val, &low_val)| sar_val < low_val)
        .count();

    let total_count = sar.len() - 3;

    // Most SAR values should be below lows in uptrend
    assert!(
        below_count as f64 / total_count as f64 > 0.5,
        "Expected majority of SAR below lows in uptrend, got {} / {}",
        below_count,
        total_count
    );
}

#[test]
fn test_parabolic_sar_downtrend_behavior() {
    // Clear downtrend data
    let high = arr1(&[
        110.0, 109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0,
    ]);
    let low = arr1(&[
        109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0,
    ]);

    let psar = ParabolicSAR::new(0.02, 0.02, 0.2).unwrap();
    let sar = psar.calculate_hl(high.view(), low.view()).unwrap();

    // In downtrend, most SAR values should be above the highs (after initial convergence)
    let above_count = sar
        .iter()
        .skip(3) // Skip first few for convergence
        .zip(high.iter().skip(3))
        .filter(|&(&sar_val, &high_val)| sar_val > high_val)
        .count();

    let total_count = sar.len() - 3;

    // Most SAR values should be above highs in downtrend
    assert!(
        above_count as f64 / total_count as f64 > 0.5,
        "Expected majority of SAR above highs in downtrend, got {} / {}",
        above_count,
        total_count
    );
}
