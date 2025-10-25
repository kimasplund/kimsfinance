//! Trend Indicators
//!
//! Implements 2 trend indicators:
//! - Parabolic SAR (Stop and Reverse)
//! - Pivot Points (support/resistance levels)

use super::core::{
    Indicator, IndicatorError, IndicatorOutput, IndicatorResult, validate_lengths,
    validate_min_periods,
};
use ndarray::{Array1, ArrayView1};

/// Parabolic SAR (Stop and Reverse)
///
/// Trailing stop indicator that follows price trends.
/// Provides potential reversal points.
pub struct ParabolicSAR {
    af_start: f64,
    af_increment: f64,
    af_max: f64,
}

impl ParabolicSAR {
    pub fn new(af_start: f64, af_increment: f64, af_max: f64) -> Result<Self, IndicatorError> {
        if af_start <= 0.0 || af_start > af_max {
            return Err(IndicatorError::InvalidParameter {
                name: "af_start",
                value: af_start.to_string(),
            });
        }
        if af_increment <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "af_increment",
                value: af_increment.to_string(),
            });
        }
        if af_max <= af_start {
            return Err(IndicatorError::InvalidParameter {
                name: "af_max",
                value: af_max.to_string(),
            });
        }

        Ok(Self {
            af_start,
            af_increment,
            af_max,
        })
    }

    /// Calculate Parabolic SAR with high and low prices
    pub fn calculate_hl<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low])?;
        validate_min_periods(n, 2)?;

        let mut sar = Array1::from_elem(n, f64::NAN);

        // Initialize
        let mut is_long = true;
        let mut af = self.af_start;
        let mut ep = high[0]; // Extreme point
        sar[0] = low[0];

        for i in 1..n {
            // Calculate SAR for this period
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1]);

            // Check for reversal
            if is_long {
                // In uptrend
                if low[i] <= sar[i] {
                    // Reversal to downtrend
                    is_long = false;
                    sar[i] = ep; // SAR becomes the extreme point
                    ep = low[i];
                    af = self.af_start;
                } else {
                    // Continue uptrend
                    if high[i] > ep {
                        ep = high[i];
                        af = (af + self.af_increment).min(self.af_max);
                    }

                    // SAR cannot be above prior two lows
                    if i >= 2 {
                        sar[i] = sar[i].min(low[i - 1]).min(low[i - 2]);
                    } else if i >= 1 {
                        sar[i] = sar[i].min(low[i - 1]);
                    }
                }
            } else {
                // In downtrend
                if high[i] >= sar[i] {
                    // Reversal to uptrend
                    is_long = true;
                    sar[i] = ep; // SAR becomes the extreme point
                    ep = high[i];
                    af = self.af_start;
                } else {
                    // Continue downtrend
                    if low[i] < ep {
                        ep = low[i];
                        af = (af + self.af_increment).min(self.af_max);
                    }

                    // SAR cannot be below prior two highs
                    if i >= 2 {
                        sar[i] = sar[i].max(high[i - 1]).max(high[i - 2]);
                    } else if i >= 1 {
                        sar[i] = sar[i].max(high[i - 1]);
                    }
                }
            }
        }

        Ok(sar)
    }
}

impl Indicator for ParabolicSAR {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "Parabolic SAR requires high and low. Use calculate_hl()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn name(&self) -> &'static str {
        "Parabolic SAR"
    }
}

/// Pivot Points
///
/// Support and resistance levels calculated from previous period's OHLC.
/// Returns pivot point, resistance levels (R1, R2, R3), and support levels (S1, S2, S3).
pub struct PivotPoints;

impl PivotPoints {
    pub fn new() -> Self {
        Self
    }

    /// Calculate pivot points from a single period's OHLC
    ///
    /// Returns [PP, R1, R2, R3, S1, S2, S3]
    pub fn calculate_single(&self, high: f64, low: f64, close: f64) -> [f64; 7] {
        // Pivot Point = (H + L + C) / 3
        let pp = (high + low + close) / 3.0;

        // Resistance levels
        let r1 = 2.0 * pp - low;
        let r2 = pp + (high - low);
        let r3 = high + 2.0 * (pp - low);

        // Support levels
        let s1 = 2.0 * pp - high;
        let s2 = pp - (high - low);
        let s3 = low - 2.0 * (high - pp);

        [pp, r1, r2, r3, s1, s2, s3]
    }

    /// Calculate pivot points for each period using previous period's data
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> Result<IndicatorOutput, IndicatorError> {
        let n = validate_lengths(&[high, low, close])?;

        let mut pp = Array1::from_elem(n, f64::NAN);
        let mut r1 = Array1::from_elem(n, f64::NAN);
        let mut r2 = Array1::from_elem(n, f64::NAN);
        let mut r3 = Array1::from_elem(n, f64::NAN);
        let mut s1 = Array1::from_elem(n, f64::NAN);
        let mut s2 = Array1::from_elem(n, f64::NAN);
        let mut s3 = Array1::from_elem(n, f64::NAN);

        // Calculate pivots using previous period's data
        for i in 1..n {
            let levels = self.calculate_single(high[i - 1], low[i - 1], close[i - 1]);
            pp[i] = levels[0];
            r1[i] = levels[1];
            r2[i] = levels[2];
            r3[i] = levels[3];
            s1[i] = levels[4];
            s2[i] = levels[5];
            s3[i] = levels[6];
        }

        Ok(IndicatorOutput {
            primary: pp,
            secondary: vec![r1, r2, r3, s1, s2, s3],
            metadata: None,
        })
    }
}

impl Indicator for PivotPoints {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "Pivot Points require high, low, close. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn name(&self) -> &'static str {
        "Pivot Points"
    }
}

/// Fibonacci Retracement Levels
///
/// Calculates Fibonacci retracement levels based on swing high and low.
/// Returns levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
pub struct FibonacciRetracement;

impl FibonacciRetracement {
    pub fn new() -> Self {
        Self
    }

    /// Calculate Fibonacci levels from swing high to swing low
    ///
    /// Returns [0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%]
    pub fn calculate_levels(&self, high: f64, low: f64) -> [f64; 7] {
        let diff = high - low;

        [
            high,                // 0% (swing high)
            high - 0.236 * diff, // 23.6%
            high - 0.382 * diff, // 38.2%
            high - 0.500 * diff, // 50%
            high - 0.618 * diff, // 61.8%
            high - 0.786 * diff, // 78.6%
            low,                 // 100% (swing low)
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_parabolic_sar() {
        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0,
        ]);

        let psar = ParabolicSAR::new(0.02, 0.02, 0.2).unwrap();
        let result = psar.calculate_hl(high.view(), low.view()).unwrap();

        // SAR should have values
        assert!(!result[1].is_nan());
        assert!(result[1] > 0.0);

        // SAR should be within price range
        for i in 1..result.len() {
            assert!(result[i] >= low[i] - 10.0); // Allow some flexibility
            assert!(result[i] <= high[i] + 10.0);
        }
    }

    #[test]
    fn test_pivot_points_single() {
        let pp = PivotPoints::new();
        let levels = pp.calculate_single(110.0, 100.0, 105.0);

        // PP = (110 + 100 + 105) / 3 = 105
        assert!((levels[0] - 105.0).abs() < 1e-10);

        // R1 = 2*PP - low = 2*105 - 100 = 110
        assert!((levels[1] - 110.0).abs() < 1e-10);

        // S1 = 2*PP - high = 2*105 - 110 = 100
        assert!((levels[4] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_pivot_points_series() {
        let high = arr1(&[110.0, 115.0, 120.0]);
        let low = arr1(&[105.0, 110.0, 115.0]);
        let close = arr1(&[108.0, 112.0, 118.0]);

        let pp = PivotPoints::new();
        let result = pp
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        // Should have 6 secondary outputs (R1, R2, R3, S1, S2, S3)
        assert_eq!(result.secondary.len(), 6);

        // First value should be NaN (no previous period)
        assert!(result.primary[0].is_nan());

        // Second value should be calculated from first period
        assert!(!result.primary[1].is_nan());
    }

    #[test]
    fn test_fibonacci_retracement() {
        let fib = FibonacciRetracement::new();
        let levels = fib.calculate_levels(100.0, 50.0);

        // 0% level = high
        assert!((levels[0] - 100.0).abs() < 1e-10);

        // 100% level = low
        assert!((levels[6] - 50.0).abs() < 1e-10);

        // 50% level = midpoint
        assert!((levels[3] - 75.0).abs() < 1e-10);

        // 61.8% level
        let expected_618 = 100.0 - 0.618 * (100.0 - 50.0);
        assert!((levels[4] - expected_618).abs() < 1e-10);
    }
}
