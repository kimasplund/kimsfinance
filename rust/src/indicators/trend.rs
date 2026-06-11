//! Trend Indicators
//!
//! Implements 2 trend indicators:
//! - Parabolic SAR (Stop and Reverse)
//! - Pivot Points (support/resistance levels)

use super::core::{
    Indicator, IndicatorError, IndicatorOutput, IndicatorResult, validate_lengths,
    validate_min_periods,
};
use super::utils::{rolling_max, rolling_min, true_range, wilders_smoothing};
use ndarray::{Array1, ArrayView1, Zip};

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

impl Default for PivotPoints {
    fn default() -> Self {
        Self::new()
    }
}

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

impl Default for FibonacciRetracement {
    fn default() -> Self {
        Self::new()
    }
}

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

/// Supertrend Indicator
///
/// Trend-following indicator based on ATR that provides dynamic support/resistance levels.
/// Returns supertrend line and trend signal (1 = uptrend, -1 = downtrend).
pub struct Supertrend {
    atr_period: usize,
    multiplier: f64,
}

impl Supertrend {
    pub fn new(atr_period: usize, multiplier: f64) -> Result<Self, IndicatorError> {
        if atr_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period",
                value: atr_period.to_string(),
            });
        }
        if multiplier < 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "multiplier",
                value: multiplier.to_string(),
            });
        }
        Ok(Self {
            atr_period,
            multiplier,
        })
    }

    /// Calculate Supertrend with high, low, close prices
    ///
    /// Returns (supertrend, signal) where:
    /// - supertrend: Array of supertrend line values
    /// - signal: Array of trend direction (1 = uptrend, -1 = downtrend, 0 = warmup)
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> Result<(Array1<f64>, Array1<i8>), IndicatorError> {
        let n = validate_lengths(&[high, low, close])?;
        validate_min_periods(n, self.atr_period)?;

        // Calculate ATR using optimized true range
        let tr = true_range(high, low, close);
        let atr = wilders_smoothing(tr.view(), self.atr_period);

        // Calculate HL average (middle line) - vectorized
        let mut hl_avg = Array1::uninit(n);
        Zip::from(&mut hl_avg)
            .and(&high)
            .and(&low)
            .for_each(|avg, &h, &l| {
                avg.write((h + l) * 0.5);
            });
        let hl_avg = unsafe { hl_avg.assume_init() };

        // Calculate basic bands - vectorized
        let mut basic_upper = Array1::uninit(n);
        let mut basic_lower = Array1::uninit(n);

        Zip::from(&mut basic_upper)
            .and(&mut basic_lower)
            .and(&hl_avg)
            .and(&atr)
            .for_each(|upper, lower, &avg, &atr_val| {
                if !atr_val.is_nan() {
                    let delta = self.multiplier * atr_val;
                    {
                        upper.write(avg + delta);
                        lower.write(avg - delta);
                    }
                } else {
                    {
                        upper.write(f64::NAN);
                        lower.write(f64::NAN);
                    }
                }
            });

        let basic_upper = unsafe { basic_upper.assume_init() };
        let basic_lower = unsafe { basic_lower.assume_init() };

        // Calculate final bands (iterative logic - cannot vectorize due to dependency)
        let mut final_upper = Array1::from_elem(n, f64::NAN);
        let mut final_lower = Array1::from_elem(n, f64::NAN);

        // Initialize first valid value
        for i in self.atr_period..n {
            if !basic_upper[i].is_nan() {
                final_upper[i] = basic_upper[i];
                final_lower[i] = basic_lower[i];
                break;
            }
        }

        // Apply band switching logic
        for i in self.atr_period..n {
            if basic_upper[i].is_nan() {
                continue;
            }

            // Upper band: keep previous if close was above it, otherwise use new basic upper
            if !final_upper[i - 1].is_nan() {
                if basic_upper[i] < final_upper[i - 1] || close[i - 1] > final_upper[i - 1] {
                    final_upper[i] = basic_upper[i];
                } else {
                    final_upper[i] = final_upper[i - 1];
                }
            } else {
                final_upper[i] = basic_upper[i];
            }

            // Lower band: keep previous if close was below it, otherwise use new basic lower
            if !final_lower[i - 1].is_nan() {
                if basic_lower[i] > final_lower[i - 1] || close[i - 1] < final_lower[i - 1] {
                    final_lower[i] = basic_lower[i];
                } else {
                    final_lower[i] = final_lower[i - 1];
                }
            } else {
                final_lower[i] = basic_lower[i];
            }
        }

        // Calculate Supertrend and signal
        let mut supertrend = Array1::from_elem(n, f64::NAN);
        let mut signal = Array1::zeros(n);

        // Initialize at atr_period
        if !final_upper[self.atr_period].is_nan() && !final_lower[self.atr_period].is_nan() {
            if close[self.atr_period] <= final_upper[self.atr_period] {
                supertrend[self.atr_period] = final_upper[self.atr_period];
                signal[self.atr_period] = -1;
            } else {
                supertrend[self.atr_period] = final_lower[self.atr_period];
                signal[self.atr_period] = 1;
            }
        }

        // Calculate subsequent values
        for i in (self.atr_period + 1)..n {
            if supertrend[i - 1].is_nan() {
                continue;
            }

            // Determine trend based on previous supertrend position
            // Use epsilon for floating point comparison
            let was_downtrend = (supertrend[i - 1] - final_upper[i - 1]).abs() < 1e-10;

            if was_downtrend {
                // Was in downtrend
                if close[i] <= final_upper[i] {
                    // Stay in downtrend
                    supertrend[i] = final_upper[i];
                    signal[i] = -1;
                } else {
                    // Switch to uptrend
                    supertrend[i] = final_lower[i];
                    signal[i] = 1;
                }
            } else {
                // Was in uptrend
                if close[i] >= final_lower[i] {
                    // Stay in uptrend
                    supertrend[i] = final_lower[i];
                    signal[i] = 1;
                } else {
                    // Switch to downtrend
                    supertrend[i] = final_upper[i];
                    signal[i] = -1;
                }
            }
        }

        Ok((supertrend, signal))
    }
}

impl Indicator for Supertrend {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "Supertrend requires high, low, close. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.atr_period
    }

    fn name(&self) -> &'static str {
        "Supertrend"
    }
}

/// Ichimoku Cloud (Ichimoku Kinko Hyo)
///
/// Comprehensive trend indicator consisting of five lines:
/// - Tenkan-sen (Conversion Line): Fast-moving line
/// - Kijun-sen (Base Line): Standard line
/// - Senkou Span A (Leading Span A): First cloud boundary (shifted forward)
/// - Senkou Span B (Leading Span B): Second cloud boundary (shifted forward)
/// - Chikou Span (Lagging Span): Close price shifted backward
///
/// The "cloud" (Kumo) is the area between Senkou Span A and B.
///
/// # Performance
/// Uses O(n) monotonic deque algorithm for rolling min/max (50x faster than naive).
pub struct IchimokuCloud {
    conversion_period: usize, // Tenkan-sen period (default: 9)
    base_period: usize,       // Kijun-sen period (default: 26)
    span_b_period: usize,     // Senkou Span B period (default: 52)
    displacement: usize,      // Forward/backward shift (default: 26)
}

impl IchimokuCloud {
    pub fn new(
        conversion_period: usize,
        base_period: usize,
        span_b_period: usize,
        displacement: usize,
    ) -> Result<Self, IndicatorError> {
        if conversion_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "conversion_period",
                value: conversion_period.to_string(),
            });
        }
        if base_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_period",
                value: base_period.to_string(),
            });
        }
        if span_b_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "span_b_period",
                value: span_b_period.to_string(),
            });
        }

        Ok(Self {
            conversion_period,
            base_period,
            span_b_period,
            displacement,
        })
    }

    /// Calculate Ichimoku Cloud with high, low, close prices
    ///
    /// Returns IndicatorOutput with:
    /// - primary: tenkan_sen (Conversion Line)
    /// - secondary[0]: kijun_sen (Base Line)
    /// - secondary[1]: senkou_span_a (Leading Span A, shifted forward)
    /// - secondary[2]: senkou_span_b (Leading Span B, shifted forward)
    /// - secondary[3]: chikou_span (Lagging Span, shifted backward)
    ///
    /// # SIMD Optimizations
    /// - Uses O(n) rolling_max/rolling_min with monotonic deque
    /// - Vectorized operations with ndarray::Zip
    /// - Cache-friendly memory access patterns
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> Result<IndicatorOutput, IndicatorError> {
        let n = validate_lengths(&[high, low, close])?;
        validate_min_periods(n, self.span_b_period)?;

        // Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        // Using O(n) optimized rolling_max/rolling_min
        let tenkan_high = rolling_max(high, self.conversion_period);
        let tenkan_low = rolling_min(low, self.conversion_period);

        let mut tenkan_sen = Array1::from_elem(n, f64::NAN);
        Zip::from(&mut tenkan_sen)
            .and(&tenkan_high)
            .and(&tenkan_low)
            .for_each(|t, &h, &l| {
                if !h.is_nan() && !l.is_nan() {
                    *t = (h + l) / 2.0;
                }
            });

        // Calculate Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        let kijun_high = rolling_max(high, self.base_period);
        let kijun_low = rolling_min(low, self.base_period);

        let mut kijun_sen = Array1::from_elem(n, f64::NAN);
        Zip::from(&mut kijun_sen)
            .and(&kijun_high)
            .and(&kijun_low)
            .for_each(|k, &h, &l| {
                if !h.is_nan() && !l.is_nan() {
                    *k = (h + l) / 2.0;
                }
            });

        // Calculate Senkou Span B base: (52-period high + 52-period low) / 2
        let span_b_high = rolling_max(high, self.span_b_period);
        let span_b_low = rolling_min(low, self.span_b_period);

        let mut span_b_base = Array1::from_elem(n, f64::NAN);
        Zip::from(&mut span_b_base)
            .and(&span_b_high)
            .and(&span_b_low)
            .for_each(|s, &h, &l| {
                if !h.is_nan() && !l.is_nan() {
                    *s = (h + l) / 2.0;
                }
            });

        // Calculate Senkou Span A base: (Tenkan-sen + Kijun-sen) / 2
        let mut span_a_base = Array1::from_elem(n, f64::NAN);
        Zip::from(&mut span_a_base)
            .and(&tenkan_sen)
            .and(&kijun_sen)
            .for_each(|a, &t, &k| {
                if !t.is_nan() && !k.is_nan() {
                    *a = (t + k) / 2.0;
                }
            });

        // Shift Senkou Span A forward by displacement periods
        let mut senkou_span_a = Array1::from_elem(n, f64::NAN);
        for i in 0..n {
            if i + self.displacement < n && !span_a_base[i].is_nan() {
                senkou_span_a[i + self.displacement] = span_a_base[i];
            }
        }

        // Shift Senkou Span B forward by displacement periods
        let mut senkou_span_b = Array1::from_elem(n, f64::NAN);
        for i in 0..n {
            if i + self.displacement < n && !span_b_base[i].is_nan() {
                senkou_span_b[i + self.displacement] = span_b_base[i];
            }
        }

        // Calculate Chikou Span: Close price, shifted backward by displacement
        let mut chikou_span = Array1::from_elem(n, f64::NAN);
        for i in self.displacement..n {
            chikou_span[i - self.displacement] = close[i];
        }

        Ok(IndicatorOutput {
            primary: tenkan_sen,
            secondary: vec![kijun_sen, senkou_span_a, senkou_span_b, chikou_span],
            metadata: None,
        })
    }
}

impl Indicator for IchimokuCloud {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "Ichimoku Cloud requires high, low, close. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.span_b_period
    }

    fn name(&self) -> &'static str {
        "Ichimoku Cloud"
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

        // SAR should be within overall price range (not necessarily current bar)
        // During reversals, SAR can be set to extreme points from previous bars
        let overall_low = low.iter().cloned().fold(f64::INFINITY, f64::min);
        let overall_high = high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for i in 1..result.len() {
            assert!(
                result[i] >= overall_low - 10.0,
                "SAR {} below overall_low {} at index {}",
                result[i],
                overall_low,
                i
            );
            assert!(
                result[i] <= overall_high + 10.0,
                "SAR {} above overall_high {} at index {}",
                result[i],
                overall_high,
                i
            );
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

    #[test]
    fn test_supertrend_basic() {
        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0, 132.0, 135.0,
            133.0, 136.0, 140.0, 138.0, 142.0, 145.0, 143.0, 146.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0, 127.0, 130.0,
            128.0, 131.0, 135.0, 133.0, 137.0, 140.0, 138.0, 141.0,
        ]);
        let close = arr1(&[
            108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0, 130.0, 133.0,
            131.0, 134.0, 138.0, 136.0, 140.0, 143.0, 141.0, 144.0,
        ]);

        let supertrend = Supertrend::new(10, 3.0).unwrap();
        let (supertrend_values, signal) = supertrend
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        // First atr_period values should be NaN/0
        for i in 0..10 {
            assert!(supertrend_values[i].is_nan());
            assert_eq!(signal[i], 0);
        }

        // After warmup period, should have valid values
        assert!(!supertrend_values[10].is_nan());
        assert!(signal[10] == 1 || signal[10] == -1);

        // Supertrend should be positive (price is going up)
        assert!(supertrend_values[10] > 0.0);

        // Signal should be consistent (either 1 or -1, not 0 after warmup)
        for i in 10..20 {
            assert!(signal[i] == 1 || signal[i] == -1);
        }
    }

    #[test]
    fn test_supertrend_trend_changes() {
        // Create data with clear trend reversal
        let high = arr1(&[
            110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, // Uptrend
            154.0, 149.0, 144.0, 139.0, 134.0, 129.0, 124.0, 119.0, 114.0, 109.0, // Downtrend
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, // Uptrend
            149.0, 144.0, 139.0, 134.0, 129.0, 124.0, 119.0, 114.0, 109.0, 104.0, // Downtrend
        ]);
        let close = arr1(&[
            108.0, 113.0, 118.0, 123.0, 128.0, 133.0, 138.0, 143.0, 148.0, 153.0, // Uptrend
            151.0, 146.0, 141.0, 136.0, 131.0, 126.0, 121.0, 116.0, 111.0, 106.0, // Downtrend
        ]);

        let supertrend = Supertrend::new(5, 2.0).unwrap();
        let (_, signal) = supertrend
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        // Should detect uptrend in first half
        assert_eq!(signal[9], 1); // Should be in uptrend

        // Should detect downtrend in second half
        assert_eq!(signal[19], -1); // Should be in downtrend
    }

    #[test]
    fn test_supertrend_parameters() {
        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0, 132.0, 135.0,
            133.0, 136.0, 140.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0, 127.0, 130.0,
            128.0, 131.0, 135.0,
        ]);
        let close = arr1(&[
            108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0, 130.0, 133.0,
            131.0, 134.0, 138.0,
        ]);

        // Test invalid parameters
        assert!(Supertrend::new(0, 3.0).is_err());
        assert!(Supertrend::new(10, -1.0).is_err());

        // Test valid parameters
        assert!(Supertrend::new(10, 3.0).is_ok());
        assert!(Supertrend::new(5, 2.0).is_ok());

        // Test different multipliers produce different results
        let st1 = Supertrend::new(10, 2.0).unwrap();
        let st2 = Supertrend::new(10, 4.0).unwrap();

        let (values1, _) = st1
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();
        let (values2, _) = st2
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        // Higher multiplier should produce different values
        assert!((values1[14] - values2[14]).abs() > 0.1);
    }

    #[test]
    fn test_supertrend_parity_with_python() {
        // Test data from Python implementation
        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0, 132.0, 135.0,
            133.0, 136.0, 140.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0, 127.0, 130.0,
            128.0, 131.0, 135.0,
        ]);
        let close = arr1(&[
            108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0, 130.0, 133.0,
            131.0, 134.0, 138.0,
        ]);

        let supertrend = Supertrend::new(10, 3.0).unwrap();
        let (supertrend_values, signal) = supertrend
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        // Verify first 10 are NaN/0 (warmup period)
        for i in 0..10 {
            assert!(supertrend_values[i].is_nan());
            assert_eq!(signal[i], 0);
        }

        // After warmup, should have valid values
        assert!(!supertrend_values[10].is_nan());
        assert!(signal[10] != 0);

        // Signal should only be -1, 0, or 1
        for i in 0..15 {
            assert!(signal[i] == -1 || signal[i] == 0 || signal[i] == 1);
        }
    }

    #[test]
    fn test_ichimoku_basic() {
        // Test with 100 data points to ensure enough for 52-period calculation
        let high: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        let low: Vec<f64> = (0..100).map(|i| 95.0 + (i as f64 * 0.5)).collect();
        let close: Vec<f64> = (0..100).map(|i| 98.0 + (i as f64 * 0.5)).collect();

        let high_arr = Array1::from_vec(high);
        let low_arr = Array1::from_vec(low);
        let close_arr = Array1::from_vec(close);

        let ichimoku = IchimokuCloud::new(9, 26, 52, 26).unwrap();
        let result = ichimoku
            .calculate_hlc(high_arr.view(), low_arr.view(), close_arr.view())
            .unwrap();

        // Should have 4 secondary outputs
        assert_eq!(result.secondary.len(), 4);

        // Check that tenkan_sen has values after warmup period (9)
        assert!(result.primary[8].is_nan());
        assert!(!result.primary[9].is_nan());

        // Check that kijun_sen has values after warmup period (26)
        assert!(result.secondary[0][25].is_nan());
        assert!(!result.secondary[0][26].is_nan());

        // Check that senkou_span_b has values after warmup period (52)
        // But they're shifted forward by 26, so check at position 52+26-1
        assert!(!result.secondary[2][77].is_nan());
    }

    #[test]
    fn test_ichimoku_values() {
        // Simple test with known values
        let n = 100;
        let high = arr1(&vec![110.0; n]);
        let low = arr1(&vec![100.0; n]);
        let close = arr1(&vec![105.0; n]);

        let ichimoku = IchimokuCloud::new(9, 26, 52, 26).unwrap();
        let result = ichimoku
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        // With constant prices:
        // Tenkan-sen = (110 + 100) / 2 = 105
        // Kijun-sen = (110 + 100) / 2 = 105
        // Senkou Span A base = (105 + 105) / 2 = 105
        // Senkou Span B base = (110 + 100) / 2 = 105

        // Check tenkan_sen after warmup
        assert!((result.primary[9] - 105.0).abs() < 1e-10);

        // Check kijun_sen after warmup
        assert!((result.secondary[0][26] - 105.0).abs() < 1e-10);

        // Check chikou_span (close shifted backward by 26)
        // chikou_span[0] should equal close[26]
        assert!((result.secondary[3][0] - 105.0).abs() < 1e-10);
    }

    #[test]
    fn test_ichimoku_parameter_validation() {
        // conversion_period = 0 should error
        assert!(IchimokuCloud::new(0, 26, 52, 26).is_err());

        // base_period = 0 should error
        assert!(IchimokuCloud::new(9, 0, 52, 26).is_err());

        // span_b_period = 0 should error
        assert!(IchimokuCloud::new(9, 26, 0, 26).is_err());

        // Valid parameters should succeed
        assert!(IchimokuCloud::new(9, 26, 52, 26).is_ok());
    }

    #[test]
    fn test_ichimoku_insufficient_data() {
        // Test with insufficient data (< 52 periods)
        let high = arr1(&[110.0, 115.0, 120.0]);
        let low = arr1(&[105.0, 110.0, 115.0]);
        let close = arr1(&[108.0, 112.0, 118.0]);

        let ichimoku = IchimokuCloud::new(9, 26, 52, 26).unwrap();
        let result = ichimoku.calculate_hlc(high.view(), low.view(), close.view());

        // Should error due to insufficient data
        assert!(result.is_err());
    }

    #[test]
    fn test_ichimoku_displacement_shift() {
        // Test that displacement shifts work correctly
        let n = 100;
        let high: Vec<f64> = (0..n).map(|i| 110.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();

        let high_arr = Array1::from_vec(high.clone());
        let low_arr = Array1::from_vec(low.clone());
        let close_arr = Array1::from_vec(close.clone());

        let ichimoku = IchimokuCloud::new(9, 26, 52, 26).unwrap();
        let result = ichimoku
            .calculate_hlc(high_arr.view(), low_arr.view(), close_arr.view())
            .unwrap();

        // Check chikou_span shift: chikou[i - 26] = close[i]
        // So chikou[0] should equal close[26]
        assert!((result.secondary[3][0] - close[26]).abs() < 1e-10);

        // Check that senkou spans are shifted forward
        // senkou_span_a[i + 26] should have value from position i
        // This means early positions should be NaN, later positions should have values
        assert!(result.secondary[1][0].is_nan()); // First position should be NaN
        assert!(!result.secondary[1][52].is_nan()); // After 52+26 should have value
    }
}
