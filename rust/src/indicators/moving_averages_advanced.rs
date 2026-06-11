//! Advanced Moving Averages
//!
//! Implements 5 advanced adaptive and specialized moving averages:
//! - KAMA (Kaufman Adaptive Moving Average)
//! - MAMA (MESA Adaptive Moving Average)
//! - Zero Lag EMA
//! - McGinley Dynamic
//! - LSMA (Least Squares Moving Average) / Linear Regression

use super::core::{Indicator, IndicatorError, IndicatorResult, MultiResult, validate_min_periods};
use super::utils::ema;
use ndarray::{Array1, ArrayView1, s};

/// Kaufman Adaptive Moving Average (KAMA)
///
/// KAMA adapts to market noise and volatility.
/// Formula:
/// - ER = Direction / Volatility (Efficiency Ratio)
/// - SC = [ER * (FastSC - SlowSC) + SlowSC]^2 (Smoothing Constant)
/// - KAMA = KAMA[prev] + SC * (Price - KAMA[prev])
pub struct KAMA {
    period: usize,
    fast_period: usize,
    slow_period: usize,
}

impl KAMA {
    pub fn new(
        period: usize,
        fast_period: usize,
        slow_period: usize,
    ) -> Result<Self, IndicatorError> {
        if period == 0 || fast_period == 0 || slow_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: format!("{}/{}/{}", period, fast_period, slow_period),
            });
        }
        if fast_period >= slow_period {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period",
                value: "must be < slow_period".to_string(),
            });
        }
        Ok(Self {
            period,
            fast_period,
            slow_period,
        })
    }
}

impl Indicator for KAMA {
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult {
        validate_min_periods(prices.len(), self.period + 1)?;

        let n = prices.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        // Calculate smoothing constants
        let fast_sc = 2.0 / (self.fast_period as f64 + 1.0);
        let slow_sc = 2.0 / (self.slow_period as f64 + 1.0);

        // Initialize KAMA with first price
        result[self.period] = prices[self.period];

        for i in (self.period + 1)..n {
            // Direction: absolute change over period
            let direction = (prices[i] - prices[i - self.period]).abs();

            // Volatility: sum of absolute price changes
            let mut volatility = 0.0;
            for j in (i - self.period + 1)..=i {
                volatility += (prices[j] - prices[j - 1]).abs();
            }

            // Efficiency Ratio
            let er = if volatility > 0.0 {
                direction / volatility
            } else {
                0.0
            };

            // Smoothing Constant
            let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);

            // KAMA calculation
            result[i] = result[i - 1] + sc * (prices[i] - result[i - 1]);
        }

        Ok(result)
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn name(&self) -> &'static str {
        "KAMA"
    }
}

/// MESA Adaptive Moving Average (MAMA)
///
/// Developed by John Ehlers, uses Hilbert Transform to adapt to market cycles.
/// Returns both MAMA and FAMA (Following Adaptive Moving Average).
///
/// This is a simplified version using phase change detection.
pub struct MAMA {
    fast_limit: f64,
    slow_limit: f64,
}

impl MAMA {
    pub fn new(fast_limit: f64, slow_limit: f64) -> Result<Self, IndicatorError> {
        if fast_limit <= 0.0 || slow_limit <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "limit",
                value: format!("{}/{}", fast_limit, slow_limit),
            });
        }
        if fast_limit <= slow_limit {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_limit",
                value: "must be > slow_limit".to_string(),
            });
        }
        Ok(Self {
            fast_limit,
            slow_limit,
        })
    }

    /// Calculate MAMA with prices
    ///
    /// Returns MultiResult with:
    /// - "mama": MAMA values
    /// - "fama": FAMA values
    pub fn calculate_multi(&self, prices: ArrayView1<f64>) -> MultiResult {
        validate_min_periods(prices.len(), 6)?;

        let n = prices.len();
        let mut mama = Array1::from_elem(n, f64::NAN);
        let mut fama = Array1::from_elem(n, f64::NAN);

        // Initialize
        mama[5] = prices[5];
        fama[5] = prices[5];

        for i in 6..n {
            // Simplified phase detection using price momentum
            let momentum = (prices[i] - prices[i - 5]).abs();
            let avg_range = (0..5)
                .map(|j| (prices[i - j] - prices[i - j - 1]).abs())
                .sum::<f64>()
                / 5.0;

            // Adaptive alpha based on momentum
            let phase = if avg_range > 0.0 {
                (momentum / avg_range).min(1.0)
            } else {
                0.5
            };

            let alpha = self.fast_limit * phase + self.slow_limit * (1.0 - phase);
            let alpha = alpha.max(self.slow_limit).min(self.fast_limit);

            // MAMA calculation
            mama[i] = alpha * prices[i] + (1.0 - alpha) * mama[i - 1];

            // FAMA follows MAMA
            let fama_alpha = alpha * 0.5;
            fama[i] = fama_alpha * mama[i] + (1.0 - fama_alpha) * fama[i - 1];
        }

        use super::core::IndicatorOutput;
        Ok(IndicatorOutput {
            primary: mama,
            secondary: vec![fama],
            metadata: None,
        })
    }
}

impl Indicator for MAMA {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "MAMA returns multiple outputs. Use calculate_multi()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        6
    }

    fn name(&self) -> &'static str {
        "MAMA"
    }
}

// Note: MAMA has its own calculate_multi method defined above, not implementing the trait here
// as it would cause recursion. Python bindings will call the method directly.

/// Zero Lag EMA
///
/// ZLEMA removes lag from EMA by adjusting for past price movement.
/// Formula:
/// ZLEMA = EMA(Price + (Price - Price[lag]))
/// where lag = (period - 1) / 2
pub struct ZeroLagEMA {
    period: usize,
}

impl ZeroLagEMA {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }
}

impl Indicator for ZeroLagEMA {
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult {
        let lag = (self.period - 1) / 2;
        validate_min_periods(prices.len(), lag + 1)?;

        let n = prices.len();
        let mut adjusted_prices = Array1::zeros(n);

        // Adjust prices for lag
        for i in 0..n {
            if i >= lag {
                adjusted_prices[i] = prices[i] + (prices[i] - prices[i - lag]);
            } else {
                adjusted_prices[i] = prices[i];
            }
        }

        // Apply EMA to adjusted prices
        Ok(ema(adjusted_prices.view(), self.period))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn name(&self) -> &'static str {
        "ZeroLagEMA"
    }
}

/// McGinley Dynamic
///
/// Adaptive moving average that adjusts speed based on market volatility.
/// Formula:
/// MD = MD[prev] + (Price - MD[prev]) / (N * (Price / MD[prev])^4)
///
/// Automatically speeds up in trending markets, slows in ranging.
pub struct McGinleyDynamic {
    period: usize,
}

impl McGinleyDynamic {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }
}

impl Indicator for McGinleyDynamic {
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult {
        validate_min_periods(prices.len(), 2)?;

        let n = prices.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        // Initialize with first price
        result[0] = prices[0];

        for i in 1..n {
            if result[i - 1] > 0.0 {
                let ratio = prices[i] / result[i - 1];
                let n_factor = self.period as f64 * ratio.powi(4);

                result[i] = result[i - 1] + (prices[i] - result[i - 1]) / n_factor.max(1.0);
            } else {
                result[i] = prices[i];
            }
        }

        Ok(result)
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn name(&self) -> &'static str {
        "McGinleyDynamic"
    }
}

/// Least Squares Moving Average (LSMA)
///
/// Also known as Linear Regression or End Point Moving Average.
/// Fits a linear regression line and returns the endpoint value.
///
/// More responsive than SMA but smoother than EMA.
pub struct LSMA {
    period: usize,
}

impl LSMA {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Perform linear regression on price slice and return endpoint
    fn linear_regression(&self, prices: &[f64]) -> f64 {
        let n = prices.len() as f64;
        let sum_x: f64 = (0..prices.len()).map(|i| i as f64).sum();
        let sum_y: f64 = prices.iter().sum();
        let sum_xy: f64 = prices.iter().enumerate().map(|(i, &p)| i as f64 * p).sum();
        let sum_x2: f64 = (0..prices.len()).map(|i| (i as f64).powi(2)).sum();

        // Calculate slope and intercept
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Return endpoint value (x = n - 1)
        slope * (n - 1.0) + intercept
    }
}

impl Indicator for LSMA {
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult {
        validate_min_periods(prices.len(), self.period)?;

        let n = prices.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        for i in (self.period - 1)..n {
            let window = &prices.slice(s![(i - self.period + 1)..=i]).to_vec();
            result[i] = self.linear_regression(window);
        }

        Ok(result)
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "LSMA"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_kama() {
        let prices = arr1(&[
            100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0, 108.0, 112.0, 111.0, 115.0,
        ]);

        let kama = KAMA::new(5, 2, 30).unwrap();
        let result = kama.calculate(prices.view()).unwrap();

        assert_eq!(result.len(), 12);
        assert!(result[5].is_finite());
        // KAMA should track price direction
        assert!(result[11] > result[6]);
    }

    #[test]
    fn test_mama() {
        let prices = arr1(&[
            100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0, 108.0, 112.0,
        ]);

        let mama = MAMA::new(0.5, 0.05).unwrap();
        let result = mama.calculate_multi(prices.view()).unwrap();

        let mama_values = &result.primary;
        let fama_values = &result.secondary[0];

        assert_eq!(mama_values.len(), 10);
        assert_eq!(fama_values.len(), 10);

        // FAMA should follow MAMA (be closer to previous values)
        assert!(fama_values[9].is_finite());
    }

    #[test]
    fn test_zero_lag_ema() {
        let prices = arr1(&[100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0]);

        let zlema = ZeroLagEMA::new(5).unwrap();
        let result = zlema.calculate(prices.view()).unwrap();

        assert_eq!(result.len(), 8);
        // Should have values after warmup
        assert!(result[5].is_finite());
    }

    #[test]
    fn test_mcginley_dynamic() {
        let prices = arr1(&[100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0]);

        let md = McGinleyDynamic::new(5).unwrap();
        let result = md.calculate(prices.view()).unwrap();

        assert_eq!(result.len(), 8);
        // Should start at first price
        assert_eq!(result[0], 100.0);
        // Should track upward trend
        assert!(result[7] > result[0]);
    }

    #[test]
    fn test_lsma() {
        let prices = arr1(&[100.0, 102.0, 104.0, 106.0, 108.0]);

        let lsma = LSMA::new(5).unwrap();
        let result = lsma.calculate(prices.view()).unwrap();

        assert_eq!(result.len(), 5);
        // For perfectly linear data, LSMA should match the trend
        assert!((result[4] - 108.0).abs() < 0.01);
    }

    #[test]
    fn test_lsma_trend_following() {
        // Test with actual trend data
        let prices = arr1(&[100.0, 101.0, 102.5, 103.0, 105.0, 104.0, 107.0]);

        let lsma = LSMA::new(5).unwrap();
        let result = lsma.calculate(prices.view()).unwrap();

        // First 4 values should be NaN
        assert!(result[0].is_nan());
        assert!(result[3].is_nan());

        // Should have values starting at index 4
        assert!(result[4].is_finite());
        assert!(result[6].is_finite());

        // Should follow upward trend
        assert!(result[6] > result[4]);
    }
}
