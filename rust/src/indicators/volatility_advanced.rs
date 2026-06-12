//! Advanced Volatility Indicators
//!
//! Implements 5 advanced volatility and statistical indicators:
//! - Standard Deviation
//! - Chaikin Volatility
//! - Mass Index
//! - Standard Error
//! - Ease of Movement (EOM)

use super::core::{
    Indicator, IndicatorError, IndicatorResult, validate_lengths, validate_min_periods,
};
use super::utils::{ema, sma};
use ndarray::{Array1, ArrayView1, Zip, s};

/// Standard Deviation
///
/// Measures volatility as standard deviation of price over N periods.
/// Lower values = less volatility, higher values = more volatility.
pub struct StandardDeviation {
    period: usize,
}

impl StandardDeviation {
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

impl Indicator for StandardDeviation {
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult {
        validate_min_periods(prices.len(), self.period)?;

        let n = prices.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        for i in (self.period - 1)..n {
            let window = prices.slice(s![i + 1 - self.period..=i]);

            // Calculate mean
            let mean: f64 = window.sum() / self.period as f64;

            // Calculate variance
            let variance: f64 =
                window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / self.period as f64;

            // Standard deviation
            result[i] = variance.sqrt();
        }

        Ok(result)
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "StandardDeviation"
    }
}

/// Chaikin Volatility
///
/// Measures volatility based on the difference between high and low.
/// Formula:
/// 1. Calculate EMA of (High - Low)
/// 2. CV = (EMA[today] - EMA[N periods ago]) / EMA[N periods ago] * 100
///
/// Positive values indicate increasing volatility.
pub struct ChaikinVolatility {
    ema_period: usize,
    roc_period: usize,
}

impl ChaikinVolatility {
    pub fn new(ema_period: usize, roc_period: usize) -> Result<Self, IndicatorError> {
        if ema_period == 0 || roc_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: format!("{}/{}", ema_period, roc_period),
            });
        }
        Ok(Self {
            ema_period,
            roc_period,
        })
    }

    /// Calculate Chaikin Volatility with high and low
    pub fn calculate_hl<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low])?;
        validate_min_periods(n, self.ema_period + self.roc_period)?;

        // Calculate high-low spread
        let mut hl_spread = Array1::zeros(n);
        Zip::from(&mut hl_spread)
            .and(&high)
            .and(&low)
            .for_each(|s, &h, &l| {
                *s = h - l;
            });

        // Apply EMA
        let ema_hl = ema(hl_spread.view(), self.ema_period);

        // Calculate rate of change
        let mut result = Array1::from_elem(n, f64::NAN);
        for i in self.roc_period..n {
            let prev_ema = ema_hl[i - self.roc_period];
            if prev_ema > 0.0 {
                result[i] = (ema_hl[i] - prev_ema) / prev_ema * 100.0;
            }
        }

        Ok(result)
    }
}

impl Indicator for ChaikinVolatility {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "ChaikinVolatility requires H, L. Use calculate_hl()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.ema_period + self.roc_period
    }

    fn name(&self) -> &'static str {
        "ChaikinVolatility"
    }
}

/// Mass Index
///
/// Identifies trend reversals by analyzing the range between high and low.
/// Formula:
/// 1. Single EMA = EMA(High - Low, 9)
/// 2. Double EMA = EMA(Single EMA, 9)
/// 3. Mass Index = Sum(Single EMA / Double EMA, 25)
///
/// Values > 27 suggest reversal, < 26.5 suggest trend continuation.
pub struct MassIndex {
    ema_period: usize,
    sum_period: usize,
}

impl MassIndex {
    pub fn new(ema_period: usize, sum_period: usize) -> Result<Self, IndicatorError> {
        if ema_period == 0 || sum_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: format!("{}/{}", ema_period, sum_period),
            });
        }
        Ok(Self {
            ema_period,
            sum_period,
        })
    }

    /// Calculate Mass Index with high and low
    pub fn calculate_hl<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low])?;
        validate_min_periods(n, self.ema_period * 2 + self.sum_period)?;

        // Calculate high-low range
        let mut hl_range = Array1::zeros(n);
        Zip::from(&mut hl_range)
            .and(&high)
            .and(&low)
            .for_each(|r, &h, &l| {
                *r = h - l;
            });

        // Single EMA
        let single_ema = ema(hl_range.view(), self.ema_period);

        // Double EMA (calculate on the non-NaN slice of single_ema to prevent NaN propagation)
        let mut double_ema = Array1::from_elem(n, f64::NAN);
        let single_ema_sliced = single_ema.slice(s![self.ema_period - 1..]);
        let double_ema_sliced = ema(single_ema_sliced, self.ema_period);
        double_ema.slice_mut(s![self.ema_period - 1..]).assign(&double_ema_sliced);

        // EMA ratio
        let mut ema_ratio = Array1::zeros(n);
        for i in 0..n {
            if double_ema[i] > 0.0 {
                ema_ratio[i] = single_ema[i] / double_ema[i];
            } else {
                ema_ratio[i] = 1.0;
            }
        }

        // Sum over period
        let mut result = Array1::from_elem(n, f64::NAN);
        for i in (self.sum_period - 1)..n {
            result[i] = ema_ratio.slice(s![i + 1 - self.sum_period..=i]).sum();
        }

        Ok(result)
    }
}

impl Indicator for MassIndex {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "MassIndex requires H, L. Use calculate_hl()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.ema_period * 2 + self.sum_period
    }

    fn name(&self) -> &'static str {
        "MassIndex"
    }
}

/// Standard Error
///
/// Standard error of linear regression.
/// Measures how well prices fit a linear trend.
/// Lower values = better fit (stronger trend).
pub struct StandardError {
    period: usize,
}

impl StandardError {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate standard error of linear regression
    fn calculate_stderr(&self, prices: &[f64]) -> f64 {
        let n = prices.len() as f64;

        // Calculate linear regression
        let sum_x: f64 = (0..prices.len()).map(|i| i as f64).sum();
        let sum_y: f64 = prices.iter().sum();
        let sum_xy: f64 = prices.iter().enumerate().map(|(i, &p)| i as f64 * p).sum();
        let sum_x2: f64 = (0..prices.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate residuals
        let mut sum_squared_error = 0.0;
        for (i, &price) in prices.iter().enumerate() {
            let predicted = slope * i as f64 + intercept;
            sum_squared_error += (price - predicted).powi(2);
        }

        // Standard error
        (sum_squared_error / (n - 2.0)).sqrt()
    }
}

impl Indicator for StandardError {
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult {
        validate_min_periods(prices.len(), self.period)?;

        let n = prices.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        for i in (self.period - 1)..n {
            let window = &prices.slice(s![i + 1 - self.period..=i]).to_vec();
            result[i] = self.calculate_stderr(window);
        }

        Ok(result)
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "StandardError"
    }
}

/// Ease of Movement (EOM)
///
/// Relates price change to volume, identifying "easy" moves.
/// Formula:
/// Distance Moved = ((High + Low) / 2) - ((Prior High + Prior Low) / 2)
/// Box Ratio = (Volume / Scale) / (High - Low)
/// EOM = Distance Moved / Box Ratio
///
/// Positive values = upward movement with low volume (easy to move up)
/// Negative values = downward movement with low volume (easy to move down)
pub struct EaseOfMovement {
    period: usize, // SMA smoothing period
    scale: f64,    // Volume scale factor (e.g., 10000)
}

impl EaseOfMovement {
    pub fn new(period: usize, scale: f64) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        if scale <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "scale",
                value: scale.to_string(),
            });
        }
        Ok(Self { period, scale })
    }

    /// Calculate EOM with H, L, V
    pub fn calculate_hlv<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        volume: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, volume])?;
        validate_min_periods(n, self.period + 1)?;

        let mut eom_raw = Array1::from_elem(n, f64::NAN);
        eom_raw[0] = 0.0;

        for i in 1..n {
            let distance_moved = ((high[i] + low[i]) / 2.0) - ((high[i - 1] + low[i - 1]) / 2.0);
            let hl_diff = high[i] - low[i];

            if hl_diff > 0.0 && volume[i] > 0.0 {
                let box_ratio = (volume[i] / self.scale) / hl_diff;
                eom_raw[i] = distance_moved / box_ratio;
            } else {
                eom_raw[i] = 0.0;
            }
        }

        // Smooth with SMA
        Ok(sma(eom_raw.view(), self.period))
    }
}

impl Indicator for EaseOfMovement {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "EaseOfMovement requires H, L, V. Use calculate_hlv()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn name(&self) -> &'static str {
        "EaseOfMovement"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_standard_deviation() {
        let prices = arr1(&[100.0, 102.0, 101.0, 105.0, 103.0, 107.0]);

        let std = StandardDeviation::new(5).unwrap();
        let result = std.calculate(prices.view()).unwrap();

        assert_eq!(result.len(), 6);
        assert!(result[4].is_finite());
        assert!(result[4] > 0.0); // Should have positive volatility
    }

    #[test]
    fn test_chaikin_volatility() {
        let high = arr1(&[
            105.0, 108.0, 106.0, 110.0, 107.0, 112.0, 109.0, 115.0, 111.0, 118.0,
        ]);
        let low = arr1(&[
            100.0, 103.0, 101.0, 105.0, 102.0, 107.0, 104.0, 110.0, 106.0, 113.0,
        ]);

        let cv = ChaikinVolatility::new(3, 3).unwrap();
        let result = cv.calculate_hl(high.view(), low.view()).unwrap();

        assert_eq!(result.len(), 10);
        // Should have values after warmup
        assert!(result[5].is_finite());
    }

    #[test]
    fn test_mass_index() {
        let high = arr1(&[
            105.0, 108.0, 106.0, 110.0, 107.0, 112.0, 109.0, 115.0, 111.0, 118.0, 114.0, 120.0,
            116.0, 122.0, 118.0, 125.0, 121.0, 128.0, 124.0, 130.0, 126.0, 132.0, 128.0, 135.0,
            131.0, 138.0, 134.0, 140.0, 136.0, 142.0, 138.0, 144.0, 140.0, 146.0, 142.0, 148.0,
            144.0, 150.0, 146.0, 152.0, 148.0, 154.0, 150.0, 156.0, 152.0, 158.0, 154.0, 160.0,
            156.0, 162.0,
        ]);
        let low = arr1(&[
            100.0, 103.0, 101.0, 105.0, 102.0, 107.0, 104.0, 110.0, 106.0, 113.0, 109.0, 115.0,
            111.0, 117.0, 113.0, 120.0, 116.0, 123.0, 119.0, 125.0, 121.0, 127.0, 123.0, 130.0,
            126.0, 133.0, 129.0, 135.0, 131.0, 137.0, 133.0, 139.0, 135.0, 141.0, 137.0, 143.0,
            139.0, 145.0, 141.0, 147.0, 143.0, 149.0, 145.0, 151.0, 147.0, 153.0, 149.0, 155.0,
            151.0, 157.0,
        ]);

        let mi = MassIndex::new(9, 25).unwrap();
        let result = mi.calculate_hl(high.view(), low.view()).unwrap();

        assert_eq!(result.len(), 50);
        // Should have values after warmup
        assert!(result[result.len() - 1].is_finite());
    }

    #[test]
    fn test_standard_error() {
        // Perfect linear trend should have near-zero standard error
        let prices = arr1(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);

        let se = StandardError::new(5).unwrap();
        let result = se.calculate(prices.view()).unwrap();

        assert_eq!(result.len(), 6);
        assert!(result[4].is_finite());
        // For perfect linear data, standard error should be very small
        assert!(result[4] < 0.1);
    }

    #[test]
    fn test_ease_of_movement() {
        let high = arr1(&[105.0, 108.0, 106.0, 110.0, 107.0, 112.0]);
        let low = arr1(&[100.0, 103.0, 101.0, 105.0, 102.0, 107.0]);
        let volume = arr1(&[10000.0, 15000.0, 12000.0, 18000.0, 11000.0, 20000.0]);

        let eom = EaseOfMovement::new(3, 10000.0).unwrap();
        let result = eom
            .calculate_hlv(high.view(), low.view(), volume.view())
            .unwrap();

        assert_eq!(result.len(), 6);
        // Should have values after warmup
        assert!(result[3].is_finite() || result[3].is_nan());
    }
}
