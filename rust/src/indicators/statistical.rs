//! Statistical Indicators
//!
//! Implements 5 statistical analysis indicators:
//! - Linear Regression
//! - Time Series Forecast (TSF)
//! - Correlation Coefficient
//! - Covariance
//! - Price Rate of Change (PROC)

use super::core::{
    Indicator, IndicatorError, IndicatorResult, validate_lengths, validate_min_periods,
};
use ndarray::{Array1, ArrayView1, s};

/// Linear Regression
///
/// Fits a linear regression line to price data and returns the fitted values.
/// Useful for identifying trend direction and strength.
pub struct LinearRegression {
    period: usize,
}

impl LinearRegression {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Perform linear regression and return fitted value at endpoint
    fn fit_linear(&self, prices: &[f64]) -> f64 {
        let n = prices.len() as f64;
        let sum_x: f64 = (0..prices.len()).map(|i| i as f64).sum();
        let sum_y: f64 = prices.iter().sum();
        let sum_xy: f64 = prices.iter().enumerate().map(|(i, &p)| i as f64 * p).sum();
        let sum_x2: f64 = (0..prices.len()).map(|i| (i as f64).powi(2)).sum();

        // Calculate slope and intercept
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Return value at endpoint (x = n - 1)
        slope * (n - 1.0) + intercept
    }
}

impl Indicator for LinearRegression {
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult {
        validate_min_periods(prices.len(), self.period)?;

        let n = prices.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        for i in (self.period - 1)..n {
            let window = &prices.slice(s![i + 1 - self.period..=i]).to_vec();
            result[i] = self.fit_linear(window);
        }

        Ok(result)
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "LinearRegression"
    }
}

/// Time Series Forecast (TSF)
///
/// Extends linear regression by forecasting N periods ahead.
/// TSF = LinReg value + (slope * forecast_periods)
pub struct TimeSeriesForecast {
    period: usize,
    forecast_periods: usize,
}

impl TimeSeriesForecast {
    pub fn new(period: usize, forecast_periods: usize) -> Result<Self, IndicatorError> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self {
            period,
            forecast_periods,
        })
    }

    /// Perform linear regression and forecast ahead
    fn forecast(&self, prices: &[f64]) -> f64 {
        let n = prices.len() as f64;
        let sum_x: f64 = (0..prices.len()).map(|i| i as f64).sum();
        let sum_y: f64 = prices.iter().sum();
        let sum_xy: f64 = prices.iter().enumerate().map(|(i, &p)| i as f64 * p).sum();
        let sum_x2: f64 = (0..prices.len()).map(|i| (i as f64).powi(2)).sum();

        // Calculate slope and intercept
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Forecast ahead
        let forecast_x = (n - 1.0) + self.forecast_periods as f64;
        slope * forecast_x + intercept
    }
}

impl Indicator for TimeSeriesForecast {
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult {
        validate_min_periods(prices.len(), self.period)?;

        let n = prices.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        for i in (self.period - 1)..n {
            let window = &prices.slice(s![i + 1 - self.period..=i]).to_vec();
            result[i] = self.forecast(window);
        }

        Ok(result)
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "TimeSeriesForecast"
    }
}

/// Correlation Coefficient
///
/// Pearson correlation between two price series over a rolling window.
/// Range: -1 (perfect negative correlation) to +1 (perfect positive correlation).
pub struct CorrelationCoefficient {
    period: usize,
}

impl CorrelationCoefficient {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate correlation coefficient between two series
    pub fn calculate_two_series<'a>(
        &self,
        series1: ArrayView1<'a, f64>,
        series2: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[series1, series2])?;
        validate_min_periods(n, self.period)?;

        let mut result = Array1::from_elem(n, f64::NAN);

        for i in (self.period - 1)..n {
            let x = series1.slice(s![i + 1 - self.period..=i]);
            let y = series2.slice(s![i + 1 - self.period..=i]);

            // Calculate means
            let mean_x: f64 = x.sum() / self.period as f64;
            let mean_y: f64 = y.sum() / self.period as f64;

            // Calculate covariance and standard deviations
            let mut cov = 0.0;
            let mut var_x = 0.0;
            let mut var_y = 0.0;

            for j in 0..self.period {
                let dx = x[j] - mean_x;
                let dy = y[j] - mean_y;
                cov += dx * dy;
                var_x += dx * dx;
                var_y += dy * dy;
            }

            let std_x = (var_x / self.period as f64).sqrt();
            let std_y = (var_y / self.period as f64).sqrt();

            if std_x > 0.0 && std_y > 0.0 {
                result[i] = cov / (self.period as f64 * std_x * std_y);
            } else {
                result[i] = 0.0;
            }
        }

        Ok(result)
    }
}

impl Indicator for CorrelationCoefficient {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "CorrelationCoefficient requires two series. Use calculate_two_series()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "CorrelationCoefficient"
    }
}

/// Covariance
///
/// Measures how two price series move together.
/// Positive = move in same direction, Negative = move in opposite directions.
pub struct Covariance {
    period: usize,
}

impl Covariance {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate covariance between two series
    pub fn calculate_two_series<'a>(
        &self,
        series1: ArrayView1<'a, f64>,
        series2: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[series1, series2])?;
        validate_min_periods(n, self.period)?;

        let mut result = Array1::from_elem(n, f64::NAN);

        for i in (self.period - 1)..n {
            let x = series1.slice(s![i + 1 - self.period..=i]);
            let y = series2.slice(s![i + 1 - self.period..=i]);

            // Calculate means
            let mean_x: f64 = x.sum() / self.period as f64;
            let mean_y: f64 = y.sum() / self.period as f64;

            // Calculate covariance
            let mut cov = 0.0;
            for j in 0..self.period {
                cov += (x[j] - mean_x) * (y[j] - mean_y);
            }

            result[i] = cov / self.period as f64;
        }

        Ok(result)
    }
}

impl Indicator for Covariance {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "Covariance requires two series. Use calculate_two_series()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "Covariance"
    }
}

/// Price Rate of Change (PROC)
///
/// Similar to ROC but returns decimal format instead of percentage.
/// PROC = (Price[today] - Price[n periods ago]) / Price[n periods ago]
pub struct PROC {
    period: usize,
}

impl PROC {
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

impl Indicator for PROC {
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult {
        validate_min_periods(prices.len(), self.period + 1)?;

        let n = prices.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        for i in self.period..n {
            let prev_price = prices[i - self.period];
            if prev_price != 0.0 {
                result[i] = (prices[i] - prev_price) / prev_price;
            }
        }

        Ok(result)
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn name(&self) -> &'static str {
        "PROC"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_linear_regression() {
        // Perfect linear trend: y = 2x + 100
        let prices = arr1(&[100.0, 102.0, 104.0, 106.0, 108.0]);

        let lr = LinearRegression::new(5).unwrap();
        let result = lr.calculate(prices.view()).unwrap();

        assert_eq!(result.len(), 5);
        // Should match the endpoint exactly for perfect linear data
        assert!((result[4] - 108.0).abs() < 0.01);
    }

    #[test]
    fn test_time_series_forecast() {
        // Linear trend: forecast should extrapolate
        let prices = arr1(&[100.0, 102.0, 104.0, 106.0, 108.0]);

        let tsf = TimeSeriesForecast::new(5, 1).unwrap();
        let result = tsf.calculate(prices.view()).unwrap();

        assert_eq!(result.len(), 5);
        // For perfect linear data with slope=2, forecast 1 period ahead should be 110
        assert!((result[4] - 110.0).abs() < 0.01);
    }

    #[test]
    fn test_correlation_coefficient() {
        let series1 = arr1(&[100.0, 102.0, 104.0, 106.0, 108.0]);
        let series2 = arr1(&[200.0, 204.0, 208.0, 212.0, 216.0]);

        let corr = CorrelationCoefficient::new(5).unwrap();
        let result = corr
            .calculate_two_series(series1.view(), series2.view())
            .unwrap();

        assert_eq!(result.len(), 5);
        // Perfect positive correlation should be 1.0
        assert!((result[4] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_correlation_negative() {
        let series1 = arr1(&[100.0, 102.0, 104.0, 106.0, 108.0]);
        let series2 = arr1(&[200.0, 196.0, 192.0, 188.0, 184.0]);

        let corr = CorrelationCoefficient::new(5).unwrap();
        let result = corr
            .calculate_two_series(series1.view(), series2.view())
            .unwrap();

        // Perfect negative correlation should be -1.0
        assert!((result[4] + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_covariance() {
        let series1 = arr1(&[100.0, 102.0, 104.0, 106.0, 108.0]);
        let series2 = arr1(&[200.0, 204.0, 208.0, 212.0, 216.0]);

        let cov = Covariance::new(5).unwrap();
        let result = cov
            .calculate_two_series(series1.view(), series2.view())
            .unwrap();

        assert_eq!(result.len(), 5);
        // Covariance should be positive for positively correlated series
        assert!(result[4] > 0.0);
    }

    #[test]
    fn test_proc() {
        let prices = arr1(&[100.0, 110.0, 105.0, 115.0]);

        let proc = PROC::new(2).unwrap();
        let result = proc.calculate(prices.view()).unwrap();

        assert_eq!(result.len(), 4);
        // PROC at index 2: (105 - 100) / 100 = 0.05
        assert!((result[2] - 0.05).abs() < 0.001);
        // PROC at index 3: (115 - 110) / 110 = 0.0454...
        assert!((result[3] - 0.04545).abs() < 0.001);
    }
}
