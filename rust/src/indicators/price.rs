//! Price-Based Indicators
//!
//! Implements 5 simple price aggregation indicators:
//! - Typical Price: (H + L + C) / 3
//! - Median Price: (H + L) / 2
//! - Weighted Close: (H + L + 2*C) / 4
//! - Average Price: (O + H + L + C) / 4
//! - True Range: max(H-L, |H-PC|, |L-PC|)
//!
//! These are often used as building blocks for other indicators.

use super::core::{Indicator, IndicatorError, IndicatorResult, validate_lengths};
use ndarray::{Array1, ArrayView1, Zip};

/// Typical Price
///
/// Typical Price = (High + Low + Close) / 3
///
/// Often used as a proxy for "average" price during a period.
pub struct TypicalPrice;

impl Default for TypicalPrice {
    fn default() -> Self {
        Self::new()
    }
}

impl TypicalPrice {
    pub fn new() -> Self {
        Self
    }

    /// Calculate Typical Price with H, L, C
    ///
    /// Optimized with SIMD-friendly vectorized operations.
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, close])?;

        let mut result = Array1::zeros(n);

        // SIMD-optimized single-pass calculation
        Zip::from(&mut result)
            .and(&high)
            .and(&low)
            .and(&close)
            .for_each(|r, &h, &l, &c| {
                *r = (h + l + c) / 3.0;
            });

        Ok(result)
    }
}

impl Indicator for TypicalPrice {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "TypicalPrice requires H, L, C. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "TypicalPrice"
    }
}

/// Median Price
///
/// Median Price = (High + Low) / 2
///
/// Simple midpoint of the range, often used in pivot calculations.
pub struct MedianPrice;

impl Default for MedianPrice {
    fn default() -> Self {
        Self::new()
    }
}

impl MedianPrice {
    pub fn new() -> Self {
        Self
    }

    /// Calculate Median Price with H, L
    pub fn calculate_hl<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low])?;

        let mut result = Array1::zeros(n);

        Zip::from(&mut result)
            .and(&high)
            .and(&low)
            .for_each(|r, &h, &l| {
                *r = (h + l) / 2.0;
            });

        Ok(result)
    }
}

impl Indicator for MedianPrice {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "MedianPrice requires H, L. Use calculate_hl()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "MedianPrice"
    }
}

/// Weighted Close
///
/// Weighted Close = (High + Low + 2*Close) / 4
///
/// Gives more weight to the closing price.
pub struct WeightedClose;

impl Default for WeightedClose {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightedClose {
    pub fn new() -> Self {
        Self
    }

    /// Calculate Weighted Close with H, L, C
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, close])?;

        let mut result = Array1::zeros(n);

        Zip::from(&mut result)
            .and(&high)
            .and(&low)
            .and(&close)
            .for_each(|r, &h, &l, &c| {
                *r = (h + l + 2.0 * c) / 4.0;
            });

        Ok(result)
    }
}

impl Indicator for WeightedClose {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "WeightedClose requires H, L, C. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "WeightedClose"
    }
}

/// Average Price
///
/// Average Price = (Open + High + Low + Close) / 4
///
/// Simple average of all OHLC prices.
pub struct AveragePrice;

impl Default for AveragePrice {
    fn default() -> Self {
        Self::new()
    }
}

impl AveragePrice {
    pub fn new() -> Self {
        Self
    }

    /// Calculate Average Price with O, H, L, C
    pub fn calculate_ohlc<'a>(
        &self,
        open: ArrayView1<'a, f64>,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[open, high, low, close])?;

        let mut result = Array1::zeros(n);

        Zip::from(&mut result)
            .and(&open)
            .and(&high)
            .and(&low)
            .and(&close)
            .for_each(|r, &o, &h, &l, &c| {
                *r = (o + h + l + c) / 4.0;
            });

        Ok(result)
    }
}

impl Indicator for AveragePrice {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "AveragePrice requires O, H, L, C. Use calculate_ohlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "AveragePrice"
    }
}

/// True Range
///
/// TR = max(H - L, |H - PC|, |L - PC|)
/// where PC = previous close
///
/// Core component of ATR and other volatility indicators.
pub struct TrueRange;

impl Default for TrueRange {
    fn default() -> Self {
        Self::new()
    }
}

impl TrueRange {
    pub fn new() -> Self {
        Self
    }

    /// Calculate True Range with H, L, C
    ///
    /// First value is simply High - Low (no previous close available).
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, close])?;

        let mut result = Array1::zeros(n);

        // First value: H - L
        result[0] = high[0] - low[0];

        // SIMD-optimized calculation with branchless max
        for i in 1..n {
            let hl = high[i] - low[i];
            let hpc = (high[i] - close[i - 1]).abs();
            let lpc = (low[i] - close[i - 1]).abs();

            // Branchless: max of three values
            result[i] = hl.max(hpc).max(lpc);
        }

        Ok(result)
    }
}

impl Indicator for TrueRange {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "TrueRange requires H, L, C. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "TrueRange"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_typical_price() {
        let high = arr1(&[105.0, 108.0, 106.0]);
        let low = arr1(&[100.0, 103.0, 101.0]);
        let close = arr1(&[102.0, 105.0, 104.0]);

        let tp = TypicalPrice::new();
        let result = tp
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        assert_eq!(result.len(), 3);
        assert!((result[0] - 102.333).abs() < 0.01);
        assert!((result[1] - 105.333).abs() < 0.01);
        assert!((result[2] - 103.666).abs() < 0.01);
    }

    #[test]
    fn test_median_price() {
        let high = arr1(&[105.0, 108.0]);
        let low = arr1(&[100.0, 103.0]);

        let mp = MedianPrice::new();
        let result = mp.calculate_hl(high.view(), low.view()).unwrap();

        assert_eq!(result[0], 102.5);
        assert_eq!(result[1], 105.5);
    }

    #[test]
    fn test_weighted_close() {
        let high = arr1(&[105.0]);
        let low = arr1(&[100.0]);
        let close = arr1(&[104.0]);

        let wc = WeightedClose::new();
        let result = wc
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        assert_eq!(result[0], (105.0 + 100.0 + 2.0 * 104.0) / 4.0);
    }

    #[test]
    fn test_average_price() {
        let open = arr1(&[101.0]);
        let high = arr1(&[105.0]);
        let low = arr1(&[100.0]);
        let close = arr1(&[104.0]);

        let ap = AveragePrice::new();
        let result = ap
            .calculate_ohlc(open.view(), high.view(), low.view(), close.view())
            .unwrap();

        assert_eq!(result[0], (101.0 + 105.0 + 100.0 + 104.0) / 4.0);
    }

    #[test]
    fn test_true_range() {
        let high = arr1(&[105.0, 110.0, 108.0]);
        let low = arr1(&[100.0, 105.0, 103.0]);
        let close = arr1(&[102.0, 107.0, 106.0]);

        let tr = TrueRange::new();
        let result = tr
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        // First: H - L = 105 - 100 = 5
        assert_eq!(result[0], 5.0);

        // Second: max(110-105=5, |110-102|=8, |105-102|=3) = 8
        assert_eq!(result[1], 8.0);

        // Third: max(108-103=5, |108-107|=1, |103-107|=4) = 5
        assert_eq!(result[2], 5.0);
    }
}
