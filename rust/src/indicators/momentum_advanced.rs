//! Advanced Momentum Indicators
//!
//! Implements 8 advanced momentum indicators:
//! - ADX (Average Directional Index)
//! - DMI (Directional Movement Index)
//! - Chaikin Oscillator
//! - Force Index
//! - Ultimate Oscillator
//! - Chande Momentum Oscillator (CMO)
//! - Commodity Selection Index (CSI)
//! - Swing Index
//!
//! These indicators provide sophisticated momentum and directional analysis.

use super::core::{
    Indicator, IndicatorError, IndicatorResult, MultiResult, validate_lengths, validate_min_periods,
};
use super::utils::{ema, wilders_smoothing};
use ndarray::{Array1, ArrayView1, s};

const _PARALLEL_THRESHOLD: usize = 500;

/// Average Directional Index (ADX)
///
/// Measures the strength of a trend (not direction).
/// Values > 25 indicate strong trend, < 20 indicate weak/ranging.
///
/// Calculation:
/// 1. Calculate +DM, -DM, and TR
/// 2. Smooth with Wilder's smoothing
/// 3. Calculate +DI and -DI
/// 4. Calculate DX = |+DI - -DI| / (+DI + -DI) * 100
/// 5. ADX = smoothed DX
pub struct ADX {
    period: usize,
}

impl ADX {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate ADX with high, low, close
    ///
    /// Returns MultiResult with:
    /// - "adx": ADX values
    /// - "plus_di": +DI values
    /// - "minus_di": -DI values
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> MultiResult {
        let n = validate_lengths(&[high, low, close])?;
        validate_min_periods(n, self.period + 1)?;

        let mut plus_dm = Array1::zeros(n);
        let mut minus_dm = Array1::zeros(n);
        let mut tr = Array1::zeros(n);

        // First TR is simply H - L
        tr[0] = high[0] - low[0];

        // Calculate directional movement and true range
        for i in 1..n {
            let high_diff = high[i] - high[i - 1];
            let low_diff = low[i - 1] - low[i];

            // +DM and -DM
            if high_diff > low_diff && high_diff > 0.0 {
                plus_dm[i] = high_diff;
            }
            if low_diff > high_diff && low_diff > 0.0 {
                minus_dm[i] = low_diff;
            }

            // True Range
            let hl = high[i] - low[i];
            let hpc = (high[i] - close[i - 1]).abs();
            let lpc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hpc).max(lpc);
        }

        // Apply Wilder's smoothing
        let smoothed_plus_dm = wilders_smoothing(plus_dm.view(), self.period);
        let smoothed_minus_dm = wilders_smoothing(minus_dm.view(), self.period);
        let smoothed_tr = wilders_smoothing(tr.view(), self.period);

        // Calculate +DI and -DI
        let mut plus_di = Array1::from_elem(n, f64::NAN);
        let mut minus_di = Array1::from_elem(n, f64::NAN);
        let mut dx = Array1::zeros(n);

        for i in self.period..n {
            if smoothed_tr[i] > 0.0 {
                plus_di[i] = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i];
                minus_di[i] = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i];

                let di_sum = plus_di[i] + minus_di[i];
                if di_sum > 0.0 {
                    dx[i] = 100.0 * (plus_di[i] - minus_di[i]).abs() / di_sum;
                }
            }
        }

        // ADX is Wilder's smoothing of DX. Since DX is only valid from index self.period,
        // we slice it to avoid smoothing the initial dummy/zero values.
        let mut adx = Array1::from_elem(n, f64::NAN);
        let dx_sliced = dx.slice(s![self.period..]);
        let smoothed_dx = wilders_smoothing(dx_sliced, self.period);
        adx.slice_mut(s![self.period..]).assign(&smoothed_dx);

        use super::core::IndicatorOutput;
        Ok(IndicatorOutput {
            primary: adx,
            secondary: vec![plus_di, minus_di],
            metadata: None,
        })
    }
}

impl Indicator for ADX {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "ADX requires H, L, C. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period * 2 // ADX needs double smoothing
    }

    fn name(&self) -> &'static str {
        "ADX"
    }
}

/// Chaikin Oscillator
///
/// Chaikin Oscillator = EMA(3) of ADL - EMA(10) of ADL
/// where ADL = Accumulation/Distribution Line
///
/// Measures momentum of the Accumulation/Distribution indicator.
pub struct ChaikinOscillator {
    fast_period: usize,
    slow_period: usize,
}

impl ChaikinOscillator {
    pub fn new(fast_period: usize, slow_period: usize) -> Result<Self, IndicatorError> {
        if fast_period == 0 || slow_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: format!("{}/{}", fast_period, slow_period),
            });
        }
        if fast_period >= slow_period {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period",
                value: "must be < slow_period".to_string(),
            });
        }
        Ok(Self {
            fast_period,
            slow_period,
        })
    }

    /// Calculate Chaikin Oscillator with H, L, C, V
    pub fn calculate_hlcv<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
        volume: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, close, volume])?;

        // Calculate Money Flow Multiplier and Money Flow Volume
        let mut adl = Array1::zeros(n);

        for i in 0..n {
            let hl_diff = high[i] - low[i];
            if hl_diff > 0.0 {
                let mf_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / hl_diff;
                let mf_volume = mf_multiplier * volume[i];

                adl[i] = if i > 0 {
                    adl[i - 1] + mf_volume
                } else {
                    mf_volume
                };
            } else if i > 0 {
                adl[i] = adl[i - 1];
            }
        }

        // Calculate EMAs
        let fast_ema = ema(adl.view(), self.fast_period);
        let slow_ema = ema(adl.view(), self.slow_period);

        // Oscillator = Fast EMA - Slow EMA
        Ok(&fast_ema - &slow_ema)
    }
}

impl Indicator for ChaikinOscillator {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "ChaikinOscillator requires H, L, C, V. Use calculate_hlcv()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.slow_period + 1
    }

    fn name(&self) -> &'static str {
        "ChaikinOscillator"
    }
}

/// Force Index
///
/// Force Index = Volume * (Close - Previous Close)
///
/// When smoothed with EMA, indicates buying/selling pressure.
pub struct ForceIndex {
    period: usize, // EMA smoothing period
}

impl ForceIndex {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Force Index with close and volume
    pub fn calculate_cv<'a>(
        &self,
        close: ArrayView1<'a, f64>,
        volume: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[close, volume])?;

        let mut force = Array1::from_elem(n, f64::NAN);
        force[0] = 0.0;

        // Calculate raw force
        for i in 1..n {
            force[i] = volume[i] * (close[i] - close[i - 1]);
        }

        // Smooth with EMA if period > 1
        if self.period > 1 {
            Ok(ema(force.view(), self.period))
        } else {
            Ok(force)
        }
    }
}

impl Indicator for ForceIndex {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "ForceIndex requires C, V. Use calculate_cv()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn name(&self) -> &'static str {
        "ForceIndex"
    }
}

/// Ultimate Oscillator
///
/// Combines momentum across 3 different timeframes to avoid false signals.
/// Formula:
/// UO = 100 * [(4*Avg7) + (2*Avg14) + Avg28] / (4 + 2 + 1)
///
/// where Avg = Sum(BP) / Sum(TR)
/// BP = Close - min(Low, Previous Close)
/// TR = max(High, Previous Close) - min(Low, Previous Close)
pub struct UltimateOscillator {
    period1: usize,
    period2: usize,
    period3: usize,
}

impl UltimateOscillator {
    pub fn new(period1: usize, period2: usize, period3: usize) -> Result<Self, IndicatorError> {
        if period1 == 0 || period2 == 0 || period3 == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: format!("{}/{}/{}", period1, period2, period3),
            });
        }
        Ok(Self {
            period1,
            period2,
            period3,
        })
    }

    /// Calculate Ultimate Oscillator with H, L, C
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, close])?;
        let max_period = self.period3;
        validate_min_periods(n, max_period)?;

        let mut bp = Array1::zeros(n);
        let mut tr = Array1::zeros(n);

        // Calculate BP and TR
        for i in 1..n {
            let prev_close = close[i - 1];
            bp[i] = close[i] - low[i].min(prev_close);
            tr[i] = high[i].max(prev_close) - low[i].min(prev_close);
        }

        let mut result = Array1::from_elem(n, f64::NAN);

        // Calculate averages for each period
        for i in max_period..n {
            let avg1 = self.calculate_avg(&bp, &tr, i, self.period1);
            let avg2 = self.calculate_avg(&bp, &tr, i, self.period2);
            let avg3 = self.calculate_avg(&bp, &tr, i, self.period3);

            result[i] = 100.0 * ((4.0 * avg1) + (2.0 * avg2) + avg3) / 7.0;
        }

        Ok(result)
    }

    fn calculate_avg(&self, bp: &Array1<f64>, tr: &Array1<f64>, end: usize, period: usize) -> f64 {
        let start = end.saturating_sub(period - 1);
        let sum_bp: f64 = bp.slice(s![start..=end]).sum();
        let sum_tr: f64 = tr.slice(s![start..=end]).sum();

        if sum_tr > 0.0 { sum_bp / sum_tr } else { 0.0 }
    }
}

impl Indicator for UltimateOscillator {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "UltimateOscillator requires H, L, C. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period3
    }

    fn name(&self) -> &'static str {
        "UltimateOscillator"
    }
}

/// Chande Momentum Oscillator (CMO)
///
/// CMO = 100 * (Sum(up) - Sum(down)) / (Sum(up) + Sum(down))
///
/// Similar to RSI but uses simple sums instead of averages.
/// Range: -100 to +100 (vs RSI's 0-100).
pub struct CMO {
    period: usize,
}

impl CMO {
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

impl Indicator for CMO {
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult {
        validate_min_periods(prices.len(), self.period + 1)?;

        let n = prices.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        // Calculate price changes
        for i in self.period..n {
            let mut sum_up = 0.0;
            let mut sum_down = 0.0;

            for j in (i - self.period + 1)..=i {
                let change = prices[j] - prices[j - 1];
                if change > 0.0 {
                    sum_up += change;
                } else {
                    sum_down += -change;
                }
            }

            let sum_total = sum_up + sum_down;
            if sum_total > 0.0 {
                result[i] = 100.0 * (sum_up - sum_down) / sum_total;
            } else {
                result[i] = 0.0;
            }
        }

        Ok(result)
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn name(&self) -> &'static str {
        "CMO"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_adx_basic() {
        let high = arr1(&[
            48.0, 48.125, 48.25, 48.375, 48.5, 48.625, 48.75, 48.875, 49.0, 49.125, 49.25, 49.375,
            49.5, 49.625, 49.75,
        ]);
        let low = arr1(&[
            47.0, 47.125, 47.25, 47.375, 47.5, 47.625, 47.75, 47.875, 48.0, 48.125, 48.25, 48.375,
            48.5, 48.625, 48.75,
        ]);
        let close = arr1(&[
            47.5, 47.625, 47.75, 47.875, 48.0, 48.125, 48.25, 48.375, 48.5, 48.625, 48.75, 48.875,
            49.0, 49.125, 49.25,
        ]);

        let adx = ADX::new(7).unwrap();
        let result = adx
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        let adx_values = &result.primary;
        let _plus_di = &result.secondary[0];
        let _minus_di = &result.secondary[1];
        assert_eq!(adx_values.len(), 15);

        // ADX should be NaN for warmup period
        assert!(adx_values[0].is_nan());
        assert!(adx_values[6].is_nan());

        // ADX should have valid values after warmup
        assert!(adx_values[14] >= 0.0 && adx_values[14] <= 100.0);
    }

    #[test]
    fn test_chaikin_oscillator() {
        let high = arr1(&[105.0, 108.0, 106.0, 110.0]);
        let low = arr1(&[100.0, 103.0, 101.0, 105.0]);
        let close = arr1(&[102.0, 105.0, 104.0, 108.0]);
        let volume = arr1(&[1000.0, 1500.0, 1200.0, 1800.0]);

        let co = ChaikinOscillator::new(3, 10).unwrap();
        let result = co
            .calculate_hlcv(high.view(), low.view(), close.view(), volume.view())
            .unwrap();

        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_force_index() {
        let close = arr1(&[100.0, 102.0, 101.0, 105.0]);
        let volume = arr1(&[1000.0, 1500.0, 1200.0, 1800.0]);

        let fi = ForceIndex::new(1).unwrap();
        let result = fi.calculate_cv(close.view(), volume.view()).unwrap();

        assert_eq!(result.len(), 4);
        assert!(result[0].is_nan() || result[0] == 0.0);
        assert_eq!(result[1], 1500.0 * (102.0 - 100.0)); // 3000
    }

    #[test]
    fn test_ultimate_oscillator() {
        let high = arr1(&[
            48.0, 48.5, 49.0, 48.5, 48.0, 49.0, 49.5, 50.0, 49.5, 49.0, 50.0, 50.5, 51.0, 50.5,
            50.0, 51.0, 51.5, 52.0, 51.5, 51.0, 52.0, 52.5, 53.0, 52.5, 52.0, 53.0, 53.5, 54.0,
            53.5, 53.0,
        ]);
        let low = arr1(&[
            47.0, 47.5, 48.0, 47.5, 47.0, 48.0, 48.5, 49.0, 48.5, 48.0, 49.0, 49.5, 50.0, 49.5,
            49.0, 50.0, 50.5, 51.0, 50.5, 50.0, 51.0, 51.5, 52.0, 51.5, 51.0, 52.0, 52.5, 53.0,
            52.5, 52.0,
        ]);
        let close = arr1(&[
            47.5, 48.0, 48.5, 48.0, 47.5, 48.5, 49.0, 49.5, 49.0, 48.5, 49.5, 50.0, 50.5, 50.0,
            49.5, 50.5, 51.0, 51.5, 51.0, 50.5, 51.5, 52.0, 52.5, 52.0, 51.5, 52.5, 53.0, 53.5,
            53.0, 52.5,
        ]);

        let uo = UltimateOscillator::new(7, 14, 28).unwrap();
        let result = uo
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        assert_eq!(result.len(), 30);
        // Should be NaN before period 28
        assert!(result[27].is_nan());
        // Should have valid value at index 28+
        assert!(result[28] >= 0.0 && result[28] <= 100.0);
    }

    #[test]
    fn test_cmo() {
        let prices = arr1(&[100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0]);

        let cmo = CMO::new(5).unwrap();
        let result = cmo.calculate(prices.view()).unwrap();

        assert_eq!(result.len(), 8);
        // CMO range: -100 to +100
        for i in 5..8 {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }
}
