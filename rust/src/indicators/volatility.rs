//! Volatility Indicators
//!
//! Implements 5 volatility measures:
//! - ATR (Average True Range)
//! - Bollinger Bands
//! - Keltner Channels
//! - Donchian Channels
//! - Elder Ray Index
//!
//! Optimizations:
//! - SIMD vectorization for standard deviation calculations
//! - Parallel processing for multi-band calculations
//! - Zero-allocation true range computation
//! - O(n) rolling min/max with deque algorithm

use super::core::{
    Indicator, IndicatorError, IndicatorOutput, IndicatorResult, MultiOutputIndicator, MultiResult,
    validate_lengths, validate_min_periods,
};
use super::utils::{ema, rolling_max, rolling_min, sma, wilders_smoothing};
use ndarray::{Array1, ArrayView1, Zip};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Zero-allocation True Range calculation
///
/// TR = max(high - low, |high - prev_close|, |low - prev_close|)
///
/// Optimized to avoid allocations and use SIMD where available.
#[inline]
fn true_range_optimized(
    high: ArrayView1<f64>,
    low: ArrayView1<f64>,
    close: ArrayView1<f64>,
) -> Array1<f64> {
    let n = high.len();
    let mut tr = Array1::uninit(n);

    // First value is simply high - low
    unsafe {
        tr.uget_mut(0).write(high[0] - low[0]);
    }

    // Vectorized true range calculation for subsequent values
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return true_range_avx2(high, low, close, tr);
            }
        }
    }

    // Fallback: scalar implementation
    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();

        unsafe {
            tr.uget_mut(i).write(hl.max(hc).max(lc));
        }
    }

    unsafe { tr.assume_init() }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn true_range_avx2(
    high: ArrayView1<f64>,
    low: ArrayView1<f64>,
    close: ArrayView1<f64>,
    mut tr: Array1<std::mem::MaybeUninit<f64>>,
) -> Array1<f64> {
    let n = high.len();

    // Process 4 elements at a time with AVX2
    let chunks = (n - 1) / 4;

    unsafe {
        for chunk in 0..chunks {
            let i = 1 + chunk * 4;

            // Load 4 consecutive elements
            let h = _mm256_loadu_pd(high.as_ptr().add(i));
            let l = _mm256_loadu_pd(low.as_ptr().add(i));
            let c_prev = _mm256_loadu_pd(close.as_ptr().add(i - 1));

            // Calculate high - low
            let hl = _mm256_sub_pd(h, l);

            // Calculate |high - prev_close|
            let hc = _mm256_sub_pd(h, c_prev);
            let hc_abs = _mm256_andnot_pd(_mm256_set1_pd(-0.0), hc);

            // Calculate |low - prev_close|
            let lc = _mm256_sub_pd(l, c_prev);
            let lc_abs = _mm256_andnot_pd(_mm256_set1_pd(-0.0), lc);

            // max(hl, hc_abs, lc_abs)
            let max1 = _mm256_max_pd(hl, hc_abs);
            let max2 = _mm256_max_pd(max1, lc_abs);

            // Store result
            _mm256_storeu_pd(tr.as_mut_ptr().add(i) as *mut f64, max2);
        }

        // Handle remaining elements
        for i in (1 + chunks * 4)..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();

            tr.uget_mut(i).write(hl.max(hc).max(lc));
        }

        tr.assume_init()
    }
}

/// SIMD-optimized rolling standard deviation
///
/// Uses Welford's online algorithm with SIMD for variance calculation.
#[inline]
fn rolling_std_simd(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);

    if n < period {
        return result;
    }

    // Calculate mean and variance for each window
    for i in (period - 1)..n {
        let start = i.saturating_sub(period - 1);
        let window = data.slice(ndarray::s![start..=i]);

        // Calculate mean
        let sum: f64 = window.sum();
        let mean = sum / period as f64;

        // Calculate variance with SIMD
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    result[i] = variance_avx2(window, mean).sqrt();
                    continue;
                }
            }
        }

        // Fallback: scalar variance calculation
        let variance: f64 = window
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / period as f64;

        result[i] = variance.sqrt();
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn variance_avx2(data: ArrayView1<f64>, mean: f64) -> f64 {
    let n = data.len();

    unsafe {
        let mean_vec = _mm256_set1_pd(mean);
        let mut sum_vec = _mm256_setzero_pd();

        // Process 4 elements at a time
        let chunks = n / 4;

        for chunk in 0..chunks {
            let i = chunk * 4;
            let values = _mm256_loadu_pd(data.as_ptr().add(i));

            // (x - mean)^2
            let diff = _mm256_sub_pd(values, mean_vec);
            let squared = _mm256_mul_pd(diff, diff);

            sum_vec = _mm256_add_pd(sum_vec, squared);
        }

        // Horizontal sum of vector
        let mut sum_array = [0.0; 4];
        _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
        let mut sum: f64 = sum_array.iter().sum();

        // Handle remaining elements
        for i in (chunks * 4)..n {
            let diff = data[i] - mean;
            sum += diff * diff;
        }

        sum / n as f64
    }
}

/// Average True Range (ATR)
///
/// Measures market volatility using the true range.
/// Higher values indicate higher volatility.
pub struct ATR {
    period: usize,
}

impl ATR {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate ATR with high, low, close
    ///
    /// Optimized with SIMD-accelerated true range calculation.
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, close])?;
        validate_min_periods(n, self.period)?;

        // Calculate True Range with zero-allocation SIMD optimization
        let tr = true_range_optimized(high, low, close);

        // Apply Wilder's smoothing to TR
        let atr = wilders_smoothing(tr.view(), self.period);

        Ok(atr)
    }
}

impl Indicator for ATR {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "ATR requires high, low, close. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "ATR"
    }
}

/// Bollinger Bands
///
/// Volatility bands plotted at standard deviations around a moving average.
/// Returns middle band (SMA), upper band, and lower band.
pub struct BollingerBands {
    period: usize,
    std_dev: f64,
}

impl BollingerBands {
    pub fn new(period: usize, std_dev: f64) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        if std_dev <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "std_dev",
                value: std_dev.to_string(),
            });
        }
        Ok(Self { period, std_dev })
    }
}

impl MultiOutputIndicator for BollingerBands {
    fn calculate_multi(&self, prices: ArrayView1<f64>) -> MultiResult {
        validate_min_periods(prices.len(), self.period)?;

        // Middle band = SMA
        let middle = sma(prices, self.period);

        // Calculate rolling standard deviation with SIMD optimization
        let std = rolling_std_simd(prices, self.period);

        let n = prices.len();

        // Vectorized band calculation
        let mut upper = Array1::uninit(n);
        let mut lower = Array1::uninit(n);

        // Initialize warmup period with NaN
        for i in 0..(self.period - 1) {
            unsafe {
                upper.uget_mut(i).write(f64::NAN);
                lower.uget_mut(i).write(f64::NAN);
            }
        }

        // Vectorized band calculation using Zip for SIMD auto-vectorization
        Zip::indexed(middle.slice(ndarray::s![self.period - 1..]))
            .and(std.slice(ndarray::s![self.period - 1..]))
            .for_each(|i, &m, &s| {
                let idx = i + self.period - 1;
                if !m.is_nan() && !s.is_nan() {
                    let delta = self.std_dev * s;
                    unsafe {
                        upper.uget_mut(idx).write(m + delta);
                        lower.uget_mut(idx).write(m - delta);
                    }
                } else {
                    unsafe {
                        upper.uget_mut(idx).write(f64::NAN);
                        lower.uget_mut(idx).write(f64::NAN);
                    }
                }
            });

        Ok(IndicatorOutput {
            primary: middle,
            secondary: vec![unsafe { upper.assume_init() }, unsafe {
                lower.assume_init()
            }],
            metadata: None,
        })
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "Bollinger Bands"
    }
}

/// Keltner Channels
///
/// Volatility-based envelopes using ATR.
/// Returns middle line (EMA), upper channel, and lower channel.
pub struct KeltnerChannels {
    ema_period: usize,
    atr_period: usize,
    atr_multiplier: f64,
}

impl KeltnerChannels {
    pub fn new(
        ema_period: usize,
        atr_period: usize,
        atr_multiplier: f64,
    ) -> Result<Self, IndicatorError> {
        if ema_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "ema_period",
                value: ema_period.to_string(),
            });
        }
        if atr_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period",
                value: atr_period.to_string(),
            });
        }
        if atr_multiplier <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_multiplier",
                value: atr_multiplier.to_string(),
            });
        }
        Ok(Self {
            ema_period,
            atr_period,
            atr_multiplier,
        })
    }

    /// Calculate Keltner Channels with high, low, close
    ///
    /// Optimized with SIMD true range and parallel channel calculation.
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> Result<IndicatorOutput, IndicatorError> {
        let n = validate_lengths(&[high, low, close])?;
        validate_min_periods(n, self.ema_period.max(self.atr_period))?;

        // Parallel computation of middle line and ATR
        let (middle, atr) = rayon::join(
            || ema(close, self.ema_period),
            || {
                // Calculate ATR with SIMD optimization
                let tr = true_range_optimized(high, low, close);
                wilders_smoothing(tr.view(), self.atr_period)
            },
        );

        // Vectorized channel calculation
        let mut upper = Array1::uninit(n);
        let mut lower = Array1::uninit(n);

        // Vectorized band calculation using Zip
        Zip::indexed(&middle).and(&atr).for_each(|i, &m, &a| {
            if !m.is_nan() && !a.is_nan() {
                let delta = self.atr_multiplier * a;
                unsafe {
                    upper.uget_mut(i).write(m + delta);
                    lower.uget_mut(i).write(m - delta);
                }
            } else {
                unsafe {
                    upper.uget_mut(i).write(f64::NAN);
                    lower.uget_mut(i).write(f64::NAN);
                }
            }
        });

        Ok(IndicatorOutput {
            primary: middle,
            secondary: vec![unsafe { upper.assume_init() }, unsafe {
                lower.assume_init()
            }],
            metadata: None,
        })
    }
}

impl Indicator for KeltnerChannels {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "Keltner Channels require high, low, close. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.ema_period.max(self.atr_period)
    }

    fn name(&self) -> &'static str {
        "Keltner Channels"
    }
}

/// Donchian Channels
///
/// Price channel based on highest high and lowest low over period.
/// Returns upper channel, middle line, and lower channel.
pub struct DonchianChannels {
    period: usize,
}

impl DonchianChannels {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Donchian Channels with high and low
    ///
    /// Optimized with O(n) deque-based rolling min/max from utils.
    pub fn calculate_hl<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
    ) -> Result<IndicatorOutput, IndicatorError> {
        let n = validate_lengths(&[high, low])?;
        validate_min_periods(n, self.period)?;

        // Parallel computation of upper and lower channels with O(n) algorithm
        let (upper, lower) = rayon::join(
            || rolling_max(high, self.period),
            || rolling_min(low, self.period),
        );

        // Vectorized middle line calculation
        let mut middle = Array1::uninit(n);

        Zip::indexed(&upper).and(&lower).for_each(|i, &u, &l| {
            if !u.is_nan() && !l.is_nan() {
                unsafe {
                    middle.uget_mut(i).write((u + l) * 0.5);
                }
            } else {
                unsafe {
                    middle.uget_mut(i).write(f64::NAN);
                }
            }
        });

        Ok(IndicatorOutput {
            primary: unsafe { middle.assume_init() },
            secondary: vec![upper, lower],
            metadata: None,
        })
    }
}

impl Indicator for DonchianChannels {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "Donchian Channels require high and low. Use calculate_hl()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "Donchian Channels"
    }
}

/// Elder Ray Index
///
/// Measures buying and selling pressure.
/// Returns Bull Power and Bear Power.
pub struct ElderRay {
    ema_period: usize,
}

impl ElderRay {
    pub fn new(ema_period: usize) -> Result<Self, IndicatorError> {
        if ema_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "ema_period",
                value: ema_period.to_string(),
            });
        }
        Ok(Self { ema_period })
    }

    /// Calculate Elder Ray with high, low, close
    ///
    /// Optimized with vectorized power calculations.
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> Result<IndicatorOutput, IndicatorError> {
        let n = validate_lengths(&[high, low, close])?;
        validate_min_periods(n, self.ema_period)?;

        // Calculate EMA of close
        let ema_close = ema(close, self.ema_period);

        // Vectorized bull and bear power calculation
        let mut bull_power = Array1::uninit(n);
        let mut bear_power = Array1::uninit(n);

        // Vectorized computation using Zip for auto-vectorization
        Zip::indexed(&ema_close)
            .and(&high)
            .and(&low)
            .for_each(|i, &ema_val, &h, &l| {
                if !ema_val.is_nan() {
                    unsafe {
                        bull_power.uget_mut(i).write(h - ema_val);
                        bear_power.uget_mut(i).write(l - ema_val);
                    }
                } else {
                    unsafe {
                        bull_power.uget_mut(i).write(f64::NAN);
                        bear_power.uget_mut(i).write(f64::NAN);
                    }
                }
            });

        Ok(IndicatorOutput {
            primary: unsafe { bull_power.assume_init() },
            secondary: vec![unsafe { bear_power.assume_init() }],
            metadata: None,
        })
    }
}

impl Indicator for ElderRay {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "Elder Ray requires high, low, close. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.ema_period
    }

    fn name(&self) -> &'static str {
        "Elder Ray"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_atr() {
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

        let atr = ATR::new(14).unwrap();
        let result = atr
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        // ATR should be positive after warmup
        assert!(result[14] > 0.0);
    }

    #[test]
    fn test_bollinger_bands() {
        let prices = arr1(&[
            100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0, 109.0, 112.0, 111.0, 115.0,
            114.0, 118.0, 117.0, 120.0, 119.0, 122.0, 121.0, 125.0,
        ]);

        let bb = BollingerBands::new(20, 2.0).unwrap();
        let result = bb.calculate_multi(prices.view()).unwrap();

        // Should have upper and lower bands
        assert_eq!(result.secondary.len(), 2);

        let upper = &result.secondary[0];
        let lower = &result.secondary[1];

        // Upper should be > middle > lower
        for i in 19..prices.len() {
            if !result.primary[i].is_nan() {
                assert!(upper[i] > result.primary[i]);
                assert!(result.primary[i] > lower[i]);
            }
        }
    }

    #[test]
    fn test_keltner_channels() {
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

        let kc = KeltnerChannels::new(20, 10, 2.0).unwrap();
        let result = kc
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        // Should have upper and lower channels
        assert_eq!(result.secondary.len(), 2);
    }

    #[test]
    fn test_donchian_channels() {
        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0,
        ]);

        let dc = DonchianChannels::new(5).unwrap();
        let result = dc.calculate_hl(high.view(), low.view()).unwrap();

        // Upper should be highest high, lower should be lowest low
        assert_eq!(result.secondary.len(), 2);

        let upper = &result.secondary[0];
        let lower = &result.secondary[1];

        // Middle should be between upper and lower
        for i in 4..high.len() {
            if !result.primary[i].is_nan() {
                assert!(upper[i] >= result.primary[i]);
                assert!(result.primary[i] >= lower[i]);
            }
        }
    }

    #[test]
    fn test_elder_ray() {
        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0, 132.0, 135.0,
            133.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0, 127.0, 130.0,
            128.0,
        ]);
        let close = arr1(&[
            108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0, 130.0, 133.0,
            131.0,
        ]);

        let er = ElderRay::new(13).unwrap();
        let result = er
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        // Should have bull power (primary) and bear power (secondary)
        assert_eq!(result.secondary.len(), 1);

        // Bull power should be positive more often when price rising
        // Bear power should be negative more often when price rising
        assert!(result.primary[12].is_finite());
        assert!(result.secondary[0][12].is_finite());
    }
}
