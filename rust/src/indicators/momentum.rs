//! Momentum Indicators
//!
//! Implements 8 momentum oscillators with SIMD and parallel processing:
//! - RSI (Relative Strength Index) - SIMD optimized gain/loss separation
//! - ROC (Rate of Change) - Parallel vectorized computation
//! - TSI (True Strength Index) - SIMD double smoothing
//! - Williams %R - Parallel rolling window operations
//! - Stochastic Oscillator - Parallel high/low finding
//! - Aroon Indicator - Optimized argmax/argmin search
//! - CCI (Commodity Channel Index) - SIMD typical price calculation
//! - MACD (Moving Average Convergence Divergence) - SIMD EMA operations
//!
//! Performance targets: 3-5x faster than NumPy for <1,000 rows

use super::core::{
    Indicator, IndicatorError, IndicatorOutput, IndicatorResult, MultiOutputIndicator, MultiResult,
    validate_lengths, validate_min_periods,
};
use super::utils::{diff, ema, rolling_max, rolling_min, sma, wilders_smoothing};
use ndarray::{Array1, ArrayView1, Zip, s};
use rayon::prelude::*;

// Threshold for parallel processing (tune based on benchmarks)
const PARALLEL_THRESHOLD: usize = 500;

/// Relative Strength Index (RSI)
///
/// Momentum oscillator measuring speed and magnitude of price changes.
/// Values range from 0-100, with >70 indicating overbought, <30 oversold.
pub struct RSI {
    period: usize,
}

impl RSI {
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

impl Indicator for RSI {
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult {
        validate_min_periods(prices.len(), self.period + 1)?;

        let n = prices.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        // Calculate price changes
        let delta = diff(prices);

        // SIMD-optimized gain/loss separation using Zip for vectorization
        let mut gains = Array1::zeros(n);
        let mut losses = Array1::zeros(n);

        // Use ndarray's Zip for SIMD-friendly operations
        Zip::from(&mut gains.slice_mut(s![1..]))
            .and(&mut losses.slice_mut(s![1..]))
            .and(&delta.slice(s![1..]))
            .for_each(|g, l, &d| {
                // Branchless: gains = max(d, 0), losses = max(-d, 0)
                *g = d.max(0.0);
                *l = (-d).max(0.0);
            });

        // Apply Wilder's smoothing
        let avg_gain = wilders_smoothing(gains.view(), self.period);
        let avg_loss = wilders_smoothing(losses.view(), self.period);

        // SIMD-optimized RSI calculation
        if n > PARALLEL_THRESHOLD {
            // Parallel computation for large datasets
            let rsi_slice: Vec<f64> = (self.period..n)
                .into_par_iter()
                .map(|i| {
                    if avg_loss[i] == 0.0 {
                        // No losses: distinguish a genuine all-gains move (RSI 100)
                        // from a FLAT series where there are also no gains. A flat
                        // window is directionless, so RSI is neutral (50) -- returning
                        // 100 here wrongly flags a flat market as maximally overbought.
                        if avg_gain[i] == 0.0 { 50.0 } else { 100.0 }
                    } else {
                        let rs = avg_gain[i] / avg_loss[i];
                        100.0 - (100.0 / (1.0 + rs))
                    }
                })
                .collect();
            result
                .slice_mut(s![self.period..])
                .assign(&Array1::from(rsi_slice));
        } else {
            // Sequential with potential auto-vectorization
            Zip::from(&mut result.slice_mut(s![self.period..]))
                .and(&avg_gain.slice(s![self.period..]))
                .and(&avg_loss.slice(s![self.period..]))
                .for_each(|r, &gain, &loss| {
                    *r = if loss == 0.0 {
                        // Flat window (no gains AND no losses) is directionless -> 50;
                        // genuine all-gains -> 100. See the parallel branch above.
                        if gain == 0.0 { 50.0 } else { 100.0 }
                    } else {
                        let rs = gain / loss;
                        100.0 - (100.0 / (1.0 + rs))
                    };
                });
        }

        Ok(result)
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn name(&self) -> &'static str {
        "RSI"
    }
}

/// Rate of Change (ROC)
///
/// Measures percentage change in price over N periods.
/// ROC = ((current - previous) / previous) * 100
pub struct ROC {
    period: usize,
}

impl ROC {
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

impl Indicator for ROC {
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult {
        validate_min_periods(prices.len(), self.period + 1)?;

        let n = prices.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        // Parallel vectorized ROC calculation for large datasets
        if n > PARALLEL_THRESHOLD {
            let roc_values: Vec<f64> = (self.period..n)
                .into_par_iter()
                .map(|i| {
                    let prev_price = prices[i - self.period];
                    if prev_price != 0.0 {
                        ((prices[i] - prev_price) / prev_price) * 100.0
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            result
                .slice_mut(s![self.period..])
                .assign(&Array1::from(roc_values));
        } else {
            // SIMD-friendly sequential computation using raw slices
            let prices_slice = prices.as_slice().unwrap();
            let result_slice = result.as_slice_mut().unwrap();

            for i in self.period..n {
                let prev_price = prices_slice[i - self.period];
                if prev_price != 0.0 {
                    let curr_price = prices_slice[i];
                    // Avoid repeated indexing - helps auto-vectorization
                    result_slice[i] = ((curr_price - prev_price) / prev_price) * 100.0;
                }
            }
        }

        Ok(result)
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn name(&self) -> &'static str {
        "ROC"
    }
}

/// Williams %R
///
/// Momentum indicator showing where current price is relative to high-low range.
/// Values range from -100 to 0, with -80 to -100 indicating oversold, -0 to -20 overbought.
pub struct WilliamsR {
    period: usize,
}

impl WilliamsR {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Williams %R with high, low, close
    /// Optimized with parallel rolling window operations
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, close])?;
        validate_min_periods(n, self.period)?;

        let highest_high = rolling_max(high, self.period);
        let lowest_low = rolling_min(low, self.period);

        let mut result = Array1::from_elem(n, f64::NAN);

        // SIMD-friendly vectorized computation
        if n > PARALLEL_THRESHOLD {
            // Parallel computation for large datasets
            let williams_values: Vec<f64> = ((self.period - 1)..n)
                .into_par_iter()
                .map(|i| {
                    let hh = highest_high[i];
                    let ll = lowest_low[i];
                    let range = hh - ll;

                    if range > 0.0 {
                        ((hh - close[i]) / range) * -100.0
                    } else {
                        // Flat window (hh==ll): %R is 0/0. Use the range midpoint
                        // (-50 on the [-100,0] scale) -- the neutral convention also
                        // used by the GPU kernel and mirroring RSI/Stochastic flat=neutral.
                        -50.0
                    }
                })
                .collect();
            result
                .slice_mut(s![(self.period - 1)..])
                .assign(&Array1::from(williams_values));
        } else {
            // Vectorized sequential computation using Zip
            Zip::from(&mut result.slice_mut(s![(self.period - 1)..]))
                .and(&highest_high.slice(s![(self.period - 1)..]))
                .and(&lowest_low.slice(s![(self.period - 1)..]))
                .and(&close.slice(s![(self.period - 1)..]))
                .for_each(|r, &hh, &ll, &c| {
                    let range = hh - ll;
                    *r = if range > 0.0 {
                        ((hh - c) / range) * -100.0
                    } else {
                        -50.0 // flat window -> neutral midpoint (see parallel branch)
                    };
                });
        }

        Ok(result)
    }
}

impl Indicator for WilliamsR {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "Williams %R requires high, low, close. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "Williams %R"
    }
}

/// Stochastic Oscillator
///
/// Compares closing price to price range over period.
/// Returns %K and %D lines (0-100 range).
pub struct Stochastic {
    k_period: usize,
    d_period: usize,
}

impl Stochastic {
    pub fn new(k_period: usize, d_period: usize) -> Result<Self, IndicatorError> {
        if k_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "k_period",
                value: k_period.to_string(),
            });
        }
        if d_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "d_period",
                value: d_period.to_string(),
            });
        }
        Ok(Self { k_period, d_period })
    }

    /// Calculate stochastic with high, low, close
    /// Optimized with SIMD vectorization for %K calculation
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> Result<IndicatorOutput, IndicatorError> {
        let n = validate_lengths(&[high, low, close])?;
        validate_min_periods(n, self.k_period)?;

        let highest_high = rolling_max(high, self.k_period);
        let lowest_low = rolling_min(low, self.k_period);

        // Calculate %K with SIMD optimization
        let mut k = Array1::from_elem(n, f64::NAN);

        if n > PARALLEL_THRESHOLD {
            // Parallel %K calculation
            let k_values: Vec<f64> = ((self.k_period - 1)..n)
                .into_par_iter()
                .map(|i| {
                    let hh = highest_high[i];
                    let ll = lowest_low[i];
                    let range = hh - ll;

                    if range > 0.0 {
                        ((close[i] - ll) / range) * 100.0
                    } else {
                        // Flat window (hh==ll): %K is 0/0. Use the range midpoint (50)
                        // -- the neutral convention used by the GPU kernel; also keeps
                        // %D = sma(%K) finite instead of NaN-poisoning the flat region.
                        50.0
                    }
                })
                .collect();
            k.slice_mut(s![(self.k_period - 1)..])
                .assign(&Array1::from(k_values));
        } else {
            // Vectorized sequential computation
            Zip::from(&mut k.slice_mut(s![(self.k_period - 1)..]))
                .and(&highest_high.slice(s![(self.k_period - 1)..]))
                .and(&lowest_low.slice(s![(self.k_period - 1)..]))
                .and(&close.slice(s![(self.k_period - 1)..]))
                .for_each(|k_val, &hh, &ll, &c| {
                    let range = hh - ll;
                    *k_val = if range > 0.0 {
                        ((c - ll) / range) * 100.0
                    } else {
                        50.0 // flat window -> neutral midpoint (see parallel branch)
                    };
                });
        }

        // Calculate %D (SMA of %K)
        let d = sma(k.view(), self.d_period);

        Ok(IndicatorOutput {
            primary: k,
            secondary: vec![d],
            metadata: None,
        })
    }
}

impl Indicator for Stochastic {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "Stochastic requires high, low, close. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.k_period + self.d_period - 1
    }

    fn name(&self) -> &'static str {
        "Stochastic"
    }
}

/// Aroon Indicator
///
/// Identifies trend changes and strength.
/// Returns Aroon Up and Aroon Down (0-100 range).
pub struct Aroon {
    period: usize,
}

impl Aroon {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Aroon with high and low prices
    /// Optimized argmax/argmin search with parallel processing
    pub fn calculate_hl<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
    ) -> Result<IndicatorOutput, IndicatorError> {
        let n = validate_lengths(&[high, low])?;
        validate_min_periods(n, self.period)?;

        let mut aroon_up = Array1::from_elem(n, f64::NAN);
        let mut aroon_down = Array1::from_elem(n, f64::NAN);

        if n > PARALLEL_THRESHOLD {
            // Parallel computation for large datasets
            let aroon_values: Vec<(f64, f64)> = ((self.period - 1)..n)
                .into_par_iter()
                .map(|i| {
                    let window_start = i - self.period + 1;

                    // Optimized argmax/argmin using iterator methods
                    let high_window = high.slice(s![window_start..=i]);
                    let low_window = low.slice(s![window_start..=i]);

                    // Most-recent occurrence of the window max/min. Mirror the
                    // sequential path's tie-break EXACTLY (`>=` for max, `<=` for
                    // min, both keeping the LATER index). The previous
                    // `rev().max_by(...)` returned the OLDEST index on tied highs
                    // (max_by yields the last element on ties, which after `.rev()`
                    // is index 0), so identical input gave different Aroon-Up
                    // depending only on whether n crossed PARALLEL_THRESHOLD.
                    let periods_since_high = high_window
                        .iter()
                        .enumerate()
                        .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                            if v >= bv { (i, v) } else { (bi, bv) }
                        })
                        .0;

                    let periods_since_low = low_window
                        .iter()
                        .enumerate()
                        .fold((0usize, f64::INFINITY), |(bi, bv), (i, &v)| {
                            if v <= bv { (i, v) } else { (bi, bv) }
                        })
                        .0;

                    // Aroon-Up = 100 * (period - bars_since_high) / period. Here
                    // periods_since_high is the window INDEX of the most-recent high
                    // (0 = oldest .. period-1 = newest), i.e. exactly (period-1 -
                    // bars_since_high), so the formula reduces to
                    // 100 * periods_since_high / (period-1). The previous
                    // `(period-1 - periods_since_high)` was inverted: a newest-bar
                    // high gave Aroon-Up 0 instead of 100.
                    let period_f = (self.period - 1) as f64;
                    let up = (periods_since_high as f64 / period_f) * 100.0;
                    let down = (periods_since_low as f64 / period_f) * 100.0;

                    (up, down)
                })
                .collect();

            for (offset, (up, down)) in aroon_values.into_iter().enumerate() {
                let i = self.period - 1 + offset;
                aroon_up[i] = up;
                aroon_down[i] = down;
            }
        } else {
            // Sequential optimized computation
            for i in (self.period - 1)..n {
                let window_start = i - self.period + 1;

                // Cache-friendly single-pass argmax/argmin
                let mut periods_since_high = 0;
                let mut periods_since_low = 0;
                let mut max_val = f64::NEG_INFINITY;
                let mut min_val = f64::INFINITY;

                // Single loop for both max and min
                for j in 0..self.period {
                    let idx = window_start + j;
                    let h_val = high[idx];
                    let l_val = low[idx];

                    if h_val >= max_val {
                        max_val = h_val;
                        periods_since_high = j;
                    }
                    if l_val <= min_val {
                        min_val = l_val;
                        periods_since_low = j;
                    }
                }

                // See the parallel branch: Aroon-Up = 100 * periods_since_high /
                // (period-1); the old `(period-1 - ...)` was inverted.
                let period_f = (self.period - 1) as f64;
                aroon_up[i] = (periods_since_high as f64 / period_f) * 100.0;
                aroon_down[i] = (periods_since_low as f64 / period_f) * 100.0;
            }
        }

        Ok(IndicatorOutput {
            primary: aroon_up,
            secondary: vec![aroon_down],
            metadata: None,
        })
    }
}

impl Indicator for Aroon {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "Aroon requires high and low. Use calculate_hl()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "Aroon"
    }
}

/// Commodity Channel Index (CCI)
///
/// Measures deviation from average price.
/// Typical values range from -100 to +100, with >+100 overbought, <-100 oversold.
pub struct CCI {
    period: usize,
}

impl CCI {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate CCI with high, low, close
    /// Optimized with SIMD typical price calculation and parallel mean deviation
    pub fn calculate_hlc<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, close])?;
        validate_min_periods(n, self.period)?;

        // SIMD-optimized Typical Price calculation: (H + L + C) / 3
        let mut tp = Array1::zeros(n);
        const ONE_THIRD: f64 = 1.0 / 3.0;

        Zip::from(&mut tp)
            .and(&high)
            .and(&low)
            .and(&close)
            .for_each(|tp_val, &h, &l, &c| {
                // Use multiplication instead of division for SIMD efficiency
                *tp_val = (h + l + c) * ONE_THIRD;
            });

        // Calculate SMA of TP
        let sma_tp = sma(tp.view(), self.period);

        let mut result = Array1::from_elem(n, f64::NAN);

        if n > PARALLEL_THRESHOLD {
            // Parallel mean deviation and CCI calculation
            let cci_values: Vec<f64> = ((self.period - 1)..n)
                .into_par_iter()
                .map(|i| {
                    let window_start = i + 1 - self.period; // Reorder to avoid underflow
                    let sma_val = sma_tp[i];

                    // Vectorized mean deviation calculation
                    let mean_dev: f64 = tp
                        .slice(s![window_start..=i])
                        .iter()
                        .map(|&tp_val| (tp_val - sma_val).abs())
                        .sum::<f64>()
                        / self.period as f64;

                    if mean_dev > 0.0 {
                        (tp[i] - sma_val) / (0.015 * mean_dev)
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            result
                .slice_mut(s![(self.period - 1)..])
                .assign(&Array1::from(cci_values));
        } else {
            // Sequential optimized computation
            let period_f = self.period as f64;
            let constant = 0.015;

            for i in (self.period - 1)..n {
                let window_start = i + 1 - self.period; // Reorder to avoid underflow
                let sma_val = sma_tp[i];

                // Cache-friendly mean deviation with unrolled accumulation
                let mut mean_dev = 0.0;
                for j in window_start..=i {
                    mean_dev += (tp[j] - sma_val).abs();
                }
                mean_dev /= period_f;

                if mean_dev > 0.0 {
                    result[i] = (tp[i] - sma_val) / (constant * mean_dev);
                }
            }
        }

        Ok(result)
    }
}

impl Indicator for CCI {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "CCI requires high, low, close. Use calculate_hlc()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "CCI"
    }
}

/// MACD (Moving Average Convergence Divergence)
///
/// Trend-following momentum indicator.
/// Returns MACD line, signal line, and histogram.
pub struct MACD {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
}

impl MACD {
    pub fn new(
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> Result<Self, IndicatorError> {
        if fast_period == 0 || slow_period == 0 || signal_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "periods",
                value: format!("fast={fast_period}, slow={slow_period}, signal={signal_period}"),
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
            signal_period,
        })
    }
}

impl MultiOutputIndicator for MACD {
    fn calculate_multi(&self, prices: ArrayView1<f64>) -> MultiResult {
        validate_min_periods(prices.len(), self.slow_period + self.signal_period)?;

        let n = prices.len();

        // Calculate EMAs (these are already optimized in utils)
        let ema_fast = ema(prices, self.fast_period);
        let ema_slow = ema(prices, self.slow_period);

        // SIMD-optimized MACD line calculation: fast EMA - slow EMA
        let mut macd_line = Array1::from_elem(n, f64::NAN);

        Zip::from(&mut macd_line)
            .and(&ema_fast)
            .and(&ema_slow)
            .for_each(|m, &fast, &slow| {
                // Branchless: NaN propagates automatically
                if !fast.is_nan() && !slow.is_nan() {
                    *m = fast - slow;
                }
            });

        // Signal line = EMA of MACD line
        let signal_line = ema(macd_line.view(), self.signal_period);

        // SIMD-optimized histogram calculation: MACD - Signal
        let mut histogram = Array1::from_elem(n, f64::NAN);

        if n > PARALLEL_THRESHOLD {
            // Parallel histogram calculation for large datasets
            let hist_values: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let macd_val = macd_line[i];
                    let signal_val = signal_line[i];
                    if !macd_val.is_nan() && !signal_val.is_nan() {
                        macd_val - signal_val
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            histogram.assign(&Array1::from(hist_values));
        } else {
            // Vectorized sequential computation
            Zip::from(&mut histogram)
                .and(&macd_line)
                .and(&signal_line)
                .for_each(|h, &macd_val, &signal_val| {
                    if !macd_val.is_nan() && !signal_val.is_nan() {
                        *h = macd_val - signal_val;
                    }
                });
        }

        Ok(IndicatorOutput {
            primary: macd_line,
            secondary: vec![signal_line, histogram],
            metadata: None,
        })
    }

    fn min_periods(&self) -> usize {
        self.slow_period + self.signal_period
    }

    fn name(&self) -> &'static str {
        "MACD"
    }
}

/// True Strength Index (TSI)
///
/// Double-smoothed momentum oscillator.
/// Values range typically from -100 to +100.
pub struct TSI {
    long_period: usize,
    short_period: usize,
    signal_period: usize,
}

impl TSI {
    pub fn new(
        long_period: usize,
        short_period: usize,
        signal_period: usize,
    ) -> Result<Self, IndicatorError> {
        if long_period == 0 || short_period == 0 || signal_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "periods",
                value: format!("long={long_period}, short={short_period}, signal={signal_period}"),
            });
        }
        Ok(Self {
            long_period,
            short_period,
            signal_period,
        })
    }
}

impl MultiOutputIndicator for TSI {
    fn calculate_multi(&self, prices: ArrayView1<f64>) -> MultiResult {
        validate_min_periods(prices.len(), self.long_period + self.short_period)?;

        let n = prices.len();

        // Price change (momentum)
        let momentum = diff(prices);

        // Double smooth momentum (EMA twice)
        let momentum_ema_long = ema(momentum.view(), self.long_period);
        let momentum_ema_short = ema(momentum_ema_long.view(), self.short_period);

        // SIMD-optimized absolute momentum calculation
        let mut abs_momentum = Array1::zeros(n);
        Zip::from(&mut abs_momentum)
            .and(&momentum)
            .for_each(|abs_m, &m| {
                *abs_m = m.abs();
            });

        // Double smooth absolute momentum
        let abs_ema_long = ema(abs_momentum.view(), self.long_period);
        let abs_ema_short = ema(abs_ema_long.view(), self.short_period);

        // SIMD-optimized TSI calculation: 100 * (momentum / abs_momentum)
        let mut tsi = Array1::from_elem(n, f64::NAN);

        if n > PARALLEL_THRESHOLD {
            // Parallel TSI calculation
            let tsi_values: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mom_val = momentum_ema_short[i];
                    let abs_val = abs_ema_short[i];

                    if abs_val != 0.0 && !mom_val.is_nan() {
                        100.0 * (mom_val / abs_val)
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            tsi.assign(&Array1::from(tsi_values));
        } else {
            // Vectorized sequential computation
            Zip::from(&mut tsi)
                .and(&momentum_ema_short)
                .and(&abs_ema_short)
                .for_each(|t, &mom_val, &abs_val| {
                    *t = if abs_val != 0.0 && !mom_val.is_nan() {
                        100.0 * (mom_val / abs_val)
                    } else {
                        f64::NAN
                    };
                });
        }

        // Signal line = EMA of TSI
        let signal = ema(tsi.view(), self.signal_period);

        Ok(IndicatorOutput {
            primary: tsi,
            secondary: vec![signal],
            metadata: None,
        })
    }

    fn min_periods(&self) -> usize {
        self.long_period + self.short_period
    }

    fn name(&self) -> &'static str {
        "TSI"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_rsi() {
        let prices = arr1(&[
            44.0, 44.5, 45.0, 44.8, 45.5, 46.0, 45.8, 46.5, 47.0, 46.8, 47.5, 48.0, 47.8, 48.5,
            49.0,
        ]);
        let rsi = RSI::new(14).unwrap();
        let result = rsi.calculate(prices.view()).unwrap();

        // RSI should produce values in 0-100 range after warmup
        assert!(result[14] >= 0.0 && result[14] <= 100.0);
    }

    #[test]
    fn test_roc() {
        let prices = arr1(&[100.0, 105.0, 110.0, 115.0, 120.0]);
        let roc = ROC::new(1).unwrap();
        let result = roc.calculate(prices.view()).unwrap();

        // ROC for index 1: (105-100)/100 * 100 = 5%
        assert!((result[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_williams_r() {
        let high = arr1(&[110.0, 115.0, 120.0, 118.0, 122.0]);
        let low = arr1(&[105.0, 110.0, 115.0, 113.0, 117.0]);
        let close = arr1(&[108.0, 112.0, 118.0, 115.0, 120.0]);

        let williams = WilliamsR::new(3).unwrap();
        let result = williams
            .calculate_hlc(high.view(), low.view(), close.view())
            .unwrap();

        // Williams %R should be between -100 and 0
        for i in 2..result.len() {
            if !result[i].is_nan() {
                assert!(result[i] >= -100.0 && result[i] <= 0.0);
            }
        }
    }

    #[test]
    fn test_macd() {
        let prices = arr1(&[
            100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0, 109.0, 112.0, 111.0, 115.0,
            114.0, 118.0, 117.0, 120.0, 119.0, 122.0, 121.0, 125.0, 124.0, 128.0, 127.0, 130.0,
            129.0, 132.0, 131.0, 135.0, 134.0, 138.0, 137.0, 140.0, 139.0, 142.0, 141.0,
            145.0, // Added 6 more points (total 36)
        ]);

        let macd = MACD::new(12, 26, 9).unwrap();
        let result = macd.calculate_multi(prices.view()).unwrap();

        // Should have MACD line, signal line, and histogram
        assert_eq!(result.secondary.len(), 2);
        // Check last value since we now have enough data
        assert!(result.primary[35].is_finite());
    }
}
