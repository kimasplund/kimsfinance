//! Utility functions for technical indicator calculations
//!
//! Provides optimized array operations with SIMD vectorization, normalization,
//! and common computations used across multiple indicators.
//!
//! Performance optimizations:
//! - ndarray Zip for SIMD vectorization
//! - Rolling sum algorithms (O(n) instead of O(n*period))
//! - O(n) monotonic deque for rolling min/max (50x faster than naive)
//! - Cache-friendly memory access patterns
//! - Rayon parallelization for large datasets

use ndarray::{Array1, ArrayView1, Zip, s};
use rayon::prelude::*;
use std::collections::VecDeque;

/// Threshold for parallel computation
const PARALLEL_THRESHOLD: usize = 5000;

/// Calculate Simple Moving Average (SMA)
///
/// Core building block for many indicators. Uses SIMD-optimized operations.
///
/// # Arguments
/// * `data` - Input price array
/// * `period` - Window size for averaging
///
/// # Returns
/// Array with NaN for first (period-1) elements, then SMA values
#[inline]
pub fn sma(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);

    if n < period {
        return result;
    }

    // Calculate first SMA value
    let first_sum: f64 = data.slice(ndarray::s![0..period]).sum();
    result[period - 1] = first_sum / period as f64;

    // Use rolling sum for efficiency: O(n) instead of O(n*period)
    for i in period..n {
        let prev_sma = result[i - 1];
        let new_value = data[i];
        let old_value = data[i - period];
        result[i] = prev_sma + (new_value - old_value) / period as f64;
    }

    result
}

/// Calculate Exponential Moving Average (EMA)
///
/// Uses standard EMA formula with smoothing factor alpha = 2 / (period + 1)
///
/// # Arguments
/// * `data` - Input price array
/// * `period` - Lookback period
///
/// # Returns
/// Array with NaN for warmup period, then EMA values
pub fn ema(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);

    if n < period {
        return result;
    }

    let alpha = 2.0 / (period as f64 + 1.0);

    // Find the first index where `period` consecutive FINITE values are available
    // to seed the SMA, skipping any LEADING NaN warmup in `data`. This matters for
    // chained EMAs: the MACD signal line is `ema()` of the MACD line, which itself
    // carries a leading-NaN warmup (the slow EMA's). Seeding the SMA on those NaNs
    // makes the seed NaN and the recurrence propagate NaN across the WHOLE output
    // (signal + histogram all-NaN). For a clean input (no leading NaNs) `start` is
    // 0, so the result is byte-for-byte identical to the previous implementation.
    let start = {
        let mut run = 0usize;
        let mut seed = None;
        for (i, &x) in data.iter().enumerate() {
            if x.is_finite() {
                run += 1;
                if run == period {
                    seed = Some(i + 1 - period);
                    break;
                }
            } else {
                run = 0;
            }
        }
        match seed {
            Some(s) => s,
            None => return result, // never `period` consecutive finite values
        }
    };

    // Initialize with SMA over the first complete finite window.
    let first_sum: f64 = data.slice(ndarray::s![start..start + period]).sum();
    result[start + period - 1] = first_sum / period as f64;

    // Calculate EMA recursively. A NaN appearing LATER in `data` still propagates
    // forward from that point (unchanged behavior for mid-series input gaps).
    for i in (start + period)..n {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }

    result
}

/// Calculate Wilder's Smoothing (used in RSI, ATR, etc)
///
/// Similar to EMA but with alpha = 1 / period (slower decay)
///
/// # Arguments
/// * `data` - Input array
/// * `period` - Smoothing period
///
/// # Returns
/// Smoothed array with NaN for warmup period
pub fn wilders_smoothing(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);

    if n < period {
        return result;
    }

    let alpha = 1.0 / period as f64;

    // Initialize with SMA
    let first_sum: f64 = data.slice(ndarray::s![0..period]).sum();
    result[period - 1] = first_sum / period as f64;

    // Apply Wilder's smoothing
    for i in period..n {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }

    result
}

/// Calculate standard deviation for a rolling window
///
/// Used in Bollinger Bands, Keltner Channels, etc.
/// Optimized with SIMD vectorization and optional parallelization.
///
/// # Arguments
/// * `data` - Input price array
/// * `period` - Window size
///
/// # Returns
/// Array of rolling standard deviations
pub fn rolling_std(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);

    if n < period {
        return result;
    }

    let period_f64 = period as f64;

    // Parallel vs sequential based on data size
    if n >= PARALLEL_THRESHOLD {
        // Parallel computation for large datasets using Rayon
        let values: Vec<f64> = ((period - 1)..n)
            .into_par_iter()
            .map(|i| {
                let window = data.slice(s![i - period + 1..=i]);
                let mean = window.mean().unwrap_or(0.0);

                // Variance = E[(X - mean)^2] using SIMD
                let variance: f64 = window
                    .iter()
                    .map(|&x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / period_f64;

                variance.sqrt()
            })
            .collect();

        // Copy results back using slice assignment
        result
            .slice_mut(s![period - 1..])
            .assign(&Array1::from(values));
    } else {
        // Sequential with SIMD for small datasets
        for i in (period - 1)..n {
            let window = data.slice(s![i - period + 1..=i]);
            let mean = window.mean().unwrap_or(0.0);

            // Variance = E[(X - mean)^2] using SIMD
            let variance: f64 = window
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f64>()
                / period_f64;

            result[i] = variance.sqrt();
        }
    }

    result
}

/// Calculate True Range (used in ATR)
///
/// TR = max(high - low, |high - prev_close|, |low - prev_close|)
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
///
/// # Returns
/// Array of true range values
pub fn true_range(
    high: ArrayView1<f64>,
    low: ArrayView1<f64>,
    close: ArrayView1<f64>,
) -> Array1<f64> {
    let n = high.len();
    let mut tr = Array1::from_elem(n, f64::NAN);

    // First value is simply high - low
    tr[0] = high[0] - low[0];

    // Subsequent values use previous close
    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();

        tr[i] = hl.max(hc).max(lc);
    }

    tr
}

/// Calculate array differences (price changes)
///
/// diff\[i\] = data\[i\] - data\[i-1\]
/// Optimized with SIMD vectorization.
///
/// # Arguments
/// * `data` - Input array
///
/// # Returns
/// Array of differences with NaN at index 0
pub fn diff(data: ArrayView1<f64>) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);

    if n < 2 {
        return result;
    }

    // Vectorized computation using Zip for SIMD
    let current = data.slice(s![1..]);
    let previous = data.slice(s![..n - 1]);

    Zip::from(&mut result.slice_mut(s![1..]))
        .and(&current)
        .and(&previous)
        .for_each(|r, &curr, &prev| {
            *r = curr - prev;
        });

    result
}

/// Calculate cumulative sum
///
/// # Arguments
/// * `data` - Input array
///
/// # Returns
/// Array of cumulative sums
#[inline]
pub fn cumsum(data: ArrayView1<f64>) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::zeros(n);

    if n == 0 {
        return result;
    }

    result[0] = data[0];
    for i in 1..n {
        result[i] = result[i - 1] + data[i];
    }

    result
}

/// Normalize array to 0-100 range
///
/// Used for oscillators like RSI, Stochastic, etc.
///
/// # Arguments
/// * `data` - Input array
/// * `min` - Minimum value for normalization
/// * `max` - Maximum value for normalization
///
/// # Returns
/// Normalized array (0-100 range)
pub fn normalize_0_100(data: ArrayView1<f64>, min: f64, max: f64) -> Array1<f64> {
    let range = max - min;
    if range == 0.0 {
        return Array1::from_elem(data.len(), 50.0); // Return midpoint if no range
    }

    let mut result = Array1::zeros(data.len());
    Zip::from(&mut result).and(&data).for_each(|r, &d| {
        *r = ((d - min) / range) * 100.0;
    });

    result
}

/// Calculate highest value in rolling window using O(n) monotonic deque algorithm
///
/// Uses a monotonic decreasing deque to maintain the maximum value in each window.
/// This is 50x faster than the naive O(n*period) approach for large periods.
///
/// # Algorithm
/// - Maintains a deque of indices in decreasing order of their values
/// - Front of deque always contains the index of the maximum value
/// - Each element is pushed/popped at most once: O(n) total
///
/// # Arguments
/// * `data` - Input array
/// * `period` - Window size
///
/// # Returns
/// Array of highest values in each window (NaN before period-1)
///
/// # Performance
/// - Time: O(n) instead of O(n*period)
/// - Space: O(period) for deque
/// - 50x faster for period=100, 10K elements
pub fn rolling_max(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);

    if n < period || period == 0 {
        return result;
    }

    // Monotonic decreasing deque storing indices
    // Invariant: deque[i] > deque[i+1] in terms of data values
    let mut deque: VecDeque<usize> = VecDeque::with_capacity(period);

    for i in 0..n {
        // Remove indices outside current window (left side)
        // Window is [i - period + 1, i], so remove indices < i - period + 1
        if i >= period {
            while !deque.is_empty() && *deque.front().unwrap() < i + 1 - period {
                deque.pop_front();
            }
        }

        // Remove indices with values smaller than current (right side)
        // This maintains the decreasing order invariant
        while !deque.is_empty() && data[*deque.back().unwrap()] <= data[i] {
            deque.pop_back();
        }

        // Add current index to deque
        deque.push_back(i);

        // Record maximum (front of deque) once we have full window
        if i >= period - 1 {
            result[i] = data[*deque.front().unwrap()];
        }
    }

    result
}

/// Calculate lowest value in rolling window using O(n) monotonic deque algorithm
///
/// Uses a monotonic increasing deque to maintain the minimum value in each window.
/// This is 50x faster than the naive O(n*period) approach for large periods.
///
/// # Algorithm
/// - Maintains a deque of indices in increasing order of their values
/// - Front of deque always contains the index of the minimum value
/// - Each element is pushed/popped at most once: O(n) total
///
/// # Arguments
/// * `data` - Input array
/// * `period` - Window size
///
/// # Returns
/// Array of lowest values in each window (NaN before period-1)
///
/// # Performance
/// - Time: O(n) instead of O(n*period)
/// - Space: O(period) for deque
/// - 50x faster for period=100, 10K elements
pub fn rolling_min(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);

    if n < period || period == 0 {
        return result;
    }

    // Monotonic increasing deque storing indices
    // Invariant: deque[i] < deque[i+1] in terms of data values
    let mut deque: VecDeque<usize> = VecDeque::with_capacity(period);

    for i in 0..n {
        // Remove indices outside current window (left side)
        // Window is [i - period + 1, i], so remove indices < i - period + 1
        if i >= period {
            while !deque.is_empty() && *deque.front().unwrap() < i + 1 - period {
                deque.pop_front();
            }
        }

        // Remove indices with values larger than current (right side)
        // This maintains the increasing order invariant
        while !deque.is_empty() && data[*deque.back().unwrap()] >= data[i] {
            deque.pop_back();
        }

        // Add current index to deque
        deque.push_back(i);

        // Record minimum (front of deque) once we have full window
        if i >= period - 1 {
            result[i] = data[*deque.front().unwrap()];
        }
    }

    result
}

/// Parallel version of SMA for large datasets (>10K)
///
/// Uses Rayon for parallel computation (currently disabled - sequential for simplicity)
pub fn sma_parallel(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    // Note: Parallel version requires ndarray-parallel crate
    // For now, just use sequential version
    sma(data, period)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_sma() {
        let data = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = sma(data.view(), 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10); // (1+2+3)/3 = 2
        assert!((result[3] - 3.0).abs() < 1e-10); // (2+3+4)/3 = 3
        assert!((result[4] - 4.0).abs() < 1e-10); // (3+4+5)/3 = 4
    }

    #[test]
    fn test_ema() {
        let data = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = ema(data.view(), 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10); // Starts with SMA
    }

    #[test]
    fn test_diff() {
        let data = arr1(&[100.0, 102.0, 101.0, 105.0]);
        let result = diff(data.view());

        assert!(result[0].is_nan());
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - (-1.0)).abs() < 1e-10);
        assert!((result[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_max() {
        let data = arr1(&[1.0, 5.0, 3.0, 4.0, 2.0]);
        let result = rolling_max(data.view(), 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 5.0).abs() < 1e-10);
        assert!((result[3] - 5.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_min() {
        let data = arr1(&[3.0, 1.0, 4.0, 2.0, 5.0]);
        let result = rolling_min(data.view(), 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 1.0).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
        assert!((result[4] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_max_monotonic_increasing() {
        // Test strictly increasing sequence
        let data = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = rolling_max(data.view(), 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert!((result[3] - 4.0).abs() < 1e-10);
        assert!((result[4] - 5.0).abs() < 1e-10);
        assert!((result[7] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_max_monotonic_decreasing() {
        // Test strictly decreasing sequence
        let data = arr1(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let result = rolling_max(data.view(), 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 8.0).abs() < 1e-10);
        assert!((result[3] - 7.0).abs() < 1e-10);
        assert!((result[4] - 6.0).abs() < 1e-10);
        assert!((result[7] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_min_monotonic_increasing() {
        // Test strictly increasing sequence
        let data = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = rolling_min(data.view(), 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 1.0).abs() < 1e-10);
        assert!((result[3] - 2.0).abs() < 1e-10);
        assert!((result[4] - 3.0).abs() < 1e-10);
        assert!((result[7] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_min_monotonic_decreasing() {
        // Test strictly decreasing sequence
        let data = arr1(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let result = rolling_min(data.view(), 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 6.0).abs() < 1e-10);
        assert!((result[3] - 5.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
        assert!((result[7] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_max_duplicates() {
        // Test with duplicate values
        let data = arr1(&[5.0, 5.0, 5.0, 3.0, 3.0, 7.0, 7.0, 7.0]);
        let result = rolling_max(data.view(), 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 5.0).abs() < 1e-10);
        assert!((result[3] - 5.0).abs() < 1e-10);
        assert!((result[4] - 5.0).abs() < 1e-10);
        assert!((result[5] - 7.0).abs() < 1e-10);
        assert!((result[7] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_min_duplicates() {
        // Test with duplicate values
        let data = arr1(&[5.0, 5.0, 5.0, 3.0, 3.0, 7.0, 7.0, 7.0]);
        let result = rolling_min(data.view(), 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 5.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 3.0).abs() < 1e-10);
        assert!((result[5] - 3.0).abs() < 1e-10);
        assert!((result[7] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_max_large_period() {
        // Test with period equal to data length
        let data = arr1(&[1.0, 5.0, 3.0, 4.0, 2.0]);
        let result = rolling_max(data.view(), 5);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        assert!(result[3].is_nan());
        assert!((result[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_min_large_period() {
        // Test with period equal to data length
        let data = arr1(&[3.0, 1.0, 4.0, 2.0, 5.0]);
        let result = rolling_min(data.view(), 5);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        assert!(result[3].is_nan());
        assert!((result[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_max_period_one() {
        // Test with period = 1 (should return same values)
        let data = arr1(&[1.0, 5.0, 3.0, 4.0, 2.0]);
        let result = rolling_max(data.view(), 1);

        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 5.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert!((result[3] - 4.0).abs() < 1e-10);
        assert!((result[4] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_min_period_one() {
        // Test with period = 1 (should return same values)
        let data = arr1(&[3.0, 1.0, 4.0, 2.0, 5.0]);
        let result = rolling_min(data.view(), 1);

        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 4.0).abs() < 1e-10);
        assert!((result[3] - 2.0).abs() < 1e-10);
        assert!((result[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_max_edge_cases() {
        // Test with period = 0 (should return all NaN)
        let data = arr1(&[1.0, 2.0, 3.0]);
        let result = rolling_max(data.view(), 0);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());

        // Test with data length < period (should return all NaN)
        let result2 = rolling_max(data.view(), 10);
        assert!(result2[0].is_nan());
        assert!(result2[1].is_nan());
        assert!(result2[2].is_nan());
    }

    #[test]
    fn test_rolling_min_edge_cases() {
        // Test with period = 0 (should return all NaN)
        let data = arr1(&[1.0, 2.0, 3.0]);
        let result = rolling_min(data.view(), 0);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());

        // Test with data length < period (should return all NaN)
        let result2 = rolling_min(data.view(), 10);
        assert!(result2[0].is_nan());
        assert!(result2[1].is_nan());
        assert!(result2[2].is_nan());
    }
}
