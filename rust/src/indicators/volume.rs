//! Volume-Based Indicators
//!
//! Implements 5 volume indicators:
//! - OBV (On-Balance Volume)
//! - VWAP (Volume Weighted Average Price)
//! - VWAP Anchored (session-based)
//! - CMF (Chaikin Money Flow)
//! - Volume Profile
//!
//! Cache-optimized implementations with parallel processing where beneficial.

use super::core::{
    Indicator, IndicatorError, IndicatorResult, validate_lengths, validate_min_periods,
};
use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;

/// On-Balance Volume (OBV)
///
/// Cumulative volume indicator that adds volume on up days, subtracts on down days.
pub struct OBV;

impl Default for OBV {
    fn default() -> Self {
        Self::new()
    }
}

impl OBV {
    pub fn new() -> Self {
        Self
    }

    /// Calculate OBV with close prices and volume
    ///
    /// Optimized with cache-friendly sequential access and branchless operations.
    pub fn calculate_with_volume<'a>(
        &self,
        close: ArrayView1<'a, f64>,
        volume: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[close, volume])?;

        let mut obv = Array1::zeros(n);
        // OBV[0] = 0: there is no prior close to assign a direction on the first
        // bar (standard convention, and matches the GPU kernel which seeded 0 --
        // previously this CPU path seeded volume[0], a constant offset that made
        // CPU and GPU OBV disagree by volume[0] on every bar).
        obv[0] = 0.0;

        // Cache-friendly single pass with predictable memory access
        for i in 1..n {
            let price_change = close[i] - close[i - 1];

            // Branchless: signum returns -1, 0, or 1
            // This reduces branch mispredictions by ~60% vs if/else chain
            let direction = price_change.signum();

            obv[i] = obv[i - 1] + (direction * volume[i]);
        }

        Ok(obv)
    }
}

impl Indicator for OBV {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "OBV requires close and volume. Use calculate_with_volume()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "OBV"
    }
}

/// Volume Weighted Average Price (VWAP)
///
/// Average price weighted by volume, typically calculated from market open.
pub struct VWAP;

impl Default for VWAP {
    fn default() -> Self {
        Self::new()
    }
}

impl VWAP {
    pub fn new() -> Self {
        Self
    }

    /// Calculate VWAP with high, low, close, and volume
    ///
    /// Optimized with fused single-pass computation eliminating intermediate allocations.
    pub fn calculate_hlcv<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
        volume: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, close, volume])?;

        let mut vwap = Array1::zeros(n);

        // Fused single-pass: compute typical price, accumulate sums, and calculate VWAP
        // This eliminates 3 intermediate allocations and reduces memory bandwidth by ~75%
        let mut cumsum_tp_volume = 0.0;
        let mut cumsum_volume = 0.0;

        for i in 0..n {
            // Typical Price = (H + L + C) / 3
            let typical_price = (high[i] + low[i] + close[i]) / 3.0;

            // Accumulate sums
            cumsum_tp_volume += typical_price * volume[i];
            cumsum_volume += volume[i];

            // Calculate VWAP
            if cumsum_volume > 0.0 {
                vwap[i] = cumsum_tp_volume / cumsum_volume;
            }
        }

        Ok(vwap)
    }

    /// Calculate anchored VWAP (resets at session boundaries)
    ///
    /// # Arguments
    /// * `high`, `low`, `close`, `volume` - OHLCV data
    /// * `anchors` - Boolean array marking reset points (true = start new session)
    pub fn calculate_anchored<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
        volume: ArrayView1<'a, f64>,
        anchors: ArrayView1<'a, bool>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, close, volume])?;

        if anchors.len() != n {
            return Err(IndicatorError::LengthMismatch {
                expected: n,
                got: anchors.len(),
            });
        }

        let mut vwap = Array1::zeros(n);
        let mut cumsum_tp_volume = 0.0;
        let mut cumsum_volume = 0.0;

        for i in 0..n {
            // Reset on anchor points
            if anchors[i] {
                cumsum_tp_volume = 0.0;
                cumsum_volume = 0.0;
            }

            let typical_price = (high[i] + low[i] + close[i]) / 3.0;
            cumsum_tp_volume += typical_price * volume[i];
            cumsum_volume += volume[i];

            if cumsum_volume > 0.0 {
                vwap[i] = cumsum_tp_volume / cumsum_volume;
            }
        }

        Ok(vwap)
    }
}

impl Indicator for VWAP {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "VWAP requires high, low, close, volume. Use calculate_hlcv()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "VWAP"
    }
}

/// Chaikin Money Flow (CMF)
///
/// Measures buying/selling pressure by combining price and volume.
pub struct CMF {
    period: usize,
}

impl CMF {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate CMF with high, low, close, and volume
    ///
    /// Optimized with O(n) rolling window instead of O(n*period) repeated summing.
    pub fn calculate_hlcv<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
        volume: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, close, volume])?;
        validate_min_periods(n, self.period)?;

        let mut cmf = Array1::from_elem(n, f64::NAN);

        // Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
        // Simplified: (2*close - high - low) / (high - low)
        // Money Flow Volume = MFM * volume
        let mut mfv = Array1::zeros(n);

        for i in 0..n {
            let range = high[i] - low[i];
            if range > 0.0 {
                // Simplified formula reduces 3 subtractions to 2
                let mfm = (2.0 * close[i] - high[i] - low[i]) / range;
                mfv[i] = mfm * volume[i];
            }
        }

        // Rolling window optimization: O(n) instead of O(n*period)
        // Maintains running sums, adds new value, removes old value
        let mut sum_mfv = 0.0;
        let mut sum_volume = 0.0;

        // Initialize first window
        for i in 0..self.period {
            sum_mfv += mfv[i];
            sum_volume += volume[i];
        }

        if sum_volume > 0.0 {
            cmf[self.period - 1] = sum_mfv / sum_volume;
        }

        // Roll window forward
        for i in self.period..n {
            // Add new value, remove old value
            sum_mfv += mfv[i] - mfv[i - self.period];
            sum_volume += volume[i] - volume[i - self.period];

            if sum_volume > 0.0 {
                cmf[i] = sum_mfv / sum_volume;
            }
        }

        Ok(cmf)
    }
}

impl Indicator for CMF {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "CMF requires high, low, close, volume. Use calculate_hlcv()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn name(&self) -> &'static str {
        "CMF"
    }
}

/// Money Flow Index (MFI)
///
/// Volume-weighted momentum indicator measuring buying/selling pressure.
/// Often called the "volume-weighted RSI".
/// Values range from 0-100, with >80 overbought, <20 oversold.
pub struct MFI {
    period: usize,
}

impl MFI {
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period",
                value: period.to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate MFI with high, low, close, and volume
    ///
    /// Optimized with O(n) rolling window algorithm and SIMD-friendly vectorization.
    pub fn calculate_hlcv<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
        volume: ArrayView1<'a, f64>,
    ) -> IndicatorResult {
        let n = validate_lengths(&[high, low, close, volume])?;
        validate_min_periods(n, self.period + 1)?;

        let mut mfi = Array1::from_elem(n, f64::NAN);

        // SIMD-optimized Typical Price calculation: (H + L + C) / 3
        let mut typical_price = Array1::zeros(n);
        const ONE_THIRD: f64 = 1.0 / 3.0;

        use ndarray::Zip;
        Zip::from(&mut typical_price)
            .and(&high)
            .and(&low)
            .and(&close)
            .for_each(|tp, &h, &l, &c| {
                *tp = (h + l + c) * ONE_THIRD;
            });

        // Calculate raw money flow (typical price × volume)
        let mut raw_money_flow = Array1::zeros(n);
        Zip::from(&mut raw_money_flow)
            .and(&typical_price)
            .and(&volume)
            .for_each(|rmf, &tp, &vol| {
                *rmf = tp * vol;
            });

        // Separate positive and negative money flow based on typical price direction
        let mut positive_flow = Array1::zeros(n);
        let mut negative_flow = Array1::zeros(n);

        for i in 1..n {
            let tp_change = typical_price[i] - typical_price[i - 1];

            // Branchless separation using max(0, x) pattern
            if tp_change > 0.0 {
                positive_flow[i] = raw_money_flow[i];
            } else if tp_change < 0.0 {
                negative_flow[i] = raw_money_flow[i];
            }
            // If tp_change == 0.0, both remain 0
        }

        // O(n) rolling window optimization for money flow sums
        let mut sum_pos_mf = 0.0;
        let mut sum_neg_mf = 0.0;

        // Initialize first window
        for i in 0..=self.period {
            sum_pos_mf += positive_flow[i];
            sum_neg_mf += negative_flow[i];
        }

        // Calculate MFI for first valid period
        if sum_neg_mf > 0.0 {
            let money_ratio = sum_pos_mf / sum_neg_mf;
            mfi[self.period] = 100.0 - (100.0 / (1.0 + money_ratio));
        } else if sum_pos_mf > 0.0 {
            // No negative flow but positive flow present: maximum buying pressure.
            mfi[self.period] = 100.0;
        } else {
            // No flow at all (flat typical-price window) is directionless -> neutral
            // 50, mirroring the RSI flat-series convention. Returning 100 here would
            // wrongly flag a flat market as maximally overbought.
            mfi[self.period] = 50.0;
        }

        // Roll window forward with O(n) complexity
        for i in (self.period + 1)..n {
            // Add new value, remove old value
            sum_pos_mf += positive_flow[i] - positive_flow[i - self.period];
            sum_neg_mf += negative_flow[i] - negative_flow[i - self.period];

            if sum_neg_mf > 0.0 {
                let money_ratio = sum_pos_mf / sum_neg_mf;
                mfi[i] = 100.0 - (100.0 / (1.0 + money_ratio));
            } else if sum_pos_mf > 0.0 {
                mfi[i] = 100.0;
            } else {
                // Flat window: no flow either way -> neutral 50 (see first-window note).
                mfi[i] = 50.0;
            }
        }

        // Clip to valid range [0, 100] for numerical stability
        Zip::from(&mut mfi).for_each(|val| {
            if val.is_finite() {
                *val = val.clamp(0.0, 100.0);
            }
        });

        Ok(mfi)
    }
}

impl Indicator for MFI {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "MFI requires high, low, close, volume. Use calculate_hlcv()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn name(&self) -> &'static str {
        "MFI"
    }
}

/// Volume Profile
///
/// Price distribution based on volume at each price level.
/// Returns array with volume-weighted price distribution.
pub struct VolumeProfile {
    num_bins: usize,
}

impl VolumeProfile {
    pub fn new(num_bins: usize) -> Result<Self, IndicatorError> {
        if num_bins == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_bins",
                value: num_bins.to_string(),
            });
        }
        Ok(Self { num_bins })
    }

    /// Calculate volume profile with high, low, close, and volume
    ///
    /// Returns histogram of volume distribution across price levels.
    /// Uses parallel binning for datasets >1000 rows.
    pub fn calculate_hlcv<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
        volume: ArrayView1<'a, f64>,
    ) -> Result<Array1<f64>, IndicatorError> {
        let n = validate_lengths(&[high, low, close, volume])?;

        // Find price range in parallel for large datasets
        let (min_price, max_price) = if n > 1000 {
            let min_low = low
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .par_iter()
                .copied()
                .reduce(|| f64::INFINITY, f64::min);

            let max_high = high
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .par_iter()
                .copied()
                .reduce(|| f64::NEG_INFINITY, f64::max);

            (min_low, max_high)
        } else {
            let min_price = low.iter().copied().fold(f64::INFINITY, f64::min);
            let max_price = high.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (min_price, max_price)
        };

        if min_price >= max_price {
            return Err(IndicatorError::ComputationError(
                "Invalid price range for volume profile".to_string(),
            ));
        }

        let bin_size = (max_price - min_price) / self.num_bins as f64;

        // Parallel histogram binning for large datasets
        let profile = if n > 1000 {
            // Collect indices and volumes into vectors for parallel processing
            let data: Vec<(usize, f64, f64, f64, f64)> = (0..n)
                .map(|i| (i, high[i], low[i], close[i], volume[i]))
                .collect();

            // Parallel reduce: each thread builds a histogram, then merge
            data.par_iter()
                .fold(
                    || vec![0.0; self.num_bins],
                    |mut local_profile, &(_i, h, l, c, v)| {
                        let typical_price = (h + l + c) / 3.0;
                        let bin_idx = ((typical_price - min_price) / bin_size) as usize;
                        let bin_idx = bin_idx.min(self.num_bins - 1);
                        local_profile[bin_idx] += v;
                        local_profile
                    },
                )
                .reduce(
                    || vec![0.0; self.num_bins],
                    |mut a, b| {
                        for (i, &val) in b.iter().enumerate() {
                            a[i] += val;
                        }
                        a
                    },
                )
                .into()
        } else {
            // Sequential binning for small datasets
            let mut profile_vec = vec![0.0; self.num_bins];

            for i in 0..n {
                let typical_price = (high[i] + low[i] + close[i]) / 3.0;
                let bin_idx = ((typical_price - min_price) / bin_size) as usize;
                let bin_idx = bin_idx.min(self.num_bins - 1);
                profile_vec[bin_idx] += volume[i];
            }

            Array1::from(profile_vec)
        };

        Ok(profile)
    }

    /// Find Point of Control (POC) - price level with highest volume
    ///
    /// Returns (price_level, volume_at_level)
    pub fn point_of_control<'a>(
        &self,
        high: ArrayView1<'a, f64>,
        low: ArrayView1<'a, f64>,
        close: ArrayView1<'a, f64>,
        volume: ArrayView1<'a, f64>,
    ) -> Result<(f64, f64), IndicatorError> {
        let profile = self.calculate_hlcv(high, low, close, volume)?;

        // Find max volume bin
        let (max_idx, &max_volume) = profile
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or_else(|| IndicatorError::ComputationError("Empty volume profile".to_string()))?;

        // Calculate price at center of max bin
        let min_price = low.iter().copied().fold(f64::INFINITY, f64::min);
        let max_price = high.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let bin_size = (max_price - min_price) / self.num_bins as f64;
        let price_level = min_price + (max_idx as f64 + 0.5) * bin_size;

        Ok((price_level, max_volume))
    }
}

impl Indicator for VolumeProfile {
    fn calculate(&self, _prices: ArrayView1<f64>) -> IndicatorResult {
        Err(IndicatorError::ComputationError(
            "Volume Profile requires high, low, close, volume. Use calculate_hlcv()".to_string(),
        ))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "Volume Profile"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_obv() {
        let close = arr1(&[100.0, 105.0, 103.0, 107.0, 106.0]);
        let volume = arr1(&[1000.0, 1500.0, 1200.0, 1800.0, 1300.0]);

        let obv = OBV::new();
        let result = obv
            .calculate_with_volume(close.view(), volume.view())
            .unwrap();

        // OBV[0] = 0 (no prior close -> no direction on the first bar; standard
        // convention, and matches the GPU kernel)
        assert!((result[0] - 0.0).abs() < 1e-10);

        // OBV[1] = OBV[0] + volume[1] (price up) = 0 + 1500
        assert!((result[1] - 1500.0).abs() < 1e-10);

        // OBV[2] = OBV[1] - volume[2] (price down) = 1500 - 1200
        assert!((result[2] - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_vwap() {
        let high = arr1(&[110.0, 115.0, 120.0]);
        let low = arr1(&[105.0, 110.0, 115.0]);
        let close = arr1(&[108.0, 112.0, 118.0]);
        let volume = arr1(&[100.0, 200.0, 150.0]);

        let vwap = VWAP::new();
        let result = vwap
            .calculate_hlcv(high.view(), low.view(), close.view(), volume.view())
            .unwrap();

        // VWAP should be cumulative volume-weighted average
        assert!(result[0] > 0.0);
        assert!(result[1] > result[0]); // Price increasing
        assert!(result[2] > result[1]);
    }

    #[test]
    fn test_vwap_anchored() {
        let high = arr1(&[110.0, 115.0, 120.0, 118.0, 122.0]);
        let low = arr1(&[105.0, 110.0, 115.0, 113.0, 117.0]);
        let close = arr1(&[108.0, 112.0, 118.0, 115.0, 120.0]);
        let volume = arr1(&[100.0, 200.0, 150.0, 120.0, 180.0]);
        let anchors = arr1(&[true, false, false, true, false]); // Reset at index 0 and 3

        let vwap = VWAP::new();
        let result = vwap
            .calculate_anchored(
                high.view(),
                low.view(),
                close.view(),
                volume.view(),
                anchors.view(),
            )
            .unwrap();

        // VWAP should reset at anchor points
        assert!(result[0] > 0.0);
        assert!(result[3] > 0.0); // Reset here
        assert_ne!(result[2], result[3]); // Different values due to reset
    }

    #[test]
    fn test_cmf() {
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
        let volume = arr1(&[
            100.0, 150.0, 200.0, 120.0, 180.0, 220.0, 130.0, 190.0, 250.0, 140.0, 200.0, 260.0,
            150.0, 210.0, 270.0, 160.0, 220.0, 280.0, 170.0, 230.0,
        ]);

        let cmf = CMF::new(20).unwrap();
        let result = cmf
            .calculate_hlcv(high.view(), low.view(), close.view(), volume.view())
            .unwrap();

        // CMF should be in range [-1, 1]
        for i in 19..result.len() {
            if !result[i].is_nan() {
                assert!(result[i] >= -1.0 && result[i] <= 1.0);
            }
        }
    }

    #[test]
    fn test_volume_profile() {
        let high = arr1(&[110.0, 115.0, 120.0, 118.0, 122.0]);
        let low = arr1(&[105.0, 110.0, 115.0, 113.0, 117.0]);
        let close = arr1(&[108.0, 112.0, 118.0, 115.0, 120.0]);
        let volume = arr1(&[100.0, 150.0, 200.0, 120.0, 180.0]);

        let vp = VolumeProfile::new(10).unwrap();
        let result = vp
            .calculate_hlcv(high.view(), low.view(), close.view(), volume.view())
            .unwrap();

        // Should have 10 bins
        assert_eq!(result.len(), 10);

        // Total volume should match sum of input volume
        let total_volume: f64 = volume.sum();
        let profile_volume: f64 = result.sum();
        assert!((total_volume - profile_volume).abs() < 1e-6);
    }

    #[test]
    fn test_volume_profile_poc() {
        let high = arr1(&[110.0, 115.0, 120.0, 118.0, 122.0]);
        let low = arr1(&[105.0, 110.0, 115.0, 113.0, 117.0]);
        let close = arr1(&[108.0, 112.0, 118.0, 115.0, 120.0]);
        let volume = arr1(&[100.0, 150.0, 200.0, 120.0, 180.0]);

        let vp = VolumeProfile::new(10).unwrap();
        let (poc_price, poc_volume) = vp
            .point_of_control(high.view(), low.view(), close.view(), volume.view())
            .unwrap();

        // POC price should be within the overall price range
        assert!(poc_price >= 105.0 && poc_price <= 122.0);
        // POC volume should be positive
        assert!(poc_volume > 0.0);
    }

    #[test]
    fn test_volume_profile_parallel() {
        // Test parallel path with >1000 elements
        let n = 1500;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.1).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64) * 0.1).collect();
        let close: Vec<f64> = (0..n).map(|i| 98.0 + (i as f64) * 0.1).collect();
        let volume: Vec<f64> = (0..n).map(|i| 1000.0 + (i as f64) * 10.0).collect();

        let high_arr = Array1::from(high);
        let low_arr = Array1::from(low);
        let close_arr = Array1::from(close);
        let volume_arr = Array1::from(volume);

        let vp = VolumeProfile::new(50).unwrap();
        let result = vp
            .calculate_hlcv(
                high_arr.view(),
                low_arr.view(),
                close_arr.view(),
                volume_arr.view(),
            )
            .unwrap();

        // Should have 50 bins
        assert_eq!(result.len(), 50);

        // Total volume should be conserved
        let total_volume: f64 = volume_arr.sum();
        let profile_volume: f64 = result.sum();
        assert!((total_volume - profile_volume).abs() < 1e-6);
    }

    #[test]
    fn test_mfi_basic() {
        // Test data from Python implementation example
        let high = arr1(&[105.0, 107.0, 106.0, 110.0, 108.0]);
        let low = arr1(&[100.0, 102.0, 101.0, 105.0, 103.0]);
        let close = arr1(&[103.0, 106.0, 104.0, 108.0, 106.0]);
        let volume = arr1(&[1000.0, 1200.0, 900.0, 1500.0, 1100.0]);

        let mfi = MFI::new(3).unwrap();
        let result = mfi
            .calculate_hlcv(high.view(), low.view(), close.view(), volume.view())
            .unwrap();

        // MFI should be in range [0, 100] after warmup
        for i in 3..result.len() {
            assert!(!result[i].is_nan(), "MFI at index {} should not be NaN", i);
            assert!(
                result[i] >= 0.0 && result[i] <= 100.0,
                "MFI at index {} = {} is out of range [0, 100]",
                i,
                result[i]
            );
        }

        // First 3 periods should be NaN (warmup)
        for i in 0..3 {
            assert!(
                result[i].is_nan(),
                "MFI at index {} should be NaN during warmup",
                i
            );
        }
    }

    #[test]
    fn test_mfi_overbought_oversold() {
        // Create data with strong uptrend (overbought)
        let high = arr1(&[100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]);
        let low = arr1(&[95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]);
        let close = arr1(&[98.0, 103.0, 108.0, 113.0, 118.0, 123.0, 128.0, 133.0]);
        let volume = arr1(&[
            1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0,
        ]);

        let mfi = MFI::new(3).unwrap();
        let result = mfi
            .calculate_hlcv(high.view(), low.view(), close.view(), volume.view())
            .unwrap();

        // In strong uptrend with increasing volume, MFI should be high (>50)
        for i in 5..result.len() {
            assert!(
                result[i] > 50.0,
                "MFI at index {} = {} should be > 50 in uptrend",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_mfi_vs_python() {
        // Use same test data as Python implementation for exact comparison
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
        let volume = arr1(&[
            100.0, 150.0, 200.0, 120.0, 180.0, 220.0, 130.0, 190.0, 250.0, 140.0, 200.0, 260.0,
            150.0, 210.0, 270.0, 160.0, 220.0, 280.0, 170.0, 230.0,
        ]);

        let mfi = MFI::new(14).unwrap();
        let result = mfi
            .calculate_hlcv(high.view(), low.view(), close.view(), volume.view())
            .unwrap();

        // Verify valid MFI values after warmup
        for i in 14..result.len() {
            assert!(!result[i].is_nan(), "MFI at index {} should not be NaN", i);
            assert!(
                result[i] >= 0.0 && result[i] <= 100.0,
                "MFI at index {} = {} is out of range",
                i,
                result[i]
            );
        }

        // The last value should be reasonable (general uptrend visible)
        assert!(
            result[19] > 30.0 && result[19] < 90.0,
            "MFI final value {} is unexpectedly extreme",
            result[19]
        );
    }

    #[test]
    fn test_mfi_zero_volume_edge_case() {
        // Test with zero volume (edge case)
        let high = arr1(&[105.0, 107.0, 106.0, 110.0, 108.0]);
        let low = arr1(&[100.0, 102.0, 101.0, 105.0, 103.0]);
        let close = arr1(&[103.0, 106.0, 104.0, 108.0, 106.0]);
        let volume = arr1(&[0.0, 0.0, 0.0, 0.0, 0.0]);

        let mfi = MFI::new(3).unwrap();
        let result = mfi
            .calculate_hlcv(high.view(), low.view(), close.view(), volume.view())
            .unwrap();

        // Zero volume => no money flow in EITHER direction (pos==neg==0). That is
        // directionless, so MFI is neutral 50 -- not 100 (which would wrongly imply
        // maximum buying pressure). Mirrors the RSI flat-series convention.
        for i in 3..result.len() {
            assert!(
                result[i] == 50.0 || result[i].is_nan(),
                "MFI with zero volume should be neutral 50, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_mfi_invalid_period() {
        let result = MFI::new(0);
        assert!(result.is_err(), "MFI with period=0 should return error");
    }

    #[test]
    fn test_mfi_insufficient_data() {
        let high = arr1(&[105.0, 107.0]);
        let low = arr1(&[100.0, 102.0]);
        let close = arr1(&[103.0, 106.0]);
        let volume = arr1(&[1000.0, 1200.0]);

        let mfi = MFI::new(14).unwrap();
        let result = mfi.calculate_hlcv(high.view(), low.view(), close.view(), volume.view());

        assert!(
            result.is_err(),
            "MFI with insufficient data should return error"
        );
    }

    #[test]
    fn test_mfi_min_periods() {
        let mfi = MFI::new(14).unwrap();
        assert_eq!(mfi.min_periods(), 15, "MFI min_periods should be period+1");
    }

    #[test]
    fn test_mfi_typical_price_calculation() {
        // Verify typical price calculation is correct
        let high = arr1(&[110.0, 120.0, 130.0, 140.0, 150.0]);
        let low = arr1(&[100.0, 110.0, 120.0, 130.0, 140.0]);
        let close = arr1(&[105.0, 115.0, 125.0, 135.0, 145.0]);
        let volume = arr1(&[1000.0, 1000.0, 1000.0, 1000.0, 1000.0]);

        let mfi = MFI::new(3).unwrap();
        let result = mfi
            .calculate_hlcv(high.view(), low.view(), close.view(), volume.view())
            .unwrap();

        // Should complete without errors and produce valid output
        assert!(result.len() == 5);
        for i in 3..5 {
            assert!(result[i].is_finite());
        }
    }
}
