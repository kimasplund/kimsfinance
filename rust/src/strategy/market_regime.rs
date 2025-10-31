//! Market Regime Detection for Adaptive Strategy Parameters
//!
//! Classifies market conditions into regimes based on:
//! - Trend: Bull/Bear/Sideways (50-day SMA slope)
//! - Volatility: Low/High (20-day ATR percentile)
//!
//! Used to adapt strategy parameters dynamically based on market conditions.

use crate::strategy::spot_data::{SpotDataError, SpotDataLoader};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Bull market with low volatility (best conditions for credit spreads)
    BullLowVol,
    /// Bull market with high volatility (reduce risk)
    BullHighVol,
    /// Bear market with low volatility (minimal trading)
    BearLowVol,
    /// Bear market with high volatility (avoid or defensive)
    BearHighVol,
    /// Sideways market (choppy, range-bound)
    Sideways,
}

impl fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarketRegime::BullLowVol => write!(f, "Bull/LowVol"),
            MarketRegime::BullHighVol => write!(f, "Bull/HighVol"),
            MarketRegime::BearLowVol => write!(f, "Bear/LowVol"),
            MarketRegime::BearHighVol => write!(f, "Bear/HighVol"),
            MarketRegime::Sideways => write!(f, "Sideways"),
        }
    }
}

/// Trend direction classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Bull,
    Bear,
    Sideways,
}

/// Volatility level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityLevel {
    Low,
    High,
}

/// Market regime detector
pub struct RegimeDetector {
    /// SMA period for trend detection (default: 50 days)
    trend_period: usize,

    /// ATR lookback for volatility percentile (default: 252 days = 1 year)
    volatility_lookback: usize,

    /// Low volatility threshold (percentile, default: 20th)
    low_vol_percentile: f64,

    /// High volatility threshold (percentile, default: 80th)
    high_vol_percentile: f64,

    /// Bull/Bear threshold for SMA slope (%, default: 2% over period)
    trend_threshold_pct: f64,
}

impl Default for RegimeDetector {
    fn default() -> Self {
        Self {
            trend_period: 50,
            volatility_lookback: 252,
            low_vol_percentile: 20.0,
            high_vol_percentile: 80.0,
            trend_threshold_pct: 2.0,
        }
    }
}

impl RegimeDetector {
    /// Create new regime detector with custom parameters
    pub fn new(
        trend_period: usize,
        volatility_lookback: usize,
        low_vol_percentile: f64,
        high_vol_percentile: f64,
        trend_threshold_pct: f64,
    ) -> Self {
        Self {
            trend_period,
            volatility_lookback,
            low_vol_percentile,
            high_vol_percentile,
            trend_threshold_pct,
        }
    }

    /// Detect market regime for a symbol on a date
    pub fn detect_regime(
        &self,
        spot_loader: &mut SpotDataLoader,
        symbol: &str,
        date: NaiveDate,
    ) -> Result<MarketRegime, SpotDataError> {
        let trend = self.calculate_trend(spot_loader, symbol, date)?;
        let volatility = self.calculate_volatility_level(spot_loader, symbol, date)?;

        let regime = match (trend, volatility) {
            (TrendDirection::Bull, VolatilityLevel::Low) => MarketRegime::BullLowVol,
            (TrendDirection::Bull, VolatilityLevel::High) => MarketRegime::BullHighVol,
            (TrendDirection::Bear, VolatilityLevel::Low) => MarketRegime::BearLowVol,
            (TrendDirection::Bear, VolatilityLevel::High) => MarketRegime::BearHighVol,
            (TrendDirection::Sideways, _) => MarketRegime::Sideways,
        };

        Ok(regime)
    }

    /// Calculate trend direction using SMA slope
    pub fn calculate_trend(
        &self,
        spot_loader: &mut SpotDataLoader,
        symbol: &str,
        date: NaiveDate,
    ) -> Result<TrendDirection, SpotDataError> {
        // Get bars for the trend period
        let bars = spot_loader.load_symbol(symbol)?;

        // Find the index of the date
        let date_idx = bars
            .iter()
            .position(|b| b.date == date)
            .ok_or_else(|| SpotDataError::NotFound(format!("No data for {} on {}", symbol, date)))?;

        // Need at least trend_period bars
        if date_idx < self.trend_period {
            return Err(SpotDataError::InsufficientData(format!(
                "Need {} days of data before {}, only have {}",
                self.trend_period, date, date_idx
            )));
        }

        // Calculate SMA for the period
        let start_idx = date_idx - self.trend_period + 1;
        let closes: Vec<f64> = bars[start_idx..=date_idx].iter().map(|b| b.close).collect();
        let sma: f64 = closes.iter().sum::<f64>() / closes.len() as f64;

        // Compare current price to SMA at start of period
        let current_price = bars[date_idx].close;
        let period_start_price = bars[start_idx].close;

        // Calculate % change from period start
        let change_pct = ((current_price - period_start_price) / period_start_price) * 100.0;

        // Classify trend
        if change_pct > self.trend_threshold_pct {
            Ok(TrendDirection::Bull)
        } else if change_pct < -self.trend_threshold_pct {
            Ok(TrendDirection::Bear)
        } else {
            Ok(TrendDirection::Sideways)
        }
    }

    /// Calculate volatility level using ATR percentile
    pub fn calculate_volatility_level(
        &self,
        spot_loader: &mut SpotDataLoader,
        symbol: &str,
        date: NaiveDate,
    ) -> Result<VolatilityLevel, SpotDataError> {
        // Get current ATR
        let current_atr = spot_loader.calculate_atr(symbol, date)?;

        // Get historical ATRs for the lookback period
        // Clone the dates we need to avoid borrowing issues
        let bars = spot_loader.load_symbol(symbol)?;
        let date_idx = bars
            .iter()
            .position(|b| b.date == date)
            .ok_or_else(|| SpotDataError::NotFound(format!("No data for {} on {}", symbol, date)))?;

        // Need at least volatility_lookback + 20 bars (20 for ATR calculation)
        let required_bars = self.volatility_lookback + 20;
        if date_idx < required_bars {
            return Err(SpotDataError::InsufficientData(format!(
                "Need {} days of data before {}, only have {}",
                required_bars, date, date_idx
            )));
        }

        // Extract dates we need before borrowing spot_loader again
        let start_idx = date_idx - self.volatility_lookback + 1;
        let dates_to_check: Vec<NaiveDate> = bars[start_idx..=date_idx]
            .iter()
            .map(|b| b.date)
            .collect();

        // Calculate ATRs for the lookback period
        let mut historical_atrs = Vec::new();
        for date in dates_to_check {
            if let Ok(atr) = spot_loader.calculate_atr(symbol, date) {
                historical_atrs.push(atr);
            }
        }

        if historical_atrs.is_empty() {
            return Err(SpotDataError::InsufficientData(
                "No ATR data available for volatility calculation".to_string(),
            ));
        }

        // Sort ATRs to calculate percentiles
        let mut sorted_atrs = historical_atrs.clone();
        sorted_atrs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate percentile thresholds
        let low_vol_idx = ((self.low_vol_percentile / 100.0) * sorted_atrs.len() as f64) as usize;
        let high_vol_idx = ((self.high_vol_percentile / 100.0) * sorted_atrs.len() as f64) as usize;

        let low_vol_threshold = sorted_atrs[low_vol_idx.min(sorted_atrs.len() - 1)];
        let high_vol_threshold = sorted_atrs[high_vol_idx.min(sorted_atrs.len() - 1)];

        // Classify volatility
        if current_atr < low_vol_threshold {
            Ok(VolatilityLevel::Low)
        } else if current_atr > high_vol_threshold {
            Ok(VolatilityLevel::High)
        } else {
            // Medium volatility - treat as low (conservative)
            Ok(VolatilityLevel::Low)
        }
    }

    /// Get regime statistics for a date range (useful for backtesting analysis)
    pub fn get_regime_stats(
        &self,
        spot_loader: &mut SpotDataLoader,
        symbol: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<RegimeStats, SpotDataError> {
        let bars = spot_loader.load_symbol(symbol)?;

        let trading_dates: Vec<NaiveDate> = bars
            .iter()
            .map(|b| b.date)
            .filter(|d| *d >= start_date && *d <= end_date)
            .collect();

        let mut stats = RegimeStats::default();

        for date in trading_dates {
            if let Ok(regime) = self.detect_regime(spot_loader, symbol, date) {
                stats.total_days += 1;
                match regime {
                    MarketRegime::BullLowVol => stats.bull_low_vol += 1,
                    MarketRegime::BullHighVol => stats.bull_high_vol += 1,
                    MarketRegime::BearLowVol => stats.bear_low_vol += 1,
                    MarketRegime::BearHighVol => stats.bear_high_vol += 1,
                    MarketRegime::Sideways => stats.sideways += 1,
                }
            }
        }

        Ok(stats)
    }
}

/// Regime statistics for analysis
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RegimeStats {
    pub total_days: usize,
    pub bull_low_vol: usize,
    pub bull_high_vol: usize,
    pub bear_low_vol: usize,
    pub bear_high_vol: usize,
    pub sideways: usize,
}

impl RegimeStats {
    /// Get percentage of days in each regime
    pub fn percentages(&self) -> (f64, f64, f64, f64, f64) {
        let total = self.total_days as f64;
        (
            (self.bull_low_vol as f64 / total) * 100.0,
            (self.bull_high_vol as f64 / total) * 100.0,
            (self.bear_low_vol as f64 / total) * 100.0,
            (self.bear_high_vol as f64 / total) * 100.0,
            (self.sideways as f64 / total) * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_enum() {
        let regime = MarketRegime::BullLowVol;
        assert_eq!(regime.to_string(), "Bull/LowVol");
    }

    #[test]
    fn test_regime_detector_default() {
        let detector = RegimeDetector::default();
        assert_eq!(detector.trend_period, 50);
        assert_eq!(detector.volatility_lookback, 252);
    }

    #[test]
    #[ignore] // Requires actual data files
    fn test_detect_regime() {
        let mut loader = SpotDataLoader::new("data/yfinance/ohlcv").unwrap();
        let detector = RegimeDetector::default();
        let date = NaiveDate::from_ymd_opt(2020, 6, 1).unwrap();

        let regime = detector.detect_regime(&mut loader, "SPY", date).unwrap();
        println!("SPY regime on {}: {}", date, regime);
    }

    #[test]
    #[ignore] // Requires actual data files
    fn test_regime_stats() {
        let mut loader = SpotDataLoader::new("data/yfinance/ohlcv").unwrap();
        let detector = RegimeDetector::default();

        let start_date = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let end_date = NaiveDate::from_ymd_opt(2020, 12, 31).unwrap();

        let stats = detector.get_regime_stats(&mut loader, "SPY", start_date, end_date).unwrap();
        let (bull_low, bull_high, bear_low, bear_high, sideways) = stats.percentages();

        println!("2020 Regime Distribution:");
        println!("  Bull/LowVol: {:.1}%", bull_low);
        println!("  Bull/HighVol: {:.1}%", bull_high);
        println!("  Bear/LowVol: {:.1}%", bear_low);
        println!("  Bear/HighVol: {:.1}%", bear_high);
        println!("  Sideways: {:.1}%", sideways);
    }
}
