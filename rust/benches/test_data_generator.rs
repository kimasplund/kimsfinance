//! Test Data Generator for Batch Backtest Benchmarks
//!
//! Generates realistic OHLCV data with configurable characteristics:
//! - Trends (upward, downward, sideways)
//! - Volatility patterns (high, medium, low)
//! - Seasonal patterns (daily, weekly cycles)
//! - Random walk with drift
//!
//! Used by batch_backtest_benchmark.rs to ensure reproducible benchmarks.

use rand::prelude::*;
use rand::SeedableRng;
use std::f64::consts::PI;

/// OHLCV dataset structure
#[derive(Clone, Debug)]
pub struct OHLCVData {
    pub timestamps: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl OHLCVData {
    /// Get number of candles
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

/// Market regime for realistic data generation
#[derive(Clone, Copy, Debug)]
pub enum MarketRegime {
    /// Strong uptrend (bull market)
    BullTrend,
    /// Strong downtrend (bear market)
    BearTrend,
    /// Sideways movement (range-bound)
    Sideways,
    /// High volatility (e.g., news events)
    HighVolatility,
    /// Low volatility (quiet market)
    LowVolatility,
}

/// Configuration for data generation
#[derive(Clone, Debug)]
pub struct DataGeneratorConfig {
    /// Number of candles to generate
    pub n_candles: usize,
    /// Market regime
    pub regime: MarketRegime,
    /// Base price (e.g., 100.0 for $100)
    pub base_price: f64,
    /// Trend strength (-0.1 to 0.1 per candle)
    pub trend_strength: f64,
    /// Volatility factor (0.5 = 50% of base)
    pub volatility: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for DataGeneratorConfig {
    fn default() -> Self {
        Self {
            n_candles: 10000,
            regime: MarketRegime::Sideways,
            base_price: 100.0,
            trend_strength: 0.0005,
            volatility: 0.02,
            seed: 12345,
        }
    }
}

impl DataGeneratorConfig {
    /// Create config for bull market
    pub fn bull_market(n_candles: usize, seed: u64) -> Self {
        Self {
            n_candles,
            regime: MarketRegime::BullTrend,
            base_price: 100.0,
            trend_strength: 0.001,
            volatility: 0.015,
            seed,
        }
    }

    /// Create config for bear market
    pub fn bear_market(n_candles: usize, seed: u64) -> Self {
        Self {
            n_candles,
            regime: MarketRegime::BearTrend,
            base_price: 100.0,
            trend_strength: -0.0008,
            volatility: 0.02,
            seed,
        }
    }

    /// Create config for sideways market
    pub fn sideways_market(n_candles: usize, seed: u64) -> Self {
        Self {
            n_candles,
            regime: MarketRegime::Sideways,
            base_price: 100.0,
            trend_strength: 0.0,
            volatility: 0.01,
            seed,
        }
    }

    /// Create config for high volatility market
    pub fn high_volatility(n_candles: usize, seed: u64) -> Self {
        Self {
            n_candles,
            regime: MarketRegime::HighVolatility,
            base_price: 100.0,
            trend_strength: 0.0,
            volatility: 0.05,
            seed,
        }
    }
}

/// Generate realistic OHLCV data with configurable characteristics
pub fn generate_realistic_ohlcv(config: &DataGeneratorConfig) -> OHLCVData {
    let mut rng = StdRng::seed_from_u64(config.seed);

    let n = config.n_candles;
    let mut timestamps = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    let base_time = 1_640_000_000i64; // Start: 2021-12-20
    let mut current_price = config.base_price;

    for i in 0..n {
        timestamps.push(base_time + (i as i64 * 60)); // 1-minute bars

        // Price movement components
        let trend = config.trend_strength;
        let cycle = match config.regime {
            MarketRegime::BullTrend | MarketRegime::BearTrend => {
                // Add daily cycle
                config.volatility * (i as f64 * 2.0 * PI / 1440.0).sin()
            }
            MarketRegime::Sideways => {
                // Strong mean reversion
                config.volatility * 2.0 * (i as f64 * 2.0 * PI / 200.0).sin()
            }
            MarketRegime::HighVolatility | MarketRegime::LowVolatility => {
                // Random noise dominates
                0.0
            }
        };

        // Random walk
        let noise = rng.gen_range(-config.volatility..config.volatility);

        // Update current price
        let price_change = trend + cycle + noise;
        current_price *= 1.0 + price_change;

        // Generate OHLC for this candle
        let candle_range = config.volatility * rng.gen_range(0.5..1.5);
        let o = current_price + rng.gen_range(-candle_range..candle_range);
        let c = current_price + rng.gen_range(-candle_range..candle_range);
        let h = o.max(c) * (1.0 + rng.gen_range(0.0..candle_range * 0.5));
        let l = o.min(c) * (1.0 - rng.gen_range(0.0..candle_range * 0.5));

        open.push(o);
        high.push(h);
        low.push(l);
        close.push(c);

        // Volume: mean-reverting around base volume
        let base_volume = 5000.0;
        let volume_noise = rng.gen_range(0.5..1.5);
        volume.push(base_volume * volume_noise);
    }

    OHLCVData {
        timestamps,
        open,
        high,
        low,
        close,
        volume,
    }
}

/// Generate simple random walk (fastest, for quick tests)
pub fn generate_simple_random_walk(n_candles: usize, seed: u64) -> OHLCVData {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut timestamps = Vec::with_capacity(n_candles);
    let mut close = Vec::with_capacity(n_candles);

    let base_time = 1_640_000_000i64;
    let mut current_price = 100.0;

    for i in 0..n_candles {
        timestamps.push(base_time + (i as i64 * 60));

        // Simple random walk
        let change = rng.gen_range(-0.02..0.02);
        current_price *= 1.0 + change;
        close.push(current_price);
    }

    // Generate OHLV from close prices
    let open: Vec<f64> = close.iter().enumerate()
        .map(|(i, &c)| if i == 0 { c } else { close[i - 1] })
        .collect();

    let high: Vec<f64> = close.iter().zip(&open)
        .map(|(&c, &o)| c.max(o) * (1.0 + rng.gen_range(0.0..0.01)))
        .collect();

    let low: Vec<f64> = close.iter().zip(&open)
        .map(|(&c, &o)| c.min(o) * (1.0 - rng.gen_range(0.0..0.01)))
        .collect();

    let volume: Vec<f64> = (0..n_candles)
        .map(|_| rng.gen_range(1000.0..10000.0))
        .collect();

    OHLCVData {
        timestamps,
        open,
        high,
        low,
        close,
        volume,
    }
}

/// Generate trending market (for testing strategies that need trends)
pub fn generate_trending_market(n_candles: usize, uptrend: bool, seed: u64) -> OHLCVData {
    let config = if uptrend {
        DataGeneratorConfig::bull_market(n_candles, seed)
    } else {
        DataGeneratorConfig::bear_market(n_candles, seed)
    };

    generate_realistic_ohlcv(&config)
}

/// Generate ranging market (for testing mean reversion strategies)
pub fn generate_ranging_market(n_candles: usize, seed: u64) -> OHLCVData {
    let config = DataGeneratorConfig::sideways_market(n_candles, seed);
    generate_realistic_ohlcv(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_simple_random_walk() {
        let data = generate_simple_random_walk(1000, 12345);
        assert_eq!(data.len(), 1000);
        assert!(data.close[0] > 0.0);
        assert_eq!(data.open.len(), data.close.len());
    }

    #[test]
    fn test_generate_realistic_ohlcv() {
        let config = DataGeneratorConfig::default();
        let data = generate_realistic_ohlcv(&config);

        assert_eq!(data.len(), config.n_candles);

        // Validate OHLC relationships
        for i in 0..data.len() {
            assert!(data.high[i] >= data.open[i]);
            assert!(data.high[i] >= data.close[i]);
            assert!(data.low[i] <= data.open[i]);
            assert!(data.low[i] <= data.close[i]);
        }
    }

    #[test]
    fn test_reproducibility() {
        let data1 = generate_simple_random_walk(100, 42);
        let data2 = generate_simple_random_walk(100, 42);

        assert_eq!(data1.close, data2.close);
    }

    #[test]
    fn test_bull_market_has_upward_trend() {
        let data = generate_trending_market(1000, true, 123);

        // Check that price generally increases
        let start_price = data.close[0];
        let end_price = data.close[data.len() - 1];

        assert!(
            end_price > start_price,
            "Bull market should have upward trend: start={}, end={}",
            start_price,
            end_price
        );
    }

    #[test]
    fn test_bear_market_has_downward_trend() {
        let data = generate_trending_market(1000, false, 456);

        // Check that price generally decreases
        let start_price = data.close[0];
        let end_price = data.close[data.len() - 1];

        assert!(
            end_price < start_price,
            "Bear market should have downward trend: start={}, end={}",
            start_price,
            end_price
        );
    }
}
