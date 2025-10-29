//! Test Data Generators for Heston Integration Tests
//!
//! Provides synthetic market data, option chains, and backtesting scenarios
//! for comprehensive integration testing of the Heston-Backtest pipeline.

use chrono::Utc;
use kimsfinance_core::quantitative::heston::{HestonParams, OptionQuote, OptionType};

/// Market regime for generating realistic test scenarios
#[derive(Debug, Clone, Copy)]
pub enum MarketRegime {
    /// Uptrend: +2% drift, 30% vol
    Trending,
    /// Sideways: 0% drift, 20% vol
    RangeBound,
    /// High volatility: 0% drift, 80% vol (crypto-like)
    Volatile,
}

/// Generate synthetic BTC price path using GBM
///
/// # Arguments
///
/// * `num_candles` - Number of OHLCV bars to generate
/// * `regime` - Market regime (Trending/RangeBound/Volatile)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// Tuple of (timestamps, open, high, low, close, volume)
pub fn generate_btc_ohlcv(
    num_candles: usize,
    regime: MarketRegime,
    seed: u64,
) -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let (drift, volatility) = match regime {
        MarketRegime::Trending => (0.02, 0.30),
        MarketRegime::RangeBound => (0.0, 0.20),
        MarketRegime::Volatile => (0.0, 0.80),
    };

    let mut timestamps = Vec::with_capacity(num_candles);
    let mut open = Vec::with_capacity(num_candles);
    let mut high = Vec::with_capacity(num_candles);
    let mut low = Vec::with_capacity(num_candles);
    let mut close = Vec::with_capacity(num_candles);
    let mut volume = Vec::with_capacity(num_candles);

    let start_time = Utc::now().timestamp() - (num_candles as i64 * 60);
    let mut current_price = 50000.0;

    for i in 0..num_candles {
        timestamps.push(start_time + i as i64 * 60);

        // GBM: dS = μS dt + σS dW
        let dt = 1.0 / 525600.0; // 1 minute in years
        let drift_term = drift * current_price * dt;
        let vol_term = volatility * current_price * dt.sqrt() * rng.sample(rand_distr::StandardNormal);

        let open_price = current_price;
        current_price += drift_term + vol_term;
        let close_price = current_price;

        // Add intrabar volatility for high/low
        let range = volatility * current_price * dt.sqrt();
        let high_price = current_price.max(open_price) + range * rng.gen::<f64>();
        let low_price = current_price.min(open_price) - range * rng.gen::<f64>();

        open.push(open_price);
        high.push(high_price);
        low.push(low_price);
        close.push(close_price);

        // Volume proportional to volatility
        let base_volume = 100.0;
        let vol_multiplier = 1.0 + (high_price - low_price) / close_price;
        volume.push(base_volume * vol_multiplier);
    }

    (timestamps, open, high, low, close, volume)
}

/// Generate synthetic options chain for testing
///
/// # Arguments
///
/// * `spot_price` - Current BTC spot price
/// * `num_strikes` - Number of strike prices (centered around spot)
/// * `expiry_days` - Days to expiration
/// * `params` - Heston parameters for consistent pricing
///
/// # Returns
///
/// Vector of OptionQuote with synthetic market data
pub fn generate_options_chain(
    spot_price: f64,
    num_strikes: usize,
    expiry_days: i64,
    params: &HestonParams,
) -> Vec<OptionQuote> {
    let now = Utc::now().timestamp();
    let expiration = now + (expiry_days * 24 * 3600);

    let mut options = Vec::with_capacity(num_strikes * 2); // calls + puts

    // Strike range: 80% to 120% of spot
    let strike_min = spot_price * 0.80;
    let strike_max = spot_price * 1.20;
    let strike_step = (strike_max - strike_min) / (num_strikes - 1) as f64;

    for i in 0..num_strikes {
        let strike = strike_min + i as f64 * strike_step;

        // Call option
        options.push(OptionQuote {
            underlying: "BTC".to_string(),
            strike,
            expiration,
            option_type: OptionType::Call,
            spot_price,
            risk_free_rate: 0.05,
            bid: None, // Will be populated by pricer
            ask: None,
            last: None,
            implied_vol: Some(params.v0.sqrt()), // Initial vol estimate
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        });

        // Put option
        options.push(OptionQuote {
            underlying: "BTC".to_string(),
            strike,
            expiration,
            option_type: OptionType::Put,
            spot_price,
            risk_free_rate: 0.05,
            bid: None,
            ask: None,
            last: None,
            implied_vol: Some(params.v0.sqrt()),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        });
    }

    options
}

/// Generate test Heston parameters for different market conditions
pub fn test_heston_params(regime: MarketRegime) -> HestonParams {
    match regime {
        MarketRegime::Trending => {
            // Low vol environment
            HestonParams::new(
                2.5,  // kappa (mean reversion speed)
                0.04, // theta (long-term var, 20% vol)
                0.2,  // sigma (vol of vol)
                -0.5, // rho (correlation)
                0.04, // v0 (initial var, 20% vol)
            )
            .expect("Valid Heston params for Trending")
        }
        MarketRegime::RangeBound => {
            // Moderate vol
            HestonParams::new(
                2.0,  // kappa
                0.09, // theta (30% vol)
                0.3,  // sigma
                -0.7, // rho (leverage effect)
                0.09, // v0 (30% vol)
            )
            .expect("Valid Heston params for RangeBound")
        }
        MarketRegime::Volatile => {
            // High vol (crypto crash)
            HestonParams::new(
                1.5,  // kappa
                0.64, // theta (80% vol)
                0.5,  // sigma
                -0.9, // rho (strong leverage effect)
                0.81, // v0 (90% vol)
            )
            .expect("Valid Heston params for Volatile")
        }
    }
}

/// Generate parameter sweep for strategy optimization
///
/// # Arguments
///
/// * `strategy_type` - Type of strategy (determines param ranges)
/// * `num_combinations` - Total number of parameter combinations to generate
///
/// # Returns
///
/// Vector of parameter vectors suitable for BatchBacktestSweep
pub fn generate_strategy_params(
    strategy_type: kimsfinance_core::backtest::batch::StrategyType,
    num_combinations: usize,
) -> Vec<Vec<f64>> {
    use kimsfinance_core::backtest::batch::StrategyType;

    match strategy_type {
        StrategyType::RsiCrossover => {
            // Parameters: [rsi_period, buy_threshold, sell_threshold]
            let mut params = Vec::new();
            let step = (num_combinations as f64).sqrt().ceil() as usize;

            for buy_thresh in 20..=(20 + step) {
                for sell_thresh in 70..=(70 + step) {
                    if params.len() < num_combinations {
                        params.push(vec![14.0, buy_thresh as f64, sell_thresh as f64]);
                    }
                }
            }
            params
        }
        StrategyType::LongStraddle => {
            // Parameters: [vol_threshold, breakeven_pct]
            let mut params = Vec::new();
            let step = (num_combinations as f64).sqrt().ceil() as usize;

            for vol_thresh in 5..=(5 + step) {
                for breakeven in 10..=(10 + step) {
                    if params.len() < num_combinations {
                        params.push(vec![vol_thresh as f64 / 100.0, breakeven as f64 / 100.0]);
                    }
                }
            }
            params
        }
        StrategyType::ShortStraddle => {
            // Parameters: [vol_threshold, max_loss_pct]
            let mut params = Vec::new();
            let step = (num_combinations as f64).sqrt().ceil() as usize;

            for vol_thresh in 10..=(10 + step) {
                for max_loss in 20..=(20 + step) {
                    if params.len() < num_combinations {
                        params.push(vec![vol_thresh as f64 / 100.0, max_loss as f64 / 100.0]);
                    }
                }
            }
            params
        }
        StrategyType::VolatilityArbitrage => {
            // Parameters: [vol_threshold, hedge_delta, min_edge]
            let mut params = Vec::new();
            let step = (num_combinations as f64).cbrt().ceil() as usize;

            for vol_thresh in 5..=(5 + step) {
                for hedge in 8..=(8 + step) {
                    for min_edge in 2..=(2 + step) {
                        if params.len() < num_combinations {
                            params.push(vec![
                                vol_thresh as f64 / 100.0,
                                hedge as f64 / 10.0,
                                min_edge as f64 / 100.0,
                            ]);
                        }
                    }
                }
            }
            params
        }
        _ => {
            // Default: simple grid for other strategies
            (0..num_combinations)
                .map(|i| vec![10.0 + i as f64, 20.0 + i as f64])
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_btc_ohlcv() {
        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(100, MarketRegime::Trending, 42);

        assert_eq!(timestamps.len(), 100);
        assert_eq!(open.len(), 100);
        assert_eq!(high.len(), 100);
        assert_eq!(low.len(), 100);
        assert_eq!(close.len(), 100);
        assert_eq!(volume.len(), 100);

        // Verify OHLC constraints
        for i in 0..100 {
            assert!(high[i] >= open[i], "High must be >= open");
            assert!(high[i] >= close[i], "High must be >= close");
            assert!(low[i] <= open[i], "Low must be <= open");
            assert!(low[i] <= close[i], "Low must be <= close");
            assert!(volume[i] > 0.0, "Volume must be positive");
        }
    }

    #[test]
    fn test_generate_options_chain() {
        let params = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(50000.0, 10, 30, &params);

        assert_eq!(options.len(), 20); // 10 calls + 10 puts

        // Verify strike distribution
        let calls: Vec<_> = options
            .iter()
            .filter(|o| o.option_type == OptionType::Call)
            .collect();
        let puts: Vec<_> = options
            .iter()
            .filter(|o| o.option_type == OptionType::Put)
            .collect();

        assert_eq!(calls.len(), 10);
        assert_eq!(puts.len(), 10);

        // Verify strikes span 80-120% of spot
        let min_strike = options.iter().map(|o| o.strike).fold(f64::INFINITY, f64::min);
        let max_strike = options
            .iter()
            .map(|o| o.strike)
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(min_strike >= 40000.0, "Min strike should be ~80% of spot");
        assert!(max_strike <= 60000.0, "Max strike should be ~120% of spot");
    }

    #[test]
    fn test_heston_params_validity() {
        let params_trend = test_heston_params(MarketRegime::Trending);
        let params_range = test_heston_params(MarketRegime::RangeBound);
        let params_vol = test_heston_params(MarketRegime::Volatile);

        // All should satisfy Feller condition: 2 * kappa * theta >= sigma^2
        assert!(2.0 * params_trend.kappa * params_trend.theta >= params_trend.sigma.powi(2));
        assert!(2.0 * params_range.kappa * params_range.theta >= params_range.sigma.powi(2));
        assert!(2.0 * params_vol.kappa * params_vol.theta >= params_vol.sigma.powi(2));
    }

    #[test]
    fn test_generate_strategy_params() {
        use kimsfinance_core::backtest::batch::StrategyType;

        let params_rsi = generate_strategy_params(StrategyType::RsiCrossover, 100);
        assert_eq!(params_rsi.len(), 100);
        assert_eq!(params_rsi[0].len(), 3); // [rsi_period, buy_thresh, sell_thresh]

        let params_straddle = generate_strategy_params(StrategyType::LongStraddle, 100);
        assert_eq!(params_straddle.len(), 100);
        assert_eq!(params_straddle[0].len(), 2); // [vol_threshold, breakeven_pct]
    }
}
