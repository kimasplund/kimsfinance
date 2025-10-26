//! Comprehensive tests for all trading strategies
//!
//! Tests strategy signal generation, parameter validation, and backtesting.

use kimsfinance_core::backtest::{BacktestEngine, Strategy};
use kimsfinance_core::strategies::*;
use ndarray::Array1;

fn generate_test_data(n: usize) -> (Vec<i64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut timestamps = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    let mut price = 100.0;
    for i in 0..n {
        timestamps.push((i * 60) as i64);

        let change = ((i as f64 * 0.1).sin() * 2.0) + (if i % 20 < 10 { 0.5 } else { -0.5 });
        price += change;

        let o = price + (i as f64 * 0.01).sin() * 0.5;
        let c = price - (i as f64 * 0.01).sin() * 0.5;
        let h = price.max(o).max(c) + 0.5;
        let l = price.min(o).min(c) - 0.5;

        open.push(o);
        high.push(h);
        low.push(l);
        close.push(c);
        volume.push(1000.0 + (i as f64 * 100.0));
    }

    (
        timestamps,
        Array1::from_vec(open),
        Array1::from_vec(high),
        Array1::from_vec(low),
        Array1::from_vec(close),
        Array1::from_vec(volume),
    )
}

fn generate_trending_data(n: usize, trend: f64) -> (Vec<i64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut timestamps = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    let mut price = 100.0;
    for i in 0..n {
        timestamps.push((i * 60) as i64);

        price += trend + ((i as f64 * 0.1).sin() * 0.5);

        let o = price;
        let c = price + trend;
        let h = o.max(c) + 0.5;
        let l = o.min(c) - 0.5;

        open.push(o);
        high.push(h);
        low.push(l);
        close.push(c);
        volume.push(1000.0 + (i as f64 * 100.0));
    }

    (
        timestamps,
        Array1::from_vec(open),
        Array1::from_vec(high),
        Array1::from_vec(low),
        Array1::from_vec(close),
        Array1::from_vec(volume),
    )
}

// ============================================================================
// MOMENTUM STRATEGY TESTS
// ============================================================================

#[test]
fn test_rsi_mean_reversion_default() {
    let strategy = momentum::RSIMeanReversion::default();
    assert_eq!(strategy.rsi_period, 14);
    assert_eq!(strategy.buy_threshold, 30.0);
    assert_eq!(strategy.sell_threshold, 50.0);

    let indicators = strategy.indicators();
    assert_eq!(indicators.len(), 1);

    let grid = strategy.parameters();
    assert!(!grid.is_empty());
}

#[test]
fn test_rsi_mean_reversion_backtest() {
    let (timestamps, open, high, low, close, volume) = generate_test_data(500);
    let mut strategy = momentum::RSIMeanReversion::default();
    let engine = BacktestEngine::new();

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.final_equity >= 0.0, true);
    assert!(result.equity_curve.len() > 0);
}

#[test]
fn test_rsi_oversold_overbought() {
    let strategy = momentum::RSIOversoldOverbought::default();
    assert_eq!(strategy.rsi_period, 14);
    assert_eq!(strategy.oversold_threshold, 20.0);
    assert_eq!(strategy.overbought_threshold, 80.0);
}

#[test]
fn test_macd_trend_following() {
    let strategy = momentum::MACDTrendFollowing::default();
    assert_eq!(strategy.fast_period, 12);
    assert_eq!(strategy.slow_period, 26);
    assert_eq!(strategy.signal_period, 9);

    let indicators = strategy.indicators();
    assert_eq!(indicators.len(), 1);
}

#[test]
fn test_macd_divergence() {
    let strategy = momentum::MACDDivergence::default();
    assert_eq!(strategy.fast_period, 12);
    assert_eq!(strategy.histogram_threshold, 0.0);
}

#[test]
fn test_stochastic_oscillator() {
    let strategy = momentum::StochasticOscillator::default();
    assert_eq!(strategy.k_period, 14);
    assert_eq!(strategy.d_period, 3);
    assert_eq!(strategy.oversold_threshold, 20.0);
    assert_eq!(strategy.overbought_threshold, 80.0);
}

#[test]
fn test_roc_breakout() {
    let strategy = momentum::ROCBreakout::default();
    assert_eq!(strategy.roc_period, 12);
    assert_eq!(strategy.buy_threshold, 2.0);
    assert_eq!(strategy.sell_threshold, -2.0);
}

#[test]
fn test_cci_reversal() {
    let strategy = momentum::CCIReversal::default();
    assert_eq!(strategy.cci_period, 20);
    assert_eq!(strategy.oversold_threshold, -100.0);
    assert_eq!(strategy.overbought_threshold, 100.0);
}

// ============================================================================
// TREND STRATEGY TESTS
// ============================================================================

#[test]
fn test_ema_crossover_default() {
    let strategy = trend::EMACrossover::default();
    assert_eq!(strategy.fast_period, 50);
    assert_eq!(strategy.slow_period, 200);

    let indicators = strategy.indicators();
    assert_eq!(indicators.len(), 2);
}

#[test]
fn test_ema_crossover_backtest_trending() {
    let (timestamps, open, high, low, close, volume) = generate_trending_data(300, 0.2);
    let mut strategy = trend::EMACrossover::new(10, 50);
    let engine = BacktestEngine::new();

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.num_trades > 0);
}

#[test]
fn test_triple_ema_trend() {
    let strategy = trend::TripleEMATrend::default();
    assert_eq!(strategy.short_period, 8);
    assert_eq!(strategy.medium_period, 21);
    assert_eq!(strategy.long_period, 55);

    let indicators = strategy.indicators();
    assert_eq!(indicators.len(), 3);
}

#[test]
fn test_donchian_breakout() {
    let strategy = trend::DonchianBreakout::default();
    assert_eq!(strategy.channel_period, 20);
}

#[test]
fn test_keltner_trend() {
    let strategy = trend::KeltnerTrend::default();
    assert_eq!(strategy.ema_period, 20);
    assert_eq!(strategy.atr_period, 10);
    assert_eq!(strategy.atr_multiplier, 2.0);
}

// ============================================================================
// VOLATILITY STRATEGY TESTS
// ============================================================================

#[test]
fn test_bollinger_squeeze_default() {
    let strategy = volatility::BollingerBandsSqueeze::default();
    assert_eq!(strategy.period, 20);
    assert_eq!(strategy.std_dev, 2.0);
    assert_eq!(strategy.squeeze_threshold, 0.05);
}

#[test]
fn test_bollinger_expansion() {
    let strategy = volatility::BollingerBandsExpansion::default();
    assert_eq!(strategy.period, 20);
    assert_eq!(strategy.std_dev, 2.0);
    assert_eq!(strategy.exit_at_middle, true);
}

#[test]
fn test_atr_volatility_breakout() {
    let strategy = volatility::ATRVolatilityBreakout::default();
    assert_eq!(strategy.atr_period, 14);
    assert_eq!(strategy.breakout_multiplier, 2.0);
    assert_eq!(strategy.min_atr_pct, 0.005);
}

// ============================================================================
// COMPOSITE STRATEGY TESTS
// ============================================================================

#[test]
fn test_rsi_with_atr_default() {
    let strategy = composite::RSIWithATR::default();
    assert_eq!(strategy.rsi_period, 14);
    assert_eq!(strategy.atr_period, 14);
    assert_eq!(strategy.min_atr_pct, 0.005);

    let indicators = strategy.indicators();
    assert_eq!(indicators.len(), 2);
}

#[test]
fn test_rsi_with_atr_backtest() {
    let (timestamps, open, high, low, close, volume) = generate_test_data(500);
    let mut strategy = composite::RSIWithATR::default();
    let engine = BacktestEngine::new();

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(result.is_ok());
}

#[test]
fn test_macd_with_ema() {
    let strategy = composite::MACDWithEMA::default();
    assert_eq!(strategy.macd_fast, 12);
    assert_eq!(strategy.macd_slow, 26);
    assert_eq!(strategy.trend_ema_period, 200);

    let indicators = strategy.indicators();
    assert_eq!(indicators.len(), 2);
}

#[test]
fn test_bollinger_with_stochastic() {
    let strategy = composite::BollingerWithStochastic::default();
    assert_eq!(strategy.bb_period, 20);
    assert_eq!(strategy.stoch_k_period, 14);

    let indicators = strategy.indicators();
    assert_eq!(indicators.len(), 2);
}

#[test]
fn test_triple_confirmation() {
    let strategy = composite::TripleConfirmation::default();
    assert_eq!(strategy.rsi_period, 14);
    assert_eq!(strategy.macd_fast, 12);
    assert_eq!(strategy.ema_period, 50);

    let indicators = strategy.indicators();
    assert_eq!(indicators.len(), 3);
}

#[test]
fn test_volatility_momentum() {
    let strategy = composite::VolatilityMomentum::default();
    assert_eq!(strategy.atr_period, 14);
    assert_eq!(strategy.roc_period, 12);
    assert_eq!(strategy.min_atr_pct, 0.005);

    let indicators = strategy.indicators();
    assert_eq!(indicators.len(), 2);
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

#[test]
fn test_all_momentum_strategies_backtest() {
    let (timestamps, open, high, low, close, volume) = generate_test_data(400);
    let engine = BacktestEngine::new();

    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(momentum::RSIMeanReversion::default()),
        Box::new(momentum::RSIOversoldOverbought::default()),
        Box::new(momentum::MACDTrendFollowing::default()),
        Box::new(momentum::StochasticOscillator::default()),
        Box::new(momentum::ROCBreakout::default()),
        Box::new(momentum::CCIReversal::default()),
    ];

    for mut strategy in strategies {
        let result = engine.run(
            strategy.as_mut(),
            &timestamps,
            &open,
            &high,
            &low,
            &close,
            &volume,
        );
        assert!(result.is_ok(), "Strategy failed: {:?}", result.err());
    }
}

#[test]
fn test_all_trend_strategies_backtest() {
    let (timestamps, open, high, low, close, volume) = generate_trending_data(400, 0.1);
    let engine = BacktestEngine::new();

    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(trend::EMACrossover::new(20, 50)),
        Box::new(trend::TripleEMATrend::default()),
        Box::new(trend::DonchianBreakout::default()),
        Box::new(trend::KeltnerTrend::default()),
    ];

    for mut strategy in strategies {
        let result = engine.run(
            strategy.as_mut(),
            &timestamps,
            &open,
            &high,
            &low,
            &close,
            &volume,
        );
        assert!(result.is_ok(), "Strategy failed: {:?}", result.err());
    }
}

#[test]
fn test_all_volatility_strategies_backtest() {
    let (timestamps, open, high, low, close, volume) = generate_test_data(400);
    let engine = BacktestEngine::new();

    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(volatility::BollingerBandsSqueeze::default()),
        Box::new(volatility::BollingerBandsExpansion::default()),
        Box::new(volatility::ATRVolatilityBreakout::default()),
    ];

    for mut strategy in strategies {
        let result = engine.run(
            strategy.as_mut(),
            &timestamps,
            &open,
            &high,
            &low,
            &close,
            &volume,
        );
        assert!(result.is_ok(), "Strategy failed: {:?}", result.err());
    }
}

#[test]
fn test_all_composite_strategies_backtest() {
    let (timestamps, open, high, low, close, volume) = generate_test_data(500);
    let engine = BacktestEngine::new();

    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(composite::RSIWithATR::default()),
        Box::new(composite::MACDWithEMA::default()),
        Box::new(composite::BollingerWithStochastic::default()),
        Box::new(composite::TripleConfirmation::default()),
        Box::new(composite::VolatilityMomentum::default()),
    ];

    for mut strategy in strategies {
        let result = engine.run(
            strategy.as_mut(),
            &timestamps,
            &open,
            &high,
            &low,
            &close,
            &volume,
        );
        assert!(result.is_ok(), "Strategy failed: {:?}", result.err());
    }
}

#[test]
fn test_parameter_grid_coverage() {
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(momentum::RSIMeanReversion::default()),
        Box::new(trend::EMACrossover::default()),
        Box::new(volatility::BollingerBandsSqueeze::default()),
        Box::new(composite::RSIWithATR::default()),
    ];

    for strategy in strategies {
        let grid = strategy.parameters();
        assert!(grid.size() > 1, "Strategy should have optimization parameters");
    }
}
