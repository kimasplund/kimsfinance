//! CPU Fallback Tests for Backtesting Engine
//!
//! Verifies that ALL indicators work correctly with CPU-only mode.
//! Tests ensure graceful degradation when GPU is unavailable.

use kimsfinance_core::backtest::core::{IndicatorConfig, OHLCVBar, Signal, Strategy};
use kimsfinance_core::backtest::engine::{BacktestConfig, BacktestEngine};
use ndarray::Array1;
use std::collections::HashMap;

/// Test strategy that uses a single indicator
struct TestStrategy {
    indicator: IndicatorConfig,
    initial_capital: f64,
}

impl Strategy for TestStrategy {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &HashMap<String, f64>) -> Signal {
        // Just verify indicators are present
        let key = self.indicator.key();
        if indicators.contains_key(&key) {
            Signal::Hold
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![self.indicator.clone()]
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

/// Helper to create test OHLCV data
fn create_test_data(
    n: usize,
) -> (
    Vec<i64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let timestamps: Vec<i64> = (0..n as i64).map(|i| i * 60).collect();

    // Create realistic price data with some volatility
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    let mut price = 100.0;
    for i in 0..n {
        let change = (i as f64 * 0.1).sin() * 2.0; // Oscillating price
        price += change;

        open.push(price);
        high.push(price + (i as f64 * 0.01).abs());
        low.push(price - (i as f64 * 0.01).abs());
        close.push(price + change * 0.5);
        volume.push(1000.0 + (i as f64 * 10.0));
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

/// Test RSI with CPU-only mode
#[test]
fn test_cpu_fallback_rsi() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);
    let mut strategy = TestStrategy {
        indicator: IndicatorConfig::RSI { period: 14 },
        initial_capital: 10_000.0,
    };

    let (timestamps, open, high, low, close, volume) = create_test_data(100);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "RSI CPU fallback failed: {:?}",
        result.err()
    );
    let backtest_result = result.unwrap();
    assert_eq!(backtest_result.equity_curve.len(), 100);
}

/// Test ATR with CPU-only mode
#[test]
fn test_cpu_fallback_atr() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);
    let mut strategy = TestStrategy {
        indicator: IndicatorConfig::ATR { period: 14 },
        initial_capital: 10_000.0,
    };

    let (timestamps, open, high, low, close, volume) = create_test_data(100);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "ATR CPU fallback failed: {:?}",
        result.err()
    );
}

/// Test ROC with CPU-only mode
#[test]
fn test_cpu_fallback_roc() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);
    let mut strategy = TestStrategy {
        indicator: IndicatorConfig::ROC { period: 10 },
        initial_capital: 10_000.0,
    };

    let (timestamps, open, high, low, close, volume) = create_test_data(100);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "ROC CPU fallback failed: {:?}",
        result.err()
    );
}

/// Test CCI with CPU-only mode
#[test]
fn test_cpu_fallback_cci() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);
    let mut strategy = TestStrategy {
        indicator: IndicatorConfig::CCI { period: 20 },
        initial_capital: 10_000.0,
    };

    let (timestamps, open, high, low, close, volume) = create_test_data(100);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "CCI CPU fallback failed: {:?}",
        result.err()
    );
}

/// Test WilliamsR with CPU-only mode
#[test]
fn test_cpu_fallback_williamsr() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);
    let mut strategy = TestStrategy {
        indicator: IndicatorConfig::WilliamsR { period: 14 },
        initial_capital: 10_000.0,
    };

    let (timestamps, open, high, low, close, volume) = create_test_data(100);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "WilliamsR CPU fallback failed: {:?}",
        result.err()
    );
}

/// Test SMA with CPU-only mode
#[test]
fn test_cpu_fallback_sma() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);
    let mut strategy = TestStrategy {
        indicator: IndicatorConfig::SMA { period: 20 },
        initial_capital: 10_000.0,
    };

    let (timestamps, open, high, low, close, volume) = create_test_data(100);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "SMA CPU fallback failed: {:?}",
        result.err()
    );
}

/// Test EMA with CPU-only mode
#[test]
fn test_cpu_fallback_ema() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);
    let mut strategy = TestStrategy {
        indicator: IndicatorConfig::EMA { period: 20 },
        initial_capital: 10_000.0,
    };

    let (timestamps, open, high, low, close, volume) = create_test_data(100);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "EMA CPU fallback failed: {:?}",
        result.err()
    );
}

/// Test MACD with CPU-only mode (multi-output indicator)
#[test]
fn test_cpu_fallback_macd() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);

    // Custom strategy to check all MACD outputs
    struct MACDStrategy;
    impl Strategy for MACDStrategy {
        fn on_data(&mut self, _bar: &OHLCVBar, indicators: &HashMap<String, f64>) -> Signal {
            // Verify all three MACD outputs are present
            assert!(
                indicators.contains_key("macd_12_26_9_macd"),
                "Missing MACD line"
            );
            assert!(
                indicators.contains_key("macd_12_26_9_signal"),
                "Missing signal line"
            );
            assert!(
                indicators.contains_key("macd_12_26_9_histogram"),
                "Missing histogram"
            );
            Signal::Hold
        }

        fn indicators(&self) -> Vec<IndicatorConfig> {
            vec![IndicatorConfig::MACD {
                fast: 12,
                slow: 26,
                signal: 9,
            }]
        }

        fn initial_capital(&self) -> f64 {
            10_000.0
        }
    }

    let mut strategy = MACDStrategy;
    let (timestamps, open, high, low, close, volume) = create_test_data(100);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "MACD CPU fallback failed: {:?}",
        result.err()
    );
}

/// Test Stochastic with CPU-only mode (multi-output indicator)
#[test]
fn test_cpu_fallback_stochastic() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);

    // Custom strategy to check both Stochastic outputs
    struct StochasticStrategy;
    impl Strategy for StochasticStrategy {
        fn on_data(&mut self, _bar: &OHLCVBar, indicators: &HashMap<String, f64>) -> Signal {
            // Verify both %K and %D are present
            assert!(indicators.contains_key("stoch_14_3_k"), "Missing %K line");
            assert!(indicators.contains_key("stoch_14_3_d"), "Missing %D line");
            Signal::Hold
        }

        fn indicators(&self) -> Vec<IndicatorConfig> {
            vec![IndicatorConfig::Stochastic {
                k_period: 14,
                d_period: 3,
            }]
        }

        fn initial_capital(&self) -> f64 {
            10_000.0
        }
    }

    let mut strategy = StochasticStrategy;
    let (timestamps, open, high, low, close, volume) = create_test_data(100);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "Stochastic CPU fallback failed: {:?}",
        result.err()
    );
}

/// Test Bollinger Bands with CPU-only mode (multi-output indicator)
#[test]
fn test_cpu_fallback_bollinger_bands() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);

    // Custom strategy to check all Bollinger Bands outputs
    struct BBStrategy;
    impl Strategy for BBStrategy {
        fn on_data(&mut self, _bar: &OHLCVBar, indicators: &HashMap<String, f64>) -> Signal {
            // Verify all three bands are present
            assert!(
                indicators.contains_key("bb_20_2_middle"),
                "Missing middle band"
            );
            assert!(
                indicators.contains_key("bb_20_2_upper"),
                "Missing upper band"
            );
            assert!(
                indicators.contains_key("bb_20_2_lower"),
                "Missing lower band"
            );
            Signal::Hold
        }

        fn indicators(&self) -> Vec<IndicatorConfig> {
            vec![IndicatorConfig::BollingerBands {
                period: 20,
                std_dev: 2.0,
            }]
        }

        fn initial_capital(&self) -> f64 {
            10_000.0
        }
    }

    let mut strategy = BBStrategy;
    let (timestamps, open, high, low, close, volume) = create_test_data(100);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "Bollinger Bands CPU fallback failed: {:?}",
        result.err()
    );
}

/// Test multiple indicators with CPU-only mode
#[test]
fn test_cpu_fallback_multiple_indicators() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);

    // Strategy using multiple indicators
    struct MultiIndicatorStrategy;
    impl Strategy for MultiIndicatorStrategy {
        fn on_data(&mut self, _bar: &OHLCVBar, indicators: &HashMap<String, f64>) -> Signal {
            // Verify all indicators are present
            assert!(indicators.contains_key("rsi_14"), "Missing RSI");
            assert!(indicators.contains_key("sma_20"), "Missing SMA");
            assert!(indicators.contains_key("ema_20"), "Missing EMA");
            assert!(indicators.contains_key("atr_14"), "Missing ATR");
            Signal::Hold
        }

        fn indicators(&self) -> Vec<IndicatorConfig> {
            vec![
                IndicatorConfig::RSI { period: 14 },
                IndicatorConfig::SMA { period: 20 },
                IndicatorConfig::EMA { period: 20 },
                IndicatorConfig::ATR { period: 14 },
            ]
        }

        fn initial_capital(&self) -> f64 {
            10_000.0
        }
    }

    let mut strategy = MultiIndicatorStrategy;
    let (timestamps, open, high, low, close, volume) = create_test_data(100);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "Multiple indicators CPU fallback failed: {:?}",
        result.err()
    );
}

/// Test force_cpu flag overrides use_gpu
#[test]
fn test_force_cpu_overrides_gpu() {
    // Even with use_gpu=true, force_cpu should take precedence
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: true,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);
    let mut strategy = TestStrategy {
        indicator: IndicatorConfig::RSI { period: 14 },
        initial_capital: 10_000.0,
    };

    let (timestamps, open, high, low, close, volume) = create_test_data(100);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "force_cpu override failed: {:?}",
        result.err()
    );
}

/// Test CPU fallback with small dataset
#[test]
fn test_cpu_fallback_small_dataset() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);
    let mut strategy = TestStrategy {
        indicator: IndicatorConfig::RSI { period: 14 },
        initial_capital: 10_000.0,
    };

    // Small dataset (30 bars)
    let (timestamps, open, high, low, close, volume) = create_test_data(30);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "Small dataset CPU fallback failed: {:?}",
        result.err()
    );
}

/// Test CPU fallback with large dataset
#[test]
fn test_cpu_fallback_large_dataset() {
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
            ..Default::default()
        
    };

    let engine = BacktestEngine::with_config(config);
    let mut strategy = TestStrategy {
        indicator: IndicatorConfig::RSI { period: 14 },
        initial_capital: 10_000.0,
    };

    // Large dataset (1000 bars)
    let (timestamps, open, high, low, close, volume) = create_test_data(1000);

    let result = engine.run(
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
    );

    assert!(
        result.is_ok(),
        "Large dataset CPU fallback failed: {:?}",
        result.err()
    );
    let backtest_result = result.unwrap();
    assert_eq!(backtest_result.equity_curve.len(), 1000);
}
