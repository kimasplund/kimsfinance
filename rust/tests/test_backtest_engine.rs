//! Integration test for backtesting engine
//!
//! Tests a simple RSI crossover strategy:
//! - Buy when RSI < 30 (oversold)
//! - Sell when RSI > 70 (overbought)

use kimsfinance_core::backtest::{
    BacktestConfig, BacktestEngine, IndicatorConfig, IndicatorValues, OHLCVBar, Signal, Strategy,
};
use ndarray::Array1;

/// Simple RSI crossover strategy
struct RSIStrategy {
    rsi_period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
}

impl Strategy for RSIStrategy {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi_key = format!("rsi_{}", self.rsi_period);

        let rsi = indicators.get(&rsi_key).copied().unwrap_or(50.0);

        // Skip NaN values (warmup period)
        if rsi.is_nan() {
            return Signal::Hold;
        }

        if rsi < self.buy_threshold {
            Signal::Buy
        } else if rsi > self.sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::RSI {
            period: self.rsi_period,
        }]
    }

    fn initial_capital(&self) -> f64 {
        10_000.0
    }
}

#[test]
fn test_rsi_strategy_cpu_only() {
    // Generate synthetic OHLCV data (trending upward)
    let n = 100;
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);
    let mut timestamps = Vec::with_capacity(n);

    let base_price = 100.0;
    for i in 0..n {
        let t = i as f64;
        // Create oscillating price (sine wave) to generate RSI crossovers
        let price = base_price + (t * 0.3).sin() * 20.0; // Large oscillation
        let spread = 2.0;

        timestamps.push(i as i64);
        high.push(price + spread);
        low.push(price - spread);
        open.push(price - spread * 0.5);
        close.push(price + spread * 0.5);
        volume.push(1000.0 + (t * 0.2).sin() * 200.0);
    }

    let high = Array1::from_vec(high);
    let low = Array1::from_vec(low);
    let open = Array1::from_vec(open);
    let close = Array1::from_vec(close);
    let volume = Array1::from_vec(volume);

    // Create strategy (wider thresholds to ensure trades)
    let mut strategy = RSIStrategy {
        rsi_period: 14,
        buy_threshold: 40.0,  // Buy when RSI < 40
        sell_threshold: 60.0, // Sell when RSI > 60
    };

    // Create backtesting engine (CPU-only)
    let config = BacktestConfig {
        use_gpu: false, // Force CPU-only
        ..Default::default()
    };
    let engine = BacktestEngine::with_config(config);

    // Run backtest
    let result = engine
        .run(
            &mut strategy,
            &timestamps,
            &open,
            &high,
            &low,
            &close,
            &volume,
        )
        .expect("Backtest failed");

    // Verify results
    println!("Backtest Results:");
    println!("  Initial Capital: $10,000");
    println!("  Final Equity: ${:.2}", result.final_equity);
    println!("  Total Return: {:.2}%", result.total_return);
    println!("  Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", result.max_drawdown);
    println!("  Win Rate: {:.2}%", result.win_rate);
    println!("  Number of Trades: {}", result.num_trades);
    println!("  Profit Factor: {:.2}", result.profit_factor);

    // Basic sanity checks
    assert!(
        result.equity_curve.len() == n,
        "Equity curve length mismatch"
    );
    assert!(result.final_equity > 0.0, "Final equity should be positive");
    assert!(
        result.num_trades > 0,
        "Should have executed at least one trade"
    );

    // Equity curve should never be negative (position sizing is 100%)
    for equity in &result.equity_curve {
        assert!(*equity > 0.0, "Equity should never go negative");
    }

    println!("\nTest passed! Backtesting engine works correctly.");
}

#[cfg(feature = "gpu")]
#[test]
fn test_rsi_strategy_gpu() {
    use kimsfinance_core::gpu::GpuDevice;

    // Try to initialize GPU
    if GpuDevice::new().is_err() {
        println!("GPU not available, skipping GPU test");
        return;
    }

    // Generate synthetic OHLCV data (trending upward)
    let n = 1000; // Larger dataset for GPU
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);
    let mut timestamps = Vec::with_capacity(n);

    let base_price = 100.0;
    for i in 0..n {
        let t = i as f64;
        let price = base_price + t * 0.5 + (t * 0.1).sin() * 5.0;
        let spread = 2.0;

        timestamps.push(i as i64);
        high.push(price + spread);
        low.push(price - spread);
        open.push(price - spread * 0.5);
        close.push(price + spread * 0.5);
        volume.push(1000.0 + (t * 0.2).sin() * 200.0);
    }

    let high = Array1::from_vec(high);
    let low = Array1::from_vec(low);
    let open = Array1::from_vec(open);
    let close = Array1::from_vec(close);
    let volume = Array1::from_vec(volume);

    // Create strategy (wider thresholds to ensure trades)
    let mut strategy = RSIStrategy {
        rsi_period: 14,
        buy_threshold: 40.0,  // Buy when RSI < 40
        sell_threshold: 60.0, // Sell when RSI > 60
    };

    // Create backtesting engine (GPU-enabled)
    let config = BacktestConfig {
        use_gpu: true,
        ..Default::default()
    };
    let engine = BacktestEngine::with_config(config);

    // Run backtest
    let result = engine
        .run(
            &mut strategy,
            &timestamps,
            &open,
            &high,
            &low,
            &close,
            &volume,
        )
        .expect("GPU backtest failed");

    // Verify results
    println!("GPU Backtest Results:");
    println!("  Final Equity: ${:.2}", result.final_equity);
    println!("  Total Return: {:.2}%", result.total_return);
    println!("  Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", result.max_drawdown);
    println!("  Number of Trades: {}", result.num_trades);

    assert!(result.equity_curve.len() == n);
    assert!(result.final_equity > 0.0);
    assert!(result.num_trades > 0);

    println!("\nGPU test passed!");
}
