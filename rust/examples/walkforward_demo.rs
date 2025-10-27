//! Walk-Forward Analysis Demo
//!
//! Demonstrates walk-forward analysis for out-of-sample testing and overfitting detection.
//!
//! # Features Demonstrated
//!
//! - Rolling window walk-forward analysis
//! - Train/test split optimization
//! - Overfitting detection
//! - Parameter stability analysis
//! - In-sample vs out-of-sample comparison
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example walkforward_demo
//! ```

use kimsfinance_core::backtest::{
    BacktestEngine, IndicatorConfig, IndicatorValues, OHLCVBar, ParameterGrid, ParameterRange,
    Signal, Strategy, WalkForwardAnalyzer, WalkForwardConfig,
};
use ndarray::Array1;
use std::collections::HashMap;

/// Simple RSI strategy for demonstration
#[derive(Debug, Clone)]
struct RSIStrategy {
    rsi_period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
}

impl Strategy for RSIStrategy {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi_key = format!("rsi_{}", self.rsi_period);
        let rsi = match indicators.get(&rsi_key) {
            Some(&value) if !value.is_nan() => value,
            _ => return Signal::Hold,
        };

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

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "rsi_period",
            ParameterRange::Int {
                min: 10,
                max: 20,
                step: 2,
            },
        );
        grid.add_range(
            "buy_threshold",
            ParameterRange::Float {
                min: 20.0,
                max: 40.0,
                step: 5.0,
            },
        );
        grid.add_range(
            "sell_threshold",
            ParameterRange::Float {
                min: 60.0,
                max: 80.0,
                step: 5.0,
            },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        10_000.0
    }
}

/// Generate synthetic OHLCV data for demonstration
fn generate_test_data(
    n: usize,
) -> (
    Vec<i64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let mut timestamps = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    let mut price = 100.0;
    let start_time = 1_600_000_000i64;

    for i in 0..n {
        timestamps.push(start_time + (i as i64 * 3600));

        // Generate price movement (simplified)
        let change = ((i as f64 * 0.1).sin() * 2.0) + (((i as f64) * 0.05).cos() * 1.0);
        price += change;

        let o = price;
        let h = price + (price * 0.01);
        let l = price - (price * 0.01);
        let c = price + (change * 0.5);

        open.push(o);
        high.push(h);
        low.push(l);
        close.push(c);
        volume.push(1000.0 + (i as f64 * 10.0));

        price = c;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Walk-Forward Analysis Demo ===\n");

    // Generate synthetic data (1 year of hourly data)
    let n = 365 * 24;
    let (timestamps, open, high, low, close, volume) = generate_test_data(n);

    println!("Generated {} bars of synthetic data", n);

    // Create backtest engine
    let engine = BacktestEngine::new();

    // Create strategy
    let mut strategy = RSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    // Get parameter grid
    let param_grid = strategy.parameters();

    println!("Parameter grid size: {} combinations\n", param_grid.size());

    // Configure walk-forward analysis
    let wf_config = WalkForwardConfig {
        train_window: 252 * 24, // 1 year training (252 trading days * 24 hours)
        test_window: 63 * 24,   // 1 quarter testing (63 days * 24 hours)
        step_size: 21 * 24,     // 1 month step (21 days * 24 hours)
        anchored: false,        // Rolling window (not expanding)
        min_bars: 300 * 24,     // Minimum 300 days required
    };

    println!("Walk-Forward Configuration:");
    println!(
        "  Train window: {} bars ({} days)",
        wf_config.train_window,
        wf_config.train_window / 24
    );
    println!(
        "  Test window: {} bars ({} days)",
        wf_config.test_window,
        wf_config.test_window / 24
    );
    println!(
        "  Step size: {} bars ({} days)",
        wf_config.step_size,
        wf_config.step_size / 24
    );
    println!(
        "  Mode: {}",
        if wf_config.anchored {
            "Anchored (expanding)"
        } else {
            "Rolling"
        }
    );
    println!("  Expected windows: {}\n", wf_config.num_splits(n));

    // Create analyzer
    let analyzer = WalkForwardAnalyzer::new(wf_config);

    // Run walk-forward analysis
    println!("Running walk-forward analysis...\n");

    let result = analyzer.analyze(
        &engine,
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &param_grid,
    )?;

    // Display results
    println!("\n=== Walk-Forward Results ===\n");
    println!("{}", result.summary());

    println!("\n=== Window Details ===");
    for (i, window) in result.windows.iter().enumerate() {
        println!("\nWindow {}:", i + 1);
        println!("  Train: bars {}-{}", window.train_start, window.train_end);
        println!("  Test:  bars {}-{}", window.test_start, window.test_end);
        println!("  Optimized Parameters:");
        for (key, value) in &window.optimized_params {
            println!("    {}: {:.2}", key, value);
        }
        println!("  In-Sample:");
        println!("    Sharpe: {:.3}", window.in_sample_result.sharpe_ratio);
        println!("    Return: {:.2}%", window.in_sample_result.total_return);
        println!("    Max DD: {:.2}%", window.in_sample_result.max_drawdown);
        println!("  Out-of-Sample:");
        println!(
            "    Sharpe: {:.3}",
            window.out_of_sample_result.sharpe_ratio
        );
        println!(
            "    Return: {:.2}%",
            window.out_of_sample_result.total_return
        );
        println!(
            "    Max DD: {:.2}%",
            window.out_of_sample_result.max_drawdown
        );
        println!("  Efficiency Ratio: {:.3}", window.efficiency_ratio());

        let prev = if i > 0 {
            Some(&result.windows[i - 1])
        } else {
            None
        };
        println!("  Stability Score: {:.3}", window.stability_score(prev));
    }

    println!("\n=== Overfitting Analysis ===");
    println!("Overfitting detected: {}", result.is_overfitted());
    println!(
        "Performance degradation: {:.1}%",
        result.degradation_percent
    );

    if result.efficiency_ratio > 0.8 {
        println!("\nStatus: EXCELLENT - Strategy generalizes well to unseen data");
    } else if result.efficiency_ratio > 0.6 {
        println!("\nStatus: GOOD - Acceptable out-of-sample performance");
    } else if result.efficiency_ratio > 0.4 {
        println!("\nStatus: FAIR - Some overfitting detected");
    } else {
        println!("\nStatus: POOR - Significant overfitting, strategy may not work in live trading");
    }

    if result.avg_stability < 0.5 {
        println!(
            "\nWARNING: Low parameter stability - parameters change significantly between windows"
        );
        println!("This suggests the strategy may be curve-fitting to market noise.");
    }

    println!("\n=== Recommendations ===");
    if result.is_overfitted() {
        println!("- Consider simplifying the strategy to reduce overfitting");
        println!("- Increase out-of-sample testing period");
        println!("- Add transaction costs and slippage to make testing more realistic");
        println!("- Try parameter averaging instead of optimization on each window");
    } else {
        println!("- Strategy shows good robustness across time periods");
        println!("- Consider forward testing with paper trading before live deployment");
        println!("- Monitor performance metrics regularly for degradation");
    }

    Ok(())
}
