//! Integration test for GPU parameter sweep
//!
//! Tests the 3D GPU kernel integration for parameter optimization:
//! - Multiple RSI periods (10, 12, 14, 16, 18, 20)
//! - Multiple buy/sell thresholds (20-40, 60-80)
//! - Verifies GPU sweep returns correct number of results
//! - Compares best result against individual backtests

use kimsfinance_core::backtest::{
    BacktestConfig, BacktestEngine, IndicatorConfig, IndicatorValues, OHLCVBar, ParameterGrid,
    ParameterRange, Signal, Strategy,
};
use ndarray::Array1;

/// RSI crossover strategy with configurable parameters
struct RSIStrategy {
    rsi_period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
}

impl Strategy for RSIStrategy {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi_key = format!("rsi_{}", self.rsi_period);
        let rsi = indicators.get(&rsi_key).copied().unwrap_or(50.0);

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

/// Generate synthetic OHLCV data with oscillating price pattern
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
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    let base_price = 100.0;
    for i in 0..n {
        let t = i as f64;
        // Create oscillating price (sine wave) to generate RSI crossovers
        let price = base_price + (t * 0.3).sin() * 20.0; // Large oscillation for RSI signals
        let spread = 2.0;

        timestamps.push(i as i64);
        high.push(price + spread);
        low.push(price - spread);
        open.push(price - spread * 0.5);
        close.push(price + spread * 0.5);
        volume.push(1000.0 + (t * 0.2).sin() * 200.0);
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

#[cfg(feature = "gpu")]
#[test]
fn test_parameter_sweep_gpu() {
    use kimsfinance_core::gpu::GpuDevice;

    // Check GPU availability
    if GpuDevice::new().is_err() {
        println!("GPU not available, skipping GPU parameter sweep test");
        return;
    }

    // Generate test data (500 bars for faster testing)
    let (timestamps, open, high, low, close, volume) = generate_test_data(500);

    // Create parameter grid
    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 2,
        },
    ); // 6 periods: 10, 12, 14, 16, 18, 20
    grid.add_range(
        "buy_threshold",
        ParameterRange::Float {
            min: 25.0,
            max: 35.0,
            step: 5.0,
        },
    ); // 3 thresholds: 25, 30, 35

    let total_combinations = grid.size();
    assert_eq!(
        total_combinations, 18,
        "Expected 6 periods × 3 thresholds = 18"
    );

    // Create engine with GPU enabled
    let config = BacktestConfig {
        use_gpu: true,
        ..Default::default()
    };
    let engine = BacktestEngine::with_config(config);

    // Create base strategy (parameters will be overridden by sweep)
    let mut strategy = RSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    // Run parameter sweep
    println!(
        "Running GPU parameter sweep with {} combinations...",
        total_combinations
    );
    let results = engine
        .run_sweep(
            &mut strategy,
            &timestamps,
            &open,
            &high,
            &low,
            &close,
            &volume,
            &grid,
        )
        .expect("GPU parameter sweep failed");

    // Verify we got results for all combinations
    assert_eq!(
        results.len(),
        total_combinations,
        "Should return result for each parameter combination"
    );

    // Verify results are sorted by fitness (best first)
    for i in 1..results.len() {
        assert!(
            results[i - 1].fitness() >= results[i].fitness(),
            "Results should be sorted by fitness score (descending)"
        );
    }

    // Print top 5 results
    println!("\nTop 5 Parameter Combinations:");
    for (i, result) in results.iter().take(5).enumerate() {
        println!(
            "{}. RSI Period: {}, Buy Threshold: {:.1}",
            i + 1,
            result.parameters.get("rsi_period").unwrap(),
            result.parameters.get("buy_threshold").unwrap()
        );
        println!(
            "   Sharpe: {:.2}, Max DD: {:.2}%, Trades: {}, Fitness: {:.4}",
            result.sharpe_ratio,
            result.max_drawdown,
            result.num_trades,
            result.fitness()
        );
    }

    // Verify all results have valid parameters
    for result in &results {
        assert!(result.parameters.contains_key("rsi_period"));
        assert!(result.parameters.contains_key("buy_threshold"));

        let period = result.parameters["rsi_period"] as usize;
        assert!(
            (10..=20).contains(&period) && period % 2 == 0,
            "Invalid RSI period: {}",
            period
        );

        let threshold = result.parameters["buy_threshold"];
        assert!(
            (25.0..=35.0).contains(&threshold),
            "Invalid buy threshold: {}",
            threshold
        );
    }

    // Verify equity curves are valid
    for result in &results {
        assert_eq!(
            result.equity_curve.len(),
            timestamps.len(),
            "Equity curve length should match data length"
        );

        // All equity values should be positive (we're using full position sizing)
        for (i, &equity) in result.equity_curve.iter().enumerate() {
            assert!(
                equity > 0.0,
                "Equity at index {} should be positive, got {}",
                i,
                equity
            );
        }
    }

    println!("\nGPU parameter sweep test passed!");
    println!(
        "Best result: Sharpe = {:.2}, Fitness = {:.4}",
        results[0].sharpe_ratio,
        results[0].fitness()
    );
}

#[test]
fn test_parameter_sweep_cpu() {
    // Generate smaller test data for CPU (faster)
    let (timestamps, open, high, low, close, volume) = generate_test_data(200);

    // Create smaller parameter grid for CPU test
    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 12,
            max: 16,
            step: 2,
        },
    ); // 3 periods: 12, 14, 16
    grid.add_range(
        "buy_threshold",
        ParameterRange::Float {
            min: 30.0,
            max: 35.0,
            step: 5.0,
        },
    ); // 2 thresholds: 30, 35

    let total_combinations = grid.size();
    assert_eq!(
        total_combinations, 6,
        "Expected 3 periods × 2 thresholds = 6"
    );

    // Create engine (will use CPU if no GPU)
    let config = BacktestConfig {
        use_gpu: false, // Force CPU
        ..Default::default()
    };
    let engine = BacktestEngine::with_config(config);

    // Create base strategy
    let mut strategy = RSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    // Run parameter sweep
    println!(
        "Running CPU parameter sweep with {} combinations...",
        total_combinations
    );
    let results = engine
        .run_sweep(
            &mut strategy,
            &timestamps,
            &open,
            &high,
            &low,
            &close,
            &volume,
            &grid,
        )
        .expect("CPU parameter sweep failed");

    // Verify we got results
    assert_eq!(
        results.len(),
        total_combinations,
        "Should return result for each parameter combination"
    );

    // Verify results are sorted
    for i in 1..results.len() {
        assert!(
            results[i - 1].fitness() >= results[i].fitness(),
            "Results should be sorted by fitness score"
        );
    }

    println!("\nCPU parameter sweep test passed!");
    println!(
        "Best parameters: period={}, threshold={:.1}",
        results[0].parameters.get("rsi_period").unwrap(),
        results[0].parameters.get("buy_threshold").unwrap()
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_sweep_vs_individual_backtest() {
    use kimsfinance_core::gpu::GpuDevice;

    // Check GPU availability
    if GpuDevice::new().is_err() {
        println!("GPU not available, skipping comparison test");
        return;
    }

    // Generate test data
    let (timestamps, open, high, low, close, volume) = generate_test_data(300);

    // Create single-parameter grid (just RSI period)
    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 14,
            max: 14,
            step: 1,
        },
    ); // Only period 14

    let config = BacktestConfig {
        use_gpu: true,
        ..Default::default()
    };
    let engine = BacktestEngine::with_config(config);

    // Run sweep with single parameter
    let mut strategy_sweep = RSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    let sweep_results = engine
        .run_sweep(
            &mut strategy_sweep,
            &timestamps,
            &open,
            &high,
            &low,
            &close,
            &volume,
            &grid,
        )
        .expect("Sweep failed");

    assert_eq!(sweep_results.len(), 1);

    // Run individual backtest with same parameters
    let mut strategy_individual = RSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    let individual_result = engine
        .run(
            &mut strategy_individual,
            &timestamps,
            &open,
            &high,
            &low,
            &close,
            &volume,
        )
        .expect("Individual backtest failed");

    // Compare results (should be very close)
    let sweep_result = &sweep_results[0];

    println!("\nComparing sweep vs individual backtest:");
    println!(
        "Sweep:      Sharpe={:.4}, MaxDD={:.2}%, Trades={}",
        sweep_result.sharpe_ratio, sweep_result.max_drawdown, sweep_result.num_trades
    );
    println!(
        "Individual: Sharpe={:.4}, MaxDD={:.2}%, Trades={}",
        individual_result.sharpe_ratio,
        individual_result.max_drawdown,
        individual_result.num_trades
    );

    // Results should be very similar (allow small differences due to rounding)
    assert!(
        (sweep_result.sharpe_ratio - individual_result.sharpe_ratio).abs() < 0.01,
        "Sharpe ratios should match"
    );
    assert!(
        (sweep_result.max_drawdown - individual_result.max_drawdown).abs() < 0.1,
        "Max drawdowns should match"
    );
    assert_eq!(
        sweep_result.num_trades, individual_result.num_trades,
        "Trade counts should match"
    );

    println!("Comparison test passed!");
}

#[test]
fn test_parameter_grid_validation() {
    // Test empty grid
    let empty_grid = ParameterGrid::new();
    assert!(empty_grid.is_empty());
    assert_eq!(empty_grid.size(), 1);

    // Test single parameter
    let mut single_grid = ParameterGrid::new();
    single_grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 5,
        },
    );
    assert_eq!(single_grid.size(), 3); // 10, 15, 20

    // Test multiple parameters
    let mut multi_grid = ParameterGrid::new();
    multi_grid.add_range(
        "period",
        ParameterRange::Int {
            min: 10,
            max: 14,
            step: 2,
        },
    );
    multi_grid.add_range(
        "threshold",
        ParameterRange::Float {
            min: 20.0,
            max: 30.0,
            step: 10.0,
        },
    );
    assert_eq!(multi_grid.size(), 3 * 2); // 3 periods × 2 thresholds

    println!("Parameter grid validation test passed!");
}
