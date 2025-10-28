//! Integration tests for GPU batch backtesting API
//!
//! Tests the complete 4-phase pipeline with actual GPU execution.
//! Run with: `cargo test --test test_batch_api --features gpu -- --ignored`

use kimsfinance_core::backtest::{BacktestConfig, BatchBacktestSweep, StrategyType};
use kimsfinance_core::gpu::device::GpuDevice;
use ndarray::Array1;
use std::sync::Arc;

/// Generate synthetic OHLCV data for testing
fn generate_test_data(
    n_candles: usize,
) -> (
    Vec<i64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let mut timestamps = Vec::with_capacity(n_candles);
    let mut open = Vec::with_capacity(n_candles);
    let mut high = Vec::with_capacity(n_candles);
    let mut low = Vec::with_capacity(n_candles);
    let mut close = Vec::with_capacity(n_candles);
    let mut volume = Vec::with_capacity(n_candles);

    let mut price = 100.0;
    for i in 0..n_candles {
        timestamps.push(i as i64 * 3600); // 1 hour intervals

        // Random walk with mean reversion
        let change = (i as f64 * 0.1).sin() * 2.0;
        price += change;
        price = price.max(50.0).min(150.0); // Keep in reasonable range

        open.push(price);
        high.push(price + (i as f64 * 0.3).sin().abs() * 2.0);
        low.push(price - (i as f64 * 0.3).cos().abs() * 2.0);
        close.push(price + (i as f64 * 0.2).sin() * 1.0);
        volume.push(1000.0 + (i as f64 * 0.1).cos() * 200.0);
    }

    (
        timestamps,
        Array1::from(open),
        Array1::from(high),
        Array1::from(low),
        Array1::from(close),
        Array1::from(volume),
    )
}

#[test]
#[ignore] // Requires GPU
fn test_batch_api_10_strategies() {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    // Generate test data
    let (timestamps, open, high, low, close, volume) = generate_test_data(1000);

    // Create 10 RSI strategies with different parameters
    let params: Vec<Vec<f64>> = (20..30)
        .map(|buy_thresh| vec![14.0, buy_thresh as f64, (100 - buy_thresh) as f64])
        .collect();

    assert_eq!(params.len(), 10);

    // Execute batch backtest
    let result = BatchBacktestSweep::new(device)
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .parameters_batch(&params)
        .config(BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
        })
        .execute();

    assert!(result.is_ok(), "Batch backtest failed: {:?}", result.err());

    let result = result.unwrap();

    // Validate results
    assert_eq!(result.results.len(), 10);
    assert!(
        result.total_time_ms < 1000.0,
        "Too slow: {}ms",
        result.total_time_ms
    );
    assert!(
        result.vram_used_mb < 100.0,
        "Too much VRAM: {}MB",
        result.vram_used_mb
    );

    // Check that all metrics are valid
    for (i, res) in result.results.iter().enumerate() {
        assert!(
            res.sharpe_ratio.is_finite(),
            "Strategy {} has invalid Sharpe: {}",
            i,
            res.sharpe_ratio
        );
        assert!(
            res.max_drawdown >= -1.0 && res.max_drawdown <= 0.0,
            "Strategy {} has invalid drawdown: {}",
            i,
            res.max_drawdown
        );
        assert!(
            res.win_rate >= 0.0 && res.win_rate <= 1.0,
            "Strategy {} has invalid win rate: {}",
            i,
            res.win_rate
        );
        assert_eq!(
            res.equity_curve.len(),
            1000,
            "Strategy {} has wrong equity curve length",
            i
        );
    }

    println!(
        "✅ 10 strategies processed in {:.2}ms",
        result.total_time_ms
    );
    println!("   GPU time: {:.2}ms", result.gpu_time_ms);
    println!("   VRAM used: {:.2} MB", result.vram_used_mb);
    println!("   Speedup: {:.1}x", result.speedup());
}

#[test]
#[ignore] // Requires GPU
fn test_batch_api_100_strategies() {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    // Generate test data
    let (timestamps, open, high, low, close, volume) = generate_test_data(1000);

    // Create 100 RSI strategies (10 buy × 10 sell thresholds)
    let mut params = Vec::new();
    for buy in 20..30 {
        for sell in 70..80 {
            params.push(vec![14.0, buy as f64, sell as f64]);
        }
    }

    assert_eq!(params.len(), 100);

    // Execute batch backtest
    let result = BatchBacktestSweep::new(device)
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .parameters_batch(&params)
        .execute();

    assert!(result.is_ok(), "Batch backtest failed: {:?}", result.err());

    let result = result.unwrap();

    // Validate results
    assert_eq!(result.results.len(), 100);
    assert!(
        result.total_time_ms < 500.0,
        "Too slow: {}ms",
        result.total_time_ms
    );

    // Performance validation: Should be much faster than sequential
    let expected_sequential_time = 100.0 * 10.0; // 100 strategies × 10ms each
    let speedup = expected_sequential_time / result.total_time_ms;
    assert!(
        speedup > 5.0,
        "Speedup too low: {:.1}x (expected >5x)",
        speedup
    );

    println!(
        "✅ 100 strategies processed in {:.2}ms",
        result.total_time_ms
    );
    println!("   Speedup: {:.1}x vs sequential", speedup);
}

#[test]
#[ignore] // Requires GPU and significant VRAM
fn test_batch_api_1000_strategies() {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    // Generate larger dataset
    let (timestamps, open, high, low, close, volume) = generate_test_data(10_000);

    // Create 1000 RSI strategies
    let mut params = Vec::new();
    for i in 0..1000 {
        let buy = 20.0 + (i % 20) as f64;
        let sell = 70.0 + (i % 20) as f64;
        params.push(vec![14.0, buy, sell]);
    }

    assert_eq!(params.len(), 1000);

    // Execute batch backtest
    let result = BatchBacktestSweep::new(device)
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .parameters_batch(&params)
        .execute();

    assert!(result.is_ok(), "Batch backtest failed: {:?}", result.err());

    let result = result.unwrap();

    // Validate results
    assert_eq!(result.results.len(), 1000);
    assert!(
        result.total_time_ms < 300.0,
        "Too slow: {}ms (target <300ms)",
        result.total_time_ms
    );
    assert!(
        result.vram_used_mb < 1000.0,
        "Too much VRAM: {}MB (target <1GB)",
        result.vram_used_mb
    );

    // Performance validation: Target 40x speedup
    let expected_sequential_time = 1000.0 * 10.0; // 10 seconds
    let speedup = expected_sequential_time / result.total_time_ms;
    assert!(
        speedup > 20.0,
        "Speedup too low: {:.1}x (target >20x)",
        speedup
    );

    result.print_summary();
}

#[test]
#[ignore] // Requires GPU
fn test_batch_api_error_handling() {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    // Test: No strategy type set
    let result = BatchBacktestSweep::new(device.clone())
        .parameters_batch(&vec![vec![14.0, 25.0, 75.0]])
        .execute();
    assert!(result.is_err());
    assert!(format!("{:?}", result.err().unwrap()).contains("Strategy type not set"));

    // Test: No data set
    let result = BatchBacktestSweep::new(device.clone())
        .strategy_type(StrategyType::RsiCrossover)
        .parameters_batch(&vec![vec![14.0, 25.0, 75.0]])
        .execute();
    assert!(result.is_err());
    assert!(format!("{:?}", result.err().unwrap()).contains("Data not set"));

    // Test: No parameters
    let (timestamps, open, high, low, close, volume) = generate_test_data(100);
    let result = BatchBacktestSweep::new(device.clone())
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .execute();
    assert!(result.is_err());
    assert!(format!("{:?}", result.err().unwrap()).contains("No parameters"));

    println!("✅ Error handling validated");
}

#[test]
#[ignore] // Requires GPU
fn test_batch_api_edge_cases() {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    // Test: Single strategy
    let (timestamps, open, high, low, close, volume) = generate_test_data(100);
    let result = BatchBacktestSweep::new(device.clone())
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .parameters_batch(&vec![vec![14.0, 25.0, 75.0]])
        .execute();
    assert!(result.is_ok());
    assert_eq!(result.unwrap().results.len(), 1);

    // Test: Very few candles
    let (timestamps, open, high, low, close, volume) = generate_test_data(50);
    let result = BatchBacktestSweep::new(device.clone())
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .parameters_batch(&vec![vec![14.0, 25.0, 75.0], vec![14.0, 30.0, 70.0]])
        .execute();
    assert!(result.is_ok());

    println!("✅ Edge cases validated");
}

#[test]
#[ignore] // Requires GPU
fn test_batch_api_results_sorted_by_fitness() {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    let (timestamps, open, high, low, close, volume) = generate_test_data(500);

    // Create diverse strategies (some should perform better than others)
    let params: Vec<Vec<f64>> = vec![
        vec![14.0, 25.0, 75.0], // Reasonable
        vec![14.0, 10.0, 90.0], // Very wide bands (worse)
        vec![14.0, 30.0, 70.0], // Tighter bands
        vec![14.0, 40.0, 60.0], // Very tight (likely worse)
    ];

    let result = BatchBacktestSweep::new(device)
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .parameters_batch(&params)
        .execute()
        .expect("Batch backtest failed");

    assert_eq!(result.results.len(), 4);

    // Verify results are sorted by fitness (highest first)
    for i in 0..result.results.len() - 1 {
        let fitness_current = result.results[i].fitness();
        let fitness_next = result.results[i + 1].fitness();
        assert!(
            fitness_current >= fitness_next,
            "Results not sorted: fitness[{}]={:.2} < fitness[{}]={:.2}",
            i,
            fitness_current,
            i + 1,
            fitness_next
        );
    }

    println!("✅ Results correctly sorted by fitness");
    println!("   Best fitness: {:.2}", result.results[0].fitness());
    println!(
        "   Worst fitness: {:.2}",
        result.results[result.results.len() - 1].fitness()
    );
}
