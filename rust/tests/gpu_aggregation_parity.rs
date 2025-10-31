//! GPU Aggregation Parity Tests
//!
//! Validates that GPU aggregation produces identical results to CPU aggregation
//! across various scenarios and edge cases.

#![cfg(feature = "gpu")]

use kimsfinance_core::binance::{Timeframe, Trade, aggregate_trades_to_candles};
use kimsfinance_core::gpu::{AggregationEngine, EngineSelector, GpuAggregator};

/// Generate test trades for validation
fn generate_test_trades(n: usize, timeframe_spread: usize) -> Vec<Trade> {
    let mut trades = Vec::with_capacity(n);

    let base_price = 50_000.0;
    let base_time = 1_600_000_000_000i64;
    let timeframe_ms = (timeframe_spread * 60 * 1000) as i64; // Minutes to ms

    for i in 0..n {
        // Distribute trades across multiple candles
        let candle_idx = i / (n / timeframe_spread).max(1);
        let timestamp = base_time + (candle_idx as i64 * timeframe_ms) + (i as i64 * 100);

        // Vary price within candle
        let price_variation = ((i % 100) as f64 / 100.0) * 100.0; // ±50
        let price = base_price + price_variation - 50.0;

        trades.push(Trade {
            trade_id: i as u64,
            price,
            quantity: 0.1 + (i as f64 * 0.001),
            quote_quantity: price * (0.1 + (i as f64 * 0.001)),
            timestamp_ms: timestamp,
            is_buyer_maker: i % 2 == 0,
        });
    }

    trades
}

/// Helper to compare two candle arrays within tolerance
fn assert_candles_equal(
    cpu_candles: &[kimsfinance_core::binance::Candle],
    gpu_candles: &[kimsfinance_core::binance::Candle],
    tolerance: f64,
) {
    assert_eq!(
        cpu_candles.len(),
        gpu_candles.len(),
        "Candle count mismatch: CPU={}, GPU={}",
        cpu_candles.len(),
        gpu_candles.len()
    );

    for (i, (cpu, gpu)) in cpu_candles.iter().zip(gpu_candles.iter()).enumerate() {
        assert_eq!(
            cpu.timestamp, gpu.timestamp,
            "Candle {} timestamp mismatch",
            i
        );

        assert!(
            (cpu.open - gpu.open).abs() < tolerance,
            "Candle {} open mismatch: CPU={}, GPU={}, diff={}",
            i,
            cpu.open,
            gpu.open,
            (cpu.open - gpu.open).abs()
        );

        assert!(
            (cpu.high - gpu.high).abs() < tolerance,
            "Candle {} high mismatch: CPU={}, GPU={}, diff={}",
            i,
            cpu.high,
            gpu.high,
            (cpu.high - gpu.high).abs()
        );

        assert!(
            (cpu.low - gpu.low).abs() < tolerance,
            "Candle {} low mismatch: CPU={}, GPU={}, diff={}",
            i,
            cpu.low,
            gpu.low,
            (cpu.low - gpu.low).abs()
        );

        assert!(
            (cpu.close - gpu.close).abs() < tolerance,
            "Candle {} close mismatch: CPU={}, GPU={}, diff={}",
            i,
            cpu.close,
            gpu.close,
            (cpu.close - gpu.close).abs()
        );

        assert!(
            (cpu.volume - gpu.volume).abs() < tolerance,
            "Candle {} volume mismatch: CPU={}, GPU={}, diff={}",
            i,
            cpu.volume,
            gpu.volume,
            (cpu.volume - gpu.volume).abs()
        );

        assert!(
            (cpu.quote_volume - gpu.quote_volume).abs() < tolerance,
            "Candle {} quote_volume mismatch: CPU={}, GPU={}, diff={}",
            i,
            cpu.quote_volume,
            gpu.quote_volume,
            (cpu.quote_volume - gpu.quote_volume).abs()
        );

        assert_eq!(
            cpu.num_trades, gpu.num_trades,
            "Candle {} num_trades mismatch",
            i
        );
    }
}

#[test]
#[ignore] // Requires GPU
fn test_gpu_aggregation_parity_small() {
    let aggregator = GpuAggregator::new().expect("GPU not available");
    let trades = generate_test_trades(1_000, 10);
    let timeframe = Timeframe::minutes(5);

    let cpu_candles = aggregate_trades_to_candles(&trades, timeframe);
    let gpu_candles = aggregator
        .aggregate_trades(&trades, timeframe)
        .expect("GPU aggregation failed");

    assert_candles_equal(&cpu_candles, &gpu_candles, 1e-10);
}

#[test]
#[ignore] // Requires GPU
fn test_gpu_aggregation_parity_medium() {
    let aggregator = GpuAggregator::new().expect("GPU not available");
    let trades = generate_test_trades(50_000, 100);
    let timeframe = Timeframe::minutes(5);

    let cpu_candles = aggregate_trades_to_candles(&trades, timeframe);
    let gpu_candles = aggregator
        .aggregate_trades(&trades, timeframe)
        .expect("GPU aggregation failed");

    assert_candles_equal(&cpu_candles, &gpu_candles, 1e-10);
}

#[test]
#[ignore] // Requires GPU
fn test_gpu_aggregation_parity_large() {
    let aggregator = GpuAggregator::new().expect("GPU not available");
    let trades = generate_test_trades(100_000, 1000);
    let timeframe = Timeframe::minutes(5);

    let cpu_candles = aggregate_trades_to_candles(&trades, timeframe);
    let gpu_candles = aggregator
        .aggregate_trades(&trades, timeframe)
        .expect("GPU aggregation failed");

    assert_candles_equal(&cpu_candles, &gpu_candles, 1e-10);
}

#[test]
#[ignore] // Requires GPU
fn test_gpu_aggregation_parity_single_candle() {
    let aggregator = GpuAggregator::new().expect("GPU not available");
    let trades = generate_test_trades(1_000, 1);
    let timeframe = Timeframe::minutes(60);

    let cpu_candles = aggregate_trades_to_candles(&trades, timeframe);
    let gpu_candles = aggregator
        .aggregate_trades(&trades, timeframe)
        .expect("GPU aggregation failed");

    assert_candles_equal(&cpu_candles, &gpu_candles, 1e-10);
}

#[test]
#[ignore] // Requires GPU
fn test_gpu_aggregation_parity_many_candles() {
    let aggregator = GpuAggregator::new().expect("GPU not available");
    let trades = generate_test_trades(10_000, 1000);
    let timeframe = Timeframe::minutes(1);

    let cpu_candles = aggregate_trades_to_candles(&trades, timeframe);
    let gpu_candles = aggregator
        .aggregate_trades(&trades, timeframe)
        .expect("GPU aggregation failed");

    assert_candles_equal(&cpu_candles, &gpu_candles, 1e-10);
}

#[test]
#[ignore] // Requires GPU
fn test_gpu_aggregation_empty_trades() {
    let aggregator = GpuAggregator::new().expect("GPU not available");
    let trades = vec![];
    let timeframe = Timeframe::minutes(5);

    let cpu_candles = aggregate_trades_to_candles(&trades, timeframe);
    let gpu_candles = aggregator
        .aggregate_trades(&trades, timeframe)
        .expect("GPU aggregation failed");

    assert!(cpu_candles.is_empty());
    assert!(gpu_candles.is_empty());
}

#[test]
#[ignore] // Requires GPU
fn test_gpu_aggregation_single_trade() {
    let aggregator = GpuAggregator::new().expect("GPU not available");
    let trades = vec![Trade {
        trade_id: 1,
        price: 50_000.0,
        quantity: 0.1,
        quote_quantity: 5_000.0,
        timestamp_ms: 1_600_000_000_000,
        is_buyer_maker: false,
    }];
    let timeframe = Timeframe::minutes(5);

    let cpu_candles = aggregate_trades_to_candles(&trades, timeframe);
    let gpu_candles = aggregator
        .aggregate_trades(&trades, timeframe)
        .expect("GPU aggregation failed");

    assert_candles_equal(&cpu_candles, &gpu_candles, 1e-10);
}

#[test]
fn test_engine_selector_auto() {
    let selector = EngineSelector::default();

    // Small dataset: should select CPU
    let small_trades = generate_test_trades(1_000, 10);
    assert_eq!(
        selector.select_engine(small_trades.len()),
        AggregationEngine::CPU
    );

    // Large dataset: should select GPU (if available)
    let large_trades = generate_test_trades(100_000, 1000);
    let engine = selector.select_engine(large_trades.len());

    if selector.is_gpu_available() {
        assert_eq!(engine, AggregationEngine::GPU);
    } else {
        assert_eq!(engine, AggregationEngine::CPU);
    }
}

#[test]
fn test_engine_selector_aggregate_small() {
    let selector = EngineSelector::default();
    let trades = generate_test_trades(1_000, 10);
    let timeframe = Timeframe::minutes(5);

    let candles = selector
        .aggregate_trades(&trades, timeframe)
        .expect("Aggregation failed");

    assert!(!candles.is_empty());
}

#[test]
#[ignore] // Requires GPU and takes time
fn test_engine_selector_aggregate_large() {
    let selector = EngineSelector::default();
    let trades = generate_test_trades(100_000, 1000);
    let timeframe = Timeframe::minutes(5);

    let candles = selector
        .aggregate_trades(&trades, timeframe)
        .expect("Aggregation failed");

    assert!(!candles.is_empty());
}
