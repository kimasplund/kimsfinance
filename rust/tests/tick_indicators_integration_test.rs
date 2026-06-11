//! Integration test for tick-level indicator calculations
//!
//! Tests the complete workflow: Trade stream → Aggregation → Indicators

use kimsfinance_core::binance::{Timeframe, Trade};
use kimsfinance_core::indicators::{
    EMA, RSI, SMA, TickIndicatorEngine, calculate_indicator_from_trades,
};

/// Helper to create test trade
fn make_trade(price: f64, timestamp_ms: i64) -> Trade {
    Trade {
        trade_id: 0,
        price,
        quantity: 1.0,
        quote_quantity: price,
        timestamp_ms,
        is_buyer_maker: false,
    }
}

#[test]
fn test_tick_indicator_engine_basic_workflow() {
    let mut engine = TickIndicatorEngine::new(Timeframe::minutes(1));

    // Create 50 trades across 50 minutes
    for i in 0..50 {
        let trade = make_trade(100.0 + i as f64, 1609459200000 + (i * 60000));
        engine.update(&trade);
    }

    assert_eq!(engine.num_trades(), 50);
    assert_eq!(engine.num_candles(), 50);

    // Calculate SMA(20)
    let sma = SMA::new(20).unwrap();
    let sma_values = engine.calculate_indicator(&sma).unwrap();

    assert_eq!(sma_values.len(), 50);
    // First 19 should be NaN
    assert!(sma_values[18].is_nan());
    // 20th onwards should have values
    assert!(!sma_values[19].is_nan());
}

#[test]
fn test_tick_indicator_engine_multiple_indicators() {
    let mut engine = TickIndicatorEngine::new(Timeframe::minutes(1));

    // Create 100 trades with price oscillation
    for i in 0..100 {
        let price = 100.0 + ((i as f64 * 0.1).sin() * 10.0);
        let trade = make_trade(price, 1609459200000 + (i * 60000));
        engine.update(&trade);
    }

    // Calculate multiple indicators
    let sma20 = SMA::new(20).unwrap();
    let ema12 = EMA::new(12).unwrap();
    let rsi14 = RSI::new(14).unwrap();

    let sma_result = engine.calculate_indicator(&sma20).unwrap();
    let ema_result = engine.calculate_indicator(&ema12).unwrap();
    let rsi_result = engine.calculate_indicator(&rsi14).unwrap();

    assert_eq!(sma_result.len(), 100);
    assert_eq!(ema_result.len(), 100);
    assert_eq!(rsi_result.len(), 100);

    // Verify values exist after warmup
    assert!(!sma_result[30].is_nan());
    assert!(!ema_result[30].is_nan());
    assert!(!rsi_result[30].is_nan());

    // RSI should be between 0 and 100
    for i in 30..100 {
        if !rsi_result[i].is_nan() {
            assert!(
                rsi_result[i] >= 0.0 && rsi_result[i] <= 100.0,
                "RSI at index {} = {} (should be 0-100)",
                i,
                rsi_result[i]
            );
        }
    }
}

#[test]
fn test_calculate_indicator_from_trades_helper() {
    // Create 50 trades
    let trades: Vec<Trade> = (0..50)
        .map(|i| make_trade(100.0 + i as f64, 1609459200000 + (i * 60000)))
        .collect();

    let sma = SMA::new(20).unwrap();
    let result = calculate_indicator_from_trades(&trades, Timeframe::minutes(1), &sma);

    assert!(result.is_ok());
    let values = result.unwrap();
    assert_eq!(values.len(), 50);
    assert!(values[18].is_nan());
    assert!(!values[19].is_nan());
}

#[test]
fn test_tick_aggregation_to_candles() {
    let mut engine = TickIndicatorEngine::new(Timeframe::minutes(5));

    // Create 25 trades across 25 minutes (should aggregate to 5 candles)
    for i in 0..25 {
        let trade = make_trade(100.0 + i as f64, 1609459200000 + (i * 60000));
        engine.update(&trade);
    }

    assert_eq!(engine.num_trades(), 25);
    assert_eq!(engine.num_candles(), 5); // 25 minutes / 5 minutes per candle

    let candles = engine.get_candles();
    assert_eq!(candles.len(), 5);

    // First candle should have 5 trades
    assert_eq!(candles[0].num_trades, 5);
    assert_eq!(candles[0].open, 100.0);
    assert_eq!(candles[0].close, 104.0); // Last trade in first 5 minutes
}

#[test]
fn test_tick_engine_cache_behavior() {
    let mut engine = TickIndicatorEngine::new(Timeframe::minutes(1));

    // Add first trade
    let trade1 = make_trade(100.0, 1609459200000);
    engine.update(&trade1);

    let candles1 = engine.get_candles();
    assert_eq!(candles1.len(), 1);

    // Add another trade in different minute
    let trade2 = make_trade(101.0, 1609459260000);
    engine.update(&trade2);

    let candles2 = engine.get_candles();
    assert_eq!(candles2.len(), 2); // Cache invalidated, rebuilt
}

#[test]
fn test_tick_engine_clear() {
    let mut engine = TickIndicatorEngine::new(Timeframe::minutes(1));

    // Add trades
    for i in 0..10 {
        let trade = make_trade(100.0, 1609459200000 + (i * 60000));
        engine.update(&trade);
    }

    assert_eq!(engine.num_trades(), 10);
    assert_eq!(engine.num_candles(), 10);

    // Clear and verify
    engine.clear();
    assert_eq!(engine.num_trades(), 0);
    assert_eq!(engine.num_candles(), 0);
}

#[test]
fn test_tick_engine_batch_update() {
    let mut engine = TickIndicatorEngine::new(Timeframe::minutes(1));

    // Create batch of trades
    let trades: Vec<Trade> = (0..20)
        .map(|i| make_trade(100.0 + i as f64, 1609459200000 + (i * 60000)))
        .collect();

    engine.update_batch(&trades);

    assert_eq!(engine.num_trades(), 20);
    assert_eq!(engine.num_candles(), 20);
}

#[test]
fn test_rsi_strategy_simulation() {
    // Simulate a simple RSI strategy using tick indicators
    let mut engine = TickIndicatorEngine::new(Timeframe::minutes(1));

    // Create 100 trades with price variation
    for i in 0..100 {
        let price = 100.0 + ((i as f64 * 0.2).sin() * 20.0); // Oscillating between 80-120
        let trade = make_trade(price, 1609459200000 + (i * 60000));
        engine.update(&trade);
    }

    let rsi = RSI::new(14).unwrap();
    let rsi_values = engine.calculate_indicator(&rsi).unwrap();

    // Check for RSI signals
    let mut buy_signals = 0;
    let mut sell_signals = 0;

    for &rsi_val in rsi_values.iter().skip(20) {
        // Skip warmup
        if !rsi_val.is_nan() {
            if rsi_val < 30.0 {
                buy_signals += 1;
            } else if rsi_val > 70.0 {
                sell_signals += 1;
            }
        }
    }

    // Should have generated some signals
    println!(
        "Buy signals: {}, Sell signals: {}",
        buy_signals, sell_signals
    );
    assert!(buy_signals > 0 || sell_signals > 0);
}

#[test]
fn test_sma_crossover_strategy_simulation() {
    // Simulate SMA crossover strategy
    let mut engine = TickIndicatorEngine::new(Timeframe::minutes(1));

    // Create uptrend followed by downtrend
    for i in 0..100 {
        let price = if i < 50 {
            100.0 + i as f64 * 0.5 // Uptrend
        } else {
            125.0 - (i - 50) as f64 * 0.5 // Downtrend
        };
        let trade = make_trade(price, 1609459200000 + (i * 60000));
        engine.update(&trade);
    }

    // Calculate fast and slow SMA
    let sma_fast = SMA::new(10).unwrap();
    let sma_slow = SMA::new(20).unwrap();

    let fast_values = engine.calculate_indicator(&sma_fast).unwrap();
    let slow_values = engine.calculate_indicator(&sma_slow).unwrap();

    // Check for crossovers
    let mut crossovers = 0;
    for i in 20..100 {
        if !fast_values[i].is_nan()
            && !slow_values[i].is_nan()
            && !fast_values[i - 1].is_nan()
            && !slow_values[i - 1].is_nan()
        {
            let prev_diff = fast_values[i - 1] - slow_values[i - 1];
            let curr_diff = fast_values[i] - slow_values[i];

            if prev_diff * curr_diff < 0.0 {
                // Sign change = crossover
                crossovers += 1;
            }
        }
    }

    println!("Detected {} SMA crossovers", crossovers);
    assert!(crossovers > 0);
}

#[test]
fn test_empty_trades_error_handling() {
    let mut engine = TickIndicatorEngine::new(Timeframe::minutes(1));

    let sma = SMA::new(20).unwrap();
    let result = engine.calculate_indicator(&sma);

    assert!(result.is_err());
}

#[test]
fn test_insufficient_data_graceful_handling() {
    let mut engine = TickIndicatorEngine::new(Timeframe::minutes(1));

    // Only 5 trades, but RSI needs 14
    for i in 0..5 {
        let trade = make_trade(100.0, 1609459200000 + (i * 60000));
        engine.update(&trade);
    }

    let rsi = RSI::new(14).unwrap();
    let result = engine.calculate_indicator(&rsi);

    // Should succeed but have NaN values
    assert!(result.is_ok());
    let values = result.unwrap();
    assert_eq!(values.len(), 5);
    // All values should be NaN (insufficient data)
    for val in values.iter() {
        assert!(val.is_nan());
    }
}
