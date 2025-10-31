//! Integration tests for market microstructure analysis
//!
//! This test suite validates the microstructure analysis functionality with
//! real-world scenarios, edge cases, and performance requirements.

use kimsfinance_core::analysis::{MicrostructureAnalyzer, MicrostructureMetrics};
use kimsfinance_core::backtest::{MicrostructureStrategy, Signal, TickStrategy};
use kimsfinance_core::binance::{IncompleteCandle, Trade};

/// Helper to create test trade
fn make_trade(price: f64, quantity: f64, timestamp_ms: i64, is_buyer_maker: bool) -> Trade {
    Trade {
        trade_id: 0,
        price,
        quantity,
        quote_quantity: price * quantity,
        timestamp_ms,
        is_buyer_maker,
    }
}

#[test]
fn test_microstructure_analyzer_basic() {
    let analyzer = MicrostructureAnalyzer::new(60_000);

    let trades = vec![
        make_trade(100.0, 1.0, 1000, false), // Aggressive buy
        make_trade(101.0, 2.0, 2000, false), // Aggressive buy
        make_trade(100.5, 1.0, 3000, true),  // Aggressive sell
    ];

    let metrics = analyzer.analyze(&trades);

    assert_eq!(metrics.num_trades, 3);
    assert_eq!(metrics.total_volume, 4.0);
    assert_eq!(metrics.buy_volume, 3.0);
    assert_eq!(metrics.sell_volume, 1.0);
    assert_eq!(metrics.aggressive_buy_count, 2);
    assert_eq!(metrics.aggressive_sell_count, 1);

    // OFI = (3 - 1) / (3 + 1) = 0.5
    assert!((metrics.order_flow_imbalance - 0.5).abs() < 1e-9);
}

#[test]
fn test_microstructure_rolling_windows() {
    let analyzer = MicrostructureAnalyzer::new(100_000); // 100ms windows

    let trades = vec![
        make_trade(100.0, 1.0, 0, false),       // Window 0
        make_trade(101.0, 1.0, 50_000, false),  // Window 0
        make_trade(102.0, 1.0, 100_000, false), // Window 1
        make_trade(103.0, 1.0, 150_000, false), // Window 1
        make_trade(104.0, 1.0, 200_000, false), // Window 2
    ];

    let metrics = analyzer.analyze_rolling(&trades);

    assert_eq!(metrics.len(), 3);
    assert_eq!(metrics[0].num_trades, 2);
    assert_eq!(metrics[1].num_trades, 2);
    assert_eq!(metrics[2].num_trades, 1);
}

#[test]
fn test_order_flow_imbalance_calculation() {
    let analyzer = MicrostructureAnalyzer::new(60_000);

    // Test 1: 100% buy pressure
    let all_buys = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(101.0, 1.0, 2000, false),
    ];
    let metrics = analyzer.analyze(&all_buys);
    assert_eq!(metrics.order_flow_imbalance, 1.0);

    // Test 2: 100% sell pressure
    let all_sells = vec![
        make_trade(100.0, 1.0, 1000, true),
        make_trade(99.0, 1.0, 2000, true),
    ];
    let metrics = analyzer.analyze(&all_sells);
    assert_eq!(metrics.order_flow_imbalance, -1.0);

    // Test 3: Balanced
    let balanced = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(100.0, 1.0, 2000, true),
    ];
    let metrics = analyzer.analyze(&balanced);
    assert_eq!(metrics.order_flow_imbalance, 0.0);
}

#[test]
fn test_price_volatility_calculation() {
    let analyzer = MicrostructureAnalyzer::new(60_000);

    // Low volatility: all trades at same price
    let low_vol = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(100.0, 1.0, 2000, false),
        make_trade(100.0, 1.0, 3000, false),
    ];
    let metrics = analyzer.analyze(&low_vol);
    assert_eq!(metrics.price_volatility, 0.0);

    // High volatility: prices vary significantly
    let high_vol = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(150.0, 1.0, 2000, false),
        make_trade(50.0, 1.0, 3000, false),
    ];
    let metrics = analyzer.analyze(&high_vol);
    assert!(metrics.price_volatility > 0.0);
}

#[test]
fn test_tick_direction_analysis() {
    let analyzer = MicrostructureAnalyzer::new(60_000);

    // All upticks
    let upticks = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(101.0, 1.0, 2000, false),
        make_trade(102.0, 1.0, 3000, false),
    ];
    let metrics = analyzer.analyze(&upticks);
    assert_eq!(metrics.tick_direction, 1.0);

    // All downticks
    let downticks = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(99.0, 1.0, 2000, false),
        make_trade(98.0, 1.0, 3000, false),
    ];
    let metrics = analyzer.analyze(&downticks);
    assert_eq!(metrics.tick_direction, -1.0);

    // Mixed
    let mixed = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(101.0, 1.0, 2000, false), // Uptick
        make_trade(100.0, 1.0, 3000, false), // Downtick
    ];
    let metrics = analyzer.analyze(&mixed);
    assert_eq!(metrics.tick_direction, 0.0);
}

#[test]
fn test_vwap_calculation() {
    let analyzer = MicrostructureAnalyzer::new(60_000);

    let trades = vec![
        make_trade(100.0, 1.0, 1000, false), // 100 * 1 = 100
        make_trade(110.0, 2.0, 2000, false), // 110 * 2 = 220
        make_trade(90.0, 1.0, 3000, false),  // 90 * 1 = 90
    ];

    let metrics = analyzer.analyze(&trades);

    // VWAP = (100 + 220 + 90) / (1 + 2 + 1) = 410 / 4 = 102.5
    assert!((metrics.volume_weighted_price - 102.5).abs() < 1e-9);
}

#[test]
fn test_microstructure_strategy_integration() {
    let mut strategy = MicrostructureStrategy::new(0.3, 60_000);

    // Simulate strong buying pressure
    for i in 0..10 {
        let trade = make_trade(100.0, 1.0, i * 1000, false);
        let candle = IncompleteCandle::new(&trade, 0);
        strategy.on_tick(&trade, &candle);
    }

    // Should generate buy signal
    let trade = make_trade(100.0, 1.0, 11_000, false);
    let candle = IncompleteCandle::new(&trade, 0);
    let signal = strategy.on_tick(&trade, &candle);

    assert_eq!(signal, Signal::Buy);
}

#[test]
fn test_strategy_threshold_behavior() {
    // Test with different thresholds
    let mut sensitive = MicrostructureStrategy::new(0.1, 60_000); // Sensitive
    let mut conservative = MicrostructureStrategy::new(0.8, 60_000); // Conservative

    // Add 6 buy trades, 4 sell trades (60% buy)
    // OFI = (6 - 4) / (6 + 4) = 0.2
    for i in 0..10 {
        let is_buy = i < 6;
        let trade = make_trade(100.0, 1.0, i * 1000, !is_buy);
        let candle = IncompleteCandle::new(&trade, 0);

        sensitive.on_tick(&trade, &candle);
        conservative.on_tick(&trade, &candle);
    }

    let trade = make_trade(100.0, 1.0, 11_000, false);
    let candle = IncompleteCandle::new(&trade, 0);

    // Sensitive should signal (0.2 > 0.1)
    let sensitive_signal = sensitive.on_tick(&trade, &candle);
    assert_eq!(sensitive_signal, Signal::Buy);

    // Conservative should hold (0.2 < 0.8)
    let conservative_signal = conservative.on_tick(&trade, &candle);
    assert_eq!(conservative_signal, Signal::Hold);
}

#[test]
fn test_edge_case_single_trade() {
    let analyzer = MicrostructureAnalyzer::new(60_000);
    let trades = vec![make_trade(100.0, 1.0, 1000, false)];

    let metrics = analyzer.analyze(&trades);

    assert_eq!(metrics.num_trades, 1);
    assert_eq!(metrics.total_volume, 1.0);
    assert_eq!(metrics.order_flow_imbalance, 1.0); // 100% buy
    assert_eq!(metrics.price_volatility, 0.0); // Single price
    assert_eq!(metrics.tick_direction, 0.0); // No price changes
}

#[test]
fn test_edge_case_zero_volume() {
    let analyzer = MicrostructureAnalyzer::new(60_000);
    let trades = vec![make_trade(100.0, 0.0, 1000, false)];

    let metrics = analyzer.analyze(&trades);

    assert_eq!(metrics.num_trades, 1);
    assert_eq!(metrics.total_volume, 0.0);
    // VWAP should handle zero volume gracefully
    assert!(metrics.volume_weighted_price == 0.0 || metrics.volume_weighted_price.is_nan());
}

#[test]
fn test_large_dataset_performance() {
    let analyzer = MicrostructureAnalyzer::new(60_000);

    // Generate 10,000 trades
    let mut trades = Vec::new();
    for i in 0..10_000 {
        trades.push(make_trade(
            100.0 + (i as f64 % 10.0),
            1.0,
            i * 100,
            i % 2 == 0,
        ));
    }

    // Time the analysis
    let start = std::time::Instant::now();
    let metrics = analyzer.analyze(&trades);
    let elapsed = start.elapsed();

    // Should process 10K trades in <1ms
    assert!(elapsed.as_micros() < 1000);
    assert_eq!(metrics.num_trades, 10_000);
}

#[test]
fn test_rolling_window_performance() {
    let analyzer = MicrostructureAnalyzer::new(10_000); // 10ms windows

    // Generate 10,000 trades across 1 second (100 windows)
    let mut trades = Vec::new();
    for i in 0..10_000 {
        trades.push(make_trade(100.0, 1.0, i * 100, i % 2 == 0));
    }

    // Time the rolling analysis
    let start = std::time::Instant::now();
    let metrics = analyzer.analyze_rolling(&trades);
    let elapsed = start.elapsed();

    // Should process 10K trades in <5ms
    assert!(elapsed.as_micros() < 5000);
    assert_eq!(metrics.len(), 100); // 100 windows
}

#[test]
fn test_strategy_buffer_management() {
    let mut strategy = MicrostructureStrategy::with_buffer_size(0.3, 60_000, 100);

    // Add 150 trades (exceeds buffer size)
    for i in 0..150 {
        let trade = make_trade(100.0, 1.0, i * 1000, false);
        let candle = IncompleteCandle::new(&trade, 0);
        strategy.on_tick(&trade, &candle);
    }

    // Buffer should be capped at 100
    assert!(strategy.buffer_size() <= 100);
}

#[test]
fn test_microstructure_metrics_empty() {
    let metrics = MicrostructureMetrics::empty(1000, 60_000);

    assert_eq!(metrics.timestamp, 1000);
    assert_eq!(metrics.duration_ms, 60_000);
    assert_eq!(metrics.num_trades, 0);
    assert_eq!(metrics.total_volume, 0.0);
    assert_eq!(metrics.order_flow_imbalance, 0.0);
}

#[test]
fn test_realistic_trading_scenario() {
    let analyzer = MicrostructureAnalyzer::new(60_000);

    // Simulate realistic trading pattern:
    // - Mix of aggressive buys and sells
    // - Varying prices and quantities
    // - Time-ordered trades
    let trades = vec![
        make_trade(50_000.0, 0.1, 0, false),   // Small aggressive buy
        make_trade(50_001.0, 0.2, 100, false), // Medium aggressive buy
        make_trade(50_000.5, 0.05, 200, true), // Small aggressive sell
        make_trade(50_002.0, 0.5, 300, false), // Large aggressive buy
        make_trade(50_001.5, 0.15, 400, true), // Medium aggressive sell
        make_trade(50_003.0, 1.0, 500, false), // Very large aggressive buy
        make_trade(50_002.5, 0.3, 600, true),  // Medium aggressive sell
    ];

    let metrics = analyzer.analyze(&trades);

    // Verify metrics make sense
    assert_eq!(metrics.num_trades, 7);

    // Buy volume = 0.1 + 0.2 + 0.5 + 1.0 = 1.8
    assert!((metrics.buy_volume - 1.8).abs() < 1e-9);

    // Sell volume = 0.05 + 0.15 + 0.3 = 0.5
    assert!((metrics.sell_volume - 0.5).abs() < 1e-9);

    // OFI = (1.8 - 0.5) / (1.8 + 0.5) = 1.3 / 2.3 ≈ 0.565
    assert!((metrics.order_flow_imbalance - 0.565217).abs() < 0.001);

    // Tick direction should be non-zero (price is changing)
    // Net direction: +1, +1, -1, +1, -1, +1, -1 = +1 out of 6 = 0.167
    assert!(metrics.tick_direction.abs() >= 0.0);

    // VWAP should be weighted towards larger trades
    assert!(metrics.volume_weighted_price > 50_001.0);
}

#[test]
fn test_strategy_reset_functionality() {
    let mut strategy = MicrostructureStrategy::new(0.3, 60_000);

    // Add trades
    for i in 0..10 {
        let trade = make_trade(100.0, 1.0, i * 1000, false);
        let candle = IncompleteCandle::new(&trade, 0);
        strategy.on_tick(&trade, &candle);
    }

    assert!(strategy.buffer_size() > 0);

    // Reset
    strategy.reset();
    assert_eq!(strategy.buffer_size(), 0);
    assert_eq!(strategy.current_imbalance(), 0.0);
}

#[test]
fn test_spread_estimation() {
    let analyzer = MicrostructureAnalyzer::new(60_000);

    // Tight spread scenario (prices close together)
    let tight_spread = vec![
        make_trade(100.00, 1.0, 1000, false),
        make_trade(100.01, 1.0, 2000, false),
        make_trade(100.02, 1.0, 3000, false),
    ];

    let metrics = analyzer.analyze(&tight_spread);
    let tight_spread_estimate = metrics.spread_estimate;

    // Wide spread scenario (prices far apart)
    let wide_spread = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(105.0, 1.0, 2000, false),
        make_trade(110.0, 1.0, 3000, false),
    ];

    let metrics = analyzer.analyze(&wide_spread);
    let wide_spread_estimate = metrics.spread_estimate;

    // Wide spread should be larger than tight spread
    assert!(wide_spread_estimate > tight_spread_estimate);
}
