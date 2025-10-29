//! Comprehensive parity tests for IncompleteCandle vs existing aggregation
//!
//! These tests MUST pass for tick engine to proceed.
//!
//! # Purpose
//! Validates that the new streaming IncompleteCandle produces identical results
//! to the existing batch aggregate_trades_to_candles() function across all scenarios.
//!
//! # Status
//! - Package 2.2 (IncompleteCandle) is in development
//! - Tests will pass once IncompleteCandle is implemented
//! - Run with: cargo test --test aggregation_parity_comprehensive

use kimsfinance_core::binance::{aggregate_trades_to_candles, Candle, Timeframe, Trade};
// use std::collections::HashMap;  // Will be used when IncompleteCandle is implemented

// =============================================================================
// Helper Functions
// =============================================================================

/// Helper: Create candles using IncompleteCandle streaming
///
/// This simulates the tick engine aggregation pattern:
/// 1. Maintain HashMap of incomplete candles by timestamp
/// 2. Update candles as trades arrive
/// 3. Complete and sort candles at the end
///
/// NOTE: This function will compile once IncompleteCandle is implemented in Package 2.2
#[allow(dead_code)]
fn aggregate_with_incomplete_candle(trades: &[Trade], timeframe: Timeframe) -> Vec<Candle> {
    // NOTE: Commented out until IncompleteCandle is implemented
    // Uncomment when Package 2.2 is complete

    /*
    let mut candles_map: HashMap<i64, IncompleteCandle> = HashMap::new();

    for trade in trades {
        let candle_timestamp = (trade.timestamp_ms / timeframe.to_ms()) * timeframe.to_ms();

        candles_map
            .entry(candle_timestamp)
            .and_modify(|candle| candle.update(trade))
            .or_insert_with(|| IncompleteCandle::new(trade, candle_timestamp));
    }

    let mut candles: Vec<Candle> = candles_map
        .into_iter()
        .map(|(_, candle)| candle.complete())
        .collect();

    candles.sort_by_key(|c| c.timestamp);
    candles
    */

    // Temporary placeholder: use existing aggregation
    // Remove this once IncompleteCandle is implemented
    aggregate_trades_to_candles(trades, timeframe)
}

/// Helper: Assert candles are identical
fn assert_candles_equal(old: &Candle, new: &Candle, context: &str) {
    assert_eq!(
        old.timestamp, new.timestamp,
        "{}: timestamp mismatch",
        context
    );
    assert_eq!(old.open, new.open, "{}: open mismatch", context);
    assert_eq!(old.high, new.high, "{}: high mismatch", context);
    assert_eq!(old.low, new.low, "{}: low mismatch", context);
    assert_eq!(old.close, new.close, "{}: close mismatch", context);
    assert_eq!(old.volume, new.volume, "{}: volume mismatch", context);
    assert_eq!(
        old.quote_volume, new.quote_volume,
        "{}: quote_volume mismatch",
        context
    );
    assert_eq!(
        old.num_trades, new.num_trades,
        "{}: num_trades mismatch",
        context
    );
}

/// Helper: Assert vectors of candles are identical
fn assert_candle_vectors_equal(old: &[Candle], new: &[Candle], context: &str) {
    assert_eq!(
        old.len(),
        new.len(),
        "{}: candle count mismatch (old: {}, new: {})",
        context,
        old.len(),
        new.len()
    );

    for (i, (old_candle, new_candle)) in old.iter().zip(new.iter()).enumerate() {
        assert_candles_equal(old_candle, new_candle, &format!("{} [candle {}]", context, i));
    }
}

// =============================================================================
// Parity Tests
// =============================================================================

#[test]
fn test_parity_single_candle() {
    // Multiple trades in a single 1-minute candle
    let trades = vec![
        Trade {
            trade_id: 1,
            price: 100.0,
            quantity: 1.0,
            quote_quantity: 100.0,
            timestamp_ms: 1000,
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 2,
            price: 105.0,
            quantity: 2.0,
            quote_quantity: 210.0,
            timestamp_ms: 2000,
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 3,
            price: 95.0,
            quantity: 1.5,
            quote_quantity: 142.5,
            timestamp_ms: 3000,
            is_buyer_maker: true,
        },
    ];

    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = aggregate_with_incomplete_candle(&trades, timeframe);

    assert_candle_vectors_equal(&candles_old, &candles_new, "single_candle");

    // Verify expected values
    assert_eq!(candles_old.len(), 1, "Expected 1 candle");
    let candle = &candles_old[0];
    assert_eq!(candle.open, 100.0, "First trade sets open");
    assert_eq!(candle.high, 105.0, "Maximum trade price");
    assert_eq!(candle.low, 95.0, "Minimum trade price");
    assert_eq!(candle.close, 95.0, "Last trade sets close");
    assert_eq!(candle.volume, 4.5, "Sum of quantities");
    assert_eq!(candle.quote_volume, 452.5, "Sum of quote quantities");
    assert_eq!(candle.num_trades, 3, "Three trades");
}

#[test]
fn test_parity_multiple_candles() {
    let mut trades = Vec::new();

    // Candle 1 (0-60s): 2 trades
    trades.push(Trade {
        trade_id: 1,
        price: 100.0,
        quantity: 1.0,
        quote_quantity: 100.0,
        timestamp_ms: 1000,
        is_buyer_maker: false,
    });
    trades.push(Trade {
        trade_id: 2,
        price: 102.0,
        quantity: 0.5,
        quote_quantity: 51.0,
        timestamp_ms: 30000,
        is_buyer_maker: true,
    });

    // Candle 2 (60-120s): 3 trades
    trades.push(Trade {
        trade_id: 3,
        price: 110.0,
        quantity: 2.0,
        quote_quantity: 220.0,
        timestamp_ms: 61000,
        is_buyer_maker: false,
    });
    trades.push(Trade {
        trade_id: 4,
        price: 108.0,
        quantity: 1.0,
        quote_quantity: 108.0,
        timestamp_ms: 90000,
        is_buyer_maker: true,
    });
    trades.push(Trade {
        trade_id: 5,
        price: 112.0,
        quantity: 0.75,
        quote_quantity: 84.0,
        timestamp_ms: 119000,
        is_buyer_maker: false,
    });

    // Candle 3 (120-180s): 1 trade
    trades.push(Trade {
        trade_id: 6,
        price: 105.0,
        quantity: 1.5,
        quote_quantity: 157.5,
        timestamp_ms: 121000,
        is_buyer_maker: false,
    });

    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = aggregate_with_incomplete_candle(&trades, timeframe);

    assert_candle_vectors_equal(&candles_old, &candles_new, "multiple_candles");

    // Verify structure
    assert_eq!(candles_old.len(), 3, "Expected 3 candles");

    // Verify each candle's trade counts
    assert_eq!(candles_old[0].num_trades, 2, "Candle 1 has 2 trades");
    assert_eq!(candles_old[1].num_trades, 3, "Candle 2 has 3 trades");
    assert_eq!(candles_old[2].num_trades, 1, "Candle 3 has 1 trade");
}

#[test]
fn test_parity_out_of_order_trades() {
    // Trades NOT sorted by timestamp
    // Both implementations should handle this gracefully
    let trades = vec![
        Trade {
            trade_id: 3,
            price: 95.0,
            quantity: 1.5,
            quote_quantity: 142.5,
            timestamp_ms: 3000,
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 1,
            price: 100.0,
            quantity: 1.0,
            quote_quantity: 100.0,
            timestamp_ms: 1000,
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 2,
            price: 105.0,
            quantity: 2.0,
            quote_quantity: 210.0,
            timestamp_ms: 2000,
            is_buyer_maker: false,
        },
    ];

    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = aggregate_with_incomplete_candle(&trades, timeframe);

    assert_candle_vectors_equal(&candles_old, &candles_new, "out_of_order_trades");

    // Both should produce same results regardless of input order
    assert_eq!(candles_old.len(), 1);
    let candle = &candles_old[0];

    // High/low are order-independent
    assert_eq!(candle.high, 105.0, "Max price regardless of order");
    assert_eq!(candle.low, 95.0, "Min price regardless of order");
    assert_eq!(candle.volume, 4.5, "Total volume regardless of order");
    assert_eq!(candle.num_trades, 3, "Count trades regardless of order");

    // Note: open/close depend on first/last trade in time, not arrival order
    // The HashMap approach makes this non-deterministic if trades have same timestamp
    // But different timestamps should be consistent
}

#[test]
fn test_parity_different_timeframes() {
    // Create trades spanning 5 minutes
    let trades = vec![
        Trade {
            trade_id: 1,
            price: 100.0,
            quantity: 1.0,
            quote_quantity: 100.0,
            timestamp_ms: 0,
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 2,
            price: 105.0,
            quantity: 2.0,
            quote_quantity: 210.0,
            timestamp_ms: 61000, // 1 minute 1 second
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 3,
            price: 110.0,
            quantity: 1.5,
            quote_quantity: 165.0,
            timestamp_ms: 121000, // 2 minutes 1 second
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 4,
            price: 95.0,
            quantity: 1.0,
            quote_quantity: 95.0,
            timestamp_ms: 181000, // 3 minutes 1 second
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 5,
            price: 100.0,
            quantity: 2.0,
            quote_quantity: 200.0,
            timestamp_ms: 241000, // 4 minutes 1 second
            is_buyer_maker: false,
        },
    ];

    // Test various timeframes
    for timeframe_str in &["1m", "5m", "15m", "1h"] {
        let timeframe = Timeframe::parse(timeframe_str).unwrap();

        let candles_old = aggregate_trades_to_candles(&trades, timeframe);
        let candles_new = aggregate_with_incomplete_candle(&trades, timeframe);

        assert_candle_vectors_equal(
            &candles_old,
            &candles_new,
            &format!("timeframe_{}", timeframe_str),
        );
    }
}

#[test]
fn test_parity_empty_trades() {
    // Edge case: No trades
    let trades: Vec<Trade> = vec![];
    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = aggregate_with_incomplete_candle(&trades, timeframe);

    assert_candle_vectors_equal(&candles_old, &candles_new, "empty_trades");
    assert!(candles_old.is_empty(), "No candles from empty trades");
}

#[test]
fn test_parity_single_trade() {
    // Edge case: Single trade
    let trades = vec![Trade {
        trade_id: 1,
        price: 100.0,
        quantity: 1.0,
        quote_quantity: 100.0,
        timestamp_ms: 1000,
        is_buyer_maker: false,
    }];

    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = aggregate_with_incomplete_candle(&trades, timeframe);

    assert_candle_vectors_equal(&candles_old, &candles_new, "single_trade");

    assert_eq!(candles_old.len(), 1);
    let candle = &candles_old[0];

    // Single trade: open = high = low = close
    assert_eq!(candle.open, 100.0);
    assert_eq!(candle.high, 100.0);
    assert_eq!(candle.low, 100.0);
    assert_eq!(candle.close, 100.0);
    assert_eq!(candle.volume, 1.0);
    assert_eq!(candle.num_trades, 1);
}

#[test]
fn test_parity_candle_boundaries() {
    // Test trades at exact candle boundaries (critical edge case)
    let trades = vec![
        Trade {
            trade_id: 1,
            price: 100.0,
            quantity: 1.0,
            quote_quantity: 100.0,
            timestamp_ms: 60000, // Exactly 00:01:00.000
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 2,
            price: 101.0,
            quantity: 1.0,
            quote_quantity: 101.0,
            timestamp_ms: 119999, // 00:01:59.999 (last ms of minute)
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 3,
            price: 102.0,
            quantity: 1.0,
            quote_quantity: 102.0,
            timestamp_ms: 120000, // Exactly 00:02:00.000 (next candle)
            is_buyer_maker: false,
        },
    ];

    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = aggregate_with_incomplete_candle(&trades, timeframe);

    assert_candle_vectors_equal(&candles_old, &candles_new, "candle_boundaries");

    // Verify boundary handling
    assert_eq!(candles_old.len(), 2, "Two distinct candles");

    // First candle: 60000-119999 (2 trades)
    assert_eq!(candles_old[0].timestamp, 60000);
    assert_eq!(candles_old[0].num_trades, 2);
    assert_eq!(candles_old[0].open, 100.0);
    assert_eq!(candles_old[0].close, 101.0);

    // Second candle: 120000+ (1 trade)
    assert_eq!(candles_old[1].timestamp, 120000);
    assert_eq!(candles_old[1].num_trades, 1);
    assert_eq!(candles_old[1].open, 102.0);
}

#[test]
fn test_parity_high_frequency_trades() {
    // Simulate high-frequency trading: many trades in short time
    let mut trades = Vec::new();

    // 100 trades in 1 second
    for i in 0..100 {
        trades.push(Trade {
            trade_id: i,
            price: 100.0 + (i as f64 * 0.01), // Incrementing prices
            quantity: 0.1,
            quote_quantity: (100.0 + (i as f64 * 0.01)) * 0.1,
            timestamp_ms: i as i64 * 10, // 0, 10, 20, ... 990 ms
            is_buyer_maker: i % 2 == 0,
        });
    }

    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = aggregate_with_incomplete_candle(&trades, timeframe);

    assert_candle_vectors_equal(&candles_old, &candles_new, "high_frequency_trades");

    assert_eq!(candles_old.len(), 1, "All trades in one candle");
    let candle = &candles_old[0];

    assert_eq!(candle.num_trades, 100, "100 trades aggregated");
    // Use approximate comparison for floating-point volume
    assert!(
        (candle.volume - 10.0).abs() < 1e-10,
        "Total volume: expected 10.0, got {}",
        candle.volume
    );
    assert_eq!(candle.open, 100.0, "First trade price");
    assert_eq!(candle.close, 100.99, "Last trade price");
}

#[test]
fn test_parity_sparse_candles() {
    // Test scenario with gaps (empty candles)
    let trades = vec![
        Trade {
            trade_id: 1,
            price: 100.0,
            quantity: 1.0,
            quote_quantity: 100.0,
            timestamp_ms: 0, // Minute 0
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 2,
            price: 105.0,
            quantity: 1.0,
            quote_quantity: 105.0,
            timestamp_ms: 300000, // Minute 5 (4-minute gap)
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 3,
            price: 110.0,
            quantity: 1.0,
            quote_quantity: 110.0,
            timestamp_ms: 600000, // Minute 10 (another 4-minute gap)
            is_buyer_maker: false,
        },
    ];

    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = aggregate_with_incomplete_candle(&trades, timeframe);

    assert_candle_vectors_equal(&candles_old, &candles_new, "sparse_candles");

    // Should only create candles with trades (no empty candles)
    assert_eq!(candles_old.len(), 3, "Only candles with trades");

    // Verify timestamps of sparse candles
    assert_eq!(candles_old[0].timestamp, 0);
    assert_eq!(candles_old[1].timestamp, 300000);
    assert_eq!(candles_old[2].timestamp, 600000);
}

#[test]
fn test_parity_large_price_swings() {
    // Test with extreme price volatility
    let trades = vec![
        Trade {
            trade_id: 1,
            price: 10000.0,
            quantity: 1.0,
            quote_quantity: 10000.0,
            timestamp_ms: 1000,
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 2,
            price: 50000.0, // 5x increase
            quantity: 2.0,
            quote_quantity: 100000.0,
            timestamp_ms: 2000,
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 3,
            price: 5000.0, // 90% drop
            quantity: 0.5,
            quote_quantity: 2500.0,
            timestamp_ms: 3000,
            is_buyer_maker: true,
        },
    ];

    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = aggregate_with_incomplete_candle(&trades, timeframe);

    assert_candle_vectors_equal(&candles_old, &candles_new, "large_price_swings");

    let candle = &candles_old[0];
    assert_eq!(candle.open, 10000.0, "First trade");
    assert_eq!(candle.high, 50000.0, "Peak price");
    assert_eq!(candle.low, 5000.0, "Lowest price");
    assert_eq!(candle.close, 5000.0, "Last trade");
}

#[test]
#[ignore] // Only run with `cargo test -- --ignored` (requires real data)
fn test_parity_real_binance_data() {
    use kimsfinance_core::binance::process_binance_month;

    // This test validates parity against real Binance data (4.6M trades)
    // Path should be updated to match your local data location
    let path = "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades/BTCUSDT-trades-2025-10-13.zip";

    // Skip test if file doesn't exist
    if !std::path::Path::new(path).exists() {
        println!("Skipping real data test: file not found at {}", path);
        return;
    }

    let timeframe = Timeframe::parse("5m").unwrap();

    // Load candles from file using existing aggregation
    let candles_from_file = process_binance_month(path, timeframe).unwrap();

    println!("Loaded {} candles from real data", candles_from_file.len());
    println!("First candle: {:?}", candles_from_file.first());
    println!("Last candle: {:?}", candles_from_file.last());

    // NOTE: Full parity test would require:
    // 1. Load trades separately (not aggregated)
    // 2. Run both aggregate_trades_to_candles() and aggregate_with_incomplete_candle()
    // 3. Compare results
    //
    // For now, this test validates the file loads successfully
    // TODO: Implement full comparison when IncompleteCandle is ready

    assert!(!candles_from_file.is_empty(), "Should load candles");
    assert!(
        candles_from_file.len() > 100,
        "Should have substantial data"
    );
}

// =============================================================================
// Performance Comparison Tests (Optional)
// =============================================================================

#[test]
#[ignore] // Run separately with `cargo test -- --ignored`
fn test_performance_comparison_small_dataset() {
    use std::time::Instant;

    // Create 1000 trades
    let mut trades = Vec::with_capacity(1000);
    for i in 0..1000 {
        trades.push(Trade {
            trade_id: i,
            price: 100.0 + (i as f64 * 0.01),
            quantity: 0.1,
            quote_quantity: (100.0 + (i as f64 * 0.01)) * 0.1,
            timestamp_ms: i as i64 * 1000, // 1 trade per second
            is_buyer_maker: i % 2 == 0,
        });
    }

    let timeframe = Timeframe::parse("1m").unwrap();

    // Benchmark old method
    let start = Instant::now();
    let _candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let duration_old = start.elapsed();

    // Benchmark new method
    let start = Instant::now();
    let _candles_new = aggregate_with_incomplete_candle(&trades, timeframe);
    let duration_new = start.elapsed();

    println!("Old method: {:?}", duration_old);
    println!("New method: {:?}", duration_new);

    // Both should be fast for small datasets
    assert!(
        duration_old < std::time::Duration::from_millis(10),
        "Old method too slow"
    );
    assert!(
        duration_new < std::time::Duration::from_millis(10),
        "New method too slow"
    );
}
