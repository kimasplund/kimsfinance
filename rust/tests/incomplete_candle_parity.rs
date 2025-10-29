//! 100% Parity validation between IncompleteCandle and existing aggregation
//!
//! This is CRITICAL - if this fails, tick engine is blocked.
//!
//! These tests ensure that `IncompleteCandle` produces IDENTICAL results to
//! the existing `CandleBuilder` used in `aggregate_trades_to_candles()`.

use kimsfinance_core::binance::{
    aggregate_trades_to_candles, IncompleteCandle, Timeframe, Trade,
};

/// Helper to create test trades
fn make_trade(
    trade_id: u64,
    price: f64,
    quantity: f64,
    timestamp_ms: i64,
    is_buyer_maker: bool,
) -> Trade {
    Trade {
        trade_id,
        price,
        quantity,
        quote_quantity: price * quantity,
        timestamp_ms,
        is_buyer_maker,
    }
}

/// Build candles using IncompleteCandle (streaming approach)
fn build_candles_with_incomplete(trades: &[Trade], timeframe: Timeframe) -> Vec<kimsfinance_core::binance::Candle> {
    if trades.is_empty() {
        return Vec::new();
    }

    let timeframe_ms = timeframe.to_ms();
    let mut candles = Vec::new();
    let mut current_candle: Option<IncompleteCandle> = None;

    for trade in trades {
        let candle_timestamp = (trade.timestamp_ms / timeframe_ms) * timeframe_ms;

        match &mut current_candle {
            Some(candle) if candle.timestamp == candle_timestamp => {
                // Same candle, update it
                candle.update(trade);
            }
            _ => {
                // New candle, finalize previous if exists
                if let Some(candle) = current_candle.take() {
                    candles.push(candle.complete());
                }
                current_candle = Some(IncompleteCandle::new(trade, candle_timestamp));
            }
        }
    }

    // Finalize last candle
    if let Some(candle) = current_candle {
        candles.push(candle.complete());
    }

    // Sort by timestamp (same as aggregate_trades_to_candles)
    candles.sort_unstable_by_key(|c| c.timestamp);

    candles
}

/// Assert two candles are identical (with floating-point tolerance)
fn assert_candles_equal(
    old: &kimsfinance_core::binance::Candle,
    new: &kimsfinance_core::binance::Candle,
    context: &str,
) {
    assert_eq!(old.timestamp, new.timestamp, "{}: timestamp mismatch", context);
    assert!(
        (old.open - new.open).abs() < 1e-9,
        "{}: open mismatch: {} vs {}",
        context,
        old.open,
        new.open
    );
    assert!(
        (old.high - new.high).abs() < 1e-9,
        "{}: high mismatch: {} vs {}",
        context,
        old.high,
        new.high
    );
    assert!(
        (old.low - new.low).abs() < 1e-9,
        "{}: low mismatch: {} vs {}",
        context,
        old.low,
        new.low
    );
    assert!(
        (old.close - new.close).abs() < 1e-9,
        "{}: close mismatch: {} vs {}",
        context,
        old.close,
        new.close
    );
    assert!(
        (old.volume - new.volume).abs() < 1e-9,
        "{}: volume mismatch: {} vs {}",
        context,
        old.volume,
        new.volume
    );
    assert!(
        (old.quote_volume - new.quote_volume).abs() < 1e-9,
        "{}: quote_volume mismatch: {} vs {}",
        context,
        old.quote_volume,
        new.quote_volume
    );
    assert_eq!(
        old.num_trades, new.num_trades,
        "{}: num_trades mismatch",
        context
    );
}

#[test]
fn test_parity_empty_trades() {
    let trades: Vec<Trade> = vec![];
    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = build_candles_with_incomplete(&trades, timeframe);

    assert_eq!(candles_old.len(), candles_new.len());
    assert!(candles_old.is_empty());
    assert!(candles_new.is_empty());
}

#[test]
fn test_parity_single_trade() {
    let trades = vec![make_trade(1, 100.0, 1.0, 1000, false)];
    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = build_candles_with_incomplete(&trades, timeframe);

    assert_eq!(candles_old.len(), 1);
    assert_eq!(candles_new.len(), 1);
    assert_candles_equal(&candles_old[0], &candles_new[0], "single trade");
}

#[test]
fn test_parity_multiple_trades_same_candle() {
    let trades = vec![
        make_trade(1, 100.0, 1.0, 1000, false),
        make_trade(2, 105.0, 2.0, 2000, false),
        make_trade(3, 95.0, 1.5, 3000, false),
    ];
    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = build_candles_with_incomplete(&trades, timeframe);

    assert_eq!(candles_old.len(), 1);
    assert_eq!(candles_new.len(), 1);
    assert_candles_equal(&candles_old[0], &candles_new[0], "multiple trades same candle");
}

#[test]
fn test_parity_multiple_candles() {
    let trades = vec![
        make_trade(1, 100.0, 1.0, 0, false),        // Candle 1
        make_trade(2, 101.0, 1.0, 60_000, false),   // Candle 2
        make_trade(3, 102.0, 1.0, 120_000, false),  // Candle 3
        make_trade(4, 103.0, 1.0, 180_000, false),  // Candle 4
    ];
    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = build_candles_with_incomplete(&trades, timeframe);

    assert_eq!(candles_old.len(), 4);
    assert_eq!(candles_new.len(), 4);

    for (i, (old, new)) in candles_old.iter().zip(candles_new.iter()).enumerate() {
        assert_candles_equal(old, new, &format!("candle {}", i));
    }
}

#[test]
fn test_parity_out_of_order_trades() {
    // Trades NOT sorted by timestamp (aggregate_trades_to_candles handles this)
    let trades = vec![
        make_trade(2, 101.0, 1.0, 60_000, false),  // Minute 1
        make_trade(1, 100.0, 1.0, 0, false),       // Minute 0
        make_trade(3, 102.0, 1.0, 120_000, false), // Minute 2
    ];
    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = build_candles_with_incomplete(&trades, timeframe);

    assert_eq!(candles_old.len(), 3);
    assert_eq!(candles_new.len(), 3);

    // Both should be sorted by timestamp
    for (i, (old, new)) in candles_old.iter().zip(candles_new.iter()).enumerate() {
        assert_candles_equal(old, new, &format!("out-of-order candle {}", i));
    }
}

#[test]
fn test_parity_five_minute_timeframe() {
    let trades = vec![
        make_trade(1, 100.0, 1.0, 0, false),         // 00:00 (5m candle 1)
        make_trade(2, 101.0, 1.0, 60_000, false),    // 00:01 (5m candle 1)
        make_trade(3, 102.0, 1.0, 300_000, false),   // 00:05 (5m candle 2)
        make_trade(4, 103.0, 1.0, 600_000, false),   // 00:10 (5m candle 3)
    ];
    let timeframe = Timeframe::parse("5m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = build_candles_with_incomplete(&trades, timeframe);

    assert_eq!(candles_old.len(), 3);
    assert_eq!(candles_new.len(), 3);

    for (i, (old, new)) in candles_old.iter().zip(candles_new.iter()).enumerate() {
        assert_candles_equal(old, new, &format!("5m candle {}", i));
    }
}

#[test]
fn test_parity_complex_scenario() {
    // Complex scenario with multiple trades per candle, gaps, and price movements
    let trades = vec![
        // Candle 1: 00:00-00:00:59
        make_trade(1, 100.0, 1.0, 0, false),
        make_trade(2, 105.0, 2.0, 10_000, false),
        make_trade(3, 95.0, 1.5, 50_000, false),
        // Candle 2: 00:01-00:01:59
        make_trade(4, 97.0, 0.5, 60_000, false),
        make_trade(5, 103.0, 1.0, 90_000, true),
        // Gap: No trades at 00:02
        // Candle 3: 00:03-00:03:59
        make_trade(6, 102.0, 2.5, 180_000, false),
        make_trade(7, 101.0, 0.8, 200_000, true),
        make_trade(8, 104.0, 1.2, 230_000, false),
    ];
    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = build_candles_with_incomplete(&trades, timeframe);

    assert_eq!(candles_old.len(), 3); // 3 candles (no candle for minute 2)
    assert_eq!(candles_new.len(), 3);

    for (i, (old, new)) in candles_old.iter().zip(candles_new.iter()).enumerate() {
        assert_candles_equal(old, new, &format!("complex candle {}", i));
    }
}

#[test]
fn test_parity_candle_boundaries() {
    // Test trades at exact candle boundaries
    let trades = vec![
        make_trade(1, 100.0, 1.0, 0, false),        // Exactly 00:00:00.000
        make_trade(2, 101.0, 1.0, 59_999, false),   // Last ms of minute 0
        make_trade(3, 102.0, 1.0, 60_000, false),   // Exactly 00:01:00.000
        make_trade(4, 103.0, 1.0, 119_999, false),  // Last ms of minute 1
        make_trade(5, 104.0, 1.0, 120_000, false),  // Exactly 00:02:00.000
    ];
    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = build_candles_with_incomplete(&trades, timeframe);

    assert_eq!(candles_old.len(), 3);
    assert_eq!(candles_new.len(), 3);

    for (i, (old, new)) in candles_old.iter().zip(candles_new.iter()).enumerate() {
        assert_candles_equal(old, new, &format!("boundary candle {}", i));
    }
}

#[test]
fn test_parity_high_low_accumulation() {
    // Specifically test high/low tracking across multiple trades
    let trades = vec![
        make_trade(1, 100.0, 1.0, 1000, false), // Open
        make_trade(2, 110.0, 1.0, 2000, false), // New high
        make_trade(3, 95.0, 1.0, 3000, false),  // New low
        make_trade(4, 105.0, 1.0, 4000, false), // Middle
        make_trade(5, 120.0, 1.0, 5000, false), // New high
        make_trade(6, 90.0, 1.0, 6000, false),  // New low
        make_trade(7, 102.0, 1.0, 7000, false), // Close
    ];
    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = build_candles_with_incomplete(&trades, timeframe);

    assert_eq!(candles_old.len(), 1);
    assert_eq!(candles_new.len(), 1);

    let old = &candles_old[0];
    let new = &candles_new[0];

    assert_eq!(old.open, 100.0);
    assert_eq!(new.open, 100.0);
    assert_eq!(old.high, 120.0);
    assert_eq!(new.high, 120.0);
    assert_eq!(old.low, 90.0);
    assert_eq!(new.low, 90.0);
    assert_eq!(old.close, 102.0);
    assert_eq!(new.close, 102.0);

    assert_candles_equal(old, new, "high/low accumulation");
}

#[test]
fn test_parity_volume_accumulation() {
    // Test volume and quote_volume accumulation
    let trades = vec![
        make_trade(1, 100.0, 1.5, 1000, false),   // volume=1.5, quote=150
        make_trade(2, 105.0, 2.3, 2000, false),   // volume=2.3, quote=241.5
        make_trade(3, 95.0, 0.7, 3000, false),    // volume=0.7, quote=66.5
    ];
    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = build_candles_with_incomplete(&trades, timeframe);

    assert_eq!(candles_old.len(), 1);
    assert_eq!(candles_new.len(), 1);

    let old = &candles_old[0];
    let new = &candles_new[0];

    // Total volume: 1.5 + 2.3 + 0.7 = 4.5
    assert!((old.volume - 4.5).abs() < 1e-9);
    assert!((new.volume - 4.5).abs() < 1e-9);

    // Total quote_volume: 150 + 241.5 + 66.5 = 458
    assert!((old.quote_volume - 458.0).abs() < 1e-9);
    assert!((new.quote_volume - 458.0).abs() < 1e-9);

    assert_candles_equal(old, new, "volume accumulation");
}

#[test]
fn test_parity_large_batch() {
    // Generate 1000 trades across 10 candles
    // Each candle has 100 trades, with timestamps spread across <1 minute
    let mut trades = Vec::new();
    let timeframe_ms = 60_000i64; // 1 minute
    let base_timestamp = 1_000_000_020_000i64; // Aligned to 1-minute boundary

    for candle_idx in 0..10 {
        // Calculate candle start timestamp (aligned to timeframe)
        let candle_start = base_timestamp + (candle_idx * timeframe_ms);

        for trade_idx in 0..100 {
            let price = 100.0 + (candle_idx as f64) + (trade_idx as f64 / 100.0);
            let quantity = 1.0 + (trade_idx as f64 / 100.0);
            // Use 550ms per trade to ensure we stay within 1 minute (100 * 550 = 55,000ms < 60,000ms)
            let timestamp = candle_start + (trade_idx * 550);

            trades.push(make_trade(
                (candle_idx * 100 + trade_idx) as u64,
                price,
                quantity,
                timestamp,
                trade_idx % 2 == 0,
            ));
        }
    }

    let timeframe = Timeframe::parse("1m").unwrap();

    let candles_old = aggregate_trades_to_candles(&trades, timeframe);
    let candles_new = build_candles_with_incomplete(&trades, timeframe);

    assert_eq!(candles_old.len(), 10);
    assert_eq!(candles_new.len(), 10);

    for (i, (old, new)) in candles_old.iter().zip(candles_new.iter()).enumerate() {
        assert_candles_equal(old, new, &format!("large batch candle {}", i));
    }
}

#[test]
fn test_parity_various_timeframes() {
    let trades = vec![
        make_trade(1, 100.0, 1.0, 0, false),
        make_trade(2, 101.0, 1.0, 60_000, false),
        make_trade(3, 102.0, 1.0, 120_000, false),
        make_trade(4, 103.0, 1.0, 300_000, false),
        make_trade(5, 104.0, 1.0, 3_600_000, false),
    ];

    let timeframes = vec![
        Timeframe::parse("1m").unwrap(),
        Timeframe::parse("5m").unwrap(),
        Timeframe::parse("15m").unwrap(),
        Timeframe::parse("1h").unwrap(),
    ];

    for timeframe in timeframes {
        let candles_old = aggregate_trades_to_candles(&trades, timeframe);
        let candles_new = build_candles_with_incomplete(&trades, timeframe);

        assert_eq!(
            candles_old.len(),
            candles_new.len(),
            "Timeframe {:?} length mismatch",
            timeframe
        );

        for (i, (old, new)) in candles_old.iter().zip(candles_new.iter()).enumerate() {
            assert_candles_equal(old, new, &format!("timeframe {:?} candle {}", timeframe, i));
        }
    }
}

#[test]
#[ignore] // Only run with `cargo test -- --ignored`
fn test_parity_real_binance_data() {
    use kimsfinance_core::binance::process_binance_month;

    // Test with real Binance data (1 day = ~4.6M trades)
    let path = "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades/BTCUSDT-trades-2025-10-13.zip";
    
    // Skip if file doesn't exist
    if !std::path::Path::new(path).exists() {
        println!("SKIPPED: Real Binance data file not found at {}", path);
        return;
    }

    let timeframe = Timeframe::parse("5m").unwrap();

    // Load all trades for streaming comparison
    println!("Loading real Binance data from {}", path);
    let candles_old = process_binance_month(path, timeframe).unwrap();
    println!("Loaded {} candles using aggregate_trades_to_candles", candles_old.len());

    // TODO: Implement streaming version when IncompleteCandle is integrated into process_binance_month
    // For now, this test just validates that the existing function works
    
    assert!(!candles_old.is_empty(), "Should have produced candles from real data");
    println!("✓ Real data test passed: {} candles produced", candles_old.len());
}
