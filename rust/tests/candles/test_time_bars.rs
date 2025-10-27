//! Time Bar Aggregation Tests
//!
//! Validates time-based candle aggregation with various intervals.
//! Tests OHLCV correctness, edge cases, and sequential processing.

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{execute_batch, GpuDevice, TimeBarBatch};

#[cfg(feature = "gpu")]
#[test]
fn test_time_bars_1m_aggregation() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Create synthetic trade data (timestamp, price, volume)
    // 10 trades spanning 3 minutes
    let trades = vec![
        // Minute 1 (0-59s): 3 trades
        (0.0, 100.0, 10.0),     // First trade sets open
        (30.0, 105.0, 15.0),    // High of minute
        (50.0, 98.0, 20.0),     // Low & close

        // Minute 2 (60-119s): 4 trades
        (60.0, 99.0, 12.0),     // Open
        (75.0, 102.0, 18.0),    // High
        (90.0, 96.0, 14.0),     // Low
        (110.0, 101.0, 16.0),   // Close

        // Minute 3 (120-179s): 3 trades
        (120.0, 100.0, 11.0),
        (140.0, 103.0, 19.0),
        (165.0, 102.0, 13.0),
    ];

    // Convert to flat arrays: [timestamps, prices, volumes]
    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    // Create 1-minute time bar batch
    let mut batch = TimeBarBatch::new();
    batch.add_task(data, 60); // 60 seconds interval

    let results = execute_batch(&device, &batch)?;
    assert_eq!(results.len(), 1, "Expected 1 result");

    // Result should contain 3 candles (3 minutes) * 5 OHLCV values = 15 values
    let candles = &results[0];
    assert_eq!(candles.len(), 15, "Expected 15 values (3 candles * 5 OHLCV)");

    // Verify first candle (minute 1)
    let open1 = candles[0];
    let high1 = candles[1];
    let low1 = candles[2];
    let close1 = candles[3];
    let volume1 = candles[4];

    assert_eq!(open1, 100.0, "Candle 1 open");
    assert_eq!(high1, 105.0, "Candle 1 high");
    assert_eq!(low1, 98.0, "Candle 1 low");
    assert_eq!(close1, 98.0, "Candle 1 close");
    assert_eq!(volume1, 45.0, "Candle 1 volume (10+15+20)");

    // Verify second candle (minute 2)
    let open2 = candles[5];
    let high2 = candles[6];
    let low2 = candles[7];
    let close2 = candles[8];
    let volume2 = candles[9];

    assert_eq!(open2, 99.0, "Candle 2 open");
    assert_eq!(high2, 102.0, "Candle 2 high");
    assert_eq!(low2, 96.0, "Candle 2 low");
    assert_eq!(close2, 101.0, "Candle 2 close");
    assert_eq!(volume2, 60.0, "Candle 2 volume (12+18+14+16)");

    println!("✅ 1-minute aggregation test passed");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_time_bars_5m_aggregation() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // 15 trades spanning 10 minutes
    let trades = vec![
        // First 5 minutes (0-299s): 8 trades
        (0.0, 100.0, 10.0),
        (60.0, 102.0, 12.0),
        (120.0, 104.0, 14.0),    // High
        (180.0, 98.0, 11.0),     // Low
        (240.0, 101.0, 13.0),
        (280.0, 103.0, 15.0),    // Close
        (285.0, 102.0, 12.0),
        (295.0, 103.5, 14.0),

        // Second 5 minutes (300-599s): 7 trades
        (300.0, 103.0, 11.0),    // Open
        (360.0, 105.0, 13.0),
        (420.0, 107.0, 15.0),    // High
        (480.0, 104.0, 12.0),
        (540.0, 102.0, 10.0),    // Low
        (580.0, 106.0, 14.0),    // Close
        (595.0, 105.5, 13.0),
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    // Create 5-minute time bar batch
    let mut batch = TimeBarBatch::new();
    batch.add_task(data, 300); // 300 seconds (5 minutes)

    let results = execute_batch(&device, &batch)?;

    // Should have 2 candles (10 minutes / 5 minutes)
    let candles = &results[0];
    assert_eq!(candles.len(), 10, "Expected 10 values (2 candles * 5 OHLCV)");

    // Verify aggregation
    let open1 = candles[0];
    let high1 = candles[1];
    let low1 = candles[2];

    assert_eq!(open1, 100.0, "5-min candle 1 open");
    assert_eq!(high1, 104.0, "5-min candle 1 high");
    assert_eq!(low1, 98.0, "5-min candle 1 low");

    println!("✅ 5-minute aggregation test passed");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_time_bars_1h_aggregation() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // 12 trades spanning 2 hours
    let mut trades = Vec::new();

    // Hour 1 (0-3599s): 6 trades
    for i in 0..6 {
        let time = (i * 600) as f64; // Every 10 minutes
        let price = 100.0 + (i as f64);
        let volume = 10.0 + (i as f64);
        trades.push((time, price, volume));
    }

    // Hour 2 (3600-7199s): 6 trades
    for i in 0..6 {
        let time = 3600.0 + (i * 600) as f64;
        let price = 105.0 + (i as f64);
        let volume = 15.0 + (i as f64);
        trades.push((time, price, volume));
    }

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    // Create 1-hour time bar batch
    let mut batch = TimeBarBatch::new();
    batch.add_task(data, 3600); // 3600 seconds (1 hour)

    let results = execute_batch(&device, &batch)?;

    // Should have 2 candles (2 hours)
    let candles = &results[0];
    assert_eq!(candles.len(), 10, "Expected 10 values (2 candles * 5 OHLCV)");

    // Verify hour 1
    let open1 = candles[0];
    let high1 = candles[1];
    let low1 = candles[2];
    let close1 = candles[3];

    assert_eq!(open1, 100.0, "Hour 1 open");
    assert_eq!(high1, 105.0, "Hour 1 high");
    assert_eq!(low1, 100.0, "Hour 1 low");
    assert_eq!(close1, 105.0, "Hour 1 close");

    println!("✅ 1-hour aggregation test passed");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_time_bars_empty_bucket() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Trades with gap (no trades in minute 2)
    let trades = vec![
        // Minute 1: 2 trades
        (10.0, 100.0, 10.0),
        (50.0, 102.0, 12.0),

        // Minute 2: EMPTY (60-119s)

        // Minute 3: 2 trades
        (130.0, 103.0, 11.0),
        (170.0, 104.0, 13.0),
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = TimeBarBatch::new();
    batch.add_task(data, 60);

    let results = execute_batch(&device, &batch)?;
    let candles = &results[0];

    // Should handle empty buckets gracefully
    // Empty bucket should either:
    // - Be skipped (resulting in 2 candles)
    // - Use previous close (resulting in 3 candles with flat middle)
    assert!(candles.len() >= 10, "Should handle empty buckets");

    println!("✅ Empty bucket handling test passed");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_time_bars_single_trade_per_bucket() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Exactly 1 trade per minute
    let trades = vec![
        (10.0, 100.0, 10.0),
        (70.0, 101.0, 11.0),
        (130.0, 102.0, 12.0),
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = TimeBarBatch::new();
    batch.add_task(data, 60);

    let results = execute_batch(&device, &batch)?;
    let candles = &results[0];

    assert_eq!(candles.len(), 15, "Expected 3 candles");

    // With single trade: open = high = low = close = price
    for i in 0..3 {
        let offset = i * 5;
        let open = candles[offset];
        let high = candles[offset + 1];
        let low = candles[offset + 2];
        let close = candles[offset + 3];

        assert_eq!(open, high, "Single trade: open == high");
        assert_eq!(high, low, "Single trade: high == low");
        assert_eq!(low, close, "Single trade: low == close");
    }

    println!("✅ Single trade per bucket test passed");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_time_bars_volume_accumulation() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Multiple trades with known volumes
    let trades = vec![
        (10.0, 100.0, 5.0),
        (20.0, 101.0, 10.0),
        (30.0, 102.0, 15.0),
        (40.0, 103.0, 20.0),
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = TimeBarBatch::new();
    batch.add_task(data, 60);

    let results = execute_batch(&device, &batch)?;
    let candles = &results[0];

    // Verify volume accumulation
    let total_volume = candles[4]; // 5th value is volume
    let expected = 5.0 + 10.0 + 15.0 + 20.0;

    assert_eq!(total_volume, expected, "Volume should accumulate");

    println!("✅ Volume accumulation test passed");
    Ok(())
}

#[cfg(not(feature = "gpu"))]
#[test]
fn test_time_bars_gpu_feature_required() {
    println!("⚠️  Time bar tests require 'gpu' feature");
}
