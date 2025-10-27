//! Volume and Tick Bar Tests
//!
//! Validates volume-based and tick-based candle aggregation.
//! Tests bar formation logic, threshold handling, and edge cases.

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{execute_batch, GpuDevice, TickBarBatch, VolumeBarBatch};

// ============================================================================
// Volume Bar Tests
// ============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_volume_bars_fixed_threshold() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Trades with known volumes, target: 50 volume per bar
    let trades = vec![
        (0.0, 100.0, 20.0),  // Bar 1 starts
        (1.0, 102.0, 15.0),  // Bar 1: 20 + 15 = 35
        (2.0, 104.0, 15.0),  // Bar 1 closes: 20 + 15 + 15 = 50
        (3.0, 103.0, 25.0),  // Bar 2 starts
        (4.0, 105.0, 25.0),  // Bar 2 closes: 25 + 25 = 50
        (5.0, 106.0, 10.0),  // Bar 3 starts (incomplete)
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = VolumeBarBatch::new();
    batch.add_task(data, 50.0); // 50 volume threshold

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    // Should have 2 complete bars (+ possibly 1 incomplete)
    assert!(bars.len() >= 10, "Expected at least 2 complete bars");

    // Bar 1: First 3 trades (volume = 50)
    let bar1_open = bars[0];
    let bar1_high = bars[1];
    let bar1_low = bars[2];
    let bar1_close = bars[3];
    let bar1_volume = bars[4];

    assert_eq!(bar1_open, 100.0, "Bar 1 open");
    assert_eq!(bar1_high, 104.0, "Bar 1 high");
    assert_eq!(bar1_low, 100.0, "Bar 1 low");
    assert_eq!(bar1_close, 104.0, "Bar 1 close");
    assert_eq!(bar1_volume, 50.0, "Bar 1 volume");

    // Bar 2: Next 2 trades (volume = 50)
    let bar2_open = bars[5];
    let bar2_close = bars[8];
    let bar2_volume = bars[9];

    assert_eq!(bar2_open, 103.0, "Bar 2 open");
    assert_eq!(bar2_close, 105.0, "Bar 2 close");
    assert_eq!(bar2_volume, 50.0, "Bar 2 volume");

    println!("✅ Fixed volume threshold test passed");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_volume_bars_large_single_trade() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Single trade exceeds threshold
    let trades = vec![
        (0.0, 100.0, 20.0),
        (1.0, 102.0, 150.0), // Single trade > 50 threshold, should form its own bar
        (2.0, 103.0, 30.0),
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = VolumeBarBatch::new();
    batch.add_task(data, 50.0);

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    // Should handle oversized trade gracefully
    // Either: 1 bar with 150 volume, or split across multiple bars
    assert!(bars.len() >= 5, "Should process large single trade");

    // Find the bar with large volume
    let large_bar_exists = (0..bars.len() / 5)
        .any(|i| bars[i * 5 + 4] >= 100.0);

    assert!(large_bar_exists, "Large volume trade should be captured");

    println!("✅ Large single trade handling verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_volume_bars_accumulation() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Many small trades accumulating to threshold
    let trades: Vec<(f64, f64, f64)> = (0..10)
        .map(|i| (i as f64, 100.0 + i as f64, 5.0))
        .collect();

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = VolumeBarBatch::new();
    batch.add_task(data, 25.0); // 25 volume threshold

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    // 10 trades * 5 volume = 50 total, should produce 2 bars
    assert_eq!(bars.len(), 10, "Expected 2 complete bars");

    // Verify accumulation
    let bar1_volume = bars[4];
    let bar2_volume = bars[9];

    assert_eq!(bar1_volume, 25.0, "Bar 1 accumulated volume");
    assert_eq!(bar2_volume, 25.0, "Bar 2 accumulated volume");

    println!("✅ Volume accumulation test passed");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_volume_bars_ohlc_correctness() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Specific price movements within bar
    let trades = vec![
        (0.0, 100.0, 10.0), // Open
        (1.0, 105.0, 10.0), // High
        (2.0, 98.0, 10.0),  // Low
        (3.0, 102.0, 10.0), // Close (40 volume)
        (4.0, 103.0, 5.0),  // Next bar starts
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = VolumeBarBatch::new();
    batch.add_task(data, 40.0);

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    let open = bars[0];
    let high = bars[1];
    let low = bars[2];
    let close = bars[3];

    assert_eq!(open, 100.0, "OHLC Open");
    assert_eq!(high, 105.0, "OHLC High");
    assert_eq!(low, 98.0, "OHLC Low");
    assert_eq!(close, 102.0, "OHLC Close");

    println!("✅ Volume bar OHLC correctness verified");
    Ok(())
}

// ============================================================================
// Tick Bar Tests
// ============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_tick_bars_fixed_count() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // 10 trades, bar every 3 ticks
    let trades: Vec<(f64, f64, f64)> = (0..10)
        .map(|i| (i as f64, 100.0 + i as f64, 10.0))
        .collect();

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = TickBarBatch::new();
    batch.add_task(data, 3); // 3 ticks per bar

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    // 10 trades / 3 ticks = 3 complete bars (+ 1 incomplete)
    assert_eq!(bars.len(), 15, "Expected 3 complete bars");

    // Bar 1: Ticks 0, 1, 2
    let bar1_open = bars[0];
    let bar1_close = bars[3];

    assert_eq!(bar1_open, 100.0, "Bar 1 open (tick 0)");
    assert_eq!(bar1_close, 102.0, "Bar 1 close (tick 2)");

    // Bar 2: Ticks 3, 4, 5
    let bar2_open = bars[5];
    let bar2_close = bars[8];

    assert_eq!(bar2_open, 103.0, "Bar 2 open (tick 3)");
    assert_eq!(bar2_close, 105.0, "Bar 2 close (tick 5)");

    println!("✅ Fixed tick count test passed");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_tick_bars_volume_aggregation() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Varying volumes
    let trades = vec![
        (0.0, 100.0, 5.0),
        (1.0, 101.0, 10.0),
        (2.0, 102.0, 15.0),  // Bar 1 closes (3 ticks)
        (3.0, 103.0, 20.0),
        (4.0, 104.0, 25.0),
        (5.0, 105.0, 30.0),  // Bar 2 closes (3 ticks)
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = TickBarBatch::new();
    batch.add_task(data, 3);

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    // Bar 1 volume: 5 + 10 + 15 = 30
    let bar1_volume = bars[4];
    assert_eq!(bar1_volume, 30.0, "Bar 1 volume");

    // Bar 2 volume: 20 + 25 + 30 = 75
    let bar2_volume = bars[9];
    assert_eq!(bar2_volume, 75.0, "Bar 2 volume");

    println!("✅ Tick bar volume aggregation verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_tick_bars_single_tick_per_bar() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // 1 tick per bar (each trade forms its own bar)
    let trades = vec![
        (0.0, 100.0, 10.0),
        (1.0, 102.0, 12.0),
        (2.0, 104.0, 14.0),
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = TickBarBatch::new();
    batch.add_task(data, 1); // 1 tick per bar

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    assert_eq!(bars.len(), 15, "Expected 3 bars (1 tick each)");

    // Each bar should have O=H=L=C
    for i in 0..3 {
        let offset = i * 5;
        let open = bars[offset];
        let high = bars[offset + 1];
        let low = bars[offset + 2];
        let close = bars[offset + 3];

        assert_eq!(open, high, "Single tick: O=H at bar {}", i);
        assert_eq!(high, low, "Single tick: H=L at bar {}", i);
        assert_eq!(low, close, "Single tick: L=C at bar {}", i);
    }

    println!("✅ Single tick per bar test passed");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_tick_bars_high_low_tracking() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Specific price movements
    let trades = vec![
        (0.0, 100.0, 10.0),
        (1.0, 110.0, 10.0), // High of bar
        (2.0, 95.0, 10.0),  // Low of bar
        (3.0, 105.0, 10.0), // Close
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = TickBarBatch::new();
    batch.add_task(data, 4); // All 4 ticks in one bar

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    let high = bars[1];
    let low = bars[2];

    assert_eq!(high, 110.0, "High tracking");
    assert_eq!(low, 95.0, "Low tracking");

    println!("✅ Tick bar high/low tracking verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_tick_bars_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Process 3 symbols with different tick counts
    let symbols = vec![
        // Symbol 1: 5 ticks, bar every 2
        vec![
            (0.0, 100.0, 10.0),
            (1.0, 101.0, 11.0),
            (2.0, 102.0, 12.0),
            (3.0, 103.0, 13.0),
            (4.0, 104.0, 14.0),
        ],
        // Symbol 2: 4 ticks, bar every 2
        vec![
            (0.0, 50.0, 5.0),
            (1.0, 51.0, 6.0),
            (2.0, 52.0, 7.0),
            (3.0, 53.0, 8.0),
        ],
        // Symbol 3: 6 ticks, bar every 2
        vec![
            (0.0, 200.0, 20.0),
            (1.0, 201.0, 21.0),
            (2.0, 202.0, 22.0),
            (3.0, 203.0, 23.0),
            (4.0, 204.0, 24.0),
            (5.0, 205.0, 25.0),
        ],
    ];

    let mut batch = TickBarBatch::new();

    for trades in symbols {
        let mut data = Vec::new();
        let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
        let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
        let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

        data.extend(&timestamps);
        data.extend(&prices);
        data.extend(&volumes);

        batch.add_task(data, 2); // 2 ticks per bar
    }

    let results = execute_batch(&device, &batch)?;

    assert_eq!(results.len(), 3, "Expected 3 symbol results");

    // Symbol 1: 5 ticks / 2 = 2 complete bars
    assert_eq!(results[0].len(), 10, "Symbol 1: 2 bars");

    // Symbol 2: 4 ticks / 2 = 2 complete bars
    assert_eq!(results[1].len(), 10, "Symbol 2: 2 bars");

    // Symbol 3: 6 ticks / 2 = 3 complete bars
    assert_eq!(results[2].len(), 15, "Symbol 3: 3 bars");

    println!("✅ Batch processing verified");
    Ok(())
}

#[cfg(not(feature = "gpu"))]
#[test]
fn test_volume_tick_bars_gpu_feature_required() {
    println!("⚠️  Volume and tick bar tests require 'gpu' feature");
}
