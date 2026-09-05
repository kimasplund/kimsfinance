//! Range and Renko Bar Tests
//!
//! Validates range-based and Renko brick formation.
//! Tests price movement logic, trending/ranging scenarios, and brick formation.

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{GpuDevice, RangeBarBatch, RenkoBatch, execute_batch};

// ============================================================================
// Range Bar Tests
// ============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_range_bars_fixed_range() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Trades with specific price movements, target range: 5.0
    let trades = vec![
        (0.0, 100.0, 10.0), // Bar 1 starts at 100
        (1.0, 102.0, 11.0),
        (2.0, 105.0, 12.0), // Bar 1 closes (range = 105 - 100 = 5)
        (3.0, 106.0, 13.0), // Bar 2 starts at ~105
        (4.0, 110.0, 14.0), // Bar 2 closes (range = 110 - 105 = 5)
        (5.0, 111.0, 15.0), // Bar 3 starts
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = RangeBarBatch::new();
    batch.add_task(data, 5.0); // 5.0 price range threshold

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    // Should have 2 complete bars
    assert!(bars.len() >= 10, "Expected at least 2 complete bars");

    // Bar 1: Range should be ~5.0
    let bar1_high = bars[1];
    let bar1_low = bars[2];
    let bar1_range = bar1_high - bar1_low;

    assert!(
        (bar1_range - 5.0).abs() < 0.1,
        "Bar 1 range should be ~5.0, got {:.2}",
        bar1_range
    );

    // Bar 2: Range should be ~5.0
    let bar2_high = bars[6];
    let bar2_low = bars[7];
    let bar2_range = bar2_high - bar2_low;

    assert!(
        (bar2_range - 5.0).abs() < 0.1,
        "Bar 2 range should be ~5.0, got {:.2}",
        bar2_range
    );

    println!("✅ Fixed range threshold test passed");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_range_bars_uptrend() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Strong uptrend with consistent range
    let trades = vec![
        (0.0, 100.0, 10.0),
        (1.0, 101.0, 11.0),
        (2.0, 103.0, 12.0),
        (3.0, 103.5, 13.0), // Range bar closes (H-L = ~3.5)
        (4.0, 104.0, 14.0),
        (5.0, 106.0, 15.0),
        (6.0, 107.5, 16.0), // Next bar (range ~3.5)
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = RangeBarBatch::new();
    batch.add_task(data, 3.5);

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    // Verify uptrend: Close > Open for all bars
    let num_bars = bars.len() / 5;
    for i in 0..num_bars {
        let offset = i * 5;
        let open = bars[offset];
        let close = bars[offset + 3];

        assert!(
            close >= open,
            "Uptrend: Close should be >= Open in bar {}",
            i
        );
    }

    println!("✅ Range bar uptrend handling verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_range_bars_downtrend() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Strong downtrend
    let trades = vec![
        (0.0, 100.0, 10.0),
        (1.0, 98.0, 11.0),
        (2.0, 96.0, 12.0),
        (3.0, 95.0, 13.0), // Range bar closes (100 - 95 = 5)
        (4.0, 94.0, 14.0),
        (5.0, 92.0, 15.0),
        (6.0, 90.0, 16.0), // Next bar
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = RangeBarBatch::new();
    batch.add_task(data, 5.0);

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    // Verify downtrend: Close < Open for bars
    let num_bars = bars.len() / 5;
    assert!(num_bars >= 1, "Should have at least 1 bar");

    let open = bars[0];
    let close = bars[3];

    assert!(close < open, "Downtrend: Close should be < Open");

    println!("✅ Range bar downtrend handling verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_range_bars_ranging_market() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Sideways/ranging market (oscillating within range)
    let trades = vec![
        (0.0, 100.0, 10.0),
        (1.0, 102.0, 11.0),
        (2.0, 98.0, 12.0),
        (3.0, 103.0, 13.0), // High
        (4.0, 97.0, 14.0),  // Low (range = 6)
        (5.0, 100.0, 15.0), // Back to middle
        (6.0, 102.0, 16.0),
        (7.0, 98.0, 17.0),
        (8.0, 104.0, 18.0), // Another oscillation
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = RangeBarBatch::new();
    batch.add_task(data, 6.0);

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    // Should form bars based on range, not direction
    assert!(bars.len() >= 5, "Should capture ranging movement");

    // In ranging market, body (|close - open|) should be small relative to range
    let num_bars = bars.len() / 5;
    for i in 0..num_bars {
        let offset = i * 5;
        let open = bars[offset];
        let high = bars[offset + 1];
        let low = bars[offset + 2];
        let close = bars[offset + 3];

        let range = high - low;
        let body = (close - open).abs();

        // In ranging market, body < range (long wicks)
        assert!(body <= range, "Body should be <= range in bar {}", i);
    }

    println!("✅ Range bar ranging market handling verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_range_bars_small_range() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Small price movements (low volatility)
    let trades: Vec<(f64, f64, f64)> = (0..20)
        .map(|i| (i as f64, 100.0 + (i as f64 * 0.05), 10.0))
        .collect();

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = RangeBarBatch::new();
    batch.add_task(data, 0.5); // Small range threshold

    let results = execute_batch(&device, &batch)?;
    let bars = &results[0];

    // Should form multiple bars with small range
    assert!(bars.len() >= 5, "Should form bars in low volatility");

    println!("✅ Small range handling verified");
    Ok(())
}

// ============================================================================
// Renko Bar Tests
// ============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_renko_brick_formation_uptrend() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Clear uptrend for brick formation, brick size = 5.0
    let trades = vec![
        (0.0, 100.0, 10.0), // Brick 1 base
        (1.0, 105.0, 11.0), // Brick 1 completes (100 -> 105)
        (2.0, 110.0, 12.0), // Brick 2 completes (105 -> 110)
        (3.0, 115.0, 13.0), // Brick 3 completes (110 -> 115)
        (4.0, 117.0, 14.0), // Incomplete brick
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = RenkoBatch::new();
    batch.add_task(data, 5.0); // 5.0 brick size

    let results = execute_batch(&device, &batch)?;
    let bricks = &results[0];

    // Should have 3 complete bricks
    assert_eq!(bricks.len(), 15, "Expected 3 complete bricks");

    // Verify brick structure (all bricks should be uniform size in uptrend)
    for i in 0..3 {
        let offset = i * 5;
        let open = bricks[offset];
        let high = bricks[offset + 1];
        let low = bricks[offset + 2];
        let close = bricks[offset + 3];

        // In uptrend Renko: high = close, low = open
        assert_eq!(high, close, "Uptrend brick {}: high = close", i);
        assert_eq!(low, open, "Uptrend brick {}: low = open", i);

        // Brick size should be 5.0
        let brick_size = close - open;
        assert_eq!(brick_size, 5.0, "Brick {} size should be 5.0", i);
    }

    // Verify sequential brick alignment
    let brick1_close = bricks[3];
    let brick2_open = bricks[5];
    assert_eq!(brick1_close, brick2_open, "Bricks should connect");

    println!("✅ Renko brick formation (uptrend) verified");
    Ok(())
}

#[cfg(feature =="gpu")]
#[test]
fn test_renko_brick_formation_downtrend() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Clear downtrend
    let trades = vec![
        (0.0, 100.0, 10.0),
        (1.0, 95.0, 11.0), // Brick 1: 100 -> 95
        (2.0, 90.0, 12.0), // Brick 2: 95 -> 90
        (3.0, 85.0, 13.0), // Brick 3: 90 -> 85
        (4.0, 83.0, 14.0), // Incomplete
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = RenkoBatch::new();
    batch.add_task(data, 5.0);

    let results = execute_batch(&device, &batch)?;
    let bricks = &results[0];

    assert_eq!(bricks.len(), 15, "Expected 3 downtrend bricks");

    // Verify downtrend brick structure
    for i in 0..3 {
        let offset = i * 5;
        let open = bricks[offset];
        let high = bricks[offset + 1];
        let low = bricks[offset + 2];
        let close = bricks[offset + 3];

        // In downtrend Renko: high = open, low = close
        assert_eq!(high, open, "Downtrend brick {}: high = open", i);
        assert_eq!(low, close, "Downtrend brick {}: low = close", i);

        // Brick size should be -5.0 (negative)
        let brick_size = close - open;
        assert_eq!(brick_size, -5.0, "Brick {} size should be -5.0", i);
    }

    println!("✅ Renko brick formation (downtrend) verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_renko_reversal_detection() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Uptrend followed by reversal
    let trades = vec![
        (0.0, 100.0, 10.0),
        (1.0, 105.0, 11.0), // Up brick
        (2.0, 110.0, 12.0), // Up brick
        (3.0, 100.0, 13.0), // Reversal! (110 -> 100 = -10, two down bricks)
        (4.0, 95.0, 14.0),  // Continuation down
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = RenkoBatch::new();
    batch.add_task(data, 5.0);

    let results = execute_batch(&device, &batch)?;
    let bricks = &results[0];

    // Should have: 2 up bricks + 2 down bricks (reversal) + 1 down brick
    assert!(bricks.len() >= 20, "Should detect reversal bricks");

    // Verify first 2 are up bricks
    let brick1_close = bricks[3];
    let brick1_open = bricks[0];
    assert!(brick1_close > brick1_open, "Brick 1 should be up");

    let brick2_close = bricks[8];
    let brick2_open = bricks[5];
    assert!(brick2_close > brick2_open, "Brick 2 should be up");

    // Later bricks should be down (after reversal)
    // Check if any brick has close < open
    let has_down_brick = (0..bricks.len() / 5).any(|i| {
        let offset = i * 5;
        bricks[offset + 3] < bricks[offset]
    });

    assert!(
        has_down_brick,
        "Should have downtrend bricks after reversal"
    );

    println!("✅ Renko reversal detection verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_renko_noise_filtering() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Noisy data with small oscillations
    let trades = vec![
        (0.0, 100.0, 10.0),
        (1.0, 101.0, 11.0), // +1 (noise)
        (2.0, 99.0, 12.0),  // -2 (noise)
        (3.0, 102.0, 13.0), // +3 (noise)
        (4.0, 105.0, 14.0), // +3, total movement = 5, brick forms
        (5.0, 104.0, 15.0), // -1 (noise)
        (6.0, 110.0, 16.0), // +6, another brick
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = RenkoBatch::new();
    batch.add_task(data, 5.0);

    let results = execute_batch(&device, &batch)?;
    let bricks = &results[0];

    // Should filter noise and produce clean bricks
    assert_eq!(bricks.len(), 10, "Expected 2 clean bricks (noise filtered)");

    println!("✅ Renko noise filtering verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_renko_multiple_brick_jump() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Large price jump forming multiple bricks at once
    let trades = vec![
        (0.0, 100.0, 10.0),
        (1.0, 120.0, 11.0), // Jump of 20 = 4 bricks (brick size 5)
        (2.0, 122.0, 12.0),
    ];

    let mut data = Vec::new();
    let timestamps: Vec<f64> = trades.iter().map(|t| t.0).collect();
    let prices: Vec<f64> = trades.iter().map(|t| t.1).collect();
    let volumes: Vec<f64> = trades.iter().map(|t| t.2).collect();

    data.extend(&timestamps);
    data.extend(&prices);
    data.extend(&volumes);

    let mut batch = RenkoBatch::new();
    batch.add_task(data, 5.0);

    let results = execute_batch(&device, &batch)?;
    let bricks = &results[0];

    // Should create 4 bricks: 100->105, 105->110, 110->115, 115->120
    assert_eq!(bricks.len(), 20, "Expected 4 bricks from large jump");

    // Verify all 4 bricks have correct size
    for i in 0..4 {
        let offset = i * 5;
        let open = bricks[offset];
        let close = bricks[offset + 3];
        let expected_open = 100.0 + (i as f64 * 5.0);
        let expected_close = 105.0 + (i as f64 * 5.0);

        assert_eq!(open, expected_open, "Brick {} open", i);
        assert_eq!(close, expected_close, "Brick {} close", i);
    }

    println!("✅ Renko multiple brick jump handling verified");
    Ok(())
}

#[cfg(not(feature = "gpu"))]
#[test]
fn test_range_renko_gpu_feature_required() {
    println!("⚠️  Range and Renko bar tests require 'gpu' feature");
}
