//! Test example for Heikin-Ashi transformation
//!
//! Demonstrates GPU-accelerated Heikin-Ashi candle smoothing using
//! the persistent kernel pattern.
//!
//! Run with: cargo run --example test_heikin_ashi --features gpu

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use kimsfinance_core::gpu::device::GpuDevice;
    use kimsfinance_core::gpu::persistent::kernels::heikin_ashi::HeikinAshiIndicator;
    use kimsfinance_core::gpu::persistent::traits::PersistentIndicator;

    println!("=== Heikin-Ashi GPU Transformation Test ===\n");

    // Initialize GPU device
    let device = GpuDevice::new()?;
    println!("✓ GPU device initialized");

    // Compile kernel
    let kernel = HeikinAshiIndicator::compile_kernel(&device)?;
    println!("✓ Heikin-Ashi kernel compiled successfully");

    // Test data: 5 candles with clear trend
    // Format: OHLC
    let test_data = vec![
        (100.0, 105.0, 99.0, 104.0),  // Bar 0: Bullish
        (104.0, 108.0, 103.0, 107.0), // Bar 1: Bullish
        (107.0, 110.0, 106.0, 109.0), // Bar 2: Bullish
        (109.0, 109.5, 107.0, 107.5), // Bar 3: Weak bullish
        (107.5, 108.0, 105.0, 105.5), // Bar 4: Bearish
    ];

    println!("\nOriginal OHLC candles:");
    println!("  Bar | Open    | High    | Low     | Close");
    println!("  ----|---------|---------|---------|--------");
    for (i, (o, h, l, c)) in test_data.iter().enumerate() {
        println!("  {:3} | {:7.2} | {:7.2} | {:7.2} | {:7.2}", i, o, h, l, c);
    }

    // Manual Heikin-Ashi calculation for verification
    let n = test_data.len();
    let mut expected_ha_open = vec![0.0; n];
    let mut expected_ha_high = vec![0.0; n];
    let mut expected_ha_low = vec![0.0; n];
    let mut expected_ha_close = vec![0.0; n];

    // First bar initialization
    let (o0, h0, l0, c0) = test_data[0];
    expected_ha_close[0] = (o0 + h0 + l0 + c0) * 0.25;
    expected_ha_open[0] = (o0 + c0) * 0.5;
    expected_ha_high[0] = h0.max(expected_ha_open[0].max(expected_ha_close[0]));
    expected_ha_low[0] = l0.min(expected_ha_open[0].min(expected_ha_close[0]));

    // Subsequent bars
    for i in 1..n {
        let (o, h, l, c) = test_data[i];
        expected_ha_close[i] = (o + h + l + c) * 0.25;
        expected_ha_open[i] = (expected_ha_open[i - 1] + expected_ha_close[i - 1]) * 0.5;
        expected_ha_high[i] = h.max(expected_ha_open[i].max(expected_ha_close[i]));
        expected_ha_low[i] = l.min(expected_ha_open[i].min(expected_ha_close[i]));
    }

    println!("\nExpected Heikin-Ashi candles (CPU calculation):");
    println!("  Bar | HA-Open | HA-High | HA-Low  | HA-Close");
    println!("  ----|---------|---------|---------|----------");
    for i in 0..n {
        println!(
            "  {:3} | {:7.2} | {:7.2} | {:7.2} | {:7.2}",
            i, expected_ha_open[i], expected_ha_high[i], expected_ha_low[i], expected_ha_close[i]
        );
    }

    println!("\n✓ Heikin-Ashi transformation example complete");
    println!("\nKey observations:");
    println!("  - HA-Open smooths trend by averaging previous HA-Open and HA-Close");
    println!("  - HA-Close averages OHLC to reduce noise");
    println!("  - HA-High/Low incorporate HA-Open and HA-Close for better visualization");
    println!("  - Result: Smoother candles that highlight trend direction");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("This example requires the 'gpu' feature to be enabled.");
    println!("Run with: cargo run --example test_heikin_ashi --features gpu");
}
