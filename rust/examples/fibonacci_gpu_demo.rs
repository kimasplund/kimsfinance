use kimsfinance_core::gpu::{GpuDevice, fibonacci_gpu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Fibonacci Retracement GPU Demo ===\n");

    // Initialize GPU device
    let device = GpuDevice::new()?;
    println!("GPU Device initialized successfully\n");

    // Generate sample data: uptrend then downtrend
    println!("Generating sample OHLC data (100 candles)...");
    let n = 100;
    let high: Vec<f64> = (0..n)
        .map(|i| {
            if i < 50 {
                // Uptrend: 100 -> 150
                100.0 + (i as f64 * 1.0)
            } else {
                // Downtrend: 150 -> 110
                150.0 - ((i - 50) as f64 * 0.8)
            }
        })
        .collect();

    let low: Vec<f64> = (0..n)
        .map(|i| {
            if i < 50 {
                // Uptrend: 95 -> 145
                95.0 + (i as f64 * 1.0)
            } else {
                // Downtrend: 145 -> 105
                145.0 - ((i - 50) as f64 * 0.8)
            }
        })
        .collect();

    // Calculate Fibonacci Retracement with 20-period lookback
    println!("Calculating Fibonacci Retracement (lookback=20)...\n");
    let lookback_period = 20;

    let start = std::time::Instant::now();
    let result = fibonacci_gpu(&device, &high, &low, lookback_period, None)?;
    let elapsed = start.elapsed();

    println!(
        "Calculation completed in {:.2}ms\n",
        elapsed.as_secs_f64() * 1000.0
    );

    // Display results for selected time points
    println!("=== Fibonacci Retracement Levels ===\n");

    let display_indices = [25, 50, 75, 99];
    for &idx in &display_indices {
        println!(
            "Time point {} (High: {:.2}, Low: {:.2}):",
            idx, high[idx], low[idx]
        );
        println!("  0.0%  (Swing High):  {:.2}", result.level_0[idx]);
        println!("  23.6% Retracement:   {:.2}", result.level_236[idx]);
        println!("  38.2% Retracement:   {:.2}", result.level_382[idx]);
        println!("  50.0% Retracement:   {:.2}", result.level_500[idx]);
        println!("  61.8% Retracement:   {:.2}", result.level_618[idx]);
        println!("  100.0% (Swing Low):  {:.2}", result.level_100[idx]);
        println!();
    }

    // Verify golden ratio (61.8%) calculation
    let idx = 50;
    let range = result.level_0[idx] - result.level_100[idx];
    let golden_ratio_offset = result.level_0[idx] - result.level_618[idx];
    let ratio = golden_ratio_offset / range;
    println!("=== Golden Ratio Verification at index {} ===", idx);
    println!("Swing High: {:.2}", result.level_0[idx]);
    println!("Swing Low:  {:.2}", result.level_100[idx]);
    println!("Range:      {:.2}", range);
    println!("61.8% offset from high: {:.2}", golden_ratio_offset);
    println!("Calculated ratio: {:.4} (expected: 0.6180)", ratio);
    println!();

    // Large dataset performance test
    println!("=== Performance Test ===\n");
    let large_n = 100_000;
    println!("Generating large dataset ({} candles)...", large_n);
    let large_high: Vec<f64> = (0..large_n)
        .map(|i| {
            let x = i as f64 * 0.01;
            110.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();
    let large_low: Vec<f64> = (0..large_n)
        .map(|i| {
            let x = i as f64 * 0.01;
            95.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();

    println!(
        "Calculating Fibonacci Retracement for {} candles...",
        large_n
    );
    let iterations = 10;
    let mut total_time = std::time::Duration::ZERO;

    for i in 0..iterations {
        let start = std::time::Instant::now();
        let _result = fibonacci_gpu(&device, &large_high, &large_low, 20, None)?;
        let elapsed = start.elapsed();
        total_time += elapsed;

        if i == 0 {
            println!(
                "  First run (includes compilation): {:.2}ms",
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }

    let avg_time = total_time / iterations;
    let candles_per_sec = large_n as f64 / avg_time.as_secs_f64();

    println!("\nPerformance Results ({} iterations):", iterations);
    println!(
        "  Average time: {:.2}ms ({:.2}μs)",
        avg_time.as_secs_f64() * 1000.0,
        avg_time.as_micros()
    );
    println!("  Throughput: {:.2} candles/sec", candles_per_sec);
    println!(
        "  Throughput: {:.2} M candles/sec",
        candles_per_sec / 1_000_000.0
    );

    // Expected: 10-25x speedup over CPU for 100K candles
    println!("\n=== Speedup Analysis ===");
    println!("Target: 10-25x speedup over CPU for large datasets");
    println!("Typical CPU time for 100K candles: ~2-3ms");
    println!(
        "GPU time achieved: {:.2}ms",
        avg_time.as_secs_f64() * 1000.0
    );

    let speedup_vs_cpu = 2.5 / avg_time.as_secs_f64();
    println!("Estimated speedup: {:.1}x", speedup_vs_cpu);

    if speedup_vs_cpu >= 10.0 {
        println!("✓ Target speedup achieved!");
    } else {
        println!("⚠ Target speedup not reached (may vary by hardware)");
    }

    Ok(())
}
