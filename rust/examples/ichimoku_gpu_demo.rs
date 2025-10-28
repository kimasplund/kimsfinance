//! Ichimoku Cloud GPU Demo
//!
//! Demonstrates GPU-accelerated Ichimoku Cloud calculation with
//! real-world market data patterns.
//!
//! # Running
//!
//! ```bash
//! cargo run --example ichimoku_gpu_demo --features gpu
//! ```

use kimsfinance_core::gpu::{GpuDevice, ichimoku_gpu};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ichimoku Cloud GPU Demo ===\n");

    // Initialize GPU device
    let device = Arc::new(GpuDevice::new()?);
    println!("GPU Device initialized successfully");
    println!("Device: {:?}\n", device.device.name()?);

    // Generate synthetic market data (trending upward with volatility)
    let n = 10_000;
    println!("Generating {} candles of synthetic market data...", n);

    let high: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 100.0 + (i as f64 * 0.05);
            let cycle = (i as f64 * 0.01).sin() * 5.0;
            let noise = ((i as f64 * 0.1).sin() * (i as f64 * 0.05).cos()) * 2.0;
            trend + cycle + noise + 5.0 // High is above close
        })
        .collect();

    let low: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 100.0 + (i as f64 * 0.05);
            let cycle = (i as f64 * 0.01).sin() * 5.0;
            let noise = ((i as f64 * 0.1).sin() * (i as f64 * 0.05).cos()) * 2.0;
            trend + cycle + noise - 5.0 // Low is below close
        })
        .collect();

    let close: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 100.0 + (i as f64 * 0.05);
            let cycle = (i as f64 * 0.01).sin() * 5.0;
            let noise = ((i as f64 * 0.1).sin() * (i as f64 * 0.05).cos()) * 2.0;
            trend + cycle + noise
        })
        .collect();

    println!("Price range:");
    println!("  High:  {:.2} - {:.2}", high[0], high[n - 1]);
    println!("  Low:   {:.2} - {:.2}", low[0], low[n - 1]);
    println!("  Close: {:.2} - {:.2}\n", close[0], close[n - 1]);

    // Calculate Ichimoku Cloud on GPU
    println!("Calculating Ichimoku Cloud on GPU...");
    let start = std::time::Instant::now();
    let result = ichimoku_gpu(device, &high, &low, &close, None)?;
    let elapsed = start.elapsed();

    println!(
        "✓ Calculation complete in {:.2}ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "  Throughput: {:.0} candles/sec\n",
        n as f64 / elapsed.as_secs_f64()
    );

    // Display results for recent candles
    println!("=== Ichimoku Cloud Values (Last 10 Candles) ===\n");
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Index", "Tenkan-sen", "Kijun-sen", "Span A", "Span B", "Chikou"
    );
    println!("{}", "-".repeat(78));

    for i in (n - 10)..n {
        println!(
            "{:>6} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
            i,
            result.tenkan_sen[i],
            result.kijun_sen[i],
            result.senkou_span_a[i],
            result.senkou_span_b[i],
            result.chikou_span[i]
        );
    }

    // Analyze cloud signals
    println!("\n=== Cloud Analysis (Latest Position) ===\n");
    let latest_idx = n - 1;

    let tenkan = result.tenkan_sen[latest_idx];
    let kijun = result.kijun_sen[latest_idx];
    let span_a = result.senkou_span_a[latest_idx];
    let span_b = result.senkou_span_b[latest_idx];
    let price = close[latest_idx];

    println!("Current Price: {:.2}", price);
    println!("Tenkan-sen (Conversion Line): {:.2}", tenkan);
    println!("Kijun-sen (Base Line): {:.2}", kijun);
    println!("Senkou Span A (Leading Span A): {:.2}", span_a);
    println!("Senkou Span B (Leading Span B): {:.2}", span_b);

    // Determine cloud color
    let cloud_color = if span_a > span_b {
        "Bullish (Green)"
    } else {
        "Bearish (Red)"
    };
    println!("\nCloud Color: {}", cloud_color);

    // Determine price position relative to cloud
    let cloud_top = span_a.max(span_b);
    let cloud_bottom = span_a.min(span_b);

    let price_position = if price > cloud_top {
        "Above Cloud (Bullish)"
    } else if price < cloud_bottom {
        "Below Cloud (Bearish)"
    } else {
        "Inside Cloud (Neutral/Consolidation)"
    };
    println!("Price Position: {}", price_position);

    // TK Cross signal
    let tk_signal = if tenkan > kijun {
        "Tenkan above Kijun (Bullish)"
    } else if tenkan < kijun {
        "Tenkan below Kijun (Bearish)"
    } else {
        "Tenkan equals Kijun (Neutral)"
    };
    println!("TK Cross: {}", tk_signal);

    // Calculate cloud thickness (support/resistance strength)
    let cloud_thickness = (span_a - span_b).abs();
    println!(
        "\nCloud Thickness: {:.2} ({:.1}% of price)",
        cloud_thickness,
        (cloud_thickness / price) * 100.0
    );

    // Statistics
    println!("\n=== Statistics ===\n");

    // Count valid values (non-NaN)
    let valid_tenkan = result.tenkan_sen.iter().filter(|x| x.is_finite()).count();
    let valid_kijun = result.kijun_sen.iter().filter(|x| x.is_finite()).count();
    let valid_span_a = result
        .senkou_span_a
        .iter()
        .filter(|x| x.is_finite() && **x != 0.0)
        .count();
    let valid_span_b = result
        .senkou_span_b
        .iter()
        .filter(|x| x.is_finite() && **x != 0.0)
        .count();
    let valid_chikou = result.chikou_span.iter().filter(|x| x.is_finite()).count();

    println!("Valid data points:");
    println!(
        "  Tenkan-sen:    {}/{} ({:.1}%)",
        valid_tenkan,
        n,
        (valid_tenkan as f64 / n as f64) * 100.0
    );
    println!(
        "  Kijun-sen:     {}/{} ({:.1}%)",
        valid_kijun,
        n,
        (valid_kijun as f64 / n as f64) * 100.0
    );
    println!(
        "  Senkou Span A: {}/{} ({:.1}%)",
        valid_span_a,
        n,
        (valid_span_a as f64 / n as f64) * 100.0
    );
    println!(
        "  Senkou Span B: {}/{} ({:.1}%)",
        valid_span_b,
        n,
        (valid_span_b as f64 / n as f64) * 100.0
    );
    println!(
        "  Chikou Span:   {}/{} ({:.1}%)",
        valid_chikou,
        n,
        (valid_chikou as f64 / n as f64) * 100.0
    );

    println!("\n✓ Demo complete!");

    Ok(())
}
