//! VWAP Anchored GPU Demo
//!
//! Demonstrates GPU-accelerated VWAP Anchored indicator with:
//! - Custom anchor points (session start, intraday pivots)
//! - Multiple anchor scenarios (daily, hourly sessions)
//! - Performance comparison: GPU vs CPU
//! - Visual output with ASCII charts
//!
//! # Run
//!
//! ```bash
//! cargo run --example vwap_anchored_gpu_demo --features gpu
//! ```

use kimsfinance_core::gpu::{GpuDevice, vwap_anchored_gpu};
use ndarray::Array1;
use std::time::Instant;

/// Generate sample intraday OHLCV data with session breaks
fn generate_intraday_data(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    // Simulate 3 trading sessions
    let session_length = n / 3;

    for session in 0..3 {
        let base_price = 100.0 + session as f64 * 5.0; // Each session starts higher
        let trend = if session % 2 == 0 { 1.0 } else { -0.5 }; // Alternate trends

        for i in 0..session_length {
            let t = i as f64 * 0.1;
            let price_offset = trend * i as f64 * 0.05 + 3.0 * (t * 0.5).sin();

            high.push(base_price + price_offset + 2.0);
            low.push(base_price + price_offset - 2.0);
            close.push(base_price + price_offset);
            volume.push(1000.0 + 500.0 * (t * 0.3).sin().abs());
        }
    }

    (
        Array1::from(high),
        Array1::from(low),
        Array1::from(close),
        Array1::from(volume),
    )
}

/// CPU-only VWAP Anchored implementation (for comparison)
fn vwap_anchored_cpu(
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    volume: &Array1<f64>,
    anchor_index: usize,
) -> Array1<f64> {
    let n = high.len();
    let mut vwap = Array1::from_elem(n, f64::NAN);

    // Calculate typical prices
    let mut typical_price = Array1::zeros(n);
    for i in 0..n {
        typical_price[i] = (high[i] + low[i] + close[i]) / 3.0;
    }

    // Cumulative sums from anchor
    let mut cumsum_tpv = typical_price[anchor_index] * volume[anchor_index];
    let mut cumsum_volume = volume[anchor_index];

    if cumsum_volume > 0.0 {
        vwap[anchor_index] = cumsum_tpv / cumsum_volume;
    }

    for i in (anchor_index + 1)..n {
        cumsum_tpv += typical_price[i] * volume[i];
        cumsum_volume += volume[i];

        if cumsum_volume > 0.0 {
            vwap[i] = cumsum_tpv / cumsum_volume;
        }
    }

    vwap
}

/// Simple ASCII chart visualization
fn print_ascii_chart(prices: &[f64], vwap: &[f64], title: &str, start: usize, end: usize) {
    println!("\n{}", title);
    println!("{}", "=".repeat(60));

    let data_slice = &prices[start..end];
    let vwap_slice = &vwap[start..end];

    let min_price = data_slice
        .iter()
        .chain(vwap_slice.iter())
        .filter(|x| x.is_finite())
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let max_price = data_slice
        .iter()
        .chain(vwap_slice.iter())
        .filter(|x| x.is_finite())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let height = 20;
    let width = end - start;

    for row in 0..height {
        let price_level = max_price - (max_price - min_price) * row as f64 / height as f64;
        print!("{:6.2} |", price_level);

        for col in 0..width {
            let price = data_slice[col];
            let vwap_val = vwap_slice[col];

            let price_dist = (price - price_level).abs();
            let vwap_dist = (vwap_val - price_level).abs();

            let step = (max_price - min_price) / height as f64;

            if price_dist < step * 0.5 {
                print!("*");
            } else if vwap_val.is_finite() && vwap_dist < step * 0.5 {
                print!("-");
            } else {
                print!(" ");
            }
        }

        println!();
    }

    println!("       {}", "-".repeat(width));
    println!(
        "       {}   (* = Close, - = VWAP)",
        " ".repeat(width / 2 - 10)
    );
}

fn main() {
    println!("VWAP Anchored GPU Demo");
    println!("======================\n");

    // Initialize GPU
    let device = match GpuDevice::new() {
        Ok(dev) => {
            println!("✓ GPU initialized successfully");
            dev
        }
        Err(e) => {
            eprintln!("✗ Failed to initialize GPU: {:?}", e);
            eprintln!("This example requires NVIDIA GPU with CUDA support");
            return;
        }
    };

    // Generate sample data
    let n = 30_000; // 3 trading sessions × 10,000 candles
    println!("✓ Generating {} candles of intraday data...", n);
    let (high, low, close, volume) = generate_intraday_data(n);

    // === Scenario 1: Single Session VWAP (anchor at start) ===
    println!("\n" + &"=".repeat(60));
    println!("Scenario 1: Single Session VWAP (anchor at 0)");
    println!("{}", "=".repeat(60));

    let anchor1 = 0;
    let start = Instant::now();
    let vwap1_gpu = vwap_anchored_gpu(&device, &high, &low, &close, &volume, anchor1, None)
        .expect("GPU VWAP calculation failed");
    let gpu_time1 = start.elapsed();

    let start = Instant::now();
    let vwap1_cpu = vwap_anchored_cpu(&high, &low, &close, &volume, anchor1);
    let cpu_time1 = start.elapsed();

    println!("GPU Time: {:.2}ms", gpu_time1.as_secs_f64() * 1000.0);
    println!("CPU Time: {:.2}ms", cpu_time1.as_secs_f64() * 1000.0);
    println!(
        "Speedup:  {:.2}x",
        cpu_time1.as_secs_f64() / gpu_time1.as_secs_f64()
    );

    // Print sample values
    println!("\nSample VWAP values (first 10 candles):");
    for i in 0..10 {
        println!(
            "  Candle {}: Close={:.2}, VWAP={:.2}",
            i, close[i], vwap1_gpu[i]
        );
    }

    // === Scenario 2: Multi-Session VWAP (anchors at session starts) ===
    println!("\n" + &"=".repeat(60));
    println!("Scenario 2: Multi-Session VWAP");
    println!("{}", "=".repeat(60));

    let session_length = n / 3;
    let anchors = vec![0, session_length, 2 * session_length];

    for (session_num, &anchor) in anchors.iter().enumerate() {
        println!("\nSession {} (anchor at {}):", session_num + 1, anchor);

        let start = Instant::now();
        let vwap_gpu =
            vwap_anchored_gpu(&device, &high, &low, &close, &volume, anchor, None).unwrap();
        let gpu_time = start.elapsed();

        let start = Instant::now();
        let vwap_cpu = vwap_anchored_cpu(&high, &low, &close, &volume, anchor);
        let cpu_time = start.elapsed();

        println!("  GPU: {:.2}ms", gpu_time.as_secs_f64() * 1000.0);
        println!("  CPU: {:.2}ms", cpu_time.as_secs_f64() * 1000.0);
        println!(
            "  Speedup: {:.2}x",
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
        );

        // Verify GPU vs CPU match
        let mut max_diff = 0.0;
        for i in anchor..n {
            let diff = (vwap_gpu[i] - vwap_cpu[i]).abs();
            if diff.is_finite() {
                max_diff = max_diff.max(diff);
            }
        }
        println!("  Max GPU-CPU difference: {:.10}", max_diff);
    }

    // === Scenario 3: Intraday Pivot VWAP (anchor at mid-session) ===
    println!("\n" + &"=".repeat(60));
    println!("Scenario 3: Intraday Pivot VWAP");
    println!("{}", "=".repeat(60));

    let pivot_anchor = session_length + session_length / 2; // Mid-second session
    println!("Pivot anchor at candle {}", pivot_anchor);

    let start = Instant::now();
    let vwap_pivot_gpu =
        vwap_anchored_gpu(&device, &high, &low, &close, &volume, pivot_anchor, None).unwrap();
    let gpu_time = start.elapsed();

    let start = Instant::now();
    let vwap_pivot_cpu = vwap_anchored_cpu(&high, &low, &close, &volume, pivot_anchor);
    let cpu_time = start.elapsed();

    println!("GPU Time: {:.2}ms", gpu_time.as_secs_f64() * 1000.0);
    println!("CPU Time: {:.2}ms", cpu_time.as_secs_f64() * 1000.0);
    println!(
        "Speedup:  {:.2}x",
        cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
    );

    // === Visualization ===
    println!("\n" + &"=".repeat(60));
    println!("Visualization: Session 2 with Pivot VWAP");
    println!("{}", "=".repeat(60));

    let viz_start = session_length;
    let viz_end = viz_start + 100; // Show 100 candles

    print_ascii_chart(
        close.as_slice().unwrap(),
        vwap_pivot_gpu.as_slice().unwrap(),
        "Close Price vs VWAP Anchored (Pivot at mid-session)",
        viz_start,
        viz_end,
    );

    // === Performance Summary ===
    println!("\n" + &"=".repeat(60));
    println!("Performance Summary");
    println!("{}", "=".repeat(60));

    // Large dataset test
    let n_large = 100_000;
    println!("\nLarge dataset test (n={}):", n_large);
    let (high_large, low_large, close_large, volume_large) = generate_intraday_data(n_large);
    let anchor_large = n_large / 10;

    let iterations = 10;
    let mut gpu_times = Vec::new();
    let mut cpu_times = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = vwap_anchored_gpu(
            &device,
            &high_large,
            &low_large,
            &close_large,
            &volume_large,
            anchor_large,
            None,
        )
        .unwrap();
        gpu_times.push(start.elapsed().as_secs_f64());

        let start = Instant::now();
        let _ = vwap_anchored_cpu(
            &high_large,
            &low_large,
            &close_large,
            &volume_large,
            anchor_large,
        );
        cpu_times.push(start.elapsed().as_secs_f64());
    }

    let avg_gpu = gpu_times.iter().sum::<f64>() / iterations as f64;
    let avg_cpu = cpu_times.iter().sum::<f64>() / iterations as f64;
    let speedup = avg_cpu / avg_gpu;

    println!("Average GPU time: {:.2}ms", avg_gpu * 1000.0);
    println!("Average CPU time: {:.2}ms", avg_cpu * 1000.0);
    println!("Average speedup:  {:.2}x", speedup);
    println!(
        "Throughput:       {:.0} candles/sec",
        n_large as f64 / avg_gpu
    );

    if speedup >= 5.0 {
        println!("\n✓ Performance target met: {:.2}x >= 5.0x", speedup);
    } else {
        println!(
            "\n⚠ Performance target missed: {:.2}x < 5.0x (expected for 100K candles)",
            speedup
        );
    }

    println!("\n" + &"=".repeat(60));
    println!("Demo completed successfully!");
    println!("{}", "=".repeat(60));
}
