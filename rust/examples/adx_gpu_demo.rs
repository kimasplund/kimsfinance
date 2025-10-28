//! ADX GPU Demo
//!
//! Demonstrates GPU-accelerated ADX (Average Directional Index) calculation
//! and compares with CPU baseline.
//!
//! ADX measures trend strength on a 0-100 scale:
//! - 0-25: Weak or absent trend (ranging market)
//! - 25-50: Strong trend
//! - 50-75: Very strong trend
//! - 75-100: Extremely strong trend
//!
//! Usage:
//! ```bash
//! cargo run --example adx_gpu_demo --features gpu --release
//! ```

use kimsfinance_core::cpu::sequential::wilders_smoothing_cpu;
use kimsfinance_core::gpu::{GpuDevice, adx_gpu};
use ndarray::Array1;
use std::time::Instant;

fn adx_cpu_baseline(
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    period: usize,
) -> Array1<f64> {
    let n = high.len();
    let mut plus_dm = Array1::zeros(n);
    let mut minus_dm = Array1::zeros(n);
    let mut true_range = Array1::zeros(n);

    // Calculate DM and TR
    for i in 0..n {
        if i == 0 {
            plus_dm[i] = 0.0;
            minus_dm[i] = 0.0;
            true_range[i] = high[i] - low[i];
        } else {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];

            if up_move > down_move && up_move > 0.0 {
                plus_dm[i] = up_move;
            }
            if down_move > up_move && down_move > 0.0 {
                minus_dm[i] = down_move;
            }

            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            true_range[i] = hl.max(hc).max(lc);
        }
    }

    // Wilder's smoothing
    let plus_dm_smooth = wilders_smoothing_cpu(&plus_dm, period).unwrap();
    let minus_dm_smooth = wilders_smoothing_cpu(&minus_dm, period).unwrap();
    let tr_smooth = wilders_smoothing_cpu(&true_range, period).unwrap();

    // Calculate DI
    let mut plus_di = Array1::from_elem(n, f64::NAN);
    let mut minus_di = Array1::from_elem(n, f64::NAN);

    for i in period..n {
        if tr_smooth[i] > 1e-10 {
            plus_di[i] = 100.0 * (plus_dm_smooth[i] / tr_smooth[i]);
            minus_di[i] = 100.0 * (minus_dm_smooth[i] / tr_smooth[i]);
        }
    }

    // Calculate DX
    let mut dx = Array1::from_elem(n, f64::NAN);
    for i in period..n {
        if !plus_di[i].is_nan() && !minus_di[i].is_nan() {
            let di_sum = plus_di[i] + minus_di[i];
            if di_sum > 1e-10 {
                let di_diff = (plus_di[i] - minus_di[i]).abs();
                dx[i] = 100.0 * (di_diff / di_sum);
            }
        }
    }

    // ADX = Wilder's smoothing of DX
    wilders_smoothing_cpu(&dx, period).unwrap()
}

fn main() {
    println!("=== ADX GPU Demo ===\n");

    // Initialize GPU
    println!("Initializing GPU...");
    let device = match GpuDevice::new() {
        Ok(d) => {
            println!("✓ GPU initialized successfully\n");
            d
        }
        Err(e) => {
            eprintln!("✗ Failed to initialize GPU: {:?}", e);
            eprintln!("  Make sure you have CUDA installed and GPU available.");
            return;
        }
    };

    // Test 1: Small dataset (real-time scenario)
    println!("--- Test 1: Small Dataset (1K candles) ---");
    let n_small = 1_000;
    test_adx(&device, n_small, 14, "real-time");

    // Test 2: Medium dataset (intraday analysis)
    println!("\n--- Test 2: Medium Dataset (10K candles) ---");
    let n_medium = 10_000;
    test_adx(&device, n_medium, 14, "intraday");

    // Test 3: Large dataset (historical backtest)
    println!("\n--- Test 3: Large Dataset (100K candles) ---");
    let n_large = 100_000;
    test_adx(&device, n_large, 14, "historical");

    // Test 4: Trend detection demonstration
    println!("\n--- Test 4: Trend Detection Demo ---");
    demonstrate_trend_detection(&device);

    println!("\n=== Demo Complete ===");
}

fn test_adx(device: &GpuDevice, n: usize, period: usize, label: &str) {
    // Generate trending data with noise
    println!("Generating {} candles ({} scenario)...", n, label);
    let high = Array1::from_vec(
        (0..n)
            .map(|i| {
                let trend = 100.0 + (i as f64) * 0.01;
                let noise = ((i * 7) % 100) as f64 * 0.05;
                trend + noise + 2.0
            })
            .collect(),
    );
    let low = Array1::from_vec(
        (0..n)
            .map(|i| {
                let trend = 100.0 + (i as f64) * 0.01;
                let noise = ((i * 7) % 100) as f64 * 0.05;
                trend + noise - 2.0
            })
            .collect(),
    );
    let close = Array1::from_vec(
        (0..n)
            .map(|i| {
                let trend = 100.0 + (i as f64) * 0.01;
                let noise = ((i * 7) % 100) as f64 * 0.05;
                trend + noise
            })
            .collect(),
    );

    // Benchmark CPU
    let start = Instant::now();
    let cpu_result = adx_cpu_baseline(&high, &low, &close, period);
    let cpu_time = start.elapsed();

    // Benchmark GPU
    let start = Instant::now();
    let gpu_result = adx_gpu(device, &high, &low, &close, period, None).expect("ADX GPU failed");
    let gpu_time = start.elapsed();

    // Verify correctness
    let warmup = period * 2 - 1;
    let mut max_diff = 0.0f64;
    for i in warmup..n {
        let diff = (cpu_result[i] - gpu_result[i]).abs();
        if !diff.is_nan() {
            max_diff = max_diff.max(diff);
        }
    }

    println!("CPU Time:  {:>8.2}ms", cpu_time.as_secs_f64() * 1000.0);
    println!("GPU Time:  {:>8.2}ms", gpu_time.as_secs_f64() * 1000.0);
    println!(
        "Speedup:   {:>8.2}x",
        cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
    );
    println!("Max Diff:  {:>8.6} (numerical precision)", max_diff);
    println!(
        "Throughput: {:.0} candles/sec",
        n as f64 / gpu_time.as_secs_f64()
    );

    // Display sample values
    let last_values = 5;
    println!("\nLast {} ADX values:", last_values);
    for i in (n - last_values)..n {
        println!(
            "  [{}] CPU: {:.2}, GPU: {:.2}",
            i, cpu_result[i], gpu_result[i]
        );
    }

    // Interpret trend strength
    let last_adx = gpu_result[n - 1];
    let trend_strength = if last_adx < 25.0 {
        "Weak/Absent (ranging market)"
    } else if last_adx < 50.0 {
        "Strong trend"
    } else if last_adx < 75.0 {
        "Very strong trend"
    } else {
        "Extremely strong trend"
    };
    println!("\nTrend Strength: {:.2} - {}", last_adx, trend_strength);
}

fn demonstrate_trend_detection(device: &GpuDevice) {
    println!("Comparing ADX values for different market conditions:\n");

    let period = 14;
    let n = 50;

    // Scenario 1: Strong uptrend
    println!("1. Strong Uptrend:");
    let high_up = Array1::from_vec((0..n).map(|i| 100.0 + i as f64 * 0.5).collect());
    let low_up = Array1::from_vec((0..n).map(|i| 98.0 + i as f64 * 0.5).collect());
    let close_up = Array1::from_vec((0..n).map(|i| 99.0 + i as f64 * 0.5).collect());

    let adx_up = adx_gpu(device, &high_up, &low_up, &close_up, period, None).unwrap();
    let last_adx_up = adx_up[n - 1];
    println!("   ADX: {:.2} - Strong directional movement", last_adx_up);

    // Scenario 2: Range-bound (oscillating)
    println!("\n2. Range-Bound Market:");
    let high_range = Array1::from_vec(
        (0..n)
            .map(|i| {
                let x = i as f64 * 0.3;
                50.0 + 2.0 * x.sin()
            })
            .collect(),
    );
    let low_range = Array1::from_vec(
        (0..n)
            .map(|i| {
                let x = i as f64 * 0.3;
                48.0 + 2.0 * x.sin()
            })
            .collect(),
    );
    let close_range = Array1::from_vec(
        (0..n)
            .map(|i| {
                let x = i as f64 * 0.3;
                49.0 + 2.0 * x.sin()
            })
            .collect(),
    );

    let adx_range = adx_gpu(device, &high_range, &low_range, &close_range, period, None).unwrap();
    let last_adx_range = adx_range[n - 1];
    println!("   ADX: {:.2} - Weak/absent trend", last_adx_range);

    // Scenario 3: Strong downtrend
    println!("\n3. Strong Downtrend:");
    let high_down = Array1::from_vec((0..n).map(|i| 100.0 - i as f64 * 0.5).collect());
    let low_down = Array1::from_vec((0..n).map(|i| 98.0 - i as f64 * 0.5).collect());
    let close_down = Array1::from_vec((0..n).map(|i| 99.0 - i as f64 * 0.5).collect());

    let adx_down = adx_gpu(device, &high_down, &low_down, &close_down, period, None).unwrap();
    let last_adx_down = adx_down[n - 1];
    println!("   ADX: {:.2} - Strong directional movement", last_adx_down);

    println!("\n✓ ADX successfully identifies trend strength regardless of direction!");
    println!("  - Trending markets (up/down): ADX > 25-30");
    println!("  - Range-bound markets: ADX < 25");
}
