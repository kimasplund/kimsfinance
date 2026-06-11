//! Validate ATR GPU Performance
//!
//! This example demonstrates proper GPU benchmarking methodology and validates
//! the performance claims from PR #8 (ATR async optimization).
//!
//! # Expected Results
//!
//! **Cold Start** (includes PTX compilation):
//! - First run: 80-150ms
//! - Dominated by NVRTC compilation overhead
//!
//! **Warm Performance** (after warmup):
//! - v0.2.1 (async): ~145μs for 100K candles
//! - Should match Jules' PR #8 claim
//!
//! # Methodology
//!
//! 1. **Cold start**: Measures first invocation (includes compilation)
//! 2. **Warmup**: 5 iterations to prime CUDA cache
//! 3. **Warm timing**: 100 iterations averaged (statistical validity)
//! 4. **Synchronization**: Always sync before stopping timer
//!
//! # Usage
//!
//! ```bash
//! # Run with proper release optimizations
//! cargo run --release --example validate_atr_performance --features gpu
//!
//! # Run with more detailed output
//! RUST_LOG=debug cargo run --release --example validate_atr_performance --features gpu
//! ```
//!
//! # See Also
//!
//! - `docs/GPU_PERFORMANCE_TESTING_GUIDE.md` - Comprehensive benchmarking guide
//! - `src/gpu/atr.rs` - ATR implementation with performance notes
//! - `docs/GPU_PROFILING_RESULTS.md` - Understanding compilation overhead

use kimsfinance_core::gpu::atr::atr_gpu;
use kimsfinance_core::gpu::device::GpuDevice;
use ndarray::Array1;
use std::time::Instant;

/// Measure cold start performance (includes compilation overhead)
fn measure_cold_start(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    period: usize,
) -> u128 {
    let start = Instant::now();
    let _ = atr_gpu(device, high, low, close, period, None).expect("ATR calculation failed");
    device.synchronize().expect("Stream sync failed");
    start.elapsed().as_micros()
}

/// Warmup phase - prime CUDA cache (PTX + SASS compilation)
fn warmup(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    period: usize,
    iterations: usize,
) {
    for _ in 0..iterations {
        let _ = atr_gpu(device, high, low, close, period, None).expect("Warmup failed");
    }
    device.synchronize().expect("Warmup sync failed");
}

/// Measure warm performance (average of many iterations)
fn measure_warm_average(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    period: usize,
    iterations: usize,
) -> (u128, u128, u128, u128, Vec<u128>) {
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = atr_gpu(device, high, low, close, period, None).expect("Warm timing failed");
        device.synchronize().expect("Warm sync failed");
        times.push(start.elapsed().as_micros());
    }

    // Calculate statistics
    let mean = times.iter().sum::<u128>() / times.len() as u128;
    let variance = times
        .iter()
        .map(|&t| (t as i128 - mean as i128).pow(2) as u128)
        .sum::<u128>()
        / times.len() as u128;
    let stddev = (variance as f64).sqrt() as u128;
    let min = *times.iter().min().unwrap();
    let max = *times.iter().max().unwrap();

    (mean, stddev, min, max, times)
}

/// Generate synthetic OHLC data for testing
fn generate_ohlc_data(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64 % 10.0)).collect());
    let low = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64 % 10.0)).collect());
    let close = Array1::from_vec((0..n).map(|i| 99.0 + (i as f64 % 10.0)).collect());
    (high, low, close)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║         ATR GPU Performance Validation                     ║");
    println!("║         Methodology: Cold Start + Warmup + Warm Timing     ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize GPU
    println!("🔧 Initializing GPU...");
    let device = GpuDevice::new()?;
    println!("✅ GPU initialized (device {})", device.device_id);
    println!();

    // Test configuration
    let n = 100_000;
    let period = 14;
    let warmup_iterations = 5;
    let timing_iterations = 100;

    println!("📊 Test Configuration:");
    println!("   Candles:          {:>10}", n);
    println!("   Period:           {:>10}", period);
    println!("   Warmup runs:      {:>10}", warmup_iterations);
    println!("   Timing runs:      {:>10}", timing_iterations);
    println!();

    // Generate test data
    println!("📈 Generating synthetic OHLC data...");
    let (high, low, close) = generate_ohlc_data(n);
    println!("✅ Data generated");
    println!();

    // === PHASE 1: Cold Start Measurement ===
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Phase 1: Cold Start (includes CUDA JIT compilation)      ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("⏱️  Measuring cold start performance...");

    let cold_us = measure_cold_start(&device, &high, &low, &close, period);
    let cold_ms = cold_us as f64 / 1000.0;

    println!("✅ Cold start complete");
    println!();
    println!("   Result: {:.2} ms ({} μs)", cold_ms, cold_us);
    println!();

    if cold_us > 50_000 && cold_us < 300_000 {
        println!("   ✅ Within expected range (50-300ms)");
        println!("      This includes PTX compilation (~50-200ms)");
    } else if cold_us < 50_000 {
        println!("   ⚠️  Suspiciously fast - kernel may be cached from previous run");
    } else {
        println!("   ⚠️  Slower than expected - check GPU/driver health");
    }
    println!();

    // === PHASE 2: Warmup ===
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Phase 2: Warmup (prime CUDA cache)                       ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("🔄 Running {} warmup iterations...", warmup_iterations);

    let warmup_start = Instant::now();
    warmup(&device, &high, &low, &close, period, warmup_iterations);
    let warmup_total = warmup_start.elapsed().as_micros();

    println!("✅ Warmup complete");
    println!();
    println!("   Total time:   {} μs", warmup_total);
    println!(
        "   Per iteration: {} μs",
        warmup_total / warmup_iterations as u128
    );
    println!();

    // === PHASE 3: Warm Performance Measurement ===
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Phase 3: Warm Performance (actual runtime)               ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "⏱️  Measuring warm performance ({} iterations)...",
        timing_iterations
    );

    let (mean, stddev, min, max, times) =
        measure_warm_average(&device, &high, &low, &close, period, timing_iterations);

    println!("✅ Warm timing complete");
    println!();

    // === RESULTS ===
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Performance Results                                       ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Cold Start (includes compilation):                       ║");
    println!(
        "║    First run:      {:>10.2} ms                       ║",
        cold_ms
    );
    println!("╠════════════════════════════════════════════════════════════╣");
    println!(
        "║  Warm Performance (n={}):                             ║",
        timing_iterations
    );
    println!(
        "║    Mean:           {:>10} μs                       ║",
        mean
    );
    println!(
        "║    Std Dev:        {:>10} μs                       ║",
        stddev
    );
    println!(
        "║    Min:            {:>10} μs                       ║",
        min
    );
    println!(
        "║    Max:            {:>10} μs                       ║",
        max
    );
    println!(
        "║    Range:          {:>10} μs                       ║",
        max - min
    );
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Performance per Candle:                                   ║");
    println!(
        "║    Time/candle:    {:>10.3} μs                       ║",
        mean as f64 / n as f64
    );
    println!(
        "║    Throughput:     {:>10.0} candles/sec            ║",
        n as f64 / (mean as f64 / 1_000_000.0)
    );
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // === VALIDATION ===
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Validation: PR #8 Claim (v0.2.1 async optimization)      ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Expected Performance:                                     ║");
    println!("║    Target:    ~145 μs (Jules' claim)                       ║");
    println!("║    Range:     130-160 μs (±10% tolerance)                  ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Actual Performance:                                       ║");
    println!(
        "║    Mean:      {} μs                                     ║",
        mean
    );
    println!("╠════════════════════════════════════════════════════════════╣");

    let target = 145;
    let lower_bound = 130;
    let upper_bound = 160;

    if mean >= lower_bound && mean <= upper_bound {
        println!("║  Status:      ✅ VALIDATED                                 ║");
        println!("║               Performance matches expected range           ║");
        let deviation = ((mean as f64 - target as f64) / target as f64 * 100.0).abs();
        println!(
            "║               Deviation: {:.1}% from target                ║",
            deviation
        );
    } else if mean < lower_bound {
        println!("║  Status:      🚀 EXCEEDS EXPECTATIONS                      ║");
        let speedup = target as f64 / mean as f64;
        println!(
            "║               {:.2}x faster than target!                 ║",
            speedup
        );
    } else {
        println!("║  Status:      ⚠️  NEEDS INVESTIGATION                      ║");
        let slowdown = mean as f64 / target as f64;
        println!(
            "║               {:.2}x slower than expected                ║",
            slowdown
        );
        println!("║                                                            ║");
        println!("║  Possible causes:                                          ║");
        println!("║  - System under load (check GPU utilization)              ║");
        println!("║  - Thermal throttling (check GPU temperature)             ║");
        println!("║  - Different hardware (expected: RTX 3500 Ada)            ║");
        println!("║  - Debug build (ensure --release flag)                    ║");
    }
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // === STATISTICAL ANALYSIS ===
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Statistical Analysis                                      ║");
    println!("╠════════════════════════════════════════════════════════════╣");

    let coefficient_of_variation = (stddev as f64 / mean as f64) * 100.0;
    println!(
        "║  Coefficient of Variation: {:.2}%                        ║",
        coefficient_of_variation
    );

    if coefficient_of_variation < 5.0 {
        println!("║    ✅ Excellent stability (<5%)                            ║");
    } else if coefficient_of_variation < 10.0 {
        println!("║    ✅ Good stability (5-10%)                               ║");
    } else {
        println!("║    ⚠️  High variance (>10%) - check for interference      ║");
    }
    println!("╠════════════════════════════════════════════════════════════╣");

    // Calculate percentiles
    let mut sorted_times = times.clone();
    sorted_times.sort_unstable();
    let p50 = sorted_times[sorted_times.len() / 2];
    let p95 = sorted_times[sorted_times.len() * 95 / 100];
    let p99 = sorted_times[sorted_times.len() * 99 / 100];

    println!("║  Percentiles:                                              ║");
    println!("║    p50 (median): {:>10} μs                       ║", p50);
    println!("║    p95:          {:>10} μs                       ║", p95);
    println!("║    p99:          {:>10} μs                       ║", p99);
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // === SCALING ANALYSIS ===
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Scaling Analysis (multiple dataset sizes)                ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("  n       | Cold (ms) | Warm (μs) | μs/candle | Throughput");
    println!("----------|-----------|-----------|-----------|-------------");

    for test_n in [100, 1_000, 10_000, 100_000] {
        let (test_high, test_low, test_close) = generate_ohlc_data(test_n);

        // Skip cold start for non-first runs (already compiled)
        let test_cold_us = if test_n == 100 {
            measure_cold_start(&device, &test_high, &test_low, &test_close, period)
        } else {
            0 // Already compiled
        };

        // Quick warmup
        warmup(&device, &test_high, &test_low, &test_close, period, 3);

        // Measure warm performance (fewer iterations for smaller datasets)
        let iterations = if test_n < 10_000 { 50 } else { 100 };
        let (test_mean, _, _, _, _) = measure_warm_average(
            &device,
            &test_high,
            &test_low,
            &test_close,
            period,
            iterations,
        );

        let us_per_candle = test_mean as f64 / test_n as f64;
        let throughput = test_n as f64 / (test_mean as f64 / 1_000_000.0);

        println!(
            "{:>9} | {:>9.1} | {:>9} | {:>9.3} | {:>10.0} c/s",
            test_n,
            test_cold_us as f64 / 1000.0,
            test_mean,
            us_per_candle,
            throughput
        );
    }
    println!();

    // === RECOMMENDATIONS ===
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Key Takeaways                                             ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  1. Cold Start vs Warm Performance:                        ║");
    println!(
        "║     - Cold: {:.1}ms (includes compilation)             ║",
        cold_ms
    );
    println!(
        "║     - Warm: {} μs (actual runtime)                      ║",
        mean
    );
    println!(
        "║     - Ratio: {:.0}x slower without warmup               ║",
        cold_ms * 1000.0 / mean as f64
    );
    println!("║                                                            ║");
    println!("║  2. Why Warmup Matters:                                    ║");
    println!("║     - First run includes PTX → SASS compilation            ║");
    println!("║     - Subsequent runs use cached kernels                   ║");
    println!("║     - Always warmup before benchmarking!                   ║");
    println!("║                                                            ║");
    println!("║  3. Performance Validation:                                ║");

    if mean >= lower_bound && mean <= upper_bound {
        println!("║     - ✅ Jules' 145μs claim VALIDATED                      ║");
        println!("║     - Async optimization working as expected              ║");
    }

    println!("║                                                            ║");
    println!("║  4. Synchronization is Critical:                           ║");
    println!("║     - GPU kernel launches are async (~10μs)                ║");
    println!("║     - Must synchronize before stopping timer               ║");
    println!("║     - Without sync: only measures launch overhead          ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    println!("📚 For more details, see:");
    println!("   - docs/GPU_PERFORMANCE_TESTING_GUIDE.md");
    println!("   - src/gpu/atr.rs (implementation notes)");
    println!("   - docs/GPU_PROFILING_RESULTS.md (compilation overhead)");
    println!();

    Ok(())
}
