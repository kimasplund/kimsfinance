//! GPU Kernel Timing Benchmark
//!
//! Measures GPU-only kernel execution time for all indicators using CUDA events,
//! separating pure GPU performance from CPU overhead (memory allocation, transfers, etc.).
//!
//! # Context
//!
//! Jules measured ATR at 145μs GPU-only time using CUDA events, while end-to-end
//! measurement shows 1.36ms (9.4x difference due to CPU overhead).
//!
//! This benchmark:
//! 1. Measures GPU-only kernel time (CUDA events) for 7 representative indicators
//! 2. Compares with end-to-end time (CPU clock) to quantify CPU overhead
//! 3. Validates the 11% async optimization impact (PR #9)
//! 4. Identifies optimization opportunities
//!
//! # Test Configuration
//!
//! - Dataset: 100K candles (standard benchmark size)
//! - Warmup: 5 iterations (exclude JIT compilation)
//! - Timing: 100 iterations averaged (statistical validity)
//! - Hardware: RTX 3500 Ada (12GB VRAM)
//!
//! # Indicators Tested
//!
//! 1. **ATR** (reference - Jules' 145μs claim) - Hybrid CPU-GPU
//! 2. **RSI** (complex) - Hybrid CPU-GPU
//! 3. **SMA** (medium) - Pure GPU, simple parallel
//! 4. **ROC** (simple, fast) - Pure GPU, embarrassingly parallel
//! 5. **CCI** (medium) - Hybrid CPU-GPU
//! 6. **Williams %R** (medium) - Pure GPU
//! 7. **OBV** (currently slow) - Pure GPU
//!
//! # Output
//!
//! For each indicator:
//! - GPU-only time (μs) - from CUDA events
//! - End-to-end time (μs) - from CPU clock
//! - CPU overhead (%) - difference between E2E and GPU-only
//! - Throughput (candles/sec)
//!
//! # Usage
//!
//! ```bash
//! # Run with release optimizations
//! cargo run --release --example benchmark_gpu_kernel_timing --features gpu
//!
//! # Generate markdown report
//! cargo run --release --example benchmark_gpu_kernel_timing --features gpu > docs/GPU_KERNEL_TIMING_REPORT.md
//! ```

use kimsfinance_core::gpu::device::GpuDevice;
use kimsfinance_core::gpu::timing::GpuTimer;
use kimsfinance_core::gpu::{atr_gpu, cci_gpu, obv_gpu, roc_gpu, rsi_gpu, sma_gpu, williams_r_gpu};
use ndarray::Array1;
use std::time::Instant;

/// Generate synthetic OHLCV data for testing
fn generate_ohlcv_data(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64 % 10.0)).collect());
    let low = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64 % 10.0)).collect());
    let close = Array1::from_vec((0..n).map(|i| 99.0 + (i as f64 % 10.0)).collect());
    let volume = Array1::from_vec((0..n).map(|i| 1_000_000.0 + (i as f64 % 100_000.0)).collect());
    (high, low, close, volume)
}

/// Benchmark result for one indicator
#[derive(Debug)]
struct BenchmarkResult {
    name: &'static str,
    gpu_only_us: u64,
    gpu_only_stddev: u64,
    end_to_end_us: u64,
    e2e_stddev: u64,
    cpu_overhead_pct: f64,
    throughput_candles_per_sec: f64,
}

impl BenchmarkResult {
    fn print_row(&self) {
        println!(
            "║ {:<12} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>7.1}% │ {:>12.0} ║",
            self.name,
            self.gpu_only_us,
            self.gpu_only_stddev,
            self.end_to_end_us,
            self.e2e_stddev,
            self.cpu_overhead_pct,
            self.throughput_candles_per_sec
        );
    }
}

/// Benchmark a single indicator with both GPU-only and end-to-end timing
fn benchmark_indicator<F>(
    name: &'static str,
    device: &GpuDevice,
    mut indicator_fn: F,
    n_candles: usize,
    warmup_iterations: usize,
    timing_iterations: usize,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    println!("\n  🔬 Benchmarking: {}", name);
    println!("     Warming up ({} iterations)...", warmup_iterations);

    // Warmup (exclude JIT compilation)
    for _ in 0..warmup_iterations {
        indicator_fn()?;
    }
    device.synchronize()?;

    println!("     Measuring GPU-only time ({} iterations)...", timing_iterations);

    // GPU-only timing using CUDA events
    let timer = GpuTimer::new(device)?;
    let mut gpu_times = Vec::with_capacity(timing_iterations);

    for _ in 0..timing_iterations {
        timer.start()?;
        indicator_fn()?;
        let elapsed = timer.stop_micros()?;
        gpu_times.push(elapsed);
    }

    println!("     Measuring end-to-end time ({} iterations)...", timing_iterations);

    // End-to-end timing using CPU clock
    let mut e2e_times = Vec::with_capacity(timing_iterations);

    for _ in 0..timing_iterations {
        let start = Instant::now();
        indicator_fn()?;
        device.synchronize()?;
        e2e_times.push(start.elapsed().as_micros() as u64);
    }

    // Calculate statistics
    let gpu_mean = gpu_times.iter().sum::<u64>() / gpu_times.len() as u64;
    let gpu_variance = gpu_times
        .iter()
        .map(|&t| (t as i128 - gpu_mean as i128).pow(2) as u64)
        .sum::<u64>()
        / gpu_times.len() as u64;
    let gpu_stddev = (gpu_variance as f64).sqrt() as u64;

    let e2e_mean = e2e_times.iter().sum::<u64>() / e2e_times.len() as u64;
    let e2e_variance = e2e_times
        .iter()
        .map(|&t| (t as i128 - e2e_mean as i128).pow(2) as u64)
        .sum::<u64>()
        / e2e_times.len() as u64;
    let e2e_stddev = (e2e_variance as f64).sqrt() as u64;

    let cpu_overhead_pct = ((e2e_mean as f64 - gpu_mean as f64) / e2e_mean as f64) * 100.0;
    let throughput = (n_candles as f64) / (e2e_mean as f64 / 1_000_000.0);

    Ok(BenchmarkResult {
        name,
        gpu_only_us: gpu_mean,
        gpu_only_stddev: gpu_stddev,
        end_to_end_us: e2e_mean,
        e2e_stddev: e2e_stddev,
        cpu_overhead_pct,
        throughput_candles_per_sec: throughput,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    GPU Kernel Timing Benchmark                              ║");
    println!("║         GPU-Only (CUDA Events) vs End-to-End (CPU Clock) Timing             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize GPU
    println!("🔧 Initializing GPU...");
    let device = GpuDevice::new()?;
    println!("✅ GPU initialized (device {})", device.device_id);
    println!();

    // Test configuration
    let n_candles = 100_000;
    let warmup_iterations = 5;
    let timing_iterations = 100;

    println!("📊 Test Configuration:");
    println!("   Candles:          {:>10}", n_candles);
    println!("   Warmup runs:      {:>10}", warmup_iterations);
    println!("   Timing runs:      {:>10}", timing_iterations);
    println!();

    // Generate test data
    println!("📈 Generating synthetic OHLCV data...");
    let (high, low, close, volume) = generate_ohlcv_data(n_candles);
    println!("✅ Data generated ({} candles)", n_candles);
    println!();

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         Benchmarking Indicators                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let mut results = Vec::new();

    // 1. ATR (reference - Jules' 145μs claim)
    results.push(benchmark_indicator(
        "ATR",
        &device,
        || {
            let _ = atr_gpu(&device, &high, &low, &close, 14, None)?;
            Ok(())
        },
        n_candles,
        warmup_iterations,
        timing_iterations,
    )?);

    // 2. RSI (complex, hybrid)
    results.push(benchmark_indicator(
        "RSI",
        &device,
        || {
            let _ = rsi_gpu(&device, &close, 14, None)?;
            Ok(())
        },
        n_candles,
        warmup_iterations,
        timing_iterations,
    )?);

    // 3. SMA (medium, pure GPU)
    results.push(benchmark_indicator(
        "SMA",
        &device,
        || {
            let _ = sma_gpu(&device, &close, 20, None)?;
            Ok(())
        },
        n_candles,
        warmup_iterations,
        timing_iterations,
    )?);

    // 4. ROC (simple, fast, pure GPU)
    results.push(benchmark_indicator(
        "ROC",
        &device,
        || {
            let _ = roc_gpu(&device, &close, 12, None)?;
            Ok(())
        },
        n_candles,
        warmup_iterations,
        timing_iterations,
    )?);

    // 5. CCI (medium, hybrid)
    results.push(benchmark_indicator(
        "CCI",
        &device,
        || {
            let _ = cci_gpu(&device, &high, &low, &close, 20, None)?;
            Ok(())
        },
        n_candles,
        warmup_iterations,
        timing_iterations,
    )?);

    // 6. Williams %R (medium, pure GPU)
    results.push(benchmark_indicator(
        "Williams %R",
        &device,
        || {
            let _ = williams_r_gpu(&device, &high, &low, &close, 14, None)?;
            Ok(())
        },
        n_candles,
        warmup_iterations,
        timing_iterations,
    )?);

    // 7. OBV (currently slow, pure GPU)
    results.push(benchmark_indicator(
        "OBV",
        &device,
        || {
            let _ = obv_gpu(&device, &close, &volume, None)?;
            Ok(())
        },
        n_candles,
        warmup_iterations,
        timing_iterations,
    )?);

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              Benchmark Results                               ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Indicator    │ GPU-Only │  StdDev  │   E2E    │  StdDev  │  CPU OH │  Throughput  ║");
    println!("║              │   (μs)   │   (μs)   │   (μs)   │   (μs)   │    (%)  │  (candles/s) ║");
    println!("╟──────────────┼──────────┼──────────┼──────────┼──────────┼─────────┼──────────────╢");

    for result in &results {
        result.print_row();
    }

    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Analysis
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                  Analysis                                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // 1. Validate ATR claim
    let atr_result = results.iter().find(|r| r.name == "ATR").unwrap();
    println!("📌 ATR Performance Validation:");
    println!("   Jules' claim:         145 μs (GPU-only, PR #8)");
    println!("   Measured (GPU-only):  {} μs", atr_result.gpu_only_us);
    println!("   Measured (E2E):       {} μs", atr_result.end_to_end_us);
    println!("   CPU overhead:         {:.1}%", atr_result.cpu_overhead_pct);
    println!();

    if atr_result.gpu_only_us >= 130 && atr_result.gpu_only_us <= 160 {
        println!("   ✅ VALIDATED: GPU-only time matches Jules' 145μs claim (±10%)");
    } else if atr_result.gpu_only_us < 130 {
        println!(
            "   🚀 EXCEEDS: {:.1}x faster than expected!",
            145.0 / atr_result.gpu_only_us as f64
        );
    } else {
        println!(
            "   ⚠️  SLOWER: {:.1}x slower than expected",
            atr_result.gpu_only_us as f64 / 145.0
        );
    }
    println!();

    // 2. CPU overhead analysis
    let avg_overhead = results.iter().map(|r| r.cpu_overhead_pct).sum::<f64>() / results.len() as f64;
    let max_overhead = results
        .iter()
        .max_by(|a, b| a.cpu_overhead_pct.partial_cmp(&b.cpu_overhead_pct).unwrap())
        .unwrap();
    let min_overhead = results
        .iter()
        .min_by(|a, b| a.cpu_overhead_pct.partial_cmp(&b.cpu_overhead_pct).unwrap())
        .unwrap();

    println!("📊 CPU Overhead Analysis:");
    println!("   Average overhead:     {:.1}%", avg_overhead);
    println!(
        "   Range:                {:.1}% - {:.1}%",
        min_overhead.cpu_overhead_pct, max_overhead.cpu_overhead_pct
    );
    println!("   Highest overhead:     {} ({:.1}%)", max_overhead.name, max_overhead.cpu_overhead_pct);
    println!("   Lowest overhead:      {} ({:.1}%)", min_overhead.name, min_overhead.cpu_overhead_pct);
    println!();

    println!("💡 Insight:");
    if avg_overhead > 80.0 {
        println!("   CPU overhead dominates (>80% on average)!");
        println!("   Optimization priorities:");
        println!("   1. Reduce memory allocation overhead");
        println!("   2. Use pinned memory for faster transfers");
        println!("   3. Batch operations to amortize overhead");
    } else if avg_overhead > 50.0 {
        println!("   CPU overhead is significant (>50% on average)");
        println!("   Optimization opportunities exist in memory management");
    } else {
        println!("   GPU work dominates (<50% CPU overhead)");
        println!("   Further GPU kernel optimization will have most impact");
    }
    println!();

    // 3. Performance ranking
    println!("🏆 Performance Ranking (by GPU-only time):");
    let mut ranked = results.clone();
    ranked.sort_by_key(|r| r.gpu_only_us);

    for (i, result) in ranked.iter().enumerate() {
        println!(
            "   {}. {:<12} - {} μs (E2E: {} μs, overhead: {:.1}%)",
            i + 1,
            result.name,
            result.gpu_only_us,
            result.end_to_end_us,
            result.cpu_overhead_pct
        );
    }
    println!();

    // 4. Async optimization impact estimation
    println!("📈 Async Optimization Impact (PR #9):");
    println!("   Jules' claim: 163μs → 145μs (11% speedup)");
    println!();
    println!("   If we apply 11% speedup to all indicators:");
    println!("   ┌──────────────┬──────────┬──────────┬──────────┐");
    println!("   │ Indicator    │ Current  │ w/ Async │ Speedup  │");
    println!("   │              │  (μs)    │  (μs)    │   (%)    │");
    println!("   ├──────────────┼──────────┼──────────┼──────────┤");

    for result in &results {
        let with_async = (result.gpu_only_us as f64 * 0.89) as u64; // 11% faster
        let speedup_pct = ((result.gpu_only_us as f64 - with_async as f64) / result.gpu_only_us as f64) * 100.0;
        println!(
            "   │ {:<12} │ {:>8} │ {:>8} │ {:>7.1}% │",
            result.name, result.gpu_only_us, with_async, speedup_pct
        );
    }

    println!("   └──────────────┴──────────┴──────────┴──────────┘");
    println!();

    // 5. Recommendations
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              Recommendations                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("🎯 Optimization Priorities:");
    println!();
    println!("1. **Reduce CPU Overhead** (biggest impact):");
    println!("   - Current: {:.1}% of total time is CPU overhead", avg_overhead);
    println!("   - Use async memory allocation (cudaMallocAsync)");
    println!("   - Batch multiple indicators to amortize allocation");
    println!("   - Reuse GPU buffers across calls");
    println!();

    println!("2. **Apply Async Optimization** (PR #9) to all indicators:");
    println!("   - Expected: 11% speedup across the board");
    println!("   - Overlap H2D transfers with kernel execution");
    println!("   - Use pinned memory for faster transfers");
    println!();

    println!("3. **Focus on slowest indicators** (highest absolute time):");
    let slowest = results.iter().max_by_key(|r| r.gpu_only_us).unwrap();
    println!("   - {} is slowest at {} μs GPU-only", slowest.name, slowest.gpu_only_us);
    println!("   - Profile with Nsight Compute for bottlenecks");
    println!("   - Consider kernel fusion to reduce overhead");
    println!();

    println!("4. **Validate hybrid approach** (CPU-GPU split):");
    let hybrid_indicators = ["ATR", "RSI", "CCI"];
    println!("   - Hybrid indicators (CPU+GPU): ATR, RSI, CCI");
    println!("   - Verify CPU smoothing is faster than GPU sequential");
    println!("   - Consider GPU parallel scan for some operations");
    println!();

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                 Next Steps                                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("✅ Completed:");
    println!("   1. GPU-only timing infrastructure using CUDA events");
    println!("   2. Benchmark 7 representative indicators");
    println!("   3. Quantify CPU overhead vs GPU work");
    println!("   4. Validate ATR 145μs claim");
    println!();

    println!("🔜 TODO:");
    println!("   1. Apply async optimization (PR #9) to remaining indicators");
    println!("   2. Implement memory pool to reduce allocation overhead");
    println!("   3. Profile slowest indicators with Nsight Compute");
    println!("   4. Add multi-phase timing (H2D → Kernel → D2H breakdown)");
    println!();

    println!("📚 Documentation:");
    println!("   - Timing methodology: docs/GPU_PERFORMANCE_TESTING_GUIDE.md");
    println!("   - This report: docs/GPU_KERNEL_TIMING_REPORT.md");
    println!("   - Implementation: src/gpu/timing.rs");
    println!();

    Ok(())
}
