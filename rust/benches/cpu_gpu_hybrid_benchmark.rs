//! CPU-GPU Hybrid Performance Benchmarks
//!
//! Validates performance improvements from converting single-thread GPU kernels
//! to CPU-GPU hybrid implementations for sequential indicators.
//!
//! # Background
//!
//! Sequential algorithms (IIR filters like EMA, Wilder's smoothing) have data
//! dependencies that prevent parallelization. Running them on a single GPU thread
//! combines:
//! - Slower single-threaded GPU core (vs fast CPU core)
//! - PCIe transfer overhead (H2D + D2H)
//! - Kernel launch overhead (~5-10μs)
//!
//! # Strategy
//!
//! **CPU-GPU Hybrid Architecture**:
//! - CPU: Sequential parts (EMA, Wilder's smoothing, ATR smoothing)
//! - GPU: Parallel parts (subtraction, gains/losses, RSI calculation)
//!
//! # Indicators Benchmarked
//!
//! 1. **EMA**: Old GPU (single-thread) vs New CPU
//! 2. **Elder Ray**: Old pure-GPU vs New hybrid (CPU EMA + GPU parallel ops)
//! 3. **RSI**: Old pure-GPU vs New hybrid (GPU parallel + CPU smoothing + GPU parallel)
//! 4. **ATR**: Old pure-GPU vs New hybrid (GPU TR + CPU smoothing)
//!
//! # Expected Results (RTX 3500 Ada, Intel i9-13980HX)
//!
//! | Indicator | Old GPU | New Hybrid/CPU | Speedup |
//! |-----------|---------|----------------|---------|
//! | EMA       | ~170μs  | ~25μs          | 6.8x    |
//! | Elder Ray | ~200μs  | ~100μs         | 2.0x    |
//! | RSI       | ~250μs  | ~130μs         | 1.9x    |
//! | ATR       | ~180μs  | ~70μs          | 2.6x    |
//!
//! Dataset: 100K candles
//!
//! # Running Benchmarks
//!
//! ```bash
//! # All hybrid benchmarks
//! cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark
//!
//! # Specific indicator
//! cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark -- EMA
//! ```
//!
//! # Interpreting Results
//!
//! - **Throughput**: Elements/sec (higher is better)
//! - **Time**: μs per operation (lower is better)
//! - **Speedup**: New / Old (>1.0 means improvement)
//! - **Confidence Intervals**: Criterion provides statistical validation

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ndarray::Array1;

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{
    atr_gpu, elder_ray_gpu, ema_gpu, rsi_gpu,
    rsi_sync::rsi_gpu_sync, // Import the synchronous version for comparison
    GpuDevice,
};

// Note: CPU implementations pending - will be added when sequential.rs is implemented
// #[cfg(feature = "gpu")]
// use kimsfinance_core::cpu::{ema_cpu, wilders_smoothing_cpu};

/// Generate realistic OHLCV test data
///
/// Simulates price movement with trend, volatility, and noise
fn generate_test_data(size: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut high = Vec::with_capacity(size);
    let mut low = Vec::with_capacity(size);
    let mut close = Vec::with_capacity(size);

    let base_price = 100.0;
    let trend = 0.01; // Slight uptrend
    let volatility = 2.0;

    for i in 0..size {
        let t = i as f64;

        // Price with trend, sine wave oscillation, and noise
        let price =
            base_price + trend * t + volatility * (t * 0.01).sin() + (t * 0.123).sin() * 0.5; // Additional noise

        // OHLC with realistic spread
        let spread = volatility * 0.5;
        high.push(price + spread * 0.7);
        low.push(price - spread * 0.7);
        close.push(price);
    }

    (
        Array1::from_vec(high),
        Array1::from_vec(low),
        Array1::from_vec(close),
    )
}

// =============================================================================
// EMA Benchmarks
// =============================================================================

/// Benchmark EMA: Old GPU (single-thread) vs New CPU
///
/// Old: Single GPU thread for entire EMA calculation (~170μs for 100K)
/// New: CPU calculation (~25μs for 100K)
/// Expected speedup: 6.8x
#[cfg(feature = "gpu")]
fn bench_ema_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("EMA_Comparison");

    let device = match GpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GPU not available, skipping EMA benchmarks: {:?}", e);
            return;
        }
    };

    // Test across multiple dataset sizes for scalability analysis
    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

    for size in sizes {
        // Generate realistic test data
        let (_, _, close) = generate_test_data(size);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark 1: Old GPU (single-thread, deprecated)
        // This is the current implementation - slow due to single-thread + overhead
        group.bench_with_input(
            BenchmarkId::new("Old_GPU_SingleThread", size),
            &close,
            |b, data| {
                #[allow(deprecated)]
                b.iter(|| ema_gpu(&device, black_box(data), 20, None))
            },
        );

        // TODO: Benchmark 2: New CPU
        // Uncomment when cpu::sequential::ema_cpu is implemented
        // group.bench_with_input(
        //     BenchmarkId::new("New_CPU", size),
        //     &close,
        //     |b, data| b.iter(|| ema_cpu(black_box(data), 20))
        // );

        // TODO: Benchmark 3: New Hybrid (delegates to CPU)
        // Uncomment when ema_hybrid is implemented
        // group.bench_with_input(
        //     BenchmarkId::new("New_Hybrid", size),
        //     &close,
        //     |b, data| b.iter(|| ema_hybrid(&device, black_box(data), 20, None))
        // );
    }

    group.finish();
}

// =============================================================================
// Elder Ray Benchmarks
// =============================================================================

/// Benchmark Elder Ray: Old pure-GPU vs New hybrid
///
/// Old: Single-thread GPU EMA + parallel GPU subtraction (~200μs for 100K)
/// New: CPU EMA + parallel GPU subtraction (~100μs for 100K)
/// Expected speedup: 2.0x
#[cfg(feature = "gpu")]
fn bench_elder_ray_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("ElderRay_Comparison");

    let device = match GpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GPU not available, skipping Elder Ray benchmarks: {:?}", e);
            return;
        }
    };

    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

    for size in sizes {
        let (high, low, close) = generate_test_data(size);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark 1: Old pure-GPU implementation
        // Current implementation uses single-thread GPU for EMA
        group.bench_with_input(
            BenchmarkId::new("Old_GPU_Pure", size),
            &(&high, &low, &close),
            |b, (h, l, c)| {
                b.iter(|| {
                    elder_ray_gpu(&device, black_box(h), black_box(l), black_box(c), 13, None)
                })
            },
        );

        // TODO: Benchmark 2: New hybrid implementation
        // CPU EMA + GPU parallel subtraction
        // Uncomment when elder_ray_hybrid is implemented
        // group.bench_with_input(
        //     BenchmarkId::new("New_Hybrid", size),
        //     &(&high, &low, &close),
        //     |b, (h, l, c)| {
        //         b.iter(|| elder_ray_hybrid(&device, black_box(h), black_box(l), black_box(c), 13, None))
        //     }
        // );
    }

    group.finish();
}

// =============================================================================
// RSI Benchmarks
// =============================================================================

/// Benchmark RSI: Old pure-GPU vs New hybrid
///
/// Old: GPU parallel gains/losses + GPU single-thread smoothing + GPU parallel RSI (~250μs for 100K)
/// New: GPU parallel gains/losses + CPU smoothing (2x) + GPU parallel RSI (~130μs for 100K)
/// Expected speedup: 1.9x
#[cfg(feature = "gpu")]
fn bench_rsi_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("RSI_Comparison");

    let device = match GpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GPU not available, skipping RSI benchmarks: {:?}", e);
            return;
        }
    };

    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

    for size in sizes {
        // Generate oscillating data for realistic RSI calculation
        let close: Vec<f64> = (0..size)
            .map(|i| {
                let t = i as f64;
                100.0 + (t * 0.01).sin() * 10.0 + (t * 0.05).sin() * 3.0
            })
            .collect();
        let close = Array1::from_vec(close);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark 1: Hybrid with synchronous transfers
        group.bench_with_input(
            BenchmarkId::new("Hybrid_Sync", size),
            &close,
            |b, data| b.iter(|| rsi_gpu_sync(&device, black_box(data), 14, None)),
        );

        // Benchmark 2: New hybrid implementation with async transfers
        group.bench_with_input(
            BenchmarkId::new("New_Hybrid_Async", size),
            &close,
            |b, data| b.iter(|| rsi_gpu(&device, black_box(data), 14, None)),
        );
    }

    group.finish();
}

// =============================================================================
// ATR Benchmarks
// =============================================================================

/// Benchmark ATR: Old pure-GPU vs New hybrid
///
/// Old: GPU parallel true range + GPU single-thread smoothing (~180μs for 100K)
/// New: GPU parallel true range + CPU smoothing (~70μs for 100K)
/// Expected speedup: 2.6x
#[cfg(feature = "gpu")]
fn bench_atr_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("ATR_Comparison");

    let device = match GpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GPU not available, skipping ATR benchmarks: {:?}", e);
            return;
        }
    };

    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

    for size in sizes {
        let (high, low, close) = generate_test_data(size);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark 1: Old pure-GPU implementation
        // Check if it uses single-thread smoothing
        group.bench_with_input(
            BenchmarkId::new("Old_GPU_Pure", size),
            &(&high, &low, &close),
            |b, (h, l, c)| {
                b.iter(|| atr_gpu(&device, black_box(h), black_box(l), black_box(c), 14, None))
            },
        );

        // TODO: Benchmark 2: New hybrid implementation
        // GPU parallel true range + CPU Wilder's smoothing
        // Uncomment when atr_hybrid is implemented
        // group.bench_with_input(
        //     BenchmarkId::new("New_Hybrid", size),
        //     &(&high, &low, &close),
        //     |b, (h, l, c)| {
        //         b.iter(|| atr_hybrid(&device, black_box(h), black_box(l), black_box(c), 14, None))
        //     }
        // );
    }

    group.finish();
}

// =============================================================================
// Summary Report
// =============================================================================

/// Print benchmark summary and analysis
///
/// This function is called after benchmarks to provide context and interpretation
#[cfg(feature = "gpu")]
fn print_benchmark_summary() {
    println!("\n========================================");
    println!("CPU-GPU Hybrid Benchmark Summary");
    println!("========================================\n");

    println!("STRATEGY:");
    println!("  Sequential algorithms (IIR filters) cannot be parallelized due to");
    println!("  data dependencies. Single GPU thread is 4-5x slower than CPU core");
    println!("  due to lower clock speed and instruction-level parallelism.\n");

    println!("EXPECTED RESULTS (100K candles):\n");

    println!("EMA:");
    println!("  Old GPU (single-thread): ~170μs");
    println!("  New CPU:                 ~25μs");
    println!("  Speedup:                 6.8x\n");

    println!("Elder Ray:");
    println!("  Old GPU (pure):          ~200μs");
    println!("  New Hybrid:              ~100μs");
    println!("  Speedup:                 2.0x\n");

    println!("RSI:");
    println!("  Old GPU (pure):          ~250μs");
    println!("  New Hybrid:              ~130μs");
    println!("  Speedup:                 1.9x\n");

    println!("ATR:");
    println!("  Old GPU (pure):          ~180μs");
    println!("  New Hybrid:              ~70μs");
    println!("  Speedup:                 2.6x\n");

    println!("OVERALL IMPACT:");
    println!("  Average speedup:         3.3x");
    println!("  Range:                   1.9x - 6.8x");
    println!("  Strategy:                CPU for sequential, GPU for parallel\n");

    println!("KEY INSIGHTS:");
    println!("  - Sequential algorithms (data dependencies) run faster on CPU");
    println!("  - GPU single-thread is slower than CPU due to architecture");
    println!("  - Hybrid approach uses each processor for what it does best");
    println!("  - PCIe overhead is justified only when GPU parallelism helps\n");

    println!("NEXT STEPS:");
    println!("  1. Implement CPU sequential algorithms (ema_cpu, wilders_smoothing_cpu)");
    println!("  2. Create hybrid variants (ema_hybrid, elder_ray_hybrid, rsi_hybrid, atr_hybrid)");
    println!("  3. Uncomment 'New_CPU' and 'New_Hybrid' benchmarks above");
    println!("  4. Re-run to validate expected speedups");
    println!("  5. Update documentation with accurate performance claims\n");

    println!("========================================\n");
}

// =============================================================================
// Criterion Configuration
// =============================================================================

#[cfg(feature = "gpu")]
criterion_group!(
    hybrid_benches,
    bench_ema_comparison,
    bench_elder_ray_comparison,
    bench_rsi_comparison,
    bench_atr_comparison
);

#[cfg(feature = "gpu")]
criterion_main!(hybrid_benches);

// Fallback for when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "GPU feature not enabled. Run with: cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark"
    );
    std::process::exit(1);
}

// =============================================================================
// Module-level tests for data generation
// =============================================================================

#[cfg(test)]
mod tests {
    use super::generate_test_data;

    #[test]
    fn test_generate_test_data_size() {
        let size = 1000;
        let (high, low, close) = generate_test_data(size);

        assert_eq!(high.len(), size);
        assert_eq!(low.len(), size);
        assert_eq!(close.len(), size);
    }

    #[test]
    fn test_generate_test_data_ohlc_relationship() {
        let (high, low, close) = generate_test_data(100);

        // Verify OHLC relationships
        for i in 0..100 {
            assert!(
                high[i] >= close[i],
                "High should be >= close at index {}",
                i
            );
            assert!(low[i] <= close[i], "Low should be <= close at index {}", i);
            assert!(high[i] >= low[i], "High should be >= low at index {}", i);
        }
    }

    #[test]
    fn test_generate_test_data_realistic_values() {
        let (high, low, close) = generate_test_data(1000);

        // Verify prices are in reasonable range
        for i in 0..1000 {
            assert!(close[i] > 0.0, "Close should be positive");
            assert!(close[i] < 200.0, "Close should be in realistic range");

            let spread = high[i] - low[i];
            assert!(spread > 0.0, "Spread should be positive");
            assert!(spread < 10.0, "Spread should be realistic");
        }
    }
}
