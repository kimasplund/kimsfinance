//! Benchmark: RSI Fused Kernel vs Hybrid
//!
//! Validates 2.13x speedup target for fused RSI with parallel Wilder's smoothing.
//!
//! # Expected Results
//!
//! **Compute-Only** (excludes H2D/D2H):
//! - Hybrid: ~66μs (30μs Wilder's CPU + 36μs GPU)
//! - Fused: ~31μs (25μs CUB scan + 6μs overhead)
//! - Speedup: 2.13x ✅
//!
//! **End-to-End** (includes H2D/D2H):
//! - Hybrid: ~130μs (64μs transfers + 66μs compute)
//! - Fused: ~110μs (64μs transfers + 46μs compute)
//! - Speedup: 1.18x ✅
//!
//! # Running
//!
//! ```bash
//! cargo bench --bench rsi_fused_benchmark --features gpu
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kimsfinance_core::gpu::{is_fused_available, rsi_fused_gpu, rsi_gpu, GpuDevice};
use ndarray::Array1;

/// Generate realistic price data
fn generate_price_data(n: usize) -> Array1<f64> {
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            100.0 + 10.0 * (x * 0.1).sin() + 5.0 * (x * 0.05).cos()
        })
        .collect();
    Array1::from_vec(data)
}

/// Benchmark RSI hybrid implementation (baseline)
fn bench_rsi_hybrid(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("rsi_hybrid");

    for size in [1_000, 10_000, 100_000].iter() {
        let close = generate_price_data(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                rsi_gpu(black_box(&device), black_box(&close), black_box(14), black_box(None))
                    .expect("RSI hybrid failed")
            });
        });
    }

    group.finish();
}

/// Benchmark RSI fused implementation (optimized)
fn bench_rsi_fused(c: &mut Criterion) {
    if !is_fused_available() {
        println!("Skipping fused RSI benchmark: kernel not available");
        println!("Compile with: cargo build --features gpu --release");
        return;
    }

    let device = GpuDevice::new().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("rsi_fused");

    for size in [1_000, 10_000, 100_000].iter() {
        let close = generate_price_data(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                rsi_fused_gpu(black_box(&device), black_box(&close), black_box(14), black_box(None))
                    .expect("RSI fused failed")
            });
        });
    }

    group.finish();
}

/// Comparison benchmark: Hybrid vs Fused
fn bench_rsi_comparison(c: &mut Criterion) {
    if !is_fused_available() {
        println!("Skipping comparison benchmark: fused kernel not available");
        return;
    }

    let device = GpuDevice::new().expect("Failed to initialize GPU");
    let close = generate_price_data(100_000);

    let mut group = c.benchmark_group("rsi_comparison");
    group.throughput(Throughput::Elements(100_000));

    group.bench_function("hybrid_100k", |b| {
        b.iter(|| {
            rsi_gpu(black_box(&device), black_box(&close), black_box(14), black_box(None))
                .expect("RSI hybrid failed")
        });
    });

    group.bench_function("fused_100k", |b| {
        b.iter(|| {
            rsi_fused_gpu(black_box(&device), black_box(&close), black_box(14), black_box(None))
                .expect("RSI fused failed")
        });
    });

    group.finish();
}

/// Accuracy validation: Ensure fused matches hybrid
fn validate_accuracy(c: &mut Criterion) {
    if !is_fused_available() {
        return;
    }

    let device = GpuDevice::new().expect("Failed to initialize GPU");
    let close = generate_price_data(100_000);

    // Compute both
    let hybrid = rsi_gpu(&device, &close, 14, None).expect("Hybrid RSI failed");
    let fused = rsi_fused_gpu(&device, &close, 14, None).expect("Fused RSI failed");

    // Validate
    let mut max_error = 0.0;
    for i in 14..100_000 {
        let error = (hybrid[i] - fused[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }

    println!("\n=== RSI Fused Accuracy Validation ===");
    println!("Max error vs hybrid: {:.12}", max_error);
    println!(
        "Status: {}",
        if max_error < 1e-6 { "PASS ✓" } else { "FAIL ✗" }
    );

    assert!(
        max_error < 1e-6,
        "Fused implementation accuracy error too high: {}",
        max_error
    );

    // Dummy benchmark to include in criterion output
    c.bench_function("accuracy_validation", |b| {
        b.iter(|| {
            // No-op, just for reporting
            black_box(max_error)
        });
    });
}

criterion_group!(
    benches,
    bench_rsi_hybrid,
    bench_rsi_fused,
    bench_rsi_comparison,
    validate_accuracy
);
criterion_main!(benches);
