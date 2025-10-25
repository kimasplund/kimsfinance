//! Benchmark: Kernel Launch Overhead Comparison
//!
//! Measures launch overhead reduction from persistent kernels.
//!
//! # Methodology
//!
//! 1. **Traditional approach**: Launch kernel N times (one per task)
//! 2. **Persistent approach**: Launch kernel once, process N tasks in loop
//!
//! # Expected Results
//!
//! - Traditional: N × (~5-10μs) overhead
//! - Persistent: 1 × (~10μs) overhead
//! - Speedup: 50-90% for N > 10

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kimsfinance_core::gpu::{GpuDevice, roc_gpu};
use ndarray::Array1;

/// Generate test data for benchmarking
fn generate_test_data(n: usize) -> Array1<f64> {
    Array1::linspace(100.0, 200.0, n)
}

/// Benchmark traditional multi-launch approach
fn bench_traditional_launches(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required for this benchmark");
    
    let mut group = c.benchmark_group("launch_overhead");
    
    // Test with varying number of launches (simulating batch indicator calculations)
    for num_launches in [1, 5, 10, 20, 50, 100].iter() {
        let data = generate_test_data(1000); // Small dataset to emphasize launch overhead
        
        group.throughput(Throughput::Elements(*num_launches as u64));
        group.bench_with_input(
            BenchmarkId::new("traditional", num_launches),
            num_launches,
            |b, &n| {
                b.iter(|| {
                    // Launch kernel N times (simulating N indicator calculations)
                    for _ in 0..n {
                        let _result = roc_gpu(&device, &data, 14, None)
                            .expect("ROC GPU failed");
                        black_box(&_result);
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark persistent kernel approach (future implementation)
///
/// NOTE: This is a placeholder. Once persistent kernels are implemented,
/// we'll measure actual overhead reduction.
fn bench_persistent_kernel(c: &mut Criterion) {
    // TODO: Implement persistent kernel benchmark once infrastructure is ready
    // This will use PersistentKernelManager.execute_batch()
    
    let mut group = c.benchmark_group("persistent_kernel_placeholder");
    group.bench_function("placeholder", |b| {
        b.iter(|| {
            // Placeholder - shows what we're aiming for
            black_box(0);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_traditional_launches,
    bench_persistent_kernel,
);
criterion_main!(benches);
