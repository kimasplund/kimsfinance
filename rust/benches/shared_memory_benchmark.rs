//! Benchmark: Shared Memory vs Global Memory for Rolling Window Indicators
//!
//! This benchmark compares performance of shared memory variants against
//! global memory (baseline) implementations for rolling window indicators.
//!
//! Expected Results:
//! - Small periods (<20): 0-3% improvement (maybe regression)
//! - Medium periods (20-50): 0-5% improvement
//! - Large periods (>50): 0-3% improvement
//!
//! Rationale: Shared memory provides minimal benefit because adjacent threads
//! have minimal data overlap (1 element out of `period`). Global memory access
//! is already coalesced and L1/L2 cache handles this pattern efficiently.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::{GpuDevice, sma_gpu, sma_gpu_shared};
use ndarray::Array1;

fn bench_sma_shared_vs_global(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("SMA_SharedVsGlobal");
    group.sample_size(50); // Reduce sample size for faster benchmarking

    let sizes = [10_000, 100_000];
    let periods = [10, 20, 50, 100, 200];

    for &n in sizes.iter() {
        for &period in periods.iter() {
            let close = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.001).collect());

            // Benchmark global memory
            group.bench_with_input(
                BenchmarkId::new("Global", format!("n{}_p{}", n, period)),
                &(&device, &close, period),
                |b, (dev, data, per)| {
                    b.iter(|| {
                        sma_gpu(black_box(dev), black_box(data), black_box(*per), None).unwrap()
                    });
                },
            );

            // Benchmark shared memory
            group.bench_with_input(
                BenchmarkId::new("Shared", format!("n{}_p{}", n, period)),
                &(&device, &close, period),
                |b, (dev, data, per)| {
                    b.iter(|| {
                        sma_gpu_shared(black_box(dev), black_box(data), black_box(*per), None)
                            .unwrap()
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_sma_shared_vs_global);
criterion_main!(benches);
