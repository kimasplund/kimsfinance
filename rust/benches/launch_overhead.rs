//! Benchmark: Kernel Launch Overhead Comparison
//!
//! Measures launch overhead reduction from persistent kernels vs traditional multi-launch.
//!
//! # Methodology
//!
//! 1. **Traditional approach**: Launch kernel N times (one per task)
//!    - Each task: Launch → Execute → Sync → Result
//!    - Total overhead: N × (~5-10μs) per launch
//!    - Use case: Multiple independent indicator calculations
//!
//! 2. **Persistent approach**: Launch kernel once, process N tasks in loop
//!    - Single launch: Launch → [Task 1 → Task 2 → ... → Task N] → Result
//!    - Total overhead: 1 × (~10μs) + task switching cost
//!    - Use case: Batch indicator processing (e.g., RSI with multiple periods)
//!
//! # Expected Results
//!
//! - **Traditional (10 tasks)**: 10 × 10μs = ~100μs overhead
//! - **Persistent (10 tasks)**: 1 × 10μs = ~10-20μs overhead
//! - **Target Speedup**: 80-90% overhead reduction for N ≥ 10
//! - **Overall Speedup**: 2-4x for small datasets where overhead dominates
//!
//! # Statistical Validation
//!
//! - Sample size: n = 100 iterations (Criterion default)
//! - Confidence intervals: 95% (Criterion default)
//! - Variance analysis: Coefficient of variation should be <10% for stable measurements
//! - Significance: p < 0.05 for speedup claims (t-test)
//!
//! # Dataset Sizes
//!
//! We test multiple dataset sizes to identify where persistent kernels provide value:
//! - **1,000 candles**: Launch overhead dominates (persistent wins big)
//! - **10,000 candles**: Mixed (overhead + compute)
//! - **100,000 candles**: Compute dominates (traditional may be faster)

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::{
    GpuDevice, PersistentKernelManager, TaskBatch, execute_batch, roc_gpu,
};
use ndarray::Array1;

/// Generate test data for benchmarking
fn generate_test_data(n: usize) -> Vec<f64> {
    (0..n).map(|i| 100.0 + (i as f64) * 0.1).collect()
}

/// Benchmark: Traditional Multi-Launch Approach
///
/// Launches ROC kernel N times, each with its own data and period.
/// This simulates calculating multiple indicators in sequence.
fn bench_traditional_launches(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required for this benchmark");

    let mut group = c.benchmark_group("traditional_launches");
    group.sample_size(100); // Sufficient for statistical significance

    // Test with varying number of tasks (1, 5, 10, 20, 50, 100)
    for num_tasks in [1, 5, 10, 20, 50, 100].iter() {
        // Use small dataset to emphasize launch overhead
        let data_size = 1000;

        group.throughput(Throughput::Elements(*num_tasks as u64));
        group.bench_with_input(BenchmarkId::new("tasks", num_tasks), num_tasks, |b, &n| {
            // Pre-generate all test data
            let datasets: Vec<_> = (0..n)
                .map(|_| Array1::from_vec(generate_test_data(data_size)))
                .collect();
            let periods: Vec<_> = (0..n).map(|i| 14 + i % 10).collect(); // Vary periods

            b.iter(|| {
                // Launch kernel N times (one per task)
                for i in 0..n {
                    let result =
                        roc_gpu(&device, &datasets[i], periods[i], None).expect("ROC GPU failed");
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark: Persistent Kernel Approach
///
/// Launches kernel once and processes all tasks in a loop using cooperative groups.
/// This is the key innovation: single launch overhead for N tasks.
fn bench_persistent_kernel(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required for this benchmark");

    let mut group = c.benchmark_group("persistent_kernel");
    group.sample_size(100);

    for num_tasks in [1, 5, 10, 20, 50, 100].iter() {
        let data_size = 1000;

        group.throughput(Throughput::Elements(*num_tasks as u64));
        group.bench_with_input(BenchmarkId::new("tasks", num_tasks), num_tasks, |b, &n| {
            b.iter(|| {
                // Create batch with all tasks
                let mut batch = TaskBatch::new();
                for i in 0..n {
                    let data = generate_test_data(data_size);
                    let period = 14 + i % 10; // Vary periods like traditional
                    batch.add_task(data, period);
                }

                // Single kernel launch for all tasks
                let results = execute_batch(&device, &batch).expect("Batch execution failed");
                black_box(results);
            });
        });
    }

    group.finish();
}

/// Benchmark: Direct Comparison at Key Operating Point (10 tasks)
///
/// This is the most important benchmark: 10 tasks is typical for multi-indicator backtests.
fn bench_overhead_reduction_10_tasks(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");

    let data_size = 1000;
    let num_tasks = 10;

    c.bench_function("overhead_traditional_10", |b| {
        let datasets: Vec<_> = (0..num_tasks)
            .map(|_| Array1::from_vec(generate_test_data(data_size)))
            .collect();
        let periods: Vec<_> = (0..num_tasks).map(|i| 14 + i % 10).collect();

        b.iter(|| {
            for i in 0..num_tasks {
                let result =
                    roc_gpu(&device, &datasets[i], periods[i], None).expect("ROC GPU failed");
                black_box(result);
            }
        });
    });

    c.bench_function("overhead_persistent_10", |b| {
        b.iter(|| {
            let mut batch = TaskBatch::new();
            for i in 0..num_tasks {
                let data = generate_test_data(data_size);
                let period = 14 + i % 10;
                batch.add_task(data, period);
            }
            // Execute batch with persistent kernel
            let results = execute_batch(&device, &batch).expect("Batch execution failed");
            black_box(results);
        });
    });
}

/// Benchmark: Scaling Analysis (1K, 10K, 100K candles)
///
/// Tests how speedup changes with dataset size.
/// Expected: Persistent wins for small datasets, may lose for large datasets.
fn bench_dataset_size_scaling(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");

    let mut group = c.benchmark_group("dataset_size_scaling");
    group.sample_size(50); // Reduce for large datasets

    for data_size in [1_000, 10_000, 100_000].iter() {
        let num_tasks = 10; // Fixed at 10 tasks

        group.throughput(Throughput::Elements(*data_size as u64 * num_tasks as u64));

        // Traditional approach
        group.bench_with_input(
            BenchmarkId::new("traditional", data_size),
            data_size,
            |b, &n| {
                let datasets: Vec<_> = (0..num_tasks)
                    .map(|_| Array1::from_vec(generate_test_data(n)))
                    .collect();
                let periods: Vec<_> = (0..num_tasks).map(|i| 14 + i % 10).collect();

                b.iter(|| {
                    for i in 0..num_tasks {
                        let result = roc_gpu(&device, &datasets[i], periods[i], None)
                            .expect("ROC GPU failed");
                        black_box(result);
                    }
                });
            },
        );

        // Persistent approach
        group.bench_with_input(
            BenchmarkId::new("persistent", data_size),
            data_size,
            |b, &n| {
                b.iter(|| {
                    let mut batch = TaskBatch::new();
                    for i in 0..num_tasks {
                        let data = generate_test_data(n);
                        let period = 14 + i % 10;
                        batch.add_task(data, period);
                    }
                    // Execute batch
                    let results = execute_batch(&device, &batch).expect("Batch execution failed");
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Throughput Measurement (tasks/second)
///
/// Measures maximum throughput achievable with each approach.
fn bench_throughput(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");

    let mut group = c.benchmark_group("throughput");
    group.sample_size(50);

    let data_size = 1000;
    let num_tasks = 100; // Large batch for throughput test

    group.throughput(Throughput::Elements(num_tasks as u64));

    group.bench_function("traditional_throughput", |b| {
        let datasets: Vec<_> = (0..num_tasks)
            .map(|_| Array1::from_vec(generate_test_data(data_size)))
            .collect();
        let periods: Vec<_> = (0..num_tasks).map(|i| 14 + i % 10).collect();

        b.iter(|| {
            for i in 0..num_tasks {
                let result =
                    roc_gpu(&device, &datasets[i], periods[i], None).expect("ROC GPU failed");
                black_box(result);
            }
        });
    });

    group.bench_function("persistent_throughput", |b| {
        b.iter(|| {
            let mut batch = TaskBatch::new();
            for i in 0..num_tasks {
                let data = generate_test_data(data_size);
                let period = 14 + i % 10;
                batch.add_task(data, period);
            }
            // Execute batch
            let results = execute_batch(&device, &batch).expect("Batch execution failed");
            black_box(results);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_traditional_launches,
    bench_persistent_kernel,
    bench_overhead_reduction_10_tasks,
    bench_dataset_size_scaling,
    bench_throughput,
);
criterion_main!(benches);
