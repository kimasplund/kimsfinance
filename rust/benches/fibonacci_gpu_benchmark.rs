use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::{GpuDevice, fibonacci_gpu};

fn bench_fibonacci_gpu(c: &mut Criterion) {
    // Initialize GPU device
    let device = GpuDevice::new().expect("Failed to initialize GPU device");

    let mut group = c.benchmark_group("fibonacci_gpu");

    // Test different dataset sizes to measure scaling
    for size in [1_000, 10_000, 100_000, 500_000].iter() {
        // Generate test data with sine wave pattern
        let high: Vec<f64> = (0..*size)
            .map(|i| {
                let x = i as f64 * 0.01;
                110.0 + 10.0 * (x * 0.1).sin()
            })
            .collect();
        let low: Vec<f64> = (0..*size)
            .map(|i| {
                let x = i as f64 * 0.01;
                95.0 + 10.0 * (x * 0.1).sin()
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("lookback_20", size), size, |b, _| {
            b.iter(|| {
                fibonacci_gpu(
                    &device,
                    black_box(&high),
                    black_box(&low),
                    black_box(20),
                    None,
                )
                .unwrap()
            })
        });

        group.bench_with_input(BenchmarkId::new("lookback_50", size), size, |b, _| {
            b.iter(|| {
                fibonacci_gpu(
                    &device,
                    black_box(&high),
                    black_box(&low),
                    black_box(50),
                    None,
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

fn bench_fibonacci_gpu_throughput(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU device");

    // Large dataset for throughput measurement
    let n = 100_000;
    let high: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            110.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();
    let low: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            95.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();

    c.bench_function("fibonacci_gpu_throughput_100k", |b| {
        b.iter(|| {
            fibonacci_gpu(
                &device,
                black_box(&high),
                black_box(&low),
                black_box(20),
                None,
            )
            .unwrap()
        })
    });

    // Calculate and report throughput
    let iterations = 100;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        fibonacci_gpu(&device, &high, &low, 20, None).unwrap();
    }
    let elapsed = start.elapsed();
    let avg_time_us = elapsed.as_micros() / iterations;
    let candles_per_sec = (n * iterations) as f64 / elapsed.as_secs_f64();

    println!("\n=== Fibonacci GPU Throughput ===");
    println!("Dataset size: {} candles", n);
    println!("Average time: {} μs", avg_time_us);
    println!("Throughput: {:.2} candles/sec", candles_per_sec);
    println!(
        "Throughput: {:.2} M candles/sec",
        candles_per_sec / 1_000_000.0
    );
}

criterion_group!(benches, bench_fibonacci_gpu, bench_fibonacci_gpu_throughput);
criterion_main!(benches);
