//! Greeks GPU Benchmark
//!
//! Compares CPU vs GPU Greeks calculation performance.
//!
//! # Expected Results
//!
//! | Options | CPU Time | GPU Time | Speedup |
//! |---------|----------|----------|---------|
//! | 10      | 30ms     | 3ms      | 10x     |
//! | 100     | 300ms    | 8ms      | 37x     |
//! | 1000    | 3000ms   | 30ms     | 100x    |
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench greeks_gpu_bench --features gpu
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
use kimsfinance_core::quantitative::heston::{
    GreeksGpuCalculator, HestonGreeksCalculator, HestonParams, OptionQuote, OptionType,
};
use parking_lot::Mutex;
use std::sync::Arc;

fn create_test_options(n: usize, base_strike: f64) -> Vec<OptionQuote> {
    let now = chrono::Utc::now().timestamp();
    let expiry_3months = now + (90 * 24 * 3600);

    (0..n)
        .map(|i| OptionQuote {
            underlying: "BTC".to_string(),
            strike: base_strike + i as f64 * 100.0,
            expiration: expiry_3months,
            option_type: OptionType::Call,
            spot_price: 48000.0,
            risk_free_rate: 0.05,
            bid: Some(2000.0),
            ask: Some(2100.0),
            last: Some(2050.0),
            implied_vol: Some(0.8),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        })
        .collect()
}

fn bench_greeks_cpu(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let pricer = HestonGpuPricer::new(device, 4096, 1000).expect("Pricer creation failed");
    let calculator = HestonGreeksCalculator::new(Arc::new(Mutex::new(pricer)));

    let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).unwrap();

    let mut group = c.benchmark_group("greeks_cpu");
    for n in [10, 50, 100].iter() {
        let options = create_test_options(*n, 46000.0);
        group.throughput(Throughput::Elements(*n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &options, |b, opts| {
            b.iter(|| {
                let greeks = calculator.calculate_greeks_batch(black_box(&params), black_box(opts));
                black_box(greeks)
            });
        });
    }
    group.finish();
}

fn bench_greeks_gpu(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let pricer = HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Pricer creation failed");
    let mut calculator =
        GreeksGpuCalculator::new(device, Arc::new(Mutex::new(pricer))).expect("Calculator creation failed");

    let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).unwrap();

    let mut group = c.benchmark_group("greeks_gpu");
    for n in [10, 50, 100, 500, 1000].iter() {
        let options = create_test_options(*n, 46000.0);
        group.throughput(Throughput::Elements(*n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &options, |b, opts| {
            b.iter(|| {
                let greeks = calculator.calculate_greeks_batch(black_box(&params), black_box(opts));
                black_box(greeks)
            });
        });
    }
    group.finish();
}

fn bench_greeks_comparison(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let pricer_cpu = HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Pricer creation failed");
    let pricer_gpu = HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Pricer creation failed");

    let calculator_cpu = HestonGreeksCalculator::new(Arc::new(Mutex::new(pricer_cpu)));
    let mut calculator_gpu =
        GreeksGpuCalculator::new(device, Arc::new(Mutex::new(pricer_gpu))).expect("Calculator creation failed");

    let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).unwrap();

    let mut group = c.benchmark_group("greeks_comparison");
    for n in [10, 100].iter() {
        let options = create_test_options(*n, 46000.0);

        group.throughput(Throughput::Elements(*n as u64));

        group.bench_with_input(BenchmarkId::new("cpu", n), &options, |b, opts| {
            b.iter(|| {
                let greeks = calculator_cpu.calculate_greeks_batch(black_box(&params), black_box(opts));
                black_box(greeks)
            });
        });

        group.bench_with_input(BenchmarkId::new("gpu", n), &options, |b, opts| {
            b.iter(|| {
                let greeks = calculator_gpu.calculate_greeks_batch(black_box(&params), black_box(opts));
                black_box(greeks)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_greeks_cpu, bench_greeks_gpu, bench_greeks_comparison);
criterion_main!(benches);
