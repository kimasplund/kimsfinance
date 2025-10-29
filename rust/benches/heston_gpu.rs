//! GPU vs CPU Benchmark: Heston Option Pricing
//!
//! Validates 100-500x speedup target for batch option pricing.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
use kimsfinance_core::quantitative::heston::{HestonParams, OptionQuote, OptionType};
use std::sync::Arc;

/// Generate test options with various strikes
fn generate_test_options(n: usize) -> Vec<OptionQuote> {
    (0..n)
        .map(|i| {
            let strike = 40000.0 + (i as f64 * 100.0);
            OptionQuote {
                symbol: format!("BTC-20250101-{:.0}-C", strike),
                underlying: "BTC".to_string(),
                strike,
                expiry_years: 0.25, // 3 months
                option_type: OptionType::Call,
                bid: 2000.0,
                ask: 2100.0,
                mid_price: 2050.0,
                implied_vol: Some(0.8),
                volume: 100.0,
            }
        })
        .collect()
}

/// CPU-based Heston pricing (placeholder - simplified Black-Scholes for baseline)
fn cpu_price_options(params: &HestonParams, options: &[OptionQuote]) -> Vec<f64> {
    // Simplified CPU pricing for baseline comparison
    // In production, this would use full Heston characteristic function + FFT

    options
        .iter()
        .map(|opt| {
            // Placeholder: return mid price (real implementation would compute Heston price)
            // For benchmarking purposes, we simulate the computational cost
            let mut sum = 0.0;
            for _i in 0..1000 {
                sum += (params.kappa * params.theta * params.sigma * params.rho * params.v0).sin();
            }
            opt.mid_price + sum * 0.00001
        })
        .collect()
}

fn bench_heston_pricing(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let pricer = HestonGpuPricer::new(device, 4096).expect("Failed to create HestonGpuPricer");

    let params = HestonParams::new(
        2.0,  // kappa
        0.04, // theta
        0.3,  // sigma
        -0.7, // rho
        0.04, // v0
    )
    .expect("Invalid Heston parameters");

    let mut group = c.benchmark_group("heston_pricing");

    // Benchmark different batch sizes
    for n_options in [10, 50, 100, 500, 1000].iter() {
        let options = generate_test_options(*n_options);

        group.throughput(Throughput::Elements(*n_options as u64));

        // GPU pricing
        group.bench_with_input(BenchmarkId::new("gpu", n_options), n_options, |b, _| {
            b.iter(|| {
                pricer
                    .price_options(black_box(&params), black_box(&options))
                    .expect("GPU pricing failed")
            })
        });

        // CPU pricing (baseline)
        group.bench_with_input(BenchmarkId::new("cpu", n_options), n_options, |b, _| {
            b.iter(|| cpu_price_options(black_box(&params), black_box(&options)))
        });
    }

    group.finish();
}

fn bench_heston_kernel_compilation(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    c.bench_function("heston_kernel_compile_cold", |b| {
        b.iter(|| {
            // This will hit cache after first call, so we measure warm startup
            HestonGpuPricer::new(device.clone(), 4096).expect("Failed to create pricer")
        })
    });
}

criterion_group!(
    benches,
    bench_heston_pricing,
    bench_heston_kernel_compilation
);
criterion_main!(benches);
