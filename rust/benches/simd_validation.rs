use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use kimsfinance_core::backtest::metrics::calculate_sharpe_ratio;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Scalar return calculation (reference baseline)
fn calculate_returns_scalar_baseline(equity_curve: &[f64]) -> Vec<f64> {
    let mut returns = Vec::with_capacity(equity_curve.len() - 1);
    for i in 1..equity_curve.len() {
        if equity_curve[i - 1] != 0.0 {
            let ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1];
            returns.push(ret);
        }
    }
    returns
}

/// AVX2-optimized return calculation (FIXED - Agent 2)
///
/// This implements the performance fix: Direct SIMD store without scalar loop bottleneck.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn calculate_returns_avx2_baseline(equity_curve: &[f64]) -> Vec<f64> {
    let n = equity_curve.len();
    let mut returns: Vec<f64> = Vec::with_capacity(n - 1);
    let num_chunks = (n - 1) / 4;
    let remainder_start = 1 + num_chunks * 4;

    // Reserve space to prevent reallocations
    returns.reserve(n - 1);

    // Process 4 elements at a time with AVX2
    unsafe {
        for chunk in 0..num_chunks {
            let i = 1 + chunk * 4;

            // Load current and previous equity values
            let curr = _mm256_loadu_pd(equity_curve.as_ptr().add(i));
            let prev = _mm256_loadu_pd(equity_curve.as_ptr().add(i - 1));

            // Calculate (curr - prev) / prev
            let diff = _mm256_sub_pd(curr, prev);
            let ret = _mm256_div_pd(diff, prev);

            // CRITICAL FIX: Store directly without scalar loop!
            let old_len = returns.len();
            returns.set_len(old_len + 4);
            _mm256_storeu_pd(returns.as_mut_ptr().add(old_len), ret);
        }
    }

    // Process remaining elements with scalar code
    for i in remainder_start..n {
        if equity_curve[i - 1] != 0.0 {
            let ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1];
            returns.push(ret);
        }
    }

    // Filter out NaN/Inf in post-processing
    returns.retain(|&r| r.is_finite());

    returns
}

fn bench_returns_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("returns_calculation");

    for size in [100, 1_000, 10_000] {
        let equity: Vec<f64> = (0..size).map(|i| 1000.0 + i as f64 * 0.1).collect();

        group.bench_with_input(BenchmarkId::new("scalar", size), &equity, |b, eq| {
            b.iter(|| calculate_returns_scalar_baseline(black_box(eq)))
        });

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            group.bench_with_input(BenchmarkId::new("avx2", size), &equity, |b, eq| {
                b.iter(|| unsafe { calculate_returns_avx2_baseline(black_box(eq)) })
            });
        }

        group.bench_with_input(BenchmarkId::new("full_sharpe", size), &equity, |b, eq| {
            b.iter(|| calculate_sharpe_ratio(black_box(eq)))
        });
    }

    group.finish();
}

fn bench_variance_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("variance_calculation");

    for size in [100, 1_000, 10_000] {
        let returns: Vec<f64> = (0..size).map(|i| (i as f64 * 0.001).sin()).collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;

        group.bench_with_input(
            BenchmarkId::new("iterator_map", size),
            &returns,
            |b, rets| {
                b.iter(|| {
                    let variance =
                        rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
                    black_box(variance)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_returns_calculation,
    bench_variance_calculation
);
criterion_main!(benches);
