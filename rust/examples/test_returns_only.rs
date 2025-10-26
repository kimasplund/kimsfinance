/// Test SIMD speedup for returns calculation only
///
/// This isolates the returns calculation from Sharpe ratio to measure pure SIMD benefit.

use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Scalar return calculation
fn calculate_returns_scalar(equity_curve: &[f64]) -> Vec<f64> {
    let mut returns = Vec::with_capacity(equity_curve.len() - 1);
    for i in 1..equity_curve.len() {
        if equity_curve[i - 1] != 0.0 {
            let ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1];
            returns.push(ret);
        }
    }
    returns
}

/// SIMD return calculation (FIXED)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn calculate_returns_simd(equity_curve: &[f64]) -> Vec<f64> {
    let n = equity_curve.len();
    let mut returns: Vec<f64> = Vec::with_capacity(n - 1);
    let num_chunks = (n - 1) / 4;
    let remainder_start = 1 + num_chunks * 4;

    returns.reserve(n - 1);

    unsafe {
        for chunk in 0..num_chunks {
            let i = 1 + chunk * 4;

            let curr = _mm256_loadu_pd(equity_curve.as_ptr().add(i));
            let prev = _mm256_loadu_pd(equity_curve.as_ptr().add(i - 1));

            let diff = _mm256_sub_pd(curr, prev);
            let ret = _mm256_div_pd(diff, prev);

            // Direct store (NO scalar loop!)
            let old_len = returns.len();
            returns.set_len(old_len + 4);
            _mm256_storeu_pd(returns.as_mut_ptr().add(old_len), ret);
        }
    }

    for i in remainder_start..n {
        if equity_curve[i - 1] != 0.0 {
            let ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1];
            returns.push(ret);
        }
    }

    returns.retain(|&r| r.is_finite());
    returns
}

fn main() {
    println!("=== Returns Calculation SIMD Speedup ===\n");

    for size in [100, 1_000, 10_000, 100_000] {
        println!("Dataset size: {}", size);

        let equity: Vec<f64> = (0..size)
            .map(|i| 10000.0 + i as f64 * 10.0 + (i as f64 * 0.1).sin() * 100.0)
            .collect();

        // Warmup
        for _ in 0..5 {
            let _ = calculate_returns_scalar(&equity);
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                unsafe {
                    let _ = calculate_returns_simd(&equity);
                }
            }
        }

        // Benchmark scalar
        let iterations = if size < 1000 { 100000 } else if size < 10000 { 10000 } else { 1000 };
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(calculate_returns_scalar(std::hint::black_box(&equity)));
        }
        let scalar_time = start.elapsed();
        let scalar_ns_per_op = scalar_time.as_nanos() / iterations as u128;

        println!("  Scalar:  {:>8} ns/op", scalar_ns_per_op);

        // Benchmark SIMD
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            let start = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    std::hint::black_box(calculate_returns_simd(std::hint::black_box(&equity)));
                }
            }
            let simd_time = start.elapsed();
            let simd_ns_per_op = simd_time.as_nanos() / iterations as u128;

            println!("  SIMD:    {:>8} ns/op", simd_ns_per_op);

            let speedup = scalar_ns_per_op as f64 / simd_ns_per_op as f64;
            println!("  Speedup: {:.2}x", speedup);

            if speedup >= 2.0 {
                println!("  Status:  ✓ Excellent (2x+ speedup)");
            } else if speedup > 1.0 {
                println!("  Status:  ✓ SIMD faster (but <2x)");
            } else {
                println!("  Status:  ✗ SIMD slower (BUG!)");
            }
        }

        println!();
    }
}
