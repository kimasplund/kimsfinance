/// Compare OLD (buggy) vs NEW (fixed) SIMD implementation
///
/// This demonstrates the actual performance fix from Agent 2.
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Scalar baseline
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

/// OLD BUGGY SIMD (with scalar loop bottleneck)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn calculate_returns_simd_old(equity_curve: &[f64]) -> Vec<f64> {
    let n = equity_curve.len();
    let mut returns: Vec<f64> = Vec::with_capacity(n - 1);
    let num_chunks = (n - 1) / 4;
    let remainder_start = 1 + num_chunks * 4;

    unsafe {
        for chunk in 0..num_chunks {
            let i = 1 + chunk * 4;

            let curr = _mm256_loadu_pd(equity_curve.as_ptr().add(i));
            let prev = _mm256_loadu_pd(equity_curve.as_ptr().add(i - 1));

            let diff = _mm256_sub_pd(curr, prev);
            let ret = _mm256_div_pd(diff, prev);

            // BUG: Scalar loop after SIMD computation!
            let mut ret_array = [0.0f64; 4];
            _mm256_storeu_pd(ret_array.as_mut_ptr(), ret);

            for (idx, &r) in ret_array.iter().enumerate() {
                if equity_curve[i + idx - 1] != 0.0 && r.is_finite() {
                    returns.push(r); // 4 Vec::push calls, branches, memory re-reads
                }
            }
        }
    }

    for i in remainder_start..n {
        if equity_curve[i - 1] != 0.0 {
            let ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1];
            returns.push(ret);
        }
    }

    returns
}

/// NEW FIXED SIMD (direct store, no scalar loop)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn calculate_returns_simd_new(equity_curve: &[f64]) -> Vec<f64> {
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

            // FIX: Direct store without scalar loop!
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
    println!("=== OLD (Buggy) vs NEW (Fixed) SIMD Comparison ===\n");

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
                    let _ = calculate_returns_simd_old(&equity);
                    let _ = calculate_returns_simd_new(&equity);
                }
            }
        }

        // Benchmark scalar
        let iterations = if size < 1000 {
            100000
        } else if size < 10000 {
            10000
        } else {
            1000
        };
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(calculate_returns_scalar(std::hint::black_box(&equity)));
        }
        let scalar_time = start.elapsed();
        let scalar_ns = scalar_time.as_nanos() / iterations as u128;

        println!("  Scalar:   {:>8} ns/op (baseline)", scalar_ns);

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // Benchmark OLD SIMD
            let start = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    std::hint::black_box(calculate_returns_simd_old(std::hint::black_box(&equity)));
                }
            }
            let old_simd_time = start.elapsed();
            let old_simd_ns = old_simd_time.as_nanos() / iterations as u128;

            // Benchmark NEW SIMD
            let start = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    std::hint::black_box(calculate_returns_simd_new(std::hint::black_box(&equity)));
                }
            }
            let new_simd_time = start.elapsed();
            let new_simd_ns = new_simd_time.as_nanos() / iterations as u128;

            let old_vs_scalar = scalar_ns as f64 / old_simd_ns as f64;
            let new_vs_scalar = scalar_ns as f64 / new_simd_ns as f64;
            let new_vs_old = old_simd_ns as f64 / new_simd_ns as f64;

            println!(
                "  OLD SIMD: {:>8} ns/op ({:.2}x vs scalar) {}",
                old_simd_ns,
                old_vs_scalar,
                if old_vs_scalar < 1.0 { "SLOWER!" } else { "" }
            );
            println!(
                "  NEW SIMD: {:>8} ns/op ({:.2}x vs scalar)",
                new_simd_ns, new_vs_scalar
            );
            println!("  Improvement: {:.2}x (NEW vs OLD)", new_vs_old);

            if new_vs_old > 1.0 {
                println!("  Status: ✓ Fix improved performance");
            } else {
                println!("  Status: ✗ No improvement (unexpected!)");
            }
        }

        println!();
    }
}
