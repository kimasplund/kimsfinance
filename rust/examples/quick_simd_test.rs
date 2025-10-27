/// Quick SIMD performance validation for Agent 2
///
/// This is a micro-benchmark to quickly validate the SIMD fix performance.
use kimsfinance_core::backtest::metrics::{
    calculate_sharpe_ratio_scalar, calculate_sharpe_ratio_simd,
};
use std::time::Instant;

fn main() {
    println!("=== Quick SIMD Performance Validation ===\n");

    // Test with different dataset sizes
    for size in [100, 1_000, 10_000, 100_000] {
        println!("Dataset size: {}", size);

        // Create test equity curve (upward trend with some noise)
        let equity: Vec<f64> = (0..size)
            .map(|i| 10000.0 + i as f64 * 10.0 + (i as f64 * 0.1).sin() * 100.0)
            .collect();

        // Warmup
        for _ in 0..5 {
            let _ = calculate_sharpe_ratio_scalar(&equity);
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                let _ = calculate_sharpe_ratio_simd(&equity);
            }
        }

        // Benchmark scalar
        let iterations = if size < 1000 {
            10000
        } else if size < 10000 {
            1000
        } else {
            100
        };
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(calculate_sharpe_ratio_scalar(std::hint::black_box(&equity)));
        }
        let scalar_time = start.elapsed();
        let scalar_ns_per_op = scalar_time.as_nanos() / iterations as u128;

        println!("  Scalar:  {:>8} ns/op", scalar_ns_per_op);

        // Benchmark SIMD
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            let start = Instant::now();
            for _ in 0..iterations {
                std::hint::black_box(calculate_sharpe_ratio_simd(std::hint::black_box(&equity)));
            }
            let simd_time = start.elapsed();
            let simd_ns_per_op = simd_time.as_nanos() / iterations as u128;

            println!("  SIMD:    {:>8} ns/op", simd_ns_per_op);

            let speedup = scalar_ns_per_op as f64 / simd_ns_per_op as f64;
            println!("  Speedup: {:.2}x", speedup);

            if speedup > 1.0 {
                println!("  Status:  ✓ SIMD faster");
            } else {
                println!("  Status:  ✗ SIMD slower (BUG!)");
            }
        } else {
            println!("  SIMD:    Not available (AVX2 not detected)");
        }

        println!();
    }
}
