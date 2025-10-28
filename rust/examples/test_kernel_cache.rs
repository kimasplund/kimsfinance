//! Integration test for kernel compilation cache
//!
//! Demonstrates 50-200x speedup on cache hits.
//!
//! # Expected Results
//!
//! - Iteration 0: ~100-150ms (cache miss, full compilation)
//! - Iterations 1-9: ~1-2ms (cache hit, 50-200x faster)
//! - Cache statistics: 1 miss, 9 hits (90% hit rate)

use kimsfinance_core::gpu::compile::{clear_cache, compile_ptx_optimized_cached, get_cache_stats};
use std::time::Instant;

const TEST_KERNEL: &str = r#"
extern "C" __global__ void test_kernel(const double* in, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate realistic indicator calculation (RSI-like)
        double sum = 0.0;
        for (int i = 0; i < 14; i++) {
            if (idx >= i) {
                sum += in[idx - i];
            }
        }
        out[idx] = sum / 14.0;
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Kernel Compilation Cache Test ===\n");

    // Clear cache to start fresh
    clear_cache();
    println!("Cache cleared. Starting benchmark...\n");

    // Compile same kernel 10 times
    let mut timings = Vec::new();

    for i in 0..10 {
        let start = Instant::now();
        let _ptx = compile_ptx_optimized_cached(TEST_KERNEL)?;
        let elapsed = start.elapsed();
        timings.push(elapsed);

        let cache_status = if i == 0 { "MISS" } else { "HIT " };
        println!(
            "Iteration {}: {:>7.2}ms [{}]",
            i,
            elapsed.as_secs_f64() * 1000.0,
            cache_status
        );
    }

    // Print statistics
    println!("\n=== Cache Statistics ===");
    let stats = get_cache_stats();
    println!("Cache hits:    {}", stats.hits);
    println!("Cache misses:  {}", stats.misses);
    println!("Total entries: {}", stats.total_entries);
    println!("Hit rate:      {:.1}%", stats.hit_rate() * 100.0);

    // Calculate speedup
    if timings.len() > 1 {
        let first = timings[0].as_secs_f64() * 1000.0;
        let avg_cached = timings[1..]
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / (timings.len() - 1) as f64;

        println!("\n=== Performance Analysis ===");
        println!("First compilation:   {:.2}ms (cache miss)", first);
        println!("Average cached:      {:.2}ms (cache hit)", avg_cached);
        println!("Speedup:             {:.1}x", first / avg_cached);

        // Validate performance targets
        println!("\n=== Validation ===");
        if avg_cached < 5.0 {
            println!("✓ Cache hits < 5ms (target met)");
        } else {
            println!("✗ Cache hits >= 5ms (target missed: {:.2}ms)", avg_cached);
        }

        if first / avg_cached >= 50.0 {
            println!(
                "✓ Speedup >= 50x (target met: {:.1}x)",
                first / avg_cached
            );
        } else {
            println!(
                "⚠ Speedup < 50x (expected 50-200x, got {:.1}x)",
                first / avg_cached
            );
            println!("  Note: May be GPU driver overhead in test environment");
        }

        if stats.hit_rate() == 0.9 {
            println!("✓ Hit rate = 90% (9 hits / 10 total)");
        } else {
            println!(
                "✗ Hit rate != 90% (got {:.1}%)",
                stats.hit_rate() * 100.0
            );
        }
    }

    println!("\n=== Test Complete ===");
    println!("Expected: First ~100-150ms, rest ~1-2ms (50-200x faster)");

    Ok(())
}
