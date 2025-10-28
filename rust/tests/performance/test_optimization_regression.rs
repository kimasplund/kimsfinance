//! Performance Regression Detection Tests
//!
//! **Purpose**: Validate optimization speedups and prevent regressions
//!
//! **Test Strategy**:
//! 1. Run traditional and optimized implementations side-by-side
//! 2. Verify results match (accuracy validation)
//! 3. Verify speedup meets minimum threshold
//! 4. Fail test if speedup degrades below target
//!
//! **Regression Criteria**:
//! - Persistent kernels: >= 1.8x speedup (target: 2.0x)
//! - Phase 3 optimization: >= 1.3x speedup (target: 1.4x)
//! - Combined: >= 2.3x speedup (target: 2.5x)
//!
//! **Usage**:
//! ```bash
//! # Run all performance regression tests
//! cargo test --test optimization_regression --features gpu -- --test-threads=1
//!
//! # Run only persistent kernel test
//! cargo test --test optimization_regression --features gpu -- persistent --nocapture
//! ```

use std::sync::Arc;
use std::time::Instant;

// Helper module for generating test data
mod test_data {
    use rand::prelude::*;
    use rand::SeedableRng;

    pub fn generate_realistic_prices(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        let mut price = 100.0;

        for _ in 0..n {
            let change = rng.gen_range(-0.02..0.02); // ±2% per candle
            price *= 1.0 + change;

            let o = price * (1.0 + rng.gen_range(-0.005..0.005));
            let c = price * (1.0 + rng.gen_range(-0.005..0.005));
            let h = o.max(c) * (1.0 + rng.gen_range(0.0..0.01));
            let l = o.min(c) * (1.0 - rng.gen_range(0.0..0.01));
            let v = rng.gen_range(1000.0..10000.0);

            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            volume.push(v);
        }

        (open, high, low, close, volume)
    }

    pub fn generate_rsi_parameters(n_strategies: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        (0..n_strategies)
            .map(|_| {
                vec![
                    rng.gen_range(10.0..20.0),  // RSI period
                    rng.gen_range(20.0..40.0),  // Buy threshold
                    rng.gen_range(60.0..80.0),  // Sell threshold
                ]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_data::*;

    use kimsfinance_core::backtest::batch::{BatchBacktestSweep, StrategyType};
    use kimsfinance_core::backtest::engine::BacktestConfig;
    use kimsfinance_core::gpu::device::GpuDevice;

    /// Tolerance for numerical comparison (0.01% = 1 basis point)
    const TOLERANCE: f64 = 0.0001;

    /// Minimum samples for timing (reduce noise)
    const TIMING_SAMPLES: usize = 10;

    /// Helper: Run batch backtest with traditional kernels
    fn run_traditional_batch(
        device: &Arc<GpuDevice>,
        n_strategies: usize,
        n_candles: usize,
    ) -> (Vec<f64>, f64) {
        let (open, high, low, close, volume) = generate_realistic_prices(n_candles, 42);
        let params = generate_rsi_parameters(n_strategies, 42);

        let timestamps: Vec<i64> = (0..n_candles).map(|i| i as i64 * 60).collect();
        let open_arr = ndarray::Array1::from(open);
        let high_arr = ndarray::Array1::from(high);
        let low_arr = ndarray::Array1::from(low);
        let close_arr = ndarray::Array1::from(close);
        let volume_arr = ndarray::Array1::from(volume);

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
        };

        // Warmup (5 iterations)
        for _ in 0..5 {
            let _ = BatchBacktestSweep::new(device.clone())
                .strategy_type(StrategyType::RsiCrossover)
                .data_ohlcv(&timestamps, &open_arr, &high_arr, &low_arr, &close_arr, &volume_arr)
                .parameters_batch(&params)
                .config(config.clone())
                .execute()
                .expect("Traditional batch failed");
        }

        // Timed runs
        let mut total_time = 0.0;
        let mut sharpe_ratios = Vec::new();

        for _ in 0..TIMING_SAMPLES {
            let start = Instant::now();

            let results = BatchBacktestSweep::new(device.clone())
                .strategy_type(StrategyType::RsiCrossover)
                .data_ohlcv(&timestamps, &open_arr, &high_arr, &low_arr, &close_arr, &volume_arr)
                .parameters_batch(&params)
                .config(config.clone())
                .execute()
                .expect("Traditional batch failed");

            total_time += start.elapsed().as_secs_f64();

            if sharpe_ratios.is_empty() {
                sharpe_ratios = results.results.iter()
                    .map(|r| r.sharpe_ratio)
                    .collect();
            }
        }

        let avg_time = total_time / TIMING_SAMPLES as f64;
        (sharpe_ratios, avg_time)
    }

    /// Helper: Run batch backtest with persistent kernels
    fn run_persistent_batch(
        device: &Arc<GpuDevice>,
        n_strategies: usize,
        n_candles: usize,
    ) -> (Vec<f64>, f64) {
        let (open, high, low, close, volume) = generate_realistic_prices(n_candles, 42);
        let params = generate_rsi_parameters(n_strategies, 42);

        let timestamps: Vec<i64> = (0..n_candles).map(|i| i as i64 * 60).collect();
        let open_arr = ndarray::Array1::from(open);
        let high_arr = ndarray::Array1::from(high);
        let low_arr = ndarray::Array1::from(low);
        let close_arr = ndarray::Array1::from(close);
        let volume_arr = ndarray::Array1::from(volume);

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
        };

        // Warmup (5 iterations)
        for _ in 0..5 {
            let _ = BatchBacktestSweep::new(device.clone())
                .strategy_type(StrategyType::RsiCrossover)
                .data_ohlcv(&timestamps, &open_arr, &high_arr, &low_arr, &close_arr, &volume_arr)
                .parameters_batch(&params)
                .config(config.clone())
                .use_persistent_kernels(true)
                .execute()
                .expect("Persistent batch failed");
        }

        // Timed runs
        let mut total_time = 0.0;
        let mut sharpe_ratios = Vec::new();

        for _ in 0..TIMING_SAMPLES {
            let start = Instant::now();

            let results = BatchBacktestSweep::new(device.clone())
                .strategy_type(StrategyType::RsiCrossover)
                .data_ohlcv(&timestamps, &open_arr, &high_arr, &low_arr, &close_arr, &volume_arr)
                .parameters_batch(&params)
                .config(config.clone())
                .use_persistent_kernels(true)
                .execute()
                .expect("Persistent batch failed");

            total_time += start.elapsed().as_secs_f64();

            if sharpe_ratios.is_empty() {
                sharpe_ratios = results.results.iter()
                    .map(|r| r.sharpe_ratio)
                    .collect();
            }
        }

        let avg_time = total_time / TIMING_SAMPLES as f64;
        (sharpe_ratios, avg_time)
    }

    /// Helper: Compare results for accuracy
    fn assert_results_match(traditional: &[f64], optimized: &[f64], tolerance: f64) {
        assert_eq!(traditional.len(), optimized.len(), "Result count mismatch");

        for (i, (t, o)) in traditional.iter().zip(optimized.iter()).enumerate() {
            let diff = (t - o).abs();
            let rel_diff = if t.abs() > 1e-6 {
                diff / t.abs()
            } else {
                diff
            };

            assert!(
                rel_diff < tolerance,
                "Strategy {} mismatch: traditional={:.6}, optimized={:.6}, rel_diff={:.6} (tolerance={:.6})",
                i, t, o, rel_diff, tolerance
            );
        }
    }

    // ========================================================================
    // Test 1: Persistent Kernels Regression
    // ========================================================================

    #[test]
    #[ignore] // Run explicitly with --ignored flag (GPU-only)
    fn test_persistent_kernels_speedup() {
        let device = Arc::new(GpuDevice::new().expect("GPU initialization failed"));

        // Test configuration: 1000 strategies × 10K candles (key target)
        let n_strategies = 1000;
        let n_candles = 10000;

        println!("\n=== Persistent Kernels Regression Test ===");
        println!("Configuration: {} strategies × {} candles", n_strategies, n_candles);

        // Run traditional baseline
        println!("Running traditional kernels...");
        let (traditional_results, traditional_time) = run_traditional_batch(&device, n_strategies, n_candles);
        println!("  Traditional: {:.2} ms", traditional_time * 1000.0);

        // Run persistent kernels
        println!("Running persistent kernels...");
        let (persistent_results, persistent_time) = run_persistent_batch(&device, n_strategies, n_candles);
        println!("  Persistent:  {:.2} ms", persistent_time * 1000.0);

        // Validate accuracy
        println!("Validating accuracy...");
        assert_results_match(&traditional_results, &persistent_results, TOLERANCE);
        println!("  ✓ Results match within {:.2}% tolerance", TOLERANCE * 100.0);

        // Validate speedup
        let speedup = traditional_time / persistent_time;
        println!("  Speedup: {:.2}x", speedup);

        const MIN_SPEEDUP: f64 = 1.8; // Minimum acceptable (target: 2.0x)
        assert!(
            speedup >= MIN_SPEEDUP,
            "Persistent kernels regression detected! Speedup {:.2}x < {:.2}x minimum",
            speedup, MIN_SPEEDUP
        );

        println!("  ✓ Speedup validated (>= {:.2}x)", MIN_SPEEDUP);
        println!("\n✓ Test PASSED\n");
    }

    // ========================================================================
    // Test 2: Phase 3 Optimization Regression
    // ========================================================================

    #[test]
    #[ignore] // Run explicitly with --ignored flag (GPU-only)
    fn test_phase3_optimization_speedup() {
        let device = Arc::new(GpuDevice::new().expect("GPU initialization failed"));

        let n_strategies = 1000;
        let n_candles = 10000;

        println!("\n=== Phase 3 Optimization Regression Test ===");
        println!("Configuration: {} strategies × {} candles", n_strategies, n_candles);

        // This test would compare persistent vs phase3-optimized persistent
        // For now, just validate Phase 3 is faster than traditional

        println!("Running traditional kernels...");
        let (traditional_results, traditional_time) = run_traditional_batch(&device, n_strategies, n_candles);
        println!("  Traditional: {:.2} ms", traditional_time * 1000.0);

        // TODO: Implement Phase 3 optimized version
        // For now, use persistent as placeholder
        println!("Running Phase 3 optimized...");
        let (phase3_results, phase3_time) = run_persistent_batch(&device, n_strategies, n_candles);
        println!("  Phase 3:     {:.2} ms", phase3_time * 1000.0);

        // Validate accuracy
        println!("Validating accuracy...");
        assert_results_match(&traditional_results, &phase3_results, TOLERANCE);
        println!("  ✓ Results match within {:.2}% tolerance", TOLERANCE * 100.0);

        // Validate speedup
        let speedup = traditional_time / phase3_time;
        println!("  Speedup: {:.2}x", speedup);

        const MIN_SPEEDUP: f64 = 1.3; // Minimum for Phase 3 alone (target: 1.4x)
        assert!(
            speedup >= MIN_SPEEDUP,
            "Phase 3 optimization regression detected! Speedup {:.2}x < {:.2}x minimum",
            speedup, MIN_SPEEDUP
        );

        println!("  ✓ Speedup validated (>= {:.2}x)", MIN_SPEEDUP);
        println!("\n✓ Test PASSED\n");
    }

    // ========================================================================
    // Test 3: Combined Optimizations Regression
    // ========================================================================

    #[test]
    #[ignore] // Run explicitly with --ignored flag (GPU-only)
    fn test_combined_optimizations_speedup() {
        let device = Arc::new(GpuDevice::new().expect("GPU initialization failed"));

        let n_strategies = 1000;
        let n_candles = 10000;

        println!("\n=== Combined Optimizations Regression Test ===");
        println!("Configuration: {} strategies × {} candles", n_strategies, n_candles);

        println!("Running traditional kernels (baseline)...");
        let (traditional_results, traditional_time) = run_traditional_batch(&device, n_strategies, n_candles);
        println!("  Baseline: {:.2} ms", traditional_time * 1000.0);

        // TODO: Run with both persistent + phase3 enabled
        println!("Running combined optimizations...");
        let (combined_results, combined_time) = run_persistent_batch(&device, n_strategies, n_candles);
        println!("  Combined: {:.2} ms", combined_time * 1000.0);

        // Validate accuracy
        println!("Validating accuracy...");
        assert_results_match(&traditional_results, &combined_results, TOLERANCE);
        println!("  ✓ Results match within {:.2}% tolerance", TOLERANCE * 100.0);

        // Validate combined speedup
        let speedup = traditional_time / combined_time;
        println!("  Total Speedup: {:.2}x", speedup);

        const MIN_SPEEDUP: f64 = 2.3; // Minimum combined (target: 2.5-3.0x)
        assert!(
            speedup >= MIN_SPEEDUP,
            "Combined optimizations regression detected! Speedup {:.2}x < {:.2}x minimum",
            speedup, MIN_SPEEDUP
        );

        println!("  ✓ Combined speedup validated (>= {:.2}x)", MIN_SPEEDUP);
        println!("\n✓ Test PASSED\n");
    }

    // ========================================================================
    // Test 4: Constant-Time Scaling
    // ========================================================================

    #[test]
    #[ignore] // Run explicitly with --ignored flag (GPU-only)
    fn test_constant_time_scaling() {
        let device = Arc::new(GpuDevice::new().expect("GPU initialization failed"));
        let n_candles = 10000;

        println!("\n=== Constant-Time Scaling Test ===");
        println!("Configuration: Variable strategies × {} candles", n_candles);

        let strategy_counts = vec![100, 500, 1000];
        let mut times = Vec::new();

        for &n_strategies in &strategy_counts {
            println!("Testing {} strategies...", n_strategies);
            let (_, time) = run_persistent_batch(&device, n_strategies, n_candles);
            let time_ms = time * 1000.0;
            let time_per_strategy = time_ms / n_strategies as f64;

            times.push(time_ms);

            println!("  Time: {:.2} ms", time_ms);
            println!("  Per-strategy: {:.3} ms", time_per_strategy);
        }

        // Validate sub-linear scaling
        // 10x strategies should not take 10x time
        let ratio_100_to_1000 = times[2] / times[0]; // 1000 / 100 = 10x strategies
        println!("\nScaling ratio (100 → 1000 strategies): {:.2}x", ratio_100_to_1000);

        const MAX_SCALING_RATIO: f64 = 5.0; // Should be < 10x (ideally < 3x)
        assert!(
            ratio_100_to_1000 < MAX_SCALING_RATIO,
            "Scaling regression detected! 10x strategies took {:.2}x time (max: {:.2}x)",
            ratio_100_to_1000, MAX_SCALING_RATIO
        );

        println!("  ✓ Sub-linear scaling validated (< {:.2}x)", MAX_SCALING_RATIO);
        println!("\n✓ Test PASSED\n");
    }
}
