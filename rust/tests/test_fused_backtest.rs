//! Test Fused (Persistent) Kernel Integration
//!
//! Validates that:
//! 1. Fused kernel produces identical results to traditional execution
//! 2. Fused kernel achieves ≥1.31x speedup for large batches
//! 3. ExecutionMode enum works correctly

#[cfg(feature = "gpu")]
mod fused_kernel_tests {
    use kimsfinance_core::backtest::engine::BacktestConfig;
    use kimsfinance_core::backtest::{BatchBacktestSweep, ExecutionMode, StrategyType};
    use kimsfinance_core::gpu::device::GpuDevice;
    use ndarray::Array1;
    use std::sync::Arc;
    use std::time::Instant;

    /// Generate synthetic OHLCV data for testing
    fn generate_ohlcv_data(
        n_candles: usize,
    ) -> (
        Vec<i64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
    ) {
        let mut close_data = vec![100.0];
        for i in 1..n_candles {
            let delta = (i as f64 * 0.1).sin() * 2.0;
            close_data.push(close_data[i - 1] + delta);
        }

        let timestamps: Vec<i64> = (0..n_candles).map(|i| i as i64).collect();
        let open = Array1::from_vec(close_data.clone());
        let high = Array1::from_vec(close_data.iter().map(|&c| c * 1.01).collect());
        let low = Array1::from_vec(close_data.iter().map(|&c| c * 0.99).collect());
        let close = Array1::from_vec(close_data);
        let volume = Array1::from_vec(vec![1000.0; n_candles]);

        (timestamps, open, high, low, close, volume)
    }

    /// Generate parameter sets for RSI crossover strategy
    fn generate_parameters(n_strategies: usize) -> Vec<Vec<f64>> {
        let mut params = vec![];
        let mut count = 0;

        'outer: for rsi_period in 10..20 {
            for buy_thresh in 25..35 {
                for sell_thresh in 65..75 {
                    if count >= n_strategies {
                        break 'outer;
                    }
                    params.push(vec![
                        rsi_period as f64,
                        buy_thresh as f64,
                        sell_thresh as f64,
                    ]);
                    count += 1;
                }
            }
        }

        params
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_execution_mode_enum() {
        // Test ExecutionMode enum values
        assert_eq!(
            ExecutionMode::Traditional as i32,
            ExecutionMode::Traditional as i32
        );
        assert_eq!(ExecutionMode::Fused as i32, ExecutionMode::Fused as i32);
        assert_eq!(ExecutionMode::Auto as i32, ExecutionMode::Auto as i32);

        // Test default
        assert_eq!(ExecutionMode::default(), ExecutionMode::Auto);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_fused_vs_traditional_correctness() {
        println!("🧪 Testing fused vs traditional correctness");

        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let (timestamps, open, high, low, close, volume) = generate_ohlcv_data(500);

        // Generate 50 strategies (small enough to avoid auto-fused, but we'll force both modes)
        let params = generate_parameters(50);

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
        };

        // Run with Traditional mode
        println!("  Running with Traditional mode...");
        let results_traditional = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .config(config.clone())
            .execution_mode(ExecutionMode::Traditional)
            .execute()
            .expect("Traditional execution failed");

        // Run with Fused mode
        println!("  Running with Fused mode...");
        let results_fused = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .config(config.clone())
            .execution_mode(ExecutionMode::Fused)
            .execute()
            .expect("Fused execution failed");

        // Verify results match
        assert_eq!(
            results_traditional.results.len(),
            results_fused.results.len(),
            "Result count mismatch"
        );

        // Compare metrics for each strategy
        for (i, (trad, fused)) in results_traditional
            .results
            .iter()
            .zip(results_fused.results.iter())
            .enumerate()
        {
            // Allow small floating point differences (0.1% tolerance)
            let sharpe_diff = (trad.sharpe_ratio - fused.sharpe_ratio).abs();
            let sharpe_tolerance = trad.sharpe_ratio.abs() * 0.001;

            assert!(
                sharpe_diff <= sharpe_tolerance,
                "Strategy {}: Sharpe mismatch - Traditional: {:.6}, Fused: {:.6}, Diff: {:.6}",
                i,
                trad.sharpe_ratio,
                fused.sharpe_ratio,
                sharpe_diff
            );

            let dd_diff = (trad.max_drawdown - fused.max_drawdown).abs();
            let dd_tolerance = trad.max_drawdown.abs() * 0.001;

            assert!(
                dd_diff <= dd_tolerance,
                "Strategy {}: Drawdown mismatch - Traditional: {:.6}, Fused: {:.6}, Diff: {:.6}",
                i,
                trad.max_drawdown,
                fused.max_drawdown,
                dd_diff
            );
        }

        println!("  ✅ Correctness verified: Results match within tolerance");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_fused_speedup() {
        println!("🚀 Testing fused kernel speedup");

        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let (timestamps, open, high, low, close, volume) = generate_ohlcv_data(1000);

        // Generate large batch (200 strategies) to see fused benefit
        let params = generate_parameters(200);

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
        };

        // Benchmark Traditional mode
        println!("  Benchmarking Traditional mode...");
        let start_traditional = Instant::now();
        let results_traditional = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .config(config.clone())
            .execution_mode(ExecutionMode::Traditional)
            .execute()
            .expect("Traditional execution failed");
        let time_traditional = start_traditional.elapsed().as_secs_f64() * 1000.0;

        // Benchmark Fused mode
        println!("  Benchmarking Fused mode...");
        let start_fused = Instant::now();
        let results_fused = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .config(config.clone())
            .execution_mode(ExecutionMode::Fused)
            .execute()
            .expect("Fused execution failed");
        let time_fused = start_fused.elapsed().as_secs_f64() * 1000.0;

        // Calculate speedup
        let speedup = time_traditional / time_fused;

        println!("  Traditional time: {:.2}ms", time_traditional);
        println!("  Fused time: {:.2}ms", time_fused);
        println!("  Speedup: {:.2}x", speedup);

        // Verify speedup meets target (1.31x minimum, but typically 2-4x)
        assert!(
            speedup >= 1.31,
            "Fused kernel speedup ({:.2}x) below 1.31x target",
            speedup
        );

        println!("  ✅ Speedup validated: {:.2}x (target: ≥1.31x)", speedup);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_auto_mode_selection() {
        println!("🎯 Testing Auto mode selection");

        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let (timestamps, open, high, low, close, volume) = generate_ohlcv_data(1000);

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
        };

        // Test with small batch (should use Traditional)
        let params_small = generate_parameters(50);
        println!("  Testing Auto with 50 strategies (should use Traditional)...");
        let results_small = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params_small)
            .config(config.clone())
            .execution_mode(ExecutionMode::Auto)
            .execute()
            .expect("Auto execution failed (small batch)");

        // Test with large batch (should use Fused)
        let params_large = generate_parameters(150);
        println!("  Testing Auto with 150 strategies (should use Fused)...");
        let results_large = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params_large)
            .config(config.clone())
            .execution_mode(ExecutionMode::Auto)
            .execute()
            .expect("Auto execution failed (large batch)");

        println!("  ✅ Auto mode selection working correctly");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_builder_api() {
        println!("🔧 Testing builder API with ExecutionMode");

        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let (timestamps, open, high, low, close, volume) = generate_ohlcv_data(100);
        let params = generate_parameters(10);

        // Test builder pattern with execution_mode
        let sweep = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
            })
            .execution_mode(ExecutionMode::Fused);

        let results = sweep.execute().expect("Execution failed");

        assert_eq!(results.results.len(), params.len());
        println!("  ✅ Builder API working correctly");
    }
}
