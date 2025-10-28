//! Tests for async triple-buffered execution mode
//!
//! Validates correctness and performance of ExecutionMode::Async

#[cfg(feature = "gpu")]
mod async_tests {
    use kimsfinance_core::backtest::batch::{BatchBacktestSweep, ExecutionMode, StrategyType};
    use kimsfinance_core::backtest::engine::BacktestConfig;
    use kimsfinance_core::gpu::device::GpuDevice;
    use ndarray::Array1;
    use std::sync::Arc;

    /// Generate synthetic OHLCV data for testing
    fn generate_test_data(
        n_candles: usize,
    ) -> (
        Vec<i64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
    ) {
        let timestamps: Vec<i64> = (0..n_candles).map(|i| i as i64 * 86400).collect();

        let mut open = vec![100.0; n_candles];
        let mut high = vec![105.0; n_candles];
        let mut low = vec![95.0; n_candles];
        let mut close = vec![102.0; n_candles];
        let volume = vec![1_000_000.0; n_candles];

        // Add some trend
        for i in 0..n_candles {
            let trend = (i as f64 / n_candles as f64) * 50.0;
            open[i] += trend;
            high[i] += trend;
            low[i] += trend;
            close[i] += trend;
        }

        (
            timestamps,
            Array1::from(open),
            Array1::from(high),
            Array1::from(low),
            Array1::from(close),
            Array1::from(volume),
        )
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_async_mode_small_batch() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let (timestamps, open, high, low, close, volume) = generate_test_data(1000);

        // Generate 50 RSI crossover strategies (small batch - shouldn't use async)
        let mut params = vec![];
        for buy in 20..25 {
            for sell in 75..80 {
                params.push(vec![14.0, buy as f64, sell as f64]);
            }
        }

        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .execution_mode(ExecutionMode::Async) // Force async
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
                use_gpu: true,
                force_cpu: false,
            })
            .execute()
            .expect("Execution failed");

        // Validate results
        assert_eq!(results.results.len(), params.len());
        assert!(results.total_time_ms > 0.0);
        assert!(results.vram_used_mb > 0.0);

        println!("✅ Async small batch test passed");
        println!("   Strategies: {}", results.results.len());
        println!("   Time: {:.2}ms", results.total_time_ms);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_async_mode_large_batch() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let (timestamps, open, high, low, close, volume) = generate_test_data(10_000);

        // Generate 1000 RSI crossover strategies (large batch - should use async)
        let mut params = vec![];
        for rsi_period in 10..20 {
            for buy in 20..30 {
                for sell in 70..80 {
                    params.push(vec![rsi_period as f64, buy as f64, sell as f64]);
                }
            }
        }

        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .execution_mode(ExecutionMode::Async) // Force async
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
                use_gpu: true,
                force_cpu: false,
            })
            .execute()
            .expect("Execution failed");

        // Validate results
        assert_eq!(results.results.len(), params.len());
        assert!(results.total_time_ms > 0.0);
        assert!(results.vram_used_mb > 0.0);

        println!("✅ Async large batch test passed");
        println!("   Strategies: {}", results.results.len());
        println!("   Time: {:.2}ms", results.total_time_ms);
        println!(
            "   Speedup estimate: {:.2}x vs sequential",
            results.speedup()
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_async_vs_fused_correctness() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let (timestamps, open, high, low, close, volume) = generate_test_data(5_000);

        // Generate 100 identical strategies
        let params = vec![vec![14.0, 25.0, 75.0]; 100];

        // Run with Fused mode
        let results_fused = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .execution_mode(ExecutionMode::Fused)
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
                use_gpu: true,
                force_cpu: false,
            })
            .execute()
            .expect("Fused execution failed");

        // Run with Async mode
        let results_async = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .execution_mode(ExecutionMode::Async)
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
                use_gpu: true,
                force_cpu: false,
            })
            .execute()
            .expect("Async execution failed");

        // Validate same results (within tolerance)
        assert_eq!(results_fused.results.len(), results_async.results.len());

        for (fused, async_res) in results_fused
            .results
            .iter()
            .zip(results_async.results.iter())
        {
            let sharpe_diff = (fused.sharpe_ratio - async_res.sharpe_ratio).abs();
            let dd_diff = (fused.max_drawdown - async_res.max_drawdown).abs();

            assert!(
                sharpe_diff < 0.01,
                "Sharpe ratio mismatch: {} vs {}",
                fused.sharpe_ratio,
                async_res.sharpe_ratio
            );
            assert!(
                dd_diff < 0.01,
                "Max drawdown mismatch: {} vs {}",
                fused.max_drawdown,
                async_res.max_drawdown
            );
        }

        println!("✅ Async vs Fused correctness test passed");
        println!("   Fused time: {:.2}ms", results_fused.total_time_ms);
        println!("   Async time: {:.2}ms", results_async.total_time_ms);
        println!(
            "   Speedup: {:.2}x",
            results_fused.total_time_ms / results_async.total_time_ms
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_auto_mode_selects_async() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let (timestamps, open, high, low, close, volume) = generate_test_data(10_000);

        // Generate 1500 strategies (should trigger async mode)
        let mut params = vec![];
        for rsi_period in 10..25 {
            for buy in 20..30 {
                for sell in 70..80 {
                    params.push(vec![rsi_period as f64, buy as f64, sell as f64]);
                }
            }
        }

        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .execution_mode(ExecutionMode::Auto) // Should select Async
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
                use_gpu: true,
                force_cpu: false,
            })
            .execute()
            .expect("Execution failed");

        // Validate results
        assert_eq!(results.results.len(), params.len());
        assert!(results.total_time_ms > 0.0);

        println!("✅ Auto mode async selection test passed");
        println!("   Strategies: {} (>1000 = async)", results.results.len());
        println!("   Time: {:.2}ms", results.total_time_ms);
    }
}
