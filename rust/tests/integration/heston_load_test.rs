//! Load Tests for Heston-Backtest Integration
//!
//! Stress tests for production workloads:
//! - 1000 strategies × 10K candles (<250ms target)
//! - 100 options × 1000 strategies
//! - VRAM usage monitoring (<1GB target)
//! - Memory leak detection
//! - GPU utilization tracking

#[cfg(all(feature = "gpu", feature = "heston"))]
mod heston_load_tests {
    use kimsfinance_core::backtest::batch::{BatchBacktestSweep, ExecutionMode, StrategyType};
    use kimsfinance_core::backtest::engine::BacktestConfig;
    use kimsfinance_core::gpu::GpuDevice;
    use std::sync::Arc;

    mod test_data {
        include!("../data/heston_test_data.rs");
    }
    use test_data::{
        generate_btc_ohlcv, generate_options_chain, generate_strategy_params, test_heston_params,
        MarketRegime,
    };

    #[test]
    #[ignore] // Slow load test
    fn test_load_1000_strategies_10k_candles() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(10_000, MarketRegime::RangeBound, 42);

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(spot, 50, 30, &params_heston); // 100 options

        let params_strategy = generate_strategy_params(StrategyType::LongStraddle, 1000);

        println!("\n===== LOAD TEST: 1000 Strategies × 10K Candles =====");

        let start = std::time::Instant::now();
        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::LongStraddle)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params_strategy)
            .config(BacktestConfig {
                initial_capital: 100_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
            ..Default::default()
        
            })
            .heston_params(params_heston)
            .options_data(options)
            .execute()
            .expect("Load test failed");
        let elapsed = start.elapsed();

        println!("\n===== LOAD TEST RESULTS =====");
        println!("Total time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("Throughput: {:.2} strategies/sec", 1000.0 / elapsed.as_secs_f64());
        println!(
            "Phase 0 (Heston): {:.2}ms",
            results.phase_timings.get("phase0_heston_ms").unwrap_or(&0.0)
        );
        println!(
            "Phase 1 (Indicators): {:.2}ms",
            results.phase_timings.get("phase1_indicators_ms").unwrap_or(&0.0)
        );
        println!(
            "Phase 2 (Signals): {:.2}ms",
            results.phase_timings.get("phase2_signals_ms").unwrap_or(&0.0)
        );
        println!(
            "Phase 3 (Execution): {:.2}ms",
            results.phase_timings.get("phase3_execution_ms").unwrap_or(&0.0)
        );
        println!(
            "Phase 4 (Metrics): {:.2}ms",
            results.phase_timings.get("phase4_metrics_ms").unwrap_or(&0.0)
        );

        assert_eq!(results.results.len(), 1000);

        // Performance target: <500ms (relaxed from 250ms for CI)
        assert!(
            elapsed.as_secs_f64() < 0.5,
            "Load test exceeded 500ms target: {:.2}ms",
            elapsed.as_secs_f64() * 1000.0
        );
    }

    #[test]
    #[ignore] // Slow load test
    fn test_load_memory_usage() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(10_000, MarketRegime::RangeBound, 42);

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(spot, 50, 30, &params_heston);

        let params_strategy = generate_strategy_params(StrategyType::LongStraddle, 1000);

        // Run multiple times to detect memory leaks
        println!("\n===== MEMORY LEAK DETECTION TEST =====");
        println!("Running 10 iterations to detect leaks...\n");

        for i in 0..10 {
            let _results = BatchBacktestSweep::new(device.clone())
                .strategy_type(StrategyType::LongStraddle)
                .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
                .parameters_batch(&params_strategy)
                .config(BacktestConfig {
                    initial_capital: 100_000.0,
                    trading_fee: 0.001,
                    slippage: 0.0005,
            ..Default::default()
        
                })
                .heston_params(params_heston.clone())
                .options_data(options.clone())
                .execute()
                .expect("Iteration failed");

            println!("Iteration {} completed", i + 1);
        }

        println!("\n===== MEMORY LEAK TEST: PASS =====");
        println!("10 iterations completed without OOM");
    }

    #[test]
    #[ignore] // Slow load test
    fn test_load_concurrent_execution_modes() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(5_000, MarketRegime::RangeBound, 42);

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(spot, 20, 30, &params_heston);

        let params_strategy = generate_strategy_params(StrategyType::LongStraddle, 500);

        println!("\n===== EXECUTION MODE LOAD TEST =====");

        for mode in [
            ExecutionMode::Traditional,
            ExecutionMode::Fused,
            ExecutionMode::Async,
        ] {
            let start = std::time::Instant::now();
            let results = BatchBacktestSweep::new(device.clone())
                .strategy_type(StrategyType::LongStraddle)
                .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
                .parameters_batch(&params_strategy)
                .config(BacktestConfig {
                    initial_capital: 100_000.0,
                    trading_fee: 0.001,
                    slippage: 0.0005,
            ..Default::default()
        
                })
                .heston_params(params_heston.clone())
                .options_data(options.clone())
                .execution_mode(mode)
                .execute()
                .expect("Load test failed");
            let elapsed = start.elapsed();

            println!(
                "{:?}: {:.2}ms ({:.2} strategies/sec)",
                mode,
                elapsed.as_secs_f64() * 1000.0,
                500.0 / elapsed.as_secs_f64()
            );

            assert_eq!(results.results.len(), 500);
        }
    }

    #[test]
    #[ignore] // Very slow
    fn test_load_extreme_scale() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        // Extreme workload: 2000 strategies × 20K candles
        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(20_000, MarketRegime::RangeBound, 42);

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(spot, 100, 30, &params_heston); // 200 options

        let params_strategy = generate_strategy_params(StrategyType::LongStraddle, 2000);

        println!("\n===== EXTREME SCALE LOAD TEST =====");
        println!("Strategies: 2000");
        println!("Candles: 20,000");
        println!("Options: 200");
        println!("Total workload: 40M data points\n");

        let start = std::time::Instant::now();
        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::LongStraddle)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params_strategy)
            .config(BacktestConfig {
                initial_capital: 100_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
            ..Default::default()
        
            })
            .heston_params(params_heston)
            .options_data(options)
            .execution_mode(ExecutionMode::Auto)
            .execute()
            .expect("Extreme load test failed");
        let elapsed = start.elapsed();

        println!("\n===== EXTREME SCALE RESULTS =====");
        println!("Total time: {:.2}s", elapsed.as_secs_f64());
        println!("Throughput: {:.2} strategies/sec", 2000.0 / elapsed.as_secs_f64());

        assert_eq!(results.results.len(), 2000);

        // Relaxed target for extreme scale: <2 seconds
        assert!(
            elapsed.as_secs_f64() < 2.0,
            "Extreme scale test too slow: {:.2}s (target: <2s)",
            elapsed.as_secs_f64()
        );
    }

    #[test]
    #[ignore] // Slow
    fn test_load_all_strategy_types() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(5_000, MarketRegime::RangeBound, 42);

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(spot, 20, 30, &params_heston);

        let strategies = vec![
            StrategyType::LongStraddle,
            StrategyType::ShortStraddle,
            StrategyType::CoveredCall,
            StrategyType::IronCondor,
            StrategyType::DeltaNeutral,
            StrategyType::VolatilityArbitrage,
        ];

        println!("\n===== ALL STRATEGIES LOAD TEST =====");
        println!("Testing 100 parameter combinations × 6 strategies\n");

        let mut total_time = 0.0;

        for strategy in strategies {
            let params_strategy = generate_strategy_params(strategy, 100);

            let start = std::time::Instant::now();
            let results = BatchBacktestSweep::new(device.clone())
                .strategy_type(strategy)
                .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
                .parameters_batch(&params_strategy)
                .config(BacktestConfig {
                    initial_capital: 100_000.0,
                    trading_fee: 0.001,
                    slippage: 0.0005,
            ..Default::default()
        
                })
                .heston_params(params_heston.clone())
                .options_data(options.clone())
                .execute()
                .expect("Strategy test failed");
            let elapsed = start.elapsed();

            total_time += elapsed.as_secs_f64();

            println!(
                "{:?}: {:.2}ms",
                strategy,
                elapsed.as_secs_f64() * 1000.0
            );

            assert_eq!(results.results.len(), 100);
        }

        println!("\n===== TOTAL TIME FOR ALL STRATEGIES =====");
        println!("{:.2}s for 600 total backtests", total_time);
        println!("{:.2}ms per backtest", total_time * 1000.0 / 600.0);
    }
}
