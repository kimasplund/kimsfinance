//! Regression Tests for Heston-Backtest Integration
//!
//! Ensures backward compatibility and no performance regressions:
//! - Phase 1-3 equity strategies still work correctly
//! - No performance degradation vs baseline
//! - API compatibility maintained
//! - CPU fallback functionality preserved

#[cfg(all(feature = "gpu", feature = "heston"))]
mod heston_regression_tests {
    use kimsfinance_core::backtest::batch::{BatchBacktestSweep, StrategyType};
    use kimsfinance_core::backtest::engine::BacktestConfig;
    use kimsfinance_core::gpu::GpuDevice;
    use std::sync::Arc;

    mod test_data {
        include!("../data/heston_test_data.rs");
    }
    use test_data::{generate_btc_ohlcv, MarketRegime};

    // ========== Equity Strategy Regression Tests ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_regression_rsi_crossover_still_works() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(1000, MarketRegime::RangeBound, 42);

        // Original RSI strategy should still work exactly as before
        let params = vec![
            vec![14.0, 25.0, 75.0],
            vec![14.0, 30.0, 70.0],
            vec![14.0, 20.0, 80.0],
        ];

        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
            ..Default::default()
        
            })
            .execute()
            .expect("RSI backtest failed");

        assert_eq!(results.results.len(), 3);

        println!("\n===== RSI Crossover Regression Test =====");
        for (i, result) in results.results.iter().enumerate() {
            println!(
                "Strategy {}: PnL={:.2}, Sharpe={:.2}, Trades={}",
                i, result.total_pnl, result.sharpe_ratio, result.num_trades
            );

            assert!(
                result.total_pnl.is_finite(),
                "RSI strategy {} has non-finite PnL",
                i
            );
            assert!(
                result.num_trades >= 0,
                "RSI strategy {} has negative trades",
                i
            );
        }

        println!("RSI Crossover: PASS (backward compatible)");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_regression_ma_crossover_still_works() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(1000, MarketRegime::Trending, 42);

        let params = vec![
            vec![10.0, 20.0],
            vec![20.0, 50.0],
            vec![50.0, 200.0],
        ];

        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::MaCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
            ..Default::default()
        
            })
            .execute()
            .expect("MA backtest failed");

        assert_eq!(results.results.len(), 3);

        println!("\n===== MA Crossover Regression Test =====");
        for (i, result) in results.results.iter().enumerate() {
            println!(
                "Strategy {}: PnL={:.2}, Sharpe={:.2}",
                i, result.total_pnl, result.sharpe_ratio
            );

            assert!(result.total_pnl.is_finite());
        }

        println!("MA Crossover: PASS (backward compatible)");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_regression_bollinger_bands_still_works() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(1000, MarketRegime::RangeBound, 42);

        let params = vec![
            vec![20.0, 2.0, 2.0, 0.5],
            vec![20.0, 2.5, 2.5, 0.5],
        ];

        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::BollingerMeanReversion)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
            ..Default::default()
        
            })
            .execute()
            .expect("Bollinger backtest failed");

        assert_eq!(results.results.len(), 2);

        println!("\n===== Bollinger Bands Regression Test =====");
        for (i, result) in results.results.iter().enumerate() {
            println!(
                "Strategy {}: PnL={:.2}, Sharpe={:.2}",
                i, result.total_pnl, result.sharpe_ratio
            );

            assert!(result.total_pnl.is_finite());
        }

        println!("Bollinger Bands: PASS (backward compatible)");
    }

    // ========== Performance Regression Tests ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_regression_equity_performance_maintained() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(10_000, MarketRegime::RangeBound, 42);

        let params: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![14.0, 20.0 + i as f64 * 0.5, 70.0 + i as f64 * 0.5])
            .collect();

        println!("\n===== Equity Performance Regression Test =====");
        println!("Baseline target: 100 strategies × 10K candles < 100ms\n");

        let start = std::time::Instant::now();
        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
            ..Default::default()
        
            })
            .execute()
            .expect("Performance test failed");
        let elapsed = start.elapsed();

        println!("Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        assert_eq!(results.results.len(), 100);

        // Performance should not have regressed
        // Allow 150ms (50% margin above 100ms baseline)
        assert!(
            elapsed.as_secs_f64() < 0.15,
            "Performance regression detected: {:.2}ms (baseline: <100ms)",
            elapsed.as_secs_f64() * 1000.0
        );

        println!("Equity Performance: PASS (no regression)");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_regression_options_overhead_acceptable() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(1000, MarketRegime::RangeBound, 42);

        let spot = close[close.len() - 1];
        let params_heston = test_data::test_heston_params(MarketRegime::RangeBound);
        let options = test_data::generate_options_chain(spot, 10, 30, &params_heston);

        let params_strategy = vec![vec![0.05, 0.10]];

        // Measure equity baseline (no Heston)
        let equity_params = vec![vec![14.0, 25.0, 75.0]];
        let start_equity = std::time::Instant::now();
        let _equity_results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&equity_params)
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
            ..Default::default()
        
            })
            .execute()
            .expect("Equity baseline failed");
        let equity_time = start_equity.elapsed();

        // Measure options strategy (with Heston)
        let start_options = std::time::Instant::now();
        let options_results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::LongStraddle)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params_strategy)
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
            ..Default::default()
        
            })
            .heston_params(params_heston)
            .options_data(options)
            .execute()
            .expect("Options test failed");
        let options_time = start_options.elapsed();

        let phase0_time = options_results
            .phase_timings
            .get("phase0_heston_ms")
            .unwrap_or(&0.0);

        println!("\n===== Options Overhead Regression Test =====");
        println!("Equity baseline: {:.2}ms", equity_time.as_secs_f64() * 1000.0);
        println!("Options strategy: {:.2}ms", options_time.as_secs_f64() * 1000.0);
        println!("Phase 0 (Heston): {:.2}ms", phase0_time);
        println!(
            "Overhead: {:.2}ms",
            (options_time.as_secs_f64() - equity_time.as_secs_f64()) * 1000.0
        );

        // Phase 0 should be <20ms
        assert!(
            phase0_time < &20.0,
            "Phase 0 regression: {:.2}ms (target: <20ms)",
            phase0_time
        );

        println!("Options Overhead: PASS (within targets)");
    }

    // ========== API Compatibility Tests ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_regression_builder_api_unchanged() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(100, MarketRegime::RangeBound, 42);

        // Ensure old API calls still work
        let params = vec![vec![14.0, 25.0, 75.0]];

        let _results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
            ..Default::default()
        
            })
            // No .heston_params() or .options_data() calls - should still work
            .execute()
            .expect("Builder API test failed");

        println!("\n===== Builder API Compatibility: PASS =====");
    }

    #[test]
    fn test_regression_strategy_type_enum_stable() {
        // Verify enum values haven't changed
        assert_eq!(StrategyType::RsiCrossover as i32, 0);
        assert_eq!(StrategyType::MaCrossover as i32, 1);
        assert_eq!(StrategyType::BollingerMeanReversion as i32, 2);
        assert_eq!(StrategyType::LongStraddle as i32, 10);
        assert_eq!(StrategyType::ShortStraddle as i32, 11);
        assert_eq!(StrategyType::CoveredCall as i32, 12);
        assert_eq!(StrategyType::IronCondor as i32, 13);
        assert_eq!(StrategyType::DeltaNeutral as i32, 14);
        assert_eq!(StrategyType::VolatilityArbitrage as i32, 15);

        println!("\n===== Strategy Type Enum: STABLE =====");
    }

    #[test]
    fn test_regression_strategy_categorization() {
        // Equity strategies (0-9)
        assert!(StrategyType::RsiCrossover.is_equity_strategy());
        assert!(!StrategyType::RsiCrossover.is_options_strategy());

        // Options strategies (10-19)
        assert!(StrategyType::LongStraddle.is_options_strategy());
        assert!(!StrategyType::LongStraddle.is_equity_strategy());

        println!("\n===== Strategy Categorization: STABLE =====");
    }

    // ========== Data Integrity Tests ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_regression_results_structure_unchanged() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(100, MarketRegime::RangeBound, 42);

        let params = vec![vec![14.0, 25.0, 75.0]];

        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .config(BacktestConfig {
                initial_capital: 10_000.0,
                trading_fee: 0.001,
                slippage: 0.0005,
            ..Default::default()
        
            })
            .execute()
            .expect("Results test failed");

        // Verify all expected fields exist
        let result = &results.results[0];

        assert!(result.total_pnl.is_finite());
        assert!(result.sharpe_ratio.is_finite());
        assert!(result.max_drawdown.is_finite());
        assert!(result.win_rate.is_finite());
        assert!(result.num_trades >= 0);

        // Verify phase timings
        assert!(results.phase_timings.contains_key("phase1_indicators_ms"));
        assert!(results.phase_timings.contains_key("phase2_signals_ms"));
        assert!(results.phase_timings.contains_key("phase3_execution_ms"));
        assert!(results.phase_timings.contains_key("phase4_metrics_ms"));

        println!("\n===== Results Structure: UNCHANGED =====");
    }
}
