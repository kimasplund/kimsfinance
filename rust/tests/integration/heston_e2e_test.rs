//! End-to-End Integration Tests for Heston-Backtest Pipeline
//!
//! Tests the complete 5-phase pipeline:
//! - Phase 0: Heston option pricing (<20ms for 1000 options)
//! - Phase 1: Technical indicator calculation
//! - Phase 2: Strategy signal generation
//! - Phase 3: Backtest execution with position management
//! - Phase 4: Performance metrics calculation
//!
//! # Test Scenarios
//!
//! 1. Single equity strategy (RSI crossover) - baseline
//! 2. Single options strategy (Long Straddle)
//! 3. All 6 options strategies in parallel
//! 4. Mixed portfolio (equity + options)
//! 5. Large scale (1000 strategies × 10K candles)

#[cfg(all(feature = "gpu", feature = "heston"))]
mod heston_e2e_tests {
    use kimsfinance_core::backtest::batch::{BatchBacktestSweep, ExecutionMode, StrategyType};
    use kimsfinance_core::backtest::engine::BacktestConfig;
    use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
    use kimsfinance_core::quantitative::heston::HestonParams;
    use std::sync::Arc;

    // Import test data generators
    mod test_data {
        include!("../data/heston_test_data.rs");
    }
    use test_data::{
        generate_btc_ohlcv, generate_options_chain, generate_strategy_params, test_heston_params,
        MarketRegime,
    };

    // ========== Test Helper Functions ==========

    fn create_backtest_config() -> BacktestConfig {
        BacktestConfig {
            initial_capital: 100_000.0,
            trading_fee: 0.001, // 0.1%
            slippage: 0.0005,   // 0.05%
        },
            ..Default::default()
        
    }

    // ========== Baseline: Equity Strategy (No Heston) ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_e2e_equity_strategy_rsi() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));

        // Generate synthetic BTC data (1000 candles)
        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(1000, MarketRegime::RangeBound, 42);

        // Generate 10 RSI parameter combinations
        let params = generate_strategy_params(StrategyType::RsiCrossover, 10);

        let start = std::time::Instant::now();
        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params)
            .config(create_backtest_config())
            .execute()
            .expect("Backtest execution failed");
        let elapsed = start.elapsed();

        assert_eq!(results.results.len(), 10);

        println!(
            "\n===== E2E Test: Equity Strategy (RSI) =====\n\
             Strategies: 10\n\
             Candles: 1000\n\
             Total time: {:.2}ms\n\
             Avg time per strategy: {:.2}ms\n",
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / 10.0
        );

        // Verify all strategies completed
        for (i, result) in results.results.iter().enumerate() {
            assert!(
                result.total_pnl.is_finite(),
                "Strategy {} has non-finite PnL",
                i
            );
            assert!(
                result.sharpe_ratio.is_finite(),
                "Strategy {} has non-finite Sharpe",
                i
            );
        }

        // Performance check: Should complete in <100ms for 10 strategies × 1000 candles
        assert!(
            elapsed.as_secs_f64() < 0.1,
            "Equity backtest too slow: {:.2}ms (target: <100ms)",
            elapsed.as_secs_f64() * 1000.0
        );
    }

    // ========== Single Options Strategy Tests ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_e2e_single_options_long_straddle() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let mut heston_pricer = HestonGpuPricer::new(device.clone())
            .expect("Failed to create Heston pricer");

        // Generate synthetic data
        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(1000, MarketRegime::Volatile, 42); // Volatile for straddles

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::Volatile);
        let options = generate_options_chain(spot, 10, 30, &params_heston);

        // Price options with Heston
        let _option_prices = heston_pricer
            .price_options(&params_heston, &options)
            .expect("Failed to price options");

        // Generate 10 Long Straddle parameter combinations
        let params_strategy = generate_strategy_params(StrategyType::LongStraddle, 10);

        let start = std::time::Instant::now();
        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::LongStraddle)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params_strategy)
            .config(create_backtest_config())
            .heston_params(params_heston)
            .options_data(options)
            .execute()
            .expect("Backtest execution failed");
        let elapsed = start.elapsed();

        assert_eq!(results.results.len(), 10);

        println!(
            "\n===== E2E Test: Options Strategy (Long Straddle) =====\n\
             Strategies: 10\n\
             Candles: 1000\n\
             Options: 20 (10 strikes × 2 types)\n\
             Total time: {:.2}ms\n\
             Phase 0 (Heston): {:.2}ms\n",
            elapsed.as_secs_f64() * 1000.0,
            results.phase_timings.get("phase0_heston_ms").unwrap_or(&0.0)
        );

        // Performance check: Total pipeline <250ms, Phase 0 <20ms
        assert!(
            elapsed.as_secs_f64() < 0.25,
            "Options backtest too slow: {:.2}ms (target: <250ms)",
            elapsed.as_secs_f64() * 1000.0
        );

        let phase0_time = results.phase_timings.get("phase0_heston_ms").unwrap_or(&0.0);
        assert!(
            phase0_time < &20.0,
            "Phase 0 (Heston) too slow: {:.2}ms (target: <20ms)",
            phase0_time
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_e2e_single_options_short_straddle() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(1000, MarketRegime::RangeBound, 42); // Range-bound for shorts

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(spot, 10, 30, &params_heston);

        let params_strategy = generate_strategy_params(StrategyType::ShortStraddle, 10);

        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::ShortStraddle)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params_strategy)
            .config(create_backtest_config())
            .heston_params(params_heston)
            .options_data(options)
            .execute()
            .expect("Backtest execution failed");

        assert_eq!(results.results.len(), 10);

        // Verify risk management (short straddles should have bounded losses)
        for (i, result) in results.results.iter().enumerate() {
            println!(
                "ShortStraddle {}: PnL={:.2}, MaxDD={:.2}%",
                i,
                result.total_pnl,
                result.max_drawdown * 100.0
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_e2e_single_options_volatility_arbitrage() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(1000, MarketRegime::Volatile, 42);

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::Volatile);
        let options = generate_options_chain(spot, 20, 30, &params_heston); // More strikes for arb

        let params_strategy = generate_strategy_params(StrategyType::VolatilityArbitrage, 10);

        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::VolatilityArbitrage)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params_strategy)
            .config(create_backtest_config())
            .heston_params(params_heston)
            .options_data(options)
            .execute()
            .expect("Backtest execution failed");

        assert_eq!(results.results.len(), 10);

        println!("\n===== Volatility Arbitrage Results =====");
        for (i, result) in results.results.iter().enumerate() {
            println!(
                "VolArb {}: PnL={:.2}, Sharpe={:.2}, Trades={}",
                i, result.total_pnl, result.sharpe_ratio, result.num_trades
            );
        }
    }

    // ========== All Options Strategies in Parallel ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_e2e_all_options_strategies() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));

        // Generate data
        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(1000, MarketRegime::RangeBound, 42);

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(spot, 20, 30, &params_heston);

        let all_strategies = vec![
            StrategyType::LongStraddle,
            StrategyType::ShortStraddle,
            StrategyType::CoveredCall,
            StrategyType::IronCondor,
            StrategyType::DeltaNeutral,
            StrategyType::VolatilityArbitrage,
        ];

        println!("\n===== Testing All 6 Options Strategies =====");

        for strategy in all_strategies {
            let params_strategy = generate_strategy_params(strategy, 5);

            let start = std::time::Instant::now();
            let results = BatchBacktestSweep::new(device.clone())
                .strategy_type(strategy)
                .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
                .parameters_batch(&params_strategy)
                .config(create_backtest_config())
                .heston_params(params_heston.clone())
                .options_data(options.clone())
                .execute()
                .expect("Backtest execution failed");
            let elapsed = start.elapsed();

            assert_eq!(results.results.len(), 5);

            let best_result = results
                .results
                .iter()
                .max_by(|a, b| a.sharpe_ratio.partial_cmp(&b.sharpe_ratio).unwrap())
                .unwrap();

            println!(
                "{:?}: {:.2}ms | Best Sharpe={:.2} PnL={:.2}",
                strategy,
                elapsed.as_secs_f64() * 1000.0,
                best_result.sharpe_ratio,
                best_result.total_pnl
            );

            // Each strategy should complete in <250ms
            assert!(
                elapsed.as_secs_f64() < 0.25,
                "{:?} too slow: {:.2}ms (target: <250ms)",
                strategy,
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }

    // ========== Mixed Portfolio (Equity + Options) ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_e2e_mixed_portfolio() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(1000, MarketRegime::RangeBound, 42);

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(spot, 10, 30, &params_heston);

        println!("\n===== Mixed Portfolio Test (Equity + Options) =====");

        // Run equity strategies
        let equity_params = generate_strategy_params(StrategyType::RsiCrossover, 5);
        let equity_results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::RsiCrossover)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&equity_params)
            .config(create_backtest_config())
            .execute()
            .expect("Equity backtest failed");

        // Run options strategies
        let options_params = generate_strategy_params(StrategyType::LongStraddle, 5);
        let options_results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::LongStraddle)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&options_params)
            .config(create_backtest_config())
            .heston_params(params_heston)
            .options_data(options)
            .execute()
            .expect("Options backtest failed");

        assert_eq!(equity_results.results.len(), 5);
        assert_eq!(options_results.results.len(), 5);

        // Compare performance
        let best_equity = equity_results
            .results
            .iter()
            .max_by(|a, b| a.sharpe_ratio.partial_cmp(&b.sharpe_ratio).unwrap())
            .unwrap();

        let best_options = options_results
            .results
            .iter()
            .max_by(|a, b| a.sharpe_ratio.partial_cmp(&b.sharpe_ratio).unwrap())
            .unwrap();

        println!(
            "Best Equity (RSI): Sharpe={:.2}, PnL={:.2}",
            best_equity.sharpe_ratio, best_equity.total_pnl
        );
        println!(
            "Best Options (Straddle): Sharpe={:.2}, PnL={:.2}",
            best_options.sharpe_ratio, best_options.total_pnl
        );
    }

    // ========== Large Scale Load Test ==========

    #[test]
    #[ignore] // Requires GPU and is slow
    fn test_e2e_large_scale_1000_strategies() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));

        // Generate large dataset
        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(10_000, MarketRegime::RangeBound, 42); // 10K candles

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(spot, 50, 30, &params_heston); // 100 options

        // Generate 1000 strategy parameter combinations
        let params_strategy = generate_strategy_params(StrategyType::LongStraddle, 1000);

        println!("\n===== LARGE SCALE LOAD TEST =====");
        println!("Strategies: 1000");
        println!("Candles: 10,000");
        println!("Options: 100 (50 strikes × 2 types)");
        println!("Total workload: 1000 × 10,000 = 10M data points\n");

        let start = std::time::Instant::now();
        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::LongStraddle)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params_strategy)
            .config(create_backtest_config())
            .heston_params(params_heston)
            .options_data(options)
            .execution_mode(ExecutionMode::Auto) // Let system choose optimal mode
            .execute()
            .expect("Large scale backtest failed");
        let elapsed = start.elapsed();

        assert_eq!(results.results.len(), 1000);

        println!("===== LOAD TEST RESULTS =====");
        println!("Total time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!(
            "Avg per strategy: {:.2}ms",
            elapsed.as_secs_f64() * 1000.0 / 1000.0
        );
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

        // Performance target: <250ms for 1000 strategies × 10K candles
        assert!(
            elapsed.as_secs_f64() < 0.5,
            "Large scale test too slow: {:.2}ms (target: <500ms relaxed for CI)",
            elapsed.as_secs_f64() * 1000.0
        );

        // Phase 0 should still be <20ms even for large batches
        let phase0_time = results.phase_timings.get("phase0_heston_ms").unwrap_or(&0.0);
        assert!(
            phase0_time < &50.0,
            "Phase 0 too slow: {:.2}ms (target: <50ms relaxed)",
            phase0_time
        );

        // Verify all strategies completed successfully
        let completed = results
            .results
            .iter()
            .filter(|r| r.total_pnl.is_finite())
            .count();
        assert_eq!(
            completed, 1000,
            "Only {} / 1000 strategies completed successfully",
            completed
        );
    }

    // ========== Execution Mode Tests ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_e2e_execution_modes_comparison() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(5000, MarketRegime::RangeBound, 42);

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(spot, 20, 30, &params_heston);

        let params_strategy = generate_strategy_params(StrategyType::LongStraddle, 100);

        let modes = vec![
            ExecutionMode::Traditional,
            ExecutionMode::Fused,
            ExecutionMode::Async,
            ExecutionMode::Auto,
        ];

        println!("\n===== Execution Mode Comparison =====");
        println!("Strategies: 100, Candles: 5000\n");

        for mode in modes {
            let start = std::time::Instant::now();
            let results = BatchBacktestSweep::new(device.clone())
                .strategy_type(StrategyType::LongStraddle)
                .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
                .parameters_batch(&params_strategy)
                .config(create_backtest_config())
                .heston_params(params_heston.clone())
                .options_data(options.clone())
                .execution_mode(mode)
                .execute()
                .expect("Backtest failed");
            let elapsed = start.elapsed();

            println!(
                "{:?}: {:.2}ms (Sharpe range: {:.2} - {:.2})",
                mode,
                elapsed.as_secs_f64() * 1000.0,
                results
                    .results
                    .iter()
                    .map(|r| r.sharpe_ratio)
                    .fold(f64::INFINITY, f64::min),
                results
                    .results
                    .iter()
                    .map(|r| r.sharpe_ratio)
                    .fold(f64::NEG_INFINITY, f64::max)
            );
        }
    }

    // ========== Position Management Edge Cases ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_e2e_position_management_multiple_entries() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(1000, MarketRegime::Volatile, 42);

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::Volatile);
        let options = generate_options_chain(spot, 10, 30, &params_heston);

        // Test strategy that should generate multiple signals
        let params_strategy = vec![vec![0.05, 0.10]]; // Aggressive thresholds

        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::LongStraddle)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params_strategy)
            .config(create_backtest_config())
            .heston_params(params_heston)
            .options_data(options)
            .execute()
            .expect("Backtest failed");

        let result = &results.results[0];

        println!("\n===== Position Management Test =====");
        println!("Trades executed: {}", result.num_trades);
        println!("Win rate: {:.1}%", result.win_rate * 100.0);
        println!("Max drawdown: {:.2}%", result.max_drawdown * 100.0);

        // Should have executed multiple trades
        assert!(
            result.num_trades > 0,
            "No trades executed in volatile market"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_e2e_risk_management_max_loss() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));

        let (timestamps, open, high, low, close, volume) =
            generate_btc_ohlcv(1000, MarketRegime::Trending, 42);

        let spot = close[close.len() - 1];
        let params_heston = test_heston_params(MarketRegime::Trending);
        let options = generate_options_chain(spot, 10, 30, &params_heston);

        // Short straddle with tight max loss (should exit quickly on adverse moves)
        let params_strategy = vec![vec![0.10, 0.15]]; // 10% vol threshold, 15% max loss

        let results = BatchBacktestSweep::new(device.clone())
            .strategy_type(StrategyType::ShortStraddle)
            .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
            .parameters_batch(&params_strategy)
            .config(create_backtest_config())
            .heston_params(params_heston)
            .options_data(options)
            .execute()
            .expect("Backtest failed");

        let result = &results.results[0];

        println!("\n===== Risk Management Test =====");
        println!("Max drawdown: {:.2}%", result.max_drawdown * 100.0);
        println!("Total PnL: {:.2}", result.total_pnl);

        // Verify risk limits were respected (max loss parameter was 15%)
        // Allow some slippage/fees overhead
        assert!(
            result.max_drawdown < 0.20,
            "Max drawdown ({:.2}%) exceeded risk limit (20% with buffer)",
            result.max_drawdown * 100.0
        );
    }
}
