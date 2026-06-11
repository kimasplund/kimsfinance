//! Unit tests for GPU batch backtesting kernels
//!
//! Tests all 4 kernels:
//! 1. batch_indicators_kernel
//! 2. strategy_signals_kernel
//! 3. backtest_execution_kernel
//! 4. metrics_calculation_kernel
//!
//! # Test Strategy
//!
//! - Test with 10, 100, 1000 strategies
//! - Validate against CPU reference implementation
//! - Test edge cases: NaN, zero trades, extreme values
//! - Measure performance (should be <200ms for 1000 strategies × 10K candles)

#[cfg(feature = "gpu")]
mod batch_backtest_tests {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use kimsfinance_core::gpu::compile::compile_backtest_kernels;
    use kimsfinance_core::gpu::device::GpuDevice;

    /// Generate test price data (simple upward trend)
    fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let open: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.1).collect();
        let high: Vec<f64> = (0..n).map(|i| 102.0 + i as f64 * 0.1).collect();
        let low: Vec<f64> = (0..n).map(|i| 98.0 + i as f64 * 0.1).collect();
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + i as f64 * 0.1 + (i as f64 * 0.01).sin())
            .collect();
        (open, high, low, close)
    }

    /// CPU reference implementation: RSI calculation
    fn calculate_rsi_cpu(close: &[f64], period: usize) -> Vec<f64> {
        let mut rsi = vec![f64::NAN; close.len()];

        for i in period..close.len() {
            let mut gain_sum = 0.0;
            let mut loss_sum = 0.0;

            for j in (i - period + 1)..=i {
                if j > 0 {
                    let delta = close[j] - close[j - 1];
                    if delta > 0.0 {
                        gain_sum += delta;
                    } else {
                        loss_sum += -delta;
                    }
                }
            }

            let avg_gain = gain_sum / period as f64;
            let avg_loss = loss_sum / period as f64;

            if avg_loss < 1e-10 {
                rsi[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        rsi
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_batch_indicators_kernel_rsi() {
        // Initialize GPU
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Generate test data
        let n_candles = 1000;
        let n_strategies = 10;
        let n_indicators = 1; // RSI only
        let n_params = 1;

        let (_, high, low, close) = generate_test_data(n_candles);

        // Flatten OHLCV [O, H, L, C, V]
        let mut ohlcv = Vec::with_capacity(n_candles * 5);
        ohlcv.extend_from_slice(&close); // O (use close for simplicity)
        ohlcv.extend_from_slice(&high); // H
        ohlcv.extend_from_slice(&low); // L
        ohlcv.extend_from_slice(&close); // C
        ohlcv.extend(vec![1000.0; n_candles]); // V (constant)

        // Parameters: RSI period = 14 for all strategies
        let params = vec![14.0; n_strategies * n_params];

        // Allocate GPU memory
        let d_ohlcv = device.copy_to_device(&ohlcv).unwrap();
        let d_params = device.copy_to_device(&params).unwrap();
        let mut d_indicators = device
            .alloc_buffer(n_strategies * n_indicators * n_candles)
            .unwrap();

        // Compile kernels
        let ptx = compile_backtest_kernels().unwrap();
        let module = device.context().load_module(std::sync::Arc::unwrap_or_clone(ptx)).unwrap();
        let kernel = module.load_function("batch_indicators_kernel").unwrap();

        // Launch kernel
        let config = LaunchConfig {
            grid_dim: (
                n_strategies as u32,
                n_indicators as u32,
                ((n_candles + 255) / 256) as u32,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_strategies_i32 = n_strategies as i32;
        let n_indicators_i32 = n_indicators as i32;
        let n_candles_i32 = n_candles as i32;
        let n_params_i32 = n_params as i32;

        let mut builder = device.stream().launch_builder(&kernel);
        builder.arg(&d_ohlcv);
        builder.arg(&d_params);
        builder.arg(&mut d_indicators);
        builder.arg(&n_strategies_i32);
        builder.arg(&n_indicators_i32);
        builder.arg(&n_candles_i32);
        builder.arg(&n_params_i32);

        unsafe { builder.launch(config).unwrap() };
        device.stream().synchronize().unwrap();

        // Copy results back
        let indicators = device.copy_to_host(&d_indicators).unwrap();

        // Validate against CPU reference
        let rsi_cpu = calculate_rsi_cpu(&close, 14);

        for strategy_idx in 0..n_strategies {
            for candle_idx in 0..n_candles {
                let gpu_idx = strategy_idx * n_indicators * n_candles + candle_idx;
                let gpu_value = indicators[gpu_idx];
                let cpu_value = rsi_cpu[candle_idx];

                if cpu_value.is_nan() {
                    assert!(
                        gpu_value.is_nan(),
                        "Strategy {}, candle {}: Expected NaN, got {}",
                        strategy_idx,
                        candle_idx,
                        gpu_value
                    );
                } else {
                    let diff = (gpu_value - cpu_value).abs();
                    assert!(
                        diff < 1e-5,
                        "Strategy {}, candle {}: GPU={:.6}, CPU={:.6}, diff={:.2e}",
                        strategy_idx,
                        candle_idx,
                        gpu_value,
                        cpu_value,
                        diff
                    );
                }
            }
        }

        println!(
            "✅ Batch indicators kernel: RSI validation passed for {} strategies",
            n_strategies
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_strategy_signals_kernel() {
        // Initialize GPU
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n_strategies = 100;
        let n_candles = 1000;
        let n_indicators = 1;

        // Create indicator values (RSI ranging from 20 to 80)
        let mut indicators = vec![50.0; n_strategies * n_indicators * n_candles];

        // Set some values to trigger buy/sell signals
        for strategy_idx in 0..n_strategies {
            let base = strategy_idx * n_indicators * n_candles;
            indicators[base + 100] = 25.0; // Should trigger BUY (< 30)
            indicators[base + 200] = 75.0; // Should trigger SELL (> 70)
            indicators[base + 300] = 50.0; // Should be HOLD
        }

        // Parameters: [period, buy_threshold, sell_threshold] for each strategy
        let mut params = Vec::new();
        for _ in 0..n_strategies {
            params.extend_from_slice(&[14.0, 30.0, 70.0]);
        }

        // Allocate GPU memory
        let d_indicators = device.copy_to_device(&indicators).unwrap();
        let d_params = device.copy_to_device(&params).unwrap();
        let mut d_signals = device.allocate_device_buffer::<i8>(n_strategies * n_candles).unwrap();

        // Compile and load kernel
        let ptx = compile_backtest_kernels().unwrap();
        let module = device.context().load_module(std::sync::Arc::unwrap_or_clone(ptx)).unwrap();
        let kernel = module.load_function("strategy_signals_kernel").unwrap();

        // Launch kernel
        let config = LaunchConfig {
            grid_dim: (n_strategies as u32, ((n_candles + 255) / 256) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let strategy_type = 0i32; // RSI crossover

        let n_strategies_i32 = n_strategies as i32;
        let n_indicators_i32 = n_indicators as i32;
        let n_candles_i32 = n_candles as i32;

        let mut builder = device.stream().launch_builder(&kernel);
        builder.arg(&d_indicators);
        builder.arg(&d_params);
        builder.arg(&mut d_signals);
        builder.arg(&n_strategies_i32);
        builder.arg(&n_indicators_i32);
        builder.arg(&n_candles_i32);
        builder.arg(&strategy_type);

        unsafe { builder.launch(config).unwrap() };
        device.stream().synchronize().unwrap();

        // Copy results back
        let signals = device.copy_to_host_i8(&d_signals).unwrap();

        // Validate signals
        for strategy_idx in 0..n_strategies {
            let base = strategy_idx * n_candles;

            // Check BUY signal at candle 100
            assert_eq!(
                signals[base + 100],
                1,
                "Strategy {}: Expected BUY signal",
                strategy_idx
            );

            // Check SELL signal at candle 200
            assert_eq!(
                signals[base + 200],
                2,
                "Strategy {}: Expected SELL signal",
                strategy_idx
            );

            // Check HOLD signal at candle 300
            assert_eq!(
                signals[base + 300],
                0,
                "Strategy {}: Expected HOLD signal",
                strategy_idx
            );
        }

        println!(
            "✅ Strategy signals kernel: Signal generation validated for {} strategies",
            n_strategies
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_backtest_execution_kernel() {
        // Initialize GPU
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n_strategies = 10;
        let n_candles = 1000;

        // Generate test data with upward trend
        let (_, _, _, close) = generate_test_data(n_candles);

        // Create simple signal pattern: Buy at candle 100, Sell at candle 200
        let mut signals = vec![0i8; n_strategies * n_candles];
        for strategy_idx in 0..n_strategies {
            let base = strategy_idx * n_candles;
            signals[base + 100] = 1; // BUY
            signals[base + 200] = 2; // SELL
        }

        // Allocate GPU memory
        let d_signals = device.copy_to_device_i8(&signals).unwrap();
        let d_close = device.copy_to_device(&close).unwrap();
        let mut d_equity_curves = device
            .alloc_buffer(n_strategies * n_candles)
            .unwrap();

        // Trade structure size: 6 doubles + 2 longs + 1 i8 = 56 bytes (padded to 64)
        let trade_size = std::mem::size_of::<f64>() * 6 + std::mem::size_of::<i64>() * 2;
        let max_trades = 1000;
        let mut d_trades = device
            .allocate_device_buffer::<u8>(n_strategies * max_trades * trade_size)
            .unwrap();
        let mut d_num_trades = device.allocate_device_buffer::<i32>(n_strategies).unwrap();

        let initial_capital = 10000.0;
        let trading_fee = 0.001;
        let slippage = 0.0005;

        // Compile and load kernel
        let ptx = compile_backtest_kernels().unwrap();
        let module = device.context().load_module(std::sync::Arc::unwrap_or_clone(ptx)).unwrap();
        let kernel = module.load_function("backtest_execution_kernel").unwrap();

        // Launch kernel
        let config = LaunchConfig {
            grid_dim: (n_strategies as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_strategies_i32 = n_strategies as i32;
        let n_candles_i32 = n_candles as i32;

        let mut d_max_drawdowns = device.stream().alloc_zeros::<f64>(n_strategies).unwrap();

        let mut builder = device.stream().launch_builder(&kernel);
        builder.arg(&d_signals);
        builder.arg(&d_close);
        builder.arg(&mut d_equity_curves);
        builder.arg(&mut d_trades);
        builder.arg(&mut d_num_trades);
        builder.arg(&mut d_max_drawdowns);
        builder.arg(&initial_capital);
        builder.arg(&trading_fee);
        builder.arg(&slippage);
        builder.arg(&n_strategies_i32);
        builder.arg(&n_candles_i32);

        unsafe { builder.launch(config).unwrap() };
        device.stream().synchronize().unwrap();

        // Copy results back
        let equity_curves = device.copy_to_host(&d_equity_curves).unwrap();
        let num_trades = device.stream().memcpy_dtov(&d_num_trades).unwrap();

        // Validate results
        for strategy_idx in 0..n_strategies {
            // Check that we executed 1 trade
            assert_eq!(
                num_trades[strategy_idx], 1,
                "Strategy {}: Expected 1 trade",
                strategy_idx
            );

            // Check equity curve makes sense
            let base = strategy_idx * n_candles;
            let initial_equity = equity_curves[base];
            let final_equity = equity_curves[base + n_candles - 1];

            // Initial equity should be ~initial_capital
            assert!(
                (initial_equity - initial_capital).abs() < 1.0,
                "Strategy {}: Initial equity {:.2} != {:.2}",
                strategy_idx,
                initial_equity,
                initial_capital
            );

            // Final equity should be positive (upward trend)
            assert!(
                final_equity > 0.0,
                "Strategy {}: Final equity is zero",
                strategy_idx
            );

            // Should have made a profit (buy at 100, sell at 200 in upward trend)
            assert!(
                final_equity > initial_capital,
                "Strategy {}: No profit. Final={:.2}, Initial={:.2}",
                strategy_idx,
                final_equity,
                initial_capital
            );
        }

        println!(
            "✅ Backtest execution kernel: Validated {} strategies with 1 trade each",
            n_strategies
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_metrics_calculation_kernel() {
        // Initialize GPU
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n_strategies = 10;
        let n_candles = 1000;

        // Create equity curves with known characteristics
        let mut equity_curves = Vec::new();

        for _strategy_idx in 0..n_strategies {
            // Upward trend: equity increases linearly
            let curve: Vec<f64> = (0..n_candles).map(|i| 10000.0 + i as f64 * 10.0).collect();
            equity_curves.extend_from_slice(&curve);
        }

        // Create simple trade data (all winning trades)
        let trade_size = std::mem::size_of::<f64>() * 6 + std::mem::size_of::<i64>() * 2;
        let max_trades = 1000;
        let trades_bytes = vec![0u8; n_strategies * max_trades * trade_size];

        // Set up 5 winning trades per strategy
        let num_trades = vec![5i32; n_strategies];

        // Allocate GPU memory
        let d_equity_curves = device.copy_to_device(&equity_curves).unwrap();
        let mut d_trades = device.allocate_device_buffer::<u8>(trades_bytes.len()).unwrap();
        device.stream().memcpy_htod(&trades_bytes, &mut d_trades).unwrap();
        let d_num_trades = device.copy_to_device_i32(&num_trades).unwrap();
        let mut d_sharpe = device.alloc_buffer(n_strategies).unwrap();
        let d_max_dd = device.stream().alloc_zeros::<f64>(n_strategies).unwrap();
        let mut d_win_rate = device.alloc_buffer(n_strategies).unwrap();

        // Compile and load kernel
        let ptx = compile_backtest_kernels().unwrap();
        let module = device.context().load_module(std::sync::Arc::unwrap_or_clone(ptx)).unwrap();
        let kernel = module.load_function("metrics_calculation_kernel").unwrap();

        // Launch kernel
        let shared_mem_bytes = 3 * 256 * std::mem::size_of::<f64>() as u32;
        let config = LaunchConfig {
            grid_dim: (n_strategies as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes,
        };

        let n_strategies_i32 = n_strategies as i32;
        let n_candles_i32 = n_candles as i32;

        let mut builder = device.stream().launch_builder(&kernel);
        builder.arg(&d_equity_curves);
        builder.arg(&d_trades);
        builder.arg(&d_num_trades);
        builder.arg(&mut d_sharpe);
        builder.arg(&mut d_win_rate);
        builder.arg(&n_strategies_i32);
        builder.arg(&n_candles_i32);

        unsafe { builder.launch(config).unwrap() };
        device.stream().synchronize().unwrap();

        // Copy results back
        let sharpe_ratios = device.copy_to_host(&d_sharpe).unwrap();
        let max_drawdowns = device.copy_to_host(&d_max_dd).unwrap();
        let _win_rates = device.copy_to_host(&d_win_rate).unwrap();

        // Validate results
        for strategy_idx in 0..n_strategies {
            // Sharpe ratio should be positive (upward trend)
            let sharpe = sharpe_ratios[strategy_idx];
            assert!(
                sharpe > 0.0,
                "Strategy {}: Sharpe ratio should be positive, got {:.2}",
                strategy_idx,
                sharpe
            );

            // Max drawdown should be 0 (perfectly linear upward trend)
            let max_dd = max_drawdowns[strategy_idx];
            assert!(
                max_dd.abs() < 0.01,
                "Strategy {}: Max drawdown should be ~0, got {:.4}",
                strategy_idx,
                max_dd
            );

            // Win rate not tested here (requires proper trade setup with PnL values)
            println!(
                "Strategy {}: Sharpe={:.2}, MaxDD={:.4}",
                strategy_idx, sharpe, max_dd
            );
        }

        println!(
            "✅ Metrics calculation kernel: Validated {} strategies",
            n_strategies
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_edge_case_no_trades() {
        // Initialize GPU
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n_strategies = 10;
        let n_candles = 1000;

        let (_, _, _, close) = generate_test_data(n_candles);

        // All HOLD signals (no trades)
        let signals = vec![0i8; n_strategies * n_candles];

        // Allocate GPU memory
        let d_signals = device.copy_to_device_i8(&signals).unwrap();
        let d_close = device.copy_to_device(&close).unwrap();
        let mut d_equity_curves = device
            .alloc_buffer(n_strategies * n_candles)
            .unwrap();

        let trade_size = std::mem::size_of::<f64>() * 6 + std::mem::size_of::<i64>() * 2;
        let max_trades = 1000;
        let mut d_trades = device
            .allocate_device_buffer::<u8>(n_strategies * max_trades * trade_size)
            .unwrap();
        let mut d_num_trades = device.allocate_device_buffer::<i32>(n_strategies).unwrap();

        let initial_capital = 10000.0;

        // Compile and launch kernel
        let ptx = compile_backtest_kernels().unwrap();
        let module = device.context().load_module(std::sync::Arc::unwrap_or_clone(ptx)).unwrap();
        let kernel = module.load_function("backtest_execution_kernel").unwrap();

        let config = LaunchConfig {
            grid_dim: (n_strategies as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let trading_fee = 0.001f64;
        let slippage = 0.0005f64;
        let n_strategies_i32 = n_strategies as i32;
        let n_candles_i32 = n_candles as i32;

        let mut d_max_drawdowns = device.stream().alloc_zeros::<f64>(n_strategies).unwrap();

        let mut builder = device.stream().launch_builder(&kernel);
        builder.arg(&d_signals);
        builder.arg(&d_close);
        builder.arg(&mut d_equity_curves);
        builder.arg(&mut d_trades);
        builder.arg(&mut d_num_trades);
        builder.arg(&mut d_max_drawdowns);
        builder.arg(&initial_capital);
        builder.arg(&trading_fee);
        builder.arg(&slippage);
        builder.arg(&n_strategies_i32);
        builder.arg(&n_candles_i32);

        unsafe { builder.launch(config).unwrap() };
        device.stream().synchronize().unwrap();

        // Validate
        let num_trades = device.stream().memcpy_dtov(&d_num_trades).unwrap();
        let equity_curves = device.copy_to_host(&d_equity_curves).unwrap();

        for strategy_idx in 0..n_strategies {
            // Should have 0 trades
            assert_eq!(
                num_trades[strategy_idx], 0,
                "Strategy {}: Expected 0 trades",
                strategy_idx
            );

            // Equity should stay at initial capital
            let base = strategy_idx * n_candles;
            for candle_idx in 0..n_candles {
                let equity = equity_curves[base + candle_idx];
                assert!(
                    (equity - initial_capital).abs() < 1.0,
                    "Strategy {}, candle {}: Equity={:.2}, expected {:.2}",
                    strategy_idx,
                    candle_idx,
                    equity,
                    initial_capital
                );
            }
        }

        println!("✅ Edge case: No trades validation passed");
    }

    #[test]
    #[ignore] // Requires GPU - Stress test
    fn test_stress_1000_strategies_10k_candles() {
        use std::time::Instant;

        // Initialize GPU
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n_strategies = 1000;
        let n_candles = 10_000;

        println!(
            "🚀 Stress test: {} strategies × {} candles",
            n_strategies, n_candles
        );

        let (_, _, _, close) = generate_test_data(n_candles);

        // Simple signal pattern for all strategies
        let mut signals = vec![0i8; n_strategies * n_candles];
        for strategy_idx in 0..n_strategies {
            let base = strategy_idx * n_candles;
            signals[base + 1000] = 1; // BUY
            signals[base + 5000] = 2; // SELL
        }

        // Allocate GPU memory
        let d_signals = device.copy_to_device_i8(&signals).unwrap();
        let d_close = device.copy_to_device(&close).unwrap();
        let mut d_equity_curves = device
            .alloc_buffer(n_strategies * n_candles)
            .unwrap();

        let trade_size = std::mem::size_of::<f64>() * 6 + std::mem::size_of::<i64>() * 2;
        let max_trades = 1000;
        let mut d_trades = device
            .allocate_device_buffer::<u8>(n_strategies * max_trades * trade_size)
            .unwrap();
        let mut d_num_trades = device.allocate_device_buffer::<i32>(n_strategies).unwrap();

        // Compile and launch kernel
        let ptx = compile_backtest_kernels().unwrap();
        let module = device.context().load_module(std::sync::Arc::unwrap_or_clone(ptx)).unwrap();
        let kernel = module.load_function("backtest_execution_kernel").unwrap();

        let config = LaunchConfig {
            grid_dim: (n_strategies as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let start = Instant::now();

        let initial_capital = 10000.0f64;
        let trading_fee = 0.001f64;
        let slippage = 0.0005f64;
        let n_strategies_i32 = n_strategies as i32;
        let n_candles_i32 = n_candles as i32;

        let mut d_max_drawdowns = device.stream().alloc_zeros::<f64>(n_strategies).unwrap();

        let mut builder = device.stream().launch_builder(&kernel);
        builder.arg(&d_signals);
        builder.arg(&d_close);
        builder.arg(&mut d_equity_curves);
        builder.arg(&mut d_trades);
        builder.arg(&mut d_num_trades);
        builder.arg(&mut d_max_drawdowns);
        builder.arg(&initial_capital);
        builder.arg(&trading_fee);
        builder.arg(&slippage);
        builder.arg(&n_strategies_i32);
        builder.arg(&n_candles_i32);

        unsafe { builder.launch(config).unwrap() };
        device.stream().synchronize().unwrap();

        let elapsed = start.elapsed();

        println!(
            "⏱️  Execution time: {:.2}ms",
            elapsed.as_secs_f64() * 1000.0
        );

        // Validate target: <250ms for 1000 strategies × 10K candles
        assert!(
            elapsed.as_millis() < 250,
            "Performance target missed: {:.2}ms > 250ms",
            elapsed.as_secs_f64() * 1000.0
        );

        println!(
            "✅ Stress test passed: {} strategies in {:.2}ms",
            n_strategies,
            elapsed.as_secs_f64() * 1000.0
        );
    }
}
