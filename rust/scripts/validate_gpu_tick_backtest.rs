//! GPU vs CPU Tick Backtest Validation Script
//!
//! # Purpose
//!
//! Validates that GPU tick backtest matches CPU implementation exactly.
//!
//! # Validation Criteria
//!
//! - **Accuracy**: <0.01% deviation in equity curves
//! - **Trades**: Same number of trades executed
//! - **P&L**: Matching trade P&L within tolerance
//! - **Metrics**: Matching Sharpe ratio, max drawdown, win rate
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --features gpu --bin validate_gpu_tick_backtest
//! ```

use kimsfinance_core::backtest::{BacktestConfig, Signal, TickEngine};
use kimsfinance_core::binance::{Timeframe, Trade};
use kimsfinance_core::gpu::tick_backtest_batch::{TickBacktestBatch, BacktestConfig as GpuBacktestConfig};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU vs CPU Tick Backtest Validation");
    println!("=====================================\n");

    // ========================================================================
    // CONFIGURATION
    // ========================================================================

    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,     // 0.1%
        slippage: 0.0005,       // 0.05%
        ..Default::default()
    };

    let gpu_config = GpuBacktestConfig {
        initial_capital: config.initial_capital,
        trading_fee: config.trading_fee,
        slippage: config.slippage,
        execution_delay_ms: 0,  // No delay for CPU comparison
    };

    // ========================================================================
    // GENERATE TEST DATA
    // ========================================================================

    println!("Generating test data...");

    let n_ticks = 10_000;
    let mut prices = Vec::with_capacity(n_ticks);
    let mut timestamps = Vec::with_capacity(n_ticks);

    // Generate realistic price movement (random walk with drift)
    let mut price = 100.0;
    for i in 0..n_ticks {
        timestamps.push((i as i64) * 1000); // 1 second per tick
        prices.push(price);

        // Random walk
        let change = if i % 3 == 0 {
            0.05
        } else if i % 5 == 0 {
            -0.03
        } else {
            0.01
        };
        price += change;
        price = price.max(50.0); // Floor price
    }

    // Generate signals (simple momentum strategy)
    let mut signals = Vec::with_capacity(n_ticks);
    let mut in_position = false;

    for i in 0..n_ticks {
        let signal = if i % 100 == 0 && !in_position {
            in_position = true;
            Signal::Buy
        } else if i % 100 == 50 && in_position {
            in_position = false;
            Signal::Sell
        } else {
            Signal::Hold
        };
        signals.push(signal);
    }

    println!("Generated {} ticks with {} signals", n_ticks,
             signals.iter().filter(|s| **s != Signal::Hold).count());

    // ========================================================================
    // RUN CPU BACKTEST
    // ========================================================================

    println!("\nRunning CPU backtest...");

    // Convert to Trade format for CPU engine
    let trades: Vec<Trade> = prices.iter().zip(timestamps.iter())
        .enumerate()
        .map(|(i, (price, time))| Trade {
            id: i as u64,
            price: *price,
            qty: 1.0,
            time: *time,
            is_buyer_maker: false,
        })
        .collect();

    // Create simple strategy that returns our signals
    struct TestStrategy {
        signals: Vec<Signal>,
    }

    impl kimsfinance_core::backtest::TickStrategy for TestStrategy {
        fn on_tick(&mut self, _candle: &kimsfinance_core::binance::IncompleteCandle,
                   _trade: &Trade, tick_idx: usize) -> Signal {
            if tick_idx < self.signals.len() {
                self.signals[tick_idx]
            } else {
                Signal::Hold
            }
        }

        fn on_candle_complete(&mut self, _candle: &kimsfinance_core::binance::Candle) {}
    }

    let mut strategy = TestStrategy {
        signals: signals.clone(),
    };

    let engine = TickEngine::new(config);
    let timeframe = Timeframe::parse("1m")?;

    let cpu_start = Instant::now();
    let cpu_result = engine.run(&mut strategy, &trades, timeframe)?;
    let cpu_elapsed = cpu_start.elapsed();

    println!("CPU Results:");
    println!("  Final Equity: ${:.2}", cpu_result.final_equity);
    println!("  Total Return: {:.2}%", cpu_result.total_return);
    println!("  Sharpe Ratio: {:.2}", cpu_result.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", cpu_result.max_drawdown);
    println!("  Win Rate: {:.2}%", cpu_result.win_rate * 100.0);
    println!("  Num Trades: {}", cpu_result.num_trades);
    println!("  Execution Time: {:?}", cpu_elapsed);

    // ========================================================================
    // RUN GPU BACKTEST
    // ========================================================================

    println!("\nRunning GPU backtest...");

    let gpu_backtest = TickBacktestBatch::new(gpu_config)?;

    // Warm-up run (JIT compilation)
    let _ = gpu_backtest.run_batch(&[signals.clone()], &prices, &timestamps)?;

    let gpu_start = Instant::now();
    let gpu_results = gpu_backtest.run_batch(&[signals.clone()], &prices, &timestamps)?;
    let gpu_elapsed = gpu_start.elapsed();

    let gpu_result = &gpu_results[0];

    println!("GPU Results:");
    println!("  Final Equity: ${:.2}", gpu_result.final_equity);
    println!("  Total Return: {:.2}%", gpu_result.total_return);
    println!("  Sharpe Ratio: {:.2}", gpu_result.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", gpu_result.max_drawdown);
    println!("  Win Rate: {:.2}%", gpu_result.win_rate * 100.0);
    println!("  Num Trades: {}", gpu_result.num_trades);
    println!("  Execution Time: {:?}", gpu_elapsed);

    // ========================================================================
    // VALIDATION
    // ========================================================================

    println!("\nValidation Results:");
    println!("===================\n");

    let tolerance = 0.01; // 1% tolerance for floating point differences

    // Final equity
    let equity_diff = ((gpu_result.final_equity - cpu_result.final_equity).abs()
                       / cpu_result.final_equity) * 100.0;
    println!("Final Equity Difference: {:.4}%", equity_diff);
    if equity_diff < tolerance {
        println!("  ✓ PASS: Within {:.2}% tolerance", tolerance);
    } else {
        println!("  ✗ FAIL: Exceeds {:.2}% tolerance", tolerance);
    }

    // Total return
    let return_diff = (gpu_result.total_return - cpu_result.total_return).abs();
    println!("\nTotal Return Difference: {:.4}%", return_diff);
    if return_diff < tolerance {
        println!("  ✓ PASS: Within {:.2}% tolerance", tolerance);
    } else {
        println!("  ✗ FAIL: Exceeds {:.2}% tolerance", tolerance);
    }

    // Sharpe ratio
    let sharpe_diff = ((gpu_result.sharpe_ratio - cpu_result.sharpe_ratio).abs()
                       / cpu_result.sharpe_ratio.abs().max(0.01)) * 100.0;
    println!("\nSharpe Ratio Difference: {:.4}%", sharpe_diff);
    if sharpe_diff < tolerance * 10.0 {  // 10% tolerance for Sharpe (more sensitive)
        println!("  ✓ PASS: Within {:.2}% tolerance", tolerance * 10.0);
    } else {
        println!("  ✗ FAIL: Exceeds {:.2}% tolerance", tolerance * 10.0);
    }

    // Max drawdown
    let dd_diff = (gpu_result.max_drawdown - cpu_result.max_drawdown).abs();
    println!("\nMax Drawdown Difference: {:.4}%", dd_diff);
    if dd_diff < tolerance {
        println!("  ✓ PASS: Within {:.2}% tolerance", tolerance);
    } else {
        println!("  ✗ FAIL: Exceeds {:.2}% tolerance", tolerance);
    }

    // Win rate
    let wr_diff = (gpu_result.win_rate - cpu_result.win_rate).abs();
    println!("\nWin Rate Difference: {:.4}", wr_diff);
    if wr_diff < tolerance / 100.0 {
        println!("  ✓ PASS: Within {:.4} tolerance", tolerance / 100.0);
    } else {
        println!("  ✗ FAIL: Exceeds {:.4} tolerance", tolerance / 100.0);
    }

    // Number of trades
    let trades_match = gpu_result.num_trades == cpu_result.num_trades;
    println!("\nNumber of Trades: GPU={}, CPU={}", gpu_result.num_trades, cpu_result.num_trades);
    if trades_match {
        println!("  ✓ PASS: Exact match");
    } else {
        println!("  ✗ FAIL: Mismatch");
    }

    // ========================================================================
    // PERFORMANCE COMPARISON
    // ========================================================================

    println!("\nPerformance Comparison:");
    println!("=======================\n");

    let speedup = cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64();
    println!("CPU Time: {:?}", cpu_elapsed);
    println!("GPU Time: {:?}", gpu_elapsed);
    println!("Speedup: {:.2}x", speedup);

    if speedup > 1.0 {
        println!("  ✓ GPU is faster!");
    } else {
        println!("  Note: GPU slower for single strategy (expected)");
        println!("        GPU excels at parallel execution (10-20 strategies)");
    }

    // ========================================================================
    // PARALLEL BENCHMARK
    // ========================================================================

    println!("\nParallel Benchmark (10 Strategies):");
    println!("=====================================\n");

    // Run 10 strategies in parallel
    let parallel_signals = vec![signals.clone(); 10];

    let parallel_start = Instant::now();
    let parallel_results = gpu_backtest.run_batch(&parallel_signals, &prices, &timestamps)?;
    let parallel_elapsed = parallel_start.elapsed();

    println!("GPU Parallel Time: {:?}", parallel_elapsed);
    println!("Strategies Processed: {}", parallel_results.len());

    let theoretical_cpu_time = cpu_elapsed.as_secs_f64() * 10.0;
    let parallel_speedup = theoretical_cpu_time / parallel_elapsed.as_secs_f64();
    println!("Parallel Speedup: {:.2}x vs sequential CPU", parallel_speedup);

    // ========================================================================
    // THROUGHPUT BENCHMARK
    // ========================================================================

    println!("\nThroughput Benchmark:");
    println!("=====================\n");

    let throughput = gpu_backtest.benchmark_throughput(10, 100_000, 2, 5)?;
    println!("Throughput: {:.2} M ticks/sec", throughput / 1e6);
    println!("Throughput: {:.2} B ticks/sec", throughput / 1e9);

    if throughput > 1e9 {
        println!("  ✓ PASS: Exceeds 1 B ticks/sec target");
    } else {
        println!("  ✗ FAIL: Below 1 B ticks/sec target");
    }

    // ========================================================================
    // FINAL VERDICT
    // ========================================================================

    println!("\nFinal Verdict:");
    println!("==============\n");

    let all_pass = equity_diff < tolerance
                   && return_diff < tolerance
                   && sharpe_diff < tolerance * 10.0
                   && dd_diff < tolerance
                   && wr_diff < tolerance / 100.0
                   && trades_match
                   && throughput > 1e9;

    if all_pass {
        println!("✓ ALL TESTS PASSED!");
        println!("  GPU implementation matches CPU within tolerance");
        println!("  Performance target achieved (>1B ticks/sec)");
    } else {
        println!("✗ SOME TESTS FAILED");
        println!("  Review validation results above");
    }

    Ok(())
}
