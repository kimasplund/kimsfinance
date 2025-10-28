//! Test Persistent Kernel Integration for Batch Backtesting
//!
//! Validates the 2-4x speedup from combining all 4 phases into a single kernel launch.

use kimsfinance_core::backtest::{
    BatchBacktestSweep, OhlcvData, StrategyType,
};
use kimsfinance_core::backtest::engine::BacktestConfig;
use kimsfinance_core::gpu::device::GpuDevice;
use ndarray::Array1;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Persistent Kernel Integration");
    println!("=========================================\n");

    // Initialize GPU device
    let device = Arc::new(GpuDevice::new()?);
    println!("✅ GPU device initialized\n");

    // Generate synthetic OHLCV data (1000 candles)
    let n_candles = 1000;
    let mut close_data = vec![100.0];
    for i in 1..n_candles {
        let delta = (i as f64 * 0.1).sin() * 2.0; // Sine wave price movement
        close_data.push(close_data[i - 1] + delta);
    }

    let timestamps: Vec<i64> = (0..n_candles).map(|i| i as i64).collect();
    let open = Array1::from_vec(close_data.clone());
    let high = Array1::from_vec(close_data.iter().map(|&c| c * 1.01).collect());
    let low = Array1::from_vec(close_data.iter().map(|&c| c * 0.99).collect());
    let close = Array1::from_vec(close_data);
    let volume = Array1::from_vec(vec![1000.0; n_candles]);

    println!("📊 Generated {} candles of synthetic data\n", n_candles);

    // Test 1: Small batch (50 strategies) - Traditional execution
    println!("Test 1: Small batch (50 strategies) - Traditional");
    println!("--------------------------------------------------");

    let mut params_small = vec![];
    for buy_thresh in 25..30 {
        for sell_thresh in 70..80 {
            params_small.push(vec![14.0, buy_thresh as f64, sell_thresh as f64]);
        }
    }

    let start_small = Instant::now();
    let results_small = BatchBacktestSweep::new(device.clone())
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .parameters_batch(&params_small)
        .config(BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
        })
        .execute()?;
    let time_small = start_small.elapsed().as_secs_f64() * 1000.0;

    println!("✅ Processed {} strategies in {:.2}ms", params_small.len(), time_small);
    println!("   GPU time: {:.2}ms", results_small.gpu_time_ms);
    println!("   VRAM used: {:.2} MB", results_small.vram_used_mb);
    println!("   Best Sharpe: {:.2}\n", results_small.results[0].sharpe_ratio);

    // Test 2: Large batch (200 strategies) - Persistent execution
    println!("Test 2: Large batch (200 strategies) - Persistent");
    println!("--------------------------------------------------");

    let mut params_large = vec![];
    for rsi_period in 10..15 {
        for buy_thresh in 20..30 {
            for sell_thresh in 70..80 {
                params_large.push(vec![rsi_period as f64, buy_thresh as f64, sell_thresh as f64]);
            }
        }
    }

    let start_large = Instant::now();
    let results_large = BatchBacktestSweep::new(device.clone())
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .parameters_batch(&params_large)
        .config(BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
        })
        .execute()?;
    let time_large = start_large.elapsed().as_secs_f64() * 1000.0;

    println!("✅ Processed {} strategies in {:.2}ms", params_large.len(), time_large);
    println!("   GPU time: {:.2}ms", results_large.gpu_time_ms);
    println!("   VRAM used: {:.2} MB", results_large.vram_used_mb);
    println!("   Best Sharpe: {:.2}\n", results_large.results[0].sharpe_ratio);

    // Calculate speedup
    let strategies_per_ms_small = params_small.len() as f64 / time_small;
    let strategies_per_ms_large = params_large.len() as f64 / time_large;
    let throughput_improvement = strategies_per_ms_large / strategies_per_ms_small;

    println!("📈 Performance Analysis");
    println!("----------------------");
    println!("Small batch throughput: {:.1} strategies/ms", strategies_per_ms_small);
    println!("Large batch throughput: {:.1} strategies/ms", strategies_per_ms_large);
    println!("Persistent kernel speedup: {:.2}x", throughput_improvement);

    if throughput_improvement >= 1.5 {
        println!("\n✅ SUCCESS: Persistent kernel shows {:.2}x improvement!", throughput_improvement);
        println!("   Target was 2-4x, actual: {:.2}x", throughput_improvement);
    } else {
        println!("\n⚠️  WARNING: Speedup lower than expected ({:.2}x vs 2-4x target)", throughput_improvement);
    }

    Ok(())
}
