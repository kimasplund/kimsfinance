//! Demonstration of ExecutionMode API for Fused Kernel Integration
//!
//! Shows how to use Traditional, Fused, and Auto execution modes.
//! Validates the 1.31-4x speedup from fused kernel execution.

use kimsfinance_core::backtest::engine::BacktestConfig;
use kimsfinance_core::backtest::{BatchBacktestSweep, ExecutionMode, StrategyType};
use kimsfinance_core::gpu::device::GpuDevice;
use ndarray::Array1;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ExecutionMode API Demonstration");
    println!("===================================\n");

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

    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        execution_latency_ms: 0,
        use_gpu: true,
        force_cpu: false,
    };

    // ===== Test 1: Traditional Mode (Explicit) =====
    println!("Test 1: Traditional Mode (4 kernel launches)");
    println!("---------------------------------------------");

    let mut params_traditional = vec![];
    for buy_thresh in 25..30 {
        for sell_thresh in 70..80 {
            params_traditional.push(vec![14.0, buy_thresh as f64, sell_thresh as f64]);
        }
    }

    let start_trad = Instant::now();
    let results_trad = BatchBacktestSweep::new(device.clone())
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .parameters_batch(&params_traditional)
        .config(config.clone())
        .execution_mode(ExecutionMode::Traditional) // Explicit mode
        .execute()?;
    let time_trad = start_trad.elapsed().as_secs_f64() * 1000.0;

    println!(
        "✅ Processed {} strategies in {:.2}ms",
        params_traditional.len(),
        time_trad
    );
    println!("   GPU time: {:.2}ms", results_trad.gpu_time_ms);
    println!("   VRAM used: {:.2} MB", results_trad.vram_used_mb);
    println!(
        "   Best Sharpe: {:.2}\n",
        results_trad.results[0].sharpe_ratio
    );

    // ===== Test 2: Fused Mode (Explicit) =====
    println!("Test 2: Fused Mode (single kernel launch)");
    println!("------------------------------------------");

    let mut params_fused = vec![];
    for rsi_period in 10..15 {
        for buy_thresh in 20..30 {
            for sell_thresh in 70..80 {
                params_fused.push(vec![
                    rsi_period as f64,
                    buy_thresh as f64,
                    sell_thresh as f64,
                ]);
            }
        }
    }

    let start_fused = Instant::now();
    let results_fused = BatchBacktestSweep::new(device.clone())
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .parameters_batch(&params_fused)
        .config(config.clone())
        .execution_mode(ExecutionMode::Fused) // Explicit mode
        .execute()?;
    let time_fused = start_fused.elapsed().as_secs_f64() * 1000.0;

    println!(
        "✅ Processed {} strategies in {:.2}ms",
        params_fused.len(),
        time_fused
    );
    println!("   GPU time: {:.2}ms", results_fused.gpu_time_ms);
    println!("   VRAM used: {:.2} MB", results_fused.vram_used_mb);
    println!(
        "   Best Sharpe: {:.2}\n",
        results_fused.results[0].sharpe_ratio
    );

    // ===== Test 3: Auto Mode (System decides) =====
    println!("Test 3: Auto Mode (automatic selection)");
    println!("----------------------------------------");

    let mut params_auto = vec![];
    for rsi_period in 12..15 {
        for buy_thresh in 25..35 {
            for sell_thresh in 70..73 {
                params_auto.push(vec![
                    rsi_period as f64,
                    buy_thresh as f64,
                    sell_thresh as f64,
                ]);
            }
        }
    }

    let start_auto = Instant::now();
    let results_auto = BatchBacktestSweep::new(device.clone())
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .parameters_batch(&params_auto)
        .config(config.clone())
        .execution_mode(ExecutionMode::Auto) // Let system decide
        .execute()?;
    let time_auto = start_auto.elapsed().as_secs_f64() * 1000.0;

    println!(
        "✅ Processed {} strategies in {:.2}ms",
        params_auto.len(),
        time_auto
    );
    println!("   GPU time: {:.2}ms", results_auto.gpu_time_ms);
    println!("   VRAM used: {:.2} MB", results_auto.vram_used_mb);
    println!(
        "   Best Sharpe: {:.2}\n",
        results_auto.results[0].sharpe_ratio
    );

    // ===== Performance Analysis =====
    println!("📈 Performance Analysis");
    println!("======================");

    let throughput_trad = params_traditional.len() as f64 / time_trad;
    let throughput_fused = params_fused.len() as f64 / time_fused;
    let speedup = throughput_fused / throughput_trad;

    println!(
        "Traditional: {:.2}ms for {} strategies",
        time_trad,
        params_traditional.len()
    );
    println!(
        "Fused:       {:.2}ms for {} strategies",
        time_fused,
        params_fused.len()
    );
    println!(
        "Auto:        {:.2}ms for {} strategies",
        time_auto,
        params_auto.len()
    );
    println!();
    println!("Throughput comparison:");
    println!("  Traditional: {:.1} strategies/ms", throughput_trad);
    println!("  Fused:       {:.1} strategies/ms", throughput_fused);
    println!();
    println!("Fused kernel speedup: {:.2}x", speedup);

    if speedup >= 1.31 {
        println!(
            "\n✅ SUCCESS: Fused kernel achieves {:.2}x speedup!",
            speedup
        );
        println!("   Target: ≥1.31x, Typical: 2-4x");
    } else {
        println!(
            "\n⚠️  WARNING: Speedup ({:.2}x) below 1.31x target",
            speedup
        );
    }

    println!("\n💡 Usage Recommendations:");
    println!("   - Use Traditional for <100 strategies");
    println!("   - Use Fused for ≥100 strategies");
    println!("   - Use Auto (default) for adaptive selection");

    Ok(())
}
