//! Euler Search GPU Optimizer Demo
//!
//! Demonstrates QuantConnect-style iterative grid refinement with GPU acceleration.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --features gpu --example euler_search_demo
//! ```
//!
//! # Expected Output
//!
//! - Converges in 5-10 iterations
//! - 90% fewer evaluations than exhaustive grid search
//! - <250ms per iteration for 1000 parameter sets

#[cfg(feature = "gpu")]
use kimsfinance_core::backtest::{BacktestConfig, EulerSearchOptimizer, StrategyType};
#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::device::GpuDevice;
#[cfg(feature = "gpu")]
use ndarray::Array1;
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
fn generate_realistic_data(
    num_candles: usize,
) -> (
    Vec<i64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    println!(
        "📊 Generating {} candles of realistic OHLCV data...",
        num_candles
    );

    let timestamps: Vec<i64> = (0..num_candles)
        .map(|i| 1609459200 + (i as i64 * 3600)) // Start: 2021-01-01, hourly candles
        .collect();

    // Realistic price movement: trend + mean reversion + noise
    let base_price = 100.0;
    let mut prices = Vec::with_capacity(num_candles);
    let mut price = base_price;

    for i in 0..num_candles {
        // Long-term trend (sine wave)
        let trend = ((i as f64 / 100.0).sin() * 20.0) / 100.0;

        // Mean reversion
        let mean_reversion = -(price - base_price) * 0.01;

        // Random walk component
        let random = ((i * 1664525 + 1013904223) % 65536) as f64 / 65536.0 - 0.5;

        // Combine components
        let change = trend + mean_reversion + random * 0.5;
        price += change;

        prices.push(price);
    }

    let close = Array1::from_vec(prices);

    // Generate OHLC from close
    let open = close.mapv(|c| {
        let offset = ((c * 123.456) as i64 % 100) as f64 / 100.0 - 0.5;
        c + offset * 0.5
    });

    let high = close.mapv(|c| {
        let offset = ((c * 234.567) as i64 % 100) as f64 / 100.0;
        c + offset * 0.8
    });

    let low = close.mapv(|c| {
        let offset = ((c * 345.678) as i64 % 100) as f64 / 100.0;
        c - offset * 0.8
    });

    let volume = Array1::from_iter((0..num_candles).map(|i| {
        let base_vol = 1_000_000.0;
        let variation = ((i * 2654435761) % 1000) as f64 / 1000.0;
        base_vol * (0.7 + variation * 0.6)
    }));

    (timestamps, open, high, low, close, volume)
}

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Euler Search GPU Optimizer Demo\n");
    println!("═══════════════════════════════════════════════════════════");

    // Initialize GPU
    println!("\n1️⃣  Initializing GPU device...");
    let device = Arc::new(GpuDevice::new()?);
    println!("   ✓ GPU initialized successfully");

    // Generate data
    println!("\n2️⃣  Generating test data...");
    let num_candles = 5000;
    let (timestamps, open, high, low, close, volume) = generate_realistic_data(num_candles);
    println!("   ✓ Generated {} candles", num_candles);

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001, // 0.1%
        slippage: 0.0005,   // 0.05%
        execution_latency_ms: 10,
        use_gpu: true,
        force_cpu: false,
    };

    println!("\n3️⃣  Configuring Euler Search optimizer...");
    println!("   Strategy: RSI Crossover");
    println!("   Segment amount: 4 (QuantConnect default)");
    println!("   Max iterations: 15");
    println!("   Batch size: 1000");
    println!("   Early stopping: 3 iterations");

    // Create optimizer
    let mut optimizer = EulerSearchOptimizer::new(device.clone())
        .segment_amount(4)
        .max_iterations(15)
        .batch_size(1000)
        .early_stopping_patience(Some(3));

    // Define parameter search space
    println!("\n4️⃣  Defining parameter search space...");
    optimizer.add_parameter("rsi_period", 5.0, 30.0, 5.0, 1.0);
    println!("   • RSI Period: [5.0, 30.0], step=5.0, min_step=1.0");

    optimizer.add_parameter("buy_threshold", 20.0, 40.0, 5.0, 1.0);
    println!("   • Buy Threshold: [20.0, 40.0], step=5.0, min_step=1.0");

    optimizer.add_parameter("sell_threshold", 60.0, 80.0, 5.0, 1.0);
    println!("   • Sell Threshold: [60.0, 80.0], step=5.0, min_step=1.0");

    // Calculate grid search comparison
    let grid_points = 6 * 5 * 5; // (30-5)/5+1 × (40-20)/5+1 × (80-60)/5+1
    println!(
        "\n   📊 Grid Search would require {} evaluations",
        grid_points
    );

    // Run optimization
    println!("\n5️⃣  Running Euler Search optimization...");
    println!("   (GPU-accelerated batch backtesting)\n");

    let start = std::time::Instant::now();
    let result = optimizer.optimize(
        StrategyType::RsiCrossover,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
        config,
    )?;
    let wall_time = start.elapsed();

    // Print results
    println!("\n═══════════════════════════════════════════════════════════");
    println!("6️⃣  OPTIMIZATION RESULTS");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("🎯 Best Parameters Found:");
    println!(
        "   • RSI Period: {:.1}",
        result.best_parameters.get("rsi_period").unwrap()
    );
    println!(
        "   • Buy Threshold: {:.1}",
        result.best_parameters.get("buy_threshold").unwrap()
    );
    println!(
        "   • Sell Threshold: {:.1}",
        result.best_parameters.get("sell_threshold").unwrap()
    );

    println!("\n📈 Performance Metrics:");
    println!("   • Best Fitness (Sharpe): {:.4}", result.best_fitness);
    println!(
        "   • Converged: {}",
        if result.is_converged() {
            "✓ Yes"
        } else {
            "✗ No"
        }
    );

    println!("\n⚡ Optimization Statistics:");
    println!("   • Iterations: {}", result.iterations);
    println!("   • Total Evaluations: {}", result.total_evaluations);
    println!(
        "   • Speedup vs Grid: {:.2}x ({} → {} evals)",
        result.grid_search_speedup(6),
        grid_points,
        result.total_evaluations
    );
    println!(
        "   • Evaluation Reduction: {:.1}%",
        (1.0 - result.total_evaluations as f64 / grid_points as f64) * 100.0
    );

    println!("\n⏱️  Timing:");
    println!("   • GPU Time: {:.2}ms", result.total_gpu_time_ms);
    println!("   • Total Time: {:.2}ms", result.total_time_ms);
    println!("   • Wall Clock: {:.2}ms", wall_time.as_secs_f64() * 1000.0);
    println!(
        "   • Avg Iteration: {:.2}ms",
        result.total_time_ms / result.iterations as f64
    );
    println!(
        "   • GPU Utilization: {:.1}%",
        (result.total_gpu_time_ms / result.total_time_ms) * 100.0
    );

    println!("\n📊 Convergence History:");
    for (i, &fitness) in result.convergence_history.iter().enumerate() {
        let bar_length = ((fitness + 5.0) / 10.0 * 40.0).max(0.0).min(40.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("   Iter {:2}: {:6.3} {}", i, fitness, bar);
    }

    println!("\n🔍 Refinement History:");
    for step in result.refinement_history.iter().take(5) {
        println!("\n   Iteration {}:", step.iteration);
        println!("      Evaluations: {}", step.num_evaluations);
        println!("      Best Fitness: {:.4}", step.best_fitness);

        for (param_name, &step_size) in &step.step_sizes {
            let (min, max) = step.search_ranges.get(param_name).unwrap();
            println!(
                "      {}: [{:.2}, {:.2}], step={:.3}",
                param_name, min, max, step_size
            );
        }
    }

    if result.refinement_history.len() > 5 {
        println!(
            "   ... ({} more iterations)",
            result.refinement_history.len() - 5
        );
    }

    // Performance validation
    println!("\n═══════════════════════════════════════════════════════════");
    println!("7️⃣  PERFORMANCE VALIDATION");
    println!("═══════════════════════════════════════════════════════════\n");

    let avg_iter_time = result.total_time_ms / result.iterations as f64;
    let target_met = avg_iter_time < 250.0;

    println!("   Target: <250ms per iteration (1000 params)");
    println!("   Actual: {:.2}ms per iteration", avg_iter_time);
    println!(
        "   Status: {}",
        if target_met { "✓ PASS" } else { "✗ FAIL" }
    );

    let speedup = result.grid_search_speedup(6);
    let speedup_met = speedup >= 2.0;

    println!("\n   Target: ≥2x speedup vs grid search");
    println!("   Actual: {:.2}x speedup", speedup);
    println!(
        "   Status: {}",
        if speedup_met { "✓ PASS" } else { "✗ FAIL" }
    );

    let eval_reduction = (1.0 - result.total_evaluations as f64 / grid_points as f64) * 100.0;
    let reduction_met = eval_reduction >= 50.0;

    println!("\n   Target: ≥50% evaluation reduction");
    println!("   Actual: {:.1}% reduction", eval_reduction);
    println!(
        "   Status: {}",
        if reduction_met {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );

    println!("\n═══════════════════════════════════════════════════════════");

    if target_met && speedup_met && reduction_met {
        println!("✅ ALL PERFORMANCE TARGETS MET!");
    } else {
        println!("⚠️  Some performance targets not met (may vary by hardware)");
    }

    println!("═══════════════════════════════════════════════════════════\n");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("❌ This example requires the 'gpu' feature.");
    eprintln!("   Run with: cargo run --release --features gpu --example euler_search_demo");
    std::process::exit(1);
}
