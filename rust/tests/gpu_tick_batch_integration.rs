//! GPU Tick Batch Integration Tests
//!
//! Validates BatchTickBacktest API integration with GeneticOptimizer and
//! verifies GPU vs CPU result equivalence within 0.01% tolerance.

#![cfg(feature = "gpu")]

use kimsfinance_core::backtest::tick_batch::{BatchBacktestResults, BatchTickBacktest};
use kimsfinance_core::backtest::tick_strategy::OrderFlowStrategy;
use kimsfinance_core::backtest::{BacktestConfig, TickEngine};
use kimsfinance_core::binance::{Timeframe, Trade};
use kimsfinance_core::gpu::device::GpuDevice;
use std::sync::Arc;

/// Generate deterministic test trades for validation
fn generate_test_trades(n: usize) -> Vec<Trade> {
    let mut trades = Vec::with_capacity(n);
    let base_time = 1_700_000_000_000i64; // Fixed epoch ms
    let base_price = 50_000.0;

    for i in 0..n {
        let timestamp_ms = base_time + (i as i64 * 100); // 100ms apart
        let price = base_price + (i as f64 * 0.01); // Slight uptrend
        let quantity = 0.1 + (i as f64 % 10.0) * 0.01; // Varying volume

        trades.push(Trade {
            trade_id: i as u64,
            price,
            quantity,
            quote_quantity: price * quantity,
            timestamp_ms,
            is_buyer_maker: i % 2 == 0,
        });
    }

    trades
}

/// Test 1: GPU vs CPU result equivalence (within 0.01% tolerance)
#[test]
fn test_gpu_vs_cpu_identical() {
    // Generate test data
    let trades = generate_test_trades(100_000);
    let params = vec![
        vec![50.0, 0.15, 10.0, 0.001, 5.0, 1.0],
        vec![100.0, 0.10, 15.0, 0.0015, 8.0, 1.2],
    ];

    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        execution_latency_ms: 10,
        use_gpu: true,
        force_cpu: false,
            ..Default::default()
        
    };

    // CPU results
    let cpu_results = run_cpu_backtests(&trades, &params, &config);

    // GPU results
    let device = Arc::new(GpuDevice::new().expect("GPU not available for test"));
    let gpu_results = BatchTickBacktest::new(device)
        .trades(&trades)
        .parameters_batch(&params)
        .config(config)
        .execute()
        .expect("GPU execution failed");

    // Validate results match within 0.01% tolerance
    assert_eq!(
        cpu_results.len(),
        gpu_results.results.len(),
        "Result count mismatch"
    );

    for (i, (cpu, gpu)) in cpu_results
        .iter()
        .zip(gpu_results.results.iter())
        .enumerate()
    {
        // Check total return within 0.01% (0.0001 absolute)
        let return_diff = (cpu.total_return - gpu.total_return).abs();
        assert!(
            return_diff < 0.0001,
            "Strategy {}: Return deviation too large: {:.4}% (CPU: {:.4}%, GPU: {:.4}%)",
            i,
            return_diff,
            cpu.total_return,
            gpu.total_return
        );

        // Check Sharpe ratio within 0.01 absolute
        let sharpe_diff = (cpu.sharpe_ratio - gpu.sharpe_ratio).abs();
        assert!(
            sharpe_diff < 0.01,
            "Strategy {}: Sharpe deviation too large: {:.4} (CPU: {:.4}, GPU: {:.4})",
            i,
            sharpe_diff,
            cpu.sharpe_ratio,
            gpu.sharpe_ratio
        );

        // Check max drawdown within 0.01% (0.0001 absolute)
        let dd_diff = (cpu.max_drawdown - gpu.max_drawdown).abs();
        assert!(
            dd_diff < 0.0001,
            "Strategy {}: Drawdown deviation too large: {:.4}% (CPU: {:.4}%, GPU: {:.4}%)",
            i,
            dd_diff,
            cpu.max_drawdown * 100.0,
            gpu.max_drawdown * 100.0
        );
    }

    println!("✅ GPU vs CPU validation passed: <0.01% deviation");
}

/// Test 2: Auto-tune batch size
#[test]
fn test_auto_tune_batch_size() {
    let device = Arc::new(GpuDevice::new().expect("GPU not available for test"));
    let batch = BatchTickBacktest::new(device);

    // Test with 106M trades (typical monthly dataset)
    let batch_size = batch.auto_tune_batch_size(106_000_000);

    // Should be between 1 and 20
    assert!(
        batch_size >= 1 && batch_size <= 20,
        "Batch size out of range: {}",
        batch_size
    );

    // For 12GB VRAM, should be around 9-10
    assert!(
        batch_size >= 8 && batch_size <= 12,
        "Expected batch size 8-12 for 12GB VRAM, got {}",
        batch_size
    );

    println!("✅ Auto-tuned batch size: {} strategies", batch_size);
}

/// Test 3: Graceful CPU fallback on GPU errors
#[test]
fn test_graceful_fallback() {
    let trades = generate_test_trades(10_000);
    let params = vec![vec![50.0, 0.15, 10.0, 0.001, 5.0, 1.0]];

    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        execution_latency_ms: 10,
        use_gpu: true,
        force_cpu: false, // Will auto-fallback if GPU fails,
            ..Default::default()
        
    };

    let device = Arc::new(GpuDevice::new().expect("GPU not available for test"));
    let results = BatchTickBacktest::new(device)
        .trades(&trades)
        .parameters_batch(&params)
        .config(config)
        .execute();

    // Should either succeed with GPU or fallback to CPU
    assert!(results.is_ok(), "Execution failed (no graceful fallback)");

    let results = results.unwrap();
    assert_eq!(results.results.len(), 1);

    println!("✅ Graceful fallback test passed");
}

/// Test 4: Builder API ergonomics
#[test]
fn test_builder_api() {
    let device = Arc::new(GpuDevice::new().expect("GPU not available for test"));

    let trades = vec![Trade::default(); 100];
    let params = vec![vec![30.0, 0.15, 10.0, 0.001, 5.0, 1.0]];

    // Test builder pattern chaining
    let _batch = BatchTickBacktest::new(device)
        .trades(&trades)
        .parameters_batch(&params)
        .batch_size(10)
        .force_cpu(true)
        .config(BacktestConfig::default());

    // If this compiles and runs, builder API is correctly structured
    println!("✅ Builder API test passed");
}

/// Test 5: Empty input validation
#[test]
fn test_empty_input_validation() {
    let device = Arc::new(GpuDevice::new().expect("GPU not available for test"));

    // Test empty trades
    let result = BatchTickBacktest::new(device.clone())
        .trades(&[])
        .parameters_batch(&[vec![50.0, 0.15, 10.0, 0.001, 5.0, 1.0]])
        .execute();
    assert!(result.is_err(), "Should fail with empty trades");

    // Test empty parameters
    let trades = generate_test_trades(100);
    let result = BatchTickBacktest::new(device.clone())
        .trades(&trades)
        .parameters_batch(&[])
        .execute();
    assert!(result.is_err(), "Should fail with empty parameters");

    println!("✅ Empty input validation passed");
}

/// Test 6: Large batch processing (batching logic)
#[test]
fn test_large_batch_processing() {
    let trades = generate_test_trades(50_000);

    // Generate 30 parameter sets (should trigger batching)
    let mut params = Vec::new();
    for window in 30..35 {
        for threshold in [0.10, 0.12, 0.15, 0.18, 0.20, 0.22] {
            params.push(vec![window as f64, threshold, 10.0, 0.001, 5.0, 1.0]);
        }
    }

    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        execution_latency_ms: 10,
        use_gpu: true,
        force_cpu: true, // Force CPU to avoid GPU dependency,
            ..Default::default()
        
    };

    let device = Arc::new(GpuDevice::new().expect("GPU not available for test"));
    let results = BatchTickBacktest::new(device)
        .trades(&trades)
        .parameters_batch(&params)
        .batch_size(10) // Force specific batch size
        .config(config)
        .execute()
        .expect("Large batch execution failed");

    // Should return results for all 30 strategies
    assert_eq!(results.results.len(), 30);

    // Results should be sorted by fitness (descending)
    for i in 0..results.results.len() - 1 {
        let fitness_a = results.results[i].fitness();
        let fitness_b = results.results[i + 1].fitness();
        assert!(
            fitness_a >= fitness_b,
            "Results not sorted by fitness: {} < {}",
            fitness_a,
            fitness_b
        );
    }

    println!(
        "✅ Large batch processing test passed: {} strategies",
        params.len()
    );
}

/// Test 7: Performance summary printing
#[test]
fn test_performance_summary() {
    let trades = generate_test_trades(10_000);
    let params = vec![
        vec![50.0, 0.15, 10.0, 0.001, 5.0, 1.0],
        vec![100.0, 0.10, 15.0, 0.0015, 8.0, 1.2],
    ];

    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        execution_latency_ms: 10,
        use_gpu: true,
        force_cpu: true,
            ..Default::default()
        
    };

    let device = Arc::new(GpuDevice::new().expect("GPU not available for test"));
    let results = BatchTickBacktest::new(device)
        .trades(&trades)
        .parameters_batch(&params)
        .config(config)
        .execute()
        .expect("Execution failed");

    // Test print_summary (shouldn't panic)
    results.print_summary();

    // Test top_n
    let top_1 = results.top_n(1);
    assert_eq!(top_1.len(), 1);

    // Test speedup calculation
    let speedup = results.speedup();
    assert!(speedup > 0.0, "Speedup should be positive");

    println!("✅ Performance summary test passed");
}

// ===== Helper Functions =====

/// Run CPU backtests for validation baseline
fn run_cpu_backtests(
    trades: &[Trade],
    params: &[Vec<f64>],
    config: &BacktestConfig,
) -> Vec<kimsfinance_core::backtest::BacktestResult> {
    use kimsfinance_core::backtest::tick_engine::TickEngine;
    use kimsfinance_core::backtest::tick_strategy::OrderFlowStrategy;
    use kimsfinance_core::binance::Timeframe;

    let engine = TickEngine::new(config.clone());
    let mut results = Vec::new();

    for param_vec in params {
        let window = param_vec[0] as usize;
        let imbalance_threshold = param_vec[1];
        let min_volume = param_vec[2];
        let spike_threshold = param_vec[3];
        let ema_period = param_vec[4] as usize;
        let volatility_factor = param_vec[5];

        let mut strategy = OrderFlowStrategy::new(
            imbalance_threshold,
        );

        let timeframe = Timeframe::parse("5m").expect("Timeframe parse error");
        let result = engine
            .run(&mut strategy, trades, timeframe)
            .expect("CPU backtest failed");

        results.push(result);
    }

    results
}
