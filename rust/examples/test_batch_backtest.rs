use kimsfinance_core::backtest::{BacktestConfig, BatchBacktestSweep, StrategyType};
use kimsfinance_core::gpu::GpuDevice;
use ndarray::Array1;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing BatchBacktestSweep construction...");

    // Initialize GPU
    let device = Arc::new(GpuDevice::new()?);
    println!("GPU initialized");

    // Create simple test data
    let n_candles = 100;
    let timestamps: Vec<i64> = (0..n_candles).map(|i| i as i64).collect();
    let prices: Vec<f64> = (0..n_candles).map(|i| 100.0 + i as f64 * 0.1).collect();

    let open = Array1::from(prices.clone());
    let high = Array1::from(prices.iter().map(|&p| p + 1.0).collect::<Vec<_>>());
    let low = Array1::from(prices.iter().map(|&p| p - 1.0).collect::<Vec<_>>());
    let close = Array1::from(prices);
    let volume = Array1::from(vec![1000.0; n_candles]);

    println!("Test data created");

    // Create RSI parameters for 10 strategies
    let mut rsi_params = Vec::new();
    for period in 10..20 {
        rsi_params.push(vec![period as f64, 70.0, 30.0]);
    }

    println!("Parameters created: {} strategies", rsi_params.len());

    // Create config
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: true,
        force_cpu: false,
    };

    println!("Config created");

    // Try to construct BatchBacktestSweep
    let sweep = BatchBacktestSweep::new(device.clone())
        .strategy_type(StrategyType::RsiCrossover)
        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
        .parameters_batch(&rsi_params)
        .config(config.clone());

    println!("BatchBacktestSweep constructed");

    // Try to execute
    println!("Executing backtest...");
    let results = sweep.execute()?;

    println!("Success! Results for {} strategies", results.results.len());
    println!(
        "GPU time: {:.2}ms, Total time: {:.2}ms",
        results.gpu_time_ms, results.total_time_ms
    );

    Ok(())
}
