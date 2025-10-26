//! Parameter Sweep Demo
//!
//! Demonstrates the Parameter Sweep Batch API for indicator optimization.
//!
//! This example shows:
//! 1. Basic parameter sweep (RSI periods 10-20)
//! 2. Parameter sweep with optimization metrics
//! 3. Finding the best parameter value
//! 4. Multi-indicator comparison
//! 5. Custom metric implementation
//!
//! Run with: cargo run --example parameter_sweep_demo --features gpu

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{
    GpuDevice, IndicatorData, IndicatorType, OptimizationMetric, ParameterSweep,
};
#[cfg(feature = "gpu")]
use ndarray::Array1;
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Parameter Sweep Batch API Demo ===\n");

    // Initialize GPU device
    let device = Arc::new(GpuDevice::new()?);
    println!("✓ GPU device initialized\n");

    // Generate sample price data (upward trend with volatility)
    let n = 5_000;
    let close = generate_sample_data(n);
    println!("✓ Generated {} price samples\n", n);

    // Demo 1: Basic parameter sweep (RSI periods 10-20)
    println!("--- Demo 1: Basic RSI Parameter Sweep ---");
    demo_basic_sweep(&device, &close)?;

    // Demo 2: Parameter sweep with Sharpe ratio optimization
    println!("\n--- Demo 2: RSI with Sharpe Ratio Optimization ---");
    demo_sharpe_optimization(&device, &close)?;

    // Demo 3: Compare multiple optimization metrics
    println!("\n--- Demo 3: Compare Optimization Metrics ---");
    demo_compare_metrics(&device, &close)?;

    // Demo 4: Multi-indicator sweep comparison
    println!("\n--- Demo 4: Multi-Indicator Parameter Sweep ---");
    demo_multi_indicator(&device, &close)?;

    // Demo 5: Custom metric (prefer higher final values)
    println!("\n--- Demo 5: Custom Optimization Metric ---");
    demo_custom_metric(&device, &close)?;

    println!("\n=== Demo Complete ===");

    Ok(())
}

#[cfg(feature = "gpu")]
fn generate_sample_data(n: usize) -> Array1<f64> {
    use std::f64::consts::PI;

    let mut prices = Vec::with_capacity(n);
    let base_price = 100.0;

    for i in 0..n {
        let t = i as f64;
        // Upward trend + cyclical pattern + noise
        let trend = base_price + t * 0.02;
        let cycle = 10.0 * (2.0 * PI * t / 200.0).sin();
        let noise = (t * 0.123).sin() * 2.0;
        prices.push(trend + cycle + noise);
    }

    Array1::from_vec(prices)
}

#[cfg(feature = "gpu")]
fn demo_basic_sweep(
    device: &Arc<GpuDevice>,
    close: &Array1<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let sweep = ParameterSweep::new(device.clone())
        .indicator(IndicatorType::RSI)
        .parameter_range(10..=20)
        .data_close(close)
        .execute()?;

    println!("Swept RSI periods 10-20");
    println!("Results:");
    for (period, result) in sweep.iter().take(5) {
        let valid_count = result.iter().filter(|&&x| !x.is_nan()).count();
        let avg = result
            .iter()
            .filter(|&&x| !x.is_nan())
            .sum::<f64>()
            / valid_count as f64;
        println!("  RSI({:2}): {:4} valid values, avg={:.2}", period, valid_count, avg);
    }
    println!("  ... ({} more parameters)", sweep.parameters.len() - 5);

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_sharpe_optimization(
    device: &Arc<GpuDevice>,
    close: &Array1<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let sweep = ParameterSweep::new(device.clone())
        .indicator(IndicatorType::RSI)
        .parameter_range(10..=30)
        .data_close(close)
        .metric(OptimizationMetric::Sharpe)
        .execute()?;

    // Display top 5 parameters
    let mut metrics_with_params: Vec<_> = sweep
        .parameters
        .iter()
        .zip(sweep.metrics.as_ref().unwrap().iter())
        .map(|(&p, &m)| (p, m))
        .collect();
    metrics_with_params.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top 5 RSI periods by Sharpe ratio:");
    for (i, (period, sharpe)) in metrics_with_params.iter().take(5).enumerate() {
        println!("  {}. RSI({:2}): Sharpe = {:.4}", i + 1, period, sharpe);
    }

    let best = sweep.find_optimal()?;
    println!("\n✓ Optimal parameter: RSI({}) with Sharpe = {:.4}", best.parameter, best.score);

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_compare_metrics(
    device: &Arc<GpuDevice>,
    close: &Array1<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let metrics = vec![
        ("Sharpe", OptimizationMetric::Sharpe),
        ("MaxDrawdown", OptimizationMetric::MaxDrawdown),
        ("WinRate", OptimizationMetric::WinRate),
        ("ProfitFactor", OptimizationMetric::ProfitFactor),
    ];

    println!("Optimal RSI period by different metrics:");
    for (name, metric) in metrics {
        let sweep = ParameterSweep::new(device.clone())
            .indicator(IndicatorType::RSI)
            .parameter_range(10..=30)
            .data_close(close)
            .metric(metric)
            .execute()?;

        let best = sweep.find_optimal()?;
        println!("  {:<15}: RSI({:2}) with score = {:.4}", name, best.parameter, best.score);
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_multi_indicator(
    device: &Arc<GpuDevice>,
    close: &Array1<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let indicators = vec![
        ("RSI", IndicatorType::RSI),
        ("SMA", IndicatorType::SMA),
        ("EMA", IndicatorType::EMA),
        ("WMA", IndicatorType::WMA),
    ];

    println!("Optimal period (10-30) by Sharpe ratio:");
    for (name, indicator) in indicators {
        let sweep = ParameterSweep::new(device.clone())
            .indicator(indicator)
            .parameter_range(10..=30)
            .data_close(close)
            .metric(OptimizationMetric::Sharpe)
            .execute()?;

        let best = sweep.find_optimal()?;
        println!("  {:<5}: period={:2}, Sharpe={:.4}", name, best.parameter, best.score);
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_custom_metric(
    device: &Arc<GpuDevice>,
    close: &Array1<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Custom metric: Prefer indicator with highest final value
    let custom_metric = Arc::new(|values: &Array1<f64>| -> f64 {
        values
            .iter()
            .rev()
            .find(|&&x| !x.is_nan())
            .copied()
            .unwrap_or(0.0)
    });

    let sweep = ParameterSweep::new(device.clone())
        .indicator(IndicatorType::RSI)
        .parameter_range(10..=30)
        .data_close(close)
        .metric(OptimizationMetric::Custom(custom_metric))
        .execute()?;

    let best = sweep.find_optimal()?;
    println!("Custom metric (highest final value):");
    println!("  Optimal: RSI({}) with final value = {:.2}", best.parameter, best.score);

    // Show top 3
    let mut metrics_with_params: Vec<_> = sweep
        .parameters
        .iter()
        .zip(sweep.metrics.as_ref().unwrap().iter())
        .map(|(&p, &m)| (p, m))
        .collect();
    metrics_with_params.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n  Top 3:");
    for (i, (period, score)) in metrics_with_params.iter().take(3).enumerate() {
        println!("    {}. RSI({:2}): final value = {:.2}", i + 1, period, score);
    }

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("ERROR: This example requires the 'gpu' feature");
    eprintln!("Run with: cargo run --example parameter_sweep_demo --features gpu");
    std::process::exit(1);
}
