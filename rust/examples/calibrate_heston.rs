//! Example: Calibrate Heston Model to Market Option Prices
//!
//! This example demonstrates how to:
//! 1. Initialize GPU-accelerated Heston pricer
//! 2. Load market option data
//! 3. Calibrate model parameters using L-BFGS-B
//! 4. Evaluate calibration quality
//!
//! # Running
//!
//! ```bash
//! cargo run --example calibrate_heston --features heston
//! ```
//!
//! # Requirements
//!
//! - NVIDIA GPU with CUDA support
//! - Feature flag: `heston` (enables GPU + argmin)

#[cfg(feature = "heston")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use kimsfinance_core::gpu::GpuDevice;
    use kimsfinance_core::gpu::heston_pricing::HestonGpuPricer;
    use kimsfinance_core::quantitative::heston::{
        CalibrationResult, HestonCalibrator, HestonParams, OptionQuote, OptionType, ParameterBounds,
    };
    use std::sync::Arc;

    println!("=== Heston Model Calibration Example ===\n");

    // Step 1: Initialize GPU device and pricer
    println!("Initializing GPU device...");
    let device = Arc::new(GpuDevice::new()?);
    let gpu_pricer = Arc::new(HestonGpuPricer::new(device, 4096)?);
    println!("✓ GPU device initialized\n");

    // Step 2: Create synthetic market data
    // In production, load from IBKR, Deribit, or other data source
    println!("Generating synthetic market data...");
    let market_options = generate_synthetic_market_data();
    println!("✓ Loaded {} option quotes\n", market_options.len());

    // Step 3: Set initial parameter guess
    println!("Setting initial parameter guess...");
    let initial_params = HestonParams {
        kappa: 1.5,  // Mean reversion speed
        theta: 0.05, // Long-term variance (22.4% vol)
        sigma: 0.4,  // Vol of vol
        rho: -0.5,   // Correlation
        v0: 0.05,    // Initial variance (22.4% vol)
    };
    println!("  Initial κ (kappa): {:.4}", initial_params.kappa);
    println!("  Initial θ (theta): {:.4}", initial_params.theta);
    println!("  Initial σ (sigma): {:.4}", initial_params.sigma);
    println!("  Initial ρ (rho):   {:.4}", initial_params.rho);
    println!("  Initial v₀:        {:.4}\n", initial_params.v0);

    // Step 4: Create calibrator with custom settings
    println!("Creating calibrator...");
    let calibrator = HestonCalibrator::new(gpu_pricer, market_options, initial_params)?
        .with_bounds(ParameterBounds::default())?
        .with_max_iterations(100)
        .with_tolerance(1e-6);
    println!("✓ Calibrator configured\n");

    // Step 5: Run calibration
    println!("Running calibration (this may take 1-5 seconds)...");
    let start = std::time::Instant::now();
    let result = calibrator.calibrate()?;
    let elapsed = start.elapsed();
    println!("✓ Calibration complete in {:.2}s\n", elapsed.as_secs_f64());

    // Step 6: Display results
    print_calibration_results(&result);

    // Step 7: Validate calibration quality
    println!("\n=== Calibration Quality ===");
    let rmse = result.rmse();
    let acceptable = result.is_acceptable(1.0);

    println!("RMSE: {:.6}", rmse);
    println!(
        "Mean Error per Option: {:.6}",
        result.mean_error_per_option()
    );
    println!(
        "Quality: {}",
        if acceptable {
            "✓ ACCEPTABLE"
        } else {
            "⚠ POOR"
        }
    );

    if let Some(grad_norm) = result.gradient_norm {
        println!("Gradient Norm: {:.2e}", grad_norm);
    }

    Ok(())
}

#[cfg(feature = "heston")]
fn generate_synthetic_market_data() -> Vec<kimsfinance_core::quantitative::heston::OptionQuote> {
    use kimsfinance_core::quantitative::heston::{OptionQuote, OptionType};

    // Generate 20 options with various strikes around ATM
    let spot = 50000.0;
    let strikes = vec![
        45000.0, 46000.0, 47000.0, 48000.0, 49000.0, 50000.0, // ATM
        51000.0, 52000.0, 53000.0, 54000.0, 55000.0, 46500.0, 47500.0, 48500.0, 49500.0, 50500.0,
        51500.0, 52500.0, 53500.0, 54500.0,
    ];

    let expiration = chrono::Utc::now().timestamp() + (30 * 24 * 3600); // 30 days

    strikes
        .into_iter()
        .map(|strike| {
            // Synthetic "market" prices based on simple heuristics
            let moneyness = strike / spot;
            let intrinsic = (spot - strike).max(0.0);
            let time_value = 2000.0 * (1.0 - (moneyness - 1.0).abs());
            let mid_price = intrinsic + time_value.max(500.0);

            OptionQuote {
                underlying: "BTC".to_string(),
                strike,
                expiration,
                option_type: OptionType::Call,
                spot_price: spot,
                risk_free_rate: 0.05,
                bid: Some(mid_price * 0.98),
                ask: Some(mid_price * 1.02),
                last: None,
                implied_vol: Some(0.8),
                volume: 100.0,
                open_interest: 500.0,
                greeks: None,
            }
        })
        .collect()
}

#[cfg(feature = "heston")]
fn print_calibration_results(result: &kimsfinance_core::quantitative::heston::CalibrationResult) {
    println!("=== Calibration Results ===");
    println!("\nOptimized Parameters:");
    println!(
        "  κ (kappa):  {:.6}  [mean reversion speed]",
        result.params.kappa
    );
    println!(
        "  θ (theta):  {:.6}  [long-term variance, vol={:.2}%]",
        result.params.theta,
        result.params.long_term_vol() * 100.0
    );
    println!("  σ (sigma):  {:.6}  [vol of vol]", result.params.sigma);
    println!("  ρ (rho):    {:.6}  [correlation]", result.params.rho);
    println!(
        "  v₀:         {:.6}  [initial variance, vol={:.2}%]",
        result.params.v0,
        result.params.current_vol() * 100.0
    );

    println!("\nOptimization Statistics:");
    println!("  Iterations:     {}", result.iterations);
    println!("  Converged:      {}", result.converged);
    println!("  Final SSE:      {:.6}", result.final_error);
    println!("  Options Used:   {}", result.n_options);
}

#[cfg(not(feature = "heston"))]
fn main() {
    eprintln!("This example requires the 'heston' feature flag.");
    eprintln!("Run with: cargo run --example calibrate_heston --features heston");
    std::process::exit(1);
}
