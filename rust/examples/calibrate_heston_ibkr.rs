//! Calibrate Heston model using real IBKR options data
//!
//! Requirements:
//! - IBKR paper trading running (port 7497 or 4002)
//! - GPU available
//! - Market data subscriptions active
//!
//! Run: cargo run --example calibrate_heston_ibkr --features heston,data-ibkr --release

use kimsfinance_core::data::OptionsDataProvider;
use kimsfinance_core::data::ibkr::{IbkrConfig, IbkrConnector};
use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
use kimsfinance_core::quantitative::heston::{HestonCalibrator, HestonParams};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Heston Calibration with IBKR Data ===\n");

    // Step 1: Connect to IBKR
    println!("Step 1/5: Connecting to IBKR paper trading...");
    let config = IbkrConfig::default();
    let connector = IbkrConnector::connect(config).await?;
    println!("✓ Connected to IBKR\n");

    // Step 2: Fetch real options data
    println!("Step 2/5: Fetching AAPL options from IBKR...");
    println!("(This may take 30-60 seconds)\n");

    let mut options = connector.fetch_options_chain("AAPL").await?;

    println!("✓ Received {} options", options.len());

    // Filter for high-quality data for calibration
    println!("\nFiltering for calibration quality:");
    println!("- Must have implied volatility");
    println!("- Must have bid and ask prices");
    println!("- Volume > 10 (liquid)");
    println!("- Bid-ask spread < 20% (reasonable pricing)");

    options.retain(|opt| {
        if let (Some(bid), Some(ask), Some(_iv)) = (opt.bid, opt.ask, opt.implied_vol) {
            let mid = (bid + ask) / 2.0;
            let spread_pct = if mid > 0.0 { (ask - bid) / mid } else { 1.0 };

            opt.volume > 10.0 && spread_pct < 0.20
        } else {
            false
        }
    });

    println!(
        "✓ {} liquid options with good data quality\n",
        options.len()
    );

    if options.is_empty() {
        eprintln!("✗ No suitable options found for calibration");
        eprintln!("\nThis may be because:");
        eprintln!("1. Market is closed (try during trading hours)");
        eprintln!("2. Market data subscription not active");
        eprintln!("3. Not enough liquid options at this time");
        return Ok(());
    }

    // Display sample data
    println!("Sample calibration data (first 5 options):");
    println!("{:-<100}", "");
    println!(
        "{:<12} {:<8} {:<10} {:<10} {:<10} {:<10}",
        "Expiration", "Type", "Strike", "Mid Price", "IV", "Volume"
    );
    println!("{:-<100}", "");

    for opt in options.iter().take(5) {
        let expiration = chrono::DateTime::from_timestamp(opt.expiration, 0)
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        let option_type = match opt.option_type {
            kimsfinance_core::quantitative::heston::OptionType::Call => "CALL",
            kimsfinance_core::quantitative::heston::OptionType::Put => "PUT",
        };

        let mid = opt
            .bid
            .and_then(|b| opt.ask.map(|a| (b + a) / 2.0))
            .unwrap_or(0.0);
        let iv = opt.implied_vol.unwrap_or(0.0);

        println!(
            "{:<12} {:<8} ${:<9.2} ${:<9.2} {:<9.1}% {:<10.0}",
            expiration,
            option_type,
            opt.strike,
            mid,
            iv * 100.0,
            opt.volume
        );
    }
    println!("{:-<100}\n", "");

    // Step 3: Setup GPU calibration
    println!("Step 3/5: Initializing GPU device...");
    let device = Arc::new(GpuDevice::new()?);
    println!("✓ GPU device: {}", device.name());
    println!("✓ Compute capability: {}", device.compute_capability());

    let gpu_pricer = Arc::new(HestonGpuPricer::new(device, 4096)?);
    println!("✓ GPU pricer initialized with 4096 paths\n");

    // Step 4: Setup calibration with initial guess
    println!("Step 4/5: Setting up calibration...");

    // Use market-typical initial parameters for equity options
    let initial_params = HestonParams::new(
        2.0,  // kappa: mean reversion speed
        0.04, // theta: long-term variance (20% long-term vol)
        0.3,  // sigma: vol-of-vol
        -0.7, // rho: negative correlation (leverage effect)
        0.04, // v0: initial variance (20% current vol)
    )?;

    println!("Initial parameters:");
    println!(
        "  κ (kappa): {:.4} - Mean reversion speed",
        initial_params.kappa
    );
    println!(
        "  θ (theta): {:.4} - Long-term variance ({:.1}% LT vol)",
        initial_params.theta,
        (initial_params.theta.sqrt() * 100.0)
    );
    println!("  σ (sigma): {:.4} - Vol-of-vol", initial_params.sigma);
    println!("  ρ (rho):   {:.4} - Correlation", initial_params.rho);
    println!(
        "  v₀:        {:.4} - Initial variance ({:.1}% current vol)\n",
        initial_params.v0,
        (initial_params.v0.sqrt() * 100.0)
    );

    // Step 5: Calibrate
    println!("Step 5/5: Calibrating Heston model...");
    println!("(This may take 1-2 minutes depending on GPU and data size)\n");

    let calibrator = HestonCalibrator::new(gpu_pricer, options.clone(), initial_params);

    let result = calibrator.calibrate()?;

    // Display results
    println!("\n{:=<80}", "");
    println!("=== Calibration Results ===");
    println!("{:=<80}", "");

    println!("\nConvergence:");
    println!(
        "  Status: {}",
        if result.converged {
            "✓ CONVERGED"
        } else {
            "✗ NOT CONVERGED"
        }
    );
    println!("  Iterations: {}", result.iterations);
    println!("  Final Error: {:.6}", result.final_error);
    println!("  Time: {:.2}s", result.time_seconds);

    println!("\nCalibrated Parameters:");
    println!(
        "  κ (kappa): {:.4} - Mean reversion speed",
        result.params.kappa
    );
    println!(
        "  θ (theta): {:.4} - Long-term variance ({:.1}% LT vol)",
        result.params.theta,
        (result.params.theta.sqrt() * 100.0)
    );
    println!("  σ (sigma): {:.4} - Vol-of-vol", result.params.sigma);
    println!(
        "  ρ (rho):   {:.4} - Correlation (leverage effect)",
        result.params.rho
    );
    println!(
        "  v₀:        {:.4} - Initial variance ({:.1}% current vol)",
        result.params.v0,
        (result.params.v0.sqrt() * 100.0)
    );

    // Validate Feller condition
    let feller_lhs = 2.0 * result.params.kappa * result.params.theta;
    let feller_rhs = result.params.sigma * result.params.sigma;

    println!("\nFeller Condition (2κθ > σ²):");
    println!("  2κθ = {:.6}", feller_lhs);
    println!("  σ²  = {:.6}", feller_rhs);

    match result.params.validate() {
        Ok(_) => println!("  ✓ Feller condition satisfied (variance stays positive)"),
        Err(e) => println!("  ✗ Feller condition violated: {}", e),
    }

    // Compare with initial guess
    println!("\nParameter Changes from Initial Guess:");
    println!(
        "  κ: {:.4} → {:.4} ({:+.1}%)",
        initial_params.kappa,
        result.params.kappa,
        (result.params.kappa - initial_params.kappa) / initial_params.kappa * 100.0
    );
    println!(
        "  θ: {:.4} → {:.4} ({:+.1}%)",
        initial_params.theta,
        result.params.theta,
        (result.params.theta - initial_params.theta) / initial_params.theta * 100.0
    );
    println!(
        "  σ: {:.4} → {:.4} ({:+.1}%)",
        initial_params.sigma,
        result.params.sigma,
        (result.params.sigma - initial_params.sigma) / initial_params.sigma * 100.0
    );
    println!(
        "  ρ: {:.4} → {:.4} ({:+.1}%)",
        initial_params.rho,
        result.params.rho,
        (result.params.rho - initial_params.rho) / initial_params.rho.abs() * 100.0
    );
    println!(
        "  v₀: {:.4} → {:.4} ({:+.1}%)",
        initial_params.v0,
        result.params.v0,
        (result.params.v0 - initial_params.v0) / initial_params.v0 * 100.0
    );

    // Interpretation
    println!("\n=== Interpretation ===");

    if result.params.rho < -0.5 {
        println!(
            "✓ Strong negative correlation (ρ={:.2}) indicates leverage effect:",
            result.params.rho
        );
        println!("  - Volatility increases when stock price drops");
        println!("  - Typical for equity options");
    }

    let mean_reversion_days = 1.0 / result.params.kappa * 252.0;
    println!(
        "✓ Mean reversion half-life: ~{:.0} trading days",
        mean_reversion_days * 0.693
    );

    let lt_vol_pct = result.params.theta.sqrt() * 100.0;
    let current_vol_pct = result.params.v0.sqrt() * 100.0;

    if current_vol_pct > lt_vol_pct {
        println!(
            "✓ Current vol ({:.1}%) above long-term ({:.1}%) - expect vol to decrease",
            current_vol_pct, lt_vol_pct
        );
    } else {
        println!(
            "✓ Current vol ({:.1}%) below long-term ({:.1}%) - expect vol to increase",
            current_vol_pct, lt_vol_pct
        );
    }

    if result.params.sigma > 0.5 {
        println!(
            "⚠ High vol-of-vol (σ={:.2}) - volatility is very volatile",
            result.params.sigma
        );
    }

    println!("\n{:=<80}", "");
    println!("✓ Calibration complete! Parameters saved in CalibrationResult");
    println!("{:=<80}\n", "");

    Ok(())
}
