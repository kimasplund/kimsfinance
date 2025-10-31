//! Test FFT-based option pricing
//!
//! Validates that FFT pricing matches known Black-Scholes prices
//! when Heston parameters approach BS limit (σ→0, ρ→0)
//!
//! Run with: cargo run --example test_fft_pricing --features heston --release

use kimsfinance_core::gpu::GpuDevice;
use kimsfinance_core::gpu::heston_pricing::HestonGpuPricer;
use kimsfinance_core::quantitative::heston::{HestonParams, OptionQuote, OptionType};
use std::sync::Arc;
use std::time::SystemTime;

/// Black-Scholes call price formula (for validation)
fn black_scholes_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return (s - k).max(0.0);
    }

    // Normal CDF approximation (accurate to 1e-7)
    fn norm_cdf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let prob = d
            * t
            * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

        if x >= 0.0 { 1.0 - prob } else { prob }
    }

    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
}

/// Black-Scholes put price (put-call parity)
fn black_scholes_put(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    let call = black_scholes_call(s, k, t, r, sigma);
    call - s + k * (-r * t).exp()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FFT Pricing Validation ===\n");

    // Initialize GPU
    let device = Arc::new(GpuDevice::new().map_err(|e| {
        eprintln!("GPU initialization failed: {:?}", e);
        eprintln!("This example requires GPU support. Run with --features heston");
        e
    })?);

    // Create pricer with 4096 FFT points (good accuracy vs speed tradeoff)
    let mut pricer = HestonGpuPricer::new(device, 4096, 100)?;

    // Heston params approaching Black-Scholes limit
    // (very small vol-of-vol and zero correlation)
    let bs_params = HestonParams::new(
        5.0,   // kappa (fast mean reversion)
        0.04,  // theta (20% long-term vol)
        0.001, // sigma (very small vol of vol → BS limit)
        0.0,   // rho (zero correlation → BS limit)
        0.04,  // v0 (20% current vol)
    )?;

    // Test cases: ATM call and put
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs() as i64;
    let one_year_later = now + (365 * 24 * 3600);

    let atm_call = OptionQuote {
        underlying: "TEST".to_string(),
        strike: 100.0,
        spot_price: 100.0,
        expiration: one_year_later,
        option_type: OptionType::Call,
        risk_free_rate: 0.05,
        bid: None,
        ask: None,
        last: None,
        implied_vol: None,
        volume: 0.0,
        open_interest: 0.0,
        greeks: None,
    };

    let atm_put = OptionQuote {
        option_type: OptionType::Put,
        ..atm_call.clone()
    };

    // ITM call (S > K)
    let itm_call = OptionQuote {
        spot_price: 110.0,
        option_type: OptionType::Call,
        ..atm_call.clone()
    };

    // OTM put (S > K)
    let otm_put = OptionQuote {
        spot_price: 110.0,
        option_type: OptionType::Put,
        ..atm_call.clone()
    };

    let options = vec![
        atm_call.clone(),
        atm_put.clone(),
        itm_call.clone(),
        otm_put.clone(),
    ];

    // Price with Heston FFT
    println!("Pricing with Heston GPU FFT...");
    let heston_prices = pricer.price_options(&bs_params, &options)?;

    // Price with Black-Scholes (reference)
    let sigma = bs_params.v0.sqrt(); // 0.2 (20% vol)
    let tau = 1.0; // 1 year

    let bs_atm_call = black_scholes_call(100.0, 100.0, tau, 0.05, sigma);
    let bs_atm_put = black_scholes_put(100.0, 100.0, tau, 0.05, sigma);
    let bs_itm_call = black_scholes_call(110.0, 100.0, tau, 0.05, sigma);
    let bs_otm_put = black_scholes_put(110.0, 100.0, tau, 0.05, sigma);

    let bs_prices = vec![bs_atm_call, bs_atm_put, bs_itm_call, bs_otm_put];
    let labels = vec!["ATM Call", "ATM Put", "ITM Call", "OTM Put"];

    // Compare results
    println!(
        "\n{:<12} {:>12} {:>12} {:>10} {:>8}",
        "Option", "Heston FFT", "Black-Scholes", "Error", "Error %"
    );
    println!("{}", "-".repeat(60));

    let mut max_error_pct: f64 = 0.0;
    let mut all_passed = true;

    for i in 0..options.len() {
        let heston_price = heston_prices[i];
        let bs_price = bs_prices[i];
        let error = (heston_price - bs_price).abs();
        let error_pct = if bs_price > 0.0 {
            (error / bs_price * 100.0).abs()
        } else {
            0.0
        };

        max_error_pct = max_error_pct.max(error_pct);

        println!(
            "{:<12} ${:>11.4} ${:>11.4} ${:>9.4} {:>7.2}%",
            labels[i], heston_price, bs_price, error, error_pct
        );

        // Validate: error should be < 1%
        if error_pct > 1.0 {
            all_passed = false;
        }
    }

    println!("\n{}", "=".repeat(60));

    // Put-call parity check
    let parity_lhs = heston_prices[0] - heston_prices[1]; // C - P
    let parity_rhs = options[0].spot_price - options[0].strike * (-0.05 * tau).exp(); // S - K*exp(-rT)
    let parity_error = (parity_lhs - parity_rhs).abs();

    println!("\nPut-Call Parity Check:");
    println!("  C - P = ${:.4}", parity_lhs);
    println!("  S - K·exp(-r·T) = ${:.4}", parity_rhs);
    println!("  Error = ${:.4}", parity_error);

    if parity_error < 0.1 {
        println!("  ✓ Put-call parity satisfied");
    } else {
        println!("  ✗ Put-call parity violated!");
        all_passed = false;
    }

    // Final verdict
    println!("\n{}", "=".repeat(60));
    println!("Maximum error: {:.3}%", max_error_pct);

    if all_passed && max_error_pct < 1.0 {
        println!("\n✓ FFT pricing validated successfully!");
        println!("  - All errors < 1%");
        println!("  - Put-call parity satisfied");
        println!("  - Ready for calibration");
        Ok(())
    } else {
        println!("\n✗ FFT pricing validation failed!");
        if max_error_pct >= 1.0 {
            println!(
                "  - Error too high: {:.3}% (threshold: 1.0%)",
                max_error_pct
            );
        }
        if parity_error >= 0.1 {
            println!("  - Put-call parity violated");
        }
        Err("Validation failed".into())
    }
}
