//! Test GPU-accelerated Heston option pricer
//!
//! Run with: cargo run --example test_heston_pricer --features gpu --release

use chrono::Utc;
use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
use kimsfinance_core::quantitative::heston::{HestonParams, OptionQuote, OptionType};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Heston GPU Option Pricer Test ===\n");

    // Initialize GPU device
    println!("Initializing GPU device...");
    let device = Arc::new(GpuDevice::new()?);
    println!("✓ GPU device initialized\n");

    // Create Heston pricer with FFT size 4096
    println!("Compiling Heston characteristic function kernel...");
    let start = Instant::now();
    let mut pricer = HestonGpuPricer::new(device, 4096, 1024)?;
    let compile_time = start.elapsed();
    println!("✓ Kernel compiled in {:?}\n", compile_time);

    // Define Heston parameters (typical BTC parameters)
    let params = HestonParams::new(
        2.0,  // kappa: mean reversion speed
        0.04, // theta: long-term variance (20% vol)
        0.3,  // sigma: vol of vol
        -0.7, // rho: correlation (leverage effect)
        0.04, // v0: initial variance (20% current vol)
    )?;

    println!("Heston Parameters:");
    println!("  κ (kappa):  {:.2} (mean reversion speed)", params.kappa);
    println!(
        "  θ (theta):  {:.4} (long-term variance = {:.1}% vol)",
        params.theta,
        params.long_term_vol() * 100.0
    );
    println!("  σ (sigma):  {:.2} (vol of vol)", params.sigma);
    println!("  ρ (rho):    {:.2} (correlation)", params.rho);
    println!(
        "  v₀ (v0):    {:.4} (initial variance = {:.1}% vol)\n",
        params.v0,
        params.current_vol() * 100.0
    );

    // Current time for expiry calculation
    let now = Utc::now().timestamp();
    let expiry_3months = now + (90 * 24 * 3600); // 90 days from now

    // Test 1: Single option pricing
    println!("Test 1: Single Option Pricing");
    println!("-------------------------------");
    let single_option = OptionQuote {
        underlying: "BTC".to_string(),
        strike: 50000.0,
        expiration: expiry_3months,
        option_type: OptionType::Call,
        spot_price: 48000.0,
        risk_free_rate: 0.05,
        bid: Some(2000.0),
        ask: Some(2100.0),
        last: Some(2050.0),
        implied_vol: Some(0.8),
        volume: 100.0,
        open_interest: 500.0,
        greeks: None,
    };

    let start = Instant::now();
    let price = pricer.price_options(&params, &[single_option.clone()])?;
    let elapsed = start.elapsed();

    println!("Option: BTC Call");
    println!("  Strike: ${:.2}", single_option.strike);
    println!("  Spot: ${:.2}", single_option.spot_price);
    println!("  Expiry: {:.3} years", single_option.time_to_expiry(now));
    println!("  Calculated Price: ${:.2}", price[0]);
    println!(
        "  Time: {:?} ({:.3}ms)\n",
        elapsed,
        elapsed.as_secs_f64() * 1000.0
    );

    // Test 2: Batch pricing (10 options)
    println!("Test 2: Batch Pricing (10 options)");
    println!("-----------------------------------");
    let options_10: Vec<OptionQuote> = (45000..45010)
        .step_by(100)
        .map(|strike| OptionQuote {
            underlying: "BTC".to_string(),
            strike: strike as f64,
            expiration: expiry_3months,
            option_type: OptionType::Call,
            spot_price: 48000.0,
            risk_free_rate: 0.05,
            bid: Some(2000.0),
            ask: Some(2100.0),
            last: Some(2050.0),
            implied_vol: Some(0.8),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        })
        .collect();

    let start = Instant::now();
    let prices_10 = pricer.price_options(&params, &options_10)?;
    let elapsed = start.elapsed();

    println!(
        "Priced {} options in {:?} ({:.3}ms)",
        prices_10.len(),
        elapsed,
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "Per-option time: {:.3}μs\n",
        elapsed.as_secs_f64() * 1_000_000.0 / prices_10.len() as f64
    );

    // Test 3: Batch pricing (100 options)
    println!("Test 3: Batch Pricing (100 options)");
    println!("------------------------------------");
    let options_100: Vec<OptionQuote> = (40000..40100)
        .map(|strike| OptionQuote {
            underlying: "BTC".to_string(),
            strike: strike as f64,
            expiration: expiry_3months,
            option_type: OptionType::Call,
            spot_price: 48000.0,
            risk_free_rate: 0.05,
            bid: Some(2000.0),
            ask: Some(2100.0),
            last: Some(2050.0),
            implied_vol: Some(0.8),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        })
        .collect();

    let start = Instant::now();
    let prices_100 = pricer.price_options(&params, &options_100)?;
    let elapsed = start.elapsed();

    println!(
        "Priced {} options in {:?} ({:.3}ms)",
        prices_100.len(),
        elapsed,
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "Per-option time: {:.3}μs",
        elapsed.as_secs_f64() * 1_000_000.0 / prices_100.len() as f64
    );
    println!(
        "Target: <3ms ✓ {}",
        if elapsed.as_millis() < 3 {
            "(PASSED)"
        } else {
            "(NEEDS OPTIMIZATION)"
        }
    );
    println!();

    // Test 4: Larger batch (500 options)
    println!("Test 4: Batch Pricing (500 options)");
    println!("------------------------------------");
    let options_500: Vec<OptionQuote> = (40000..40500)
        .map(|strike| OptionQuote {
            underlying: "BTC".to_string(),
            strike: strike as f64,
            expiration: expiry_3months,
            option_type: OptionType::Call,
            spot_price: 48000.0,
            risk_free_rate: 0.05,
            bid: Some(2000.0),
            ask: Some(2100.0),
            last: Some(2050.0),
            implied_vol: Some(0.8),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        })
        .collect();

    let start = Instant::now();
    let prices_500 = pricer.price_options(&params, &options_500)?;
    let elapsed = start.elapsed();

    println!(
        "Priced {} options in {:?} ({:.3}ms)",
        prices_500.len(),
        elapsed,
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "Per-option time: {:.3}μs",
        elapsed.as_secs_f64() * 1_000_000.0 / prices_500.len() as f64
    );
    println!(
        "Target: <10ms ✓ {}",
        if elapsed.as_millis() < 10 {
            "(PASSED)"
        } else {
            "(NEEDS OPTIMIZATION)"
        }
    );
    println!();

    println!("=== All Tests Complete ===");

    Ok(())
}
