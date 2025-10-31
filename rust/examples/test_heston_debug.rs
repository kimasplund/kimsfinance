//! Minimal Heston characteristic function debug test
//!
//! Run with: cargo run --example test_heston_debug --features heston --release

use kimsfinance_core::gpu::GpuDevice;
use kimsfinance_core::gpu::heston_pricing::HestonGpuPricer;
use kimsfinance_core::quantitative::heston::{HestonParams, OptionQuote, OptionType};
use std::sync::Arc;
use std::time::SystemTime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Heston Characteristic Function Debug Test ===\n");

    // Initialize GPU
    let device = Arc::new(GpuDevice::new().map_err(|e| {
        eprintln!("GPU initialization failed: {:?}", e);
        eprintln!("This example requires GPU support. Run with --features heston");
        e
    })?);

    // Create pricer with small FFT size for faster debug
    let mut pricer = HestonGpuPricer::new(device, 16, 1)?; // 16 points, 1 option

    // Heston params
    let params = HestonParams::new(
        5.0,  // kappa
        0.04, // theta (20% long-term vol)
        0.3,  // sigma (vol of vol)
        -0.5, // rho (correlation)
        0.04, // v0 (20% current vol)
    )?;

    // Single ATM call option
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs() as i64;
    let one_year_later = now + (365 * 24 * 3600);

    let option = OptionQuote {
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

    let options = vec![option];

    // Price - this will trigger the kernel and print debug output
    println!("Calling price_options...\n");
    let prices = pricer.price_options(&params, &options)?;

    println!("\n=== Results ===");
    println!("Price: ${:.4}", prices[0]);

    Ok(())
}
