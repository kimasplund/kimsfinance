use chrono::Utc;
use kimsfinance_core::gpu::GpuDevice;
use kimsfinance_core::gpu::heston_pricing::HestonGpuPricer;
use kimsfinance_core::quantitative::heston::{HestonParams, OptionQuote, OptionType};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Heston GPU buffer initialization...");

    let device = Arc::new(GpuDevice::new()?);
    let mut pricer = HestonGpuPricer::new(device, 4096, 10)?;

    let params = HestonParams::new(
        2.0,  // kappa
        0.04, // theta
        0.3,  // sigma
        -0.7, // rho
        0.04, // v0
    )?;

    let options = vec![OptionQuote {
        underlying: "SPX".to_string(),
        strike: 100.0,
        expiration: Utc::now().timestamp() + 86400 * 30,
        option_type: OptionType::Call,
        spot_price: 100.0,
        risk_free_rate: 0.05,
        bid: None,
        ask: None,
        last: None,
        implied_vol: None,
        volume: 0.0,
        open_interest: 0.0,
        greeks: None,
    }];

    println!("\nPricing {} options...\n", options.len());
    let prices = pricer.price_options(&params, &options)?;

    println!("\nResults:");
    for (i, price) in prices.iter().enumerate() {
        println!("  Option {}: ${:.4}", i, price);
    }

    Ok(())
}
