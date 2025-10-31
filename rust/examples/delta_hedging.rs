//! Delta Hedging Strategy Example
//!
//! Demonstrates portfolio delta hedging using Greeks calculation.
//! Maintains delta-neutral portfolio by hedging with underlying shares.
//!
//! # Strategy
//!
//! 1. Calculate Greeks for each option position
//! 2. Compute portfolio delta (sum of individual deltas)
//! 3. Hedge to target delta (typically 0 for delta-neutral)
//!
//! # Run
//!
//! ```bash
//! cargo run --example delta_hedging --features gpu --release
//! ```

use chrono::Utc;
use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
use kimsfinance_core::quantitative::heston::{
    DeltaHedgingStrategy, HestonGreeksCalculator, HestonParams, OptionPosition, OptionQuote,
    OptionType, PortfolioGreeks,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Delta Hedging Strategy Demo ===\n");

    // Initialize GPU device
    println!("Initializing GPU device...");
    let device = Arc::new(GpuDevice::new()?);
    println!("✓ GPU device initialized\n");

    // Create Heston pricer
    println!("Compiling Heston pricer...");
    let start = Instant::now();
    let pricer = HestonGpuPricer::new(device, 4096)?;
    println!("✓ Pricer compiled in {:?}\n", start.elapsed());

    // Create Greeks calculator
    let calculator = HestonGreeksCalculator::new(Arc::new(Mutex::new(pricer)));

    // Define Heston parameters (calibrated from market)
    let params = HestonParams::new(
        2.0,  // kappa
        0.04, // theta
        0.3,  // sigma
        -0.7, // rho
        0.04, // v0
    )?;

    println!("Heston Parameters:");
    println!("  Current Vol: {:.1}%", params.current_vol() * 100.0);
    println!("  Long-term Vol: {:.1}%\n", params.long_term_vol() * 100.0);

    // Create example portfolio
    let now = Utc::now().timestamp();
    let expiry_3months = now + (90 * 24 * 3600);

    println!("=== Portfolio Positions ===");
    let portfolio = vec![
        OptionPosition {
            option: create_option(48000.0, expiry_3months),
            quantity: 10, // Long 10 calls @ 48K
        },
        OptionPosition {
            option: create_option(50000.0, expiry_3months),
            quantity: -5, // Short 5 calls @ 50K
        },
        OptionPosition {
            option: create_option(52000.0, expiry_3months),
            quantity: 8, // Long 8 calls @ 52K
        },
    ];

    for (i, pos) in portfolio.iter().enumerate() {
        let action = if pos.quantity > 0 { "Long" } else { "Short" };
        println!(
            "Position {}: {} {} x ${:.0} Call",
            i + 1,
            action,
            pos.quantity.abs(),
            pos.option.strike
        );
    }
    println!();

    // Calculate Greeks for each position
    println!("Calculating Greeks...");
    let start = Instant::now();
    let options: Vec<_> = portfolio.iter().map(|p| p.option.clone()).collect();
    let greeks = calculator.calculate_greeks_batch(&params, &options)?;
    let elapsed = start.elapsed();
    println!("✓ Greeks calculated in {:?}\n", elapsed);

    // Display individual position Greeks
    println!("=== Position Greeks ===");
    println!(
        "{:<12} {:<8} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "Strike", "Qty", "Delta", "Gamma", "Vega", "Theta", "Rho"
    );
    println!("{}", "-".repeat(80));

    for (pos, greek) in portfolio.iter().zip(greeks.iter()) {
        println!(
            "${:<11.0} {:<8} {:<10.4} {:<10.6} {:<10.2} {:<10.2} {:<10.2}",
            pos.option.strike,
            pos.quantity,
            greek.delta.unwrap_or(0.0),
            greek.gamma.unwrap_or(0.0),
            greek.vega.unwrap_or(0.0),
            greek.theta.unwrap_or(0.0),
            greek.rho_greek.unwrap_or(0.0)
        );
    }
    println!();

    // Calculate portfolio-level Greeks
    let port_greeks = DeltaHedgingStrategy::calculate_portfolio_greeks(&portfolio, &greeks);

    println!("=== Portfolio Greeks (Aggregated) ===");
    println!("  Delta: {:.2}", port_greeks.delta);
    println!("  Gamma: {:.4}", port_greeks.gamma);
    println!("  Vega:  {:.2}", port_greeks.vega);
    println!("  Theta: {:.2} (per day)", port_greeks.theta);
    println!("  Rho:   {:.2}\n", port_greeks.rho);

    // Calculate hedge recommendation
    let strategy = DeltaHedgingStrategy::new(0.0); // Delta-neutral target
    let hedge = strategy.calculate_hedge(&portfolio, &greeks);

    println!("=== Hedge Recommendation ===");
    println!("  Current Delta: {:.2}", hedge.current_delta);
    println!("  Target Delta:  {:.2}", hedge.target_delta);
    println!(
        "  Action: {} {} BTC shares",
        if hedge.underlying_shares > 0 {
            "BUY"
        } else {
            "SELL"
        },
        hedge.underlying_shares.abs()
    );
    println!("  Rationale: {}\n", hedge.reason);

    // Simulate hedge execution
    println!("=== After Hedging ===");
    let hedged_delta = hedge.current_delta + hedge.underlying_shares as f64;
    println!(
        "  Portfolio Delta: {:.2} → {:.2}",
        hedge.current_delta, hedged_delta
    );
    println!(
        "  Status: {}",
        if hedged_delta.abs() < 0.1 {
            "✓ Delta-neutral achieved"
        } else {
            "⚠ Residual delta remains"
        }
    );
    println!();

    // Risk analysis
    println!("=== Risk Analysis ===");
    if port_greeks.gamma.abs() > 0.01 {
        println!("  ⚠ High Gamma: Portfolio delta sensitive to price moves");
        println!("    Consider re-hedging frequently or gamma hedging");
    } else {
        println!("  ✓ Low Gamma: Delta hedge stable");
    }

    if port_greeks.vega.abs() > 100.0 {
        println!("  ⚠ High Vega: Portfolio exposed to volatility changes");
        println!("    Consider vega hedging with additional options");
    } else {
        println!("  ✓ Low Vega: Limited volatility exposure");
    }

    if port_greeks.theta < -100.0 {
        println!(
            "  ⚠ High Theta Decay: Losing ${:.2}/day from time decay",
            -port_greeks.theta
        );
        println!("    Monitor daily and consider adjusting positions");
    } else if port_greeks.theta > 0.0 {
        println!(
            "  ✓ Positive Theta: Earning ${:.2}/day from time decay",
            port_greeks.theta
        );
    } else {
        println!("  ✓ Minimal Theta: Limited time decay");
    }
    println!();

    // Monitoring recommendations
    println!("=== Monitoring Recommendations ===");
    println!("1. Re-calculate Greeks: Every 4-8 hours or after 1% spot move");
    println!("2. Re-hedge Delta: When |delta| > 10 (or your risk tolerance)");
    println!("3. Vega Monitoring: Watch for IV regime changes");
    println!("4. Gamma Management: Increase hedge frequency if gamma > 0.05");
    println!("5. Theta Decay: Daily P&L attribution to time decay\n");

    println!("=== Notes ===");
    println!("1. Greeks are calculated using finite differences (GPU-accelerated)");
    println!("2. In production, use real-time market data and Greeks");
    println!("3. Consider transaction costs when re-hedging");
    println!("4. Implement automated delta monitoring and hedging");
    println!("5. Use multiple Greeks for comprehensive risk management\n");

    Ok(())
}

/// Create option quote for testing
fn create_option(strike: f64, expiration: i64) -> OptionQuote {
    let spot = 48000.0;
    let ttm = 0.25;
    let iv = 0.2;
    let intrinsic = (spot - strike).max(0.0);
    let time_value = iv * spot * ttm.sqrt() * 0.4;
    let mid_price = intrinsic + time_value;

    OptionQuote {
        underlying: "BTC".to_string(),
        strike,
        expiration,
        option_type: OptionType::Call,
        spot_price: spot,
        risk_free_rate: 0.05,
        bid: Some(mid_price - 50.0),
        ask: Some(mid_price + 50.0),
        last: Some(mid_price),
        implied_vol: Some(iv),
        volume: 100.0,
        open_interest: 500.0,
        greeks: None,
    }
}
