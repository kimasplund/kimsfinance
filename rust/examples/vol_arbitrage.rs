//! Vol Arbitrage Strategy Example
//!
//! Demonstrates volatility arbitrage using calibrated Heston model.
//! Identifies options mispriced based on model IV vs market IV.
//!
//! # Strategy
//!
//! 1. Calculate model IV from calibrated Heston parameters
//! 2. Compare with market implied volatility
//! 3. Generate BUY signal if market IV < model IV (underpriced)
//! 4. Generate SELL signal if market IV > model IV (overpriced)
//!
//! # Run
//!
//! ```bash
//! cargo run --example vol_arbitrage --features gpu --release
//! ```

use chrono::Utc;
use kimsfinance_core::quantitative::heston::{
    HestonParams, OptionQuote, OptionType, TradeSignal, VolArbitrageStrategy,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vol Arbitrage Strategy Demo ===\n");

    // Define calibrated Heston parameters (example for BTC)
    // In production, these would come from Heston calibrator
    let params = HestonParams::new(
        2.0,  // kappa: mean reversion speed
        0.04, // theta: long-term variance (20% vol)
        0.3,  // sigma: vol of vol
        -0.7, // rho: correlation (leverage effect)
        0.04, // v0: initial variance (20% current vol)
    )?;

    println!("Calibrated Heston Parameters:");
    println!("  κ (kappa):  {:.2}", params.kappa);
    println!(
        "  θ (theta):  {:.4} ({:.1}% vol)",
        params.theta,
        params.long_term_vol() * 100.0
    );
    println!("  σ (sigma):  {:.2}", params.sigma);
    println!("  ρ (rho):    {:.2}", params.rho);
    println!(
        "  v₀ (v0):    {:.4} ({:.1}% vol)",
        params.v0,
        params.current_vol() * 100.0
    );
    println!("\nModel IV: {:.1}%\n", params.current_vol() * 100.0);

    // Generate synthetic options chain (normally from exchange API)
    let now = Utc::now().timestamp();
    let expiry_3months = now + (90 * 24 * 3600);

    // Create options with varying market IVs
    let options = vec![
        // Underpriced options (market IV < model IV)
        create_option(48000.0, 0.15, expiry_3months), // 15% IV
        create_option(49000.0, 0.16, expiry_3months), // 16% IV
        create_option(47000.0, 0.14, expiry_3months), // 14% IV
        // Fairly priced options (market IV ≈ model IV)
        create_option(50000.0, 0.20, expiry_3months), // 20% IV
        create_option(51000.0, 0.21, expiry_3months), // 21% IV
        // Overpriced options (market IV > model IV)
        create_option(52000.0, 0.26, expiry_3months), // 26% IV
        create_option(53000.0, 0.28, expiry_3months), // 28% IV
        create_option(54000.0, 0.30, expiry_3months), // 30% IV
        // Extreme mispricings
        create_option(45000.0, 0.12, expiry_3months), // 12% IV (8pp underpriced)
        create_option(55000.0, 0.35, expiry_3months), // 35% IV (15pp overpriced)
    ];

    println!(
        "Analyzing {} options from synthetic chain...\n",
        options.len()
    );

    // Create vol arbitrage strategy with 5% threshold
    let strategy = VolArbitrageStrategy::new(5.0); // 5 percentage points

    // Generate trade signals
    let start = Instant::now();
    let signals = strategy.generate_signals(&options, &params);
    let elapsed = start.elapsed();

    println!("=== Vol Arbitrage Opportunities ===");
    println!("Found {} signals in {:?}\n", signals.len(), elapsed);

    if signals.is_empty() {
        println!("No arbitrage opportunities found (all options fairly priced)\n");
        return Ok(());
    }

    // Display top 10 opportunities
    println!("Top Opportunities (sorted by edge):");
    println!(
        "{:<8} {:<10} {:<12} {:<15} {:<8}",
        "Action", "Strike", "Market IV", "Model IV", "Edge"
    );
    println!("{}", "-".repeat(65));

    for (i, signal) in signals.iter().take(10).enumerate() {
        match signal {
            TradeSignal::Buy { option, edge, .. } => {
                let market_iv = option.implied_vol.unwrap_or(0.0);
                println!(
                    "{:<8} ${:<9.0} {:<11.1}% {:<14.1}% {:<7.1}pp",
                    "BUY",
                    option.strike,
                    market_iv * 100.0,
                    params.current_vol() * 100.0,
                    edge
                );
            }
            TradeSignal::Sell { option, edge, .. } => {
                let market_iv = option.implied_vol.unwrap_or(0.0);
                println!(
                    "{:<8} ${:<9.0} {:<11.1}% {:<14.1}% {:<7.1}pp",
                    "SELL",
                    option.strike,
                    market_iv * 100.0,
                    params.current_vol() * 100.0,
                    edge
                );
            }
        }

        if i == 4 && signals.len() > 5 {
            println!("{}", "-".repeat(65));
        }
    }

    println!();

    // Summary statistics
    let buy_signals: Vec<_> = signals
        .iter()
        .filter(|s| matches!(s, TradeSignal::Buy { .. }))
        .collect();
    let sell_signals: Vec<_> = signals
        .iter()
        .filter(|s| matches!(s, TradeSignal::Sell { .. }))
        .collect();

    println!("=== Summary ===");
    println!("Total signals: {}", signals.len());
    println!("  Buy signals:  {} (market underpriced)", buy_signals.len());
    println!("  Sell signals: {} (market overpriced)", sell_signals.len());

    // Calculate average edge
    let total_edge: f64 = signals
        .iter()
        .map(|s| match s {
            TradeSignal::Buy { edge, .. } | TradeSignal::Sell { edge, .. } => *edge,
        })
        .sum();
    let avg_edge = total_edge / signals.len() as f64;
    println!("  Average edge: {:.2}pp\n", avg_edge);

    // Best opportunity
    if let Some(best) = signals.first() {
        println!("=== Best Opportunity ===");
        match best {
            TradeSignal::Buy {
                option,
                reason,
                edge,
            } => {
                println!("Action: BUY");
                println!("Strike: ${:.0}", option.strike);
                println!("Edge: {:.2}pp", edge);
                println!("Reason: {}", reason);
            }
            TradeSignal::Sell {
                option,
                reason,
                edge,
            } => {
                println!("Action: SELL");
                println!("Strike: ${:.0}", option.strike);
                println!("Edge: {:.2}pp", edge);
                println!("Reason: {}", reason);
            }
        }
    }

    println!("\n=== Notes ===");
    println!("1. This is a synthetic example using dummy prices");
    println!("2. In production, integrate with exchange API (e.g., Deribit)");
    println!("3. Calibrate Heston parameters from market data");
    println!("4. Consider transaction costs, slippage, and risk limits");
    println!("5. Implement position sizing and portfolio risk management\n");

    Ok(())
}

/// Create option quote with specified parameters
fn create_option(strike: f64, market_iv: f64, expiration: i64) -> OptionQuote {
    // Simple Black-Scholes approximation for mid price (placeholder)
    let spot = 48000.0;
    let ttm = 0.25_f64; // 3 months
    let intrinsic = (spot - strike).max(0.0);
    let time_value = market_iv * spot * ttm.sqrt() * 0.4; // Rough approximation
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
        implied_vol: Some(market_iv),
        volume: 100.0,
        open_interest: 500.0,
        greeks: None,
    }
}
