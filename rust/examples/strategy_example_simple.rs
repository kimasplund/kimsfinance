//! Simple example of loading and analyzing historical options data
//!
//! This demonstrates:
//! - Loading historical options chains from parquet
//! - Filtering contracts by DTE and delta
//! - Analyzing spreads and liquidity
//!
//! Run with:
//! ```bash
//! cargo run --release --features data-downloaders --example strategy_example_simple
//! ```

use chrono::NaiveDate;
use kimsfinance_core::strategy::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Options Strategy Framework Demo ===\n");

    // Initialize data loader
    let data_dir = "data/yfinance/options_historical";
    println!("Loading data from: {}\n", data_dir);

    let mut loader = OptionsDataLoader::new(data_dir)?;

    // Get statistics
    println!("📊 Database Statistics:");
    let stats = loader.get_stats()?;
    for (symbol, count) in &stats {
        println!("  {}: {} days", symbol, count);
    }
    println!();

    // Load AAPL chain for a specific date
    let symbol = "AAPL";
    let date = NaiveDate::from_ymd_opt(2020, 6, 1).expect("Invalid date");

    println!("📈 Loading {} chain for {}...", symbol, date);
    let contracts = loader.load_chain(symbol, date)?;
    println!("  Total contracts: {}\n", contracts.len());

    // Analyze contracts
    analyze_chain(&contracts);

    // Filter for potential credit spread candidates
    println!("\n🎯 Credit Spread Candidates:");
    println!("  (30-45 DTE, 0.15-0.35 delta)");

    let filter = ContractFilter {
        option_type: Some(OptionType::Put),
        dte_range: Some((30, 45)),
        delta_range: Some((0.15, 0.35)),
        min_volume: Some(10.0),
        min_open_interest: Some(100.0),
        ..Default::default()
    };

    let filtered = loader.filter_contracts(&contracts, &filter);
    println!("  Found {} candidates\n", filtered.len());

    // Show top 10 by open interest
    let mut candidates = filtered.clone();
    candidates.sort_by(|a, b| b.open_interest.partial_cmp(&a.open_interest).unwrap());

    println!("  Top 10 by Open Interest:");
    println!(
        "  {:>6} {:>10} {:>8} {:>8} {:>10} {:>12}",
        "Strike", "Mid", "Delta", "DTE", "Volume", "OI"
    );
    println!("  {}", "-".repeat(64));

    for contract in candidates.iter().take(10) {
        let delta = contract.delta.unwrap_or(0.0);
        println!(
            "  ${:>5.2} ${:>9.2} {:>7.3} {:>7} {:>10.0} {:>12.0}",
            contract.strike,
            contract.mid_price(),
            delta,
            contract.dte,
            contract.volume,
            contract.open_interest
        );
    }

    // Demo: Analyze a potential bull put spread
    if candidates.len() >= 2 {
        println!("\n💡 Example Bull Put Spread:");
        let short_put = &candidates[0]; // Higher strike (more premium)
        let long_put = &candidates[1]; // Lower strike (protection)

        let credit = short_put.mid_price() - long_put.mid_price();
        let width = (short_put.strike - long_put.strike).abs();
        let max_risk = width - credit;
        let risk_reward = max_risk / credit;

        println!(
            "  Short ${:.2} PUT @ ${:.2} premium",
            short_put.strike,
            short_put.mid_price()
        );
        println!(
            "  Long  ${:.2} PUT @ ${:.2} premium",
            long_put.strike,
            long_put.mid_price()
        );
        println!(
            "  Net Credit: ${:.2} (per contract = ${})",
            credit,
            credit * 100.0
        );
        println!(
            "  Max Risk:   ${:.2} (per contract = ${})",
            max_risk,
            max_risk * 100.0
        );
        println!("  Risk/Reward: {:.2}:1", risk_reward);
        println!(
            "  Win Prob (est): {:.1}%",
            (1.0 - short_put.delta.unwrap_or(0.3).abs()) * 100.0
        );
    }

    Ok(())
}

fn analyze_chain(contracts: &[OptionContract]) {
    let calls: Vec<_> = contracts
        .iter()
        .filter(|c| c.option_type == OptionType::Call)
        .collect();
    let puts: Vec<_> = contracts
        .iter()
        .filter(|c| c.option_type == OptionType::Put)
        .collect();

    println!("📋 Chain Analysis:");
    println!("  Calls: {}", calls.len());
    println!("  Puts:  {}", puts.len());

    // Expirations
    let mut expirations: Vec<_> = contracts.iter().map(|c| c.expiration).collect();
    expirations.sort();
    expirations.dedup();
    println!("  Expirations: {}", expirations.len());

    // DTE range
    if let (Some(min_dte), Some(max_dte)) = (
        contracts.iter().map(|c| c.dte).min(),
        contracts.iter().map(|c| c.dte).max(),
    ) {
        println!("  DTE range: {} to {}", min_dte, max_dte);
    }

    // Strike range
    if let (Some(min_strike), Some(max_strike)) = (
        contracts
            .iter()
            .map(|c| c.strike)
            .min_by(|a, b| a.partial_cmp(b).unwrap()),
        contracts
            .iter()
            .map(|c| c.strike)
            .max_by(|a, b| a.partial_cmp(b).unwrap()),
    ) {
        println!("  Strike range: ${:.2} to ${:.2}", min_strike, max_strike);
    }

    // Average volume and OI
    let avg_volume: f64 = contracts.iter().map(|c| c.volume).sum::<f64>() / contracts.len() as f64;
    let avg_oi: f64 =
        contracts.iter().map(|c| c.open_interest).sum::<f64>() / contracts.len() as f64;

    println!("  Avg Volume: {:.0}", avg_volume);
    println!("  Avg Open Interest: {:.0}", avg_oi);

    // Liquidity score (contracts with tight spreads and high volume)
    let liquid_contracts = contracts
        .iter()
        .filter(|c| {
            c.spread_pct() < 10.0 && // Spread < 10% of mid
            c.volume > 10.0 &&
            c.open_interest > 100.0
        })
        .count();

    println!(
        "  Liquid contracts: {} ({:.1}%)",
        liquid_contracts,
        (liquid_contracts as f64 / contracts.len() as f64) * 100.0
    );
}
