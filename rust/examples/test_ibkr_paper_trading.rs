//! Test IBKR paper trading connection
//!
//! Requirements:
//! - TWS or IB Gateway running on paper trading account
//! - Port 7497 (or 4002) accessible
//! - Market data subscriptions active
//!
//! Run: cargo run --example test_ibkr_paper_trading --features data-ibkr --release

use kimsfinance_core::data::OptionsDataProvider;
use kimsfinance_core::data::ibkr::{IbkrConfig, IbkrConnector};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IBKR Paper Trading Test ===\n");

    // 1. Connect to paper trading
    let config = IbkrConfig::default(); // 127.0.0.1:7497

    println!(
        "Connecting to IBKR paper trading at {}:{}...",
        config.host, config.port
    );

    let connector = IbkrConnector::connect(config).await.map_err(|e| {
        eprintln!("✗ Connection failed: {}", e);
        eprintln!("\nTroubleshooting:");
        eprintln!("1. Check if TWS or IB Gateway is running");
        eprintln!("2. Verify port 7497 is configured for paper trading");
        eprintln!("   (File → Global Configuration → API → Settings)");
        eprintln!("3. Ensure 'Enable ActiveX and Socket Clients' is checked");
        eprintln!("4. Try port 4002 if 7497 doesn't work (edit IbkrConfig)");
        e
    })?;

    println!("✓ Connected successfully!\n");

    // 2. Test fetching AAPL options
    println!("Fetching AAPL option chain...");
    println!("(This may take 30-60 seconds depending on market conditions)\n");

    let options = connector.fetch_options_chain("AAPL").await.map_err(|e| {
        eprintln!("✗ Error: {}", e);
        eprintln!("\nNote: This may be due to:");
        eprintln!("1. Market data subscription not active");
        eprintln!("2. No options available for AAPL (unlikely)");
        eprintln!("3. API rate limiting");
        eprintln!("4. Market closed (try during trading hours)");
        e
    })?;

    println!("✓ Found {} options", options.len());

    // Show first 10 options
    println!("\nSample options (first 10):");
    println!("{:-<100}", "");
    println!(
        "{:<12} {:<8} {:<10} {:<12} {:<12} {:<12} {:<10}",
        "Expiration", "Type", "Strike", "Bid", "Ask", "IV", "Volume"
    );
    println!("{:-<100}", "");

    for (_i, opt) in options.iter().take(10).enumerate() {
        let expiration = chrono::DateTime::from_timestamp(opt.expiration, 0)
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        let option_type = match opt.option_type {
            kimsfinance_core::quantitative::heston::OptionType::Call => "CALL",
            kimsfinance_core::quantitative::heston::OptionType::Put => "PUT",
        };

        let bid = opt
            .bid
            .map(|b| format!("${:.2}", b))
            .unwrap_or_else(|| "-".to_string());
        let ask = opt
            .ask
            .map(|a| format!("${:.2}", a))
            .unwrap_or_else(|| "-".to_string());
        let iv = opt
            .implied_vol
            .map(|v| format!("{:.1}%", v * 100.0))
            .unwrap_or_else(|| "-".to_string());

        println!(
            "{:<12} {:<8} ${:<9.2} {:<12} {:<12} {:<12} {:<10.0}",
            expiration, option_type, opt.strike, bid, ask, iv, opt.volume
        );
    }

    if options.len() > 10 {
        println!("{:-<100}", "");
        println!("... and {} more options", options.len() - 10);
    }

    println!("\n=== Statistics ===");

    let calls = options
        .iter()
        .filter(|opt| {
            matches!(
                opt.option_type,
                kimsfinance_core::quantitative::heston::OptionType::Call
            )
        })
        .count();
    let puts = options
        .iter()
        .filter(|opt| {
            matches!(
                opt.option_type,
                kimsfinance_core::quantitative::heston::OptionType::Put
            )
        })
        .count();

    println!("Calls: {}", calls);
    println!("Puts: {}", puts);

    let with_iv = options
        .iter()
        .filter(|opt| opt.implied_vol.is_some())
        .count();
    let with_greeks = options.iter().filter(|opt| opt.greeks.is_some()).count();
    let with_volume = options.iter().filter(|opt| opt.volume > 0.0).count();

    println!("With IV: {}", with_iv);
    println!("With Greeks: {}", with_greeks);
    println!("With Volume: {}", with_volume);

    if with_iv > 0 {
        let avg_iv = options
            .iter()
            .filter_map(|opt| opt.implied_vol)
            .sum::<f64>()
            / with_iv as f64;
        println!("Average IV: {:.1}%", avg_iv * 100.0);
    }

    println!("\n✓ IBKR integration test PASSED!");

    Ok(())
}
