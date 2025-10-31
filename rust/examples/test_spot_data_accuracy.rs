//! Test Spot Data Accuracy
//!
//! Verifies that spot prices from OHLCV data match option strikes within 1%.
//! Also demonstrates ATR and Bollinger Band calculations.
//!
//! Run with:
//! ```bash
//! cargo run --release --features data-downloaders --example test_spot_data_accuracy
//! ```

use chrono::NaiveDate;
use kimsfinance_core::strategy::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Spot Data Accuracy Test ===\n");

    // Configuration
    let options_dir = "data/yfinance/options_historical";
    let spot_dir = "../data/yfinance/ohlcv";
    let symbol = "AAPL";

    // Test dates (various points in time)
    let test_dates = vec![
        NaiveDate::from_ymd_opt(2020, 1, 15).unwrap(),
        NaiveDate::from_ymd_opt(2020, 6, 1).unwrap(),
        NaiveDate::from_ymd_opt(2021, 1, 15).unwrap(),
        NaiveDate::from_ymd_opt(2022, 1, 15).unwrap(),
        NaiveDate::from_ymd_opt(2023, 1, 15).unwrap(),
    ];

    println!("Configuration:");
    println!("  Symbol: {}", symbol);
    println!("  Test Dates: {}", test_dates.len());
    println!();

    // Initialize loaders
    println!("Loading data...");
    let mut options_loader = OptionsDataLoader::new(options_dir)?;
    let mut spot_loader = SpotDataLoader::new(spot_dir)?;

    // Get spot data range
    let (min_date, max_date) = spot_loader.get_date_range(symbol)?;
    println!("  Spot data range: {} to {}", min_date, max_date);

    // Get options data stats
    let stats = options_loader.get_stats()?;
    if let Some(days) = stats.get(symbol) {
        println!("  Options data: {} days", days);
    }
    println!();

    // Test each date
    println!("=== Spot Price Accuracy Tests ===\n");

    let mut total_diff_pct = 0.0;
    let mut valid_tests = 0;

    for test_date in &test_dates {
        println!("Date: {}", test_date);

        // Get spot price from OHLCV
        match spot_loader.get_spot_price(symbol, *test_date) {
            Ok(spot_price) => {
                println!("  Spot price (OHLCV): ${:.2}", spot_price);

                // Load options chain for this date
                match options_loader.load_chain(symbol, *test_date) {
                    Ok(contracts) => {
                        let puts: Vec<_> = contracts
                            .iter()
                            .filter(|c| c.option_type == OptionType::Put)
                            .collect();

                        // Find ATM put (closest to spot price)
                        if let Some(atm_put) = puts.iter().min_by(|a, b| {
                            let diff_a = (a.strike - spot_price).abs();
                            let diff_b = (b.strike - spot_price).abs();
                            diff_a.partial_cmp(&diff_b).unwrap()
                        }) {
                            println!("  ATM strike (closest PUT): ${:.2}", atm_put.strike);

                            let diff_pct =
                                ((spot_price - atm_put.strike) / atm_put.strike * 100.0).abs();
                            println!("  Difference: {:.2}%", diff_pct);

                            if diff_pct < 1.0 {
                                println!("  ✅ PASS - Within 1% tolerance");
                            } else {
                                println!("  ⚠️  WARNING - Exceeds 1% tolerance");
                            }

                            total_diff_pct += diff_pct;
                            valid_tests += 1;

                            // Also check delta-based estimate
                            if let Some(delta_put) = puts.iter().find(|p| {
                                p.delta
                                    .map(|d| d.abs() > 0.45 && d.abs() < 0.55)
                                    .unwrap_or(false)
                            }) {
                                println!("  50-delta PUT strike: ${:.2}", delta_put.strike);
                                let delta_diff_pct =
                                    ((spot_price - delta_put.strike) / delta_put.strike * 100.0)
                                        .abs();
                                println!("  50-delta diff: {:.2}%", delta_diff_pct);
                            }
                        }
                    }
                    Err(e) => {
                        println!("  ⚠️  No options data: {}", e);
                    }
                }

                // Calculate ATR
                match spot_loader.calculate_atr(symbol, *test_date) {
                    Ok(atr) => {
                        let atr_pct = (atr / spot_price) * 100.0;
                        println!("  20-day ATR: ${:.2} ({:.2}% of spot)", atr, atr_pct);
                    }
                    Err(e) => {
                        println!("  ATR calculation failed: {}", e);
                    }
                }

                // Calculate Bollinger Bands
                match spot_loader.calculate_bollinger_bands(symbol, *test_date, 2.0) {
                    Ok((upper, lower, width)) => {
                        let width_pct = (width / spot_price) * 100.0;
                        println!("  Bollinger Bands (2σ):");
                        println!("    Upper: ${:.2}", upper);
                        println!("    Lower: ${:.2}", lower);
                        println!("    Width: ${:.2} ({:.2}% of spot)", width, width_pct);
                    }
                    Err(e) => {
                        println!("  Bollinger Bands calculation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("  ⚠️  Spot price not available: {}", e);
            }
        }

        println!();
    }

    // Summary
    println!("=== Summary ===");
    if valid_tests > 0 {
        let avg_diff_pct = total_diff_pct / valid_tests as f64;
        println!("Valid tests: {}/{}", valid_tests, test_dates.len());
        println!("Average difference: {:.2}%", avg_diff_pct);

        if avg_diff_pct < 1.0 {
            println!("✅ PASS - Spot prices are accurate within 1% tolerance");
        } else {
            println!("⚠️  WARNING - Average difference exceeds 1%");
        }
    } else {
        println!("❌ FAIL - No valid tests completed");
    }

    println!("\n=== Additional Spot Data Info ===");

    // Show recent OHLCV data
    if let Ok(bars) = spot_loader.load_symbol(symbol) {
        println!("Total OHLCV bars: {}", bars.len());
        println!("\nRecent data (last 5 days):");
        for bar in bars.iter().rev().take(5).rev() {
            println!(
                "  {} - O: ${:.2}, H: ${:.2}, L: ${:.2}, C: ${:.2}, V: {:.0}",
                bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume
            );
        }
    }

    println!("\n=== Test Complete ===");

    Ok(())
}
