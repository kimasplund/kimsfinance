//! Demonstration of date range utilities and file discovery
//!
//! This example shows how to:
//! 1. Parse date ranges
//! 2. Generate month and day lists
//! 3. Discover Binance trade data files by date range
//!
//! # Usage
//! ```bash
//! cargo run --example demo_date_range_discovery
//! ```

use kimsfinance_core::binance::{BinanceDataFinder, DateRange};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Date Range Utilities Demo ===\n");

    // Example 1: Parse a date range
    println!("1. Parsing date range:");
    let range = DateRange::parse("2021-01-01", "2021-03-31")?;
    println!("   Start: {}", range.start);
    println!("   End: {}", range.end);
    println!("   Number of days: {}", range.num_days());
    println!("   Number of months: {}", range.num_months());
    println!();

    // Example 2: Generate month strings
    println!("2. Generate month strings:");
    let months = range.months();
    println!("   Months: {:?}", months);
    println!();

    // Example 3: Generate day strings (first few days)
    println!("3. Generate day strings (first 5):");
    let days = range.days();
    println!("   First 5 days: {:?}", &days[..5.min(days.len())]);
    println!("   Total days: {}", days.len());
    println!();

    // Example 4: File discovery (demonstration only - requires actual files)
    println!("4. File discovery (demo - adjust path for your system):");
    println!("   Example code:");
    println!("   ```rust");
    println!("   let finder = BinanceDataFinder::new(\"/data/binance\");");
    println!("   let range = DateRange::parse(\"2021-01-01\", \"2021-03-31\")?;");
    println!();
    println!("   // Find all files for date range");
    println!("   let files = finder.find_by_date_range(&range)?;");
    println!("   println!(\"Found {{}} files\", files.len());");
    println!();
    println!("   // Find BTCUSDT files only");
    println!("   let btc_files = finder.find_by_symbol_and_range(\"BTCUSDT\", &range)?;");
    println!("   ```");
    println!();

    // Example 5: Cross-year ranges
    println!("5. Cross-year date range:");
    let year_range = DateRange::parse("2020-12-01", "2021-02-28")?;
    println!("   Range: {} to {}", year_range.start, year_range.end);
    println!("   Months: {:?}", year_range.months());
    println!();

    // Example 6: Single day range
    println!("6. Single day range:");
    let single_day = DateRange::parse("2021-01-15", "2021-01-15")?;
    println!("   Range: {} to {}", single_day.start, single_day.end);
    println!("   Days: {:?}", single_day.days());
    println!("   Months: {:?}", single_day.months());
    println!();

    // Example 7: Long range stats
    println!("7. Full year range:");
    let full_year = DateRange::parse("2021-01-01", "2021-12-31")?;
    println!("   Number of months: {}", full_year.num_months());
    println!("   Number of days: {}", full_year.num_days());
    println!();

    // Example 8: Leap year handling
    println!("8. Leap year handling:");
    let leap_range = DateRange::parse("2020-02-28", "2020-03-01")?;
    println!("   Range: {} to {}", leap_range.start, leap_range.end);
    println!("   Days: {:?}", leap_range.days());
    println!("   (Includes leap day 2020-02-29)");
    println!();

    println!("=== Demo Complete ===");
    println!();
    println!("Integration Example:");
    println!("For actual file discovery, place Binance trade data in a directory:");
    println!("  /data/binance/BTCUSDT-trades-2021-01.zip");
    println!("  /data/binance/BTCUSDT-trades-2021-02.zip");
    println!("  /data/binance/BTCUSDT-trades-2021-03.zip");
    println!();
    println!("Then use BinanceDataFinder to discover them:");
    println!("  let finder = BinanceDataFinder::new(\"/data/binance\");");
    println!("  let range = DateRange::parse(\"2021-01-01\", \"2021-03-31\")?;");
    println!("  let files = finder.find_by_date_range(&range)?;");

    Ok(())
}
