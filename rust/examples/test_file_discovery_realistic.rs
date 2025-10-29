//! Realistic example of finding Binance trade data files
//!
//! This example creates a temporary directory with sample files
//! and demonstrates file discovery across multiple months.
//!
//! # Usage
//! ```bash
//! cargo run --example test_file_discovery_realistic
//! ```

use kimsfinance_core::binance::{BinanceDataFinder, DateRange};
use std::error::Error;
use std::fs::File;
use tempfile::TempDir;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Realistic File Discovery Example ===\n");

    // Create temporary directory with sample files
    let temp_dir = TempDir::new()?;
    let data_path = temp_dir.path();

    println!("1. Creating sample Binance data files:");
    let sample_files = vec![
        "BTCUSDT-trades-2021-01.zip",
        "BTCUSDT-trades-2021-02.zip",
        "BTCUSDT-trades-2021-03.zip",
        "BTCUSDT-trades-2021-04.zip",
        "ETHUSDT-trades-2021-01.zip",
        "ETHUSDT-trades-2021-02.zip",
        "ETHUSDT-trades-2021-03.zip",
        "SOLUSDT-trades-2021-02.zip",
        "BNBUSDT-trades-2021-01.zip",
        "unrelated-file.txt",
    ];

    for filename in &sample_files {
        File::create(data_path.join(filename))?;
        println!("   Created: {}", filename);
    }
    println!();

    // Initialize finder
    let finder = BinanceDataFinder::new(data_path);

    // Test 1: Find all files for date range (2021-01 to 2021-03)
    println!("2. Finding files for date range 2021-01-01 to 2021-03-31:");
    let range = DateRange::parse("2021-01-01", "2021-03-31")?;
    let files = finder.find_by_date_range(&range)?;
    println!("   Found {} files:", files.len());
    for file in &files {
        println!("     - {}", file.file_name().unwrap().to_string_lossy());
    }
    println!();

    // Test 2: Find BTCUSDT files only
    println!("3. Finding BTCUSDT files for 2021-01-01 to 2021-03-31:");
    let btc_files = finder.find_by_symbol_and_range("BTCUSDT", &range)?;
    println!("   Found {} BTCUSDT files:", btc_files.len());
    for file in &btc_files {
        println!("     - {}", file.file_name().unwrap().to_string_lossy());
    }
    println!();

    // Test 3: Find ETHUSDT files only
    println!("4. Finding ETHUSDT files for 2021-01-01 to 2021-03-31:");
    let eth_files = finder.find_by_symbol_and_range("ETHUSDT", &range)?;
    println!("   Found {} ETHUSDT files:", eth_files.len());
    for file in &eth_files {
        println!("     - {}", file.file_name().unwrap().to_string_lossy());
    }
    println!();

    // Test 4: Find files for single month
    println!("5. Finding all files for February 2021 only:");
    let feb_range = DateRange::parse("2021-02-01", "2021-02-28")?;
    let feb_files = finder.find_by_date_range(&feb_range)?;
    println!("   Found {} files:", feb_files.len());
    for file in &feb_files {
        println!("     - {}", file.file_name().unwrap().to_string_lossy());
    }
    println!();

    // Test 5: Pattern matching
    println!("6. Using pattern matching for BTCUSDT files:");
    let btc_pattern_files = finder.find_files("BTCUSDT-*.zip")?;
    println!("   Found {} files with pattern:", btc_pattern_files.len());
    for file in &btc_pattern_files {
        println!("     - {}", file.file_name().unwrap().to_string_lossy());
    }
    println!();

    // Test 6: No matches scenario
    println!("7. Testing no matches (ADAUSDT for 2021-01 to 2021-03):");
    let no_matches = finder.find_by_symbol_and_range("ADAUSDT", &range)?;
    println!("   Found {} files (expected 0)", no_matches.len());
    println!();

    // Test 7: Extended range (includes April)
    println!("8. Extended range (2021-01-01 to 2021-04-30):");
    let extended_range = DateRange::parse("2021-01-01", "2021-04-30")?;
    let extended_files = finder.find_by_symbol_and_range("BTCUSDT", &extended_range)?;
    println!("   Found {} BTCUSDT files:", extended_files.len());
    for file in &extended_files {
        println!("     - {}", file.file_name().unwrap().to_string_lossy());
    }
    println!();

    // Summary statistics
    println!("=== Summary ===");
    println!("Date range analysis:");
    println!("  - Original range: 2021-01-01 to 2021-03-31");
    println!("  - Number of months: {}", range.num_months());
    println!("  - Number of days: {}", range.num_days());
    println!("  - Months covered: {:?}", range.months());
    println!();
    println!("File discovery results:");
    println!("  - Total files in Q1 2021: {}", files.len());
    println!("  - BTCUSDT files: {}", btc_files.len());
    println!("  - ETHUSDT files: {}", eth_files.len());
    println!("  - February-only files: {}", feb_files.len());
    println!();
    println!("Integration ready!");
    println!("This code can be used to:");
    println!("  1. Discover multi-month Binance trade data");
    println!("  2. Filter by trading pair symbol");
    println!("  3. Process data sequentially or in batch");

    Ok(())
}
