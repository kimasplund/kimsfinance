//! Test Yahoo Finance Chunked Downloader
//!
//! Downloads historical stock data from Yahoo Finance with automatic chunking.
//! Completely free - no API key or subscription required!
//!
//! Usage:
//! ```bash
//! cargo run --release --features data-downloaders --example test_yahoo_chunked
//! ```

use kimsfinance_core::data::downloaders::{DownloadConfig, YahooDownloader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Yahoo Finance Chunked Downloader Test ===\n");
    println!("✅ FREE - No API key required!\n");

    let config = DownloadConfig::default();
    let downloader = YahooDownloader::new(config);

    // Test 1: Download 1 year of AAPL data in 3-month chunks
    println!("--- Test 1: AAPL (1 year, 3-month chunks) ---\n");

    match downloader
        .download_stock_chunked("AAPL", "2024-01-01", "2024-12-31", 3)
        .await
    {
        Ok(path) => {
            println!("\n✓ SUCCESS: AAPL data saved to {}\n", path.display());
        }
        Err(e) => {
            eprintln!("✗ FAILED: {}\n", e);
        }
    }

    // Test 2: Download 5 years of TSLA data in 6-month chunks
    println!("--- Test 2: TSLA (5 years, 6-month chunks) ---\n");

    match downloader
        .download_stock_chunked("TSLA", "2020-01-01", "2024-12-31", 6)
        .await
    {
        Ok(path) => {
            println!("\n✓ SUCCESS: TSLA data saved to {}\n", path.display());
        }
        Err(e) => {
            eprintln!("✗ FAILED: {}\n", e);
        }
    }

    // Test 3: Download 10 years of SPY data in 12-month (yearly) chunks
    println!("--- Test 3: SPY (10 years, yearly chunks) ---\n");

    match downloader
        .download_stock_chunked("SPY", "2015-01-01", "2024-12-31", 12)
        .await
    {
        Ok(path) => {
            println!("\n✓ SUCCESS: SPY data saved to {}\n", path.display());
        }
        Err(e) => {
            eprintln!("✗ FAILED: {}\n", e);
        }
    }

    // Test 4: Download AAPL options (first 12 expirations)
    println!("--- Test 4: AAPL Options (12 expirations) ---\n");

    match downloader.download_options_chunked("AAPL", Some(12)).await {
        Ok(path) => {
            println!("\n✓ SUCCESS: AAPL options saved to {}\n", path.display());
        }
        Err(e) => {
            eprintln!("✗ FAILED: {}\n", e);
        }
    }

    println!("=== Test Complete ===\n");

    println!("Expected file structure:");
    println!("data/yahoo/");
    println!("  ├── stocks/");
    println!("  │   ├── AAPL/daily/");
    println!("  │   │   ├── 2024-01-01_to_2024-03-31_daily.parquet");
    println!("  │   │   ├── 2024-04-01_to_2024-06-30_daily.parquet");
    println!("  │   │   ├── 2024-07-01_to_2024-09-30_daily.parquet");
    println!("  │   │   └── 2024-10-01_to_2024-12-31_daily.parquet");
    println!("  │   ├── TSLA/daily/");
    println!("  │   │   ├── 2020-01-01_to_2020-06-30_daily.parquet");
    println!("  │   │   ├── 2020-07-01_to_2020-12-31_daily.parquet");
    println!("  │   │   └── ... (10 total chunks)");
    println!("  │   └── SPY/daily/");
    println!("  │       ├── 2015-01-01_to_2015-12-31_daily.parquet");
    println!("  │       ├── 2016-01-01_to_2016-12-31_daily.parquet");
    println!("  │       └── ... (10 yearly files)");
    println!("  └── options/");
    println!("      └── AAPL/");
    println!("          ├── 2024-12-20_options.parquet");
    println!("          ├── 2025-01-17_options.parquet");
    println!("          ├── 2025-02-21_options.parquet");
    println!("          └── ... (12 expiration files)");
    println!("\nNote: Stock files use date ranges for easy reusability!");
    println!("      Option files use expiration dates for complete chains!");

    Ok(())
}
