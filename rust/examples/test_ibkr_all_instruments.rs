//! Test IBKR Chunked Downloader for All Instrument Types
//!
//! Demonstrates downloading historical data for:
//! - Stocks (AAPL)
//! - Futures/Commodities (Gold, Oil)
//! - Forex (EUR/USD)
//! - Options (AAPL Call)
//!
//! Usage:
//! ```bash
//! cargo run --release --features data-ibkr,data-downloaders --example test_ibkr_all_instruments
//! ```

use kimsfinance_core::data::ibkr::{IbkrConfig, IbkrHistoricalDownloader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IBKR All Instruments Chunked Downloader Test ===\n");

    // Connect to IBKR TWS/Gateway
    let config = IbkrConfig {
        host: "127.0.0.1".to_string(),
        port: 7497,
        client_id: 7,
    };

    println!("Connecting to IBKR at {}:{}...", config.host, config.port);
    let downloader = IbkrHistoricalDownloader::connect(config).await?;
    println!("✓ Connected!\n");

    // Test 1: Download stock data (already tested in test_ibkr_chunked)
    println!("--- Test 1: Stock Data (AAPL) ---");
    println!("Downloading 6 months of 1-hour bars\n");

    match downloader
        .download_stock_chunked("AAPL", "6 M", "1 hour")
        .await
    {
        Ok(path) => {
            println!("✓ SUCCESS: Stock data saved to {}\n", path.display());
        }
        Err(e) => {
            eprintln!("✗ FAILED: {}\n", e);
        }
    }

    // Test 2: Download futures/commodities data
    println!("--- Test 2: Futures/Commodities (Gold) ---");
    println!("Downloading 1 year of gold futures (GC) with 1-hour bars\n");

    match downloader
        .download_futures_chunked("GC", "1 Y", "1 hour")
        .await
    {
        Ok(path) => {
            println!("✓ SUCCESS: Gold futures data saved to {}\n", path.display());
        }
        Err(e) => {
            eprintln!("✗ FAILED: {}", e);
            eprintln!("Note: Futures require market data subscription from IBKR\n");
        }
    }

    println!("--- Test 3: Futures/Commodities (Oil) ---");
    println!("Downloading 6 months of crude oil futures (CL) with 1-hour bars\n");

    match downloader
        .download_futures_chunked("CL", "6 M", "1 hour")
        .await
    {
        Ok(path) => {
            println!("✓ SUCCESS: Oil futures data saved to {}\n", path.display());
        }
        Err(e) => {
            eprintln!("✗ FAILED: {}", e);
            eprintln!("Note: Futures require market data subscription from IBKR\n");
        }
    }

    // Test 4: Download forex data
    println!("--- Test 4: Forex (EUR/USD) ---");
    println!("Downloading 1 year of EUR/USD with 1-hour bars\n");

    match downloader
        .download_forex_chunked("EUR", "USD", "1 Y", "1 hour")
        .await
    {
        Ok(path) => {
            println!("✓ SUCCESS: EUR/USD data saved to {}\n", path.display());
        }
        Err(e) => {
            eprintln!("✗ FAILED: {}", e);
            eprintln!("Note: Forex requires IDEALPRO market data subscription from IBKR\n");
        }
    }

    // Test 5: Download options data
    println!("--- Test 5: Options (AAPL Call) ---");
    println!("Downloading 3 months of AAPL $150 Call expiring Dec 20, 2024\n");

    match downloader
        .download_options_chunked("AAPL", "20241220", 150.0, "C", "3 M", "1 hour")
        .await
    {
        Ok(path) => {
            println!("✓ SUCCESS: Options data saved to {}\n", path.display());
        }
        Err(e) => {
            eprintln!("✗ FAILED: {}", e);
            eprintln!("Note: Options require market data subscription from IBKR\n");
        }
    }

    println!("=== Test Complete ===\n");

    println!("Expected file structure:");
    println!("data/ibkr/");
    println!("  ├── stocks/AAPL/");
    println!("  │   └── 2024-01-01_to_2024-06-30_1hour.parquet");
    println!("  ├── futures/GC/");
    println!("  │   ├── 2024-01-01_to_2024-03-31_1hour.parquet");
    println!("  │   ├── 2024-04-01_to_2024-06-30_1hour.parquet");
    println!("  │   └── 2024-07-01_to_2024-09-30_1hour.parquet");
    println!("  ├── futures/CL/");
    println!("  │   └── 2024-04-01_to_2024-09-30_1hour.parquet");
    println!("  ├── forex/EURUSD/");
    println!("  │   ├── 2024-01-01_to_2024-03-31_1hour.parquet");
    println!("  │   └── 2024-04-01_to_2024-06-30_1hour.parquet");
    println!("  └── options/AAPL_20241220_150C/");
    println!("      └── 2024-07-01_to_2024-09-30_1hour.parquet");

    Ok(())
}
