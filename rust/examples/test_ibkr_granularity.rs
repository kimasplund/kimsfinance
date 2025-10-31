//! Test IBKR Data Granularity
//!
//! Tests the smallest timeframe (1 second bars) and various durations
//! to understand IBKR's historical data limits.
//!
//! Usage:
//! ```bash
//! cargo run --release --features data-ibkr,data-downloaders --example test_ibkr_granularity
//! ```

use kimsfinance_core::data::ibkr::{IbkrConfig, IbkrHistoricalDownloader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IBKR Data Granularity Test ===\n");

    // Connect to IBKR TWS/Gateway
    let config = IbkrConfig {
        host: "127.0.0.1".to_string(),
        port: 7497,
        client_id: 3, // Different client ID
    };

    println!("Connecting to IBKR at {}:{}...", config.host, config.port);
    let downloader = IbkrHistoricalDownloader::connect(config).await?;
    println!("✓ Connected!\n");

    // Test 1: 1 second bars - very short duration
    println!("--- Test 1: AAPL 1-second bars (1 day) ---");
    match downloader.download_stock("AAPL", "1 D", "1 sec").await {
        Ok(path) => {
            let metadata = std::fs::metadata(&path)?;
            println!("✓ Success! Saved to: {}", path.display());
            println!("  File size: {} bytes\n", metadata.len());
        }
        Err(e) => println!("✗ Failed: {}\n", e),
    }

    // Test 2: 5 second bars - 1 week
    println!("--- Test 2: AAPL 5-second bars (1 week) ---");
    match downloader.download_stock("AAPL", "1 W", "5 secs").await {
        Ok(path) => {
            let metadata = std::fs::metadata(&path)?;
            println!("✓ Success! Saved to: {}", path.display());
            println!("  File size: {} bytes\n", metadata.len());
        }
        Err(e) => println!("✗ Failed: {}\n", e),
    }

    // Test 3: 1 minute bars - 1 month
    println!("--- Test 3: AAPL 1-minute bars (1 month) ---");
    match downloader.download_stock("AAPL", "1 M", "1 min").await {
        Ok(path) => {
            let metadata = std::fs::metadata(&path)?;
            println!("✓ Success! Saved to: {}", path.display());
            println!("  File size: {} bytes\n", metadata.len());
        }
        Err(e) => println!("✗ Failed: {}\n", e),
    }

    // Test 4: 5 minute bars - 6 months
    println!("--- Test 4: AAPL 5-minute bars (6 months) ---");
    match downloader.download_stock("AAPL", "6 M", "5 mins").await {
        Ok(path) => {
            let metadata = std::fs::metadata(&path)?;
            println!("✓ Success! Saved to: {}", path.display());
            println!("  File size: {} bytes\n", metadata.len());
        }
        Err(e) => println!("✗ Failed: {}\n", e),
    }

    // Test 5: 1 hour bars - 1 year
    println!("--- Test 5: AAPL 1-hour bars (1 year) ---");
    match downloader.download_stock("AAPL", "1 Y", "1 hour").await {
        Ok(path) => {
            let metadata = std::fs::metadata(&path)?;
            println!("✓ Success! Saved to: {}", path.display());
            println!("  File size: {} bytes\n", metadata.len());
        }
        Err(e) => println!("✗ Failed: {}\n", e),
    }

    println!("=== Granularity Test Complete ===");
    println!("\nKey Findings:");
    println!("- Smallest bar size: 1 second");
    println!("- Duration limits depend on bar size (IBKR restrictions)");
    println!("- Check file sizes above to see how much data was returned");

    Ok(())
}
