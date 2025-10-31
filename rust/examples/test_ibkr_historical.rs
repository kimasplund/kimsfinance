//! Test IBKR Historical Data Downloader
//!
//! Tests downloading historical data from IBKR (works even when markets are closed)
//!
//! Usage:
//! ```bash
//! cargo run --release --features data-ibkr,data-downloaders --example test_ibkr_historical
//! ```

use kimsfinance_core::data::ibkr::{IbkrConfig, IbkrHistoricalDownloader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IBKR Historical Data Downloader Test ===\n");

    // Connect to IBKR TWS/Gateway
    let config = IbkrConfig {
        host: "127.0.0.1".to_string(),
        port: 7497,   // Paper trading port (adjust if needed)
        client_id: 2, // Use different client ID from options connector
    };

    println!("Connecting to IBKR at {}:{}...", config.host, config.port);
    let downloader = IbkrHistoricalDownloader::connect(config).await?;
    println!("✓ Connected!\n");

    // Test 1: Download stock data
    println!("--- Test 1: Downloading AAPL stock data (1 month, daily bars) ---");
    match downloader.download_stock("AAPL", "1 M", "1 day").await {
        Ok(path) => println!("✓ Success! Saved to: {}\n", path.display()),
        Err(e) => println!("✗ Failed: {}\n", e),
    }

    // Test 2: Download forex data
    println!("--- Test 2: Downloading EUR/USD forex data (1 week, hourly bars) ---");
    match downloader
        .download_forex("EUR", "USD", "1 W", "1 hour")
        .await
    {
        Ok(path) => println!("✓ Success! Saved to: {}\n", path.display()),
        Err(e) => println!("✗ Failed: {}\n", e),
    }

    // Test 3: Download futures data
    println!("--- Test 3: Downloading ES futures data (1 month, 1 hour bars) ---");
    match downloader.download_futures("ES", "1 M", "1 hour").await {
        Ok(path) => println!("✓ Success! Saved to: {}\n", path.display()),
        Err(e) => println!("✗ Failed: {}\n", e),
    }

    // Test 4: Download crypto data
    println!("--- Test 4: Downloading BTC crypto data (1 week, 1 day bars) ---");
    match downloader.download_crypto("BTC", "1 W", "1 day").await {
        Ok(path) => println!("✓ Success! Saved to: {}\n", path.display()),
        Err(e) => println!("✗ Failed: {}\n", e),
    }

    println!("=== Test Complete ===");

    Ok(())
}
