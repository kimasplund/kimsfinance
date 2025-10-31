//! Test IBKR Historical Data Limits
//!
//! Probes the actual duration limits for different bar sizes
//! to understand how far back IBKR data goes.
//!
//! Usage:
//! ```bash
//! cargo run --release --features data-ibkr,data-downloaders --example test_ibkr_limits
//! ```

use kimsfinance_core::data::ibkr::{IbkrConfig, IbkrHistoricalDownloader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IBKR Historical Data Limits Test ===\n");

    // Connect to IBKR TWS/Gateway
    let config = IbkrConfig {
        host: "127.0.0.1".to_string(),
        port: 7497,
        client_id: 4,
    };

    println!("Connecting to IBKR at {}:{}...", config.host, config.port);
    let downloader = IbkrHistoricalDownloader::connect(config).await?;
    println!("✓ Connected!\n");

    println!("Testing 1-minute bar limits for AAPL...\n");

    // Test increasing durations for 1-minute bars
    let test_durations = vec![
        ("1 M", "1 month"),
        ("2 M", "2 months"),
        ("3 M", "3 months"),
        ("6 M", "6 months"),
        ("1 Y", "1 year"),
        ("2 Y", "2 years"),
    ];

    for (duration, label) in test_durations {
        println!("--- Testing: {} of 1-minute bars ---", label);
        match downloader.download_stock("AAPL", duration, "1 min").await {
            Ok(path) => {
                let metadata = std::fs::metadata(&path)?;

                // Count bars by reading the parquet file
                println!("✓ Success!");
                println!("  Duration: {}", label);
                println!("  File: {}", path.display());
                println!("  Size: {} KB", metadata.len() / 1024);

                // Try to parse the parquet file to count rows
                if let Ok(file) = std::fs::File::open(&path) {
                    use parquet::file::reader::{FileReader, SerializedFileReader};
                    if let Ok(reader) = SerializedFileReader::new(file) {
                        let metadata = reader.metadata();
                        let num_rows: i64 = metadata.file_metadata().num_rows();
                        println!("  Bars: {}", num_rows);

                        // Calculate approximate days of trading data
                        // Assume ~6.5 hours trading day * 60 minutes = 390 minutes per day
                        let trading_days = num_rows as f64 / 390.0;
                        println!("  Approx trading days: {:.1}", trading_days);
                    }
                }
                println!();
            }
            Err(e) => {
                println!("✗ Failed: {}", e);
                println!("  -> This might be the limit!\n");
                break;
            }
        }

        // Small delay between requests
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }

    println!("\n=== Limits Test Complete ===");
    println!("\nNext steps:");
    println!("- Try even longer durations if 2Y succeeded");
    println!("- Test other bar sizes (5min, 15min, 1hour) with longer durations");

    Ok(())
}
