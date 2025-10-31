//! Refined IBKR Historical Data Limits Test
//!
//! Binary search to find the exact limit for 1-minute bars
//!
//! Usage:
//! ```bash
//! cargo run --release --features data-ibkr,data-downloaders --example test_ibkr_limits_refined
//! ```

use kimsfinance_core::data::ibkr::{IbkrConfig, IbkrHistoricalDownloader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Refined IBKR Data Limits Test ===\n");

    let config = IbkrConfig {
        host: "127.0.0.1".to_string(),
        port: 7497,
        client_id: 5,
    };

    println!("Connecting to IBKR...");
    let downloader = IbkrHistoricalDownloader::connect(config).await?;
    println!("✓ Connected!\n");

    println!("Finding exact limit for 1-minute bars (AAPL)...\n");

    // We know: 2 months works (8 weeks), 3 months times out (12 weeks)
    // Let's test week by week between 8-12 weeks
    let test_weeks = vec![8, 9, 10, 11, 12];

    for weeks in test_weeks {
        let duration = format!("{} W", weeks);
        let trading_days = weeks * 5; // Rough estimate

        println!(
            "--- Testing: {} weeks (~{} trading days) ---",
            weeks, trading_days
        );

        match downloader.download_stock("AAPL", &duration, "1 min").await {
            Ok(path) => {
                if let Ok(metadata) = std::fs::metadata(&path) {
                    println!("✓ Success!");
                    println!("  Duration: {} weeks", weeks);
                    println!("  File size: {} KB", metadata.len() / 1024);

                    // Count bars
                    if let Ok(file) = std::fs::File::open(&path) {
                        use parquet::file::reader::{FileReader, SerializedFileReader};
                        if let Ok(reader) = SerializedFileReader::new(file) {
                            let num_rows = reader.metadata().file_metadata().num_rows();
                            let trading_days = num_rows as f64 / 390.0;
                            println!("  Bars: {}", num_rows);
                            println!("  Trading days: {:.1}", trading_days);
                        }
                    }
                    println!();
                }
            }
            Err(e) => {
                println!("✗ Failed: {}", e);
                println!(
                    "  -> Maximum appears to be {} weeks ({} months)\n",
                    weeks - 1,
                    (weeks - 1) as f64 / 4.0
                );
                break;
            }
        }

        // Small delay between requests
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
    }

    // Also test if we can get a full year of 5-minute bars
    println!("\n--- Bonus: Testing 1 year of 5-minute bars ---");
    match downloader.download_stock("AAPL", "1 Y", "5 mins").await {
        Ok(path) => {
            if let Ok(metadata) = std::fs::metadata(&path) {
                println!("✓ Success!");
                println!("  File size: {} KB", metadata.len() / 1024);

                if let Ok(file) = std::fs::File::open(&path) {
                    use parquet::file::reader::{FileReader, SerializedFileReader};
                    if let Ok(reader) = SerializedFileReader::new(file) {
                        let num_rows = reader.metadata().file_metadata().num_rows();
                        println!("  Bars: {}", num_rows);
                    }
                }
            }
        }
        Err(e) => println!("✗ Failed: {}", e),
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
