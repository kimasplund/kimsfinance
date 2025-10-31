//! Test IBKR Chunked Historical Data Downloader
//!
//! Downloads a full year of 1-minute bars using automatic chunking.
//!
//! Usage:
//! ```bash
//! cargo run --release --features data-ibkr,data-downloaders --example test_ibkr_chunked
//! ```

use kimsfinance_core::data::ibkr::{IbkrConfig, IbkrHistoricalDownloader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IBKR Chunked Downloader Test ===\n");

    // Connect to IBKR TWS/Gateway
    let config = IbkrConfig {
        host: "127.0.0.1".to_string(),
        port: 7497,
        client_id: 6,
    };

    println!("Connecting to IBKR at {}:{}...", config.host, config.port);
    let downloader = IbkrHistoricalDownloader::connect(config).await?;
    println!("✓ Connected!\n");

    // Test 1: Download a full year of 1-minute bars (requires chunking)
    println!("--- Test 1: Full year of 1-minute bars (AAPL) ---");
    println!("This will make ~4 requests (3 months each) with rate limiting\n");

    match downloader
        .download_stock_chunked("AAPL", "1 Y", "1 min")
        .await
    {
        Ok(path) => {
            if let Ok(metadata) = std::fs::metadata(&path) {
                println!("\n✓ SUCCESS!");
                println!("  File: {}", path.display());
                println!("  Size: {} MB", metadata.len() / 1024 / 1024);

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
            }
        }
        Err(e) => {
            eprintln!("✗ FAILED: {}", e);
            return Err(e.into());
        }
    }

    // Test 2: Compare with non-chunked request (should be same result for small duration)
    println!("\n--- Test 2: Comparing chunked vs regular for 1 month ---");

    match downloader.download_stock("AAPL", "1 M", "1 min").await {
        Ok(path) => {
            if let Ok(file) = std::fs::File::open(&path) {
                use parquet::file::reader::{FileReader, SerializedFileReader};
                if let Ok(reader) = SerializedFileReader::new(file) {
                    let bars_regular = reader.metadata().file_metadata().num_rows();
                    println!("Regular download: {} bars", bars_regular);
                }
            }
        }
        Err(e) => println!("Regular download failed: {}", e),
    }

    match downloader
        .download_stock_chunked("AAPL", "1 M", "1 min")
        .await
    {
        Ok(path) => {
            if let Ok(file) = std::fs::File::open(&path) {
                use parquet::file::reader::{FileReader, SerializedFileReader};
                if let Ok(reader) = SerializedFileReader::new(file) {
                    let bars_chunked = reader.metadata().file_metadata().num_rows();
                    println!("Chunked download: {} bars", bars_chunked);
                    println!("(Should be the same - chunked detects no chunking needed)");
                }
            }
        }
        Err(e) => println!("Chunked download failed: {}", e),
    }

    println!("\n=== Test Complete ===");
    println!("\nNote: Full year download takes ~45 seconds (3x delays + processing time)");

    Ok(())
}
