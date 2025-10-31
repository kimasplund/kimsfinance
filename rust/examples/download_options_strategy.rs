//! Download Options Data for Strategy Development
//!
//! Downloads AAPL and MSFT options chains for strategy development.
//! All available expirations will be downloaded.
//!
//! Usage:
//! ```bash
//! cargo run --release --features data-downloaders --example download_options_strategy
//! ```

use kimsfinance_core::data::downloaders::{DownloadConfig, YahooDownloader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Options Strategy Data Download ===\n");
    println!("✅ FREE - Yahoo Finance (no API key required)\n");

    let config = DownloadConfig::default();
    let downloader = YahooDownloader::new(config);

    // Download AAPL options (first 20 expirations to avoid rate limiting)
    println!("--- Downloading AAPL Options ---\n");

    match downloader.download_options_chunked("AAPL", Some(20)).await {
        Ok(path) => {
            println!("\n✓ SUCCESS: AAPL options saved to {}\n", path.display());

            // Count files
            if let Ok(entries) = std::fs::read_dir(&path) {
                let count = entries.count();
                println!("  Total expirations downloaded: {}\n", count);
            }
        }
        Err(e) => {
            eprintln!("✗ FAILED: {}\n", e);
            return Err(e.into());
        }
    }

    // Download MSFT options (first 20 expirations to avoid rate limiting)
    println!("--- Downloading MSFT Options ---\n");

    match downloader.download_options_chunked("MSFT", Some(20)).await {
        Ok(path) => {
            println!("\n✓ SUCCESS: MSFT options saved to {}\n", path.display());

            // Count files
            if let Ok(entries) = std::fs::read_dir(&path) {
                let count = entries.count();
                println!("  Total expirations downloaded: {}\n", count);
            }
        }
        Err(e) => {
            eprintln!("✗ FAILED: {}\n", e);
            return Err(e.into());
        }
    }

    println!("=== Download Complete ===\n");

    println!("Data location:");
    println!("data/yahoo/options/");
    println!("  ├── AAPL/");
    println!("  │   ├── 2024-12-20_options.parquet");
    println!("  │   ├── 2025-01-17_options.parquet");
    println!("  │   └── ... (all available expirations)");
    println!("  └── MSFT/");
    println!("      ├── 2024-12-20_options.parquet");
    println!("      ├── 2025-01-17_options.parquet");
    println!("      └── ... (all available expirations)");

    println!("\nEach file contains:");
    println!("  - All strikes for that expiration");
    println!("  - Both calls and puts");
    println!("  - Full options chain data (bid, ask, volume, OI, Greeks, etc.)");

    println!("\nReady for strategy development! 🚀");

    Ok(())
}
