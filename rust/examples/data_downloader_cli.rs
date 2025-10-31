//! Data Downloader CLI
//!
//! Download market data from multiple sources to organized Parquet files.
//!
//! # Usage
//!
//! ```bash
//! # Download Binance BTC spot trades for 2024
//! cargo run --release --example data_downloader_cli -- binance spot BTCUSDT 2024
//!
//! # Download Yahoo Finance stock data
//! cargo run --release --example data_downloader_cli -- yahoo stock AAPL 2024-01-01 2024-12-31
//!
//! # Download Yahoo Finance options chain
//! cargo run --release --example data_downloader_cli -- yahoo options AAPL
//! ```

use kimsfinance_core::data::downloaders::{BinanceDownloader, DownloadConfig, YahooDownloader};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        print_usage();
        return Ok(());
    }

    let source = &args[1];
    let command = &args[2];

    let config = DownloadConfig {
        base_path: "data".into(),
        parallel_downloads: 4,
        verify_checksums: true,
        resume: true,
    };

    let separator = "=".repeat(80);
    println!("{}", separator);
    println!("KIMSFINANCE DATA DOWNLOADER");
    println!("{}", separator);
    println!();

    match source.as_str() {
        "binance" => handle_binance(command, &args[3..], config).await?,
        "yahoo" => handle_yahoo(command, &args[3..], config).await?,
        _ => {
            eprintln!("Unknown source: {}", source);
            print_usage();
        }
    }

    let separator = "=".repeat(80);
    println!();
    println!("{}", separator);
    println!("DOWNLOAD COMPLETE");
    println!("{}", separator);

    Ok(())
}

async fn handle_binance(
    command: &str,
    args: &[String],
    config: DownloadConfig,
) -> Result<(), Box<dyn Error>> {
    let downloader = BinanceDownloader::new(config);

    match command {
        "spot" | "futures" => {
            if args.len() < 2 {
                eprintln!("Usage: binance {} <SYMBOL> <YEAR> [MONTH]", command);
                return Ok(());
            }

            let symbol = &args[0];
            let year: u32 = args[1].parse()?;
            let month: Option<u32> = args.get(2).and_then(|m| m.parse().ok());

            println!("Source: Binance {} Market", command);
            println!("Symbol: {}", symbol);
            println!(
                "Period: {} {}",
                year,
                month.map_or("(all months)".to_string(), |m| format!("month {}", m))
            );
            println!();

            // Download trades
            let downloaded = if command == "spot" {
                downloader.download_spot_trades(symbol, year, month).await?
            } else {
                downloader
                    .download_futures_trades(symbol, year, month)
                    .await?
            };

            println!("\n✓ Downloaded {} files", downloaded.len());

            // Aggregate to OHLCV
            println!("\n--- Aggregating to OHLCV Parquet ---\n");

            for timeframe in ["1m", "5m", "15m", "1h"] {
                println!("Timeframe: {}", timeframe);
                let parquet_files = downloader
                    .aggregate_to_ohlcv(symbol, timeframe, year)
                    .await?;
                println!("✓ Created {} Parquet files\n", parquet_files.len());
            }
        }
        _ => {
            eprintln!("Unknown Binance command: {}", command);
            eprintln!("Valid commands: spot, futures");
        }
    }

    Ok(())
}

async fn handle_yahoo(
    command: &str,
    args: &[String],
    config: DownloadConfig,
) -> Result<(), Box<dyn Error>> {
    let downloader = YahooDownloader::new(config);

    match command {
        "stock" => {
            if args.len() < 3 {
                eprintln!("Usage: yahoo stock <SYMBOL> <START_DATE> <END_DATE>");
                eprintln!("Example: yahoo stock AAPL 2024-01-01 2024-12-31");
                return Ok(());
            }

            let symbol = &args[0];
            let start_date = &args[1];
            let end_date = &args[2];

            println!("Source: Yahoo Finance");
            println!("Symbol: {}", symbol);
            println!("Period: {} to {}", start_date, end_date);
            println!();

            let path = downloader
                .download_stock(symbol, start_date, end_date)
                .await?;
            println!("\n✓ Saved to: {}", path.display());
        }
        "options" => {
            if args.is_empty() {
                eprintln!("Usage: yahoo options <SYMBOL> [EXPIRATION]");
                eprintln!("Example: yahoo options AAPL");
                eprintln!("Example: yahoo options AAPL 2024-03-15");
                return Ok(());
            }

            let symbol = &args[0];
            let expiration = args.get(1).map(|s| s.as_str());

            println!("Source: Yahoo Finance Options");
            println!("Symbol: {}", symbol);
            if let Some(exp) = expiration {
                println!("Expiration: {}", exp);
            } else {
                println!("Expiration: Nearest available");
            }
            println!();

            let path = downloader
                .download_options_chain(symbol, expiration)
                .await?;
            println!("\n✓ Saved to: {}", path.display());
        }
        _ => {
            eprintln!("Unknown Yahoo command: {}", command);
            eprintln!("Valid commands: stock, options");
        }
    }

    Ok(())
}

fn print_usage() {
    println!("Usage: data_downloader_cli <SOURCE> <COMMAND> [ARGS...]");
    println!();
    println!("Sources:");
    println!("  binance     Binance Vision historical data");
    println!("  yahoo       Yahoo Finance stocks + options");
    println!();
    println!("Binance Commands:");
    println!("  spot <SYMBOL> <YEAR> [MONTH]          Download spot trades");
    println!("  futures <SYMBOL> <YEAR> [MONTH]       Download futures trades");
    println!();
    println!("Yahoo Commands:");
    println!("  stock <SYMBOL> <START> <END>          Download stock data (YYYY-MM-DD)");
    println!("  options <SYMBOL> [EXPIRATION]         Download options chain");
    println!();
    println!("Examples:");
    println!("  data_downloader_cli binance spot BTCUSDT 2024");
    println!("  data_downloader_cli binance futures ETHUSDT 2024 1");
    println!("  data_downloader_cli yahoo stock AAPL 2024-01-01 2024-12-31");
    println!("  data_downloader_cli yahoo options SPY");
    println!();
    println!("Output Directory Structure:");
    println!("  data/");
    println!("  ├── binance/");
    println!("  │   ├── spot/BTCUSDT/");
    println!("  │   │   ├── trades/2024-01.zip");
    println!("  │   │   └── ohlcv/1m/2024-01.parquet");
    println!("  │   └── futures/ETHUSDT/");
    println!("  └── yahoo/");
    println!("      ├── stocks/AAPL/daily/2024.parquet");
    println!("      └── options/SPY/chain/2024-03-15.parquet");
}
