//! Data Downloaders Module
//!
//! Multi-source market data downloaders with organized Parquet storage.
//!
//! # Supported Sources
//!
//! - **Binance Vision**: Free historical spot/futures trade data
//! - **Yahoo Finance**: Free stocks + options data (via yfinance)
//! - **IBKR**: Interactive Brokers data (requires account)
//!
//! # Directory Structure
//!
//! ```text
//! data/
//! ├── binance/
//! │   ├── spot/BTCUSDT/trades/2024-01.parquet
//! │   └── futures/BTCUSDT/ohlcv/1m/2024-01.parquet
//! ├── yahoo/
//! │   ├── stocks/AAPL/daily/2024.parquet
//! │   └── options/AAPL/chain/2024-01-19.parquet
//! └── ibkr/
//!     └── options/SPY/chain/2024-01-19.parquet
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use kimsfinance_core::data::downloaders::{BinanceDownloader, DownloadConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = DownloadConfig {
//!     base_path: "data".into(),
//!     parallel_downloads: 4,
//!     verify_checksums: true,
//! };
//!
//! let downloader = BinanceDownloader::new(config);
//!
//! // Download BTC spot trades for 2024
//! downloader.download_spot_trades("BTCUSDT", 2024, None).await?;
//!
//! // Convert to 5-minute OHLCV Parquet
//! downloader.aggregate_to_ohlcv("BTCUSDT", "5m", 2024).await?;
//! # Ok(())
//! # }
//! ```

pub mod binance;
pub mod common;
pub mod yahoo;

pub use binance::BinanceDownloader;
pub use common::{DownloadConfig, DownloadError, DownloadProgress, Downloader};
pub use yahoo::YahooDownloader;
