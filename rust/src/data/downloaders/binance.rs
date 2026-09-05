//! Binance Vision data downloader
//!
//! Downloads historical trade data from Binance Public Data and converts to Parquet.
//!
//! # Data Sources
//!
//! - Spot trades: https://data.binance.vision/data/spot/monthly/trades/
//! - Futures trades: https://data.binance.vision/data/futures/um/monthly/trades/
//!
//! # Example
//!
//! ```rust,no_run
//! use kimsfinance_core::data::downloaders::{BinanceDownloader, DownloadConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = DownloadConfig::default();
//! let downloader = BinanceDownloader::new(config);
//!
//! // Download BTC futures trades for Jan 2024
//! downloader.download_futures_trades("BTCUSDT", 2024, Some(1)).await?;
//!
//! // Aggregate to 5m OHLCV Parquet
//! downloader.aggregate_to_ohlcv("BTCUSDT", "5m", 2024).await?;
//! # Ok(())
//! # }
//! ```

use super::common::{DownloadConfig, DownloadError, DownloadProgress, Downloader};
use crate::binance::{Candle, Timeframe, process_binance_month};
use async_trait::async_trait;
use chrono::{Datelike, NaiveDate};
use reqwest::Client;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Binance data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinanceMarketType {
    Spot,
    Futures,
}

/// Binance Vision downloader
pub struct BinanceDownloader {
    config: DownloadConfig,
    client: Client,
    progress: Arc<RwLock<Option<DownloadProgress>>>,
}

impl BinanceDownloader {
    /// Create new Binance downloader
    pub fn new(config: DownloadConfig) -> Self {
        let client = Client::builder()
            .user_agent("kimsfinance/0.2.0")
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap();

        Self {
            config,
            client,
            progress: Arc::new(RwLock::new(None)),
        }
    }

    /// Download spot market trades
    ///
    /// # Arguments
    ///
    /// - `symbol`: Trading pair (e.g., "BTCUSDT")
    /// - `year`: Year to download
    /// - `month`: Optional month (1-12), downloads all months if None
    pub async fn download_spot_trades(
        &self,
        symbol: &str,
        year: u32,
        month: Option<u32>,
    ) -> Result<Vec<PathBuf>, DownloadError> {
        self.download_trades(BinanceMarketType::Spot, symbol, year, month)
            .await
    }

    /// Download futures market trades
    pub async fn download_futures_trades(
        &self,
        symbol: &str,
        year: u32,
        month: Option<u32>,
    ) -> Result<Vec<PathBuf>, DownloadError> {
        self.download_trades(BinanceMarketType::Futures, symbol, year, month)
            .await
    }

    /// Download trades for a specific market type
    async fn download_trades(
        &self,
        market_type: BinanceMarketType,
        symbol: &str,
        year: u32,
        month: Option<u32>,
    ) -> Result<Vec<PathBuf>, DownloadError> {
        let months = match month {
            Some(m) => vec![m],
            None => (1..=12).collect(),
        };

        let mut downloaded = Vec::new();

        for m in months {
            let url = self.build_url(market_type, symbol, year, m);
            let output_path = self.get_output_path(market_type, symbol, year, m, "trades");

            // Create directory
            if let Some(parent) = output_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            // Skip if already exists and resume is enabled
            if self.config.resume && output_path.exists() {
                println!("✓ Already downloaded: {}", output_path.display());
                downloaded.push(output_path);
                continue;
            }

            println!("Downloading: {} → {}", url, output_path.display());

            // Download ZIP file
            let response = self
                .client
                .get(&url)
                .send()
                .await
                .map_err(|e| DownloadError::Network(e.to_string()))?;

            if !response.status().is_success() {
                if response.status().as_u16() == 404 {
                    println!("⚠️  No data for {}-{:02}", year, m);
                    continue;
                }
                return Err(DownloadError::ApiError(format!(
                    "HTTP {}: {}",
                    response.status(),
                    response.text().await.unwrap_or_default()
                )));
            }

            let bytes = response
                .bytes()
                .await
                .map_err(|e| DownloadError::Network(e.to_string()))?;

            // Save ZIP file
            tokio::fs::write(&output_path, &bytes).await?;

            println!(
                "✓ Downloaded: {} ({:.2} MB)",
                output_path.display(),
                bytes.len() as f64 / 1_048_576.0
            );

            downloaded.push(output_path);
        }

        Ok(downloaded)
    }

    /// Aggregate downloaded trades to OHLCV Parquet
    ///
    /// # Arguments
    ///
    /// - `symbol`: Trading pair
    /// - `timeframe`: Timeframe string (e.g., "1m", "5m", "1h")
    /// - `year`: Year to process
    pub async fn aggregate_to_ohlcv(
        &self,
        symbol: &str,
        timeframe: &str,
        year: u32,
    ) -> Result<Vec<PathBuf>, DownloadError> {
        let tf =
            Timeframe::parse(timeframe).map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;

        let mut output_paths = Vec::new();

        for month in 1..=12 {
            let zip_path =
                self.get_output_path(BinanceMarketType::Spot, symbol, year, month, "trades");

            if !zip_path.exists() {
                continue;
            }

            println!(
                "Aggregating: {} → {} candles",
                zip_path.display(),
                timeframe
            );

            // Process with existing infrastructure
            let candles = process_binance_month(&zip_path, tf)
                .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;

            if candles.is_empty() {
                println!("⚠️  No candles generated for {}-{:02}", year, month);
                continue;
            }

            // Save to Parquet
            let parquet_path =
                self.get_ohlcv_path(BinanceMarketType::Spot, symbol, timeframe, year, month);

            if let Some(parent) = parquet_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            self.write_candles_parquet(&candles, &parquet_path).await?;

            println!(
                "✓ Written: {} ({} candles, {:.2} MB)",
                parquet_path.display(),
                candles.len(),
                tokio::fs::metadata(&parquet_path).await?.len() as f64 / 1_048_576.0
            );

            output_paths.push(parquet_path);
        }

        Ok(output_paths)
    }

    /// Build Binance Vision URL
    fn build_url(
        &self,
        market_type: BinanceMarketType,
        symbol: &str,
        year: u32,
        month: u32,
    ) -> String {
        let base = "https://data.binance.vision/data";
        let market = match market_type {
            BinanceMarketType::Spot => "spot",
            BinanceMarketType::Futures => "futures/um",
        };

        format!(
            "{}/{}/monthly/trades/{}/{}-trades-{}-{:02}.zip",
            base, market, symbol, symbol, year, month
        )
    }

    /// Get output path for downloaded ZIP
    fn get_output_path(
        &self,
        market_type: BinanceMarketType,
        symbol: &str,
        year: u32,
        month: u32,
        data_type: &str,
    ) -> PathBuf {
        let market = match market_type {
            BinanceMarketType::Spot => "spot",
            BinanceMarketType::Futures => "futures",
        };

        self.config
            .base_path
            .join("binance")
            .join(market)
            .join(symbol)
            .join(data_type)
            .join(format!("{}-{:02}.zip", year, month))
    }

    /// Get output path for OHLCV Parquet
    fn get_ohlcv_path(
        &self,
        market_type: BinanceMarketType,
        symbol: &str,
        timeframe: &str,
        year: u32,
        month: u32,
    ) -> PathBuf {
        let market = match market_type {
            BinanceMarketType::Spot => "spot",
            BinanceMarketType::Futures => "futures",
        };

        self.config
            .base_path
            .join("binance")
            .join(market)
            .join(symbol)
            .join("ohlcv")
            .join(timeframe)
            .join(format!("{}-{:02}.parquet", year, month))
    }

    /// Write candles to Parquet file
    async fn write_candles_parquet(
        &self,
        candles: &[Candle],
        path: &Path,
    ) -> Result<(), DownloadError> {
        use arrow::array::{Float64Array, Int64Array, UInt32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::fs::File;
        use std::sync::Arc;

        // Create schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("open", DataType::Float64, false),
            Field::new("high", DataType::Float64, false),
            Field::new("low", DataType::Float64, false),
            Field::new("close", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
            Field::new("num_trades", DataType::UInt32, false),
        ]));

        // Extract columns
        let timestamps: Vec<i64> = candles.iter().map(|c| c.timestamp).collect();
        let opens: Vec<f64> = candles.iter().map(|c| c.open).collect();
        let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
        let num_trades: Vec<u32> = candles.iter().map(|c| c.num_trades as u32).collect();

        // Create arrays
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(timestamps)),
                Arc::new(Float64Array::from(opens)),
                Arc::new(Float64Array::from(highs)),
                Arc::new(Float64Array::from(lows)),
                Arc::new(Float64Array::from(closes)),
                Arc::new(Float64Array::from(volumes)),
                Arc::new(UInt32Array::from(num_trades)),
            ],
        )
        .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;

        // Write Parquet with compression
        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();

        let mut writer = ArrowWriter::try_new(file, schema, Some(props))
            .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;

        writer
            .write(&batch)
            .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;

        writer
            .close()
            .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;

        Ok(())
    }
}

#[async_trait]
impl Downloader for BinanceDownloader {
    async fn download(
        &self,
        symbol: &str,
        start_date: NaiveDate,
        end_date: Option<NaiveDate>,
    ) -> Result<PathBuf, DownloadError> {
        // Download all months in range
        let start_year = start_date.year() as u32;
        let start_month = start_date.month();

        let end = end_date.unwrap_or_else(|| chrono::Local::now().date_naive());
        let end_year = end.year() as u32;
        let end_month = end.month();

        for year in start_year..=end_year {
            let month_start = if year == start_year { start_month } else { 1 };
            let month_end = if year == end_year { end_month } else { 12 };

            for month in month_start..=month_end {
                self.download_spot_trades(symbol, year, Some(month)).await?;
            }
        }

        // Return base directory
        Ok(self
            .config
            .base_path
            .join("binance")
            .join("spot")
            .join(symbol))
    }

    fn progress(&self) -> Option<DownloadProgress> {
        self.progress.try_read().ok().and_then(|p| p.clone())
    }

    async fn cancel(&self) {
        // TODO: Implement cancellation logic
    }
}
