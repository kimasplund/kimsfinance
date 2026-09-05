//! IBKR Historical Data Downloader
//!
//! Downloads historical market data for all supported instrument types:
//! - Stocks (US equities, international) ✅ Tested & Working
//! - Forex/FX (currency pairs) ⚠️ Requires market data subscription
//! - Futures (commodities, indices, bonds) ⚠️ Requires market data subscription
//! - Crypto (BTC, ETH via PAXOS) ❌ Not supported (ibapi crate limitation)
//! - Indices (SPX, NDX, etc.) ⚠️ Requires market data subscription
//! - Bonds (US Treasuries, corporate) ⚠️ Requires market data subscription
//!
//! # Requirements
//!
//! - TWS (Trader Workstation) or IB Gateway running
//! - IBKR account with historical data access
//! - Market data subscriptions for non-stock instruments:
//!   - Forex: IDEALPRO exchange subscription
//!   - Futures: Exchange-specific subscriptions (CME, NYMEX, etc.)
//!   - Crypto: PAXOS subscription + ibapi crate AGGTRADES support
//!
//! # Known Limitations
//!
//! - Crypto downloads not supported: The `ibapi` Rust crate (v2.0.0) doesn't support
//!   the "AGGTRADES" WhatToShow variant required by IBKR for crypto data.
//! - Forex/Futures require paid market data subscriptions from IBKR
//! - Historical data works 24/7 (even when markets closed) as long as TWS/Gateway is running
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::data::ibkr::{IbkrHistoricalDownloader, IbkrConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = IbkrConfig::default();
//! let downloader = IbkrHistoricalDownloader::connect(config).await?;
//!
//! // Download stock data
//! downloader.download_stock("AAPL", "1 Y", "1 day").await?;
//!
//! // Download forex data
//! downloader.download_forex("EUR", "USD", "1 M", "1 hour").await?;
//!
//! // Download futures data
//! downloader.download_futures("ES", "1 M", "15 mins").await?;
//! # Ok(())
//! # }
//! ```

use super::IbkrConfig;
use crate::data::downloaders::DownloadError;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "data-ibkr")]
use ibapi::contracts::Contract;
#[cfg(feature = "data-ibkr")]
use ibapi::market_data::TradingHours;
#[cfg(feature = "data-ibkr")]
use ibapi::market_data::historical::{Bar, BarSize, Duration, HistoricalData, WhatToShow};
#[cfg(feature = "data-ibkr")]
use ibapi::prelude::*;

/// Instrument type for IBKR downloads
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrumentType {
    Stock,
    Forex,
    Futures,
    Crypto,
    Index,
    Bond,
}

/// IBKR historical data downloader
#[cfg(feature = "data-ibkr")]
pub struct IbkrHistoricalDownloader {
    pub(crate) client: Arc<Client>,
    pub(crate) config: IbkrConfig,
    pub(crate) base_path: PathBuf,
}

#[cfg(feature = "data-ibkr")]
impl IbkrHistoricalDownloader {
    /// Connect to IBKR TWS/Gateway for historical data downloads
    pub async fn connect(config: IbkrConfig) -> Result<Self, DownloadError> {
        let address = format!("{}:{}", config.host, config.port);

        let client = Client::connect(&address, config.client_id)
            .await
            .map_err(|e| DownloadError::Network(format!("IBKR connection failed: {}", e)))?;

        println!("✓ Connected to IBKR at {}", address);

        Ok(Self {
            client: Arc::new(client),
            config,
            base_path: PathBuf::from("data/ibkr"),
        })
    }

    /// Download historical stock data
    ///
    /// # Arguments
    ///
    /// - `symbol`: Stock ticker (e.g., "AAPL", "TSLA")
    /// - `duration`: Duration string (e.g., "1 Y", "6 M", "1 W", "5 D")
    /// - `bar_size`: Bar size (e.g., "1 day", "1 hour", "5 mins", "1 min")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// downloader.download_stock("AAPL", "1 Y", "1 day").await?;
    /// ```
    pub async fn download_stock(
        &self,
        symbol: &str,
        duration: &str,
        bar_size: &str,
    ) -> Result<PathBuf, DownloadError> {
        println!("Downloading stock data: {}", symbol);

        // Build stock contract
        let contract = Contract::stock(symbol).build();

        // Download historical data
        let bars = self
            .fetch_historical_data(&contract, duration, bar_size, "TRADES")
            .await?;

        // Save to Parquet
        let output_path = self
            .base_path
            .join("stocks")
            .join(symbol)
            .join(format!("{}.parquet", bar_size.replace(" ", "_")));

        self.save_bars_parquet(&bars, &output_path).await?;

        println!("✓ Saved: {} ({} bars)", output_path.display(), bars.len());

        Ok(output_path)
    }

    /// Download historical forex data
    ///
    /// # Arguments
    ///
    /// - `base`: Base currency (e.g., "EUR", "GBP")
    /// - `quote`: Quote currency (e.g., "USD", "JPY")
    /// - `duration`: Duration string
    /// - `bar_size`: Bar size
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// downloader.download_forex("EUR", "USD", "1 M", "1 hour").await?;
    /// ```
    pub async fn download_forex(
        &self,
        base: &str,
        quote: &str,
        duration: &str,
        bar_size: &str,
    ) -> Result<PathBuf, DownloadError> {
        let pair = format!("{}/{}", base, quote);
        println!("Downloading forex data: {}", pair);

        // Build forex contract
        // Note: May require market data subscriptions for specific forex pairs
        let contract = Contract::forex(base, quote).build();

        // Download historical data
        let bars = self
            .fetch_historical_data(
                &contract, duration, bar_size, "MIDPOINT", // Forex uses midpoint
            )
            .await?;

        // Save to Parquet
        let output_path = self
            .base_path
            .join("forex")
            .join(&pair.replace("/", ""))
            .join(format!("{}.parquet", bar_size.replace(" ", "_")));

        self.save_bars_parquet(&bars, &output_path).await?;

        println!("✓ Saved: {} ({} bars)", output_path.display(), bars.len());

        Ok(output_path)
    }

    /// Download historical futures data
    ///
    /// # Arguments
    ///
    /// - `symbol`: Futures symbol (e.g., "ES", "NQ", "CL")
    /// - `duration`: Duration string
    /// - `bar_size`: Bar size
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// downloader.download_futures("ES", "1 M", "15 mins").await?;
    /// ```
    pub async fn download_futures(
        &self,
        symbol: &str,
        duration: &str,
        bar_size: &str,
    ) -> Result<PathBuf, DownloadError> {
        println!("Downloading futures data: {}", symbol);

        // Build futures contract (front month)
        // Note: May require market data subscriptions for specific futures
        let contract = Contract::futures(symbol).front_month().build();

        // Download historical data
        let bars = self
            .fetch_historical_data(&contract, duration, bar_size, "TRADES")
            .await?;

        // Save to Parquet
        let output_path = self
            .base_path
            .join("futures")
            .join(symbol)
            .join(format!("{}.parquet", bar_size.replace(" ", "_")));

        self.save_bars_parquet(&bars, &output_path).await?;

        println!("✓ Saved: {} ({} bars)", output_path.display(), bars.len());

        Ok(output_path)
    }

    /// Download historical crypto data
    ///
    /// # Arguments
    ///
    /// - `symbol`: Crypto symbol (e.g., "BTC", "ETH", "LTC")
    /// - `duration`: Duration string
    /// - `bar_size`: Bar size
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// downloader.download_crypto("BTC", "1 M", "1 hour").await?;
    /// ```
    pub async fn download_crypto(
        &self,
        symbol: &str,
        duration: &str,
        bar_size: &str,
    ) -> Result<PathBuf, DownloadError> {
        println!("Downloading crypto data: {}", symbol);

        // Build crypto contract
        let contract = Contract::crypto(symbol).build();

        // Download historical data
        // NOTE: IBKR's API requires "AGGTRADES" for crypto, but the ibapi Rust crate (v2.0.0)
        // doesn't support this WhatToShow variant yet. Crypto downloads will fail until this
        // is added to the crate. Using TRADES for now which will return an error from IBKR.
        let bars = self
            .fetch_historical_data(
                &contract, duration, bar_size,
                "TRADES", // Should be "AGGTRADES" but not supported by ibapi crate
            )
            .await?;

        // Save to Parquet
        let output_path = self
            .base_path
            .join("crypto")
            .join(symbol)
            .join(format!("{}.parquet", bar_size.replace(" ", "_")));

        self.save_bars_parquet(&bars, &output_path).await?;

        println!("✓ Saved: {} ({} bars)", output_path.display(), bars.len());

        Ok(output_path)
    }

    /// Fetch historical data from IBKR
    pub(crate) async fn fetch_historical_data(
        &self,
        contract: &Contract,
        duration: &str,
        bar_size: &str,
        what_to_show: &str,
    ) -> Result<Vec<Bar>, DownloadError> {
        use tokio::time::timeout;

        // Parse bar size
        let bar_size_enum = Self::parse_bar_size(bar_size)?;

        // Parse what to show
        let what_enum = Self::parse_what_to_show(what_to_show)?;

        // Request historical data with timeout
        // Increased timeout for large requests (e.g., years of minute data)
        let result: HistoricalData = timeout(
            tokio::time::Duration::from_secs(120),
            self.client.historical_data(
                contract,
                None, // end_datetime: use current time
                Self::parse_duration(duration)?,
                bar_size_enum,
                Some(what_enum),
                TradingHours::Regular, // use regular trading hours
            ),
        )
        .await
        .map_err(|_| DownloadError::Network("Timeout fetching historical data".to_string()))?
        .map_err(|e| DownloadError::Network(format!("IBKR error: {}", e)))?;

        Ok(result.bars)
    }

    /// Parse duration string to IBKR duration
    pub(crate) fn parse_duration(duration: &str) -> Result<Duration, DownloadError> {
        // Duration format: "1 Y", "6 M", "1 W", "5 D"
        let parts: Vec<&str> = duration.trim().split_whitespace().collect();
        if parts.len() != 2 {
            return Err(DownloadError::InvalidFormat(format!(
                "Invalid duration: {}",
                duration
            )));
        }

        let value: i32 = parts[0].parse().map_err(|_| {
            DownloadError::InvalidFormat(format!("Invalid duration value: {}", parts[0]))
        })?;

        let duration = match parts[1] {
            "Y" | "year" | "years" => Duration::years(value),
            "M" | "month" | "months" => Duration::months(value),
            "W" | "week" | "weeks" => Duration::weeks(value),
            "D" | "day" | "days" => Duration::days(value),
            _ => {
                return Err(DownloadError::InvalidFormat(format!(
                    "Invalid duration unit: {}",
                    parts[1]
                )));
            }
        };

        Ok(duration)
    }

    /// Parse bar size string to IBKR bar size enum
    pub(crate) fn parse_bar_size(bar_size: &str) -> Result<BarSize, DownloadError> {
        let size = match bar_size.trim().to_lowercase().as_str() {
            "1 sec" | "1sec" => BarSize::Sec,
            "5 secs" | "5secs" => BarSize::Sec5,
            "15 secs" | "15secs" => BarSize::Sec15,
            "30 secs" | "30secs" => BarSize::Sec30,
            "1 min" | "1min" => BarSize::Min,
            "2 mins" | "2mins" => BarSize::Min2,
            "3 mins" | "3mins" => BarSize::Min3,
            "5 mins" | "5mins" => BarSize::Min5,
            "15 mins" | "15mins" => BarSize::Min15,
            "20 mins" | "20mins" => BarSize::Min20,
            "30 mins" | "30mins" => BarSize::Min30,
            "1 hour" | "1hour" => BarSize::Hour,
            "1 day" | "1day" => BarSize::Day,
            "1 week" | "1week" => BarSize::Week,
            "1 month" | "1month" => BarSize::Month,
            _ => {
                return Err(DownloadError::InvalidFormat(format!(
                    "Invalid bar size: {}",
                    bar_size
                )));
            }
        };

        Ok(size)
    }

    /// Parse what to show string to IBKR enum
    pub(crate) fn parse_what_to_show(what: &str) -> Result<WhatToShow, DownloadError> {
        let show = match what.trim().to_uppercase().as_str() {
            "TRADES" => WhatToShow::Trades,
            "MIDPOINT" => WhatToShow::MidPoint,
            "BID" => WhatToShow::Bid,
            "ASK" => WhatToShow::Ask,
            "BID_ASK" => WhatToShow::BidAsk,
            _ => {
                return Err(DownloadError::InvalidFormat(format!(
                    "Invalid what_to_show: {}",
                    what
                )));
            }
        };

        Ok(show)
    }

    /// Save historical bars to Parquet file
    pub(crate) async fn save_bars_parquet(
        &self,
        bars: &[Bar],
        path: &PathBuf,
    ) -> Result<(), DownloadError> {
        use arrow::array::{Float64Array, Int64Array, TimestampMillisecondArray};
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::fs::File;

        // Create directory
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Create schema
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "timestamp",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                false,
            ),
            Field::new("open", DataType::Float64, false),
            Field::new("high", DataType::Float64, false),
            Field::new("low", DataType::Float64, false),
            Field::new("close", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
            Field::new("wap", DataType::Float64, false), // Weighted average price
            Field::new("count", DataType::Int64, false), // Number of trades
        ]));

        // Extract columns
        let timestamps: Vec<i64> = bars
            .iter()
            .map(|b| b.date.unix_timestamp() * 1000) // Convert seconds to milliseconds
            .collect();
        let opens: Vec<f64> = bars.iter().map(|b| b.open).collect();
        let highs: Vec<f64> = bars.iter().map(|b| b.high).collect();
        let lows: Vec<f64> = bars.iter().map(|b| b.low).collect();
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
        let waps: Vec<f64> = bars.iter().map(|b| b.wap).collect();
        let counts: Vec<i64> = bars.iter().map(|b| b.count as i64).collect();

        // Create record batch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(TimestampMillisecondArray::from(timestamps)),
                Arc::new(Float64Array::from(opens)),
                Arc::new(Float64Array::from(highs)),
                Arc::new(Float64Array::from(lows)),
                Arc::new(Float64Array::from(closes)),
                Arc::new(Float64Array::from(volumes)),
                Arc::new(Float64Array::from(waps)),
                Arc::new(Int64Array::from(counts)),
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
