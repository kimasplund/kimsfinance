//! Chunked Historical Data Downloader
//!
//! Downloads large amounts of historical data by splitting requests into chunks
//! to work around IBKR's bar count limits (~20,000-25,000 bars per request).
//!
//! # Supported Instruments
//!
//! - **Stocks**: `download_stock_chunked()` - US equities, international stocks
//! - **Futures**: `download_futures_chunked()` - Commodities (gold, oil), indices, bonds
//! - **Forex**: `download_forex_chunked()` - Currency pairs (EUR/USD, GBP/JPY)
//! - **Options**: `download_options_chunked()` - Equity options with strike/expiration
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
//! // Download a full year of 1-minute bars (chunked automatically)
//! downloader.download_stock_chunked("AAPL", "1 Y", "1 min").await?;
//!
//! // Download gold futures
//! downloader.download_futures_chunked("GC", "1 Y", "1 hour").await?;
//!
//! // Download forex data
//! downloader.download_forex_chunked("EUR", "USD", "1 Y", "1 hour").await?;
//!
//! // Download options data
//! downloader.download_options_chunked("AAPL", "20241220", 150.0, "C", "6 M", "1 hour").await?;
//! # Ok(())
//! # }
//! ```

use super::{IbkrConfig, IbkrHistoricalDownloader};
use crate::data::downloaders::DownloadError;
use std::path::PathBuf;

#[cfg(feature = "data-ibkr")]
use ibapi::contracts::Contract;
#[cfg(feature = "data-ibkr")]
use ibapi::market_data::TradingHours;
#[cfg(feature = "data-ibkr")]
use ibapi::market_data::historical::{Bar, BarSize, Duration, WhatToShow};
#[cfg(feature = "data-ibkr")]
use ibapi::prelude::*;
#[cfg(feature = "data-ibkr")]
use time::OffsetDateTime;

#[cfg(feature = "data-ibkr")]
impl IbkrHistoricalDownloader {
    /// Download historical stock data with automatic chunking for large requests
    ///
    /// # Arguments
    ///
    /// - `symbol`: Stock ticker (e.g., "AAPL", "TSLA")
    /// - `total_duration`: Total duration to download (e.g., "1 Y")
    /// - `bar_size`: Bar size (e.g., "1 min", "5 mins", "1 hour")
    ///
    /// # Chunking Strategy
    ///
    /// - 1-minute bars: Max 3 months per chunk
    /// - 5-minute bars: Max 6 months per chunk
    /// - 15-minute bars: Max 1 year per chunk
    /// - Hourly+ bars: No chunking needed
    ///
    /// # File Organization
    ///
    /// Each chunk is saved as a separate file with date range in name:
    /// - `AAPL/2024-01-02_to_2024-03-31_1min.parquet` (Chunk 1)
    /// - `AAPL/2024-04-01_to_2024-06-30_1min.parquet` (Chunk 2)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Downloads a full year by making 4 requests of 3 months each
    /// downloader.download_stock_chunked("AAPL", "1 Y", "1 min").await?;
    /// ```
    ///
    /// # Returns
    ///
    /// Returns the directory path containing all chunk files
    pub async fn download_stock_chunked(
        &self,
        symbol: &str,
        total_duration: &str,
        bar_size: &str,
    ) -> Result<PathBuf, DownloadError> {
        println!("Downloading stock data with chunking: {}", symbol);
        println!("Total duration: {}, Bar size: {}", total_duration, bar_size);

        // Build stock contract
        let contract = Contract::stock(symbol).build();

        // Download with chunking (returns Vec of (bars, start_date, end_date))
        let chunks = self
            .fetch_historical_data_chunked_separate(&contract, total_duration, bar_size, "TRADES")
            .await?;

        println!("✓ Downloaded {} chunks", chunks.len());

        // Save each chunk as a separate file
        let base_dir = self.base_path.join("stocks").join(symbol);
        let bar_size_str = bar_size.replace(" ", "");

        for (chunk_bars, start_date, end_date) in chunks {
            if chunk_bars.is_empty() {
                continue;
            }

            // Format dates as YYYY-MM-DD
            let start_str = format!(
                "{:04}-{:02}-{:02}",
                start_date.year(),
                start_date.month() as u8,
                start_date.day()
            );
            let end_str = format!(
                "{:04}-{:02}-{:02}",
                end_date.year(),
                end_date.month() as u8,
                end_date.day()
            );

            let filename = format!("{}_to_{}_{}.parquet", start_str, end_str, bar_size_str);
            let output_path = base_dir.join(&filename);

            self.save_bars_parquet(&chunk_bars, &output_path).await?;

            println!("✓ Saved: {} ({} bars)", filename, chunk_bars.len());
        }

        println!("✓ All chunks saved to: {}", base_dir.display());

        Ok(base_dir)
    }

    /// Download historical futures data with automatic chunking
    ///
    /// # Arguments
    ///
    /// - `symbol`: Futures symbol (e.g., "GC" for gold, "CL" for oil, "ES" for S&P 500)
    /// - `total_duration`: Total duration to download (e.g., "1 Y")
    /// - `bar_size`: Bar size (e.g., "1 min", "5 mins", "1 hour")
    ///
    /// # Commodities Examples
    ///
    /// - Gold: `download_futures_chunked("GC", "1 Y", "1 hour")`
    /// - Oil: `download_futures_chunked("CL", "1 Y", "1 hour")`
    /// - Silver: `download_futures_chunked("SI", "1 Y", "1 hour")`
    /// - Natural Gas: `download_futures_chunked("NG", "1 Y", "1 hour")`
    ///
    /// # Returns
    ///
    /// Returns the directory path containing all chunk files
    pub async fn download_futures_chunked(
        &self,
        symbol: &str,
        total_duration: &str,
        bar_size: &str,
    ) -> Result<PathBuf, DownloadError> {
        println!("Downloading futures data with chunking: {}", symbol);
        println!("Total duration: {}, Bar size: {}", total_duration, bar_size);

        // Build futures contract (front month)
        let contract = Contract::futures(symbol).front_month().build();

        // Download with chunking
        let chunks = self
            .fetch_historical_data_chunked_separate(&contract, total_duration, bar_size, "TRADES")
            .await?;

        println!("✓ Downloaded {} chunks", chunks.len());

        // Save each chunk as a separate file
        let base_dir = self.base_path.join("futures").join(symbol);
        let bar_size_str = bar_size.replace(" ", "");

        for (chunk_bars, start_date, end_date) in chunks {
            if chunk_bars.is_empty() {
                continue;
            }

            // Format dates as YYYY-MM-DD
            let start_str = format!(
                "{:04}-{:02}-{:02}",
                start_date.year(),
                start_date.month() as u8,
                start_date.day()
            );
            let end_str = format!(
                "{:04}-{:02}-{:02}",
                end_date.year(),
                end_date.month() as u8,
                end_date.day()
            );

            let filename = format!("{}_to_{}_{}.parquet", start_str, end_str, bar_size_str);
            let output_path = base_dir.join(&filename);

            self.save_bars_parquet(&chunk_bars, &output_path).await?;

            println!("✓ Saved: {} ({} bars)", filename, chunk_bars.len());
        }

        println!("✓ All chunks saved to: {}", base_dir.display());

        Ok(base_dir)
    }

    /// Download historical forex data with automatic chunking
    ///
    /// # Arguments
    ///
    /// - `base_currency`: Base currency (e.g., "EUR", "GBP")
    /// - `quote_currency`: Quote currency (e.g., "USD", "JPY")
    /// - `total_duration`: Total duration to download (e.g., "1 Y")
    /// - `bar_size`: Bar size (e.g., "1 min", "5 mins", "1 hour")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// downloader.download_forex_chunked("EUR", "USD", "1 Y", "1 hour").await?;
    /// ```
    ///
    /// # Returns
    ///
    /// Returns the directory path containing all chunk files
    pub async fn download_forex_chunked(
        &self,
        base_currency: &str,
        quote_currency: &str,
        total_duration: &str,
        bar_size: &str,
    ) -> Result<PathBuf, DownloadError> {
        let pair = format!("{}{}", base_currency, quote_currency);
        println!("Downloading forex data with chunking: {}", pair);
        println!("Total duration: {}, Bar size: {}", total_duration, bar_size);

        // Build forex contract
        let contract = Contract::forex(base_currency, quote_currency).build();

        // Download with chunking
        let chunks = self
            .fetch_historical_data_chunked_separate(
                &contract,
                total_duration,
                bar_size,
                "MIDPOINT", // Forex typically uses MIDPOINT
            )
            .await?;

        println!("✓ Downloaded {} chunks", chunks.len());

        // Save each chunk as a separate file
        let base_dir = self.base_path.join("forex").join(&pair);
        let bar_size_str = bar_size.replace(" ", "");

        for (chunk_bars, start_date, end_date) in chunks {
            if chunk_bars.is_empty() {
                continue;
            }

            // Format dates as YYYY-MM-DD
            let start_str = format!(
                "{:04}-{:02}-{:02}",
                start_date.year(),
                start_date.month() as u8,
                start_date.day()
            );
            let end_str = format!(
                "{:04}-{:02}-{:02}",
                end_date.year(),
                end_date.month() as u8,
                end_date.day()
            );

            let filename = format!("{}_to_{}_{}.parquet", start_str, end_str, bar_size_str);
            let output_path = base_dir.join(&filename);

            self.save_bars_parquet(&chunk_bars, &output_path).await?;

            println!("✓ Saved: {} ({} bars)", filename, chunk_bars.len());
        }

        println!("✓ All chunks saved to: {}", base_dir.display());

        Ok(base_dir)
    }

    /// Download historical options data with automatic chunking
    ///
    /// # Arguments
    ///
    /// - `underlying`: Underlying symbol (e.g., "AAPL", "SPY")
    /// - `expiration`: Expiration date in YYYYMMDD format (e.g., "20241220")
    /// - `strike`: Strike price (e.g., 150.0)
    /// - `right`: "C" for Call or "P" for Put
    /// - `total_duration`: Total duration to download (e.g., "1 Y")
    /// - `bar_size`: Bar size (e.g., "1 min", "5 mins", "1 hour")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // AAPL $150 Call expiring Dec 20, 2024
    /// downloader.download_options_chunked(
    ///     "AAPL", "20241220", 150.0, "C", "6 M", "1 hour"
    /// ).await?;
    /// ```
    ///
    /// # Returns
    ///
    /// Returns the directory path containing all chunk files
    pub async fn download_options_chunked(
        &self,
        underlying: &str,
        expiration: &str,
        strike: f64,
        right: &str,
        total_duration: &str,
        bar_size: &str,
    ) -> Result<PathBuf, DownloadError> {
        println!(
            "Downloading options data with chunking: {} {} {} {}",
            underlying, expiration, strike, right
        );
        println!("Total duration: {}, Bar size: {}", total_duration, bar_size);

        // Build options contract
        let contract = Contract {
            symbol: Symbol::from(underlying),
            security_type: SecurityType::Option,
            exchange: Exchange::from("SMART"),
            currency: Currency::from("USD"),
            strike,
            right: right.to_string(),
            last_trade_date_or_contract_month: expiration.to_string(),
            multiplier: "100".to_string(),
            ..Default::default()
        };

        // Download with chunking
        let chunks = self
            .fetch_historical_data_chunked_separate(&contract, total_duration, bar_size, "TRADES")
            .await?;

        println!("✓ Downloaded {} chunks", chunks.len());

        // Save each chunk as a separate file
        let option_name = format!("{}_{}_{}{}", underlying, expiration, strike, right);
        let base_dir = self.base_path.join("options").join(&option_name);
        let bar_size_str = bar_size.replace(" ", "");

        for (chunk_bars, start_date, end_date) in chunks {
            if chunk_bars.is_empty() {
                continue;
            }

            // Format dates as YYYY-MM-DD
            let start_str = format!(
                "{:04}-{:02}-{:02}",
                start_date.year(),
                start_date.month() as u8,
                start_date.day()
            );
            let end_str = format!(
                "{:04}-{:02}-{:02}",
                end_date.year(),
                end_date.month() as u8,
                end_date.day()
            );

            let filename = format!("{}_to_{}_{}.parquet", start_str, end_str, bar_size_str);
            let output_path = base_dir.join(&filename);

            self.save_bars_parquet(&chunk_bars, &output_path).await?;

            println!("✓ Saved: {} ({} bars)", filename, chunk_bars.len());
        }

        println!("✓ All chunks saved to: {}", base_dir.display());

        Ok(base_dir)
    }

    /// Fetch historical data with automatic chunking, returning separate chunks
    ///
    /// Returns Vec of (bars, start_date, end_date) for each chunk
    async fn fetch_historical_data_chunked_separate(
        &self,
        contract: &Contract,
        total_duration: &str,
        bar_size: &str,
        what_to_show: &str,
    ) -> Result<Vec<(Vec<Bar>, OffsetDateTime, OffsetDateTime)>, DownloadError> {
        use tokio::time::{Duration as TokioDuration, sleep, timeout};

        // Determine chunk size based on bar size
        let chunk_duration = self.get_chunk_duration(bar_size)?;

        // Calculate total duration in days
        let total_days = self.duration_to_days(total_duration)?;
        let chunk_days = self.duration_to_days(&chunk_duration)?;

        // If no chunking needed, use single request
        if total_days <= chunk_days {
            println!("No chunking needed (within single request limit)");
            let bars = self
                .fetch_historical_data(contract, total_duration, bar_size, what_to_show)
                .await?;

            // Get start and end dates from bars
            if let (Some(first), Some(last)) = (bars.first(), bars.last()) {
                let start_date = first.date;
                let end_date = last.date;
                return Ok(vec![(bars, start_date, end_date)]);
            } else {
                return Ok(vec![]);
            }
        }

        // Calculate number of chunks
        let num_chunks = (total_days as f64 / chunk_days as f64).ceil() as usize;
        println!(
            "Splitting into {} chunks of {} days each",
            num_chunks, chunk_days
        );

        let mut all_chunks = Vec::new();
        let mut current_end = OffsetDateTime::now_utc();

        // Parse bar size and what_to_show for IBKR API
        let bar_size_enum = Self::parse_bar_size(bar_size)?;
        let what_enum = Self::parse_what_to_show(what_to_show)?;
        let duration_enum = Self::parse_duration(&chunk_duration)?;

        for chunk_num in 0..num_chunks {
            println!("\n--- Chunk {}/{} ---", chunk_num + 1, num_chunks);
            println!("End time: {}", current_end);

            // Request historical data for this chunk
            let result = timeout(
                TokioDuration::from_secs(120),
                self.client.historical_data(
                    contract,
                    Some(current_end),
                    duration_enum.clone(),
                    bar_size_enum,
                    Some(what_enum),
                    TradingHours::Regular,
                ),
            )
            .await
            .map_err(|_| DownloadError::Network("Timeout fetching chunk".to_string()))?
            .map_err(|e| DownloadError::Network(format!("IBKR error: {}", e)))?;

            let chunk_bars = result.bars;
            println!("✓ Chunk {}: {} bars", chunk_num + 1, chunk_bars.len());

            if chunk_bars.is_empty() {
                println!("No more data available");
                break;
            }

            // Get start and end dates from this chunk
            if let (Some(first), Some(last)) = (chunk_bars.first(), chunk_bars.last()) {
                let start_date = first.date;
                let end_date = last.date;
                all_chunks.push((chunk_bars.clone(), start_date, end_date));

                // Set next end time to earliest bar in this chunk
                current_end = start_date;
            }

            // Rate limiting: 10-second delay between chunks
            if chunk_num < num_chunks - 1 {
                println!("Waiting 10 seconds (rate limiting)...");
                sleep(TokioDuration::from_secs(10)).await;
            }
        }

        println!("\n✓ Downloaded {} chunks", all_chunks.len());

        Ok(all_chunks)
    }

    /// Fetch historical data with automatic chunking (legacy method - merges all chunks)
    #[allow(dead_code)]
    async fn fetch_historical_data_chunked(
        &self,
        contract: &Contract,
        total_duration: &str,
        bar_size: &str,
        what_to_show: &str,
    ) -> Result<Vec<Bar>, DownloadError> {
        use tokio::time::{Duration as TokioDuration, sleep, timeout};

        // Determine chunk size based on bar size
        let chunk_duration = self.get_chunk_duration(bar_size)?;

        // Calculate total duration in days
        let total_days = self.duration_to_days(total_duration)?;
        let chunk_days = self.duration_to_days(&chunk_duration)?;

        // If no chunking needed, use single request
        if total_days <= chunk_days {
            println!("No chunking needed (within single request limit)");
            return self
                .fetch_historical_data(contract, total_duration, bar_size, what_to_show)
                .await;
        }

        // Calculate number of chunks
        let num_chunks = (total_days as f64 / chunk_days as f64).ceil() as usize;
        println!(
            "Splitting into {} chunks of {} days each",
            num_chunks, chunk_days
        );

        let mut all_bars = Vec::new();
        let mut current_end = OffsetDateTime::now_utc();

        // Parse bar size and what_to_show for IBKR API
        let bar_size_enum = Self::parse_bar_size(bar_size)?;
        let what_enum = Self::parse_what_to_show(what_to_show)?;
        let duration_enum = Self::parse_duration(&chunk_duration)?;

        for chunk_num in 0..num_chunks {
            println!("\n--- Chunk {}/{} ---", chunk_num + 1, num_chunks);
            println!("End time: {}", current_end);

            // Request historical data for this chunk
            let result = timeout(
                TokioDuration::from_secs(120),
                self.client.historical_data(
                    contract,
                    Some(current_end), // Use specific end time for this chunk
                    duration_enum.clone(),
                    bar_size_enum,
                    Some(what_enum),
                    TradingHours::Regular,
                ),
            )
            .await
            .map_err(|_| DownloadError::Network("Timeout fetching chunk".to_string()))?
            .map_err(|e| DownloadError::Network(format!("IBKR error: {}", e)))?;

            let chunk_bars = result.bars;
            println!("✓ Chunk {}: {} bars", chunk_num + 1, chunk_bars.len());

            if chunk_bars.is_empty() {
                println!("No more data available");
                break;
            }

            // Find the earliest bar in this chunk to set next end time
            if let Some(earliest_bar) = chunk_bars.first() {
                current_end = earliest_bar.date;
            }

            all_bars.extend(chunk_bars);

            // Rate limiting: 10-second delay between chunks (IBKR limit: ~60 req/10min)
            if chunk_num < num_chunks - 1 {
                println!("Waiting 10 seconds (rate limiting)...");
                sleep(TokioDuration::from_secs(10)).await;
            }
        }

        // Sort bars by date (oldest first)
        all_bars.sort_by_key(|b| b.date);

        // Remove duplicates (bars at chunk boundaries)
        all_bars.dedup_by_key(|b| b.date.unix_timestamp());

        println!("\n✓ Total bars after merging: {}", all_bars.len());

        Ok(all_bars)
    }

    /// Get appropriate chunk duration based on bar size
    fn get_chunk_duration(&self, bar_size: &str) -> Result<String, DownloadError> {
        let duration = match bar_size.trim().to_lowercase().as_str() {
            "1 min" | "1min" => "3 M", // 3 months for 1-minute bars
            "2 mins" | "2mins" => "3 M",
            "3 mins" | "3mins" => "3 M",
            "5 mins" | "5mins" => "6 M",   // 6 months for 5-minute bars
            "15 mins" | "15mins" => "1 Y", // 1 year for 15-minute bars
            "30 mins" | "30mins" => "1 Y",
            "1 hour" | "1hour" => "2 Y", // 2 years for hourly bars
            _ => "1 Y",                  // 1 year for all other bars
        };

        Ok(duration.to_string())
    }

    /// Convert duration string to approximate days
    fn duration_to_days(&self, duration: &str) -> Result<i32, DownloadError> {
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

        let days = match parts[1] {
            "Y" | "year" | "years" => value * 365,
            "M" | "month" | "months" => value * 30,
            "W" | "week" | "weeks" => value * 7,
            "D" | "day" | "days" => value,
            _ => {
                return Err(DownloadError::InvalidFormat(format!(
                    "Invalid duration unit: {}",
                    parts[1]
                )));
            }
        };

        Ok(days)
    }
}
