//! Yahoo Finance data downloader
//!
//! Downloads stocks and options data from Yahoo Finance API.
//! Inspired by pandas-datareader: https://github.com/pydata/pandas-datareader
//!
//! # Features
//!
//! - ✅ Historical stock prices (daily/intraday)
//! - ✅ Options chains with Greeks
//! - ✅ Real-time quotes
//! - ✅ No API key required (free)
//!
//! # Example
//!
//! ```rust,no_run
//! use kimsfinance_core::data::downloaders::{YahooDownloader, DownloadConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = DownloadConfig::default();
//! let downloader = YahooDownloader::new(config);
//!
//! // Download AAPL stock data
//! downloader.download_stock("AAPL", "2024-01-01", "2024-12-31").await?;
//!
//! // Download options chain
//! downloader.download_options_chain("AAPL", None).await?;
//! # Ok(())
//! # }
//! ```

use super::common::{DownloadConfig, DownloadError, DownloadProgress, Downloader};
use crate::quantitative::heston::{OptionQuote, OptionType};
use async_trait::async_trait;
use chrono::{Datelike, NaiveDate};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Stock OHLCV data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockQuote {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
    pub adj_close: f64,
}

/// Options chain data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionsChain {
    pub symbol: String,
    pub expiration_date: String,
    pub calls: Vec<OptionQuote>,
    pub puts: Vec<OptionQuote>,
    pub underlying_price: f64,
}

/// Yahoo Finance downloader
pub struct YahooDownloader {
    config: DownloadConfig,
    client: Client,
    progress: Arc<RwLock<Option<DownloadProgress>>>,
}

impl YahooDownloader {
    /// Retry helper with exponential backoff for rate limiting
    async fn retry_with_backoff<F, Fut, T>(
        &self,
        operation: F,
        _max_retries: u32, // Keep parameter for backward compatibility but don't use it
    ) -> Result<T, DownloadError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, DownloadError>>,
    {
        const MAX_BACKOFF: u64 = 120; // Maximum backoff in seconds
        let mut retry_count = 0;

        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(DownloadError::ApiError(ref msg)) if msg.contains("429") => {
                    retry_count += 1;

                    // Exponential backoff: 2, 4, 8, 16, 32, 64, 120, 120, 120...
                    // After reaching MAX_BACKOFF (120s), keep retrying every 120s indefinitely
                    let delay_secs = std::cmp::min(2u64.pow(retry_count), MAX_BACKOFF);

                    if delay_secs < MAX_BACKOFF {
                        println!(
                            "⏳ Rate limited (429). Exponential backoff: waiting {} seconds... (attempt {})",
                            delay_secs, retry_count
                        );
                    } else {
                        println!(
                            "⏳ Rate limited (429). Steady retry: waiting {} seconds... (attempt {})",
                            delay_secs, retry_count
                        );
                    }

                    tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Create new Yahoo Finance downloader
    pub fn new(config: DownloadConfig) -> Self {
        let client = Client::builder()
            .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .timeout(std::time::Duration::from_secs(30))
            .cookie_store(true) // Enable cookie jar for session management
            .build()
            .unwrap();

        Self {
            config,
            client,
            progress: Arc::new(RwLock::new(None)),
        }
    }

    /// Download historical stock data
    ///
    /// # Arguments
    ///
    /// - `symbol`: Stock ticker (e.g., "AAPL", "SPY")
    /// - `start_date`: Start date (YYYY-MM-DD)
    /// - `end_date`: End date (YYYY-MM-DD)
    pub async fn download_stock(
        &self,
        symbol: &str,
        start_date: &str,
        end_date: &str,
    ) -> Result<PathBuf, DownloadError> {
        println!(
            "Downloading {} stock data: {} to {}",
            symbol, start_date, end_date
        );

        // Parse dates
        let start = NaiveDate::parse_from_str(start_date, "%Y-%m-%d")
            .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;
        let end = NaiveDate::parse_from_str(end_date, "%Y-%m-%d")
            .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;

        // Convert to Unix timestamps
        let start_ts = start.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
        let end_ts = end.and_hms_opt(23, 59, 59).unwrap().and_utc().timestamp();

        // Build Yahoo Finance URL
        let url = format!(
            "https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history",
            symbol, start_ts, end_ts
        );

        // Download CSV with retry logic for rate limiting
        let csv_data = self.retry_with_backoff(|| async {
            let response = self
                .client
                .get(&url)
                .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")
                .header("Accept-Language", "en-US,en;q=0.5")
                .header("Accept-Encoding", "gzip, deflate, br")
                .header("Referer", "https://finance.yahoo.com/")
                .header("DNT", "1")
                .header("Connection", "keep-alive")
                .header("Upgrade-Insecure-Requests", "1")
                .header("Sec-Fetch-Dest", "document")
                .header("Sec-Fetch-Mode", "navigate")
                .header("Sec-Fetch-Site", "same-site")
                .send()
                .await
                .map_err(|e| DownloadError::Network(e.to_string()))?;

            if !response.status().is_success() {
                return Err(DownloadError::ApiError(format!(
                    "HTTP {}: {}",
                    response.status(),
                    response.text().await.unwrap_or_default()
                )));
            }

            let csv_data = response
                .text()
                .await
                .map_err(|e| DownloadError::Network(e.to_string()))?;

            Ok(csv_data)
        }, 4).await?; // Max 4 retries (total 5 attempts)

        // Parse CSV to quotes
        let quotes = self.parse_stock_csv(&csv_data)?;

        // Save to Parquet
        let year = start.year();
        let output_path = self.get_stock_path(symbol, year);

        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        self.write_stock_parquet(&quotes, &output_path).await?;

        println!(
            "✓ Written: {} ({} quotes, {:.2} KB)",
            output_path.display(),
            quotes.len(),
            tokio::fs::metadata(&output_path).await?.len() as f64 / 1024.0
        );

        Ok(output_path)
    }

    /// Download historical stock data with automatic chunking
    ///
    /// Splits large date ranges into monthly chunks for better file organization
    /// and reusability across projects.
    ///
    /// # Arguments
    ///
    /// - `symbol`: Stock ticker (e.g., "AAPL", "TSLA")
    /// - `start_date`: Start date in YYYY-MM-DD format
    /// - `end_date`: End date in YYYY-MM-DD format
    /// - `chunk_months`: Number of months per chunk (default: 3)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kimsfinance_core::data::downloaders::{YahooDownloader, DownloadConfig};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let downloader = YahooDownloader::new(DownloadConfig::default());
    /// // Download 5 years of data in 3-month chunks
    /// downloader.download_stock_chunked("AAPL", "2020-01-01", "2024-12-31", 3).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Returns
    ///
    /// Returns the directory path containing all chunk files
    pub async fn download_stock_chunked(
        &self,
        symbol: &str,
        start_date: &str,
        end_date: &str,
        chunk_months: u32,
    ) -> Result<PathBuf, DownloadError> {
        println!(
            "Downloading {} stock data with chunking: {} to {}",
            symbol, start_date, end_date
        );
        println!("Chunk size: {} months", chunk_months);

        // Parse dates
        let start = NaiveDate::parse_from_str(start_date, "%Y-%m-%d")
            .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;
        let end = NaiveDate::parse_from_str(end_date, "%Y-%m-%d")
            .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;

        // Calculate chunks
        let mut current_start = start;
        let mut chunks = Vec::new();

        while current_start < end {
            // Calculate chunk end date (chunk_months later or end_date, whichever is earlier)
            let chunk_end = std::cmp::min(
                current_start
                    .checked_add_months(chrono::Months::new(chunk_months))
                    .ok_or_else(|| DownloadError::InvalidFormat("Date overflow".to_string()))?
                    .pred_opt()
                    .unwrap_or(current_start),
                end,
            );

            chunks.push((current_start, chunk_end));

            // Move to next chunk
            current_start = chunk_end.succ_opt().unwrap_or(end);
        }

        println!("Splitting into {} chunks", chunks.len());

        // Base directory for this symbol
        let base_dir = self
            .config
            .base_path
            .join("yahoo")
            .join("stocks")
            .join(symbol)
            .join("daily");

        tokio::fs::create_dir_all(&base_dir).await?;

        // Download each chunk
        for (i, (chunk_start, chunk_end)) in chunks.iter().enumerate() {
            println!("\n--- Chunk {}/{} ---", i + 1, chunks.len());
            println!("Date range: {} to {}", chunk_start, chunk_end);

            // Download this chunk
            let chunk_start_str = chunk_start.format("%Y-%m-%d").to_string();
            let chunk_end_str = chunk_end.format("%Y-%m-%d").to_string();

            // Convert to Unix timestamps
            let start_ts = chunk_start
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp();
            let end_ts = chunk_end
                .and_hms_opt(23, 59, 59)
                .unwrap()
                .and_utc()
                .timestamp();

            // Build Yahoo Finance URL
            let url = format!(
                "https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history",
                symbol, start_ts, end_ts
            );

            // Download CSV with retry logic
            let csv_data = self.retry_with_backoff(|| async {
                let response = self
                    .client
                    .get(&url)
                    .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")
                    .header("Accept-Language", "en-US,en;q=0.5")
                    .header("Accept-Encoding", "gzip, deflate, br")
                    .header("Referer", "https://finance.yahoo.com/")
                    .header("DNT", "1")
                    .header("Connection", "keep-alive")
                    .header("Upgrade-Insecure-Requests", "1")
                    .header("Sec-Fetch-Dest", "document")
                    .header("Sec-Fetch-Mode", "navigate")
                    .header("Sec-Fetch-Site", "same-site")
                    .send()
                    .await
                    .map_err(|e| DownloadError::Network(e.to_string()))?;

                if !response.status().is_success() {
                    return Err(DownloadError::ApiError(format!(
                        "HTTP {}: {}",
                        response.status(),
                        response.text().await.unwrap_or_default()
                    )));
                }

                let csv_data = response
                    .text()
                    .await
                    .map_err(|e| DownloadError::Network(e.to_string()))?;

                Ok(csv_data)
            }, 4).await?;

            // Parse CSV to quotes
            let quotes = self.parse_stock_csv(&csv_data)?;

            if quotes.is_empty() {
                println!("No data in this chunk, skipping...");
                continue;
            }

            // Save with date range in filename
            let filename = format!("{}_to_{}_daily.parquet", chunk_start_str, chunk_end_str);
            let output_path = base_dir.join(&filename);

            self.write_stock_parquet(&quotes, &output_path).await?;

            println!(
                "✓ Saved: {} ({} quotes, {:.2} KB)",
                filename,
                quotes.len(),
                tokio::fs::metadata(&output_path).await?.len() as f64 / 1024.0
            );

            // Rate limiting: 1 second delay between chunks
            if i < chunks.len() - 1 {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }

        println!("\n✓ All chunks saved to: {}", base_dir.display());

        Ok(base_dir)
    }

    /// Download options chain for a symbol
    ///
    /// # Arguments
    ///
    /// - `symbol`: Stock ticker
    /// - `expiration`: Optional expiration date (YYYY-MM-DD), fetches nearest if None
    pub async fn download_options_chain(
        &self,
        symbol: &str,
        expiration: Option<&str>,
    ) -> Result<PathBuf, DownloadError> {
        println!("Downloading {} options chain", symbol);

        // Get available expiration dates with retry logic
        let expirations = self
            .retry_with_backoff(|| async { self.get_option_expirations(symbol).await }, 4)
            .await?;

        if expirations.is_empty() {
            return Err(DownloadError::ApiError(
                "No option expirations available".to_string(),
            ));
        }

        // Select expiration
        let target_exp = match expiration {
            Some(exp) => exp.to_string(),
            None => expirations[0].clone(), // Nearest expiration
        };

        println!("Using expiration: {}", target_exp);

        // Fetch options chain
        let chain = self.fetch_options_chain(symbol, &target_exp).await?;

        // Save to Parquet
        let output_path = self.get_options_path(symbol, &target_exp);

        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        self.write_options_parquet(&chain, &output_path).await?;

        println!(
            "✓ Written: {} ({} calls, {} puts)",
            output_path.display(),
            chain.calls.len(),
            chain.puts.len()
        );

        Ok(output_path)
    }

    /// Download multiple options chains with automatic chunking by expiration
    ///
    /// Downloads all available option expirations or a specified number of nearest expirations.
    /// Each expiration is saved as a separate file for easy reusability.
    ///
    /// # Arguments
    ///
    /// - `symbol`: Stock ticker (e.g., "AAPL", "TSLA")
    /// - `max_expirations`: Maximum number of expirations to download (None = all)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kimsfinance_core::data::downloaders::{YahooDownloader, DownloadConfig};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let downloader = YahooDownloader::new(DownloadConfig::default());
    /// // Download first 12 expirations for AAPL
    /// downloader.download_options_chunked("AAPL", Some(12)).await?;
    ///
    /// // Download all available expirations
    /// downloader.download_options_chunked("TSLA", None).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Returns
    ///
    /// Returns the directory path containing all expiration files
    pub async fn download_options_chunked(
        &self,
        symbol: &str,
        max_expirations: Option<usize>,
    ) -> Result<PathBuf, DownloadError> {
        println!("Downloading {} options chains with chunking", symbol);

        // Get available expiration dates with retry logic
        let mut expirations = self
            .retry_with_backoff(|| async { self.get_option_expirations(symbol).await }, 4)
            .await?;

        if expirations.is_empty() {
            return Err(DownloadError::ApiError(
                "No option expirations available".to_string(),
            ));
        }

        // Limit to max_expirations if specified
        if let Some(max) = max_expirations {
            expirations.truncate(max);
        }

        println!("Found {} expirations to download", expirations.len());

        // Base directory for this symbol
        let base_dir = self
            .config
            .base_path
            .join("yahoo")
            .join("options")
            .join(symbol);

        tokio::fs::create_dir_all(&base_dir).await?;

        // Download each expiration
        for (i, expiration) in expirations.iter().enumerate() {
            println!(
                "\n--- Expiration {}/{}: {} ---",
                i + 1,
                expirations.len(),
                expiration
            );

            // Fetch options chain for this expiration
            let chain = self
                .retry_with_backoff(
                    || async { self.fetch_options_chain(symbol, expiration).await },
                    4,
                )
                .await?;

            // Save with expiration date in filename
            let filename = format!("{}_options.parquet", expiration);
            let output_path = base_dir.join(&filename);

            self.write_options_parquet(&chain, &output_path).await?;

            println!(
                "✓ Saved: {} ({} calls, {} puts)",
                filename,
                chain.calls.len(),
                chain.puts.len()
            );

            // Rate limiting: 1 second delay between expirations
            if i < expirations.len() - 1 {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }

        println!("\n✓ All expirations saved to: {}", base_dir.display());

        Ok(base_dir)
    }

    /// Get available option expiration dates
    async fn get_option_expirations(&self, symbol: &str) -> Result<Vec<String>, DownloadError> {
        // Yahoo Finance options endpoint
        let url = format!(
            "https://query1.finance.yahoo.com/v7/finance/options/{}",
            symbol
        );

        let response = self
            .client
            .get(&url)
            .header("Accept", "application/json")
            .header("Accept-Language", "en-US,en;q=0.5")
            .header("Referer", "https://finance.yahoo.com/")
            .header("DNT", "1")
            .header("Connection", "keep-alive")
            .header("Sec-Fetch-Dest", "empty")
            .header("Sec-Fetch-Mode", "cors")
            .header("Sec-Fetch-Site", "same-site")
            .send()
            .await
            .map_err(|e| DownloadError::Network(e.to_string()))?;

        if !response.status().is_success() {
            return Err(DownloadError::ApiError(format!(
                "HTTP {}: Failed to fetch options expirations",
                response.status()
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;

        // Parse expirationDates from response
        // Structure: {"optionChain": {"result": [{"expirationDates": [timestamp1, timestamp2, ...]}]}}
        let expirations = json
            .get("optionChain")
            .and_then(|chain| chain.get("result"))
            .and_then(|result| result.as_array())
            .and_then(|arr| arr.first())
            .and_then(|first| first.get("expirationDates"))
            .and_then(|exp| exp.as_array())
            .ok_or_else(|| {
                DownloadError::InvalidFormat("Invalid options API response".to_string())
            })?;

        // Convert Unix timestamps to YYYY-MM-DD strings
        let mut dates = Vec::new();
        for exp in expirations {
            if let Some(timestamp) = exp.as_i64() {
                let datetime = chrono::DateTime::from_timestamp(timestamp, 0)
                    .ok_or_else(|| DownloadError::InvalidFormat("Invalid timestamp".to_string()))?;
                dates.push(datetime.format("%Y-%m-%d").to_string());
            }
        }

        Ok(dates)
    }

    /// Fetch full options chain for an expiration
    async fn fetch_options_chain(
        &self,
        symbol: &str,
        expiration: &str,
    ) -> Result<OptionsChain, DownloadError> {
        // Convert date string to Unix timestamp
        let date = NaiveDate::parse_from_str(expiration, "%Y-%m-%d")
            .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;
        let timestamp = date.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();

        // Yahoo Finance options chain endpoint with date parameter
        let url = format!(
            "https://query1.finance.yahoo.com/v7/finance/options/{}?date={}",
            symbol, timestamp
        );

        let response = self
            .client
            .get(&url)
            .header("Accept", "application/json")
            .header("Accept-Language", "en-US,en;q=0.5")
            .header("Referer", "https://finance.yahoo.com/")
            .header("DNT", "1")
            .header("Connection", "keep-alive")
            .header("Sec-Fetch-Dest", "empty")
            .header("Sec-Fetch-Mode", "cors")
            .header("Sec-Fetch-Site", "same-site")
            .send()
            .await
            .map_err(|e| DownloadError::Network(e.to_string()))?;

        if !response.status().is_success() {
            return Err(DownloadError::ApiError(format!(
                "HTTP {}: Failed to fetch options chain",
                response.status()
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;

        // Parse response structure
        // {"optionChain": {"result": [{"quote": {"regularMarketPrice": 150.0}, "options": [{"calls": [...], "puts": [...]}]}]}}
        let result = json
            .get("optionChain")
            .and_then(|chain| chain.get("result"))
            .and_then(|result| result.as_array())
            .and_then(|arr| arr.first())
            .ok_or_else(|| {
                DownloadError::InvalidFormat("Invalid options chain response".to_string())
            })?;

        // Get underlying price
        let underlying_price = result
            .get("quote")
            .and_then(|quote| quote.get("regularMarketPrice"))
            .and_then(|price| price.as_f64())
            .unwrap_or(0.0);

        // Get options data
        let options_data = result
            .get("options")
            .and_then(|opts| opts.as_array())
            .and_then(|arr| arr.first())
            .ok_or_else(|| DownloadError::InvalidFormat("No options data found".to_string()))?;

        // Parse calls
        let calls = self.parse_options_array(
            symbol,
            options_data.get("calls").and_then(|c| c.as_array()),
            OptionType::Call,
            expiration,
            underlying_price,
        )?;

        // Parse puts
        let puts = self.parse_options_array(
            symbol,
            options_data.get("puts").and_then(|p| p.as_array()),
            OptionType::Put,
            expiration,
            underlying_price,
        )?;

        Ok(OptionsChain {
            symbol: symbol.to_string(),
            expiration_date: expiration.to_string(),
            calls,
            puts,
            underlying_price,
        })
    }

    /// Parse options array from Yahoo Finance JSON
    fn parse_options_array(
        &self,
        symbol: &str,
        options: Option<&Vec<serde_json::Value>>,
        option_type: OptionType,
        expiration: &str,
        underlying_price: f64,
    ) -> Result<Vec<OptionQuote>, DownloadError> {
        let options = match options {
            Some(opts) => opts,
            None => return Ok(Vec::new()),
        };

        let mut quotes = Vec::new();

        for opt in options {
            // Extract fields from Yahoo Finance format
            let strike = opt.get("strike").and_then(|s| s.as_f64()).unwrap_or(0.0);
            let bid = opt.get("bid").and_then(|b| b.as_f64());
            let ask = opt.get("ask").and_then(|a| a.as_f64());
            let last_price = opt.get("lastPrice").and_then(|l| l.as_f64());
            let volume = opt.get("volume").and_then(|v| v.as_u64());
            let open_interest = opt.get("openInterest").and_then(|oi| oi.as_u64());
            let implied_volatility = opt.get("impliedVolatility").and_then(|iv| iv.as_f64());

            // Convert expiration date to Unix timestamp
            let expiration_date = NaiveDate::parse_from_str(expiration, "%Y-%m-%d")
                .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;
            let expiration_timestamp = expiration_date
                .and_hms_opt(16, 0, 0) // Options expire at 4pm ET
                .unwrap()
                .and_utc()
                .timestamp();

            let quote = OptionQuote {
                underlying: symbol.to_string(),
                strike,
                expiration: expiration_timestamp,
                option_type,
                spot_price: underlying_price,
                risk_free_rate: 0.05, // TODO: Fetch real risk-free rate
                bid,
                ask,
                last: last_price,
                implied_vol: implied_volatility,
                volume: volume.unwrap_or(0) as f64,
                open_interest: open_interest.unwrap_or(0) as f64,
                greeks: None, // Yahoo doesn't provide Greeks directly
            };

            quotes.push(quote);
        }

        Ok(quotes)
    }

    /// Parse Yahoo Finance stock CSV
    fn parse_stock_csv(&self, csv: &str) -> Result<Vec<StockQuote>, DownloadError> {
        let mut quotes = Vec::new();

        for (i, line) in csv.lines().enumerate() {
            if i == 0 {
                continue; // Skip header
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 7 {
                continue;
            }

            // Parse: Date,Open,High,Low,Close,Adj Close,Volume
            let date = NaiveDate::parse_from_str(parts[0], "%Y-%m-%d")
                .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;

            let quote = StockQuote {
                timestamp: date.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp() * 1000,
                open: parts[1].parse().unwrap_or(0.0),
                high: parts[2].parse().unwrap_or(0.0),
                low: parts[3].parse().unwrap_or(0.0),
                close: parts[4].parse().unwrap_or(0.0),
                adj_close: parts[5].parse().unwrap_or(0.0),
                volume: parts[6].parse().unwrap_or(0),
            };

            quotes.push(quote);
        }

        Ok(quotes)
    }

    /// Get output path for stock data
    fn get_stock_path(&self, symbol: &str, year: i32) -> PathBuf {
        self.config
            .base_path
            .join("yahoo")
            .join("stocks")
            .join(symbol)
            .join("daily")
            .join(format!("{}.parquet", year))
    }

    /// Get output path for options chain
    fn get_options_path(&self, symbol: &str, expiration: &str) -> PathBuf {
        self.config
            .base_path
            .join("yahoo")
            .join("options")
            .join(symbol)
            .join("chain")
            .join(format!("{}.parquet", expiration))
    }

    /// Write stock quotes to Parquet
    async fn write_stock_parquet(
        &self,
        quotes: &[StockQuote],
        path: &Path,
    ) -> Result<(), DownloadError> {
        use arrow::array::{Float64Array, Int64Array, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::fs::File;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("open", DataType::Float64, false),
            Field::new("high", DataType::Float64, false),
            Field::new("low", DataType::Float64, false),
            Field::new("close", DataType::Float64, false),
            Field::new("adj_close", DataType::Float64, false),
            Field::new("volume", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    quotes.iter().map(|q| q.timestamp),
                )),
                Arc::new(Float64Array::from_iter_values(
                    quotes.iter().map(|q| q.open),
                )),
                Arc::new(Float64Array::from_iter_values(
                    quotes.iter().map(|q| q.high),
                )),
                Arc::new(Float64Array::from_iter_values(quotes.iter().map(|q| q.low))),
                Arc::new(Float64Array::from_iter_values(
                    quotes.iter().map(|q| q.close),
                )),
                Arc::new(Float64Array::from_iter_values(
                    quotes.iter().map(|q| q.adj_close),
                )),
                Arc::new(UInt64Array::from_iter_values(
                    quotes.iter().map(|q| q.volume),
                )),
            ],
        )
        .map_err(|e| DownloadError::InvalidFormat(e.to_string()))?;

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

    /// Write options chain to Parquet
    async fn write_options_parquet(
        &self,
        chain: &OptionsChain,
        path: &Path,
    ) -> Result<(), DownloadError> {
        use arrow::array::{Float64Array, StringArray, UInt8Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::fs::File;
        use std::sync::Arc;

        // Combine calls and puts
        let mut all_options: Vec<&OptionQuote> = chain.calls.iter().collect();
        all_options.extend(chain.puts.iter());

        if all_options.is_empty() {
            // Create empty file
            File::create(path)?;
            return Ok(());
        }

        // Create schema for options data matching OptionQuote struct
        let schema = Arc::new(Schema::new(vec![
            Field::new("underlying", DataType::Utf8, false),
            Field::new("strike", DataType::Float64, false),
            Field::new("expiration", DataType::Int64, false),
            Field::new("option_type", DataType::Utf8, false),
            Field::new("spot_price", DataType::Float64, false),
            Field::new("risk_free_rate", DataType::Float64, false),
            Field::new("bid", DataType::Float64, true),
            Field::new("ask", DataType::Float64, true),
            Field::new("last", DataType::Float64, true),
            Field::new("implied_vol", DataType::Float64, true),
            Field::new("volume", DataType::Float64, false),
            Field::new("open_interest", DataType::Float64, false),
            // Greeks (if available)
            Field::new("delta", DataType::Float64, true),
            Field::new("gamma", DataType::Float64, true),
            Field::new("vega", DataType::Float64, true),
            Field::new("theta", DataType::Float64, true),
            Field::new("rho", DataType::Float64, true),
        ]));

        // Extract columns from OptionQuote fields
        let underlyings: Vec<String> = all_options.iter().map(|o| o.underlying.clone()).collect();
        let strikes: Vec<f64> = all_options.iter().map(|o| o.strike).collect();
        let expirations: Vec<i64> = all_options.iter().map(|o| o.expiration).collect();
        let option_types: Vec<String> = all_options
            .iter()
            .map(|o| match o.option_type {
                OptionType::Call => "call".to_string(),
                OptionType::Put => "put".to_string(),
            })
            .collect();
        let spot_prices: Vec<f64> = all_options.iter().map(|o| o.spot_price).collect();
        let risk_free_rates: Vec<f64> = all_options.iter().map(|o| o.risk_free_rate).collect();
        let bids: Vec<Option<f64>> = all_options.iter().map(|o| o.bid).collect();
        let asks: Vec<Option<f64>> = all_options.iter().map(|o| o.ask).collect();
        let lasts: Vec<Option<f64>> = all_options.iter().map(|o| o.last).collect();
        let implied_vols: Vec<Option<f64>> = all_options.iter().map(|o| o.implied_vol).collect();
        let volumes: Vec<f64> = all_options.iter().map(|o| o.volume).collect();
        let open_interests: Vec<f64> = all_options.iter().map(|o| o.open_interest).collect();

        // Extract Greeks from Option<Greeks> (flatten Option<Option<f64>> to Option<f64>)
        let deltas: Vec<Option<f64>> = all_options
            .iter()
            .map(|o| o.greeks.as_ref().and_then(|g| g.delta))
            .collect();
        let gammas: Vec<Option<f64>> = all_options
            .iter()
            .map(|o| o.greeks.as_ref().and_then(|g| g.gamma))
            .collect();
        let vegas: Vec<Option<f64>> = all_options
            .iter()
            .map(|o| o.greeks.as_ref().and_then(|g| g.vega))
            .collect();
        let thetas: Vec<Option<f64>> = all_options
            .iter()
            .map(|o| o.greeks.as_ref().and_then(|g| g.theta))
            .collect();
        let rhos: Vec<Option<f64>> = all_options
            .iter()
            .map(|o| o.greeks.as_ref().and_then(|g| g.rho_greek))
            .collect();

        // Create arrays
        use arrow::array::Int64Array;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(underlyings)),
                Arc::new(Float64Array::from(strikes)),
                Arc::new(Int64Array::from(expirations)),
                Arc::new(StringArray::from(option_types)),
                Arc::new(Float64Array::from(spot_prices)),
                Arc::new(Float64Array::from(risk_free_rates)),
                Arc::new(Float64Array::from(bids)),
                Arc::new(Float64Array::from(asks)),
                Arc::new(Float64Array::from(lasts)),
                Arc::new(Float64Array::from(implied_vols)),
                Arc::new(Float64Array::from(volumes)),
                Arc::new(Float64Array::from(open_interests)),
                Arc::new(Float64Array::from(deltas)),
                Arc::new(Float64Array::from(gammas)),
                Arc::new(Float64Array::from(vegas)),
                Arc::new(Float64Array::from(thetas)),
                Arc::new(Float64Array::from(rhos)),
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
impl Downloader for YahooDownloader {
    async fn download(
        &self,
        symbol: &str,
        start_date: NaiveDate,
        end_date: Option<NaiveDate>,
    ) -> Result<PathBuf, DownloadError> {
        let end = end_date.unwrap_or_else(|| chrono::Local::now().date_naive());

        self.download_stock(
            symbol,
            &start_date.format("%Y-%m-%d").to_string(),
            &end.format("%Y-%m-%d").to_string(),
        )
        .await
    }

    fn progress(&self) -> Option<DownloadProgress> {
        self.progress.try_read().ok().and_then(|p| p.clone())
    }

    async fn cancel(&self) {
        // TODO: Implement cancellation
    }
}
