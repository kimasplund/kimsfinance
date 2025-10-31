//! Common types for data connectors

use crate::quantitative::heston::OptionQuote;
use async_trait::async_trait;
use thiserror::Error;

/// Errors that can occur when fetching options data
#[derive(Debug, Error)]
pub enum DataError {
    /// Connection to data provider failed
    #[error("Connection failed: {0}")]
    ConnectionError(String),

    /// Authentication with data provider failed
    #[error("Authentication failed: {0}")]
    AuthError(String),

    /// API returned an error
    #[error("API error: {0}")]
    ApiError(String),

    /// Failed to parse data from API response
    #[error("Data parsing error: {0}")]
    ParseError(String),

    /// Network timeout occurred
    #[error("Request timeout: {0}")]
    Timeout(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    /// Data validation failed
    #[error("Data validation failed: {0}")]
    ValidationError(String),
}

/// Unified trait for options data providers
///
/// All data connectors (IBKR, Deribit, etc.) implement this trait to provide
/// a consistent API for fetching options market data.
#[async_trait]
pub trait OptionsDataProvider: Send + Sync {
    /// Fetch current options chain for given underlying
    ///
    /// # Arguments
    ///
    /// * `underlying` - Ticker symbol (e.g., "AAPL", "BTC", "ETH")
    ///
    /// # Returns
    ///
    /// Vector of `OptionQuote` containing market data, implied volatility, and Greeks
    ///
    /// # Errors
    ///
    /// Returns `DataError` if:
    /// - Connection fails
    /// - Authentication fails
    /// - API returns error
    /// - Data parsing fails
    async fn fetch_options_chain(&self, underlying: &str) -> Result<Vec<OptionQuote>, DataError>;

    /// Fetch historical volatility data for calibration
    ///
    /// # Arguments
    ///
    /// * `underlying` - Ticker symbol
    /// * `days` - Number of days of historical data
    ///
    /// # Returns
    ///
    /// Vector of (timestamp, volatility) tuples
    ///
    /// # Errors
    ///
    /// Returns `DataError` if historical data is unavailable or API fails
    async fn fetch_historical_volatility(
        &self,
        underlying: &str,
        days: u32,
    ) -> Result<Vec<(i64, f64)>, DataError>;

    /// Subscribe to real-time options data updates
    ///
    /// # Arguments
    ///
    /// * `underlying` - Ticker symbol to subscribe to
    ///
    /// # Errors
    ///
    /// Returns `DataError` if subscription fails
    async fn subscribe_to_updates(&mut self, underlying: &str) -> Result<(), DataError>;
}
