//! Interactive Brokers (IBKR) Options Data Connector
//!
//! Fetches real-time options data from Interactive Brokers using the TWS API.
//!
//! # Requirements
//!
//! - TWS (Trader Workstation) or IB Gateway running locally
//! - Funded IBKR account (minimum $500)
//! - Market data subscriptions:
//!   - Level 1 data for underlying securities
//!   - Options data (OPRA for US options)
//!
//! # Configuration
//!
//! Default connection settings:
//! - Paper trading: `127.0.0.1:4002` (or `7497` depending on TWS config)
//! - Live trading: `127.0.0.1:7496`
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::data::ibkr::{IbkrConnector, IbkrConfig};
//! use kimsfinance_core::data::OptionsDataProvider;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = IbkrConfig {
//!     host: "127.0.0.1".to_string(),
//!     port: 7497, // Paper trading (check your TWS/Gateway settings)
//!     client_id: 1,
//! };
//!
//! let connector = IbkrConnector::connect(config).await?;
//! let options = connector.fetch_options_chain("AAPL").await?;
//!
//! for option in options {
//!     println!("Strike {}: {:?}", option.strike, option.option_type);
//! }
//! # Ok(())
//! # }
//! ```

// Historical data downloader for all instrument types (requires arrow/parquet)
#[cfg(all(feature = "data-ibkr", feature = "data-downloaders"))]
pub mod historical;

// Chunked downloader for large historical data requests
#[cfg(all(feature = "data-ibkr", feature = "data-downloaders"))]
pub mod chunked;

#[cfg(all(feature = "data-ibkr", feature = "data-downloaders"))]
pub use historical::{IbkrHistoricalDownloader, InstrumentType};

use crate::data::common::{DataError, OptionsDataProvider};
use crate::quantitative::heston::{Greeks, OptionQuote, OptionType};
use async_trait::async_trait;

#[cfg(feature = "data-ibkr")]
use ibapi::contracts::{OptionChain, tick_types::TickType};
#[cfg(feature = "data-ibkr")]
use ibapi::prelude::*;
#[cfg(feature = "data-ibkr")]
use std::sync::Arc;
#[cfg(feature = "data-ibkr")]
use tokio::time::{Duration, timeout};

/// IBKR connection configuration
#[derive(Debug, Clone)]
pub struct IbkrConfig {
    /// TWS/Gateway host (default: "127.0.0.1")
    pub host: String,
    /// TWS/Gateway port
    /// - Paper trading: 7497 or 4002 (depends on your TWS/Gateway configuration)
    /// - Live trading: 7496 or 4001
    pub port: u16,
    /// Unique client ID (1-32 for most users)
    pub client_id: i32,
}

impl Default for IbkrConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 7497, // Paper trading (common port)
            client_id: 1,
        }
    }
}

/// IBKR options data connector
///
/// Connects to Interactive Brokers TWS or IB Gateway to fetch real-time options market data.
///
/// # Implementation Status
///
/// - [x] Connection to TWS/Gateway
/// - [x] Fetch option chains (strikes, expirations)
/// - [x] Fetch market data (bid/ask/last/IV/Greeks)
/// - [x] Filter liquid options
/// - [ ] Historical volatility (not directly available from IBKR)
/// - [ ] Real-time streaming updates
#[cfg(feature = "data-ibkr")]
pub struct IbkrConnector {
    client: Arc<Client>,
    config: IbkrConfig,
}

#[cfg(feature = "data-ibkr")]
impl IbkrConnector {
    /// Connect to IBKR TWS/Gateway
    ///
    /// # Arguments
    ///
    /// * `config` - Connection configuration (host, port, client_id)
    ///
    /// # Errors
    ///
    /// Returns `DataError::ConnectionError` if connection fails
    pub async fn connect(config: IbkrConfig) -> Result<Self, DataError> {
        let address = format!("{}:{}", config.host, config.port);

        println!("Connecting to IBKR at {}...", address);

        let client = Client::connect(&address, config.client_id)
            .await
            .map_err(|e| {
                DataError::ConnectionError(format!(
                    "Failed to connect to IBKR at {}: {:?}",
                    address, e
                ))
            })?;

        println!("✓ Connected to IBKR successfully!");

        Ok(Self {
            client: Arc::new(client),
            config,
        })
    }

    /// Fetch option chain for a given symbol
    ///
    /// # Process
    ///
    /// 1. Request option parameters (strikes, expirations) from IBKR
    /// 2. For each option contract, request market data (bid/ask/IV/Greeks)
    /// 3. Parse responses into `OptionQuote` structs
    /// 4. Filter for liquid options (volume > 0, valid prices)
    ///
    /// # Returns
    ///
    /// Vector of option quotes with market data
    async fn fetch_option_chain_impl(&self, symbol: &str) -> Result<Vec<OptionQuote>, DataError> {
        println!("Fetching option chain for {}...", symbol);

        // Step 1: Get option chain parameters (strikes, expirations)
        let option_chains = self.fetch_option_parameters(symbol).await?;

        if option_chains.is_empty() {
            return Err(DataError::ApiError(format!(
                "No option chains found for {}",
                symbol
            )));
        }

        println!("✓ Found {} option chain(s)", option_chains.len());

        // Step 2: Get spot price for underlying
        let spot_price = self.fetch_spot_price(symbol).await?;
        println!("✓ Spot price: ${:.2}", spot_price);

        // Step 3: Build contracts and fetch market data
        let mut all_options = Vec::new();

        for chain in option_chains {
            println!(
                "\nProcessing chain: {} (expiry count: {}, strike count: {})",
                chain.trading_class,
                chain.expirations.len(),
                chain.strikes.len()
            );

            // Limit to first 3 expirations to avoid overwhelming API
            for expiration in chain.expirations.iter().take(3) {
                // Limit to strikes near ATM (±20%)
                let atm_strikes: Vec<f64> = chain
                    .strikes
                    .iter()
                    .filter(|&&strike| {
                        let pct_diff = (strike - spot_price).abs() / spot_price;
                        pct_diff < 0.20
                    })
                    .copied()
                    .collect();

                println!(
                    "  Expiration {}: {} ATM strikes (within 20%)",
                    expiration,
                    atm_strikes.len()
                );

                for strike in atm_strikes.iter().take(10) {
                    // Fetch both calls and puts
                    for option_type in [OptionType::Call, OptionType::Put] {
                        match self
                            .fetch_option_data(
                                symbol,
                                expiration,
                                *strike,
                                option_type,
                                spot_price,
                                &chain,
                            )
                            .await
                        {
                            Ok(option) => all_options.push(option),
                            Err(e) => {
                                // Warn but continue (some options may not have data)
                                eprintln!(
                                    "    Warning: Failed to fetch {} ${} {:?}: {}",
                                    expiration, strike, option_type, e
                                );
                            }
                        }
                    }
                }
            }
        }

        println!(
            "\n✓ Successfully fetched {} options with market data",
            all_options.len()
        );

        // Step 4: Filter for liquid options
        let liquid_options: Vec<OptionQuote> = all_options
            .into_iter()
            .filter(|opt| {
                opt.bid.is_some()
                    && opt.ask.is_some()
                    && opt.implied_vol.is_some()
                    && opt.volume > 0.0
            })
            .collect();

        println!(
            "✓ {} liquid options (with IV, bid/ask, volume > 0)",
            liquid_options.len()
        );

        Ok(liquid_options)
    }

    /// Fetch option chain parameters (strikes, expirations)
    async fn fetch_option_parameters(&self, symbol: &str) -> Result<Vec<OptionChain>, DataError> {
        let security_type = SecurityType::Stock;
        let exchange = "SMART";
        let contract_id = 0;

        let mut chain_stream = self
            .client
            .option_chain(symbol, exchange, security_type, contract_id)
            .await
            .map_err(|e| DataError::ApiError(format!("Failed to request option chain: {:?}", e)))?;

        let mut chains = Vec::new();

        // Collect option chains with timeout
        let timeout_duration = Duration::from_secs(10);
        while let Ok(Some(result)) = timeout(timeout_duration, chain_stream.next()).await {
            match result {
                Ok(chain) => chains.push(chain),
                Err(e) => {
                    eprintln!("Error in option chain stream: {:?}", e);
                    break;
                }
            }
        }

        Ok(chains)
    }

    /// Fetch spot price for underlying
    async fn fetch_spot_price(&self, symbol: &str) -> Result<f64, DataError> {
        let contract = Contract::stock(symbol).build();

        let mut subscription = self
            .client
            .market_data(&contract)
            .snapshot()
            .subscribe()
            .await
            .map_err(|e| {
                DataError::ApiError(format!("Failed to subscribe to market data: {:?}", e))
            })?;

        let mut spot_price = None;
        let timeout_duration = Duration::from_secs(5);

        while let Ok(Some(result)) = timeout(timeout_duration, subscription.next()).await {
            match result {
                Ok(TickTypes::Price(price)) => {
                    if matches!(price.tick_type, TickType::Last | TickType::Close) {
                        spot_price = Some(price.price);
                        break;
                    }
                }
                Ok(TickTypes::PriceSize(price_size)) => {
                    if matches!(price_size.price_tick_type, TickType::Last | TickType::Close) {
                        spot_price = Some(price_size.price);
                        break;
                    }
                }
                Err(e) => {
                    return Err(DataError::ApiError(format!(
                        "Error fetching spot price: {:?}",
                        e
                    )));
                }
                _ => {}
            }
        }

        spot_price
            .ok_or_else(|| DataError::ApiError(format!("Failed to get spot price for {}", symbol)))
    }

    /// Fetch market data for a specific option contract
    async fn fetch_option_data(
        &self,
        symbol: &str,
        expiration: &str,
        strike: f64,
        option_type: OptionType,
        spot_price: f64,
        chain: &OptionChain,
    ) -> Result<OptionQuote, DataError> {
        // Build option contract
        let right = match option_type {
            OptionType::Call => "C",
            OptionType::Put => "P",
        };

        let contract = Contract {
            symbol: Symbol::from(symbol),
            security_type: SecurityType::Option,
            exchange: Exchange::from(&chain.exchange),
            currency: Currency::from("USD"),
            strike,
            right: right.to_string(),
            last_trade_date_or_contract_month: expiration.to_string(),
            multiplier: chain.multiplier.clone(),
            ..Default::default()
        };

        // Request market data with IV and Greeks
        let mut subscription = self
            .client
            .market_data(&contract)
            .generic_ticks(&["106"]) // Request IV and Greeks (tick type 106 includes Greeks)
            .snapshot()
            .subscribe()
            .await
            .map_err(|e| {
                DataError::ApiError(format!("Failed to subscribe to option data: {:?}", e))
            })?;

        let mut bid = None;
        let mut ask = None;
        let mut last = None;
        let mut volume = 0.0;
        let mut implied_vol = None;
        let mut greeks = Greeks::default();

        let timeout_duration = Duration::from_secs(3);
        let start = tokio::time::Instant::now();

        while start.elapsed() < timeout_duration {
            match timeout(Duration::from_millis(500), subscription.next()).await {
                Ok(Some(Ok(tick))) => {
                    match tick {
                        TickTypes::Price(price) => match price.tick_type {
                            TickType::Bid => bid = Some(price.price),
                            TickType::Ask => ask = Some(price.price),
                            TickType::Last => last = Some(price.price),
                            _ => {}
                        },
                        TickTypes::Size(size) => {
                            if size.tick_type == TickType::Volume {
                                volume = size.size;
                            }
                        }
                        TickTypes::Generic(generic) => {
                            // Handle implied volatility from generic ticks
                            if generic.tick_type == TickType::OptionImpliedVol {
                                implied_vol = Some(generic.value);
                            }
                        }
                        TickTypes::OptionComputation(computation) => {
                            // Greeks come from OptionComputation
                            if computation.implied_volatility.is_some() {
                                implied_vol = computation.implied_volatility;
                            }
                            greeks.delta = computation.delta;
                            greeks.gamma = computation.gamma;
                            greeks.vega = computation.vega;
                            greeks.theta = computation.theta;
                        }
                        _ => {}
                    }
                }
                Ok(Some(Err(e))) => {
                    return Err(DataError::ApiError(format!(
                        "Error in option data stream: {:?}",
                        e
                    )));
                }
                Ok(None) => break,
                Err(_) => {
                    // Timeout on next(), check if we have enough data
                    if bid.is_some() || ask.is_some() || last.is_some() {
                        break;
                    }
                }
            }
        }

        // Parse expiration to Unix timestamp
        let expiration_ts = self.parse_expiration_to_timestamp(expiration)?;

        Ok(OptionQuote {
            underlying: symbol.to_string(),
            strike,
            expiration: expiration_ts,
            option_type,
            spot_price,
            risk_free_rate: 0.05, // TODO: Fetch from market or config
            bid,
            ask,
            last,
            implied_vol,
            volume,
            open_interest: 0.0, // IBKR doesn't provide OI in real-time data
            greeks: Some(greeks),
        })
    }

    /// Parse IBKR expiration string (YYYYMMDD) to Unix timestamp
    fn parse_expiration_to_timestamp(&self, expiration: &str) -> Result<i64, DataError> {
        use chrono::NaiveDate;

        // Parse YYYYMMDD format
        if expiration.len() != 8 {
            return Err(DataError::ParseError(format!(
                "Invalid expiration format: {}",
                expiration
            )));
        }

        let year: i32 = expiration[0..4].parse().map_err(|_| {
            DataError::ParseError(format!("Invalid year in expiration: {}", expiration))
        })?;
        let month: u32 = expiration[4..6].parse().map_err(|_| {
            DataError::ParseError(format!("Invalid month in expiration: {}", expiration))
        })?;
        let day: u32 = expiration[6..8].parse().map_err(|_| {
            DataError::ParseError(format!("Invalid day in expiration: {}", expiration))
        })?;

        let date = NaiveDate::from_ymd_opt(year, month, day).ok_or_else(|| {
            DataError::ParseError(format!("Invalid date: {}-{}-{}", year, month, day))
        })?;

        // Convert to midnight UTC timestamp
        let datetime = date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| DataError::ParseError("Failed to create datetime".to_string()))?;

        Ok(datetime.and_utc().timestamp())
    }
}

#[cfg(feature = "data-ibkr")]
#[async_trait]
impl OptionsDataProvider for IbkrConnector {
    async fn fetch_options_chain(&self, underlying: &str) -> Result<Vec<OptionQuote>, DataError> {
        self.fetch_option_chain_impl(underlying).await
    }

    async fn fetch_historical_volatility(
        &self,
        _underlying: &str,
        _days: u32,
    ) -> Result<Vec<(i64, f64)>, DataError> {
        // IBKR doesn't provide direct historical IV API
        // Would need to reconstruct from historical option prices
        Err(DataError::ApiError(
            "Historical IV not directly supported for IBKR. Use Deribit or reconstruct from historical option prices.".to_string()
        ))
    }

    async fn subscribe_to_updates(&mut self, _underlying: &str) -> Result<(), DataError> {
        // TODO: Implement streaming updates if needed
        Ok(())
    }
}

#[cfg(not(feature = "data-ibkr"))]
pub struct IbkrConnector;

#[cfg(not(feature = "data-ibkr"))]
impl IbkrConnector {
    pub async fn connect(_config: IbkrConfig) -> Result<Self, DataError> {
        Err(DataError::ConfigError(
            "IBKR connector requires 'data-ibkr' feature flag".to_string(),
        ))
    }
}
