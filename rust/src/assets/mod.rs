//! Asset Type System for Multi-Asset Trading
//!
//! Comprehensive support for different asset classes with their specific market conventions,
//! data structures, and trading rules.
//!
//! # Supported Asset Classes
//! - **Equity (Stocks)**: Corporate actions, tick size rules, market hours
//! - **Futures**: Contract specifications, expiration, roll-over logic
//! - **Options**: Greeks, strike chains, expiration, American/European
//! - **Forex**: Currency pairs, pip values, cross rates, session times
//! - **Crypto**: 24/7 trading, exchange-specific conventions
//! - **CFD**: Contract for Difference with leverage and margin
//! - **Index**: Composite instruments, cash-settled
//!
//! # Architecture
//! ```text
//! AssetType (enum)
//!   ↓
//! AssetSpec (specifications)
//!   ↓
//! Asset (trait) ← implemented by specific asset types
//!   ├── validate_price()
//!   ├── normalize_symbol()
//!   ├── calculate_value()
//!   └── is_market_open()
//! ```
//!
//! # Example
//! ```rust,ignore
//! use kimsfinance_core::assets::*;
//!
//! // Create equity asset
//! let aapl = EquityAsset::new("AAPL", Exchange::Nasdaq);
//!
//! // Validate price with tick size rules
//! let valid_price = aapl.validate_price(150.05); // Ok
//! let invalid_price = aapl.validate_price(150.001); // Error: invalid tick size
//!
//! // Create futures contract
//! let es = FuturesContract::new(
//!     "ES",
//!     Expiration::Monthly(2025, 3), // March 2025
//!     50.0,  // $50 multiplier
//!     0.25,  // 0.25 point tick size
//! );
//!
//! // Calculate contract value
//! let value = es.calculate_value(5000.0); // 5000 * 50 = $250,000
//! ```

pub mod cfd;
pub mod crypto;
pub mod equity;
pub mod forex;
pub mod futures;
pub mod index;
pub mod options;
pub mod specs;

pub use cfd::*;
pub use crypto::*;
pub use equity::*;
pub use forex::*;
pub use futures::*;
pub use index::*;
pub use options::*;
pub use specs::*;

use chrono::{DateTime, Datelike, NaiveTime, Utc, Weekday};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Asset class enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetType {
    /// Equity instruments (stocks)
    Equity,
    /// Futures contracts
    Futures,
    /// Options contracts
    Options,
    /// Foreign exchange (forex)
    Forex,
    /// Cryptocurrencies
    Crypto,
    /// Contracts for Difference
    CFD,
    /// Market indices
    Index,
}

impl fmt::Display for AssetType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssetType::Equity => write!(f, "EQUITY"),
            AssetType::Futures => write!(f, "FUTURES"),
            AssetType::Options => write!(f, "OPTIONS"),
            AssetType::Forex => write!(f, "FOREX"),
            AssetType::Crypto => write!(f, "CRYPTO"),
            AssetType::CFD => write!(f, "CFD"),
            AssetType::Index => write!(f, "INDEX"),
        }
    }
}

/// Asset validation and calculation errors
#[derive(Debug, Error)]
pub enum AssetError {
    #[error("Invalid price: {0}")]
    InvalidPrice(String),

    #[error("Invalid tick size: {0}")]
    InvalidTickSize(String),

    #[error("Market closed: {0}")]
    MarketClosed(String),

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("Contract expired: {0}")]
    ContractExpired(String),

    #[error("Invalid strike price: {0}")]
    InvalidStrike(String),

    #[error("Invalid expiration: {0}")]
    InvalidExpiration(String),

    #[error("Asset specification error: {0}")]
    SpecificationError(String),

    #[error("Conversion error: {0}")]
    ConversionError(String),
}

pub type AssetResult<T> = Result<T, AssetError>;

/// Core trait for all asset types
pub trait Asset: Send + Sync {
    /// Get asset type
    fn asset_type(&self) -> AssetType;

    /// Get symbol/identifier
    fn symbol(&self) -> &str;

    /// Validate price according to asset-specific rules
    fn validate_price(&self, price: f64) -> AssetResult<f64>;

    /// Normalize symbol to standard format
    fn normalize_symbol(&self, symbol: &str) -> AssetResult<String>;

    /// Calculate contract/position value
    fn calculate_value(&self, price: f64, quantity: f64) -> AssetResult<f64>;

    /// Check if market is currently open
    fn is_market_open(&self, timestamp: DateTime<Utc>) -> bool;

    /// Get minimum tick size
    fn tick_size(&self) -> f64;

    /// Get minimum quantity increment
    fn quantity_increment(&self) -> f64;

    /// Get contract multiplier (futures/options)
    fn contract_multiplier(&self) -> f64 {
        1.0
    }

    /// Get asset specification
    fn specification(&self) -> &AssetSpec;
}

/// Trading session (market hours)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSession {
    /// Session name (e.g., "Regular", "Pre-Market", "After-Hours")
    pub name: String,

    /// Days of week (Monday=1, Sunday=7)
    pub days: Vec<Weekday>,

    /// Session start time (UTC)
    pub start_time: NaiveTime,

    /// Session end time (UTC)
    pub end_time: NaiveTime,
}

impl TradingSession {
    /// Check if timestamp is within this session
    pub fn is_active(&self, timestamp: DateTime<Utc>) -> bool {
        let weekday = timestamp.weekday();
        let time = timestamp.time();

        self.days.contains(&weekday) && time >= self.start_time && time <= self.end_time
    }
}

/// Exchange/venue identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Exchange {
    // US Equity Exchanges
    NYSE,
    Nasdaq,
    AMEX,
    ARCA,
    BATS,

    // Futures Exchanges
    CME,   // Chicago Mercantile Exchange
    CBOT,  // Chicago Board of Trade
    NYMEX, // New York Mercantile Exchange
    COMEX, // Commodity Exchange
    CBOE,  // Chicago Board Options Exchange
    ICE,   // Intercontinental Exchange

    // Crypto Exchanges
    Binance,
    Coinbase,
    Kraken,
    FTX,
    Bybit,

    // Forex
    Forex,

    // Other
    OTC,
    Custom(u32),
}

impl fmt::Display for Exchange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Exchange::NYSE => write!(f, "NYSE"),
            Exchange::Nasdaq => write!(f, "NASDAQ"),
            Exchange::AMEX => write!(f, "AMEX"),
            Exchange::ARCA => write!(f, "ARCA"),
            Exchange::BATS => write!(f, "BATS"),
            Exchange::CME => write!(f, "CME"),
            Exchange::CBOT => write!(f, "CBOT"),
            Exchange::NYMEX => write!(f, "NYMEX"),
            Exchange::COMEX => write!(f, "COMEX"),
            Exchange::CBOE => write!(f, "CBOE"),
            Exchange::ICE => write!(f, "ICE"),
            Exchange::Binance => write!(f, "BINANCE"),
            Exchange::Coinbase => write!(f, "COINBASE"),
            Exchange::Kraken => write!(f, "KRAKEN"),
            Exchange::FTX => write!(f, "FTX"),
            Exchange::Bybit => write!(f, "BYBIT"),
            Exchange::Forex => write!(f, "FOREX"),
            Exchange::OTC => write!(f, "OTC"),
            Exchange::Custom(id) => write!(f, "CUSTOM_{}", id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_type_display() {
        assert_eq!(AssetType::Equity.to_string(), "EQUITY");
        assert_eq!(AssetType::Futures.to_string(), "FUTURES");
        assert_eq!(AssetType::Options.to_string(), "OPTIONS");
    }

    #[test]
    fn test_exchange_display() {
        assert_eq!(Exchange::NYSE.to_string(), "NYSE");
        assert_eq!(Exchange::CME.to_string(), "CME");
        assert_eq!(Exchange::Binance.to_string(), "BINANCE");
    }
}
