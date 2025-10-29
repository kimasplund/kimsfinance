//! Options Data Connectors
//!
//! Unified interface for fetching options market data from multiple sources:
//! - IBKR (Interactive Brokers) for equity options
//! - Deribit for crypto options (BTC/ETH)
//!
//! # Architecture
//!
//! All connectors implement the `OptionsDataProvider` trait, providing a uniform
//! API for the Heston calibrator regardless of data source.
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::data::{OptionsDataProvider, deribit::DeribitConnector};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Connect to Deribit
//! let connector = DeribitConnector::connect().await?;
//!
//! // Fetch BTC options chain
//! let options = connector.fetch_options_chain("BTC").await?;
//!
//! println!("Found {} options", options.len());
//! # Ok(())
//! # }
//! ```

pub mod common;

#[cfg(feature = "data-deribit")]
pub mod deribit;

#[cfg(feature = "data-ibkr")]
pub mod ibkr;

pub use common::DataError;

// Re-export trait from common module
pub use common::OptionsDataProvider;
