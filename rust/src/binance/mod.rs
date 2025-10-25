//! Binance trade data aggregation module
//!
//! This module provides high-performance aggregation of tick-level Binance trade data
//! into OHLCV candles for GPU indicator calculations.
//!
//! # Features
//! - Zero-allocation CSV parsing
//! - Memory-efficient streaming for large datasets (52GB+ trade data)
//! - Fast HashMap-based aggregation
//! - Support for multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)
//! - ZIP archive processing for Binance monthly exports
//!
//! # Example
//! ```no_run
//! use kimsfinance_core::binance::{Timeframe, process_binance_month};
//!
//! let candles = process_binance_month("BTCUSDT-trades-2021-01.zip", Timeframe::FiveMinutes)?;
//! println!("Aggregated {} candles", candles.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod trades;

pub use trades::{
    BinanceError, Candle, ParseError, Timeframe, Trade, aggregate_trades_to_candles,
    parse_trade_csv, process_binance_month, stream_aggregate_csv,
};
