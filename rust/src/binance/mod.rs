//! Binance trade data aggregation module
//!
//! This module provides high-performance aggregation of tick-level Binance trade data
//! into OHLCV candles for GPU indicator calculations.
//!
//! # Features
//! - Zero-allocation CSV parsing
//! - Memory-efficient streaming for large datasets (52GB+ trade data)
//! - Fast HashMap-based aggregation
//! - Flexible timeframe system supporting any duration (5m, 3m, 45s, 2h, etc.)
//! - ZIP archive processing for Binance monthly exports
//! - Date range utilities for multi-month data processing
//! - File discovery for batch processing
//!
//! # Example
//! ```no_run
//! use kimsfinance_core::binance::{Timeframe, process_binance_month};
//!
//! // Parse flexible timeframes
//! let tf = Timeframe::parse("5m").unwrap();
//! let candles = process_binance_month("BTCUSDT-trades-2021-01.zip", tf)?;
//! println!("Aggregated {} candles", candles.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod batch;
pub mod date_utils;
pub mod discovery;
pub mod incomplete_candle;
pub mod timeframe;
pub mod trades;

// Parquet loader for tick-level data (requires data-downloaders feature)
#[cfg(feature = "data-downloaders")]
pub mod parquet_loader;

pub use batch::{process_binance_directory, process_binance_months};
pub use date_utils::{DateRange, ParseError as DateParseError};
pub use discovery::BinanceDataFinder;
pub use incomplete_candle::IncompleteCandle;
#[allow(deprecated)]
pub use timeframe::{ParseError as TimeframeParseError, Timeframe, TimeframeEnum};
pub use trades::{
    BinanceError, Candle, ParseError as CsvParseError, Trade, aggregate_trades_to_candles,
    parse_trade_csv, process_binance_month, stream_aggregate_csv,
};

// Re-export Parquet loader functions when feature enabled
#[cfg(feature = "data-downloaders")]
pub use parquet_loader::{load_parquet_file, load_parquet_month};

// GPU-accelerated aggregation (optional, feature-gated)
#[cfg(feature = "gpu")]
pub use crate::gpu::{AggregationEngine, EngineSelector, GpuAggregator};

/// Process Binance month with GPU-accelerated aggregation
///
/// Uses GPU for OHLCV aggregation on large datasets (>10K trades).
/// Automatically falls back to CPU if GPU unavailable or dataset too small.
#[cfg(feature = "gpu")]
pub fn process_binance_month_gpu<P: AsRef<std::path::Path>>(
    zip_path: P,
    timeframe: Timeframe,
) -> Result<Vec<Candle>, BinanceError> {
    use crate::gpu::EngineSelector;

    // Read trades from ZIP
    let mut trades = Vec::new();
    let file = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if file.name().ends_with(".csv") {
            use std::io::{BufRead, BufReader};
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if let Ok(trade) = parse_trade_csv(&line) {
                    trades.push(trade);
                }
            }
        }
    }

    // Use engine selector to choose GPU or CPU based on data size
    let selector = EngineSelector::new();
    selector.aggregate_trades(&trades, timeframe)
}
