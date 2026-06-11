//! Parquet file loader for tick-level trade data
//!
//! Loads Binance tick data from month-partitioned Parquet files using Apache Arrow
//! for zero-copy reads and maximum performance.
//!
//! # Features
//! - Zero-copy reads via Arrow RecordBatch
//! - Batch processing for efficient memory usage
//! - Target: 10-20M records/sec loading speed
//! - Memory-efficient for 100M+ trade datasets
//!
//! # Schema
//! Expected Parquet schema:
//! - `id`: UInt64 (trade ID)
//! - `price`: Float64
//! - `qty`: Float64
//! - `quote_qty`: Float64
//! - `time`: Int64 (Unix timestamp ms)
//! - `is_buyer_maker`: Boolean
//!
//! # Example
//! ```no_run
//! use kimsfinance_core::binance::load_parquet_month;
//!
//! let trades = load_parquet_month(
//!     "/path/to/trades_parquet/2024-01",
//!     Some(1_000_000) // Limit to 1M trades
//! )?;
//!
//! println!("Loaded {} trades", trades.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::binance::{BinanceError, Trade};
use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, UInt64Array};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Load tick data from a single Parquet file
///
/// Uses zero-copy Arrow RecordBatch reads for maximum performance.
/// Processes batches of ~10K records at a time to balance memory usage
/// and throughput.
///
/// # Performance
/// - Target: 10-20M records/sec
/// - Zero-copy reads via Arrow
/// - Batch processing (default batch size from Parquet metadata)
///
/// # Errors
/// Returns `BinanceError` if:
/// - File cannot be opened (`IoError`)
/// - Parquet parsing fails (`ParseError`)
/// - Required columns are missing (`InvalidData`)
/// - Column types don't match expected schema (`InvalidData`)
///
/// # Example
/// ```no_run
/// use kimsfinance_core::binance::load_parquet_file;
///
/// let trades = load_parquet_file(
///     "/path/to/BTCUSDT-trades-2024-01-01.parquet"
/// )?;
///
/// assert!(trades.len() > 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn load_parquet_file<P: AsRef<Path>>(parquet_path: P) -> Result<Vec<Trade>, BinanceError> {
    let path = parquet_path.as_ref();

    // Open file
    let file = File::open(path).map_err(|e| {
        BinanceError::IoError(std::io::Error::new(
            e.kind(),
            format!("Failed to open Parquet file {:?}: {}", path, e),
        ))
    })?;

    // Build Parquet reader
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        BinanceError::ParseError(format!("Failed to open Parquet file {:?}: {}", path, e))
    })?;

    let reader = builder.build().map_err(|e| {
        BinanceError::ParseError(format!(
            "Failed to build Parquet reader for {:?}: {}",
            path, e
        ))
    })?;

    let mut trades = Vec::new();

    // Process record batches (zero-copy)
    for batch_result in reader {
        let batch = batch_result.map_err(|e| {
            BinanceError::ParseError(format!("Failed to read batch from {:?}: {}", path, e))
        })?;

        // Extract columns using zero-copy Arrow arrays
        let ids = extract_uint64_column(&batch, "id")?;
        let prices = extract_float64_column(&batch, "price")?;
        let quantities = extract_float64_column(&batch, "qty")?;
        let quote_qtys = extract_float64_column(&batch, "quote_qty")?;
        let timestamps = extract_int64_column(&batch, "time")?;
        let is_buyer_makers = extract_boolean_column(&batch, "is_buyer_maker")?;

        // Convert Arrow arrays to Trade structs
        for i in 0..batch.num_rows() {
            trades.push(Trade {
                trade_id: ids.value(i),
                price: prices.value(i),
                quantity: quantities.value(i),
                quote_quantity: quote_qtys.value(i),
                timestamp_ms: timestamps.value(i),
                is_buyer_maker: is_buyer_makers.value(i),
            });
        }
    }

    Ok(trades)
}

/// Load all Parquet files from a month directory
///
/// Discovers all `.parquet` files in the directory, loads them, and sorts
/// trades by timestamp. Supports optional trade limit for testing or
/// memory-constrained environments.
///
/// # Performance
/// - Parallel file loading (sequential for now, Rayon parallel later)
/// - Memory-mapped I/O via Arrow
/// - Target: 5-10M records/sec aggregate
///
/// # File Pattern
/// Expected directory structure:
/// ```text
/// /trades_parquet/2024-01/
///   ├── BTCUSDT-trades-2024-01-01.parquet
///   ├── BTCUSDT-trades-2024-01-02.parquet
///   └── ...
/// ```
///
/// # Arguments
/// - `month_dir`: Path to directory containing Parquet files
/// - `max_trades`: Optional limit on total trades to load
///
/// # Errors
/// Returns `BinanceError` if:
/// - Directory doesn't exist or can't be read
/// - No `.parquet` files found
/// - Any file fails to load
///
/// # Example
/// ```no_run
/// use kimsfinance_core::binance::load_parquet_month;
///
/// // Load entire month
/// let all_trades = load_parquet_month(
///     "/path/to/trades_parquet/2024-01",
///     None
/// )?;
///
/// // Load first 100K trades for testing
/// let sample = load_parquet_month(
///     "/path/to/trades_parquet/2024-01",
///     Some(100_000)
/// )?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn load_parquet_month<P: AsRef<Path>>(
    month_dir: P,
    max_trades: Option<usize>,
) -> Result<Vec<Trade>, BinanceError> {
    let month_path = month_dir.as_ref();

    // Find all .parquet files in directory
    let mut parquet_files: Vec<PathBuf> = std::fs::read_dir(month_path)
        .map_err(|e| {
            BinanceError::IoError(std::io::Error::new(
                e.kind(),
                format!("Failed to read directory {:?}: {}", month_path, e),
            ))
        })?
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "parquet").unwrap_or(false))
        .collect();

    if parquet_files.is_empty() {
        return Err(BinanceError::InvalidData(format!(
            "No Parquet files found in directory {:?}",
            month_path
        )));
    }

    // Sort files by name for chronological order
    parquet_files.sort();

    let mut all_trades = Vec::new();

    // Load each file sequentially (TODO: parallelize with Rayon)
    for file_path in parquet_files {
        let trades = load_parquet_file(&file_path)?;

        if let Some(limit) = max_trades {
            let remaining = limit.saturating_sub(all_trades.len());
            if remaining == 0 {
                break;
            }
            all_trades.extend(trades.into_iter().take(remaining));
        } else {
            all_trades.extend(trades);
        }
    }

    // Sort by timestamp (files may not be perfectly chronological)
    all_trades.sort_unstable_by_key(|t| t.timestamp_ms);

    Ok(all_trades)
}

// ============================================================================
// Helper functions for Arrow column extraction
// ============================================================================

/// Extract UInt64 column from RecordBatch
fn extract_uint64_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a UInt64Array, BinanceError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| {
            BinanceError::InvalidData(format!(
                "Missing required column '{}'. Available columns: {:?}",
                name,
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>()
            ))
        })?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| {
            BinanceError::InvalidData(format!(
                "Column '{}' has incorrect type (expected UInt64)",
                name
            ))
        })
}

/// Extract Float64 column from RecordBatch
fn extract_float64_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a Float64Array, BinanceError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| {
            BinanceError::InvalidData(format!(
                "Missing required column '{}'. Available columns: {:?}",
                name,
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>()
            ))
        })?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            BinanceError::InvalidData(format!(
                "Column '{}' has incorrect type (expected Float64)",
                name
            ))
        })
}

/// Extract Int64 column from RecordBatch
fn extract_int64_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a Int64Array, BinanceError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| {
            BinanceError::InvalidData(format!(
                "Missing required column '{}'. Available columns: {:?}",
                name,
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>()
            ))
        })?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            BinanceError::InvalidData(format!(
                "Column '{}' has incorrect type (expected Int64)",
                name
            ))
        })
}

/// Extract Boolean column from RecordBatch
fn extract_boolean_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a BooleanArray, BinanceError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| {
            BinanceError::InvalidData(format!(
                "Missing required column '{}'. Available columns: {:?}",
                name,
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>()
            ))
        })?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| {
            BinanceError::InvalidData(format!(
                "Column '{}' has incorrect type (expected Boolean)",
                name
            ))
        })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires actual dataset
    fn test_load_parquet_file_btcusdt() {
        let path = "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades_parquet/2024-01/BTCUSDT-trades-2024-01-01.parquet";

        let trades = load_parquet_file(path).expect("Failed to load Parquet file");

        // Validate non-empty
        assert!(!trades.is_empty(), "Expected trades, got empty result");

        // Validate first trade has realistic data
        let first = &trades[0];
        assert!(first.price > 0.0, "Price should be positive");
        assert!(first.quantity > 0.0, "Quantity should be positive");
        assert!(
            first.quote_quantity > 0.0,
            "Quote quantity should be positive"
        );
        assert!(
            first.timestamp_ms > 1_600_000_000_000,
            "Timestamp should be realistic (2020+)"
        );

        println!("Loaded {} trades from single file", trades.len());
        println!("First trade: {:?}", first);
    }

    #[test]
    #[ignore] // Requires actual dataset
    fn test_load_parquet_month_btcusdt() {
        let dir = "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades_parquet/2024-01";

        let trades = load_parquet_month(dir, Some(100_000)).expect("Failed to load month");

        // Should respect max_trades limit
        assert_eq!(trades.len(), 100_000, "Expected exactly 100K trades");

        // Verify sorted by timestamp
        for i in 1..trades.len() {
            assert!(
                trades[i].timestamp_ms >= trades[i - 1].timestamp_ms,
                "Trades should be sorted by timestamp"
            );
        }

        println!("Successfully loaded and sorted 100K trades");
        println!(
            "Time range: {} to {}",
            trades.first().unwrap().timestamp_ms,
            trades.last().unwrap().timestamp_ms
        );
    }

    #[test]
    #[ignore] // Requires actual dataset
    fn test_load_parquet_month_no_limit() {
        let dir = "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades_parquet/2024-01";

        let trades = load_parquet_month(dir, None).expect("Failed to load month");

        // Should load all trades from January 2024
        assert!(
            trades.len() > 1_000_000,
            "Expected >1M trades for full month"
        );

        println!("Loaded {} trades from entire month", trades.len());
    }

    #[test]
    fn test_load_parquet_nonexistent_file() {
        let result = load_parquet_file("/nonexistent/path/to/file.parquet");

        assert!(result.is_err(), "Should fail on nonexistent file");

        if let Err(BinanceError::IoError(_)) = result {
            // Correct error type
        } else {
            panic!("Expected IoError, got {:?}", result);
        }
    }

    #[test]
    fn test_load_parquet_nonexistent_directory() {
        let result = load_parquet_month("/nonexistent/directory", None);

        assert!(result.is_err(), "Should fail on nonexistent directory");
    }
}
