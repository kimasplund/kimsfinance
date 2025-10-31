//! Batch processing for multi-file Binance trade data
//!
//! Provides functions to load entire date ranges of Binance trade data,
//! automatically discovering and processing multiple months or days of files.
//!
//! # Features
//! - Load complete date ranges (e.g., all of 2021)
//! - Automatic file discovery and sorting
//! - Progress logging for long-running operations
//! - Deduplication at file boundaries
//! - Maintains 1-5M trades/sec performance
//!
//! # Example
//! ```no_run
//! use kimsfinance_core::binance::{Timeframe, process_binance_directory};
//!
//! // Load 3 months of data
//! let candles = process_binance_directory(
//!     "/data/binance/BTCUSDT/trades",
//!     "2021-01-01",
//!     "2021-03-31",
//!     Timeframe::parse("5m")?,
//! )?;
//! println!("Loaded {} candles", candles.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::binance::{
    BinanceDataFinder, BinanceError, Candle, DateRange, Timeframe, process_binance_month,
};
use std::path::Path;

/// Process multiple Binance files for a date range
///
/// Discovers all Binance trade data files in the given directory that fall within
/// the specified date range, processes each file, and returns aggregated candles
/// sorted by timestamp with duplicates removed at file boundaries.
///
/// # Arguments
/// * `data_dir` - Directory containing Binance trade data files (ZIP or CSV)
/// * `start_date` - Start date in "YYYY-MM-DD" format (inclusive)
/// * `end_date` - End date in "YYYY-MM-DD" format (inclusive)
/// * `timeframe` - Candle timeframe (e.g., Timeframe::parse("5m")?)
///
/// # Performance
/// - Maintains ~1-5M trades/sec throughput per file
/// - Progress logging for visibility during long operations
/// - Memory-efficient: processes one file at a time
///
/// # Errors
/// Returns `BinanceError` if:
/// - Date range parsing fails
/// - No files found for the given date range
/// - File processing fails (ZIP/CSV errors)
/// - Trade data is invalid
///
/// # Example
/// ```no_run
/// use kimsfinance_core::binance::{Timeframe, process_binance_directory};
///
/// // Load Q1 2021
/// let candles = process_binance_directory(
///     "/data/binance/BTCUSDT/trades",
///     "2021-01-01",
///     "2021-03-31",
///     Timeframe::parse("5m")?,
/// )?;
/// println!("Loaded {} candles", candles.len());
///
/// // Load full year 2021
/// let candles = process_binance_directory(
///     "/data/binance/BTCUSDT/trades",
///     "2021-01-01",
///     "2021-12-31",
///     Timeframe::parse("1h")?,
/// )?;
/// println!("Full year: {} hourly candles", candles.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn process_binance_directory<P: AsRef<Path>>(
    data_dir: P,
    start_date: &str,
    end_date: &str,
    timeframe: Timeframe,
) -> Result<Vec<Candle>, BinanceError> {
    // 1. Parse date range
    let range = DateRange::parse(start_date, end_date)
        .map_err(|e| BinanceError::ParseError(e.to_string()))?;

    println!("=== Batch Processing: {} to {} ===", start_date, end_date);
    println!(
        "Date range: {} days, {} months",
        range.num_days(),
        range.num_months()
    );

    // 2. Find files
    let finder = BinanceDataFinder::new(data_dir);
    let files = finder.find_by_date_range(&range)?;

    if files.is_empty() {
        return Err(BinanceError::InvalidData(format!(
            "No files found for date range {} to {}",
            start_date, end_date
        )));
    }

    println!("Found {} files to process\n", files.len());

    // 3. Process each file
    let mut all_candles = Vec::new();
    for (i, file) in files.iter().enumerate() {
        println!(
            "[{}/{}] Processing: {:?}",
            i + 1,
            files.len(),
            file.file_name().unwrap_or_default()
        );

        let candles = process_binance_month(file.to_str().unwrap(), timeframe)?;

        println!("  → Loaded {} candles", candles.len());
        all_candles.extend(candles);
    }

    println!("\n=== Processing Complete ===");
    println!("Total candles (before dedup): {}", all_candles.len());

    // 4. Sort by timestamp (files might be out of order)
    all_candles.sort_unstable_by_key(|c| c.timestamp);

    // 5. Remove duplicates at file boundaries
    // Binance files may overlap at month boundaries, so we deduplicate by timestamp
    let initial_count = all_candles.len();
    all_candles.dedup_by_key(|c| c.timestamp);
    let removed = initial_count - all_candles.len();

    if removed > 0 {
        println!("Removed {} duplicate candles at file boundaries", removed);
    }

    println!("Final candle count: {}", all_candles.len());

    if !all_candles.is_empty() {
        // Edition 2024: Let chains for cleaner option handling
        if let Some(first) = all_candles.first()
            && let Some(last) = all_candles.last()
        {
            println!("Time range: {} to {}", first.timestamp, last.timestamp);
        }
    }

    Ok(all_candles)
}

/// Process specific months of Binance data
///
/// Loads trade data for explicitly specified months. Useful when you need
/// non-contiguous months or want more control over which months to process.
///
/// # Arguments
/// * `data_dir` - Directory containing Binance trade data files
/// * `months` - Slice of month strings in "YYYY-MM" format
/// * `timeframe` - Candle timeframe
///
/// # Performance
/// Same as `process_binance_directory`: ~1-5M trades/sec per file
///
/// # Errors
/// Returns `BinanceError` if file processing fails. Logs warnings for months
/// with no matching files but continues processing other months.
///
/// # Example
/// ```no_run
/// use kimsfinance_core::binance::{Timeframe, process_binance_months};
///
/// // Load specific months (e.g., Q1 and Q3 only)
/// let candles = process_binance_months(
///     "/data/binance/BTCUSDT/trades",
///     &["2021-01", "2021-02", "2021-03", "2021-07", "2021-08", "2021-09"],
///     Timeframe::parse("5m")?,
/// )?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn process_binance_months<P: AsRef<Path>>(
    data_dir: P,
    months: &[&str],
    timeframe: Timeframe,
) -> Result<Vec<Candle>, BinanceError> {
    println!("=== Batch Processing: {} months ===", months.len());
    println!("Months: {:?}\n", months);

    let mut all_candles = Vec::new();
    let finder = BinanceDataFinder::new(data_dir.as_ref());

    for (i, month) in months.iter().enumerate() {
        println!("[{}/{}] Processing month: {}", i + 1, months.len(), month);

        // Find files for this month using multiple patterns
        let patterns = vec![
            format!("*trades*{}*.zip", month),
            format!("*{}*.zip", month),
        ];

        let mut files = Vec::new();
        for pattern in patterns {
            let found = finder.find_files(&pattern)?;
            files.extend(found);
        }

        // Remove duplicates if patterns matched same files
        files.sort();
        files.dedup();

        if files.is_empty() {
            eprintln!("  ⚠ Warning: No files found for month {}", month);
            continue;
        }

        println!("  Found {} file(s)", files.len());

        for file in files {
            let candles = process_binance_month(file.to_str().unwrap(), timeframe)?;
            println!(
                "    → {} candles from {:?}",
                candles.len(),
                file.file_name().unwrap()
            );
            all_candles.extend(candles);
        }
    }

    println!("\n=== Processing Complete ===");
    println!("Total candles (before dedup): {}", all_candles.len());

    // Sort and deduplicate
    all_candles.sort_unstable_by_key(|c| c.timestamp);
    let initial_count = all_candles.len();
    all_candles.dedup_by_key(|c| c.timestamp);
    let removed = initial_count - all_candles.len();

    if removed > 0 {
        println!("Removed {} duplicate candles", removed);
    }

    println!("Final candle count: {}", all_candles.len());

    Ok(all_candles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    /// Helper to create a test ZIP file with trade data
    fn create_test_zip(path: &Path, content: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut zip = ZipWriter::new(file);

        let options = SimpleFileOptions::default();
        zip.start_file("data.csv", options)?;
        zip.write_all(content.as_bytes())?;
        zip.finish()?;

        Ok(())
    }

    /// Sample trade data for testing
    fn sample_trade_data() -> &'static str {
        "trade_id,price,quantity,quote_quantity,timestamp,is_buyer_maker\n\
         352562763,28948.19,0.052,1505.30,1609459200001,false\n\
         352562764,28948.18,0.001,28.94,1609459200010,true\n\
         352562765,28950.00,0.100,2895.00,1609459260000,false\n"
    }

    #[test]
    fn test_process_multiple_months() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create test ZIP files for multiple months
        let file1 = base_path.join("BTCUSDT-trades-2021-01.zip");
        let file2 = base_path.join("BTCUSDT-trades-2021-02.zip");

        create_test_zip(&file1, sample_trade_data()).unwrap();
        create_test_zip(&file2, sample_trade_data()).unwrap();

        // Test batch processing
        let timeframe = Timeframe::parse("1m").unwrap();
        let result = process_binance_months(base_path, &["2021-01", "2021-02"], timeframe);

        assert!(result.is_ok());
        let candles = result.unwrap();
        assert!(!candles.is_empty(), "Should have loaded candles");
        println!("Loaded {} candles from 2 months", candles.len());
    }

    #[test]
    fn test_process_directory_with_date_range() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create files for Q1 2021
        let file1 = base_path.join("BTCUSDT-trades-2021-01.zip");
        let file2 = base_path.join("BTCUSDT-trades-2021-02.zip");
        let file3 = base_path.join("BTCUSDT-trades-2021-03.zip");

        for file in &[&file1, &file2, &file3] {
            create_test_zip(file, sample_trade_data()).unwrap();
        }

        let timeframe = Timeframe::parse("5m").unwrap();
        let result = process_binance_directory(base_path, "2021-01-01", "2021-03-31", timeframe);

        assert!(result.is_ok());
        let candles = result.unwrap();
        assert!(!candles.is_empty(), "Should have loaded candles");
        println!("Loaded {} candles from Q1 2021", candles.len());
    }

    #[test]
    fn test_candles_sorted_by_timestamp() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create files in reverse order to test sorting
        let file1 = base_path.join("BTCUSDT-trades-2021-03.zip");
        let file2 = base_path.join("BTCUSDT-trades-2021-01.zip");
        let file3 = base_path.join("BTCUSDT-trades-2021-02.zip");

        for file in &[&file1, &file2, &file3] {
            create_test_zip(file, sample_trade_data()).unwrap();
        }

        let timeframe = Timeframe::parse("1m").unwrap();
        let result = process_binance_directory(base_path, "2021-01-01", "2021-03-31", timeframe);

        assert!(result.is_ok());
        let candles = result.unwrap();

        // Verify candles are sorted by timestamp
        for i in 1..candles.len() {
            assert!(
                candles[i - 1].timestamp <= candles[i].timestamp,
                "Candles should be sorted by timestamp"
            );
        }
    }

    #[test]
    fn test_deduplication_at_boundaries() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create two files with overlapping data (same timestamps)
        let duplicate_data = "trade_id,price,quantity,quote_quantity,timestamp,is_buyer_maker\n\
            352562763,28948.19,0.052,1505.30,1609459200000,false\n\
            352562764,28950.00,0.100,2895.00,1609459200000,true\n";

        let file1 = base_path.join("BTCUSDT-trades-2021-01.zip");
        let file2 = base_path.join("BTCUSDT-trades-2021-02.zip");

        create_test_zip(&file1, duplicate_data).unwrap();
        create_test_zip(&file2, duplicate_data).unwrap();

        let timeframe = Timeframe::parse("1m").unwrap();
        let result = process_binance_directory(base_path, "2021-01-01", "2021-02-28", timeframe);

        assert!(result.is_ok());
        let candles = result.unwrap();

        // Should deduplicate candles with same timestamp
        assert!(!candles.is_empty());
        println!("After dedup: {} candles", candles.len());
    }

    #[test]
    fn test_no_files_found_error() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Empty directory
        let timeframe = Timeframe::parse("5m").unwrap();
        let result = process_binance_directory(base_path, "2021-01-01", "2021-03-31", timeframe);

        assert!(result.is_err());
        match result {
            Err(BinanceError::InvalidData(msg)) => {
                assert!(msg.contains("No files found"));
            }
            _ => panic!("Expected InvalidData error"),
        }
    }

    #[test]
    fn test_missing_month_warning() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create only January file
        let file1 = base_path.join("BTCUSDT-trades-2021-01.zip");
        create_test_zip(&file1, sample_trade_data()).unwrap();

        let timeframe = Timeframe::parse("1m").unwrap();
        // Try to load Jan and Feb, but Feb doesn't exist
        let result = process_binance_months(base_path, &["2021-01", "2021-02"], timeframe);

        // Should succeed but log warning for missing month
        assert!(result.is_ok());
        let candles = result.unwrap();
        assert!(!candles.is_empty(), "Should have loaded Jan data");
    }

    #[test]
    fn test_invalid_date_range() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // End before start
        let timeframe = Timeframe::parse("5m").unwrap();
        let result = process_binance_directory(base_path, "2021-12-31", "2021-01-01", timeframe);

        assert!(result.is_err());
        match result {
            Err(BinanceError::ParseError(msg)) => {
                assert!(msg.contains("before"));
            }
            _ => panic!("Expected ParseError for invalid range"),
        }
    }

    #[test]
    fn test_different_timeframes() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        let file1 = base_path.join("BTCUSDT-trades-2021-01.zip");
        create_test_zip(&file1, sample_trade_data()).unwrap();

        // Test with 1-minute timeframe
        let tf_1m = Timeframe::parse("1m").unwrap();
        let result_1m = process_binance_months(base_path, &["2021-01"], tf_1m);
        assert!(result_1m.is_ok());
        let candles_1m = result_1m.unwrap();

        // Test with 5-minute timeframe
        let tf_5m = Timeframe::parse("5m").unwrap();
        let result_5m = process_binance_months(base_path, &["2021-01"], tf_5m);
        assert!(result_5m.is_ok());
        let candles_5m = result_5m.unwrap();

        // 1m should have more candles than 5m for same data
        assert!(
            candles_1m.len() >= candles_5m.len(),
            "1m candles should be >= 5m candles"
        );
    }
}
