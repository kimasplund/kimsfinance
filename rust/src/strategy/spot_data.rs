//! Spot price data loader from OHLCV parquet files
//!
//! Loads historical OHLCV data and provides:
//! - Spot price retrieval by symbol and date
//! - 20-day ATR (Average True Range) calculation
//! - 20-day Bollinger Band width calculation
//!
//! Data format:
//! ```text
//! data/yfinance/ohlcv/
//!   ├── AAPL.parquet
//!   ├── SPY.parquet
//!   ├── TSLA.parquet
//!   └── QQQ.parquet
//! ```

use chrono::NaiveDate;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[cfg(feature = "data-downloaders")]
use parquet::file::reader::{FileReader, SerializedFileReader};
#[cfg(feature = "data-downloaders")]
use parquet::record::RowAccessor;

#[derive(Debug, Error)]
pub enum SpotDataError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(feature = "data-downloaders")]
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Data not found: {0}")]
    NotFound(String),

    #[error("Insufficient data for calculation: {0}")]
    InsufficientData(String),
}

/// OHLCV data point
#[derive(Debug, Clone)]
pub struct OhlcvBar {
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl OhlcvBar {
    /// Calculate true range for this bar given previous close
    pub fn true_range(&self, prev_close: Option<f64>) -> f64 {
        let range = self.high - self.low;

        if let Some(prev) = prev_close {
            let high_prev = (self.high - prev).abs();
            let low_prev = (self.low - prev).abs();
            range.max(high_prev).max(low_prev)
        } else {
            range
        }
    }
}

/// Spot data loader for OHLCV historical data
pub struct SpotDataLoader {
    /// Base directory for OHLCV data
    base_dir: PathBuf,

    /// Cached data by symbol
    cache: HashMap<String, Vec<OhlcvBar>>,

    /// ATR period (default: 20 days)
    atr_period: usize,

    /// Bollinger Band period (default: 20 days)
    bb_period: usize,
}

impl SpotDataLoader {
    /// Create new spot data loader
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self, SpotDataError> {
        let base_dir = base_dir.as_ref().to_path_buf();

        if !base_dir.exists() {
            return Err(SpotDataError::NotFound(format!(
                "Base directory does not exist: {}",
                base_dir.display()
            )));
        }

        Ok(Self {
            base_dir,
            cache: HashMap::new(),
            atr_period: 20,
            bb_period: 20,
        })
    }

    /// Load OHLCV data for a symbol
    pub fn load_symbol(&mut self, symbol: &str) -> Result<&Vec<OhlcvBar>, SpotDataError> {
        // Check cache first
        if self.cache.contains_key(symbol) {
            return Ok(self.cache.get(symbol).unwrap());
        }

        // Load from file
        let file_path = self.base_dir.join(format!("{}.parquet", symbol));

        if !file_path.exists() {
            return Err(SpotDataError::NotFound(format!(
                "No data found for symbol: {}",
                symbol
            )));
        }

        let bars = self.load_parquet_file(&file_path)?;
        self.cache.insert(symbol.to_string(), bars);

        Ok(self.cache.get(symbol).unwrap())
    }

    /// Get spot price for a symbol on a specific date
    pub fn get_spot_price(&mut self, symbol: &str, date: NaiveDate) -> Result<f64, SpotDataError> {
        let bars = self.load_symbol(symbol)?;

        // Find exact date or closest prior date
        let bar = bars
            .iter()
            .filter(|b| b.date <= date)
            .max_by_key(|b| b.date)
            .ok_or_else(|| {
                SpotDataError::NotFound(format!("No data for {} on or before {}", symbol, date))
            })?;

        Ok(bar.close)
    }

    /// Get OHLCV bar for a specific date
    pub fn get_bar(&mut self, symbol: &str, date: NaiveDate) -> Result<&OhlcvBar, SpotDataError> {
        let bars = self.load_symbol(symbol)?;

        bars.iter()
            .find(|b| b.date == date)
            .ok_or_else(|| SpotDataError::NotFound(format!("No data for {} on {}", symbol, date)))
    }

    /// Calculate 20-day ATR (Average True Range) for a symbol on a date
    pub fn calculate_atr(&mut self, symbol: &str, date: NaiveDate) -> Result<f64, SpotDataError> {
        // Extract period first to avoid borrow issues
        let atr_period = self.atr_period;
        let bars = self.load_symbol(symbol)?;

        // Find the index of the date
        let date_idx = bars.iter().position(|b| b.date == date).ok_or_else(|| {
            SpotDataError::NotFound(format!("No data for {} on {}", symbol, date))
        })?;

        // Need at least atr_period bars
        if date_idx < atr_period {
            return Err(SpotDataError::InsufficientData(format!(
                "Need {} days of data before {}, only have {}",
                atr_period, date, date_idx
            )));
        }

        // Calculate true ranges for the period
        let start_idx = date_idx - atr_period + 1;
        let mut true_ranges = Vec::new();

        for i in start_idx..=date_idx {
            let prev_close = if i > 0 { Some(bars[i - 1].close) } else { None };
            true_ranges.push(bars[i].true_range(prev_close));
        }

        // Calculate average
        let atr = true_ranges.iter().sum::<f64>() / true_ranges.len() as f64;
        Ok(atr)
    }

    /// Calculate 20-day Bollinger Band width for a symbol on a date
    /// Returns (upper_band, lower_band, width)
    pub fn calculate_bollinger_bands(
        &mut self,
        symbol: &str,
        date: NaiveDate,
        num_std: f64,
    ) -> Result<(f64, f64, f64), SpotDataError> {
        // Extract period first to avoid borrow issues
        let bb_period = self.bb_period;
        let bars = self.load_symbol(symbol)?;

        // Find the index of the date
        let date_idx = bars.iter().position(|b| b.date == date).ok_or_else(|| {
            SpotDataError::NotFound(format!("No data for {} on {}", symbol, date))
        })?;

        // Need at least bb_period bars
        if date_idx < bb_period - 1 {
            return Err(SpotDataError::InsufficientData(format!(
                "Need {} days of data before {}, only have {}",
                bb_period,
                date,
                date_idx + 1
            )));
        }

        // Get closing prices for the period
        let start_idx = date_idx - bb_period + 1;
        let closes: Vec<f64> = bars[start_idx..=date_idx].iter().map(|b| b.close).collect();

        // Calculate mean (middle band)
        let mean = closes.iter().sum::<f64>() / closes.len() as f64;

        // Calculate standard deviation
        let variance =
            closes.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / closes.len() as f64;
        let std_dev = variance.sqrt();

        // Calculate bands
        let upper_band = mean + (num_std * std_dev);
        let lower_band = mean - (num_std * std_dev);
        let width = upper_band - lower_band;

        Ok((upper_band, lower_band, width))
    }

    /// Get all available dates for a symbol
    pub fn get_available_dates(&mut self, symbol: &str) -> Result<Vec<NaiveDate>, SpotDataError> {
        let bars = self.load_symbol(symbol)?;
        Ok(bars.iter().map(|b| b.date).collect())
    }

    /// Get date range for a symbol
    pub fn get_date_range(
        &mut self,
        symbol: &str,
    ) -> Result<(NaiveDate, NaiveDate), SpotDataError> {
        let bars = self.load_symbol(symbol)?;

        if bars.is_empty() {
            return Err(SpotDataError::NotFound(format!(
                "No data for symbol: {}",
                symbol
            )));
        }

        let min_date = bars.iter().map(|b| b.date).min().unwrap();
        let max_date = bars.iter().map(|b| b.date).max().unwrap();

        Ok((min_date, max_date))
    }

    /// Load parquet file
    #[cfg(feature = "data-downloaders")]
    fn load_parquet_file(&self, path: &Path) -> Result<Vec<OhlcvBar>, SpotDataError> {
        use std::fs;

        let file = fs::File::open(path)?;
        let reader = SerializedFileReader::new(file)?;
        let iter = reader.get_row_iter(None)?;

        let mut bars = Vec::new();

        for record_result in iter {
            let record = record_result?;
            let bar = self.parse_record(&record)?;
            bars.push(bar);
        }

        // Sort by date
        bars.sort_by_key(|b| b.date);

        Ok(bars)
    }

    /// Fallback for when parquet feature is not enabled
    #[cfg(not(feature = "data-downloaders"))]
    fn load_parquet_file(&self, _path: &Path) -> Result<Vec<OhlcvBar>, SpotDataError> {
        Err(SpotDataError::Parse(
            "Parquet loading requires 'data-downloaders' feature".to_string(),
        ))
    }

    /// Parse parquet record into OhlcvBar
    #[cfg(feature = "data-downloaders")]
    fn parse_record(&self, record: &parquet::record::Row) -> Result<OhlcvBar, SpotDataError> {
        // Helper to find column index by name
        let find_column_index = |name: &str| -> Option<usize> {
            record
                .get_column_iter()
                .enumerate()
                .find(|(_, (col_name, _))| col_name.as_str() == name)
                .map(|(idx, _)| idx)
        };

        // Helper to get f64 field
        let get_f64 = |name: &str| -> Result<f64, SpotDataError> {
            let idx = find_column_index(name)
                .ok_or_else(|| SpotDataError::Parse(format!("Missing field: {}", name)))?;
            record
                .get_double(idx)
                .map_err(|e| SpotDataError::Parse(format!("Error reading field {}: {}", name, e)))
        };

        // Parse date (timestamp in nanoseconds)
        let date_idx = find_column_index("Date")
            .ok_or_else(|| SpotDataError::Parse("Missing field: Date".to_string()))?;

        let date_nanos = record
            .get_long(date_idx)
            .map_err(|e| SpotDataError::Parse(format!("Error reading Date: {}", e)))?;

        // Convert nanoseconds to NaiveDate
        let date_secs = date_nanos / 1_000_000_000;
        let date = chrono::DateTime::from_timestamp(date_secs, 0)
            .ok_or_else(|| SpotDataError::Parse(format!("Invalid timestamp: {}", date_nanos)))?
            .date_naive();

        // Get volume (could be Long or Double depending on the parquet file)
        let volume_idx = find_column_index("Volume")
            .ok_or_else(|| SpotDataError::Parse("Missing field: Volume".to_string()))?;
        let volume = record.get_double(volume_idx)
            .or_else(|_| record.get_long(volume_idx).map(|v| v as f64))
            .map_err(|e| SpotDataError::Parse(format!("Error reading Volume: {}", e)))?;

        Ok(OhlcvBar {
            date,
            open: get_f64("Open")?,
            high: get_f64("High")?,
            low: get_f64("Low")?,
            close: get_f64("Close")?,
            volume,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires actual data files
    fn test_load_symbol() {
        let mut loader = SpotDataLoader::new("data/yfinance/ohlcv").unwrap();
        let bars = loader.load_symbol("AAPL").unwrap();
        assert!(!bars.is_empty());
    }

    #[test]
    #[ignore] // Requires actual data files
    fn test_get_spot_price() {
        let mut loader = SpotDataLoader::new("data/yfinance/ohlcv").unwrap();
        let date = NaiveDate::from_ymd_opt(2020, 1, 2).unwrap();
        let price = loader.get_spot_price("AAPL", date).unwrap();
        assert!(price > 0.0);
    }

    #[test]
    #[ignore] // Requires actual data files
    fn test_calculate_atr() {
        let mut loader = SpotDataLoader::new("data/yfinance/ohlcv").unwrap();
        let date = NaiveDate::from_ymd_opt(2020, 2, 1).unwrap();
        let atr = loader.calculate_atr("AAPL", date).unwrap();
        assert!(atr > 0.0);
        println!("AAPL ATR on {}: {:.2}", date, atr);
    }

    #[test]
    #[ignore] // Requires actual data files
    fn test_calculate_bollinger_bands() {
        let mut loader = SpotDataLoader::new("data/yfinance/ohlcv").unwrap();
        let date = NaiveDate::from_ymd_opt(2020, 2, 1).unwrap();
        let (upper, lower, width) = loader.calculate_bollinger_bands("AAPL", date, 2.0).unwrap();
        assert!(upper > lower);
        assert!(width > 0.0);
        println!(
            "AAPL Bollinger Bands on {}: Upper={:.2}, Lower={:.2}, Width={:.2}",
            date, upper, lower, width
        );
    }
}
