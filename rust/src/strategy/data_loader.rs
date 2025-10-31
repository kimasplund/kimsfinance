//! Historical options data loader from parquet files
//!
//! Loads options chain data from our historical database format:
//! ```text
//! data/yfinance/options_historical/
//!   ├── AAPL/
//!   │   ├── 2016-01-04.parquet
//!   │   ├── 2016-01-05.parquet
//!   │   └── ...
//!   ├── SPY/
//!   └── ...
//! ```

use crate::strategy::types::*;
use chrono::NaiveDate;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[cfg(feature = "data-downloaders")]
use parquet::file::reader::{FileReader, SerializedFileReader};
#[cfg(feature = "data-downloaders")]
use parquet::record::RowAccessor;

#[derive(Debug, Error)]
pub enum DataLoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(feature = "data-downloaders")]
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Data not found: {0}")]
    NotFound(String),
}

/// Historical options data loader
pub struct OptionsDataLoader {
    /// Base directory for options data
    base_dir: PathBuf,

    /// Cached data by (symbol, date)
    cache: HashMap<(String, NaiveDate), Vec<OptionContract>>,
}

impl OptionsDataLoader {
    /// Create new data loader
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self, DataLoaderError> {
        let base_dir = base_dir.as_ref().to_path_buf();

        if !base_dir.exists() {
            return Err(DataLoaderError::NotFound(format!(
                "Base directory does not exist: {}",
                base_dir.display()
            )));
        }

        Ok(Self {
            base_dir,
            cache: HashMap::new(),
        })
    }

    /// Load options chain for a specific symbol and date
    pub fn load_chain(
        &mut self,
        symbol: &str,
        date: NaiveDate,
    ) -> Result<Vec<OptionContract>, DataLoaderError> {
        // Check cache first
        let cache_key = (symbol.to_string(), date);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Load from file
        let file_path = self
            .base_dir
            .join(symbol)
            .join(format!("{}.parquet", date.format("%Y-%m-%d")));

        if !file_path.exists() {
            return Err(DataLoaderError::NotFound(format!(
                "No data found for {} on {}",
                symbol, date
            )));
        }

        let contracts = self.load_parquet_file(&file_path)?;

        // Cache result
        self.cache.insert(cache_key, contracts.clone());

        Ok(contracts)
    }

    /// Load options data for a date range
    pub fn load_range(
        &mut self,
        symbol: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<HashMap<NaiveDate, Vec<OptionContract>>, DataLoaderError> {
        let symbol_dir = self.base_dir.join(symbol);

        if !symbol_dir.exists() {
            return Err(DataLoaderError::NotFound(format!(
                "No data directory for symbol: {}",
                symbol
            )));
        }

        let mut result = HashMap::new();

        // Iterate through all parquet files in date range
        for entry in fs::read_dir(&symbol_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Check if file has .parquet extension
            let is_parquet = path
                .extension()
                .map(|ext| ext == "parquet")
                .unwrap_or(false);

            if !is_parquet {
                continue;
            }

            // Parse date from filename (YYYY-MM-DD.parquet)
            if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
                if let Ok(date) = NaiveDate::parse_from_str(filename, "%Y-%m-%d") {
                    if date >= start_date && date <= end_date {
                        let contracts = self.load_chain(symbol, date)?;
                        result.insert(date, contracts);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Get all available dates for a symbol
    pub fn get_available_dates(&self, symbol: &str) -> Result<Vec<NaiveDate>, DataLoaderError> {
        let symbol_dir = self.base_dir.join(symbol);

        if !symbol_dir.exists() {
            return Err(DataLoaderError::NotFound(format!(
                "No data directory for symbol: {}",
                symbol
            )));
        }

        let mut dates = Vec::new();

        for entry in fs::read_dir(&symbol_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Check if file has .parquet extension
            let is_parquet = path
                .extension()
                .map(|ext| ext == "parquet")
                .unwrap_or(false);

            if !is_parquet {
                continue;
            }

            if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
                if let Ok(date) = NaiveDate::parse_from_str(filename, "%Y-%m-%d") {
                    dates.push(date);
                }
            }
        }

        dates.sort();
        Ok(dates)
    }

    /// Get statistics about available data
    pub fn get_stats(&self) -> Result<HashMap<String, usize>, DataLoaderError> {
        let mut stats = HashMap::new();

        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(symbol) = path.file_name().and_then(|s| s.to_str()) {
                    let count = fs::read_dir(&path)?
                        .filter(|e| {
                            e.as_ref()
                                .ok()
                                .and_then(|entry| {
                                    entry.path().extension().map(|ext| ext == "parquet")
                                })
                                .unwrap_or(false)
                        })
                        .count();
                    stats.insert(symbol.to_string(), count);
                }
            }
        }

        Ok(stats)
    }

    /// Load parquet file
    #[cfg(feature = "data-downloaders")]
    fn load_parquet_file(&self, path: &Path) -> Result<Vec<OptionContract>, DataLoaderError> {
        let file = fs::File::open(path)?;
        let reader = SerializedFileReader::new(file)?;
        let iter = reader.get_row_iter(None)?;

        let mut contracts = Vec::new();

        for record_result in iter {
            let record = record_result?;

            // Parse record into OptionContract
            let contract = self.parse_record(&record)?;
            contracts.push(contract);
        }

        Ok(contracts)
    }

    /// Fallback for when parquet feature is not enabled
    #[cfg(not(feature = "data-downloaders"))]
    fn load_parquet_file(&self, _path: &Path) -> Result<Vec<OptionContract>, DataLoaderError> {
        Err(DataLoaderError::Parse(
            "Parquet loading requires 'data-downloaders' feature".to_string(),
        ))
    }

    /// Parse parquet record into OptionContract
    #[cfg(feature = "data-downloaders")]
    fn parse_record(
        &self,
        record: &parquet::record::Row,
    ) -> Result<OptionContract, DataLoaderError> {
        // Helper to find column index by name
        let find_column_index = |name: &str| -> Option<usize> {
            record
                .get_column_iter()
                .enumerate()
                .find(|(_, (col_name, _))| col_name.as_str() == name)
                .map(|(idx, _)| idx)
        };

        // Helper to get string field
        let get_string = |name: &str| -> Result<String, DataLoaderError> {
            let idx = find_column_index(name)
                .ok_or_else(|| DataLoaderError::Parse(format!("Missing field: {}", name)))?;
            record
                .get_string(idx)
                .map(|s| s.to_string())
                .map_err(|e| DataLoaderError::Parse(format!("Error reading field {}: {}", name, e)))
        };

        // Helper to get f64 field
        let get_f64 = |name: &str| -> Result<f64, DataLoaderError> {
            let idx = find_column_index(name)
                .ok_or_else(|| DataLoaderError::Parse(format!("Missing field: {}", name)))?;
            record
                .get_double(idx)
                .map_err(|e| DataLoaderError::Parse(format!("Error reading field {}: {}", name, e)))
        };

        // Helper to get optional f64 field
        let get_opt_f64 = |name: &str| -> Option<f64> {
            find_column_index(name).and_then(|idx| record.get_double(idx).ok())
        };

        // Parse option type
        let option_type_str = get_string("optionType")?;
        let option_type = match option_type_str.to_lowercase().as_str() {
            "call" => OptionType::Call,
            "put" => OptionType::Put,
            _ => {
                return Err(DataLoaderError::Parse(format!(
                    "Invalid option type: {}",
                    option_type_str
                )));
            }
        };

        // Parse dates
        let snapshot_date_str = get_string("snapshotDate")?;
        let snapshot_date = NaiveDate::parse_from_str(&snapshot_date_str, "%Y-%m-%d")
            .map_err(|e| DataLoaderError::Parse(format!("Invalid snapshot date: {}", e)))?;

        let expiration_str = get_string("expiration")?;
        let expiration = NaiveDate::parse_from_str(&expiration_str, "%Y-%m-%d")
            .map_err(|e| DataLoaderError::Parse(format!("Invalid expiration date: {}", e)))?;

        // Calculate DTE
        let dte = (expiration - snapshot_date).num_days() as i32;

        // Get contract symbol or construct it if missing
        let contract_symbol = get_string("contractSymbol").unwrap_or_else(|_| {
            // Construct contract symbol from components if not available
            format!(
                "{}_{}",
                get_string("symbol").unwrap_or_default(),
                expiration.format("%y%m%d")
            )
        });

        Ok(OptionContract {
            symbol: get_string("symbol")?,
            contract_symbol,
            strike: get_f64("strike")?,
            expiration,
            option_type,
            snapshot_date,
            bid: get_f64("bid").unwrap_or(0.0),
            ask: get_f64("ask").unwrap_or(0.0),
            last_price: get_f64("lastPrice").unwrap_or(0.0),
            volume: get_f64("volume").unwrap_or(0.0),
            open_interest: get_f64("openInterest").unwrap_or(0.0),
            delta: get_opt_f64("delta"),
            gamma: get_opt_f64("gamma"),
            theta: get_opt_f64("theta"),
            vega: get_opt_f64("vega"),
            rho: get_opt_f64("rho"),
            implied_volatility: get_opt_f64("impliedVolatility"),
            dte,
        })
    }

    /// Filter contracts by criteria
    pub fn filter_contracts(
        &self,
        contracts: &[OptionContract],
        filter: &ContractFilter,
    ) -> Vec<OptionContract> {
        contracts
            .iter()
            .filter(|c| {
                // Option type
                if let Some(opt_type) = filter.option_type {
                    if c.option_type != opt_type {
                        return false;
                    }
                }

                // DTE range
                if let Some((min, max)) = filter.dte_range {
                    if c.dte < min || c.dte > max {
                        return false;
                    }
                }

                // Delta range
                if let Some((min, max)) = filter.delta_range {
                    if let Some(delta) = c.delta {
                        let abs_delta = delta.abs();
                        if abs_delta < min || abs_delta > max {
                            return false;
                        }
                    } else {
                        return false; // No delta data
                    }
                }

                // Strike range
                if let Some((min, max)) = filter.strike_range {
                    if c.strike < min || c.strike > max {
                        return false;
                    }
                }

                // Minimum volume
                if let Some(min_vol) = filter.min_volume {
                    if c.volume < min_vol {
                        return false;
                    }
                }

                // Minimum open interest
                if let Some(min_oi) = filter.min_open_interest {
                    if c.open_interest < min_oi {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect()
    }
}

/// Filter for selecting options contracts
#[derive(Debug, Clone, Default)]
pub struct ContractFilter {
    /// Option type (None = both)
    pub option_type: Option<OptionType>,

    /// DTE range (min, max)
    pub dte_range: Option<(i32, i32)>,

    /// Delta range (min, max) - absolute value
    pub delta_range: Option<(f64, f64)>,

    /// Strike range (min, max)
    pub strike_range: Option<(f64, f64)>,

    /// Minimum volume
    pub min_volume: Option<f64>,

    /// Minimum open interest
    pub min_open_interest: Option<f64>,
}

impl ContractFilter {
    /// Create filter for specific delta range and DTE
    pub fn delta_dte(delta_min: f64, delta_max: f64, dte_min: i32, dte_max: i32) -> Self {
        Self {
            delta_range: Some((delta_min, delta_max)),
            dte_range: Some((dte_min, dte_max)),
            ..Default::default()
        }
    }

    /// Create filter for specific strike range
    pub fn strike_range(min: f64, max: f64) -> Self {
        Self {
            strike_range: Some((min, max)),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires actual data files
    fn test_load_chain() {
        let mut loader = OptionsDataLoader::new("data/yfinance/options_historical").unwrap();
        let date = NaiveDate::from_ymd_opt(2020, 1, 2).unwrap();
        let contracts = loader.load_chain("AAPL", date).unwrap();

        assert!(!contracts.is_empty());
        assert!(contracts.iter().all(|c| c.symbol == "AAPL"));
        assert!(contracts.iter().all(|c| c.snapshot_date == date));
    }

    #[test]
    #[ignore] // Requires actual data files
    fn test_filter_contracts() {
        let mut loader = OptionsDataLoader::new("data/yfinance/options_historical").unwrap();
        let date = NaiveDate::from_ymd_opt(2020, 1, 2).unwrap();
        let contracts = loader.load_chain("AAPL", date).unwrap();

        let filter = ContractFilter::delta_dte(0.15, 0.35, 30, 45);
        let filtered = loader.filter_contracts(&contracts, &filter);

        assert!(!filtered.is_empty());
        assert!(filtered.iter().all(|c| c.dte >= 30 && c.dte <= 45));
    }
}
