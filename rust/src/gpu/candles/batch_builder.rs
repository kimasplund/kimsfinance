//! Batch Builder for Candle Aggregation Tasks
//!
//! Simplifies creation of TaskBatch for candle aggregation with multiple symbols
//! and parameters. Provides fluent API for batch construction.
//!
//! # Example
//!
//! ```rust,no_run
//! # use kimsfinance_core::gpu::candles::{CandleBatchBuilder, TradeData, TimeBarParams};
//! let btc_trades = TradeData::from_csv("btc_trades.csv")?;
//! let eth_trades = TradeData::from_csv("eth_trades.csv")?;
//!
//! // Build batch with multiple symbols
//! let batch = CandleBatchBuilder::new()
//!     .add_symbol("BTCUSDT", btc_trades, TimeBarParams::five_minutes())
//!     .add_symbol("ETHUSDT", eth_trades, TimeBarParams::one_hour())
//!     .build();
//!
//! // Execute batch with single kernel launch
//! let results = execute_batch(&device, &batch)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use super::traits::CandleAggregator;
use super::types::TradeData;
use crate::gpu::persistent::TaskBatch;

/// Batch builder for candle aggregation tasks
///
/// Generic over aggregator type `A` to ensure type safety when building batches.
/// Provides fluent API for adding multiple symbols with different parameters.
///
/// # Type Safety
///
/// The generic parameter `A` ensures:
/// - Correct parameters for the chosen aggregator
/// - Type-safe batch construction
/// - Zero-cost abstraction
///
/// # Examples
///
/// ```rust,no_run
/// # use kimsfinance_core::gpu::candles::{CandleBatchBuilder, TradeData, TimeBarAggregator, TimeBarParams};
/// let trades = TradeData::from_csv("trades.csv")?;
///
/// // Time bar batch
/// let time_batch = CandleBatchBuilder::<TimeBarAggregator>::new()
///     .add_symbol("BTCUSDT", trades.clone(), TimeBarParams::one_minute())
///     .add_symbol("BTCUSDT", trades.clone(), TimeBarParams::five_minutes())
///     .add_symbol("BTCUSDT", trades, TimeBarParams::one_hour())
///     .build();
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CandleBatchBuilder<A: CandleAggregator> {
    batch: TaskBatch<A>,
    symbol_names: Vec<String>,
}

impl<A: CandleAggregator> CandleBatchBuilder<A> {
    /// Create new empty batch builder
    pub fn new() -> Self {
        Self {
            batch: TaskBatch::new(),
            symbol_names: Vec::new(),
        }
    }

    /// Add symbol with trade data and parameters
    ///
    /// # Arguments
    ///
    /// * `name` - Symbol name (e.g., "BTCUSDT") for identification
    /// * `trades` - Trade data to aggregate
    /// * `params` - Aggregator-specific parameters
    ///
    /// # Returns
    ///
    /// Self reference for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kimsfinance_core::gpu::candles::{CandleBatchBuilder, TradeData, TimeBarAggregator, TimeBarParams};
    /// let trades = TradeData::from_csv("trades.csv")?;
    ///
    /// let batch = CandleBatchBuilder::<TimeBarAggregator>::new()
    ///     .add_symbol("BTCUSDT", trades, TimeBarParams::one_minute())
    ///     .build();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_symbol(mut self, name: &str, trades: TradeData, params: A::Params) -> Self {
        // Convert TradeData to concatenated buffer format for GPU
        let input_data = trades.concat_buffers();
        self.batch.add_task(input_data, params);
        self.symbol_names.push(name.to_string());
        self
    }

    /// Add symbol with mutable reference (alternative API)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kimsfinance_core::gpu::candles::{CandleBatchBuilder, TradeData, TimeBarAggregator, TimeBarParams};
    /// let trades = TradeData::from_csv("trades.csv")?;
    ///
    /// let mut builder = CandleBatchBuilder::<TimeBarAggregator>::new();
    /// builder.add_symbol_mut("BTCUSDT", trades, TimeBarParams::one_minute());
    /// let batch = builder.build();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_symbol_mut(
        &mut self,
        name: &str,
        trades: TradeData,
        params: A::Params,
    ) -> &mut Self {
        let input_data = trades.concat_buffers();
        self.batch.add_task(input_data, params);
        self.symbol_names.push(name.to_string());
        self
    }

    /// Build final task batch
    ///
    /// Consumes the builder and returns the constructed TaskBatch.
    ///
    /// # Returns
    ///
    /// TaskBatch ready for execution
    pub fn build(self) -> TaskBatch<A> {
        self.batch
    }

    /// Get symbol names in batch order
    ///
    /// Useful for matching results back to symbols after execution.
    ///
    /// # Returns
    ///
    /// Reference to vector of symbol names
    pub fn symbol_names(&self) -> &[String] {
        &self.symbol_names
    }

    /// Get number of tasks in batch
    pub fn len(&self) -> usize {
        self.batch.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.batch.is_empty()
    }
}

impl<A: CandleAggregator> Default for CandleBatchBuilder<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-symbol result wrapper
///
/// Pairs symbol names with their aggregated candle results for easy identification.
#[derive(Debug, Clone)]
pub struct SymbolCandleResult<O> {
    pub symbol: String,
    pub candles: O,
}

/// Execute batch and return results with symbol names
///
/// Convenience function that pairs results with symbol names from the builder.
///
/// # Arguments
///
/// * `device` - GPU device
/// * `builder` - Batch builder (consumed to get symbol names)
///
/// # Returns
///
/// Vector of (symbol, candles) pairs
///
/// # Example
///
/// ```rust,no_run
/// # use kimsfinance_core::gpu::{GpuDevice, candles::{CandleBatchBuilder, TradeData, TimeBarAggregator, TimeBarParams, execute_batch_with_symbols}};
/// let device = GpuDevice::new()?;
/// let trades = TradeData::from_csv("trades.csv")?;
///
/// let builder = CandleBatchBuilder::<TimeBarAggregator>::new()
///     .add_symbol("BTCUSDT", trades.clone(), TimeBarParams::one_minute())
///     .add_symbol("ETHUSDT", trades, TimeBarParams::one_minute());
///
/// let results = execute_batch_with_symbols(&device, builder)?;
///
/// for result in results {
///     println!("{}: {} candles", result.symbol, result.candles.len());
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn execute_batch_with_symbols<A: CandleAggregator>(
    device: &crate::gpu::device::GpuDevice,
    builder: CandleBatchBuilder<A>,
) -> Result<Vec<SymbolCandleResult<Vec<f64>>>, crate::gpu::device::GpuError> {
    let symbol_names = builder.symbol_names.clone();
    let batch = builder.build();

    let results = crate::gpu::persistent::execute_batch(device, &batch)?;

    Ok(symbol_names
        .into_iter()
        .zip(results)
        .map(|(symbol, candles)| SymbolCandleResult { symbol, candles })
        .collect())
}

/// Batch builder for generic trade data aggregation
///
/// Simplifies common use cases where you have multiple CSV files to process.
pub struct MultiFileBatchBuilder<A: CandleAggregator> {
    builder: CandleBatchBuilder<A>,
}

impl<A: CandleAggregator> MultiFileBatchBuilder<A> {
    /// Create new multi-file batch builder
    pub fn new() -> Self {
        Self {
            builder: CandleBatchBuilder::new(),
        }
    }

    /// Add symbol from CSV file
    ///
    /// Loads trades from CSV and adds to batch in one step.
    ///
    /// # Arguments
    ///
    /// * `csv_path` - Path to CSV file
    /// * `symbol` - Symbol name
    /// * `params` - Aggregator parameters
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kimsfinance_core::gpu::candles::{MultiFileBatchBuilder, TimeBarAggregator, TimeBarParams};
    /// let batch = MultiFileBatchBuilder::<TimeBarAggregator>::new()
    ///     .add_from_csv("btc_trades.csv", "BTCUSDT", TimeBarParams::one_minute())?
    ///     .add_from_csv("eth_trades.csv", "ETHUSDT", TimeBarParams::one_minute())?
    ///     .build();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_from_csv<P: AsRef<std::path::Path>>(
        mut self,
        csv_path: P,
        symbol: &str,
        params: A::Params,
    ) -> Result<Self, super::csv_loader::CsvError> {
        let trades = TradeData::from_csv_enhanced(csv_path)?;
        self.builder = self.builder.add_symbol(symbol, trades, params);
        Ok(self)
    }

    /// Add symbol from CSV with mutable reference
    pub fn add_from_csv_mut<P: AsRef<std::path::Path>>(
        &mut self,
        csv_path: P,
        symbol: &str,
        params: A::Params,
    ) -> Result<&mut Self, super::csv_loader::CsvError> {
        let trades = TradeData::from_csv_enhanced(csv_path)?;
        self.builder.add_symbol_mut(symbol, trades, params);
        Ok(self)
    }

    /// Add symbol from filtered CSV
    ///
    /// Useful when CSV contains multiple symbols.
    pub fn add_from_csv_filtered<P: AsRef<std::path::Path>>(
        mut self,
        csv_path: P,
        symbol: &str,
        params: A::Params,
    ) -> Result<Self, super::csv_loader::CsvError> {
        let trades = TradeData::from_csv_filtered(csv_path, symbol)?;
        self.builder = self.builder.add_symbol(symbol, trades, params);
        Ok(self)
    }

    /// Build final batch
    pub fn build(self) -> CandleBatchBuilder<A> {
        self.builder
    }

    /// Get symbol names
    pub fn symbol_names(&self) -> &[String] {
        self.builder.symbol_names()
    }

    /// Get number of tasks
    pub fn len(&self) -> usize {
        self.builder.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.builder.is_empty()
    }
}

impl<A: CandleAggregator> Default for MultiFileBatchBuilder<A> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::candles::{TimeBarAggregator, TimeBarParams};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_trades() -> TradeData {
        TradeData {
            timestamps: vec![1609459200000, 1609459201000, 1609459202000],
            symbols: vec!["BTCUSDT".to_string(); 3],
            prices: vec![29000.0, 29010.0, 29005.0],
            volumes: vec![1.0, 2.0, 1.5],
            sides: vec![1, -1, 1], // buy, sell, buy
        }
    }

    #[test]
    fn test_batch_builder_fluent_api() {
        let trades1 = create_test_trades();
        let trades2 = create_test_trades();

        let batch = CandleBatchBuilder::<TimeBarAggregator>::new()
            .add_symbol("BTCUSDT", trades1, TimeBarParams::one_minute())
            .add_symbol("ETHUSDT", trades2, TimeBarParams::five_minutes())
            .build();

        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_batch_builder_mutable_api() {
        let trades1 = create_test_trades();
        let trades2 = create_test_trades();

        let mut builder = CandleBatchBuilder::<TimeBarAggregator>::new();
        builder.add_symbol_mut("BTCUSDT", trades1, TimeBarParams::one_minute());
        builder.add_symbol_mut("ETHUSDT", trades2, TimeBarParams::five_minutes());

        assert_eq!(builder.len(), 2);
        assert_eq!(builder.symbol_names().len(), 2);
        assert_eq!(builder.symbol_names()[0], "BTCUSDT");
        assert_eq!(builder.symbol_names()[1], "ETHUSDT");
    }

    #[test]
    fn test_multi_file_batch_builder() {
        let mut file1 = NamedTempFile::new().unwrap();
        writeln!(file1, "timestamp,symbol,price,volume,side").unwrap();
        writeln!(file1, "1609459200000,BTCUSDT,29000.0,1.0,buy").unwrap();
        file1.flush().unwrap();

        let mut file2 = NamedTempFile::new().unwrap();
        writeln!(file2, "timestamp,symbol,price,volume,side").unwrap();
        writeln!(file2, "1609459200000,ETHUSDT,2000.0,1.0,buy").unwrap();
        file2.flush().unwrap();

        let batch_builder = MultiFileBatchBuilder::<TimeBarAggregator>::new()
            .add_from_csv(file1.path(), "BTCUSDT", TimeBarParams::one_minute())
            .unwrap()
            .add_from_csv(file2.path(), "ETHUSDT", TimeBarParams::one_minute())
            .unwrap()
            .build();

        assert_eq!(batch_builder.len(), 2);
        assert_eq!(batch_builder.symbol_names()[0], "BTCUSDT");
        assert_eq!(batch_builder.symbol_names()[1], "ETHUSDT");
    }

    #[test]
    fn test_symbol_names_tracking() {
        let trades = create_test_trades();

        let builder = CandleBatchBuilder::<TimeBarAggregator>::new()
            .add_symbol("BTC", trades.clone(), TimeBarParams::one_minute())
            .add_symbol("ETH", trades.clone(), TimeBarParams::five_minutes())
            .add_symbol("SOL", trades, TimeBarParams::one_hour());

        let names = builder.symbol_names();
        assert_eq!(names.len(), 3);
        assert_eq!(names[0], "BTC");
        assert_eq!(names[1], "ETH");
        assert_eq!(names[2], "SOL");
    }
}
