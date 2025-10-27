//! GPU-Accelerated Custom Candle Generation
//!
//! Implements persistent kernel pattern for trade-to-candle aggregation with multiple candle types.
//!
//! # Architecture
//!
//! - **Trade Data Ingestion**: CSV → GPU buffers (timestamps, prices, volumes, sides)
//! - **Candle Types**: Time bars, Volume bars, Tick bars, Range bars, Heikin-Ashi, Renko
//! - **Batch Processing**: Multiple symbols processed in single kernel launch
//! - **Persistent Kernels**: Eliminates 90% of launch overhead for batch operations
//!
//! # Performance Targets
//!
//! - Time bars: 50-100x vs CPU (highly parallel groupby operations)
//! - Volume/Tick bars: 20-50x vs CPU (sequential per-symbol, parallel across symbols)
//! - Heikin-Ashi: 30-70x vs CPU (simple transformations, highly parallel)
//! - Range/Renko bars: 10-30x vs CPU (more sequential logic)
//!
//! # Example: Load Trades CSV and Create 1-Minute Candles
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::candles::*;
//!
//! // Load trades from CSV
//! let trades = TradeData::from_csv("btc_trades.csv")?;
//!
//! // Create time bar batch (1-minute candles)
//! let mut batch = TimeBarBatch::new();
//! batch.add_task(
//!     trades.concat_buffers(), // [timestamps, prices, volumes]
//!     TimeBarParams { interval_seconds: 60 }
//! );
//!
//! // Execute on GPU with persistent kernel (single launch for all symbols!)
//! let device = GpuDevice::new()?;
//! let candles = execute_batch(&device, &batch)?;
//!
//! // Result: OHLCV candles aggregated by minute
//! ```
//!
//! # Example: Batch Process Multiple Symbols
//!
//! ```rust,ignore
//! // Load multiple CSV files
//! let btc_trades = TradeData::from_csv("btc_trades.csv")?;
//! let eth_trades = TradeData::from_csv("eth_trades.csv")?;
//! let sol_trades = TradeData::from_csv("sol_trades.csv")?;
//!
//! // Create batch with all symbols (single GPU launch!)
//! let mut batch = TimeBarBatch::new();
//! batch.add_task(btc_trades.concat_buffers(), TimeBarParams { interval_seconds: 300 }); // 5m
//! batch.add_task(eth_trades.concat_buffers(), TimeBarParams { interval_seconds: 300 });
//! batch.add_task(sol_trades.concat_buffers(), TimeBarParams { interval_seconds: 300 });
//!
//! // Process all 3 symbols in single persistent kernel launch (90% overhead reduction!)
//! let candles = execute_batch(&device, &batch)?;
//! // candles[0] = BTC 5m candles
//! // candles[1] = ETH 5m candles
//! // candles[2] = SOL 5m candles
//! ```
//!
//! # Example: Heikin-Ashi from Existing OHLCV
//!
//! ```rust,ignore
//! // Already have OHLCV candles, convert to Heikin-Ashi
//! let mut batch = HeikinAshiBatch::new();
//! let ohlcv_concat = concat_ohlcv(&open, &high, &low, &close);
//! batch.add_task(ohlcv_concat, ());
//!
//! let ha_candles = execute_batch(&device, &batch)?;
//! // Result: Smoothed Heikin-Ashi candles for trend following
//! ```

#[cfg(feature = "gpu")]
pub mod types;

#[cfg(feature = "gpu")]
pub mod traits;

// Candle type implementations
#[cfg(feature = "gpu")]
pub mod time_bars;

#[cfg(feature = "gpu")]
pub mod volume_bars;

#[cfg(feature = "gpu")]
pub mod tick_bars;

#[cfg(feature = "gpu")]
pub mod range_bars;

#[cfg(feature = "gpu")]
pub mod heikin_ashi;

#[cfg(feature = "gpu")]
pub mod renko;

#[cfg(feature = "gpu")]
pub mod csv_loader;

#[cfg(feature = "gpu")]
pub mod batch_builder;

// Re-export core types
#[cfg(feature = "gpu")]
pub use types::{OHLCVCandle, TradeData, TradeSide};

#[cfg(feature = "gpu")]
pub use traits::CandleAggregator;

// Re-export CSV loader types
#[cfg(feature = "gpu")]
pub use csv_loader::{CsvError, CsvFormat};

// Re-export batch builder
#[cfg(feature = "gpu")]
pub use batch_builder::{CandleBatchBuilder, MultiFileBatchBuilder, SymbolCandleResult, execute_batch_with_symbols};

// Re-export persistent kernel execution (reuses existing infrastructure)
#[cfg(feature = "gpu")]
pub use super::persistent::execute_batch;

// Type aliases for candle batches
#[cfg(feature = "gpu")]
pub use time_bars::{TimeBarAggregator, TimeBarBatch, TimeBarParams};

#[cfg(feature = "gpu")]
pub use volume_bars::{VolumeBarAggregator, VolumeBarParams};

#[cfg(feature = "gpu")]
pub use tick_bars::{TickBarAggregator, TickBarParams};

#[cfg(feature = "gpu")]
pub use range_bars::{RangeBarAggregator, RangeBarParams};

#[cfg(feature = "gpu")]
pub use heikin_ashi::HeikinAshiAggregator;

#[cfg(feature = "gpu")]
pub use renko::{RenkoAggregator, RenkoParams};

// Batch type aliases (using TaskBatch from persistent module)
#[cfg(feature = "gpu")]
use super::persistent::TaskBatch;

#[cfg(feature = "gpu")]
pub type VolumeBarBatch = TaskBatch<VolumeBarAggregator>;

#[cfg(feature = "gpu")]
pub type TickBarBatch = TaskBatch<TickBarAggregator>;

#[cfg(feature = "gpu")]
pub type RangeBarBatch = TaskBatch<RangeBarAggregator>;

#[cfg(feature = "gpu")]
pub type HeikinAshiBatch = TaskBatch<HeikinAshiAggregator>;

#[cfg(feature = "gpu")]
pub type RenkoBatch = TaskBatch<RenkoAggregator>;
