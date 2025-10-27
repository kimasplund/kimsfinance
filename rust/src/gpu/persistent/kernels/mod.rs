//! Persistent kernel implementations for technical indicators
//!
//! This module contains CUDA kernel implementations for various technical indicators
//! that support the persistent kernel pattern for batch processing.
//!
//! # Available Indicators
//!
//! ## Single-Output Indicators
//! - **RSI** (Relative Strength Index): Momentum oscillator measuring speed and magnitude of price changes
//! - **ATR** (Average True Range): Volatility indicator measuring market volatility
//! - **ROC** (Rate of Change): Momentum indicator showing percentage change over period
//!
//! ## Multi-Output Indicators
//! - **MACD** (Moving Average Convergence Divergence): Trend-following momentum indicator
//!   - Outputs: MACD line, signal line, histogram
//!
//! # Performance Benefits
//!
//! Persistent kernels reduce launch overhead from O(N) to O(1):
//! - **Traditional**: N indicators × 10μs = 10N μs overhead
//! - **Persistent**: 1 launch × 10μs = 10μs overhead
//! - **Speedup**: 2-4x for N ≥ 10 tasks
//!
//! # Example
//!
//! ```rust,no_run
//! use kimsfinance_core::gpu::persistent::kernels::rsi::RsiIndicator;
//! use kimsfinance_core::gpu::persistent::traits::PersistentIndicator;
//! use kimsfinance_core::gpu::GpuDevice;
//!
//! let device = GpuDevice::new()?;
//! let kernel = RsiIndicator::compile_kernel(&device)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod atr;
pub mod macd;
pub mod roc;
pub mod rsi;

// Re-export indicator types for convenience
pub use atr::AtrIndicator;
pub use macd::{MacdIndicator, MacdParams};
pub use roc::RocIndicator;
pub use rsi::RsiIndicator;
