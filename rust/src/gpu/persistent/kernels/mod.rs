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
//! - **SMA** (Simple Moving Average): Average of close prices over period
//! - **EMA** (Exponential Moving Average): Weighted average favoring recent prices
//! - **Williams %R**: Momentum indicator measuring overbought/oversold levels (range: -100 to 0)
//! - **CCI** (Commodity Channel Index): Measures deviation from statistical average
//! - **OBV** (On-Balance Volume): Cumulative momentum indicator relating volume to price changes
//! - **CMF** (Chaikin Money Flow): Volume-based accumulation/distribution indicator
//! - **VWMA** (Volume-Weighted Moving Average): Moving average weighted by trading volume
//! - **WMA** (Weighted Moving Average): Moving average with linearly decreasing weights
//!
//! ## Multi-Output Indicators
//! - **MACD** (Moving Average Convergence Divergence): Trend-following momentum indicator
//!   - Outputs: MACD line, signal line, histogram
//! - **Bollinger Bands**: Volatility bands based on standard deviation
//!   - Outputs: upper band, middle band (SMA), lower band
//! - **Stochastic**: Momentum oscillator measuring price position within high-low range
//!   - Outputs: %K line (fast), %D line (slow)
//! - **Donchian Channels**: Breakout indicator using rolling max/min
//!   - Outputs: upper, middle, lower
//! - **Keltner Channels**: Volatility-based envelopes around EMA
//!   - Outputs: upper, middle, lower
//! - **Aroon**: Time-based momentum indicator
//!   - Outputs: aroon_up, aroon_down, oscillator
//! - **Elder Ray**: Bull/Bear power relative to EMA
//!   - Outputs: bull power, bear power
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

pub mod aroon;
pub mod atr;
pub mod bollinger;
pub mod cci;
pub mod cmf;
pub mod donchian;
pub mod elder_ray;
pub mod ema;
pub mod keltner;
pub mod macd;
pub mod obv;
pub mod roc;
pub mod rsi;
pub mod sma;
pub mod stochastic;
pub mod vwma;
pub mod williams_r;
pub mod wma;

// Re-export indicator types for convenience
pub use aroon::AroonIndicator;
pub use atr::AtrIndicator;
pub use bollinger::{BollingerIndicator, BollingerParams};
pub use cci::CciIndicator;
pub use cmf::CmfIndicator;
pub use donchian::DonchianIndicator;
pub use elder_ray::ElderRayIndicator;
pub use ema::EmaIndicator;
pub use keltner::{KeltnerIndicator, KeltnerParams};
pub use macd::{MacdIndicator, MacdParams};
pub use obv::ObvIndicator;
pub use roc::RocIndicator;
pub use rsi::RsiIndicator;
pub use sma::SmaIndicator;
pub use stochastic::{StochasticIndicator, StochasticParams};
pub use vwma::VwmaIndicator;
pub use williams_r::WilliamsRIndicator;
pub use wma::WmaIndicator;
