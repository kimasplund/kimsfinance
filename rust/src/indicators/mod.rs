//! Technical Indicators Module
//!
//! Comprehensive collection of 30+ technical indicators implemented in Rust
//! for maximum performance with Python bindings via PyO3.
//!
//! # Architecture
//!
//! - **Core**: Trait system and error types
//! - **Utils**: Optimized array operations (SMA, EMA, rolling stats)
//! - **Moving Averages**: SMA, EMA, WMA, VWMA, DEMA, TEMA, HMA
//! - **Momentum**: RSI, ROC, TSI, Williams%R, Stochastic, Aroon, CCI, MACD
//! - **Volatility**: ATR, Bollinger Bands, Keltner Channels, Donchian, Elder Ray
//! - **Volume**: OBV, VWAP, CMF, Volume Profile
//! - **Trend**: Parabolic SAR, Pivot Points, Fibonacci
//!
//! # Performance Strategy
//!
//! For datasets:
//! - **< 1,000 rows**: Individual indicator calls (3-4x faster than Python)
//! - **> 1,000 rows**: Batch API to minimize FFI overhead
//!
//! # Example
//!
//! ```rust
//! use ndarray::arr1;
//! use indicators::{Indicator, RSI};
//!
//! let prices = arr1(&[100.0, 102.0, 101.0, 105.0, 103.0, 107.0]);
//! let rsi = RSI::new(14).unwrap();
//! let result = rsi.calculate(prices.view()).unwrap();
//! ```

pub mod core;
pub mod momentum;
pub mod moving_averages;
pub mod trend;
pub mod utils;
pub mod volatility;
pub mod volume;

// Re-export commonly used types
pub use core::{Indicator, IndicatorError, MultiOutputIndicator};

// Re-export all indicators
pub use momentum::{Aroon, CCI, MACD, ROC, RSI, Stochastic, TSI, WilliamsR};
pub use moving_averages::{DEMA, EMA, HMA, SMA, TEMA, VWMA, WMA};
pub use trend::{FibonacciRetracement, ParabolicSAR, PivotPoints};
pub use volatility::{ATR, BollingerBands, DonchianChannels, ElderRay, KeltnerChannels};
pub use volume::{CMF, MFI, OBV, VWAP, VolumeProfile};
