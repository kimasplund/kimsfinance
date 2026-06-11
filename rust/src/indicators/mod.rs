//! Technical Indicators Module
//!
//! Comprehensive collection of 65+ technical indicators implemented in Rust
//! for maximum performance with Python bindings via PyO3.
//!
//! # Architecture
//!
//! - **Core**: Trait system and error types
//! - **Utils**: Optimized array operations (SMA, EMA, rolling stats)
//! - **Moving Averages**: SMA, EMA, WMA, VWMA, DEMA, TEMA, HMA, KAMA, MAMA, ZeroLagEMA, McGinley, LSMA
//! - **Momentum**: RSI, ROC, TSI, Williams%R, Stochastic, Aroon, CCI, MACD, ADX, CMO, Force Index, Ultimate Oscillator, Chaikin Oscillator
//! - **Volatility**: ATR, Bollinger Bands, Keltner Channels, Donchian, Elder Ray, Standard Deviation, Chaikin Volatility, Mass Index, Standard Error, EOM
//! - **Volume**: OBV, VWAP, CMF, MFI, Volume Profile
//! - **Trend**: Parabolic SAR, Pivot Points, Fibonacci, Supertrend, Ichimoku Cloud
//! - **Price**: Typical Price, Median Price, Weighted Close, Average Price, True Range
//! - **Statistical**: Linear Regression, Time Series Forecast, Correlation, Covariance, PROC
//! - **Candlestick Patterns**: 35+ patterns (Hammer, Doji, Engulfing, Stars, etc.)
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

pub mod candlestick;
pub mod core;
pub mod momentum;
pub mod momentum_advanced;
pub mod moving_averages;
pub mod moving_averages_advanced;
pub mod price;
pub mod statistical;
pub mod tick_indicators;
pub mod trend;
pub mod utils;
pub mod volatility;
pub mod volatility_advanced;
pub mod volume;

// Re-export commonly used types
pub use core::{Indicator, IndicatorError, MultiOutputIndicator};
pub use tick_indicators::{TickIndicatorEngine, calculate_indicator_from_trades};

// Re-export all indicators
pub use momentum::{Aroon, CCI, MACD, ROC, RSI, Stochastic, TSI, WilliamsR};
pub use momentum_advanced::{ADX, CMO, ChaikinOscillator, ForceIndex, UltimateOscillator};
pub use moving_averages::{DEMA, EMA, HMA, SMA, TEMA, VWMA, WMA};
pub use moving_averages_advanced::{KAMA, LSMA, MAMA, McGinleyDynamic, ZeroLagEMA};
pub use price::{AveragePrice, MedianPrice, TrueRange, TypicalPrice, WeightedClose};
pub use statistical::{
    CorrelationCoefficient, Covariance, LinearRegression, PROC, TimeSeriesForecast,
};
pub use trend::{FibonacciRetracement, ParabolicSAR, PivotPoints};
pub use volatility::{ATR, BollingerBands, DonchianChannels, ElderRay, KeltnerChannels};
pub use volatility_advanced::{
    ChaikinVolatility, EaseOfMovement, MassIndex, StandardDeviation, StandardError,
};
pub use volume::{CMF, MFI, OBV, VWAP, VolumeProfile};

// Re-export candlestick pattern types
pub use candlestick::{CandlestickPattern, PatternConfig, PatternDetection, recognize_patterns};
