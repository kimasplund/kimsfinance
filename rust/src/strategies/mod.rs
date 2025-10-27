//! Production-ready trading strategies library
//!
//! This module provides a comprehensive collection of battle-tested trading strategies
//! organized by category:
//!
//! - **Momentum**: RSI mean reversion, MACD trend following, Stochastic, ROC breakout
//! - **Trend Following**: EMA crossovers, MA ribbons, Donchian breakout, Keltner trend
//! - **Volatility**: Bollinger Bands, ATR breakout, Elder Ray
//! - **Composite**: Multi-indicator strategies combining momentum, trend, and volatility
//!
//! All strategies include:
//! - Default parameters based on industry best practices
//! - Parameter ranges for optimization
//! - Risk management (stop loss, take profit)
//! - Expected market conditions documentation
//! - Performance characteristics
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::strategies::momentum::RSIMeanReversion;
//! use kimsfinance_core::backtest::BacktestEngine;
//!
//! let mut strategy = RSIMeanReversion::default();
//! let engine = BacktestEngine::new();
//! let result = engine.run(&mut strategy, &timestamps, &open, &high, &low, &close, &volume)?;
//!
//! println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
//! ```

pub mod composite;
pub mod momentum;
pub mod trend;
pub mod volatility;

// Re-export all strategies for convenience
pub use momentum::{
    CCIReversal, MACDDivergence, MACDTrendFollowing, ROCBreakout, RSIMeanReversion,
    RSIOversoldOverbought, StochasticOscillator,
};

pub use trend::{DonchianBreakout, EMACrossover, KeltnerTrend, TripleEMATrend};

pub use volatility::{ATRVolatilityBreakout, BollingerBandsExpansion, BollingerBandsSqueeze};

pub use composite::{
    BollingerWithStochastic, MACDWithEMA, RSIWithATR, TripleConfirmation, VolatilityMomentum,
};
