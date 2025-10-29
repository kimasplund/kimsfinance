//! Heston Stochastic Volatility Model
//!
//! GPU-accelerated option pricing and calibration using the Heston model.

#[cfg(feature = "gpu")]
pub mod greeks;
pub mod model;
pub mod strategies;

#[cfg(feature = "heston")]
pub mod calibration;
#[cfg(feature = "heston")]
pub mod constraints;
#[cfg(feature = "heston")]
pub mod objective;

#[cfg(feature = "gpu")]
pub use greeks::{GreeksError, HestonGreeksCalculator};
pub use model::{Greeks, HestonParams, OptionQuote, OptionType, ValidationError};
pub use strategies::{
    DeltaHedgingStrategy, HedgeRecommendation, OptionPosition, PortfolioGreeks, TradeSignal,
    VolArbitrageStrategy,
};

#[cfg(feature = "heston")]
pub use calibration::{CalibrationError, CalibrationResult, HestonCalibrator};
#[cfg(feature = "heston")]
pub use constraints::ParameterBounds;
#[cfg(feature = "heston")]
pub use objective::HestonObjective;
