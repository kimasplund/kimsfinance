//! Heston Stochastic Volatility Model
//!
//! GPU-accelerated option pricing and calibration using the Heston model.

pub mod black_scholes;
#[cfg(feature = "gpu")]
pub mod greeks;
// PHASE 3c: GPU Greeks module (enabled after Phase 1 complete)
#[cfg(feature = "gpu")]
pub mod greeks_gpu;
pub mod model;
pub mod strategies;
// PHASE 3c: GPU Strategies module (enabled after Phase 1 complete)
#[cfg(feature = "gpu")]
pub mod strategies_gpu;

#[cfg(feature = "heston")]
pub mod calibration;
#[cfg(feature = "heston")]
pub mod constraints;
#[cfg(feature = "heston")]
pub mod objective;

pub use black_scholes::BlackScholesPricer;
#[cfg(feature = "gpu")]
pub use greeks::{GreeksError, HestonGreeksCalculator};
// PHASE 3c: GPU Greeks exports (enabled after Phase 1 complete)
#[cfg(feature = "gpu")]
pub use greeks_gpu::GreeksGpuCalculator;
pub use model::{Greeks, HestonParams, OptionQuote, OptionType, ValidationError};
pub use strategies::{
    DeltaHedgingStrategy, HedgeRecommendation, OptionPosition, PortfolioGreeks, TradeSignal,
    VolArbitrageStrategy,
};
// PHASE 3c: GPU Strategies exports (enabled after Phase 1 complete)
#[cfg(feature = "gpu")]
pub use strategies_gpu::{StraddleParams, StraddleSignal, StraddleStrategyGpu};

#[cfg(feature = "heston")]
pub use calibration::{CalibrationError, CalibrationResult, HestonCalibrator};
#[cfg(feature = "heston")]
pub use constraints::ParameterBounds;
#[cfg(feature = "heston")]
pub use objective::HestonObjective;
