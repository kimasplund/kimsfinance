//! Heston Stochastic Volatility Model
//!
//! GPU-accelerated option pricing and calibration using the Heston model.

pub mod black_scholes;
#[cfg(feature = "gpu")]
pub mod greeks;
// PHASE 2: Temporarily disabled greeks_gpu - API incompatibility (not part of Phase 2)
// TODO: Fix greeks_gpu.rs to use current cudarc API after Phase 2 complete
// #[cfg(feature = "gpu")]
// pub mod greeks_gpu;
pub mod model;
pub mod strategies;
// PHASE 2: Temporarily disabled strategies_gpu - API incompatibility (not part of Phase 2)
// #[cfg(feature = "gpu")]
// pub mod strategies_gpu;

#[cfg(feature = "heston")]
pub mod calibration;
#[cfg(feature = "heston")]
pub mod constraints;
#[cfg(feature = "heston")]
pub mod objective;

pub use black_scholes::BlackScholesPricer;
#[cfg(feature = "gpu")]
pub use greeks::{GreeksError, HestonGreeksCalculator};
// PHASE 2: Temporarily disabled greeks_gpu exports
// #[cfg(feature = "gpu")]
// pub use greeks_gpu::GreeksGpuCalculator;
pub use model::{Greeks, HestonParams, OptionQuote, OptionType, ValidationError};
pub use strategies::{
    DeltaHedgingStrategy, HedgeRecommendation, OptionPosition, PortfolioGreeks, TradeSignal,
    VolArbitrageStrategy,
};
// PHASE 2: Temporarily disabled strategies_gpu exports
// #[cfg(feature = "gpu")]
// pub use strategies_gpu::{StraddleParams, StraddleSignal, StraddleStrategyGpu};

#[cfg(feature = "heston")]
pub use calibration::{CalibrationError, CalibrationResult, HestonCalibrator};
#[cfg(feature = "heston")]
pub use constraints::ParameterBounds;
#[cfg(feature = "heston")]
pub use objective::HestonObjective;
