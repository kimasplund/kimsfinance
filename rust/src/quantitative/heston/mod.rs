//! Heston Stochastic Volatility Model
//!
//! GPU-accelerated option pricing and calibration using the Heston model.

pub mod black_scholes;
#[cfg(feature = "heston")]
pub mod greeks;
// PHASE 3c: GPU Greeks module (enabled after Phase 1 complete)
#[cfg(feature = "heston")]
pub mod greeks_gpu;
pub mod model;
pub mod strategies;
// PHASE 3c: GPU Strategies module (enabled after Phase 1 complete)
#[cfg(feature = "heston")]
pub mod strategies_gpu;
// PHASE 3a: Delta-Neutral and Vol Arbitrage GPU Strategies
#[cfg(feature = "heston")]
pub mod strategies_delta_neutral;
#[cfg(feature = "heston")]
pub mod strategies_vol_arbitrage;

// PHASE 4: Options Execution Engine
pub mod execution;

#[cfg(feature = "heston")]
pub mod calibration;
#[cfg(feature = "heston")]
pub mod constraints;
#[cfg(feature = "heston")]
pub mod objective;

pub use black_scholes::BlackScholesPricer;
#[cfg(feature = "heston")]
pub use greeks::{GreeksError, HestonGreeksCalculator};
// PHASE 3c: GPU Greeks exports (enabled after Phase 1 complete)
#[cfg(feature = "heston")]
pub use greeks_gpu::GreeksGpuCalculator;
pub use model::{Greeks, HestonParams, OptionQuote, OptionType, ValidationError};
pub use strategies::{
    DeltaHedgingStrategy, HedgeRecommendation, OptionPosition, PortfolioGreeks, TradeSignal,
    VolArbitrageStrategy,
};
// PHASE 3c: GPU Strategies exports (enabled after Phase 1 complete)
#[cfg(feature = "heston")]
pub use strategies_gpu::{
    CoveredCallParams, CoveredCallSignal, CoveredCallStrategyGpu, IronCondorParams,
    IronCondorSignal, IronCondorStrategyGpu, StraddleParams, StraddleSignal, StraddleStrategyGpu,
};
// PHASE 3a: Delta-Neutral and Vol Arbitrage GPU Strategy exports
#[cfg(feature = "heston")]
pub use strategies_delta_neutral::{
    DeltaNeutralParams, DeltaNeutralSignal, DeltaNeutralStrategyGpu, RebalanceSignal,
};
#[cfg(feature = "heston")]
pub use strategies_vol_arbitrage::{
    EdgeMonitor, VolArbitrageParams, VolArbitragePnL, VolArbitrageSignal, VolArbitrageStrategyGpu,
};

// PHASE 4: Execution Engine exports
pub use execution::{
    ExecutionConfig, ExecutionEngine, ExecutionError, ExecutionReport, ExpirationEvent,
    ExpirationHandler, MarketData, OptionSignal, PerformanceMetrics, PnLTracker, SignalType,
    TimeStepResult, Trade,
};

#[cfg(feature = "heston")]
pub use calibration::{CalibrationError, CalibrationResult, HestonCalibrator};
#[cfg(feature = "heston")]
pub use constraints::ParameterBounds;
#[cfg(feature = "heston")]
pub use objective::HestonObjective;
