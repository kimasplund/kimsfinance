//! GPU-accelerated backtesting engine for algorithmic trading
//!
//! # Overview
//!
//! This module provides a high-performance backtesting framework with:
//! - GPU-accelerated indicator calculation using 2D/3D CUDA kernels
//! - CPU-only fallback for environments without CUDA
//! - Parameter optimization with genetic algorithms
//! - Hybrid precision (FP8 exploration → FP64 refinement)
//! - QuantConnect-style Strategy trait for user-defined logic
//!
//! # Architecture
//!
//! ```text
//! Strategy (user-defined)
//!    ↓ implements
//! on_data() → Signal (Buy/Sell/Hold)
//!    ↓ collected by
//! BacktestEngine → BacktestResult (metrics)
//!    ↓ optimized by
//! GeneticOptimizer → OptimalParameters
//! ```
//!
//! # GPU Acceleration
//!
//! - 2D kernels: Batch processing across multiple assets
//! - 3D kernels: Parameter sweeps (Period × Asset × Candle)
//! - Auto-detection: Falls back to CPU if no GPU available
//! - Hybrid precision: FP8 for exploration, FP64 for refinement
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::backtest::{BacktestEngine, Strategy, Signal, OHLCVBar, IndicatorValues};
//!
//! struct SimpleRSI {
//!     rsi_period: usize,
//!     buy_threshold: f64,
//!     sell_threshold: f64,
//! }
//!
//! impl Strategy for SimpleRSI {
//!     fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
//!         let rsi = indicators.get("rsi_14").unwrap_or(&50.0);
//!         if *rsi < self.buy_threshold {
//!             Signal::Buy
//!         } else if *rsi > self.sell_threshold {
//!             Signal::Sell
//!         } else {
//!             Signal::Hold
//!         }
//!     }
//!
//!     fn indicators(&self) -> Vec<IndicatorConfig> {
//!         vec![IndicatorConfig::RSI { period: self.rsi_period }]
//!     }
//! }
//!
//! // Run backtest
//! let engine = BacktestEngine::new();
//! let strategy = SimpleRSI { rsi_period: 14, buy_threshold: 30.0, sell_threshold: 70.0 };
//! let result = engine.run(&mut strategy, &ohlcv_data)?;
//!
//! println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
//! println!("Max Drawdown: {:.2}%", result.max_drawdown);
//! ```

pub mod core;
pub mod engine;
pub mod metrics;
pub mod optimizer;
pub mod sweep;
pub mod walkforward;
pub mod multi_objective;
pub mod portfolio;

// Re-export main types for convenience
pub use core::{
    BacktestResult, IndicatorConfig, IndicatorValues, OHLCVBar, ParameterGrid, ParameterRange,
    Signal, Strategy, Trade, TradeDirection,
};
pub use engine::{BacktestEngine, BacktestConfig};
pub use metrics::{
    calculate_max_drawdown, calculate_sharpe_ratio, calculate_win_rate, calculate_sortino_ratio,
    calculate_calmar_ratio, calculate_profit_factor,
};
pub use optimizer::{GeneticOptimizer, OptimizerResult};
pub use walkforward::{WalkForwardAnalyzer, WalkForwardConfig, WalkForwardResult, WalkForwardWindow};
pub use multi_objective::{
    MultiObjectiveOptimizer, MultiObjectiveResult, Objective, Solution,
};
pub use portfolio::{
    PortfolioBacktest, PortfolioConfig, PortfolioResult, PortfolioStrategy, AssetData,
    AllocationStrategy, RebalanceFrequency,
};

#[cfg(feature = "gpu")]
pub use sweep::run_parameter_sweep_gpu;

#[cfg(not(feature = "gpu"))]
pub use sweep::run_parameter_sweep_cpu;
