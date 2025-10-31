//! Options Trading Strategy Framework
//!
//! GPU-accelerated backtesting and parameter optimization for options strategies.
//!
//! ## Features
//! - Historical data loading from parquet files
//! - Multiple strategy types (credit spreads, iron condors, etc.)
//! - GPU-accelerated parameter sweeps
//! - Comprehensive performance metrics
//! - Risk management and position sizing
//!
//! ## Architecture
//! ```text
//! Data Layer (Parquet)
//!   ↓
//! Strategy Engine
//!   ├── Entry Rules
//!   ├── Exit Rules
//!   └── Position Management
//!   ↓
//! Backtest Engine (CPU/GPU)
//!   ├── P&L Calculation
//!   ├── Risk Metrics
//!   └── Performance Analytics
//!   ↓
//! Parameter Sweep (GPU-accelerated)
//!   ├── Grid Search
//!   ├── Walk-Forward Analysis
//!   └── Out-of-Sample Testing
//! ```

pub mod backtest;
pub mod black_scholes;
pub mod data_loader;
pub mod market_regime;
pub mod metrics;
pub mod spot_data;
pub mod strategies;
pub mod transaction_costs;
pub mod types;

#[cfg(feature = "gpu")]
pub mod gpu_sweep;

pub use backtest::*;
pub use black_scholes::*;
pub use data_loader::*;
pub use market_regime::*;
pub use metrics::*;
pub use spot_data::*;
pub use strategies::*;
pub use transaction_costs::*;
pub use types::*;

#[cfg(feature = "gpu")]
pub use gpu_sweep::*;
