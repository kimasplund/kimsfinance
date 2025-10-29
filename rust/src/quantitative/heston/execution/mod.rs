//! Options Execution Engine
//!
//! Complete execution system for options strategies with P&L tracking,
//! expiration handling, and position management.
//!
//! # Architecture
//!
//! ```text
//! ExecutionEngine
//!    ├── PositionManager (track positions, Greeks, cash)
//!    ├── ExpirationHandler (auto-exercise, settlement)
//!    └── PnLTracker (realized/unrealized P&L, metrics)
//! ```
//!
//! # Performance
//!
//! - Handle 1000 positions in <50ms
//! - Expiration checks in <10ms
//! - P&L updates in <20ms
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::quantitative::heston::execution::{
//!     ExecutionEngine, ExecutionConfig
//! };
//!
//! let config = ExecutionConfig {
//!     trading_fee: 0.001,
//!     slippage: 0.0005,
//!     max_position_size: 100,
//!     margin_requirement: 0.2,
//! };
//!
//! let mut engine = ExecutionEngine::new(10_000.0, config);
//!
//! // Execute signals
//! let trades = engine.execute_signals(&signals, &market_data)?;
//!
//! // Process time step (update positions, check expirations)
//! let result = engine.process_time_step(current_time, &market_data)?;
//!
//! // Get performance report
//! let report = engine.get_execution_report();
//! println!("Total P&L: ${:.2}", report.total_pnl);
//! println!("Sharpe: {:.2}", report.sharpe_ratio);
//! ```

pub mod engine;
pub mod expiration;
pub mod pnl_tracker;
pub mod position_manager;

pub use engine::{ExecutionConfig, ExecutionEngine, ExecutionReport, TimeStepResult};
pub use expiration::{ExpirationEvent, ExpirationHandler};
pub use pnl_tracker::{PnLTracker, PerformanceMetrics};
pub use position_manager::{OptionPosition, PositionManager, PositionUpdate};

use crate::quantitative::heston::{Greeks, OptionType};
use std::collections::HashMap;
use thiserror::Error;

/// Execution errors
#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("Insufficient capital: need ${0:.2}, have ${1:.2}")]
    InsufficientCapital(f64, f64),

    #[error("Position not found: {0}")]
    PositionNotFound(String),

    #[error("Invalid position size: {0}")]
    InvalidPositionSize(i32),

    #[error("Margin requirement not met: need ${0:.2}, have ${1:.2}")]
    MarginRequirementNotMet(f64, f64),

    #[error("Maximum position size exceeded: {0} > {1}")]
    MaxPositionSizeExceeded(usize, usize),

    #[error("Invalid execution config: {0}")]
    InvalidConfig(String),
}

/// Market data snapshot for execution
#[derive(Debug, Clone)]
pub struct MarketData {
    /// Current underlying price
    pub underlying_price: f64,

    /// Option prices by position ID
    pub option_prices: HashMap<String, f64>,

    /// Greeks by position ID
    pub option_greeks: HashMap<String, Greeks>,

    /// Current timestamp
    pub timestamp: i64,
}

/// Trade executed by the engine
#[derive(Debug, Clone)]
pub struct Trade {
    /// Unique trade ID
    pub trade_id: String,

    /// Position ID this trade affects
    pub position_id: String,

    /// Timestamp of trade
    pub timestamp: i64,

    /// Option type
    pub option_type: OptionType,

    /// Strike price
    pub strike: f64,

    /// Expiration timestamp
    pub expiration: i64,

    /// Quantity traded (positive = buy, negative = sell)
    pub quantity: i32,

    /// Execution price
    pub price: f64,

    /// Trading fee paid
    pub fee: f64,

    /// Total cost (including fees)
    pub total_cost: f64,

    /// Realized P&L (for closing trades)
    pub realized_pnl: Option<f64>,
}

impl Trade {
    /// Check if this is an opening trade
    pub fn is_opening(&self) -> bool {
        self.realized_pnl.is_none()
    }

    /// Check if this is a closing trade
    pub fn is_closing(&self) -> bool {
        self.realized_pnl.is_some()
    }
}

/// Option signal for execution
#[derive(Debug, Clone)]
pub struct OptionSignal {
    /// Option type
    pub option_type: OptionType,

    /// Strike price
    pub strike: f64,

    /// Expiration timestamp
    pub expiration: i64,

    /// Signal type
    pub signal_type: SignalType,

    /// Recommended quantity (positive = buy, negative = sell)
    pub quantity: i32,

    /// Signal strength (0.0 to 1.0)
    pub strength: f64,
}

/// Signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Open long position
    OpenLong,

    /// Open short position
    OpenShort,

    /// Close position
    Close,

    /// Adjust position (add or reduce)
    Adjust,
}
