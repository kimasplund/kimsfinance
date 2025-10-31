//! Core data types for options strategy framework

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Option type (Call or Put)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

impl fmt::Display for OptionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptionType::Call => write!(f, "CALL"),
            OptionType::Put => write!(f, "PUT"),
        }
    }
}

/// Options contract with full data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionContract {
    /// Underlying symbol (e.g., "AAPL")
    pub symbol: String,

    /// Contract symbol (e.g., "AAPL241220P00450000")
    pub contract_symbol: String,

    /// Strike price
    pub strike: f64,

    /// Expiration date
    pub expiration: NaiveDate,

    /// Option type (Call/Put)
    pub option_type: OptionType,

    /// Snapshot date (when data was collected)
    pub snapshot_date: NaiveDate,

    /// Pricing data
    pub bid: f64,
    pub ask: f64,
    pub last_price: f64,

    /// Market data
    pub volume: f64,
    pub open_interest: f64,

    /// Greeks
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub theta: Option<f64>,
    pub vega: Option<f64>,
    pub rho: Option<f64>,

    /// Implied volatility
    pub implied_volatility: Option<f64>,

    /// Days to expiration
    pub dte: i32,
}

impl OptionContract {
    /// Get mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }

    /// Get bid-ask spread
    pub fn spread(&self) -> f64 {
        self.ask - self.bid
    }

    /// Get spread as percentage of mid
    pub fn spread_pct(&self) -> f64 {
        let mid = self.mid_price();
        if mid > 0.0 {
            self.spread() / mid * 100.0
        } else {
            0.0
        }
    }

    /// Check if option is in-the-money given spot price
    pub fn is_itm(&self, spot_price: f64) -> bool {
        match self.option_type {
            OptionType::Call => spot_price > self.strike,
            OptionType::Put => spot_price < self.strike,
        }
    }

    /// Get intrinsic value given spot price
    pub fn intrinsic_value(&self, spot_price: f64) -> f64 {
        match self.option_type {
            OptionType::Call => (spot_price - self.strike).max(0.0),
            OptionType::Put => (self.strike - spot_price).max(0.0),
        }
    }

    /// Get extrinsic (time) value
    pub fn extrinsic_value(&self, spot_price: f64) -> f64 {
        self.mid_price() - self.intrinsic_value(spot_price)
    }

    /// Get moneyness (spot / strike for calls, strike / spot for puts)
    pub fn moneyness(&self, spot_price: f64) -> f64 {
        match self.option_type {
            OptionType::Call => spot_price / self.strike,
            OptionType::Put => self.strike / spot_price,
        }
    }
}

/// Position side (Long or Short)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
}

/// A leg in an options strategy (single contract position)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionLeg {
    /// The contract
    pub contract: OptionContract,

    /// Position side (long/short)
    pub side: PositionSide,

    /// Number of contracts
    pub quantity: i32,

    /// Entry price (per contract)
    pub entry_price: f64,

    /// Exit price (if closed, per contract)
    pub exit_price: Option<f64>,

    /// Entry date
    pub entry_date: NaiveDate,

    /// Exit date (if closed)
    pub exit_date: Option<NaiveDate>,
}

impl OptionLeg {
    /// Get P&L for this leg (per contract)
    pub fn pnl_per_contract(&self) -> f64 {
        if let Some(exit_price) = self.exit_price {
            match self.side {
                PositionSide::Long => exit_price - self.entry_price,
                PositionSide::Short => self.entry_price - exit_price,
            }
        } else {
            0.0 // Position still open
        }
    }

    /// Get total P&L for this leg (quantity * per-contract P&L)
    pub fn total_pnl(&self) -> f64 {
        self.pnl_per_contract() * self.quantity as f64 * 100.0 // Options are per 100 shares
    }

    /// Get current value (mark-to-market)
    pub fn current_value(&self, current_price: f64) -> f64 {
        let value = match self.side {
            PositionSide::Long => current_price,
            PositionSide::Short => -current_price,
        };
        value * self.quantity as f64 * 100.0
    }

    /// Get unrealized P&L given current price
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        let pnl = match self.side {
            PositionSide::Long => current_price - self.entry_price,
            PositionSide::Short => self.entry_price - current_price,
        };
        pnl * self.quantity as f64 * 100.0
    }
}

/// Multi-leg options position (spread, iron condor, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionsPosition {
    /// Unique position ID
    pub id: String,

    /// Strategy name
    pub strategy: String,

    /// Legs in this position
    pub legs: Vec<OptionLeg>,

    /// Entry date
    pub entry_date: NaiveDate,

    /// Exit date (if closed)
    pub exit_date: Option<NaiveDate>,

    /// Max profit potential
    pub max_profit: Option<f64>,

    /// Max loss potential
    pub max_loss: Option<f64>,
}

impl OptionsPosition {
    /// Get total entry cost (debit paid or credit received)
    pub fn entry_cost(&self) -> f64 {
        self.legs
            .iter()
            .map(|leg| {
                let cost = match leg.side {
                    PositionSide::Long => -leg.entry_price, // Paid (negative)
                    PositionSide::Short => leg.entry_price, // Received (positive)
                };
                cost * leg.quantity as f64 * 100.0
            })
            .sum()
    }

    /// Get total realized P&L (for closed positions)
    pub fn realized_pnl(&self) -> f64 {
        self.legs
            .iter()
            .filter(|leg| leg.exit_price.is_some())
            .map(|leg| leg.total_pnl())
            .sum()
    }

    /// Get total unrealized P&L given current prices
    pub fn unrealized_pnl(&self, current_prices: &[f64]) -> f64 {
        self.legs
            .iter()
            .zip(current_prices.iter())
            .map(|(leg, &price)| leg.unrealized_pnl(price))
            .sum()
    }

    /// Check if position is closed
    pub fn is_closed(&self) -> bool {
        self.exit_date.is_some() && self.legs.iter().all(|leg| leg.exit_price.is_some())
    }

    /// Get days held
    pub fn days_held(&self) -> i32 {
        if let Some(exit_date) = self.exit_date {
            (exit_date - self.entry_date).num_days() as i32
        } else {
            0
        }
    }

    /// Get return on capital (%)
    pub fn return_on_capital(&self) -> f64 {
        let capital = self.entry_cost().abs();
        if capital > 0.0 {
            (self.realized_pnl() / capital) * 100.0
        } else {
            0.0
        }
    }
}

/// Strategy parameters for backtesting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParams {
    /// Strategy name
    pub name: String,

    /// Days to expiration range for entry
    pub dte_min: i32,
    pub dte_max: i32,

    /// Delta range for option selection
    pub delta_min: f64,
    pub delta_max: f64,

    /// Profit target (% of max profit)
    pub profit_target_pct: Option<f64>,

    /// Stop loss (% of max loss)
    pub stop_loss_pct: Option<f64>,

    /// Maximum days to hold
    pub max_hold_days: Option<i32>,

    /// Position sizing (% of capital per trade)
    pub position_size_pct: f64,

    /// Minimum credit received (for credit spreads)
    pub min_credit: Option<f64>,

    /// Commission per contract (e.g., $0.65)
    pub commission_per_contract: f64,

    /// Slippage in ticks (e.g., 1.0 = $0.05 per contract)
    pub slippage_ticks: f64,

    /// Apply bid-ask spread modeling
    pub apply_bid_ask_spread: bool,

    /// Additional strategy-specific parameters
    pub custom_params: std::collections::HashMap<String, f64>,
}

impl Default for StrategyParams {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            dte_min: 30,
            dte_max: 45,
            delta_min: 0.15,
            delta_max: 0.35,
            profit_target_pct: Some(50.0),
            stop_loss_pct: Some(200.0),
            max_hold_days: Some(21),
            position_size_pct: 10.0,
            min_credit: Some(0.30),
            commission_per_contract: 0.65, // $0.65 per contract (retail broker)
            slippage_ticks: 1.0,           // 1 tick = $0.05
            apply_bid_ask_spread: true,    // Use realistic bid/ask
            custom_params: std::collections::HashMap::new(),
        }
    }
}

/// Backtest result for a single parameter set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Strategy parameters used
    pub params: StrategyParams,

    /// Number of trades executed
    pub num_trades: usize,

    /// Total P&L
    pub total_pnl: f64,

    /// Win rate (%)
    pub win_rate: f64,

    /// Average win
    pub avg_win: f64,

    /// Average loss
    pub avg_loss: f64,

    /// Maximum drawdown
    pub max_drawdown: f64,

    /// Sharpe ratio
    pub sharpe_ratio: f64,

    /// Sortino ratio
    pub sortino_ratio: f64,

    /// Profit factor
    pub profit_factor: f64,

    /// Maximum consecutive losses
    pub max_consecutive_losses: i32,

    /// Average days in trade
    pub avg_days_in_trade: f64,

    /// Return on capital (%)
    pub return_on_capital: f64,

    /// All positions
    pub positions: Vec<OptionsPosition>,
}
