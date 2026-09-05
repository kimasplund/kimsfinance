//! Position Manager
//!
//! Tracks option positions, underlying holdings, and cash balance.

use super::{ExecutionError, MarketData};
use crate::quantitative::heston::{Greeks, OptionType};
use std::collections::HashMap;

/// Option position in portfolio
#[derive(Debug, Clone)]
pub struct OptionPosition {
    /// Unique position identifier
    pub position_id: String,

    /// Option type
    pub option_type: OptionType,

    /// Strike price
    pub strike: f64,

    /// Expiration timestamp
    pub expiration: i64,

    /// Quantity (positive = long, negative = short)
    pub quantity: i32,

    /// Entry price per contract
    pub entry_price: f64,

    /// Entry timestamp
    pub entry_time: i64,

    /// Current market price
    pub current_price: f64,

    /// Unrealized P&L
    pub pnl: f64,

    /// Current Greeks
    pub greeks: Greeks,
}

impl OptionPosition {
    /// Create new option position
    pub fn new(
        position_id: String,
        option_type: OptionType,
        strike: f64,
        expiration: i64,
        quantity: i32,
        entry_price: f64,
        entry_time: i64,
    ) -> Self {
        Self {
            position_id,
            option_type,
            strike,
            expiration,
            quantity,
            entry_price,
            entry_time,
            current_price: entry_price,
            pnl: 0.0,
            greeks: Greeks::default(),
        }
    }

    /// Update position with current market data
    pub fn update(&mut self, current_price: f64, greeks: Greeks) {
        self.current_price = current_price;
        self.greeks = greeks;
        self.pnl = (current_price - self.entry_price) * (self.quantity as f64) * 100.0;
    }

    /// Calculate intrinsic value at expiration
    pub fn intrinsic_value(&self, underlying_price: f64) -> f64 {
        let intrinsic = match self.option_type {
            OptionType::Call => (underlying_price - self.strike).max(0.0),
            OptionType::Put => (self.strike - underlying_price).max(0.0),
        };
        intrinsic * (self.quantity.abs() as f64) * 100.0
    }

    /// Check if position is long
    pub fn is_long(&self) -> bool {
        self.quantity > 0
    }

    /// Check if position is short
    pub fn is_short(&self) -> bool {
        self.quantity < 0
    }

    /// Check if option is in the money
    pub fn is_itm(&self, underlying_price: f64) -> bool {
        match self.option_type {
            OptionType::Call => underlying_price > self.strike,
            OptionType::Put => underlying_price < self.strike,
        }
    }
}

/// Position update notification
#[derive(Debug, Clone)]
pub struct PositionUpdate {
    pub position_id: String,
    pub update_type: UpdateType,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateType {
    Opened,
    Closed,
    Updated,
    Expired,
    Assigned,
}

/// Portfolio Greeks aggregated across all positions
#[derive(Debug, Clone, Copy, Default)]
pub struct PortfolioGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

/// Position Manager
///
/// Tracks all option positions, underlying holdings, and cash balance.
pub struct PositionManager {
    /// All active positions (position_id -> OptionPosition)
    positions: HashMap<String, OptionPosition>,

    /// Underlying shares held (for hedging)
    underlying_position: f64,

    /// Current cash balance
    cash: f64,

    /// Initial capital
    initial_capital: f64,

    /// Position counter for generating IDs
    position_counter: usize,
}

impl PositionManager {
    /// Create new position manager
    pub fn new(initial_capital: f64) -> Self {
        Self {
            positions: HashMap::new(),
            underlying_position: 0.0,
            cash: initial_capital,
            initial_capital,
            position_counter: 0,
        }
    }

    /// Open new option position
    #[allow(clippy::too_many_arguments)] // public API: signature is documented and used by callers
    pub fn open_position(
        &mut self,
        option_type: OptionType,
        strike: f64,
        expiration: i64,
        quantity: i32,
        entry_price: f64,
        entry_time: i64,
        fee: f64,
    ) -> Result<String, ExecutionError> {
        if quantity == 0 {
            return Err(ExecutionError::InvalidPositionSize(quantity));
        }

        // Calculate total cost
        let contract_cost = entry_price * (quantity.abs() as f64) * 100.0;
        let total_cost = contract_cost + fee;

        // Check if we have enough capital
        // For long positions: need to pay premium + fee
        // For short positions: receive premium - fee, but need margin
        let required_capital = if quantity > 0 {
            total_cost
        } else {
            fee // Only pay fee, receive premium
        };

        if self.cash < required_capital {
            return Err(ExecutionError::InsufficientCapital(
                required_capital,
                self.cash,
            ));
        }

        // Generate position ID
        self.position_counter += 1;
        let position_id = format!(
            "{:?}_{}_{}_{}",
            option_type, strike, expiration, self.position_counter
        );

        // Create position
        let position = OptionPosition::new(
            position_id.clone(),
            option_type,
            strike,
            expiration,
            quantity,
            entry_price,
            entry_time,
        );

        // Update cash
        if quantity > 0 {
            self.cash -= total_cost; // Pay premium + fee
        } else {
            self.cash += contract_cost - fee; // Receive premium - fee
        }

        // Store position
        self.positions.insert(position_id.clone(), position);

        Ok(position_id)
    }

    /// Close existing position
    pub fn close_position(
        &mut self,
        position_id: &str,
        exit_price: f64,
        _exit_time: i64,
        fee: f64,
    ) -> Result<f64, ExecutionError> {
        let position = self
            .positions
            .remove(position_id)
            .ok_or_else(|| ExecutionError::PositionNotFound(position_id.to_string()))?;

        // Calculate realized P&L
        let price_diff = exit_price - position.entry_price;
        let contract_value = price_diff * (position.quantity as f64) * 100.0;
        let realized_pnl = contract_value - fee;

        // Update cash
        if position.quantity > 0 {
            // Long position: receive exit price - fee
            self.cash += exit_price * (position.quantity as f64) * 100.0 - fee;
        } else {
            // Short position: pay exit price + fee
            self.cash -= exit_price * (position.quantity.abs() as f64) * 100.0 + fee;
        }

        Ok(realized_pnl)
    }

    /// Update all positions with current market data
    pub fn update_positions(&mut self, market_data: &MarketData) {
        for position in self.positions.values_mut() {
            if let Some(&price) = market_data.option_prices.get(&position.position_id) {
                let greeks = market_data
                    .option_greeks
                    .get(&position.position_id)
                    .copied()
                    .unwrap_or_default();
                position.update(price, greeks);
            }
        }
    }

    /// Handle expiration for all positions
    pub fn handle_expirations(
        &mut self,
        current_time: i64,
        underlying_price: f64,
    ) -> Vec<(String, f64)> {
        let mut expired_positions = Vec::new();

        // Find expired positions
        let expired_ids: Vec<String> = self
            .positions
            .iter()
            .filter(|(_, pos)| pos.expiration <= current_time)
            .map(|(id, _)| id.clone())
            .collect();

        // Process each expiration
        for position_id in expired_ids {
            if let Some(position) = self.positions.remove(&position_id) {
                let settlement = self.calculate_settlement(&position, underlying_price);
                self.cash += settlement;
                expired_positions.push((position_id, settlement));
            }
        }

        expired_positions
    }

    /// Calculate settlement value for expired position
    fn calculate_settlement(&self, position: &OptionPosition, underlying_price: f64) -> f64 {
        let intrinsic = match position.option_type {
            OptionType::Call => (underlying_price - position.strike).max(0.0),
            OptionType::Put => (position.strike - underlying_price).max(0.0),
        };

        // Long positions receive intrinsic value
        // Short positions pay intrinsic value
        intrinsic * (position.quantity as f64) * 100.0
    }

    /// Calculate total unrealized P&L across all positions
    pub fn calculate_total_pnl(&self) -> f64 {
        self.positions.values().map(|pos| pos.pnl).sum()
    }

    /// Calculate portfolio Greeks (aggregated)
    pub fn get_portfolio_greeks(&self) -> PortfolioGreeks {
        let mut portfolio = PortfolioGreeks::default();

        for position in self.positions.values() {
            let qty = position.quantity as f64;
            portfolio.delta += qty * position.greeks.delta.unwrap_or(0.0) * 100.0;
            portfolio.gamma += qty * position.greeks.gamma.unwrap_or(0.0) * 100.0;
            portfolio.vega += qty * position.greeks.vega.unwrap_or(0.0) * 100.0;
            portfolio.theta += qty * position.greeks.theta.unwrap_or(0.0) * 100.0;
            portfolio.rho += qty * position.greeks.rho_greek.unwrap_or(0.0) * 100.0;
        }

        portfolio
    }

    /// Get current cash balance
    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// Get total equity (cash + unrealized P&L)
    pub fn equity(&self) -> f64 {
        self.cash + self.calculate_total_pnl()
    }

    /// Get number of active positions
    pub fn position_count(&self) -> usize {
        self.positions.len()
    }

    /// Get reference to all positions
    pub fn positions(&self) -> &HashMap<String, OptionPosition> {
        &self.positions
    }

    /// Get position by ID
    pub fn get_position(&self, position_id: &str) -> Option<&OptionPosition> {
        self.positions.get(position_id)
    }

    /// Check if position exists
    pub fn has_position(&self, position_id: &str) -> bool {
        self.positions.contains_key(position_id)
    }

    /// Get initial capital
    pub fn initial_capital(&self) -> f64 {
        self.initial_capital
    }

    /// Get underlying position (shares held for hedging)
    pub fn underlying_position(&self) -> f64 {
        self.underlying_position
    }

    /// Update underlying position (for delta hedging)
    pub fn set_underlying_position(&mut self, shares: f64) {
        self.underlying_position = shares;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantitative::heston::OptionType;

    #[test]
    fn test_open_long_call() {
        let mut manager = PositionManager::new(10_000.0);

        let position_id = manager
            .open_position(
                OptionType::Call,
                100.0,
                1735689600, // Some future timestamp
                1,          // 1 contract
                5.0,        // $5 premium
                1735000000,
                1.0, // $1 fee
            )
            .unwrap();

        assert_eq!(manager.position_count(), 1);
        assert_eq!(manager.cash(), 10_000.0 - 500.0 - 1.0); // Initial - premium - fee
        assert!(manager.has_position(&position_id));
    }

    #[test]
    fn test_open_short_put() {
        let mut manager = PositionManager::new(10_000.0);

        let position_id = manager
            .open_position(
                OptionType::Put,
                100.0,
                1735689600,
                -1, // Short 1 contract
                5.0,
                1735000000,
                1.0,
            )
            .unwrap();

        assert_eq!(manager.position_count(), 1);
        assert_eq!(manager.cash(), 10_000.0 + 500.0 - 1.0); // Initial + premium - fee
        assert!(manager.has_position(&position_id));
    }

    #[test]
    fn test_close_position() {
        let mut manager = PositionManager::new(10_000.0);

        let position_id = manager
            .open_position(OptionType::Call, 100.0, 1735689600, 1, 5.0, 1735000000, 1.0)
            .unwrap();

        let realized_pnl = manager
            .close_position(&position_id, 7.0, 1735100000, 1.0)
            .unwrap();

        assert_eq!(manager.position_count(), 0);
        assert!((realized_pnl - (2.0 * 100.0 - 1.0)).abs() < 0.01);
    }

    #[test]
    fn test_insufficient_capital() {
        let mut manager = PositionManager::new(100.0);

        let result = manager.open_position(
            OptionType::Call,
            100.0,
            1735689600,
            1,
            5.0, // Needs $501
            1735000000,
            1.0,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_expiration_itm_call() {
        let mut manager = PositionManager::new(10_000.0);

        manager
            .open_position(OptionType::Call, 100.0, 1735000000, 1, 5.0, 1734900000, 1.0)
            .unwrap();

        let expired = manager.handle_expirations(1735100000, 110.0);

        assert_eq!(expired.len(), 1);
        assert_eq!(manager.position_count(), 0);
        // ITM call: (110 - 100) * 100 = $1000 intrinsic value
        assert!((expired[0].1 - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_expiration_otm_call() {
        let mut manager = PositionManager::new(10_000.0);

        manager
            .open_position(OptionType::Call, 100.0, 1735000000, 1, 5.0, 1734900000, 1.0)
            .unwrap();

        let expired = manager.handle_expirations(1735100000, 95.0);

        assert_eq!(expired.len(), 1);
        assert_eq!(manager.position_count(), 0);
        // OTM call: expires worthless
        assert!((expired[0].1 - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_portfolio_greeks() {
        let mut manager = PositionManager::new(10_000.0);

        let position_id = manager
            .open_position(OptionType::Call, 100.0, 1735689600, 1, 5.0, 1735000000, 1.0)
            .unwrap();

        let greeks = Greeks {
            delta: Some(0.5),
            gamma: Some(0.02),
            vega: Some(0.1),
            theta: Some(-0.05),
            rho_greek: Some(0.03),
        };

        let market_data = MarketData {
            underlying_price: 105.0,
            option_prices: [(position_id.clone(), 7.0)].iter().cloned().collect(),
            option_greeks: [(position_id.clone(), greeks)].iter().cloned().collect(),
            timestamp: 1735100000,
        };

        manager.update_positions(&market_data);

        let portfolio_greeks = manager.get_portfolio_greeks();
        assert!((portfolio_greeks.delta - 50.0).abs() < 0.01); // 0.5 * 100
        assert!((portfolio_greeks.gamma - 2.0).abs() < 0.01); // 0.02 * 100
    }
}
