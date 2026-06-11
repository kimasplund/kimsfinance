//! Execution Engine
//!
//! Orchestrates position management, expiration handling, and P&L tracking.

use super::{
    ExecutionError, MarketData, OptionSignal, SignalType, Trade,
    pnl_tracker::{PerformanceMetrics, PnLTracker},
    position_manager::PositionManager,
};
use crate::quantitative::heston::OptionType;
use std::time::Instant;

/// Execution configuration
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Trading fee per contract
    pub trading_fee: f64,

    /// Slippage (as fraction of price)
    pub slippage: f64,

    /// Maximum number of positions
    pub max_position_size: usize,

    /// Margin requirement for short positions (as fraction)
    pub margin_requirement: f64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            trading_fee: 1.0, // $1 per contract
            slippage: 0.0005, // 0.05%
            max_position_size: 100,
            margin_requirement: 0.2, // 20%
        }
    }
}

impl ExecutionConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), ExecutionError> {
        if self.trading_fee < 0.0 {
            return Err(ExecutionError::InvalidConfig(
                "Trading fee must be non-negative".to_string(),
            ));
        }
        if self.slippage < 0.0 || self.slippage > 1.0 {
            return Err(ExecutionError::InvalidConfig(
                "Slippage must be between 0 and 1".to_string(),
            ));
        }
        if self.max_position_size == 0 {
            return Err(ExecutionError::InvalidConfig(
                "Max position size must be positive".to_string(),
            ));
        }
        if self.margin_requirement < 0.0 || self.margin_requirement > 1.0 {
            return Err(ExecutionError::InvalidConfig(
                "Margin requirement must be between 0 and 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Execution Engine
///
/// Coordinates position management, P&L tracking, and expiration handling.
pub struct ExecutionEngine {
    position_manager: PositionManager,
    pnl_tracker: PnLTracker,
    config: ExecutionConfig,
}

impl ExecutionEngine {
    /// Create new execution engine
    pub fn new(initial_capital: f64, config: ExecutionConfig) -> Result<Self, ExecutionError> {
        config.validate()?;

        Ok(Self {
            position_manager: PositionManager::new(initial_capital),
            pnl_tracker: PnLTracker::new(initial_capital),
            config,
        })
    }

    /// Execute option signals
    ///
    /// Converts signals into actual trades with fees and slippage.
    pub fn execute_signals(
        &mut self,
        signals: &[OptionSignal],
        market_data: &MarketData,
    ) -> Result<Vec<Trade>, ExecutionError> {
        let mut trades = Vec::new();

        for signal in signals {
            match self.execute_signal(signal, market_data) {
                Ok(Some(trade)) => trades.push(trade),
                Ok(None) => {} // Signal filtered out
                Err(e) => {
                    // Log error but continue with other signals
                    eprintln!("Failed to execute signal: {}", e);
                }
            }
        }

        Ok(trades)
    }

    /// Execute single signal
    fn execute_signal(
        &mut self,
        signal: &OptionSignal,
        market_data: &MarketData,
    ) -> Result<Option<Trade>, ExecutionError> {
        // Check position limit
        if self.position_manager.position_count() >= self.config.max_position_size {
            return Err(ExecutionError::MaxPositionSizeExceeded(
                self.position_manager.position_count(),
                self.config.max_position_size,
            ));
        }

        match signal.signal_type {
            SignalType::OpenLong | SignalType::OpenShort => {
                self.execute_open_signal(signal, market_data)
            }
            SignalType::Close => self.execute_close_signal(signal, market_data),
            SignalType::Adjust => {
                // For adjust, treat as close + open
                // For simplicity, not implemented in Phase 4
                Ok(None)
            }
        }
    }

    /// Execute open signal (long or short)
    fn execute_open_signal(
        &mut self,
        signal: &OptionSignal,
        market_data: &MarketData,
    ) -> Result<Option<Trade>, ExecutionError> {
        // Get execution price with slippage
        let base_price = market_data.underlying_price * 0.05; // Simplified: 5% of underlying
        let execution_price = self.apply_slippage(base_price, signal.quantity > 0);

        let quantity = if signal.signal_type == SignalType::OpenShort {
            -signal.quantity.abs()
        } else {
            signal.quantity.abs()
        };

        // Calculate fee
        let fee = self.config.trading_fee * quantity.abs() as f64;
        let total_cost = execution_price * (quantity.abs() as f64) * 100.0 + fee;

        // Open position
        let position_id = self.position_manager.open_position(
            signal.option_type,
            signal.strike,
            signal.expiration,
            quantity,
            execution_price,
            market_data.timestamp,
            fee,
        )?;

        // Create trade record
        let trade = Trade {
            trade_id: format!("trade_{}", market_data.timestamp),
            position_id,
            timestamp: market_data.timestamp,
            option_type: signal.option_type,
            strike: signal.strike,
            expiration: signal.expiration,
            quantity,
            price: execution_price,
            fee,
            total_cost,
            realized_pnl: None,
        };

        self.pnl_tracker.record_trade(trade.clone());

        Ok(Some(trade))
    }

    /// Execute close signal
    fn execute_close_signal(
        &mut self,
        signal: &OptionSignal,
        market_data: &MarketData,
    ) -> Result<Option<Trade>, ExecutionError> {
        // Find matching position to close
        let position_id = self.find_position_to_close(signal)?;

        // Get execution price with slippage
        let base_price = market_data.underlying_price * 0.05; // Simplified

        // Extract needed position data before mutable borrow
        let (quantity, option_type, strike, expiration) = {
            let position = self
                .position_manager
                .get_position(&position_id)
                .ok_or_else(|| ExecutionError::PositionNotFound(position_id.clone()))?;
            (
                position.quantity,
                position.option_type,
                position.strike,
                position.expiration,
            )
        };

        let execution_price = self.apply_slippage(base_price, quantity < 0);

        // Calculate fee
        let fee = self.config.trading_fee * quantity.abs() as f64;

        // Close position
        let realized_pnl = self.position_manager.close_position(
            &position_id,
            execution_price,
            market_data.timestamp,
            fee,
        )?;

        // Create trade record
        let trade = Trade {
            trade_id: format!("trade_{}", market_data.timestamp),
            position_id,
            timestamp: market_data.timestamp,
            option_type,
            strike,
            expiration,
            quantity: -quantity, // Opposite of original position
            price: execution_price,
            fee,
            total_cost: execution_price * (quantity.abs() as f64) * 100.0 + fee,
            realized_pnl: Some(realized_pnl),
        };

        self.pnl_tracker.record_trade(trade.clone());

        Ok(Some(trade))
    }

    /// Find position to close based on signal
    fn find_position_to_close(&self, signal: &OptionSignal) -> Result<String, ExecutionError> {
        // Find first matching position
        for (position_id, position) in self.position_manager.positions() {
            if position.option_type == signal.option_type
                && position.strike == signal.strike
                && position.expiration == signal.expiration
            {
                return Ok(position_id.clone());
            }
        }

        Err(ExecutionError::PositionNotFound(format!(
            "{:?} {} @ {}",
            signal.option_type, signal.strike, signal.expiration
        )))
    }

    /// Apply slippage to price
    fn apply_slippage(&self, price: f64, is_buy: bool) -> f64 {
        if is_buy {
            price * (1.0 + self.config.slippage)
        } else {
            price * (1.0 - self.config.slippage)
        }
    }

    /// Process single time step
    ///
    /// Updates positions, checks expirations, and updates P&L.
    pub fn process_time_step(
        &mut self,
        current_time: i64,
        market_data: &MarketData,
    ) -> Result<TimeStepResult, ExecutionError> {
        let start = Instant::now();

        // Update positions with current market data
        self.position_manager.update_positions(market_data);

        // Check for expirations
        let expirations = self
            .position_manager
            .handle_expirations(current_time, market_data.underlying_price);

        // Record expiration events
        for (position_id, settlement) in &expirations {
            // Record as closed trade
            let trade = Trade {
                trade_id: format!("expiration_{}", current_time),
                position_id: position_id.clone(),
                timestamp: current_time,
                option_type: OptionType::Call, // Placeholder
                strike: 0.0,
                expiration: current_time,
                quantity: 0,
                price: 0.0,
                fee: 0.0,
                total_cost: 0.0,
                realized_pnl: Some(*settlement),
            };
            self.pnl_tracker.record_trade(trade);
        }

        // Update unrealized P&L
        let positions: Vec<_> = self
            .position_manager
            .positions()
            .values()
            .cloned()
            .collect();
        let current_equity = self.position_manager.equity();
        self.pnl_tracker
            .update_unrealized_pnl(&positions, current_equity);

        let duration = start.elapsed();

        Ok(TimeStepResult {
            timestamp: current_time,
            expirations: expirations.len(),
            current_equity,
            unrealized_pnl: self.pnl_tracker.unrealized_pnl(),
            realized_pnl: self.pnl_tracker.realized_pnl(),
            position_count: self.position_manager.position_count(),
            processing_time_us: duration.as_micros() as u64,
        })
    }

    /// Get execution report
    pub fn get_execution_report(&self) -> ExecutionReport {
        let metrics = self.pnl_tracker.get_metrics();

        ExecutionReport {
            initial_capital: self.position_manager.initial_capital(),
            current_equity: self.position_manager.equity(),
            cash: self.position_manager.cash(),
            metrics,
            position_count: self.position_manager.position_count(),
            total_trades: self.pnl_tracker.trade_count(),
        }
    }

    /// Get position manager reference
    pub fn position_manager(&self) -> &PositionManager {
        &self.position_manager
    }

    /// Get P&L tracker reference
    pub fn pnl_tracker(&self) -> &PnLTracker {
        &self.pnl_tracker
    }
}

/// Time step processing result
#[derive(Debug, Clone)]
pub struct TimeStepResult {
    pub timestamp: i64,
    pub expirations: usize,
    pub current_equity: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub position_count: usize,
    pub processing_time_us: u64,
}

/// Execution report
#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub initial_capital: f64,
    pub current_equity: f64,
    pub cash: f64,
    pub metrics: PerformanceMetrics,
    pub position_count: usize,
    pub total_trades: usize,
}

impl ExecutionReport {
    /// Format report as human-readable string
    pub fn to_string(&self) -> String {
        format!(
            "Execution Report:\n\
             ================\n\
             Initial Capital: ${:.2}\n\
             Current Equity: ${:.2}\n\
             Cash: ${:.2}\n\
             Active Positions: {}\n\
             Total Trades: {}\n\
             \n\
             {}",
            self.initial_capital,
            self.current_equity,
            self.cash,
            self.position_count,
            self.total_trades,
            self.metrics.to_string()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_market_data(timestamp: i64, underlying_price: f64) -> MarketData {
        MarketData {
            underlying_price,
            option_prices: Default::default(),
            option_greeks: Default::default(),
            timestamp,
        }
    }

    fn create_test_signal(
        signal_type: SignalType,
        option_type: OptionType,
        strike: f64,
        quantity: i32,
    ) -> OptionSignal {
        OptionSignal {
            option_type,
            strike,
            expiration: 1735689600,
            signal_type,
            quantity,
            strength: 0.8,
        }
    }

    #[test]
    fn test_engine_creation() {
        let config = ExecutionConfig::default();
        let engine = ExecutionEngine::new(10_000.0, config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let mut config = ExecutionConfig::default();
        config.slippage = 1.5; // Invalid

        let result = ExecutionEngine::new(10_000.0, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_open_long() {
        let config = ExecutionConfig::default();
        let mut engine = ExecutionEngine::new(10_000.0, config).unwrap();

        let signal = create_test_signal(SignalType::OpenLong, OptionType::Call, 100.0, 1);
        let market_data = create_test_market_data(1735000000, 100.0);

        let trades = engine.execute_signals(&[signal], &market_data).unwrap();

        assert_eq!(trades.len(), 1);
        assert!(trades[0].is_opening());
        assert_eq!(engine.position_manager().position_count(), 1);
    }

    #[test]
    fn test_execute_close() {
        let config = ExecutionConfig::default();
        let mut engine = ExecutionEngine::new(10_000.0, config).unwrap();

        // Open position
        let open_signal = create_test_signal(SignalType::OpenLong, OptionType::Call, 100.0, 1);
        let market_data = create_test_market_data(1735000000, 100.0);
        engine
            .execute_signals(&[open_signal], &market_data)
            .unwrap();

        // Close position
        let close_signal = create_test_signal(SignalType::Close, OptionType::Call, 100.0, 1);
        let market_data = create_test_market_data(1735100000, 105.0);
        let trades = engine
            .execute_signals(&[close_signal], &market_data)
            .unwrap();

        assert_eq!(trades.len(), 1);
        assert!(trades[0].is_closing());
        assert_eq!(engine.position_manager().position_count(), 0);
    }

    #[test]
    fn test_max_position_limit() {
        let mut config = ExecutionConfig::default();
        config.max_position_size = 2;

        let mut engine = ExecutionEngine::new(10_000.0, config).unwrap();
        let market_data = create_test_market_data(1735000000, 100.0);

        // Open 2 positions (should succeed)
        let signal1 = create_test_signal(SignalType::OpenLong, OptionType::Call, 100.0, 1);
        let signal2 = create_test_signal(SignalType::OpenLong, OptionType::Call, 110.0, 1);

        engine
            .execute_signals(&[signal1, signal2], &market_data)
            .unwrap();
        assert_eq!(engine.position_manager().position_count(), 2);

        // Try to open 3rd position (should fail)
        let signal3 = create_test_signal(SignalType::OpenLong, OptionType::Call, 120.0, 1);
        let _result = engine.execute_signals(&[signal3], &market_data);

        // Should still have 2 positions (3rd failed)
        assert_eq!(engine.position_manager().position_count(), 2);
    }

    #[test]
    fn test_process_time_step() {
        let config = ExecutionConfig::default();
        let mut engine = ExecutionEngine::new(10_000.0, config).unwrap();

        // Open position
        let signal = create_test_signal(SignalType::OpenLong, OptionType::Call, 100.0, 1);
        let market_data = create_test_market_data(1735000000, 100.0);
        engine.execute_signals(&[signal], &market_data).unwrap();

        // Process time step
        let market_data = create_test_market_data(1735100000, 105.0);
        let result = engine.process_time_step(1735100000, &market_data).unwrap();

        assert_eq!(result.position_count, 1);
        assert!(result.processing_time_us > 0);
    }

    #[test]
    fn test_expiration_handling() {
        let config = ExecutionConfig::default();
        let mut engine = ExecutionEngine::new(10_000.0, config).unwrap();

        // Open position with near expiration
        let signal = create_test_signal(SignalType::OpenLong, OptionType::Call, 100.0, 1);
        let market_data = create_test_market_data(1735000000, 100.0);
        engine.execute_signals(&[signal], &market_data).unwrap();

        // Process past expiration
        let market_data = create_test_market_data(1735700000, 110.0); // Past expiration
        let result = engine.process_time_step(1735700000, &market_data).unwrap();

        assert_eq!(result.expirations, 1);
        assert_eq!(result.position_count, 0);
    }

    #[test]
    fn test_execution_report() {
        let config = ExecutionConfig::default();
        let mut engine = ExecutionEngine::new(10_000.0, config).unwrap();

        // Execute some trades
        let signal = create_test_signal(SignalType::OpenLong, OptionType::Call, 100.0, 1);
        let market_data = create_test_market_data(1735000000, 100.0);
        engine.execute_signals(&[signal], &market_data).unwrap();

        let report = engine.get_execution_report();

        assert_eq!(report.initial_capital, 10_000.0);
        assert_eq!(report.position_count, 1);
        assert_eq!(report.total_trades, 1);
    }

    #[test]
    fn test_slippage_application() {
        let mut config = ExecutionConfig::default();
        config.slippage = 0.01; // 1%

        let engine = ExecutionEngine::new(10_000.0, config).unwrap();

        let buy_price = engine.apply_slippage(100.0, true);
        assert_eq!(buy_price, 101.0);

        let sell_price = engine.apply_slippage(100.0, false);
        assert_eq!(sell_price, 99.0);
    }
}
