//! P&L Tracker
//!
//! Tracks realized and unrealized P&L with performance metrics.

use super::Trade;
use super::position_manager::OptionPosition;
use std::collections::HashMap;

/// P&L Tracker
///
/// Maintains comprehensive profit & loss records with risk-adjusted metrics.
pub struct PnLTracker {
    /// All trades executed
    trades: Vec<Trade>,

    /// Daily P&L by date (timestamp -> P&L)
    daily_pnl: HashMap<i64, f64>,

    /// Cumulative P&L over time
    cumulative_pnl: f64,

    /// Realized P&L (closed positions)
    realized_pnl: f64,

    /// Unrealized P&L (open positions)
    unrealized_pnl: f64,

    /// Peak equity reached
    peak_capital: f64,

    /// Maximum drawdown experienced
    max_drawdown: f64,

    /// Initial capital
    initial_capital: f64,

    /// Number of winning trades
    winning_trades: usize,

    /// Number of losing trades
    losing_trades: usize,

    /// Total profit from winning trades
    gross_profit: f64,

    /// Total loss from losing trades
    gross_loss: f64,
}

impl PnLTracker {
    /// Create new P&L tracker
    pub fn new(initial_capital: f64) -> Self {
        Self {
            trades: Vec::new(),
            daily_pnl: HashMap::new(),
            cumulative_pnl: 0.0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            peak_capital: initial_capital,
            max_drawdown: 0.0,
            initial_capital,
            winning_trades: 0,
            losing_trades: 0,
            gross_profit: 0.0,
            gross_loss: 0.0,
        }
    }

    /// Record new trade
    pub fn record_trade(&mut self, trade: Trade) {
        // Update realized P&L if closing trade
        if let Some(pnl) = trade.realized_pnl {
            self.realized_pnl += pnl;
            self.cumulative_pnl += pnl;

            // Track winning/losing trades
            if pnl > 0.0 {
                self.winning_trades += 1;
                self.gross_profit += pnl;
            } else if pnl < 0.0 {
                self.losing_trades += 1;
                self.gross_loss += pnl.abs();
            }

            // Update daily P&L
            let day = self.timestamp_to_day(trade.timestamp);
            *self.daily_pnl.entry(day).or_insert(0.0) += pnl;
        }

        self.trades.push(trade);
    }

    /// Update unrealized P&L from current positions
    pub fn update_unrealized_pnl(&mut self, positions: &[OptionPosition], current_equity: f64) {
        self.unrealized_pnl = positions.iter().map(|pos| pos.pnl).sum();

        // Update peak and drawdown
        if current_equity > self.peak_capital {
            self.peak_capital = current_equity;
        }

        let current_drawdown = (self.peak_capital - current_equity) / self.peak_capital;
        if current_drawdown > self.max_drawdown {
            self.max_drawdown = current_drawdown;
        }
    }

    /// Calculate daily P&L for specific date
    pub fn calculate_daily_pnl(&self, date: i64) -> f64 {
        let day = self.timestamp_to_day(date);
        self.daily_pnl.get(&day).copied().unwrap_or(0.0)
    }

    /// Get Sharpe ratio (annualized)
    ///
    /// Measures risk-adjusted return.
    /// Formula: (Mean Return - Risk-Free Rate) / Std Dev of Returns
    pub fn get_sharpe_ratio(&self) -> f64 {
        if self.daily_pnl.is_empty() {
            return 0.0;
        }

        let daily_returns: Vec<f64> = self.daily_pnl.values().copied().collect();

        let mean_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let variance = daily_returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / daily_returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        // Annualize: assume 252 trading days
        let annualized_return = mean_return * 252.0;
        let annualized_std_dev = std_dev * (252.0_f64).sqrt();

        annualized_return / annualized_std_dev
    }

    /// Get Sortino ratio (annualized)
    ///
    /// Like Sharpe but only penalizes downside volatility.
    /// Formula: (Mean Return - Risk-Free Rate) / Downside Deviation
    pub fn get_sortino_ratio(&self) -> f64 {
        if self.daily_pnl.is_empty() {
            return 0.0;
        }

        let daily_returns: Vec<f64> = self.daily_pnl.values().copied().collect();

        let mean_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;

        // Downside deviation: only negative returns
        let downside_variance = daily_returns
            .iter()
            .filter(|&&r| r < 0.0)
            .map(|r| r.powi(2))
            .sum::<f64>()
            / daily_returns.len() as f64;
        let downside_dev = downside_variance.sqrt();

        if downside_dev == 0.0 {
            return 0.0;
        }

        // Annualize
        let annualized_return = mean_return * 252.0;
        let annualized_downside_dev = downside_dev * (252.0_f64).sqrt();

        annualized_return / annualized_downside_dev
    }

    /// Get maximum drawdown (as decimal, e.g., 0.25 = 25%)
    pub fn get_max_drawdown(&self) -> f64 {
        self.max_drawdown
    }

    /// Get win rate (%)
    pub fn get_win_rate(&self) -> f64 {
        let total_closed = self.winning_trades + self.losing_trades;
        if total_closed == 0 {
            return 0.0;
        }
        (self.winning_trades as f64 / total_closed as f64) * 100.0
    }

    /// Get profit factor (gross profit / gross loss)
    pub fn get_profit_factor(&self) -> f64 {
        if self.gross_loss == 0.0 {
            if self.gross_profit > 0.0 {
                return f64::INFINITY;
            }
            return 0.0;
        }
        self.gross_profit / self.gross_loss
    }

    /// Get total return (%)
    pub fn get_total_return(&self) -> f64 {
        if self.initial_capital == 0.0 {
            return 0.0;
        }
        (self.cumulative_pnl / self.initial_capital) * 100.0
    }

    /// Get realized P&L
    pub fn realized_pnl(&self) -> f64 {
        self.realized_pnl
    }

    /// Get unrealized P&L
    pub fn unrealized_pnl(&self) -> f64 {
        self.unrealized_pnl
    }

    /// Get total P&L (realized + unrealized)
    pub fn total_pnl(&self) -> f64 {
        self.realized_pnl + self.unrealized_pnl
    }

    /// Get number of trades
    pub fn trade_count(&self) -> usize {
        self.trades.len()
    }

    /// Get number of closed trades
    pub fn closed_trade_count(&self) -> usize {
        self.winning_trades + self.losing_trades
    }

    /// Get all trades
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            total_pnl: self.total_pnl(),
            realized_pnl: self.realized_pnl,
            unrealized_pnl: self.unrealized_pnl,
            total_return: self.get_total_return(),
            sharpe_ratio: self.get_sharpe_ratio(),
            sortino_ratio: self.get_sortino_ratio(),
            max_drawdown: self.max_drawdown,
            win_rate: self.get_win_rate(),
            profit_factor: self.get_profit_factor(),
            total_trades: self.trade_count(),
            winning_trades: self.winning_trades,
            losing_trades: self.losing_trades,
        }
    }

    /// Convert timestamp to day (strip time component)
    fn timestamp_to_day(&self, timestamp: i64) -> i64 {
        // Round down to start of day (86400 seconds per day)
        (timestamp / 86400) * 86400
    }
}

/// Performance metrics summary
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_pnl: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
}

impl std::fmt::Display for PerformanceMetrics {
    /// Format metrics as human-readable string
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Performance Metrics:\n\
             Total P&L: ${:.2}\n\
             Realized P&L: ${:.2}\n\
             Unrealized P&L: ${:.2}\n\
             Total Return: {:.2}%\n\
             Sharpe Ratio: {:.2}\n\
             Sortino Ratio: {:.2}\n\
             Max Drawdown: {:.2}%\n\
             Win Rate: {:.1}%\n\
             Profit Factor: {:.2}\n\
             Total Trades: {} (W: {}, L: {})",
            self.total_pnl,
            self.realized_pnl,
            self.unrealized_pnl,
            self.total_return,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.win_rate,
            self.profit_factor,
            self.total_trades,
            self.winning_trades,
            self.losing_trades
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantitative::heston::OptionType;

    fn create_test_trade(realized_pnl: Option<f64>, timestamp: i64) -> Trade {
        Trade {
            trade_id: "test".to_string(),
            position_id: "pos1".to_string(),
            timestamp,
            option_type: OptionType::Call,
            strike: 100.0,
            expiration: 1735689600,
            quantity: 1,
            price: 5.0,
            fee: 1.0,
            total_cost: 501.0,
            realized_pnl,
        }
    }

    #[test]
    fn test_record_winning_trade() {
        let mut tracker = PnLTracker::new(10_000.0);

        let trade = create_test_trade(Some(200.0), 1735000000);
        tracker.record_trade(trade);

        assert_eq!(tracker.realized_pnl(), 200.0);
        assert_eq!(tracker.winning_trades, 1);
        assert_eq!(tracker.losing_trades, 0);
        assert_eq!(tracker.gross_profit, 200.0);
    }

    #[test]
    fn test_record_losing_trade() {
        let mut tracker = PnLTracker::new(10_000.0);

        let trade = create_test_trade(Some(-150.0), 1735000000);
        tracker.record_trade(trade);

        assert_eq!(tracker.realized_pnl(), -150.0);
        assert_eq!(tracker.winning_trades, 0);
        assert_eq!(tracker.losing_trades, 1);
        assert_eq!(tracker.gross_loss, 150.0);
    }

    #[test]
    fn test_win_rate() {
        let mut tracker = PnLTracker::new(10_000.0);

        tracker.record_trade(create_test_trade(Some(200.0), 1735000000));
        tracker.record_trade(create_test_trade(Some(-100.0), 1735100000));
        tracker.record_trade(create_test_trade(Some(150.0), 1735200000));

        assert!((tracker.get_win_rate() - (200.0 / 3.0)).abs() < 0.01);
    }

    #[test]
    fn test_profit_factor() {
        let mut tracker = PnLTracker::new(10_000.0);

        tracker.record_trade(create_test_trade(Some(300.0), 1735000000));
        tracker.record_trade(create_test_trade(Some(-100.0), 1735100000));

        assert!((tracker.get_profit_factor() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_total_return() {
        let mut tracker = PnLTracker::new(10_000.0);

        tracker.record_trade(create_test_trade(Some(1000.0), 1735000000));

        assert_eq!(tracker.get_total_return(), 10.0); // 1000/10000 * 100
    }

    #[test]
    fn test_sharpe_ratio() {
        let mut tracker = PnLTracker::new(10_000.0);

        // Add trades with varying returns
        tracker.record_trade(create_test_trade(Some(100.0), 1735000000));
        tracker.record_trade(create_test_trade(Some(150.0), 1735086400));
        tracker.record_trade(create_test_trade(Some(-50.0), 1735172800));
        tracker.record_trade(create_test_trade(Some(200.0), 1735259200));

        let sharpe = tracker.get_sharpe_ratio();
        assert!(sharpe > 0.0); // Positive returns should have positive Sharpe
    }

    #[test]
    fn test_sortino_ratio() {
        let mut tracker = PnLTracker::new(10_000.0);

        // Add trades with varying returns
        tracker.record_trade(create_test_trade(Some(100.0), 1735000000));
        tracker.record_trade(create_test_trade(Some(150.0), 1735086400));
        tracker.record_trade(create_test_trade(Some(-50.0), 1735172800));

        let sortino = tracker.get_sortino_ratio();
        assert!(sortino > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let mut tracker = PnLTracker::new(10_000.0);

        // Simulate equity curve
        tracker.update_unrealized_pnl(&[], 11_000.0); // Peak
        tracker.update_unrealized_pnl(&[], 9_000.0); // Drawdown

        let dd = tracker.get_max_drawdown();
        assert!((dd - (2000.0 / 11000.0)).abs() < 0.01);
    }

    #[test]
    fn test_daily_pnl() {
        let mut tracker = PnLTracker::new(10_000.0);

        // Same day trades
        tracker.record_trade(create_test_trade(Some(100.0), 1735000000));
        tracker.record_trade(create_test_trade(Some(50.0), 1735001000));

        // Different day
        tracker.record_trade(create_test_trade(Some(75.0), 1735100000));

        let day1_pnl = tracker.calculate_daily_pnl(1735000000);
        assert_eq!(day1_pnl, 150.0);

        let day2_pnl = tracker.calculate_daily_pnl(1735100000);
        assert_eq!(day2_pnl, 75.0);
    }

    #[test]
    fn test_unrealized_pnl() {
        let mut tracker = PnLTracker::new(10_000.0);

        let mut position = OptionPosition::new(
            "pos1".to_string(),
            OptionType::Call,
            100.0,
            1735689600,
            1,
            5.0,
            1735000000,
        );

        position.pnl = 250.0;

        tracker.update_unrealized_pnl(&[position], 10_250.0);

        assert_eq!(tracker.unrealized_pnl(), 250.0);
        assert_eq!(tracker.total_pnl(), 250.0);
    }

    #[test]
    fn test_performance_metrics() {
        let mut tracker = PnLTracker::new(10_000.0);

        tracker.record_trade(create_test_trade(Some(500.0), 1735000000));
        tracker.record_trade(create_test_trade(Some(-200.0), 1735100000));

        let metrics = tracker.get_metrics();

        assert_eq!(metrics.total_pnl, 300.0);
        assert_eq!(metrics.realized_pnl, 300.0);
        assert_eq!(metrics.winning_trades, 1);
        assert_eq!(metrics.losing_trades, 1);
        assert!((metrics.win_rate - 50.0).abs() < 0.01);
    }
}
