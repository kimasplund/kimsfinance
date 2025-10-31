//! Performance metrics (Sharpe ratio, max drawdown, win rate, etc.)

use crate::strategy::types::*;
use chrono::NaiveDate;

/// Performance metrics for a backtest
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_pnl: f64,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub profit_factor: f64,
    pub max_consecutive_losses: i32,
    pub avg_days_in_trade: f64,
    pub return_on_capital: f64,
}

impl PerformanceMetrics {
    /// Calculate all performance metrics from closed positions and daily capital
    pub fn calculate(
        positions: &[OptionsPosition],
        daily_capital: &[(NaiveDate, f64)],
        initial_capital: f64,
    ) -> Self {
        // Calculate P&L for each position
        let mut pnls: Vec<f64> = Vec::new();
        let mut wins: Vec<f64> = Vec::new();
        let mut losses: Vec<f64> = Vec::new();
        let mut days_in_trade: Vec<f64> = Vec::new();

        for position in positions {
            let mut pnl = 0.0;

            // Calculate P&L for each leg
            for leg in &position.legs {
                let entry_price = leg.entry_price;
                let exit_price = leg.exit_price.unwrap_or(leg.entry_price);

                match leg.side {
                    PositionSide::Short => {
                        pnl += (entry_price - exit_price) * 100.0; // per 100 shares
                    }
                    PositionSide::Long => {
                        pnl += (exit_price - entry_price) * 100.0;
                    }
                }
            }

            pnls.push(pnl);

            if pnl > 0.0 {
                wins.push(pnl);
            } else {
                losses.push(pnl.abs());
            }

            // Days in trade
            if let Some(exit_date) = position.exit_date {
                let days = (exit_date - position.entry_date).num_days() as f64;
                days_in_trade.push(days);
            }
        }

        // Total P&L
        let total_pnl: f64 = pnls.iter().sum();

        // Win rate
        let num_wins = wins.len();
        let num_trades = positions.len();
        let win_rate = if num_trades > 0 {
            (num_wins as f64 / num_trades as f64) * 100.0
        } else {
            0.0
        };

        // Average win/loss
        let avg_win = if !wins.is_empty() {
            wins.iter().sum::<f64>() / wins.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losses.is_empty() {
            losses.iter().sum::<f64>() / losses.len() as f64
        } else {
            0.0
        };

        // Profit factor
        let total_wins: f64 = wins.iter().sum();
        let total_losses: f64 = losses.iter().sum();
        let profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else if total_wins > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Max consecutive losses
        let mut max_consecutive_losses = 0;
        let mut current_consecutive = 0;
        for pnl in &pnls {
            if *pnl < 0.0 {
                current_consecutive += 1;
                max_consecutive_losses = max_consecutive_losses.max(current_consecutive);
            } else {
                current_consecutive = 0;
            }
        }

        // Average days in trade
        let avg_days_in_trade = if !days_in_trade.is_empty() {
            days_in_trade.iter().sum::<f64>() / days_in_trade.len() as f64
        } else {
            0.0
        };

        // Max drawdown from daily capital
        let max_drawdown = Self::calculate_max_drawdown(daily_capital);

        // Sharpe ratio from daily returns
        let sharpe_ratio = Self::calculate_sharpe_ratio(daily_capital, initial_capital);

        // Sortino ratio from daily returns
        let sortino_ratio = Self::calculate_sortino_ratio(daily_capital, initial_capital);

        // Return on capital
        let return_on_capital = if initial_capital > 0.0 {
            (total_pnl / initial_capital) * 100.0
        } else {
            0.0
        };

        Self {
            total_pnl,
            win_rate,
            avg_win,
            avg_loss,
            max_drawdown,
            sharpe_ratio,
            sortino_ratio,
            profit_factor,
            max_consecutive_losses,
            avg_days_in_trade,
            return_on_capital,
        }
    }

    /// Calculate maximum drawdown from daily capital
    fn calculate_max_drawdown(daily_capital: &[(NaiveDate, f64)]) -> f64 {
        if daily_capital.is_empty() {
            return 0.0;
        }

        let mut max_capital: f64 = daily_capital[0].1;
        let mut max_drawdown: f64 = 0.0;

        for (_, capital) in daily_capital {
            max_capital = max_capital.max(*capital);
            let drawdown: f64 = max_capital - capital;
            max_drawdown = max_drawdown.max(drawdown);
        }

        max_drawdown
    }

    /// Calculate Sharpe ratio from daily returns
    ///
    /// Sharpe = (mean return - risk-free rate) / std dev of returns
    /// We assume risk-free rate = 0 for simplicity
    fn calculate_sharpe_ratio(daily_capital: &[(NaiveDate, f64)], _initial_capital: f64) -> f64 {
        if daily_capital.len() < 2 {
            return 0.0;
        }

        // Calculate daily returns
        let mut returns: Vec<f64> = Vec::new();
        for i in 1..daily_capital.len() {
            let prev_capital = daily_capital[i - 1].1;
            let curr_capital = daily_capital[i].1;

            if prev_capital > 0.0 {
                let ret = (curr_capital - prev_capital) / prev_capital;
                returns.push(ret);
            }
        }

        if returns.is_empty() {
            return 0.0;
        }

        // Mean return
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

        // Standard deviation
        let variance = returns
            .iter()
            .map(|r| {
                let diff = r - mean_return;
                diff * diff
            })
            .sum::<f64>()
            / returns.len() as f64;

        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            // Annualize (252 trading days)
            mean_return * (252.0_f64).sqrt() / std_dev
        } else {
            0.0
        }
    }

    /// Calculate Sortino ratio from daily returns
    ///
    /// Sortino = (mean return - risk-free rate) / downside deviation
    /// Only considers downside volatility (negative returns)
    fn calculate_sortino_ratio(daily_capital: &[(NaiveDate, f64)], _initial_capital: f64) -> f64 {
        if daily_capital.len() < 2 {
            return 0.0;
        }

        // Calculate daily returns
        let mut returns: Vec<f64> = Vec::new();
        for i in 1..daily_capital.len() {
            let prev_capital = daily_capital[i - 1].1;
            let curr_capital = daily_capital[i].1;

            if prev_capital > 0.0 {
                let ret = (curr_capital - prev_capital) / prev_capital;
                returns.push(ret);
            }
        }

        if returns.is_empty() {
            return 0.0;
        }

        // Mean return
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

        // Downside deviation (only negative returns)
        let downside_returns: Vec<f64> = returns.iter().filter(|r| **r < 0.0).copied().collect();

        if downside_returns.is_empty() {
            return f64::INFINITY; // No downside = infinite Sortino
        }

        let downside_variance =
            downside_returns.iter().map(|r| r * r).sum::<f64>() / downside_returns.len() as f64;

        let downside_dev = downside_variance.sqrt();

        if downside_dev > 0.0 {
            // Annualize (252 trading days)
            mean_return * (252.0_f64).sqrt() / downside_dev
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn test_max_drawdown() {
        let daily_capital = vec![
            (NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(), 10000.0),
            (NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(), 11000.0),
            (NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(), 10500.0),
            (NaiveDate::from_ymd_opt(2024, 1, 4).unwrap(), 9000.0),
            (NaiveDate::from_ymd_opt(2024, 1, 5).unwrap(), 9500.0),
        ];

        let max_dd = PerformanceMetrics::calculate_max_drawdown(&daily_capital);
        assert!((max_dd - 2000.0).abs() < 0.01); // 11000 - 9000 = 2000
    }

    #[test]
    fn test_win_rate() {
        let positions = vec![
            create_test_position(100.0), // Win
            create_test_position(-50.0), // Loss
            create_test_position(75.0),  // Win
        ];

        let metrics = PerformanceMetrics::calculate(&positions, &[], 10000.0);
        assert!((metrics.win_rate - 66.67).abs() < 0.1); // 2/3 = 66.67%
    }

    fn create_test_position(pnl: f64) -> OptionsPosition {
        let entry_date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let exit_date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();

        let leg = OptionLeg {
            contract: OptionContract {
                symbol: "TEST".to_string(),
                contract_symbol: "TEST_CONTRACT".to_string(),
                strike: 100.0,
                expiration: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                option_type: OptionType::Put,
                snapshot_date: entry_date,
                bid: 1.0,
                ask: 1.1,
                last_price: 1.05,
                volume: 100.0,
                open_interest: 1000.0,
                delta: Some(-0.25),
                gamma: None,
                theta: None,
                vega: None,
                rho: None,
                implied_volatility: Some(0.2),
                dte: 30,
            },
            side: PositionSide::Short,
            quantity: 1,
            entry_price: 1.05,
            exit_price: Some(1.05 - (pnl / 100.0)),
            entry_date,
            exit_date: Some(exit_date),
        };

        OptionsPosition {
            id: "TEST_POS".to_string(),
            strategy: "TEST".to_string(),
            legs: vec![leg],
            entry_date,
            exit_date: Some(exit_date),
            max_profit: Some(pnl),
            max_loss: Some(-pnl),
        }
    }
}
