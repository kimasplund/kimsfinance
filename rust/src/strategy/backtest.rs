//! Backtesting engine for options strategies

use crate::strategy::data_loader::*;
use crate::strategy::market_regime::*;
use crate::strategy::metrics::*;
use crate::strategy::spot_data::*;
use crate::strategy::strategies::*;
use crate::strategy::transaction_costs::*;
use crate::strategy::types::*;
use chrono::NaiveDate;
use std::collections::HashMap;

/// Backtest engine for options strategies
pub struct BacktestEngine {
    data_loader: OptionsDataLoader,
    spot_loader: SpotDataLoader,
    initial_capital: f64,
    /// Maximum risk per trade as percentage of capital (default: 5.0%)
    pub max_risk_per_trade_pct: f64,
    /// Maximum concurrent positions (default: 10)
    pub max_concurrent_positions: usize,
    /// Maximum margin utilization as percentage of capital (default: 50.0%)
    pub max_margin_utilization_pct: f64,
}

impl BacktestEngine {
    /// Create new backtest engine with default risk limits
    pub fn new(
        data_loader: OptionsDataLoader,
        spot_loader: SpotDataLoader,
        initial_capital: f64,
    ) -> Self {
        Self {
            data_loader,
            spot_loader,
            initial_capital,
            max_risk_per_trade_pct: 5.0,        // 5% max risk per trade
            max_concurrent_positions: 10,        // Max 10 open positions
            max_margin_utilization_pct: 50.0,   // Use max 50% of capital as margin
        }
    }

    /// Create new backtest engine with custom risk limits
    pub fn new_with_limits(
        data_loader: OptionsDataLoader,
        spot_loader: SpotDataLoader,
        initial_capital: f64,
        max_risk_per_trade_pct: f64,
        max_concurrent_positions: usize,
        max_margin_utilization_pct: f64,
    ) -> Self {
        Self {
            data_loader,
            spot_loader,
            initial_capital,
            max_risk_per_trade_pct,
            max_concurrent_positions,
            max_margin_utilization_pct,
        }
    }

    /// Get spot price for a symbol on a date
    pub fn get_spot_price(&mut self, symbol: &str, date: NaiveDate) -> Result<f64, SpotDataError> {
        self.spot_loader.get_spot_price(symbol, date)
    }

    /// Get 20-day ATR for a symbol on a date
    pub fn get_atr(&mut self, symbol: &str, date: NaiveDate) -> Result<f64, SpotDataError> {
        self.spot_loader.calculate_atr(symbol, date)
    }

    /// Get Bollinger Band width for a symbol on a date
    pub fn get_bollinger_width(
        &mut self,
        symbol: &str,
        date: NaiveDate,
    ) -> Result<f64, SpotDataError> {
        let (_, _, width) = self
            .spot_loader
            .calculate_bollinger_bands(symbol, date, 2.0)?;
        Ok(width)
    }

    /// Calculate required margin for a position
    /// For vertical spreads: margin = (short_strike - long_strike) * 100 per contract
    pub fn calculate_required_margin(&self, position: &OptionsPosition) -> f64 {
        // For bull put spread: margin = width of spread
        // Short leg is at higher strike, long leg at lower strike
        if position.legs.len() == 2 {
            let short_leg = position.legs.iter().find(|l| l.side == PositionSide::Short);
            let long_leg = position.legs.iter().find(|l| l.side == PositionSide::Long);

            if let (Some(short), Some(long)) = (short_leg, long_leg) {
                let width = (short.contract.strike - long.contract.strike).abs();
                return width * 100.0 * short.quantity as f64; // Per 100 shares
            }
        }

        // Fallback: use max loss if available
        position.max_loss.map(|ml| ml.abs()).unwrap_or(0.0)
    }

    /// Get available margin (capital - used margin)
    pub fn get_available_margin(&self, current_capital: f64, positions: &[OptionsPosition]) -> f64 {
        let used_margin: f64 = positions
            .iter()
            .map(|p| self.calculate_required_margin(p))
            .sum();

        current_capital - used_margin
    }

    /// Get current margin utilization percentage
    pub fn get_margin_utilization_pct(&self, current_capital: f64, positions: &[OptionsPosition]) -> f64 {
        let used_margin: f64 = positions
            .iter()
            .map(|p| self.calculate_required_margin(p))
            .sum();

        if current_capital > 0.0 {
            (used_margin / current_capital) * 100.0
        } else {
            0.0
        }
    }

    /// Check if we can enter a new position given risk limits
    fn can_enter_position(
        &self,
        required_margin: f64,
        max_risk: f64,
        current_capital: f64,
        positions: &[OptionsPosition],
    ) -> (bool, Option<String>) {
        // Check 1: Maximum concurrent positions
        if positions.len() >= self.max_concurrent_positions {
            return (
                false,
                Some(format!(
                    "Max concurrent positions reached ({}/{})",
                    positions.len(),
                    self.max_concurrent_positions
                )),
            );
        }

        // Check 2: Margin utilization limit
        let used_margin: f64 = positions
            .iter()
            .map(|p| self.calculate_required_margin(p))
            .sum();
        let total_margin_after = used_margin + required_margin;
        let margin_util_after = (total_margin_after / self.initial_capital) * 100.0;

        if margin_util_after > self.max_margin_utilization_pct {
            return (
                false,
                Some(format!(
                    "Margin limit exceeded ({:.1}% > {:.1}%)",
                    margin_util_after, self.max_margin_utilization_pct
                )),
            );
        }

        // Check 3: Risk per trade limit
        let risk_pct = (max_risk / self.initial_capital) * 100.0;
        if risk_pct > self.max_risk_per_trade_pct {
            return (
                false,
                Some(format!(
                    "Risk per trade exceeded ({:.2}% > {:.2}%)",
                    risk_pct, self.max_risk_per_trade_pct
                )),
            );
        }

        // Check 4: Insufficient capital for margin
        if required_margin > current_capital {
            return (
                false,
                Some(format!(
                    "Insufficient capital (need ${:.2}, have ${:.2})",
                    required_margin, current_capital
                )),
            );
        }

        (true, None)
    }

    /// Run backtest for a bull put spread strategy
    pub fn run_bull_put_spread(
        &mut self,
        symbol: &str,
        strategy: &BullPutSpread,
        params: &StrategyParams,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<BacktestResult, DataLoaderError> {
        println!(
            "Running backtest: {} from {} to {}",
            symbol, start_date, end_date
        );

        let mut capital = self.initial_capital;
        let mut positions: Vec<OptionsPosition> = Vec::new();
        let mut closed_positions: Vec<OptionsPosition> = Vec::new();
        let mut daily_capital: Vec<(NaiveDate, f64)> = Vec::new();

        // Get all available trading dates in range
        let available_dates = self.data_loader.get_available_dates(symbol)?;
        let trading_dates: Vec<_> = available_dates
            .into_iter()
            .filter(|d| *d >= start_date && *d <= end_date)
            .collect();

        println!("  Trading days: {}", trading_dates.len());

        // Create transaction cost model from params
        let cost_model = TransactionCostModel::new_custom(
            params.commission_per_contract,
            0.50, // Leg fee (exchange fees)
            params.slippage_ticks,
            0.05, // Tick size
            params.apply_bid_ask_spread,
        );

        println!("\n  Transaction Costs:");
        println!(
            "    Commission: ${:.2}/contract",
            params.commission_per_contract
        );
        println!("    Leg fee: $0.50/leg");
        println!(
            "    Slippage: {} ticks (${:.2})",
            params.slippage_ticks,
            params.slippage_ticks * 0.05
        );
        println!(
            "    Bid-ask spread: {}",
            if params.apply_bid_ask_spread {
                "Enabled"
            } else {
                "Disabled"
            }
        );
        println!(
            "    Entry cost per spread: ${:.2}",
            cost_model.entry_cost(2)
        );
        println!("    Exit cost per spread: ${:.2}", cost_model.exit_cost(2));
        println!(
            "    Round trip cost: ${:.2}\n",
            cost_model.round_trip_cost(2)
        );

        // Walk forward through each trading day
        for current_date in trading_dates.iter() {
            // Load options chain for current date
            let contracts = match self.data_loader.load_chain(symbol, *current_date) {
                Ok(c) => c,
                Err(_) => {
                    // Skip if data not available
                    continue;
                }
            };

            // Build price map for exit checks
            let mut price_map: HashMap<String, (f64, f64)> = HashMap::new();
            for contract in &contracts {
                price_map.insert(
                    contract.contract_symbol.clone(),
                    (contract.bid, contract.ask),
                );
            }

            // Check existing positions for exits
            let mut positions_to_close: Vec<usize> = Vec::new();
            for (idx, position) in positions.iter().enumerate() {
                if let Some(reason) = strategy.should_close(position, *current_date, &price_map) {
                    positions_to_close.push(idx);

                    // Calculate P&L with realistic prices (bid/ask + slippage)
                    let mut pnl = 0.0;
                    for leg in &position.legs {
                        if let Some((bid, ask)) = price_map.get(&leg.contract.contract_symbol) {
                            // Use transaction cost model for exit prices
                            let exit_price = match leg.side {
                                PositionSide::Short => {
                                    // Closing short: buy at ask + slippage
                                    cost_model.exit_price(*bid, *ask, true)
                                }
                                PositionSide::Long => {
                                    // Closing long: sell at bid - slippage
                                    cost_model.exit_price(*bid, *ask, false)
                                }
                            };

                            match leg.side {
                                PositionSide::Short => {
                                    pnl += leg.entry_price - exit_price;
                                }
                                PositionSide::Long => {
                                    pnl += exit_price - leg.entry_price;
                                }
                            }
                        }
                    }

                    // Deduct exit transaction costs (commission + leg fees)
                    let exit_costs = cost_model.exit_cost(position.legs.len() as u32);
                    let pnl_after_costs = (pnl * 100.0) - exit_costs; // Options are per 100 shares

                    capital += pnl_after_costs;

                    if pnl_after_costs > 0.0 {
                        println!(
                            "  {} - CLOSE WIN: {} (${:.2} gross, ${:.2} net) - {}",
                            current_date,
                            position.id,
                            pnl * 100.0,
                            pnl_after_costs,
                            reason
                        );
                    } else {
                        println!(
                            "  {} - CLOSE LOSS: {} (${:.2} gross, ${:.2} net) - {}",
                            current_date,
                            position.id,
                            pnl * 100.0,
                            pnl_after_costs,
                            reason
                        );
                    }
                }
            }

            // Close positions (in reverse order to maintain indices)
            for idx in positions_to_close.iter().rev() {
                let mut position = positions.remove(*idx);
                position.exit_date = Some(*current_date);

                // Set exit prices on legs (using realistic prices with bid/ask + slippage)
                for leg in &mut position.legs {
                    if let Some((bid, ask)) = price_map.get(&leg.contract.contract_symbol) {
                        let exit_price = match leg.side {
                            PositionSide::Short => cost_model.exit_price(*bid, *ask, true),
                            PositionSide::Long => cost_model.exit_price(*bid, *ask, false),
                        };
                        leg.exit_price = Some(exit_price);
                        leg.exit_date = Some(*current_date);
                    }
                }

                closed_positions.push(position);
            }

            // Look for new entry opportunities
            let puts: Vec<_> = contracts
                .iter()
                .filter(|c| c.option_type == OptionType::Put)
                .cloned()
                .collect();

            // Only proceed with entry checks if we haven't hit max positions
            if positions.len() < self.max_concurrent_positions {

                // Debug: show first day stats
                if trading_dates.iter().position(|d| d == current_date) == Some(0) {
                    println!("  First day - Total puts: {}", puts.len());
                    let with_delta: Vec<_> = puts.iter().filter(|p| p.delta.is_some()).collect();
                    println!("  Puts with delta: {}", with_delta.len());
                    let liquid_strict: Vec<_> = puts
                        .iter()
                        .filter(|p| p.volume >= 10.0 && p.open_interest >= 100.0)
                        .collect();
                    println!("  Puts with volume>=10, OI>=100: {}", liquid_strict.len());
                    let liquid_relaxed: Vec<_> = puts
                        .iter()
                        .filter(|p| p.volume >= 1.0 && p.open_interest >= 10.0)
                        .collect();
                    println!("  Puts with volume>=1, OI>=10: {}", liquid_relaxed.len());
                }

                // Get real spot price from OHLCV data
                let spot_price = match self.spot_loader.get_spot_price(symbol, *current_date) {
                    Ok(price) => {
                        // Debug: show spot price vs ATM estimate on first day
                        if trading_dates.iter().position(|d| d == current_date) == Some(0) {
                            let atm_estimate = if let Some(put) = puts.iter().find(|p| {
                                p.delta.map(|d| d.abs()).unwrap_or(0.0) > 0.45
                                    && p.delta.map(|d| d.abs()).unwrap_or(0.0) < 0.55
                            }) {
                                put.strike
                            } else {
                                puts.iter().map(|p| p.strike).sum::<f64>() / puts.len() as f64
                            };
                            let diff_pct = ((price - atm_estimate) / atm_estimate * 100.0).abs();
                            println!("  Spot price (OHLCV): ${:.2}", price);
                            println!("  ATM estimate: ${:.2}", atm_estimate);
                            println!("  Difference: {:.2}%", diff_pct);
                        }
                        price
                    }
                    Err(_) => {
                        // Fallback to ATM approximation if spot data not available
                        if let Some(put) = puts.iter().find(|p| {
                            p.delta.map(|d| d.abs()).unwrap_or(0.0) > 0.45
                                && p.delta.map(|d| d.abs()).unwrap_or(0.0) < 0.55
                        }) {
                            put.strike
                        } else {
                            puts.iter().map(|p| p.strike).sum::<f64>() / puts.len() as f64
                        }
                    }
                };

                let candidates = strategy.find_candidates(&puts, spot_price);

                // Debug: log candidates
                if trading_dates.iter().position(|d| d == current_date) == Some(0) {
                    println!("  Spot price estimate: ${:.2}", spot_price);
                    println!("  Candidates found: {}", candidates.len());
                }

                if let Some((short_put, long_put)) = candidates.first() {
                    let credit = short_put.mid_price() - long_put.mid_price();
                    let width = short_put.strike - long_put.strike;
                    let max_risk = width - credit;
                    let required_margin = width * 100.0; // Margin = width of spread * 100
                    let max_risk_dollars = max_risk * 100.0; // Per contract

                    // Debug: show position sizing on first day
                    if trading_dates.iter().position(|d| d == current_date) == Some(0) {
                        println!("  First candidate:");
                        println!(
                            "    Short PUT: ${:.2} @ ${:.2}",
                            short_put.strike,
                            short_put.mid_price()
                        );
                        println!(
                            "    Long PUT: ${:.2} @ ${:.2}",
                            long_put.strike,
                            long_put.mid_price()
                        );
                        println!("    Credit: ${:.2}", credit * 100.0);
                        println!("    Width: ${:.2}", width);
                        println!("    Max risk: ${:.2}", max_risk_dollars);
                        println!("    Required margin: ${:.2}", required_margin);
                        println!("    Position size limit: {:.2}%", params.position_size_pct);

                        let current_margin_util = self.get_margin_utilization_pct(capital, &positions);
                        println!("    Current margin utilization: {:.1}%", current_margin_util);
                        println!("    Max margin utilization: {:.1}%", self.max_margin_utilization_pct);
                        println!("    Max risk per trade: {:.1}%", self.max_risk_per_trade_pct);
                        println!("    Open positions: {}/{}", positions.len(), self.max_concurrent_positions);
                    }

                    // Check all risk limits
                    let (can_enter, rejection_reason) = self.can_enter_position(
                        required_margin,
                        max_risk_dollars,
                        capital,
                        &positions,
                    );

                    if can_enter {
                        // Calculate realistic entry prices with bid/ask + slippage
                        let short_entry_price =
                            cost_model.entry_price(short_put.bid, short_put.ask, true);
                        let long_entry_price =
                            cost_model.entry_price(long_put.bid, long_put.ask, false);
                        let realistic_credit = short_entry_price - long_entry_price;

                        // Deduct entry transaction costs immediately
                        let entry_costs = cost_model.entry_cost(2);
                        capital -= entry_costs;

                        // Create position with adjusted entry prices
                        let mut position = strategy.create_position(
                            short_put.clone(),
                            long_put.clone(),
                            *current_date,
                        );

                        // Override entry prices with realistic values
                        position.legs[0].entry_price = short_entry_price;
                        position.legs[1].entry_price = long_entry_price;

                        // Recalculate max profit/loss with realistic credit
                        let width = short_put.strike - long_put.strike;
                        let max_risk_realistic = width - realistic_credit;
                        position.max_profit = Some(realistic_credit);
                        position.max_loss = Some(-max_risk_realistic);

                        println!(
                            "  {} - ENTER: {} (credit: ${:.2}, realistic: ${:.2}, costs: ${:.2}, risk: ${:.2}, margin: ${:.2})",
                            current_date,
                            position.id,
                            credit * 100.0,
                            realistic_credit * 100.0,
                            entry_costs,
                            max_risk_realistic * 100.0,
                            required_margin
                        );

                        positions.push(position);
                    } else if let Some(reason) = rejection_reason {
                        // Log rejection on first occurrence
                        if trading_dates.iter().position(|d| d == current_date) == Some(0)
                            || positions.is_empty()
                        {
                            println!("  {} - REJECT: {}", current_date, reason);
                        }
                    }
                }
            }

            // Record daily capital
            daily_capital.push((*current_date, capital));
        }

        // Close any remaining open positions at final price
        if let Some(final_date) = trading_dates.last() {
            if let Ok(final_contracts) = self.data_loader.load_chain(symbol, *final_date) {
                let mut price_map: HashMap<String, (f64, f64)> = HashMap::new();
                for contract in &final_contracts {
                    price_map.insert(
                        contract.contract_symbol.clone(),
                        (contract.bid, contract.ask),
                    );
                }

                for mut position in positions.drain(..) {
                    let mut pnl = 0.0;
                    for leg in &position.legs {
                        if let Some((bid, ask)) = price_map.get(&leg.contract.contract_symbol) {
                            // Use transaction cost model for exit prices
                            let exit_price = match leg.side {
                                PositionSide::Short => cost_model.exit_price(*bid, *ask, true),
                                PositionSide::Long => cost_model.exit_price(*bid, *ask, false),
                            };
                            match leg.side {
                                PositionSide::Short => {
                                    pnl += leg.entry_price - exit_price;
                                }
                                PositionSide::Long => {
                                    pnl += exit_price - leg.entry_price;
                                }
                            }
                        }
                    }

                    // Deduct exit transaction costs
                    let exit_costs = cost_model.exit_cost(position.legs.len() as u32);
                    let pnl_after_costs = (pnl * 100.0) - exit_costs;

                    capital += pnl_after_costs;
                    position.exit_date = Some(*final_date);

                    // Set exit prices on legs
                    for leg in &mut position.legs {
                        if let Some((bid, ask)) = price_map.get(&leg.contract.contract_symbol) {
                            let exit_price = match leg.side {
                                PositionSide::Short => cost_model.exit_price(*bid, *ask, true),
                                PositionSide::Long => cost_model.exit_price(*bid, *ask, false),
                            };
                            leg.exit_price = Some(exit_price);
                            leg.exit_date = Some(*final_date);
                        }
                    }

                    closed_positions.push(position);
                }
            }
        }

        // Calculate performance metrics
        let metrics =
            PerformanceMetrics::calculate(&closed_positions, &daily_capital, self.initial_capital);

        let result = BacktestResult {
            params: params.clone(),
            num_trades: closed_positions.len(),
            total_pnl: metrics.total_pnl,
            win_rate: metrics.win_rate,
            avg_win: metrics.avg_win,
            avg_loss: metrics.avg_loss,
            max_drawdown: metrics.max_drawdown,
            sharpe_ratio: metrics.sharpe_ratio,
            sortino_ratio: metrics.sortino_ratio,
            profit_factor: metrics.profit_factor,
            max_consecutive_losses: metrics.max_consecutive_losses,
            avg_days_in_trade: metrics.avg_days_in_trade,
            return_on_capital: metrics.return_on_capital,
            positions: closed_positions,
        };

        println!("\n=== Backtest Complete ===");
        println!("Total Trades: {}", result.num_trades);
        println!("Total P&L: ${:.2}", result.total_pnl);
        println!("Win Rate: {:.1}%", result.win_rate);
        println!("Avg Win: ${:.2}", result.avg_win);
        println!("Avg Loss: ${:.2}", result.avg_loss);
        println!("Max Drawdown: ${:.2}", result.max_drawdown);
        println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
        println!("Return on Capital: {:.1}%", result.return_on_capital);
        println!("\n=== Risk Management ===");
        println!("Max Risk Per Trade: {:.1}%", self.max_risk_per_trade_pct);
        println!("Max Concurrent Positions: {}", self.max_concurrent_positions);
        println!("Max Margin Utilization: {:.1}%\n", self.max_margin_utilization_pct);

        Ok(result)
    }

    /// Run backtest with regime-adaptive parameters
    ///
    /// Detects market regime at start of each day and adapts strategy parameters accordingly.
    /// This can significantly improve performance by:
    /// - Increasing aggression in favorable conditions (BullLowVol)
    /// - Reducing risk in volatile conditions (BullHighVol)
    /// - Avoiding poor conditions (BearLowVol, BearHighVol)
    pub fn run_bull_put_spread_adaptive(
        &mut self,
        symbol: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<BacktestResult, DataLoaderError> {
        println!(
            "Running ADAPTIVE backtest: {} from {} to {}",
            symbol, start_date, end_date
        );

        let mut capital = self.initial_capital;
        let mut positions: Vec<OptionsPosition> = Vec::new();
        let mut closed_positions: Vec<OptionsPosition> = Vec::new();
        let mut daily_capital: Vec<(NaiveDate, f64)> = Vec::new();

        // Create regime detector
        let regime_detector = RegimeDetector::default();

        // Track regime changes
        let mut current_regime: Option<MarketRegime> = None;
        let mut regime_changes = 0;

        // Get all available trading dates in range
        let available_dates = self.data_loader.get_available_dates(symbol)?;
        let trading_dates: Vec<_> = available_dates
            .into_iter()
            .filter(|d| *d >= start_date && *d <= end_date)
            .collect();

        println!("  Trading days: {}", trading_dates.len());

        // Walk forward through each trading day
        for current_date in trading_dates.iter() {
            // Detect market regime for this day
            let regime = match regime_detector.detect_regime(&mut self.spot_loader, symbol, *current_date) {
                Ok(r) => r,
                Err(SpotDataError::InsufficientData(_)) => {
                    // Skip days without enough data for regime detection
                    continue;
                }
                Err(e) => {
                    println!("  {} - Warning: Could not detect regime: {}", current_date, e);
                    continue;
                }
            };

            // Log regime changes
            if current_regime.is_none() || current_regime.unwrap() != regime {
                if let Some(prev_regime) = current_regime {
                    println!("  {} - REGIME CHANGE: {} -> {}", current_date, prev_regime, regime);
                } else {
                    println!("  {} - INITIAL REGIME: {}", current_date, regime);
                }
                current_regime = Some(regime);
                regime_changes += 1;
            }

            // Check if we should trade in this regime
            if !should_trade_in_regime(regime) {
                // Skip trading in unfavorable regimes (bear markets)
                continue;
            }

            // Get regime-adapted parameters
            let params = regime_adapted_bull_put_params(regime);
            let strategy = BullPutSpread::new(params.clone());

            // Create transaction cost model
            let cost_model = TransactionCostModel::new_custom(
                params.commission_per_contract,
                0.50,
                params.slippage_ticks,
                0.05,
                params.apply_bid_ask_spread,
            );

            // Load options chain for current date
            let contracts = match self.data_loader.load_chain(symbol, *current_date) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Build price map for exit checks
            let mut price_map: HashMap<String, (f64, f64)> = HashMap::new();
            for contract in &contracts {
                price_map.insert(
                    contract.contract_symbol.clone(),
                    (contract.bid, contract.ask),
                );
            }

            // Check existing positions for exits (using their original params)
            let mut positions_to_close: Vec<usize> = Vec::new();
            for (idx, position) in positions.iter().enumerate() {
                // Reconstruct strategy with position's original params
                let pos_params = self.get_params_from_position_name(&position.strategy);
                let pos_strategy = BullPutSpread::new(pos_params);

                if let Some(reason) = pos_strategy.should_close(position, *current_date, &price_map) {
                    positions_to_close.push(idx);

                    // Calculate P&L
                    let mut pnl = 0.0;
                    for leg in &position.legs {
                        if let Some((bid, ask)) = price_map.get(&leg.contract.contract_symbol) {
                            let exit_price = match leg.side {
                                PositionSide::Short => cost_model.exit_price(*bid, *ask, true),
                                PositionSide::Long => cost_model.exit_price(*bid, *ask, false),
                            };
                            match leg.side {
                                PositionSide::Short => pnl += leg.entry_price - exit_price,
                                PositionSide::Long => pnl += exit_price - leg.entry_price,
                            }
                        }
                    }

                    let exit_costs = cost_model.exit_cost(position.legs.len() as u32);
                    let pnl_after_costs = (pnl * 100.0) - exit_costs;
                    capital += pnl_after_costs;

                    if pnl_after_costs > 0.0 {
                        println!(
                            "  {} - CLOSE WIN: {} (${:.2} net) - {}",
                            current_date, position.id, pnl_after_costs, reason
                        );
                    } else {
                        println!(
                            "  {} - CLOSE LOSS: {} (${:.2} net) - {}",
                            current_date, position.id, pnl_after_costs, reason
                        );
                    }
                }
            }

            // Close positions
            for idx in positions_to_close.iter().rev() {
                let mut position = positions.remove(*idx);
                position.exit_date = Some(*current_date);

                for leg in &mut position.legs {
                    if let Some((bid, ask)) = price_map.get(&leg.contract.contract_symbol) {
                        let exit_price = match leg.side {
                            PositionSide::Short => cost_model.exit_price(*bid, *ask, true),
                            PositionSide::Long => cost_model.exit_price(*bid, *ask, false),
                        };
                        leg.exit_price = Some(exit_price);
                        leg.exit_date = Some(*current_date);
                    }
                }

                closed_positions.push(position);
            }

            // Look for new entries if we have capacity
            if positions.len() < self.max_concurrent_positions {
                let puts: Vec<_> = contracts
                    .iter()
                    .filter(|c| c.option_type == OptionType::Put)
                    .cloned()
                    .collect();

                let spot_price = match self.spot_loader.get_spot_price(symbol, *current_date) {
                    Ok(price) => price,
                    Err(_) => {
                        if let Some(put) = puts.iter().find(|p| {
                            p.delta.map(|d| d.abs()).unwrap_or(0.0) > 0.45
                                && p.delta.map(|d| d.abs()).unwrap_or(0.0) < 0.55
                        }) {
                            put.strike
                        } else {
                            puts.iter().map(|p| p.strike).sum::<f64>() / puts.len() as f64
                        }
                    }
                };

                let candidates = strategy.find_candidates(&puts, spot_price);

                if let Some((short_put, long_put)) = candidates.first() {
                    let credit = short_put.mid_price() - long_put.mid_price();
                    let width = short_put.strike - long_put.strike;
                    let max_risk = width - credit;
                    let required_margin = max_risk * 100.0;
                    let max_risk_dollars = required_margin;

                    let (can_enter, _rejection_reason) = self.can_enter_position(
                        required_margin,
                        max_risk_dollars,
                        capital,
                        &positions,
                    );

                    if can_enter {
                        let short_entry_price = cost_model.entry_price(short_put.bid, short_put.ask, true);
                        let long_entry_price = cost_model.entry_price(long_put.bid, long_put.ask, false);
                        let realistic_credit = short_entry_price - long_entry_price;
                        let entry_costs = cost_model.entry_cost(2);
                        capital -= entry_costs;

                        let mut position = strategy.create_position(
                            short_put.clone(),
                            long_put.clone(),
                            *current_date,
                        );

                        position.legs[0].entry_price = short_entry_price;
                        position.legs[1].entry_price = long_entry_price;

                        let max_risk_realistic = width - realistic_credit;
                        position.max_profit = Some(realistic_credit);
                        position.max_loss = Some(-max_risk_realistic);

                        println!(
                            "  {} - ENTER [{}]: {} (credit: ${:.2}, risk: ${:.2})",
                            current_date,
                            regime,
                            position.id,
                            realistic_credit * 100.0,
                            max_risk_realistic * 100.0
                        );

                        positions.push(position);
                    }
                }
            }

            daily_capital.push((*current_date, capital));
        }

        // Close remaining positions
        if let Some(final_date) = trading_dates.last() {
            if let Ok(final_contracts) = self.data_loader.load_chain(symbol, *final_date) {
                let params = default_bull_put_params();
                let cost_model = TransactionCostModel::new_custom(
                    params.commission_per_contract,
                    0.50,
                    params.slippage_ticks,
                    0.05,
                    params.apply_bid_ask_spread,
                );

                let mut price_map: HashMap<String, (f64, f64)> = HashMap::new();
                for contract in &final_contracts {
                    price_map.insert(
                        contract.contract_symbol.clone(),
                        (contract.bid, contract.ask),
                    );
                }

                for mut position in positions.drain(..) {
                    let mut pnl = 0.0;
                    for leg in &position.legs {
                        if let Some((bid, ask)) = price_map.get(&leg.contract.contract_symbol) {
                            let exit_price = match leg.side {
                                PositionSide::Short => cost_model.exit_price(*bid, *ask, true),
                                PositionSide::Long => cost_model.exit_price(*bid, *ask, false),
                            };
                            match leg.side {
                                PositionSide::Short => pnl += leg.entry_price - exit_price,
                                PositionSide::Long => pnl += exit_price - leg.entry_price,
                            }
                        }
                    }

                    let exit_costs = cost_model.exit_cost(position.legs.len() as u32);
                    let pnl_after_costs = (pnl * 100.0) - exit_costs;
                    capital += pnl_after_costs;
                    position.exit_date = Some(*final_date);

                    for leg in &mut position.legs {
                        if let Some((bid, ask)) = price_map.get(&leg.contract.contract_symbol) {
                            let exit_price = match leg.side {
                                PositionSide::Short => cost_model.exit_price(*bid, *ask, true),
                                PositionSide::Long => cost_model.exit_price(*bid, *ask, false),
                            };
                            leg.exit_price = Some(exit_price);
                            leg.exit_date = Some(*final_date);
                        }
                    }

                    closed_positions.push(position);
                }
            }
        }

        // Calculate metrics
        let metrics = PerformanceMetrics::calculate(&closed_positions, &daily_capital, self.initial_capital);

        // Use default params for result (since we used multiple param sets)
        let mut result_params = default_bull_put_params();
        result_params.name = "BullPutSpread_Adaptive".to_string();

        let result = BacktestResult {
            params: result_params,
            num_trades: closed_positions.len(),
            total_pnl: metrics.total_pnl,
            win_rate: metrics.win_rate,
            avg_win: metrics.avg_win,
            avg_loss: metrics.avg_loss,
            max_drawdown: metrics.max_drawdown,
            sharpe_ratio: metrics.sharpe_ratio,
            sortino_ratio: metrics.sortino_ratio,
            profit_factor: metrics.profit_factor,
            max_consecutive_losses: metrics.max_consecutive_losses,
            avg_days_in_trade: metrics.avg_days_in_trade,
            return_on_capital: metrics.return_on_capital,
            positions: closed_positions,
        };

        println!("\n=== ADAPTIVE Backtest Complete ===");
        println!("Regime Changes: {}", regime_changes);
        println!("Total Trades: {}", result.num_trades);
        println!("Total P&L: ${:.2}", result.total_pnl);
        println!("Win Rate: {:.1}%", result.win_rate);
        println!("Avg Win: ${:.2}", result.avg_win);
        println!("Avg Loss: ${:.2}", result.avg_loss);
        println!("Max Drawdown: ${:.2}", result.max_drawdown);
        println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
        println!("Return on Capital: {:.1}%\n", result.return_on_capital);

        Ok(result)
    }

    /// Helper to reconstruct params from position strategy name
    fn get_params_from_position_name(&self, strategy_name: &str) -> StrategyParams {
        if strategy_name.contains("BullLowVol") {
            regime_adapted_bull_put_params(MarketRegime::BullLowVol)
        } else if strategy_name.contains("BullHighVol") {
            regime_adapted_bull_put_params(MarketRegime::BullHighVol)
        } else if strategy_name.contains("Sideways") {
            regime_adapted_bull_put_params(MarketRegime::Sideways)
        } else if strategy_name.contains("BearLowVol") {
            regime_adapted_bull_put_params(MarketRegime::BearLowVol)
        } else if strategy_name.contains("BearHighVol") {
            regime_adapted_bull_put_params(MarketRegime::BearHighVol)
        } else {
            default_bull_put_params()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_engine_creation() {
        let data_dir = "data/yfinance/options_historical";
        let spot_dir = "data/yfinance/ohlcv";
        if let Ok(loader) = OptionsDataLoader::new(data_dir) {
            if let Ok(spot_loader) = SpotDataLoader::new(spot_dir) {
                let engine = BacktestEngine::new(loader, spot_loader, 10000.0);
                assert_eq!(engine.initial_capital, 10000.0);
            }
        }
    }
}
