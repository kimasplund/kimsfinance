//! Strategy implementations (spreads, iron condors, etc.)

use crate::strategy::market_regime::MarketRegime;
use crate::strategy::types::*;
use chrono::NaiveDate;
use std::collections::HashMap;

/// Bull Put Spread Strategy
///
/// A credit spread where you:
/// - Sell a higher-strike put (collect premium)
/// - Buy a lower-strike put (protection)
///
/// Profit when stock stays above short strike at expiration.
/// Max profit = credit received
/// Max loss = strike width - credit
#[derive(Debug, Clone)]
pub struct BullPutSpread {
    params: StrategyParams,
}

impl BullPutSpread {
    /// Create new bull put spread strategy with parameters
    pub fn new(params: StrategyParams) -> Self {
        Self { params }
    }

    /// Find suitable bull put spread candidates
    ///
    /// Returns pairs of (short_put, long_put) that meet criteria:
    /// - DTE in specified range
    /// - Delta in specified range for short put
    /// - Long put provides protection (lower strike, lower delta)
    /// - Minimum credit requirement met
    pub fn find_candidates(
        &self,
        puts: &[OptionContract],
        spot_price: f64,
    ) -> Vec<(OptionContract, OptionContract)> {
        let mut candidates = Vec::new();

        // Filter puts by DTE and delta for short leg
        let short_candidates: Vec<_> = puts
            .iter()
            .filter(|p| {
                // Must be put
                if p.option_type != OptionType::Put {
                    return false;
                }

                // DTE range
                if p.dte < self.params.dte_min || p.dte > self.params.dte_max {
                    return false;
                }

                // Delta range for short put
                if let Some(delta) = p.delta {
                    let abs_delta = delta.abs();
                    if abs_delta < self.params.delta_min || abs_delta > self.params.delta_max {
                        return false;
                    }
                } else {
                    return false;
                }

                // Must be OTM
                if p.strike >= spot_price {
                    return false;
                }

                // Skip liquidity checks for historical data (volume/OI often not available)
                // In production, you would want:
                // if p.volume < 10.0 || p.open_interest < 100.0 { return false; }

                true
            })
            .cloned()
            .collect();

        // For each short candidate, find suitable long protection
        for short_put in short_candidates.iter() {
            let protection_delta_max = self.params.delta_min * 0.5; // Long put should have ~half the delta

            let long_candidates: Vec<_> = puts
                .iter()
                .filter(|p| {
                    // Same expiration
                    if p.expiration != short_put.expiration {
                        return false;
                    }

                    // Lower strike (protection)
                    if p.strike >= short_put.strike {
                        return false;
                    }

                    // Delta range for long put (lower delta)
                    if let Some(delta) = p.delta {
                        let abs_delta = delta.abs();
                        if abs_delta > protection_delta_max {
                            return false;
                        }
                    } else {
                        return false;
                    }

                    // Skip liquidity checks for historical data (volume/OI often not available)
                    // In production, you would want:
                    // if p.volume < 5.0 || p.open_interest < 50.0 { return false; }

                    true
                })
                .cloned()
                .collect();

            // Find best long put (closest to desired protection level)
            if let Some(long_put) = long_candidates.first() {
                let credit = short_put.mid_price() - long_put.mid_price();

                // Check minimum credit if specified
                if let Some(min_credit) = self.params.min_credit {
                    if credit < min_credit {
                        continue;
                    }
                }

                // Valid spread
                candidates.push((short_put.clone(), long_put.clone()));
            }
        }

        candidates
    }

    /// Create a position from a bull put spread
    pub fn create_position(
        &self,
        short_put: OptionContract,
        long_put: OptionContract,
        entry_date: NaiveDate,
    ) -> OptionsPosition {
        let short_price = short_put.mid_price();
        let long_price = long_put.mid_price();
        let credit = short_price - long_price;
        let width = short_put.strike - long_put.strike;
        let max_risk = width - credit;

        let position_id = format!(
            "BPS_{}_{}_{}_{}",
            short_put.symbol,
            short_put.strike,
            long_put.strike,
            entry_date.format("%Y%m%d")
        );

        let short_leg = OptionLeg {
            contract: short_put,
            side: PositionSide::Short,
            quantity: 1,
            entry_price: short_price,
            exit_price: None,
            entry_date,
            exit_date: None,
        };

        let long_leg = OptionLeg {
            contract: long_put,
            side: PositionSide::Long,
            quantity: 1,
            entry_price: long_price,
            exit_price: None,
            entry_date,
            exit_date: None,
        };

        OptionsPosition {
            id: position_id,
            strategy: "BullPutSpread".to_string(),
            legs: vec![short_leg, long_leg],
            entry_date,
            exit_date: None,
            max_profit: Some(credit),
            max_loss: Some(-max_risk),
        }
    }

    /// Check if position should be closed based on exit criteria
    ///
    /// Returns Some(reason) if should close, None otherwise
    pub fn should_close(
        &self,
        position: &OptionsPosition,
        current_date: NaiveDate,
        current_prices: &HashMap<String, (f64, f64)>, // contract_symbol -> (bid, ask)
    ) -> Option<String> {
        // Days in trade
        let days_in_trade = (current_date - position.entry_date).num_days() as i32;

        // Check max hold days
        if let Some(max_days) = self.params.max_hold_days {
            if days_in_trade >= max_days {
                return Some(format!("Max hold days reached: {}", days_in_trade));
            }
        }

        // Check if at expiration
        let expiration = position.legs[0].contract.expiration;
        if current_date >= expiration {
            return Some("At expiration".to_string());
        }

        // Calculate current P&L
        let mut current_value = 0.0;
        let entry_credit = position.max_profit.unwrap_or(0.0);

        for leg in &position.legs {
            if let Some((bid, ask)) = current_prices.get(&leg.contract.contract_symbol) {
                let current_price = (bid + ask) / 2.0;
                match leg.side {
                    PositionSide::Short => {
                        // Short position: profit when price decreases
                        current_value += leg.entry_price - current_price;
                    }
                    PositionSide::Long => {
                        // Long position: profit when price increases
                        current_value += current_price - leg.entry_price;
                    }
                }
            }
        }

        let pnl = current_value;
        let pnl_pct = if entry_credit.abs() > 0.001 {
            (pnl / entry_credit) * 100.0
        } else {
            0.0
        };

        // Check profit target
        if let Some(profit_target) = self.params.profit_target_pct {
            if pnl_pct >= profit_target {
                return Some(format!("Profit target hit: {:.1}%", pnl_pct));
            }
        }

        // Check stop loss
        if let Some(stop_loss) = self.params.stop_loss_pct {
            if pnl_pct <= -stop_loss {
                return Some(format!("Stop loss hit: {:.1}%", pnl_pct));
            }
        }

        None
    }
}

/// Create default bull put spread parameters
pub fn default_bull_put_params() -> StrategyParams {
    let mut custom = HashMap::new();
    custom.insert("strike_width_pct".to_string(), 5.0); // 5% wide spreads

    StrategyParams {
        name: "BullPutSpread".to_string(),
        dte_min: 30,
        dte_max: 45,
        delta_min: 0.15,
        delta_max: 0.35,
        profit_target_pct: Some(50.0), // Take profit at 50% of max profit
        stop_loss_pct: Some(200.0),    // Stop out at 200% of credit (max loss)
        max_hold_days: Some(42),       // Don't hold to expiration
        position_size_pct: 100.0,      // Allow up to 100% capital per trade (for backtest)
        min_credit: Some(0.20),        // Minimum $0.20 credit
        commission_per_contract: 0.65, // $0.65 per contract (retail broker)
        slippage_ticks: 1.0,           // 1 tick = $0.05
        apply_bid_ask_spread: true,    // Use realistic bid/ask
        custom_params: custom,
    }
}

/// Create regime-adapted bull put spread parameters
///
/// Adjusts strategy parameters based on market regime:
///
/// **BullLowVol** (ideal conditions):
/// - Delta: 0.30-0.40 (aggressive, closer to ATM)
/// - Profit target: 40% (take profits earlier)
/// - Stop loss: 200%
/// - Max hold: 35 days
///
/// **BullHighVol** (reduce risk):
/// - Delta: 0.15-0.25 (conservative, further OTM)
/// - Profit target: 60% (wait for higher gains to justify risk)
/// - Stop loss: 150% (tighter stops in volatile markets)
/// - Max hold: 30 days
///
/// **Sideways** (moderate):
/// - Delta: 0.20-0.30 (balanced)
/// - Profit target: 50%
/// - Stop loss: 200%
/// - Max hold: 40 days
///
/// **BearLowVol/BearHighVol** (defensive or skip):
/// - Delta: 0.10-0.20 (very conservative, far OTM)
/// - Profit target: 70% (need more conviction)
/// - Stop loss: 100% (exit quickly)
/// - Max hold: 21 days
pub fn regime_adapted_bull_put_params(regime: MarketRegime) -> StrategyParams {
    let mut params = default_bull_put_params();

    match regime {
        MarketRegime::BullLowVol => {
            // Ideal conditions - aggressive positioning
            params.name = "BullPutSpread_BullLowVol".to_string();
            params.delta_min = 0.30;
            params.delta_max = 0.40;
            params.profit_target_pct = Some(40.0);
            params.stop_loss_pct = Some(200.0);
            params.max_hold_days = Some(35);
            params.min_credit = Some(0.25); // Higher credit for higher delta
        }
        MarketRegime::BullHighVol => {
            // Bull but volatile - reduce risk
            params.name = "BullPutSpread_BullHighVol".to_string();
            params.delta_min = 0.15;
            params.delta_max = 0.25;
            params.profit_target_pct = Some(60.0);
            params.stop_loss_pct = Some(150.0); // Tighter stops
            params.max_hold_days = Some(30);
            params.min_credit = Some(0.20);
        }
        MarketRegime::Sideways => {
            // Choppy market - moderate approach
            params.name = "BullPutSpread_Sideways".to_string();
            params.delta_min = 0.20;
            params.delta_max = 0.30;
            params.profit_target_pct = Some(50.0);
            params.stop_loss_pct = Some(200.0);
            params.max_hold_days = Some(40);
            params.min_credit = Some(0.20);
        }
        MarketRegime::BearLowVol | MarketRegime::BearHighVol => {
            // Bear market - very conservative or skip
            params.name = format!("BullPutSpread_{}", regime);
            params.delta_min = 0.10;
            params.delta_max = 0.20;
            params.profit_target_pct = Some(70.0); // Need more conviction
            params.stop_loss_pct = Some(100.0); // Exit quickly
            params.max_hold_days = Some(21);
            params.min_credit = Some(0.15);
        }
    }

    params
}

/// Check if we should trade in the current regime
///
/// Returns true if the regime is favorable for bull put spreads
pub fn should_trade_in_regime(regime: MarketRegime) -> bool {
    match regime {
        MarketRegime::BullLowVol => true,   // Best conditions
        MarketRegime::BullHighVol => true,  // Good but reduce risk
        MarketRegime::Sideways => true,     // Moderate conditions
        MarketRegime::BearLowVol => false,  // Skip bear markets
        MarketRegime::BearHighVol => false, // Definitely skip
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn test_bull_put_spread_creation() {
        let params = default_bull_put_params();
        let strategy = BullPutSpread::new(params);

        let short_put = OptionContract {
            symbol: "SPY".to_string(),
            contract_symbol: "SPY241220P00450000".to_string(),
            strike: 450.0,
            expiration: NaiveDate::from_ymd_opt(2024, 12, 20).unwrap(),
            option_type: OptionType::Put,
            snapshot_date: NaiveDate::from_ymd_opt(2024, 11, 15).unwrap(),
            bid: 1.80,
            ask: 1.90,
            last_price: 1.85,
            volume: 1000.0,
            open_interest: 5000.0,
            delta: Some(-0.25),
            gamma: Some(0.01),
            theta: Some(-0.05),
            vega: Some(0.10),
            rho: Some(-0.02),
            implied_volatility: Some(0.15),
            dte: 35,
        };

        let long_put = OptionContract {
            symbol: "SPY".to_string(),
            contract_symbol: "SPY241220P00445000".to_string(),
            strike: 445.0,
            expiration: NaiveDate::from_ymd_opt(2024, 12, 20).unwrap(),
            option_type: OptionType::Put,
            snapshot_date: NaiveDate::from_ymd_opt(2024, 11, 15).unwrap(),
            bid: 1.20,
            ask: 1.30,
            last_price: 1.25,
            volume: 500.0,
            open_interest: 2500.0,
            delta: Some(-0.12),
            gamma: Some(0.008),
            theta: Some(-0.03),
            vega: Some(0.08),
            rho: Some(-0.01),
            implied_volatility: Some(0.14),
            dte: 35,
        };

        let entry_date = NaiveDate::from_ymd_opt(2024, 11, 15).unwrap();
        let position = strategy.create_position(short_put, long_put, entry_date);

        assert_eq!(position.legs.len(), 2);
        assert_eq!(position.legs[0].side, PositionSide::Short);
        assert_eq!(position.legs[1].side, PositionSide::Long);

        let credit = 1.85 - 1.25;
        assert!((position.max_profit.unwrap() - credit).abs() < 0.01);

        let max_risk = 5.0 - credit;
        assert!((position.max_loss.unwrap() + max_risk).abs() < 0.01);
    }
}
