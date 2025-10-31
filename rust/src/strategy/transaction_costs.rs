//! Transaction cost modeling for realistic backtesting
//!
//! Models all real-world trading costs for options:
//! - Commissions: Per-contract fees
//! - Leg fees: Per-leg fees
//! - Slippage: Market impact (configurable ticks)
//! - Bid-ask spread: Entry uses ask, exit uses bid
//!
//! ## Typical Costs (Options Trading)
//!
//! - **Commission**: $0.65 per contract (standard retail broker)
//! - **Leg fee**: $0.50 per leg (exchange/clearing fees)
//! - **Slippage**: 1-2 ticks ($0.05-$0.10 per contract)
//! - **Bid-ask spread**: Naturally modeled by using bid for sells, ask for buys
//!
//! ## Example: Bull Put Spread Entry
//!
//! ```text
//! Entry:
//! - Short PUT @ $2.50 ask (would prefer $2.55 bid)
//! - Long PUT @ $0.10 bid (would prefer $0.05 ask)
//! - Commission: $0.65 × 2 = $1.30
//! - Leg fees: $0.50 × 2 = $1.00
//! - Slippage: $0.05 × 2 = $0.10 (1 tick per leg)
//! - Total cost: $2.40 per spread
//! ```
//!
//! ## Realistic Impact
//!
//! - **Cost per round trip**: $4.80-$7.00 per spread
//! - **Expected profit drag**: 15-30% on high-frequency strategies
//! - **Win rate impact**: Reduces by 5-10% (small winners become losers)

use serde::{Deserialize, Serialize};

/// Transaction cost model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionCostModel {
    /// Commission per contract (e.g., $0.65)
    pub commission_per_contract: f64,

    /// Leg fee per leg (e.g., $0.50)
    pub leg_fee_per_leg: f64,

    /// Slippage in ticks (e.g., 1.0 = $0.05 per contract)
    pub slippage_ticks: f64,

    /// Tick size (e.g., $0.05 for options)
    pub tick_size: f64,

    /// Apply bid-ask spread modeling (use bid for exits, ask for entries)
    pub apply_bid_ask_spread: bool,
}

impl TransactionCostModel {
    /// Create new transaction cost model with typical retail broker costs
    pub fn new_retail_broker() -> Self {
        Self {
            commission_per_contract: 0.65, // $0.65 per contract (typical)
            leg_fee_per_leg: 0.50,         // $0.50 per leg (exchange fees)
            slippage_ticks: 1.0,           // 1 tick slippage
            tick_size: 0.05,               // $0.05 tick size
            apply_bid_ask_spread: true,    // Use realistic bid/ask
        }
    }

    /// Create model with no costs (for comparison)
    pub fn new_zero_costs() -> Self {
        Self {
            commission_per_contract: 0.0,
            leg_fee_per_leg: 0.0,
            slippage_ticks: 0.0,
            tick_size: 0.05,
            apply_bid_ask_spread: false,
        }
    }

    /// Create model with custom costs
    pub fn new_custom(
        commission_per_contract: f64,
        leg_fee_per_leg: f64,
        slippage_ticks: f64,
        tick_size: f64,
        apply_bid_ask_spread: bool,
    ) -> Self {
        Self {
            commission_per_contract,
            leg_fee_per_leg,
            slippage_ticks,
            tick_size,
            apply_bid_ask_spread,
        }
    }

    /// Calculate total entry cost for a spread (2 legs)
    ///
    /// # Arguments
    ///
    /// * `num_legs` - Number of legs in the spread (e.g., 2 for bull put spread)
    ///
    /// # Returns
    ///
    /// Total cost in dollars
    #[inline]
    pub fn entry_cost(&self, num_legs: u32) -> f64 {
        let commission = self.commission_per_contract * num_legs as f64;
        let leg_fees = self.leg_fee_per_leg * num_legs as f64;
        let slippage = self.slippage_ticks * self.tick_size * num_legs as f64;

        commission + leg_fees + slippage
    }

    /// Calculate total exit cost for a spread (2 legs)
    ///
    /// # Arguments
    ///
    /// * `num_legs` - Number of legs in the spread
    ///
    /// # Returns
    ///
    /// Total cost in dollars
    #[inline]
    pub fn exit_cost(&self, num_legs: u32) -> f64 {
        // Same as entry cost
        self.entry_cost(num_legs)
    }

    /// Calculate total round-trip cost (entry + exit)
    ///
    /// # Arguments
    ///
    /// * `num_legs` - Number of legs in the spread
    ///
    /// # Returns
    ///
    /// Total round-trip cost in dollars
    #[inline]
    pub fn round_trip_cost(&self, num_legs: u32) -> f64 {
        self.entry_cost(num_legs) + self.exit_cost(num_legs)
    }

    /// Apply entry price adjustment with slippage
    ///
    /// For short positions: use ask price + slippage (worse fill)
    /// For long positions: use ask price + slippage (worse fill)
    ///
    /// # Arguments
    ///
    /// * `bid` - Bid price
    /// * `ask` - Ask price
    /// * `is_short` - True if short position, false if long
    ///
    /// # Returns
    ///
    /// Adjusted entry price
    #[inline]
    pub fn entry_price(&self, bid: f64, ask: f64, is_short: bool) -> f64 {
        if !self.apply_bid_ask_spread {
            // Use mid price without spread modeling
            return (bid + ask) / 2.0;
        }

        let base_price = if is_short {
            // Short position: receive bid price (sell to market)
            bid
        } else {
            // Long position: pay ask price (buy from market)
            ask
        };

        // Apply slippage (always against us)
        let slippage = self.slippage_ticks * self.tick_size;
        if is_short {
            // Worse fill for short: receive less
            base_price - slippage
        } else {
            // Worse fill for long: pay more
            base_price + slippage
        }
    }

    /// Apply exit price adjustment with slippage
    ///
    /// For closing short: use ask price + slippage (worse fill)
    /// For closing long: use bid price - slippage (worse fill)
    ///
    /// # Arguments
    ///
    /// * `bid` - Bid price
    /// * `ask` - Ask price
    /// * `is_closing_short` - True if closing a short position
    ///
    /// # Returns
    ///
    /// Adjusted exit price
    #[inline]
    pub fn exit_price(&self, bid: f64, ask: f64, is_closing_short: bool) -> f64 {
        if !self.apply_bid_ask_spread {
            // Use mid price without spread modeling
            return (bid + ask) / 2.0;
        }

        let base_price = if is_closing_short {
            // Closing short: buy at ask (pay market)
            ask
        } else {
            // Closing long: sell at bid (receive from market)
            bid
        };

        // Apply slippage (always against us)
        let slippage = self.slippage_ticks * self.tick_size;
        if is_closing_short {
            // Worse fill: pay more to close short
            base_price + slippage
        } else {
            // Worse fill: receive less to close long
            base_price - slippage
        }
    }
}

impl Default for TransactionCostModel {
    fn default() -> Self {
        Self::new_retail_broker()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retail_broker_costs() {
        let model = TransactionCostModel::new_retail_broker();

        // Entry cost for 2-leg spread
        let entry_cost = model.entry_cost(2);
        // $0.65 * 2 + $0.50 * 2 + $0.05 * 2 = $2.40
        assert_eq!(entry_cost, 2.40);

        // Exit cost (same as entry)
        let exit_cost = model.exit_cost(2);
        assert_eq!(exit_cost, 2.40);

        // Round trip cost
        let round_trip = model.round_trip_cost(2);
        assert_eq!(round_trip, 4.80);
    }

    #[test]
    fn test_zero_costs() {
        let model = TransactionCostModel::new_zero_costs();

        assert_eq!(model.entry_cost(2), 0.0);
        assert_eq!(model.exit_cost(2), 0.0);
        assert_eq!(model.round_trip_cost(2), 0.0);
    }

    #[test]
    fn test_entry_price_short_with_spread() {
        let model = TransactionCostModel::new_retail_broker();

        // Short position: receive bid - slippage
        let price = model.entry_price(2.50, 2.55, true);
        // Bid: 2.50, minus slippage: 2.50 - 0.05 = 2.45
        assert_eq!(price, 2.45);
    }

    #[test]
    fn test_entry_price_long_with_spread() {
        let model = TransactionCostModel::new_retail_broker();

        // Long position: pay ask + slippage
        let price = model.entry_price(0.08, 0.12, false);
        // Ask: 0.12, plus slippage: 0.12 + 0.05 = 0.17
        assert_eq!(price, 0.17);
    }

    #[test]
    fn test_exit_price_closing_short() {
        let model = TransactionCostModel::new_retail_broker();

        // Closing short: buy at ask + slippage
        let price = model.exit_price(0.05, 0.10, true);
        // Ask: 0.10, plus slippage: 0.10 + 0.05 = 0.15
        assert_eq!(price, 0.15);
    }

    #[test]
    fn test_exit_price_closing_long() {
        let model = TransactionCostModel::new_retail_broker();

        // Closing long: sell at bid - slippage
        let price = model.exit_price(2.45, 2.50, false);
        // Bid: 2.45, minus slippage: 2.45 - 0.05 = 2.40
        assert_eq!(price, 2.40);
    }

    #[test]
    fn test_entry_price_without_spread_modeling() {
        let mut model = TransactionCostModel::new_retail_broker();
        model.apply_bid_ask_spread = false;

        // Should use mid price for both
        let price_short = model.entry_price(2.50, 2.60, true);
        let price_long = model.entry_price(2.50, 2.60, false);

        assert_eq!(price_short, 2.55);
        assert_eq!(price_long, 2.55);
    }

    #[test]
    fn test_realistic_spread_scenario() {
        let model = TransactionCostModel::new_retail_broker();

        // Bull put spread entry:
        // - Short PUT @ bid=$2.50, ask=$2.55
        // - Long PUT @ bid=$0.08, ask=$0.12

        let short_entry = model.entry_price(2.50, 2.55, true);
        let long_entry = model.entry_price(0.08, 0.12, false);

        // Credit received (worse than mid prices)
        let credit = short_entry - long_entry;
        // Expected: (2.50 - 0.05) - (0.12 + 0.05) = 2.45 - 0.17 = 2.28
        assert_eq!(credit, 2.28);

        // Compare to mid prices (no costs)
        let mid_short = (2.50 + 2.55) / 2.0;
        let mid_long = (0.08 + 0.12) / 2.0;
        let mid_credit = mid_short - mid_long;
        // Expected: 2.525 - 0.10 = 2.425
        assert_eq!(mid_credit, 2.425);

        // Cost drag from bid-ask + slippage
        let price_drag = mid_credit - credit;
        // Expected: 2.425 - 2.28 = 0.145
        assert_eq!(price_drag, 0.145);

        // Total entry cost (fees)
        let entry_cost = model.entry_cost(2);
        assert_eq!(entry_cost, 2.40);

        // Total entry impact (price drag + fees)
        let total_entry_impact = (price_drag * 100.0) + entry_cost;
        // Expected: (0.145 * 100) + 2.40 = 14.50 + 2.40 = 16.90
        assert_eq!(total_entry_impact, 16.90);
    }

    #[test]
    fn test_custom_costs() {
        // Professional trader (lower costs)
        let model = TransactionCostModel::new_custom(
            0.25, // $0.25 per contract
            0.25, // $0.25 per leg
            0.5,  // 0.5 tick slippage
            0.05, // $0.05 tick size
            true, // Use bid/ask
        );

        let entry_cost = model.entry_cost(2);
        // $0.25 * 2 + $0.25 * 2 + $0.025 * 2 = $1.05
        assert_eq!(entry_cost, 1.05);

        let round_trip = model.round_trip_cost(2);
        assert_eq!(round_trip, 2.10);
    }
}
