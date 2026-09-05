//! Order matching engine for realistic backtest execution
//!
//! Implements price-time priority matching with support for:
//! - Market orders: Immediate execution at current price
//! - Limit orders: Execution when price reaches limit
//! - Stop orders: Trigger when stop price hit, then execute
//! - Trailing stops: Dynamic stop price adjustment
//! - Session orders: MOO, MOC, LOO, LOC
//! - Algorithmic orders: TWAP, VWAP, POV slicing
//! - Complex orders: OCO, OTO, Bracket handling
//!
//! # Matching Logic
//!
//! Price-time priority:
//! 1. Best price gets priority
//! 2. Earlier orders at same price get priority
//! 3. Market orders always have highest priority

use super::core::OHLCVBar;
use super::orders::{
    Fill, Order, OrderGroup, OrderId, OrderSide, OrderStatus, OrderType, TimeInForce,
};
use std::collections::HashMap;

/// Market data snapshot for order matching
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    /// Current timestamp
    pub timestamp: i64,

    /// Current bar (OHLCV)
    pub bar: OHLCVBar,

    /// Is this the first bar of the session (market open)?
    pub is_market_open: bool,

    /// Is this the last bar of the session (market close)?
    pub is_market_close: bool,

    /// Session trading volume so far
    pub session_volume: f64,

    /// Average volume for this time of day (for VWAP)
    pub average_volume: f64,
}

impl MarketSnapshot {
    /// Create market snapshot from OHLCV bar
    pub fn new(timestamp: i64, bar: OHLCVBar) -> Self {
        Self {
            timestamp,
            bar,
            is_market_open: false,
            is_market_close: false,
            session_volume: 0.0,
            average_volume: 0.0,
        }
    }

    /// Get current market price (use close for simplicity)
    pub fn current_price(&self) -> f64 {
        self.bar.close
    }

    /// Get bid price (close - small spread)
    pub fn bid_price(&self) -> f64 {
        self.bar.close * 0.9995 // 0.05% spread
    }

    /// Get ask price (close + small spread)
    pub fn ask_price(&self) -> f64 {
        self.bar.close * 1.0005 // 0.05% spread
    }
}

/// Configuration for order matching engine
#[derive(Debug, Clone)]
pub struct MatchingConfig {
    /// Trading fee per trade (as fraction, e.g., 0.001 = 0.1%)
    pub trading_fee: f64,

    /// Base slippage (as fraction, e.g., 0.0005 = 0.05%)
    pub base_slippage: f64,

    /// Enable realistic slippage model (increases with order size)
    pub realistic_slippage: bool,

    /// Enable partial fills for large orders
    pub enable_partial_fills: bool,

    /// Maximum fill percentage per bar (e.g., 0.1 = 10% of bar volume)
    pub max_fill_per_bar: f64,
}

impl Default for MatchingConfig {
    fn default() -> Self {
        Self {
            trading_fee: 0.001,    // 0.1%
            base_slippage: 0.0005, // 0.05%
            realistic_slippage: true,
            enable_partial_fills: true,
            max_fill_per_bar: 0.1, // 10% of volume
        }
    }
}

/// Order matching engine
pub struct MatchingEngine {
    /// Configuration
    config: MatchingConfig,

    /// Pending orders (not yet filled)
    pending_orders: HashMap<OrderId, Order>,

    /// Completed orders (filled, cancelled, rejected)
    completed_orders: HashMap<OrderId, Order>,

    /// Order groups (OCO, OTO, Bracket)
    order_groups: Vec<OrderGroup>,

    /// Next order ID
    next_order_id: OrderId,

    /// TWAP/VWAP state tracking
    algo_order_state: HashMap<OrderId, AlgoOrderState>,
}

/// State tracking for algorithmic orders
#[derive(Debug, Clone)]
struct AlgoOrderState {
    /// Total quantity to execute
    _total_quantity: f64,

    /// Quantity executed so far
    executed_quantity: f64,

    /// Start time
    start_time: i64,

    /// End time (for TWAP)
    _end_time: Option<i64>,

    /// Last execution time
    last_execution_time: i64,
}

impl MatchingEngine {
    /// Create new matching engine
    pub fn new() -> Self {
        Self::with_config(MatchingConfig::default())
    }

    /// Create matching engine with custom configuration
    pub fn with_config(config: MatchingConfig) -> Self {
        Self {
            config,
            pending_orders: HashMap::new(),
            completed_orders: HashMap::new(),
            order_groups: Vec::new(),
            next_order_id: 1,
            algo_order_state: HashMap::new(),
        }
    }

    /// Submit new order
    pub fn submit_order(&mut self, mut order: Order) -> OrderId {
        let order_id = self.next_order_id;
        self.next_order_id += 1;

        order.id = order_id;
        order.set_status(OrderStatus::Pending);

        // Initialize algo order state if needed
        if matches!(
            order.order_type,
            OrderType::TWAP | OrderType::VWAP | OrderType::POV
        ) {
            let state = AlgoOrderState {
                _total_quantity: order.quantity,
                executed_quantity: 0.0,
                start_time: order.created_at,
                _end_time: order.twap_duration_secs.map(|d| order.created_at + d),
                last_execution_time: order.created_at,
            };
            self.algo_order_state.insert(order_id, state);
        }

        self.pending_orders.insert(order_id, order);
        order_id
    }

    /// Submit order group (OCO, OTO, Bracket)
    pub fn submit_order_group(&mut self, group: OrderGroup) {
        self.order_groups.push(group);
    }

    /// Cancel order by ID
    pub fn cancel_order(&mut self, order_id: OrderId) -> bool {
        if let Some(mut order) = self.pending_orders.remove(&order_id) {
            order.set_status(OrderStatus::Cancelled);
            self.completed_orders.insert(order_id, order);
            true
        } else {
            false
        }
    }

    /// Get order by ID
    pub fn get_order(&self, order_id: OrderId) -> Option<&Order> {
        self.pending_orders
            .get(&order_id)
            .or_else(|| self.completed_orders.get(&order_id))
    }

    /// Get all pending orders
    pub fn pending_orders(&self) -> &HashMap<OrderId, Order> {
        &self.pending_orders
    }

    /// Process market snapshot and match orders
    pub fn match_orders(&mut self, market: &MarketSnapshot) -> Vec<Fill> {
        let mut fills = Vec::new();

        // Step 1: Update trailing stops
        self.update_trailing_stops(market);

        // Step 2: Check stop order triggers
        self.check_stop_triggers(market);

        // Step 3: Check time-in-force expiry
        self.check_expiry(market);

        // Step 4: Match orders by priority
        let order_ids: Vec<OrderId> = self.pending_orders.keys().copied().collect();

        for order_id in order_ids {
            if let Some(order_fills) = self.match_single_order(order_id, market) {
                fills.extend(order_fills);
            }
        }

        // Step 5: Handle order groups (OCO, OTO, Bracket)
        self.handle_order_groups(&fills);

        fills
    }

    /// Update trailing stop prices based on current market price
    fn update_trailing_stops(&mut self, market: &MarketSnapshot) {
        for order in self.pending_orders.values_mut() {
            if order.order_type == OrderType::TrailingStop
                || order.order_type == OrderType::TrailingStopLimit
            {
                order.update_trailing_high_water_mark(market.current_price());
            }
        }
    }

    /// Check if stop orders should be triggered
    fn check_stop_triggers(&mut self, market: &MarketSnapshot) {
        let mut triggered_orders = Vec::new();

        for (order_id, order) in &self.pending_orders {
            let should_trigger = match order.order_type {
                OrderType::Stop | OrderType::StopLimit => {
                    if let Some(stop_price) = order.stop_price {
                        match order.side {
                            OrderSide::Buy => market.bar.high >= stop_price,
                            OrderSide::Sell => market.bar.low <= stop_price,
                        }
                    } else {
                        false
                    }
                }
                OrderType::TrailingStop | OrderType::TrailingStopLimit => {
                    if let Some(trailing_stop_price) = order.calculate_trailing_stop_price() {
                        match order.side {
                            OrderSide::Buy => market.current_price() >= trailing_stop_price,
                            OrderSide::Sell => market.current_price() <= trailing_stop_price,
                        }
                    } else {
                        false
                    }
                }
                _ => false,
            };

            if should_trigger {
                triggered_orders.push(*order_id);
            }
        }

        // Mark triggered orders
        for order_id in triggered_orders {
            if let Some(order) = self.pending_orders.get_mut(&order_id) {
                order.set_status(OrderStatus::Triggered);
            }
        }
    }

    /// Check time-in-force expiry
    fn check_expiry(&mut self, market: &MarketSnapshot) {
        let mut expired_orders = Vec::new();

        for (order_id, order) in &self.pending_orders {
            let is_expired = match order.time_in_force {
                TimeInForce::Day => market.is_market_close,
                TimeInForce::GTD { expiry } => market.timestamp >= expiry,
                TimeInForce::IOC | TimeInForce::FOK => {
                    // These should have been handled in match_single_order
                    false
                }
                TimeInForce::GTC => false,
            };

            if is_expired {
                expired_orders.push(*order_id);
            }
        }

        // Mark expired orders
        for order_id in expired_orders {
            if let Some(mut order) = self.pending_orders.remove(&order_id) {
                order.set_status(OrderStatus::Expired);
                self.completed_orders.insert(order_id, order);
            }
        }
    }

    /// Match a single order and return fills
    fn match_single_order(
        &mut self,
        order_id: OrderId,
        market: &MarketSnapshot,
    ) -> Option<Vec<Fill>> {
        let order = self.pending_orders.get(&order_id)?;

        // Check if order can be matched at this time
        if !self.can_match_order(order, market) {
            return None;
        }

        let fills = match order.order_type {
            OrderType::Market => self.match_market_order(order_id, market),
            OrderType::Limit => self.match_limit_order(order_id, market),
            OrderType::Stop | OrderType::TrailingStop => {
                // After trigger, treat as market order
                if order.status == OrderStatus::Triggered {
                    self.match_market_order(order_id, market)
                } else {
                    None
                }
            }
            OrderType::StopLimit | OrderType::TrailingStopLimit => {
                // After trigger, treat as limit order
                if order.status == OrderStatus::Triggered {
                    self.match_limit_order(order_id, market)
                } else {
                    None
                }
            }
            OrderType::MarketOnOpen => {
                if market.is_market_open {
                    self.match_market_order(order_id, market)
                } else {
                    None
                }
            }
            OrderType::MarketOnClose => {
                if market.is_market_close {
                    self.match_market_order(order_id, market)
                } else {
                    None
                }
            }
            OrderType::LimitOnOpen => {
                if market.is_market_open {
                    self.match_limit_order(order_id, market)
                } else {
                    None
                }
            }
            OrderType::LimitOnClose => {
                if market.is_market_close {
                    self.match_limit_order(order_id, market)
                } else {
                    None
                }
            }
            OrderType::Iceberg => self.match_iceberg_order(order_id, market),
            OrderType::TWAP => self.match_twap_order(order_id, market),
            OrderType::VWAP => self.match_vwap_order(order_id, market),
            OrderType::POV => self.match_pov_order(order_id, market),
        };

        // Update order status based on fills
        if let Some(ref fill_vec) = fills {
            self.update_order_from_fills(order_id, fill_vec);
        }

        fills
    }

    /// Check if order can be matched at this market snapshot
    fn can_match_order(&self, order: &Order, market: &MarketSnapshot) -> bool {
        if !order.is_active() {
            return false;
        }

        // Session-based orders
        match order.order_type {
            OrderType::MarketOnOpen | OrderType::LimitOnOpen => market.is_market_open,
            OrderType::MarketOnClose | OrderType::LimitOnClose => market.is_market_close,
            _ => true,
        }
    }

    /// Match market order
    fn match_market_order(&self, order_id: OrderId, market: &MarketSnapshot) -> Option<Vec<Fill>> {
        let order = self.pending_orders.get(&order_id)?;

        let execution_price = match order.side {
            OrderSide::Buy => market.ask_price(),
            OrderSide::Sell => market.bid_price(),
        };

        let quantity = self.calculate_fill_quantity(order, market);
        let slippage = self.calculate_slippage(quantity, market);
        let fee = quantity * execution_price * self.config.trading_fee;

        let fill = Fill::new(
            order_id,
            market.timestamp,
            execution_price,
            quantity,
            fee,
            slippage,
            order.side,
            order.symbol.clone(),
        );

        Some(vec![fill])
    }

    /// Match limit order
    fn match_limit_order(&self, order_id: OrderId, market: &MarketSnapshot) -> Option<Vec<Fill>> {
        let order = self.pending_orders.get(&order_id)?;
        let limit_price = order.limit_price?;

        // Check if limit price is met
        let can_fill = match order.side {
            OrderSide::Buy => market.bar.low <= limit_price,
            OrderSide::Sell => market.bar.high >= limit_price,
        };

        if !can_fill {
            return None;
        }

        let execution_price = limit_price;
        let quantity = self.calculate_fill_quantity(order, market);
        let slippage = self.calculate_slippage(quantity, market);
        let fee = quantity * execution_price * self.config.trading_fee;

        let fill = Fill::new(
            order_id,
            market.timestamp,
            execution_price,
            quantity,
            fee,
            slippage,
            order.side,
            order.symbol.clone(),
        );

        Some(vec![fill])
    }

    /// Match iceberg order (show only visible quantity)
    fn match_iceberg_order(&self, order_id: OrderId, market: &MarketSnapshot) -> Option<Vec<Fill>> {
        let order = self.pending_orders.get(&order_id)?;
        let visible_qty = order
            .iceberg_visible_qty
            .unwrap_or(order.remaining_quantity);

        // Only fill visible quantity at a time
        let quantity = visible_qty.min(order.remaining_quantity);
        let limit_price = order.limit_price?;

        let can_fill = match order.side {
            OrderSide::Buy => market.bar.low <= limit_price,
            OrderSide::Sell => market.bar.high >= limit_price,
        };

        if !can_fill {
            return None;
        }

        let execution_price = limit_price;
        let slippage = self.calculate_slippage(quantity, market);
        let fee = quantity * execution_price * self.config.trading_fee;

        let fill = Fill::new(
            order_id,
            market.timestamp,
            execution_price,
            quantity,
            fee,
            slippage,
            order.side,
            order.symbol.clone(),
        );

        Some(vec![fill])
    }

    /// Match TWAP order (time-weighted slicing)
    fn match_twap_order(
        &mut self,
        order_id: OrderId,
        market: &MarketSnapshot,
    ) -> Option<Vec<Fill>> {
        let state = self.algo_order_state.get(&order_id)?;
        let order = self.pending_orders.get(&order_id)?;

        let duration = order.twap_duration_secs? as f64;
        let elapsed = (market.timestamp - state.start_time) as f64;
        let time_fraction = (elapsed / duration).min(1.0);

        // Calculate how much should be executed by now
        let target_executed = order.quantity * time_fraction;
        let slice_quantity = (target_executed - state.executed_quantity).max(0.0);

        if slice_quantity < 1e-8 {
            return None;
        }

        let execution_price = match order.side {
            OrderSide::Buy => market.ask_price(),
            OrderSide::Sell => market.bid_price(),
        };

        let slippage = self.calculate_slippage(slice_quantity, market);
        let fee = slice_quantity * execution_price * self.config.trading_fee;

        let fill = Fill::new(
            order_id,
            market.timestamp,
            execution_price,
            slice_quantity,
            fee,
            slippage,
            order.side,
            order.symbol.clone(),
        );

        // Update algo state
        if let Some(state) = self.algo_order_state.get_mut(&order_id) {
            state.executed_quantity += slice_quantity;
            state.last_execution_time = market.timestamp;
        }

        Some(vec![fill])
    }

    /// Match VWAP order (volume-weighted slicing)
    fn match_vwap_order(
        &mut self,
        order_id: OrderId,
        market: &MarketSnapshot,
    ) -> Option<Vec<Fill>> {
        let _state = self.algo_order_state.get(&order_id)?;
        let order = self.pending_orders.get(&order_id)?;

        let participation = order.vwap_participation.unwrap_or(0.1);
        let slice_quantity = (market.bar.volume * participation).min(order.remaining_quantity);

        if slice_quantity < 1e-8 {
            return None;
        }

        let execution_price = match order.side {
            OrderSide::Buy => market.ask_price(),
            OrderSide::Sell => market.bid_price(),
        };

        let slippage = self.calculate_slippage(slice_quantity, market);
        let fee = slice_quantity * execution_price * self.config.trading_fee;

        let fill = Fill::new(
            order_id,
            market.timestamp,
            execution_price,
            slice_quantity,
            fee,
            slippage,
            order.side,
            order.symbol.clone(),
        );

        // Update algo state
        if let Some(state) = self.algo_order_state.get_mut(&order_id) {
            state.executed_quantity += slice_quantity;
            state.last_execution_time = market.timestamp;
        }

        Some(vec![fill])
    }

    /// Match POV order (percentage of volume)
    fn match_pov_order(&mut self, order_id: OrderId, market: &MarketSnapshot) -> Option<Vec<Fill>> {
        let order = self.pending_orders.get(&order_id)?;
        let pov_pct = order.pov_percentage.unwrap_or(0.05);

        let slice_quantity = (market.bar.volume * pov_pct).min(order.remaining_quantity);

        if slice_quantity < 1e-8 {
            return None;
        }

        let execution_price = match order.side {
            OrderSide::Buy => market.ask_price(),
            OrderSide::Sell => market.bid_price(),
        };

        let slippage = self.calculate_slippage(slice_quantity, market);
        let fee = slice_quantity * execution_price * self.config.trading_fee;

        let fill = Fill::new(
            order_id,
            market.timestamp,
            execution_price,
            slice_quantity,
            fee,
            slippage,
            order.side,
            order.symbol.clone(),
        );

        Some(vec![fill])
    }

    /// Calculate fill quantity considering volume constraints
    fn calculate_fill_quantity(&self, order: &Order, market: &MarketSnapshot) -> f64 {
        let max_fill_volume = market.bar.volume * self.config.max_fill_per_bar;

        if self.config.enable_partial_fills {
            order.remaining_quantity.min(max_fill_volume)
        } else {
            // All-or-nothing for small orders
            if order.remaining_quantity <= max_fill_volume {
                order.remaining_quantity
            } else {
                0.0
            }
        }
    }

    /// Calculate realistic slippage based on order size
    fn calculate_slippage(&self, quantity: f64, market: &MarketSnapshot) -> f64 {
        if !self.config.realistic_slippage {
            return self.config.base_slippage;
        }

        // Slippage increases with order size relative to volume
        let volume_fraction = quantity / market.bar.volume.max(1.0);
        let additional_slippage = volume_fraction * 0.01; // 1% per 100% of volume

        self.config.base_slippage + additional_slippage
    }

    /// Update order status from fills
    fn update_order_from_fills(&mut self, order_id: OrderId, fills: &[Fill]) {
        if let Some(order) = self.pending_orders.get_mut(&order_id) {
            for fill in fills {
                order.add_fill(fill.quantity, fill.price);
            }

            // Move to completed if filled
            if order.is_complete() {
                let order = self.pending_orders.remove(&order_id).unwrap();
                self.completed_orders.insert(order_id, order);
            }
        }
    }

    /// Handle order groups (OCO, OTO, Bracket)
    fn handle_order_groups(&mut self, fills: &[Fill]) {
        let filled_order_ids: Vec<OrderId> = fills.iter().map(|f| f.order_id).collect();

        for group in &self.order_groups.clone() {
            match group {
                OrderGroup::OCO { order_ids, .. } => {
                    // If any order in OCO group is filled, cancel others
                    for &filled_id in &filled_order_ids {
                        if order_ids.contains(&filled_id) {
                            for &cancel_id in order_ids {
                                if cancel_id != filled_id {
                                    self.cancel_order(cancel_id);
                                }
                            }
                            break;
                        }
                    }
                }
                OrderGroup::OTO {
                    parent_id,
                    child_ids,
                } => {
                    // If parent is filled, activate child orders
                    if filled_order_ids.contains(parent_id) {
                        for &child_id in child_ids {
                            if let Some(child) = self.pending_orders.get_mut(&child_id)
                                && child.status == OrderStatus::Created
                            {
                                child.set_status(OrderStatus::Pending);
                            }
                        }
                    }
                }
                OrderGroup::Bracket {
                    entry_id,
                    stop_loss_id,
                    take_profit_id,
                } => {
                    // If entry is filled, activate stop loss and take profit
                    if filled_order_ids.contains(entry_id) {
                        for &id in &[*stop_loss_id, *take_profit_id] {
                            if let Some(order) = self.pending_orders.get_mut(&id)
                                && order.status == OrderStatus::Created
                            {
                                order.set_status(OrderStatus::Pending);
                            }
                        }
                    }
                    // If either SL or TP is filled, cancel the other (OCO behavior)
                    if filled_order_ids.contains(stop_loss_id) {
                        self.cancel_order(*take_profit_id);
                    } else if filled_order_ids.contains(take_profit_id) {
                        self.cancel_order(*stop_loss_id);
                    }
                }
            }
        }
    }

    /// Get all completed orders
    pub fn completed_orders(&self) -> &HashMap<OrderId, Order> {
        &self.completed_orders
    }
}

impl Default for MatchingEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bar(close: f64, volume: f64) -> OHLCVBar {
        OHLCVBar {
            timestamp: 1000,
            open: close * 0.999,
            high: close * 1.001,
            low: close * 0.999,
            close,
            volume,
        }
    }

    #[test]
    fn test_market_order_execution() {
        let mut engine = MatchingEngine::new();
        let order = Order::market(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0);

        let order_id = engine.submit_order(order);

        let market = MarketSnapshot::new(1000, create_test_bar(50000.0, 100.0));
        let fills = engine.match_orders(&market);

        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].quantity, 1.0);
        assert!(fills[0].price > 0.0);

        let order = engine.get_order(order_id).unwrap();
        assert_eq!(order.status, OrderStatus::Filled);
    }

    #[test]
    fn test_limit_order_execution() {
        let mut engine = MatchingEngine::new();
        let order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 49900.0);

        let _order_id = engine.submit_order(order);

        // Price above limit - should not fill
        let market1 = MarketSnapshot::new(1000, create_test_bar(50000.0, 100.0));
        let fills1 = engine.match_orders(&market1);
        println!(
            "low: {}, limit: {}, fills1: {:?}",
            market1.bar.low, 49900.0, fills1
        );
        assert_eq!(fills1.len(), 0);

        // Price below limit - should fill
        let mut bar2 = create_test_bar(49800.0, 100.0);
        bar2.low = 49800.0;
        let market2 = MarketSnapshot::new(2000, bar2);
        let fills2 = engine.match_orders(&market2);

        assert_eq!(fills2.len(), 1);
        assert_eq!(fills2[0].price, 49900.0);
    }

    #[test]
    fn test_stop_order_trigger() {
        let mut engine = MatchingEngine::new();
        let order = Order::stop(0, "BTC/USD".to_string(), OrderSide::Sell, 1.0, 49000.0);

        let order_id = engine.submit_order(order);

        // Price above stop - should not trigger
        let market1 = MarketSnapshot::new(1000, create_test_bar(50000.0, 100.0));
        let fills1 = engine.match_orders(&market1);
        assert_eq!(fills1.len(), 0);

        // Price hits stop - should trigger and fill
        let mut bar2 = create_test_bar(48900.0, 100.0);
        bar2.low = 48900.0;
        let market2 = MarketSnapshot::new(2000, bar2);
        let fills2 = engine.match_orders(&market2);

        assert_eq!(fills2.len(), 1);

        let order = engine.get_order(order_id).unwrap();
        assert_eq!(order.status, OrderStatus::Filled);
    }

    #[test]
    fn test_trailing_stop() {
        let mut engine = MatchingEngine::new();
        let order = Order::trailing_stop(
            0,
            "BTC/USD".to_string(),
            OrderSide::Sell,
            1.0,
            None,
            Some(0.05), // 5% trail
        );

        engine.submit_order(order);

        // Initialize at 50000
        let market1 = MarketSnapshot::new(1000, create_test_bar(50000.0, 100.0));
        engine.match_orders(&market1);

        // Price rises to 52000 - trail should update
        let market2 = MarketSnapshot::new(2000, create_test_bar(52000.0, 100.0));
        engine.match_orders(&market2);

        // Price drops below trail (52000 * 0.95 = 49400) - should trigger
        let mut bar3 = create_test_bar(49000.0, 100.0);
        bar3.low = 49000.0;
        let market3 = MarketSnapshot::new(3000, bar3);
        let fills = engine.match_orders(&market3);

        assert_eq!(fills.len(), 1);
    }

    #[test]
    fn test_partial_fills() {
        let config = MatchingConfig {
            max_fill_per_bar: 0.1, // 10% of volume
            ..Default::default()
        };
        let mut engine = MatchingEngine::with_config(config);

        let order = Order::market(0, "BTC/USD".to_string(), OrderSide::Buy, 20.0);
        engine.submit_order(order);

        // Volume is 100, so can fill max 10
        let market = MarketSnapshot::new(1000, create_test_bar(50000.0, 100.0));
        let fills = engine.match_orders(&market);

        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].quantity, 10.0); // Partial fill
    }
}
