//! Order types and order management for backtesting
//!
//! Implements 12+ order types matching LEAN/QuantConnect capabilities:
//! - Basic: Market, Limit, Stop, Stop-Limit
//! - Session-based: MOO, MOC, LOO, LOC
//! - Dynamic: Trailing Stop, Trailing Stop-Limit
//! - Advanced: Iceberg, OCO, OTO, Bracket
//! - Algorithmic: TWAP, VWAP, POV
//!
//! # Architecture
//!
//! ```text
//! Strategy → SubmitOrder → OrderManager → Pending Queue
//!                                ↓
//! Market Data → MatchingEngine → Fills → Portfolio Update
//!                                ↓
//! Order Lifecycle (cancellations, expiry, triggers)
//! ```

use serde::{Deserialize, Serialize};

/// Unique order identifier
pub type OrderId = u64;

/// Order type enumeration - supports 12+ order types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    /// Market order - execute immediately at best available price
    Market,

    /// Limit order - execute only at limit price or better
    Limit,

    /// Stop order - becomes market order when stop price is hit
    Stop,

    /// Stop-Limit order - becomes limit order when stop price is hit
    StopLimit,

    /// Market-on-Open - execute at market open price
    MarketOnOpen,

    /// Market-on-Close - execute at market close price
    MarketOnClose,

    /// Limit-on-Open - limit order at market open
    LimitOnOpen,

    /// Limit-on-Close - limit order at market close
    LimitOnClose,

    /// Trailing stop - stop price follows market (absolute or percentage)
    TrailingStop,

    /// Trailing stop-limit - trailing stop that becomes limit order
    TrailingStopLimit,

    /// Iceberg order - only show partial quantity
    Iceberg,

    /// TWAP (Time-Weighted Average Price) - split over time
    TWAP,

    /// VWAP (Volume-Weighted Average Price) - split by volume
    VWAP,

    /// POV (Percentage of Volume) - participate at percentage of market volume
    POV,
}

/// Order side - buy or sell
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order status lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    /// Order created but not yet submitted
    Created,

    /// Order submitted and pending execution
    Pending,

    /// Order partially filled
    PartiallyFilled,

    /// Order completely filled
    Filled,

    /// Order cancelled by user or system
    Cancelled,

    /// Order rejected (invalid parameters, insufficient funds, etc.)
    Rejected,

    /// Order expired (time-in-force reached)
    Expired,

    /// Order triggered (stop/trailing orders)
    Triggered,
}

/// Time-in-force specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TimeInForce {
    /// Good Till Cancelled - remains active until filled or cancelled
    #[default]
    GTC,

    /// Good Till Date - remains active until specified date
    GTD { expiry: i64 }, // Unix timestamp

    /// Day order - expires at end of trading day
    Day,

    /// Immediate or Cancel - fill immediately or cancel remaining
    IOC,

    /// Fill or Kill - fill entire order immediately or cancel
    FOK,
}

/// Order parameters for different order types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Order {
    /// Unique order ID
    pub id: OrderId,

    /// Order type
    pub order_type: OrderType,

    /// Buy or sell
    pub side: OrderSide,

    /// Order quantity
    pub quantity: f64,

    /// Limit price (for limit orders)
    pub limit_price: Option<f64>,

    /// Stop price (for stop orders)
    pub stop_price: Option<f64>,

    /// Trailing amount (absolute price units for trailing orders)
    pub trailing_amount: Option<f64>,

    /// Trailing percentage (0.0-1.0 for trailing orders)
    pub trailing_percent: Option<f64>,

    /// Time-in-force specification
    pub time_in_force: TimeInForce,

    /// Order status
    pub status: OrderStatus,

    /// Creation timestamp
    pub created_at: i64,

    /// Last update timestamp
    pub updated_at: i64,

    /// Filled quantity
    pub filled_quantity: f64,

    /// Remaining quantity
    pub remaining_quantity: f64,

    /// Average fill price
    pub average_fill_price: f64,

    /// Symbol/asset being traded
    pub symbol: String,

    /// Parent order ID (for child orders in OCO, OTO, Bracket)
    pub parent_order_id: Option<OrderId>,

    /// Child order IDs (for parent orders)
    pub child_order_ids: Vec<OrderId>,

    /// Iceberg visible quantity (for iceberg orders)
    pub iceberg_visible_qty: Option<f64>,

    /// TWAP duration in seconds (for TWAP orders)
    pub twap_duration_secs: Option<i64>,

    /// VWAP target participation rate (0.0-1.0)
    pub vwap_participation: Option<f64>,

    /// POV target percentage (0.0-1.0)
    pub pov_percentage: Option<f64>,

    /// High water mark for trailing stops (tracks best price)
    pub trailing_high_water_mark: Option<f64>,
}

impl Order {
    /// Create a new market order
    pub fn market(id: OrderId, symbol: String, side: OrderSide, quantity: f64) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id,
            order_type: OrderType::Market,
            side,
            quantity,
            limit_price: None,
            stop_price: None,
            trailing_amount: None,
            trailing_percent: None,
            time_in_force: TimeInForce::GTC,
            status: OrderStatus::Pending,
            created_at: now,
            updated_at: now,
            filled_quantity: 0.0,
            remaining_quantity: quantity,
            average_fill_price: 0.0,
            symbol,
            parent_order_id: None,
            child_order_ids: Vec::new(),
            iceberg_visible_qty: None,
            twap_duration_secs: None,
            vwap_participation: None,
            pov_percentage: None,
            trailing_high_water_mark: None,
        }
    }

    /// Create a new limit order
    pub fn limit(
        id: OrderId,
        symbol: String,
        side: OrderSide,
        quantity: f64,
        limit_price: f64,
    ) -> Self {
        let mut order = Self::market(id, symbol, side, quantity);
        order.order_type = OrderType::Limit;
        order.limit_price = Some(limit_price);
        order
    }

    /// Create a new stop order
    pub fn stop(
        id: OrderId,
        symbol: String,
        side: OrderSide,
        quantity: f64,
        stop_price: f64,
    ) -> Self {
        let mut order = Self::market(id, symbol, side, quantity);
        order.order_type = OrderType::Stop;
        order.stop_price = Some(stop_price);
        order
    }

    /// Create a new stop-limit order
    pub fn stop_limit(
        id: OrderId,
        symbol: String,
        side: OrderSide,
        quantity: f64,
        stop_price: f64,
        limit_price: f64,
    ) -> Self {
        let mut order = Self::market(id, symbol, side, quantity);
        order.order_type = OrderType::StopLimit;
        order.stop_price = Some(stop_price);
        order.limit_price = Some(limit_price);
        order
    }

    /// Create a new trailing stop order
    pub fn trailing_stop(
        id: OrderId,
        symbol: String,
        side: OrderSide,
        quantity: f64,
        trailing_amount: Option<f64>,
        trailing_percent: Option<f64>,
    ) -> Self {
        let mut order = Self::market(id, symbol, side, quantity);
        order.order_type = OrderType::TrailingStop;
        order.trailing_amount = trailing_amount;
        order.trailing_percent = trailing_percent;
        order
    }

    /// Create a new iceberg order
    pub fn iceberg(
        id: OrderId,
        symbol: String,
        side: OrderSide,
        quantity: f64,
        limit_price: f64,
        visible_qty: f64,
    ) -> Self {
        let mut order = Self::limit(id, symbol, side, quantity, limit_price);
        order.order_type = OrderType::Iceberg;
        order.iceberg_visible_qty = Some(visible_qty);
        order
    }

    /// Create a new TWAP order
    pub fn twap(
        id: OrderId,
        symbol: String,
        side: OrderSide,
        quantity: f64,
        duration_secs: i64,
    ) -> Self {
        let mut order = Self::market(id, symbol, side, quantity);
        order.order_type = OrderType::TWAP;
        order.twap_duration_secs = Some(duration_secs);
        order
    }

    /// Create a new VWAP order
    pub fn vwap(
        id: OrderId,
        symbol: String,
        side: OrderSide,
        quantity: f64,
        participation: f64,
    ) -> Self {
        let mut order = Self::market(id, symbol, side, quantity);
        order.order_type = OrderType::VWAP;
        order.vwap_participation = Some(participation);
        order
    }

    /// Create a new POV order
    pub fn pov(
        id: OrderId,
        symbol: String,
        side: OrderSide,
        quantity: f64,
        percentage: f64,
    ) -> Self {
        let mut order = Self::market(id, symbol, side, quantity);
        order.order_type = OrderType::POV;
        order.pov_percentage = Some(percentage);
        order
    }

    /// Update order status
    pub fn set_status(&mut self, status: OrderStatus) {
        self.status = status;
        self.updated_at = chrono::Utc::now().timestamp();
    }

    /// Record a partial fill
    pub fn add_fill(&mut self, quantity: f64, price: f64) {
        let total_filled = self.filled_quantity + quantity;
        // Update average fill price using weighted average
        self.average_fill_price =
            (self.average_fill_price * self.filled_quantity + price * quantity) / total_filled;

        self.filled_quantity = total_filled;
        self.remaining_quantity = self.quantity - total_filled;
        self.updated_at = chrono::Utc::now().timestamp();

        // Update status
        if self.remaining_quantity <= 1e-8 {
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartiallyFilled;
        }
    }

    /// Check if order is active (can be matched)
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Pending | OrderStatus::PartiallyFilled | OrderStatus::Triggered
        )
    }

    /// Check if order is complete (filled or terminated)
    pub fn is_complete(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Filled
                | OrderStatus::Cancelled
                | OrderStatus::Rejected
                | OrderStatus::Expired
        )
    }

    /// Update trailing stop high water mark
    pub fn update_trailing_high_water_mark(&mut self, current_price: f64) {
        match self.trailing_high_water_mark {
            Some(hwm) => {
                if (self.side == OrderSide::Sell && current_price > hwm)
                    || (self.side == OrderSide::Buy && current_price < hwm)
                {
                    self.trailing_high_water_mark = Some(current_price);
                }
            }
            None => {
                self.trailing_high_water_mark = Some(current_price);
            }
        }
    }

    /// Calculate current trailing stop price
    pub fn calculate_trailing_stop_price(&self) -> Option<f64> {
        let hwm = self.trailing_high_water_mark?;

        if let Some(trailing_pct) = self.trailing_percent {
            // Percentage-based trailing
            match self.side {
                OrderSide::Sell => Some(hwm * (1.0 - trailing_pct)),
                OrderSide::Buy => Some(hwm * (1.0 + trailing_pct)),
            }
        } else if let Some(trailing_amt) = self.trailing_amount {
            // Absolute amount-based trailing
            match self.side {
                OrderSide::Sell => Some(hwm - trailing_amt),
                OrderSide::Buy => Some(hwm + trailing_amt),
            }
        } else {
            None
        }
    }
}

/// Order fill record
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fill {
    /// Order ID that was filled
    pub order_id: OrderId,

    /// Fill timestamp
    pub timestamp: i64,

    /// Fill price
    pub price: f64,

    /// Fill quantity
    pub quantity: f64,

    /// Trading fee
    pub fee: f64,

    /// Slippage applied
    pub slippage: f64,

    /// Order side
    pub side: OrderSide,

    /// Symbol
    pub symbol: String,
}

impl Fill {
    /// Create new fill record
    #[allow(clippy::too_many_arguments)] // public API: signature is documented and used by callers
    pub fn new(
        order_id: OrderId,
        timestamp: i64,
        price: f64,
        quantity: f64,
        fee: f64,
        slippage: f64,
        side: OrderSide,
        symbol: String,
    ) -> Self {
        Self {
            order_id,
            timestamp,
            price,
            quantity,
            fee,
            slippage,
            side,
            symbol,
        }
    }
}

/// Complex order groups
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderGroup {
    /// One-Cancels-Other: When one order fills, cancel the others
    OCO {
        group_id: u64,
        order_ids: Vec<OrderId>,
    },

    /// One-Triggers-Other: When parent fills, submit child orders
    OTO {
        parent_id: OrderId,
        child_ids: Vec<OrderId>,
    },

    /// Bracket order: Entry + stop loss + take profit
    Bracket {
        entry_id: OrderId,
        stop_loss_id: OrderId,
        take_profit_id: OrderId,
    },
}

impl OrderGroup {
    /// Create OCO group
    pub fn oco(group_id: u64, order_ids: Vec<OrderId>) -> Self {
        Self::OCO {
            group_id,
            order_ids,
        }
    }

    /// Create OTO group
    pub fn oto(parent_id: OrderId, child_ids: Vec<OrderId>) -> Self {
        Self::OTO {
            parent_id,
            child_ids,
        }
    }

    /// Create bracket order group
    pub fn bracket(entry_id: OrderId, stop_loss_id: OrderId, take_profit_id: OrderId) -> Self {
        Self::Bracket {
            entry_id,
            stop_loss_id,
            take_profit_id,
        }
    }
}

/// Order builder for ergonomic order creation
pub struct OrderBuilder {
    id: OrderId,
    symbol: String,
    side: OrderSide,
    quantity: f64,
    order_type: OrderType,
    limit_price: Option<f64>,
    stop_price: Option<f64>,
    trailing_amount: Option<f64>,
    trailing_percent: Option<f64>,
    time_in_force: TimeInForce,
    iceberg_visible_qty: Option<f64>,
    twap_duration_secs: Option<i64>,
    vwap_participation: Option<f64>,
    pov_percentage: Option<f64>,
}

impl OrderBuilder {
    /// Create new order builder
    pub fn new(id: OrderId, symbol: String, side: OrderSide, quantity: f64) -> Self {
        Self {
            id,
            symbol,
            side,
            quantity,
            order_type: OrderType::Market,
            limit_price: None,
            stop_price: None,
            trailing_amount: None,
            trailing_percent: None,
            time_in_force: TimeInForce::GTC,
            iceberg_visible_qty: None,
            twap_duration_secs: None,
            vwap_participation: None,
            pov_percentage: None,
        }
    }

    /// Set order type to market
    pub fn market(mut self) -> Self {
        self.order_type = OrderType::Market;
        self
    }

    /// Set order type to limit
    pub fn limit(mut self, price: f64) -> Self {
        self.order_type = OrderType::Limit;
        self.limit_price = Some(price);
        self
    }

    /// Set order type to stop
    pub fn stop(mut self, price: f64) -> Self {
        self.order_type = OrderType::Stop;
        self.stop_price = Some(price);
        self
    }

    /// Set time-in-force
    pub fn tif(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }

    /// Build the order
    pub fn build(self) -> Order {
        let now = chrono::Utc::now().timestamp();
        Order {
            id: self.id,
            order_type: self.order_type,
            side: self.side,
            quantity: self.quantity,
            limit_price: self.limit_price,
            stop_price: self.stop_price,
            trailing_amount: self.trailing_amount,
            trailing_percent: self.trailing_percent,
            time_in_force: self.time_in_force,
            status: OrderStatus::Pending,
            created_at: now,
            updated_at: now,
            filled_quantity: 0.0,
            remaining_quantity: self.quantity,
            average_fill_price: 0.0,
            symbol: self.symbol,
            parent_order_id: None,
            child_order_ids: Vec::new(),
            iceberg_visible_qty: self.iceberg_visible_qty,
            twap_duration_secs: self.twap_duration_secs,
            vwap_participation: self.vwap_participation,
            pov_percentage: self.pov_percentage,
            trailing_high_water_mark: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_order_creation() {
        let order = Order::market(1, "BTC/USD".to_string(), OrderSide::Buy, 1.0);
        assert_eq!(order.order_type, OrderType::Market);
        assert_eq!(order.quantity, 1.0);
        assert_eq!(order.remaining_quantity, 1.0);
        assert!(order.is_active());
    }

    #[test]
    fn test_limit_order_creation() {
        let order = Order::limit(1, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 50000.0);
        assert_eq!(order.order_type, OrderType::Limit);
        assert_eq!(order.limit_price, Some(50000.0));
    }

    #[test]
    fn test_partial_fill() {
        let mut order = Order::market(1, "BTC/USD".to_string(), OrderSide::Buy, 10.0);

        order.add_fill(3.0, 50000.0);
        assert_eq!(order.filled_quantity, 3.0);
        assert_eq!(order.remaining_quantity, 7.0);
        assert_eq!(order.status, OrderStatus::PartiallyFilled);

        order.add_fill(7.0, 50100.0);
        assert_eq!(order.filled_quantity, 10.0);
        assert_eq!(order.remaining_quantity, 0.0);
        assert_eq!(order.status, OrderStatus::Filled);
        assert!(order.is_complete());
    }

    #[test]
    fn test_average_fill_price() {
        let mut order = Order::market(1, "BTC/USD".to_string(), OrderSide::Buy, 10.0);

        order.add_fill(5.0, 50000.0);
        assert_eq!(order.average_fill_price, 50000.0);

        order.add_fill(5.0, 50100.0);
        assert_eq!(order.average_fill_price, 50050.0); // (50000*5 + 50100*5) / 10
    }

    #[test]
    fn test_trailing_stop_percentage() {
        let mut order = Order::trailing_stop(
            1,
            "BTC/USD".to_string(),
            OrderSide::Sell,
            1.0,
            None,
            Some(0.05), // 5% trailing
        );

        // Initialize high water mark
        order.update_trailing_high_water_mark(50000.0);
        assert_eq!(order.trailing_high_water_mark, Some(50000.0));

        // Price rises - update HWM
        order.update_trailing_high_water_mark(51000.0);
        assert_eq!(order.trailing_high_water_mark, Some(51000.0));

        // Calculate trailing stop
        let stop_price = order.calculate_trailing_stop_price().unwrap();
        assert!((stop_price - 48450.0).abs() < 1.0); // 51000 * 0.95
    }

    #[test]
    fn test_order_builder() {
        let order = OrderBuilder::new(1, "BTC/USD".to_string(), OrderSide::Buy, 1.0)
            .limit(50000.0)
            .tif(TimeInForce::Day)
            .build();

        assert_eq!(order.order_type, OrderType::Limit);
        assert_eq!(order.limit_price, Some(50000.0));
        assert_eq!(order.time_in_force, TimeInForce::Day);
    }
}
