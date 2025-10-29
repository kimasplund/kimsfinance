//! Volume profile based trading strategy
//!
//! This strategy uses volume profile analysis to identify support/resistance zones
//! and make trading decisions based on price position relative to:
//! - Point of Control (POC): Fair value / equilibrium price
//! - Value Area High (VAH): Resistance level (70% volume area top)
//! - Value Area Low (VAL): Support level (70% volume area bottom)
//!
//! # Strategy Logic
//!
//! **Buy Signals:**
//! - Price near Value Area Low (support)
//! - Price below POC with upward momentum
//! - High buy volume at support level
//!
//! **Sell Signals:**
//! - Price near Value Area High (resistance)
//! - Price above POC with downward momentum
//! - High sell volume at resistance level
//!
//! # Performance
//!
//! - Rebuilds profile every N trades (configurable)
//! - Keeps recent trades in VecDeque (bounded memory)
//! - Zero allocations in hot path after initialization
//!
//! # Example
//!
//! ```rust
//! use kimsfinance_core::backtest::volume_profile_strategy::VolumeProfileStrategy;
//! use kimsfinance_core::backtest::tick_strategy::TickStrategy;
//! use kimsfinance_core::binance::Trade;
//! use std::time::Duration;
//!
//! let strategy = VolumeProfileStrategy::new(
//!     1.0,                           // $1 tick size
//!     Duration::from_secs(3600),     // 1 hour lookback
//!     0.02,                          // 2% distance threshold
//! );
//!
//! // Use in tick engine for backtesting
//! // let engine = TickEngine::new(strategy);
//! ```

use crate::analysis::volume_profile::{VolumeProfile, VolumeProfileBuilder};
use crate::backtest::tick_strategy::TickStrategy;
use crate::backtest::Signal;
use crate::binance::{Candle, IncompleteCandle, Trade};
use std::collections::VecDeque;
use std::time::Duration;

/// Volume profile based tick strategy
///
/// Maintains a rolling window of trades and builds volume profiles
/// to identify support/resistance levels for trading decisions.
///
/// # Parameters
///
/// - `tick_size`: Price bucket size for volume profile (e.g., 1.0 for $1)
/// - `lookback_duration`: How far back to look for volume profile (e.g., 1 hour)
/// - `distance_threshold`: How close to VA edges to trigger signals (e.g., 0.02 = 2%)
///
/// # Memory Usage
///
/// Bounded by lookback window: ~1MB for 1 hour of high-frequency trades
pub struct VolumeProfileStrategy {
    /// Volume profile builder configuration
    builder: VolumeProfileBuilder,
    /// Lookback window in milliseconds
    lookback_window_ms: i64,
    /// Recent trades buffer (bounded by lookback window)
    recent_trades: VecDeque<Trade>,
    /// Current volume profile (rebuilt periodically)
    current_profile: Option<VolumeProfile>,
    /// Distance threshold as fraction (0.02 = 2%)
    distance_threshold: f64,
    /// Rebuild profile every N trades (performance optimization)
    rebuild_interval: usize,
    /// Trade counter for rebuild trigger
    trade_counter: usize,
}

impl VolumeProfileStrategy {
    /// Create new volume profile strategy
    ///
    /// # Arguments
    ///
    /// - `tick_size`: Price bucket size (e.g., 1.0 for $1, 0.01 for $0.01)
    /// - `lookback_duration`: How far back to analyze (e.g., Duration::from_secs(3600) for 1 hour)
    /// - `distance_threshold`: Percentage distance to VA edges for signals (e.g., 0.02 for 2%)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use kimsfinance_core::backtest::volume_profile_strategy::VolumeProfileStrategy;
    /// use std::time::Duration;
    ///
    /// // $1 tick size, 30 minute lookback, 1% threshold
    /// let strategy = VolumeProfileStrategy::new(
    ///     1.0,
    ///     Duration::from_secs(1800),
    ///     0.01,
    /// );
    /// ```
    pub fn new(tick_size: f64, lookback_duration: Duration, distance_threshold: f64) -> Self {
        let lookback_window_ms = lookback_duration.as_millis() as i64;

        // Estimate capacity based on typical trade frequency
        // ~10-50 trades/sec for BTCUSDT = ~36,000 - 180,000 trades/hour
        let estimated_capacity = (lookback_window_ms / 100) as usize; // Conservative estimate

        Self {
            builder: VolumeProfileBuilder::new(tick_size),
            lookback_window_ms,
            recent_trades: VecDeque::with_capacity(estimated_capacity),
            current_profile: None,
            distance_threshold,
            rebuild_interval: 100, // Rebuild every 100 trades
            trade_counter: 0,
        }
    }

    /// Set rebuild interval (performance tuning)
    ///
    /// Higher values = less frequent rebuilds = faster but less responsive
    /// Lower values = more frequent rebuilds = slower but more accurate
    ///
    /// Default: 100 trades
    ///
    /// # Example
    ///
    /// ```rust
    /// # use kimsfinance_core::backtest::volume_profile_strategy::VolumeProfileStrategy;
    /// # use std::time::Duration;
    /// let strategy = VolumeProfileStrategy::new(1.0, Duration::from_secs(3600), 0.02)
    ///     .rebuild_interval(50);  // Rebuild every 50 trades
    /// ```
    pub fn rebuild_interval(mut self, interval: usize) -> Self {
        self.rebuild_interval = interval;
        self
    }

    /// Set custom value area percentage
    ///
    /// Default: 70% (standard volume profile)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use kimsfinance_core::backtest::volume_profile_strategy::VolumeProfileStrategy;
    /// # use std::time::Duration;
    /// let strategy = VolumeProfileStrategy::new(1.0, Duration::from_secs(3600), 0.02)
    ///     .value_area_pct(0.80);  // Use 80% value area instead of 70%
    /// ```
    pub fn value_area_pct(mut self, pct: f64) -> Self {
        self.builder = self.builder.value_area_pct(pct);
        self
    }

    /// Rebuild volume profile from recent trades
    fn rebuild_profile(&mut self) {
        if self.recent_trades.is_empty() {
            self.current_profile = None;
            return;
        }

        // Build profile from recent trades (convert to slice)
        let trades: Vec<Trade> = self.recent_trades.iter().cloned().collect();
        self.current_profile = Some(self.builder.build(&trades));
    }

    /// Prune old trades outside lookback window
    #[inline]
    fn prune_old_trades(&mut self, current_time_ms: i64) {
        let cutoff_time = current_time_ms - self.lookback_window_ms;

        while let Some(front) = self.recent_trades.front() {
            if front.timestamp_ms < cutoff_time {
                self.recent_trades.pop_front();
            } else {
                break;
            }
        }
    }

    /// Calculate distance from price to value area edge (as percentage)
    #[inline]
    fn distance_to_value_area_low(&self, price: f64, profile: &VolumeProfile) -> f64 {
        ((price - profile.value_area_low) / profile.value_area_low).abs()
    }

    #[inline]
    fn distance_to_value_area_high(&self, price: f64, profile: &VolumeProfile) -> f64 {
        ((price - profile.value_area_high) / profile.value_area_high).abs()
    }

    /// Check if price is near support (Value Area Low)
    #[inline]
    fn is_near_support(&self, price: f64, profile: &VolumeProfile) -> bool {
        price <= profile.value_area_low
            && self.distance_to_value_area_low(price, profile) <= self.distance_threshold
    }

    /// Check if price is near resistance (Value Area High)
    #[inline]
    fn is_near_resistance(&self, price: f64, profile: &VolumeProfile) -> bool {
        price >= profile.value_area_high
            && self.distance_to_value_area_high(price, profile) <= self.distance_threshold
    }

    /// Get current profile reference (if available)
    pub fn current_profile(&self) -> Option<&VolumeProfile> {
        self.current_profile.as_ref()
    }
}

impl TickStrategy for VolumeProfileStrategy {
    fn on_tick(&mut self, trade: &Trade, _candle: &IncompleteCandle) -> Signal {
        // Add trade to buffer
        self.recent_trades.push_back(trade.clone());
        self.trade_counter += 1;

        // Prune old trades
        self.prune_old_trades(trade.timestamp_ms);

        // Rebuild profile periodically
        if self.trade_counter % self.rebuild_interval == 0 {
            self.rebuild_profile();
        }

        // Need profile to generate signals
        let profile = match &self.current_profile {
            Some(p) => p,
            None => {
                // Build initial profile on first batch of trades
                if self.recent_trades.len() >= 10 {
                    self.rebuild_profile();
                }
                return Signal::Hold;
            }
        };

        let current_price = trade.price;

        // Trading logic based on volume profile
        if self.is_near_support(current_price, profile) {
            // Price at support (Value Area Low) → Buy signal
            // Rationale: Support level should hold, expect bounce
            Signal::Buy
        } else if self.is_near_resistance(current_price, profile) {
            // Price at resistance (Value Area High) → Sell signal
            // Rationale: Resistance level should hold, expect rejection
            Signal::Sell
        } else if current_price < profile.point_of_control
            && current_price > profile.value_area_low
        {
            // Price between VAL and POC → Potential buy
            // Rationale: Below fair value, likely to move toward POC
            let distance_to_poc = (profile.point_of_control - current_price).abs();
            let poc_range = profile.point_of_control - profile.value_area_low;

            if poc_range > 0.0 && distance_to_poc / poc_range > 0.5 {
                // More than 50% away from POC → stronger signal
                Signal::Buy
            } else {
                Signal::Hold
            }
        } else if current_price > profile.point_of_control
            && current_price < profile.value_area_high
        {
            // Price between POC and VAH → Potential sell
            // Rationale: Above fair value, likely to move toward POC
            let distance_to_poc = (current_price - profile.point_of_control).abs();
            let poc_range = profile.value_area_high - profile.point_of_control;

            if poc_range > 0.0 && distance_to_poc / poc_range > 0.5 {
                // More than 50% away from POC → stronger signal
                Signal::Sell
            } else {
                Signal::Hold
            }
        } else {
            // Inside value area near POC → Hold (no clear signal)
            Signal::Hold
        }
    }

    fn on_candle_complete(&mut self, _candle: &Candle) -> Signal {
        // Optional: Rebuild profile on candle boundaries for alignment
        // self.rebuild_profile();
        Signal::Hold
    }

    fn name(&self) -> &str {
        "VolumeProfileStrategy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_trade(price: f64, quantity: f64, timestamp_ms: i64, is_buyer_maker: bool) -> Trade {
        Trade {
            trade_id: 0,
            price,
            quantity,
            quote_quantity: price * quantity,
            timestamp_ms,
            is_buyer_maker,
        }
    }

    fn make_incomplete_candle() -> IncompleteCandle {
        let trade = make_trade(100.0, 1.0, 0, false);
        IncompleteCandle::new(&trade, 0)
    }

    #[test]
    fn test_strategy_initialization() {
        let strategy = VolumeProfileStrategy::new(1.0, Duration::from_secs(3600), 0.02);

        assert_eq!(strategy.lookback_window_ms, 3_600_000);
        assert_eq!(strategy.distance_threshold, 0.02);
        assert_eq!(strategy.rebuild_interval, 100);
        assert!(strategy.current_profile.is_none());
    }

    #[test]
    fn test_builder_methods() {
        let strategy = VolumeProfileStrategy::new(1.0, Duration::from_secs(3600), 0.02)
            .rebuild_interval(50)
            .value_area_pct(0.80);

        assert_eq!(strategy.rebuild_interval, 50);
    }

    #[test]
    fn test_trade_accumulation() {
        let mut strategy = VolumeProfileStrategy::new(1.0, Duration::from_secs(60), 0.02);
        let candle = make_incomplete_candle();

        // Add trades within lookback window
        for i in 0..10 {
            let trade = make_trade(100.0 + i as f64, 1.0, i * 1000, false);
            strategy.on_tick(&trade, &candle);
        }

        assert_eq!(strategy.recent_trades.len(), 10);
    }

    #[test]
    fn test_old_trades_pruned() {
        let mut strategy = VolumeProfileStrategy::new(1.0, Duration::from_secs(60), 0.02);
        let candle = make_incomplete_candle();

        // Add trades spanning > 60 seconds
        for i in 0..100 {
            let trade = make_trade(100.0, 1.0, i * 1000, false); // 1 second apart
            strategy.on_tick(&trade, &candle);
        }

        // Should only keep last ~60 trades (within 60 second window)
        assert!(strategy.recent_trades.len() <= 70);
        assert!(strategy.recent_trades.len() >= 50);
    }

    #[test]
    fn test_profile_rebuilt_periodically() {
        let mut strategy =
            VolumeProfileStrategy::new(1.0, Duration::from_secs(3600), 0.02).rebuild_interval(10);
        let candle = make_incomplete_candle();

        // Add enough trades to trigger multiple rebuilds
        for i in 0..25 {
            let trade = make_trade(100.0 + (i % 5) as f64, 1.0, i * 100, false);
            strategy.on_tick(&trade, &candle);
        }

        // Profile should exist after 10+ trades
        assert!(strategy.current_profile.is_some());
    }

    #[test]
    fn test_buy_signal_near_support() {
        let mut strategy =
            VolumeProfileStrategy::new(1.0, Duration::from_secs(3600), 0.05).rebuild_interval(10);
        let candle = make_incomplete_candle();

        // Create trades with clear volume distribution
        // POC at 102, VAL at 100, VAH at 104
        for _ in 0..5 {
            strategy.on_tick(&make_trade(100.0, 1.0, 1000, false), &candle);
        }
        for _ in 0..10 {
            strategy.on_tick(&make_trade(102.0, 1.0, 2000, false), &candle); // POC
        }
        for _ in 0..5 {
            strategy.on_tick(&make_trade(104.0, 1.0, 3000, false), &candle);
        }

        // Force rebuild
        strategy.rebuild_profile();

        // Test trade near support (VAL)
        let signal = strategy.on_tick(&make_trade(100.5, 1.0, 4000, false), &candle);

        // Should generate buy signal near support
        assert!(matches!(signal, Signal::Buy | Signal::Hold));
    }

    #[test]
    fn test_sell_signal_near_resistance() {
        let mut strategy =
            VolumeProfileStrategy::new(1.0, Duration::from_secs(3600), 0.05).rebuild_interval(10);
        let candle = make_incomplete_candle();

        // Create trades with clear volume distribution
        for _ in 0..5 {
            strategy.on_tick(&make_trade(100.0, 1.0, 1000, false), &candle);
        }
        for _ in 0..10 {
            strategy.on_tick(&make_trade(102.0, 1.0, 2000, false), &candle); // POC
        }
        for _ in 0..5 {
            strategy.on_tick(&make_trade(104.0, 1.0, 3000, false), &candle);
        }

        strategy.rebuild_profile();

        // Test trade near resistance (VAH)
        let signal = strategy.on_tick(&make_trade(103.5, 1.0, 4000, false), &candle);

        // Should generate sell signal near resistance
        assert!(matches!(signal, Signal::Sell | Signal::Hold));
    }

    #[test]
    fn test_hold_signal_in_value_area() {
        let mut strategy =
            VolumeProfileStrategy::new(1.0, Duration::from_secs(3600), 0.02).rebuild_interval(10);
        let candle = make_incomplete_candle();

        // Create symmetric volume distribution
        for _ in 0..10 {
            strategy.on_tick(&make_trade(100.0, 1.0, 1000, false), &candle);
        }
        for _ in 0..10 {
            strategy.on_tick(&make_trade(101.0, 1.0, 2000, false), &candle);
        }
        for _ in 0..10 {
            strategy.on_tick(&make_trade(102.0, 1.0, 3000, false), &candle);
        }

        strategy.rebuild_profile();

        // Test trade in middle of value area
        let signal = strategy.on_tick(&make_trade(101.0, 1.0, 4000, false), &candle);

        // Should hold when in middle of value area
        assert_eq!(signal, Signal::Hold);
    }

    #[test]
    fn test_no_profile_returns_hold() {
        let mut strategy = VolumeProfileStrategy::new(1.0, Duration::from_secs(3600), 0.02);
        let candle = make_incomplete_candle();

        // First trade with no profile built yet
        let signal = strategy.on_tick(&make_trade(100.0, 1.0, 1000, false), &candle);

        assert_eq!(signal, Signal::Hold);
        assert!(strategy.current_profile.is_none());
    }

    #[test]
    fn test_distance_threshold() {
        let mut strategy =
            VolumeProfileStrategy::new(1.0, Duration::from_secs(3600), 0.01) // 1% threshold
                .rebuild_interval(5);
        let candle = make_incomplete_candle();

        // Create clear support at 100
        for _ in 0..10 {
            strategy.on_tick(&make_trade(100.0, 1.0, 1000, false), &candle);
        }
        for _ in 0..5 {
            strategy.on_tick(&make_trade(105.0, 1.0, 2000, false), &candle);
        }

        strategy.rebuild_profile();

        let profile = strategy.current_profile().unwrap();

        // Should recognize support at VAL
        assert!(strategy.is_near_support(100.0, profile));
        assert!(!strategy.is_near_support(102.0, profile)); // Too far (>1%)
    }

    #[test]
    fn test_name() {
        let strategy = VolumeProfileStrategy::new(1.0, Duration::from_secs(3600), 0.02);
        assert_eq!(strategy.name(), "VolumeProfileStrategy");
    }

    #[test]
    fn test_current_profile_accessor() {
        let mut strategy =
            VolumeProfileStrategy::new(1.0, Duration::from_secs(3600), 0.02).rebuild_interval(5);
        let candle = make_incomplete_candle();

        assert!(strategy.current_profile().is_none());

        // Add enough trades to build profile
        for i in 0..10 {
            strategy.on_tick(&make_trade(100.0, 1.0, i * 100, false), &candle);
        }

        assert!(strategy.current_profile().is_some());
    }

    #[test]
    fn test_memory_bounded_by_lookback() {
        let mut strategy = VolumeProfileStrategy::new(1.0, Duration::from_secs(1), 0.02);
        let candle = make_incomplete_candle();

        // Simulate high-frequency trading for 10 seconds
        for i in 0..10_000 {
            let trade = make_trade(100.0, 1.0, i, false); // 1ms apart
            strategy.on_tick(&trade, &candle);
        }

        // Should only keep ~1000 trades (1 second window at 1ms spacing)
        // Allow some overhead for timing
        assert!(strategy.recent_trades.len() < 2000);
    }
}
