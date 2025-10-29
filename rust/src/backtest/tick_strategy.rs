//! Tick-level trading strategy trait and example implementations
//!
//! This module provides the `TickStrategy` trait for processing individual trades
//! as they arrive in tick-by-tick backtesting or live trading scenarios.
//!
//! # Key Features
//!
//! - **Sub-candle Resolution**: React to price movements within candle formation
//! - **Order Flow Analysis**: Track aggressive buyer/seller activity
//! - **Volume Spikes**: Detect unusual trading activity
//! - **Intra-Candle Patterns**: Identify momentum shifts before candle completion
//!
//! # Architecture
//!
//! ```text
//! Trade Stream → TickStrategy::on_tick() → Signal
//!       ↓
//! IncompleteCandle (updated incrementally)
//!       ↓
//! Candle Complete → TickStrategy::on_candle_complete() → Signal
//! ```
//!
//! # Performance
//!
//! - `on_tick()` will be called millions of times per backtest
//! - Target: <1μs per call for hot path strategies
//! - Use `on_tick_batch()` for 10-50% speedup via batching
//!
//! # Example
//!
//! ```rust
//! use kimsfinance_core::backtest::tick_strategy::TickStrategy;
//! use kimsfinance_core::backtest::Signal;
//! use kimsfinance_core::binance::{Trade, IncompleteCandle};
//!
//! struct MomentumStrategy {
//!     threshold: f64,  // 0.5 = 0.5% price change
//! }
//!
//! impl TickStrategy for MomentumStrategy {
//!     fn on_tick(&mut self, trade: &Trade, candle: &IncompleteCandle) -> Signal {
//!         if candle.open == 0.0 {
//!             return Signal::Hold;
//!         }
//!
//!         let change_pct = (trade.price - candle.open) / candle.open * 100.0;
//!
//!         if change_pct > self.threshold {
//!             Signal::Buy
//!         } else if change_pct < -self.threshold {
//!             Signal::Sell
//!         } else {
//!             Signal::Hold
//!         }
//!     }
//!
//!     fn name(&self) -> &str {
//!         "MomentumStrategy"
//!     }
//! }
//! ```

use crate::backtest::Signal;
use crate::binance::{Candle, IncompleteCandle, Trade};

/// Strategy that processes individual trades (ticks) as they arrive
///
/// This trait enables intra-candle trading decisions based on:
/// - Price movements within the candle
/// - Order flow patterns (aggressive buy/sell pressure)
/// - Volume accumulation and spikes
/// - High-frequency micro-patterns
///
/// # Design Philosophy
///
/// Unlike traditional candle-based strategies that only see OHLCV after completion,
/// tick strategies have visibility into the candle formation process. This enables:
///
/// - **Early Detection**: Identify trends before candle close
/// - **Order Flow**: Track aggressive buyer vs seller activity
/// - **Liquidity Analysis**: Detect large orders and volume spikes
/// - **Tick Patterns**: Recognize high-frequency microstructure
///
/// # Performance Considerations
///
/// `on_tick()` is a hot path that will be called millions of times. Guidelines:
///
/// - Keep logic simple and fast (<1μs per call)
/// - Avoid allocations in the hot path
/// - Use batching via `on_tick_batch()` when possible
/// - Leverage `IncompleteCandle`'s 2.31ns update performance
///
/// # State Management
///
/// Strategies can maintain internal state across ticks:
/// - Accumulate volume deltas
/// - Track rolling averages
/// - Build internal tick buffers
///
/// Use `on_candle_complete()` to reset state between candles.
pub trait TickStrategy {
    /// Called for every trade (tick) as it arrives
    ///
    /// # Arguments
    ///
    /// - `trade`: The current trade tick
    /// - `candle`: The incomplete candle being built (includes all trades up to this point)
    ///
    /// # Returns
    ///
    /// Trading signal: Buy, Sell, Hold, Short, or Cover
    ///
    /// # Performance
    ///
    /// This will be called millions of times in backtesting. Target: <1μs per call.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use kimsfinance_core::backtest::tick_strategy::TickStrategy;
    /// # use kimsfinance_core::backtest::Signal;
    /// # use kimsfinance_core::binance::{Trade, IncompleteCandle};
    /// #
    /// # struct MyStrategy;
    /// # impl TickStrategy for MyStrategy {
    /// fn on_tick(&mut self, trade: &Trade, candle: &IncompleteCandle) -> Signal {
    ///     // React to every trade
    ///     if candle.close > candle.open * 1.005 {
    ///         Signal::Buy  // 0.5% price increase within candle
    ///     } else {
    ///         Signal::Hold
    ///     }
    /// }
    /// # fn name(&self) -> &str { "MyStrategy" }
    /// # }
    /// ```
    fn on_tick(&mut self, trade: &Trade, candle: &IncompleteCandle) -> Signal;

    /// Called when a candle completes (optional hook)
    ///
    /// Use this for:
    /// - End-of-candle analysis
    /// - State reset for next candle
    /// - Cleanup of internal buffers
    ///
    /// # Default Implementation
    ///
    /// Does nothing and returns `Signal::Hold`. Override if needed.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use kimsfinance_core::backtest::tick_strategy::TickStrategy;
    /// # use kimsfinance_core::backtest::Signal;
    /// # use kimsfinance_core::binance::{Trade, IncompleteCandle, Candle};
    /// #
    /// # struct MyStrategy { volume_accumulator: f64 }
    /// # impl TickStrategy for MyStrategy {
    /// #     fn on_tick(&mut self, _trade: &Trade, _candle: &IncompleteCandle) -> Signal { Signal::Hold }
    /// fn on_candle_complete(&mut self, _candle: &Candle) -> Signal {
    ///     // Reset state for next candle
    ///     self.volume_accumulator = 0.0;
    ///     Signal::Hold
    /// }
    /// # fn name(&self) -> &str { "MyStrategy" }
    /// # }
    /// ```
    fn on_candle_complete(&mut self, _candle: &Candle) -> Signal {
        Signal::Hold
    }

    /// Called with batch of ticks (optional optimization)
    ///
    /// Override this if you want to process ticks in batches for efficiency.
    /// Batching can reduce function call overhead by 10-50%.
    ///
    /// # Default Implementation
    ///
    /// Calls `on_tick()` for each trade and returns the last signal.
    ///
    /// # Performance
    ///
    /// Batching is beneficial when:
    /// - Processing thousands of ticks per second
    /// - Strategy logic benefits from vectorization
    /// - Function call overhead is significant
    ///
    /// # Example
    ///
    /// ```rust
    /// # use kimsfinance_core::backtest::tick_strategy::TickStrategy;
    /// # use kimsfinance_core::backtest::Signal;
    /// # use kimsfinance_core::binance::{Trade, IncompleteCandle};
    /// #
    /// # struct MyStrategy;
    /// # impl TickStrategy for MyStrategy {
    /// #     fn on_tick(&mut self, _trade: &Trade, _candle: &IncompleteCandle) -> Signal { Signal::Hold }
    /// fn on_tick_batch(&mut self, trades: &[Trade], candle: &IncompleteCandle) -> Signal {
    ///     // Process all trades at once (potential for vectorization)
    ///     let total_volume: f64 = trades.iter().map(|t| t.quantity).sum();
    ///
    ///     if total_volume > 100.0 {
    ///         Signal::Buy
    ///     } else {
    ///         Signal::Hold
    ///     }
    /// }
    /// # fn name(&self) -> &str { "MyStrategy" }
    /// # }
    /// ```
    fn on_tick_batch(&mut self, trades: &[Trade], candle: &IncompleteCandle) -> Signal {
        let mut last_signal = Signal::Hold;
        for trade in trades {
            last_signal = self.on_tick(trade, candle);
        }
        last_signal
    }

    /// Get strategy name for logging/reporting
    ///
    /// Used in backtest results and performance reports.
    ///
    /// # Default Implementation
    ///
    /// Returns `"UnnamedStrategy"`. Override to provide a descriptive name.
    fn name(&self) -> &str {
        "UnnamedStrategy"
    }
}

// ====================================================================================
// Example Strategy 1: Intra-Candle Momentum
// ====================================================================================

/// Trades on price momentum within a single candle
///
/// # Strategy Logic
///
/// - **Entry**: Price moves >threshold% from candle open
/// - **Direction**: Buy if up-momentum, Sell if down-momentum
/// - **Reset**: On each new candle
///
/// # Use Cases
///
/// - Capturing breakouts within candle formation
/// - Early trend detection before candle close
/// - Scalping sub-candle price movements
///
/// # Example
///
/// ```rust
/// use kimsfinance_core::backtest::tick_strategy::IntraCandleMomentum;
///
/// // Buy when price rises 0.5% within candle
/// let strategy = IntraCandleMomentum::new(0.5);
/// ```
///
/// # Performance
///
/// - Zero allocations per tick
/// - Simple arithmetic: <50ns per call
/// - Suitable for high-frequency backtesting
#[derive(Debug, Clone)]
pub struct IntraCandleMomentum {
    /// Threshold as percentage (e.g., 0.5 = 0.5%)
    threshold_pct: f64,
    name: String,
}

impl IntraCandleMomentum {
    /// Create new intra-candle momentum strategy
    ///
    /// # Arguments
    ///
    /// - `threshold_pct`: Minimum price change percentage to trigger signal (e.g., 0.5 for 0.5%)
    ///
    /// # Example
    ///
    /// ```rust
    /// use kimsfinance_core::backtest::tick_strategy::IntraCandleMomentum;
    ///
    /// let strategy = IntraCandleMomentum::new(0.5);  // 0.5% threshold
    /// ```
    pub fn new(threshold_pct: f64) -> Self {
        Self {
            threshold_pct,
            name: format!("IntraCandleMomentum({}%)", threshold_pct),
        }
    }
}

impl TickStrategy for IntraCandleMomentum {
    fn on_tick(&mut self, trade: &Trade, candle: &IncompleteCandle) -> Signal {
        if candle.open == 0.0 {
            return Signal::Hold;
        }

        let price_change_pct = (trade.price - candle.open) / candle.open * 100.0;

        if price_change_pct > self.threshold_pct {
            Signal::Buy
        } else if price_change_pct < -self.threshold_pct {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ====================================================================================
// Example Strategy 2: Volume Spike
// ====================================================================================

/// Trades on volume spikes within candle
///
/// # Strategy Logic
///
/// - Tracks average volume per trade
/// - Signals when volume spike occurs (e.g., 3x average)
/// - Direction based on price direction during spike
///
/// # Use Cases
///
/// - Detecting institutional order flow
/// - Identifying liquidity events
/// - Trading on sudden volume surges
///
/// # Example
///
/// ```rust
/// use kimsfinance_core::backtest::tick_strategy::VolumeSpikeStrategy;
///
/// // Signal on 3x volume spike
/// let strategy = VolumeSpikeStrategy::new(3.0);
/// ```
///
/// # Performance
///
/// - Zero allocations per tick
/// - Simple division and comparison: <100ns per call
#[derive(Debug, Clone)]
pub struct VolumeSpikeStrategy {
    /// Multiplier for volume spike (e.g., 3.0 = 3x average)
    spike_multiplier: f64,
    /// Running average volume per trade
    avg_volume: f64,
    name: String,
}

impl VolumeSpikeStrategy {
    /// Create new volume spike strategy
    ///
    /// # Arguments
    ///
    /// - `spike_multiplier`: Volume multiplier to trigger signal (e.g., 3.0 for 3x average)
    ///
    /// # Example
    ///
    /// ```rust
    /// use kimsfinance_core::backtest::tick_strategy::VolumeSpikeStrategy;
    ///
    /// let strategy = VolumeSpikeStrategy::new(3.0);  // 3x volume spike
    /// ```
    pub fn new(spike_multiplier: f64) -> Self {
        Self {
            spike_multiplier,
            avg_volume: 0.0,
            name: format!("VolumeSpike({}x)", spike_multiplier),
        }
    }
}

impl TickStrategy for VolumeSpikeStrategy {
    fn on_tick(&mut self, trade: &Trade, candle: &IncompleteCandle) -> Signal {
        // Calculate average volume EXCLUDING current trade
        // (candle already includes current trade after update)
        if candle.num_trades > 1 {
            // Average of previous trades (before this one)
            self.avg_volume = (candle.volume - trade.quantity) / (candle.num_trades - 1) as f64;
        } else if candle.num_trades == 1 {
            // First trade - no average yet
            self.avg_volume = trade.quantity;
            return Signal::Hold;
        }

        // Check for spike (comparing current trade against previous average)
        if self.avg_volume > 0.0 && trade.quantity > self.avg_volume * self.spike_multiplier {
            // Volume spike detected! Trade in direction of price movement
            if trade.price > candle.open {
                Signal::Buy
            } else if trade.price < candle.open {
                Signal::Sell
            } else {
                Signal::Hold
            }
        } else {
            Signal::Hold
        }
    }

    fn on_candle_complete(&mut self, _candle: &Candle) -> Signal {
        // Reset for next candle
        self.avg_volume = 0.0;
        Signal::Hold
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ====================================================================================
// Example Strategy 3: Order Flow (using is_buyer_maker)
// ====================================================================================

/// Trades based on aggressive buy/sell flow
///
/// # Strategy Logic
///
/// Uses `is_buyer_maker` field to detect aggressive orders:
/// - `false` = Buyer is taker (aggressive buy, bullish)
/// - `true` = Seller is taker (aggressive sell, bearish)
///
/// Accumulates buy vs sell volume and signals when imbalance exceeds threshold.
///
/// # Order Flow Terminology
///
/// - **Maker**: Places limit order (passive, adds liquidity)
/// - **Taker**: Executes market order (aggressive, removes liquidity)
/// - **Buyer Maker**: Buyer placed limit order → Seller executed (bearish)
/// - **Seller Maker**: Seller placed limit order → Buyer executed (bullish)
///
/// # Use Cases
///
/// - Detecting institutional accumulation/distribution
/// - Trading on aggressive order flow imbalances
/// - Identifying hidden buying/selling pressure
///
/// # Example
///
/// ```rust
/// use kimsfinance_core::backtest::tick_strategy::OrderFlowStrategy;
///
/// // Signal when 5 BTC imbalance detected
/// let strategy = OrderFlowStrategy::new(5.0);
/// ```
///
/// # Performance
///
/// - Zero allocations per tick
/// - Simple accumulation: <50ns per call
#[derive(Debug, Clone)]
pub struct OrderFlowStrategy {
    /// Imbalance threshold (BTC)
    imbalance_threshold: f64,
    /// Buy volume accumulator (buyers are takers)
    buy_volume: f64,
    /// Sell volume accumulator (sellers are takers)
    sell_volume: f64,
    name: String,
}

impl OrderFlowStrategy {
    /// Create new order flow strategy
    ///
    /// # Arguments
    ///
    /// - `imbalance_threshold`: Volume imbalance to trigger signal (in base asset units)
    ///
    /// # Example
    ///
    /// ```rust
    /// use kimsfinance_core::backtest::tick_strategy::OrderFlowStrategy;
    ///
    /// let strategy = OrderFlowStrategy::new(5.0);  // 5 BTC imbalance
    /// ```
    pub fn new(imbalance_threshold: f64) -> Self {
        Self {
            imbalance_threshold,
            buy_volume: 0.0,
            sell_volume: 0.0,
            name: format!("OrderFlow({})", imbalance_threshold),
        }
    }

    /// Get current order flow delta (buy volume - sell volume)
    ///
    /// Positive = More aggressive buying
    /// Negative = More aggressive selling
    pub fn delta(&self) -> f64 {
        self.buy_volume - self.sell_volume
    }
}

impl TickStrategy for OrderFlowStrategy {
    fn on_tick(&mut self, trade: &Trade, _candle: &IncompleteCandle) -> Signal {
        // Accumulate buy/sell volume
        if trade.is_buyer_maker {
            // Seller is taker (aggressive sell)
            self.sell_volume += trade.quantity;
        } else {
            // Buyer is taker (aggressive buy)
            self.buy_volume += trade.quantity;
        }

        // Check imbalance
        let delta = self.delta();

        if delta > self.imbalance_threshold {
            Signal::Buy // More aggressive buying
        } else if delta < -self.imbalance_threshold {
            Signal::Sell // More aggressive selling
        } else {
            Signal::Hold
        }
    }

    fn on_candle_complete(&mut self, _candle: &Candle) -> Signal {
        // Reset accumulators for next candle
        self.buy_volume = 0.0;
        self.sell_volume = 0.0;
        Signal::Hold
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_trade(price: f64, quantity: f64, is_buyer_maker: bool) -> Trade {
        Trade {
            trade_id: 1,
            price,
            quantity,
            quote_quantity: price * quantity,
            timestamp_ms: 1000,
            is_buyer_maker,
        }
    }

    #[test]
    fn test_intra_candle_momentum_buy() {
        let mut strategy = IntraCandleMomentum::new(0.5);

        let trade1 = create_test_trade(100.0, 1.0, false);
        let mut candle = IncompleteCandle::new(&trade1, 0);

        // Price rises 0.6% -> should signal Buy
        let trade2 = create_test_trade(100.6, 1.0, false);
        candle.update(&trade2);

        let signal = strategy.on_tick(&trade2, &candle);
        assert_eq!(signal, Signal::Buy);
    }

    #[test]
    fn test_intra_candle_momentum_sell() {
        let mut strategy = IntraCandleMomentum::new(0.5);

        let trade1 = create_test_trade(100.0, 1.0, false);
        let mut candle = IncompleteCandle::new(&trade1, 0);

        // Price falls 0.6% -> should signal Sell
        let trade2 = create_test_trade(99.4, 1.0, false);
        candle.update(&trade2);

        let signal = strategy.on_tick(&trade2, &candle);
        assert_eq!(signal, Signal::Sell);
    }

    #[test]
    fn test_intra_candle_momentum_hold() {
        let mut strategy = IntraCandleMomentum::new(0.5);

        let trade1 = create_test_trade(100.0, 1.0, false);
        let mut candle = IncompleteCandle::new(&trade1, 0);

        // Price rises only 0.3% -> should Hold
        let trade2 = create_test_trade(100.3, 1.0, false);
        candle.update(&trade2);

        let signal = strategy.on_tick(&trade2, &candle);
        assert_eq!(signal, Signal::Hold);
    }

    #[test]
    fn test_volume_spike_detection() {
        let mut strategy = VolumeSpikeStrategy::new(3.0);

        let trade1 = create_test_trade(100.0, 1.0, false);
        let mut candle = IncompleteCandle::new(&trade1, 0);

        // Small trade (no spike)
        let trade2 = create_test_trade(101.0, 1.0, false);
        candle.update(&trade2);
        let signal = strategy.on_tick(&trade2, &candle);
        assert_eq!(signal, Signal::Hold);

        // HUGE trade (3x+ average) with price up
        let trade3 = create_test_trade(102.0, 10.0, false);
        candle.update(&trade3);
        let signal = strategy.on_tick(&trade3, &candle);
        assert_eq!(signal, Signal::Buy);
    }

    #[test]
    fn test_volume_spike_sell_signal() {
        let mut strategy = VolumeSpikeStrategy::new(3.0);

        let trade1 = create_test_trade(100.0, 1.0, false);
        let mut candle = IncompleteCandle::new(&trade1, 0);

        let trade2 = create_test_trade(101.0, 1.0, false);
        candle.update(&trade2);

        // HUGE trade (3x+ average) with price down
        let trade3 = create_test_trade(98.0, 10.0, false);
        candle.update(&trade3);
        let signal = strategy.on_tick(&trade3, &candle);
        assert_eq!(signal, Signal::Sell);
    }

    #[test]
    fn test_order_flow_buy_signal() {
        let mut strategy = OrderFlowStrategy::new(5.0);

        let trade1 = create_test_trade(100.0, 1.0, false);
        let candle = IncompleteCandle::new(&trade1, 0);

        // 10 BTC aggressive buying (is_buyer_maker = false)
        for _ in 0..10 {
            let trade = create_test_trade(100.0, 1.0, false);
            let signal = strategy.on_tick(&trade, &candle);
            if strategy.delta() > 5.0 {
                assert_eq!(signal, Signal::Buy);
            }
        }
    }

    #[test]
    fn test_order_flow_sell_signal() {
        let mut strategy = OrderFlowStrategy::new(5.0);

        let trade1 = create_test_trade(100.0, 1.0, true);
        let candle = IncompleteCandle::new(&trade1, 0);

        // 10 BTC aggressive selling (is_buyer_maker = true)
        for _ in 0..10 {
            let trade = create_test_trade(100.0, 1.0, true);
            let signal = strategy.on_tick(&trade, &candle);
            if strategy.delta() < -5.0 {
                assert_eq!(signal, Signal::Sell);
            }
        }
    }

    #[test]
    fn test_order_flow_delta_calculation() {
        let mut strategy = OrderFlowStrategy::new(10.0);

        let candle_start = create_test_trade(100.0, 1.0, false);
        let candle = IncompleteCandle::new(&candle_start, 0);

        // 5 BTC buy
        for _ in 0..5 {
            let trade = create_test_trade(100.0, 1.0, false);
            strategy.on_tick(&trade, &candle);
        }

        // 2 BTC sell
        for _ in 0..2 {
            let trade = create_test_trade(100.0, 1.0, true);
            strategy.on_tick(&trade, &candle);
        }

        // Delta = 5 - 2 = 3.0
        assert_eq!(strategy.delta(), 3.0);
    }

    #[test]
    fn test_candle_complete_reset() {
        let mut strategy = OrderFlowStrategy::new(5.0);

        let trade = create_test_trade(100.0, 10.0, false);
        let candle = IncompleteCandle::new(&trade, 0);
        strategy.on_tick(&trade, &candle);

        assert!(strategy.buy_volume > 0.0);

        // Complete candle -> should reset
        let complete_candle = candle.complete();
        strategy.on_candle_complete(&complete_candle);

        assert_eq!(strategy.buy_volume, 0.0);
        assert_eq!(strategy.sell_volume, 0.0);
    }

    #[test]
    fn test_volume_spike_candle_reset() {
        let mut strategy = VolumeSpikeStrategy::new(3.0);

        let trade = create_test_trade(100.0, 10.0, false);
        let candle = IncompleteCandle::new(&trade, 0);
        strategy.on_tick(&trade, &candle);

        assert!(strategy.avg_volume > 0.0);

        // Complete candle -> should reset
        let complete_candle = candle.complete();
        strategy.on_candle_complete(&complete_candle);

        assert_eq!(strategy.avg_volume, 0.0);
    }

    #[test]
    fn test_strategy_names() {
        let momentum = IntraCandleMomentum::new(0.5);
        let volume = VolumeSpikeStrategy::new(3.0);
        let order_flow = OrderFlowStrategy::new(5.0);

        assert_eq!(momentum.name(), "IntraCandleMomentum(0.5%)");
        assert_eq!(volume.name(), "VolumeSpike(3x)");
        assert_eq!(order_flow.name(), "OrderFlow(5)");
    }

    #[test]
    fn test_batch_processing() {
        let mut strategy = IntraCandleMomentum::new(0.5);

        let trades = vec![
            create_test_trade(100.0, 1.0, false),
            create_test_trade(100.3, 1.0, false),
            create_test_trade(100.6, 1.0, false), // Exceeds threshold
        ];

        let mut candle = IncompleteCandle::new(&trades[0], 0);
        for trade in &trades[1..] {
            candle.update(trade);
        }

        let signal = strategy.on_tick_batch(&trades, &candle);
        assert_eq!(signal, Signal::Buy); // Last signal should be Buy
    }

    #[test]
    fn test_zero_open_price_handling() {
        let mut strategy = IntraCandleMomentum::new(0.5);

        let trade = create_test_trade(100.0, 1.0, false);
        let mut candle = IncompleteCandle::new(&trade, 0);
        candle.open = 0.0; // Edge case: zero open price

        let signal = strategy.on_tick(&trade, &candle);
        assert_eq!(signal, Signal::Hold); // Should safely handle division by zero
    }
}
