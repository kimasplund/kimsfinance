//! Example: Using tick indicators with TickStrategy
//!
//! This example demonstrates how to build a trading strategy that uses technical
//! indicators calculated from tick-level trade data.
//!
//! Run with:
//! ```bash
//! cargo run --example tick_indicators_strategy
//! ```

use kimsfinance_core::backtest::tick_strategy::TickStrategy;
use kimsfinance_core::backtest::Signal;
use kimsfinance_core::binance::{Candle, IncompleteCandle, Timeframe, Trade};
use kimsfinance_core::indicators::{TickIndicatorEngine, RSI, SMA};

/// RSI-based strategy using tick indicators
///
/// Strategy Logic:
/// - Aggregates tick data to 5-minute candles
/// - Calculates RSI(14) on aggregated candles
/// - Buy when RSI < 30 (oversold)
/// - Sell when RSI > 70 (overbought)
struct RSITickStrategy {
    engine: TickIndicatorEngine,
    rsi: RSI,
    oversold_threshold: f64,
    overbought_threshold: f64,
}

impl RSITickStrategy {
    fn new() -> Self {
        Self {
            engine: TickIndicatorEngine::new(Timeframe::minutes(5)),
            rsi: RSI::new(14).expect("Failed to create RSI(14)"),
            oversold_threshold: 30.0,
            overbought_threshold: 70.0,
        }
    }
}

impl TickStrategy for RSITickStrategy {
    fn on_tick(&mut self, trade: &Trade, _candle: &IncompleteCandle) -> Signal {
        // Update indicator engine with new tick
        self.engine.update(trade);

        // Calculate RSI on aggregated candles
        if let Ok(rsi_values) = self.engine.calculate_indicator(&self.rsi) {
            if let Some(&last_rsi) = rsi_values.last() {
                if !last_rsi.is_nan() {
                    if last_rsi < self.oversold_threshold {
                        return Signal::Buy;
                    } else if last_rsi > self.overbought_threshold {
                        return Signal::Sell;
                    }
                }
            }
        }

        Signal::Hold
    }

    fn name(&self) -> &str {
        "RSITickStrategy"
    }
}

/// SMA Crossover strategy using tick indicators
///
/// Strategy Logic:
/// - Fast SMA(10) crosses above Slow SMA(20) → Buy
/// - Fast SMA(10) crosses below Slow SMA(20) → Sell
struct SMACrossoverStrategy {
    engine: TickIndicatorEngine,
    sma_fast: SMA,
    sma_slow: SMA,
    prev_fast: Option<f64>,
    prev_slow: Option<f64>,
}

impl SMACrossoverStrategy {
    fn new() -> Self {
        Self {
            engine: TickIndicatorEngine::new(Timeframe::minutes(1)),
            sma_fast: SMA::new(10).expect("Failed to create SMA(10)"),
            sma_slow: SMA::new(20).expect("Failed to create SMA(20)"),
            prev_fast: None,
            prev_slow: None,
        }
    }
}

impl TickStrategy for SMACrossoverStrategy {
    fn on_tick(&mut self, trade: &Trade, _candle: &IncompleteCandle) -> Signal {
        self.engine.update(trade);

        // Calculate both SMAs
        let fast_result = self.engine.calculate_indicator(&self.sma_fast);
        let slow_result = self.engine.calculate_indicator(&self.sma_slow);

        if let (Ok(fast_values), Ok(slow_values)) = (fast_result, slow_result) {
            if let (Some(&curr_fast), Some(&curr_slow)) = (fast_values.last(), slow_values.last())
            {
                if !curr_fast.is_nan() && !curr_slow.is_nan() {
                    // Check for crossover
                    if let (Some(prev_fast), Some(prev_slow)) = (self.prev_fast, self.prev_slow) {
                        let prev_diff = prev_fast - prev_slow;
                        let curr_diff = curr_fast - curr_slow;

                        // Bullish crossover (fast crosses above slow)
                        if prev_diff <= 0.0 && curr_diff > 0.0 {
                            self.prev_fast = Some(curr_fast);
                            self.prev_slow = Some(curr_slow);
                            return Signal::Buy;
                        }
                        // Bearish crossover (fast crosses below slow)
                        else if prev_diff >= 0.0 && curr_diff < 0.0 {
                            self.prev_fast = Some(curr_fast);
                            self.prev_slow = Some(curr_slow);
                            return Signal::Sell;
                        }
                    }

                    // Update previous values
                    self.prev_fast = Some(curr_fast);
                    self.prev_slow = Some(curr_slow);
                }
            }
        }

        Signal::Hold
    }

    fn on_candle_complete(&mut self, _candle: &Candle) -> Signal {
        // Could reset state here if needed
        Signal::Hold
    }

    fn name(&self) -> &str {
        "SMACrossoverStrategy"
    }
}

/// Multi-indicator strategy combining RSI and SMA
///
/// Strategy Logic:
/// - Buy when: RSI < 30 AND price > SMA(20)
/// - Sell when: RSI > 70 OR price < SMA(20)
struct MultiIndicatorStrategy {
    engine: TickIndicatorEngine,
    rsi: RSI,
    sma: SMA,
}

impl MultiIndicatorStrategy {
    fn new() -> Self {
        Self {
            engine: TickIndicatorEngine::new(Timeframe::minutes(5)),
            rsi: RSI::new(14).expect("Failed to create RSI(14)"),
            sma: SMA::new(20).expect("Failed to create SMA(20)"),
        }
    }
}

impl TickStrategy for MultiIndicatorStrategy {
    fn on_tick(&mut self, trade: &Trade, _candle: &IncompleteCandle) -> Signal {
        self.engine.update(trade);

        let rsi_result = self.engine.calculate_indicator(&self.rsi);
        let sma_result = self.engine.calculate_indicator(&self.sma);

        if let (Ok(rsi_values), Ok(sma_values)) = (rsi_result, sma_result) {
            if let (Some(&curr_rsi), Some(&curr_sma)) = (rsi_values.last(), sma_values.last()) {
                if !curr_rsi.is_nan() && !curr_sma.is_nan() {
                    let current_price = trade.price;

                    // Buy signal: Oversold + price above trend
                    if curr_rsi < 30.0 && current_price > curr_sma {
                        return Signal::Buy;
                    }
                    // Sell signal: Overbought OR price below trend
                    else if curr_rsi > 70.0 || current_price < curr_sma {
                        return Signal::Sell;
                    }
                }
            }
        }

        Signal::Hold
    }

    fn name(&self) -> &str {
        "MultiIndicatorStrategy"
    }
}

/// Helper function to create test trades
fn create_test_trades(num_trades: usize) -> Vec<Trade> {
    let mut trades = Vec::with_capacity(num_trades);
    let base_timestamp = 1609459200000; // 2021-01-01 00:00:00

    for i in 0..num_trades {
        // Create oscillating price pattern
        let price = 100.0 + ((i as f64 * 0.1).sin() * 20.0);

        trades.push(Trade {
            trade_id: i as u64,
            price,
            quantity: 1.0 + (i as f64 * 0.01), // Varying volume
            quote_quantity: price * (1.0 + (i as f64 * 0.01)),
            timestamp_ms: base_timestamp + ((i as i64) * 60000), // 1 trade per minute
            is_buyer_maker: i % 2 == 0,
        });
    }

    trades
}

fn main() {
    println!("==================================================");
    println!("Tick Indicators Strategy Examples");
    println!("==================================================\n");

    // Create test trade stream
    let trades = create_test_trades(100);
    println!("Created {} test trades\n", trades.len());

    // Example 1: RSI Strategy
    {
        println!("--- Example 1: RSI Strategy ---");
        let mut strategy = RSITickStrategy::new();
        let mut signals = Vec::new();

        for trade in &trades {
            let signal = strategy.on_tick(trade, &IncompleteCandle::new(trade, trade.timestamp_ms));
            if signal != Signal::Hold {
                signals.push((trade.timestamp_ms, signal.clone()));
            }
        }

        println!("RSI Strategy generated {} signals:", signals.len());
        for (timestamp, signal) in signals.iter().take(5) {
            println!("  {} → {:?}", timestamp, signal);
        }
        println!();
    }

    // Example 2: SMA Crossover Strategy
    {
        println!("--- Example 2: SMA Crossover Strategy ---");
        let mut strategy = SMACrossoverStrategy::new();
        let mut signals = Vec::new();

        for trade in &trades {
            let signal = strategy.on_tick(trade, &IncompleteCandle::new(trade, trade.timestamp_ms));
            if signal != Signal::Hold {
                signals.push((trade.timestamp_ms, signal.clone()));
            }
        }

        println!("SMA Crossover Strategy generated {} signals:", signals.len());
        for (timestamp, signal) in signals.iter().take(5) {
            println!("  {} → {:?}", timestamp, signal);
        }
        println!();
    }

    // Example 3: Multi-Indicator Strategy
    {
        println!("--- Example 3: Multi-Indicator Strategy ---");
        let mut strategy = MultiIndicatorStrategy::new();
        let mut signals = Vec::new();

        for trade in &trades {
            let signal = strategy.on_tick(trade, &IncompleteCandle::new(trade, trade.timestamp_ms));
            if signal != Signal::Hold {
                signals.push((trade.timestamp_ms, signal.clone()));
            }
        }

        println!(
            "Multi-Indicator Strategy generated {} signals:",
            signals.len()
        );
        for (timestamp, signal) in signals.iter().take(5) {
            println!("  {} → {:?}", timestamp, signal);
        }
        println!();
    }

    // Performance summary
    println!("==================================================");
    println!("Performance Summary:");
    println!("- Processed {} trades", trades.len());
    println!("- Calculated indicators on-the-fly");
    println!("- Zero manual aggregation required");
    println!("- All strategies use same indicator implementations");
    println!("==================================================");
}
