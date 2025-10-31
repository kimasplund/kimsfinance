//! Tick Strategy Demo
//!
//! Demonstrates the three example tick strategies:
//! 1. IntraCandleMomentum - Trades on price movement within candle
//! 2. VolumeSpikeStrategy - Trades on volume spikes
//! 3. OrderFlowStrategy - Trades on aggressive buy/sell pressure
//!
//! Usage:
//!     cargo run --example tick_strategy_demo

use kimsfinance_core::backtest::tick_strategy::{
    IntraCandleMomentum, OrderFlowStrategy, TickStrategy, VolumeSpikeStrategy,
};
use kimsfinance_core::binance::{IncompleteCandle, Trade};

fn main() {
    println!("=== Tick Strategy Demo ===\n");

    // Create example strategies
    let mut momentum = IntraCandleMomentum::new(0.5);
    let mut volume = VolumeSpikeStrategy::new(3.0);
    let mut order_flow = OrderFlowStrategy::new(5.0);

    println!("Strategies:");
    println!(
        "  1. {} - Signals on 0.5% price move from candle open",
        momentum.name()
    );
    println!("  2. {} - Signals on 3x volume spike", volume.name());
    println!(
        "  3. {} - Signals on 5 BTC order flow imbalance",
        order_flow.name()
    );
    println!();

    // Simulate trades
    let trades = vec![
        Trade {
            trade_id: 1,
            price: 100.0,
            quantity: 1.0,
            quote_quantity: 100.0,
            timestamp_ms: 1000,
            is_buyer_maker: false,
        },
        Trade {
            trade_id: 2,
            price: 100.5,
            quantity: 1.0,
            quote_quantity: 100.5,
            timestamp_ms: 2000,
            is_buyer_maker: false, // Aggressive buy
        },
        Trade {
            trade_id: 3,
            price: 101.0,
            quantity: 10.0, // Volume spike!
            quote_quantity: 1010.0,
            timestamp_ms: 3000,
            is_buyer_maker: false, // Aggressive buy
        },
        Trade {
            trade_id: 4,
            price: 100.8,
            quantity: 1.0,
            quote_quantity: 100.8,
            timestamp_ms: 4000,
            is_buyer_maker: true, // Aggressive sell
        },
        Trade {
            trade_id: 5,
            price: 100.6,
            quantity: 2.0,
            quote_quantity: 201.2,
            timestamp_ms: 5000,
            is_buyer_maker: false, // Aggressive buy
        },
    ];

    let mut candle = IncompleteCandle::new(&trades[0], 0);

    println!("Processing {} trades:\n", trades.len());

    for (i, trade) in trades.iter().enumerate() {
        if i > 0 {
            candle.update(trade);
        }

        println!(
            "Trade {}: price={:.2}, qty={:.2}, buyer_maker={}",
            i + 1,
            trade.price,
            trade.quantity,
            trade.is_buyer_maker
        );

        println!(
            "  Candle state: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
            candle.open, candle.high, candle.low, candle.close, candle.volume
        );

        let momentum_signal = momentum.on_tick(trade, &candle);
        let volume_signal = volume.on_tick(trade, &candle);
        let order_flow_signal = order_flow.on_tick(trade, &candle);

        println!("  Signals:");
        println!("    {} -> {:?}", momentum.name(), momentum_signal);
        println!("    {} -> {:?}", volume.name(), volume_signal);
        println!(
            "    {} -> {:?} (delta={:.2})",
            order_flow.name(),
            order_flow_signal,
            order_flow.delta()
        );
        println!();
    }

    // Complete candle
    let complete_candle = candle.complete();
    println!("=== Candle Complete ===");
    println!(
        "Final OHLCV: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
        complete_candle.open,
        complete_candle.high,
        complete_candle.low,
        complete_candle.close,
        complete_candle.volume
    );
    println!("Trades: {}", complete_candle.num_trades);

    // Reset strategies
    momentum.on_candle_complete(&complete_candle);
    volume.on_candle_complete(&complete_candle);
    order_flow.on_candle_complete(&complete_candle);

    println!("\nStrategies reset for next candle.");
    println!("Order flow delta after reset: {:.2}", order_flow.delta());
}
