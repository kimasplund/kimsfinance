//! Market Microstructure Analysis Demo
//!
//! This example demonstrates the market microstructure analysis functionality
//! using simulated trade data and real-world scenarios.
//!
//! # Features Demonstrated
//! - Order flow imbalance calculation
//! - Trade aggressiveness detection
//! - Price volatility and spread estimation
//! - Rolling window analysis
//! - Integration with TickStrategy for live trading
//!
//! # Usage
//! ```bash
//! cargo run --example microstructure_demo
//! ```

use kimsfinance_core::analysis::{MicrostructureAnalyzer, MicrostructureMetrics};
use kimsfinance_core::backtest::{MicrostructureStrategy, Signal, TickStrategy};
use kimsfinance_core::binance::{IncompleteCandle, Trade};

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Market Microstructure Analysis Demo");
    println!("═══════════════════════════════════════════════════════════\n");

    // Demo 1: Basic microstructure analysis
    demo_basic_analysis();

    // Demo 2: Rolling window analysis
    demo_rolling_windows();

    // Demo 3: Trading strategy integration
    demo_strategy_integration();

    // Demo 4: Real-world scenario simulation
    demo_realistic_scenario();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Demo Complete!");
    println!("═══════════════════════════════════════════════════════════");
}

fn demo_basic_analysis() {
    println!("─────────────────────────────────────────────────────────");
    println!("Demo 1: Basic Microstructure Analysis");
    println!("─────────────────────────────────────────────────────────\n");

    let analyzer = MicrostructureAnalyzer::new(60_000); // 1-minute window

    // Simulate a trading scenario with strong buying pressure
    let trades = vec![
        make_trade(50_000.0, 0.5, 0, false),    // Aggressive buy
        make_trade(50_001.0, 1.0, 1000, false), // Aggressive buy
        make_trade(50_002.0, 0.8, 2000, false), // Aggressive buy
        make_trade(50_001.5, 0.2, 3000, true),  // Aggressive sell
        make_trade(50_003.0, 1.5, 4000, false), // Aggressive buy
    ];

    let metrics = analyzer.analyze(&trades);

    print_metrics(&metrics);
}

fn demo_rolling_windows() {
    println!("\n─────────────────────────────────────────────────────────");
    println!("Demo 2: Rolling Window Analysis");
    println!("─────────────────────────────────────────────────────────\n");

    let analyzer = MicrostructureAnalyzer::new(30_000); // 30-second windows

    // Simulate 2 minutes of trading across 4 windows
    let mut trades = Vec::new();

    // Window 1 (0-30s): Strong buying
    for i in 0..5 {
        trades.push(make_trade(100.0, 1.0, i * 5000, false));
    }

    // Window 2 (30-60s): Balanced
    for i in 6..11 {
        trades.push(make_trade(101.0, 1.0, i * 5000, i % 2 == 0));
    }

    // Window 3 (60-90s): Strong selling
    for i in 12..17 {
        trades.push(make_trade(100.5, 1.0, i * 5000, true));
    }

    // Window 4 (90-120s): Recovery
    for i in 18..23 {
        trades.push(make_trade(101.5, 1.0, i * 5000, false));
    }

    let all_metrics = analyzer.analyze_rolling(&trades);

    println!("Analyzed {} windows:\n", all_metrics.len());

    for (i, metrics) in all_metrics.iter().enumerate() {
        println!(
            "Window {} ({}ms - {}ms):",
            i + 1,
            metrics.timestamp,
            metrics.timestamp + metrics.duration_ms
        );
        println!("  Trades: {}", metrics.num_trades);
        println!(
            "  Order Flow Imbalance: {:.3}",
            metrics.order_flow_imbalance
        );
        println!("  Aggressiveness: {:.3}", metrics.aggressiveness_ratio);
        println!(
            "  Interpretation: {}",
            interpret_ofi(metrics.order_flow_imbalance)
        );
        println!();
    }
}

fn demo_strategy_integration() {
    println!("─────────────────────────────────────────────────────────");
    println!("Demo 3: Trading Strategy Integration");
    println!("─────────────────────────────────────────────────────────\n");

    let mut strategy = MicrostructureStrategy::new(0.3, 60_000);

    println!("Testing strategy with threshold = 0.3 (30% imbalance required)\n");

    // Simulate incoming trades
    let test_scenarios = vec![
        ("Strong Buy Pressure", generate_buy_pressure(10)),
        ("Strong Sell Pressure", generate_sell_pressure(10)),
        ("Balanced Market", generate_balanced_market(10)),
    ];

    for (name, trades) in test_scenarios {
        println!("Scenario: {}", name);

        // Reset strategy for new scenario
        strategy.reset();

        // Feed trades to strategy
        let mut last_signal = Signal::Hold;
        for trade in &trades {
            let candle = IncompleteCandle::new(trade, 0);
            last_signal = strategy.on_tick(trade, &candle);
        }

        println!("  Final Signal: {:?}", last_signal);
        println!(
            "  Order Flow Imbalance: {:.3}",
            strategy.current_imbalance()
        );
        println!();
    }
}

fn demo_realistic_scenario() {
    println!("─────────────────────────────────────────────────────────");
    println!("Demo 4: Realistic Market Scenario");
    println!("─────────────────────────────────────────────────────────\n");

    let analyzer = MicrostructureAnalyzer::new(60_000);

    // Simulate a realistic trading session:
    // - Morning session starts with selling pressure
    // - Mid-day consolidation
    // - Afternoon rally with buying pressure

    let trades = vec![
        // Morning: Selling pressure
        make_trade(50_000.0, 0.5, 0, true),
        make_trade(49_950.0, 0.8, 5000, true),
        make_trade(49_900.0, 1.0, 10000, true),
        make_trade(49_920.0, 0.3, 15000, false),
        // Mid-day: Consolidation
        make_trade(49_930.0, 0.5, 20000, false),
        make_trade(49_925.0, 0.4, 25000, true),
        make_trade(49_935.0, 0.6, 30000, false),
        // Afternoon: Rally
        make_trade(49_950.0, 1.0, 35000, false),
        make_trade(49_980.0, 1.5, 40000, false),
        make_trade(50_000.0, 2.0, 45000, false),
        make_trade(49_990.0, 0.5, 50000, true),
        make_trade(50_010.0, 1.8, 55000, false),
    ];

    let metrics = analyzer.analyze(&trades);

    println!("Full Session Analysis:");
    print_metrics(&metrics);

    println!("\n📊 Market Interpretation:");
    println!(
        "  Price Action: {} → {} ({})",
        trades.first().unwrap().price,
        trades.last().unwrap().price,
        if trades.last().unwrap().price > trades.first().unwrap().price {
            "Bullish"
        } else {
            "Bearish"
        }
    );
    println!(
        "  Order Flow: {}",
        interpret_ofi(metrics.order_flow_imbalance)
    );
    println!(
        "  Price Trend: {}",
        interpret_tick_direction(metrics.tick_direction)
    );
    println!("  Spread: ${:.2}", metrics.spread_estimate);
}

// ============================================================================
// Helper Functions
// ============================================================================

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

fn print_metrics(metrics: &MicrostructureMetrics) {
    println!("📈 Microstructure Metrics:");
    println!("  ├─ Total Trades: {}", metrics.num_trades);
    println!("  ├─ Total Volume: {:.4} BTC", metrics.total_volume);
    println!("  ├─ Avg Trade Size: {:.4} BTC", metrics.avg_trade_size);
    println!("  ├─ VWAP: ${:.2}", metrics.volume_weighted_price);
    println!("  │");
    println!("  ├─ Order Flow:");
    println!("  │  ├─ Buy Volume: {:.4} BTC", metrics.buy_volume);
    println!("  │  ├─ Sell Volume: {:.4} BTC", metrics.sell_volume);
    println!(
        "  │  └─ Imbalance: {:.3} {}",
        metrics.order_flow_imbalance,
        format_ofi_indicator(metrics.order_flow_imbalance)
    );
    println!("  │");
    println!("  ├─ Trade Aggressiveness:");
    println!("  │  ├─ Aggressive Buys: {}", metrics.aggressive_buy_count);
    println!(
        "  │  ├─ Aggressive Sells: {}",
        metrics.aggressive_sell_count
    );
    println!("  │  └─ Ratio: {:.3}", metrics.aggressiveness_ratio);
    println!("  │");
    println!("  └─ Price Dynamics:");
    println!("     ├─ Volatility (σ): ${:.2}", metrics.price_volatility);
    println!("     ├─ Spread Estimate: ${:.2}", metrics.spread_estimate);
    println!(
        "     └─ Tick Direction: {:.3} {}",
        metrics.tick_direction,
        format_tick_indicator(metrics.tick_direction)
    );
}

fn format_ofi_indicator(ofi: f64) -> &'static str {
    if ofi > 0.5 {
        "🟢 (Strong Buy Pressure)"
    } else if ofi > 0.2 {
        "🔵 (Moderate Buy Pressure)"
    } else if ofi > -0.2 {
        "⚪ (Balanced)"
    } else if ofi > -0.5 {
        "🟠 (Moderate Sell Pressure)"
    } else {
        "🔴 (Strong Sell Pressure)"
    }
}

fn format_tick_indicator(tick: f64) -> &'static str {
    if tick > 0.3 {
        "⬆️ (Uptrend)"
    } else if tick < -0.3 {
        "⬇️ (Downtrend)"
    } else {
        "➡️ (Sideways)"
    }
}

fn interpret_ofi(ofi: f64) -> &'static str {
    if ofi > 0.5 {
        "Strong buying pressure detected"
    } else if ofi > 0.2 {
        "Moderate buying pressure"
    } else if ofi > -0.2 {
        "Balanced market"
    } else if ofi > -0.5 {
        "Moderate selling pressure"
    } else {
        "Strong selling pressure detected"
    }
}

fn interpret_tick_direction(tick: f64) -> &'static str {
    if tick > 0.3 {
        "Consistent upward price movement"
    } else if tick < -0.3 {
        "Consistent downward price movement"
    } else {
        "Choppy/sideways price action"
    }
}

fn generate_buy_pressure(count: usize) -> Vec<Trade> {
    (0..count)
        .map(|i| make_trade(100.0 + i as f64, 1.0, i as i64 * 1000, false))
        .collect()
}

fn generate_sell_pressure(count: usize) -> Vec<Trade> {
    (0..count)
        .map(|i| make_trade(100.0 - i as f64, 1.0, i as i64 * 1000, true))
        .collect()
}

fn generate_balanced_market(count: usize) -> Vec<Trade> {
    (0..count)
        .map(|i| make_trade(100.0, 1.0, i as i64 * 1000, i % 2 == 0))
        .collect()
}
