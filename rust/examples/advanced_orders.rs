//! Advanced order types example
//!
//! Demonstrates 12+ order types including:
//! - Market, Limit, Stop, Stop-Limit
//! - MOO, MOC, LOO, LOC
//! - Trailing stops (absolute and percentage)
//! - Iceberg orders
//! - TWAP, VWAP, POV algorithmic orders
//! - OCO, OTO, Bracket complex orders

use kimsfinance_core::backtest::{
    MarketSnapshot, MatchingEngine, OHLCVBar, Order, OrderGroup, OrderSide, OrderStatus, OrderType,
    TimeInForce,
};

fn create_test_bar(timestamp: i64, close: f64, volume: f64) -> OHLCVBar {
    OHLCVBar {
        timestamp,
        open: close * 0.99,
        high: close * 1.01,
        low: close * 0.99,
        close,
        volume,
    }
}

fn main() {
    println!("=== Advanced Order Types Demo ===\n");

    // Initialize matching engine
    let mut engine = MatchingEngine::new();

    // ========================================
    // Example 1: Basic Order Types
    // ========================================
    println!("1. Basic Order Types");
    println!("--------------------");

    // Market order - immediate execution
    let market_order = Order::market(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0);
    let market_id = engine.submit_order(market_order);
    println!("✓ Market order submitted (ID: {})", market_id);

    // Limit order - execute at specific price or better
    let limit_order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 49000.0);
    let limit_id = engine.submit_order(limit_order);
    println!("✓ Limit order submitted (ID: {}) @ $49,000", limit_id);

    // Stop order - trigger when price hits stop
    let stop_order = Order::stop(0, "BTC/USD".to_string(), OrderSide::Sell, 1.0, 48000.0);
    let stop_id = engine.submit_order(stop_order);
    println!("✓ Stop order submitted (ID: {}) @ $48,000", stop_id);

    // Stop-limit order
    let stop_limit = Order::stop_limit(
        0,
        "BTC/USD".to_string(),
        OrderSide::Sell,
        1.0,
        48000.0,
        47900.0,
    );
    let stop_limit_id = engine.submit_order(stop_limit);
    println!(
        "✓ Stop-limit order submitted (ID: {}) stop=$48,000 limit=$47,900\n",
        stop_limit_id
    );

    // ========================================
    // Example 2: Session-Based Orders
    // ========================================
    println!("2. Session-Based Orders (MOO/MOC/LOO/LOC)");
    println!("------------------------------------------");

    let mut moo_order = Order::market(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0);
    moo_order.order_type = OrderType::MarketOnOpen;
    let moo_id = engine.submit_order(moo_order);
    println!("✓ Market-on-Open order (ID: {})", moo_id);

    let mut moc_order = Order::market(0, "BTC/USD".to_string(), OrderSide::Sell, 1.0);
    moc_order.order_type = OrderType::MarketOnClose;
    let moc_id = engine.submit_order(moc_order);
    println!("✓ Market-on-Close order (ID: {})", moc_id);

    let mut loo_order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 50000.0);
    loo_order.order_type = OrderType::LimitOnOpen;
    let loo_id = engine.submit_order(loo_order);
    println!("✓ Limit-on-Open order (ID: {}) @ $50,000", loo_id);

    let mut loc_order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Sell, 1.0, 50000.0);
    loc_order.order_type = OrderType::LimitOnClose;
    let loc_id = engine.submit_order(loc_order);
    println!("✓ Limit-on-Close order (ID: {}) @ $50,000\n", loc_id);

    // ========================================
    // Example 3: Trailing Stops
    // ========================================
    println!("3. Trailing Stop Orders");
    println!("-----------------------");

    // Percentage-based trailing stop
    let trailing_pct = Order::trailing_stop(
        0,
        "BTC/USD".to_string(),
        OrderSide::Sell,
        1.0,
        None,
        Some(0.05), // 5% trailing
    );
    let trailing_pct_id = engine.submit_order(trailing_pct);
    println!(
        "✓ Trailing stop (percentage) submitted (ID: {}) - 5% trail",
        trailing_pct_id
    );

    // Absolute amount trailing stop
    let trailing_abs = Order::trailing_stop(
        0,
        "BTC/USD".to_string(),
        OrderSide::Sell,
        1.0,
        Some(2000.0), // $2000 trail
        None,
    );
    let trailing_abs_id = engine.submit_order(trailing_abs);
    println!(
        "✓ Trailing stop (absolute) submitted (ID: {}) - $2,000 trail\n",
        trailing_abs_id
    );

    // ========================================
    // Example 4: Iceberg Orders
    // ========================================
    println!("4. Iceberg Orders (Hidden Quantity)");
    println!("------------------------------------");

    let iceberg = Order::iceberg(
        0,
        "BTC/USD".to_string(),
        OrderSide::Buy,
        100.0,   // Total: 100
        50000.0, // Limit price
        10.0,    // Visible: 10
    );
    let iceberg_id = engine.submit_order(iceberg);
    println!("✓ Iceberg order submitted (ID: {})", iceberg_id);
    println!("  Total: 100 units, Visible: 10 units @ $50,000\n");

    // ========================================
    // Example 5: Algorithmic Orders (TWAP/VWAP/POV)
    // ========================================
    println!("5. Algorithmic Orders");
    println!("---------------------");

    // TWAP - Time-Weighted Average Price
    let twap = Order::twap(
        0,
        "BTC/USD".to_string(),
        OrderSide::Buy,
        1000.0, // Quantity
        3600,   // Duration: 1 hour
    );
    let twap_id = engine.submit_order(twap);
    println!("✓ TWAP order submitted (ID: {})", twap_id);
    println!("  Execute 1,000 units over 1 hour");

    // VWAP - Volume-Weighted Average Price
    let vwap = Order::vwap(
        0,
        "BTC/USD".to_string(),
        OrderSide::Buy,
        1000.0, // Quantity
        0.1,    // 10% participation rate
    );
    let vwap_id = engine.submit_order(vwap);
    println!("✓ VWAP order submitted (ID: {})", vwap_id);
    println!("  Execute 1,000 units at 10% of volume");

    // POV - Percentage of Volume
    let pov = Order::pov(
        0,
        "BTC/USD".to_string(),
        OrderSide::Buy,
        1000.0, // Quantity
        0.05,   // 5% of market volume
    );
    let pov_id = engine.submit_order(pov);
    println!("✓ POV order submitted (ID: {})", pov_id);
    println!("  Execute 1,000 units at 5% of market volume\n");

    // ========================================
    // Example 6: Time-in-Force Variations
    // ========================================
    println!("6. Time-in-Force Variations");
    println!("---------------------------");

    // Good-Till-Cancelled
    let mut gtc_order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 49000.0);
    gtc_order.time_in_force = TimeInForce::GTC;
    let gtc_id = engine.submit_order(gtc_order);
    println!(
        "✓ GTC order (ID: {}) - remains until filled or cancelled",
        gtc_id
    );

    // Good-Till-Date
    let mut gtd_order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 49000.0);
    gtd_order.time_in_force = TimeInForce::GTD { expiry: 1735689600 }; // Jan 1, 2025
    let gtd_id = engine.submit_order(gtd_order);
    println!("✓ GTD order (ID: {}) - expires Jan 1, 2025", gtd_id);

    // Day order
    let mut day_order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 49000.0);
    day_order.time_in_force = TimeInForce::Day;
    let day_id = engine.submit_order(day_order);
    println!("✓ Day order (ID: {}) - expires at market close", day_id);

    // Immediate-or-Cancel
    let mut ioc_order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 49000.0);
    ioc_order.time_in_force = TimeInForce::IOC;
    let ioc_id = engine.submit_order(ioc_order);
    println!("✓ IOC order (ID: {}) - fill immediately or cancel", ioc_id);

    // Fill-or-Kill
    let mut fok_order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 49000.0);
    fok_order.time_in_force = TimeInForce::FOK;
    let fok_id = engine.submit_order(fok_order);
    println!(
        "✓ FOK order (ID: {}) - fill entire order or cancel\n",
        fok_id
    );

    // ========================================
    // Example 7: Complex Order Groups
    // ========================================
    println!("7. Complex Order Groups");
    println!("-----------------------");

    // OCO (One-Cancels-Other)
    println!("OCO (One-Cancels-Other):");
    let buy_stop = Order::stop(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 51000.0);
    let sell_stop = Order::stop(0, "BTC/USD".to_string(), OrderSide::Sell, 1.0, 49000.0);

    let buy_stop_id = engine.submit_order(buy_stop);
    let sell_stop_id = engine.submit_order(sell_stop);

    let oco = OrderGroup::oco(1, vec![buy_stop_id, sell_stop_id]);
    engine.submit_order_group(oco);
    println!("  ✓ Buy stop @ $51,000 OR Sell stop @ $49,000");
    println!("  (When one fills, the other is cancelled)\n");

    // OTO (One-Triggers-Other)
    println!("OTO (One-Triggers-Other):");
    let entry = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 49000.0);
    let exit = Order::limit(0, "BTC/USD".to_string(), OrderSide::Sell, 1.0, 52000.0);

    let entry_id = engine.submit_order(entry);
    let exit_id = engine.submit_order(exit);

    let oto = OrderGroup::oto(entry_id, vec![exit_id]);
    engine.submit_order_group(oto);
    println!("  ✓ Entry @ $49,000 THEN Exit @ $52,000");
    println!("  (Exit order activates when entry fills)\n");

    // Bracket Order
    println!("Bracket Order (Entry + Stop Loss + Take Profit):");
    let bracket_entry = Order::market(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0);
    let stop_loss = Order::stop(0, "BTC/USD".to_string(), OrderSide::Sell, 1.0, 48000.0);
    let take_profit = Order::limit(0, "BTC/USD".to_string(), OrderSide::Sell, 1.0, 52000.0);

    let bracket_entry_id = engine.submit_order(bracket_entry);
    let stop_loss_id = engine.submit_order(stop_loss);
    let take_profit_id = engine.submit_order(take_profit);

    let bracket = OrderGroup::bracket(bracket_entry_id, stop_loss_id, take_profit_id);
    engine.submit_order_group(bracket);
    println!("  ✓ Entry: Market");
    println!("  ✓ Stop Loss: $48,000");
    println!("  ✓ Take Profit: $52,000");
    println!("  (When entry fills, SL and TP activate. When one exits, the other cancels)\n");

    // ========================================
    // Simulate Market and Show Results
    // ========================================
    println!("8. Simulation Results");
    println!("---------------------");

    // Create market snapshot at $50,000
    let bar = create_test_bar(1000, 50000.0, 1000.0);
    let mut market = MarketSnapshot::new(1000, bar);
    market.is_market_open = true;

    let fills = engine.match_orders(&market);

    println!("Market price: $50,000");
    println!("Fills executed: {}", fills.len());
    println!();

    for (i, fill) in fills.iter().enumerate() {
        println!(
            "Fill #{}: {} {} @ ${:.2} (qty: {:.2})",
            i + 1,
            if fill.side == OrderSide::Buy {
                "BUY"
            } else {
                "SELL"
            },
            fill.symbol,
            fill.price,
            fill.quantity
        );
    }

    println!();
    println!("Pending orders: {}", engine.pending_orders().len());
    println!("Completed orders: {}", engine.completed_orders().len());

    // ========================================
    // Summary
    // ========================================
    println!("\n=== Summary ===");
    println!("Order types demonstrated:");
    println!("  [1] Market, Limit, Stop, Stop-Limit");
    println!("  [2] MOO, MOC, LOO, LOC");
    println!("  [3] Trailing Stop (percentage & absolute)");
    println!("  [4] Iceberg");
    println!("  [5] TWAP, VWAP, POV");
    println!("  [6] Time-in-Force: GTC, GTD, Day, IOC, FOK");
    println!("  [7] Complex: OCO, OTO, Bracket");
    println!("\nTotal: 12+ order types implemented ✓");
}
