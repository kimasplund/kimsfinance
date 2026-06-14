//! Comprehensive tests for order management and matching engine
//!
//! Tests all 12+ order types and complex order scenarios

use kimsfinance_core::backtest::{
    Fill, MarketSnapshot, MatchingEngine, OHLCVBar, Order, OrderGroup, OrderSide, OrderStatus,
    OrderType, TimeInForce,
};

fn create_test_bar(timestamp: i64, close: f64, volume: f64, low: f64, high: f64) -> OHLCVBar {
    OHLCVBar {
        timestamp,
        open: close * 0.999,
        high,
        low,
        close,
        volume,
    }
}

fn create_market_snapshot(timestamp: i64, close: f64, volume: f64) -> MarketSnapshot {
    let bar = create_test_bar(timestamp, close, volume, close * 0.999, close * 1.001);
    MarketSnapshot::new(timestamp, bar)
}

#[test]
fn test_market_order() {
    let mut engine = MatchingEngine::new();
    let order = Order::market(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0);

    let order_id = engine.submit_order(order);
    let market = create_market_snapshot(1000, 50000.0, 100.0);
    let fills = engine.match_orders(&market);

    assert_eq!(fills.len(), 1);
    assert_eq!(fills[0].quantity, 1.0);
    assert!(fills[0].price > 0.0);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.status, OrderStatus::Filled);
    assert_eq!(order.filled_quantity, 1.0);
}

#[test]
fn test_limit_order_buy() {
    let mut engine = MatchingEngine::new();
    let order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 49900.0);

    let order_id = engine.submit_order(order);

    // Price above limit - should not fill
    let market1 = create_market_snapshot(1000, 50000.0, 100.0);
    let fills1 = engine.match_orders(&market1);
    assert_eq!(fills1.len(), 0);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.status, OrderStatus::Pending);

    // Price below limit - should fill
    let mut bar2 = create_test_bar(2000, 49800.0, 100.0, 49800.0, 50000.0);
    let market2 = MarketSnapshot::new(2000, bar2);
    let fills2 = engine.match_orders(&market2);

    assert_eq!(fills2.len(), 1);
    assert_eq!(fills2[0].price, 49900.0);
}

#[test]
fn test_limit_order_sell() {
    let mut engine = MatchingEngine::new();
    let order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Sell, 1.0, 50100.0);

    let order_id = engine.submit_order(order);

    // Price below limit - should not fill
    let market1 = create_market_snapshot(1000, 50000.0, 100.0);
    let fills1 = engine.match_orders(&market1);
    assert_eq!(fills1.len(), 0);

    // Price above limit - should fill
    let mut bar2 = create_test_bar(2000, 50200.0, 100.0, 50000.0, 50200.0);
    let market2 = MarketSnapshot::new(2000, bar2);
    let fills2 = engine.match_orders(&market2);

    assert_eq!(fills2.len(), 1);
    assert_eq!(fills2[0].price, 50100.0);
}

#[test]
fn test_stop_order() {
    let mut engine = MatchingEngine::new();
    let order = Order::stop(0, "BTC/USD".to_string(), OrderSide::Sell, 1.0, 49000.0);

    let order_id = engine.submit_order(order);

    // Price above stop - should not trigger
    let market1 = create_market_snapshot(1000, 50000.0, 100.0);
    let fills1 = engine.match_orders(&market1);
    assert_eq!(fills1.len(), 0);

    // Price hits stop - should trigger and fill
    let mut bar2 = create_test_bar(2000, 48900.0, 100.0, 48900.0, 50000.0);
    let market2 = MarketSnapshot::new(2000, bar2);
    let fills2 = engine.match_orders(&market2);

    assert_eq!(fills2.len(), 1);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.status, OrderStatus::Filled);
}

#[test]
fn test_stop_limit_order() {
    let mut engine = MatchingEngine::new();
    let order = Order::stop_limit(
        0,
        "BTC/USD".to_string(),
        OrderSide::Sell,
        1.0,
        49000.0,
        48900.0,
    );

    let order_id = engine.submit_order(order);

    // Price above stop - should not trigger
    let market1 = create_market_snapshot(1000, 50000.0, 100.0);
    let fills1 = engine.match_orders(&market1);
    assert_eq!(fills1.len(), 0);

    // Price hits stop but doesn't reach limit - triggers but doesn't fill
    let mut bar2 = create_test_bar(2000, 48800.0, 100.0, 48800.0, 48850.0);
    let market2 = MarketSnapshot::new(2000, bar2);
    let fills2 = engine.match_orders(&market2);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.status, OrderStatus::Triggered);
    assert_eq!(fills2.len(), 0);

    // Price reaches limit - should fill
    let mut bar3 = create_test_bar(3000, 48800.0, 100.0, 48800.0, 49500.0);
    let market3 = MarketSnapshot::new(3000, bar3);
    let fills3 = engine.match_orders(&market3);

    assert_eq!(fills3.len(), 1);
    assert_eq!(fills3[0].price, 48900.0);
}

#[test]
fn test_trailing_stop_sell() {
    let mut engine = MatchingEngine::new();
    let order = Order::trailing_stop(
        0,
        "BTC/USD".to_string(),
        OrderSide::Sell,
        1.0,
        None,
        Some(0.05), // 5% trail
    );

    let order_id = engine.submit_order(order);

    // Initialize at 50000 - trail = 47500
    let market1 = create_market_snapshot(1000, 50000.0, 100.0);
    engine.match_orders(&market1);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.trailing_high_water_mark, Some(50000.0));

    // Price rises to 52000 - trail should update to 49400
    let market2 = create_market_snapshot(2000, 52000.0, 100.0);
    engine.match_orders(&market2);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.trailing_high_water_mark, Some(52000.0));

    // Price drops below trail - should trigger and fill
    let mut bar3 = create_test_bar(3000, 49000.0, 100.0, 49000.0, 52000.0);
    let market3 = MarketSnapshot::new(3000, bar3);
    let fills = engine.match_orders(&market3);

    assert_eq!(fills.len(), 1);
}

#[test]
fn test_trailing_stop_buy() {
    let mut engine = MatchingEngine::new();
    let order = Order::trailing_stop(
        0,
        "BTC/USD".to_string(),
        OrderSide::Buy,
        1.0,
        Some(1000.0), // $1000 absolute trail
        None,
    );

    let order_id = engine.submit_order(order);

    // Initialize at 50000 - trail = 51000
    let market1 = create_market_snapshot(1000, 50000.0, 100.0);
    engine.match_orders(&market1);

    // Price drops to 48000 - trail should update to 49000
    let market2 = create_market_snapshot(2000, 48000.0, 100.0);
    engine.match_orders(&market2);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.trailing_high_water_mark, Some(48000.0));

    // Price rises above trail - should trigger
    let mut bar3 = create_test_bar(3000, 49500.0, 100.0, 48000.0, 49500.0);
    let market3 = MarketSnapshot::new(3000, bar3);
    let fills = engine.match_orders(&market3);

    assert_eq!(fills.len(), 1);
}

#[test]
fn test_market_on_open() {
    let mut engine = MatchingEngine::new();
    let order = Order::market(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0);

    // Manually set order type to MOO
    let mut order = order;
    order.order_type = OrderType::MarketOnOpen;

    let order_id = engine.submit_order(order);

    // Regular bar - should not fill
    let market1 = create_market_snapshot(1000, 50000.0, 100.0);
    let fills1 = engine.match_orders(&market1);
    assert_eq!(fills1.len(), 0);

    // Market open bar - should fill
    let mut market2 = create_market_snapshot(2000, 50100.0, 100.0);
    market2.is_market_open = true;
    let fills2 = engine.match_orders(&market2);

    assert_eq!(fills2.len(), 1);
}

#[test]
fn test_market_on_close() {
    let mut engine = MatchingEngine::new();
    let order = Order::market(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0);

    let mut order = order;
    order.order_type = OrderType::MarketOnClose;

    let order_id = engine.submit_order(order);

    // Regular bar - should not fill
    let market1 = create_market_snapshot(1000, 50000.0, 100.0);
    let fills1 = engine.match_orders(&market1);
    assert_eq!(fills1.len(), 0);

    // Market close bar - should fill
    let mut market2 = create_market_snapshot(2000, 50100.0, 100.0);
    market2.is_market_close = true;
    let fills2 = engine.match_orders(&market2);

    assert_eq!(fills2.len(), 1);
}

#[test]
fn test_iceberg_order() {
    let mut engine = MatchingEngine::new();
    let order = Order::iceberg(0, "BTC/USD".to_string(), OrderSide::Buy, 10.0, 50000.0, 2.0);

    let order_id = engine.submit_order(order);

    // First fill - should only fill visible quantity (2.0)
    let mut bar1 = create_test_bar(1000, 49900.0, 100.0, 49900.0, 50100.0);
    let market1 = MarketSnapshot::new(1000, bar1);
    let fills1 = engine.match_orders(&market1);

    assert_eq!(fills1.len(), 1);
    assert_eq!(fills1[0].quantity, 2.0);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.remaining_quantity, 8.0);
    assert_eq!(order.status, OrderStatus::PartiallyFilled);
}

#[test]
fn test_twap_order() {
    let mut engine = MatchingEngine::new();
    let mut order = Order::twap(0, "BTC/USD".to_string(), OrderSide::Buy, 100.0, 10); // 10 seconds
    order.created_at = 0; // Align with test timestamps

    let order_id = engine.submit_order(order);

    // At t=0, should execute 0%
    let market1 = create_market_snapshot(0, 50000.0, 1000.0);
    let fills1 = engine.match_orders(&market1);
    assert_eq!(fills1.len(), 0);

    // At t=5 (50%), should execute ~50
    let market2 = create_market_snapshot(5, 50100.0, 1000.0);
    let fills2 = engine.match_orders(&market2);
    assert!(fills2.len() > 0);
    assert!(fills2[0].quantity > 45.0 && fills2[0].quantity < 55.0);

    // At t=10 (100%), should execute remaining
    let market3 = create_market_snapshot(10, 50200.0, 1000.0);
    let fills3 = engine.match_orders(&market3);
    assert!(fills3.len() > 0);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.status, OrderStatus::Filled);
    assert!((order.filled_quantity - 100.0).abs() < 1.0);
}

#[test]
fn test_vwap_order() {
    let mut engine = MatchingEngine::new();
    let order = Order::vwap(0, "BTC/USD".to_string(), OrderSide::Buy, 100.0, 0.1); // 10% participation

    let order_id = engine.submit_order(order);

    // Bar with 500 volume - should execute 50 (10% of 500)
    let market1 = create_market_snapshot(1000, 50000.0, 500.0);
    let fills1 = engine.match_orders(&market1);

    assert_eq!(fills1.len(), 1);
    assert_eq!(fills1[0].quantity, 50.0);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.remaining_quantity, 50.0);

    // Bar with 1000 volume - should execute remaining 50
    let market2 = create_market_snapshot(2000, 50100.0, 1000.0);
    let fills2 = engine.match_orders(&market2);

    assert_eq!(fills2.len(), 1);
    assert_eq!(fills2[0].quantity, 50.0);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.status, OrderStatus::Filled);
}

#[test]
fn test_pov_order() {
    let mut engine = MatchingEngine::new();
    let order = Order::pov(0, "BTC/USD".to_string(), OrderSide::Buy, 100.0, 0.05); // 5% of volume

    let order_id = engine.submit_order(order);

    // Bar with 1000 volume - should execute 50 (5% of 1000)
    let market1 = create_market_snapshot(1000, 50000.0, 1000.0);
    let fills1 = engine.match_orders(&market1);

    assert_eq!(fills1.len(), 1);
    assert_eq!(fills1[0].quantity, 50.0);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.remaining_quantity, 50.0);

    // Bar with 2000 volume - should execute remaining 50
    let market2 = create_market_snapshot(2000, 50100.0, 2000.0);
    let fills2 = engine.match_orders(&market2);

    assert_eq!(fills2.len(), 1);
    assert_eq!(fills2[0].quantity, 50.0);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.status, OrderStatus::Filled);
}

#[test]
fn test_time_in_force_day() {
    let mut engine = MatchingEngine::new();
    let mut order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 49000.0);
    order.time_in_force = TimeInForce::Day;

    let order_id = engine.submit_order(order);

    // Regular bar - should remain pending
    let market1 = create_market_snapshot(1000, 50000.0, 100.0);
    let fills1 = engine.match_orders(&market1);
    assert_eq!(fills1.len(), 0);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.status, OrderStatus::Pending);

    // Market close - should expire
    let mut market2 = create_market_snapshot(2000, 50100.0, 100.0);
    market2.is_market_close = true;
    engine.match_orders(&market2);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.status, OrderStatus::Expired);
}

#[test]
fn test_time_in_force_gtd() {
    let mut engine = MatchingEngine::new();
    let mut order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 49000.0);
    order.time_in_force = TimeInForce::GTD { expiry: 5000 };

    let order_id = engine.submit_order(order);

    // Before expiry - should remain pending
    let market1 = create_market_snapshot(4000, 50000.0, 100.0);
    let fills1 = engine.match_orders(&market1);
    assert_eq!(fills1.len(), 0);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.status, OrderStatus::Pending);

    // After expiry - should expire
    let market2 = create_market_snapshot(6000, 50100.0, 100.0);
    engine.match_orders(&market2);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.status, OrderStatus::Expired);
}

#[test]
fn test_order_cancellation() {
    let mut engine = MatchingEngine::new();
    let order = Order::limit(0, "BTC/USD".to_string(), OrderSide::Buy, 1.0, 49000.0);

    let order_id = engine.submit_order(order);

    // Cancel order
    let cancelled = engine.cancel_order(order_id);
    assert!(cancelled);

    let order = engine.get_order(order_id).unwrap();
    assert_eq!(order.status, OrderStatus::Cancelled);

    // Should not match after cancellation
    let market = create_market_snapshot(1000, 48000.0, 100.0);
    let fills = engine.match_orders(&market);
    assert_eq!(fills.len(), 0);
}

#[test]
fn test_average_fill_price() {
    let mut engine = MatchingEngine::new();
    let order = Order::vwap(0, "BTC/USD".to_string(), OrderSide::Buy, 100.0, 0.5); // 50% participation

    engine.submit_order(order);

    // First fill at 50000
    let market1 = create_market_snapshot(1000, 50000.0, 100.0);
    engine.match_orders(&market1);

    // Second fill at 51000
    let market2 = create_market_snapshot(2000, 51000.0, 100.0);
    engine.match_orders(&market2);

    let order = engine.completed_orders().values().next().unwrap();
    assert_eq!(order.status, OrderStatus::Filled);

    // Average should be around 50500
    assert!(order.average_fill_price > 50400.0 && order.average_fill_price < 50600.0);
}
