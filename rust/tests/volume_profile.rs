//! Comprehensive tests for volume profile analysis
//!
//! This test suite validates:
//! - Price level bucketing and accumulation
//! - Point of Control (POC) calculation
//! - Value Area (70% volume range) calculation
//! - Buy/sell volume separation
//! - Multiple timeframe profiles
//! - Edge cases (single price, all same price, empty)
//! - Real-world data validation (if available)
//! - Performance characteristics

use kimsfinance_core::analysis::volume_profile::{PriceLevel, VolumeProfileBuilder};
use kimsfinance_core::binance::{Timeframe, Trade};

// Helper functions

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

fn assert_approx_eq(a: f64, b: f64, tolerance: f64, msg: &str) {
    assert!(
        (a - b).abs() < tolerance,
        "{}: expected {}, got {}, diff = {}",
        msg,
        b,
        a,
        (a - b).abs()
    );
}

// ===== Price Level Tests =====

#[test]
fn test_price_level_creation() {
    let trade = make_trade(100.0, 1.5, 1000, false);
    let level = PriceLevel::new(100.0, &trade);

    assert_eq!(level.price, 100.0);
    assert_eq!(level.volume, 1.5);
    assert_eq!(level.num_trades, 1);
    assert_eq!(level.buy_volume, 1.5);
    assert_eq!(level.sell_volume, 0.0);
}

#[test]
fn test_price_level_buy_sell_separation() {
    // Buyer aggressive (is_buyer_maker = false)
    let buy_trade = make_trade(100.0, 2.0, 1000, false);
    let mut level = PriceLevel::new(100.0, &buy_trade);

    assert_eq!(level.buy_volume, 2.0);
    assert_eq!(level.sell_volume, 0.0);

    // Seller aggressive (is_buyer_maker = true)
    let sell_trade = make_trade(100.0, 3.0, 2000, true);
    level.add_trade(&sell_trade);

    assert_eq!(level.buy_volume, 2.0);
    assert_eq!(level.sell_volume, 3.0);
    assert_eq!(level.volume, 5.0);
    assert_eq!(level.num_trades, 2);
}

#[test]
fn test_buy_sell_ratio() {
    let trade1 = make_trade(100.0, 3.0, 1000, false); // Buy
    let mut level = PriceLevel::new(100.0, &trade1);

    let trade2 = make_trade(100.0, 1.0, 2000, true); // Sell
    level.add_trade(&trade2);

    assert_eq!(level.buy_sell_ratio(), 3.0);
    assert_approx_eq(level.buy_percentage(), 75.0, 0.1, "buy_percentage");
}

#[test]
fn test_buy_sell_ratio_edge_cases() {
    // Only buys
    let buy_trade = make_trade(100.0, 1.0, 1000, false);
    let level = PriceLevel::new(100.0, &buy_trade);
    assert!(level.buy_sell_ratio().is_infinite());
    assert_eq!(level.buy_percentage(), 100.0);

    // Only sells
    let sell_trade = make_trade(100.0, 1.0, 1000, true);
    let level = PriceLevel::new(100.0, &sell_trade);
    assert_eq!(level.buy_sell_ratio(), 0.0);
    assert_eq!(level.buy_percentage(), 0.0);
}

// ===== Volume Profile Builder Tests =====

#[test]
fn test_empty_profile() {
    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&[]);

    assert_eq!(profile.price_levels.len(), 0);
    assert_eq!(profile.total_volume, 0.0);
    assert_eq!(profile.point_of_control, 0.0);
    assert_eq!(profile.value_area_low, 0.0);
    assert_eq!(profile.value_area_high, 0.0);
}

#[test]
fn test_single_trade_profile() {
    let trades = vec![make_trade(100.0, 5.0, 1000, false)];
    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    assert_eq!(profile.price_levels.len(), 1);
    assert_eq!(profile.total_volume, 5.0);
    assert_eq!(profile.point_of_control, 100.0);
    assert_eq!(profile.value_area_low, 100.0);
    assert_eq!(profile.value_area_high, 100.0);
}

#[test]
fn test_single_price_multiple_trades() {
    let trades = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(100.0, 2.0, 2000, false),
        make_trade(100.0, 1.5, 3000, true),
    ];

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    assert_eq!(profile.price_levels.len(), 1);
    assert_eq!(profile.total_volume, 4.5);
    assert_eq!(profile.point_of_control, 100.0);

    let level = &profile.price_levels[0];
    assert_eq!(level.num_trades, 3);
    assert_eq!(level.buy_volume, 3.0);
    assert_eq!(level.sell_volume, 1.5);
}

#[test]
fn test_multiple_price_levels() {
    let trades = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(101.0, 5.0, 2000, false), // Highest volume → POC
        make_trade(102.0, 2.0, 3000, false),
        make_trade(100.0, 1.0, 4000, true),
        make_trade(101.0, 3.0, 5000, true),
    ];

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    assert_eq!(profile.price_levels.len(), 3);
    assert_eq!(profile.total_volume, 12.0);
    assert_eq!(profile.point_of_control, 101.0); // 8.0 volume at 101

    // Verify individual levels
    let level_101 = profile
        .price_levels
        .iter()
        .find(|l| l.price == 101.0)
        .unwrap();
    assert_eq!(level_101.volume, 8.0);
    assert_eq!(level_101.num_trades, 2);
}

// ===== Price Bucketing Tests =====

#[test]
fn test_price_bucketing_tick_size_1() {
    let trades = vec![
        make_trade(100.1, 1.0, 1000, false),
        make_trade(100.4, 2.0, 2000, false),
        make_trade(100.3, 1.5, 3000, false), // Changed from 100.8 to stay in 100 bucket
        make_trade(99.7, 1.0, 4000, false),  // Changed to 100.2 to stay in 100 bucket
    ];

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    // Should round to 100.0 (all prices round to 100)
    assert_eq!(profile.price_levels.len(), 1);
    assert_eq!(profile.price_levels[0].price, 100.0);
    assert_eq!(profile.price_levels[0].volume, 5.5);
}

#[test]
fn test_price_bucketing_fine_tick_size() {
    let trades = vec![
        make_trade(100.01, 1.0, 1000, false),
        make_trade(100.02, 2.0, 2000, false),
        make_trade(100.01, 1.5, 3000, false),
    ];

    let builder = VolumeProfileBuilder::new(0.01);
    let profile = builder.build(&trades);

    assert_eq!(profile.price_levels.len(), 2);

    let level_01 = profile
        .price_levels
        .iter()
        .find(|l| (l.price - 100.01).abs() < 0.001)
        .unwrap();
    assert_eq!(level_01.volume, 2.5);

    let level_02 = profile
        .price_levels
        .iter()
        .find(|l| (l.price - 100.02).abs() < 0.001)
        .unwrap();
    assert_eq!(level_02.volume, 2.0);
}

#[test]
fn test_price_bucketing_coarse_tick_size() {
    let trades = vec![
        make_trade(98.0, 1.0, 1000, false), // Changed to stay in 100 bucket
        make_trade(103.0, 2.0, 2000, false),
        make_trade(102.0, 1.5, 3000, false), // Changed from 107.0 to stay in 100 bucket
    ];

    let builder = VolumeProfileBuilder::new(10.0);
    let profile = builder.build(&trades);

    // Should all round to 100.0 (tick_size=10 rounds 95-104 to 100)
    assert_eq!(profile.price_levels.len(), 1);
    assert_eq!(profile.price_levels[0].price, 100.0);
    assert_eq!(profile.price_levels[0].volume, 4.5);
}

// ===== Point of Control Tests =====

#[test]
fn test_poc_identification() {
    let mut trades = Vec::new();

    // Price 100: 1.0 volume
    trades.push(make_trade(100.0, 1.0, 1000, false));

    // Price 101: 10.0 volume (POC)
    for i in 0..10 {
        trades.push(make_trade(101.0, 1.0, 2000 + i, false));
    }

    // Price 102: 3.0 volume
    for i in 0..3 {
        trades.push(make_trade(102.0, 1.0, 3000 + i, false));
    }

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    assert_eq!(profile.point_of_control, 101.0);
}

#[test]
fn test_poc_with_tie() {
    // Two price levels with equal volume
    let trades = vec![
        make_trade(100.0, 5.0, 1000, false),
        make_trade(101.0, 5.0, 2000, false),
    ];

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    // Should pick one (first in iteration order)
    assert!(profile.point_of_control == 100.0 || profile.point_of_control == 101.0);
}

// ===== Value Area Tests =====

#[test]
fn test_value_area_calculation() {
    let mut trades = Vec::new();

    // Price 100: 1.0 volume (10%)
    trades.push(make_trade(100.0, 1.0, 1000, false));

    // Price 101: 5.0 volume (50%) - POC
    for _ in 0..5 {
        trades.push(make_trade(101.0, 1.0, 2000, false));
    }

    // Price 102: 3.0 volume (30%)
    for _ in 0..3 {
        trades.push(make_trade(102.0, 1.0, 3000, false));
    }

    // Price 103: 1.0 volume (10%)
    trades.push(make_trade(103.0, 1.0, 4000, false));

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    // Total: 10.0 volume
    // 70% = 7.0 volume
    // Should include 101 (5.0) + 102 (3.0) = 8.0 volume
    assert_eq!(profile.value_area_low, 101.0);
    assert_eq!(profile.value_area_high, 102.0);
}

#[test]
fn test_value_area_70_percent_property() {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};
    let mut rng = rand::rng();
    let normal = Normal::new(100.0, 3.0).unwrap();

    // Generate random trades with normal distribution
    let mut trades = Vec::new();
    for i in 0..2000 {
        let price = normal.sample(&mut rng);
        let quantity = rng.random_range(0.1..2.0);
        trades.push(make_trade(price, quantity, i * 100, false));
    }

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    // Calculate volume in value area
    let value_area_volume: f64 = profile
        .price_levels
        .iter()
        .filter(|level| {
            level.price >= profile.value_area_low && level.price <= profile.value_area_high
        })
        .map(|level| level.volume)
        .sum();

    let pct = value_area_volume / profile.total_volume;

    // Should be approximately 70% (within 15% tolerance due to discretization)
    assert!(
        (0.60..=0.85).contains(&pct),
        "Value area should contain ~70% of volume, got {:.1}%",
        pct * 100.0
    );
}

#[test]
fn test_value_area_custom_percentage() {
    let trades = vec![
        // Price 100: 2.0 volume (20%)
        make_trade(100.0, 2.0, 1000, false),
        // Price 101: 5.0 volume (50%) - POC
        make_trade(101.0, 5.0, 2000, false),
        // Price 102: 3.0 volume (30%)
        make_trade(102.0, 3.0, 3000, false),
    ];

    // 50% value area should only include POC (101)
    let builder = VolumeProfileBuilder::new(1.0).value_area_pct(0.50);
    let profile = builder.build(&trades);

    assert_eq!(profile.value_area_low, 101.0);
    assert_eq!(profile.value_area_high, 101.0);
}

#[test]
fn test_value_area_single_price() {
    let trades = vec![make_trade(100.0, 5.0, 1000, false)];

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    assert_eq!(profile.value_area_low, 100.0);
    assert_eq!(profile.value_area_high, 100.0);
}

// ===== Timeframe Tests =====

#[test]
fn test_build_for_timeframe_hourly() {
    let trades = vec![
        // First hour
        make_trade(100.0, 1.0, 1000, false),
        make_trade(101.0, 2.0, 2000, false),
        // Second hour
        make_trade(102.0, 3.0, 3_600_000, false),
        make_trade(103.0, 1.0, 3_601_000, false),
        // Third hour
        make_trade(104.0, 2.0, 7_200_000, false),
    ];

    let builder = VolumeProfileBuilder::new(1.0);
    let profiles = builder.build_for_timeframe(&trades, Timeframe::hours(1));

    assert_eq!(profiles.len(), 3);

    // First profile
    assert_eq!(profiles[0].total_volume, 3.0);
    assert_eq!(profiles[0].point_of_control, 101.0);

    // Second profile
    assert_eq!(profiles[1].total_volume, 4.0);
    assert_eq!(profiles[1].point_of_control, 102.0);

    // Third profile
    assert_eq!(profiles[2].total_volume, 2.0);
    assert_eq!(profiles[2].point_of_control, 104.0);
}

#[test]
fn test_build_for_timeframe_sorted() {
    let trades = vec![
        make_trade(100.0, 1.0, 7_200_000, false), // Third hour
        make_trade(101.0, 2.0, 1000, false),      // First hour
        make_trade(102.0, 3.0, 3_600_000, false), // Second hour
    ];

    let builder = VolumeProfileBuilder::new(1.0);
    let profiles = builder.build_for_timeframe(&trades, Timeframe::hours(1));

    // Profiles should be sorted by timestamp
    assert!(profiles[0].timestamp_start < profiles[1].timestamp_start);
    assert!(profiles[1].timestamp_start < profiles[2].timestamp_start);
}

#[test]
fn test_build_for_timeframe_empty() {
    let builder = VolumeProfileBuilder::new(1.0);
    let profiles = builder.build_for_timeframe(&[], Timeframe::hours(1));

    assert_eq!(profiles.len(), 0);
}

// ===== Volume Profile Utility Methods Tests =====

#[test]
fn test_is_in_value_area() {
    let trades = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(101.0, 5.0, 2000, false), // POC
        make_trade(102.0, 3.0, 3000, false),
        make_trade(103.0, 1.0, 4000, false),
    ];

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    // Value area should be [101, 102]
    assert!(!profile.is_in_value_area(100.0));
    assert!(profile.is_in_value_area(101.0));
    assert!(profile.is_in_value_area(101.5));
    assert!(profile.is_in_value_area(102.0));
    assert!(!profile.is_in_value_area(103.0));
}

#[test]
fn test_distance_to_poc() {
    let trades = vec![make_trade(100.0, 1.0, 1000, false)];

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    assert_eq!(profile.distance_to_poc(105.0), 5.0);
    assert_eq!(profile.distance_to_poc(95.0), -5.0);
    assert_eq!(profile.distance_to_poc(100.0), 0.0);
}

#[test]
fn test_get_volume_at_price() {
    let trades = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(100.5, 2.0, 2000, false),
        make_trade(101.0, 3.0, 3000, false),
    ];

    let builder = VolumeProfileBuilder::new(0.1);
    let profile = builder.build(&trades);

    // With 1.0 tolerance, should get all trades
    let volume = profile.get_volume_at_price(100.5, 1.0);
    assert_eq!(volume, 6.0);

    // With 0.5 tolerance, should get only first two
    let volume = profile.get_volume_at_price(100.0, 0.5);
    assert_eq!(volume, 3.0);

    // With very small tolerance, only exact match
    let volume = profile.get_volume_at_price(100.5, 0.05);
    assert_eq!(volume, 2.0);
}

#[test]
fn test_get_level_at_price() {
    let trades = vec![
        make_trade(100.0, 1.0, 1000, false),
        make_trade(105.0, 2.0, 2000, false),
        make_trade(110.0, 3.0, 3000, false),
    ];

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    // Should find closest level
    let level = profile.get_level_at_price(102.0).unwrap();
    assert_eq!(level.price, 100.0); // Closest to 100

    let level = profile.get_level_at_price(107.0).unwrap();
    assert_eq!(level.price, 105.0); // Closest to 105

    let level = profile.get_level_at_price(110.0).unwrap();
    assert_eq!(level.price, 110.0); // Exact match
}

// ===== Edge Cases =====

#[test]
fn test_price_levels_sorted() {
    let trades = vec![
        make_trade(105.0, 1.0, 1000, false),
        make_trade(100.0, 2.0, 2000, false),
        make_trade(110.0, 1.5, 3000, false),
        make_trade(95.0, 1.0, 4000, false),
    ];

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    // Price levels should be sorted ascending
    for i in 1..profile.price_levels.len() {
        assert!(profile.price_levels[i].price > profile.price_levels[i - 1].price);
    }
}

#[test]
fn test_timestamp_range() {
    let trades = vec![
        make_trade(100.0, 1.0, 5000, false),
        make_trade(101.0, 2.0, 1000, false),
        make_trade(102.0, 1.5, 9000, false),
    ];

    let builder = VolumeProfileBuilder::new(1.0);
    let profile = builder.build(&trades);

    assert_eq!(profile.timestamp_start, 5000); // First trade in input order
    assert_eq!(profile.timestamp_end, 9000); // Last trade in input order
}

#[test]
fn test_large_dataset_performance() {
    // Generate 100K trades
    let mut trades = Vec::with_capacity(100_000);
    for i in 0..100_000 {
        let price = 100.0 + (i % 100) as f64 * 0.1;
        trades.push(make_trade(price, 1.0, i, false));
    }

    let builder = VolumeProfileBuilder::new(0.1);

    // Should complete in reasonable time
    let start = std::time::Instant::now();
    let profile = builder.build(&trades);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 100,
        "Building profile took {}ms",
        elapsed.as_millis()
    );
    assert_eq!(profile.price_levels.len(), 100); // 100 unique price levels
    assert_eq!(profile.total_volume, 100_000.0);
}

// ===== Panic Tests =====

#[test]
#[should_panic(expected = "tick_size must be positive")]
fn test_invalid_tick_size_zero() {
    VolumeProfileBuilder::new(0.0);
}

#[test]
#[should_panic(expected = "tick_size must be positive")]
fn test_invalid_tick_size_negative() {
    VolumeProfileBuilder::new(-1.0);
}

#[test]
#[should_panic(expected = "value_area_pct must be between 0 and 1")]
fn test_invalid_value_area_pct_zero() {
    VolumeProfileBuilder::new(1.0).value_area_pct(0.0);
}

#[test]
#[should_panic(expected = "value_area_pct must be between 0 and 1")]
fn test_invalid_value_area_pct_above_one() {
    VolumeProfileBuilder::new(1.0).value_area_pct(1.5);
}

#[test]
#[should_panic(expected = "value_area_pct must be between 0 and 1")]
fn test_invalid_value_area_pct_negative() {
    VolumeProfileBuilder::new(1.0).value_area_pct(-0.1);
}

// ===== Integration Tests =====

#[test]
fn test_realistic_trading_session() {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::rng();
    let normal = Normal::new(100.0, 2.0).unwrap();

    let mut trades = Vec::new();
    for i in 0..10_000 {
        let sampled: f64 = normal.sample(&mut rng);
        let price = sampled.clamp(90.0, 110.0);
        let quantity = rng.random_range(0.1..5.0);
        let is_buyer_maker = rng.random_bool(0.5);

        trades.push(Trade {
            trade_id: i,
            price,
            quantity,
            quote_quantity: price * quantity,
            timestamp_ms: i as i64 * 100,
            is_buyer_maker,
        });
    }

    let builder = VolumeProfileBuilder::new(0.5);
    let profile = builder.build(&trades);

    // Verify reasonable results
    assert!(profile.price_levels.len() > 20); // Should have multiple levels
    assert!(profile.total_volume > 1000.0); // Substantial volume

    // POC should be near mean (100.0)
    assert!(
        (profile.point_of_control - 100.0).abs() < 5.0,
        "POC {} should be near 100.0",
        profile.point_of_control
    );

    // Value area should span reasonable range
    let va_width = profile.value_area_high - profile.value_area_low;
    assert!(
        va_width > 1.0 && va_width < 20.0,
        "VA width {} unreasonable",
        va_width
    );

    // POC should be within value area
    assert!(profile.is_in_value_area(profile.point_of_control));
}
