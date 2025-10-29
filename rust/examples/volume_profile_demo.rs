//! Volume Profile Analysis Demo
//!
//! Demonstrates volume profile analysis on simulated Binance trade data.
//!
//! This example shows:
//! - Building volume profile for a trading session
//! - Identifying Point of Control (POC) and Value Areas
//! - Visualizing price levels by volume distribution
//! - Using volume profile in a trading strategy
//!
//! # Usage
//!
//! ```bash
//! cargo run --example volume_profile_demo
//! ```
//!
//! # Output
//!
//! - Volume profile statistics (POC, VAH, VAL, total volume)
//! - Top 10 price levels by volume
//! - Buy/sell volume distribution
//! - Trading signals based on volume profile

use kimsfinance_core::analysis::volume_profile::VolumeProfileBuilder;
use kimsfinance_core::backtest::tick_strategy::TickStrategy;
use kimsfinance_core::backtest::VolumeProfileStrategy;
use kimsfinance_core::binance::{IncompleteCandle, Trade};
use std::time::Duration;

/// Generate sample trades for demonstration
///
/// Simulates a trading session with:
/// - Normal distribution around $100 (POC)
/// - High volume between $98-$102 (value area)
/// - Lower volume at extremes
fn generate_sample_trades() -> Vec<Trade> {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::thread_rng();

    // Normal distribution centered at $100 with std dev of $2
    let normal = Normal::new(100.0, 2.0).unwrap();

    let mut trades = Vec::with_capacity(10_000);
    let base_timestamp = 1_609_459_200_000i64; // 2021-01-01 00:00:00

    for i in 0..10_000 {
        let sampled: f64 = normal.sample(&mut rng);
        let price = sampled.clamp(90.0, 110.0); // Clamp to reasonable range
        let quantity = rng.gen_range(0.1..5.0);
        let is_buyer_maker = rng.gen_bool(0.5);

        trades.push(Trade {
            trade_id: i as u64,
            price,
            quantity,
            quote_quantity: price * quantity,
            timestamp_ms: base_timestamp + (i as i64 * 100), // 100ms apart
            is_buyer_maker,
        });
    }

    trades
}

fn main() {
    println!("========================================");
    println!("   Volume Profile Analysis Demo");
    println!("========================================\n");

    // Generate sample data
    println!("Generating 10,000 sample trades...");
    let trades = generate_sample_trades();
    println!("✓ Generated {} trades\n", trades.len());

    // Build volume profile with $0.50 tick size
    println!("Building volume profile ($0.50 tick size)...");
    let builder = VolumeProfileBuilder::new(0.5);
    let profile = builder.build(&trades);
    println!("✓ Profile built with {} price levels\n", profile.price_levels.len());

    // Display volume profile statistics
    println!("Volume Profile Statistics");
    println!("-------------------------");
    println!("Time Range: {} - {}", profile.timestamp_start, profile.timestamp_end);
    println!("Total Volume: {:.2} BTC", profile.total_volume);
    println!("Point of Control (POC): ${:.2}", profile.point_of_control);
    println!("Value Area High (VAH): ${:.2}", profile.value_area_high);
    println!("Value Area Low (VAL): ${:.2}", profile.value_area_low);
    println!(
        "Value Area Width: ${:.2}\n",
        profile.value_area_high - profile.value_area_low
    );

    // Show top 10 price levels by volume
    println!("Top 10 Price Levels by Volume");
    println!("------------------------------");
    println!("{:<8} {:<12} {:<8} {:<10} {:<10}", "Price", "Volume", "Trades", "Buy %", "Sell %");
    println!("{}", "-".repeat(60));

    let mut sorted_levels = profile.price_levels.clone();
    sorted_levels.sort_by(|a, b| b.volume.partial_cmp(&a.volume).unwrap());

    for (i, level) in sorted_levels.iter().take(10).enumerate() {
        let buy_pct = (level.buy_volume / level.volume * 100.0).round();
        let sell_pct = (level.sell_volume / level.volume * 100.0).round();

        let marker = if (level.price - profile.point_of_control).abs() < 0.5 {
            " ← POC"
        } else if (level.price - profile.value_area_high).abs() < 0.5 {
            " ← VAH"
        } else if (level.price - profile.value_area_low).abs() < 0.5 {
            " ← VAL"
        } else {
            ""
        };

        println!(
            "{:<8.2} {:<12.2} {:<8} {:<10.0} {:<10.0}{}",
            level.price, level.volume, level.num_trades, buy_pct, sell_pct, marker
        );
    }
    println!();

    // Demonstrate volume profile strategy
    println!("Volume Profile Trading Strategy Demo");
    println!("-------------------------------------");

    let mut strategy = VolumeProfileStrategy::new(
        0.5,                          // $0.50 tick size
        Duration::from_secs(600),     // 10 minute lookback
        0.03,                         // 3% distance threshold
    )
    .rebuild_interval(50); // Rebuild every 50 trades

    // Simulate strategy on sample trades
    let mut buy_signals = 0;
    let mut sell_signals = 0;
    let mut hold_signals = 0;

    let mut candle = IncompleteCandle::new(&trades[0], trades[0].timestamp_ms);

    println!("Processing {} trades through strategy...", trades.len());

    for trade in &trades {
        candle.update(trade);
        let signal = strategy.on_tick(trade, &candle);

        match signal {
            kimsfinance_core::backtest::Signal::Buy => buy_signals += 1,
            kimsfinance_core::backtest::Signal::Sell => sell_signals += 1,
            kimsfinance_core::backtest::Signal::Hold => hold_signals += 1,
            _ => {}
        }
    }

    println!("\nStrategy Results:");
    println!("  Buy Signals:  {} ({:.1}%)", buy_signals, buy_signals as f64 / trades.len() as f64 * 100.0);
    println!("  Sell Signals: {} ({:.1}%)", sell_signals, sell_signals as f64 / trades.len() as f64 * 100.0);
    println!("  Hold Signals: {} ({:.1}%)", hold_signals, hold_signals as f64 / trades.len() as f64 * 100.0);

    // Show current profile from strategy
    if let Some(current_profile) = strategy.current_profile() {
        println!("\nCurrent Strategy Profile:");
        println!("  POC: ${:.2}", current_profile.point_of_control);
        println!("  VAH: ${:.2}", current_profile.value_area_high);
        println!("  VAL: ${:.2}", current_profile.value_area_low);
        println!("  Total Volume: {:.2} BTC", current_profile.total_volume);
    }

    // Demonstrate multi-timeframe analysis
    println!("\n\nMulti-Timeframe Volume Profile");
    println!("-------------------------------");

    let profiles = builder.build_for_timeframe(&trades, kimsfinance_core::binance::Timeframe::seconds(300));
    println!("Built {} profiles (5-minute intervals)\n", profiles.len());

    println!("{:<5} {:<12} {:<12} {:<12} {:<12}", "Period", "POC", "VAH", "VAL", "Volume");
    println!("{}", "-".repeat(65));

    for (i, prof) in profiles.iter().enumerate() {
        println!(
            "{:<5} ${:<11.2} ${:<11.2} ${:<11.2} {:<12.2}",
            i + 1,
            prof.point_of_control,
            prof.value_area_high,
            prof.value_area_low,
            prof.total_volume
        );
    }

    println!("\n========================================");
    println!("   Volume Profile Analysis Complete");
    println!("========================================\n");

    println!("Key Insights:");
    println!("  • POC represents fair value / equilibrium price");
    println!("  • Value Area (VAH-VAL) contains 70% of volume");
    println!("  • High volume nodes act as support/resistance");
    println!("  • Strategy generates signals near VA boundaries");
    println!("\nFor more information, see docs/VOLUME_PROFILE.md");
}
