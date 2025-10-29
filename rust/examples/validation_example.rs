//! Example demonstrating the data validation module
//!
//! Run with: cargo run --example validation_example

use kimsfinance_core::binance::Trade;
use kimsfinance_core::validation::{DataQualityReport, GapDetector, OutlierDetector};

fn main() {
    println!("=== Data Validation Module Example ===\n");

    // Create sample trade data with some issues
    let mut trades = vec![];

    // Normal trades (10 seconds apart)
    for i in 0..10 {
        trades.push(Trade {
            trade_id: i as u64,
            price: 100.0 + (i as f64),
            quantity: 1.0,
            quote_quantity: 100.0 + (i as f64),
            timestamp_ms: (i * 10_000) as i64,
            is_buyer_maker: false,
        });
    }

    // Gap: 10 minute gap
    trades.push(Trade {
        trade_id: 10,
        price: 110.0,
        quantity: 1.0,
        quote_quantity: 110.0,
        timestamp_ms: 700_000, // 700 seconds = 11.67 minutes later
        is_buyer_maker: false,
    });

    // Outlier: extreme price
    trades.push(Trade {
        trade_id: 11,
        price: 1000.0, // Way above normal
        quantity: 1.0,
        quote_quantity: 1000.0,
        timestamp_ms: 710_000,
        is_buyer_maker: false,
    });

    // Duplicate ID
    trades.push(Trade {
        trade_id: 10, // Duplicate!
        price: 111.0,
        quantity: 1.0,
        quote_quantity: 111.0,
        timestamp_ms: 720_000,
        is_buyer_maker: false,
    });

    // Zero price (data corruption)
    trades.push(Trade {
        trade_id: 13,
        price: 0.0, // Invalid!
        quantity: 1.0,
        quote_quantity: 0.0,
        timestamp_ms: 730_000,
        is_buyer_maker: false,
    });

    println!("Sample data created: {} trades\n", trades.len());

    // Example 1: Gap Detection
    println!("--- Gap Detection ---");
    let gap_detector = GapDetector::new(600_000); // 10 minutes
    let gaps = gap_detector.find_gaps(&trades);
    println!("Found {} gap(s):", gaps.len());
    for gap in &gaps {
        println!("  {}", gap);
    }
    println!();

    // Example 2: Outlier Detection
    println!("--- Outlier Detection ---");
    let outlier_detector = OutlierDetector::new(3.0); // 3 standard deviations
    let outliers = outlier_detector.find_outliers(&trades);
    println!("Found {} outlier(s):", outliers.len());
    for outlier in &outliers {
        println!("  {}", outlier);
    }
    println!();

    // Example 3: Comprehensive Data Quality Report
    println!("--- Comprehensive Quality Report ---");
    let report = DataQualityReport::generate(&trades);
    report.print_summary();
    println!();

    // Example 4: Gap Statistics
    println!("--- Gap Statistics ---");
    if !gaps.is_empty() {
        let (total_gap_time, max_gap, avg_gap) = gap_detector.gap_statistics(&gaps);
        println!("Total gap time: {} ms", total_gap_time);
        println!("Maximum gap: {} ms", max_gap);
        println!("Average gap: {:.2} ms", avg_gap);
    } else {
        println!("No gaps found");
    }
    println!();

    // Example 5: Outlier Statistics
    println!("--- Outlier Statistics ---");
    if !outliers.is_empty() {
        let (count, max_z, avg_z) = outlier_detector.outlier_statistics(&outliers);
        println!("Outlier count: {}", count);
        println!("Maximum z-score: {:.2}σ", max_z);
        println!("Average z-score: {:.2}σ", avg_z);
    } else {
        println!("No outliers found");
    }
    println!();

    // Example 6: Price Statistics
    println!("--- Price Statistics ---");
    let (mean, std_dev) = outlier_detector.price_statistics(&trades);
    println!("Mean price: {:.2}", mean);
    println!("Standard deviation: {:.2}", std_dev);
    println!();

    println!("=== Example Complete ===");
}
