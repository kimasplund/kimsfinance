//! Example demonstrating flexible timeframe parsing
//!
//! Run with: cargo run --example timeframe_parsing

use kimsfinance_core::binance::{Timeframe, TimeframeEnum};

fn main() {
    println!("=== Flexible Timeframe System Demo ===\n");

    // 1. Parse various duration strings
    println!("1. Parsing duration strings:");
    let examples = vec!["5m", "1h", "45s", "2d", "7m", "33s", "15M", "4H"];

    for example in examples {
        match Timeframe::parse(example) {
            Ok(tf) => println!("   '{}' -> {}ms", example, tf.to_ms()),
            Err(e) => println!("   '{}' -> Error: {}", example, e),
        }
    }

    // 2. Direct construction
    println!("\n2. Direct construction:");
    let tf1 = Timeframe::minutes(5);
    let tf2 = Timeframe::hours(1);
    let tf3 = Timeframe::seconds(30);
    println!("   Timeframe::minutes(5) -> {}ms", tf1.to_ms());
    println!("   Timeframe::hours(1) -> {}ms", tf2.to_ms());
    println!("   Timeframe::seconds(30) -> {}ms", tf3.to_ms());

    // 3. Backward compatibility
    println!("\n3. Backward compatibility with old enum:");
    #[allow(deprecated)]
    {
        let old_tf = TimeframeEnum::FiveMinutes;
        let new_tf: Timeframe = old_tf.into();
        println!("   TimeframeEnum::FiveMinutes -> {}ms", old_tf.to_ms());
        println!("   Converted to new Timeframe -> {}ms", new_tf.to_ms());
        println!("   Same result: {}", old_tf.to_ms() == new_tf.to_ms());
    }

    // 4. Error handling
    println!("\n4. Error handling:");
    let invalid_examples = vec!["invalid", "5x", "m5", "", "0m"];
    for example in invalid_examples {
        match Timeframe::parse(example) {
            Ok(tf) => println!("   '{}' -> {}ms (unexpected success!)", example, tf.to_ms()),
            Err(e) => println!("   '{}' -> Error: {}", example, e),
        }
    }

    // 5. Equivalence
    println!("\n5. Parse equals constructor:");
    let parsed = Timeframe::parse("5m").unwrap();
    let constructed = Timeframe::minutes(5);
    println!(
        "   Timeframe::parse(\"5m\") == Timeframe::minutes(5): {}",
        parsed == constructed
    );

    println!("\n✓ Demo complete!");
}
