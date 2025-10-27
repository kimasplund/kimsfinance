//! Test MACD calculation with multi-output support
//!
//! Demonstrates the persistent kernel system handling 3 outputs:
//! - MACD line
//! - Signal line
//! - Histogram
//!
//! # Expected Behavior
//!
//! - MACD uses 1 input array (close prices)
//! - 3 output arrays (macd_line, signal_line, histogram)
//! - Type-safe: compile-time verification of output count
//!
//! # Usage
//!
//! ```bash
//! cargo run --example test_macd_multi_output
//! ```

use kimsfinance_core::gpu::{
    execute_generic_batch, GenericBatch, GpuDevice, MacdIndicator, MacdParams, PersistentIndicator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing MACD Multi-Output Support");
    println!("════════════════════════════════════");

    // Initialize GPU
    let device = GpuDevice::new()?;
    println!("✅ GPU initialized");

    // Create sample close price data (50 candles for proper MACD calculation)
    let close_prices: Vec<f64> = (0..50)
        .map(|i| 100.0 + (i as f64 * 0.5) + (i as f64 / 10.0).sin() * 5.0)
        .collect();

    println!("\n📊 Input Data:");
    println!("   Close prices (first 5): {:?}", &close_prices[0..5]);
    println!("   Close prices (last 5):  {:?}", &close_prices[45..50]);
    println!("   Total candles: {}", close_prices.len());

    // Create generic batch with MACD tasks
    let mut batch = GenericBatch::<MacdIndicator>::new();

    // Add MACD task with standard parameters (12, 26, 9)
    println!("\n🔧 Adding MACD task with standard params (12, 26, 9)");
    batch.add_single_input_task(close_prices.clone(), MacdParams::standard());

    // Add MACD task with custom parameters
    println!("🔧 Adding MACD task with custom params (5, 10, 5)");
    batch.add_single_input_task(
        close_prices.clone(),
        MacdParams {
            fast_period: 5,
            slow_period: 10,
            signal_period: 5,
        },
    );

    println!("\n✅ Batch created: {} tasks", batch.len());
    println!("   Inputs per task: {}", MacdIndicator::num_inputs());
    println!("   Outputs per task: {}", MacdIndicator::num_outputs());

    // Execute batch
    println!("\n⚡ Executing persistent kernel batch...");
    match execute_generic_batch(&device, &batch) {
        Ok(results) => {
            println!("✅ Execution successful!");
            println!("\n📈 Results:");

            for (i, task_results) in results.iter().enumerate() {
                let params = if i == 0 { "(12,26,9)" } else { "(5,10,5)" };
                println!("\n   Task {}: MACD{}", i, params);
                println!("   Output arrays: {}", task_results.len());

                if task_results.len() == 3 {
                    let macd_line = &task_results[0];
                    let signal_line = &task_results[1];
                    let histogram = &task_results[2];

                    println!("   ├─ MACD Line:   {} values", macd_line.len());
                    println!("   ├─ Signal Line: {} values", signal_line.len());
                    println!("   └─ Histogram:   {} values", histogram.len());

                    // Show last 5 valid values
                    let start = macd_line.len().saturating_sub(5);
                    println!("\n   Last 5 MACD Line:   {:?}", &macd_line[start..]);
                    println!("   Last 5 Signal Line: {:?}", &signal_line[start..]);
                    println!("   Last 5 Histogram:   {:?}", &histogram[start..]);
                }
            }

            println!("\n🎉 Multi-output MACD test completed successfully!");
        }
        Err(e) => {
            println!("❌ Execution failed: {:?}", e);
            println!("\n⚠️  Note: This requires full implementation of generic_batch executor.");
            println!("   The current implementation is a framework demonstration.");
        }
    }

    Ok(())
}
