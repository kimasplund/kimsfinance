//! Test ATR calculation with multi-input support
//!
//! Demonstrates the persistent kernel system handling 3 inputs: high, low, close.
//!
//! # Expected Behavior
//!
//! - ATR uses 3 input arrays (high, low, close)
//! - Single output array (ATR values)
//! - Type-safe: compile-time verification of input count
//!
//! # Usage
//!
//! ```bash
//! cargo run --example test_atr_multi_input
//! ```

use kimsfinance_core::gpu::GpuDevice;
use kimsfinance_core::gpu::persistent::{
    AtrIndicator, GenericBatch, PersistentIndicator, execute_generic_batch,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing ATR Multi-Input Support");
    println!("═══════════════════════════════════");

    // Initialize GPU
    let device = GpuDevice::new()?;
    println!("✅ GPU initialized");

    // Create sample OHLCV data
    let high = vec![
        102.0, 103.5, 105.0, 104.5, 106.0, 107.5, 106.0, 108.0, 109.5, 108.0,
    ];
    let low = vec![
        99.0, 100.5, 101.0, 102.0, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
    ];
    let close = vec![
        100.0, 102.0, 103.0, 103.5, 105.0, 106.0, 104.5, 107.0, 108.0, 106.5,
    ];

    println!("\n📊 Input Data:");
    println!("   High:  {:?}", &high[0..5]);
    println!("   Low:   {:?}", &low[0..5]);
    println!("   Close: {:?}", &close[0..5]);
    println!("   ... ({} candles total)", high.len());

    // Create generic batch with ATR tasks
    let mut batch = GenericBatch::<AtrIndicator>::new();

    // Add task with 3 inputs (high, low, close)
    println!("\n🔧 Adding ATR task with period=3");
    batch.add_task(vec![high.clone(), low.clone(), close.clone()], 3);

    // Add another task with different period
    println!("🔧 Adding ATR task with period=5");
    batch.add_task(vec![high.clone(), low.clone(), close.clone()], 5);

    println!("\n✅ Batch created: {} tasks", batch.len());
    println!("   Inputs per task: {}", AtrIndicator::num_inputs());
    println!("   Outputs per task: {}", AtrIndicator::num_outputs());

    // Execute batch
    println!("\n⚡ Executing persistent kernel batch...");
    match execute_generic_batch(&device, &batch) {
        Ok(results) => {
            println!("✅ Execution successful!");
            println!("\n📈 Results:");

            for (i, task_results) in results.iter().enumerate() {
                println!("\n   Task {}: ATR({})", i, if i == 0 { 3 } else { 5 });
                println!("   Output arrays: {}", task_results.len());

                if !task_results.is_empty() {
                    let atr = &task_results[0];
                    println!("   First 5 ATR values: {:?}", &atr[0..5.min(atr.len())]);
                    println!(
                        "   Last 3 ATR values:  {:?}",
                        &atr[atr.len().saturating_sub(3)..]
                    );
                }
            }

            println!("\n🎉 Multi-input ATR test completed successfully!");
        }
        Err(e) => {
            println!("❌ Execution failed: {:?}", e);
            println!("\n⚠️  Note: This requires full implementation of generic_batch executor.");
            println!("   The current implementation is a framework demonstration.");
        }
    }

    Ok(())
}
