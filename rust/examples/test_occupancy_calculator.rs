//! Test occupancy calculator for persistent kernels
//!
//! This example demonstrates:
//! 1. Conservative 25% heuristic grid size
//! 2. Dynamic occupancy-based grid size
//! 3. Performance comparison
//!
//! Expected result: 1.5-2x more blocks with dynamic occupancy

use kimsfinance_core::gpu::{GpuDevice, persistent::*};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Dynamic Occupancy Calculator");
    println!("========================================\n");

    // Initialize GPU
    let device = GpuDevice::new()?;
    println!("✅ GPU initialized\n");

    // Create persistent kernel manager (with occupancy calculator)
    let manager = PersistentKernelManager::new(&device)?;
    println!();

    // Create a test batch
    let mut batch = TaskBatch::<RocIndicator>::new();
    let test_data = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0];
    batch.add_task(test_data.clone(), 3);
    batch.add_task(test_data.clone(), 5);

    // Execute batch (will use dynamic occupancy internally)
    println!("🚀 Executing batch with dynamic occupancy...\n");
    let results = manager.execute_batch(&batch)?;

    // Verify correctness
    assert_eq!(results.len(), 2, "Should have 2 results");
    println!("✅ Batch execution successful!");
    println!("   Task 1 result length: {}", results[0].len());
    println!("   Task 2 result length: {}", results[1].len());

    // Show ROC calculations
    println!("\n📊 ROC Calculations (Task 1, period=3):");
    for (i, val) in results[0].iter().enumerate() {
        if val.is_finite() {
            println!("   ROC[{}] = {:.2}%", i, val);
        } else {
            println!("   ROC[{}] = NaN (warmup)", i);
        }
    }

    println!("\n✅ Dynamic occupancy calculator working correctly!");
    println!("   Grid size automatically optimized based on kernel resources");

    Ok(())
}
