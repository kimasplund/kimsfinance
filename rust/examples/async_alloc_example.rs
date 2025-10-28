//! Example: GPU Async Memory Allocation
//!
//! Demonstrates the AsyncAllocator API and CUDA version detection.
//!
//! Run with: cargo run --example async_alloc_example --features gpu

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{AsyncAllocator, GpuDevice, PoolStats};

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU Async Memory Allocator Demo ===\n");

    // Initialize GPU device
    println!("1. Initializing GPU device...");
    let device = GpuDevice::new()?;
    println!("   ✓ GPU initialized\n");

    // Check if async allocation is supported
    println!("2. Checking async allocation support...");
    let supports_async = device.supports_async_alloc();
    println!("   Async allocation supported: {}\n", supports_async);

    if !supports_async {
        println!("   Note: CUDA < 11.2 or pool creation failed.");
        println!("   Using standard allocation (no performance benefit).\n");
    }

    // Allocate buffers
    println!("3. Allocating GPU buffers...");
    let sizes = [1_000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        let buffer = device.alloc_async(size)?;
        println!(
            "   ✓ Allocated {} f64 elements ({:.2} MB)",
            size,
            (size * 8) as f64 / 1_048_576.0
        );
        drop(buffer); // Free immediately
    }
    println!();

    // Get statistics
    println!("4. Allocation statistics:");
    if let Some(stats) = device.async_alloc_stats() {
        print_stats(&stats);
    } else {
        println!("   No statistics available (async allocator not initialized)");
    }
    println!();

    // Trim pool (release unused memory)
    println!("5. Trimming memory pool...");
    device.trim_async_pool();
    println!("   ✓ Pool trimmed\n");

    println!("=== Demo Complete ===");
    println!("\nStatus: Infrastructure working, waiting for cudarc API support");
    println!("Expected speedup when enabled: 1.2-1.5x for allocation-heavy code");

    Ok(())
}

#[cfg(feature = "gpu")]
fn print_stats(stats: &PoolStats) {
    println!("   Allocations:       {}", stats.allocations);
    println!("   Deallocations:     {}", stats.deallocations);
    println!(
        "   Peak memory used:  {:.2} MB",
        stats.peak_bytes_used as f64 / 1_048_576.0
    );
    println!(
        "   Total allocated:   {:.2} MB",
        stats.total_bytes_allocated as f64 / 1_048_576.0
    );
    println!(
        "   Current in use:    {:.2} MB",
        stats.current_bytes_used as f64 / 1_048_576.0
    );
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("This example requires the 'gpu' feature.");
    println!("Run with: cargo run --example async_alloc_example --features gpu");
}
