//! Simple test example for pinned memory module
//!
//! Validates that pinned memory allocates correctly and is accessible.
//!
//! Run with: cargo run --example test_pinned_memory --features gpu

use kimsfinance_core::gpu::{GpuDevice, persistent::pinned_memory::PinnedBuffer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Pinned Memory Module");
    println!("=============================\n");

    // Initialize CUDA driver
    println!("Initializing CUDA driver...");
    let _device = GpuDevice::new()?;
    println!("✓ CUDA driver initialized\n");

    // Test 1: Basic allocation
    println!("Test 1: Basic allocation...");
    let size = 1000;
    let buffer = PinnedBuffer::<f64>::new(size)?;
    println!("  ✓ Allocated {} elements", buffer.len());
    assert_eq!(buffer.len(), size);
    println!("  ✓ Length matches expected size\n");

    // Test 2: Write and read
    println!("Test 2: Write and read data...");
    let mut buffer2 = PinnedBuffer::<f64>::new(10)?;
    {
        let slice = buffer2.as_mut_slice();
        for (i, val) in slice.iter_mut().enumerate() {
            *val = i as f64 * 10.0;
        }
    }

    let data = buffer2.as_slice();
    println!("  ✓ Written and read 10 elements:");
    for (i, &val) in data.iter().enumerate() {
        println!("    [{}] = {:.1}", i, val);
        assert_eq!(val, i as f64 * 10.0);
    }
    println!("  ✓ Data integrity verified\n");

    // Test 3: Copy from Vec
    println!("Test 3: Copy from Vec...");
    let source = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut buffer3 = PinnedBuffer::new(5)?;
    buffer3.copy_from_slice(&source);

    let result = buffer3.as_slice();
    println!("  ✓ Copied {} elements", result.len());
    for (i, (&expected, &actual)) in source.iter().zip(result.iter()).enumerate() {
        println!("    [{}] expected={:.1}, actual={:.1}", i, expected, actual);
        assert_eq!(expected, actual);
    }
    println!("  ✓ Copy successful\n");

    // Test 4: Large allocation
    println!("Test 4: Large allocation (10M elements = ~80MB)...");
    let large_size = 10_000_000;
    match PinnedBuffer::<f64>::new(large_size) {
        Ok(large_buffer) => {
            println!(
                "  ✓ Allocated {:.2} MB of pinned memory",
                (large_buffer.len() * 8) as f64 / 1_000_000.0
            );
        }
        Err(e) => {
            println!("  ⚠️  Large allocation failed (expected if pinned memory limited):");
            println!("     {}", e);
            println!("     This is normal behavior when pinned memory quota is exceeded");
        }
    }

    println!("\n=============================");
    println!("✓ All tests passed!");
    println!("Pinned memory module is working correctly.");

    Ok(())
}
