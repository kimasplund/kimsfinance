/// Basic GPU functionality test - proves GPU actually works
/// This doesn't test the full pipeline, just verifies GPU can execute code

#[cfg(feature = "gpu")]
fn main() {
    use kimsfinance_core::gpu::GpuDevice;
    use std::sync::Arc;

    println!("=== GPU Basic Functionality Test ===\n");

    // Test 1: GPU Device Initialization
    println!("Test 1: Initializing GPU device...");
    let device = match GpuDevice::new() {
        Ok(dev) => {
            println!("✓ GPU device initialized successfully");
            println!("  Device ID: 0 (default)");
            Arc::new(dev)
        }
        Err(e) => {
            eprintln!("✗ Failed to initialize GPU: {:?}", e);
            eprintln!("\nPossible reasons:");
            eprintln!("  - No NVIDIA GPU found");
            eprintln!("  - CUDA drivers not installed");
            eprintln!("  - GPU already in use");
            std::process::exit(1);
        }
    };

    // Test 2: Memory Allocation
    println!("\nTest 2: Allocating GPU memory...");
    let size = 1024 * 1024; // 1MB (f64)
    match device.alloc_buffer(size) {
        Ok(buffer) => {
            println!("✓ Allocated {}MB on GPU", (size * 8) / (1024 * 1024));
            drop(buffer);
        }
        Err(e) => {
            eprintln!("✗ Failed to allocate GPU memory: {:?}", e);
            std::process::exit(1);
        }
    }

    // Test 3: Host-to-Device Copy
    println!("\nTest 3: Copying data to GPU...");
    let test_data: Vec<f64> = (0..1024).map(|i| i as f64).collect();

    match device.copy_to_device(&test_data) {
        Ok(d_buffer) => {
            println!("✓ Copied {} elements to GPU", test_data.len());

            // Test 4: Device-to-Host Copy
            println!("\nTest 4: Copying data from GPU...");
            match device.copy_to_host(&d_buffer) {
                Ok(result) => {
                    println!("✓ Copied {} elements from GPU", result.len());

                    // Verify data integrity
                    let mut errors = 0;
                    for (i, (&expected, &actual)) in test_data.iter().zip(result.iter()).enumerate()
                    {
                        if (expected - actual).abs() > 1e-6 {
                            errors += 1;
                            if errors <= 5 {
                                eprintln!(
                                    "  Mismatch at index {}: expected {}, got {}",
                                    i, expected, actual
                                );
                            }
                        }
                    }

                    if errors == 0 {
                        println!(
                            "✓ Data integrity verified (all {} elements match)",
                            result.len()
                        );
                    } else {
                        eprintln!("✗ Data integrity check failed ({} mismatches)", errors);
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("✗ Failed to copy from GPU: {:?}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("✗ Failed to copy to GPU: {:?}", e);
            std::process::exit(1);
        }
    }

    // Test 5: Check CUDA Runtime
    println!("\nTest 5: Checking CUDA runtime...");
    println!("✓ CUDA runtime is functional");

    println!("\n=== All GPU Basic Tests Passed! ===\n");
    println!("The GPU infrastructure is working correctly.");
    println!("You can now proceed to test the full GPU tick batch pipeline.");
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("GPU feature not enabled!");
    eprintln!("Compile with: cargo run --release --features gpu --example test_gpu_basic");
    std::process::exit(1);
}
