//! Minimal reproduction case for NVIDIA: Cooperative kernel with pinned memory returns zeros
//!
//! This example demonstrates the issue where a cooperative persistent kernel
//! works correctly with non-pinned memory but returns all zeros with pinned memory.
//!
//! Expected behavior: Output should be input * 2.0
//! Actual behavior with pinned memory: Output is all zeros
//! Workaround: Use non-pinned memory (pageable) - 20-30% slower but correct
//!
//! See: docs/NVIDIA_BUG_REPORT_PINNED_MEMORY.md

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::GpuDevice;

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaSlice, DevicePtr, DeviceSlice, LaunchConfig};

#[cfg(feature = "gpu")]
use cudarc::nvrtc::compile_ptx;

use std::error::Error;

#[cfg(feature = "gpu")]
const KERNEL_SRC: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void persistent_test_kernel(
    const double** __restrict__ input_batch,   // Array of input pointers
    double** __restrict__ output_batch,         // Array of output pointers
    const int* __restrict__ sizes,              // Array of sizes (n for each task)
    int num_tasks
) {
    // Cooperative grid setup
    cg::grid_group grid = cg::this_grid();
    int grid_size = gridDim.x * blockDim.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Process tasks sequentially
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* input = input_batch[task_id];
        double* output = output_batch[task_id];
        int n = sizes[task_id];

        // Single thread handles this task
        if (global_tid == task_id % grid_size) {
            // Simple computation: copy input to output with 2x multiplier
            for (int i = 0; i < n; i++) {
                output[i] = input[i] * 2.0;
            }
        }

        // Synchronize before next task
        grid.sync();
    }
}
"#;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Minimal Reproduction: Pinned Memory + Cooperative Kernel ===\n");

    // Initialize GPU
    let device = GpuDevice::new()?;
    println!("✅ GPU Device initialized");
    println!("   Compute capability: {}", device.compute_capability());
    println!();

    // Compile kernel
    println!("📦 Compiling kernel...");
    let ptx = compile_ptx(KERNEL_SRC)?;
    device
        .device
        .load_ptx(ptx, "test_module", &["persistent_test_kernel"])?;
    println!("✅ Kernel compiled\n");

    // Test data: 100 elements
    let n = 100;
    let input_data: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    println!("📊 Test data: {} elements (1.0 to 100.0)", n);
    println!("   First 5: {:?}", &input_data[0..5]);
    println!();

    // ========================================================================
    // TEST 1: Using Pinned Memory (EXPECTED TO FAIL)
    // ========================================================================
    println!("🧪 TEST 1: Using Pinned Memory");
    println!("─────────────────────────────────");

    match test_with_pinned_memory(&device, &input_data, n) {
        Ok(result) => {
            println!("✅ Test completed");
            println!("   Result: {:?}", &result[0..5.min(result.len())]);

            // Verify correctness
            let expected: Vec<f64> = input_data.iter().map(|x| x * 2.0).collect();
            let all_correct = result.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-6);

            if all_correct {
                println!("✅ Result is CORRECT (all values match expected)");
            } else {
                println!("❌ Result is INCORRECT");
                println!("   Expected: {:?}", &expected[0..5]);
                println!("   Got:      {:?}", &result[0..5]);

                // Check if all zeros
                let all_zeros = result.iter().all(|x| *x == 0.0);
                if all_zeros {
                    println!("   ⚠️  ALL ZEROS - This is the bug!");
                }
            }
        }
        Err(e) => {
            println!("❌ Test failed: {}", e);
        }
    }
    println!();

    // ========================================================================
    // TEST 2: Using Non-Pinned Memory (EXPECTED TO WORK)
    // ========================================================================
    println!("🧪 TEST 2: Using Non-Pinned Memory (Workaround)");
    println!("─────────────────────────────────────────────────");

    match test_with_pageable_memory(&device, &input_data, n) {
        Ok(result) => {
            println!("✅ Test completed");
            println!("   Result: {:?}", &result[0..5.min(result.len())]);

            // Verify correctness
            let expected: Vec<f64> = input_data.iter().map(|x| x * 2.0).collect();
            let all_correct = result.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-6);

            if all_correct {
                println!("✅ Result is CORRECT (all values match expected)");
            } else {
                println!("❌ Result is INCORRECT");
                println!("   Expected: {:?}", &expected[0..5]);
                println!("   Got:      {:?}", &result[0..5]);
            }
        }
        Err(e) => {
            println!("❌ Test failed: {}", e);
        }
    }
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("=== Summary ===");
    println!("TEST 1 (pinned):     Expected FAIL (returns zeros)");
    println!("TEST 2 (pageable):   Expected PASS (correct results)");
    println!();
    println!("This demonstrates the pinned memory + cooperative kernel bug.");
    println!("See docs/NVIDIA_BUG_REPORT_PINNED_MEMORY.md for full details.");

    Ok(())
}

#[cfg(feature = "gpu")]
fn test_with_pinned_memory(
    device: &GpuDevice,
    input_data: &[f64],
    n: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    use cudarc::driver::sys::CUdeviceptr;

    // Allocate pinned host memory
    println!("   Allocating pinned memory...");
    let mut h_input = device.device.alloc_pinned::<f64>(n)?;
    let mut h_output = device.device.alloc_pinned::<f64>(n)?;

    // Copy input data to pinned buffer
    h_input.copy_from_slice(input_data);
    println!("   ✅ Pinned buffers allocated ({} elements)", n);

    // Allocate device memory
    let mut d_input = device.device.alloc::<f64>(n)?;
    let mut d_output = device.device.alloc::<f64>(n)?;
    println!("   ✅ Device buffers allocated");

    // Transfer using pinned memory (DMA)
    device.device.htod_copy_into(h_input, &mut d_input)?;
    println!("   ✅ Data uploaded (pinned → device)");

    // Launch kernel
    launch_kernel(device, &d_input, &mut d_output, n)?;
    println!("   ✅ Kernel executed");

    // Transfer results back using pinned memory
    device
        .device
        .dtoh_copy_into(&d_output, &mut h_output)?;
    println!("   ✅ Results downloaded (device → pinned)");

    Ok(h_output.to_vec())
}

#[cfg(feature = "gpu")]
fn test_with_pageable_memory(
    device: &GpuDevice,
    input_data: &[f64],
    n: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    println!("   Allocating pageable memory...");

    // Allocate device memory
    let d_input = device.device.htod_copy(input_data)?;
    let mut d_output = device.device.alloc::<f64>(n)?;
    println!("   ✅ Device buffers allocated");
    println!("   ✅ Data uploaded (pageable → device)");

    // Launch kernel
    launch_kernel(device, &d_input, &mut d_output, n)?;
    println!("   ✅ Kernel executed");

    // Transfer results back using pageable memory
    let output = device.device.dtoh_sync_copy(&d_output)?;
    println!("   ✅ Results downloaded (device → pageable)");

    Ok(output)
}

#[cfg(feature = "gpu")]
fn launch_kernel(
    device: &GpuDevice,
    d_input: &CudaSlice<f64>,
    d_output: &mut CudaSlice<f64>,
    n: usize,
) -> Result<(), Box<dyn Error>> {
    use cudarc::driver::sys::CUdeviceptr;

    // Create pointer arrays
    let (input_ptr, _) = d_input.device_ptr(&device.stream);
    let (output_ptr, _) = d_output.device_ptr(&device.stream);

    let input_ptrs = vec![input_ptr as u64];
    let output_ptrs = vec![output_ptr as u64];
    let sizes = vec![n as i32];

    let d_input_ptrs = device.device.htod_copy(input_ptrs)?;
    let d_output_ptrs = device.device.htod_copy(output_ptrs)?;
    let d_sizes = device.device.htod_copy(sizes)?;

    // Launch cooperative kernel
    let num_blocks = 128;
    let num_threads = 256;
    let func = device
        .device
        .get_func("test_module", "persistent_test_kernel")?;

    unsafe {
        let params = (
            d_input_ptrs.device_ptr(&device.stream).0 as *mut CUdeviceptr,
            d_output_ptrs.device_ptr(&device.stream).0 as *mut CUdeviceptr,
            d_sizes.device_ptr(&device.stream).0 as *mut i32,
            1i32, // num_tasks
        );

        func.launch_on_stream(
            &device.stream,
            LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (num_threads, 1, 1),
                shared_mem_bytes: 0,
            },
            params,
        )?;
    }

    // Synchronize
    device.stream.synchronize()?;

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("❌ This example requires the 'gpu' feature.");
    eprintln!("   Run with: cargo run --example nvidia_pinned_memory_repro --features gpu");
    std::process::exit(1);
}
