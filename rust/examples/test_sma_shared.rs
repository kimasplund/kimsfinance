use kimsfinance_core::gpu::{GpuDevice, sma_gpu, sma_gpu_shared};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;
    
    // Test data
    let close = Array1::from_vec(vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0]);
    let period = 3;
    
    println!("Testing SMA shared memory implementation...");
    println!("Data: {:?}", close.as_slice().unwrap());
    println!("Period: {}", period);
    
    // Global memory version
    let sma_global = sma_gpu(&device, &close, period, None)?;
    println!("\nGlobal SMA: {:?}", sma_global.as_slice().unwrap());
    
    // Shared memory version
    let sma_shared = sma_gpu_shared(&device, &close, period, None)?;
    println!("Shared SMA: {:?}", sma_shared.as_slice().unwrap());
    
    // Verify match
    let mut match_count = 0;
    for i in 0..close.len() {
        if sma_global[i].is_nan() && sma_shared[i].is_nan() {
            match_count += 1;
        } else if (sma_global[i] - sma_shared[i]).abs() < 1e-10 {
            match_count += 1;
        }
    }
    
    println!("\nMatching values: {}/{}", match_count, close.len());
    
    if match_count == close.len() {
        println!("✓ SUCCESS: Shared memory implementation is correct!");
    } else {
        println!("✗ FAILURE: Shared memory implementation has errors!");
        std::process::exit(1);
    }
    
    Ok(())
}
