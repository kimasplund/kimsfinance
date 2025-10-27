//! Test example for persistent kernel trait system
//!
//! Demonstrates compilation and trait usage for all indicator types.

use kimsfinance_core::gpu::persistent::{
    AtrIndicator, MacdIndicator, MacdParams, PersistentIndicator, RocIndicator, RsiIndicator,
};
use kimsfinance_core::gpu::{GpuDevice, GpuError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU
    let device = GpuDevice::new().map_err(|e| {
        eprintln!("GPU not available: {:?}", e);
        eprintln!("This example requires a CUDA-capable GPU");
        e
    })?;

    println!("=== Persistent Kernel Trait System Test ===\n");

    // Test ROC indicator
    println!("1. ROC Indicator:");
    println!("   Kernel name: {}", RocIndicator::kernel_name());
    println!("   Output count: {}", RocIndicator::num_outputs());
    let roc_func = RocIndicator::compile_kernel(&device)?;
    println!("   ✓ Compilation successful\n");

    // Test RSI indicator
    println!("2. RSI Indicator:");
    println!("   Kernel name: {}", RsiIndicator::kernel_name());
    println!("   Output count: {}", RsiIndicator::num_outputs());
    let rsi_func = RsiIndicator::compile_kernel(&device)?;
    println!("   ✓ Compilation successful\n");

    // Test ATR indicator
    println!("3. ATR Indicator:");
    println!("   Kernel name: {}", AtrIndicator::kernel_name());
    println!("   Output count: {}", AtrIndicator::num_outputs());
    let atr_func = AtrIndicator::compile_kernel(&device)?;
    println!("   ✓ Compilation successful\n");

    // Test MACD indicator
    println!("4. MACD Indicator:");
    println!("   Kernel name: {}", MacdIndicator::kernel_name());
    println!("   Output count: {}", MacdIndicator::num_outputs());
    let macd_func = MacdIndicator::compile_kernel(&device)?;
    println!("   ✓ Compilation successful\n");

    println!("=== All indicators compiled successfully! ===");
    println!("\nTrait system features:");
    println!("  - Generic parameter types (i32 for RSI/ROC/ATR, MacdParams for MACD)");
    println!("  - Single-output indicators (ROC, RSI, ATR)");
    println!("  - Multi-output indicators (MACD: 3 outputs)");
    println!("  - Unified compilation interface");
    println!("  - Zero-cost abstraction (static dispatch)");

    Ok(())
}
