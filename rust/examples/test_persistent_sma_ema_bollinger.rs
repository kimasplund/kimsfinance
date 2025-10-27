//! Test persistent kernel implementations for SMA, EMA, and Bollinger Bands
//!
//! This example validates that the new persistent kernel indicators:
//! 1. Compile successfully with NVRTC
//! 2. Implement the correct traits (PersistentIndicator, SingleOutputIndicator, MultiOutputIndicator)
//! 3. Have correct metadata (kernel_name, num_inputs, num_outputs)
//! 4. Can be instantiated with type-safe batch APIs
//!
//! **Build with**: `cargo build --example test_persistent_sma_ema_bollinger --features gpu`
//! **Run with**: `cargo run --example test_persistent_sma_ema_bollinger --features gpu`

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("⚠️  This example requires GPU feature enabled");
    println!("   Build with: cargo build --example test_persistent_sma_ema_bollinger --features gpu");
}

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_tests()
}

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::persistent::*;
#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::GpuDevice;

#[cfg(feature = "gpu")]
fn run_tests() -> Result<(), Box<dyn std::error::Error>> {

    println!("\n🧪 Testing Persistent Kernel Implementations: SMA, EMA, Bollinger Bands\n");

    // Initialize GPU device
    let device = match GpuDevice::new() {
        Ok(d) => {
            println!("✅ GPU device initialized");
            d
        }
        Err(e) => {
            println!("⚠️  GPU not available: {:?}", e);
            println!("   This is expected in CPU-only environments");
            return Ok(());
        }
    };

    println!("\n--- SMA Indicator ---");
    test_sma_indicator(&device)?;

    println!("\n--- EMA Indicator ---");
    test_ema_indicator(&device)?;

    println!("\n--- Bollinger Bands Indicator ---");
    test_bollinger_indicator(&device)?;

    println!("\n✅ All persistent kernel implementations validated successfully!");

    Ok(())
}

#[cfg(feature = "gpu")]
fn test_sma_indicator(device: &GpuDevice) -> Result<(), Box<dyn std::error::Error>> {
    use traits::PersistentIndicator;

    println!("Kernel name: {}", SmaIndicator::kernel_name());
    println!("Num inputs: {}", SmaIndicator::num_inputs());
    println!("Num outputs: {}", SmaIndicator::num_outputs());

    // Validate trait properties
    assert_eq!(SmaIndicator::kernel_name(), "persistent_sma_kernel");
    assert_eq!(SmaIndicator::num_inputs(), 1);
    assert_eq!(SmaIndicator::num_outputs(), 1);

    // Test kernel compilation
    print!("Compiling CUDA kernel... ");
    let kernel = SmaIndicator::compile_kernel(device)?;
    println!("✅ Success");

    // Test batch type alias
    let mut _batch = SmaBatch::new();
    _batch.add_task(vec![100.0, 101.0, 102.0, 103.0], 3);
    println!("✅ SmaBatch type alias works");

    println!("✅ SMA indicator validated");
    Ok(())
}

#[cfg(feature = "gpu")]
fn test_ema_indicator(device: &GpuDevice) -> Result<(), Box<dyn std::error::Error>> {
    use traits::PersistentIndicator;

    println!("Kernel name: {}", EmaIndicator::kernel_name());
    println!("Num inputs: {}", EmaIndicator::num_inputs());
    println!("Num outputs: {}", EmaIndicator::num_outputs());

    // Validate trait properties
    assert_eq!(EmaIndicator::kernel_name(), "persistent_ema_kernel");
    assert_eq!(EmaIndicator::num_inputs(), 1);
    assert_eq!(EmaIndicator::num_outputs(), 1);

    // Test kernel compilation
    print!("Compiling CUDA kernel... ");
    let kernel = EmaIndicator::compile_kernel(device)?;
    println!("✅ Success");

    // Test batch type alias
    let mut _batch = EmaBatch::new();
    _batch.add_task(vec![100.0, 101.0, 102.0, 103.0], 3);
    println!("✅ EmaBatch type alias works");

    println!("✅ EMA indicator validated");
    Ok(())
}

#[cfg(feature = "gpu")]
fn test_bollinger_indicator(device: &GpuDevice) -> Result<(), Box<dyn std::error::Error>> {
    use traits::{MultiOutputIndicator, PersistentIndicator};

    println!("Kernel name: {}", BollingerIndicator::kernel_name());
    println!("Num inputs: {}", BollingerIndicator::num_inputs());
    println!("Num outputs: {}", BollingerIndicator::num_outputs());

    // Validate trait properties
    assert_eq!(
        BollingerIndicator::kernel_name(),
        "persistent_bollinger_kernel"
    );
    assert_eq!(BollingerIndicator::num_inputs(), 1);
    assert_eq!(BollingerIndicator::num_outputs(), 3); // upper, middle, lower

    // Test kernel compilation
    print!("Compiling CUDA kernel... ");
    let kernel = BollingerIndicator::compile_kernel(device)?;
    println!("✅ Success");

    // Test batch type alias with struct params
    let mut _batch = BollingerBatch::new();
    _batch.add_task(vec![100.0; 50], BollingerParams::standard());
    _batch.add_task(
        vec![200.0; 50],
        BollingerParams {
            period: 10,
            std_dev: 1.5,
        },
    );
    println!("✅ BollingerBatch type alias works");
    println!("✅ BollingerParams struct params work");

    println!("✅ Bollinger Bands indicator validated");
    Ok(())
}
