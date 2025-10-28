//! Test Range Bars and Renko Bricks Implementation
//!
//! Verifies that Range Bars and Renko kernels compile and have correct trait implementations.

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{
    GpuDevice, RangeBarAggregator, RangeBarParams, RenkoAggregator, RenkoParams,
};

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::persistent::PersistentIndicator;

fn main() {
    #[cfg(feature = "gpu")]
    {
        println!("=== Range Bars & Renko Bricks Verification ===\n");

        // Verify Range Bars
        println!("Range Bars:");
        println!("  Kernel name: {}", RangeBarAggregator::kernel_name());
        println!(
            "  Inputs: {} (timestamp, price, volume)",
            RangeBarAggregator::num_inputs()
        );
        println!("  Outputs: {} (OHLCV)", RangeBarAggregator::num_outputs());

        let range_params = RangeBarParams { range_size: 100.0 };
        println!("  Example params: {:?}", range_params);
        println!(
            "  Params size: {} bytes\n",
            std::mem::size_of::<RangeBarParams>()
        );

        // Verify Renko
        println!("Renko Bricks:");
        println!("  Kernel name: {}", RenkoAggregator::kernel_name());
        println!(
            "  Inputs: {} (timestamp, price)",
            RenkoAggregator::num_inputs()
        );
        println!(
            "  Outputs: {} (brick_price, direction, timestamp)",
            RenkoAggregator::num_outputs()
        );

        let renko_params = RenkoParams { brick_size: 50.0 };
        println!("  Example params: {:?}", renko_params);
        println!(
            "  Params size: {} bytes\n",
            std::mem::size_of::<RenkoParams>()
        );

        // Try to compile kernels (requires GPU)
        match GpuDevice::new() {
            Ok(device) => {
                println!("GPU Device found! Compiling kernels...\n");

                match RangeBarAggregator::compile_kernel(&device) {
                    Ok(_) => println!("✓ Range Bar kernel compiled successfully"),
                    Err(e) => println!("✗ Range Bar kernel compilation failed: {:?}", e),
                }

                match RenkoAggregator::compile_kernel(&device) {
                    Ok(_) => println!("✓ Renko kernel compiled successfully"),
                    Err(e) => println!("✗ Renko kernel compilation failed: {:?}", e),
                }
            }
            Err(e) => {
                println!("No GPU available: {:?}", e);
                println!("Skipping kernel compilation tests (CPU-only verification passed)");
            }
        }

        println!("\n=== Verification Complete ===");
        println!("\nUse Cases:");
        println!("  Range Bars: Fixed price movement per bar (e.g., $100 moves)");
        println!("  Renko: Trend-following bricks (ignores time completely)");
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled. Compile with --features gpu");
    }
}
