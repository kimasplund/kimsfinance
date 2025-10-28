// Test persistent kernel indicator trait implementations
//
// This example verifies that Donchian, Keltner, and Aroon indicators
// correctly implement the PersistentIndicator trait.

use kimsfinance_core::gpu::persistent::{
    AroonIndicator, DonchianIndicator, KeltnerIndicator, KeltnerParams, PersistentIndicator,
};

fn main() {
    println!("Testing Persistent Kernel Indicator Traits\n");
    println!("===========================================\n");

    // Test Donchian
    println!("Donchian Channels:");
    println!("  Kernel name: {}", DonchianIndicator::kernel_name());
    println!("  Num inputs: {}", DonchianIndicator::num_inputs());
    println!("  Num outputs: {}", DonchianIndicator::num_outputs());
    assert_eq!(
        DonchianIndicator::kernel_name(),
        "persistent_donchian_kernel"
    );
    assert_eq!(DonchianIndicator::num_inputs(), 2); // high, low
    assert_eq!(DonchianIndicator::num_outputs(), 3); // upper, middle, lower
    println!("  ✓ All checks passed\n");

    // Test Keltner
    println!("Keltner Channels:");
    println!("  Kernel name: {}", KeltnerIndicator::kernel_name());
    println!("  Num inputs: {}", KeltnerIndicator::num_inputs());
    println!("  Num outputs: {}", KeltnerIndicator::num_outputs());
    assert_eq!(KeltnerIndicator::kernel_name(), "persistent_keltner_kernel");
    assert_eq!(KeltnerIndicator::num_inputs(), 1); // ema+atr concatenated
    assert_eq!(KeltnerIndicator::num_outputs(), 3); // upper, middle, lower

    let params = KeltnerParams::standard();
    println!("  Standard params: {:?}", params);
    assert_eq!(params.ema_period, 20);
    assert_eq!(params.atr_period, 10);
    assert_eq!(params.multiplier, 2.0);
    println!("  ✓ All checks passed\n");

    // Test Aroon
    println!("Aroon Indicator:");
    println!("  Kernel name: {}", AroonIndicator::kernel_name());
    println!("  Num inputs: {}", AroonIndicator::num_inputs());
    println!("  Num outputs: {}", AroonIndicator::num_outputs());
    assert_eq!(AroonIndicator::kernel_name(), "persistent_aroon_kernel");
    assert_eq!(AroonIndicator::num_inputs(), 2); // high, low
    assert_eq!(AroonIndicator::num_outputs(), 3); // up, down, oscillator
    println!("  ✓ All checks passed\n");

    println!("===========================================");
    println!("All indicator trait implementations verified!");
    println!("\nNext steps:");
    println!("  1. Implement multi-input batch allocation");
    println!("  2. Test with GenericBatch framework");
    println!("  3. Benchmark persistent vs traditional kernels");
}
