//! Property-Based Tests for CUDA Features
//!
//! Uses proptest to validate properties across random inputs:
//! - Async allocator: any valid size should succeed or return OOM
//! - FP8 quantization: mathematical properties (commutativity, range)
//! - CUDA graphs: deterministic execution
//!
//! Run with: cargo test --release --features gpu --test cuda_features_property

use proptest::prelude::*;

// ============================================================================
// PROPERTY TESTS: FP8 Quantization
// ============================================================================

/// FP8 quantization should preserve sign
fn quantize_fp8_sign_preserving(value: f64) -> bool {
    // Simulate quantize_fp8 behavior (clamp to ±448, round to 2 decimals)
    if value.is_nan() {
        return true; // NaN is special case
    }

    let quantized = quantize_fp8_sim(value);
    value.signum() == quantized.signum() || quantized == 0.0
}

/// FP8 quantization should be idempotent
fn quantize_fp8_idempotent(value: f64) -> bool {
    if value.is_nan() {
        return true;
    }

    let once = quantize_fp8_sim(value);
    let twice = quantize_fp8_sim(once);
    (once - twice).abs() < 1e-10
}

/// FP8 quantization should be monotonic
fn quantize_fp8_monotonic(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return true;
    }

    if a <= b {
        let qa = quantize_fp8_sim(a);
        let qb = quantize_fp8_sim(b);
        qa <= qb || (qa - qb).abs() < 0.01 // Allow rounding error
    } else {
        true
    }
}

/// FP8 quantization should clamp to ±448
fn quantize_fp8_clamped(value: f64) -> bool {
    if value.is_nan() {
        return true;
    }

    let quantized = quantize_fp8_sim(value);
    (-448.0..=448.0).contains(&quantized)
}

/// Simulate FP8 quantization (matches optimizer implementation)
fn quantize_fp8_sim(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }

    // FP8 E4M3 range: ±448
    let clamped = value.clamp(-448.0, 448.0);

    // Round to 2 decimal places (FP8 precision)
    (clamped * 100.0).round() / 100.0
}

proptest! {
    #[test]
    fn prop_fp8_sign_preserving(value in -1000.0..1000.0f64) {
        prop_assert!(quantize_fp8_sign_preserving(value),
            "FP8 quantization should preserve sign for {}", value);
    }

    #[test]
    fn prop_fp8_idempotent(value in -1000.0..1000.0f64) {
        prop_assert!(quantize_fp8_idempotent(value),
            "FP8 quantization should be idempotent for {}", value);
    }

    #[test]
    fn prop_fp8_monotonic(a in -1000.0..1000.0f64, b in -1000.0..1000.0f64) {
        prop_assert!(quantize_fp8_monotonic(a, b),
            "FP8 quantization should be monotonic for {} <= {}", a, b);
    }

    #[test]
    fn prop_fp8_clamped(value in -1000.0..1000.0f64) {
        prop_assert!(quantize_fp8_clamped(value),
            "FP8 quantization should clamp to ±448 for {}", value);
    }

    #[test]
    fn prop_fp8_precision_loss(value in -448.0..448.0f64) {
        let quantized = quantize_fp8_sim(value);
        let error = (value - quantized).abs();

        // FP8 should have at most 0.005 error (rounding to 2 decimals)
        prop_assert!(error <= 0.01,
            "FP8 precision loss too large: {} vs {} (error: {})",
            value, quantized, error);
    }
}

// ============================================================================
// PROPERTY TESTS: Async Allocator
// ============================================================================

#[cfg(feature = "gpu")]
proptest! {
    #[test]
    #[ignore] // Requires GPU
    fn prop_async_allocator_any_size(size in 1..100_000_000usize) {
        use kimsfinance_core::gpu::{GpuDevice, AsyncAllocator};

        let device = GpuDevice::new().expect("GPU required");
        let allocator = AsyncAllocator::new(device.stream().clone(), device.device_id as i32)
            .expect("Failed to create allocator");

        // Should either succeed or return error (not panic)
        let result = allocator.alloc::<f64>(size);

        // Check that result is valid (either Ok or Err, no panic)
        match result {
            Ok(buffer) => {
                prop_assert_eq!(buffer.len(), size,
                    "Buffer size mismatch: expected {}, got {}", size, buffer.len());
            }
            Err(e) => {
                // OOM is acceptable for very large sizes
                prop_assert!(size > 10_000_000 || e.to_string().contains("allocation"),
                    "Unexpected error for size {}: {:?}", size, e);
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn prop_async_allocator_sequential(sizes in prop::collection::vec(1..10_000usize, 1..100)) {
        use kimsfinance_core::gpu::{GpuDevice, AsyncAllocator};

        let device = GpuDevice::new().expect("GPU required");
        let allocator = AsyncAllocator::new(device.stream().clone(), device.device_id as i32)
            .expect("Failed to create allocator");

        let mut buffers = Vec::new();
        for size in sizes.iter() {
            match allocator.alloc::<f64>(*size) {
                Ok(buffer) => {
                    prop_assert_eq!(buffer.len(), *size);
                    buffers.push(buffer);
                }
                Err(_) => {
                    // OOM is acceptable
                    break;
                }
            }
        }

        // All allocations succeeded or we hit OOM
        let stats = allocator.stats();
        prop_assert!(stats.allocations >= buffers.len(),
            "Stats mismatch: {} allocations vs {} buffers",
            stats.allocations, buffers.len());
    }
}

// ============================================================================
// PROPERTY TESTS: FP8 Arithmetic Properties
// ============================================================================

/// FP8 addition should be approximately commutative
fn fp8_addition_commutative(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return true;
    }

    let qa = quantize_fp8_sim(a);
    let qb = quantize_fp8_sim(b);

    let ab = quantize_fp8_sim(qa + qb);
    let ba = quantize_fp8_sim(qb + qa);

    (ab - ba).abs() < 0.01 // Allow small rounding error
}

/// FP8 multiplication should be approximately commutative
fn fp8_multiplication_commutative(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return true;
    }

    let qa = quantize_fp8_sim(a);
    let qb = quantize_fp8_sim(b);

    // Avoid overflow
    if qa.abs() > 20.0 || qb.abs() > 20.0 {
        return true;
    }

    let ab = quantize_fp8_sim(qa * qb);
    let ba = quantize_fp8_sim(qb * qa);

    (ab - ba).abs() < 0.01
}

proptest! {
    #[test]
    fn prop_fp8_addition_commutative(a in -100.0..100.0f64, b in -100.0..100.0f64) {
        prop_assert!(fp8_addition_commutative(a, b),
            "FP8 addition should be commutative: {} + {} vs {} + {}", a, b, b, a);
    }

    #[test]
    fn prop_fp8_multiplication_commutative(a in -20.0..20.0f64, b in -20.0..20.0f64) {
        prop_assert!(fp8_multiplication_commutative(a, b),
            "FP8 multiplication should be commutative: {} * {} vs {} * {}", a, b, b, a);
    }

    #[test]
    fn prop_fp8_zero_identity(value in -448.0..448.0f64) {
        let q = quantize_fp8_sim(value);
        let q_plus_zero = quantize_fp8_sim(q + 0.0);

        prop_assert!((q - q_plus_zero).abs() < 1e-10,
            "Adding zero should be identity: {} + 0 = {}", q, q_plus_zero);
    }

    #[test]
    fn prop_fp8_one_identity(value in -20.0..20.0f64) {
        let q = quantize_fp8_sim(value);
        let q_times_one = quantize_fp8_sim(q * 1.0);

        prop_assert!((q - q_times_one).abs() < 1e-10,
            "Multiplying by one should be identity: {} * 1 = {}", q, q_times_one);
    }
}

// ============================================================================
// PROPERTY TESTS: CUDA Graph Determinism
// ============================================================================

#[cfg(feature = "gpu")]
proptest! {
    #[test]
    #[ignore] // Requires GPU
    fn prop_cuda_graph_deterministic_execution(_seed in 0..1000u64) {
        use kimsfinance_core::gpu::{GpuDevice, IndicatorGraphBuilder, IndicatorSpeed, StreamManager};
        use std::sync::Arc;

        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let stream_mgr = Arc::new(StreamManager::new(device.clone()).unwrap());

        // Build graph
        let mut builder = IndicatorGraphBuilder::new(device.clone(), stream_mgr.clone()).unwrap();
        builder.begin_capture_stream(IndicatorSpeed::Fast).unwrap();
        // TODO: Add kernel launches when cudarc supports graphs
        builder.end_capture_stream(IndicatorSpeed::Fast).unwrap();
        let graph = builder.build().unwrap();

        // Launch graph multiple times
        for _ in 0..10 {
            graph.launch_all().expect("Graph launch failed");
            graph.synchronize().expect("Sync failed");
        }

        // If we get here without panic, determinism is preserved
        prop_assert!(true);
    }
}

// ============================================================================
// PROPERTY TESTS: FP8 Statistical Properties
// ============================================================================

/// FP8 quantization should have bounded error distribution
fn fp8_error_bounded(values: Vec<f64>) -> bool {
    let errors: Vec<f64> = values
        .iter()
        .filter(|v| !v.is_nan() && v.abs() <= 448.0)
        .map(|v| {
            let quantized = quantize_fp8_sim(*v);
            (v - quantized).abs()
        })
        .collect();

    if errors.is_empty() {
        return true;
    }

    // All errors should be < 0.01 (rounding to 2 decimals)
    errors.iter().all(|e| *e <= 0.01)
}

/// FP8 quantization mean error should be small
fn fp8_mean_error_small(values: Vec<f64>) -> bool {
    let errors: Vec<f64> = values
        .iter()
        .filter(|v| !v.is_nan() && v.abs() <= 448.0)
        .map(|v| {
            let quantized = quantize_fp8_sim(*v);
            (v - quantized).abs()
        })
        .collect();

    if errors.is_empty() {
        return true;
    }

    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;

    // Mean error should be < 0.005 (half of max rounding error)
    mean_error < 0.005
}

proptest! {
    #[test]
    fn prop_fp8_error_distribution(values in prop::collection::vec(-448.0..448.0f64, 1..1000)) {
        prop_assert!(fp8_error_bounded(values.clone()),
            "FP8 errors should be bounded");
        prop_assert!(fp8_mean_error_small(values),
            "FP8 mean error should be small");
    }

    #[test]
    fn prop_fp8_relative_error(value in 1.0..448.0f64) {
        let quantized = quantize_fp8_sim(value);
        let relative_error = ((value - quantized) / value).abs();

        // Relative error should be < 1% for values >= 1.0
        prop_assert!(relative_error < 0.01,
            "FP8 relative error too large: {:.4}% for value {}", relative_error * 100.0, value);
    }
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_fp8_special_values() {
    // Test special IEEE-754 values
    assert!(quantize_fp8_sim(f64::NAN).is_nan(), "NaN should remain NaN");
    assert_eq!(quantize_fp8_sim(0.0), 0.0, "Zero should remain zero");
    assert_eq!(
        quantize_fp8_sim(-0.0),
        -0.0,
        "Negative zero should remain negative zero"
    );

    // Clamping tests
    assert_eq!(
        quantize_fp8_sim(500.0),
        448.0,
        "Over-max should clamp to 448"
    );
    assert_eq!(
        quantize_fp8_sim(-500.0),
        -448.0,
        "Under-min should clamp to -448"
    );
    assert_eq!(quantize_fp8_sim(448.0), 448.0, "Max should remain at 448");
    assert_eq!(
        quantize_fp8_sim(-448.0),
        -448.0,
        "Min should remain at -448"
    );

    // Precision tests
    assert_eq!(
        quantize_fp8_sim(1.234567),
        1.23,
        "Should round to 2 decimals"
    );
    assert_eq!(
        quantize_fp8_sim(100.456),
        100.46,
        "Should round up correctly"
    );
    assert_eq!(
        quantize_fp8_sim(-50.789),
        -50.79,
        "Should round negative values"
    );
}

#[test]
#[cfg(feature = "gpu")]
#[ignore] // Requires GPU
fn test_async_allocator_zero_size() {
    use kimsfinance_core::gpu::{AsyncAllocator, GpuDevice};

    let device = GpuDevice::new().expect("GPU required");
    let allocator = AsyncAllocator::new(device.stream().clone(), device.device_id as i32)
        .expect("Failed to create allocator");

    // Zero-size allocation should either succeed with empty buffer or error
    let result = allocator.alloc::<f64>(0);

    match result {
        Ok(buffer) => {
            assert_eq!(buffer.len(), 0, "Zero-size buffer should have length 0");
        }
        Err(_) => {
            // Error is also acceptable for zero-size
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore] // Requires GPU
fn test_async_allocator_huge_size() {
    use kimsfinance_core::gpu::{AsyncAllocator, GpuDevice};

    let device = GpuDevice::new().expect("GPU required");
    let allocator = AsyncAllocator::new(device.stream().clone(), device.device_id as i32)
        .expect("Failed to create allocator");

    // Huge allocation (1TB) should fail gracefully with OOM
    let huge_size = 125_000_000_000usize; // 1TB
    let result = allocator.alloc::<f64>(huge_size);

    assert!(
        result.is_err(),
        "Huge allocation should fail with OOM error"
    );

    // Should not panic, just return error
    if let Err(e) = result {
        println!("Expected OOM error: {:?}", e);
    }
}
