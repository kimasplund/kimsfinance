//! Tests for Batch Size Tuning Optimizations (Phase 4)
//!
//! Validates dynamic threshold calculation and per-phase block size selection.

#![cfg(feature = "gpu")]

use kimsfinance_core::backtest::batch::calculate_optimal_threshold;
use kimsfinance_core::gpu::device::GpuDevice;
use kimsfinance_core::gpu::persistent::{KernelPhase, optimal_block_size};
use std::sync::Arc;

// ===== Phase 1: Dynamic Threshold Tests =====

#[test]
fn test_dynamic_threshold_small_dataset() {
    let device = Arc::new(unsafe { std::mem::zeroed() }); // Dummy device

    // Small dataset: 10 strategies × 1000 candles = ~0.4MB
    let threshold = calculate_optimal_threshold(10, 1000, &device);

    // Should use conservative threshold (150)
    assert_eq!(threshold, 150, "Small datasets should use threshold=150");
}

#[test]
fn test_dynamic_threshold_medium_dataset() {
    let device = Arc::new(unsafe { std::mem::zeroed() }); // Dummy device

    // Medium dataset: 500 strategies × 5000 candles = ~100MB
    let threshold = calculate_optimal_threshold(500, 5000, &device);

    // Should use balanced threshold (100)
    assert_eq!(threshold, 100, "Medium datasets should use threshold=100");
}

#[test]
fn test_dynamic_threshold_large_dataset() {
    let device = Arc::new(unsafe { std::mem::zeroed() }); // Dummy device

    // Large dataset: 1000 strategies × 10000 candles = ~400MB
    let threshold = calculate_optimal_threshold(1000, 10000, &device);

    // Should use aggressive threshold (50)
    assert_eq!(threshold, 50, "Large datasets should use threshold=50");
}

#[test]
fn test_dynamic_threshold_edge_cases() {
    let device = Arc::new(unsafe { std::mem::zeroed() }); // Dummy device

    // Edge case: 0 strategies (should still return valid threshold)
    let threshold = calculate_optimal_threshold(0, 1000, &device);
    assert!(
        threshold > 0,
        "Should return valid threshold even for 0 strategies"
    );

    // Edge case: 0 candles (should still return valid threshold)
    let threshold = calculate_optimal_threshold(100, 0, &device);
    assert!(
        threshold > 0,
        "Should return valid threshold even for 0 candles"
    );
}

#[test]
fn test_threshold_progression() {
    let device = Arc::new(unsafe { std::mem::zeroed() }); // Dummy device

    // Thresholds should decrease as data size increases
    let small = calculate_optimal_threshold(10, 1000, &device); // <10MB
    let medium = calculate_optimal_threshold(500, 5000, &device); // 10-50MB
    let large = calculate_optimal_threshold(1000, 10000, &device); // >50MB

    assert!(
        large < medium && medium < small,
        "Thresholds should decrease with data size: {} < {} < {}",
        large,
        medium,
        small
    );
}

// ===== Phase 2: Block Size Selection Tests =====

#[test]
#[ignore] // Requires GPU
fn test_block_size_indicator_phase() {
    let device = GpuDevice::new().expect("GPU required");

    let block_size = optimal_block_size(KernelPhase::Indicator, &device);

    // Indicator phase: memory-bound → smaller blocks (128)
    assert_eq!(block_size, 128, "Indicator phase should use block size 128");
}

#[test]
#[ignore] // Requires GPU
fn test_block_size_signals_phase() {
    let device = GpuDevice::new().expect("GPU required");

    let block_size = optimal_block_size(KernelPhase::Signals, &device);

    // Signals phase: compute-bound → larger blocks (256)
    assert_eq!(block_size, 256, "Signals phase should use block size 256");
}

#[test]
#[ignore] // Requires GPU
fn test_block_size_execution_phase() {
    let device = GpuDevice::new().expect("GPU required");

    let block_size = optimal_block_size(KernelPhase::Execution, &device);

    // Execution phase: sequential → warp size (32)
    assert_eq!(
        block_size, 32,
        "Execution phase should use block size 32 (warp size)"
    );
}

#[test]
#[ignore] // Requires GPU
fn test_block_size_aggregation_phase() {
    let device = GpuDevice::new().expect("GPU required");

    let block_size = optimal_block_size(KernelPhase::Aggregation, &device);

    // Aggregation phase: reduction → medium blocks (64)
    assert_eq!(block_size, 64, "Aggregation phase should use block size 64");
}

#[test]
#[ignore] // Requires GPU
fn test_block_size_ordering() {
    let device = GpuDevice::new().expect("GPU required");

    let indicator = optimal_block_size(KernelPhase::Indicator, &device);
    let signals = optimal_block_size(KernelPhase::Signals, &device);
    let execution = optimal_block_size(KernelPhase::Execution, &device);
    let aggregation = optimal_block_size(KernelPhase::Aggregation, &device);

    // Verify ordering: Execution < Aggregation < Indicator < Signals
    assert!(
        execution < aggregation && aggregation < indicator && indicator < signals,
        "Block sizes should be ordered: {} < {} < {} < {}",
        execution,
        aggregation,
        indicator,
        signals
    );
}

#[test]
#[ignore] // Requires GPU
fn test_block_sizes_power_of_two() {
    let device = GpuDevice::new().expect("GPU required");

    // All block sizes should be powers of 2 for optimal GPU execution
    let phases = [
        KernelPhase::Indicator,
        KernelPhase::Signals,
        KernelPhase::Execution,
        KernelPhase::Aggregation,
    ];

    for phase in phases.iter() {
        let block_size = optimal_block_size(*phase, &device);
        assert!(
            block_size.is_power_of_two(),
            "{:?} block size {} should be power of 2",
            phase,
            block_size
        );
    }
}

// ===== Integration Tests =====

#[test]
#[ignore] // Requires GPU
fn test_threshold_and_block_size_consistency() {
    // Verify that threshold and block size functions work together correctly
    let device = Arc::new(GpuDevice::new().expect("GPU required"));

    // Test various dataset sizes
    let test_cases = vec![
        (10, 1000, 150),   // Small → threshold 150
        (500, 5000, 100),  // Medium → threshold 100
        (1000, 10000, 50), // Large → threshold 50
    ];

    for (num_strategies, num_candles, expected_threshold) in test_cases {
        let threshold = calculate_optimal_threshold(num_strategies, num_candles, &device);
        assert_eq!(
            threshold, expected_threshold,
            "Threshold mismatch for {}×{} dataset",
            num_strategies, num_candles
        );

        // Verify block sizes are valid regardless of threshold
        for phase in [
            KernelPhase::Indicator,
            KernelPhase::Signals,
            KernelPhase::Execution,
            KernelPhase::Aggregation,
        ]
        .iter()
        {
            let block_size = optimal_block_size(*phase, &device);
            assert!(
                block_size >= 32 && block_size <= 1024,
                "Block size {} for {:?} out of valid range [32, 1024]",
                block_size,
                phase
            );
        }
    }
}
