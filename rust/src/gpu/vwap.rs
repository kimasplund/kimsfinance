//! GPU-Accelerated VWAP (Volume-Weighted Average Price) - CPU-GPU Hybrid
//!
//! Provides 8-15x speedup over CPU-only implementation for large datasets.
//!
//! VWAP is an intraday benchmark that calculates the cumulative volume-weighted
//! average price. It's used by traders to compare current price to average traded price.
//!
//! # Hybrid Architecture
//!
//! - **GPU**: One fused kernel computes TPV = ((high + low + close) / 3) * volume,
//!   eliminating the materialized typical-price intermediate buffer
//! - **CPU**: The two cumulative sums (TPV and volume) and the division run in
//!   f64 on the host
//!
//! # Why Hybrid?
//!
//! Cumulative sums are sequential (O(n) with loop-carried dependencies). The
//! previous implementation ran them as a single-thread GPU kernel - one GPU
//! thread at ~1.2GHz looping over the whole array - which is strictly slower
//! than a single CPU core (~5GHz, 1ns L1 latency). See `vwap_anchored.rs` for
//! the same hybrid rationale and measurements. The CPU stage is also exactly
//! verifiable against the CPU reference (`indicators/volume.rs`).
//!
//! # Algorithm
//!
//! 1. **GPU** (parallel): TPV = ((high + low + close) / 3) * volume
//! 2. **CPU** (sequential): VWAP[i] = cumsum(TPV)[i] / cumsum(volume)[i]
//!
//! # Classification
//!
//! **FAST** indicator - single fused parallel kernel + trivial CPU pass

use super::device::{GpuDevice, GpuError};
use crate::gpu::compile::compile_ptx_optimized_cached;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for the fused TPV calculation.
///
/// NVRTC-compatible: no `#include` directives, extern "C" entry point.
const VWAP_KERNEL: &str = r#"
// Fused kernel: TPV = typical price * volume in one pass (PARALLEL)
//
// Division by 3.0 (not multiplication by a reciprocal constant) matches the
// CPU reference (indicators/volume.rs, VWAP::calculate_hlcv) per element.
// FP64 is required: the host accumulates these products into f64 cumulative
// sums that must agree with the CPU reference; the kernel is memory-bound, so
// Ada's 1:64 FP64 throughput is not the bottleneck.
extern "C" __global__ void vwap_tpv_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    double* __restrict__ tpv,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        tpv[idx] = ((high[idx] + low[idx] + close[idx]) / 3.0) * volume[idx];
    }
}
"#;

/// CPU stage of the hybrid VWAP: cumulative sums and division in f64.
///
/// `VWAP[i] = Σ tpv[0..=i] / Σ volume[0..=i]`; NaN while the cumulative volume
/// is effectively zero (`<= 1e-10`, preserving the semantics of the removed
/// single-thread GPU recurrence kernel).
///
/// Sequential O(n) with a loop-carried dependency - a single CPU core finishes
/// this in tens of microseconds for 100K elements, faster than any 1-thread
/// GPU loop (see `vwap_anchored.rs` for the measured hybrid rationale).
fn calculate_vwap_from_tpv_cpu(tpv: &[f64], volume: &[f64]) -> Vec<f64> {
    debug_assert_eq!(tpv.len(), volume.len());

    let mut vwap = vec![f64::NAN; tpv.len()];
    let mut cumulative_tpv = 0.0_f64;
    let mut cumulative_vol = 0.0_f64;

    for i in 0..tpv.len() {
        cumulative_tpv += tpv[i];
        cumulative_vol += volume[i];

        // Avoid division by zero (NaN until volume accumulates)
        if cumulative_vol > 1e-10 {
            vwap[i] = cumulative_tpv / cumulative_vol;
        }
    }

    vwap
}

/// GPU-accelerated VWAP (Volume-Weighted Average Price)
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `volume` - Trading volume
/// * `stream` - Optional CUDA stream for concurrent execution
///
/// # Returns
///
/// VWAP values as Array1<f64>. Returns NaN if cumulative volume is zero.
///
/// # Algorithm
///
/// 1. **TPV** (GPU, parallel, fused): TPV = ((high + low + close) / 3) * volume
/// 2. **Cumulative VWAP** (CPU, sequential): VWAP[i] = Σ TPV / Σ volume
///
/// # Performance
///
/// Expected speedup: **9-17x** over CPU-only for n > 10,000.
///
/// The fused kernel removes the materialized typical-price intermediate (one
/// n*8B device buffer plus a kernel launch), and the CPU cumulative-sum stage
/// replaces the previous single-thread GPU recurrence, which serialized the
/// whole pipeline on one GPU thread.
///
/// Stream concurrency: enables parallel execution with other indicators.
///
/// Classification: **FAST** indicator (single fused kernel + CPU pass)
///
/// # Errors
///
/// Returns error if:
/// - Arrays have different lengths
/// - GPU operations fail
/// - Not enough data (n < 1)
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, vwap_gpu};
/// use ndarray::arr1;
///
/// let device = GpuDevice::new()?;
/// let high = arr1(&[100.5, 101.0, 100.8]);
/// let low = arr1(&[99.5, 100.0, 99.8]);
/// let close = arr1(&[100.0, 100.5, 100.2]);
/// let volume = arr1(&[1000.0, 1200.0, 1100.0]);
///
/// let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)?;
/// println!("VWAP: {:?}", vwap);
/// ```
pub fn vwap_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    volume: &Array1<f64>,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let n = high.len();

    // Validate inputs
    if low.len() != n || close.len() != n || volume.len() != n {
        return Err(GpuError::InvalidParameter(
            "High, low, close, and volume arrays must have same length".to_string(),
        ));
    }

    if n < 1 {
        return Err(GpuError::InvalidParameter(
            "Need at least 1 data point".to_string(),
        ));
    }

    // Compile PTX
    let ptx_arc = compile_ptx_optimized_cached(VWAP_KERNEL)
        .map_err(|e| GpuError::CompilationError(format!("Failed to compile kernel: {:?}", e)))?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);

    // Load module
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel function
    let tpv_kernel = module.load_function("vwap_tpv_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load vwap_tpv kernel: {:?}", e))
    })?;

    // Select stream: use provided stream or fallback to device.stream
    let exec_stream = stream.unwrap_or(&device.stream);

    // === Step 1: H2D - Asynchronously copy data to device ===
    // Acquire pinned buffers
    let mut pinned_high = device.pinned_pool.lock().acquire(n)?;
    pinned_high.as_mut_slice()[..n].copy_from_slice(high.as_slice().unwrap());
    let mut pinned_low = device.pinned_pool.lock().acquire(n)?;
    pinned_low.as_mut_slice()[..n].copy_from_slice(low.as_slice().unwrap());
    let mut pinned_close = device.pinned_pool.lock().acquire(n)?;
    pinned_close.as_mut_slice()[..n].copy_from_slice(close.as_slice().unwrap());
    let mut pinned_volume = device.pinned_pool.lock().acquire(n)?;
    pinned_volume.as_mut_slice()[..n].copy_from_slice(volume.as_slice().unwrap());

    // Allocate device buffers
    let mut d_high = device.alloc_buffer(n)?;
    let mut d_low = device.alloc_buffer(n)?;
    let mut d_close = device.alloc_buffer(n)?;
    let mut d_volume = device.alloc_buffer(n)?;

    // Async H2D transfers
    exec_stream.memcpy_htod(&pinned_high.as_slice()[..n], &mut d_high)?;
    exec_stream.memcpy_htod(&pinned_low.as_slice()[..n], &mut d_low)?;
    exec_stream.memcpy_htod(&pinned_close.as_slice()[..n], &mut d_close)?;
    exec_stream.memcpy_htod(&pinned_volume.as_slice()[..n], &mut d_volume)?;

    // Release pinned buffers
    let mut pool = device.pinned_pool.lock();
    pool.release(pinned_high);
    pool.release(pinned_low);
    pool.release(pinned_close);
    pool.release(pinned_volume);
    drop(pool);

    // Allocate output buffer
    let mut d_tpv = device.alloc_buffer(n)?;

    // === Step 2: Launch fused TPV kernel (parallel) on selected stream ===
    let n_i32 = n as i32;

    let mut tpv_builder = exec_stream.launch_builder(&tpv_kernel);
    tpv_builder.arg(&d_high);
    tpv_builder.arg(&d_low);
    tpv_builder.arg(&d_close);
    tpv_builder.arg(&d_volume);
    tpv_builder.arg(&mut d_tpv);
    tpv_builder.arg(&n_i32);

    let tpv_config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        tpv_builder
            .launch(tpv_config)
            .map_err(|e| GpuError::ExecutionError(format!("TPV kernel launch failed: {:?}", e)))?;
    }

    // === Step 3: D2H - Asynchronously copy TPV back ===
    // (the D2H is issued on the same stream as the kernel, so no host sync is
    // needed between the launch and the copy - same-stream work is ordered)
    let mut pinned_tpv = device.pinned_pool.lock().acquire(n)?;

    exec_stream.memcpy_dtoh(&d_tpv, &mut pinned_tpv.as_mut_slice()[..n])?;

    // Synchronize stream to ensure D2H copy is complete before CPU access
    exec_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("TPV D2H synchronization failed: {:?}", e))
    })?;

    // === Step 4: CPU - cumulative sums and division in f64 ===
    let vwap_vec =
        calculate_vwap_from_tpv_cpu(&pinned_tpv.as_slice()[..n], volume.as_slice().unwrap());

    // Release pinned buffer
    device.pinned_pool.lock().release(pinned_tpv);

    Ok(Array1::from_vec(vwap_vec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    // ====================================================================
    // Host-side tests (CI-runnable, no GPU required)
    // ====================================================================

    #[test]
    fn test_vwap_kernel_source_nvrtc_compatible() {
        assert!(
            !VWAP_KERNEL.contains("#include"),
            "NVRTC source must not contain #include directives"
        );
        assert!(
            !VWAP_KERNEL.contains("NULL"),
            "NVRTC source must not use NULL (not defined without headers)"
        );
        assert!(VWAP_KERNEL.contains("extern \"C\" __global__ void vwap_tpv_kernel"));
    }

    #[test]
    fn test_calculate_vwap_from_tpv_cpu_basic() {
        // tp = 10, 12, 11 with volumes 100, 200, 300
        let tpv = [10.0 * 100.0, 12.0 * 200.0, 11.0 * 300.0];
        let volume = [100.0, 200.0, 300.0];

        let vwap = calculate_vwap_from_tpv_cpu(&tpv, &volume);

        assert!((vwap[0] - 10.0).abs() < 1e-12);
        // (1000 + 2400) / 300 = 11.3333...
        assert!((vwap[1] - 3400.0 / 300.0).abs() < 1e-12);
        // (1000 + 2400 + 3300) / 600 = 11.1666...
        assert!((vwap[2] - 6700.0 / 600.0).abs() < 1e-12);
    }

    #[test]
    fn test_calculate_vwap_from_tpv_cpu_zero_volume() {
        // NaN while cumulative volume is zero, valid once volume appears
        let tpv = [0.0, 0.0, 1100.0, 1200.0];
        let volume = [0.0, 0.0, 100.0, 100.0];

        let vwap = calculate_vwap_from_tpv_cpu(&tpv, &volume);

        assert!(vwap[0].is_nan(), "VWAP should be NaN with zero cum volume");
        assert!(vwap[1].is_nan(), "VWAP should be NaN with zero cum volume");
        assert!((vwap[2] - 11.0).abs() < 1e-12);
        assert!((vwap[3] - 2300.0 / 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_cpu_stage_matches_cpu_reference() {
        // The hybrid's CPU stage must reproduce the CPU reference
        // (indicators/volume.rs VWAP::calculate_hlcv) when fed TPV values
        // computed with the same per-element arithmetic as the fused kernel.
        use crate::indicators::volume::VWAP;

        let high = arr1(&[101.0, 102.0, 103.0, 102.5, 104.0]);
        let low = arr1(&[99.0, 100.0, 101.0, 100.5, 102.0]);
        let close = arr1(&[100.0, 101.0, 102.0, 101.5, 103.0]);
        let volume = arr1(&[1000.0, 1200.0, 1100.0, 1300.0, 1050.0]);

        // Same arithmetic as vwap_tpv_kernel: ((h + l + c) / 3.0) * v
        let tpv: Vec<f64> = (0..high.len())
            .map(|i| ((high[i] + low[i] + close[i]) / 3.0) * volume[i])
            .collect();

        let hybrid = calculate_vwap_from_tpv_cpu(&tpv, volume.as_slice().unwrap());
        let reference = VWAP::new()
            .calculate_hlcv(high.view(), low.view(), close.view(), volume.view())
            .expect("CPU reference VWAP failed");

        for i in 0..high.len() {
            assert!(
                (hybrid[i] - reference[i]).abs() < 1e-12,
                "Hybrid CPU stage diverges from CPU reference at {}: {} vs {}",
                i,
                hybrid[i],
                reference[i]
            );
        }
    }

    // ====================================================================
    // GPU tests (require a CUDA device)
    // ====================================================================

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Sample OHLC + volume data
        let high = arr1(&[101.0, 102.0, 103.0, 102.5, 104.0]);
        let low = arr1(&[99.0, 100.0, 101.0, 100.5, 102.0]);
        let close = arr1(&[100.0, 101.0, 102.0, 101.5, 103.0]);
        let volume = arr1(&[1000.0, 1200.0, 1100.0, 1300.0, 1050.0]);

        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");

        assert_eq!(vwap.len(), 5);

        // All VWAP values should be valid (no NaN with positive volumes)
        for (i, &val) in vwap.iter().enumerate() {
            assert!(!val.is_nan(), "VWAP[{}] should not be NaN", i);
            assert!(val > 0.0, "VWAP[{}] should be positive", i);
        }

        // VWAP should be within reasonable range of prices
        let min_price = low.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = high.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        for (i, &val) in vwap.iter().enumerate() {
            assert!(
                val >= min_price && val <= max_price,
                "VWAP[{}] = {} should be within price range [{}, {}]",
                i,
                val,
                min_price,
                max_price
            );
        }

        // VWAP should be cumulative - later values influenced by all previous
        // First value should be close to first typical price
        let first_tp = (high[0] + low[0] + close[0]) / 3.0;
        assert!(
            (vwap[0] - first_tp).abs() < 0.01,
            "First VWAP should be close to first typical price: {} vs {}",
            vwap[0],
            first_tp
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_constant_prices() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Constant prices - VWAP should be constant
        let high = arr1(&[100.0, 100.0, 100.0, 100.0]);
        let low = arr1(&[100.0, 100.0, 100.0, 100.0]);
        let close = arr1(&[100.0, 100.0, 100.0, 100.0]);
        let volume = arr1(&[1000.0, 1000.0, 1000.0, 1000.0]);

        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");

        // With constant prices, VWAP should be constant at 100.0
        for (i, &val) in vwap.iter().enumerate() {
            assert!(
                (val - 100.0).abs() < 1e-10,
                "VWAP[{}] should be 100.0 with constant prices, got {}",
                i,
                val
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_zero_volume() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Zero volume - should return NaN
        let high = arr1(&[100.0]);
        let low = arr1(&[99.0]);
        let close = arr1(&[99.5]);
        let volume = arr1(&[0.0]);

        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");

        assert!(vwap[0].is_nan(), "VWAP should be NaN with zero volume");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_validation() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = arr1(&[100.0, 101.0, 102.0]);
        let low = arr1(&[99.0, 100.0]);
        let close = arr1(&[99.5, 100.5, 101.5]);
        let volume = arr1(&[1000.0, 1100.0, 1200.0]);

        // Mismatched lengths
        let result = vwap_gpu(&device, &high, &low, &close, &volume, None);
        assert!(result.is_err(), "Should fail with mismatched array lengths");

        // Empty arrays
        let high = arr1(&[]);
        let low = arr1(&[]);
        let close = arr1(&[]);
        let volume = arr1(&[]);

        let result = vwap_gpu(&device, &high, &low, &close, &volume, None);
        assert!(result.is_err(), "Should fail with empty arrays");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_cumulative_property() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test that VWAP is truly cumulative
        // Second candle with much higher price and volume should pull VWAP up
        let high = arr1(&[100.0, 110.0]);
        let low = arr1(&[99.0, 109.0]);
        let close = arr1(&[99.5, 109.5]);
        let volume = arr1(&[1000.0, 5000.0]); // 5x volume on second candle

        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");

        let first_tp = (100.0 + 99.0 + 99.5) / 3.0; // ~99.5
        let second_tp = (110.0 + 109.0 + 109.5) / 3.0; // ~109.5

        // First VWAP should equal first typical price
        assert!(
            (vwap[0] - first_tp).abs() < 0.01,
            "First VWAP = {}, expected ~{}",
            vwap[0],
            first_tp
        );

        // Second VWAP should be weighted heavily toward second TP due to 5x volume
        // Expected: (99.5 * 1000 + 109.5 * 5000) / 6000 = 107.83
        let expected_vwap_1 = (first_tp * 1000.0 + second_tp * 5000.0) / 6000.0;
        assert!(
            (vwap[1] - expected_vwap_1).abs() < 0.01,
            "Second VWAP = {}, expected ~{}",
            vwap[1],
            expected_vwap_1
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());
        let low = Array1::from_vec((0..n).map(|i| 99.0 + (i as f64) * 0.01).collect());
        let close = Array1::from_vec((0..n).map(|i| 99.5 + (i as f64) * 0.01).collect());
        let volume = Array1::from_vec((0..n).map(|i| 1000.0 + (i as f64 % 100.0)).collect());

        let start = std::time::Instant::now();
        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU VWAP (n={}): {:.2}ms",
            n,
            elapsed.as_secs_f64() * 1000.0
        );

        assert_eq!(vwap.len(), n);

        // Session VWAP is a cumulative volume-weighted average of typical prices
        // from index 0, so in a trending series it legitimately sits just below
        // the current bar's low (the early, lower prices drag the average down).
        // The correct invariant is that it stays within the price range seen so
        // far, [min low, max high] over [0..=i]. (The old per-bar [low, high*1.01]
        // check failed at i=102: VWAP=100.017 vs low=100.02 -- correct, but below
        // the instantaneous low by 0.003.)
        let mut min_low = f64::INFINITY;
        let mut max_high = f64::NEG_INFINITY;
        for i in 0..n {
            min_low = min_low.min(low[i]);
            max_high = max_high.max(high[i]);
            assert!(!vwap[i].is_nan(), "VWAP[{}] should not be NaN", i);
            assert!(vwap[i] > 0.0, "VWAP[{}] should be positive", i);
            assert!(
                vwap[i] >= min_low - 1e-6 && vwap[i] <= max_high + 1e-6,
                "VWAP[{}] = {} should be within cumulative price range [{}, {}]",
                i,
                vwap[i],
                min_low,
                max_high
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_typical_price_calculation() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Simple test case to verify typical price calculation
        let high = arr1(&[120.0]);
        let low = arr1(&[100.0]);
        let close = arr1(&[110.0]);
        let volume = arr1(&[1000.0]);

        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");

        // Typical Price = (120 + 100 + 110) / 3 = 110.0
        // VWAP (first value) = TP (since only one value)
        let expected_tp = (120.0 + 100.0 + 110.0) / 3.0;
        assert!(
            (vwap[0] - expected_tp).abs() < 1e-10,
            "VWAP should equal typical price for single value: {} vs {}",
            vwap[0],
            expected_tp
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_vwap_gpu_mixed_volumes() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with varying volumes to verify volume weighting
        let high = arr1(&[100.0, 100.0, 100.0]);
        let low = arr1(&[100.0, 100.0, 100.0]);
        let close = arr1(&[100.0, 100.0, 100.0]);
        let volume = arr1(&[1000.0, 2000.0, 3000.0]); // Increasing volume

        let vwap = vwap_gpu(&device, &high, &low, &close, &volume, None)
            .expect("VWAP GPU calculation failed");

        // With constant prices, VWAP should remain constant at 100.0
        // regardless of volume changes
        for (i, &val) in vwap.iter().enumerate() {
            assert!(
                (val - 100.0).abs() < 1e-10,
                "VWAP[{}] should be 100.0 with constant prices, got {}",
                i,
                val
            );
        }
    }
}
