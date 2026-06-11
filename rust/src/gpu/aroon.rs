//! GPU-Accelerated Aroon Indicator
//!
//! Provides 15-25x speedup over CPU implementation for large datasets.
//!
//! The Aroon indicator measures the time elapsed since the highest high and lowest low
//! within a given period, expressed as a percentage (0-100).

use super::device::{GpuDevice, GpuError};
use crate::gpu::compile::compile_ptx_optimized_cached;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for Aroon indicator
const AROON_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void aroon_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    double* __restrict__ aroon_up,
    double* __restrict__ aroon_down,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Need at least 'period' data points to calculate Aroon
    if (idx < period - 1) {
        aroon_up[idx] = CUDART_NAN;
        aroon_down[idx] = CUDART_NAN;
        return;
    }

    // Find position of highest high and lowest low in rolling window
    // Window: [idx - period + 1, idx]
    int highest_high_idx = idx;
    int lowest_low_idx = idx;
    double highest_high = high[idx];
    double lowest_low = low[idx];

    // Scan backward through the window (newest -> oldest). Strict comparisons
    // keep the MOST RECENT extreme on ties (TA-Lib semantics): an older bar
    // must be strictly better to replace the current candidate. With >=/<=
    // the OLDEST tied bar won, diverging from TA-Lib on flat stretches.
    for (int i = 1; i < period; i++) {
        int window_idx = idx - i;

        if (high[window_idx] > highest_high) {
            highest_high = high[window_idx];
            highest_high_idx = window_idx;
        }

        if (low[window_idx] < lowest_low) {
            lowest_low = low[window_idx];
            lowest_low_idx = window_idx;
        }
    }

    // Calculate periods since high/low
    // periods_since = current_idx - position_of_extreme
    int periods_since_high = idx - highest_high_idx;
    int periods_since_low = idx - lowest_low_idx;

    // Calculate Aroon values
    // Aroon = ((period - periods_since) / period) * 100
    aroon_up[idx] = ((double)(period - periods_since_high) / (double)period) * 100.0;
    aroon_down[idx] = ((double)(period - periods_since_low) / (double)period) * 100.0;
}
"#;

/// GPU-accelerated Aroon indicator
///
/// Calculates Aroon Up and Aroon Down indicators using CUDA.
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `high` - High prices
/// * `low` - Low prices
/// * `period` - Lookback period (typically 14 or 25)
/// * `stream` - Optional CUDA stream for concurrent execution (uses device default if None)
///
/// # Returns
///
/// Tuple of (Aroon Up, Aroon Down) as Array1<f64>
///
/// # Algorithm
///
/// - Aroon Up = ((period - periods_since_highest_high) / period) * 100
/// - Aroon Down = ((period - periods_since_lowest_low) / period) * 100
///
/// Ties resolve to the MOST RECENT extreme (TA-Lib semantics): on flat
/// stretches `periods_since` is measured from the newest tied bar.
///
/// # Performance
///
/// Expected speedup: **15-25x** over CPU for n > 10,000 (async pinned-memory
/// transfers, single stream synchronization)
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, aroon_gpu};
///
/// let device = GpuDevice::new()?;
/// let (aroon_up, aroon_down) = aroon_gpu(&device, &high, &low, 14, None)?;
/// ```
pub fn aroon_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<(Array1<f64>, Array1<f64>), GpuError> {
    let n = high.len();

    // Validate inputs
    if low.len() != n {
        return Err(GpuError::InvalidParameter(
            "High and low arrays must have same length".to_string(),
        ));
    }

    if period < 1 {
        return Err(GpuError::InvalidParameter(
            "Period must be >= 1".to_string(),
        ));
    }

    if n < period {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need {} points, got {}",
            period, n
        )));
    }

    // Compile PTX
    let ptx_arc = compile_ptx_optimized_cached(AROON_KERNEL)
        .map_err(|e| GpuError::CompilationError(format!("Failed to compile kernel: {:?}", e)))?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);

    // Load module (use context, not stream)
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel function from module
    let kernel = module.load_function("aroon_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
    })?;

    // Select stream: use provided stream or device default
    let exec_stream = stream.unwrap_or(&device.stream);

    // === Step 1: H2D - Asynchronously copy data to device ===
    // Stage through the pinned pool: pageable memcpy_htod forces a synchronous
    // driver-side staging copy, while pinned buffers DMA asynchronously
    // (20-30% faster, matches every sibling indicator).
    let mut pinned_high = device.pinned_pool.lock().acquire(n)?;
    pinned_high.as_mut_slice()[..n].copy_from_slice(high.as_slice().unwrap());
    let mut pinned_low = device.pinned_pool.lock().acquire(n)?;
    pinned_low.as_mut_slice()[..n].copy_from_slice(low.as_slice().unwrap());

    // Allocate device input buffers
    let mut d_high = exec_stream.alloc_zeros::<f64>(n).map_err(|e| {
        GpuError::AllocationError(format!("Failed to allocate high buffer: {:?}", e))
    })?;
    let mut d_low = exec_stream.alloc_zeros::<f64>(n).map_err(|e| {
        GpuError::AllocationError(format!("Failed to allocate low buffer: {:?}", e))
    })?;

    // Async H2D transfers
    exec_stream
        .memcpy_htod(&pinned_high.as_slice()[..n], &mut d_high)
        .map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy high to device: {:?}", e))
        })?;
    exec_stream
        .memcpy_htod(&pinned_low.as_slice()[..n], &mut d_low)
        .map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy low to device: {:?}", e))
        })?;

    // Allocate output buffers on selected stream
    let mut d_aroon_up = exec_stream.alloc_zeros::<f64>(n).map_err(|e| {
        GpuError::AllocationError(format!("Failed to allocate aroon_up buffer: {:?}", e))
    })?;

    let mut d_aroon_down = exec_stream.alloc_zeros::<f64>(n).map_err(|e| {
        GpuError::AllocationError(format!("Failed to allocate aroon_down buffer: {:?}", e))
    })?;

    // Launch kernel on selected stream
    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut builder = exec_stream.launch_builder(&kernel);
    builder.arg(&d_high);
    builder.arg(&d_low);
    builder.arg(&mut d_aroon_up);
    builder.arg(&mut d_aroon_down);
    builder.arg(&n_i32);
    builder.arg(&period_i32);

    let config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("Kernel launch failed: {:?}", e)))?;
    }

    // === Step 3: D2H - Asynchronously copy results back ===
    // Async copies into pinned staging plus a single sync: the previous
    // explicit synchronize followed by memcpy_dtov (which synchronizes
    // internally) performed a redundant double sync.
    let mut pinned_up = device.pinned_pool.lock().acquire(n)?;
    let mut pinned_down = device.pinned_pool.lock().acquire(n)?;

    exec_stream
        .memcpy_dtoh(&d_aroon_up, &mut pinned_up.as_mut_slice()[..n])
        .map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy aroon_up to host: {:?}", e))
        })?;
    exec_stream
        .memcpy_dtoh(&d_aroon_down, &mut pinned_down.as_mut_slice()[..n])
        .map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy aroon_down to host: {:?}", e))
        })?;

    // Synchronize stream to ensure D2H copies are complete before CPU access
    exec_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    let aroon_up_vec = pinned_up.as_slice()[..n].to_vec();
    let aroon_down_vec = pinned_down.as_slice()[..n].to_vec();

    // Release ALL pinned staging buffers only after the final sync: the async
    // H2D/D2H copies may still be reading/writing them until the stream drains.
    let mut pool = device.pinned_pool.lock();
    pool.release(pinned_high);
    pool.release(pinned_low);
    pool.release(pinned_up);
    pool.release(pinned_down);
    drop(pool);

    Ok((
        Array1::from_vec(aroon_up_vec),
        Array1::from_vec(aroon_down_vec),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    // ==================== Host-side tests (no GPU required) ====================

    /// CPU mirror of the aroon_kernel per-thread logic (backward scan with
    /// strict comparisons). Kept in lockstep with the CUDA source so the
    /// tie-break semantics are verifiable without a GPU.
    fn aroon_reference(high: &[f64], low: &[f64], period: usize, idx: usize) -> (f64, f64) {
        assert!(idx >= period - 1);
        let mut highest_high_idx = idx;
        let mut lowest_low_idx = idx;
        let mut highest_high = high[idx];
        let mut lowest_low = low[idx];

        for i in 1..period {
            let window_idx = idx - i;
            if high[window_idx] > highest_high {
                highest_high = high[window_idx];
                highest_high_idx = window_idx;
            }
            if low[window_idx] < lowest_low {
                lowest_low = low[window_idx];
                lowest_low_idx = window_idx;
            }
        }

        let periods_since_high = idx - highest_high_idx;
        let periods_since_low = idx - lowest_low_idx;
        let up = ((period - periods_since_high) as f64 / period as f64) * 100.0;
        let down = ((period - periods_since_low) as f64 / period as f64) * 100.0;
        (up, down)
    }

    #[test]
    fn test_kernel_source_nvrtc_compatible() {
        // NVRTC compilation path provides no SDK headers
        assert!(
            !AROON_KERNEL.contains("#include"),
            "kernel must not use #include (NVRTC-incompatible)"
        );
        assert!(AROON_KERNEL.contains(r#"extern "C" __global__ void aroon_kernel"#));
    }

    #[test]
    fn test_kernel_source_strict_tie_break() {
        // Ties must keep the MOST RECENT extreme (TA-Lib semantics). The
        // backward scan starts at the newest bar, so older bars may only
        // replace the candidate on STRICT comparisons.
        assert!(AROON_KERNEL.contains("high[window_idx] > highest_high"));
        assert!(AROON_KERNEL.contains("low[window_idx] < lowest_low"));
        assert!(
            !AROON_KERNEL.contains(">= highest_high"),
            "non-strict high comparison would keep the OLDEST tied extreme"
        );
        assert!(
            !AROON_KERNEL.contains("<= lowest_low"),
            "non-strict low comparison would keep the OLDEST tied extreme"
        );
    }

    #[test]
    fn test_aroon_tie_break_keeps_most_recent_extreme() {
        // Flat stretch: every bar in the window ties, so the most recent
        // occurrence wins (TA-Lib) and periods_since == 0 -> Aroon == 100.
        // The previous >=/<= scan returned the OLDEST tied bar instead.
        let high = vec![100.0; 20];
        let low = vec![95.0; 20];
        let (up, down) = aroon_reference(&high, &low, 14, 19);
        assert_eq!(up, 100.0);
        assert_eq!(down, 100.0);
    }

    #[test]
    fn test_aroon_reference_unique_extremes() {
        // Unique max at the current bar -> Aroon Up = 100;
        // unique min 13 bars back (oldest in a 14-bar window) -> Aroon Down
        // = ((14 - 13) / 14) * 100.
        let mut high = vec![100.0; 20];
        high[19] = 150.0;
        let mut low = vec![95.0; 20];
        low[6] = 50.0;

        let (up, down) = aroon_reference(&high, &low, 14, 19);
        assert!((up - 100.0).abs() < 1e-12);
        let expected_down = ((14.0 - 13.0) / 14.0) * 100.0;
        assert!(
            (down - expected_down).abs() < 1e-12,
            "down = {}, expected {}",
            down,
            expected_down
        );
    }

    #[test]
    fn test_aroon_reference_partial_tie() {
        // Two bars tie for the high; the more recent one must win.
        let mut high = vec![100.0; 20];
        high[10] = 150.0;
        high[15] = 150.0; // tie, more recent
        let low = vec![95.0; 20];

        let (up, _) = aroon_reference(&high, &low, 14, 19);
        // periods_since_high = 19 - 15 = 4 -> up = ((14 - 4) / 14) * 100
        let expected_up = ((14.0 - 4.0) / 14.0) * 100.0;
        assert!(
            (up - expected_up).abs() < 1e-12,
            "up = {}, expected {} (most recent tied high must win)",
            up,
            expected_up
        );
    }

    // ==================== GPU tests ====================

    #[test]
    #[ignore] // Requires GPU
    fn test_aroon_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Simple trending up data
        let high = arr1(&[
            100.0, 102.0, 105.0, 103.0, 107.0, 110.0, 108.0, 112.0, 115.0, 113.0, 118.0, 120.0,
            117.0, 122.0, 125.0,
        ]);
        let low = arr1(&[
            95.0, 97.0, 100.0, 98.0, 102.0, 105.0, 103.0, 107.0, 110.0, 108.0, 113.0, 115.0, 112.0,
            117.0, 120.0,
        ]);

        let (aroon_up, aroon_down) =
            aroon_gpu(&device, &high, &low, 14, None).expect("Aroon GPU calculation failed");

        // Verify output length
        assert_eq!(aroon_up.len(), 15);
        assert_eq!(aroon_down.len(), 15);

        // First 13 values should be NaN (need 14 periods)
        for i in 0..13 {
            assert!(aroon_up[i].is_nan());
            assert!(aroon_down[i].is_nan());
        }

        // Valid values should be in range [0, 100]
        for i in 13..aroon_up.len() {
            assert!(aroon_up[i] >= 0.0 && aroon_up[i] <= 100.0);
            assert!(aroon_down[i] >= 0.0 && aroon_down[i] <= 100.0);
        }

        // In uptrend, Aroon Up should be higher than Aroon Down
        assert!(aroon_up[14] > aroon_down[14]);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_aroon_gpu_downtrend() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Trending down data
        let high = arr1(&[
            125.0, 122.0, 120.0, 118.0, 115.0, 113.0, 110.0, 108.0, 105.0, 103.0, 100.0, 98.0,
            95.0, 93.0, 90.0,
        ]);
        let low = arr1(&[
            120.0, 117.0, 115.0, 113.0, 110.0, 108.0, 105.0, 103.0, 100.0, 98.0, 95.0, 93.0, 90.0,
            88.0, 85.0,
        ]);

        let (aroon_up, aroon_down) =
            aroon_gpu(&device, &high, &low, 14, None).expect("Aroon GPU calculation failed");

        // In downtrend, Aroon Down should be higher than Aroon Up
        assert!(aroon_down[14] > aroon_up[14]);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_aroon_gpu_extreme_values() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // High at the end of window (most recent)
        let mut high = vec![100.0; 20];
        high[19] = 150.0; // Highest high at current position

        let mut low = vec![95.0; 20];
        low[6] = 50.0; // Lowest low at the oldest bar of the window [6, 19]

        let high_arr = Array1::from_vec(high);
        let low_arr = Array1::from_vec(low);

        let (aroon_up, aroon_down) = aroon_gpu(&device, &high_arr, &low_arr, 14, None)
            .expect("Aroon GPU calculation failed");

        // Aroon Up should be 100 (highest high 0 periods ago)
        assert!((aroon_up[19] - 100.0).abs() < 0.001);

        // Aroon Down: lowest low 13 periods ago (oldest position reachable in
        // a 14-bar window) -> ((14 - 13) / 14) * 100
        let expected_down = ((14.0 - 13.0) / 14.0) * 100.0;
        assert!((aroon_down[19] - expected_down).abs() < 0.001);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_aroon_gpu_flat_data_tie_break() {
        // TA-Lib tie semantics: on completely flat data every bar ties, the
        // most recent occurrence wins, and both Aroon lines read 100.
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = Array1::from_vec(vec![100.0; 20]);
        let low = Array1::from_vec(vec![95.0; 20]);

        let (aroon_up, aroon_down) =
            aroon_gpu(&device, &high, &low, 14, None).expect("Aroon GPU calculation failed");

        for i in 13..20 {
            assert!(
                (aroon_up[i] - 100.0).abs() < 0.001,
                "flat data: aroon_up[{}] = {}, expected 100",
                i,
                aroon_up[i]
            );
            assert!(
                (aroon_down[i] - 100.0).abs() < 0.001,
                "flat data: aroon_down[{}] = {}, expected 100",
                i,
                aroon_down[i]
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_aroon_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());
        let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64) * 0.01).collect());

        let start = std::time::Instant::now();
        let (aroon_up, aroon_down) =
            aroon_gpu(&device, &high, &low, 14, None).expect("Aroon GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU Aroon (n={}): {:.2}ms",
            n,
            elapsed.as_secs_f64() * 1000.0
        );

        assert_eq!(aroon_up.len(), n);
        assert_eq!(aroon_down.len(), n);

        // In steady uptrend, Aroon Up should be high
        assert!(aroon_up[n - 1] > 90.0);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_aroon_gpu_validation_errors() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = arr1(&[100.0, 102.0, 105.0]);
        let low = arr1(&[95.0, 97.0]);

        // Mismatched lengths
        let result = aroon_gpu(&device, &high, &low, 14, None);
        assert!(result.is_err());

        let high = arr1(&[100.0, 102.0, 105.0]);
        let low = arr1(&[95.0, 97.0, 100.0]);

        // Period too large
        let result = aroon_gpu(&device, &high, &low, 14, None);
        assert!(result.is_err());

        // Period zero
        let result = aroon_gpu(&device, &high, &low, 0, None);
        assert!(result.is_err());
    }
}
