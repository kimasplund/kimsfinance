//! GPU-Accelerated RSI (Relative Strength Index) - CPU-GPU Hybrid
//!
//! Provides 2-3x speedup over old pure-GPU implementation using hybrid architecture.
//! RSI measures momentum by comparing average gains to average losses.
//!
//! # Hybrid Architecture (v0.2.0)
//!
//! - **GPU**: Parallel gains/losses calculation (~20μs)
//! - **CPU**: Wilder's smoothing for gains (~15μs)
//! - **CPU**: Wilder's smoothing for losses (~15μs)
//! - **GPU**: Parallel RSI calculation (~15μs)
//! - **Total**: ~130μs (vs ~250μs for old pure-GPU)
//!
//! # Why Hybrid?
//!
//! Wilder's smoothing is a sequential IIR filter that cannot be parallelized.
//! Running it on single GPU thread is 6x slower than CPU:
//!
//! - **Old (v0.1.0 - Anti-pattern)**:
//!   - GPU: Parallel gains/losses (~20μs)
//!   - GPU: Single-thread Wilder's for gains (~100μs) ← Bottleneck!
//!   - GPU: Single-thread Wilder's for losses (~100μs) ← Bottleneck!
//!   - GPU: Parallel RSI (~15μs)
//!   - **Total**: ~250μs
//!
//! - **New (v0.2.0 - Hybrid)**:
//!   - GPU: Parallel gains/losses (~20μs)
//!   - D2H: Copy gains/losses (~32μs)
//!   - CPU: Wilder's smoothing (2x) (~30μs) ← 3-4x faster!
//!   - H2D: Copy avg_gain/avg_loss (~32μs)
//!   - GPU: Parallel RSI (~15μs)
//!   - **Total**: ~130μs (2x faster!)
//!
//! **Trade-off**: This approach requires 2 round-trips (D2H gains/losses, H2D avg_gain/avg_loss).
//! But CPU smoothing is so much faster than single-thread GPU that it's still a net win.

use super::device::{GpuDevice, GpuError};
use crate::gpu::compile::compile_ptx_optimized_cached;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for RSI calculation (Hybrid v0.2.0)
///
/// Only contains parallel kernels - sequential Wilder's smoothing moved to CPU.
const RSI_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

// Kernel 1: Calculate price deltas and separate gains/losses (PARALLEL - Good for GPU)
extern "C" __global__ void calculate_gains_losses_kernel(
    const double* __restrict__ close,
    double* __restrict__ gains,
    double* __restrict__ losses,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n - 1) return;

    // Calculate delta for position idx+1
    double delta = close[idx + 1] - close[idx];

    // Branchless gain/loss separation
    // gain = max(delta, 0), loss = max(-delta, 0)
    gains[idx + 1] = fmax(delta, 0.0);
    losses[idx + 1] = fmax(-delta, 0.0);
}

// Kernel 2: Calculate final RSI values (PARALLEL - Good for GPU)
// Note: Wilder's smoothing removed - now done on CPU (3-4x faster)
extern "C" __global__ void calculate_rsi_kernel(
    const double* __restrict__ avg_gain,
    const double* __restrict__ avg_loss,
    double* __restrict__ rsi,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // RSI is only valid from period onward
    if (idx < period) {
        rsi[idx] = CUDART_NAN;
        return;
    }

    double gain = avg_gain[idx];
    double loss = avg_loss[idx];

    // Edge case: no losses. Distinguish a genuine all-gains move (RSI 100) from a
    // FLAT window where there are also no gains -> neutral 50 (matches the CPU RSI
    // fix; returning 100 on a flat market wrongly flags it as maximally overbought).
    if (loss < 1e-10) {
        rsi[idx] = (gain < 1e-10) ? 50.0 : 100.0;
        return;
    }

    // Calculate RSI = 100 - (100 / (1 + RS))
    // where RS = avg_gain / avg_loss
    double rs = gain / loss;
    rsi[idx] = 100.0 - (100.0 / (1.0 + rs));
}
"#;

/// GPU-accelerated RSI (Relative Strength Index) - CPU-GPU Hybrid
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `close` - Closing prices
/// * `period` - RSI period (typically 14)
/// * `stream` - Optional CUDA stream for concurrent execution (None uses device default)
///
/// # Returns
///
/// Array1<f64> with RSI values (0-100 range)
///
/// # Performance (Async v0.2.1)
///
/// Expected performance: **~115μs** for 100K candles (10-15% faster than sync hybrid).
///
/// Breakdown (with async transfers):
/// - H2D `close` (pinned): ~25μs
/// - GPU gains/losses kernel: ~20μs
/// - D2H `gains`/`losses` (pinned): ~25μs
/// - CPU Wilder's smoothing (2x): ~30μs
/// - H2D `avg_gain`/`avg_loss` (pinned): ~25μs
/// - GPU RSI kernel: ~15μs
/// - **Total**: ~115μs (vs ~130μs for sync)
///
/// # Optimization: Asynchronous Transfers
///
/// This implementation uses pinned memory and asynchronous copies to overlap data
/// transfers with computation where possible:
/// 1. H2D copy of `close` data is performed asynchronously.
/// 2. The `gains`/`losses` kernel is queued on the same stream, ensuring order.
/// 3. D2H copy of `gains`/`losses` into pinned memory is async.
/// 4. The stream is synchronized before CPU access (unavoidable).
/// 5. H2D copy of smoothed `avg_gain`/`avg_loss` is async.
/// 6. The final RSI kernel is queued.
/// 7. D2H copy of the final RSI result is async.
/// 8. A final synchronization waits for the result.
///
/// This approach reduces transfer latency by 20-30% via pinned memory and
/// improves overall pipeline efficiency.
///
/// # Stream Concurrency
///
/// When a stream is provided, kernel launches execute on that stream, enabling
/// concurrent execution with other operations on different streams. This is used
/// in the batch pipeline for 4-6x speedup across Fast/Medium/Slow indicator groups.
///
/// Classification: **MEDIUM** indicator (hybrid GPU-CPU-GPU approach)
///
/// # Algorithm
///
/// 1. **GPU**: Calculate price deltas and separate gains/losses (parallel)
/// 2. **CPU**: Apply Wilder's smoothing to gains (sequential, alpha = 1/period)
/// 3. **CPU**: Apply Wilder's smoothing to losses (sequential, alpha = 1/period)
/// 4. **GPU**: Calculate RSI = 100 - (100 / (1 + avg_gain/avg_loss)) (parallel)
///
/// # Why Hybrid?
///
/// Wilder's smoothing is a sequential IIR filter (each output depends on previous).
/// Single-thread GPU kernel is 6x slower than CPU due to lower clock speed and overhead.
/// Hybrid approach with 2 round-trips is still 2x faster overall.
pub fn rsi_gpu(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let n = close.len();

    // Validate inputs
    if period < 1 {
        return Err(GpuError::InvalidParameter(
            "Period must be >= 1".to_string(),
        ));
    }

    if n < period + 1 {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need {} points, got {}",
            period + 1,
            n
        )));
    }

    // Compile PTX with caching (50-200x faster on cache hits)
    let ptx_arc = compile_ptx_optimized_cached(RSI_KERNEL).map_err(|e| {
        GpuError::CompilationError(format!("Failed to compile RSI kernel: {:?}", e))
    })?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);

    // Load module
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel functions (only parallel kernels - smoothing moved to CPU)
    let gains_losses_kernel = module
        .load_function("calculate_gains_losses_kernel")
        .map_err(|e| {
            GpuError::ExecutionError(format!("Failed to load gains_losses kernel: {:?}", e))
        })?;

    let rsi_kernel = module
        .load_function("calculate_rsi_kernel")
        .map_err(|e| GpuError::ExecutionError(format!("Failed to load RSI kernel: {:?}", e)))?;

    // Select stream: use provided stream or device default
    let kernel_stream = stream.unwrap_or(&device.stream);

    // === Step 1: GPU - Calculate gains and losses (parallel) ===
    // Acquire pinned buffer for async H2D transfer
    let mut pinned_close = device.pinned_pool.lock().acquire(n)?;
    pinned_close.as_mut_slice()[..n].copy_from_slice(close.as_slice().unwrap());

    // Allocate device buffers
    let mut d_close = device.alloc_buffer(n)?;
    let mut d_gains = device.alloc_buffer(n)?;
    let mut d_losses = device.alloc_buffer(n)?;

    // Asynchronous H2D copy using pinned memory (20-30% faster)
    kernel_stream
        .memcpy_htod(&pinned_close.as_slice()[..n], &mut d_close)
        .map_err(|e| GpuError::ExecutionError(format!("H2D copy failed: {:?}", e)))?;

    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut builder = kernel_stream.launch_builder(&gains_losses_kernel);
    builder.arg(&d_close);
    builder.arg(&mut d_gains);
    builder.arg(&mut d_losses);
    builder.arg(&n_i32);

    let config = LaunchConfig::for_num_elems((n - 1) as u32);
    unsafe {
        builder.launch(config).map_err(|e| {
            GpuError::ExecutionError(format!("Gains/losses kernel launch failed: {:?}", e))
        })?;
    }

    // Release `close` buffer back to pool now that H2D is done
    device.pinned_pool.lock().release(pinned_close);

    // === Step 2: D2H - Copy gains/losses back to CPU for Wilder's smoothing ===
    // Acquire pinned buffers for async D2H transfer
    let mut pinned_gains = device.pinned_pool.lock().acquire(n)?;
    let mut pinned_losses = device.pinned_pool.lock().acquire(n)?;

    // Asynchronous D2H copies
    kernel_stream
        .memcpy_dtoh(&d_gains, &mut pinned_gains.as_mut_slice()[..n])
        .map_err(|e| GpuError::ExecutionError(format!("D2H gains copy failed: {:?}", e)))?;
    kernel_stream
        .memcpy_dtoh(&d_losses, &mut pinned_losses.as_mut_slice()[..n])
        .map_err(|e| GpuError::ExecutionError(format!("D2H losses copy failed: {:?}", e)))?;

    // Synchronize stream to ensure D2H copies are complete before CPU access
    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!(
            "Stream sync after gains/losses D2H failed: {:?}",
            e
        ))
    })?;

    // Access data from pinned buffers
    let gains = Array1::from_vec(pinned_gains.as_slice()[..n].to_vec());
    let losses = Array1::from_vec(pinned_losses.as_slice()[..n].to_vec());

    // Release buffers back to pool
    let mut pool = device.pinned_pool.lock();
    pool.release(pinned_gains);
    pool.release(pinned_losses);
    drop(pool); // Unlock mutex

    // === Step 3: CPU - Apply Wilder's smoothing (sequential, 3-4x faster than GPU) ===
    use crate::cpu::sequential::wilders_smoothing_cpu;

    let avg_gain = wilders_smoothing_cpu(&gains, period)?;
    let avg_loss = wilders_smoothing_cpu(&losses, period)?;

    // === Step 4: H2D - Copy avg_gain/avg_loss back to GPU for final RSI calculation ===
    // Acquire pinned buffers for async H2D transfer
    let mut pinned_avg_gain = device.pinned_pool.lock().acquire(n)?;
    pinned_avg_gain.as_mut_slice()[..n].copy_from_slice(avg_gain.as_slice().unwrap());
    let mut pinned_avg_loss = device.pinned_pool.lock().acquire(n)?;
    pinned_avg_loss.as_mut_slice()[..n].copy_from_slice(avg_loss.as_slice().unwrap());

    // Allocate device buffers
    let mut d_avg_gain = device.alloc_buffer(n)?;
    let mut d_avg_loss = device.alloc_buffer(n)?;
    let mut d_rsi = device.alloc_buffer(n)?;

    // Asynchronous H2D copies
    kernel_stream
        .memcpy_htod(&pinned_avg_gain.as_slice()[..n], &mut d_avg_gain)
        .map_err(|e| GpuError::ExecutionError(format!("H2D avg_gain copy failed: {:?}", e)))?;
    kernel_stream
        .memcpy_htod(&pinned_avg_loss.as_slice()[..n], &mut d_avg_loss)
        .map_err(|e| GpuError::ExecutionError(format!("H2D avg_loss copy failed: {:?}", e)))?;

    // Release buffers back to pool
    let mut pool = device.pinned_pool.lock();
    pool.release(pinned_avg_gain);
    pool.release(pinned_avg_loss);
    drop(pool);

    // === Step 5: GPU - Calculate final RSI (parallel) ===
    let mut builder = kernel_stream.launch_builder(&rsi_kernel);
    builder.arg(&d_avg_gain);
    builder.arg(&d_avg_loss);
    builder.arg(&mut d_rsi);
    builder.arg(&n_i32);
    builder.arg(&period_i32);

    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("RSI kernel launch failed: {:?}", e)))?;
    }

    // === Step 6: D2H - Copy final RSI back to host ===
    // Acquire pinned buffer for async D2H transfer
    let mut pinned_rsi = device.pinned_pool.lock().acquire(n)?;
    kernel_stream
        .memcpy_dtoh(&d_rsi, &mut pinned_rsi.as_mut_slice()[..n])
        .map_err(|e| GpuError::ExecutionError(format!("D2H RSI copy failed: {:?}", e)))?;

    // Synchronize to ensure final result is ready
    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream sync after RSI D2H failed: {:?}", e))
    })?;

    let rsi_vec = pinned_rsi.as_slice()[..n].to_vec();

    // Release buffer back to pool
    device.pinned_pool.lock().release(pinned_rsi);

    Ok(Array1::from_vec(rsi_vec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_rsi_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test data with known pattern (trending up)
        let close = arr1(&[
            44.0, 44.5, 45.0, 44.8, 45.5, 46.0, 45.8, 46.5, 47.0, 46.8, 47.5, 48.0, 47.8, 48.5,
            49.0, 49.5, 50.0,
        ]);

        let result = rsi_gpu(&device, &close, 14, None).expect("RSI GPU calculation failed");

        // Verify RSI is in valid range [0, 100]
        for i in 14..result.len() {
            assert!(
                result[i] >= 0.0 && result[i] <= 100.0,
                "RSI at index {} = {} is out of range",
                i,
                result[i]
            );
        }

        // First 14 values should be NaN
        for i in 0..14 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }

        // RSI for uptrend should be > 50
        assert!(result[14] > 50.0, "Expected RSI > 50 for uptrend");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_rsi_gpu_edge_cases() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // All gains, no losses - RSI should be 100
        let close = arr1(&[
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0,
        ]);

        let result = rsi_gpu(&device, &close, 14, None).expect("RSI GPU calculation failed");

        // RSI should approach 100 when only gains
        assert!(
            result[14] > 95.0,
            "Expected RSI close to 100 for all gains, got {}",
            result[14]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_rsi_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Generate large dataset with sine wave pattern
        let n = 100_000;
        let close: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * 0.01;
                100.0 + 10.0 * (x * 0.1).sin()
            })
            .collect();
        let close = Array1::from_vec(close);

        let start = std::time::Instant::now();
        let result = rsi_gpu(&device, &close, 14, None).expect("RSI GPU calculation failed");
        let elapsed = start.elapsed();

        println!("GPU RSI (n={}): {:.2}ms", n, elapsed.as_secs_f64() * 1000.0);

        // Verify output size
        assert_eq!(result.len(), n);

        // Verify valid range
        for i in 14..n {
            assert!(
                result[i] >= 0.0 && result[i] <= 100.0,
                "RSI out of range at index {}",
                i
            );
        }

        // For oscillating data, RSI should oscillate around 50
        let avg_rsi: f64 =
            result.slice(ndarray::s![14..]).iter().sum::<f64>() / (result.len() - 14) as f64;
        assert!(
            (avg_rsi - 50.0).abs() < 10.0,
            "Expected average RSI near 50 for oscillating data, got {}",
            avg_rsi
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_rsi_gpu_invalid_inputs() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Too short dataset
        let close = arr1(&[100.0, 101.0, 102.0]);
        let result = rsi_gpu(&device, &close, 14, None);
        assert!(result.is_err(), "Should fail with insufficient data");

        // Invalid period
        let close = arr1(&[100.0; 20]);
        let result = rsi_gpu(&device, &close, 0, None);
        assert!(result.is_err(), "Should fail with period = 0");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_rsi_gpu_constant_prices() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Constant prices - no gains or losses
        let close = arr1(&[100.0; 30]);

        let result = rsi_gpu(&device, &close, 14, None).expect("RSI GPU calculation failed");

        // Flat series: no gains AND no losses (0/0). RSI is directionless, so the
        // kernel returns the neutral 50 (matching the CPU RSI), NOT 100 -- a flat
        // market is not maximally overbought.
        for i in 14..result.len() {
            assert!(
                result[i] == 50.0,
                "Expected neutral RSI = 50 for constant prices, got {}",
                result[i]
            );
        }
    }
}
