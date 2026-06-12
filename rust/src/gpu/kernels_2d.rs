//! 2D CUDA Kernels for Batch Processing and Multi-Indicator Fusion
//!
//! Production-ready 2D kernel implementations for:
//! - Batch processing (multiple assets in parallel)
//! - Multi-indicator fusion (calculate multiple indicators in one kernel)
//! - Cooperative shared memory loading for rolling windows
//!
//! # Performance Targets
//!
//! - Batch processing: +35-45% speedup over sequential (N_assets >= 10)
//! - Multi-indicator fusion: +30-40% speedup (N_indicators >= 3)
//! - Memory coalescing: >90% efficiency maintained
//! - GPU utilization: >75% during batch execution
//! - Async pinned memory: +11% speedup over sync transfers (batch operations benefit most)

use super::device::{GpuDevice, GpuError};
use super::compile::compile_ptx_optimized_cached;
use crate::cpu::sequential::wilders_smoothing_cpu;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use ndarray::Array1;
use rayon::prelude::*;
use std::sync::Arc;

/// CUDA kernel source for 2D batch processing
///
/// Architecture:
/// - blockIdx.x = asset index
/// - blockIdx.y = candle chunk index
/// - threadIdx.x = candle within chunk
/// - threadIdx.y = unused (always 1)
const BATCH_2D_KERNELS: &str = r#"
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)

// ============================================================================
// 2D BATCH PROCESSING KERNELS
// ============================================================================

// Kernel 1: RSI Batch (2D) - Process multiple assets in parallel
// Grid: (n_assets, (n_candles + 255) / 256, 1)
// Block: (256, 1, 1)
extern "C" __global__ void rsi_batch_2d_kernel(
    const double* __restrict__ close_batch,  // [n_assets, n_candles] row-major
    double* __restrict__ gains_batch,        // [n_assets, n_candles]
    double* __restrict__ losses_batch,       // [n_assets, n_candles]
    int n_assets,
    int n_candles
) {
    int asset_idx = blockIdx.x;
    int chunk_idx = blockIdx.y;
    int local_idx = threadIdx.x;

    int candle_idx = chunk_idx * blockDim.x + local_idx;

    if (asset_idx < n_assets && candle_idx < n_candles - 1) {
        // Global index: [asset_idx * n_candles + candle_idx]
        int idx = asset_idx * n_candles + candle_idx;
        int next_idx = idx + 1;

        // Calculate delta
        double delta = close_batch[next_idx] - close_batch[idx];

        // Separate gains and losses
        gains_batch[next_idx] = fmax(delta, 0.0);
        losses_batch[next_idx] = fmax(-delta, 0.0);
    }

    // First element is always 0 for gains/losses
    if (candle_idx == 0 && asset_idx < n_assets) {
        int idx = asset_idx * n_candles;
        gains_batch[idx] = 0.0;
        losses_batch[idx] = 0.0;
    }
}

// Kernel 2: RSI Batch Final Calculation (2D)
// After CPU Wilder's smoothing, compute final RSI values
extern "C" __global__ void rsi_batch_final_2d_kernel(
    const double* __restrict__ avg_gain_batch,  // [n_assets, n_candles]
    const double* __restrict__ avg_loss_batch,  // [n_assets, n_candles]
    double* __restrict__ rsi_batch,             // [n_assets, n_candles]
    int n_assets,
    int n_candles,
    int period
) {
    int asset_idx = blockIdx.x;
    int chunk_idx = blockIdx.y;
    int local_idx = threadIdx.x;

    int candle_idx = chunk_idx * blockDim.x + local_idx;

    if (asset_idx < n_assets && candle_idx < n_candles) {
        int idx = asset_idx * n_candles + candle_idx;

        if (candle_idx < period) {
            rsi_batch[idx] = CUDART_NAN;
        } else {
            double gain = avg_gain_batch[idx];
            double loss = avg_loss_batch[idx];

            if (loss < 1e-10) {
                rsi_batch[idx] = 100.0;
            } else {
                double rs = gain / loss;
                rsi_batch[idx] = 100.0 - (100.0 / (1.0 + rs));
            }
        }
    }
}

// Kernel 3: SMA Batch (2D) - Process multiple assets in parallel
extern "C" __global__ void sma_batch_2d_kernel(
    const double* __restrict__ close_batch,  // [n_assets, n_candles]
    double* __restrict__ sma_batch,          // [n_assets, n_candles]
    int n_assets,
    int n_candles,
    int period
) {
    int asset_idx = blockIdx.x;
    int chunk_idx = blockIdx.y;
    int local_idx = threadIdx.x;

    int candle_idx = chunk_idx * blockDim.x + local_idx;

    if (asset_idx < n_assets && candle_idx < n_candles) {
        int base_idx = asset_idx * n_candles;
        int idx = base_idx + candle_idx;

        if (candle_idx >= period - 1) {
            double sum = 0.0;

            // Sum last 'period' values
            #pragma unroll 4
            for (int j = 0; j < period; j++) {
                sum += close_batch[base_idx + candle_idx - j];
            }

            sma_batch[idx] = sum / (double)period;
        } else {
            sma_batch[idx] = CUDART_NAN;
        }
    }
}

// Kernel 4: Stochastic Batch (2D) - Process multiple assets
extern "C" __global__ void stochastic_batch_2d_kernel(
    const double* __restrict__ high_batch,   // [n_assets, n_candles]
    const double* __restrict__ low_batch,    // [n_assets, n_candles]
    const double* __restrict__ close_batch,  // [n_assets, n_candles]
    double* __restrict__ k_batch,            // [n_assets, n_candles]
    int n_assets,
    int n_candles,
    int k_period
) {
    int asset_idx = blockIdx.x;
    int chunk_idx = blockIdx.y;
    int local_idx = threadIdx.x;

    int candle_idx = chunk_idx * blockDim.x + local_idx;

    if (asset_idx < n_assets && candle_idx < n_candles) {
        int base_idx = asset_idx * n_candles;
        int idx = base_idx + candle_idx;

        if (candle_idx >= k_period - 1) {
            double highest_high = -CUDART_INF;
            double lowest_low = CUDART_INF;

            // Find highest high and lowest low in window
            for (int i = 0; i < k_period; i++) {
                int window_idx = base_idx + candle_idx - i;
                highest_high = fmax(highest_high, high_batch[window_idx]);
                lowest_low = fmin(lowest_low, low_batch[window_idx]);
            }

            // Calculate %K
            double range = highest_high - lowest_low;
            if (range > 1e-10) {
                k_batch[idx] = 100.0 * (close_batch[idx] - lowest_low) / range;
            } else {
                k_batch[idx] = 50.0;
            }
        } else {
            k_batch[idx] = CUDART_NAN;
        }
    }
}

// ============================================================================
// MULTI-INDICATOR FUSION KERNELS
// ============================================================================

// Kernel 5: Momentum Fusion (RSI + Stochastic %K + Williams %R)
// Grid: ((n_candles + 255) / 256, 1, 1)
// Block: (256, 1, 1) - each thread emits all three indicator outputs
//
// Stochastic %K and Williams %R are affine transforms of the SAME window
// extrema (highest high / lowest low over `period` bars), so the rolling
// max/min scan runs once per element and both outputs derive from the same
// registers. The previous blockDim=(256, 3) layout duplicated that scan in
// two of the three thread groups and left the RSI group idle during it.
extern "C" __global__ void momentum_fusion_2d_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    double* __restrict__ rsi_out,
    double* __restrict__ stoch_k_out,
    double* __restrict__ williams_out,
    const double* __restrict__ avg_gain,    // Pre-computed from CPU
    const double* __restrict__ avg_loss,    // Pre-computed from CPU
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // --- RSI from pre-smoothed Wilder averages ---
    if (idx < period) {
        rsi_out[idx] = CUDART_NAN;
    } else {
        double gain = avg_gain[idx];
        double loss = avg_loss[idx];

        if (loss < 1e-10) {
            rsi_out[idx] = 100.0;
        } else {
            double rs = gain / loss;
            rsi_out[idx] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    // --- Window extrema computed once; %K and %R derived from the same registers ---
    if (idx >= period - 1) {
        double highest_high = -CUDART_INF;
        double lowest_low = CUDART_INF;

        for (int i = 0; i < period; i++) {
            int window_idx = idx - i;
            highest_high = fmax(highest_high, high[window_idx]);
            lowest_low = fmin(lowest_low, low[window_idx]);
        }

        double range = highest_high - lowest_low;
        if (range > 1e-10) {
            double c = close[idx];
            stoch_k_out[idx] = 100.0 * (c - lowest_low) / range;
            williams_out[idx] = -100.0 * (highest_high - c) / range;
        } else {
            stoch_k_out[idx] = 50.0;
            williams_out[idx] = -50.0;
        }
    } else {
        stoch_k_out[idx] = CUDART_NAN;
        williams_out[idx] = CUDART_NAN;
    }
}

// Kernel 6: Volatility Fusion (ATR + Bollinger Bands)
// Computes True Range and Bollinger simultaneously
extern "C" __global__ void volatility_fusion_2d_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    double* __restrict__ true_range_out,
    double* __restrict__ bollinger_middle_out,
    double* __restrict__ bollinger_upper_out,
    double* __restrict__ bollinger_lower_out,
    int n,
    int period,
    double num_std
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int indicator_type = threadIdx.y;  // 0=ATR (True Range), 1=Bollinger

    if (idx >= n) return;

    // True Range calculation (indicator_type == 0)
    if (indicator_type == 0) {
        if (idx == 0) {
            true_range_out[idx] = high[idx] - low[idx];
        } else {
            double hl = high[idx] - low[idx];
            double hc = fabs(high[idx] - close[idx - 1]);
            double lc = fabs(low[idx] - close[idx - 1]);
            true_range_out[idx] = fmax(fmax(hl, hc), lc);
        }
    }

    // Bollinger Bands calculation (indicator_type == 1)
    else if (indicator_type == 1) {
        if (idx >= period - 1) {
            double sum = 0.0;

            // Calculate SMA
            for (int j = 0; j < period; j++) {
                sum += close[idx - j];
            }

            double sma = sum / (double)period;
            bollinger_middle_out[idx] = sma;

            // Calculate standard deviation
            double variance = 0.0;
            for (int j = 0; j < period; j++) {
                double diff = close[idx - j] - sma;
                variance += diff * diff;
            }

            double std_dev = sqrt(variance / (double)period);

            bollinger_upper_out[idx] = sma + num_std * std_dev;
            bollinger_lower_out[idx] = sma - num_std * std_dev;
        } else {
            bollinger_middle_out[idx] = CUDART_NAN;
            bollinger_upper_out[idx] = CUDART_NAN;
            bollinger_lower_out[idx] = CUDART_NAN;
        }
    }
}
"#;

/// CPU Wilder's smoothing stage for the 2D batch RSI.
///
/// `gains`/`losses` are laid out as [n_assets, n_candles]; each asset is an
/// independent series, so assets are smoothed in parallel across all cores
/// with rayon. Numerical semantics are exactly
/// `crate::cpu::sequential::wilders_smoothing_cpu` per asset.
fn wilder_smooth_batch(
    gains: &[f64],
    losses: &[f64],
    n_assets: usize,
    n_candles: usize,
    period: usize,
) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
    // par_chunks_mut panics on a zero chunk size
    if n_candles == 0 {
        return Err(GpuError::InvalidParameter(
            "n_candles must be > 0".to_string(),
        ));
    }

    let batch_size = n_assets * n_candles;

    let mut avg_gain_batch = vec![0.0f64; batch_size];
    let mut avg_loss_batch = vec![0.0f64; batch_size];

    // Chunk index == asset_idx ([n_assets, n_candles] row-major layout)
    avg_gain_batch
        .par_chunks_mut(n_candles)
        .zip(avg_loss_batch.par_chunks_mut(n_candles))
        .enumerate()
        .try_for_each(|(asset_idx, (gain_out, loss_out))| -> Result<(), GpuError> {
            let start = asset_idx * n_candles;
            let end = start + n_candles;

            let gains_arr = Array1::from_vec(gains[start..end].to_vec());
            let losses_arr = Array1::from_vec(losses[start..end].to_vec());

            let avg_gain = wilders_smoothing_cpu(&gains_arr, period)?;
            let avg_loss = wilders_smoothing_cpu(&losses_arr, period)?;

            gain_out.copy_from_slice(avg_gain.as_slice().unwrap());
            loss_out.copy_from_slice(avg_loss.as_slice().unwrap());
            Ok(())
        })?;

    Ok((avg_gain_batch, avg_loss_batch))
}

/// 2D Batch RSI calculation (multiple assets in parallel)
///
/// # Arguments
///
/// * `device` - GPU device
/// * `close_batch` - Flattened close prices [asset_0[candles], asset_1[candles], ...]
/// * `n_assets` - Number of assets
/// * `n_candles` - Number of candles per asset
/// * `period` - RSI period
///
/// # Returns
///
/// Flattened RSI values [asset_0[rsi], asset_1[rsi], ...] same layout as input
///
/// # Performance
///
/// Expected speedup: +35-45% over sequential (n_assets >= 10)
/// Async pinned memory: +11% additional speedup in batch operations
///
/// # Example
///
/// ```rust,ignore
/// let device = GpuDevice::new()?;
///
/// // Flatten 10 assets × 100K candles
/// let close_batch: Vec<f64> = assets.iter()
///     .flat_map(|a| a.close.iter().copied())
///     .collect();
///
/// let rsi_batch = rsi_batch_2d_gpu(&device, &close_batch, 10, 100_000, 14)?;
///
/// // Reshape back to Vec<Array1<f64>>
/// let results: Vec<Array1<f64>> = (0..10)
///     .map(|i| Array1::from_vec(rsi_batch[i*100_000..(i+1)*100_000].to_vec()))
///     .collect();
/// ```
pub fn rsi_batch_2d_gpu(
    device: &GpuDevice,
    close_batch: &[f64],
    n_assets: usize,
    n_candles: usize,
    period: usize,
) -> Result<Vec<f64>, GpuError> {
    // Validate inputs
    if close_batch.len() != n_assets * n_candles {
        return Err(GpuError::InvalidParameter(format!(
            "close_batch length mismatch: expected {}, got {}",
            n_assets * n_candles,
            close_batch.len()
        )));
    }

    if period < 1 {
        return Err(GpuError::InvalidParameter(
            "Period must be >= 1".to_string(),
        ));
    }

    if n_candles < period + 1 {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need {} candles, got {}",
            period + 1,
            n_candles
        )));
    }

    // Compile PTX
    let ptx_arc = compile_ptx_optimized_cached(BATCH_2D_KERNELS)
        .map_err(|e| GpuError::CompilationError(format!("Failed to compile 2D kernels: {:?}", e)))?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);

    let module = device.context().load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    let gains_losses_kernel = module.load_function("rsi_batch_2d_kernel")
        .map_err(|e| GpuError::ExecutionError(format!("Failed to load kernel: {:?}", e)))?;

    let rsi_final_kernel = module.load_function("rsi_batch_final_2d_kernel")
        .map_err(|e| GpuError::ExecutionError(format!("Failed to load kernel: {:?}", e)))?;

    // === Step 1: GPU - Calculate gains/losses (2D parallel) ===
    // Async H2D transfer for close_batch
    let batch_size = n_assets * n_candles;
    let mut pinned_close = device.pinned_pool.lock().acquire(batch_size)?;
    pinned_close.as_mut_slice()[..batch_size].copy_from_slice(close_batch);

    let mut d_close = device.alloc_buffer(batch_size)?;
    device.stream.memcpy_htod(&pinned_close.as_slice()[..batch_size], &mut d_close)?;
    device.pinned_pool.lock().release(pinned_close);

    let mut d_gains = device.alloc_buffer(batch_size)?;
    let mut d_losses = device.alloc_buffer(batch_size)?;

    let threads_per_block = 256;
    let blocks_x = n_assets as u32;
    let blocks_y = n_candles.div_ceil(256) as u32;

    let config = LaunchConfig {
        grid_dim: (blocks_x, blocks_y, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    // Scalar kernel args must outlive the launch builder (PushKernelArg
    // borrows until launch), so bind them to locals instead of temporaries.
    let n_assets_i32 = n_assets as i32;
    let n_candles_i32 = n_candles as i32;
    let period_i32 = period as i32;

    let mut builder = device.stream.launch_builder(&gains_losses_kernel);
    builder.arg(&d_close);
    builder.arg(&mut d_gains);
    builder.arg(&mut d_losses);
    builder.arg(&n_assets_i32);
    builder.arg(&n_candles_i32);

    unsafe {
        builder.launch(config).map_err(|e| {
            GpuError::ExecutionError(format!("Gains/losses kernel launch failed: {:?}", e))
        })?;
    }

    device.stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream sync after gains/losses failed: {:?}", e))
    })?;

    // === Step 2: D2H - Copy gains/losses for CPU Wilder's smoothing ===
    // Async D2H transfer for gains
    let mut pinned_gains = device.pinned_pool.lock().acquire(batch_size)?;
    device.stream.memcpy_dtoh(&d_gains, &mut pinned_gains.as_mut_slice()[..batch_size])?;

    // Async D2H transfer for losses
    let mut pinned_losses = device.pinned_pool.lock().acquire(batch_size)?;
    device.stream.memcpy_dtoh(&d_losses, &mut pinned_losses.as_mut_slice()[..batch_size])?;

    // Synchronize before CPU access
    device.stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream sync after D2H failed: {:?}", e))
    })?;

    let gains_vec = pinned_gains.as_slice()[..batch_size].to_vec();
    let losses_vec = pinned_losses.as_slice()[..batch_size].to_vec();

    device.pinned_pool.lock().release(pinned_gains);
    device.pinned_pool.lock().release(pinned_losses);

    // === Step 3: CPU - Wilder's smoothing per asset (rayon-parallel across assets) ===
    let (avg_gain_batch, avg_loss_batch) =
        wilder_smooth_batch(&gains_vec, &losses_vec, n_assets, n_candles, period)?;

    // === Step 4: H2D - Copy avg_gain/avg_loss back to GPU ===
    // Async H2D transfer for avg_gain
    let mut pinned_avg_gain = device.pinned_pool.lock().acquire(batch_size)?;
    pinned_avg_gain.as_mut_slice()[..batch_size].copy_from_slice(&avg_gain_batch);

    let mut d_avg_gain = device.alloc_buffer(batch_size)?;
    device.stream.memcpy_htod(&pinned_avg_gain.as_slice()[..batch_size], &mut d_avg_gain)?;
    device.pinned_pool.lock().release(pinned_avg_gain);

    // Async H2D transfer for avg_loss
    let mut pinned_avg_loss = device.pinned_pool.lock().acquire(batch_size)?;
    pinned_avg_loss.as_mut_slice()[..batch_size].copy_from_slice(&avg_loss_batch);

    let mut d_avg_loss = device.alloc_buffer(batch_size)?;
    device.stream.memcpy_htod(&pinned_avg_loss.as_slice()[..batch_size], &mut d_avg_loss)?;
    device.pinned_pool.lock().release(pinned_avg_loss);

    let mut d_rsi = device.alloc_buffer(batch_size)?;

    // === Step 5: GPU - Calculate final RSI (2D parallel) ===
    let mut builder = device.stream.launch_builder(&rsi_final_kernel);
    builder.arg(&d_avg_gain);
    builder.arg(&d_avg_loss);
    builder.arg(&mut d_rsi);
    builder.arg(&n_assets_i32);
    builder.arg(&n_candles_i32);
    builder.arg(&period_i32);

    unsafe {
        builder.launch(config).map_err(|e| {
            GpuError::ExecutionError(format!("RSI final kernel launch failed: {:?}", e))
        })?;
    }

    device.stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream sync after RSI final failed: {:?}", e))
    })?;

    // === Step 6: D2H - Copy final RSI ===
    // Async D2H transfer for final RSI
    let mut pinned_rsi = device.pinned_pool.lock().acquire(batch_size)?;
    device.stream.memcpy_dtoh(&d_rsi, &mut pinned_rsi.as_mut_slice()[..batch_size])?;

    // Synchronize before returning
    device.stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream sync after final RSI D2H failed: {:?}", e))
    })?;

    let result = pinned_rsi.as_slice()[..batch_size].to_vec();
    device.pinned_pool.lock().release(pinned_rsi);

    Ok(result)
}

/// 2D Batch SMA calculation (multiple assets in parallel)
///
/// Expected speedup: +35-45% over sequential (n_assets >= 10)
/// Async pinned memory: +11% additional speedup
pub fn sma_batch_2d_gpu(
    device: &GpuDevice,
    close_batch: &[f64],
    n_assets: usize,
    n_candles: usize,
    period: usize,
) -> Result<Vec<f64>, GpuError> {
    if close_batch.len() != n_assets * n_candles {
        return Err(GpuError::InvalidParameter("close_batch length mismatch".to_string()));
    }

    let ptx_arc = compile_ptx_optimized_cached(BATCH_2D_KERNELS)?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);
    let module = device.context().load_module(ptx)?;
    let kernel = module.load_function("sma_batch_2d_kernel")?;

    // Async H2D transfer for close_batch
    let batch_size = n_assets * n_candles;
    let mut pinned_close = device.pinned_pool.lock().acquire(batch_size)?;
    pinned_close.as_mut_slice()[..batch_size].copy_from_slice(close_batch);

    let mut d_close = device.alloc_buffer(batch_size)?;
    device.stream.memcpy_htod(&pinned_close.as_slice()[..batch_size], &mut d_close)?;
    device.pinned_pool.lock().release(pinned_close);

    let mut d_sma = device.alloc_buffer(batch_size)?;

    let config = LaunchConfig {
        grid_dim: (n_assets as u32, n_candles.div_ceil(256) as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // Scalar kernel args must outlive the launch builder (PushKernelArg
    // borrows until launch), so bind them to locals instead of temporaries.
    let n_assets_i32 = n_assets as i32;
    let n_candles_i32 = n_candles as i32;
    let period_i32 = period as i32;

    let mut builder = device.stream.launch_builder(&kernel);
    builder.arg(&d_close);
    builder.arg(&mut d_sma);
    builder.arg(&n_assets_i32);
    builder.arg(&n_candles_i32);
    builder.arg(&period_i32);

    unsafe { builder.launch(config)? };
    device.stream.synchronize()?;

    // Async D2H transfer for SMA
    let mut pinned_sma = device.pinned_pool.lock().acquire(batch_size)?;
    device.stream.memcpy_dtoh(&d_sma, &mut pinned_sma.as_mut_slice()[..batch_size])?;

    // Synchronize before returning
    device.stream.synchronize()?;

    let result = pinned_sma.as_slice()[..batch_size].to_vec();
    device.pinned_pool.lock().release(pinned_sma);

    Ok(result)
}

/// Multi-Indicator Fusion: Calculate RSI + Stochastic %K + Williams %R in single kernel
///
/// Expected speedup: +30-40% over 3 separate calls
/// Async pinned memory: +11% additional speedup for multi-indicator fusion
///
/// # Arguments
///
/// * `device` - GPU device
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `period` - Period for all indicators
///
/// # Returns
///
/// Tuple of (RSI, Stochastic %K, Williams %R)
#[allow(clippy::type_complexity)] // (RSI, Stochastic %K, Williams %R) tuple is the public API
pub fn momentum_fusion_2d_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    period: usize,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), GpuError> {
    let n = close.len();

    if high.len() != n || low.len() != n {
        return Err(GpuError::InvalidParameter("Array length mismatch".to_string()));
    }

    // Step 1: CPU - Calculate gains/losses and Wilder's smoothing for RSI
    // (Sequential bottleneck, cannot be fused)
    let mut gains = Array1::zeros(n);
    let mut losses = Array1::zeros(n);

    for i in 1..n {
        let delta = close[i] - close[i - 1];
        gains[i] = delta.max(0.0);
        losses[i] = (-delta).max(0.0);
    }

    let avg_gain = wilders_smoothing_cpu(&gains, period)?;
    let avg_loss = wilders_smoothing_cpu(&losses, period)?;

    // Step 2: GPU - Fused calculation of RSI, Stochastic, Williams
    let ptx_arc = compile_ptx_optimized_cached(BATCH_2D_KERNELS)?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);
    let module = device.context().load_module(ptx)?;
    let kernel = module.load_function("momentum_fusion_2d_kernel")?;

    // Async H2D transfers for all input arrays
    let mut pinned_high = device.pinned_pool.lock().acquire(n)?;
    pinned_high.as_mut_slice()[..n].copy_from_slice(high.as_slice().unwrap());
    let mut d_high = device.alloc_buffer(n)?;
    device.stream.memcpy_htod(&pinned_high.as_slice()[..n], &mut d_high)?;
    device.pinned_pool.lock().release(pinned_high);

    let mut pinned_low = device.pinned_pool.lock().acquire(n)?;
    pinned_low.as_mut_slice()[..n].copy_from_slice(low.as_slice().unwrap());
    let mut d_low = device.alloc_buffer(n)?;
    device.stream.memcpy_htod(&pinned_low.as_slice()[..n], &mut d_low)?;
    device.pinned_pool.lock().release(pinned_low);

    let mut pinned_close = device.pinned_pool.lock().acquire(n)?;
    pinned_close.as_mut_slice()[..n].copy_from_slice(close.as_slice().unwrap());
    let mut d_close = device.alloc_buffer(n)?;
    device.stream.memcpy_htod(&pinned_close.as_slice()[..n], &mut d_close)?;
    device.pinned_pool.lock().release(pinned_close);

    let mut pinned_avg_gain = device.pinned_pool.lock().acquire(n)?;
    pinned_avg_gain.as_mut_slice()[..n].copy_from_slice(avg_gain.as_slice().unwrap());
    let mut d_avg_gain = device.alloc_buffer(n)?;
    device.stream.memcpy_htod(&pinned_avg_gain.as_slice()[..n], &mut d_avg_gain)?;
    device.pinned_pool.lock().release(pinned_avg_gain);

    let mut pinned_avg_loss = device.pinned_pool.lock().acquire(n)?;
    pinned_avg_loss.as_mut_slice()[..n].copy_from_slice(avg_loss.as_slice().unwrap());
    let mut d_avg_loss = device.alloc_buffer(n)?;
    device.stream.memcpy_htod(&pinned_avg_loss.as_slice()[..n], &mut d_avg_loss)?;
    device.pinned_pool.lock().release(pinned_avg_loss);

    let mut d_rsi = device.alloc_buffer(n)?;
    let mut d_stoch_k = device.alloc_buffer(n)?;
    let mut d_williams = device.alloc_buffer(n)?;

    // Flattened 1D launch: each thread emits RSI + Stochastic %K + Williams %R
    let config = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // Scalar kernel args must outlive the launch builder (PushKernelArg
    // borrows until launch), so bind them to locals instead of temporaries.
    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut builder = device.stream.launch_builder(&kernel);
    builder.arg(&d_high);
    builder.arg(&d_low);
    builder.arg(&d_close);
    builder.arg(&mut d_rsi);
    builder.arg(&mut d_stoch_k);
    builder.arg(&mut d_williams);
    builder.arg(&d_avg_gain);
    builder.arg(&d_avg_loss);
    builder.arg(&n_i32);
    builder.arg(&period_i32);

    unsafe { builder.launch(config)? };
    device.stream.synchronize()?;

    // Async D2H transfers for all output arrays
    let mut pinned_rsi = device.pinned_pool.lock().acquire(n)?;
    device.stream.memcpy_dtoh(&d_rsi, &mut pinned_rsi.as_mut_slice()[..n])?;

    let mut pinned_stoch = device.pinned_pool.lock().acquire(n)?;
    device.stream.memcpy_dtoh(&d_stoch_k, &mut pinned_stoch.as_mut_slice()[..n])?;

    let mut pinned_williams = device.pinned_pool.lock().acquire(n)?;
    device.stream.memcpy_dtoh(&d_williams, &mut pinned_williams.as_mut_slice()[..n])?;

    // Synchronize before CPU access
    device.stream.synchronize()?;

    let rsi_vec = pinned_rsi.as_slice()[..n].to_vec();
    let stoch_vec = pinned_stoch.as_slice()[..n].to_vec();
    let williams_vec = pinned_williams.as_slice()[..n].to_vec();

    device.pinned_pool.lock().release(pinned_rsi);
    device.pinned_pool.lock().release(pinned_stoch);
    device.pinned_pool.lock().release(pinned_williams);

    Ok((
        Array1::from_vec(rsi_vec),
        Array1::from_vec(stoch_vec),
        Array1::from_vec(williams_vec),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn generate_test_asset(n: usize, seed: f64) -> Vec<f64> {
        (0..n)
            .map(|i| 100.0 + seed + (i as f64 * 0.01) + ((i as f64 * 0.1).sin() * 2.0))
            .collect()
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_rsi_batch_2d_correctness() {
        use super::super::rsi::rsi_gpu;

        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let n_assets = 5;
        let n_candles = 1000;
        let period = 14;

        // Generate test data
        let assets: Vec<Vec<f64>> = (0..n_assets)
            .map(|i| generate_test_asset(n_candles, i as f64 * 10.0))
            .collect();

        // Compute with 1D (sequential)
        let results_1d: Vec<Array1<f64>> = assets
            .iter()
            .map(|asset_data| {
                let close = Array1::from_vec(asset_data.clone());
                rsi_gpu(&device, &close, period, None).unwrap()
            })
            .collect();

        // Compute with 2D (batch)
        let close_batch: Vec<f64> = assets.iter().flatten().copied().collect();
        let rsi_batch = rsi_batch_2d_gpu(&device, &close_batch, n_assets, n_candles, period)
            .expect("2D batch RSI failed");

        // Reshape 2D results
        let results_2d: Vec<Array1<f64>> = (0..n_assets)
            .map(|i| {
                let start = i * n_candles;
                let end = start + n_candles;
                Array1::from_vec(rsi_batch[start..end].to_vec())
            })
            .collect();

        // Compare outputs
        for (asset_idx, (r1d, r2d)) in results_1d.iter().zip(results_2d.iter()).enumerate() {
            for (i, (&v1, &v2)) in r1d.iter().zip(r2d.iter()).enumerate() {
                if v1.is_nan() {
                    assert!(v2.is_nan(), "Asset {}, idx {}: Expected NaN", asset_idx, i);
                } else {
                    assert!(
                        (v1 - v2).abs() < 1e-9,
                        "Asset {}, idx {}: {:.15} vs {:.15} (diff: {:.2e})",
                        asset_idx,
                        i,
                        v1,
                        v2,
                        (v1 - v2).abs()
                    );
                }
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_batch_2d_correctness() {
        use super::super::sma::sma_gpu;

        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let n_assets = 3;
        let n_candles = 500;
        let period = 20;

        let assets: Vec<Vec<f64>> = (0..n_assets)
            .map(|i| generate_test_asset(n_candles, i as f64 * 5.0))
            .collect();

        // 1D sequential
        let results_1d: Vec<Array1<f64>> = assets
            .iter()
            .map(|asset_data| {
                let close = Array1::from_vec(asset_data.clone());
                sma_gpu(&device, &close, period, None).unwrap()
            })
            .collect();

        // 2D batch
        let close_batch: Vec<f64> = assets.iter().flatten().copied().collect();
        let sma_batch = sma_batch_2d_gpu(&device, &close_batch, n_assets, n_candles, period).unwrap();

        // Compare
        for asset_idx in 0..n_assets {
            for i in 0..n_candles {
                let v1 = results_1d[asset_idx][i];
                let v2 = sma_batch[asset_idx * n_candles + i];

                if v1.is_nan() {
                    assert!(v2.is_nan());
                } else {
                    assert!((v1 - v2).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_momentum_fusion_2d() {
        use super::super::{rsi::rsi_gpu, stochastic::stochastic_gpu, williams_r::williams_r_gpu};

        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let n = 1000;
        let period = 14;

        let high: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.02) + 2.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.02) - 2.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.02)).collect();

        let high_arr = Array1::from_vec(high);
        let low_arr = Array1::from_vec(low);
        let close_arr = Array1::from_vec(close);

        // Individual calls
        let rsi_ind = rsi_gpu(&device, &close_arr, period, None).unwrap();
        let (stoch_ind, _) = stochastic_gpu(&device, &high_arr, &low_arr, &close_arr, period, 3, None).unwrap();
        let williams_ind = williams_r_gpu(&device, &high_arr, &low_arr, &close_arr, period, None).unwrap();

        // Fused call
        let (rsi_fused, stoch_fused, williams_fused) =
            momentum_fusion_2d_gpu(&device, &high_arr, &low_arr, &close_arr, period).unwrap();

        // Compare RSI
        for i in 0..n {
            if rsi_ind[i].is_nan() {
                assert!(rsi_fused[i].is_nan());
            } else {
                assert!((rsi_ind[i] - rsi_fused[i]).abs() < 1e-9);
            }
        }

        // Compare Stochastic
        for i in 0..n {
            if stoch_ind[i].is_nan() {
                assert!(stoch_fused[i].is_nan());
            } else {
                assert!((stoch_ind[i] - stoch_fused[i]).abs() < 1e-9);
            }
        }

        // Compare Williams
        for i in 0..n {
            if williams_ind[i].is_nan() {
                assert!(williams_fused[i].is_nan());
            } else {
                assert!((williams_ind[i] - williams_fused[i]).abs() < 1e-9);
            }
        }
    }

    // ------------------------------------------------------------------
    // Host-side tests (no GPU required)
    // ------------------------------------------------------------------

    #[test]
    fn test_batch_2d_kernel_source_nvrtc_compatible() {
        assert!(
            !BATCH_2D_KERNELS.contains("#include"),
            "NVRTC kernel source must not use #include directives"
        );

        for name in [
            "rsi_batch_2d_kernel",
            "rsi_batch_final_2d_kernel",
            "sma_batch_2d_kernel",
            "stochastic_batch_2d_kernel",
            "momentum_fusion_2d_kernel",
            "volatility_fusion_2d_kernel",
        ] {
            let signature = format!("extern \"C\" __global__ void {}", name);
            assert!(
                BATCH_2D_KERNELS.contains(&signature),
                "missing kernel entry point: {}",
                name
            );
        }
    }

    #[test]
    fn test_momentum_fusion_kernel_is_flattened() {
        let start = BATCH_2D_KERNELS
            .find("momentum_fusion_2d_kernel")
            .expect("momentum fusion kernel present");
        let end = BATCH_2D_KERNELS
            .find("volatility_fusion_2d_kernel")
            .expect("volatility fusion kernel present");
        let fusion_src = &BATCH_2D_KERNELS[start..end];

        // Flattened to a single 256-thread mapping: no threadIdx.y indicator
        // selection (must match the (256, 1, 1) block_dim at the launch site).
        assert!(
            !fusion_src.contains("threadIdx.y"),
            "momentum fusion kernel must not branch on threadIdx.y"
        );

        // Window extrema computed exactly once; Stochastic %K and Williams %R
        // both derive from the same highest_high/lowest_low registers.
        assert_eq!(fusion_src.matches("highest_high = fmax").count(), 1);
        assert_eq!(fusion_src.matches("lowest_low = fmin").count(), 1);
    }

    #[test]
    fn test_wilder_smooth_batch_matches_sequential_reference() {
        let n_assets = 4;
        let n_candles = 96;
        let period = 14;

        // Deterministic non-negative gains/losses, [n_assets, n_candles]
        let gains: Vec<f64> = (0..n_assets * n_candles)
            .map(|i| (i as f64 * 0.7).sin().abs() * 2.0)
            .collect();
        let losses: Vec<f64> = (0..n_assets * n_candles)
            .map(|i| (i as f64 * 1.3).cos().abs() * 1.5)
            .collect();

        let (avg_gain_batch, avg_loss_batch) =
            wilder_smooth_batch(&gains, &losses, n_assets, n_candles, period)
                .expect("smoothing failed");

        assert_eq!(avg_gain_batch.len(), n_assets * n_candles);
        assert_eq!(avg_loss_batch.len(), n_assets * n_candles);

        for asset_idx in 0..n_assets {
            let start = asset_idx * n_candles;
            let g = Array1::from_vec(gains[start..start + n_candles].to_vec());
            let l = Array1::from_vec(losses[start..start + n_candles].to_vec());
            let expected_gain = wilders_smoothing_cpu(&g, period).unwrap();
            let expected_loss = wilders_smoothing_cpu(&l, period).unwrap();

            for i in 0..n_candles {
                for (actual, expected) in [
                    (avg_gain_batch[start + i], expected_gain[i]),
                    (avg_loss_batch[start + i], expected_loss[i]),
                ] {
                    if expected.is_nan() {
                        assert!(
                            actual.is_nan(),
                            "asset {}, candle {}: expected NaN",
                            asset_idx,
                            i
                        );
                    } else {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "asset {}, candle {}: {} vs {}",
                            asset_idx,
                            i,
                            actual,
                            expected
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_wilder_smooth_batch_propagates_errors() {
        // period > n_candles must surface the wilders_smoothing_cpu error
        let gains = vec![1.0; 8];
        let losses = vec![1.0; 8];
        assert!(wilder_smooth_batch(&gains, &losses, 1, 8, 20).is_err());
    }
}
