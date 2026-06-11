//! GPU-Accelerated Trade Aggregation
//!
//! High-performance OHLCV candle aggregation using CUDA for large trade datasets.
//!
//! # Performance
//!
//! - **Small datasets (<10K trades)**: CPU faster (kernel overhead dominates)
//! - **Medium (10-100K)**: 2-5x speedup vs CPU
//! - **Large (>100K)**: 5-10x speedup vs CPU
//! - **Crossover point**: ~10-20K trades
//! - All seven OHLCV fields (open, high, low, close, volume, quote volume,
//!   trade count) are produced on the GPU — no per-trade host post-processing.
//!
//! # Algorithm
//!
//! ```text
//! Input:  [Trade array — i64 timestamps + f64 price/qty/quote_qty (SoA)]
//!          ↓ init_ohlcv_state_kernel   (encoded ∓inf into high/low)
//!          ↓ bin_trades_kernel         (i64 ts → dense candle index)
//!          ↓ aggregate_ohlcv_kernel    (atomics + boundary open/close)
//! Output: [Dense candle slots; empty slots filtered on host]
//! ```
//!
//! ## Dense candle indexing
//!
//! Bucket ids are rebased on the GPU as `ts / timeframe_ms - first_bucket`
//! where `first_bucket = trades[0].timestamp_ms / timeframe_ms`. This:
//!
//! - makes bucket ids directly usable as output indices (the old pipeline
//!   round-tripped bucket ids to the host and remapped them with an
//!   O(n_trades × n_candles) `position()` scan),
//! - fixes the i32 overflow of the old `(int)(ts / timeframe_ms)` for
//!   sub-second timeframes on epoch-millisecond timestamps,
//! - lets the host compute `n_candles` from the first/last trade alone
//!   (trades are time-ordered) with zero device-to-host transfers.
//!
//! `n_candles = last_bucket - first_bucket + 1` counts *dense* slots, so
//! time gaps allocate empty slots that are filtered out (`num_trades == 0`)
//! when candles are constructed, preserving the sparse output contract of
//! the CPU reference (`crate::binance::aggregate_trades_to_candles`).
//!
//! ## Open/Close on GPU (segment-boundary detection)
//!
//! Because trades are time-ordered, each candle's trades occupy a contiguous
//! index range. Thread `i` compares `bucket(i)` with `bucket(i-1)`: on a
//! discontinuity it writes `open[bucket(i)] = price[i]` and
//! `close[bucket(i-1)] = price[i-1]` (plus the `i == 0` / `i == n-1` edges).
//! Exactly one thread writes each slot — no atomics, no CPU pass.
//!
//! Inputs that are *not* time-ordered fall back transparently to the CPU
//! reference implementation, which handles arbitrary order.
//!
//! ## High/Low via ordered-int atomics
//!
//! There is no native `atomicMax(double*)`. Instead of the old atomicCAS
//! retry loops, prices are mapped through an order-preserving `u64` encoding
//! (sign-flip transform of the IEEE-754 bits) so native integer
//! `atomicMax`/`atomicMin` apply. The high/low buffers are initialized by a
//! kernel to `encode(-inf)`/`encode(+inf)` — not zero, which would floor
//! highs at 0.0 for negative-price instruments (e.g., oil futures in
//! April 2020). See `encode_ordered_f64`/`decode_ordered_u64`, which mirror
//! `ordered_encode_double` in `kernels/aggregation.cu`.
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::aggregation::GpuAggregator;
//! use kimsfinance_core::binance::{Trade, Timeframe};
//!
//! let aggregator = GpuAggregator::new()?;
//! let candles = aggregator.aggregate_trades(&trades, Timeframe::minutes(5))?;
//! ```

use super::{GpuDevice, GpuError};
use crate::binance::{Candle, Timeframe, Trade};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// CUDA kernel source (also used by host-side tests to validate the
/// layout contract between this file and the kernels).
const AGGREGATION_KERNEL_SRC: &str = include_str!("kernels/aggregation.cu");

/// Threads per block for all aggregation kernels.
/// Must match the launch-configuration contract in `kernels/aggregation.cu`.
const THREADS_PER_BLOCK: usize = 256;

/// Sign bit of an IEEE-754 double — pivot of the order-preserving encoding.
const ORDERED_SIGN_BIT: u64 = 0x8000_0000_0000_0000;

/// `encode_ordered_f64(f64::NEG_INFINITY)` — atomicMax identity.
/// Must match `ORDERED_ENCODED_NEG_INF` in `kernels/aggregation.cu`.
#[cfg_attr(not(test), allow(dead_code))]
const ORDERED_ENCODED_NEG_INF: u64 = 0x000F_FFFF_FFFF_FFFF;

/// `encode_ordered_f64(f64::INFINITY)` — atomicMin identity.
/// Must match `ORDERED_ENCODED_POS_INF` in `kernels/aggregation.cu`.
#[cfg_attr(not(test), allow(dead_code))]
const ORDERED_ENCODED_POS_INF: u64 = 0xFFF0_0000_0000_0000;

/// GPU trade aggregator with CUDA kernels
pub struct GpuAggregator {
    device: GpuDevice,
    /// High/low state initialization kernel (writes encoded ∓inf)
    init_kernel: cudarc::driver::CudaFunction,
    /// Binning kernel (map trades to dense candle indices)
    binning_kernel: cudarc::driver::CudaFunction,
    /// OHLCV aggregation kernel (atomics + segment-boundary open/close)
    aggregation_kernel: cudarc::driver::CudaFunction,
}

impl GpuAggregator {
    /// Initialize GPU aggregator with compiled CUDA kernels
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - GPU initialization fails
    /// - Kernel compilation fails
    /// - No CUDA-capable device found
    pub fn new() -> Result<Self, GpuError> {
        let device = GpuDevice::new()?;

        // Compile CUDA kernels (cached: 50-200x faster on subsequent inits)
        let ptx = compile_aggregation_kernels()?;
        let module = device.context().load_module(ptx).map_err(|e| {
            GpuError::CompilationError(format!("Failed to load PTX module: {:?}", e))
        })?;

        let init_kernel = module.load_function("init_ohlcv_state_kernel").map_err(|e| {
            GpuError::CompilationError(format!("Failed to load init_ohlcv_state_kernel: {:?}", e))
        })?;

        let binning_kernel = module.load_function("bin_trades_kernel").map_err(|e| {
            GpuError::CompilationError(format!("Failed to load bin_trades_kernel: {:?}", e))
        })?;

        let aggregation_kernel = module
            .load_function("aggregate_ohlcv_kernel")
            .map_err(|e| {
                GpuError::CompilationError(format!(
                    "Failed to load aggregate_ohlcv_kernel: {:?}",
                    e
                ))
            })?;

        Ok(Self {
            device,
            init_kernel,
            binning_kernel,
            aggregation_kernel,
        })
    }

    /// Check if GPU aggregation is available
    ///
    /// # Returns
    ///
    /// - `true`: GPU available and kernels compiled
    /// - `false`: GPU not available (will fall back to CPU)
    pub fn is_available() -> bool {
        GpuDevice::new().is_ok()
    }

    /// Aggregate trades to candles on GPU
    ///
    /// # Performance
    ///
    /// - **<10K trades**: CPU faster (use CPU aggregation)
    /// - **10-100K**: 2-5x speedup vs CPU
    /// - **>100K**: 5-10x speedup vs CPU
    ///
    /// # Algorithm
    ///
    /// 1. Transfer trades to GPU (H2D, pinned staging for f64 columns)
    /// 2. Initialize high/low state (encoded ∓inf)
    /// 3. Bin trades to dense candle indices (parallel)
    /// 4. Aggregate OHLCV (atomics + segment-boundary open/close)
    /// 5. Transfer dense candle slots back, filter empties (D2H)
    ///
    /// # Preconditions
    ///
    /// Trades should be sorted by timestamp (Binance exports are). Unordered
    /// input is detected and falls back to the CPU reference implementation.
    ///
    /// # Arguments
    ///
    /// * `trades` - Input trade array
    /// * `timeframe` - Aggregation timeframe (e.g., 5 minutes)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - The timeframe is non-positive, or the trade time range spans more
    ///   candle slots than fit in i32 indexing
    /// - GPU memory allocation fails
    /// - Kernel launch fails
    /// - Memory transfer fails
    pub fn aggregate_trades(
        &self,
        trades: &[Trade],
        timeframe: Timeframe,
    ) -> Result<Vec<Candle>, GpuError> {
        if trades.is_empty() {
            return Ok(Vec::new());
        }

        let n_trades = trades.len();
        let timeframe_ms = timeframe.to_ms();
        if timeframe_ms <= 0 {
            return Err(GpuError::InvalidParameter(format!(
                "Timeframe must be positive, got {} ms",
                timeframe_ms
            )));
        }

        // Step 1: Extract trade data into separate arrays (SoA layout for GPU).
        // Timestamps stay i64 end-to-end: f64 has only 53 mantissa bits and
        // the binning kernel needs exact integer division.
        let mut timestamps: Vec<i64> = Vec::with_capacity(n_trades);
        let mut prices = Vec::with_capacity(n_trades);
        let mut quantities = Vec::with_capacity(n_trades);
        let mut quote_quantities = Vec::with_capacity(n_trades);

        for trade in trades {
            timestamps.push(trade.timestamp_ms);
            prices.push(trade.price);
            quantities.push(trade.quantity);
            quote_quantities.push(trade.quote_quantity);
        }

        // GPU open/close uses segment-boundary detection, which requires
        // time-ordered trades. Unordered input (rare; Binance exports are
        // sorted) falls back to the CPU reference implementation.
        if !timestamps_are_ordered(&timestamps) {
            return Ok(crate::binance::aggregate_trades_to_candles(
                trades, timeframe,
            ));
        }

        // Dense candle indexing: candle 0 is the first trade's bucket.
        // n_candles comes from the slice ends alone — zero D2H transfers.
        let first_bucket = timestamps[0] / timeframe_ms;
        let last_bucket = timestamps[n_trades - 1] / timeframe_ms;
        let n_candles_i64 = last_bucket - first_bucket + 1;
        if n_candles_i64 > i32::MAX as i64 {
            return Err(GpuError::InvalidParameter(format!(
                "Trade time range spans {} candle slots at {} ms timeframe \
                 (exceeds i32 indexing); use a coarser timeframe",
                n_candles_i64, timeframe_ms
            )));
        }
        let n_candles = n_candles_i64 as usize;

        // Step 2: Transfer trade data to GPU.
        // f64 columns go through pinned staging buffers for fast async H2D.
        // The staging buffers are held until after the synchronize below —
        // releasing them back to the pool before the DMA completes would let
        // another thread overwrite memory still being read by the copy engine.
        let mut pinned_prices = self.device.pinned_pool.lock().acquire(n_trades)?;
        pinned_prices.as_mut_slice()[..n_trades].copy_from_slice(&prices);

        let mut pinned_quantities = self.device.pinned_pool.lock().acquire(n_trades)?;
        pinned_quantities.as_mut_slice()[..n_trades].copy_from_slice(&quantities);

        let mut pinned_quote_quantities = self.device.pinned_pool.lock().acquire(n_trades)?;
        pinned_quote_quantities.as_mut_slice()[..n_trades].copy_from_slice(&quote_quantities);

        let mut d_prices = self.device.alloc_buffer(n_trades)?;
        let mut d_quantities = self.device.alloc_buffer(n_trades)?;
        let mut d_quote_quantities = self.device.alloc_buffer(n_trades)?;

        self.device
            .stream
            .memcpy_htod(&pinned_prices.as_slice()[..n_trades], &mut d_prices)?;
        self.device
            .stream
            .memcpy_htod(&pinned_quantities.as_slice()[..n_trades], &mut d_quantities)?;
        self.device.stream.memcpy_htod(
            &pinned_quote_quantities.as_slice()[..n_trades],
            &mut d_quote_quantities,
        )?;

        // Timestamps: synchronous pageable H2D (the pinned pool is f64-only;
        // an f64 round-trip would lose integer precision above 2^53).
        let d_timestamps = self.device.copy_to_device_i64(&timestamps)?;

        // Step 3: Allocate per-trade and per-candle device buffers.
        let mut d_bucket_ids = self
            .device
            .stream
            .alloc_zeros::<i32>(n_trades)
            .map_err(|e| {
                GpuError::AllocationError(format!(
                    "Failed to allocate {} i32 elements for bucket ids: {:?}",
                    n_trades, e
                ))
            })?;

        // High/low are stored as order-preserving u64 encodings of f64 and
        // initialized by init_ohlcv_state_kernel (zeros here are overwritten).
        let mut d_high_bits = self
            .device
            .stream
            .alloc_zeros::<u64>(n_candles)
            .map_err(|e| {
                GpuError::AllocationError(format!(
                    "Failed to allocate {} u64 elements for high bits: {:?}",
                    n_candles, e
                ))
            })?;
        let mut d_low_bits = self
            .device
            .stream
            .alloc_zeros::<u64>(n_candles)
            .map_err(|e| {
                GpuError::AllocationError(format!(
                    "Failed to allocate {} u64 elements for low bits: {:?}",
                    n_candles, e
                ))
            })?;

        let mut d_open = self.device.alloc_buffer(n_candles)?;
        let mut d_close = self.device.alloc_buffer(n_candles)?;
        let mut d_volume = self.device.alloc_buffer(n_candles)?;
        let mut d_quote_volume = self.device.alloc_buffer(n_candles)?;
        let mut d_num_trades = self
            .device
            .stream
            .alloc_zeros::<i32>(n_candles)
            .map_err(|e| {
                GpuError::AllocationError(format!(
                    "Failed to allocate {} i32 elements for num_trades: {:?}",
                    n_candles, e
                ))
            })?;

        // Launch configurations (block size must match kernels/aggregation.cu).
        let trade_cfg = LaunchConfig {
            grid_dim: (n_trades.div_ceil(THREADS_PER_BLOCK) as u32, 1, 1),
            block_dim: (THREADS_PER_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let candle_cfg = LaunchConfig {
            grid_dim: (n_candles.div_ceil(THREADS_PER_BLOCK) as u32, 1, 1),
            block_dim: (THREADS_PER_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_trades_i32 = n_trades as i32;
        let n_candles_i32 = n_candles as i32;

        // Step 4: Initialize high/low to encoded ∓inf. NOT zero — a
        // zero-initialized high buffer floors at 0.0 and corrupts results
        // for negative-price instruments.
        unsafe {
            let mut builder = self.device.stream.launch_builder(&self.init_kernel);
            builder
                .arg(&mut d_high_bits)
                .arg(&mut d_low_bits)
                .arg(&n_candles_i32)
                .launch(candle_cfg)
                .map_err(|e| {
                    GpuError::ExecutionError(format!("Init kernel launch failed: {:?}", e))
                })?;
        }

        // Step 5: Bin trades into dense candle indices.
        unsafe {
            let mut builder = self.device.stream.launch_builder(&self.binning_kernel);
            builder
                .arg(&d_timestamps)
                .arg(&mut d_bucket_ids)
                .arg(&n_trades_i32)
                .arg(&timeframe_ms)
                .arg(&first_bucket)
                .launch(trade_cfg)
                .map_err(|e| {
                    GpuError::ExecutionError(format!("Binning kernel launch failed: {:?}", e))
                })?;
        }

        // Step 6: Aggregate OHLCV (atomics + segment-boundary open/close).
        unsafe {
            let mut builder = self.device.stream.launch_builder(&self.aggregation_kernel);
            builder
                .arg(&d_prices)
                .arg(&d_quantities)
                .arg(&d_quote_quantities)
                .arg(&d_bucket_ids)
                .arg(&n_trades_i32)
                .arg(&mut d_high_bits)
                .arg(&mut d_low_bits)
                .arg(&mut d_open)
                .arg(&mut d_close)
                .arg(&mut d_volume)
                .arg(&mut d_quote_volume)
                .arg(&mut d_num_trades)
                .launch(trade_cfg)
                .map_err(|e| {
                    GpuError::ExecutionError(format!("Aggregation kernel launch failed: {:?}", e))
                })?;
        }

        // Step 7: One synchronize covers both kernel completion and the async
        // H2D DMAs — only now is it safe to return the pinned staging buffers
        // to the pool.
        self.device.synchronize()?;

        let mut pool = self.device.pinned_pool.lock();
        pool.release(pinned_prices);
        pool.release(pinned_quantities);
        pool.release(pinned_quote_quantities);
        drop(pool);

        // Step 8: Copy per-candle results back. Synchronous pageable copies:
        // the candle arrays are small relative to the trade inputs, and the
        // dense slot count can exceed the pinned pool's standard buffer size
        // for sparse data, so the pinned path is not usable here.
        let high_bits: Vec<u64> = self.device.stream.memcpy_dtov(&d_high_bits).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy high bits from device: {:?}", e))
        })?;
        let low_bits: Vec<u64> = self.device.stream.memcpy_dtov(&d_low_bits).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy low bits from device: {:?}", e))
        })?;
        let open = self.device.copy_to_host(&d_open)?;
        let close = self.device.copy_to_host(&d_close)?;
        let volume = self.device.copy_to_host(&d_volume)?;
        let quote_volume = self.device.copy_to_host(&d_quote_volume)?;
        let num_trades = self.device.copy_to_host_i32(&d_num_trades)?;

        // Step 9: Construct candles. Dense slots with no trades (time gaps)
        // are filtered to preserve the sparse output contract of the CPU
        // reference. Dense indices ascend with time, so the result is sorted.
        let mut candles = Vec::with_capacity(n_candles);
        for i in 0..n_candles {
            let trades_in_candle = num_trades[i];
            if trades_in_candle == 0 {
                continue;
            }
            candles.push(Candle {
                // (first_bucket + i) * timeframe_ms == (ts / timeframe_ms) *
                // timeframe_ms — identical to the CPU reference bucket math.
                timestamp: (first_bucket + i as i64) * timeframe_ms,
                open: open[i],
                high: decode_ordered_u64(high_bits[i]),
                low: decode_ordered_u64(low_bits[i]),
                close: close[i],
                volume: volume[i],
                quote_volume: quote_volume[i],
                num_trades: trades_in_candle as usize,
            });
        }

        Ok(candles)
    }
}

/// Helper method to copy i32 data from device to host
impl GpuDevice {
    pub fn copy_to_host_i32(&self, buffer: &CudaSlice<i32>) -> Result<Vec<i32>, GpuError> {
        self.stream.memcpy_dtov(buffer).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy i32 from device: {:?}", e))
        })
    }
}

/// Order-preserving u64 image of an f64 (sign-flip transform).
///
/// Strictly monotonic over all non-NaN doubles, enabling native integer
/// `atomicMax`/`atomicMin` on the GPU instead of atomicCAS retry loops.
/// Mirrors `ordered_encode_double` in `kernels/aggregation.cu` — the two
/// MUST stay in sync (layout contract).
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn encode_ordered_f64(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits & ORDERED_SIGN_BIT != 0 {
        !bits
    } else {
        bits | ORDERED_SIGN_BIT
    }
}

/// Inverse of `encode_ordered_f64`; decodes GPU atomicMax/atomicMin results.
#[inline]
fn decode_ordered_u64(e: u64) -> f64 {
    let bits = if e & ORDERED_SIGN_BIT != 0 {
        e & !ORDERED_SIGN_BIT
    } else {
        !e
    };
    f64::from_bits(bits)
}

/// Dense candle index for a trade timestamp.
///
/// Mirrors `bin_trades_kernel` in `kernels/aggregation.cu`: truncating i64
/// division, identical to the CPU reference bucket math in
/// `crate::binance::aggregate_trades_to_candles`.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn dense_bucket_index(timestamp_ms: i64, timeframe_ms: i64, first_bucket: i64) -> i64 {
    timestamp_ms / timeframe_ms - first_bucket
}

/// Check that timestamps are non-decreasing (segment-boundary open/close
/// detection on the GPU requires time-ordered trades).
#[inline]
fn timestamps_are_ordered(timestamps: &[i64]) -> bool {
    timestamps.windows(2).all(|w| w[0] <= w[1])
}

/// Compile CUDA kernels for trade aggregation (cached).
///
/// Uses `compile_ptx_optimized_cached` so repeated `GpuAggregator::new()`
/// calls skip NVRTC compilation entirely (50-200x faster on cache hits).
fn compile_aggregation_kernels() -> Result<cudarc::nvrtc::Ptx, GpuError> {
    let ptx_arc =
        super::compile::compile_ptx_optimized_cached(AGGREGATION_KERNEL_SRC).map_err(|e| {
            GpuError::CompilationError(format!("Failed to compile aggregation kernels: {:?}", e))
        })?;
    Ok(Arc::unwrap_or_clone(ptx_arc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binance::aggregate_trades_to_candles;

    fn make_trade(trade_id: u64, price: f64, quantity: f64, timestamp_ms: i64) -> Trade {
        Trade {
            trade_id,
            price,
            quantity,
            quote_quantity: price.abs() * quantity,
            timestamp_ms,
            is_buyer_maker: false,
        }
    }

    /// Deterministic pseudo-random ordered trade stream (no external deps).
    fn make_ordered_trades(count: usize, start_ts: i64, base_price: f64) -> Vec<Trade> {
        let mut trades = Vec::with_capacity(count);
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut ts = start_ts;
        for id in 0..count {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (state >> 11) as f64 / (1u64 << 53) as f64;
            ts += (r * 250.0) as i64; // 0-250ms inter-trade gaps
            let price = base_price + (r - 0.5) * 100.0;
            let quantity = 0.001 + r;
            trades.push(make_trade(id as u64, price, quantity, ts));
        }
        trades
    }

    // ===== GPU-gated tests =====

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_aggregator_initialization() {
        let result = GpuAggregator::new();
        assert!(result.is_ok(), "Failed to initialize GPU aggregator");
    }

    #[test]
    fn test_is_available() {
        // Should not panic
        let available = GpuAggregator::is_available();
        println!("GPU aggregation available: {}", available);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_aggregate_empty_trades() {
        let aggregator = GpuAggregator::new().expect("GPU not available");
        let trades = vec![];
        let candles = aggregator
            .aggregate_trades(&trades, Timeframe::minutes(1))
            .expect("Aggregation failed");
        assert!(candles.is_empty());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_matches_cpu_reference() {
        let aggregator = GpuAggregator::new().expect("GPU not available");
        let trades = make_ordered_trades(50_000, 1_609_459_200_000, 20_000.0);
        let timeframe = Timeframe::seconds(5);

        let gpu = aggregator
            .aggregate_trades(&trades, timeframe)
            .expect("GPU aggregation failed");
        let cpu = aggregate_trades_to_candles(&trades, timeframe);

        assert_eq!(gpu.len(), cpu.len(), "Candle count mismatch");
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            assert_eq!(g.timestamp, c.timestamp);
            assert_eq!(g.open, c.open, "open mismatch at ts {}", g.timestamp);
            assert_eq!(g.high, c.high, "high mismatch at ts {}", g.timestamp);
            assert_eq!(g.low, c.low, "low mismatch at ts {}", g.timestamp);
            assert_eq!(g.close, c.close, "close mismatch at ts {}", g.timestamp);
            assert_eq!(g.num_trades, c.num_trades);
            // atomicAdd order differs from sequential CPU summation
            let vol_tol = 1e-9 * c.volume.abs().max(1.0);
            assert!((g.volume - c.volume).abs() < vol_tol);
            let qvol_tol = 1e-9 * c.quote_volume.abs().max(1.0);
            assert!((g.quote_volume - c.quote_volume).abs() < qvol_tol);
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_negative_prices() {
        // Negative-price instrument (e.g., WTI futures, April 2020):
        // exercises the ordered-int high/low fix. The old pipeline
        // zero-initialized the high buffer, flooring highs at 0.0 for
        // all-negative prices. Prices here span [-150, -50] so every
        // candle high must remain strictly negative.
        let aggregator = GpuAggregator::new().expect("GPU not available");
        let trades = make_ordered_trades(10_000, 1_587_600_000_000, -100.0);
        let timeframe = Timeframe::seconds(1);

        let gpu = aggregator
            .aggregate_trades(&trades, timeframe)
            .expect("GPU aggregation failed");
        let cpu = aggregate_trades_to_candles(&trades, timeframe);

        assert_eq!(gpu.len(), cpu.len());
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            assert_eq!(g.high, c.high);
            assert_eq!(g.low, c.low);
            assert!(g.high < 0.0, "negative-price high must stay negative");
        }
    }

    // ===== Host-only tests (no GPU required) =====

    #[test]
    fn test_dense_bucket_index_subsecond_timeframe() {
        // 100ms candles on epoch-millisecond timestamps: the raw quotient
        // ts / timeframe_ms exceeds i32::MAX (the old kernel's i32 cast
        // overflowed), while rebased dense indices stay tiny.
        let timeframe_ms = 100_i64;
        let first_ts = 1_700_000_000_123_i64;
        let first_bucket = first_ts / timeframe_ms;
        assert!(
            first_bucket > i32::MAX as i64,
            "raw bucket id must demonstrate the old i32 overflow"
        );

        // Offsets chosen to cover same-bucket, boundary, and gap cases.
        let cases = [
            (0_i64, 0_i64),   // first trade → index 0
            (10, 0),          // same 100ms bucket (…123 → …133)
            (76, 0),          // …199: last ms of bucket 0
            (77, 1),          // …200: first ms of bucket 1
            (250, 2),         // …373: bucket 2 (gap-free run)
            (1077, 11),       // …1200 → dense index 11 (buckets 3-10 empty)
        ];
        for (offset, expected) in cases {
            let ts = first_ts + offset;
            assert_eq!(
                dense_bucket_index(ts, timeframe_ms, first_bucket),
                expected,
                "offset {} should map to dense index {}",
                offset,
                expected
            );
        }
    }

    #[test]
    fn test_dense_bucket_index_matches_cpu_reference_timestamps() {
        // Reconstructed candle timestamp (first_bucket + index) * timeframe_ms
        // must equal the CPU reference bucket math (ts / tf) * tf.
        let timeframes = [100_i64, 1_000, 60_000, 300_000];
        let first_ts = 1_609_459_200_123_i64;
        for timeframe_ms in timeframes {
            let first_bucket = first_ts / timeframe_ms;
            for offset in [0_i64, 1, 999, 59_999, 60_000, 1_234_567] {
                let ts = first_ts + offset;
                let idx = dense_bucket_index(ts, timeframe_ms, first_bucket);
                assert!(idx >= 0, "ordered trades must give non-negative indices");
                let reconstructed = (first_bucket + idx) * timeframe_ms;
                let cpu_reference = (ts / timeframe_ms) * timeframe_ms;
                assert_eq!(reconstructed, cpu_reference);
            }
        }
    }

    #[test]
    fn test_ordered_encoding_roundtrip() {
        let values = [
            f64::NEG_INFINITY,
            -1.0e308,
            -37.63, // WTI settlement, 2020-04-20
            -1.0e-9,
            -0.0,
            0.0,
            1.0e-9,
            28_948.19,
            1.0e308,
            f64::INFINITY,
        ];
        for v in values {
            let decoded = decode_ordered_u64(encode_ordered_f64(v));
            assert_eq!(
                decoded.to_bits(),
                v.to_bits(),
                "roundtrip must be bit-exact for {}",
                v
            );
        }
    }

    #[test]
    fn test_ordered_encoding_monotonic() {
        // Includes negative prices: the encoding (not a zero-floored buffer)
        // is what makes GPU high/low correct for negative-price instruments.
        let sorted = [
            f64::NEG_INFINITY,
            -1.0e308,
            -100.5,
            -37.63,
            -1.0e-9,
            0.0,
            1.0e-9,
            42.0,
            28_948.19,
            1.0e308,
            f64::INFINITY,
        ];
        for w in sorted.windows(2) {
            assert!(
                encode_ordered_f64(w[0]) < encode_ordered_f64(w[1]),
                "encoding must be strictly increasing: {} vs {}",
                w[0],
                w[1]
            );
        }
        // max/min over encodings == max/min over values
        let encoded_max = sorted.iter().map(|&v| encode_ordered_f64(v)).max().unwrap();
        let encoded_min = sorted.iter().map(|&v| encode_ordered_f64(v)).min().unwrap();
        assert_eq!(decode_ordered_u64(encoded_max), f64::INFINITY);
        assert_eq!(decode_ordered_u64(encoded_min), f64::NEG_INFINITY);
    }

    #[test]
    fn test_ordered_encoding_constants_match_kernel_source() {
        // Rust constants must equal the actual encodings...
        assert_eq!(encode_ordered_f64(f64::NEG_INFINITY), ORDERED_ENCODED_NEG_INF);
        assert_eq!(encode_ordered_f64(f64::INFINITY), ORDERED_ENCODED_POS_INF);
        // ...and the CUDA source must initialize with the same literals
        // (layout contract between aggregation.rs and aggregation.cu).
        assert!(
            AGGREGATION_KERNEL_SRC.contains("0x000FFFFFFFFFFFFFULL"),
            "kernel source must define encode(-inf) literal"
        );
        assert!(
            AGGREGATION_KERNEL_SRC.contains("0xFFF0000000000000ULL"),
            "kernel source must define encode(+inf) literal"
        );
    }

    #[test]
    fn test_kernel_source_nvrtc_compatible() {
        // Check actual directives line-by-line (the header comment may
        // legitimately mention the word "#include").
        let has_include_directive = AGGREGATION_KERNEL_SRC
            .lines()
            .any(|line| line.trim_start().starts_with("#include"));
        assert!(
            !has_include_directive,
            "NVRTC kernels must not use #include directives"
        );
        for name in [
            "init_ohlcv_state_kernel",
            "bin_trades_kernel",
            "aggregate_ohlcv_kernel",
        ] {
            let signature = format!("extern \"C\" __global__ void {}", name);
            assert!(
                AGGREGATION_KERNEL_SRC.contains(&signature),
                "kernel source must define extern \"C\" entry point {}",
                name
            );
        }
        // The never-launched volume-only kernel was deleted.
        assert!(!AGGREGATION_KERNEL_SRC.contains("aggregate_volume_only_kernel"));
        // CAS retry loops were replaced by ordered-int atomics.
        assert!(!AGGREGATION_KERNEL_SRC.contains("atomicCAS"));
        assert!(AGGREGATION_KERNEL_SRC.contains("atomicMax"));
        assert!(AGGREGATION_KERNEL_SRC.contains("atomicMin"));
    }

    #[test]
    fn test_timestamps_are_ordered() {
        assert!(timestamps_are_ordered(&[]));
        assert!(timestamps_are_ordered(&[42]));
        assert!(timestamps_are_ordered(&[1, 2, 2, 3])); // equal timestamps OK
        assert!(!timestamps_are_ordered(&[1, 3, 2]));
        assert!(!timestamps_are_ordered(&[2, 1]));
    }

    #[test]
    fn test_dense_candle_count_from_slice_ends() {
        // n_candles = last_bucket - first_bucket + 1 (host-side, zero D2H)
        let timeframe_ms = 60_000_i64;
        let trades = [
            make_trade(1, 100.0, 1.0, 1_609_459_200_000), // minute 0
            make_trade(2, 101.0, 1.0, 1_609_459_260_000), // minute 1
            make_trade(3, 102.0, 1.0, 1_609_459_500_000), // minute 5 (gap 2-4)
        ];
        let first_bucket = trades[0].timestamp_ms / timeframe_ms;
        let last_bucket = trades[trades.len() - 1].timestamp_ms / timeframe_ms;
        let n_candles = last_bucket - first_bucket + 1;
        assert_eq!(n_candles, 6, "dense slots include the empty gap buckets");

        // Each trade's dense index lies within [0, n_candles)
        for trade in &trades {
            let idx = dense_bucket_index(trade.timestamp_ms, timeframe_ms, first_bucket);
            assert!(idx >= 0 && idx < n_candles);
        }
    }
}
