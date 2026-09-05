//! GPU Tick-Level Aggregation (Sorted Boundary Detection)
//!
//! High-performance trade→OHLCV aggregation for timestamp-sorted tick streams.
//!
//! # Architecture
//!
//! ## Pipeline
//!
//! 1. **Sortedness check** (`check_sorted_kernel`): exchange tick feeds are
//!    timestamp-sorted; a single i32 flag is copied back. Unsorted input
//!    falls back to the CPU aggregator (same semantics as
//!    `crate::binance::aggregate_trades_to_candles`).
//! 2. **Bucket range** (`compute_bucket_range_kernel`): grid-stride min/max
//!    reduction over `bucket = ts / timeframe_ms`. Only two i64 scalars come
//!    back; there is no per-trade bucket-id D2H copy and no host hash pass.
//! 3. **Aggregation** (`aggregate_ohlcv_sorted_kernel`): one pass over the
//!    sorted stream. Open/close/timestamps are race-free plain stores at
//!    bucket boundaries; high/low use one hardware atomicMax/atomicMin each
//!    on an ordered-uint image of the f32 price; volume uses
//!    `atomicAdd(float)`.
//! 4. **Candle construction**: all output buffers are copied back with one
//!    stream synchronize; empty dense slots (`num_trades == 0`) are filtered
//!    on the host.
//!
//! ## Precision
//!
//! Prices and volumes stay f32 end-to-end. The input feed is f32, and Ada
//! (sm_89) executes FP64 at 1/64 the FP32 rate, so widening on the host
//! doubled transfer volume without adding information. Timestamp/bucket math
//! is i64 (exact).
//!
//! ## Memory Layout: Structure-of-Arrays (SoA)
//!
//! **Input**:
//! - `timestamps: &[i64]` - Milliseconds since epoch (sorted non-decreasing)
//! - `prices: &[f32]` - Trade execution prices
//! - `volumes: &[f32]` - Trade volumes
//! - `sides: &[i8]` - Buy/sell indicators (1/-1, reserved)
//!
//! **Output** ([`AggregatedCandles`]):
//! - `timestamps: Vec<i64>` - Candle open times (`bucket * timeframe_ms`)
//! - `open/high/low/close/volume: Vec<f32>`
//! - `num_trades: Vec<i32>` - Trade count per candle
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::tick_aggregation::TickAggregator;
//! use kimsfinance_core::gpu::device::GpuDevice;
//!
//! let device = GpuDevice::new()?;
//! let aggregator = TickAggregator::new(device)?;
//!
//! // Aggregate 106M trades to 5-minute candles
//! let candles = aggregator.aggregate(&timestamps, &prices, &volumes, &sides, 300_000)?;
//! println!("Aggregated {} trades into {} candles", timestamps.len(), candles.num_candles);
//! ```

use super::device::{GpuDevice, GpuError};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

/// Threads per block for all kernels in this module.
const THREADS_PER_BLOCK: usize = 256;

/// Grid cap for the grid-stride bucket-range reduction (1024 blocks × 256
/// threads covers any input via the kernel's grid-stride loop while keeping
/// the per-warp atomic count bounded).
const BUCKET_RANGE_MAX_BLOCKS: usize = 1024;

/// Maximum dense candle range handled on the GPU.
///
/// The aggregation kernel writes into a dense slot per bucket in
/// `[first_bucket, last_bucket]`, costing 32 bytes/slot across the seven
/// output buffers (~2.1 GB at this cap). A wider range means the data is far
/// too sparse for dense-range GPU aggregation, so we fall back to the CPU
/// hash aggregator instead of over-allocating.
const MAX_DENSE_CANDLES: i64 = 1 << 26;

/// Ordered-uint encoding of `-inf` (identity for the high/atomicMax buffer).
///
/// Layout contract mirrored from `kernels/tick_aggregation.cu`
/// (`ENCODED_NEG_INF`) - keep in sync. Written on-device by the init kernel;
/// the host mirror exists for contract validation in tests.
#[cfg_attr(not(test), allow(dead_code))]
const ENCODED_NEG_INF: u32 = 0x007F_FFFF;

/// Ordered-uint encoding of `+inf` (identity for the low/atomicMin buffer).
///
/// Layout contract mirrored from `kernels/tick_aggregation.cu`
/// (`ENCODED_POS_INF`) - keep in sync. Written on-device by the init kernel;
/// the host mirror exists for contract validation in tests.
#[cfg_attr(not(test), allow(dead_code))]
const ENCODED_POS_INF: u32 = 0xFF80_0000;

/// GPU tick aggregator using sorted boundary detection
pub struct TickAggregator {
    device: GpuDevice,
    /// Sortedness check (single i32 flag output)
    check_sorted_kernel: cudarc::driver::CudaFunction,
    /// Grid-stride bucket min/max reduction (two i64 scalar outputs)
    bucket_range_kernel: cudarc::driver::CudaFunction,
    /// High/low encoded-buffer initialization (encoded ∓inf identities)
    init_extrema_kernel: cudarc::driver::CudaFunction,
    /// Single-pass sorted OHLCV aggregation (f32)
    aggregate_sorted_kernel: cudarc::driver::CudaFunction,
    /// INT8 quantization (raw 0-255 convention; reserved for compressed output path)
    #[allow(dead_code)]
    quantize_kernel: cudarc::driver::CudaFunction,
    /// INT8 dequantization (for validation; reserved for compressed output path)
    #[allow(dead_code)]
    dequantize_kernel: cudarc::driver::CudaFunction,
}

/// Aggregated OHLCV candles (output from tick aggregation)
///
/// This structure uses SoA (Structure-of-Arrays) layout for compatibility
/// with downstream GPU kernels (orderflow analysis).
#[derive(Debug, Clone)]
pub struct AggregatedCandles {
    /// Candle start timestamps (milliseconds since epoch)
    pub timestamps: Vec<i64>,
    /// Open prices (first trade in candle, stream order)
    pub open: Vec<f32>,
    /// High prices (maximum in candle)
    pub high: Vec<f32>,
    /// Low prices (minimum in candle)
    pub low: Vec<f32>,
    /// Close prices (last trade in candle, stream order)
    pub close: Vec<f32>,
    /// Volumes (sum of trade volumes)
    pub volume: Vec<f32>,
    /// Trade counts (number of trades per candle)
    pub num_trades: Vec<i32>,
    /// Number of candles
    pub num_candles: usize,
}

impl TickAggregator {
    /// Initialize tick aggregator with compiled CUDA kernels
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - GPU initialization fails
    /// - Kernel compilation fails
    /// - No CUDA-capable device found
    pub fn new(device: GpuDevice) -> Result<Self, GpuError> {
        // Compile CUDA kernels
        let ptx = compile_tick_kernels()?;
        let module = device.context.load_module(ptx).map_err(|e| {
            GpuError::CompilationError(format!("Failed to load PTX module: {:?}", e))
        })?;

        let check_sorted_kernel = module.load_function("check_sorted_kernel").map_err(|e| {
            GpuError::CompilationError(format!("Failed to load check_sorted_kernel: {:?}", e))
        })?;

        let bucket_range_kernel = module
            .load_function("compute_bucket_range_kernel")
            .map_err(|e| {
                GpuError::CompilationError(format!(
                    "Failed to load compute_bucket_range_kernel: {:?}",
                    e
                ))
            })?;

        let init_extrema_kernel =
            module
                .load_function("init_ohlcv_extrema_kernel")
                .map_err(|e| {
                    GpuError::CompilationError(format!(
                        "Failed to load init_ohlcv_extrema_kernel: {:?}",
                        e
                    ))
                })?;

        let aggregate_sorted_kernel = module
            .load_function("aggregate_ohlcv_sorted_kernel")
            .map_err(|e| {
                GpuError::CompilationError(format!(
                    "Failed to load aggregate_ohlcv_sorted_kernel: {:?}",
                    e
                ))
            })?;

        let quantize_kernel = module
            .load_function("quantize_to_int8_kernel")
            .map_err(|e| {
                GpuError::CompilationError(format!(
                    "Failed to load quantize_to_int8_kernel: {:?}",
                    e
                ))
            })?;

        let dequantize_kernel = module
            .load_function("dequantize_from_int8_kernel")
            .map_err(|e| {
                GpuError::CompilationError(format!(
                    "Failed to load dequantize_from_int8_kernel: {:?}",
                    e
                ))
            })?;

        Ok(Self {
            device,
            check_sorted_kernel,
            bucket_range_kernel,
            init_extrema_kernel,
            aggregate_sorted_kernel,
            quantize_kernel,
            dequantize_kernel,
        })
    }

    /// Check if GPU tick aggregation is available
    pub fn is_available() -> bool {
        GpuDevice::new().is_ok()
    }

    /// Aggregate trades to OHLCV candles on GPU
    ///
    /// # Performance
    ///
    /// - **<10K trades**: CPU faster (kernel overhead dominates)
    /// - **10K-1M trades**: 5-10x speedup vs CPU
    /// - **>1M trades**: 10-20x speedup vs CPU
    ///
    /// # Fallback Behavior
    ///
    /// The GPU fast path requires a timestamp-sorted stream (exchange feeds
    /// are). Unsorted input - and pathologically sparse input whose dense
    /// bucket range exceeds [`MAX_DENSE_CANDLES`] - is aggregated on the CPU
    /// with identical semantics.
    ///
    /// # Arguments
    ///
    /// * `timestamps` - Trade timestamps (milliseconds since epoch)
    /// * `prices` - Trade prices
    /// * `volumes` - Trade volumes
    /// * `sides` - Trade sides (1=buy, -1=sell, 0=unknown; reserved)
    /// * `timeframe_ms` - Candle timeframe in milliseconds (e.g., 300_000 for 5m)
    ///
    /// # Returns
    ///
    /// Aggregated OHLCV candles in SoA layout. Open is the first trade and
    /// close the last trade of each candle in stream order; only non-empty
    /// candles are returned, sorted by timestamp.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input arrays have different lengths
    /// - `timeframe_ms` is not positive
    /// - GPU memory allocation, kernel launch, or memory transfer fails
    pub fn aggregate(
        &self,
        timestamps: &[i64],
        prices: &[f32],
        volumes: &[f32],
        _sides: &[i8], // Future use: imbalance analysis
        timeframe_ms: i64,
    ) -> Result<AggregatedCandles, GpuError> {
        let n_trades = timestamps.len();

        // Validate inputs
        if prices.len() != n_trades || volumes.len() != n_trades {
            return Err(GpuError::InvalidParameter(
                "timestamps, prices, and volumes must have same length".to_string(),
            ));
        }

        if timeframe_ms <= 0 {
            return Err(GpuError::InvalidParameter(
                "timeframe_ms must be positive".to_string(),
            ));
        }

        if n_trades == 0 {
            return Ok(empty_candles());
        }

        if n_trades > i32::MAX as usize {
            return Err(GpuError::InvalidParameter(format!(
                "n_trades {} exceeds i32::MAX kernel index range",
                n_trades
            )));
        }

        // ====================================================================
        // STEP 1: Transfer inputs to GPU
        //
        // Prices/volumes are uploaded as f32 directly (no f64 widening: it
        // halved nothing but doubled H2D volume, and Ada runs FP64 at 1/64
        // the FP32 rate so device math stays f32 as well).
        // ====================================================================

        let d_timestamps = self.copy_i64_to_device(timestamps)?;
        let d_prices = self.copy_f32_to_device(prices)?;
        let d_volumes = self.copy_f32_to_device(volumes)?;

        let n_trades_i32 = n_trades as i32;
        let threads_per_block = THREADS_PER_BLOCK;
        let blocks_per_grid = n_trades.div_ceil(threads_per_block);

        // ====================================================================
        // STEP 2: Sortedness check + bucket range (two tiny readbacks)
        // ====================================================================

        let mut d_unsorted_flag = self.device.stream.alloc_zeros::<i32>(1).map_err(|e| {
            GpuError::AllocationError(format!("Failed to allocate sorted flag: {:?}", e))
        })?;

        // Reduction identities; overwritten by atomicMin/atomicMax.
        let mut d_first_bucket = self.copy_i64_to_device(&[i64::MAX])?;
        let mut d_last_bucket = self.copy_i64_to_device(&[i64::MIN])?;

        let cfg_per_trade = LaunchConfig {
            grid_dim: (blocks_per_grid as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.device.stream.launch_builder(&self.check_sorted_kernel);
        unsafe {
            builder
                .arg(&d_timestamps)
                .arg(&mut d_unsorted_flag)
                .arg(&n_trades_i32)
                .launch(cfg_per_trade)
                .map_err(|e| {
                    GpuError::ExecutionError(format!("check_sorted_kernel launch failed: {:?}", e))
                })?;
        }

        let range_blocks = blocks_per_grid.min(BUCKET_RANGE_MAX_BLOCKS);
        let cfg_range = LaunchConfig {
            grid_dim: (range_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.device.stream.launch_builder(&self.bucket_range_kernel);
        unsafe {
            builder
                .arg(&d_timestamps)
                .arg(&n_trades_i32)
                .arg(&timeframe_ms)
                .arg(&mut d_first_bucket)
                .arg(&mut d_last_bucket)
                .launch(cfg_range)
                .map_err(|e| {
                    GpuError::ExecutionError(format!(
                        "compute_bucket_range_kernel launch failed: {:?}",
                        e
                    ))
                })?;
        }

        // Batch the three scalar readbacks behind one synchronize.
        let mut unsorted_flag = [0i32; 1];
        let mut first_bucket_host = [0i64; 1];
        let mut last_bucket_host = [0i64; 1];

        self.copy_to_host_into(&d_unsorted_flag, &mut unsorted_flag[..])?;
        self.copy_to_host_into(&d_first_bucket, &mut first_bucket_host[..])?;
        self.copy_to_host_into(&d_last_bucket, &mut last_bucket_host[..])?;
        self.synchronize()?;

        if unsorted_flag[0] != 0 {
            // Exchange feeds are sorted; tolerate unsorted input via the CPU
            // hash aggregator (identical semantics, no sortedness assumption).
            eprintln!(
                "GPU tick aggregation: input not timestamp-sorted, using CPU fallback ({} trades)",
                n_trades
            );
            return Ok(aggregate_on_cpu(timestamps, prices, volumes, timeframe_ms));
        }

        let first_bucket = first_bucket_host[0];
        let last_bucket = last_bucket_host[0];

        // Sorted, non-empty input guarantees last_bucket >= first_bucket;
        // checked math guards degenerate timestamp extremes.
        let n_candles = match last_bucket
            .checked_sub(first_bucket)
            .and_then(|span| span.checked_add(1))
        {
            Some(n) if n > 0 && n <= MAX_DENSE_CANDLES => n as usize,
            _ => {
                // Dense range too large (sparse data) or arithmetic overflow:
                // dense GPU buffers would be oversized, aggregate on CPU.
                eprintln!(
                    "GPU tick aggregation: dense candle range exceeds {} slots, using CPU fallback",
                    MAX_DENSE_CANDLES
                );
                return Ok(aggregate_on_cpu(timestamps, prices, volumes, timeframe_ms));
            }
        };

        // ====================================================================
        // STEP 3: Allocate output buffers + initialize high/low identities
        // ====================================================================

        let mut d_out_timestamps =
            self.device
                .stream
                .alloc_zeros::<i64>(n_candles)
                .map_err(|e| {
                    GpuError::AllocationError(format!(
                        "Failed to allocate {} i64 timestamps: {:?}",
                        n_candles, e
                    ))
                })?;

        let mut d_out_open = self.alloc_f32(n_candles)?;
        let mut d_out_close = self.alloc_f32(n_candles)?;
        let mut d_out_volume = self.alloc_f32(n_candles)?;

        // Encoded high/low buffers: zero-fill is NOT a valid reduction
        // identity for the ordered-uint min/max, so an init kernel writes
        // encoded(-inf)/encoded(+inf) before aggregation.
        let mut d_out_high_enc = self.alloc_u32(n_candles)?;
        let mut d_out_low_enc = self.alloc_u32(n_candles)?;

        let mut d_out_num_trades =
            self.device
                .stream
                .alloc_zeros::<i32>(n_candles)
                .map_err(|e| {
                    GpuError::AllocationError(format!(
                        "Failed to allocate {} i32 num_trades: {:?}",
                        n_candles, e
                    ))
                })?;

        let n_candles_i32 = n_candles as i32;
        let candle_blocks = n_candles.div_ceil(threads_per_block);
        let cfg_per_candle = LaunchConfig {
            grid_dim: (candle_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.device.stream.launch_builder(&self.init_extrema_kernel);
        unsafe {
            builder
                .arg(&mut d_out_high_enc)
                .arg(&mut d_out_low_enc)
                .arg(&n_candles_i32)
                .launch(cfg_per_candle)
                .map_err(|e| {
                    GpuError::ExecutionError(format!(
                        "init_ohlcv_extrema_kernel launch failed: {:?}",
                        e
                    ))
                })?;
        }

        // ====================================================================
        // STEP 4: Single-pass sorted aggregation
        // ====================================================================

        let mut builder = self
            .device
            .stream
            .launch_builder(&self.aggregate_sorted_kernel);
        unsafe {
            builder
                .arg(&d_timestamps)
                .arg(&d_prices)
                .arg(&d_volumes)
                .arg(&n_trades_i32)
                .arg(&timeframe_ms)
                .arg(&first_bucket)
                .arg(&mut d_out_timestamps)
                .arg(&mut d_out_open)
                .arg(&mut d_out_high_enc)
                .arg(&mut d_out_low_enc)
                .arg(&mut d_out_close)
                .arg(&mut d_out_volume)
                .arg(&mut d_out_num_trades)
                .launch(cfg_per_trade)
                .map_err(|e| {
                    GpuError::ExecutionError(format!(
                        "aggregate_ohlcv_sorted_kernel launch failed: {:?}",
                        e
                    ))
                })?;
        }

        // ====================================================================
        // STEP 5: Batch all D2H copies behind a single synchronize
        // ====================================================================

        let mut out_timestamps = vec![0i64; n_candles];
        let mut out_open = vec![0f32; n_candles];
        let mut out_high_enc = vec![0u32; n_candles];
        let mut out_low_enc = vec![0u32; n_candles];
        let mut out_close = vec![0f32; n_candles];
        let mut out_volume = vec![0f32; n_candles];
        let mut out_num_trades = vec![0i32; n_candles];

        self.copy_to_host_into(&d_out_timestamps, &mut out_timestamps[..])?;
        self.copy_to_host_into(&d_out_open, &mut out_open[..])?;
        self.copy_to_host_into(&d_out_high_enc, &mut out_high_enc[..])?;
        self.copy_to_host_into(&d_out_low_enc, &mut out_low_enc[..])?;
        self.copy_to_host_into(&d_out_close, &mut out_close[..])?;
        self.copy_to_host_into(&d_out_volume, &mut out_volume[..])?;
        self.copy_to_host_into(&d_out_num_trades, &mut out_num_trades[..])?;
        self.synchronize()?;

        // ====================================================================
        // STEP 6: Filter empty dense slots and decode high/low
        // ====================================================================

        Ok(build_candles_from_dense(
            &out_timestamps,
            &out_open,
            &out_high_enc,
            &out_low_enc,
            &out_close,
            &out_volume,
            &out_num_trades,
        ))
    }

    // ========================================================================
    // Helper Methods (Memory Transfers)
    // ========================================================================

    /// Copy i64 array from host to device
    fn copy_i64_to_device(&self, data: &[i64]) -> Result<CudaSlice<i64>, GpuError> {
        let n = data.len();
        self.device.stream.memcpy_stod(data).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy {} i64 to device: {:?}", n, e))
        })
    }

    /// Copy f32 array from host to device
    fn copy_f32_to_device(&self, data: &[f32]) -> Result<CudaSlice<f32>, GpuError> {
        let n = data.len();
        self.device.stream.memcpy_stod(data).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy {} f32 to device: {:?}", n, e))
        })
    }

    /// Allocate zeroed f32 device buffer
    fn alloc_f32(&self, len: usize) -> Result<CudaSlice<f32>, GpuError> {
        self.device.stream.alloc_zeros::<f32>(len).map_err(|e| {
            GpuError::AllocationError(format!("Failed to allocate {} f32 elements: {:?}", len, e))
        })
    }

    /// Allocate u32 device buffer (zeroed; semantic init done by kernel)
    fn alloc_u32(&self, len: usize) -> Result<CudaSlice<u32>, GpuError> {
        self.device.stream.alloc_zeros::<u32>(len).map_err(|e| {
            GpuError::AllocationError(format!("Failed to allocate {} u32 elements: {:?}", len, e))
        })
    }

    /// Issue a stream-ordered D2H copy into a host slice (NOT synchronized;
    /// callers batch copies and synchronize once)
    fn copy_to_host_into<T: cudarc::driver::DeviceRepr>(
        &self,
        src: &CudaSlice<T>,
        dst: &mut [T],
    ) -> Result<(), GpuError> {
        let n = dst.len();
        self.device.stream.memcpy_dtoh(src, dst).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy {} elements to host: {:?}", n, e))
        })
    }

    /// Synchronize the stream
    fn synchronize(&self) -> Result<(), GpuError> {
        self.device
            .stream
            .synchronize()
            .map_err(|e| GpuError::SynchronizationError(format!("Stream sync failed: {:?}", e)))
    }
}

/// Empty aggregation result
fn empty_candles() -> AggregatedCandles {
    AggregatedCandles {
        timestamps: Vec::new(),
        open: Vec::new(),
        high: Vec::new(),
        low: Vec::new(),
        close: Vec::new(),
        volume: Vec::new(),
        num_trades: Vec::new(),
        num_candles: 0,
    }
}

/// Decode the ordered-uint image of an f32 produced by the GPU kernels.
///
/// Inverse of `float_to_ordered_uint` in `kernels/tick_aggregation.cu`
/// (layout contract - keep in sync): values with the top bit set came from
/// non-negative floats (`bits | 0x80000000`), the rest from negative floats
/// (`!bits`).
#[inline]
fn decode_ordered_u32(encoded: u32) -> f32 {
    if encoded & 0x8000_0000 != 0 {
        f32::from_bits(encoded & 0x7FFF_FFFF)
    } else {
        f32::from_bits(!encoded)
    }
}

/// Host mirror of the CUDA `float_to_ordered_uint` transform (used by tests
/// to validate monotonicity and the init-identity constants).
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn encode_ordered_u32(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

/// Build the final candle list from dense GPU output slots.
///
/// Filters empty slots (`num_trades == 0`, i.e. buckets with no trades in
/// the dense `[first_bucket, last_bucket]` range) and decodes the encoded
/// high/low values. Dense slots are bucket-ordered, so the result is sorted
/// by timestamp, matching the CPU reference.
fn build_candles_from_dense(
    timestamps: &[i64],
    open: &[f32],
    high_enc: &[u32],
    low_enc: &[u32],
    close: &[f32],
    volume: &[f32],
    num_trades: &[i32],
) -> AggregatedCandles {
    let dense_len = timestamps.len();
    let non_empty = num_trades.iter().filter(|&&n| n > 0).count();

    let mut result = AggregatedCandles {
        timestamps: Vec::with_capacity(non_empty),
        open: Vec::with_capacity(non_empty),
        high: Vec::with_capacity(non_empty),
        low: Vec::with_capacity(non_empty),
        close: Vec::with_capacity(non_empty),
        volume: Vec::with_capacity(non_empty),
        num_trades: Vec::with_capacity(non_empty),
        num_candles: non_empty,
    };

    for c in 0..dense_len {
        if num_trades[c] > 0 {
            result.timestamps.push(timestamps[c]);
            result.open.push(open[c]);
            result.high.push(decode_ordered_u32(high_enc[c]));
            result.low.push(decode_ordered_u32(low_enc[c]));
            result.close.push(close[c]);
            result.volume.push(volume[c]);
            result.num_trades.push(num_trades[c]);
        }
    }

    result
}

/// CPU fallback aggregator (unsorted or pathologically sparse input).
///
/// Numerical semantics match `crate::binance::aggregate_trades_to_candles`
/// exactly: candle timestamp is `(ts / timeframe_ms) * timeframe_ms`, open
/// is the first and close the last trade in *stream* order, volume
/// accumulates in f64, and candles are sorted by timestamp with empty
/// buckets omitted.
fn aggregate_on_cpu(
    timestamps: &[i64],
    prices: &[f32],
    volumes: &[f32],
    timeframe_ms: i64,
) -> AggregatedCandles {
    use std::collections::HashMap;

    struct Builder {
        timestamp: i64,
        open: f32,
        high: f32,
        low: f32,
        close: f32,
        volume: f64,
        num_trades: i32,
    }

    let estimated_candles = (timestamps.len() / 1000).max(1);
    let mut builders: HashMap<i64, Builder> = HashMap::with_capacity(estimated_candles);

    for i in 0..timestamps.len() {
        let candle_timestamp = (timestamps[i] / timeframe_ms) * timeframe_ms;
        let price = prices[i];
        let volume = volumes[i] as f64;

        builders
            .entry(candle_timestamp)
            .and_modify(|b| {
                if price > b.high {
                    b.high = price;
                }
                if price < b.low {
                    b.low = price;
                }
                b.close = price; // Last trade in stream order becomes close
                b.volume += volume;
                b.num_trades += 1;
            })
            .or_insert_with(|| Builder {
                timestamp: candle_timestamp,
                open: price,
                high: price,
                low: price,
                close: price,
                volume,
                num_trades: 1,
            });
    }

    let mut sorted: Vec<Builder> = builders.into_values().collect();
    sorted.sort_unstable_by_key(|b| b.timestamp);

    let n = sorted.len();
    let mut result = AggregatedCandles {
        timestamps: Vec::with_capacity(n),
        open: Vec::with_capacity(n),
        high: Vec::with_capacity(n),
        low: Vec::with_capacity(n),
        close: Vec::with_capacity(n),
        volume: Vec::with_capacity(n),
        num_trades: Vec::with_capacity(n),
        num_candles: n,
    };

    for b in sorted {
        result.timestamps.push(b.timestamp);
        result.open.push(b.open);
        result.high.push(b.high);
        result.low.push(b.low);
        result.close.push(b.close);
        result.volume.push(b.volume as f32);
        result.num_trades.push(b.num_trades);
    }

    result
}

/// Compile CUDA kernels for tick aggregation (cached).
///
/// Uses `compile_ptx_optimized_cached` so repeated `TickAggregator::new()`
/// calls skip NVRTC compilation entirely (50-200x faster on cache hits).
fn compile_tick_kernels() -> Result<cudarc::nvrtc::Ptx, GpuError> {
    let kernel_src = include_str!("kernels/tick_aggregation.cu");

    let ptx_arc = super::compile::compile_ptx_optimized_cached(kernel_src).map_err(|e| {
        GpuError::CompilationError(format!(
            "Failed to compile tick aggregation kernels: {:?}",
            e
        ))
    })?;

    Ok(std::sync::Arc::unwrap_or_clone(ptx_arc))
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL_SRC: &str = include_str!("kernels/tick_aggregation.cu");

    /// Runtime GPU gate: returns None (test prints skip notice) when no
    /// CUDA device is present, so GPU CI runs these tests by default.
    fn gpu_test_aggregator() -> Option<TickAggregator> {
        let device = GpuDevice::new().ok()?;
        TickAggregator::new(device).ok()
    }

    // ========================================================================
    // Host-side tests (no GPU required)
    // ========================================================================

    #[test]
    fn test_kernel_source_is_nvrtc_compatible() {
        assert!(
            !KERNEL_SRC.contains("#include"),
            "NVRTC kernels must not use include directives"
        );

        for name in [
            "bin_trades_kernel",
            "check_sorted_kernel",
            "compute_bucket_range_kernel",
            "init_ohlcv_extrema_kernel",
            "aggregate_ohlcv_sorted_kernel",
            "quantize_to_int8_kernel",
            "dequantize_from_int8_kernel",
        ] {
            assert!(
                KERNEL_SRC.contains(&format!("extern \"C\" __global__ void {}", name)),
                "missing extern \"C\" __global__ entry point: {}",
                name
            );
        }

        // f32-only price path (Ada FP64 runs at 1/64 the FP32 rate)
        assert!(
            !KERNEL_SRC.contains("double"),
            "tick aggregation kernels must not use FP64"
        );

        // Shared-memory kernels previously failed PTX JIT loading on sm_89
        assert!(
            !KERNEL_SRC.contains("__shared__"),
            "tick aggregation kernels must not use shared memory"
        );

        // 64-bit warp shuffles must go through the 32-bit split helper: the
        // long long __shfl_down_sync overloads are SDK-header inline functions
        // not guaranteed by NVRTC's builtin preamble.
        assert!(KERNEL_SRC.contains("shfl_down_i64(0xFFFFFFFFu, local_min, offset)"));
        assert!(KERNEL_SRC.contains("shfl_down_i64(0xFFFFFFFFu, local_max, offset)"));
        assert!(
            !KERNEL_SRC.contains("__shfl_down_sync(0xFFFFFFFFu, local_"),
            "shuffle i64 values via shfl_down_i64, not a direct 64-bit __shfl_down_sync"
        );

        // Racy CAS helpers and the dead hash kernel must stay deleted
        assert!(!KERNEL_SRC.contains("atomicMaxDouble"));
        assert!(!KERNEL_SRC.contains("atomicMinDouble"));
        assert!(!KERNEL_SRC.contains("atomicMinTimestampAndPrice"));
        assert!(!KERNEL_SRC.contains("atomicMaxTimestampAndPrice"));
        assert!(!KERNEL_SRC.contains("aggregate_ohlcv_hash_kernel"));
    }

    #[test]
    fn test_kernel_source_quantize_uses_raw_0_255_convention() {
        // Raw 0-255 codes carried through the i8 ABI (no saturating cast)
        assert!(KERNEL_SRC.contains("(int8_t)(unsigned char)"));
        // Dequantize must read the code back through unsigned char
        assert!(KERNEL_SRC.contains("(unsigned char)in_values[idx]"));
        // Round-to-nearest, not truncation
        assert!(KERNEL_SRC.contains("__float2int_rn"));
    }

    #[test]
    fn test_ordered_uint_encoding_constants_match_kernel() {
        // Host constants mirror the CUDA defines (layout contract)
        assert!(KERNEL_SRC.contains("#define ENCODED_NEG_INF 0x007FFFFFu"));
        assert!(KERNEL_SRC.contains("#define ENCODED_POS_INF 0xFF800000u"));

        assert_eq!(encode_ordered_u32(f32::NEG_INFINITY), ENCODED_NEG_INF);
        assert_eq!(encode_ordered_u32(f32::INFINITY), ENCODED_POS_INF);
        assert_eq!(decode_ordered_u32(ENCODED_NEG_INF), f32::NEG_INFINITY);
        assert_eq!(decode_ordered_u32(ENCODED_POS_INF), f32::INFINITY);
    }

    #[test]
    fn test_ordered_uint_encoding_is_monotonic() {
        // Strictly increasing floats (including negatives and signed zero)
        let sorted = [
            f32::NEG_INFINITY,
            f32::MIN,
            -1.0e30,
            -3.5,
            -1.0e-20,
            -0.0,
            0.0,
            1.0e-20,
            2.5,
            1.0e30,
            f32::MAX,
            f32::INFINITY,
        ];

        for pair in sorted.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert!(
                encode_ordered_u32(a) < encode_ordered_u32(b),
                "encoding not monotonic for {} < {}: {:#x} >= {:#x}",
                a,
                b,
                encode_ordered_u32(a),
                encode_ordered_u32(b)
            );
        }
    }

    #[test]
    fn test_ordered_uint_encoding_roundtrips() {
        let values = [
            f32::NEG_INFINITY,
            -12345.678,
            -1.0,
            -0.0,
            0.0,
            1.0,
            100.25,
            98765.4,
            f32::INFINITY,
        ];

        for &v in &values {
            let decoded = decode_ordered_u32(encode_ordered_u32(v));
            assert_eq!(
                decoded.to_bits(),
                v.to_bits(),
                "round-trip changed bits for {}",
                v
            );
        }
    }

    #[test]
    fn test_cpu_fallback_unsorted_single_bucket() {
        // Unsorted timestamps within one 60s candle: open/close follow
        // STREAM order, matching binance::aggregate_trades_to_candles.
        let timestamps = vec![3000i64, 1000, 2000];
        let prices = vec![10.0f32, 12.0, 11.0];
        let volumes = vec![1.0f32, 2.0, 3.0];

        let candles = aggregate_on_cpu(&timestamps, &prices, &volumes, 60_000);

        assert_eq!(candles.num_candles, 1);
        assert_eq!(candles.timestamps, vec![0]);
        assert_eq!(candles.open[0], 10.0); // First in stream order
        assert_eq!(candles.close[0], 11.0); // Last in stream order
        assert_eq!(candles.high[0], 12.0);
        assert_eq!(candles.low[0], 10.0);
        assert_eq!(candles.volume[0], 6.0);
        assert_eq!(candles.num_trades[0], 3);
    }

    #[test]
    fn test_cpu_fallback_multi_bucket_with_gap() {
        let timeframe_ms = 300_000i64;
        let base = 1_609_459_200_000i64;

        // Buckets 0 and 2 populated; bucket 1 empty (must be omitted)
        let timestamps = vec![
            base,
            base + 10_000,
            base + 2 * timeframe_ms,
            base + 2 * timeframe_ms + 5_000,
        ];
        let prices = vec![100.0f32, 105.0, 120.0, 118.0];
        let volumes = vec![1.0f32, 2.0, 1.5, 0.5];

        let candles = aggregate_on_cpu(&timestamps, &prices, &volumes, timeframe_ms);

        assert_eq!(candles.num_candles, 2);
        assert_eq!(candles.timestamps, vec![base, base + 2 * timeframe_ms]);

        assert_eq!(candles.open[0], 100.0);
        assert_eq!(candles.high[0], 105.0);
        assert_eq!(candles.low[0], 100.0);
        assert_eq!(candles.close[0], 105.0);
        assert_eq!(candles.volume[0], 3.0);
        assert_eq!(candles.num_trades[0], 2);

        assert_eq!(candles.open[1], 120.0);
        assert_eq!(candles.high[1], 120.0);
        assert_eq!(candles.low[1], 118.0);
        assert_eq!(candles.close[1], 118.0); // Last trade in stream order
        assert_eq!(candles.volume[1], 2.0);
        assert_eq!(candles.num_trades[1], 2);
    }

    #[test]
    fn test_build_candles_filters_empty_dense_slots() {
        let timeframe_ms = 300_000i64;
        // Three dense slots: [populated, empty, populated]
        let timestamps = vec![0, 0, 2 * timeframe_ms];
        let open = vec![100.0f32, 0.0, 120.0];
        let high_enc = vec![
            encode_ordered_u32(105.0),
            ENCODED_NEG_INF, // Untouched init identity in the empty slot
            encode_ordered_u32(125.0),
        ];
        let low_enc = vec![
            encode_ordered_u32(99.5),
            ENCODED_POS_INF,
            encode_ordered_u32(118.0),
        ];
        let close = vec![102.0f32, 0.0, 124.0];
        let volume = vec![4.5f32, 0.0, 7.0];
        let num_trades = vec![3i32, 0, 5];

        let candles = build_candles_from_dense(
            &timestamps,
            &open,
            &high_enc,
            &low_enc,
            &close,
            &volume,
            &num_trades,
        );

        assert_eq!(candles.num_candles, 2);
        assert_eq!(candles.timestamps, vec![0, 2 * timeframe_ms]);
        assert_eq!(candles.open, vec![100.0, 120.0]);
        assert_eq!(candles.high, vec![105.0, 125.0]);
        assert_eq!(candles.low, vec![99.5, 118.0]);
        assert_eq!(candles.close, vec![102.0, 124.0]);
        assert_eq!(candles.volume, vec![4.5, 7.0]);
        assert_eq!(candles.num_trades, vec![3, 5]);
    }

    #[test]
    fn test_is_available() {
        let available = TickAggregator::is_available();
        println!("GPU tick aggregation available: {}", available);
    }

    // ========================================================================
    // GPU tests (runtime-gated or #[ignore])
    // ========================================================================

    #[test]
    #[ignore] // Requires GPU
    fn test_tick_aggregator_initialization() {
        let device = GpuDevice::new().expect("GPU not available");
        let result = TickAggregator::new(device);
        assert!(result.is_ok(), "Failed to initialize tick aggregator");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_aggregate_empty_trades() {
        let device = GpuDevice::new().expect("GPU not available");
        let aggregator = TickAggregator::new(device).expect("Failed to init aggregator");

        let timestamps = vec![];
        let prices = vec![];
        let volumes = vec![];
        let sides = vec![];

        let candles = aggregator
            .aggregate(&timestamps, &prices, &volumes, &sides, 300_000)
            .expect("Aggregation failed");

        assert_eq!(candles.num_candles, 0);
    }

    #[test]
    fn test_aggregate_simple_trades() {
        // GPU-gated at runtime (not #[ignore]) so GPU CI exercises the full
        // open/close/timestamp correctness path by default.
        let Some(aggregator) = gpu_test_aggregator() else {
            eprintln!("Skipping test_aggregate_simple_trades: GPU not available");
            return;
        };

        // Create 10 trades spanning 3 candles (5-minute timeframe)
        let timestamps = vec![
            1609459200000, // Candle 0 (00:00:00)
            1609459210000, // Candle 0 (00:00:10)
            1609459220000, // Candle 0 (00:00:20)
            1609459500000, // Candle 1 (00:05:00)
            1609459510000, // Candle 1 (00:05:10)
            1609459800000, // Candle 2 (00:10:00)
            1609459810000, // Candle 2 (00:10:10)
            1609459820000, // Candle 2 (00:10:20)
            1609459830000, // Candle 2 (00:10:30)
            1609459840000, // Candle 2 (00:10:40)
        ];

        let prices = vec![
            100.0, 105.0, 102.0, // Candle 0: open=100, high=105, low=100, close=102
            110.0, 115.0, // Candle 1: open=110, high=115, low=110, close=115
            120.0, 125.0, 118.0, 122.0,
            124.0, // Candle 2: open=120, high=125, low=118, close=124
        ];

        let volumes = vec![1.0, 2.0, 1.5, 1.0, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5];

        let sides = vec![1, 1, -1, 1, 1, 1, 1, -1, 1, 1];

        let candles = aggregator
            .aggregate(&timestamps, &prices, &volumes, &sides, 300_000)
            .expect("Aggregation failed");

        assert_eq!(candles.num_candles, 3, "Should have 3 candles");

        // Candle open times (bucket-aligned)
        assert_eq!(
            candles.timestamps,
            vec![1609459200000, 1609459500000, 1609459800000]
        );

        // Validate candle 0
        assert_eq!(candles.open[0], 100.0);
        assert_eq!(candles.high[0], 105.0);
        assert_eq!(candles.low[0], 100.0);
        assert_eq!(candles.close[0], 102.0);
        assert_eq!(candles.volume[0], 4.5);
        assert_eq!(candles.num_trades[0], 3);

        // Validate candle 1
        assert_eq!(candles.open[1], 110.0);
        assert_eq!(candles.high[1], 115.0);
        assert_eq!(candles.low[1], 110.0);
        assert_eq!(candles.close[1], 115.0);
        assert_eq!(candles.volume[1], 3.0);
        assert_eq!(candles.num_trades[1], 2);

        // Validate candle 2
        assert_eq!(candles.open[2], 120.0);
        assert_eq!(candles.high[2], 125.0);
        assert_eq!(candles.low[2], 118.0);
        assert_eq!(candles.close[2], 124.0);
        assert_eq!(candles.volume[2], 7.0);
        assert_eq!(candles.num_trades[2], 5);
    }

    #[test]
    fn test_aggregate_unsorted_uses_cpu_fallback() {
        // GPU-gated at runtime; validates the device-side sortedness check
        // routes unsorted input through the CPU aggregator.
        let Some(aggregator) = gpu_test_aggregator() else {
            eprintln!("Skipping test_aggregate_unsorted_uses_cpu_fallback: GPU not available");
            return;
        };

        let timestamps = vec![1609459220000i64, 1609459200000, 1609459210000];
        let prices = vec![102.0f32, 100.0, 105.0];
        let volumes = vec![1.5f32, 1.0, 2.0];
        let sides = vec![1i8, 1, -1];

        let candles = aggregator
            .aggregate(&timestamps, &prices, &volumes, &sides, 300_000)
            .expect("Aggregation failed");

        assert_eq!(candles.num_candles, 1);
        assert_eq!(candles.timestamps, vec![1609459200000]);
        assert_eq!(candles.open[0], 102.0); // Stream order, not time order
        assert_eq!(candles.close[0], 105.0);
        assert_eq!(candles.high[0], 105.0);
        assert_eq!(candles.low[0], 100.0);
        assert_eq!(candles.volume[0], 4.5);
        assert_eq!(candles.num_trades[0], 3);
    }

    #[test]
    #[ignore] // Requires GPU - benchmark test
    fn test_aggregate_performance() {
        use std::time::Instant;

        let device = GpuDevice::new().expect("GPU not available");
        let aggregator = TickAggregator::new(device).expect("Failed to init aggregator");

        // Generate 1M trades
        let n_trades = 1_000_000;
        let mut timestamps = Vec::with_capacity(n_trades);
        let mut prices = Vec::with_capacity(n_trades);
        let mut volumes = Vec::with_capacity(n_trades);
        let mut sides = Vec::with_capacity(n_trades);

        let base_ts = 1609459200000i64; // 2021-01-01 00:00:00
        for i in 0..n_trades {
            timestamps.push(base_ts + (i as i64) * 1000); // 1 trade per second
            prices.push(100.0 + ((i % 100) as f32) * 0.1); // Varying prices
            volumes.push(1.0 + ((i % 10) as f32) * 0.1);
            sides.push(if i % 2 == 0 { 1 } else { -1 });
        }

        // Warm-up (JIT compilation)
        let _ = aggregator
            .aggregate(
                &timestamps[..1000],
                &prices[..1000],
                &volumes[..1000],
                &sides[..1000],
                300_000,
            )
            .expect("Warm-up failed");

        // Benchmark
        let start = Instant::now();
        let candles = aggregator
            .aggregate(&timestamps, &prices, &volumes, &sides, 300_000)
            .expect("Aggregation failed");
        let duration = start.elapsed();

        let throughput = (n_trades as f64) / duration.as_secs_f64();

        println!(
            "Aggregated {} trades into {} candles",
            n_trades, candles.num_candles
        );
        println!("Duration: {:?}", duration);
        println!("Throughput: {:.2} trades/sec", throughput);
        println!("Throughput: {:.2} M trades/sec", throughput / 1_000_000.0);

        // Correctness: the aggregation must produce the right number of candles.
        // The impl buckets by `bucket = ts / timeframe_ms` over the dense range
        // [first_bucket, last_bucket], so derive the expected count the same way
        // from the test's own timestamps (robust to epoch alignment).
        let timeframe_ms = 300_000i64;
        let first_bucket = timestamps[0] / timeframe_ms;
        let last_bucket = timestamps[n_trades - 1] / timeframe_ms;
        let expected_candles = (last_bucket - first_bucket + 1) as usize;
        assert_eq!(
            candles.num_candles, expected_candles,
            "Unexpected candle count: got {}, expected {}",
            candles.num_candles, expected_candles
        );
        // The absolute trades/sec figure above is printed for reference but NOT
        // asserted: a wall-clock throughput threshold is machine-dependent (GPU
        // model, clocks, contention) and was flaky in CI/hardware runs. This is a
        // perf-classification target, not a correctness invariant.
        assert!(
            throughput.is_finite() && throughput > 0.0,
            "Throughput must be a positive, finite value, got {:.2}",
            throughput
        );
    }
}
