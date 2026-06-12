//! GPU Orderflow Feature Extraction + Signal Generation (Fused Kernels)
//!
//! High-performance batch processing of orderflow features and trading signals.
//! Features are computed ONCE per tick (not once per strategy) and immediately
//! consumed by a fused per-strategy epilogue (signals + INT8 quantization),
//! eliminating the intermediate feature write/read.
//!
//! # Normative reference
//!
//! `rust/src/cpu/orderflow.rs` defines the ground-truth numerical semantics.
//! The kernel mirrors `extract_features` line by line:
//!
//! | # | Feature                  | Definition                                       |
//! |---|--------------------------|--------------------------------------------------|
//! | 0 | buy_sell_imbalance       | buy / (buy + sell); 0.5 when total == 0          |
//! | 1 | volume_delta             | buy - sell (per-tick)                            |
//! | 2 | trade_intensity          | volume / (max(ts[i]-ts[i-1], 1) ms as seconds)   |
//! | 3 | price_velocity           | z-score of close over trailing 20-tick window    |
//! | 4 | volume_velocity          | z-score of volume over trailing 20-tick window   |
//! | 5 | cumulative_volume_delta  | inclusive prefix sum of volume_delta             |
//!
//! # Architecture (3 deterministic GPU passes)
//!
//! Feature 5 is a global prefix sum, so processing runs as:
//!
//! 1. `orderflow_block_scan_kernel` — block-local inclusive scan of
//!    (buy - sell) + per-block totals
//! 2. `orderflow_scan_block_sums_kernel` — single block turns the totals into
//!    exclusive block prefixes
//! 3. `orderflow_features_signals_kernel` (or
//!    `calibrate_feature_ranges_kernel`) — chunk + halo feature computation,
//!    fused per-strategy epilogue (the block-prefix fixup is folded in here)
//!
//! Passes 1 and 3 share the tick->block mapping: [`TICKS_PER_BLOCK`]
//! contiguous ticks per block (this module owns both launch configurations).
//!
//! # INT8 convention
//!
//! Quantized feature codes are raw 0-255 values stored through the `i8` ABI
//! (`(code as u8) as i8` — bit pattern preserved, codes >= 128 read negative).
//! Scale is `255 / (max - min)`, 0.0 for degenerate ranges (<= 1e-9) so the
//! code collapses to 0 and dequantizes to `min`. This matches
//! `gpu/quantization.rs` / `kernels/quantize_int8.cu` exactly.
//!
//! Strategies sharing identical (mins, maxs) ranges are deduplicated into
//! range groups before launch; the kernel quantizes once per group and the
//! host broadcasts group rows back to the stable per-strategy output shape
//! (all 5 default [`StrategyConfig`]s share one group).
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::orderflow_batch::{OrderflowBatchProcessor, OrderflowInput, StrategyConfig};
//! use kimsfinance_core::gpu::device::GpuDevice;
//! use std::sync::Arc;
//!
//! let device = Arc::new(GpuDevice::new()?);
//! let mut processor = OrderflowBatchProcessor::new(device)?;
//!
//! let strategies = vec![
//!     StrategyConfig::momentum(),
//!     StrategyConfig::mean_reversion(),
//!     StrategyConfig::breakout(),
//! ];
//!
//! let input = OrderflowInput {
//!     timestamps: vec![/* ms epochs */],
//!     close_prices: vec![/* f32 */],
//!     volumes: vec![/* f32 */],
//!     buy_volumes: vec![/* f32 */],
//!     sell_volumes: vec![/* f32 */],
//! };
//!
//! let results = processor.process_batch(&input, &strategies)?;
//! println!("Signals: {:?}", results.signals);   // [num_strategies][num_ticks]
//! println!("Features: {:?}", results.features); // [num_strategies][num_ticks * 6]
//! ```

use crate::gpu::device::{GpuDevice, GpuError};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// CUDA kernel source, NVRTC-JIT-compiled at runtime.
///
/// Must stay NVRTC-compatible: no `#include` directives, `extern "C"
/// __global__` entry points only (asserted by host-side tests below).
const ORDERFLOW_KERNEL_SRC: &str = include_str!("kernels/orderflow_signals_batch.cu");

/// Number of orderflow features computed per tick
pub const NUM_FEATURES: usize = 6;

/// Sliding window size for the z-score features. Mirrored by
/// `#define WINDOW_SIZE` in the kernel source (asserted in tests).
#[allow(dead_code)] // host logic no longer windows; kept for the layout-contract tests
const WINDOW_SIZE: usize = 20;

/// Ticks (= threads) per block for the scan and feature kernels. Mirrored by
/// `#define TICKS_PER_BLOCK` in the kernel source (asserted in tests); the
/// scan pass and the feature pass MUST agree on this so the per-block scan
/// prefixes line up.
const TICKS_PER_BLOCK: usize = 256;

/// Signal types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Signal {
    Hold = 0,
    Buy = 1,
    Sell = -1,
}

impl From<i8> for Signal {
    fn from(value: i8) -> Self {
        match value {
            1 => Signal::Buy,
            -1 => Signal::Sell,
            _ => Signal::Hold,
        }
    }
}

/// Strategy identifiers (hardcoded Phase 1)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum StrategyType {
    /// Simple momentum: buy when imbalance > 0.6 && volume_delta > 1000
    Momentum = 0,

    /// Mean reversion: buy when imbalance < 0.4 && volume_delta < -1000
    MeanReversion = 1,

    /// Breakout: buy when trade_intensity > 100 && price_velocity > 0.001
    Breakout = 2,

    /// Scalping: buy when imbalance > 0.55 && abs(volume_delta) < 500
    Scalping = 3,

    /// Trend following: buy when volume_delta > 5000 && price_velocity > 0.002
    TrendFollowing = 4,
}

/// Strategy configuration
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Strategy type (determines signal generation logic)
    pub strategy_type: StrategyType,

    /// Per-feature minimum values for quantization (calibrated from data)
    pub feature_mins: [f32; NUM_FEATURES],

    /// Per-feature maximum values for quantization (calibrated from data)
    pub feature_maxs: [f32; NUM_FEATURES],
}

impl StrategyConfig {
    /// Create momentum strategy with default quantization ranges
    pub fn momentum() -> Self {
        Self {
            strategy_type: StrategyType::Momentum,
            feature_mins: [0.0, -10000.0, 0.0, -1.0, -1.0, 0.0],
            feature_maxs: [1.0, 10000.0, 1000.0, 1.0, 1.0, 10000.0],
        }
    }

    /// Create mean reversion strategy with default quantization ranges
    pub fn mean_reversion() -> Self {
        Self {
            strategy_type: StrategyType::MeanReversion,
            feature_mins: [0.0, -10000.0, 0.0, -1.0, -1.0, 0.0],
            feature_maxs: [1.0, 10000.0, 1000.0, 1.0, 1.0, 10000.0],
        }
    }

    /// Create breakout strategy with default quantization ranges
    pub fn breakout() -> Self {
        Self {
            strategy_type: StrategyType::Breakout,
            feature_mins: [0.0, -10000.0, 0.0, -1.0, -1.0, 0.0],
            feature_maxs: [1.0, 10000.0, 1000.0, 1.0, 1.0, 10000.0],
        }
    }

    /// Create scalping strategy with default quantization ranges
    pub fn scalping() -> Self {
        Self {
            strategy_type: StrategyType::Scalping,
            feature_mins: [0.0, -10000.0, 0.0, -1.0, -1.0, 0.0],
            feature_maxs: [1.0, 10000.0, 1000.0, 1.0, 1.0, 10000.0],
        }
    }

    /// Create trend following strategy with default quantization ranges
    pub fn trend_following() -> Self {
        Self {
            strategy_type: StrategyType::TrendFollowing,
            feature_mins: [0.0, -10000.0, 0.0, -1.0, -1.0, 0.0],
            feature_maxs: [1.0, 10000.0, 1000.0, 1.0, 1.0, 10000.0],
        }
    }
}

/// Input data for orderflow processing (from tick aggregation)
#[derive(Debug, Clone)]
pub struct OrderflowInput {
    /// Unix timestamps in milliseconds
    pub timestamps: Vec<i64>,

    /// Close prices (OHLCV aggregated from ticks)
    pub close_prices: Vec<f32>,

    /// Total volumes
    pub volumes: Vec<f32>,

    /// Buy-side volumes (taker was buyer)
    pub buy_volumes: Vec<f32>,

    /// Sell-side volumes (taker was seller)
    pub sell_volumes: Vec<f32>,
}

impl OrderflowInput {
    /// Validate input data consistency
    pub fn validate(&self) -> Result<(), GpuError> {
        let n = self.timestamps.len();

        if n == 0 {
            return Err(GpuError::InvalidParameter("Empty input data".into()));
        }

        if self.close_prices.len() != n {
            return Err(GpuError::InvalidParameter(
                "close_prices length mismatch".into(),
            ));
        }

        if self.volumes.len() != n {
            return Err(GpuError::InvalidParameter("volumes length mismatch".into()));
        }

        if self.buy_volumes.len() != n {
            return Err(GpuError::InvalidParameter(
                "buy_volumes length mismatch".into(),
            ));
        }

        if self.sell_volumes.len() != n {
            return Err(GpuError::InvalidParameter(
                "sell_volumes length mismatch".into(),
            ));
        }

        Ok(())
    }

    /// Get number of ticks
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
}

/// Output from orderflow processing
#[derive(Debug, Clone)]
pub struct OrderflowOutput {
    /// Trading signals [num_strategies][num_ticks]
    /// i8: 1=buy, -1=sell, 0=hold
    pub signals: Vec<Vec<i8>>,

    /// Quantized features [num_strategies][num_ticks * NUM_FEATURES].
    ///
    /// Raw 0-255 codes through the i8 ABI (read back via `code as u8`).
    /// Strategies sharing identical quantization ranges receive byte-identical
    /// rows (computed once on the GPU, broadcast on the host).
    pub features: Vec<Vec<i8>>,

    /// Feature ranges used for quantization [num_strategies][NUM_FEATURES * 2]
    /// Layout: [min0, max0, min1, max1, ...]
    pub feature_ranges: Vec<[f32; NUM_FEATURES * 2]>,
}

/// `255 / (max - min)`, or 0.0 for degenerate ranges.
///
/// Same 1e-9 threshold as `quantization.rs::QuantizationCalibrator`: a zero
/// scale quantizes a constant feature to code 0, which dequantizes to `min`.
fn quantization_scale(min: f32, max: f32) -> f32 {
    let range = max - min;
    if range > 1e-9 {
        255.0 / range
    } else {
        0.0
    }
}

/// Unique quantization-range groups across strategies.
///
/// `group_mins` / `group_scales` are flattened `[num_groups * NUM_FEATURES]`
/// in first-occurrence order; `group_of[s]` maps strategy `s` to its group.
struct RangeGroups {
    group_of: Vec<usize>,
    group_mins: Vec<f32>,
    group_scales: Vec<f32>,
    num_groups: usize,
}

/// Deduplicate strategies by exact (bit-pattern) equality of their
/// quantization ranges so the kernel quantizes once per unique range set.
/// All 5 default [`StrategyConfig`]s collapse to a single group.
fn group_strategy_ranges(strategies: &[StrategyConfig]) -> RangeGroups {
    let mut keys: Vec<[u32; NUM_FEATURES * 2]> = Vec::new();
    let mut group_of = Vec::with_capacity(strategies.len());
    let mut group_mins = Vec::new();
    let mut group_scales = Vec::new();

    for strategy in strategies {
        let mut key = [0u32; NUM_FEATURES * 2];
        for i in 0..NUM_FEATURES {
            key[i] = strategy.feature_mins[i].to_bits();
            key[NUM_FEATURES + i] = strategy.feature_maxs[i].to_bits();
        }

        let idx = match keys.iter().position(|k| *k == key) {
            Some(idx) => idx,
            None => {
                keys.push(key);
                for i in 0..NUM_FEATURES {
                    group_mins.push(strategy.feature_mins[i]);
                    group_scales.push(quantization_scale(
                        strategy.feature_mins[i],
                        strategy.feature_maxs[i],
                    ));
                }
                keys.len() - 1
            }
        };
        group_of.push(idx);
    }

    RangeGroups {
        group_of,
        group_mins,
        group_scales,
        num_groups: keys.len(),
    }
}

/// `ceil(num_ticks / TICKS_PER_BLOCK)`: the tick->block mapping shared by the
/// scan pass and the feature pass (block prefixes are indexed by this).
fn num_tick_blocks(num_ticks: usize) -> usize {
    num_ticks.div_ceil(TICKS_PER_BLOCK)
}

/// GPU batch processor for orderflow features and signals
pub struct OrderflowBatchProcessor {
    device: Arc<GpuDevice>,
}

impl OrderflowBatchProcessor {
    /// Create new processor with GPU device.
    ///
    /// Kernels are NVRTC-compiled and module-cached on first use via
    /// [`GpuDevice::get_or_load_function`].
    pub fn new(device: Arc<GpuDevice>) -> Result<Self, GpuError> {
        Ok(Self { device })
    }

    /// Passes 1 + 2 of the cumulative-volume-delta scan.
    ///
    /// Returns `(cum_delta_partial [num_ticks], block_prefixes [num_blocks])`:
    /// the per-tick block-LOCAL inclusive scan of `(buy - sell)` and the
    /// exclusive prefix of the per-block totals. The pass-3 kernels fold the
    /// fixup in as `cum[i] = cum_delta_partial[i] + block_prefixes[i / TPB]`.
    fn compute_cum_delta_scan(
        &self,
        d_buy: &CudaSlice<f32>,
        d_sell: &CudaSlice<f32>,
        num_ticks: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), GpuError> {
        let num_blocks = num_tick_blocks(num_ticks);

        let mut d_partial = self
            .device
            .stream
            .alloc_zeros::<f32>(num_ticks)
            .map_err(|e| {
                GpuError::AllocationError(format!("Failed to allocate scan partials: {:?}", e))
            })?;
        let mut d_block_sums = self
            .device
            .stream
            .alloc_zeros::<f32>(num_blocks)
            .map_err(|e| {
                GpuError::AllocationError(format!("Failed to allocate block sums: {:?}", e))
            })?;
        let mut d_block_prefixes =
            self.device
                .stream
                .alloc_zeros::<f32>(num_blocks)
                .map_err(|e| {
                    GpuError::AllocationError(format!("Failed to allocate block prefixes: {:?}", e))
                })?;

        let num_ticks_i32 = num_ticks as i32;
        let num_blocks_i32 = num_blocks as i32;

        // Pass 1: block-local inclusive scans + per-block totals
        let scan_fn = self
            .device
            .get_or_load_function(ORDERFLOW_KERNEL_SRC, "orderflow_block_scan_kernel")?;
        let scan_config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (TICKS_PER_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0, // static shared memory only
        };
        let mut builder = self.device.stream.launch_builder(&scan_fn);
        builder.arg(d_buy);
        builder.arg(d_sell);
        builder.arg(&mut d_partial);
        builder.arg(&mut d_block_sums);
        builder.arg(&num_ticks_i32);
        unsafe {
            builder.launch(scan_config).map_err(|e| {
                GpuError::ExecutionError(format!("Block scan kernel launch failed: {:?}", e))
            })?;
        }

        // Pass 2: exclusive scan of the per-block totals (single block)
        let sums_fn = self
            .device
            .get_or_load_function(ORDERFLOW_KERNEL_SRC, "orderflow_scan_block_sums_kernel")?;
        let sums_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (TICKS_PER_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.device.stream.launch_builder(&sums_fn);
        builder.arg(&d_block_sums);
        builder.arg(&mut d_block_prefixes);
        builder.arg(&num_blocks_i32);
        unsafe {
            builder.launch(sums_config).map_err(|e| {
                GpuError::ExecutionError(format!("Block-sums scan kernel launch failed: {:?}", e))
            })?;
        }

        Ok((d_partial, d_block_prefixes))
    }

    /// Calibrate feature quantization ranges from input data.
    ///
    /// Computes the 6 features over real trailing-20 windows (same chunk +
    /// halo kernel math as `process_batch`) and reduces per-feature min/max.
    ///
    /// # Returns
    ///
    /// `[min0, max0, min1, max1, ...]` for each of the 6 features. Features
    /// with no finite samples report the CPU-parity fallbacks `[0.0, 1.0]`.
    pub fn calibrate_ranges(
        &mut self,
        input: &OrderflowInput,
    ) -> Result<[f32; NUM_FEATURES * 2], GpuError> {
        input.validate()?;

        let num_ticks = input.len();
        let num_blocks = num_tick_blocks(num_ticks);

        // Transfer input data to GPU
        let d_timestamps = self.device.copy_to_device_i64(&input.timestamps)?;
        let d_close_prices = self.device.copy_to_device_f32(&input.close_prices)?;
        let d_volumes = self.device.copy_to_device_f32(&input.volumes)?;
        let d_buy_volumes = self.device.copy_to_device_f32(&input.buy_volumes)?;
        let d_sell_volumes = self.device.copy_to_device_f32(&input.sell_volumes)?;

        // Feature 5 (cumulative delta) needs the global scan even during
        // calibration — its range is meaningless otherwise.
        let (d_cum_partial, d_block_prefixes) =
            self.compute_cum_delta_scan(&d_buy_volumes, &d_sell_volumes, num_ticks)?;

        let mut d_mins = self
            .device
            .stream
            .alloc_zeros::<f32>(NUM_FEATURES)
            .map_err(|e| GpuError::AllocationError(format!("Failed to allocate mins: {:?}", e)))?;
        let mut d_maxs = self
            .device
            .stream
            .alloc_zeros::<f32>(NUM_FEATURES)
            .map_err(|e| GpuError::AllocationError(format!("Failed to allocate maxs: {:?}", e)))?;

        // Initialize to +inf/-inf identity elements. The zeros from
        // alloc_zeros would clamp mins <= 0 and maxs >= 0, which is wrong for
        // the signed features (z-scores and both deltas).
        let init_fn = self
            .device
            .get_or_load_function(ORDERFLOW_KERNEL_SRC, "init_calibration_ranges_kernel")?;
        let init_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1), // NUM_FEATURES <= 32
            shared_mem_bytes: 0,
        };
        let num_features_i32 = NUM_FEATURES as i32;
        let mut builder = self.device.stream.launch_builder(&init_fn);
        builder.arg(&mut d_mins);
        builder.arg(&mut d_maxs);
        builder.arg(&num_features_i32);
        unsafe {
            builder.launch(init_config).map_err(|e| {
                GpuError::ExecutionError(format!("Range init kernel launch failed: {:?}", e))
            })?;
        }

        // Pass 3b: feature computation + per-feature min/max reduction
        let calib_fn = self
            .device
            .get_or_load_function(ORDERFLOW_KERNEL_SRC, "calibrate_feature_ranges_kernel")?;
        let calib_config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (TICKS_PER_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let num_ticks_i32 = num_ticks as i32;
        let mut builder = self.device.stream.launch_builder(&calib_fn);
        builder.arg(&d_timestamps);
        builder.arg(&d_close_prices);
        builder.arg(&d_volumes);
        builder.arg(&d_buy_volumes);
        builder.arg(&d_sell_volumes);
        builder.arg(&d_cum_partial);
        builder.arg(&d_block_prefixes);
        builder.arg(&mut d_mins);
        builder.arg(&mut d_maxs);
        builder.arg(&num_ticks_i32);
        unsafe {
            builder.launch(calib_config).map_err(|e| {
                GpuError::ExecutionError(format!("Calibration kernel launch failed: {:?}", e))
            })?;
        }

        // Synchronize and copy results
        self.device.synchronize()?;

        let mins = self.device.copy_to_host_f32(&d_mins)?;
        let maxs = self.device.copy_to_host_f32(&d_maxs)?;

        // Mirror cpu/orderflow.rs calibrate_ranges fallbacks: a feature with
        // no finite samples keeps its ±inf identity on the device; report
        // [0.0, 1.0] instead.
        let mut ranges = [0.0f32; NUM_FEATURES * 2];
        for i in 0..NUM_FEATURES {
            ranges[i * 2] = if mins[i].is_finite() { mins[i] } else { 0.0 };
            ranges[i * 2 + 1] = if maxs[i].is_finite() { maxs[i] } else { 1.0 };
        }

        Ok(ranges)
    }

    /// Process batch of orderflow data with multiple strategies (FUSED).
    ///
    /// Computes the 6 orderflow features ONCE per tick and fuses the
    /// per-strategy epilogue (threshold signals + per-range-group INT8
    /// quantization) into the same kernel.
    ///
    /// # Arguments
    ///
    /// * `input` - Tick-level data (timestamps in ms, prices/volumes f32)
    /// * `strategies` - Strategy configurations (type + quantization ranges)
    ///
    /// # Returns
    ///
    /// Signals and quantized features for all strategies. Strategies sharing
    /// identical quantization ranges receive byte-identical feature rows.
    pub fn process_batch(
        &mut self,
        input: &OrderflowInput,
        strategies: &[StrategyConfig],
    ) -> Result<OrderflowOutput, GpuError> {
        let (output, _) = self.process_batch_impl(input, strategies, false)?;
        Ok(output)
    }

    /// Full implementation; optionally copies the raw FP32 features
    /// `[num_ticks * NUM_FEATURES]` back to the host (used by the GPU-vs-CPU
    /// parity tests; skipped on the production path to avoid a 24 B/tick D2H
    /// transfer nobody reads).
    fn process_batch_impl(
        &mut self,
        input: &OrderflowInput,
        strategies: &[StrategyConfig],
        copy_features_f32: bool,
    ) -> Result<(OrderflowOutput, Option<Vec<f32>>), GpuError> {
        input.validate()?;

        if strategies.is_empty() {
            return Err(GpuError::InvalidParameter("No strategies provided".into()));
        }

        let num_strategies = strategies.len();
        let num_ticks = input.len();
        let num_blocks = num_tick_blocks(num_ticks);

        let strategy_ids: Vec<i32> = strategies.iter().map(|s| s.strategy_type as i32).collect();
        let groups = group_strategy_ranges(strategies);

        // Transfer input data to GPU
        let d_timestamps = self.device.copy_to_device_i64(&input.timestamps)?;
        let d_close_prices = self.device.copy_to_device_f32(&input.close_prices)?;
        let d_volumes = self.device.copy_to_device_f32(&input.volumes)?;
        let d_buy_volumes = self.device.copy_to_device_f32(&input.buy_volumes)?;
        let d_sell_volumes = self.device.copy_to_device_f32(&input.sell_volumes)?;
        let d_strategy_ids = self.device.copy_to_device_i32(&strategy_ids)?;
        let d_group_mins = self.device.copy_to_device_f32(&groups.group_mins)?;
        let d_group_scales = self.device.copy_to_device_f32(&groups.group_scales)?;

        // Passes 1 + 2: cumulative volume delta scan
        let (d_cum_partial, d_block_prefixes) =
            self.compute_cum_delta_scan(&d_buy_volumes, &d_sell_volumes, num_ticks)?;

        // Output buffers
        let features_f32_len = num_ticks * NUM_FEATURES;
        let features_q_len = groups.num_groups * num_ticks * NUM_FEATURES;
        let signals_len = num_strategies * num_ticks;

        let mut d_features_f32 = self
            .device
            .stream
            .alloc_zeros::<f32>(features_f32_len)
            .map_err(|e| {
                GpuError::AllocationError(format!("Failed to allocate f32 features: {:?}", e))
            })?;
        let mut d_features_q = self
            .device
            .stream
            .alloc_zeros::<i8>(features_q_len)
            .map_err(|e| {
                GpuError::AllocationError(format!("Failed to allocate features: {:?}", e))
            })?;
        let mut d_signals = self
            .device
            .stream
            .alloc_zeros::<i8>(signals_len)
            .map_err(|e| {
                GpuError::AllocationError(format!("Failed to allocate signals: {:?}", e))
            })?;

        // Pass 3a: fused features + signals + quantization
        let func = self
            .device
            .get_or_load_function(ORDERFLOW_KERNEL_SRC, "orderflow_features_signals_kernel")?;
        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (TICKS_PER_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0, // static shared memory only
        };

        let num_strategies_i32 = num_strategies as i32;
        let num_groups_i32 = groups.num_groups as i32;
        let num_ticks_i32 = num_ticks as i32;

        let mut builder = self.device.stream.launch_builder(&func);
        builder.arg(&d_timestamps);
        builder.arg(&d_close_prices);
        builder.arg(&d_volumes);
        builder.arg(&d_buy_volumes);
        builder.arg(&d_sell_volumes);
        builder.arg(&d_cum_partial);
        builder.arg(&d_block_prefixes);
        builder.arg(&d_strategy_ids);
        builder.arg(&d_group_mins);
        builder.arg(&d_group_scales);
        builder.arg(&mut d_features_f32);
        builder.arg(&mut d_features_q);
        builder.arg(&mut d_signals);
        builder.arg(&num_strategies_i32);
        builder.arg(&num_groups_i32);
        builder.arg(&num_ticks_i32);
        unsafe {
            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("Fused kernel launch failed: {:?}", e))
            })?;
        }

        // Synchronize and copy results
        self.device.synchronize()?;

        let signals_flat = self.device.copy_to_host_i8(&d_signals)?;
        let features_q_flat = self.device.copy_to_host_i8(&d_features_q)?;
        let features_f32 = if copy_features_f32 {
            Some(self.device.copy_to_host_f32(&d_features_f32)?)
        } else {
            None
        };

        // Reshape signals to [num_strategies][num_ticks]
        let signals: Vec<Vec<i8>> = (0..num_strategies)
            .map(|i| {
                let start = i * num_ticks;
                signals_flat[start..start + num_ticks].to_vec()
            })
            .collect();

        // Broadcast each strategy's range-group row back to the stable
        // per-strategy shape [num_strategies][num_ticks * NUM_FEATURES]
        let group_len = num_ticks * NUM_FEATURES;
        let features: Vec<Vec<i8>> = groups
            .group_of
            .iter()
            .map(|&g| {
                let start = g * group_len;
                features_q_flat[start..start + group_len].to_vec()
            })
            .collect();

        // Extract feature ranges
        let feature_ranges: Vec<[f32; NUM_FEATURES * 2]> = strategies
            .iter()
            .map(|s| {
                let mut ranges = [0.0f32; NUM_FEATURES * 2];
                for i in 0..NUM_FEATURES {
                    ranges[i * 2] = s.feature_mins[i];
                    ranges[i * 2 + 1] = s.feature_maxs[i];
                }
                ranges
            })
            .collect();

        Ok((
            OrderflowOutput {
                signals,
                features,
                feature_ranges,
            },
            features_f32,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    // ====================================================================
    // Deterministic pseudo-random data (no external RNG dependency)
    // ====================================================================

    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    /// Uniform f32 in [0, 1) with a 24-bit mantissa
    fn lcg_f32(state: &mut u64) -> f32 {
        ((lcg_next(state) >> 40) as f32) / 16_777_216.0
    }

    fn random_input(num_ticks: usize, seed: u64) -> OrderflowInput {
        let mut s = seed;
        let mut timestamps = Vec::with_capacity(num_ticks);
        let mut close_prices = Vec::with_capacity(num_ticks);
        let mut volumes = Vec::with_capacity(num_ticks);
        let mut buy_volumes = Vec::with_capacity(num_ticks);
        let mut sell_volumes = Vec::with_capacity(num_ticks);

        let mut ts = 1_700_000_000_000i64;
        for _ in 0..num_ticks {
            ts += 1 + (lcg_next(&mut s) % 250) as i64; // 1-250 ms gaps
            timestamps.push(ts);
            close_prices.push(50_000.0 + 500.0 * (lcg_f32(&mut s) - 0.5));
            // Occasionally large one-sided volume so signal thresholds
            // (|delta| > 1000/5000) actually fire
            let scale = if lcg_next(&mut s) % 10 == 0 {
                8_000.0
            } else {
                400.0
            };
            let buy = scale * lcg_f32(&mut s);
            let sell = scale * lcg_f32(&mut s);
            buy_volumes.push(buy);
            sell_volumes.push(sell);
            volumes.push(buy + sell);
        }

        OrderflowInput {
            timestamps,
            close_prices,
            volumes,
            buy_volumes,
            sell_volumes,
        }
    }

    // ====================================================================
    // Host mirror of the CPU reference (cpu/orderflow.rs extract_features).
    // extract_features is private to the cpu module, so the mirror is kept
    // test-local and pinned bit-for-bit against the real implementation by
    // test_reference_features_match_cpu_processor below.
    // ====================================================================

    fn window_push(window: &mut VecDeque<f32>, value: f32) {
        if window.len() == WINDOW_SIZE {
            window.pop_front();
        }
        window.push_back(value);
    }

    /// Mirrors CircularBuffer::mean / std_dev: population variance, f32
    /// accumulation in oldest -> newest order.
    fn window_zscore(window: &VecDeque<f32>, current: f32) -> f32 {
        if window.len() < 2 {
            return 0.0;
        }
        let len = window.len() as f32;
        let sum: f32 = window.iter().sum();
        let mean = sum / len;
        let variance: f32 = window.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / len;
        let std = variance.sqrt();
        if std > 1e-6 {
            (current - mean) / std
        } else {
            0.0
        }
    }

    fn reference_features(input: &OrderflowInput) -> Vec<[f32; NUM_FEATURES]> {
        let n = input.len();
        let mut out = Vec::with_capacity(n);
        let mut price_window: VecDeque<f32> = VecDeque::new();
        let mut volume_window: VecDeque<f32> = VecDeque::new();
        let mut cumulative = 0.0f32;
        let mut prev_ts = input.timestamps[0];

        for i in 0..n {
            let close = input.close_prices[i];
            let volume = input.volumes[i];
            let buy = input.buy_volumes[i];
            let sell = input.sell_volumes[i];
            let ts = input.timestamps[i];

            let total = buy + sell;
            let imbalance = if total > 0.0 { buy / total } else { 0.5 };
            let volume_delta = buy - sell;

            let dt_ms = (ts - prev_ts).max(1);
            let intensity = volume / (dt_ms as f32 / 1000.0);
            prev_ts = ts;

            window_push(&mut price_window, close);
            let price_velocity = window_zscore(&price_window, close);
            window_push(&mut volume_window, volume);
            let volume_velocity = window_zscore(&volume_window, volume);

            cumulative += volume_delta;

            out.push([
                imbalance,
                volume_delta,
                intensity,
                price_velocity,
                volume_velocity,
                cumulative,
            ]);
        }
        out
    }

    /// Mirror of cpu/orderflow.rs quantize_features (the CPU's own saturating
    /// `as i8` convention — used ONLY to pin the reference mirror against the
    /// CPU processor output; the GPU path uses the raw 0-255 convention).
    fn cpu_style_quantize(value: f32, min: f32, max: f32) -> i8 {
        let normalized = if (max - min).abs() > 1e-6 {
            ((value - min) / (max - min)).clamp(0.0, 1.0)
        } else {
            0.5
        };
        (normalized * 255.0) as i8
    }

    fn reference_signal(strategy: StrategyType, feats: &[f32; NUM_FEATURES]) -> i8 {
        let (imb, delta, intensity, pvel) = (feats[0], feats[1], feats[2], feats[3]);
        match strategy {
            StrategyType::Momentum => {
                if imb > 0.6 && delta > 1000.0 {
                    1
                } else if imb < 0.4 && delta < -1000.0 {
                    -1
                } else {
                    0
                }
            }
            StrategyType::MeanReversion => {
                if imb < 0.4 && delta < -1000.0 {
                    1
                } else if imb > 0.6 && delta > 1000.0 {
                    -1
                } else {
                    0
                }
            }
            StrategyType::Breakout => {
                if intensity > 100.0 && pvel > 0.001 {
                    1
                } else if intensity > 100.0 && pvel < -0.001 {
                    -1
                } else {
                    0
                }
            }
            StrategyType::Scalping => {
                if imb > 0.55 && delta.abs() < 500.0 {
                    1
                } else if imb < 0.45 && delta.abs() < 500.0 {
                    -1
                } else {
                    0
                }
            }
            StrategyType::TrendFollowing => {
                if delta > 5000.0 && pvel > 0.002 {
                    1
                } else if delta < -5000.0 && pvel < -0.002 {
                    -1
                } else {
                    0
                }
            }
        }
    }

    fn to_cpu_input(input: &OrderflowInput) -> crate::cpu::orderflow::OrderflowInput {
        crate::cpu::orderflow::OrderflowInput {
            timestamps: input.timestamps.clone(),
            close_prices: input.close_prices.clone(),
            volumes: input.volumes.clone(),
            buy_volumes: input.buy_volumes.clone(),
            sell_volumes: input.sell_volumes.clone(),
        }
    }

    fn to_cpu_strategy(s: &StrategyConfig) -> crate::cpu::orderflow::StrategyConfig {
        use crate::cpu::orderflow::StrategyType as CpuType;
        crate::cpu::orderflow::StrategyConfig {
            strategy_type: match s.strategy_type {
                StrategyType::Momentum => CpuType::Momentum,
                StrategyType::MeanReversion => CpuType::MeanReversion,
                StrategyType::Breakout => CpuType::Breakout,
                StrategyType::Scalping => CpuType::Scalping,
                StrategyType::TrendFollowing => CpuType::TrendFollowing,
            },
            feature_mins: s.feature_mins,
            feature_maxs: s.feature_maxs,
        }
    }

    fn all_default_strategies() -> Vec<StrategyConfig> {
        vec![
            StrategyConfig::momentum(),
            StrategyConfig::mean_reversion(),
            StrategyConfig::breakout(),
            StrategyConfig::scalping(),
            StrategyConfig::trend_following(),
        ]
    }

    // ====================================================================
    // Bit-exact host simulation of the kernel's 3-pass scan composition
    // (same Kogge-Stone warp scans + cross-warp carry + chunked pass 2)
    // ====================================================================

    /// Mirrors `block_inclusive_scan` for one TICKS_PER_BLOCK block: 32-lane
    /// Kogge-Stone scans, warp totals scanned by the same algorithm, then the
    /// preceding-warp prefix added. Input shorter than TICKS_PER_BLOCK is
    /// zero-padded exactly like out-of-range lanes in the kernel; the full
    /// padded result is returned (lane TICKS_PER_BLOCK-1 holds the chunk
    /// total the kernel writes to block_sums).
    fn simulate_block_inclusive_scan_padded(block: &[f32]) -> Vec<f32> {
        assert!(block.len() <= TICKS_PER_BLOCK);
        let num_warps = TICKS_PER_BLOCK / 32;
        let mut v = block.to_vec();
        v.resize(TICKS_PER_BLOCK, 0.0);

        let mut warp_sums = vec![0.0f32; num_warps];
        for w in 0..num_warps {
            let base = w * 32;
            let mut offset = 1;
            while offset < 32 {
                // __shfl_up_sync reads the pre-step register values
                let snapshot: Vec<f32> = v[base..base + 32].to_vec();
                for lane in offset..32 {
                    v[base + lane] = snapshot[lane] + snapshot[lane - offset];
                }
                offset <<= 1;
            }
            warp_sums[w] = v[base + 31];
        }

        let mut offset = 1;
        while offset < num_warps {
            let snapshot = warp_sums.clone();
            for lane in offset..num_warps {
                warp_sums[lane] = snapshot[lane] + snapshot[lane - offset];
            }
            offset <<= 1;
        }

        for w in 1..num_warps {
            for lane in 0..32 {
                v[w * 32 + lane] += warp_sums[w - 1];
            }
        }
        v
    }

    /// Mirrors the full pass-1 + pass-2 + fused-fixup composition:
    /// `cum[i] = block_local_inclusive[i] + exclusive_block_prefix[i / TPB]`.
    fn simulate_gpu_cumulative_scan(deltas: &[f32]) -> Vec<f32> {
        let n = deltas.len();
        let num_blocks = num_tick_blocks(n);

        // Pass 1
        let mut partial = vec![0.0f32; n];
        let mut block_sums = vec![0.0f32; num_blocks];
        for b in 0..num_blocks {
            let start = b * TICKS_PER_BLOCK;
            let end = (start + TICKS_PER_BLOCK).min(n);
            let incl = simulate_block_inclusive_scan_padded(&deltas[start..end]);
            partial[start..end].copy_from_slice(&incl[..end - start]);
            block_sums[b] = incl[TICKS_PER_BLOCK - 1];
        }

        // Pass 2: chunked exclusive scan with running carry
        let mut prefixes = vec![0.0f32; num_blocks];
        let mut carry = 0.0f32;
        let mut base = 0usize;
        while base < num_blocks {
            let end = (base + TICKS_PER_BLOCK).min(num_blocks);
            let incl = simulate_block_inclusive_scan_padded(&block_sums[base..end]);
            for i in base..end {
                let t = i - base;
                let prev = if t > 0 { incl[t - 1] } else { 0.0 };
                prefixes[i] = carry + prev;
            }
            carry += incl[TICKS_PER_BLOCK - 1];
            base += TICKS_PER_BLOCK;
        }

        // Pass-3 fixup (fused into the feature kernels)
        (0..n)
            .map(|i| partial[i] + prefixes[i / TICKS_PER_BLOCK])
            .collect()
    }

    // ====================================================================
    // Host-only tests (no GPU)
    // ====================================================================

    #[test]
    fn test_strategy_config_creation() {
        let momentum = StrategyConfig::momentum();
        assert_eq!(momentum.strategy_type, StrategyType::Momentum);
        assert_eq!(momentum.feature_mins.len(), NUM_FEATURES);

        let mean_rev = StrategyConfig::mean_reversion();
        assert_eq!(mean_rev.strategy_type, StrategyType::MeanReversion);
    }

    #[test]
    fn test_orderflow_input_validation() {
        let valid = OrderflowInput {
            timestamps: vec![1, 2, 3],
            close_prices: vec![100.0, 101.0, 102.0],
            volumes: vec![10.0, 11.0, 12.0],
            buy_volumes: vec![5.0, 6.0, 7.0],
            sell_volumes: vec![5.0, 5.0, 5.0],
        };
        assert!(valid.validate().is_ok());

        let invalid = OrderflowInput {
            timestamps: vec![1, 2, 3],
            close_prices: vec![100.0, 101.0], // Wrong length!
            volumes: vec![10.0, 11.0, 12.0],
            buy_volumes: vec![5.0, 6.0, 7.0],
            sell_volumes: vec![5.0, 5.0, 5.0],
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_signal_conversion() {
        assert_eq!(Signal::from(1), Signal::Buy);
        assert_eq!(Signal::from(-1), Signal::Sell);
        assert_eq!(Signal::from(0), Signal::Hold);
        assert_eq!(Signal::from(99), Signal::Hold); // Unknown defaults to Hold
    }

    #[test]
    fn test_block_constants_sane() {
        assert_eq!(TICKS_PER_BLOCK % 32, 0, "block must be whole warps");
        assert!(
            TICKS_PER_BLOCK / 32 <= 32,
            "cross-warp scan needs <= 32 warps"
        );
        assert!(
            TICKS_PER_BLOCK >= WINDOW_SIZE,
            "halo must fit one block's chunk"
        );
    }

    /// The kernel is NVRTC-JIT-compiled: no #include, extern "C" __global__
    /// entry points only.
    #[test]
    fn test_kernel_source_is_nvrtc_compatible() {
        assert!(
            !ORDERFLOW_KERNEL_SRC.contains("#include"),
            "NVRTC kernel must not use #include directives"
        );

        for name in [
            "orderflow_block_scan_kernel",
            "orderflow_scan_block_sums_kernel",
            "orderflow_features_signals_kernel",
            "init_calibration_ranges_kernel",
            "calibrate_feature_ranges_kernel",
        ] {
            let signature = format!("extern \"C\" __global__ void {}(", name);
            assert!(
                ORDERFLOW_KERNEL_SRC.contains(&signature),
                "missing extern \"C\" __global__ entry point: {}",
                name
            );
        }
    }

    /// The #define lines in the kernel are a layout contract with this
    /// module's launch math; keep them in lockstep.
    #[test]
    fn test_kernel_layout_contract_matches_host_constants() {
        assert!(
            ORDERFLOW_KERNEL_SRC.contains(&format!("#define WINDOW_SIZE {}", WINDOW_SIZE)),
            "kernel WINDOW_SIZE must match host const"
        );
        assert!(
            ORDERFLOW_KERNEL_SRC.contains(&format!("#define TICKS_PER_BLOCK {}", TICKS_PER_BLOCK)),
            "kernel TICKS_PER_BLOCK must match host const"
        );
        assert!(
            ORDERFLOW_KERNEL_SRC.contains(&format!("#define NUM_FEATURES {}", NUM_FEATURES)),
            "kernel NUM_FEATURES must match host const"
        );
    }

    /// Ada runs FP64 at 1/64 the FP32 rate; the only 64-bit type allowed in
    /// device code is the i64 timestamp (long long).
    #[test]
    fn test_kernel_uses_no_double_precision() {
        assert!(
            !ORDERFLOW_KERNEL_SRC.contains("double"),
            "no FP64 in device code (Ada 1:64 FP64 throughput); use float"
        );
        assert!(
            ORDERFLOW_KERNEL_SRC.contains("const long long* __restrict__ timestamps"),
            "timestamps must stay i64 until the ms-delta is taken"
        );
    }

    /// INT8 convention: raw 0-255 codes through the char ABI; a direct
    /// (char) cast saturates at 127 (the bug class fixed in quantize_int8.cu).
    #[test]
    fn test_kernel_quantize_cast_preserves_raw_codes() {
        assert!(
            ORDERFLOW_KERNEL_SRC.contains("(char)(unsigned char)q"),
            "quantize must cast through unsigned char"
        );
        assert!(
            ORDERFLOW_KERNEL_SRC.contains("roundf(q)"),
            "quantize must round half away from zero (matches Rust f32::round)"
        );
        assert!(
            !ORDERFLOW_KERNEL_SRC.contains("normalized * 255.0f + 0.5f"),
            "old truncation-based rounding must not return"
        );
    }

    /// The redesign removed the per-warp circular buffers (strided windows +
    /// data race), the O(W^2) median bubble sort, and the old entry point.
    #[test]
    fn test_kernel_has_no_legacy_artifacts() {
        for legacy in [
            "orderflow_signals_fused_kernel",
            "CircularBuffer",
            "circ_push",
            "circ_median",
            "tick += WARP_SIZE",
        ] {
            assert!(
                !ORDERFLOW_KERNEL_SRC.contains(legacy),
                "legacy artifact must not return: {}",
                legacy
            );
        }
    }

    #[test]
    fn test_quantization_scale() {
        // Normal range
        let s = quantization_scale(-1.0, 1.0);
        assert!((s - 127.5).abs() < 1e-4);

        // Matches quantization.rs: scale * range == 255
        let s = quantization_scale(-10_000.0, 10_000.0);
        assert!((s * 20_000.0 - 255.0).abs() < 1e-3);

        // Degenerate ranges quantize to code 0 (dequantizes to min)
        assert_eq!(quantization_scale(5.0, 5.0), 0.0);
        assert_eq!(quantization_scale(5.0, 5.0 + 1e-10), 0.0);
        assert_eq!(quantization_scale(5.0, 4.0), 0.0); // inverted range
    }

    #[test]
    fn test_group_strategy_ranges_dedup() {
        // All 5 defaults share ranges -> exactly one group
        let strategies = all_default_strategies();
        let groups = group_strategy_ranges(&strategies);
        assert_eq!(groups.num_groups, 1);
        assert_eq!(groups.group_of, vec![0, 0, 0, 0, 0]);
        assert_eq!(groups.group_mins.len(), NUM_FEATURES);
        assert_eq!(groups.group_scales.len(), NUM_FEATURES);
        // Spot-check a scale: feature 1 range [-10000, 10000] -> 255/20000
        assert!((groups.group_scales[1] - 255.0 / 20_000.0).abs() < 1e-9);

        // A custom range introduces a second group, in first-occurrence order
        let mut custom = StrategyConfig::momentum();
        custom.feature_maxs[2] = 2_000.0;
        let strategies = vec![
            StrategyConfig::momentum(),
            custom.clone(),
            StrategyConfig::breakout(),
            custom,
        ];
        let groups = group_strategy_ranges(&strategies);
        assert_eq!(groups.num_groups, 2);
        assert_eq!(groups.group_of, vec![0, 1, 0, 1]);
        assert_eq!(groups.group_mins.len(), 2 * NUM_FEATURES);
        assert!((groups.group_scales[NUM_FEATURES + 2] - 255.0 / 2_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_num_tick_blocks() {
        assert_eq!(num_tick_blocks(1), 1);
        assert_eq!(num_tick_blocks(TICKS_PER_BLOCK - 1), 1);
        assert_eq!(num_tick_blocks(TICKS_PER_BLOCK), 1);
        assert_eq!(num_tick_blocks(TICKS_PER_BLOCK + 1), 2);
        assert_eq!(num_tick_blocks(1_000_000), 3907);
    }

    /// Scan composition, exact: with small integer-valued f32 deltas every
    /// addition is exact (|sums| << 2^24), so the simulated 3-pass scan must
    /// equal the sequential prefix sum BIT FOR BIT regardless of association
    /// order. Catches off-by-one block prefixes, inclusive/exclusive mixups,
    /// and padding bugs across partial-block sizes.
    #[test]
    fn test_scan_composition_exact_integers() {
        for &n in &[1usize, 7, 255, 256, 257, 1317, 4096] {
            let mut s = 0xDEAD_0000 + n as u64;
            let deltas: Vec<f32> = (0..n)
                .map(|_| ((lcg_next(&mut s) % 7) as i64 - 3) as f32)
                .collect();

            let simulated = simulate_gpu_cumulative_scan(&deltas);

            let mut running = 0.0f32;
            for (i, &d) in deltas.iter().enumerate() {
                running += d;
                assert_eq!(
                    simulated[i], running,
                    "n={}: scan composition diverged at index {}",
                    n, i
                );
            }
        }
    }

    /// Scan composition, floats: the tree-order f32 result must track an f64
    /// sequential reference. A composition bug shifts results by whole block
    /// sums (O(1)-O(100) here), far above the rounding-noise tolerance.
    #[test]
    fn test_scan_composition_random_floats() {
        let n = 10_000;
        let mut s = 0xF00D;
        let deltas: Vec<f32> = (0..n).map(|_| 2.0 * lcg_f32(&mut s) - 1.0).collect();

        let simulated = simulate_gpu_cumulative_scan(&deltas);

        let mut running = 0.0f64;
        for (i, &d) in deltas.iter().enumerate() {
            running += d as f64;
            let err = (simulated[i] as f64 - running).abs();
            let tol = 0.05 + running.abs() * 1e-4;
            assert!(
                err <= tol,
                "index {}: simulated {} vs f64 reference {} (err {})",
                i,
                simulated[i],
                running,
                err
            );
        }
    }

    /// Pin the test-local reference mirror to the normative CPU processor:
    /// the mirror's features, pushed through the CPU's own quantizer and
    /// signal logic, must reproduce the processor output bit for bit. This
    /// is what makes the GPU parity tests below meaningful.
    #[test]
    fn test_reference_features_match_cpu_processor() {
        let input = random_input(2_000, 0x5EED);
        let reference = reference_features(&input);

        let strategies = all_default_strategies();
        let cpu_strategies: Vec<_> = strategies.iter().map(to_cpu_strategy).collect();
        let cpu_out = crate::cpu::orderflow::OrderflowBatchProcessor::new()
            .process_batch(&to_cpu_input(&input), &cpu_strategies)
            .expect("CPU processing failed");

        for (s, strategy) in strategies.iter().enumerate() {
            for tick in 0..input.len() {
                // Signals (function of features 0-3)
                let expected_signal = reference_signal(strategy.strategy_type, &reference[tick]);
                assert_eq!(
                    cpu_out.signals[s][tick], expected_signal,
                    "strategy {} tick {}: mirror signal diverged from CPU",
                    s, tick
                );

                // Quantized features (lossy but bit-deterministic: pins all
                // 6 mirror features against the CPU implementation)
                for f in 0..NUM_FEATURES {
                    let expected = cpu_style_quantize(
                        reference[tick][f],
                        strategy.feature_mins[f],
                        strategy.feature_maxs[f],
                    );
                    assert_eq!(
                        cpu_out.features[s][tick * NUM_FEATURES + f],
                        expected,
                        "strategy {} tick {} feature {}: mirror feature diverged from CPU",
                        s,
                        tick,
                        f
                    );
                }
            }
        }
    }

    // ====================================================================
    // GPU-vs-CPU parity tests (require a CUDA device; run with --ignored)
    // ====================================================================

    fn gpu_processor() -> Option<OrderflowBatchProcessor> {
        match GpuDevice::new() {
            Ok(device) => Some(
                OrderflowBatchProcessor::new(Arc::new(device))
                    .expect("processor creation is infallible"),
            ),
            Err(e) => {
                eprintln!("GPU not available, skipping: {:?}", e);
                None
            }
        }
    }

    /// A GPU/CPU signal mismatch is acceptable only when the reference
    /// feature sits inside the fast-math drift band around the strategy's
    /// decision threshold. Kernels compile with -use_fast_math
    /// (prec_div/prec_sqrt off): adds/subs stay exact (so the volume-delta
    /// thresholds get NO band), but every division carries ~2 ulp, and the
    /// z-score's absolute drift scales with ulp(price)/std (up to ~2e-3).
    fn signal_boundary_mismatch_ok(strategy: StrategyType, feats: &[f32; NUM_FEATURES]) -> bool {
        let (imb, intensity, pvel) = (feats[0], feats[2], feats[3]);
        const IMB_EPS: f32 = 1e-5; // one ~2-ulp division on a [0,1] value
        const INTENSITY_EPS: f32 = 0.01; // two divisions, threshold at 100
        const PVEL_EPS: f32 = 5e-3; // z-score drift band
        match strategy {
            StrategyType::Momentum | StrategyType::MeanReversion => {
                (imb - 0.6).abs() <= IMB_EPS || (imb - 0.4).abs() <= IMB_EPS
            }
            StrategyType::Scalping => {
                (imb - 0.55).abs() <= IMB_EPS || (imb - 0.45).abs() <= IMB_EPS
            }
            StrategyType::Breakout => {
                (intensity - 100.0).abs() <= INTENSITY_EPS || (pvel.abs() - 0.001).abs() <= PVEL_EPS
            }
            StrategyType::TrendFollowing => (pvel.abs() - 0.002).abs() <= PVEL_EPS,
        }
    }

    fn run_feature_signal_parity(processor: &mut OrderflowBatchProcessor, n: usize, seed: u64) {
        let input = random_input(n, seed);
        let strategies = all_default_strategies();

        let (output, features_f32) = processor
            .process_batch_impl(&input, &strategies, true)
            .expect("GPU processing failed");
        let gpu_feats = features_f32.expect("f32 features requested");
        assert_eq!(gpu_feats.len(), n * NUM_FEATURES);

        let reference = reference_features(&input);

        // |delta| prefix sums set the error scale for the cumulative feature:
        // f32 scan rounding grows with the magnitude of the intermediate
        // partial sums, not the pointwise value (which crosses zero, where a
        // purely relative bound collapses).
        let mut abs_delta_prefix = Vec::with_capacity(n);
        let mut acc = 0.0f64;
        for i in 0..n {
            acc += (input.buy_volumes[i] - input.sell_volumes[i]).abs() as f64;
            abs_delta_prefix.push(acc as f32);
        }

        // FP32 feature parity. Per-feature tolerances reflect -use_fast_math
        // compilation (see signal_boundary_mismatch_ok): feature 1 is a lone
        // exact subtraction; z-scores divide a small cancellation by std, so
        // their absolute drift scales with ulp(input)/std; the cumulative sum
        // scales with the |delta| prefix.
        for tick in 0..n {
            for f in 0..NUM_FEATURES {
                let gpu = gpu_feats[tick * NUM_FEATURES + f];
                let cpu = reference[tick][f];
                let tol = match f {
                    1 => 0.0,                                        // exact
                    3 | 4 => 2e-3f32.max(cpu.abs() * 2e-4),          // z-scores
                    5 => 1e-4f32.max(abs_delta_prefix[tick] * 2e-6), // cumulative
                    _ => 1e-4f32.max(cpu.abs() * 1e-4),              // f0, f2
                };
                assert!(
                    (gpu - cpu).abs() <= tol,
                    "n={} tick {} feature {}: gpu {} vs cpu {} (tol {})",
                    n,
                    tick,
                    f,
                    gpu,
                    cpu,
                    tol
                );
            }
        }

        // Signal parity against the normative CPU processor
        let cpu_strategies: Vec<_> = strategies.iter().map(to_cpu_strategy).collect();
        let cpu_out = crate::cpu::orderflow::OrderflowBatchProcessor::new()
            .process_batch(&to_cpu_input(&input), &cpu_strategies)
            .expect("CPU processing failed");

        for (s, strategy) in strategies.iter().enumerate() {
            for tick in 0..n {
                let gpu_sig = output.signals[s][tick];
                let cpu_sig = cpu_out.signals[s][tick];
                if gpu_sig == cpu_sig {
                    continue;
                }
                assert!(
                    signal_boundary_mismatch_ok(strategy.strategy_type, &reference[tick]),
                    "n={} strategy {} tick {}: gpu signal {} vs cpu {} (feats {:?})",
                    n,
                    s,
                    tick,
                    gpu_sig,
                    cpu_sig,
                    reference[tick]
                );
            }
        }

        // All 5 defaults share one range group -> byte-identical feature rows
        for s in 1..strategies.len() {
            assert_eq!(
                output.features[s], output.features[0],
                "default strategies must share one quantization group"
            );
        }

        // GPU INT8 codes must equal host quantization (raw 0-255 convention)
        // of the GPU's own f32 features — validates the in-kernel epilogue.
        let groups = group_strategy_ranges(&strategies);
        assert_eq!(groups.num_groups, 1);
        for tick in 0..n {
            for f in 0..NUM_FEATURES {
                let v = gpu_feats[tick * NUM_FEATURES + f];
                let q = (v - groups.group_mins[f]) * groups.group_scales[f];
                let expected = (q.round().clamp(0.0, 255.0) as u8) as i8;
                assert_eq!(
                    output.features[0][tick * NUM_FEATURES + f],
                    expected,
                    "n={} tick {} feature {}: quantized code mismatch",
                    n,
                    tick,
                    f
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU: cargo test --features gpu -- --ignored
    fn test_gpu_cpu_parity_features_and_signals() {
        let Some(mut processor) = gpu_processor() else {
            return;
        };
        // Sizes straddle the block/halo boundaries (1 block partial, exactly
        // 1 block, halo crossing into block 2, many blocks)
        for &n in &[1usize, 255, 256, 257, 1024, 10_000] {
            run_feature_signal_parity(&mut processor, n, 0xC0FFEE ^ n as u64);
        }
    }

    #[test]
    #[ignore] // Requires GPU: cargo test --features gpu -- --ignored
    fn test_gpu_calibration_matches_cpu() {
        let Some(mut processor) = gpu_processor() else {
            return;
        };

        let input = random_input(10_000, 0xCA11B);
        let gpu_ranges = processor
            .calibrate_ranges(&input)
            .expect("GPU calibration failed");

        let cpu_ranges = crate::cpu::orderflow::OrderflowBatchProcessor::new()
            .calibrate_ranges(&to_cpu_input(&input))
            .expect("CPU calibration failed");

        for f in 0..NUM_FEATURES {
            let (gmin, gmax) = (gpu_ranges[f * 2], gpu_ranges[f * 2 + 1]);
            let (cmin, cmax) = (cpu_ranges[f * 2], cpu_ranges[f * 2 + 1]);

            assert!(gmin <= gmax, "feature {}: min {} > max {}", f, gmin, gmax);

            let tol_min = 1e-3f32.max(cmin.abs() * 1e-3);
            let tol_max = 1e-3f32.max(cmax.abs() * 1e-3);
            assert!(
                (gmin - cmin).abs() <= tol_min,
                "feature {} min: gpu {} vs cpu {}",
                f,
                gmin,
                cmin
            );
            assert!(
                (gmax - cmax).abs() <= tol_max,
                "feature {} max: gpu {} vs cpu {}",
                f,
                gmax,
                cmax
            );
        }

        // Structural sanity: imbalance lives in [0, 1]; the old kernel's
        // zero-initialized buffers and one-tick windows reported [0, 0] here.
        assert!(gpu_ranges[0] >= 0.0 && gpu_ranges[1] <= 1.0);
        assert!(
            gpu_ranges[1] > gpu_ranges[0],
            "imbalance range must not collapse"
        );
    }
}
