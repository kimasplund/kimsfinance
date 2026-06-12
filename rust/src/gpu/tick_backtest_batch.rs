//! GPU Tick-Level Batch Backtest with Pending Orders Queue
//!
//! # Architecture
//!
//! - **Sequential per-strategy**: position state is path-dependent, so each
//!   strategy is walked tick by tick by a single thread
//! - **Parallel across strategies**: one THREAD per strategy packed into
//!   128-thread blocks (sized for 100s-1000s-strategy genetic workloads; the
//!   previous version launched one single-thread block per strategy)
//! - **Pending orders queue**: execution-delay simulation; orders execute at
//!   the first tick at or after `signal_time + delay`, at THAT tick's price
//! - **CPU parity**: cost model, equity marking, and end-of-data forced close
//!   mirror `rust/src/backtest/tick_engine.rs` (the normative reference);
//!   asserted by the host-side reference simulator and the `#[ignore]`
//!   GPU-vs-CPU parity test below (1e-9 on final equity, exact trade counts)
//!
//! # Signal contract
//!
//! The orderflow pipeline (`gpu/orderflow_batch.rs`, `cpu/orderflow.rs`)
//! emits i8 signals `BUY=1 / SELL=-1 / HOLD=0`. The kernel remaps raw bytes
//! at execution time: `-1` becomes SELL (close long) or, with
//! [`BacktestConfig::allow_short`], SHORT (close long + open short). The
//! legacy `0..4` enum encoding (HOLD/BUY/SELL/SHORT/COVER) is still accepted
//! unchanged. [`crate::backtest::Signal`] values are converted with an
//! explicit map (`signal_to_kernel_i8`) because the Rust enum's declaration
//! order (`Buy=0, Sell=1, Hold=2`) does NOT match the kernel encoding — the
//! old `signal as i8` cast silently sent `Buy` as HOLD.
//!
//! # Memory
//!
//! Per strategy: 1000-trade record buffer (48 B each) + metrics scalars.
//! The equity curve is optional: `run_batch_with_stride` / `run_batch_i8`
//! accept an `equity_stride` (0 = none, k = every k-th tick), so 100M-tick
//! multi-strategy runs no longer require 8 B/tick/strategy of VRAM plus a
//! multi-GB device-to-host copy. Sharpe/drawdown/returns are computed
//! incrementally in registers regardless of the stride.
//!
//! # Example
//!
//! ```rust,no_run
//! use kimsfinance_core::gpu::tick_backtest_batch::{TickBacktestBatch, BacktestConfig};
//! use kimsfinance_core::backtest::Signal;
//!
//! let config = BacktestConfig {
//!     initial_capital: 10_000.0,
//!     trading_fee: 0.001,
//!     slippage: 0.0005,
//!     execution_delay_ms: 10,
//!     allow_short: false,
//! };
//!
//! let backtest = TickBacktestBatch::new(config)?;
//!
//! // Run 10 strategies in parallel
//! let signals = vec![vec![Signal::Buy, Signal::Hold, Signal::Sell]; 10];
//! let prices = vec![100.0, 101.0, 102.0];
//! let timestamps = vec![0, 1000, 2000]; // milliseconds, non-decreasing
//!
//! let results = backtest.run_batch(&signals, &prices, &timestamps)?;
//!
//! for (i, result) in results.iter().enumerate() {
//!     println!("Strategy {}: Return={:.2}%, Sharpe={:.2}",
//!              i, result.total_return, result.sharpe_ratio);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::backtest::Signal;
use crate::gpu::{GpuDevice, GpuError};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// CUDA kernel source, NVRTC-JIT-compiled at runtime.
///
/// Must stay NVRTC-compatible: no `#include` directives, `extern "C"
/// __global__` entry points only (asserted by host-side tests below).
const TICK_BACKTEST_KERNEL_SRC: &str = include_str!("kernels/tick_backtest_batch.cu");

/// Maximum trades recorded per strategy. Mirrored by `#define MAX_TRADES`
/// in the kernel source (asserted in tests).
pub const MAX_TRADES: usize = 1000;

/// Maximum pending orders per strategy. Mirrored by
/// `#define MAX_PENDING_ORDERS` in the kernel source (asserted in tests).
pub const MAX_PENDING_ORDERS: usize = 100;

/// Default execution delay in milliseconds
pub const DEFAULT_EXECUTION_DELAY_MS: i32 = 10;

/// Threads (= strategies) per block. Mirrored by `#define THREADS_PER_BLOCK`
/// in the kernel source (asserted in tests).
const THREADS_PER_BLOCK: usize = 128;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Trade record. Layout contract with the CUDA `struct Trade`:
/// `#[repr(C)]`, 3 x f64 + 2 x i64 + i8, natural alignment 8 -> 48 bytes
/// (asserted by `test_gpu_trade_layout_contract`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuTrade {
    pub entry_price: f64,
    pub exit_price: f64,
    pub entry_time: i64,
    pub exit_time: i64,
    pub pnl: f64,
    pub direction: i8, // 1=Long, -1=Short
}

/// Backtest configuration
#[derive(Debug, Clone, Copy)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub trading_fee: f64,
    pub slippage: f64,
    pub execution_delay_ms: i32,
    /// How orderflow `-1` signals are interpreted: `false` -> SELL (close
    /// long only), `true` -> SHORT (close long + open short, matching the
    /// CPU engine's `Signal::Sell`/`Signal::Short` behavior). Legacy `0..4`
    /// encoded signals are unaffected.
    pub allow_short: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10_000.0,
            trading_fee: 0.001, // 0.1%
            slippage: 0.0005,   // 0.05%
            execution_delay_ms: DEFAULT_EXECUTION_DELAY_MS,
            allow_short: false,
        }
    }
}

/// Backtest results for a single strategy
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Mark-to-market equity at the last tick. CPU-parity quirk: if a
    /// position was force-closed at end of data, this is the PRE-close
    /// mark-to-market value, exactly like `tick_engine.rs` reports
    /// `position.equity` from the last `update_equity` call.
    pub final_equity: f64,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub num_trades: i32,
    /// Signals that arrived while the pending-order queue was full and were
    /// executed immediately instead (graceful degradation). Non-zero values
    /// mean the delay simulation was bypassed for that many orders.
    pub queue_overflows: u32,
    /// Sampled equity curve; empty when `equity_stride == 0`,
    /// `ceil(n_ticks / stride)` points otherwise (`run_batch` uses stride 1).
    pub equity_curve: Vec<f64>,
    pub trades: Vec<GpuTrade>,
}

// ============================================================================
// ENCODING HELPERS
// ============================================================================

/// Map [`crate::backtest::Signal`] to the kernel's i8 encoding
/// (HOLD=0, BUY=1, SELL=2, SHORT=3, COVER=4).
///
/// An explicit match is REQUIRED here: the Rust enum has no explicit
/// discriminants, so declaration order gives `Buy=0, Sell=1, Hold=2` — a
/// bare `signal as i8` cast (the old wrapper) sent Buy as kernel-HOLD and
/// Hold as kernel-SELL.
#[inline]
fn signal_to_kernel_i8(signal: Signal) -> i8 {
    match signal {
        Signal::Hold => 0,
        Signal::Buy => 1,
        Signal::Sell => 2,
        Signal::Short => 3,
        Signal::Cover => 4,
    }
}

/// Stored equity-curve points per strategy for a given stride
/// (0 = no curve). Mirrors the kernel's `n_eq` computation.
#[inline]
fn equity_points(n_ticks: usize, equity_stride: usize) -> usize {
    if equity_stride == 0 {
        0
    } else {
        n_ticks.div_ceil(equity_stride)
    }
}

/// Reinterpret one strategy's raw device bytes as trade records.
///
/// `read_unaligned` (not `from_raw_parts`): the `Vec<u8>` backing store has
/// no alignment guarantee for the 8-byte fields.
fn parse_strategy_trades(bytes: &[u8], num_trades: usize) -> Vec<GpuTrade> {
    let trade_size = std::mem::size_of::<GpuTrade>();
    bytes
        .chunks_exact(trade_size)
        .take(num_trades)
        .map(|chunk| {
            // SAFETY: GpuTrade is #[repr(C)] with the same field order as the
            // CUDA `struct Trade` the kernel wrote; every field type
            // (f64/i64/i8) is valid for any bit pattern, and read_unaligned
            // imposes no alignment requirement on the source pointer.
            unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const GpuTrade) }
        })
        .collect()
}

// ============================================================================
// GPU BACKTEST ENGINE
// ============================================================================

/// GPU tick-level batch backtest engine
pub struct TickBacktestBatch {
    device: Arc<GpuDevice>,
    config: BacktestConfig,
}

impl TickBacktestBatch {
    /// Create new GPU backtest engine
    ///
    /// # Arguments
    ///
    /// - `config`: Backtest configuration (fees, slippage, initial capital,
    ///   delay, short policy)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use kimsfinance_core::gpu::tick_backtest_batch::{TickBacktestBatch, BacktestConfig};
    ///
    /// let config = BacktestConfig::default();
    /// let backtest = TickBacktestBatch::new(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(config: BacktestConfig) -> Result<Self, GpuError> {
        let device = Arc::new(GpuDevice::new()?);
        Ok(Self { device, config })
    }

    /// Run batch backtest on GPU (full equity curve, stride 1)
    ///
    /// # Arguments
    ///
    /// - `signals`: Signal arrays `[N_strategies][N_ticks]`
    /// - `prices`: Price array `[N_ticks]`
    /// - `timestamps`: Timestamp array `[N_ticks]` (ms, non-decreasing)
    ///
    /// # Returns
    ///
    /// Vec of [`BacktestResult`] (one per strategy)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kimsfinance_core::gpu::tick_backtest_batch::{TickBacktestBatch, BacktestConfig};
    /// # use kimsfinance_core::backtest::Signal;
    /// # let backtest = TickBacktestBatch::new(BacktestConfig::default())?;
    /// let signals = vec![
    ///     vec![Signal::Buy, Signal::Hold, Signal::Sell],
    ///     vec![Signal::Short, Signal::Hold, Signal::Cover],
    /// ];
    /// let prices = vec![100.0, 101.0, 102.0];
    /// let timestamps = vec![0, 1000, 2000];
    ///
    /// let results = backtest.run_batch(&signals, &prices, &timestamps)?;
    /// assert_eq!(results.len(), 2); // One result per strategy
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn run_batch(
        &self,
        signals: &[Vec<Signal>],
        prices: &[f64],
        timestamps: &[i64],
    ) -> Result<Vec<BacktestResult>, GpuError> {
        self.run_batch_with_stride(signals, prices, timestamps, 1)
    }

    /// Run batch backtest with a configurable equity-curve stride.
    ///
    /// `equity_stride == 0` stores no equity curve at all (metrics are still
    /// computed incrementally on-device); `k > 0` stores every k-th tick —
    /// `ceil(n_ticks / k)` points per strategy. Use 0 or a large stride for
    /// 100M+ tick runs: a full f64 curve costs 8 B/tick/strategy of VRAM
    /// plus the matching device-to-host copy.
    pub fn run_batch_with_stride(
        &self,
        signals: &[Vec<Signal>],
        prices: &[f64],
        timestamps: &[i64],
        equity_stride: usize,
    ) -> Result<Vec<BacktestResult>, GpuError> {
        let signals_i8: Vec<Vec<i8>> = signals
            .iter()
            .map(|s| s.iter().map(|&sig| signal_to_kernel_i8(sig)).collect())
            .collect();
        self.run_batch_i8(&signals_i8, prices, timestamps, equity_stride)
    }

    /// Run batch backtest on raw i8 signal streams.
    ///
    /// This is the integration point for the orderflow pipeline
    /// ([`crate::gpu::orderflow_batch::OrderflowOutput::signals`]): accepted
    /// encodings per tick are `1` (buy), `-1` (sell-or-short per
    /// [`BacktestConfig::allow_short`]), `0` (hold), and the legacy `2..4`
    /// enum values (SELL/SHORT/COVER). Any other byte is treated as hold.
    pub fn run_batch_i8(
        &self,
        signals: &[Vec<i8>],
        prices: &[f64],
        timestamps: &[i64],
        equity_stride: usize,
    ) -> Result<Vec<BacktestResult>, GpuError> {
        let n_strategies = signals.len();
        let n_ticks = prices.len();

        // ====================================================================
        // VALIDATION
        // ====================================================================
        if n_strategies == 0 {
            return Err(GpuError::InvalidInput("No strategies provided".to_string()));
        }
        if n_ticks == 0 {
            return Err(GpuError::InvalidInput("No ticks provided".to_string()));
        }
        if n_strategies > i32::MAX as usize || n_ticks > i32::MAX as usize {
            return Err(GpuError::InvalidInput(
                "Strategy/tick counts must fit in i32".to_string(),
            ));
        }
        if equity_stride > i32::MAX as usize {
            return Err(GpuError::InvalidInput(format!(
                "equity_stride too large: {}",
                equity_stride
            )));
        }
        if timestamps.len() != n_ticks {
            return Err(GpuError::InvalidInput(format!(
                "Timestamp length mismatch: {} != {}",
                timestamps.len(),
                n_ticks
            )));
        }
        for (i, sig) in signals.iter().enumerate() {
            if sig.len() != n_ticks {
                return Err(GpuError::InvalidInput(format!(
                    "Strategy {} signal length mismatch: {} != {}",
                    i,
                    sig.len(),
                    n_ticks
                )));
            }
        }

        let signals_flat: Vec<i8> = signals.iter().flat_map(|s| s.iter().copied()).collect();

        // ====================================================================
        // ALLOCATE GPU MEMORY
        // ====================================================================

        // Inputs
        let d_signals = self.device.copy_to_device_i8(&signals_flat)?;
        let d_prices = self.device.copy_to_device(prices)?;
        let d_timestamps = self.device.copy_to_device_i64(timestamps)?;

        // Equity curve (optional). At least 1 element so the kernel always
        // receives a valid device pointer; never dereferenced when stride==0.
        let n_eq_points = equity_points(n_ticks, equity_stride);
        let equity_len = n_strategies * n_eq_points;
        let mut d_equity_curves = self.device.alloc_async(equity_len.max(1))?;

        // Trade records as raw bytes; layout contract asserted host-side
        // (GpuTrade == CUDA struct Trade == 48 bytes).
        let trade_size = std::mem::size_of::<GpuTrade>();
        let trades_bytes_len = n_strategies * MAX_TRADES * trade_size;
        let mut d_trades = self.device.alloc_async_u8(trades_bytes_len)?;

        let mut d_num_trades = self.device.alloc_async_i32(n_strategies)?;

        // Overflow counters MUST be zero-initialized (kernel atomicAdds).
        let mut d_overflows = self
            .device
            .stream
            .alloc_zeros::<u32>(n_strategies)
            .map_err(|e| {
                GpuError::AllocationError(format!("Failed to allocate overflow counters: {:?}", e))
            })?;

        // Metrics (every element written unconditionally by the kernel)
        let mut d_final_equity = self.device.alloc_async(n_strategies)?;
        let mut d_total_return = self.device.alloc_async(n_strategies)?;
        let mut d_sharpe_ratios = self.device.alloc_async(n_strategies)?;
        let mut d_max_drawdowns = self.device.alloc_async(n_strategies)?;
        let mut d_win_rates = self.device.alloc_async(n_strategies)?;

        // ====================================================================
        // LAUNCH KERNEL (module cached per device; PTX cached process-wide)
        // ====================================================================

        let kernel = self
            .device
            .get_or_load_function(TICK_BACKTEST_KERNEL_SRC, "tick_backtest_batch_kernel")?;

        // One THREAD per strategy, packed into THREADS_PER_BLOCK-thread blocks
        let grid_blocks = n_strategies.div_ceil(THREADS_PER_BLOCK);
        let launch_config = LaunchConfig {
            grid_dim: (grid_blocks as u32, 1, 1),
            block_dim: (THREADS_PER_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_strategies_i32 = n_strategies as i32;
        let n_ticks_i32 = n_ticks as i32;
        let allow_short_i32: i32 = if self.config.allow_short { 1 } else { 0 };
        let equity_stride_i32 = equity_stride as i32;

        let mut builder = self.device.stream.launch_builder(&kernel);
        builder.arg(&d_signals);
        builder.arg(&d_prices);
        builder.arg(&d_timestamps);
        builder.arg(&mut d_equity_curves);
        builder.arg(&mut d_trades);
        builder.arg(&mut d_num_trades);
        builder.arg(&mut d_overflows);
        builder.arg(&mut d_final_equity);
        builder.arg(&mut d_total_return);
        builder.arg(&mut d_sharpe_ratios);
        builder.arg(&mut d_max_drawdowns);
        builder.arg(&mut d_win_rates);
        builder.arg(&n_strategies_i32);
        builder.arg(&n_ticks_i32);
        builder.arg(&self.config.initial_capital);
        builder.arg(&self.config.trading_fee);
        builder.arg(&self.config.slippage);
        builder.arg(&self.config.execution_delay_ms);
        builder.arg(&allow_short_i32);
        builder.arg(&equity_stride_i32);
        unsafe {
            builder
                .launch(launch_config)
                .map_err(|e| GpuError::ExecutionError(format!("Kernel launch failed: {:?}", e)))?;
        }

        // ====================================================================
        // COPY RESULTS BACK
        // ====================================================================

        self.device.synchronize()?;

        let h_equity_curves = if n_eq_points > 0 {
            self.device.copy_to_host(&d_equity_curves)?
        } else {
            Vec::new()
        };
        let h_num_trades: Vec<i32> =
            self.device.stream.memcpy_dtov(&d_num_trades).map_err(|e| {
                GpuError::MemoryCopyError(format!("Failed to copy trade counts: {:?}", e))
            })?;
        let h_overflows: Vec<u32> = self.device.stream.memcpy_dtov(&d_overflows).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy overflow counts: {:?}", e))
        })?;
        let h_final_equity = self.device.copy_to_host(&d_final_equity)?;
        let h_total_return = self.device.copy_to_host(&d_total_return)?;
        let h_sharpe_ratios = self.device.copy_to_host(&d_sharpe_ratios)?;
        let h_max_drawdowns = self.device.copy_to_host(&d_max_drawdowns)?;
        let h_win_rates = self.device.copy_to_host(&d_win_rates)?;
        let h_trades_bytes = self.device.copy_to_host_u8(&d_trades)?;

        // ====================================================================
        // PACKAGE RESULTS
        // ====================================================================

        let mut results = Vec::with_capacity(n_strategies);
        let strategy_trade_bytes = MAX_TRADES * trade_size;

        for strategy_idx in 0..n_strategies {
            let equity_curve = if n_eq_points > 0 {
                let start = strategy_idx * n_eq_points;
                h_equity_curves[start..start + n_eq_points].to_vec()
            } else {
                Vec::new()
            };

            let num_trades = (h_num_trades[strategy_idx] as usize).min(MAX_TRADES);
            let byte_start = strategy_idx * strategy_trade_bytes;
            let trades = parse_strategy_trades(
                &h_trades_bytes[byte_start..byte_start + strategy_trade_bytes],
                num_trades,
            );

            results.push(BacktestResult {
                final_equity: h_final_equity[strategy_idx],
                total_return: h_total_return[strategy_idx],
                sharpe_ratio: h_sharpe_ratios[strategy_idx],
                max_drawdown: h_max_drawdowns[strategy_idx],
                win_rate: h_win_rates[strategy_idx],
                num_trades: h_num_trades[strategy_idx],
                queue_overflows: h_overflows[strategy_idx],
                equity_curve,
                trades,
            });
        }

        Ok(results)
    }

    /// Benchmark throughput (ticks/sec)
    ///
    /// # Arguments
    ///
    /// - `n_strategies`: Number of strategies to test
    /// - `n_ticks`: Number of ticks per strategy
    /// - `warmup_runs`: Number of warmup runs (JIT compilation)
    /// - `benchmark_runs`: Number of benchmark runs for averaging
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kimsfinance_core::gpu::tick_backtest_batch::{TickBacktestBatch, BacktestConfig};
    /// # let backtest = TickBacktestBatch::new(BacktestConfig::default())?;
    /// let throughput = backtest.benchmark_throughput(10, 100_000, 2, 10)?;
    /// println!("Throughput: {:.2} M ticks/sec", throughput / 1e6);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn benchmark_throughput(
        &self,
        n_strategies: usize,
        n_ticks: usize,
        warmup_runs: usize,
        benchmark_runs: usize,
    ) -> Result<f64, GpuError> {
        use std::time::Instant;

        // Generate dummy data
        let signals: Vec<Vec<Signal>> = (0..n_strategies)
            .map(|_| {
                (0..n_ticks)
                    .map(|i| {
                        if i % 100 == 0 {
                            Signal::Buy
                        } else if i % 100 == 50 {
                            Signal::Sell
                        } else {
                            Signal::Hold
                        }
                    })
                    .collect()
            })
            .collect();

        let prices: Vec<f64> = (0..n_ticks).map(|i| 100.0 + (i as f64) * 0.01).collect();
        let timestamps: Vec<i64> = (0..n_ticks).map(|i| (i as i64) * 1000).collect();

        // Warmup
        for _ in 0..warmup_runs {
            self.run_batch(&signals, &prices, &timestamps)?;
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..benchmark_runs {
            self.run_batch(&signals, &prices, &timestamps)?;
        }
        let elapsed = start.elapsed();

        let total_ticks = (n_strategies * n_ticks * benchmark_runs) as f64;
        Ok(total_ticks / elapsed.as_secs_f64())
    }
}

// ============================================================================
// TESTS
// ============================================================================

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

    /// Uniform f64 in [0, 1)
    fn lcg_f64(state: &mut u64) -> f64 {
        ((lcg_next(state) >> 11) as f64) / (1u64 << 53) as f64
    }

    fn random_market(n_ticks: usize, seed: u64) -> (Vec<f64>, Vec<i64>) {
        let mut s = seed;
        let mut prices = Vec::with_capacity(n_ticks);
        let mut timestamps = Vec::with_capacity(n_ticks);
        let mut price = 50_000.0f64;
        let mut ts = 1_700_000_000_000i64;
        for _ in 0..n_ticks {
            price *= 1.0 + 0.0005 * (lcg_f64(&mut s) - 0.5);
            ts += 1 + (lcg_next(&mut s) % 250) as i64; // 1-250 ms gaps
            prices.push(price);
            timestamps.push(ts);
        }
        (prices, timestamps)
    }

    /// Raw pipeline signals: hold-biased mix of orderflow (-1/0/1) and
    /// legacy (2/3/4) encodings.
    fn random_raw_signals(n_ticks: usize, seed: u64) -> Vec<i8> {
        let mut s = seed;
        (0..n_ticks)
            .map(|_| match lcg_next(&mut s) % 16 {
                10 | 15 => 1i8,
                11 => -1,
                12 => 2,
                13 => 3,
                14 => 4,
                _ => 0,
            })
            .collect()
    }

    // ====================================================================
    // Host reference simulator
    //
    // Numerics (open/close/equity formulas, operation order) mirror
    // rust/src/backtest/tick_engine.rs verbatim — pinned against the real
    // TickEngine by test_reference_matches_tick_engine below. The signal
    // state machine mirrors the kernel's documented contract (the one
    // intentional divergence from the CPU engine: legacy SELL=2 closes the
    // long without opening a short; route -1 + allow_short for CPU
    // Sell-opens-short behavior).
    // ====================================================================

    struct RefPosition {
        cash: f64,
        position_size: f64,
        position_value: f64,
        entry_price: f64,
        entry_timestamp: i64,
    }

    struct RefTrade {
        pnl: f64,
        direction: i8,
        entry_time: i64,
        exit_time: i64,
    }

    struct RefResult {
        final_equity: f64,
        num_trades: usize,
        overflows: u32,
        equity_curve: Vec<f64>,
        trades: Vec<RefTrade>,
    }

    /// tick_engine.rs open_position (lines 331-357), identical expressions
    fn ref_open(
        pos: &mut RefPosition,
        price: f64,
        timestamp: i64,
        direction: f64,
        fee: f64,
        slip: f64,
    ) {
        let gross_position_value = pos.cash / price;
        let fee_cost = gross_position_value * price * fee;
        let slippage_cost = gross_position_value * price * slip;
        let total_cost = fee_cost + slippage_cost;

        pos.position_size = gross_position_value * direction;
        pos.entry_price = price;
        pos.entry_timestamp = timestamp;
        pos.position_value = pos.cash - total_cost;
        pos.cash = 0.0;
    }

    /// tick_engine.rs close_position (lines 367-418), identical expressions
    fn ref_close(
        pos: &mut RefPosition,
        exit_price: f64,
        exit_timestamp: i64,
        fee: f64,
        slip: f64,
        trades: &mut Vec<RefTrade>,
    ) {
        if pos.position_size == 0.0 {
            return;
        }
        let exit_value = pos.position_size.abs() * exit_price;
        let fee_cost = exit_value * fee;
        let slippage_cost = exit_value * slip;
        let pnl = if pos.position_size > 0.0 {
            exit_value - pos.position_value
        } else {
            pos.position_value - exit_value
        };
        pos.cash += pos.position_value + pnl - fee_cost - slippage_cost;

        if trades.len() < MAX_TRADES {
            trades.push(RefTrade {
                pnl,
                direction: if pos.position_size > 0.0 { 1 } else { -1 },
                entry_time: pos.entry_timestamp,
                exit_time: exit_timestamp,
            });
        }

        pos.position_size = 0.0;
        pos.position_value = 0.0;
        pos.entry_price = 0.0;
        pos.entry_timestamp = 0;
    }

    /// tick_engine.rs update_equity (lines 426-441), identical expressions
    fn ref_equity(pos: &RefPosition, price: f64) -> f64 {
        if pos.position_size == 0.0 {
            return pos.cash;
        }
        let unrealized = if pos.position_size > 0.0 {
            (price - pos.entry_price) * pos.position_size
        } else {
            (pos.entry_price - price) * pos.position_size.abs()
        };
        pos.cash + pos.position_value + unrealized
    }

    /// Kernel remap_signal mirror: raw byte -> kernel enum value (0..4)
    fn ref_remap(raw: i8, allow_short: bool) -> i8 {
        match raw {
            -1 => {
                if allow_short {
                    3
                } else {
                    2
                }
            }
            0..=4 => raw,
            _ => 0,
        }
    }

    /// Kernel execute_signal mirror
    fn ref_execute(
        raw: i8,
        allow_short: bool,
        pos: &mut RefPosition,
        price: f64,
        timestamp: i64,
        fee: f64,
        slip: f64,
        trades: &mut Vec<RefTrade>,
    ) {
        match ref_remap(raw, allow_short) {
            1 => {
                // BUY
                if pos.position_size <= 0.0 {
                    if pos.position_size < 0.0 {
                        ref_close(pos, price, timestamp, fee, slip, trades);
                    }
                    ref_open(pos, price, timestamp, 1.0, fee, slip);
                }
            }
            2 => {
                // SELL: close long only
                if pos.position_size > 0.0 {
                    ref_close(pos, price, timestamp, fee, slip, trades);
                }
            }
            3 => {
                // SHORT
                if pos.position_size >= 0.0 {
                    if pos.position_size > 0.0 {
                        ref_close(pos, price, timestamp, fee, slip, trades);
                    }
                    ref_open(pos, price, timestamp, -1.0, fee, slip);
                }
            }
            4 => {
                // COVER
                if pos.position_size < 0.0 {
                    ref_close(pos, price, timestamp, fee, slip, trades);
                }
            }
            _ => {} // HOLD
        }
    }

    /// Full host mirror of tick_backtest_batch_kernel for one strategy.
    fn reference_backtest_i8(
        signals: &[i8],
        prices: &[f64],
        timestamps: &[i64],
        config: &BacktestConfig,
        equity_stride: usize,
    ) -> RefResult {
        let n_ticks = prices.len();
        let mut pos = RefPosition {
            cash: config.initial_capital,
            position_size: 0.0,
            position_value: 0.0,
            entry_price: 0.0,
            entry_timestamp: 0,
        };
        let mut trades: Vec<RefTrade> = Vec::new();
        let mut pending: VecDeque<(i8, i64)> = VecDeque::new(); // (raw signal, execution_time)
        let mut overflows = 0u32;
        let mut equity_curve = Vec::new();
        let mut prev_equity = config.initial_capital;

        for tick in 0..n_ticks {
            let current_time = timestamps[tick];
            let current_price = prices[tick];
            let raw = signals[tick];

            // 1) due pending orders at the CURRENT tick's price/time
            while let Some(&(pending_raw, exec_time)) = pending.front() {
                if current_time < exec_time {
                    break;
                }
                ref_execute(
                    pending_raw,
                    config.allow_short,
                    &mut pos,
                    current_price,
                    current_time,
                    config.trading_fee,
                    config.slippage,
                    &mut trades,
                );
                pending.pop_front();
            }

            // 2) queue this tick's signal
            if raw != 0 {
                if pending.len() >= MAX_PENDING_ORDERS {
                    overflows += 1;
                    ref_execute(
                        raw,
                        config.allow_short,
                        &mut pos,
                        current_price,
                        current_time,
                        config.trading_fee,
                        config.slippage,
                        &mut trades,
                    );
                } else {
                    pending.push_back((raw, current_time + config.execution_delay_ms as i64));
                }
            }

            // 3) mark-to-market
            let current_equity = ref_equity(&pos, current_price);
            if equity_stride > 0 && tick % equity_stride == 0 {
                equity_curve.push(current_equity);
            }
            prev_equity = current_equity;
        }

        // CPU parity quirk: final equity is the pre-close mark-to-market
        let final_equity = prev_equity;
        if pos.position_size != 0.0 {
            ref_close(
                &mut pos,
                prices[n_ticks - 1],
                timestamps[n_ticks - 1],
                config.trading_fee,
                config.slippage,
                &mut trades,
            );
        }

        RefResult {
            final_equity,
            num_trades: trades.len(),
            overflows,
            equity_curve,
            trades,
        }
    }

    // ====================================================================
    // Host-only tests (no GPU)
    // ====================================================================

    /// The kernel is NVRTC-JIT-compiled: no #include, extern "C" __global__
    /// entry points only.
    #[test]
    fn test_kernel_source_is_nvrtc_compatible() {
        assert!(
            !TICK_BACKTEST_KERNEL_SRC.contains("#include"),
            "NVRTC kernel must not use #include directives"
        );
        assert!(
            TICK_BACKTEST_KERNEL_SRC
                .contains("extern \"C\" __global__ void tick_backtest_batch_kernel("),
            "missing extern \"C\" __global__ entry point"
        );
    }

    /// The #define lines in the kernel are a layout contract with this
    /// module's launch math and buffer sizing; keep them in lockstep.
    #[test]
    fn test_kernel_layout_contract_matches_host_constants() {
        assert!(
            TICK_BACKTEST_KERNEL_SRC.contains(&format!("#define MAX_TRADES {}", MAX_TRADES)),
            "kernel MAX_TRADES must match host const"
        );
        assert!(
            TICK_BACKTEST_KERNEL_SRC.contains(&format!(
                "#define MAX_PENDING_ORDERS {}",
                MAX_PENDING_ORDERS
            )),
            "kernel MAX_PENDING_ORDERS must match host const"
        );
        assert!(
            TICK_BACKTEST_KERNEL_SRC
                .contains(&format!("#define THREADS_PER_BLOCK {}", THREADS_PER_BLOCK)),
            "kernel THREADS_PER_BLOCK must match host const"
        );
    }

    /// GpuTrade must match the CUDA `struct Trade` byte for byte:
    /// 3 x f64 + 2 x i64 + i8 with natural alignment 8 -> 48 bytes.
    #[test]
    fn test_gpu_trade_layout_contract() {
        assert_eq!(std::mem::size_of::<GpuTrade>(), 48);
        assert_eq!(std::mem::align_of::<GpuTrade>(), 8);
    }

    /// Fixed bugs must not return: hot-loop printf, the shared-memory
    /// pending queue, the fee/slippage double charge via price
    /// pre-adjustment, and the one-active-thread-per-block launch shape.
    #[test]
    fn test_kernel_has_no_legacy_artifacts() {
        for legacy in [
            "printf",
            "__shared__ PendingOrder",
            "1.0 + slippage + trading_fee",
            "1.0 - slippage - trading_fee",
            "threadIdx.x != 0",
        ] {
            assert!(
                !TICK_BACKTEST_KERNEL_SRC.contains(legacy),
                "legacy artifact must not return: {}",
                legacy
            );
        }
        // The remap + overflow counter must be present
        assert!(TICK_BACKTEST_KERNEL_SRC.contains("remap_signal"));
        assert!(TICK_BACKTEST_KERNEL_SRC.contains("atomicAdd(&overflow_counts[strategy_idx], 1u)"));
    }

    /// The Rust Signal enum's declaration order (Buy=0, Sell=1, Hold=2) does
    /// NOT match the kernel encoding — the explicit map is load-bearing.
    #[test]
    fn test_signal_to_kernel_encoding() {
        assert_eq!(signal_to_kernel_i8(Signal::Hold), 0);
        assert_eq!(signal_to_kernel_i8(Signal::Buy), 1);
        assert_eq!(signal_to_kernel_i8(Signal::Sell), 2);
        assert_eq!(signal_to_kernel_i8(Signal::Short), 3);
        assert_eq!(signal_to_kernel_i8(Signal::Cover), 4);
        // Document the footgun the map fixes: a bare `as i8` cast does NOT
        // produce the kernel encoding for the first three variants.
        assert_ne!(Signal::Buy as i8, signal_to_kernel_i8(Signal::Buy));
        assert_ne!(Signal::Hold as i8, signal_to_kernel_i8(Signal::Hold));
    }

    #[test]
    fn test_equity_points() {
        assert_eq!(equity_points(1000, 0), 0);
        assert_eq!(equity_points(1000, 1), 1000);
        assert_eq!(equity_points(1000, 100), 10);
        assert_eq!(equity_points(1001, 100), 11);
        assert_eq!(equity_points(1, 100), 1);
    }

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();
        assert_eq!(config.execution_delay_ms, DEFAULT_EXECUTION_DELAY_MS);
        assert!(!config.allow_short);
    }

    #[test]
    fn test_parse_strategy_trades_unaligned() {
        let trade = GpuTrade {
            entry_price: 100.0,
            exit_price: 110.0,
            entry_time: 1_700_000_000_000,
            exit_time: 1_700_000_001_000,
            pnl: 1015.0,
            direction: 1,
        };
        let size = std::mem::size_of::<GpuTrade>();
        // Deliberately misalign the record inside the byte buffer (offset 1)
        // to prove the parser has no alignment requirement.
        let mut bytes = vec![0u8; 1 + size * 2];
        let src =
            unsafe { std::slice::from_raw_parts(&trade as *const GpuTrade as *const u8, size) };
        bytes[1..1 + size].copy_from_slice(src);
        bytes[1 + size..1 + 2 * size].copy_from_slice(src);

        let parsed = parse_strategy_trades(&bytes[1..], 2);
        assert_eq!(parsed.len(), 2);
        for t in &parsed {
            assert_eq!(t.entry_price, 100.0);
            assert_eq!(t.exit_price, 110.0);
            assert_eq!(t.entry_time, 1_700_000_000_000);
            assert_eq!(t.exit_time, 1_700_000_001_000);
            assert_eq!(t.pnl, 1015.0);
            assert_eq!(t.direction, 1);
        }
    }

    /// Hand-computed fee model: fees and slippage are charged exactly once
    /// (entry: on gross position value; exit: on exit value), never via
    /// price pre-adjustment. Numbers mirror tick_engine.rs formulas.
    #[test]
    fn test_reference_fee_model_hand_computed() {
        let config = BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
            execution_delay_ms: 0,
            allow_short: false,
        };
        // Buy queued at tick 0 executes at tick 1 (pending orders run before
        // new signals are queued, so even delay 0 executes next tick — same
        // as the CPU engine). Sell queued at tick 2 executes at tick 3.
        let signals = vec![1i8, 0, -1, 0];
        let prices = vec![100.0, 100.0, 110.0, 110.0];
        let timestamps = vec![0i64, 1000, 2000, 3000];

        let r = reference_backtest_i8(&signals, &prices, &timestamps, &config, 1);

        // Entry at 100: gross = 10000/100 = 100 units; entry costs =
        // 10000 * 0.0015 = 15 -> position_value = 9985.
        // Exit at 110: exit_value = 11000; pnl = 11000 - 9985 = 1015 (gross
        // of exit costs); exit costs = 11000 * 0.0015 = 16.5;
        // cash = 9985 + 1015 - 16.5 = 10983.5.
        assert_eq!(r.num_trades, 1);
        assert_eq!(r.trades[0].direction, 1);
        assert!((r.trades[0].pnl - 1015.0).abs() < 1e-6);
        assert!((r.final_equity - 10_983.5).abs() < 1e-6);
        // Mark-to-market before the exit executes: 9985 + 10 * 100 = 10985
        assert!((r.equity_curve[2] - 10_985.0).abs() < 1e-6);
        // With the old double-charge model the entry would cost 30 and the
        // exit 33: final equity ~10952 — assert we are far from it.
        assert!(r.final_equity > 10_983.0);
    }

    /// Orderflow -1 contract: SELL (close long only) without allow_short,
    /// SHORT (close long + open short) with it.
    #[test]
    fn test_reference_orderflow_sell_contract() {
        let signals = vec![1i8, 0, -1, 0, 0];
        let prices = vec![100.0, 100.0, 110.0, 110.0, 105.0];
        let timestamps = vec![0i64, 1000, 2000, 3000, 4000];

        let long_only = BacktestConfig {
            execution_delay_ms: 0,
            allow_short: false,
            ..Default::default()
        };
        let r = reference_backtest_i8(&signals, &prices, &timestamps, &long_only, 0);
        // Buy executes tick 1; -1 -> SELL executes tick 3 closing the long.
        assert_eq!(r.num_trades, 1);
        assert_eq!(r.trades[0].direction, 1);
        assert_eq!(r.overflows, 0);
        assert!(r.equity_curve.is_empty()); // stride 0

        let with_short = BacktestConfig {
            execution_delay_ms: 0,
            allow_short: true,
            ..Default::default()
        };
        let r = reference_backtest_i8(&signals, &prices, &timestamps, &with_short, 0);
        // Same long close at tick 3, then a short opens and is force-closed
        // at end of data.
        assert_eq!(r.num_trades, 2);
        assert_eq!(r.trades[0].direction, 1);
        assert_eq!(r.trades[1].direction, -1);
        // Short opened at 110, force-closed at 105 -> profitable short
        assert!(r.trades[1].pnl > 0.0);
    }

    /// Unknown signal bytes are defensive HOLDs; legacy 2..4 pass through.
    #[test]
    fn test_reference_remap_defensive() {
        assert_eq!(ref_remap(0, false), 0);
        assert_eq!(ref_remap(1, false), 1);
        assert_eq!(ref_remap(-1, false), 2);
        assert_eq!(ref_remap(-1, true), 3);
        assert_eq!(ref_remap(2, true), 2);
        assert_eq!(ref_remap(3, false), 3);
        assert_eq!(ref_remap(4, false), 4);
        assert_eq!(ref_remap(7, false), 0);
        assert_eq!(ref_remap(-5, true), 0);
    }

    /// Queue overflow: signals beyond MAX_PENDING_ORDERS execute immediately
    /// and are counted (no signal is silently dropped).
    #[test]
    fn test_reference_pending_overflow() {
        let n = MAX_PENDING_ORDERS + 50;
        let signals = vec![1i8; n];
        let prices: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.01).collect();
        let timestamps: Vec<i64> = (0..n).map(|i| i as i64 * 1000).collect();
        let config = BacktestConfig {
            // Delay far beyond the data so no queued order ever executes
            execution_delay_ms: 2_000_000_000,
            ..Default::default()
        };

        let r = reference_backtest_i8(&signals, &prices, &timestamps, &config, 0);
        assert_eq!(r.overflows, 50);
        // The first overflow BUY opened a long (queued orders never ran);
        // the long is force-closed at end of data.
        assert_eq!(r.num_trades, 1);
    }

    /// Pin the reference simulator against the normative CPU engine: a
    /// scripted strategy replays the same signal stream through TickEngine;
    /// final equity, trade count, and the 1-in-100 sampled equity curve must
    /// agree. Sell is excluded (the kernel's legacy SELL=2 deliberately
    /// closes the long without opening a short — documented divergence);
    /// Buy/Short/Cover/Hold semantics are identical in both engines.
    #[test]
    fn test_reference_matches_tick_engine() {
        use crate::backtest::{BacktestConfig as CpuConfig, TickEngine, TickStrategy};
        use crate::binance::{IncompleteCandle, Timeframe, Trade};

        struct ScriptedStrategy {
            signals: Vec<Signal>,
            idx: usize,
        }

        impl TickStrategy for ScriptedStrategy {
            fn on_tick(&mut self, _trade: &Trade, _candle: &IncompleteCandle) -> Signal {
                let s = self.signals[self.idx];
                self.idx += 1;
                s
            }

            fn name(&self) -> &str {
                "scripted"
            }
        }

        let n_ticks = 5_000;
        let (prices, timestamps) = random_market(n_ticks, 0xACE5);

        let mut s = 0xBEEF;
        let script: Vec<Signal> = (0..n_ticks)
            .map(|_| match lcg_next(&mut s) % 12 {
                8 => Signal::Buy,
                9 => Signal::Short,
                10 => Signal::Cover,
                _ => Signal::Hold,
            })
            .collect();

        let trades: Vec<Trade> = (0..n_ticks)
            .map(|i| Trade {
                trade_id: i as u64,
                price: prices[i],
                quantity: 1.0,
                quote_quantity: prices[i],
                timestamp_ms: timestamps[i],
                is_buyer_maker: false,
            })
            .collect();

        let cpu_config = CpuConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
            execution_latency_ms: 10,
            ..Default::default()
        };
        let engine = TickEngine::new(cpu_config);
        let mut strategy = ScriptedStrategy {
            signals: script.clone(),
            idx: 0,
        };
        let timeframe = Timeframe::parse("5m").unwrap();
        let cpu_result = engine.run(&mut strategy, &trades, timeframe).unwrap();

        let gpu_config = BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
            execution_delay_ms: 10,
            allow_short: false, // irrelevant: no -1 bytes in this script
        };
        let raw: Vec<i8> = script.iter().map(|&sig| signal_to_kernel_i8(sig)).collect();
        // Stride 100 mirrors the CPU engine's idx % 100 == 0 sampling
        let reference = reference_backtest_i8(&raw, &prices, &timestamps, &gpu_config, 100);

        assert!(
            reference.num_trades > 0,
            "degenerate script: no trades executed"
        );
        assert_eq!(reference.num_trades, cpu_result.num_trades);
        assert!(
            (reference.final_equity - cpu_result.final_equity).abs()
                <= 1e-9 * cpu_result.final_equity.abs().max(1.0),
            "final equity diverged: reference {} vs TickEngine {}",
            reference.final_equity,
            cpu_result.final_equity
        );
        assert_eq!(reference.equity_curve.len(), cpu_result.equity_curve.len());
        for (i, (a, b)) in reference
            .equity_curve
            .iter()
            .zip(cpu_result.equity_curve.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                "equity curve diverged at sample {}: {} vs {}",
                i,
                a,
                b
            );
        }

        // Trade-level pinning: directions, execution timestamps, and pnl
        use crate::backtest::TradeDirection;
        for (i, (rt, ct)) in reference
            .trades
            .iter()
            .zip(cpu_result.trades.iter())
            .enumerate()
        {
            let cpu_dir = match ct.direction {
                TradeDirection::Long => 1i8,
                TradeDirection::Short => -1i8,
            };
            assert_eq!(rt.direction, cpu_dir, "trade {} direction", i);
            assert_eq!(rt.entry_time, ct.entry_time, "trade {} entry time", i);
            assert_eq!(rt.exit_time, ct.exit_time, "trade {} exit time", i);
            assert!(
                (rt.pnl - ct.pnl).abs() <= 1e-9 * ct.pnl.abs().max(1.0),
                "trade {} pnl {} vs {}",
                i,
                rt.pnl,
                ct.pnl
            );
        }
    }

    // ====================================================================
    // GPU tests (require CUDA hardware; run with --ignored)
    // ====================================================================

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_tick_backtest_batch_basic() {
        let config = BacktestConfig::default();
        let backtest = TickBacktestBatch::new(config).unwrap();

        // Simple buy-hold-sell strategy
        let signals = vec![vec![Signal::Buy, Signal::Hold, Signal::Hold, Signal::Sell]];
        let prices = vec![100.0, 101.0, 102.0, 103.0];
        let timestamps = vec![0, 1000, 2000, 3000];

        let results = backtest.run_batch(&signals, &prices, &timestamps).unwrap();
        assert_eq!(results.len(), 1);

        let result = &results[0];
        assert!(result.final_equity > config.initial_capital); // Profitable trade
        assert_eq!(result.num_trades, 1); // forced close at end of data
        assert_eq!(result.equity_curve.len(), 4);
        assert_eq!(result.queue_overflows, 0);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_tick_backtest_batch_multiple_strategies() {
        let config = BacktestConfig::default();
        let backtest = TickBacktestBatch::new(config).unwrap();

        // 3 different strategies
        let signals = vec![
            vec![Signal::Buy, Signal::Sell, Signal::Buy, Signal::Sell],
            vec![Signal::Short, Signal::Cover, Signal::Short, Signal::Cover],
            vec![Signal::Hold, Signal::Hold, Signal::Hold, Signal::Hold],
        ];
        let prices = vec![100.0, 101.0, 102.0, 103.0];
        let timestamps = vec![0, 1000, 2000, 3000];

        let results = backtest.run_batch(&signals, &prices, &timestamps).unwrap();
        assert_eq!(results.len(), 3);

        // Strategy 0: Long trades
        assert!(results[0].num_trades >= 2);

        // Strategy 1: Short trades
        assert!(results[1].num_trades >= 2);

        // Strategy 2: No trades (hold only)
        assert_eq!(results[2].num_trades, 0);
        assert_eq!(results[2].final_equity, config.initial_capital);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_tick_backtest_batch_pending_orders() {
        let config = BacktestConfig {
            execution_delay_ms: 10,
            ..Default::default()
        };
        let backtest = TickBacktestBatch::new(config).unwrap();

        // Signal at t=0 should execute at the first tick with ts >= 10
        let signals = vec![vec![Signal::Buy, Signal::Hold, Signal::Hold, Signal::Sell]];
        let prices = vec![100.0, 101.0, 102.0, 103.0];
        let timestamps = vec![0, 5, 15, 20]; // buy executes at ts=15

        let results = backtest.run_batch(&signals, &prices, &timestamps).unwrap();
        assert_eq!(results.len(), 1);

        // Should have executed at least one trade
        assert!(results[0].num_trades >= 1);
    }

    /// GPU-vs-CPU parity over random raw signal streams: final equity within
    /// 1e-9 (relative), trade counts and overflow counts exact, equity curve
    /// samples within 1e-9 for every stride.
    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_gpu_cpu_parity_random_signals() {
        let n_ticks = 5_000;
        let n_strategies = 7;
        let (prices, timestamps) = random_market(n_ticks, 0xC0FFEE);

        let signals: Vec<Vec<i8>> = (0..n_strategies)
            .map(|i| random_raw_signals(n_ticks, 0x1234_5678 + i as u64))
            .collect();

        for &allow_short in &[false, true] {
            let config = BacktestConfig {
                allow_short,
                ..Default::default()
            };
            let backtest = TickBacktestBatch::new(config).unwrap();

            for &stride in &[0usize, 1, 100] {
                let results = backtest
                    .run_batch_i8(&signals, &prices, &timestamps, stride)
                    .unwrap();
                assert_eq!(results.len(), n_strategies);

                for (i, gpu) in results.iter().enumerate() {
                    let reference =
                        reference_backtest_i8(&signals[i], &prices, &timestamps, &config, stride);

                    assert_eq!(
                        gpu.num_trades as usize, reference.num_trades,
                        "strategy {} stride {} allow_short {}: trade count",
                        i, stride, allow_short
                    );
                    assert_eq!(
                        gpu.queue_overflows, reference.overflows,
                        "strategy {} stride {} allow_short {}: overflow count",
                        i, stride, allow_short
                    );
                    assert!(
                        (gpu.final_equity - reference.final_equity).abs()
                            <= 1e-9 * reference.final_equity.abs().max(1.0),
                        "strategy {} stride {} allow_short {}: final equity {} vs {}",
                        i,
                        stride,
                        allow_short,
                        gpu.final_equity,
                        reference.final_equity
                    );
                    assert_eq!(gpu.equity_curve.len(), reference.equity_curve.len());
                    for (k, (a, b)) in gpu
                        .equity_curve
                        .iter()
                        .zip(reference.equity_curve.iter())
                        .enumerate()
                    {
                        assert!(
                            (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                            "strategy {} stride {} sample {}: {} vs {}",
                            i,
                            stride,
                            k,
                            a,
                            b
                        );
                    }
                    // Per-trade parity: pnl within 1e-9; directions and
                    // execution timestamps exact (integers)
                    for (k, (gt, rt)) in gpu.trades.iter().zip(reference.trades.iter()).enumerate()
                    {
                        assert_eq!(gt.direction, rt.direction, "trade {} direction", k);
                        assert_eq!(gt.entry_time, rt.entry_time, "trade {} entry time", k);
                        assert_eq!(gt.exit_time, rt.exit_time, "trade {} exit time", k);
                        assert!(
                            (gt.pnl - rt.pnl).abs() <= 1e-9 * rt.pnl.abs().max(1.0),
                            "trade {} pnl {} vs {}",
                            k,
                            gt.pnl,
                            rt.pnl
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore] // Requires CUDA hardware and takes time
    fn test_tick_backtest_batch_throughput() {
        let config = BacktestConfig::default();
        let backtest = TickBacktestBatch::new(config).unwrap();

        let throughput = backtest.benchmark_throughput(10, 100_000, 2, 5).unwrap();

        println!("Throughput: {:.2} M ticks/sec", throughput / 1e6);
        assert!(
            throughput > 1e8,
            "Throughput too low: {:.2} M ticks/sec",
            throughput / 1e6
        );
    }
}
