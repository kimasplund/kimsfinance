//! GPU Tick-Level Batch Backtest with Pending Orders Queue
//!
//! # Architecture
//!
//! - **Sequential per-strategy**: Maintains position state correctness
//! - **Parallel across strategies**: 10-20 strategies in parallel
//! - **Pending orders queue**: 10ms execution delay simulation (configurable)
//! - **Exact CPU matching**: <0.01% deviation from CPU backtest
//!
//! # Performance Target
//!
//! - **Throughput**: 1-1.5B ticks/sec (10-20 strategies in parallel)
//! - **Latency**: 10ms execution delay (configurable)
//! - **GPU Utilization**: >80% during execution
//!
//! # Memory Requirements
//!
//! Per strategy:
//! - Pending orders: 2KB (100 orders × 20 bytes)
//! - Trades: 20KB (1000 trades × 20 bytes)
//! - Total: ~22KB per strategy + input/output arrays
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
//! };
//!
//! let backtest = TickBacktestBatch::new(config)?;
//!
//! // Run 10 strategies in parallel
//! let signals = vec![vec![Signal::Buy, Signal::Hold, Signal::Sell]; 10];
//! let prices = vec![100.0, 101.0, 102.0];
//! let timestamps = vec![0, 1000, 2000];  // milliseconds
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

/// Maximum trades per strategy (matches CPU MAX_TRADES)
pub const MAX_TRADES: usize = 1000;

/// Maximum pending orders per strategy
pub const MAX_PENDING_ORDERS: usize = 100;

/// Default execution delay in milliseconds
pub const DEFAULT_EXECUTION_DELAY_MS: i32 = 10;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Trade record (matches GPU Trade struct)
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
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10_000.0,
            trading_fee: 0.001,     // 0.1%
            slippage: 0.0005,       // 0.05%
            execution_delay_ms: DEFAULT_EXECUTION_DELAY_MS,
        }
    }
}

/// Backtest results for a single strategy
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub final_equity: f64,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub num_trades: i32,
    pub equity_curve: Vec<f64>,
    pub trades: Vec<GpuTrade>,
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
    /// - `config`: Backtest configuration (fees, slippage, initial capital, delay)
    ///
    /// # Returns
    ///
    /// New TickBacktestBatch instance or error
    ///
    /// # Example
    ///
    /// ```rust
    /// use kimsfinance_core::gpu::tick_backtest_batch::{TickBacktestBatch, BacktestConfig};
    ///
    /// let config = BacktestConfig::default();
    /// let backtest = TickBacktestBatch::new(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(config: BacktestConfig) -> Result<Self, GpuError> {
        // Initialize CUDA device (uses cudarc internally)
        let device = Arc::new(GpuDevice::new()?);

        Ok(Self {
            device,
            config,
        })
    }

    /// Run batch backtest on GPU
    ///
    /// # Arguments
    ///
    /// - `signals`: Signal arrays [N_strategies][N_ticks]
    /// - `prices`: Price array [N_ticks]
    /// - `timestamps`: Timestamp array [N_ticks] (milliseconds)
    ///
    /// # Returns
    ///
    /// Vec of BacktestResult (one per strategy)
    ///
    /// # Errors
    ///
    /// Returns error string if:
    /// - CUDA memory allocation fails
    /// - Kernel launch fails
    /// - Results transfer fails
    ///
    /// # Performance
    ///
    /// - **Target**: 1-1.5B ticks/sec (10-20 strategies)
    /// - **GPU Utilization**: >80% during execution
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
    /// assert_eq!(results.len(), 2);  // One result per strategy
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn run_batch(
        &self,
        signals: &[Vec<Signal>],
        prices: &[f64],
        timestamps: &[i64],
    ) -> Result<Vec<BacktestResult>, GpuError> {
        let n_strategies = signals.len();
        let n_ticks = prices.len();

        // Validation
        if n_strategies == 0 {
            return Err(GpuError::InvalidInput("No strategies provided".to_string()));
        }
        if n_ticks == 0 {
            return Err(GpuError::InvalidInput("No ticks provided".to_string()));
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

        // Convert signals to i8 array (flattened)
        let signals_flat: Vec<i8> = signals
            .iter()
            .flat_map(|strategy_signals| strategy_signals.iter().map(|s| *s as i8))
            .collect();

        // ====================================================================
        // ALLOCATE GPU MEMORY
        // ====================================================================

        // Inputs - copy to device using appropriate types
        let d_signals = self.device.copy_to_device_i8(&signals_flat)?;
        let d_prices = self.device.copy_to_device(prices)?;
        let d_timestamps = self.device.copy_to_device_i64(timestamps)?;

        // Outputs - allocate device buffers
        let mut d_equity_curves = self.device.alloc_async(n_strategies * n_ticks)?;

        // For complex types like GpuTrade, we need to allocate as raw bytes
        // GpuTrade is repr(C): 6*f64 + 2*i64 + i8 = 48 + 16 + 1 = 65 bytes, padded to 72
        let trade_size_bytes = std::mem::size_of::<GpuTrade>();
        let bytes_len = n_strategies * MAX_TRADES * trade_size_bytes;
        let trades_buffer_bytes = self.device.alloc_async_u8(bytes_len)?;

        let mut d_num_trades = self.device.alloc_async_i32(n_strategies)?;

        // Metrics
        let mut d_final_equity = self.device.alloc_async(n_strategies)?;
        let mut d_total_return = self.device.alloc_async(n_strategies)?;
        let mut d_sharpe_ratios = self.device.alloc_async(n_strategies)?;
        let mut d_max_drawdowns = self.device.alloc_async(n_strategies)?;
        let mut d_win_rates = self.device.alloc_async(n_strategies)?;

        // ====================================================================
        // COMPILE AND LAUNCH KERNEL
        // ====================================================================

        // Compile kernel using cached PTX compilation
        let kernel_source = include_str!("kernels/tick_backtest_batch.cu");
        let ptx_arc = crate::gpu::compile::compile_ptx_optimized_cached(kernel_source)?;
        let ptx = std::sync::Arc::unwrap_or_clone(ptx_arc);

        // Load PTX module
        let module = self
            .device
            .context()
            .load_module(ptx)
            .map_err(|e| {
                GpuError::CompilationError(format!("Failed to load PTX module: {:?}", e))
            })?;

        // Get kernel function
        let kernel = module.load_function("tick_backtest_batch_kernel").map_err(|e| {
            GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
        })?;

        // One block per strategy, one thread per block (sequential per-strategy)
        let config = LaunchConfig {
            grid_dim: (n_strategies as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // Prepare kernel arguments
        let n_strategies_i32 = n_strategies as i32;
        let n_ticks_i32 = n_ticks as i32;

        // Launch kernel using cudarc LaunchBuilder pattern
        let mut builder = self.device.stream.launch_builder(&kernel);
        builder.push(&d_signals);
        builder.push(&d_prices);
        builder.push(&d_timestamps);
        builder.push(&mut d_equity_curves);
        builder.push(&trades_buffer_bytes);
        builder.push(&mut d_num_trades);
        builder.push(&mut d_final_equity);
        builder.push(&mut d_total_return);
        builder.push(&mut d_sharpe_ratios);
        builder.push(&mut d_max_drawdowns);
        builder.push(&mut d_win_rates);
        builder.push(&n_strategies_i32);
        builder.push(&n_ticks_i32);
        builder.push(&self.config.initial_capital);
        builder.push(&self.config.trading_fee);
        builder.push(&self.config.slippage);
        builder.push(&self.config.execution_delay_ms);

        unsafe {
            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("Kernel launch failed: {:?}", e))
            })?;
        }

        // ====================================================================
        // COPY RESULTS BACK
        // ====================================================================

        // Synchronize to ensure kernel completion
        self.device.synchronize()?;

        // Copy results from device to host using GpuDevice's copy_to_host
        let h_equity_curves = self.device.copy_to_host(&d_equity_curves)?;
        let h_num_trades = self.device.copy_to_host_i32(&d_num_trades)?;
        let h_final_equity = self.device.copy_to_host(&d_final_equity)?;
        let h_total_return = self.device.copy_to_host(&d_total_return)?;
        let h_sharpe_ratios = self.device.copy_to_host(&d_sharpe_ratios)?;
        let h_max_drawdowns = self.device.copy_to_host(&d_max_drawdowns)?;
        let h_win_rates = self.device.copy_to_host(&d_win_rates)?;

        // Copy trades buffer as bytes, then reinterpret
        let h_trades_bytes = self.device.copy_to_host_u8(&trades_buffer_bytes)?;
        let h_trades: Vec<GpuTrade> = unsafe {
            // SAFETY: GpuTrade is repr(C) and matches CUDA kernel struct layout
            std::slice::from_raw_parts(
                h_trades_bytes.as_ptr() as *const GpuTrade,
                n_strategies * MAX_TRADES,
            )
            .to_vec()
        };

        // ====================================================================
        // PACKAGE RESULTS
        // ====================================================================

        let mut results = Vec::with_capacity(n_strategies);

        for strategy_idx in 0..n_strategies {
            let equity_start = strategy_idx * n_ticks;
            let equity_end = equity_start + n_ticks;
            let equity_curve = h_equity_curves[equity_start..equity_end].to_vec();

            let num_trades = h_num_trades[strategy_idx] as usize;
            let trade_start = strategy_idx * MAX_TRADES;
            let trade_end = trade_start + num_trades.min(MAX_TRADES);
            let trades = h_trades[trade_start..trade_end].to_vec();

            results.push(BacktestResult {
                final_equity: h_final_equity[strategy_idx],
                total_return: h_total_return[strategy_idx],
                sharpe_ratio: h_sharpe_ratios[strategy_idx],
                max_drawdown: h_max_drawdowns[strategy_idx],
                win_rate: h_win_rates[strategy_idx],
                num_trades: h_num_trades[strategy_idx],
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
    /// # Returns
    ///
    /// Average throughput in ticks/sec
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
        let throughput = total_ticks / elapsed.as_secs_f64();

        Ok(throughput)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_tick_backtest_batch_basic() {
        let config = BacktestConfig::default();
        let backtest = TickBacktestBatch::new(config).unwrap();

        // Simple buy-hold-sell strategy
        let signals = vec![
            vec![Signal::Buy, Signal::Hold, Signal::Hold, Signal::Sell],
        ];
        let prices = vec![100.0, 101.0, 102.0, 103.0];
        let timestamps = vec![0, 1000, 2000, 3000];

        let results = backtest.run_batch(&signals, &prices, &timestamps).unwrap();
        assert_eq!(results.len(), 1);

        let result = &results[0];
        assert!(result.final_equity > config.initial_capital); // Profitable trade
        assert_eq!(result.num_trades, 1);
        assert_eq!(result.equity_curve.len(), 4);
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

        // Signal at t=0 should execute at t=10
        let signals = vec![
            vec![Signal::Buy, Signal::Hold, Signal::Hold, Signal::Sell],
        ];
        let prices = vec![100.0, 101.0, 102.0, 103.0];
        let timestamps = vec![0, 5, 15, 20]; // 10ms delay means buy executes between 5 and 15

        let results = backtest.run_batch(&signals, &prices, &timestamps).unwrap();
        assert_eq!(results.len(), 1);

        // Should have executed at least one trade
        assert!(results[0].num_trades >= 1);
    }

    #[test]
    #[ignore] // Requires CUDA hardware and takes time
    fn test_tick_backtest_batch_throughput() {
        let config = BacktestConfig::default();
        let backtest = TickBacktestBatch::new(config).unwrap();

        let throughput = backtest.benchmark_throughput(10, 100_000, 2, 5).unwrap();

        // Target: 1B ticks/sec minimum
        println!("Throughput: {:.2} M ticks/sec", throughput / 1e6);
        assert!(throughput > 1e9, "Throughput too low: {:.2} M ticks/sec", throughput / 1e6);
    }
}
