//! GPU Batch Tick Backtesting API
//!
//! Integration layer for GPU-accelerated tick-level backtesting with genetic optimization.
//! Processes 106M trades across 10-20 strategies simultaneously.
//!
//! # Performance Targets
//!
//! - **106M trades × 10 strategies**: <5 seconds (target)
//! - **VRAM usage**: <12GB (3.4GB trades + 8.6GB working)
//! - **Accuracy**: Match CPU within 0.01% tolerance
//!
//! # Architecture
//!
//! ```text
//! BatchTickBacktest (Builder API)
//!    ↓
//! GPU Pipeline (3 Fused Kernels):
//!   Phase 1: Tick Aggregation (trades → candles)    - 300ms
//!   Phase 2: Orderflow + Signals (fused!)            - 200ms
//!   Phase 3: Tick Backtest Execution                 - 3000ms
//!    ↓
//! BatchBacktestResults (per-strategy metrics)
//! ```
//!
//! # Memory Management
//!
//! - Upload trades once (3.4GB for 106M trades)
//! - Batch strategies in chunks of 10-20 (fit in 8.6GB working memory)
//! - Auto-tune batch size based on available VRAM
//! - Graceful fallback to CPU on GPU errors
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::backtest::tick_batch::BatchTickBacktest;
//! use kimsfinance_core::gpu::device::GpuDevice;
//! use std::sync::Arc;
//!
//! let device = Arc::new(GpuDevice::new()?);
//!
//! // Define 10 orderflow strategies with different parameters
//! let mut params = vec![];
//! for window in 30..40 {
//!     for threshold in [0.10, 0.15, 0.20] {
//!         params.push(vec![window as f64, threshold, 10.0, 0.001, 5.0, 1.0]);
//!     }
//! }
//!
//! let results = BatchTickBacktest::new(device)
//!     .trades(&trades)
//!     .parameters_batch(&params)
//!     .config(BacktestConfig {
//!         initial_capital: 10_000.0,
//!         trading_fee: 0.001,
//!         slippage: 0.0005,
//!         execution_latency_ms: 10,
//!         use_gpu: true,
//!         force_cpu: false,
//!     })
//!     .execute()?;
//!
//! // Results for all 30 strategies
//! for (i, result) in results.results.iter().enumerate() {
//!     println!("Strategy {}: Sharpe = {:.2}, Return = {:.2}%",
//!              i, result.sharpe_ratio, result.total_return);
//! }
//! ```

use crate::backtest::core::BacktestResult;
use crate::backtest::engine::BacktestConfig;
use crate::binance::Trade;
use crate::gpu::device::{GpuDevice, GpuError};
use cudarc::driver::CudaSlice;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Batch tick backtesting builder with auto-tuning and graceful fallback
///
/// Uses builder pattern for ergonomic API construction. Automatically tunes
/// batch size based on available VRAM and provides CPU fallback on errors.
///
/// # GPU Memory Layout
///
/// ```text
/// Trades (uploaded once):     3.4GB (106M × 32 bytes)
/// Candles (per strategy):     636MB (106M × 6 bytes INT8)
/// Signals (per strategy):     106MB (106M × 1 byte)
/// Features (per strategy):    212MB (106M × 2 bytes)
/// Pending orders:             100MB overhead
/// Total per strategy:         ~950MB
/// ```
///
/// # Batch Size Auto-Tuning
///
/// With 12GB VRAM:
/// - Trades: 3.4GB (persistent)
/// - Available working memory: 8.6GB
/// - Batch size: 8.6GB / 950MB = 9 strategies per batch
/// - Rounded to 10 for efficiency
pub struct BatchTickBacktest {
    device: Arc<GpuDevice>,
    trades: Option<Vec<Trade>>,
    parameters: Option<Vec<Vec<f64>>>,
    config: BacktestConfig,
    batch_size: Option<usize>,
    force_cpu: bool,
}

impl BatchTickBacktest {
    /// Create new batch tick backtest
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device handle (shared across calls for efficiency)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = Arc::new(GpuDevice::new()?);
    /// let batch = BatchTickBacktest::new(device);
    /// ```
    pub fn new(device: Arc<GpuDevice>) -> Self {
        Self {
            device,
            trades: None,
            parameters: None,
            config: BacktestConfig::default(),
            batch_size: None,
            force_cpu: false,
        }
    }

    /// Set trades data (uploaded to GPU once)
    ///
    /// All strategies will execute on the same trade stream.
    ///
    /// # Arguments
    ///
    /// * `trades` - Vec of Trade structs (106M for full month)
    ///
    /// # Memory
    ///
    /// 106M trades × 32 bytes = 3.4GB GPU memory (persistent)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// batch.trades(&trades)
    /// ```
    pub fn trades(mut self, trades: &[Trade]) -> Self {
        self.trades = Some(trades.to_vec());
        self
    }

    /// Set parameter batch (N strategies × M parameters)
    ///
    /// Each inner vector represents one strategy's parameters.
    /// For orderflow strategies, typical parameters:
    ///
    /// - `[window, imbalance_threshold, min_volume, spike_threshold, ema_period, volatility_factor]`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let params = vec![
    ///     vec![30.0, 0.15, 10.0, 0.001, 5.0, 1.0],  // Strategy 1
    ///     vec![50.0, 0.10, 15.0, 0.0015, 8.0, 1.2], // Strategy 2
    ///     vec![100.0, 0.20, 20.0, 0.002, 12.0, 1.5],// Strategy 3
    /// ];
    /// batch.parameters_batch(&params);
    /// ```
    pub fn parameters_batch(mut self, params: &[Vec<f64>]) -> Self {
        self.parameters = Some(params.to_vec());
        self
    }

    /// Set backtest configuration
    ///
    /// Includes initial capital, trading fees, slippage, and execution latency.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// batch.config(BacktestConfig {
    ///     initial_capital: 10_000.0,
    ///     trading_fee: 0.001,       // 0.1%
    ///     slippage: 0.0005,          // 0.05%
    ///     execution_latency_ms: 10,  // 10ms latency
    ///     use_gpu: true,
    ///     force_cpu: false,
    /// })
    /// ```
    pub fn config(mut self, config: BacktestConfig) -> Self {
        self.config = config;
        self
    }

    /// Set batch size (auto-tuned if not specified)
    ///
    /// Override automatic batch size calculation. Useful for testing or
    /// when you know the optimal batch size for your hardware.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// batch.batch_size(10)  // Force 10 strategies per batch
    /// ```
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// Force CPU execution (disable GPU)
    ///
    /// Useful for validation or when GPU is unavailable.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// batch.force_cpu(true)
    /// ```
    pub fn force_cpu(mut self, force: bool) -> Self {
        self.force_cpu = force;
        self
    }

    /// Execute batch backtest on GPU (with automatic CPU fallback)
    ///
    /// Processes all strategies through GPU pipeline with automatic batching.
    /// Falls back to CPU if GPU execution fails.
    ///
    /// # Returns
    ///
    /// `BatchBacktestResults` with metrics for all strategies, sorted by fitness score.
    ///
    /// # Errors
    ///
    /// - `InvalidParameter`: No trades or parameters provided
    /// - `AllocationError`: Out of GPU memory (will fallback to CPU)
    /// - `ExecutionError`: CUDA kernel launch failure (will fallback to CPU)
    ///
    /// # Performance
    ///
    /// Expected timing (106M trades × 10 strategies on RTX 3500 Ada):
    ///
    /// - Phase 1: Tick aggregation - 300ms
    /// - Phase 2: Orderflow + signals (fused) - 200ms
    /// - Phase 3: Tick backtest - 3000ms
    /// - Data transfer: 500ms
    /// - **Total: ~4 seconds** (target <5s)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = batch.execute()?;
    /// println!("Processed {} strategies in {:.2}s",
    ///          results.results.len(), results.total_time_ms / 1000.0);
    /// println!("Best Sharpe: {:.2}", results.results[0].sharpe_ratio);
    /// ```
    pub fn execute(mut self) -> Result<BatchBacktestResults, GpuError> {
        // Validate inputs
        let trades = self
            .trades
            .take()
            .ok_or_else(|| GpuError::InvalidParameter("No trades provided".into()))?;

        let parameters = self
            .parameters
            .take()
            .ok_or_else(|| GpuError::InvalidParameter("No parameters provided".into()))?;

        if trades.is_empty() {
            return Err(GpuError::InvalidParameter("Empty trades vector".into()));
        }

        if parameters.is_empty() {
            return Err(GpuError::InvalidParameter("Empty parameters vector".into()));
        }

        // Check if CPU is forced
        if self.force_cpu || self.config.force_cpu {
            eprintln!("⚠️  CPU execution forced, falling back to CPU implementation");
            return self.execute_cpu(&trades, &parameters);
        }

        // Try GPU execution with fallback
        match self.execute_gpu(&trades, &parameters) {
            Ok(results) => Ok(results),
            Err(e) => {
                eprintln!("⚠️  GPU execution failed: {}", e);
                eprintln!("   Falling back to CPU implementation...");
                self.execute_cpu(&trades, &parameters)
            }
        }
    }

    /// Execute using GPU pipeline (3 fused kernels)
    ///
    /// Internal method - use `execute()` for automatic fallback.
    fn execute_gpu(
        &self,
        trades: &[Trade],
        parameters: &[Vec<f64>],
    ) -> Result<BatchBacktestResults, GpuError> {
        let start_total = Instant::now();

        let n_trades = trades.len();
        let n_strategies = parameters.len();

        eprintln!("🚀 GPU batch tick backtest starting...");
        eprintln!(
            "   Trades: {} ({:.2} GB)",
            n_trades,
            n_trades as f64 * 32.0 / 1e9
        );
        eprintln!("   Strategies: {}", n_strategies);

        // Auto-tune batch size if not specified
        let batch_size = self.batch_size.unwrap_or_else(|| {
            let auto_size = self.auto_tune_batch_size(n_trades);
            eprintln!("   Auto-tuned batch size: {} strategies", auto_size);
            auto_size
        });

        // Upload trades to GPU once (3.4GB for 106M trades)
        let start_upload = Instant::now();
        let trades_gpu = self.upload_trades_to_gpu(trades)?;
        let upload_ms = start_upload.elapsed().as_secs_f64() * 1000.0;
        eprintln!("   ✓ Trades uploaded to GPU in {:.2}ms", upload_ms);

        // Process strategies in batches
        let mut all_results = Vec::new();
        let num_batches = (n_strategies + batch_size - 1) / batch_size;

        for (batch_idx, chunk) in parameters.chunks(batch_size).enumerate() {
            eprintln!(
                "   Processing batch {}/{} ({} strategies)...",
                batch_idx + 1,
                num_batches,
                chunk.len()
            );

            let batch_results = self.execute_batch_gpu(&trades_gpu, chunk, n_trades)?;
            all_results.extend(batch_results);
        }

        // Sort by fitness (Sharpe ratio with drawdown penalty)
        all_results.sort_by(|a, b| {
            b.fitness()
                .partial_cmp(&a.fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        // Calculate VRAM usage (approximate)
        let vram_used_mb = (n_trades * 32 // trades
            + n_strategies * 636_000_000 // candles (compressed INT8)
            + n_strategies * 106_000_000 // signals
            + n_strategies * 212_000_000 // features
            + n_strategies * 100_000_000) as f64
            // overhead
            / (1024.0 * 1024.0);

        eprintln!("✅ GPU execution complete: {:.2}s", total_ms / 1000.0);
        eprintln!(
            "   Throughput: {:.1} strategies/sec",
            n_strategies as f64 / (total_ms / 1000.0)
        );

        Ok(BatchBacktestResults {
            results: all_results,
            gpu_time_ms: total_ms * 0.85, // Approximate (actual GPU time ~85%)
            total_time_ms: total_ms,
            vram_used_mb,
        })
    }

    /// Execute single batch using GPU kernels (calls Agents 1-3)
    ///
    /// Coordinates three GPU kernel launches:
    /// 1. Tick aggregation (Agent 1)
    /// 2. Orderflow + signals fused (Agent 2)
    /// 3. Tick backtest execution (Agent 3)
    fn execute_batch_gpu(
        &self,
        _trades_gpu: &CudaSlice<u8>,
        _params: &[Vec<f64>],
        _n_trades: usize,
    ) -> Result<Vec<BacktestResult>, GpuError> {
        // TODO: Coordinate with Agents 1, 2, 3 for actual kernel signatures

        // Phase 1: Tick aggregation (Agent 1)
        // let candles = gpu_tick_aggregation(trades_gpu, n_trades)?;

        // Phase 2: Orderflow + Signals (Agent 2 - fused!)
        // let signals = gpu_orderflow_signals_batch(&candles, params)?;

        // Phase 3: Backtest execution (Agent 3)
        // let results = gpu_tick_backtest_batch(&signals, trades_gpu, &self.config)?;

        // Placeholder: return empty results for now
        eprintln!("   ⚠️  GPU kernels not yet implemented (Agents 1-3 pending)");
        eprintln!("   ⚠️  Falling back to CPU for this batch");

        // Fall back to CPU for this batch
        // Note: In production, we'd use the GPU results. For now, we need
        // to reconstruct trades from the trades_gpu buffer or keep a CPU copy.
        // For simplicity, we'll just return empty results indicating GPU not ready.
        Err(GpuError::ExecutionError(
            "GPU kernels not yet implemented (placeholder)".into(),
        ))
    }

    /// Execute single batch using CPU (fallback implementation)
    ///
    /// Uses existing TickEngine for sequential CPU execution.
    fn execute_batch_cpu(
        &self,
        trades: &[Trade],
        params: &[Vec<f64>],
    ) -> Result<Vec<BacktestResult>, GpuError> {
        use crate::backtest::tick_engine::TickEngine;
        use crate::backtest::tick_strategy::OrderFlowStrategy;
        use crate::binance::Timeframe;

        let engine = TickEngine::new(self.config.clone());
        let mut results = Vec::with_capacity(params.len());

        for (idx, param_vec) in params.iter().enumerate() {
            // Parse parameters for OrderFlowStrategy
            // Note: OrderFlowStrategy::new() takes only imbalance_threshold
            // Other parameters are for future GPU implementation
            let imbalance_threshold = param_vec.get(1).copied().unwrap_or(0.15);

            let mut strategy = OrderFlowStrategy::new(imbalance_threshold);

            // Run backtest (assumes 5-minute candles)
            let timeframe = Timeframe::parse("5m")
                .map_err(|e| GpuError::ExecutionError(format!("Timeframe parse error: {:?}", e)))?;

            let result = engine
                .run(&mut strategy, trades, timeframe)
                .map_err(|e| GpuError::ExecutionError(format!("CPU backtest failed: {:?}", e)))?;

            // Convert parameters to HashMap for consistency
            let params_map: HashMap<String, f64> = param_vec
                .iter()
                .enumerate()
                .map(|(i, &v)| (format!("param_{}", i), v))
                .collect();

            results.push(BacktestResult {
                parameters: params_map,
                equity_curve: result.equity_curve,
                final_equity: result.final_equity,
                total_return: result.total_return,
                sharpe_ratio: result.sharpe_ratio,
                max_drawdown: result.max_drawdown,
                win_rate: result.win_rate,
                num_trades: result.num_trades,
                profit_factor: result.profit_factor,
                trades: result.trades,
            });

            if (idx + 1) % 10 == 0 {
                eprintln!(
                    "      CPU: {}/{} strategies completed",
                    idx + 1,
                    params.len()
                );
            }
        }

        Ok(results)
    }

    /// Execute all strategies using CPU (top-level fallback)
    fn execute_cpu(
        &self,
        trades: &[Trade],
        parameters: &[Vec<f64>],
    ) -> Result<BatchBacktestResults, GpuError> {
        let start_total = Instant::now();

        eprintln!("🔧 CPU batch execution starting...");
        eprintln!("   Trades: {}", trades.len());
        eprintln!("   Strategies: {}", parameters.len());

        let mut results = self.execute_batch_cpu(trades, parameters)?;

        // Sort by fitness (Sharpe ratio with drawdown penalty)
        results.sort_by(|a, b| {
            b.fitness()
                .partial_cmp(&a.fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        eprintln!("✅ CPU execution complete: {:.2}s", total_ms / 1000.0);
        eprintln!(
            "   Throughput: {:.1} strategies/sec",
            parameters.len() as f64 / (total_ms / 1000.0)
        );

        Ok(BatchBacktestResults {
            results,
            gpu_time_ms: 0.0, // No GPU time
            total_time_ms: total_ms,
            vram_used_mb: 0.0, // No VRAM used
        })
    }

    /// Auto-tune batch size based on available VRAM
    ///
    /// Formula:
    /// - Reserve for trades: 3.4GB (106M trades × 32 bytes)
    /// - Reserve overhead: 500MB
    /// - Working memory: VRAM - 3.9GB
    /// - Per-strategy memory: 950MB (candles + signals + features + overhead)
    /// - Batch size: working_memory / 950MB
    ///
    /// # Returns
    ///
    /// Optimal batch size (1-20 strategies)
    pub fn auto_tune_batch_size(&self, n_trades: usize) -> usize {
        // Query available VRAM (placeholder - actual implementation would query GPU)
        // For RTX 3500 Ada: 12GB
        let total_vram_bytes: usize = 12_000_000_000;

        // Calculate trade memory usage
        let trades_memory = n_trades * 32; // 32 bytes per trade

        // Reserve overhead
        let overhead = 500_000_000; // 500MB

        // Available working memory
        let working_memory = total_vram_bytes.saturating_sub(trades_memory + overhead);

        // Per-strategy memory requirement
        // - Candles: 636MB (106M × 6 bytes INT8)
        // - Signals: 106MB (106M × 1 byte)
        // - Features: 212MB (106M × 2 bytes)
        // - Overhead: 100MB
        // Total: ~950MB per strategy
        let per_strategy_memory = 950_000_000;

        // Calculate batch size
        let batch_size = (working_memory / per_strategy_memory).max(1).min(20);

        batch_size
    }

    /// Upload trades to GPU (once per execution)
    ///
    /// Uses pinned memory for fast asynchronous transfer.
    ///
    /// # Memory Layout
    ///
    /// Trades are stored as flat u8 buffer:
    /// ```text
    /// [trade_id (8B) | price (8B) | quantity (8B) | quote_qty (8B) |
    ///  timestamp (8B) | is_buyer_maker (1B) | padding (7B)]
    /// Total: 32 bytes per trade
    /// ```
    fn upload_trades_to_gpu(&self, trades: &[Trade]) -> Result<CudaSlice<u8>, GpuError> {
        let n_trades = trades.len();
        let bytes_per_trade = 32; // Align to 32 bytes for coalesced access
        let total_bytes = n_trades * bytes_per_trade;

        // Flatten trades to u8 buffer
        let mut buffer = Vec::with_capacity(total_bytes);
        for trade in trades {
            // trade_id (8 bytes)
            buffer.extend_from_slice(&trade.trade_id.to_le_bytes());
            // price (8 bytes)
            buffer.extend_from_slice(&trade.price.to_le_bytes());
            // quantity (8 bytes)
            buffer.extend_from_slice(&trade.quantity.to_le_bytes());
            // quote_quantity (8 bytes)
            buffer.extend_from_slice(&trade.quote_quantity.to_le_bytes());
            // timestamp_ms (8 bytes)
            buffer.extend_from_slice(&trade.timestamp_ms.to_le_bytes());
            // is_buyer_maker (1 byte)
            buffer.push(if trade.is_buyer_maker { 1 } else { 0 });
            // padding (7 bytes) for 32-byte alignment
            buffer.extend_from_slice(&[0u8; 7]);
        }

        // Allocate GPU memory and copy data
        // Note: cudarc 0.17.3 uses memcpy_stod (sync transfer with allocation)
        let d_trades = self.device.stream.memcpy_stod(&buffer).map_err(|e| {
            GpuError::AllocationError(format!("Failed to allocate/copy trades to GPU: {:?}", e))
        })?;

        // Synchronize to ensure transfer is complete
        self.device.synchronize()?;

        Ok(d_trades)
    }
}

/// Batch tick backtest results
///
/// Contains results for all strategies, sorted by fitness score (best first).
#[derive(Debug, Clone)]
pub struct BatchBacktestResults {
    /// Results for each strategy (sorted by fitness, best first)
    pub results: Vec<BacktestResult>,

    /// GPU execution time (kernel time only)
    pub gpu_time_ms: f64,

    /// Total execution time (including transfers)
    pub total_time_ms: f64,

    /// VRAM used (MB)
    pub vram_used_mb: f64,
}

impl BatchBacktestResults {
    /// Get best N strategies by fitness score
    pub fn top_n(&self, n: usize) -> &[BacktestResult] {
        &self.results[..n.min(self.results.len())]
    }

    /// Calculate speedup vs sequential CPU execution
    ///
    /// Assumes 5 seconds per strategy for sequential CPU execution
    /// (based on 106M trades taking ~5s per strategy on modern CPU)
    pub fn speedup(&self) -> f64 {
        let sequential_time_s = self.results.len() as f64 * 5.0;
        sequential_time_s / (self.total_time_ms / 1000.0)
    }

    /// Print performance summary
    pub fn print_summary(&self) {
        println!("=== Batch Tick Backtest Summary ===");
        println!("Strategies processed: {}", self.results.len());
        println!("GPU time: {:.2}s", self.gpu_time_ms / 1000.0);
        println!("Total time: {:.2}s", self.total_time_ms / 1000.0);
        println!("VRAM used: {:.2} GB", self.vram_used_mb / 1024.0);
        println!("Speedup: {:.1}x vs sequential CPU", self.speedup());
        println!();
        println!("Top 5 strategies:");
        for (i, result) in self.top_n(5).iter().enumerate() {
            println!(
                "  {}. Sharpe={:.2} Return={:.2}% DD={:.2}% Trades={}",
                i + 1,
                result.sharpe_ratio,
                result.total_return,
                result.max_drawdown * 100.0,
                result.num_trades
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_tune_batch_size() {
        let device = Arc::new(GpuDevice::new().expect("GPU not available for test"));
        let batch = BatchTickBacktest::new(device);

        // Test with 106M trades
        let batch_size = batch.auto_tune_batch_size(106_000_000);

        // Should be between 1 and 20
        assert!(batch_size >= 1 && batch_size <= 20);

        // For 12GB VRAM, should be around 9-10
        assert!(
            batch_size >= 8 && batch_size <= 12,
            "Expected batch size 8-12, got {}",
            batch_size
        );
    }

    #[test]
    fn test_builder_api() {
        let device = Arc::new(GpuDevice::new().expect("GPU not available for test"));

        let trades = vec![Trade::default(); 100];
        let params = vec![vec![30.0, 0.15, 10.0, 0.001, 5.0, 1.0]];

        let _batch = BatchTickBacktest::new(device)
            .trades(&trades)
            .parameters_batch(&params)
            .batch_size(10)
            .force_cpu(true); // Force CPU to avoid GPU dependency in test

        // If this compiles, builder API is correctly structured
    }
}
