//! Backtesting execution engine with GPU/CPU auto-selection
//!
//! # Features
//!
//! - GPU-accelerated indicator calculation for large datasets
//! - CPU-only fallback for environments without CUDA
//! - Batch indicator processing for efficiency
//! - Position tracking and trade execution
//! - Performance metric calculation
//!
//! # Architecture
//!
//! ```text
//! OHLCV Data
//!   ↓
//! Indicator Calculation (GPU batch or CPU fallback)
//!   ↓
//! Bar-by-Bar Strategy Execution
//!   ↓
//! Trade Execution & Position Tracking
//!   ↓
//! Performance Metrics (Sharpe, Drawdown, Win Rate)
//! ```

use super::core::{
    BacktestResult, IndicatorConfig, OHLCVBar, Signal, Strategy, Trade, TradeDirection,
};
use super::metrics::{calculate_max_drawdown, calculate_sharpe_ratio, calculate_win_rate};
use ndarray::Array1;
use std::collections::HashMap;

#[cfg(feature = "gpu")]
use crate::gpu::{
    GpuError,
    batch::{BatchIndicatorParams, BatchIndicatorType, IndicatorResult},
    device::GpuDevice,
};

#[cfg(not(feature = "gpu"))]
use crate::cpu::sequential::GpuError;

/// Backtesting engine configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital for trading
    pub initial_capital: f64,

    /// Trading fee per trade (as a fraction, e.g., 0.001 = 0.1%)
    pub trading_fee: f64,

    /// Slippage per trade (as a fraction, e.g., 0.0005 = 0.05%)
    pub slippage: f64,

    /// Enable GPU acceleration (if available)
    pub use_gpu: bool,

    /// Force CPU-only mode (for testing and small datasets)
    pub force_cpu: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10_000.0,
            trading_fee: 0.001, // 0.1% per trade
            slippage: 0.0005,   // 0.05% slippage
            use_gpu: true,      // Auto-detect and use GPU if available
            force_cpu: false,   // Allow GPU by default
        }
    }
}

/// Backtesting execution engine
pub struct BacktestEngine {
    config: BacktestConfig,
    #[cfg(feature = "gpu")]
    gpu_device: Option<GpuDevice>,
}

impl BacktestEngine {
    /// Create new backtesting engine with default configuration
    pub fn new() -> Self {
        Self::with_config(BacktestConfig::default())
    }

    /// Create backtesting engine with custom configuration
    pub fn with_config(config: BacktestConfig) -> Self {
        #[cfg(feature = "gpu")]
        let gpu_device = if config.use_gpu && !config.force_cpu {
            GpuDevice::new().ok()
        } else {
            None
        };

        Self {
            config,
            #[cfg(feature = "gpu")]
            gpu_device,
        }
    }

    /// Get configuration reference
    pub fn config(&self) -> &BacktestConfig {
        &self.config
    }

    /// Run backtest on OHLCV data with given strategy
    ///
    /// # Arguments
    ///
    /// * `strategy` - Trading strategy implementation
    /// * `timestamps` - Unix timestamps for each bar
    /// * `open` - Open prices
    /// * `high` - High prices
    /// * `low` - Low prices
    /// * `close` - Close prices
    /// * `volume` - Trading volume
    ///
    /// # Returns
    ///
    /// BacktestResult with equity curve, trades, and performance metrics
    pub fn run(
        &self,
        strategy: &mut dyn Strategy,
        timestamps: &[i64],
        open: &Array1<f64>,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        volume: &Array1<f64>,
    ) -> Result<BacktestResult, GpuError> {
        let n = timestamps.len();

        // Validate inputs
        if n == 0 {
            return Err(GpuError::EmptyOhlcvData);
        }

        if open.len() != n
            || high.len() != n
            || low.len() != n
            || close.len() != n
            || volume.len() != n
        {
            return Err(GpuError::OhlcvLengthMismatch);
        }

        // Get required indicators from strategy
        let indicator_configs = strategy.indicators();

        // Calculate all indicators upfront (batch processing for efficiency)
        let indicator_values =
            self.calculate_indicators(&indicator_configs, high, low, close, open, volume)?;

        // Run strategy bar-by-bar
        let mut position = 0.0; // Current position size (0 = no position, 1 = full long, -1 = full short)
        let mut entry_price = 0.0;
        let mut entry_time = 0i64;
        let mut equity = strategy.initial_capital();
        let mut equity_curve = Vec::with_capacity(n);
        let mut trades = Vec::new();

        for i in 0..n {
            // Build indicator values for this bar
            // Investigation showed HashMap::clear() is 21.5% slower than allocating new
            // Reverted from pre-allocation optimization
            let mut bar_indicators = HashMap::new();
            for (key, values) in &indicator_values {
                bar_indicators.insert(key.clone(), values[i]);
            }

            // Create OHLCV bar
            let bar = OHLCVBar {
                timestamp: timestamps[i],
                open: open[i],
                high: high[i],
                low: low[i],
                close: close[i],
                volume: volume[i],
            };

            // Get trading signal from strategy
            let signal = strategy.on_data(&bar, &bar_indicators);

            // Execute trades based on signal
            match signal {
                Signal::Buy if position <= 0.0 => {
                    // Close short position if exists
                    if position < 0.0 {
                        let exit_price = close[i] * (1.0 + self.config.slippage);
                        let pnl = position * (entry_price - exit_price);
                        let pnl_percent = (entry_price - exit_price) / entry_price * 100.0;

                        trades.push(Trade {
                            entry_time,
                            exit_time: timestamps[i],
                            entry_price,
                            exit_price,
                            quantity: position.abs(),
                            direction: TradeDirection::Short,
                            pnl,
                            pnl_percent,
                        });

                        equity +=
                            pnl - (entry_price.abs() + exit_price.abs()) * self.config.trading_fee;
                    }

                    // Open long position
                    let position_size = strategy.position_size(equity, signal);
                    entry_price = close[i] * (1.0 + self.config.slippage);
                    entry_time = timestamps[i];
                    position = position_size;
                }
                Signal::Sell if position >= 0.0 => {
                    // Close long position if exists
                    if position > 0.0 {
                        let exit_price = close[i] * (1.0 - self.config.slippage);
                        let pnl = position * (exit_price - entry_price);
                        let pnl_percent = (exit_price - entry_price) / entry_price * 100.0;

                        trades.push(Trade {
                            entry_time,
                            exit_time: timestamps[i],
                            entry_price,
                            exit_price,
                            quantity: position,
                            direction: TradeDirection::Long,
                            pnl,
                            pnl_percent,
                        });

                        equity += pnl - (entry_price + exit_price) * self.config.trading_fee;
                        position = 0.0;
                    }
                }
                Signal::Short if position >= 0.0 => {
                    // Close long position if exists
                    if position > 0.0 {
                        let exit_price = close[i] * (1.0 - self.config.slippage);
                        let pnl = position * (exit_price - entry_price);
                        let pnl_percent = (exit_price - entry_price) / entry_price * 100.0;

                        trades.push(Trade {
                            entry_time,
                            exit_time: timestamps[i],
                            entry_price,
                            exit_price,
                            quantity: position,
                            direction: TradeDirection::Long,
                            pnl,
                            pnl_percent,
                        });

                        equity += pnl - (entry_price + exit_price) * self.config.trading_fee;
                    }

                    // Open short position
                    let position_size = strategy.position_size(equity, signal);
                    entry_price = close[i] * (1.0 - self.config.slippage);
                    entry_time = timestamps[i];
                    position = -position_size;
                }
                Signal::Cover if position <= 0.0 => {
                    // Close short position if exists
                    if position < 0.0 {
                        let exit_price = close[i] * (1.0 + self.config.slippage);
                        let pnl = position * (entry_price - exit_price);
                        let pnl_percent = (entry_price - exit_price) / entry_price * 100.0;

                        trades.push(Trade {
                            entry_time,
                            exit_time: timestamps[i],
                            entry_price,
                            exit_price,
                            quantity: position.abs(),
                            direction: TradeDirection::Short,
                            pnl,
                            pnl_percent,
                        });

                        equity +=
                            pnl - (entry_price.abs() + exit_price.abs()) * self.config.trading_fee;
                        position = 0.0;
                    }
                }
                Signal::Hold => {
                    // Do nothing, hold current position
                }
                _ => {
                    // Invalid signal for current position (e.g., Buy when already long)
                    // Just hold
                }
            }

            // Update equity curve (mark-to-market)
            let mut current_equity = equity;
            if position != 0.0 {
                // Add unrealized P&L
                if position > 0.0 {
                    current_equity += position * (close[i] - entry_price);
                } else {
                    current_equity += position * (entry_price - close[i]);
                }
            }
            equity_curve.push(current_equity);
        }

        // Close any remaining position at the end
        if position != 0.0 {
            let exit_price = if position > 0.0 {
                close[n - 1] * (1.0 - self.config.slippage)
            } else {
                close[n - 1] * (1.0 + self.config.slippage)
            };

            let pnl = if position > 0.0 {
                position * (exit_price - entry_price)
            } else {
                position * (entry_price - exit_price)
            };

            let pnl_percent = if position > 0.0 {
                (exit_price - entry_price) / entry_price * 100.0
            } else {
                (entry_price - exit_price) / entry_price * 100.0
            };

            trades.push(Trade {
                entry_time,
                exit_time: timestamps[n - 1],
                entry_price,
                exit_price,
                quantity: position.abs(),
                direction: if position > 0.0 {
                    TradeDirection::Long
                } else {
                    TradeDirection::Short
                },
                pnl,
                pnl_percent,
            });

            equity += pnl - (entry_price.abs() + exit_price.abs()) * self.config.trading_fee;
        }

        // Calculate performance metrics
        let final_equity = equity;
        let total_return =
            (final_equity - strategy.initial_capital()) / strategy.initial_capital() * 100.0;
        let sharpe_ratio = calculate_sharpe_ratio(&equity_curve);
        let max_drawdown = calculate_max_drawdown(&equity_curve);
        let win_rate = calculate_win_rate(&trades);
        let num_trades = trades.len();

        // Calculate profit factor
        let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f64 = trades
            .iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| t.pnl.abs())
            .sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        Ok(BacktestResult {
            parameters: HashMap::new(), // Will be filled by optimizer
            equity_curve,
            final_equity,
            total_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            num_trades,
            profit_factor,
            trades,
        })
    }

    /// Run backtest with early exit optimization for unpromising strategies
    ///
    /// # Arguments
    ///
    /// * `strategy` - Trading strategy implementation
    /// * `timestamps` - Unix timestamps for each bar
    /// * `open` - Open prices
    /// * `high` - High prices
    /// * `low` - Low prices
    /// * `close` - Close prices
    /// * `volume` - Trading volume
    /// * `min_sharpe_threshold` - Minimum acceptable Sharpe ratio (early exit if interim Sharpe < threshold * 0.5)
    ///
    /// # Returns
    ///
    /// - `Ok(Some(result))` - Strategy passed threshold, full backtest completed
    /// - `Ok(None)` - Strategy failed threshold, early exit triggered
    /// - `Err(e)` - Error during backtest execution
    ///
    /// # Performance
    ///
    /// - Checks Sharpe ratio every 10% of bars
    /// - Early exits if interim Sharpe < 50% of threshold
    /// - Saves ~70% computation for ~30% of unpromising strategies
    /// - Expected speedup: ~21% for parameter sweeps (0.3 * 0.7 = 0.21)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Exit early if Sharpe ratio is below 0.5 at checkpoints
    /// let result = engine.run_with_early_exit(
    ///     &mut strategy,
    ///     &timestamps,
    ///     &open, &high, &low, &close, &volume,
    ///     Some(1.0),  // Require minimum Sharpe of 1.0
    /// )?;
    ///
    /// if let Some(result) = result {
    ///     println!("Strategy passed: Sharpe = {:.2}", result.sharpe_ratio);
    /// } else {
    ///     println!("Strategy failed early exit");
    /// }
    /// ```
    pub fn run_with_early_exit(
        &self,
        strategy: &mut dyn Strategy,
        timestamps: &[i64],
        open: &Array1<f64>,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        volume: &Array1<f64>,
        min_sharpe_threshold: Option<f64>,
    ) -> Result<Option<BacktestResult>, GpuError> {
        let n = timestamps.len();

        // Validate inputs
        if n == 0 {
            return Err(GpuError::EmptyOhlcvData);
        }

        if open.len() != n
            || high.len() != n
            || low.len() != n
            || close.len() != n
            || volume.len() != n
        {
            return Err(GpuError::OhlcvLengthMismatch);
        }

        // If no threshold, just run normal backtest
        if min_sharpe_threshold.is_none() {
            return self
                .run(strategy, timestamps, open, high, low, close, volume)
                .map(Some);
        }

        let threshold = min_sharpe_threshold.unwrap();
        let check_interval = n / 10; // Check every 10% of bars

        // Get required indicators from strategy
        let indicator_configs = strategy.indicators();

        // Calculate all indicators upfront (batch processing for efficiency)
        let indicator_values =
            self.calculate_indicators(&indicator_configs, high, low, close, open, volume)?;

        // Run strategy bar-by-bar with early exit checks
        let mut position = 0.0;
        let mut entry_price = 0.0;
        let mut entry_time = 0i64;
        let mut equity = strategy.initial_capital();
        let mut equity_curve = Vec::with_capacity(n);
        let mut trades = Vec::new();

        for i in 0..n {
            // Early exit check every 10% of bars (after minimum 10% progress)
            if i > check_interval && i % check_interval == 0 {
                let interim_sharpe = calculate_sharpe_ratio(&equity_curve);

                // Exit if interim Sharpe is less than 50% of threshold
                // This is aggressive pruning - strategies must show promise early
                if interim_sharpe < threshold * 0.5 {
                    return Ok(None); // Early exit - unpromising strategy
                }
            }

            // Build indicator values for this bar
            let mut bar_indicators = HashMap::new();
            for (key, values) in &indicator_values {
                bar_indicators.insert(key.clone(), values[i]);
            }

            // Create OHLCV bar
            let bar = OHLCVBar {
                timestamp: timestamps[i],
                open: open[i],
                high: high[i],
                low: low[i],
                close: close[i],
                volume: volume[i],
            };

            // Get trading signal from strategy
            let signal = strategy.on_data(&bar, &bar_indicators);

            // Execute trades based on signal (same logic as run())
            match signal {
                Signal::Buy if position <= 0.0 => {
                    // Close short position if exists
                    if position < 0.0 {
                        let exit_price = close[i] * (1.0 + self.config.slippage);
                        let pnl = position * (entry_price - exit_price);
                        let pnl_percent = (entry_price - exit_price) / entry_price * 100.0;

                        trades.push(Trade {
                            entry_time,
                            exit_time: timestamps[i],
                            entry_price,
                            exit_price,
                            quantity: position.abs(),
                            direction: TradeDirection::Short,
                            pnl,
                            pnl_percent,
                        });

                        equity +=
                            pnl - (entry_price.abs() + exit_price.abs()) * self.config.trading_fee;
                    }

                    // Open long position
                    let position_size = strategy.position_size(equity, signal);
                    entry_price = close[i] * (1.0 + self.config.slippage);
                    entry_time = timestamps[i];
                    position = position_size;
                }
                Signal::Sell if position >= 0.0 => {
                    // Close long position if exists
                    if position > 0.0 {
                        let exit_price = close[i] * (1.0 - self.config.slippage);
                        let pnl = position * (exit_price - entry_price);
                        let pnl_percent = (exit_price - entry_price) / entry_price * 100.0;

                        trades.push(Trade {
                            entry_time,
                            exit_time: timestamps[i],
                            entry_price,
                            exit_price,
                            quantity: position,
                            direction: TradeDirection::Long,
                            pnl,
                            pnl_percent,
                        });

                        equity += pnl - (entry_price + exit_price) * self.config.trading_fee;
                        position = 0.0;
                    }
                }
                Signal::Short if position >= 0.0 => {
                    // Close long position if exists
                    if position > 0.0 {
                        let exit_price = close[i] * (1.0 - self.config.slippage);
                        let pnl = position * (exit_price - entry_price);
                        let pnl_percent = (exit_price - entry_price) / entry_price * 100.0;

                        trades.push(Trade {
                            entry_time,
                            exit_time: timestamps[i],
                            entry_price,
                            exit_price,
                            quantity: position,
                            direction: TradeDirection::Long,
                            pnl,
                            pnl_percent,
                        });

                        equity += pnl - (entry_price + exit_price) * self.config.trading_fee;
                    }

                    // Open short position
                    let position_size = strategy.position_size(equity, signal);
                    entry_price = close[i] * (1.0 - self.config.slippage);
                    entry_time = timestamps[i];
                    position = -position_size;
                }
                Signal::Cover if position <= 0.0 => {
                    // Close short position if exists
                    if position < 0.0 {
                        let exit_price = close[i] * (1.0 + self.config.slippage);
                        let pnl = position * (entry_price - exit_price);
                        let pnl_percent = (entry_price - exit_price) / entry_price * 100.0;

                        trades.push(Trade {
                            entry_time,
                            exit_time: timestamps[i],
                            entry_price,
                            exit_price,
                            quantity: position.abs(),
                            direction: TradeDirection::Short,
                            pnl,
                            pnl_percent,
                        });

                        equity +=
                            pnl - (entry_price.abs() + exit_price.abs()) * self.config.trading_fee;
                        position = 0.0;
                    }
                }
                Signal::Hold => {
                    // Do nothing, hold current position
                }
                _ => {
                    // Invalid signal for current position
                    // Just hold
                }
            }

            // Update equity curve (mark-to-market)
            let mut current_equity = equity;
            if position != 0.0 {
                // Add unrealized P&L
                if position > 0.0 {
                    current_equity += position * (close[i] - entry_price);
                } else {
                    current_equity += position * (entry_price - close[i]);
                }
            }
            equity_curve.push(current_equity);
        }

        // Close any remaining position at the end
        if position != 0.0 {
            let exit_price = if position > 0.0 {
                close[n - 1] * (1.0 - self.config.slippage)
            } else {
                close[n - 1] * (1.0 + self.config.slippage)
            };

            let pnl = if position > 0.0 {
                position * (exit_price - entry_price)
            } else {
                position * (entry_price - exit_price)
            };

            let pnl_percent = if position > 0.0 {
                (exit_price - entry_price) / entry_price * 100.0
            } else {
                (entry_price - exit_price) / entry_price * 100.0
            };

            trades.push(Trade {
                entry_time,
                exit_time: timestamps[n - 1],
                entry_price,
                exit_price,
                quantity: position.abs(),
                direction: if position > 0.0 {
                    TradeDirection::Long
                } else {
                    TradeDirection::Short
                },
                pnl,
                pnl_percent,
            });

            equity += pnl - (entry_price.abs() + exit_price.abs()) * self.config.trading_fee;
        }

        // Calculate performance metrics
        let final_equity = equity;
        let total_return =
            (final_equity - strategy.initial_capital()) / strategy.initial_capital() * 100.0;
        let sharpe_ratio = calculate_sharpe_ratio(&equity_curve);
        let max_drawdown = calculate_max_drawdown(&equity_curve);
        let win_rate = calculate_win_rate(&trades);
        let num_trades = trades.len();

        // Calculate profit factor
        let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f64 = trades
            .iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| t.pnl.abs())
            .sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        Ok(Some(BacktestResult {
            parameters: HashMap::new(),
            equity_curve,
            final_equity,
            total_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            num_trades,
            profit_factor,
            trades,
        }))
    }

    /// Run parameter sweep using 3D GPU kernels for optimization
    ///
    /// # Arguments
    ///
    /// * `strategy` - Trading strategy (parameters will be overridden from grid)
    /// * `timestamps` - Unix timestamps for each bar
    /// * `open` - Open prices
    /// * `high` - High prices
    /// * `low` - Low prices
    /// * `close` - Close prices
    /// * `volume` - Trading volume
    /// * `grid` - Parameter grid to sweep
    ///
    /// # Returns
    ///
    /// Vector of BacktestResult sorted by fitness score (best first)
    ///
    /// # Performance
    ///
    /// GPU mode: +40-60% speedup over sequential testing (N_combinations >= 20)
    /// CPU mode: Sequential testing (no parallelization)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use kimsfinance_core::backtest::{BacktestEngine, ParameterGrid, ParameterRange};
    ///
    /// let mut grid = ParameterGrid::new();
    /// grid.add_range("rsi_period", ParameterRange::Int { min: 10, max: 30, step: 2 });
    /// grid.add_range("buy_threshold", ParameterRange::Float { min: 20.0, max: 40.0, step: 5.0 });
    ///
    /// let engine = BacktestEngine::new();
    /// let results = engine.run_sweep(&mut strategy, &timestamps, &open, &high, &low, &close, &volume, &grid)?;
    ///
    /// println!("Best parameters: {:?}", results[0].parameters);
    /// println!("Best Sharpe: {:.2}", results[0].sharpe_ratio);
    /// ```
    #[cfg(feature = "gpu")]
    pub fn run_sweep(
        &self,
        strategy: &mut dyn Strategy,
        timestamps: &[i64],
        open: &Array1<f64>,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        volume: &Array1<f64>,
        grid: &super::core::ParameterGrid,
    ) -> Result<Vec<BacktestResult>, GpuError> {
        use super::sweep::run_parameter_sweep_gpu;

        run_parameter_sweep_gpu(
            self, strategy, timestamps, open, high, low, close, volume, grid,
        )
    }

    /// Run parameter sweep (CPU fallback)
    #[cfg(not(feature = "gpu"))]
    pub fn run_sweep(
        &self,
        strategy: &mut dyn Strategy,
        timestamps: &[i64],
        open: &Array1<f64>,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        volume: &Array1<f64>,
        grid: &super::core::ParameterGrid,
    ) -> Result<Vec<BacktestResult>, GpuError> {
        use super::sweep::run_parameter_sweep_cpu;

        run_parameter_sweep_cpu(
            self, strategy, timestamps, open, high, low, close, volume, grid,
        )
    }

    /// Calculate all required indicators (GPU batch or CPU fallback)
    fn calculate_indicators(
        &self,
        configs: &[IndicatorConfig],
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        _open: &Array1<f64>,
        _volume: &Array1<f64>,
    ) -> Result<HashMap<String, Vec<f64>>, GpuError> {
        #[cfg(feature = "gpu")]
        if let Some(device) = &self.gpu_device {
            // Use GPU batch processing
            return self.calculate_indicators_gpu(device, configs, high, low, close);
        }

        // CPU fallback
        self.calculate_indicators_cpu(configs, high, low, close)
    }

    #[cfg(feature = "gpu")]
    /// Calculate indicators using GPU batch processing
    fn calculate_indicators_gpu(
        &self,
        device: &GpuDevice,
        configs: &[IndicatorConfig],
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
    ) -> Result<HashMap<String, Vec<f64>>, GpuError> {
        use crate::gpu::batch::calculate_indicators_batch_gpu;

        // Convert IndicatorConfig to BatchIndicatorType + params
        let mut indicators = Vec::new();
        let mut params = HashMap::new();

        for config in configs {
            match config {
                IndicatorConfig::RSI { period } => {
                    indicators.push(BatchIndicatorType::RSI);
                    params.insert(
                        BatchIndicatorType::RSI,
                        BatchIndicatorParams {
                            period: Some(*period),
                            ..Default::default()
                        },
                    );
                }
                IndicatorConfig::ATR { period } => {
                    indicators.push(BatchIndicatorType::ATR);
                    params.insert(
                        BatchIndicatorType::ATR,
                        BatchIndicatorParams {
                            period: Some(*period),
                            ..Default::default()
                        },
                    );
                }
                IndicatorConfig::ROC { period } => {
                    indicators.push(BatchIndicatorType::ROC);
                    params.insert(
                        BatchIndicatorType::ROC,
                        BatchIndicatorParams {
                            period: Some(*period),
                            ..Default::default()
                        },
                    );
                }
                IndicatorConfig::CCI { period } => {
                    indicators.push(BatchIndicatorType::CCI);
                    params.insert(
                        BatchIndicatorType::CCI,
                        BatchIndicatorParams {
                            period: Some(*period),
                            ..Default::default()
                        },
                    );
                }
                IndicatorConfig::WilliamsR { period } => {
                    indicators.push(BatchIndicatorType::WilliamsR);
                    params.insert(
                        BatchIndicatorType::WilliamsR,
                        BatchIndicatorParams {
                            period: Some(*period),
                            ..Default::default()
                        },
                    );
                }
                IndicatorConfig::Stochastic { k_period, d_period } => {
                    indicators.push(BatchIndicatorType::Stochastic);
                    params.insert(
                        BatchIndicatorType::Stochastic,
                        BatchIndicatorParams {
                            k_period: Some(*k_period),
                            d_period: Some(*d_period),
                            ..Default::default()
                        },
                    );
                }
                IndicatorConfig::BollingerBands { period, std_dev } => {
                    indicators.push(BatchIndicatorType::BollingerBands);
                    params.insert(
                        BatchIndicatorType::BollingerBands,
                        BatchIndicatorParams {
                            period: Some(*period),
                            num_std: Some(*std_dev),
                            ..Default::default()
                        },
                    );
                }
                _ => {
                    // Unsupported indicator for GPU batch - fall back to CPU
                    return self.calculate_indicators_cpu(configs, high, low, close);
                }
            }
        }

        // Call GPU batch processing
        let results = calculate_indicators_batch_gpu(
            device,
            high,
            low,
            close,
            None,
            None,
            &indicators,
            &params,
        )?;

        // Convert results to HashMap<String, Vec<f64>>
        let mut output = HashMap::new();

        for config in configs {
            let key = config.key();
            match config {
                IndicatorConfig::RSI { .. } => {
                    if let Some(IndicatorResult::Single(values)) =
                        results.get(&BatchIndicatorType::RSI)
                    {
                        output.insert(key, values.to_vec());
                    }
                }
                IndicatorConfig::ATR { .. } => {
                    if let Some(IndicatorResult::Single(values)) =
                        results.get(&BatchIndicatorType::ATR)
                    {
                        output.insert(key, values.to_vec());
                    }
                }
                IndicatorConfig::ROC { .. } => {
                    if let Some(IndicatorResult::Single(values)) =
                        results.get(&BatchIndicatorType::ROC)
                    {
                        output.insert(key, values.to_vec());
                    }
                }
                IndicatorConfig::CCI { .. } => {
                    if let Some(IndicatorResult::Single(values)) =
                        results.get(&BatchIndicatorType::CCI)
                    {
                        output.insert(key, values.to_vec());
                    }
                }
                IndicatorConfig::WilliamsR { .. } => {
                    if let Some(IndicatorResult::Single(values)) =
                        results.get(&BatchIndicatorType::WilliamsR)
                    {
                        output.insert(key, values.to_vec());
                    }
                }
                IndicatorConfig::Stochastic { .. } => {
                    if let Some(IndicatorResult::Double(k_values, d_values)) =
                        results.get(&BatchIndicatorType::Stochastic)
                    {
                        output.insert(format!("{}_k", key), k_values.to_vec());
                        output.insert(format!("{}_d", key), d_values.to_vec());
                    }
                }
                IndicatorConfig::BollingerBands { .. } => {
                    if let Some(IndicatorResult::Triple(upper, middle, lower)) =
                        results.get(&BatchIndicatorType::BollingerBands)
                    {
                        output.insert(format!("{}_upper", key), upper.to_vec());
                        output.insert(format!("{}_middle", key), middle.to_vec());
                        output.insert(format!("{}_lower", key), lower.to_vec());
                    }
                }
                _ => {}
            }
        }

        Ok(output)
    }

    /// Calculate indicators using CPU implementations (fallback)
    pub(crate) fn calculate_indicators_cpu(
        &self,
        configs: &[IndicatorConfig],
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
    ) -> Result<HashMap<String, Vec<f64>>, GpuError> {
        use crate::indicators::core::{Indicator, MultiOutputIndicator};
        use crate::indicators::momentum::{CCI, MACD, ROC, RSI, Stochastic, WilliamsR};
        use crate::indicators::moving_averages::{EMA, SMA};
        use crate::indicators::volatility::{ATR, BollingerBands};

        let mut output = HashMap::new();

        for config in configs {
            let key = config.key();
            match config {
                // Momentum Indicators (single-output)
                IndicatorConfig::RSI { period } => {
                    let indicator =
                        RSI::new(*period).map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    let values = indicator
                        .calculate(close.view())
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    output.insert(key, values.to_vec());
                }
                IndicatorConfig::ROC { period } => {
                    let indicator =
                        ROC::new(*period).map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    let values = indicator
                        .calculate(close.view())
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    output.insert(key, values.to_vec());
                }
                IndicatorConfig::CCI { period } => {
                    let indicator =
                        CCI::new(*period).map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    let values = indicator
                        .calculate_hlc(high.view(), low.view(), close.view())
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    output.insert(key, values.to_vec());
                }
                IndicatorConfig::WilliamsR { period } => {
                    let indicator = WilliamsR::new(*period)
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    let values = indicator
                        .calculate_hlc(high.view(), low.view(), close.view())
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    output.insert(key, values.to_vec());
                }

                // Volatility Indicators
                IndicatorConfig::ATR { period } => {
                    let indicator =
                        ATR::new(*period).map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    let values = indicator
                        .calculate_hlc(high.view(), low.view(), close.view())
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    output.insert(key, values.to_vec());
                }

                // Moving Averages
                IndicatorConfig::SMA { period } => {
                    let indicator =
                        SMA::new(*period).map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    let values = indicator
                        .calculate(close.view())
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    output.insert(key, values.to_vec());
                }
                IndicatorConfig::EMA { period } => {
                    let indicator =
                        EMA::new(*period).map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    let values = indicator
                        .calculate(close.view())
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    output.insert(key, values.to_vec());
                }

                // Multi-output Indicators
                IndicatorConfig::MACD { fast, slow, signal } => {
                    let indicator = MACD::new(*fast, *slow, *signal)
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    let result = indicator
                        .calculate_multi(close.view())
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;

                    output.insert(format!("{}_macd", key), result.primary.to_vec());
                    output.insert(format!("{}_signal", key), result.secondary[0].to_vec());
                    output.insert(format!("{}_histogram", key), result.secondary[1].to_vec());
                }
                IndicatorConfig::Stochastic { k_period, d_period } => {
                    let indicator = Stochastic::new(*k_period, *d_period)
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    let result = indicator
                        .calculate_hlc(high.view(), low.view(), close.view())
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;

                    output.insert(format!("{}_k", key), result.primary.to_vec());
                    output.insert(format!("{}_d", key), result.secondary[0].to_vec());
                }
                IndicatorConfig::BollingerBands { period, std_dev } => {
                    let indicator = BollingerBands::new(*period, *std_dev)
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;
                    let result = indicator
                        .calculate_multi(close.view())
                        .map_err(|e| GpuError::ComputationError(e.to_string()))?;

                    output.insert(format!("{}_middle", key), result.primary.to_vec());
                    output.insert(format!("{}_upper", key), result.secondary[0].to_vec());
                    output.insert(format!("{}_lower", key), result.secondary[1].to_vec());
                }
            }
        }

        Ok(output)
    }
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Add comprehensive tests once metrics module is implemented
}
