//! Python bindings for tick-level (event-driven) back testing
//!
//! This module exposes tick-level backtesting to Python via PyO3.
//! It accepts pre-computed signals from Python and runs event-driven backtests.
//!
//! # Example Usage (Python)
//!
//! ```python
//! import kimsfinance_core
//! import numpy as np
//!
//! # Configure backtest
//! config = kimsfinance_core.TickBacktestConfig(
//!     initial_capital=10_000.0,
//!     trading_fee=0.001,      # 0.1%
//!     slippage=0.0005,        # 0.05%
//!     execution_latency_ms=10 # 10ms execution delay
//! )
//!
//! # Create backtest engine
//! engine = kimsfinance_core.TickBacktestEngine(config)
//!
//! # Prepare trade data (NumPy arrays)
//! timestamps = np.array([...], dtype=np.int64)  # Milliseconds
//! prices = np.array([...], dtype=np.float32)
//! volumes = np.array([...], dtype=np.float32)
//! is_buyer_maker = np.array([...], dtype=np.bool_)
//!
//! # Prepare signals (Python computes these)
//! # Signal values: 0=Hold, 1=Buy, 2=Sell
//! signals = np.array([0, 0, 1, 0, 0, 2, 0], dtype=np.int8)
//!
//! # Run backtest
//! result = engine.run(timestamps, prices, volumes, is_buyer_maker, signals, timeframe_ms=300_000)
//!
//! # Access results
//! print(f"Total Return: {result.total_return:.2f}%")
//! print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
//! print(f"Max Drawdown: {result.max_drawdown:.2f}%")
//! print(f"Win Rate: {result.win_rate:.2f}%")
//! print(f"Num Trades: {result.num_trades}")
//! ```

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Tick backtest configuration
#[pyclass(name = "TickBacktestConfig")]
#[derive(Clone, Debug)]
pub struct PyTickBacktestConfig {
    pub initial_capital: f64,
    pub trading_fee: f64,
    pub slippage: f64,
    pub execution_latency_ms: i64,
}

#[pymethods]
impl PyTickBacktestConfig {
    #[new]
    #[pyo3(signature = (initial_capital=10_000.0, trading_fee=0.001, slippage=0.0005, execution_latency_ms=10))]
    fn new(
        initial_capital: f64,
        trading_fee: f64,
        slippage: f64,
        execution_latency_ms: i64,
    ) -> Self {
        Self {
            initial_capital,
            trading_fee,
            slippage,
            execution_latency_ms,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TickBacktestConfig(initial_capital={:.2}, trading_fee={:.4}, slippage={:.4}, execution_latency_ms={})",
            self.initial_capital, self.trading_fee, self.slippage, self.execution_latency_ms
        )
    }
}

/// Tick backtest result
#[pyclass(name = "TickBacktestResult")]
pub struct PyTickBacktestResult {
    #[pyo3(get)]
    pub total_return: f64,
    #[pyo3(get)]
    pub sharpe_ratio: f64,
    #[pyo3(get)]
    pub max_drawdown: f64,
    #[pyo3(get)]
    pub win_rate: f64,
    #[pyo3(get)]
    pub profit_factor: f64,
    #[pyo3(get)]
    pub num_trades: usize,
    #[pyo3(get)]
    pub final_equity: f64,

    // Store equity curve and trade history
    equity_curve: Vec<f64>,
    trade_pnls: Vec<f64>,
}

#[pymethods]
impl PyTickBacktestResult {
    /// Get equity curve as NumPy array
    fn equity_curve<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        PyArray1::from_slice(py, &self.equity_curve).into()
    }

    /// Get trade P&Ls as NumPy array
    fn trade_pnls<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        PyArray1::from_slice(py, &self.trade_pnls).into()
    }

    /// Convert to dictionary
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("total_return", self.total_return)?;
        dict.set_item("sharpe_ratio", self.sharpe_ratio)?;
        dict.set_item("max_drawdown", self.max_drawdown)?;
        dict.set_item("win_rate", self.win_rate)?;
        dict.set_item("profit_factor", self.profit_factor)?;
        dict.set_item("num_trades", self.num_trades)?;
        dict.set_item("final_equity", self.final_equity)?;
        dict.set_item("equity_curve", self.equity_curve(py))?;
        dict.set_item("trade_pnls", self.trade_pnls(py))?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "TickBacktestResult(return={:.2}%, sharpe={:.2}, drawdown={:.2}%, trades={})",
            self.total_return, self.sharpe_ratio, self.max_drawdown, self.num_trades
        )
    }
}

/// Tick backtest engine (CPU-based, event-driven)
#[pyclass(name = "TickBacktestEngine")]
pub struct PyTickBacktestEngine {
    config: PyTickBacktestConfig,
}

#[pymethods]
impl PyTickBacktestEngine {
    #[new]
    fn new(config: PyTickBacktestConfig) -> Self {
        Self { config }
    }

    /// Run tick-level backtest with pre-computed signals
    ///
    /// # Arguments
    ///
    /// * `timestamps` - Trade timestamps (milliseconds, int64)
    /// * `prices` - Trade prices (float32)
    /// * `volumes` - Trade volumes (float32)
    /// * `is_buyer_maker` - True if buyer is maker (bool)
    /// * `signals` - Trading signals (0=Hold, 1=Buy, 2=Sell, int8)
    /// * `timeframe_ms` - Timeframe in milliseconds (e.g., 300_000 for 5min)
    ///
    /// # Returns
    ///
    /// TickBacktestResult with performance metrics
    ///
    /// # Example
    ///
    /// ```python
    /// timestamps = np.array([1000, 2000, 3000], dtype=np.int64)
    /// prices = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    /// volumes = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    /// is_buyer_maker = np.array([True, False, True], dtype=np.bool_)
    /// signals = np.array([0, 1, 2], dtype=np.int8)  # Hold, Buy, Sell
    ///
    /// result = engine.run(timestamps, prices, volumes, is_buyer_maker, signals, 300_000)
    /// print(f"Return: {result.total_return:.2f}%")
    /// ```
    #[pyo3(signature = (timestamps, prices, volumes, is_buyer_maker, signals, timeframe_ms))]
    fn run(
        &self,
        timestamps: PyReadonlyArray1<i64>,
        prices: PyReadonlyArray1<f32>,
        volumes: PyReadonlyArray1<f32>,
        is_buyer_maker: PyReadonlyArray1<bool>,
        signals: PyReadonlyArray1<i8>,
        timeframe_ms: i64,
    ) -> PyResult<PyTickBacktestResult> {
        // Extract arrays
        let timestamps = timestamps.as_slice()?;
        let prices = prices.as_slice()?;
        let volumes = volumes.as_slice()?;
        let is_buyer_maker = is_buyer_maker.as_slice()?;
        let signals = signals.as_slice()?;

        // Validate lengths
        let n = timestamps.len();
        if prices.len() != n || volumes.len() != n || is_buyer_maker.len() != n {
            return Err(PyRuntimeError::new_err(
                "All trade arrays must have the same length",
            ));
        }

        if signals.len() != n {
            return Err(PyRuntimeError::new_err(
                "Signals array must match trade data length",
            ));
        }

        // Run backtest
        let result = run_signal_based_backtest(
            timestamps,
            prices,
            volumes,
            is_buyer_maker,
            signals,
            timeframe_ms,
            &self.config,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Backtest failed: {}", e)))?;

        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!("TickBacktestEngine({:?})", self.config)
    }
}

/// Run signal-based backtest (internal implementation)
fn run_signal_based_backtest(
    timestamps: &[i64],
    prices: &[f32],
    _volumes: &[f32],
    _is_buyer_maker: &[bool],
    signals: &[i8],
    _timeframe_ms: i64,
    config: &PyTickBacktestConfig,
) -> Result<PyTickBacktestResult, String> {
    if timestamps.is_empty() {
        return Err("No trade data provided".to_string());
    }

    // Initialize state
    let mut equity = config.initial_capital;
    let mut cash = config.initial_capital;
    let mut position_size = 0.0f64;
    let mut entry_price = 0.0f64;
    let mut in_position = false;

    let mut equity_curve = Vec::with_capacity(timestamps.len() / 100);
    let mut trade_pnls = Vec::new();
    let mut total_pnl = 0.0;
    let mut winning_trades = 0;
    let mut losing_trades = 0;
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;

    // Track pending orders (execution latency simulation)
    let mut pending_orders: Vec<(i8, i64)> = Vec::new(); // (signal, execution_time)

    // Process each trade (event-driven)
    for (idx, &timestamp) in timestamps.iter().enumerate() {
        let price = prices[idx] as f64;

        // Execute pending orders that are ready
        pending_orders.retain(|(signal, execution_time)| {
            if timestamp >= *execution_time {
                // Execute order
                match signal {
                    1 => {
                        // Buy signal
                        if !in_position {
                            // Enter long position
                            position_size = cash / price;
                            entry_price = price;
                            cash = 0.0;
                            equity = position_size * price;
                            in_position = true;
                        }
                    }
                    2 => {
                        // Sell signal
                        if in_position {
                            // Exit long position
                            let exit_price = price * (1.0 - config.trading_fee - config.slippage);
                            cash = position_size * exit_price;
                            let pnl = cash - config.initial_capital;
                            trade_pnls.push(pnl);
                            total_pnl += pnl;

                            if pnl > 0.0 {
                                winning_trades += 1;
                                gross_profit += pnl;
                            } else {
                                losing_trades += 1;
                                gross_loss += pnl.abs();
                            }

                            equity = cash;
                            position_size = 0.0;
                            in_position = false;
                        }
                    }
                    _ => {} // Hold
                }
                false // Remove from pending
            } else {
                true // Keep in pending
            }
        });

        // Check for new signal
        let signal = signals[idx];
        if signal != 0 {
            // Add to pending orders with execution latency
            let execution_time = timestamp + config.execution_latency_ms;
            pending_orders.push((signal, execution_time));
        }

        // Update equity
        if in_position {
            equity = position_size * price;
        } else {
            equity = cash;
        }

        // Sample equity curve (every 100 ticks)
        if idx % 100 == 0 {
            equity_curve.push(equity);
        }
    }

    // Close any open position at the end
    if in_position {
        let final_price = prices[prices.len() - 1] as f64;
        let exit_price = final_price * (1.0 - config.trading_fee - config.slippage);
        cash = position_size * exit_price;
        let pnl = cash - config.initial_capital;
        trade_pnls.push(pnl);
        // total_pnl += pnl; // Unused assignment

        if pnl > 0.0 {
            winning_trades += 1;
            gross_profit += pnl;
        } else {
            // losing_trades += 1; // Unused assignment
            gross_loss += pnl.abs();
        }

        equity = cash;
    }

    // Add final equity to curve
    equity_curve.push(equity);

    // Calculate metrics
    let total_return = ((equity / config.initial_capital) - 1.0) * 100.0;
    let num_trades = trade_pnls.len();
    let win_rate = if num_trades > 0 {
        (winning_trades as f64 / num_trades as f64) * 100.0
    } else {
        0.0
    };
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        0.0
    };

    // Calculate Sharpe ratio (simplified - assumes daily returns)
    let sharpe_ratio = if !trade_pnls.is_empty() {
        let mean_pnl = trade_pnls.iter().sum::<f64>() / trade_pnls.len() as f64;
        let variance = trade_pnls
            .iter()
            .map(|&pnl| (pnl - mean_pnl).powi(2))
            .sum::<f64>()
            / trade_pnls.len() as f64;
        let std_dev = variance.sqrt();
        if std_dev > 0.0 {
            mean_pnl / std_dev * (252.0f64).sqrt() // Annualized
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Calculate max drawdown
    let mut max_equity = config.initial_capital;
    let mut max_drawdown = 0.0;
    for &eq in &equity_curve {
        if eq > max_equity {
            max_equity = eq;
        }
        let drawdown = ((max_equity - eq) / max_equity) * 100.0;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    Ok(PyTickBacktestResult {
        total_return,
        sharpe_ratio,
        max_drawdown,
        win_rate,
        profit_factor,
        num_trades,
        final_equity: equity,
        equity_curve,
        trade_pnls,
    })
}

// No need for register function - classes are exported directly in lib.rs
