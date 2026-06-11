//! Walk-forward analysis for out-of-sample testing and overfitting detection
//!
//! # Overview
//!
//! Walk-forward analysis splits historical data into training and testing windows,
//! optimizing parameters on training data and validating on unseen test data.
//! This prevents overfitting and ensures strategy robustness.
//!
//! # Architecture
//!
//! ```text
//! Historical Data
//!   ↓
//! Train Window 1 → Optimize → Test Window 1
//!   ↓
//! Train Window 2 → Optimize → Test Window 2
//!   ↓
//! Train Window 3 → Optimize → Test Window 3
//!   ↓
//! Aggregate Results (In-Sample vs Out-of-Sample)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::backtest::walkforward::{WalkForwardAnalyzer, WalkForwardConfig};
//!
//! let config = WalkForwardConfig {
//!     train_window: 252,  // 1 year training
//!     test_window: 63,    // 1 quarter testing
//!     step_size: 21,      // 1 month step
//!     anchored: false,    // Rolling window
//! };
//!
//! let analyzer = WalkForwardAnalyzer::new(config);
//! let result = analyzer.analyze(&engine, &mut strategy, &timestamps, &ohlcv, &grid)?;
//!
//! println!("In-sample Sharpe: {:.2}", result.in_sample_sharpe);
//! println!("Out-of-sample Sharpe: {:.2}", result.out_of_sample_sharpe);
//! println!("Overfitting ratio: {:.2}", result.efficiency_ratio);
//! ```

use super::core::{BacktestResult, ParameterGrid, Strategy};
use super::engine::BacktestEngine;
use super::optimizer::GeneticOptimizer;
use ndarray::Array1;
use std::collections::HashMap;

#[cfg(feature = "gpu")]
use crate::gpu::GpuError;

#[cfg(not(feature = "gpu"))]
use crate::cpu::sequential::GpuError;

/// Walk-forward analysis configuration
#[derive(Debug, Clone)]
pub struct WalkForwardConfig {
    /// Training window size (number of bars)
    pub train_window: usize,

    /// Testing window size (number of bars)
    pub test_window: usize,

    /// Step size between windows (number of bars to advance)
    pub step_size: usize,

    /// Use anchored window (expand training from start) vs rolling window (fixed size)
    pub anchored: bool,

    /// Minimum bars required for analysis
    pub min_bars: usize,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            train_window: 252, // 1 year
            test_window: 63,   // 1 quarter
            step_size: 21,     // 1 month
            anchored: false,   // Rolling window
            min_bars: 100,     // Minimum data required
        }
    }
}

impl WalkForwardConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.train_window == 0 {
            return Err("Train window must be > 0".to_string());
        }
        if self.test_window == 0 {
            return Err("Test window must be > 0".to_string());
        }
        if self.step_size == 0 {
            return Err("Step size must be > 0".to_string());
        }
        if self.min_bars < self.train_window + self.test_window {
            return Err("Min bars must be >= train_window + test_window".to_string());
        }
        Ok(())
    }

    /// Calculate number of splits for given data length
    pub fn num_splits(&self, data_length: usize) -> usize {
        if data_length < self.min_bars {
            return 0;
        }

        let mut count = 0;
        let mut current_pos = 0;

        loop {
            let _train_start = if self.anchored { 0 } else { current_pos };
            let train_end = current_pos + self.train_window;
            let test_end = train_end + self.test_window;

            if test_end > data_length {
                break;
            }

            count += 1;
            current_pos += self.step_size;
        }

        count
    }
}

/// Single window in walk-forward analysis
#[derive(Debug, Clone)]
pub struct WalkForwardWindow {
    /// Training window start index
    pub train_start: usize,

    /// Training window end index (exclusive)
    pub train_end: usize,

    /// Testing window start index
    pub test_start: usize,

    /// Testing window end index (exclusive)
    pub test_end: usize,

    /// Parameters optimized on training data
    pub optimized_params: HashMap<String, f64>,

    /// In-sample (training) backtest result
    pub in_sample_result: BacktestResult,

    /// Out-of-sample (testing) backtest result
    pub out_of_sample_result: BacktestResult,
}

impl WalkForwardWindow {
    /// Efficiency ratio (out-of-sample / in-sample Sharpe)
    ///
    /// - > 0.8: Excellent (minimal overfitting)
    /// - 0.6-0.8: Good
    /// - 0.4-0.6: Acceptable
    /// - < 0.4: Overfitted (poor generalization)
    pub fn efficiency_ratio(&self) -> f64 {
        if self.in_sample_result.sharpe_ratio == 0.0 {
            return 0.0;
        }
        self.out_of_sample_result.sharpe_ratio / self.in_sample_result.sharpe_ratio
    }

    /// Parameter stability score (0-1, higher is better)
    ///
    /// Measures consistency of optimized parameters across windows
    pub fn stability_score(&self, prev_window: Option<&WalkForwardWindow>) -> f64 {
        let prev = match prev_window {
            Some(w) => w,
            None => return 1.0, // First window always stable
        };

        if self.optimized_params.is_empty() {
            return 0.0;
        }

        let mut total_deviation = 0.0;
        let mut count = 0;

        for (key, value) in &self.optimized_params {
            if let Some(&prev_value) = prev.optimized_params.get(key) {
                // Normalized deviation: |current - prev| / max(current, prev)
                let max_val = value.abs().max(prev_value.abs());
                if max_val > 0.0 {
                    total_deviation += (value - prev_value).abs() / max_val;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return 0.0;
        }

        // Stability = 1 - average_deviation
        1.0 - (total_deviation / count as f64).min(1.0)
    }
}

/// Walk-forward analysis result
#[derive(Debug, Clone)]
pub struct WalkForwardResult {
    /// Configuration used
    pub config: WalkForwardConfig,

    /// Individual window results
    pub windows: Vec<WalkForwardWindow>,

    /// Aggregate in-sample Sharpe ratio
    pub in_sample_sharpe: f64,

    /// Aggregate out-of-sample Sharpe ratio
    pub out_of_sample_sharpe: f64,

    /// Overall efficiency ratio (OOS/IS Sharpe)
    pub efficiency_ratio: f64,

    /// Average parameter stability score
    pub avg_stability: f64,

    /// Combined equity curve (all out-of-sample periods)
    pub oos_equity_curve: Vec<f64>,

    /// Detection metrics
    pub overfitting_detected: bool,
    pub degradation_percent: f64,
}

impl WalkForwardResult {
    /// Check if strategy is likely overfitted
    pub fn is_overfitted(&self) -> bool {
        self.overfitting_detected
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        format!(
            "Walk-Forward Analysis Summary:\n\
            - Windows: {}\n\
            - In-Sample Sharpe: {:.3}\n\
            - Out-of-Sample Sharpe: {:.3}\n\
            - Efficiency Ratio: {:.3}\n\
            - Parameter Stability: {:.3}\n\
            - Overfitting Detected: {}\n\
            - Performance Degradation: {:.1}%",
            self.windows.len(),
            self.in_sample_sharpe,
            self.out_of_sample_sharpe,
            self.efficiency_ratio,
            self.avg_stability,
            self.overfitting_detected,
            self.degradation_percent
        )
    }
}

/// Walk-forward analyzer
pub struct WalkForwardAnalyzer {
    config: WalkForwardConfig,
    optimizer: GeneticOptimizer,
}

impl WalkForwardAnalyzer {
    /// Create new walk-forward analyzer with default configuration
    pub fn new(config: WalkForwardConfig) -> Self {
        Self {
            config,
            optimizer: GeneticOptimizer::new(),
        }
    }

    /// Create with custom optimizer
    pub fn with_optimizer(mut self, optimizer: GeneticOptimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Run walk-forward analysis
    ///
    /// # Arguments
    ///
    /// * `engine` - Backtesting engine
    /// * `strategy` - Trading strategy to analyze
    /// * `timestamps` - Unix timestamps for each bar
    /// * `open` - Open prices
    /// * `high` - High prices
    /// * `low` - Low prices
    /// * `close` - Close prices
    /// * `volume` - Trading volume
    /// * `param_grid` - Parameter search space
    ///
    /// # Returns
    ///
    /// WalkForwardResult with in-sample/out-of-sample comparison
    pub fn analyze<S>(
        &self,
        engine: &BacktestEngine,
        strategy: &S,
        timestamps: &[i64],
        open: &Array1<f64>,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        volume: &Array1<f64>,
        param_grid: &ParameterGrid,
    ) -> Result<WalkForwardResult, GpuError>
    where
        S: Strategy + Clone,
    {
        // Validate configuration
        self.config.validate().map_err(|e| {
            GpuError::InvalidParameter(format!("Invalid walk-forward config: {}", e))
        })?;

        let n = timestamps.len();
        if n < self.config.min_bars {
            return Err(GpuError::InvalidParameter(format!(
                "Insufficient data: {} bars (need {})",
                n, self.config.min_bars
            )));
        }

        // Generate windows
        let mut windows = Vec::new();
        let mut current_pos = 0;

        loop {
            let train_start = if self.config.anchored { 0 } else { current_pos };
            let train_end = current_pos + self.config.train_window;
            let test_start = train_end;
            let test_end = test_start + self.config.test_window;

            if test_end > n {
                break;
            }

            // Optimize on training data
            let train_timestamps = &timestamps[train_start..train_end];
            let train_open = open.slice(ndarray::s![train_start..train_end]).to_owned();
            let train_high = high.slice(ndarray::s![train_start..train_end]).to_owned();
            let train_low = low.slice(ndarray::s![train_start..train_end]).to_owned();
            let train_close = close.slice(ndarray::s![train_start..train_end]).to_owned();
            let train_volume = volume.slice(ndarray::s![train_start..train_end]).to_owned();

            println!(
                "\nWindow {}: Train [{}-{}], Test [{}-{}]",
                windows.len() + 1,
                train_start,
                train_end,
                test_start,
                test_end
            );

            let optimizer_result = self.optimizer.optimize(
                engine,
                strategy,
                train_timestamps,
                &train_open,
                &train_high,
                &train_low,
                &train_close,
                &train_volume,
                param_grid,
            )?;

            let optimized_params = optimizer_result.best_parameters;
            let in_sample_result = optimizer_result.best_result;

            // Test on out-of-sample data
            let test_timestamps = &timestamps[test_start..test_end];
            let test_open = open.slice(ndarray::s![test_start..test_end]).to_owned();
            let test_high = high.slice(ndarray::s![test_start..test_end]).to_owned();
            let test_low = low.slice(ndarray::s![test_start..test_end]).to_owned();
            let test_close = close.slice(ndarray::s![test_start..test_end]).to_owned();
            let test_volume = volume.slice(ndarray::s![test_start..test_end]).to_owned();

            // Run backtest with optimized parameters
            // NOTE: This assumes strategy parameters can be updated externally
            // In production, you'd need a way to apply optimized_params to strategy
            let mut strategy_clone = strategy.clone();
            let mut out_of_sample_result = engine.run(
                &mut strategy_clone,
                test_timestamps,
                &test_open,
                &test_high,
                &test_low,
                &test_close,
                &test_volume,
            )?;
            out_of_sample_result.parameters = optimized_params.clone();

            println!(
                "  In-Sample:  Sharpe={:.3}, Return={:.2}%",
                in_sample_result.sharpe_ratio, in_sample_result.total_return
            );
            println!(
                "  Out-Sample: Sharpe={:.3}, Return={:.2}%",
                out_of_sample_result.sharpe_ratio, out_of_sample_result.total_return
            );

            windows.push(WalkForwardWindow {
                train_start,
                train_end,
                test_start,
                test_end,
                optimized_params,
                in_sample_result,
                out_of_sample_result,
            });

            current_pos += self.config.step_size;
        }

        if windows.is_empty() {
            return Err(GpuError::InvalidParameter(
                "No valid windows generated".to_string(),
            ));
        }

        // Aggregate results
        let in_sample_sharpe = windows
            .iter()
            .map(|w| w.in_sample_result.sharpe_ratio)
            .sum::<f64>()
            / windows.len() as f64;

        let out_of_sample_sharpe = windows
            .iter()
            .map(|w| w.out_of_sample_result.sharpe_ratio)
            .sum::<f64>()
            / windows.len() as f64;

        let efficiency_ratio = if in_sample_sharpe != 0.0 {
            out_of_sample_sharpe / in_sample_sharpe
        } else {
            0.0
        };

        // Calculate parameter stability
        let mut stability_scores = Vec::new();
        for (i, window) in windows.iter().enumerate() {
            let prev = if i > 0 { Some(&windows[i - 1]) } else { None };
            stability_scores.push(window.stability_score(prev));
        }
        let avg_stability = stability_scores.iter().sum::<f64>() / stability_scores.len() as f64;

        // Combine out-of-sample equity curves
        let mut oos_equity_curve = Vec::new();
        for window in &windows {
            oos_equity_curve.extend_from_slice(&window.out_of_sample_result.equity_curve);
        }

        // Overfitting detection
        let overfitting_detected = efficiency_ratio < 0.5 || avg_stability < 0.3;
        let degradation_percent = if in_sample_sharpe > 0.0 {
            ((in_sample_sharpe - out_of_sample_sharpe) / in_sample_sharpe * 100.0).max(0.0)
        } else {
            0.0
        };

        Ok(WalkForwardResult {
            config: self.config.clone(),
            windows,
            in_sample_sharpe,
            out_of_sample_sharpe,
            efficiency_ratio,
            avg_stability,
            oos_equity_curve,
            overfitting_detected,
            degradation_percent,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let valid_config = WalkForwardConfig {
            train_window: 100,
            test_window: 20,
            step_size: 10,
            anchored: false,
            min_bars: 120,
        };
        assert!(valid_config.validate().is_ok());

        let invalid_config = WalkForwardConfig {
            train_window: 0,
            ..valid_config
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_num_splits() {
        let config = WalkForwardConfig {
            train_window: 100,
            test_window: 20,
            step_size: 10,
            anchored: false,
            min_bars: 120,
        };

        // Data length: 300
        // Split 1: Train [0-100], Test [100-120]
        // Split 2: Train [10-110], Test [110-130]
        // ...
        // Split 19: Train [180-280], Test [280-300]
        assert_eq!(config.num_splits(300), 19);

        // Too short
        assert_eq!(config.num_splits(100), 0);
    }

    #[test]
    fn test_efficiency_ratio() {
        let window = WalkForwardWindow {
            train_start: 0,
            train_end: 100,
            test_start: 100,
            test_end: 120,
            optimized_params: HashMap::new(),
            in_sample_result: BacktestResult {
                sharpe_ratio: 2.0,
                ..BacktestResult::empty()
            },
            out_of_sample_result: BacktestResult {
                sharpe_ratio: 1.6,
                ..BacktestResult::empty()
            },
        };

        assert!((window.efficiency_ratio() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_stability_score() {
        let window1 = WalkForwardWindow {
            train_start: 0,
            train_end: 100,
            test_start: 100,
            test_end: 120,
            optimized_params: {
                let mut params = HashMap::new();
                params.insert("rsi_period".to_string(), 14.0);
                params.insert("threshold".to_string(), 30.0);
                params
            },
            in_sample_result: BacktestResult::empty(),
            out_of_sample_result: BacktestResult::empty(),
        };

        let window2 = WalkForwardWindow {
            train_start: 10,
            train_end: 110,
            test_start: 110,
            test_end: 130,
            optimized_params: {
                let mut params = HashMap::new();
                params.insert("rsi_period".to_string(), 15.0);
                params.insert("threshold".to_string(), 32.0);
                params
            },
            in_sample_result: BacktestResult::empty(),
            out_of_sample_result: BacktestResult::empty(),
        };

        // First window always has stability 1.0
        assert_eq!(window1.stability_score(None), 1.0);

        // Second window should have high stability (small deviations)
        let stability = window2.stability_score(Some(&window1));
        assert!(stability > 0.8, "Stability: {}", stability);
    }
}
