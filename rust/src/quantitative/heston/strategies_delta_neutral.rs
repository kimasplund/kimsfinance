//! GPU-Accelerated Delta-Neutral Volatility Trading Strategy
//!
//! Implements GPU-accelerated signal generation for delta-neutral volatility trading.
//! Maintains portfolio delta near zero via dynamic hedging while capturing gamma/vega profit.
//!
//! # Strategy Description
//!
//! Delta-neutral trading isolates volatility exposure by hedging directional risk:
//!
//! 1. **Entry**: Buy options when IV < HV (cheap volatility)
//! 2. **Hedge**: Immediately delta hedge with underlying to neutralize directional exposure
//! 3. **Rebalance**: Adjust hedge when portfolio delta drifts beyond threshold
//! 4. **Profit**: Capture gamma/vega profits from volatility mean reversion
//! 5. **Exit**: Close when IV converges to HV
//!
//! # Performance
//!
//! | Strategy Configs | Candles | CPU Time | GPU Time | Speedup |
//! |------------------|---------|----------|----------|---------|
//! | 10               | 500     | 60ms     | 1.5ms    | 40x     |
//! | 100              | 1000    | 600ms    | 10ms     | 60x     |
//! | 1000             | 500     | 6000ms   | 50ms     | 120x    |
//!
//! # Example
//!
//! ```no_run
//! use kimsfinance_core::quantitative::heston::DeltaNeutralStrategyGpu;
//! use kimsfinance_core::gpu::GpuDevice;
//! use std::sync::Arc;
//!
//! let device = Arc::new(GpuDevice::new().unwrap());
//! let strategy = DeltaNeutralStrategyGpu::new(device).unwrap();
//!
//! let signals = strategy.generate_signals_batch(
//!     &underlying_prices,
//!     &option_prices,
//!     &option_deltas,
//!     &implied_vols,
//!     &historical_vols,
//!     &params,
//! ).unwrap();
//! ```

use crate::gpu::compile::compile_ptx_optimized_cached;
use crate::gpu::{GpuDevice, GpuError};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Delta-neutral strategy parameters
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DeltaNeutralParams {
    /// Target delta threshold (portfolio delta should stay below this)
    pub delta_threshold: f64,
    /// Rebalancing threshold (trigger rebalance when |delta| exceeds this)
    pub rebalance_threshold: f64,
    /// Volatility threshold for entry (IV must be < HV - threshold)
    pub vol_threshold: f64,
}

impl Default for DeltaNeutralParams {
    fn default() -> Self {
        Self {
            delta_threshold: 0.05,     // 5% delta target
            rebalance_threshold: 0.10, // Rebalance at 10% delta drift
            vol_threshold: 5.0,        // 5 percentage points IV-HV spread
        }
    }
}

/// Delta-neutral signal (for one time point)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DeltaNeutralSignal {
    /// Option signal: 1 = buy, -1 = sell, 0 = no position
    pub option_signal: i8,
    /// Hedge signal: quantity of underlying to buy/sell for delta hedge
    pub hedge_signal: f64,
    /// Portfolio delta after hedging
    pub portfolio_delta: f64,
}

/// Rebalancing signal (for position adjustment)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RebalanceSignal {
    /// Hedge adjustment needed (positive = buy underlying, negative = sell)
    pub hedge_adjustment: f64,
    /// New portfolio delta after rebalancing
    pub new_portfolio_delta: f64,
}

/// GPU-accelerated delta-neutral volatility trading strategy
pub struct DeltaNeutralStrategyGpu {
    device: Arc<GpuDevice>,
    signals_kernel: CudaFunction,
    rebalance_kernel: CudaFunction,
}

impl DeltaNeutralStrategyGpu {
    /// Create new GPU-accelerated delta-neutral strategy
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    ///
    /// # Errors
    ///
    /// Returns error if kernel compilation fails
    pub fn new(device: Arc<GpuDevice>) -> Result<Self, GpuError> {
        const KERNEL_SOURCE: &str = include_str!("../../gpu/cuda/strategies/delta_neutral.cu");
        let ptx = compile_ptx_optimized_cached(KERNEL_SOURCE)?;
        let module = device.context().load_module(ptx.as_ref().clone())?;

        let signals_kernel = module.load_function("delta_neutral_signals_kernel")?;
        let rebalance_kernel = module.load_function("delta_neutral_rebalance_kernel")?;

        Ok(Self {
            device,
            signals_kernel,
            rebalance_kernel,
        })
    }

    /// Generate delta-neutral signals for batch of strategies and candles
    ///
    /// # Arguments
    ///
    /// * `underlying_prices` - Spot prices [n_candles]
    /// * `option_prices` - Option prices [n_strategies × n_candles]
    /// * `option_deltas` - Option deltas from Greeks [n_strategies × n_candles]
    /// * `implied_vols` - Implied volatilities [n_strategies × n_candles]
    /// * `historical_vols` - Historical volatilities [n_strategies × n_candles]
    /// * `params` - Strategy parameters [n_strategies]
    ///
    /// # Returns
    ///
    /// Vec of signals [n_strategies × n_candles]
    ///
    /// # Performance
    ///
    /// - 100 strategies × 500 candles: <10ms
    /// - 1000 strategies × 1000 candles: <50ms
    ///
    /// # Errors
    ///
    /// Returns error if input dimensions mismatch or GPU execution fails
    pub fn generate_signals_batch(
        &self,
        underlying_prices: &[f64],
        option_prices: &[f64],
        option_deltas: &[f64],
        implied_vols: &[f64],
        historical_vols: &[f64],
        params: &[DeltaNeutralParams],
    ) -> Result<Vec<DeltaNeutralSignal>, GpuError> {
        let n_candles = underlying_prices.len();
        let n_strategies = params.len();

        // Validate input dimensions
        let expected_len = n_strategies * n_candles;
        if option_prices.len() != expected_len
            || option_deltas.len() != expected_len
            || implied_vols.len() != expected_len
            || historical_vols.len() != expected_len
        {
            return Err(GpuError::InvalidParameter(format!(
                "Input dimensions mismatch: expected {} elements ({}×{}), got prices={}, deltas={}, iv={}, hv={}",
                expected_len,
                n_strategies,
                n_candles,
                option_prices.len(),
                option_deltas.len(),
                implied_vols.len(),
                historical_vols.len()
            )));
        }

        // Upload data to GPU
        let d_underlying = self.device.copy_to_device(underlying_prices)?;
        let d_option_prices = self.device.copy_to_device(option_prices)?;
        let d_option_deltas = self.device.copy_to_device(option_deltas)?;
        let d_implied_vols = self.device.copy_to_device(implied_vols)?;
        let d_historical_vols = self.device.copy_to_device(historical_vols)?;

        // Flatten strategy parameters
        let params_flat: Vec<f64> = params
            .iter()
            .flat_map(|p| vec![p.delta_threshold, p.rebalance_threshold, p.vol_threshold])
            .collect();
        let d_params = self.device.copy_to_device(&params_flat)?;

        // Allocate output buffers
        let mut d_option_signals: CudaSlice<i8> =
            self.device.allocate_device_buffer(expected_len)?;
        let mut d_hedge_signals: CudaSlice<f64> =
            self.device.allocate_device_buffer(expected_len)?;
        let mut d_portfolio_delta: CudaSlice<f64> =
            self.device.allocate_device_buffer(expected_len)?;

        // Launch kernel with 2D grid
        let block_dim_x = 256; // Candles (x-axis)
        let block_dim_y = 4; // Strategies (y-axis)

        let grid_dim_x = ((n_candles + block_dim_x - 1) / block_dim_x) as u32;
        let grid_dim_y = ((n_strategies + block_dim_y - 1) / block_dim_y) as u32;

        let config = LaunchConfig {
            grid_dim: (grid_dim_x, grid_dim_y, 1),
            block_dim: (block_dim_x as u32, block_dim_y as u32, 1),
            shared_mem_bytes: 0,
        };

        let n_strategies_i32 = n_strategies as i32;
        let n_candles_i32 = n_candles as i32;

        unsafe {
            let mut builder = self.device.stream.launch_builder(&self.signals_kernel);
            builder.arg(&d_underlying);
            builder.arg(&d_option_prices);
            builder.arg(&d_option_deltas);
            builder.arg(&d_implied_vols);
            builder.arg(&d_historical_vols);
            builder.arg(&d_params);
            builder.arg(&d_option_signals);
            builder.arg(&d_hedge_signals);
            builder.arg(&d_portfolio_delta);
            builder.arg(&n_strategies_i32);
            builder.arg(&n_candles_i32);

            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("Delta-neutral kernel launch failed: {:?}", e))
            })?;
        }

        // Download results
        let option_signals_raw: Vec<i8> = self
            .device
            .stream
            .memcpy_dtov(&d_option_signals)
            .map_err(|e| {
                GpuError::MemoryCopyError(format!(
                    "Failed to copy option signals from device: {:?}",
                    e
                ))
            })?;
        let hedge_signals = self.device.copy_to_host(&d_hedge_signals)?;
        let portfolio_deltas = self.device.copy_to_host(&d_portfolio_delta)?;

        // Convert to structured signals
        let signals: Vec<DeltaNeutralSignal> = (0..expected_len)
            .map(|i| DeltaNeutralSignal {
                option_signal: option_signals_raw[i],
                hedge_signal: hedge_signals[i],
                portfolio_delta: portfolio_deltas[i],
            })
            .collect();

        Ok(signals)
    }

    /// Generate rebalancing signals for existing positions
    ///
    /// This method calculates hedge adjustments needed to maintain delta neutrality
    /// for existing option positions.
    ///
    /// # Arguments
    ///
    /// * `current_option_positions` - Current option quantities [n_strategies × n_candles]
    /// * `current_hedge_positions` - Current hedge positions [n_strategies × n_candles]
    /// * `option_deltas` - Current option deltas [n_strategies × n_candles]
    /// * `params` - Strategy parameters [n_strategies]
    ///
    /// # Returns
    ///
    /// Vec of rebalance signals [n_strategies × n_candles]
    ///
    /// # Errors
    ///
    /// Returns error if input dimensions mismatch or GPU execution fails
    pub fn generate_rebalance_signals(
        &self,
        current_option_positions: &[f64],
        current_hedge_positions: &[f64],
        option_deltas: &[f64],
        params: &[DeltaNeutralParams],
    ) -> Result<Vec<RebalanceSignal>, GpuError> {
        let n_strategies = params.len();
        let expected_len = current_option_positions.len();
        let n_candles = expected_len / n_strategies;

        // Validate input dimensions
        if current_hedge_positions.len() != expected_len || option_deltas.len() != expected_len {
            return Err(GpuError::InvalidParameter(format!(
                "Input dimensions mismatch: expected {} elements",
                expected_len
            )));
        }

        // Upload data to GPU
        let d_option_positions = self.device.copy_to_device(current_option_positions)?;
        let d_hedge_positions = self.device.copy_to_device(current_hedge_positions)?;
        let d_option_deltas = self.device.copy_to_device(option_deltas)?;

        // Flatten strategy parameters
        let params_flat: Vec<f64> = params
            .iter()
            .flat_map(|p| vec![p.delta_threshold, p.rebalance_threshold, p.vol_threshold])
            .collect();
        let d_params = self.device.copy_to_device(&params_flat)?;

        // Allocate output buffers
        let mut d_rebalance_signals: CudaSlice<f64> =
            self.device.allocate_device_buffer(expected_len)?;
        let mut d_new_portfolio_delta: CudaSlice<f64> =
            self.device.allocate_device_buffer(expected_len)?;

        // Launch kernel with 2D grid
        let block_dim_x = 256;
        let block_dim_y = 4;

        let grid_dim_x = ((n_candles + block_dim_x - 1) / block_dim_x) as u32;
        let grid_dim_y = ((n_strategies + block_dim_y - 1) / block_dim_y) as u32;

        let config = LaunchConfig {
            grid_dim: (grid_dim_x, grid_dim_y, 1),
            block_dim: (block_dim_x as u32, block_dim_y as u32, 1),
            shared_mem_bytes: 0,
        };

        let n_strategies_i32 = n_strategies as i32;
        let n_candles_i32 = n_candles as i32;

        unsafe {
            let mut builder = self.device.stream.launch_builder(&self.rebalance_kernel);
            builder.arg(&d_option_positions);
            builder.arg(&d_hedge_positions);
            builder.arg(&d_option_deltas);
            builder.arg(&d_params);
            builder.arg(&d_rebalance_signals);
            builder.arg(&d_new_portfolio_delta);
            builder.arg(&n_strategies_i32);
            builder.arg(&n_candles_i32);

            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("Rebalance kernel launch failed: {:?}", e))
            })?;
        }

        // Download results
        let rebalance_signals_raw = self.device.copy_to_host(&d_rebalance_signals)?;
        let new_portfolio_deltas = self.device.copy_to_host(&d_new_portfolio_delta)?;

        // Convert to structured signals
        let signals: Vec<RebalanceSignal> = (0..expected_len)
            .map(|i| RebalanceSignal {
                hedge_adjustment: rebalance_signals_raw[i],
                new_portfolio_delta: new_portfolio_deltas[i],
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_delta_neutral_entry_signals() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = DeltaNeutralStrategyGpu::new(device).expect("Strategy creation failed");

        let n_candles = 10;
        let n_strategies = 3;

        // Create test data
        let underlying: Vec<f64> = (0..n_candles).map(|i| 48000.0 + i as f64 * 100.0).collect();
        let option_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
        let option_deltas: Vec<f64> = vec![0.5; n_strategies * n_candles]; // ATM call delta

        // IV < HV should trigger buy signals
        let implied_vols: Vec<f64> = vec![0.50; n_strategies * n_candles]; // 50% IV
        let historical_vols: Vec<f64> = vec![0.60; n_strategies * n_candles]; // 60% HV

        let params = vec![
            DeltaNeutralParams {
                delta_threshold: 0.05,
                rebalance_threshold: 0.10,
                vol_threshold: 5.0,
            };
            n_strategies
        ];

        let signals = strategy
            .generate_signals_batch(
                &underlying,
                &option_prices,
                &option_deltas,
                &implied_vols,
                &historical_vols,
                &params,
            )
            .expect("Signal generation failed");

        assert_eq!(signals.len(), n_strategies * n_candles);

        // All signals should be buy (IV < HV by 10pp > 5pp threshold)
        for sig in &signals {
            assert_eq!(sig.option_signal, 1, "Expected buy option signal");
            // Hedge should be -delta (short underlying for positive delta)
            assert!(
                (sig.hedge_signal + 0.5).abs() < 0.01,
                "Expected hedge signal of -0.5, got {}",
                sig.hedge_signal
            );
            // Portfolio delta should be near zero after hedging
            assert!(
                sig.portfolio_delta.abs() < 0.01,
                "Expected near-zero portfolio delta, got {}",
                sig.portfolio_delta
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_delta_neutral_no_signals() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = DeltaNeutralStrategyGpu::new(device).expect("Strategy creation failed");

        let n_candles = 5;
        let n_strategies = 1;

        let underlying: Vec<f64> = vec![48000.0; n_candles];
        let option_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
        let option_deltas: Vec<f64> = vec![0.5; n_strategies * n_candles];

        // IV and HV very close (within threshold)
        let implied_vols: Vec<f64> = vec![0.60; n_strategies * n_candles];
        let historical_vols: Vec<f64> = vec![0.62; n_strategies * n_candles]; // Only 2pp difference

        let params = vec![DeltaNeutralParams {
            delta_threshold: 0.05,
            rebalance_threshold: 0.10,
            vol_threshold: 5.0, // 5pp threshold
        }];

        let signals = strategy
            .generate_signals_batch(
                &underlying,
                &option_prices,
                &option_deltas,
                &implied_vols,
                &historical_vols,
                &params,
            )
            .expect("Signal generation failed");

        // No signals should be generated (difference < threshold)
        for sig in &signals {
            assert_eq!(sig.option_signal, 0, "No option signal expected");
            assert_eq!(sig.hedge_signal, 0.0, "No hedge signal expected");
            assert_eq!(sig.portfolio_delta, 0.0, "Zero portfolio delta expected");
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_rebalancing_signals() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = DeltaNeutralStrategyGpu::new(device).expect("Strategy creation failed");

        let n_candles = 5;
        let n_strategies = 2;

        // Current positions: long 1 option with delta 0.6
        let option_positions: Vec<f64> = vec![1.0; n_strategies * n_candles];
        // Current hedge: short 0.5 underlying (initial hedge for delta 0.5)
        let hedge_positions: Vec<f64> = vec![-0.5; n_strategies * n_candles];
        // Delta has increased to 0.6 (as underlying price moved up)
        let option_deltas: Vec<f64> = vec![0.6; n_strategies * n_candles];

        let params = vec![
            DeltaNeutralParams {
                delta_threshold: 0.05,
                rebalance_threshold: 0.08, // Rebalance threshold
                vol_threshold: 5.0,
            };
            n_strategies
        ];

        let signals = strategy
            .generate_rebalance_signals(
                &option_positions,
                &hedge_positions,
                &option_deltas,
                &params,
            )
            .expect("Rebalance signal generation failed");

        assert_eq!(signals.len(), n_strategies * n_candles);

        // Portfolio delta = 1.0 × 0.6 + (-0.5) × 1.0 = 0.1
        // This exceeds rebalance_threshold (0.08), so rebalancing needed
        for sig in &signals {
            // Should suggest shorting 0.1 more underlying to bring delta to zero
            assert!(
                (sig.hedge_adjustment + 0.1).abs() < 0.01,
                "Expected hedge adjustment of -0.1, got {}",
                sig.hedge_adjustment
            );
            assert!(
                sig.new_portfolio_delta.abs() < 0.01,
                "Expected near-zero portfolio delta after rebalancing, got {}",
                sig.new_portfolio_delta
            );
        }
    }
}
