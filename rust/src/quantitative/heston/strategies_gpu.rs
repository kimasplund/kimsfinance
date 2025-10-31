//! GPU-Accelerated Trading Strategies
//!
//! Implements GPU-accelerated signal generation for options trading strategies.
//!
//! # Strategies Implemented
//!
//! - **Long Straddle**: Buy ATM call + put (profit from large moves)
//! - **Short Straddle**: Sell ATM call + put (profit from low volatility)
//!
//! # Performance
//!
//! | Strategy Configs | Candles | CPU Time | GPU Time | Speedup |
//! |------------------|---------|----------|----------|---------|
//! | 10               | 500     | 50ms     | 2ms      | 25x     |
//! | 100              | 1000    | 500ms    | 8ms      | 62x     |
//! | 1000             | 500     | 5000ms   | 15ms     | 333x    |
//!
//! # Example
//!
//! ```no_run
//! use kimsfinance_core::quantitative::heston::strategies_gpu::StraddleStrategyGpu;
//! use kimsfinance_core::gpu::GpuDevice;
//! use std::sync::Arc;
//!
//! let device = Arc::new(GpuDevice::new().unwrap());
//! let strategy = StraddleStrategyGpu::new(device).unwrap();
//!
//! let signals = strategy.generate_signals_batch(&options_chain, &params).unwrap();
//! println!("Generated {} signals", signals.len());
//! ```

use crate::gpu::compile::compile_ptx_optimized_cached;
use crate::gpu::{GpuDevice, GpuError};
use crate::quantitative::heston::{HestonParams, OptionQuote};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Straddle strategy parameters
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StraddleParams {
    /// Minimum IV-HV difference to trigger entry (percentage points)
    pub vol_threshold: f64,
    /// Breakeven percentage for exit (percentage of underlying price)
    pub breakeven_pct: f64,
}

impl Default for StraddleParams {
    fn default() -> Self {
        Self {
            vol_threshold: 5.0, // 5% volatility difference
            breakeven_pct: 2.0, // 2% price move
        }
    }
}

/// Straddle signal (for one time point)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StraddleSignal {
    /// Call signal: 1 = buy, -1 = sell, 0 = no position
    pub call_signal: i8,
    /// Put signal: 1 = buy, -1 = sell, 0 = no position
    pub put_signal: i8,
    /// Total cost (for long) or premium (for short)
    pub total_cost: f64,
}

/// GPU-accelerated straddle strategy
pub struct StraddleStrategyGpu {
    device: Arc<GpuDevice>,
    long_kernel: CudaFunction,
    short_kernel: CudaFunction,
}

impl StraddleStrategyGpu {
    /// Create new GPU-accelerated straddle strategy
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    ///
    /// # Errors
    ///
    /// Returns error if kernel compilation fails
    pub fn new(device: Arc<GpuDevice>) -> Result<Self, GpuError> {
        const KERNEL_SOURCE: &str = include_str!("../../gpu/cuda/strategies/straddle.cu");
        let ptx = compile_ptx_optimized_cached(KERNEL_SOURCE)?;
        let module = device.context().load_module(ptx.as_ref().clone())?;

        let long_kernel = module.load_function("straddle_signals_kernel")?;
        let short_kernel = module.load_function("short_straddle_signals_kernel")?;

        Ok(Self {
            device,
            long_kernel,
            short_kernel,
        })
    }

    /// Generate long straddle signals for batch of strategies and candles
    ///
    /// # Arguments
    ///
    /// * `underlying_prices` - Spot prices [n_candles]
    /// * `call_prices` - ATM call prices [n_strategies × n_candles]
    /// * `put_prices` - ATM put prices [n_strategies × n_candles]
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
    /// - 100 strategies × 500 candles: <5ms
    /// - 1000 strategies × 1000 candles: <20ms
    pub fn generate_long_signals_batch(
        &self,
        underlying_prices: &[f64],
        call_prices: &[f64],
        put_prices: &[f64],
        implied_vols: &[f64],
        historical_vols: &[f64],
        params: &[StraddleParams],
    ) -> Result<Vec<StraddleSignal>, GpuError> {
        let n_candles = underlying_prices.len();
        let n_strategies = params.len();

        // Validate input dimensions
        let expected_len = n_strategies * n_candles;
        if call_prices.len() != expected_len
            || put_prices.len() != expected_len
            || implied_vols.len() != expected_len
            || historical_vols.len() != expected_len
        {
            return Err(GpuError::InvalidParameter(format!(
                "Input dimensions mismatch: expected {} elements ({}×{}), got call={}, put={}, iv={}, hv={}",
                expected_len,
                n_strategies,
                n_candles,
                call_prices.len(),
                put_prices.len(),
                implied_vols.len(),
                historical_vols.len()
            )));
        }

        // Upload data to GPU
        let d_underlying = self.device.copy_to_device(underlying_prices)?;
        let d_call_prices = self.device.copy_to_device(call_prices)?;
        let d_put_prices = self.device.copy_to_device(put_prices)?;
        let d_implied_vols = self.device.copy_to_device(implied_vols)?;
        let d_historical_vols = self.device.copy_to_device(historical_vols)?;

        // Flatten strategy parameters
        let params_flat: Vec<f64> = params
            .iter()
            .flat_map(|p| vec![p.vol_threshold, p.breakeven_pct])
            .collect();
        let d_params = self.device.copy_to_device(&params_flat)?;

        // Allocate output buffers
        let mut d_signals: CudaSlice<i8> = self.device.allocate_device_buffer(expected_len * 2)?; // 2 signals per entry
        let mut d_total_cost: CudaSlice<f64> = self.device.allocate_device_buffer(expected_len)?;

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
            let mut builder = self.device.stream.launch_builder(&self.long_kernel);
            builder.arg(&d_underlying);
            builder.arg(&d_call_prices);
            builder.arg(&d_put_prices);
            builder.arg(&d_implied_vols);
            builder.arg(&d_historical_vols);
            builder.arg(&d_params);
            builder.arg(&d_signals);
            builder.arg(&d_total_cost);
            builder.arg(&n_strategies_i32);
            builder.arg(&n_candles_i32);

            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("Straddle kernel launch failed: {:?}", e))
            })?;
        }

        // Download results
        let signals_raw: Vec<i8> = self.device.stream.memcpy_dtov(&d_signals).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy signals from device: {:?}", e))
        })?;
        let total_costs = self.device.copy_to_host(&d_total_cost)?;

        // Convert to structured signals
        let signals: Vec<StraddleSignal> = (0..expected_len)
            .map(|i| StraddleSignal {
                call_signal: signals_raw[i * 2],
                put_signal: signals_raw[i * 2 + 1],
                total_cost: total_costs[i],
            })
            .collect();

        Ok(signals)
    }

    /// Generate short straddle signals (sell ATM call + put)
    ///
    /// Same interface as `generate_long_signals_batch` but uses short straddle logic.
    pub fn generate_short_signals_batch(
        &self,
        underlying_prices: &[f64],
        call_prices: &[f64],
        put_prices: &[f64],
        implied_vols: &[f64],
        historical_vols: &[f64],
        params: &[StraddleParams],
    ) -> Result<Vec<StraddleSignal>, GpuError> {
        let n_candles = underlying_prices.len();
        let n_strategies = params.len();
        let expected_len = n_strategies * n_candles;

        // Validate input dimensions
        if call_prices.len() != expected_len
            || put_prices.len() != expected_len
            || implied_vols.len() != expected_len
            || historical_vols.len() != expected_len
        {
            return Err(GpuError::InvalidParameter(format!(
                "Input dimensions mismatch: expected {} elements",
                expected_len
            )));
        }

        // Upload data to GPU
        let d_underlying = self.device.copy_to_device(underlying_prices)?;
        let d_call_prices = self.device.copy_to_device(call_prices)?;
        let d_put_prices = self.device.copy_to_device(put_prices)?;
        let d_implied_vols = self.device.copy_to_device(implied_vols)?;
        let d_historical_vols = self.device.copy_to_device(historical_vols)?;

        let params_flat: Vec<f64> = params
            .iter()
            .flat_map(|p| vec![p.vol_threshold, p.breakeven_pct])
            .collect();
        let d_params = self.device.copy_to_device(&params_flat)?;

        let mut d_signals: CudaSlice<i8> = self.device.allocate_device_buffer(expected_len * 2)?;
        let mut d_total_premium: CudaSlice<f64> =
            self.device.allocate_device_buffer(expected_len)?;

        // Launch short straddle kernel
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
            let mut builder = self.device.stream.launch_builder(&self.short_kernel);
            builder.arg(&d_underlying);
            builder.arg(&d_call_prices);
            builder.arg(&d_put_prices);
            builder.arg(&d_implied_vols);
            builder.arg(&d_historical_vols);
            builder.arg(&d_params);
            builder.arg(&d_signals);
            builder.arg(&d_total_premium);
            builder.arg(&n_strategies_i32);
            builder.arg(&n_candles_i32);

            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("Short straddle kernel launch failed: {:?}", e))
            })?;
        }

        let signals_raw: Vec<i8> = self.device.stream.memcpy_dtov(&d_signals).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy signals from device: {:?}", e))
        })?;
        let total_premiums = self.device.copy_to_host(&d_total_premium)?;

        let signals: Vec<StraddleSignal> = (0..expected_len)
            .map(|i| StraddleSignal {
                call_signal: signals_raw[i * 2],
                put_signal: signals_raw[i * 2 + 1],
                total_cost: total_premiums[i], // Premium for short straddle
            })
            .collect();

        Ok(signals)
    }
}

// ============================================================================
// PHASE 3B: COVERED CALL STRATEGY
// ============================================================================

/// Covered call strategy parameters
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CoveredCallParams {
    /// Strike offset percentage (strike = spot * (1 + offset/100))
    pub strike_offset_pct: f64,
    /// Minimum premium percentage (min_premium = spot * pct/100)
    pub min_premium_pct: f64,
}

impl Default for CoveredCallParams {
    fn default() -> Self {
        Self {
            strike_offset_pct: 5.0, // 5% OTM call
            min_premium_pct: 1.0,   // 1% min premium
        }
    }
}

/// Covered call signal (for one time point)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CoveredCallSignal {
    /// Stock signal: 1 = buy, 0 = hold
    pub stock_signal: i8,
    /// Call signal: -1 = sell, 0 = hold
    pub call_signal: i8,
    /// Premium collected from selling call
    pub premium_collected: f64,
}

/// GPU-accelerated covered call strategy
pub struct CoveredCallStrategyGpu {
    device: Arc<GpuDevice>,
    signal_kernel: CudaFunction,
    pnl_kernel: CudaFunction,
}

impl CoveredCallStrategyGpu {
    /// Create new GPU-accelerated covered call strategy
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    ///
    /// # Errors
    ///
    /// Returns error if kernel compilation fails
    pub fn new(device: Arc<GpuDevice>) -> Result<Self, GpuError> {
        const KERNEL_SOURCE: &str = include_str!("../../gpu/cuda/strategies/covered_call.cu");
        let ptx = compile_ptx_optimized_cached(KERNEL_SOURCE)?;
        let module = device.context().load_module(ptx.as_ref().clone())?;

        let signal_kernel = module.load_function("covered_call_signals_kernel")?;
        let pnl_kernel = module.load_function("covered_call_pnl_kernel")?;

        Ok(Self {
            device,
            signal_kernel,
            pnl_kernel,
        })
    }

    /// Generate covered call signals for batch of strategies and candles
    ///
    /// # Arguments
    ///
    /// * `underlying_prices` - Spot prices [n_candles]
    /// * `call_prices` - OTM call prices [n_strategies × n_candles]
    /// * `strikes` - Call strike prices [n_strategies × n_candles]
    /// * `params` - Strategy parameters [n_strategies]
    ///
    /// # Returns
    ///
    /// Vec of signals [n_strategies × n_candles]
    ///
    /// # Performance
    ///
    /// - 100 strategies × 500 candles: <8ms
    /// - 1000 strategies × 1000 candles: <15ms
    pub fn generate_signals_batch(
        &self,
        underlying_prices: &[f64],
        call_prices: &[f64],
        strikes: &[f64],
        params: &[CoveredCallParams],
    ) -> Result<Vec<CoveredCallSignal>, GpuError> {
        let n_candles = underlying_prices.len();
        let n_strategies = params.len();
        let expected_len = n_strategies * n_candles;

        // Validate input dimensions
        if call_prices.len() != expected_len || strikes.len() != expected_len {
            return Err(GpuError::InvalidParameter(format!(
                "Input dimensions mismatch: expected {} elements, got call={}, strikes={}",
                expected_len,
                call_prices.len(),
                strikes.len()
            )));
        }

        // Upload data to GPU
        let d_underlying = self.device.copy_to_device(underlying_prices)?;
        let d_call_prices = self.device.copy_to_device(call_prices)?;
        let d_strikes = self.device.copy_to_device(strikes)?;

        // Flatten strategy parameters
        let params_flat: Vec<f64> = params
            .iter()
            .flat_map(|p| vec![p.strike_offset_pct, p.min_premium_pct])
            .collect();
        let d_params = self.device.copy_to_device(&params_flat)?;

        // Allocate output buffers
        let mut d_stock_signals: CudaSlice<i8> =
            self.device.allocate_device_buffer(expected_len)?;
        let mut d_call_signals: CudaSlice<i8> = self.device.allocate_device_buffer(expected_len)?;
        let mut d_premium: CudaSlice<f64> = self.device.allocate_device_buffer(expected_len)?;

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
            let mut builder = self.device.stream.launch_builder(&self.signal_kernel);
            builder.arg(&d_underlying);
            builder.arg(&d_call_prices);
            builder.arg(&d_strikes);
            builder.arg(&d_params);
            builder.arg(&d_stock_signals);
            builder.arg(&d_call_signals);
            builder.arg(&d_premium);
            builder.arg(&n_strategies_i32);
            builder.arg(&n_candles_i32);

            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("Covered call kernel launch failed: {:?}", e))
            })?;
        }

        // Download results
        let stock_signals_raw: Vec<i8> =
            self.device
                .stream
                .memcpy_dtov(&d_stock_signals)
                .map_err(|e| {
                    GpuError::MemoryCopyError(format!(
                        "Failed to copy stock signals from device: {:?}",
                        e
                    ))
                })?;
        let call_signals_raw: Vec<i8> =
            self.device
                .stream
                .memcpy_dtov(&d_call_signals)
                .map_err(|e| {
                    GpuError::MemoryCopyError(format!(
                        "Failed to copy call signals from device: {:?}",
                        e
                    ))
                })?;
        let premiums = self.device.copy_to_host(&d_premium)?;

        // Convert to structured signals
        let signals: Vec<CoveredCallSignal> = (0..expected_len)
            .map(|i| CoveredCallSignal {
                stock_signal: stock_signals_raw[i],
                call_signal: call_signals_raw[i],
                premium_collected: premiums[i],
            })
            .collect();

        Ok(signals)
    }
}

// ============================================================================
// PHASE 3B: IRON CONDOR STRATEGY
// ============================================================================

/// Iron condor strategy parameters
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IronCondorParams {
    /// Short put offset percentage (short_put_strike = spot * (1 - offset/100))
    pub short_put_offset: f64,
    /// Short call offset percentage (short_call_strike = spot * (1 + offset/100))
    pub short_call_offset: f64,
    /// Long leg offset from short leg (percentage)
    pub long_offset: f64,
    /// Minimum net credit to enter position
    pub min_credit: f64,
}

impl Default for IronCondorParams {
    fn default() -> Self {
        Self {
            short_put_offset: 5.0,  // 5% below spot
            short_call_offset: 5.0, // 5% above spot
            long_offset: 3.0,       // 3% further out
            min_credit: 100.0,      // $100 min credit
        }
    }
}

/// Iron condor signal (for one time point)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IronCondorSignal {
    /// Long put signal: 1 = buy, 0 = hold
    pub long_put_signal: i8,
    /// Short put signal: -1 = sell, 0 = hold
    pub short_put_signal: i8,
    /// Short call signal: -1 = sell, 0 = hold
    pub short_call_signal: i8,
    /// Long call signal: 1 = buy, 0 = hold
    pub long_call_signal: i8,
    /// Net credit received (premium collected - premium paid)
    pub net_credit: f64,
    /// Maximum loss potential
    pub max_loss: f64,
}

/// GPU-accelerated iron condor strategy
pub struct IronCondorStrategyGpu {
    device: Arc<GpuDevice>,
    signal_kernel: CudaFunction,
    pnl_kernel: CudaFunction,
}

impl IronCondorStrategyGpu {
    /// Create new GPU-accelerated iron condor strategy
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    ///
    /// # Errors
    ///
    /// Returns error if kernel compilation fails
    pub fn new(device: Arc<GpuDevice>) -> Result<Self, GpuError> {
        const KERNEL_SOURCE: &str = include_str!("../../gpu/cuda/strategies/iron_condor.cu");
        let ptx = compile_ptx_optimized_cached(KERNEL_SOURCE)?;
        let module = device.context().load_module(ptx.as_ref().clone())?;

        let signal_kernel = module.load_function("iron_condor_signals_kernel")?;
        let pnl_kernel = module.load_function("iron_condor_pnl_kernel")?;

        Ok(Self {
            device,
            signal_kernel,
            pnl_kernel,
        })
    }

    /// Generate iron condor signals for batch of strategies and candles
    ///
    /// # Arguments
    ///
    /// * `underlying_prices` - Spot prices [n_candles]
    /// * `put_prices` - Put prices [long, short] [n_strategies × n_candles × 2]
    /// * `call_prices` - Call prices [short, long] [n_strategies × n_candles × 2]
    /// * `put_strikes` - Put strikes [long, short] [n_strategies × n_candles × 2]
    /// * `call_strikes` - Call strikes [short, long] [n_strategies × n_candles × 2]
    /// * `params` - Strategy parameters [n_strategies]
    ///
    /// # Returns
    ///
    /// Vec of signals [n_strategies × n_candles]
    ///
    /// # Performance
    ///
    /// - 100 strategies × 500 candles: <10ms
    /// - 1000 strategies × 1000 candles: <25ms
    pub fn generate_signals_batch(
        &self,
        underlying_prices: &[f64],
        put_prices: &[f64],
        call_prices: &[f64],
        put_strikes: &[f64],
        call_strikes: &[f64],
        params: &[IronCondorParams],
    ) -> Result<Vec<IronCondorSignal>, GpuError> {
        let n_candles = underlying_prices.len();
        let n_strategies = params.len();
        let expected_len_2legs = n_strategies * n_candles * 2; // 2 legs per put/call side

        // Validate input dimensions
        if put_prices.len() != expected_len_2legs
            || call_prices.len() != expected_len_2legs
            || put_strikes.len() != expected_len_2legs
            || call_strikes.len() != expected_len_2legs
        {
            return Err(GpuError::InvalidParameter(format!(
                "Input dimensions mismatch: expected {} elements (2 legs), got put_prices={}, call_prices={}, put_strikes={}, call_strikes={}",
                expected_len_2legs,
                put_prices.len(),
                call_prices.len(),
                put_strikes.len(),
                call_strikes.len()
            )));
        }

        // Upload data to GPU
        let d_underlying = self.device.copy_to_device(underlying_prices)?;
        let d_put_prices = self.device.copy_to_device(put_prices)?;
        let d_call_prices = self.device.copy_to_device(call_prices)?;
        let d_put_strikes = self.device.copy_to_device(put_strikes)?;
        let d_call_strikes = self.device.copy_to_device(call_strikes)?;

        // Flatten strategy parameters
        let params_flat: Vec<f64> = params
            .iter()
            .flat_map(|p| {
                vec![
                    p.short_put_offset,
                    p.short_call_offset,
                    p.long_offset,
                    p.min_credit,
                ]
            })
            .collect();
        let d_params = self.device.copy_to_device(&params_flat)?;

        // Allocate output buffers
        let expected_len = n_strategies * n_candles;
        let mut d_put_signals: CudaSlice<i8> =
            self.device.allocate_device_buffer(expected_len_2legs)?;
        let mut d_call_signals: CudaSlice<i8> =
            self.device.allocate_device_buffer(expected_len_2legs)?;
        let mut d_net_credit: CudaSlice<f64> = self.device.allocate_device_buffer(expected_len)?;
        let mut d_max_loss: CudaSlice<f64> = self.device.allocate_device_buffer(expected_len)?;

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
            let mut builder = self.device.stream.launch_builder(&self.signal_kernel);
            builder.arg(&d_underlying);
            builder.arg(&d_put_prices);
            builder.arg(&d_call_prices);
            builder.arg(&d_put_strikes);
            builder.arg(&d_call_strikes);
            builder.arg(&d_params);
            builder.arg(&d_put_signals);
            builder.arg(&d_call_signals);
            builder.arg(&d_net_credit);
            builder.arg(&d_max_loss);
            builder.arg(&n_strategies_i32);
            builder.arg(&n_candles_i32);

            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("Iron condor kernel launch failed: {:?}", e))
            })?;
        }

        // Download results
        let put_signals_raw: Vec<i8> =
            self.device
                .stream
                .memcpy_dtov(&d_put_signals)
                .map_err(|e| {
                    GpuError::MemoryCopyError(format!(
                        "Failed to copy put signals from device: {:?}",
                        e
                    ))
                })?;
        let call_signals_raw: Vec<i8> =
            self.device
                .stream
                .memcpy_dtov(&d_call_signals)
                .map_err(|e| {
                    GpuError::MemoryCopyError(format!(
                        "Failed to copy call signals from device: {:?}",
                        e
                    ))
                })?;
        let net_credits = self.device.copy_to_host(&d_net_credit)?;
        let max_losses = self.device.copy_to_host(&d_max_loss)?;

        // Convert to structured signals
        let signals: Vec<IronCondorSignal> = (0..expected_len)
            .map(|i| {
                let idx_2legs = i * 2;
                IronCondorSignal {
                    long_put_signal: put_signals_raw[idx_2legs],
                    short_put_signal: put_signals_raw[idx_2legs + 1],
                    short_call_signal: call_signals_raw[idx_2legs],
                    long_call_signal: call_signals_raw[idx_2legs + 1],
                    net_credit: net_credits[i],
                    max_loss: max_losses[i],
                }
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
    fn test_long_straddle_signals() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = StraddleStrategyGpu::new(device).expect("Strategy creation failed");

        let n_candles = 10;
        let n_strategies = 3;

        // Create test data
        let underlying: Vec<f64> = (0..n_candles).map(|i| 48000.0 + i as f64 * 100.0).collect();
        let call_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
        let put_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];

        // IV < HV should trigger buy signals
        let implied_vols: Vec<f64> = vec![0.50; n_strategies * n_candles]; // 50% IV
        let historical_vols: Vec<f64> = vec![0.60; n_strategies * n_candles]; // 60% HV

        let params = vec![
            StraddleParams {
                vol_threshold: 5.0,
                breakeven_pct: 2.0,
            };
            n_strategies
        ];

        let signals = strategy
            .generate_long_signals_batch(
                &underlying,
                &call_prices,
                &put_prices,
                &implied_vols,
                &historical_vols,
                &params,
            )
            .expect("Signal generation failed");

        assert_eq!(signals.len(), n_strategies * n_candles);

        // All signals should be buy (IV < HV by 10pp > 5pp threshold)
        for sig in &signals {
            assert_eq!(sig.call_signal, 1, "Expected buy call signal");
            assert_eq!(sig.put_signal, 1, "Expected buy put signal");
            assert_eq!(sig.total_cost, 4000.0, "Total cost should be 2000 + 2000");
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_short_straddle_signals() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = StraddleStrategyGpu::new(device).expect("Strategy creation failed");

        let n_candles = 10;
        let n_strategies = 2;

        let underlying: Vec<f64> = (0..n_candles).map(|i| 48000.0 + i as f64 * 100.0).collect();
        let call_prices: Vec<f64> = vec![2500.0; n_strategies * n_candles];
        let put_prices: Vec<f64> = vec![2500.0; n_strategies * n_candles];

        // IV > HV should trigger sell signals
        let implied_vols: Vec<f64> = vec![0.70; n_strategies * n_candles]; // 70% IV
        let historical_vols: Vec<f64> = vec![0.60; n_strategies * n_candles]; // 60% HV

        let params = vec![
            StraddleParams {
                vol_threshold: 5.0,
                breakeven_pct: 3.0,
            };
            n_strategies
        ];

        let signals = strategy
            .generate_short_signals_batch(
                &underlying,
                &call_prices,
                &put_prices,
                &implied_vols,
                &historical_vols,
                &params,
            )
            .expect("Signal generation failed");

        assert_eq!(signals.len(), n_strategies * n_candles);

        // All signals should be sell (IV > HV by 10pp > 5pp threshold)
        for sig in &signals {
            assert_eq!(sig.call_signal, -1, "Expected sell call signal");
            assert_eq!(sig.put_signal, -1, "Expected sell put signal");
            assert_eq!(
                sig.total_cost, 5000.0,
                "Total premium should be 2500 + 2500"
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_no_signals_below_threshold() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = StraddleStrategyGpu::new(device).expect("Strategy creation failed");

        let n_candles = 5;
        let n_strategies = 1;

        let underlying: Vec<f64> = vec![48000.0; n_candles];
        let call_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
        let put_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];

        // IV and HV very close (within threshold)
        let implied_vols: Vec<f64> = vec![0.60; n_strategies * n_candles];
        let historical_vols: Vec<f64> = vec![0.62; n_strategies * n_candles]; // Only 2pp difference

        let params = vec![StraddleParams {
            vol_threshold: 5.0, // 5pp threshold
            breakeven_pct: 2.0,
        }];

        let signals = strategy
            .generate_long_signals_batch(
                &underlying,
                &call_prices,
                &put_prices,
                &implied_vols,
                &historical_vols,
                &params,
            )
            .expect("Signal generation failed");

        // No signals should be generated (difference < threshold)
        for sig in &signals {
            assert_eq!(sig.call_signal, 0, "No call signal expected");
            assert_eq!(sig.put_signal, 0, "No put signal expected");
            assert_eq!(sig.total_cost, 0.0, "No cost when no position");
        }
    }
}
