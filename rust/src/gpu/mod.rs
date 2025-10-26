//! GPU-Accelerated Indicators
//!
//! Optional GPU acceleration using NVIDIA CUDA via cudarc.
//! Provides 15-50x speedup for large datasets (>10K rows).
//!
//! # Architecture
//!
//! - **Device Management**: GPU initialization, memory pools, error handling
//! - **CUDA Kernels**: Custom kernels compiled from CUDA C++ source
//! - **Indicators**: GPU-accelerated implementations with CPU fallback
//!
//! # Feature Flag
//!
//! GPU support requires the `gpu` feature:
//! ```toml
//! kimsfinance_core = { version = "0.1.0", features = ["gpu"] }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! #[cfg(feature = "gpu")]
//! use kimsfinance_core::gpu::{GpuDevice, stochastic_gpu};
//!
//! #[cfg(feature = "gpu")]
//! {
//!     let device = GpuDevice::new()?;
//!     let result = stochastic_gpu(&device, high, low, close, k_period, d_period)?;
//! }
//! ```

#[cfg(feature = "gpu")]
pub mod device;

#[cfg(feature = "gpu")]
pub mod memory_pool;

#[cfg(feature = "gpu")]
pub mod streams;

#[cfg(feature = "gpu")]
pub mod compile;

#[cfg(feature = "gpu")]
pub mod l2_cache;

#[cfg(feature = "gpu")]
pub use l2_cache::{
    calculate_l2_chunk_size, set_l2_persist_policy, clear_l2_persist_policy,
    L2CachePolicy, AccessProperty,
};

#[cfg(feature = "gpu")]
pub mod stochastic;

#[cfg(feature = "gpu")]
pub mod roc;

#[cfg(feature = "gpu")]
pub mod williams_r;

#[cfg(feature = "gpu")]
pub mod bollinger;

#[cfg(feature = "gpu")]
pub mod aroon;

#[cfg(feature = "gpu")]
pub mod atr;

#[cfg(feature = "gpu")]
pub mod cci;

#[cfg(feature = "gpu")]
pub mod keltner;

#[cfg(feature = "gpu")]
pub use device::{GpuDevice, GpuError};

#[cfg(feature = "gpu")]
pub use memory_pool::{GpuMemoryPool, IndicatorType};

#[cfg(feature = "gpu")]
pub use streams::{IndicatorSpeed, StreamManager};

#[cfg(feature = "gpu")]
pub use stochastic::stochastic_gpu;

#[cfg(feature = "gpu")]
pub use roc::roc_gpu;

#[cfg(feature = "gpu")]
pub use williams_r::williams_r_gpu;

#[cfg(feature = "gpu")]
pub use bollinger::bollinger_bands_gpu;

#[cfg(feature = "gpu")]
pub use aroon::aroon_gpu;

#[cfg(feature = "gpu")]
pub use atr::atr_gpu;

#[cfg(feature = "gpu")]
pub use cci::cci_gpu;

#[cfg(feature = "gpu")]
pub use keltner::keltner_channels_gpu;

#[cfg(feature = "gpu")]
pub mod rsi;

#[cfg(feature = "gpu")]
pub use rsi::rsi_gpu;

#[cfg(feature = "gpu")]
pub mod macd;

#[cfg(feature = "gpu")]
pub use macd::macd_gpu;

#[cfg(feature = "gpu")]
pub mod donchian;

#[cfg(feature = "gpu")]
pub use donchian::donchian_gpu;

#[cfg(feature = "gpu")]
pub mod sma;

#[cfg(feature = "gpu")]
pub use sma::{sma_gpu, sma_gpu_shared};

#[cfg(feature = "gpu")]
pub mod wma;

#[cfg(feature = "gpu")]
pub use wma::wma_gpu;

#[cfg(feature = "gpu")]
pub mod elder_ray;

#[cfg(feature = "gpu")]
pub use elder_ray::elder_ray_gpu;
#[cfg(feature = "gpu")]
pub mod ema;

#[cfg(feature = "gpu")]
pub use ema::ema_gpu;

#[cfg(feature = "gpu")]
pub mod batch;

#[cfg(feature = "gpu")]
pub use batch::{
    BatchIndicatorParams, BatchIndicatorType, IndicatorRequest, IndicatorResult,
    calculate_indicator_gpu, calculate_indicators_batch_gpu,
};

#[cfg(feature = "gpu")]
pub mod obv;

#[cfg(feature = "gpu")]
pub use obv::obv_gpu;

#[cfg(feature = "gpu")]
pub mod cmf;

#[cfg(feature = "gpu")]
pub use cmf::cmf_gpu;

#[cfg(feature = "gpu")]
pub mod vwma;

#[cfg(feature = "gpu")]
pub use vwma::vwma_gpu;

// TODO: Fix sweep module conflicts before re-enabling
// #[cfg(feature = "gpu")]
// pub mod sweep;
//
// #[cfg(feature = "gpu")]
// pub use sweep::{
//     IndicatorData, IndicatorType, OptimizationMetric, OptimalParameter, ParameterSweep,
//     SweepBatch, SweepResult,
// };

#[cfg(feature = "gpu")]
pub mod persistent;

#[cfg(feature = "gpu")]
pub use persistent::{PersistentKernelManager, TaskBatch};

#[cfg(feature = "gpu")]
pub mod cuda_graphs;

#[cfg(feature = "gpu")]
pub use cuda_graphs::{IndicatorGraph, IndicatorGraphBuilder};

#[cfg(feature = "gpu")]
pub mod kernels_3d;

#[cfg(feature = "gpu")]
pub use kernels_3d::{rsi_sweep_3d_gpu, sma_sweep_3d_gpu, sharpe_reduction_gpu, SweepResult3D};
