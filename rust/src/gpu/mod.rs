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
pub mod async_alloc;

#[cfg(feature = "gpu")]
pub mod memory_pool;

#[cfg(feature = "gpu")]
pub mod streams;

#[cfg(feature = "gpu")]
pub mod async_transfers;

#[cfg(feature = "gpu")]
pub mod triple_buffer;

#[cfg(feature = "gpu")]
pub mod compile;

#[cfg(feature = "gpu")]
pub mod l2_cache;

// TODO: Fix cudarc API compatibility issues before re-enabling
// #[cfg(feature = "gpu")]
// pub mod aggregation;
//
// #[cfg(feature = "gpu")]
// pub mod auto_select;
//
// #[cfg(feature = "gpu")]
// pub use aggregation::GpuAggregator;
//
// #[cfg(feature = "gpu")]
// pub use auto_select::{AggregationEngine, EngineSelector};

#[cfg(feature = "gpu")]
pub use l2_cache::{
    AccessProperty, L2CachePolicy, calculate_l2_chunk_size, clear_l2_persist_policy,
    set_l2_persist_policy,
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
pub use async_alloc::{AsyncAllocator, PoolStats};

#[cfg(feature = "gpu")]
pub use memory_pool::{GpuMemoryPool, IndicatorType};

#[cfg(feature = "gpu")]
pub use streams::{IndicatorSpeed, StreamManager};

#[cfg(feature = "gpu")]
pub use async_transfers::{AsyncTransferExt, CudaEvent};

#[cfg(feature = "gpu")]
pub use triple_buffer::TripleBufferedExecutor;

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
pub mod rsi_sync;

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

#[cfg(feature = "gpu")]
pub mod mfi;

#[cfg(feature = "gpu")]
pub use mfi::mfi_gpu;

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
pub use persistent::{
    AroonBatch,
    AroonIndicator,
    AtrBatch,
    AtrIndicator,
    BollingerBatch,
    BollingerIndicator,
    BollingerParams,
    CciBatch,
    CciIndicator,
    CmfBatch,
    CmfIndicator,
    // Agent 3 indicators
    DonchianBatch,
    DonchianIndicator,
    // Agent 4 indicators
    ElderRayBatch,
    ElderRayIndicator,
    EmaBatch,
    EmaIndicator,
    GenericBatch,
    KeltnerBatch,
    KeltnerIndicator,
    KeltnerParams,
    MacdBatch,
    MacdIndicator,
    MacdParams,
    ObvBatch,
    ObvIndicator,
    PersistentIndicator,
    PersistentKernelManager,
    RocBatch,
    RocIndicator,
    RsiBatch,
    RsiIndicator,
    // Agent 1 indicators
    SmaBatch,
    SmaIndicator,
    // Agent 2 indicators
    StochasticBatch,
    StochasticIndicator,
    StochasticParams,
    Task,
    TaskBatch,
    VwmaBatch,
    VwmaIndicator,
    WilliamsRBatch,
    WilliamsRIndicator,
    WmaBatch,
    WmaIndicator,
    execute_batch,
    execute_generic_batch,
};

#[cfg(feature = "gpu")]
pub mod cuda_graphs;

#[cfg(feature = "gpu")]
pub use cuda_graphs::{IndicatorGraph, IndicatorGraphBuilder};

#[cfg(feature = "gpu")]
pub mod kernels_3d;

#[cfg(feature = "gpu")]
pub use kernels_3d::{SweepResult3D, rsi_sweep_3d_gpu, sharpe_reduction_gpu, sma_sweep_3d_gpu};

#[cfg(feature = "gpu")]
pub mod candles;

#[cfg(feature = "gpu")]
pub use candles::{
    CandleAggregator, OHLCVCandle, RangeBarAggregator, RangeBarParams, RenkoAggregator,
    RenkoParams, TradeData,
};

#[cfg(feature = "gpu")]
pub mod parabolic_sar;

#[cfg(feature = "gpu")]
pub use parabolic_sar::parabolic_sar_gpu;

#[cfg(feature = "gpu")]
pub mod pivot_points;

#[cfg(feature = "gpu")]
pub use pivot_points::{PivotPointsOutput, pivot_points_gpu};

#[cfg(feature = "gpu")]
pub mod adx;

#[cfg(feature = "gpu")]
pub use adx::adx_gpu;

#[cfg(feature = "gpu")]
pub mod supertrend;

#[cfg(feature = "gpu")]
pub use supertrend::supertrend_gpu;

#[cfg(feature = "gpu")]
pub mod vwap_anchored;

#[cfg(feature = "gpu")]
pub use vwap_anchored::vwap_anchored_gpu;

#[cfg(feature = "gpu")]
pub mod fibonacci;

#[cfg(feature = "gpu")]
pub use fibonacci::{FibonacciOutput, fibonacci_gpu};

#[cfg(feature = "gpu")]
pub mod ichimoku;

#[cfg(feature = "gpu")]
pub use ichimoku::{IchimokuOutput, ichimoku_gpu};
