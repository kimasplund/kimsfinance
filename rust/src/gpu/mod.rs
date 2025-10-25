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
pub mod stochastic;

#[cfg(feature = "gpu")]
pub use device::GpuDevice;

#[cfg(feature = "gpu")]
pub use stochastic::stochastic_gpu;
