//! Deprecated alias for the hybrid RSI implementation
//!
//! This module previously contained a verbatim duplicate of the RSI kernel
//! and host flow from `gpu::rsi`. The duplicate has been removed;
//! [`rsi_gpu_sync`] is now a thin, deprecated wrapper around
//! [`crate::gpu::rsi::rsi_gpu`] and produces identical results.

use super::device::{GpuDevice, GpuError};
use cudarc::driver::CudaStream;
use ndarray::Array1;
use std::sync::Arc;

/// GPU-accelerated RSI - DEPRECATED duplicate of `rsi_gpu`
///
/// # DEPRECATED
///
/// `rsi_gpu_sync` was a byte-for-byte duplicate of `rsi_gpu` (same kernel
/// source, same host flow). Call [`crate::gpu::rsi::rsi_gpu`] directly.
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `close` - Closing prices
/// * `period` - RSI period (typically 14)
/// * `stream` - Optional CUDA stream for concurrent execution
///
/// # Returns
///
/// Array1<f64> with RSI values (0-100 range); first `period` values are NaN.
#[deprecated(
    since = "0.2.0",
    note = "rsi_gpu_sync was a verbatim duplicate of rsi_gpu; use kimsfinance_core::gpu::rsi::rsi_gpu instead"
)]
pub fn rsi_gpu_sync(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    super::rsi::rsi_gpu(device, close, period, stream)
}
