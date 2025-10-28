//! GPU-Accelerated RSI (Relative Strength Index) - Synchronous CPU-GPU Hybrid
//!
//! This file contains the original synchronous version of the RSI calculation
//! for benchmarking purposes. It uses blocking memory transfers.

use super::device::{GpuDevice, GpuError};
use crate::gpu::compile::compile_ptx_optimized_cached;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array1;
use std::sync::Arc;

const RSI_KERNEL: &str = r#"
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void calculate_gains_losses_kernel(
    const double* __restrict__ close,
    double* __restrict__ gains,
    double* __restrict__ losses,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n - 1) return;
    double delta = close[idx + 1] - close[idx];
    gains[idx + 1] = fmax(delta, 0.0);
    losses[idx + 1] = fmax(-delta, 0.0);
}

extern "C" __global__ void calculate_rsi_kernel(
    const double* __restrict__ avg_gain,
    const double* __restrict__ avg_loss,
    double* __restrict__ rsi,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (idx < period) {
        rsi[idx] = CUDART_NAN;
        return;
    }
    double gain = avg_gain[idx];
    double loss = avg_loss[idx];
    if (loss < 1e-10) {
        rsi[idx] = 100.0;
        return;
    }
    double rs = gain / loss;
    rsi[idx] = 100.0 - (100.0 / (1.0 + rs));
}
"#;

pub fn rsi_gpu_sync(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let n = close.len();
    if period < 1 {
        return Err(GpuError::InvalidParameter(
            "Period must be >= 1".to_string(),
        ));
    }
    if n < period + 1 {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need {} points, got {}",
            period + 1,
            n
        )));
    }

    let ptx_arc = compile_ptx_optimized_cached(RSI_KERNEL).map_err(|e| {
        GpuError::CompilationError(format!("Failed to compile RSI kernel: {:?}", e))
    })?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;
    let gains_losses_kernel = module
        .load_function("calculate_gains_losses_kernel")
        .map_err(|e| {
            GpuError::ExecutionError(format!("Failed to load gains_losses kernel: {:?}", e))
        })?;
    let rsi_kernel = module
        .load_function("calculate_rsi_kernel")
        .map_err(|e| GpuError::ExecutionError(format!("Failed to load RSI kernel: {:?}", e)))?;

    let kernel_stream = stream.unwrap_or(&device.stream);

    let d_close = device.copy_to_device(close.as_slice().unwrap())?;
    let mut d_gains = device.alloc_buffer(n)?;
    let mut d_losses = device.alloc_buffer(n)?;

    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut builder = kernel_stream.launch_builder(&gains_losses_kernel);
    builder.arg(&d_close);
    builder.arg(&mut d_gains);
    builder.arg(&mut d_losses);
    builder.arg(&n_i32);
    let config = LaunchConfig::for_num_elems((n - 1) as u32);
    unsafe {
        builder.launch(config).map_err(|e| {
            GpuError::ExecutionError(format!("Gains/losses kernel launch failed: {:?}", e))
        })?;
    }

    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream sync after gains/losses failed: {:?}", e))
    })?;

    let gains_vec = device.copy_to_host(&d_gains)?;
    let losses_vec = device.copy_to_host(&d_losses)?;
    let gains = Array1::from_vec(gains_vec);
    let losses = Array1::from_vec(losses_vec);

    use crate::cpu::sequential::wilders_smoothing_cpu;
    let avg_gain = wilders_smoothing_cpu(&gains, period)?;
    let avg_loss = wilders_smoothing_cpu(&losses, period)?;

    let d_avg_gain = device.copy_to_device(avg_gain.as_slice().unwrap())?;
    let d_avg_loss = device.copy_to_device(avg_loss.as_slice().unwrap())?;
    let mut d_rsi = device.alloc_buffer(n)?;

    let mut builder = kernel_stream.launch_builder(&rsi_kernel);
    builder.arg(&d_avg_gain);
    builder.arg(&d_avg_loss);
    builder.arg(&mut d_rsi);
    builder.arg(&n_i32);
    builder.arg(&period_i32);
    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("RSI kernel launch failed: {:?}", e)))?;
    }

    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream sync after RSI failed: {:?}", e))
    })?;

    let rsi_vec = device.copy_to_host(&d_rsi)?;
    Ok(Array1::from_vec(rsi_vec))
}
