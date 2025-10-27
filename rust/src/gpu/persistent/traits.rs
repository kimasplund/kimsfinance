//! Trait system for persistent kernel indicators
//!
//! Defines the interface for indicators that can run in persistent kernel mode,
//! enabling batch processing with minimal launch overhead.

use super::super::device::{GpuDevice, GpuError};
use cudarc::driver::CudaFunction;

/// Core trait for indicators that support persistent kernel execution
///
/// Indicators implementing this trait can be batched together and executed
/// with a single kernel launch, reducing overhead from ~5-10μs per task to
/// a single launch overhead shared across all tasks.
///
/// # Type Parameters
///
/// - `Params`: Parameter type for the indicator (e.g., period for RSI/ROC)
///
/// # Design Philosophy
///
/// This trait uses "interface-first" design from tree-of-thoughts analysis:
/// - Generic over parameter types (extensible)
/// - Static dispatch for zero-cost abstraction
/// - Consistent API across all indicators
pub trait PersistentIndicator: Sized {
    /// Parameter type (must be Copy for efficient GPU transfer)
    type Params: Copy + Send + Sync + std::fmt::Debug;

    /// CUDA kernel source code
    ///
    /// Must include cooperative groups and persistent kernel pattern:
    /// ```cuda
    /// for (int task_id = 0; task_id < num_tasks; task_id++) {
    ///     // Process task
    ///     grid.sync(); // Synchronize before next task
    /// }
    /// ```
    fn kernel_source() -> &'static str;

    /// Kernel function name (must match __global__ function in source)
    fn kernel_name() -> &'static str;

    /// Number of input buffers per task
    ///
    /// - RSI/ROC: 1 input (close prices)
    /// - ATR: 3 inputs (high, low, close)
    /// - MACD: 1 input (close prices)
    fn num_inputs() -> usize {
        1 // Default: single input
    }

    /// Number of output buffers per task
    ///
    /// - RSI/ATR/ROC: 1 output
    /// - MACD: 3 outputs (macd, signal, histogram)
    /// - Bollinger Bands: 3 outputs (upper, middle, lower)
    fn num_outputs() -> usize;

    /// Compile kernel for device
    ///
    /// Default implementation uses optimized compilation from compile.rs
    fn compile_kernel(device: &GpuDevice) -> Result<CudaFunction, GpuError> {
        use super::super::compile::compile_ptx_optimized;

        // Compile PTX with optimizations
        let ptx = compile_ptx_optimized(Self::kernel_source()).map_err(|e| {
            GpuError::CompilationError(format!(
                "Failed to compile {} kernel: {:?}",
                Self::kernel_name(),
                e
            ))
        })?;

        // Load module
        let module = device.context().load_module(ptx).map_err(|e| {
            GpuError::CompilationError(format!("Failed to load PTX module: {:?}", e))
        })?;

        // Load kernel function
        let func = module.load_function(Self::kernel_name()).map_err(|e| {
            GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
        })?;

        Ok(func)
    }
}

/// Marker trait for single-output indicators (RSI, ATR, ROC)
///
/// These indicators produce one output array per input.
pub trait SingleOutputIndicator: PersistentIndicator {}

/// Marker trait for multi-output indicators (MACD, Bollinger Bands)
///
/// These indicators produce multiple output arrays per input.
pub trait MultiOutputIndicator: PersistentIndicator {}
