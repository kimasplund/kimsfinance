//! Persistent EMA (Exponential Moving Average) kernel
//!
//! Implements EMA calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! ```text
//! alpha = 2 / (period + 1)
//! EMA[0..period-1] = NaN
//! EMA[period-1] = SMA(close[0..period])
//! EMA[i] = alpha * close[i] + (1-alpha) * EMA[i-1]  // Sequential dependency
//! ```
//!
//! # Performance Note
//!
//! EMA is a sequential IIR filter with data dependencies. Unlike parallel indicators,
//! this kernel uses one thread per task to handle sequential computation. However,
//! the persistent kernel pattern still provides benefits:
//! - Single kernel launch for multiple tasks (reduces overhead)
//! - Batch processing amortizes PCIe transfer costs
//! - Grid synchronization ensures correctness between tasks

use super::super::traits::{PersistentIndicator, SingleOutputIndicator};

/// EMA indicator for persistent kernel execution
pub struct EmaIndicator;

/// CUDA kernel for persistent EMA calculation
///
/// Uses single thread per task due to sequential data dependency.
/// Each task processes its data sequentially, then synchronizes before next task.
const EMA_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_ema_kernel(
    const double** __restrict__ input_batch,    // Array of input pointers (close prices)
    double** __restrict__ output_batch,          // Array of output pointers (EMA values)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of EMA periods
    int num_tasks                                // Number of tasks to process
) {
    // Get grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Process each task sequentially (persistent kernel pattern)
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* close = input_batch[task_id];
        double* ema = output_batch[task_id];
        int n = sizes[task_id];
        int period = periods[task_id];

        // Sequential calculation (one thread per task due to data dependency)
        // Use thread assignment based on task_id to distribute work across grid
        if (global_tid == task_id % grid_size) {
            // Calculate alpha (exponential smoothing factor)
            double alpha = 2.0 / (period + 1.0);
            double one_minus_alpha = 1.0 - alpha;

            // First period-1 values are NaN (not enough data for EMA)
            for (int i = 0; i < period - 1; i++) {
                ema[i] = CUDART_NAN;
            }

            // Calculate initial EMA as SMA of first `period` values
            double sum = 0.0;
            for (int i = 0; i < period; i++) {
                sum += close[i];
            }
            ema[period - 1] = sum / (double)period;

            // Apply exponential smoothing for remaining values
            // EMA[i] = alpha * close[i] + (1 - alpha) * EMA[i-1]
            for (int i = period; i < n; i++) {
                ema[i] = alpha * close[i] + one_minus_alpha * ema[i - 1];
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for EmaIndicator {
    type Params = i32; // EMA period

    fn kernel_source() -> &'static str {
        EMA_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_ema_kernel"
    }

    fn num_outputs() -> usize {
        1 // Single output: EMA values
    }
}

impl SingleOutputIndicator for EmaIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_ema_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = EmaIndicator::compile_kernel(&device);
        assert!(result.is_ok(), "EMA kernel should compile successfully");
    }

    #[test]
    fn test_ema_trait_properties() {
        assert_eq!(EmaIndicator::kernel_name(), "persistent_ema_kernel");
        assert_eq!(EmaIndicator::num_inputs(), 1);
        assert_eq!(EmaIndicator::num_outputs(), 1);
    }
}
