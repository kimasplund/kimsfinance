//! Persistent SMA (Simple Moving Average) kernel
//!
//! Implements SMA calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! SMA[i] = sum(close[i-period+1..=i]) / period
//!
//! This is an embarrassingly parallel problem - each thread calculates one SMA value
//! independently by summing the last `period` values.
//!
//! # Performance
//!
//! SMA is perfectly parallelizable, making it ideal for persistent kernel execution.
//! Each task can utilize full GPU parallelism across all data points.

use super::super::traits::{PersistentIndicator, SingleOutputIndicator};

/// SMA indicator for persistent kernel execution
pub struct SmaIndicator;

/// CUDA kernel for persistent SMA calculation
///
/// Each thread processes one data point per task. Grid-stride loop ensures
/// efficient processing of variable-length datasets.
const SMA_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_sma_kernel(
    const double** __restrict__ input_batch,    // Array of input pointers (close prices)
    double** __restrict__ output_batch,          // Array of output pointers (SMA values)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of SMA periods
    int num_tasks                                // Number of tasks to process
) {
    // Get grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Process each task sequentially (persistent kernel pattern)
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* close = input_batch[task_id];
        double* sma = output_batch[task_id];
        int n = sizes[task_id];
        int period = periods[task_id];

        // Grid-stride loop for this task's data (parallel across all threads)
        for (int idx = global_tid; idx < n; idx += grid_size) {
            if (idx < period - 1) {
                // Not enough data for SMA
                sma[idx] = CUDART_NAN;
            } else {
                // Calculate SMA: sum last `period` values and divide
                double sum = 0.0;
                for (int j = 0; j < period; j++) {
                    sum += close[idx - j];
                }
                sma[idx] = sum / (double)period;
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for SmaIndicator {
    type Params = i32; // SMA period

    fn kernel_source() -> &'static str {
        SMA_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_sma_kernel"
    }

    fn num_outputs() -> usize {
        1 // Single output: SMA values
    }
}

impl SingleOutputIndicator for SmaIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = SmaIndicator::compile_kernel(&device);
        assert!(result.is_ok(), "SMA kernel should compile successfully");
    }

    #[test]
    fn test_sma_trait_properties() {
        assert_eq!(SmaIndicator::kernel_name(), "persistent_sma_kernel");
        assert_eq!(SmaIndicator::num_inputs(), 1);
        assert_eq!(SmaIndicator::num_outputs(), 1);
    }
}
