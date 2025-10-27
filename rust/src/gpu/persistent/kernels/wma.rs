//! Persistent WMA (Weighted Moving Average) kernel
//!
//! Implements WMA calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! WMA assigns higher weight to more recent values in a linear fashion:
//! ```text
//! weights = [1, 2, 3, ..., period]
//! WMA[i] = sum(close[i-period+1..=i] * weights) / sum(weights)
//!        = sum(close[i-period+1..=i] * weights) / (period * (period + 1) / 2)
//! ```
//!
//! More recent prices have higher weight:
//! - Most recent: weight = period
//! - Oldest: weight = 1
//!
//! # Performance
//!
//! This is a FAST indicator with good parallelism:
//! - Simple rolling window
//! - Minimal branching
//! - Sequential memory access pattern
//! - Each thread operates independently
//!
//! Expected speedup: 35-55x over CPU for large datasets.

use super::super::traits::{PersistentIndicator, SingleOutputIndicator};

/// WMA indicator for persistent kernel execution
pub struct WmaIndicator;

/// CUDA kernel for persistent WMA calculation
const WMA_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_wma_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (close prices)
    double** __restrict__ output_batch,          // Array of output pointers (WMA)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of WMA periods
    int num_tasks                                // Number of tasks to process
) {
    // Get grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Process each task sequentially (persistent kernel pattern)
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* close = input_batch[task_id];  // Single input: close prices
        double* wma = output_batch[task_id];
        int n = sizes[task_id];
        int period = periods[task_id];

        // Grid-stride loop for this task's data
        for (int idx = global_tid; idx < n; idx += grid_size) {
            if (idx < period - 1) {
                // Not enough history - set to NAN
                wma[idx] = CUDART_NAN;
            } else {
                // Calculate WMA for this index
                double weighted_sum = 0.0;

                // Calculate weighted sum with linear weights
                // Weight scheme: most recent value gets 'period' weight,
                // oldest value gets 1 weight
                for (int j = 0; j < period; j++) {
                    int weight = period - j;  // Decreasing weights: period, period-1, ..., 2, 1
                    weighted_sum += close[idx - j] * weight;
                }

                // Denominator is sum of arithmetic series: period * (period + 1) / 2
                int weight_sum = period * (period + 1) / 2;
                wma[idx] = weighted_sum / weight_sum;
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for WmaIndicator {
    type Params = i32; // WMA period (typically 10, 20, or 50)

    fn kernel_source() -> &'static str {
        WMA_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_wma_kernel"
    }

    fn num_inputs() -> usize {
        1 // Single input: close prices
    }

    fn num_outputs() -> usize {
        1 // Single output: WMA values
    }
}

impl SingleOutputIndicator for WmaIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_wma_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = WmaIndicator::compile_kernel(&device);
        assert!(result.is_ok(), "WMA kernel should compile successfully");
    }

    #[test]
    fn test_wma_trait_properties() {
        assert_eq!(WmaIndicator::kernel_name(), "persistent_wma_kernel");
        assert_eq!(WmaIndicator::num_inputs(), 1);
        assert_eq!(WmaIndicator::num_outputs(), 1);
    }
}
