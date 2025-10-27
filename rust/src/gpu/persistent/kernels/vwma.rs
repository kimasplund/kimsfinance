//! Persistent VWMA (Volume-Weighted Moving Average) kernel
//!
//! Implements VWMA calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! VWMA weights prices by their trading volume:
//! ```text
//! VWMA[i] = sum(close[j] * volume[j] for j in [i-period+1..=i])
//!           / sum(volume[j] for j in [i-period+1..=i])
//! ```
//!
//! # Performance
//!
//! This is one of the fastest GPU indicators due to perfect parallelism:
//! - No rolling dependencies between windows
//! - No shared memory needed
//! - No thread synchronization required (beyond task boundaries)
//! - Each thread operates completely independently
//!
//! Expected speedup: 30-50x over CPU for large datasets.

use super::super::traits::{PersistentIndicator, SingleOutputIndicator};

/// VWMA indicator for persistent kernel execution
pub struct VwmaIndicator;

/// CUDA kernel for persistent VWMA calculation
///
/// Input buffer layout: [close(n), volume(n)] - concatenated
const VWMA_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_vwma_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (close+volume concatenated)
    double** __restrict__ output_batch,          // Array of output pointers (VWMA)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of VWMA periods
    int num_tasks                                // Number of tasks to process
) {
    // Get grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Process each task sequentially (persistent kernel pattern)
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* input = input_batch[task_id];
        int n = sizes[task_id];
        int period = periods[task_id];

        // Split input buffer: [close(n), volume(n)]
        const double* close = input;           // First n elements
        const double* volume = input + n;      // Next n elements

        double* vwma = output_batch[task_id];

        // Grid-stride loop for this task's data
        for (int idx = global_tid; idx < n; idx += grid_size) {
            if (idx < period - 1) {
                // Not enough history - set to NAN
                vwma[idx] = CUDART_NAN;
            } else {
                // Calculate VWMA for this index
                double weighted_sum = 0.0;
                double volume_sum = 0.0;

                // Calculate sum(close * volume) and sum(volume) for the window
                for (int j = 0; j < period; j++) {
                    int pos = idx - period + 1 + j;
                    double vol = volume[pos];
                    weighted_sum += close[pos] * vol;
                    volume_sum += vol;
                }

                // Handle division by zero (no volume in window)
                if (volume_sum > 1e-10) {
                    vwma[idx] = weighted_sum / volume_sum;
                } else {
                    vwma[idx] = CUDART_NAN;
                }
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for VwmaIndicator {
    type Params = i32; // VWMA period (typically 14-20)

    fn kernel_source() -> &'static str {
        VWMA_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_vwma_kernel"
    }

    fn num_inputs() -> usize {
        2 // Two inputs: close, volume
    }

    fn num_outputs() -> usize {
        1 // Single output: VWMA values
    }
}

impl SingleOutputIndicator for VwmaIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_vwma_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = VwmaIndicator::compile_kernel(&device);
        assert!(result.is_ok(), "VWMA kernel should compile successfully");
    }

    #[test]
    fn test_vwma_trait_properties() {
        assert_eq!(VwmaIndicator::kernel_name(), "persistent_vwma_kernel");
        assert_eq!(VwmaIndicator::num_inputs(), 2);
        assert_eq!(VwmaIndicator::num_outputs(), 1);
    }
}
