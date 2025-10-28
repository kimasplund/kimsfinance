//! Persistent ROC (Rate of Change) kernel
//!
//! Implements ROC calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! ROC = ((price[i] - price[i-period]) / price[i-period]) * 100
//!
//! Measures the percentage change in price over a specified period.
//!
//! # Performance
//!
//! This is the simplest indicator, making it ideal for testing the persistent kernel pattern.

use super::super::traits::{PersistentIndicator, SingleOutputIndicator};

/// ROC indicator for persistent kernel execution
pub struct RocIndicator;

/// CUDA kernel for persistent ROC calculation (from existing implementation)
const ROC_KERNEL: &str = r#"
// NVRTC Kernel - Do NOT include system headers
// NVRTC provides built-in CUDA types and functions

// Cooperative Groups API (available in NVRTC without includes)
namespace cooperative_groups {
    struct grid_group {
        __device__ void sync() const {
            __syncthreads();  // Intra-block sync
        }
    };

    __device__ inline grid_group this_grid() {
        return grid_group{};
    }
}
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_roc_kernel(
    const double** __restrict__ input_batch,    // Array of input pointers
    double** __restrict__ output_batch,          // Array of output pointers
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of ROC periods
    int num_tasks                                // Number of tasks to process
) {
    // Get grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Process each task sequentially (persistent kernel pattern)
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* input = input_batch[task_id];
        double* output = output_batch[task_id];
        int n = sizes[task_id];
        int period = periods[task_id];

        // Grid-stride loop for this task's data
        for (int idx = global_tid; idx < n; idx += grid_size) {
            if (idx < period) {
                output[idx] = CUDART_NAN;
            } else {
                // ROC = (price[i] / price[i-period] - 1) * 100
                output[idx] = (input[idx] / input[idx - period] - 1.0) * 100.0;
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for RocIndicator {
    type Params = i32; // ROC period

    fn kernel_source() -> &'static str {
        ROC_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_roc_kernel"
    }

    fn num_outputs() -> usize {
        1 // Single output: ROC values
    }
}

impl SingleOutputIndicator for RocIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_roc_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = RocIndicator::compile_kernel(&device);
        assert!(result.is_ok(), "ROC kernel should compile successfully");
    }

    #[test]
    fn test_roc_trait_properties() {
        assert_eq!(RocIndicator::kernel_name(), "persistent_roc_kernel");
        assert_eq!(RocIndicator::num_inputs(), 1);
        assert_eq!(RocIndicator::num_outputs(), 1);
    }
}
