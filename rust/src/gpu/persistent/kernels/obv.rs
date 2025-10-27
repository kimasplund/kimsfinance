//! Persistent OBV (On-Balance Volume) kernel
//!
//! Implements OBV calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! OBV is a cumulative momentum indicator:
//! 1. If close[i] > close[i-1]: delta = +volume[i]
//! 2. If close[i] < close[i-1]: delta = -volume[i]
//! 3. If close[i] == close[i-1]: delta = 0
//! 4. OBV[i] = OBV[i-1] + delta (cumulative sum)
//!
//! # Performance
//!
//! OBV has sequential dependencies (cumulative sum), making it less parallelizable
//! than other indicators. However, persistent kernel pattern still reduces launch
//! overhead for batch processing.
//!
//! # Sequential Processing
//!
//! Due to cumulative sum dependency, this kernel uses one thread per task for
//! the sequential calculation portion.

use super::super::traits::{PersistentIndicator, SingleOutputIndicator};

/// OBV indicator for persistent kernel execution
pub struct ObvIndicator;

/// CUDA kernel for persistent OBV calculation
///
/// Input buffer layout: [close(n), volume(n)] - concatenated
const OBV_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_obv_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (close+volume concatenated)
    double** __restrict__ output_batch,          // Array of output pointers (OBV)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const void* __restrict__ dummy_params,       // Unused (OBV has no parameters)
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

        // Split input buffer: [close(n), volume(n)]
        const double* close = input;           // First n elements
        const double* volume = input + n;      // Next n elements

        double* obv = output_batch[task_id];

        // Sequential calculation (one thread per task for dependency handling)
        // Use modulo to assign one thread per task
        if (global_tid == task_id % grid_size) {
            // OBV starts at 0 (no previous price to compare)
            obv[0] = 0.0;

            // Use small epsilon for floating-point comparison tolerance
            const double EPSILON = 1e-10;

            // Calculate OBV sequentially
            for (int i = 1; i < n; i++) {
                double price_change = close[i] - close[i - 1];
                double delta = 0.0;

                if (price_change > EPSILON) {
                    // Price up: add volume
                    delta = volume[i];
                } else if (price_change < -EPSILON) {
                    // Price down: subtract volume
                    delta = -volume[i];
                }
                // else: price unchanged, delta = 0

                // Cumulative sum
                obv[i] = obv[i - 1] + delta;
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for ObvIndicator {
    type Params = (); // No parameters for OBV

    fn kernel_source() -> &'static str {
        OBV_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_obv_kernel"
    }

    fn num_inputs() -> usize {
        2 // Two inputs: close, volume
    }

    fn num_outputs() -> usize {
        1 // Single output: OBV values
    }
}

impl SingleOutputIndicator for ObvIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = ObvIndicator::compile_kernel(&device);
        assert!(result.is_ok(), "OBV kernel should compile successfully");
    }

    #[test]
    fn test_obv_trait_properties() {
        assert_eq!(ObvIndicator::kernel_name(), "persistent_obv_kernel");
        assert_eq!(ObvIndicator::num_inputs(), 2);
        assert_eq!(ObvIndicator::num_outputs(), 1);
    }
}
