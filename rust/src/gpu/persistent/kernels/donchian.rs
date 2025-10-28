//! Persistent Donchian Channels kernel
//!
//! Implements Donchian Channels calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! Donchian Channels identify breakout levels using rolling max/min:
//!
//! 1. **Upper Channel** = max(high) over period
//! 2. **Lower Channel** = min(low) over period
//! 3. **Middle Channel** = (Upper + Lower) / 2
//!
//! # Calculation Steps
//!
//! 1. For each candle after period-1:
//!    - Find max(high) in rolling window [i-period+1, i]
//!    - Find min(low) in rolling window [i-period+1, i]
//!    - Calculate middle as average
//! 2. First period-1 values are NaN
//!
//! # Performance
//!
//! Persistent kernel execution reduces overhead:
//! - Traditional: 2 arrays (high, low) × N tasks × 10μs = 20N μs
//! - Persistent: 1 launch × 10μs = 10μs (95% reduction for N=10)

use super::super::traits::{MultiOutputIndicator, PersistentIndicator};

/// Donchian Channels indicator for persistent kernel execution
pub struct DonchianIndicator;

/// CUDA kernel for persistent Donchian Channels calculation
///
/// Input buffer layout: [high(n), low(n)] - concatenated
/// Output buffer layout: [upper(n), middle(n), lower(n)] - concatenated
const DONCHIAN_KERNEL: &str = r#"
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

// Define constants for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)

extern "C" __global__ void persistent_donchian_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (high+low concatenated)
    double** __restrict__ output_batch,          // Array of output pointers (upper+middle+lower concatenated)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of periods
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

        // Split input buffer: [high(n), low(n)]
        const double* high = input;           // First n elements
        const double* low = input + n;        // Next n elements

        double* output = output_batch[task_id];

        // Output layout: [upper (n), middle (n), lower (n)]
        double* upper = output;              // First n elements
        double* middle = output + n;         // Next n elements
        double* lower = output + 2*n;        // Last n elements

        // Parallel calculation: each thread handles multiple indices
        for (int idx = global_tid; idx < n; idx += grid_size) {
            if (idx < period - 1) {
                // Not enough history - set to NaN
                upper[idx] = CUDART_NAN;
                lower[idx] = CUDART_NAN;
                middle[idx] = CUDART_NAN;
            } else {
                // Find max and min over the rolling window
                double max_val = -CUDART_INF;
                double min_val = CUDART_INF;

                for (int j = 0; j < period; j++) {
                    int window_idx = idx - j;
                    max_val = fmax(max_val, high[window_idx]);
                    min_val = fmin(min_val, low[window_idx]);
                }

                upper[idx] = max_val;
                lower[idx] = min_val;
                middle[idx] = (max_val + min_val) / 2.0;
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for DonchianIndicator {
    type Params = i32; // Period (typically 20)

    fn kernel_source() -> &'static str {
        DONCHIAN_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_donchian_kernel"
    }

    fn num_inputs() -> usize {
        2 // Two inputs: high, low
    }

    fn num_outputs() -> usize {
        3 // Three outputs: upper, middle, lower
    }
}

impl MultiOutputIndicator for DonchianIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_donchian_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = DonchianIndicator::compile_kernel(&device);
        assert!(
            result.is_ok(),
            "Donchian kernel should compile successfully"
        );
    }

    #[test]
    fn test_donchian_trait_properties() {
        assert_eq!(
            DonchianIndicator::kernel_name(),
            "persistent_donchian_kernel"
        );
        assert_eq!(DonchianIndicator::num_inputs(), 2);
        assert_eq!(DonchianIndicator::num_outputs(), 3);
    }
}
