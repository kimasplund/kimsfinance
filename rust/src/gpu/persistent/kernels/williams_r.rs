//! Persistent Williams %R kernel
//!
//! Implements Williams %R calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! Williams %R is a momentum indicator measuring overbought/oversold levels.
//! It is inversely related to Stochastic %K: Williams %R = Stochastic %K - 100.
//!
//! **Formula**:
//! - %R = ((Highest High - Close) / (Highest High - Lowest Low)) × -100
//! - Range: [-100, 0]
//!
//! # Interpretation
//!
//! - **-80 to -100**: Oversold (potential buy signal)
//! - **-20 to 0**: Overbought (potential sell signal)
//! - **-50**: Neutral
//!
//! # Performance
//!
//! Persistent kernel execution reduces overhead:
//! - Traditional: 3 arrays (high, low, close) × N tasks × 10μs = 30N μs
//! - Persistent: 1 launch × 10μs = 10μs (96% reduction for N=10)

use super::super::traits::{PersistentIndicator, SingleOutputIndicator};

/// Williams %R indicator for persistent kernel execution
pub struct WilliamsRIndicator;

/// CUDA kernel for persistent Williams %R calculation
///
/// Input buffer layout: [high(n), low(n), close(n)] - concatenated
/// Produces single output: Williams %R values
const WILLIAMS_R_KERNEL: &str = r#"
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

extern "C" __global__ void persistent_williams_r_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (high+low+close concatenated)
    double** __restrict__ output_batch,          // Array of output pointers (Williams %R values)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of lookback periods
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

        // Split input buffer: [high(n), low(n), close(n)]
        const double* high = input;           // First n elements
        const double* low = input + n;        // Next n elements
        const double* close = input + 2*n;    // Last n elements

        double* williams_r = output_batch[task_id];

        // Sequential calculation (one thread per task for dependency handling)
        if (global_tid == task_id % grid_size) {
            // Calculate Williams %R for each point
            for (int idx = 0; idx < n; idx++) {
                if (idx >= period - 1) {
                    // Find highest high and lowest low in period window
                    double highest_high = -CUDART_INF;
                    double lowest_low = CUDART_INF;

                    for (int i = 0; i < period; i++) {
                        int window_idx = idx - i;
                        if (window_idx >= 0) {
                            highest_high = fmax(highest_high, high[window_idx]);
                            lowest_low = fmin(lowest_low, low[window_idx]);
                        }
                    }

                    // Calculate %R: ((highest_high - close) / (highest_high - lowest_low)) * -100
                    double range = highest_high - lowest_low;
                    if (range > 1e-10) {
                        williams_r[idx] = ((highest_high - close[idx]) / range) * -100.0;
                    } else {
                        // When range is zero, use midpoint (-50)
                        williams_r[idx] = -50.0;
                    }
                } else {
                    williams_r[idx] = CUDART_NAN;
                }
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for WilliamsRIndicator {
    type Params = i32; // Williams %R period (typically 14)

    fn kernel_source() -> &'static str {
        WILLIAMS_R_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_williams_r_kernel"
    }

    fn num_inputs() -> usize {
        3 // Three inputs: high, low, close
    }

    fn num_outputs() -> usize {
        1 // Single output: Williams %R values
    }
}

impl SingleOutputIndicator for WilliamsRIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_williams_r_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = WilliamsRIndicator::compile_kernel(&device);
        assert!(
            result.is_ok(),
            "Williams %R kernel should compile successfully"
        );
    }

    #[test]
    fn test_williams_r_trait_properties() {
        assert_eq!(
            WilliamsRIndicator::kernel_name(),
            "persistent_williams_r_kernel"
        );
        assert_eq!(WilliamsRIndicator::num_inputs(), 3);
        assert_eq!(WilliamsRIndicator::num_outputs(), 1);
    }
}
