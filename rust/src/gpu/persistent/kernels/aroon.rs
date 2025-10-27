//! Persistent Aroon indicator kernel
//!
//! Implements Aroon Up/Down calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! Aroon measures the time elapsed since the highest high and lowest low
//! within a given period, expressed as a percentage (0-100).
//!
//! 1. **Aroon Up** = ((period - periods_since_highest_high) / period) × 100
//! 2. **Aroon Down** = ((period - periods_since_lowest_low) / period) × 100
//! 3. **Aroon Oscillator** = Aroon Up - Aroon Down
//!
//! # Calculation Steps
//!
//! For each candle after period-1:
//! 1. Find index of highest high in rolling window [i-period+1, i]
//! 2. Find index of lowest low in rolling window [i-period+1, i]
//! 3. Calculate periods since each extreme
//! 4. Convert to percentage (0-100 scale)
//! 5. Calculate oscillator as difference
//!
//! # Performance
//!
//! Persistent kernel execution reduces overhead:
//! - Traditional: 2 arrays (high, low) × N tasks × 10μs = 20N μs
//! - Persistent: 1 launch × 10μs = 10μs (95% reduction for N=10)

use super::super::traits::{MultiOutputIndicator, PersistentIndicator};

/// Aroon indicator for persistent kernel execution
pub struct AroonIndicator;

/// CUDA kernel for persistent Aroon calculation
///
/// Requires two input arrays: high, low
/// Produces three outputs: aroon_up, aroon_down, oscillator (stored contiguously)
const AROON_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define constants for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_aroon_kernel(
    const double** __restrict__ high_batch,      // Array of high price pointers
    const double** __restrict__ low_batch,       // Array of low price pointers
    double** __restrict__ output_batch,          // Array of output pointers (up+down+oscillator concatenated)
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
        const double* high = high_batch[task_id];
        const double* low = low_batch[task_id];
        double* output = output_batch[task_id];
        int n = sizes[task_id];
        int period = periods[task_id];

        // Output layout: [aroon_up (n), aroon_down (n), oscillator (n)]
        double* aroon_up = output;           // First n elements
        double* aroon_down = output + n;     // Next n elements
        double* oscillator = output + 2*n;   // Last n elements

        // Parallel calculation: each thread handles multiple indices
        for (int idx = global_tid; idx < n; idx += grid_size) {
            if (idx < period - 1) {
                // Not enough history - set to NaN
                aroon_up[idx] = CUDART_NAN;
                aroon_down[idx] = CUDART_NAN;
                oscillator[idx] = CUDART_NAN;
            } else {
                // Find position of highest high and lowest low in rolling window
                // Window: [idx - period + 1, idx]
                int highest_high_idx = idx;
                int lowest_low_idx = idx;
                double highest_high = high[idx];
                double lowest_low = low[idx];

                // Scan backward through the window
                for (int i = 1; i < period; i++) {
                    int window_idx = idx - i;

                    if (high[window_idx] >= highest_high) {
                        highest_high = high[window_idx];
                        highest_high_idx = window_idx;
                    }

                    if (low[window_idx] <= lowest_low) {
                        lowest_low = low[window_idx];
                        lowest_low_idx = window_idx;
                    }
                }

                // Calculate periods since high/low
                int periods_since_high = idx - highest_high_idx;
                int periods_since_low = idx - lowest_low_idx;

                // Calculate Aroon values
                // Aroon = ((period - periods_since) / period) × 100
                double up_val = ((double)(period - periods_since_high) / (double)period) * 100.0;
                double down_val = ((double)(period - periods_since_low) / (double)period) * 100.0;

                aroon_up[idx] = up_val;
                aroon_down[idx] = down_val;
                oscillator[idx] = up_val - down_val;
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for AroonIndicator {
    type Params = i32; // Period (typically 14 or 25)

    fn kernel_source() -> &'static str {
        AROON_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_aroon_kernel"
    }

    fn num_inputs() -> usize {
        2 // Two inputs: high, low
    }

    fn num_outputs() -> usize {
        3 // Three outputs: aroon_up, aroon_down, oscillator
    }
}

impl MultiOutputIndicator for AroonIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_aroon_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = AroonIndicator::compile_kernel(&device);
        assert!(result.is_ok(), "Aroon kernel should compile successfully");
    }

    #[test]
    fn test_aroon_trait_properties() {
        assert_eq!(AroonIndicator::kernel_name(), "persistent_aroon_kernel");
        assert_eq!(AroonIndicator::num_inputs(), 2);
        assert_eq!(AroonIndicator::num_outputs(), 3);
    }
}
