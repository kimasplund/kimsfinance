//! Persistent Stochastic Oscillator kernel
//!
//! Implements Stochastic %K and %D calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! Stochastic oscillator measures momentum using price position within a high-low range.
//!
//! **%K Line** (fast stochastic):
//! - %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
//! - Calculated over k_period lookback window
//!
//! **%D Line** (slow stochastic):
//! - %D = SMA(%K, d_period)
//! - Smoothed version of %K line
//!
//! # Calculation Steps
//!
//! 1. For each candle, find highest high and lowest low over k_period
//! 2. Calculate %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
//! 3. Calculate %D as simple moving average of %K over d_period
//!
//! # Performance
//!
//! Persistent kernel execution reduces overhead:
//! - Traditional: 3 arrays (high, low, close) × N tasks × 10μs = 30N μs
//! - Persistent: 1 launch × 10μs = 10μs (96% reduction for N=10)

use super::super::traits::{MultiOutputIndicator, PersistentIndicator};

/// Stochastic indicator for persistent kernel execution
pub struct StochasticIndicator;

/// Parameters for Stochastic calculation
///
/// Standard values: k_period=14, d_period=3
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct StochasticParams {
    pub k_period: i32,
    pub d_period: i32,
}

impl StochasticParams {
    /// Create standard Stochastic parameters (14, 3)
    pub fn standard() -> Self {
        Self {
            k_period: 14,
            d_period: 3,
        }
    }

    /// Create custom Stochastic parameters
    pub fn new(k_period: i32, d_period: i32) -> Self {
        Self { k_period, d_period }
    }
}

/// CUDA kernel for persistent Stochastic calculation
///
/// Requires three input arrays: high, low, close
/// Produces two outputs: %K line, %D line
const STOCHASTIC_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define constants for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)

// Stochastic parameters struct (must match Rust layout)
struct StochasticParams {
    int k_period;
    int d_period;
};

extern "C" __global__ void persistent_stochastic_kernel(
    const double** __restrict__ high_batch,      // Array of high price pointers
    const double** __restrict__ low_batch,       // Array of low price pointers
    const double** __restrict__ close_batch,     // Array of close price pointers
    double** __restrict__ output_batch,          // Array of output pointers (%K + %D concatenated)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const StochasticParams* __restrict__ params, // Array of Stochastic parameters
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
        const double* close = close_batch[task_id];
        double* output = output_batch[task_id];
        int n = sizes[task_id];

        // Output layout: [k_line (n), d_line (n)]
        double* k_line = output;         // First n elements
        double* d_line = output + n;     // Next n elements

        StochasticParams p = params[task_id];
        int k_period = p.k_period;
        int d_period = p.d_period;

        // Sequential calculation (one thread per task for dependency handling)
        if (global_tid == task_id % grid_size) {
            // Calculate %K line
            for (int idx = 0; idx < n; idx++) {
                if (idx >= k_period - 1) {
                    // Find highest high and lowest low in k_period window
                    double highest_high = -CUDART_INF;
                    double lowest_low = CUDART_INF;

                    for (int i = 0; i < k_period; i++) {
                        int window_idx = idx - i;
                        if (window_idx >= 0) {
                            highest_high = fmax(highest_high, high[window_idx]);
                            lowest_low = fmin(lowest_low, low[window_idx]);
                        }
                    }

                    // Calculate %K: 100 * (close - lowest_low) / (highest_high - lowest_low)
                    double range = highest_high - lowest_low;
                    if (range > 1e-10) {
                        k_line[idx] = 100.0 * (close[idx] - lowest_low) / range;
                    } else {
                        // When range is zero, use midpoint (50)
                        k_line[idx] = 50.0;
                    }
                } else {
                    k_line[idx] = CUDART_NAN;
                }
            }

            // Calculate %D line (SMA of %K)
            int d_start = k_period + d_period - 2;
            for (int idx = 0; idx < n; idx++) {
                if (idx >= d_start) {
                    double sum = 0.0;
                    int count = 0;

                    for (int i = 0; i < d_period; i++) {
                        int k_idx = idx - i;
                        if (k_idx >= k_period - 1 && !isnan(k_line[k_idx])) {
                            sum += k_line[k_idx];
                            count++;
                        }
                    }

                    if (count == d_period) {
                        d_line[idx] = sum / d_period;
                    } else {
                        d_line[idx] = CUDART_NAN;
                    }
                } else {
                    d_line[idx] = CUDART_NAN;
                }
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for StochasticIndicator {
    type Params = StochasticParams;

    fn kernel_source() -> &'static str {
        STOCHASTIC_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_stochastic_kernel"
    }

    fn num_inputs() -> usize {
        3 // Three inputs: high, low, close
    }

    fn num_outputs() -> usize {
        2 // Two outputs: %K line, %D line
    }
}

impl MultiOutputIndicator for StochasticIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_stochastic_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = StochasticIndicator::compile_kernel(&device);
        assert!(
            result.is_ok(),
            "Stochastic kernel should compile successfully"
        );
    }

    #[test]
    fn test_stochastic_trait_properties() {
        assert_eq!(
            StochasticIndicator::kernel_name(),
            "persistent_stochastic_kernel"
        );
        assert_eq!(StochasticIndicator::num_inputs(), 3);
        assert_eq!(StochasticIndicator::num_outputs(), 2);
    }

    #[test]
    fn test_stochastic_params() {
        let params = StochasticParams::standard();
        assert_eq!(params.k_period, 14);
        assert_eq!(params.d_period, 3);

        let custom = StochasticParams::new(21, 5);
        assert_eq!(custom.k_period, 21);
        assert_eq!(custom.d_period, 5);
    }
}
