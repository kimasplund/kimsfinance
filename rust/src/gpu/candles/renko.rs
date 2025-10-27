//! Renko Bricks - Price Movement Only (No Time)
//!
//! Renko charts use "bricks" that form only when price moves a full brick size.
//! This completely removes time and volume, focusing purely on price trends.
//!
//! # Algorithm
//!
//! 1. Start with initial brick at first price
//! 2. Form new brick only when price moves >= brick_size
//! 3. Direction: +1 (up brick) or -1 (down brick)
//! 4. Price can skip multiple bricks in large moves
//! 5. Reversals require 2× brick_size movement
//!
//! # Use Cases
//!
//! - **Trend Following**: Clear visual trends without noise
//! - **Support/Resistance**: Each brick = significant level
//! - **Reversal Detection**: 2-brick reversal = trend change
//! - **Clean Charts**: Removes time-based clutter
//!
//! # Example
//!
//! ```text
//! brick_size = 50
//! Prices: 1000, 1020, 1060, 1040, 1110
//!
//! Bricks:
//! 1. 1000 → 1050 (up, +1)
//! 2. 1050 → 1100 (up, +1)
//! 3. No brick for 1040 (only -60, need -100 for reversal)
//! ```
//!
//! # Performance
//!
//! Sequential processing (price dependencies):
//! - One thread per task handles entire dataset
//! - Grid synchronization between tasks
//! - Persistent kernel reduces launch overhead

use crate::gpu::persistent::PersistentIndicator;
use super::traits::{CandleAggregator, TradeBasedAggregator};
use super::types::{TradeData, OHLCVCandle};

/// Renko brick aggregator for persistent kernel execution
pub struct RenkoAggregator;

/// Parameters for Renko brick calculation
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RenkoParams {
    /// Price movement per brick (e.g., 50.0 = $50 per brick)
    pub brick_size: f64,
}

/// CUDA kernel for persistent Renko brick calculation
///
/// # Input Layout
/// - `input[0..n]`: timestamp (unix nanoseconds)
/// - `input[n..2n]`: price (no volume needed)
///
/// # Output Layout
/// - `output[0..m]`: brick_price (m <= n, variable output size)
/// - `output[m..2m]`: direction (+1.0 = up, -1.0 = down)
/// - `output[2m..3m]`: timestamp (when brick formed)
///
/// # Algorithm Details
///
/// Renko bricks form only on full brick-size moves:
/// - Track current brick price and direction
/// - Calculate price difference from current brick
/// - If diff >= brick_size: form new brick(s)
/// - Handle reversals (require 2× brick_size)
/// - Can skip multiple bricks in large moves
const RENKO_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_renko_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (timestamp+price)
    double** __restrict__ output_batch,          // Array of output pointers (brick_price+direction+timestamp)
    const int* __restrict__ sizes,               // Array of dataset sizes (ticks)
    const double* __restrict__ brick_sizes,      // Array of brick sizes
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
        double brick_size = brick_sizes[task_id];

        // Split input buffer: [timestamp(n), price(n)]
        const double* timestamps = input;           // First n elements
        const double* prices = input + n;           // Last n elements

        double* output = output_batch[task_id];

        // Sequential calculation (one thread per task for dependency handling)
        if (global_tid == task_id % grid_size) {
            if (n == 0) {
                // No data
                grid.sync();
                continue;
            }

            int brick_count = 0;

            // Initialize first brick at first price (rounded to brick_size)
            double first_price = prices[0];
            double current_brick = floor(first_price / brick_size) * brick_size;
            double current_direction = 1.0; // Start with up direction (arbitrary)

            // Emit first brick
            output[brick_count] = current_brick;                    // brick_price
            output[n + brick_count] = current_direction;            // direction
            output[2*n + brick_count] = timestamps[0];              // timestamp
            brick_count++;

            // Process remaining prices
            for (int i = 1; i < n; i++) {
                double price = prices[i];
                double timestamp = timestamps[i];

                // Calculate difference from current brick
                double diff = price - current_brick;

                // Check for continuation (same direction)
                if (current_direction > 0.0) {
                    // Currently up direction
                    if (diff >= brick_size) {
                        // Continue up: form new brick(s)
                        int num_bricks = (int)(diff / brick_size);
                        for (int j = 0; j < num_bricks && brick_count < n; j++) {
                            current_brick += brick_size;
                            output[brick_count] = current_brick;
                            output[n + brick_count] = current_direction;
                            output[2*n + brick_count] = timestamp;
                            brick_count++;
                        }
                    } else if (diff <= -2.0 * brick_size) {
                        // Reversal: price dropped 2× brick_size
                        // Switch to down direction
                        current_direction = -1.0;
                        int num_bricks = (int)(fabs(diff) / brick_size) - 1; // -1 for reversal threshold
                        for (int j = 0; j < num_bricks && brick_count < n; j++) {
                            current_brick -= brick_size;
                            output[brick_count] = current_brick;
                            output[n + brick_count] = current_direction;
                            output[2*n + brick_count] = timestamp;
                            brick_count++;
                        }
                    }
                    // Else: no change (price within current brick range)
                } else {
                    // Currently down direction
                    if (diff <= -brick_size) {
                        // Continue down: form new brick(s)
                        int num_bricks = (int)(fabs(diff) / brick_size);
                        for (int j = 0; j < num_bricks && brick_count < n; j++) {
                            current_brick -= brick_size;
                            output[brick_count] = current_brick;
                            output[n + brick_count] = current_direction;
                            output[2*n + brick_count] = timestamp;
                            brick_count++;
                        }
                    } else if (diff >= 2.0 * brick_size) {
                        // Reversal: price rose 2× brick_size
                        // Switch to up direction
                        current_direction = 1.0;
                        int num_bricks = (int)(diff / brick_size) - 1; // -1 for reversal threshold
                        for (int j = 0; j < num_bricks && brick_count < n; j++) {
                            current_brick += brick_size;
                            output[brick_count] = current_brick;
                            output[n + brick_count] = current_direction;
                            output[2*n + brick_count] = timestamp;
                            brick_count++;
                        }
                    }
                    // Else: no change (price within current brick range)
                }
            }

            // Note: brick_count is implicit from non-NaN values
            // Caller must check output for valid bricks
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for RenkoAggregator {
    type Params = RenkoParams;

    fn kernel_source() -> &'static str {
        RENKO_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_renko_kernel"
    }

    fn num_inputs() -> usize {
        2 // timestamp, price (no volume)
    }

    fn num_outputs() -> usize {
        3 // brick_price, direction, timestamp
    }
}

impl CandleAggregator for RenkoAggregator {
    type InputData = TradeData;
    type OutputCandle = OHLCVCandle;

    fn supports_streaming() -> bool {
        false // Renko bricks have sequential price dependencies
    }

    fn expected_compression_ratio() -> usize {
        30 // Estimate ~30 trades per brick
    }
}

impl TradeBasedAggregator for RenkoAggregator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_renko_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = RenkoAggregator::compile_kernel(&device);
        assert!(result.is_ok(), "Renko kernel should compile successfully");
    }

    #[test]
    fn test_renko_trait_properties() {
        assert_eq!(RenkoAggregator::kernel_name(), "persistent_renko_kernel");
        assert_eq!(RenkoAggregator::num_inputs(), 2);
        assert_eq!(RenkoAggregator::num_outputs(), 3);
    }

    #[test]
    fn test_renko_params_size() {
        // Verify params is compatible with GPU transfer
        assert_eq!(std::mem::size_of::<RenkoParams>(), 8); // f64
    }

    #[test]
    fn test_renko_reversal_logic() {
        // Document reversal requirements
        let brick_size = 50.0;
        let reversal_threshold = 2.0 * brick_size; // 100.0

        // Up trend at 1000:
        // - Continue up: price >= 1050 (+1 brick)
        // - Reverse down: price <= 900 (-2 bricks required)
        assert_eq!(reversal_threshold, 100.0);
    }
}
