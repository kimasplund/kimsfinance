//! Range Bars - Fixed Price Range Aggregation
//!
//! Range bars create a new bar when price moves a fixed range (e.g., $100).
//! This removes time from the equation, focusing purely on price movement.
//!
//! # Algorithm
//!
//! 1. Start new bar when (high - low) >= range_size
//! 2. Track current bar's OHLCV
//! 3. Accumulate volume until range exceeded
//! 4. Reset for next bar
//!
//! # Use Cases
//!
//! - **Volatility-Adjusted Trading**: Equal price movement per bar
//! - **Noise Reduction**: Filters out small price fluctuations
//! - **Breakout Detection**: Clear support/resistance levels
//! - **Algorithmic Trading**: Consistent price-based signals
//!
//! # Performance
//!
//! Sequential processing (price dependencies):
//! - One thread per task handles entire dataset
//! - Grid synchronization between tasks
//! - Persistent kernel reduces launch overhead

use crate::gpu::persistent::PersistentIndicator;

/// Range Bar aggregator for persistent kernel execution
pub struct RangeBarAggregator;

/// Parameters for Range Bar calculation
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RangeBarParams {
    /// Fixed price range per bar (e.g., 100.0 = $100 move)
    pub range_size: f64,
}

/// CUDA kernel for persistent Range Bar calculation
use super::traits::{CandleAggregator, TradeBasedAggregator};
use super::types::{TradeData, OHLCVCandle};
///
/// # Input Layout
/// - `input[0..n]`: timestamp (unix nanoseconds)
/// - `input[n..2n]`: price
/// - `input[2n..3n]`: volume
///
/// # Output Layout
/// - `output[0..m]`: open (m <= n, variable output size)
/// - `output[m..2m]`: high
/// - `output[2m..3m]`: low
/// - `output[3m..4m]`: close
/// - `output[4m..5m]`: volume
///
/// # Algorithm Details
///
/// Range bars form when price movement exceeds range_size:
/// - Track current bar: open, high, low, close, volume
/// - Check if (high - low) >= range_size
/// - If yes: emit bar, start new bar
/// - If no: continue accumulating
const RANGE_BAR_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_range_bar_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (timestamp+price+volume)
    double** __restrict__ output_batch,          // Array of output pointers (OHLCV)
    const int* __restrict__ sizes,               // Array of dataset sizes (ticks)
    const double* __restrict__ range_sizes,      // Array of range sizes
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
        double range_size = range_sizes[task_id];

        // Split input buffer: [timestamp(n), price(n), volume(n)]
        const double* timestamps = input;           // First n elements
        const double* prices = input + n;           // Next n elements
        const double* volumes = input + 2*n;        // Last n elements

        double* output = output_batch[task_id];

        // Sequential calculation (one thread per task for dependency handling)
        if (global_tid == task_id % grid_size) {
            int bar_count = 0;

            // Current bar state
            double bar_open = 0.0;
            double bar_high = 0.0;
            double bar_low = 0.0;
            double bar_close = 0.0;
            double bar_volume = 0.0;
            double bar_timestamp = 0.0;
            bool bar_started = false;

            for (int i = 0; i < n; i++) {
                double price = prices[i];
                double volume = volumes[i];
                double timestamp = timestamps[i];

                // Start new bar if needed
                if (!bar_started) {
                    bar_open = price;
                    bar_high = price;
                    bar_low = price;
                    bar_close = price;
                    bar_volume = volume;
                    bar_timestamp = timestamp;
                    bar_started = true;
                } else {
                    // Update current bar
                    if (price > bar_high) bar_high = price;
                    if (price < bar_low) bar_low = price;
                    bar_close = price;
                    bar_volume += volume;
                }

                // Check if range exceeded
                double current_range = bar_high - bar_low;
                if (current_range >= range_size) {
                    // Emit bar (OHLCV format)
                    // Output layout: [open(m), high(m), low(m), close(m), volume(m)]
                    // Maximum m = n (worst case: every tick = 1 bar)
                    output[bar_count] = bar_open;                    // Open
                    output[n + bar_count] = bar_high;                // High
                    output[2*n + bar_count] = bar_low;               // Low
                    output[3*n + bar_count] = bar_close;             // Close
                    output[4*n + bar_count] = bar_volume;            // Volume

                    bar_count++;
                    bar_started = false; // Start new bar on next tick
                }
            }

            // Emit final incomplete bar if exists
            if (bar_started && bar_count < n) {
                output[bar_count] = bar_open;
                output[n + bar_count] = bar_high;
                output[2*n + bar_count] = bar_low;
                output[3*n + bar_count] = bar_close;
                output[4*n + bar_count] = bar_volume;
                bar_count++;
            }

            // Store bar count in first output element (metadata)
            // Actual bars start from index 1
            // ALTERNATIVE: Use separate output buffer for count
            // For now, bar_count is implicit from non-NaN values
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for RangeBarAggregator {
    type Params = RangeBarParams;

    fn kernel_source() -> &'static str {
        RANGE_BAR_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_range_bar_kernel"
    }

    fn num_inputs() -> usize {
        3 // timestamp, price, volume
    }

    fn num_outputs() -> usize {
        5 // OHLCV
    }
}

impl CandleAggregator for RangeBarAggregator {
    type InputData = TradeData;
    type OutputCandle = OHLCVCandle;

    fn supports_streaming() -> bool {
        false // Range bars have sequential price dependencies
    }

    fn expected_compression_ratio() -> usize {
        20 // Estimate ~20 trades per range bar
    }
}

impl TradeBasedAggregator for RangeBarAggregator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_range_bar_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = RangeBarAggregator::compile_kernel(&device);
        assert!(result.is_ok(), "Range Bar kernel should compile successfully");
    }

    #[test]
    fn test_range_bar_trait_properties() {
        assert_eq!(RangeBarAggregator::kernel_name(), "persistent_range_bar_kernel");
        assert_eq!(RangeBarAggregator::num_inputs(), 3);
        assert_eq!(RangeBarAggregator::num_outputs(), 5);
    }

    #[test]
    fn test_range_bar_params_size() {
        // Verify params is compatible with GPU transfer
        assert_eq!(std::mem::size_of::<RangeBarParams>(), 8); // f64
    }
}
