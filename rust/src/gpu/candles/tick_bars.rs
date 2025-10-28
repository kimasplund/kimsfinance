//! Tick Bar Aggregation - Fixed Trades per Bar
//!
//! Implements tick-based candle aggregation where each bar represents a fixed
//! number of trades (e.g., 100 trades per bar).
//!
//! # Algorithm
//!
//! Group every N trades into one bar:
//! 1. Divide trades into fixed-size groups
//! 2. Calculate OHLCV for each group
//! 3. Output includes trade count for validation
//! 4. Parallel-friendly: Can process multiple groups simultaneously
//!
//! # Use Cases
//!
//! - **Order flow analysis**: Equal tick bars for consistent activity analysis
//! - **Market microstructure**: Normalize across different trading speeds
//! - **High-frequency trading**: Time-independent activity patterns
//! - **Statistical analysis**: Uniform sample sizes for better statistics
//!
//! # Performance Characteristics
//!
//! - **Parallel-friendly**: Each bar can be computed independently
//! - **Block-based processing**: Multiple threads per bar for OHLC reduction
//! - **Faster than volume bars**: No sequential dependencies
//!
//! # Example
//!
//! ```rust,ignore
//! // Aggregate trades into 100-trade bars
//! let params = TickBarParams {
//!     ticks_per_bar: 100,
//! };
//!
//! // Input: [timestamp, price, volume] for 250 trades
//! // Output: 3 bars (100, 100, 50 trades each)
//! //   Bar 0: [open, high, low, close, volume, 100]
//! //   Bar 1: [open, high, low, close, volume, 100]
//! //   Bar 2: [open, high, low, close, volume, 50] (partial)
//! ```

use super::super::persistent::traits::{MultiOutputIndicator, PersistentIndicator};
use super::traits::{CandleAggregator, TradeBasedAggregator};
use super::types::{OHLCVCandle, TradeData};

/// Tick bar aggregator for persistent kernel execution
pub struct TickBarAggregator;

/// Parameters for tick bar aggregation
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TickBarParams {
    /// Fixed number of trades per bar (e.g., 100)
    pub ticks_per_bar: i32,
}

impl TickBarParams {
    /// Create new tick bar parameters
    ///
    /// # Arguments
    ///
    /// * `ticks_per_bar` - Fixed number of trades (must be > 0)
    pub fn new(ticks_per_bar: i32) -> Self {
        assert!(ticks_per_bar > 0, "Ticks per bar must be positive");
        Self { ticks_per_bar }
    }

    /// Standard 100-tick bars
    pub fn standard() -> Self {
        Self::new(100)
    }

    /// Small tick bars (10 trades)
    pub fn small() -> Self {
        Self::new(10)
    }

    /// Large tick bars (1000 trades)
    pub fn large() -> Self {
        Self::new(1000)
    }
}

/// CUDA kernel for persistent tick bar aggregation
///
/// Input buffer layout: [timestamp(n), price(n), volume(n)] - concatenated
/// Output buffer layout: [open, high, low, close, volume, trade_count] per bar
const TICK_BAR_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN and infinity constants for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)

// Tick bar parameters (matches TickBarParams)
struct TickBarParams {
    int ticks_per_bar;
};

extern "C" __global__ void persistent_tick_bars_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (timestamp+price+volume)
    double** __restrict__ output_batch,          // Array of output pointers (OHLCV+count)
    const int* __restrict__ sizes,               // Array of dataset sizes (number of trades)
    const TickBarParams* __restrict__ params,    // Array of tick bar parameters
    int num_tasks                                // Number of tasks to process
) {
    // Get grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Process each task sequentially (persistent kernel pattern)
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* input = input_batch[task_id];
        int n = sizes[task_id];  // Number of trades
        TickBarParams param = params[task_id];

        // Split input buffer: [timestamp(n), price(n), volume(n)]
        const double* timestamp = input;
        const double* price = input + n;
        const double* volume = input + 2 * n;

        double* output = output_batch[task_id];

        // Calculate number of bars (ceiling division)
        int num_bars = (n + param.ticks_per_bar - 1) / param.ticks_per_bar;

        // Parallel processing: Each thread processes one or more bars
        // Grid-stride loop for this task's bars
        for (int bar_idx = global_tid; bar_idx < num_bars; bar_idx += grid_size) {
            // Determine trade range for this bar
            int start_idx = bar_idx * param.ticks_per_bar;
            int end_idx = start_idx + param.ticks_per_bar;
            if (end_idx > n) end_idx = n;  // Handle partial final bar
            int actual_ticks = end_idx - start_idx;

            // Initialize OHLCV for this bar
            double bar_open = price[start_idx];
            double bar_high = -CUDART_INF;
            double bar_low = CUDART_INF;
            double bar_close = price[end_idx - 1];
            double bar_volume = 0.0;

            // Calculate OHLCV by scanning all trades in this bar
            for (int i = start_idx; i < end_idx; i++) {
                double p = price[i];
                double v = volume[i];

                // Update high/low
                if (p > bar_high) bar_high = p;
                if (p < bar_low) bar_low = p;

                // Accumulate volume
                bar_volume += v;
            }

            // Output bar data (6 values per bar)
            int offset = bar_idx * 6;
            output[offset + 0] = bar_open;
            output[offset + 1] = bar_high;
            output[offset + 2] = bar_low;
            output[offset + 3] = bar_close;
            output[offset + 4] = bar_volume;
            output[offset + 5] = (double)actual_ticks;  // Trade count for validation
        }

        // Fill remaining output with NaN (if num_bars < max possible)
        // This handles variable-length output gracefully
        for (int i = global_tid + num_bars; i < n; i += grid_size) {
            if (i * 6 < n * 6) {  // Ensure we don't write beyond buffer
                int offset = i * 6;
                for (int j = 0; j < 6; j++) {
                    output[offset + j] = CUDART_NAN;
                }
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for TickBarAggregator {
    type Params = TickBarParams;

    fn kernel_source() -> &'static str {
        TICK_BAR_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_tick_bars_kernel"
    }

    fn num_inputs() -> usize {
        3 // Three inputs: timestamp, price, volume
    }

    fn num_outputs() -> usize {
        6 // Six outputs per bar: O, H, L, C, V, trade_count
    }
}

impl MultiOutputIndicator for TickBarAggregator {}

impl CandleAggregator for TickBarAggregator {
    type InputData = TradeData;
    type OutputCandle = OHLCVCandle;

    fn supports_streaming() -> bool {
        true // Tick bars are deterministic based on count
    }

    fn expected_compression_ratio() -> usize {
        100 // Exactly ticks_per_bar trades per candle
    }
}

impl TradeBasedAggregator for TickBarAggregator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    fn test_tick_bar_params() {
        let params = TickBarParams::new(100);
        assert_eq!(params.ticks_per_bar, 100);

        let standard = TickBarParams::standard();
        assert_eq!(standard.ticks_per_bar, 100);

        let small = TickBarParams::small();
        assert_eq!(small.ticks_per_bar, 10);

        let large = TickBarParams::large();
        assert_eq!(large.ticks_per_bar, 1000);
    }

    #[test]
    #[should_panic(expected = "Ticks per bar must be positive")]
    fn test_tick_bar_params_validation() {
        TickBarParams::new(0);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_tick_bar_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = TickBarAggregator::compile_kernel(&device);
        assert!(
            result.is_ok(),
            "Tick bar kernel should compile successfully"
        );
    }

    #[test]
    fn test_tick_bar_trait_properties() {
        assert_eq!(
            TickBarAggregator::kernel_name(),
            "persistent_tick_bars_kernel"
        );
        assert_eq!(TickBarAggregator::num_inputs(), 3);
        assert_eq!(TickBarAggregator::num_outputs(), 6);
    }
}
