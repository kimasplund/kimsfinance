//! Volume Bar Aggregation - Fixed Volume per Bar
//!
//! Implements volume-based candle aggregation where each bar represents a fixed
//! amount of traded volume (e.g., 100 BTC, 1M USDT).
//!
//! # Algorithm
//!
//! Accumulate trades sequentially until cumulative volume >= threshold:
//! 1. Start new bar when volume threshold reached
//! 2. Track OHLCV within each bar
//! 3. Record start_time (first trade) and end_time (last trade)
//! 4. Output: [open, high, low, close, volume, start_time, end_time]
//!
//! # Use Cases
//!
//! - **Order flow analysis**: Equal volume bars for consistent liquidity analysis
//! - **Market microstructure**: Identify volume patterns independent of time
//! - **High-frequency trading**: Normalize activity across different market conditions
//!
//! # Performance Characteristics
//!
//! - **Sequential**: Volume accumulation has dependencies (can't parallelize)
//! - **Similar to OBV**: Uses one thread per task for dependency handling
//! - **Output**: Variable number of bars (depends on volume threshold)
//!
//! # Example
//!
//! ```rust,ignore
//! // Aggregate trades into 100 BTC volume bars
//! let params = VolumeBarParams {
//!     volume_per_bar: 100.0,
//! };
//!
//! // Input: [timestamp, price, volume] for each trade
//! let trades = vec![
//!     1000.0, 50000.0, 25.0,  // Trade 1
//!     1001.0, 50100.0, 30.0,  // Trade 2
//!     1002.0, 50200.0, 50.0,  // Trade 3 (total: 105 > 100, new bar starts)
//!     1003.0, 50150.0, 20.0,  // Trade 4
//! ];
//!
//! // Output: Bar 1 = [50000, 50200, 50000, 50200, 105, 1000, 1002]
//! //         Bar 2 = [50150, 50150, 50150, 50150, 20, 1003, 1003] (partial)
//! ```

use super::super::persistent::traits::{MultiOutputIndicator, PersistentIndicator};
use super::traits::{CandleAggregator, TradeBasedAggregator};
use super::types::{TradeData, OHLCVCandle};

/// Volume bar aggregator for persistent kernel execution
pub struct VolumeBarAggregator;

/// Parameters for volume bar aggregation
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VolumeBarParams {
    /// Fixed volume threshold per bar (e.g., 100.0 BTC)
    pub volume_per_bar: f64,
}

impl VolumeBarParams {
    /// Create new volume bar parameters
    ///
    /// # Arguments
    ///
    /// * `volume_per_bar` - Fixed volume threshold (must be > 0)
    pub fn new(volume_per_bar: f64) -> Self {
        assert!(volume_per_bar > 0.0, "Volume per bar must be positive");
        Self { volume_per_bar }
    }

    /// Standard 100 unit volume bars
    pub fn standard() -> Self {
        Self::new(100.0)
    }
}

/// CUDA kernel for persistent volume bar aggregation
///
/// Input buffer layout: [timestamp(n), price(n), volume(n)] - concatenated
/// Output buffer layout: [open, high, low, close, volume, start_time, end_time] per bar
const VOLUME_BAR_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN and infinity constants for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)

// Volume bar parameters (matches VolumeBarParams)
struct VolumeBarParams {
    double volume_per_bar;
};

extern "C" __global__ void persistent_volume_bars_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (timestamp+price+volume)
    double** __restrict__ output_batch,          // Array of output pointers (OHLCV+times)
    const int* __restrict__ sizes,               // Array of dataset sizes (number of trades)
    const VolumeBarParams* __restrict__ params,  // Array of volume bar parameters
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
        VolumeBarParams param = params[task_id];

        // Split input buffer: [timestamp(n), price(n), volume(n)]
        const double* timestamp = input;
        const double* price = input + n;
        const double* volume = input + 2 * n;

        double* output = output_batch[task_id];

        // Sequential calculation (one thread per task for dependency handling)
        // Use modulo to assign one thread per task
        if (global_tid == task_id % grid_size) {
            // State for current bar
            double cumulative_volume = 0.0;
            double bar_open = 0.0;
            double bar_high = -CUDART_INF;
            double bar_low = CUDART_INF;
            double bar_close = 0.0;
            double bar_volume = 0.0;
            double bar_start_time = 0.0;
            double bar_end_time = 0.0;
            bool bar_active = false;
            int bar_count = 0;

            // Maximum output bars (conservative estimate: all trades = 1 bar each)
            const int MAX_BARS = n;

            // Process each trade
            for (int i = 0; i < n; i++) {
                double t = timestamp[i];
                double p = price[i];
                double v = volume[i];

                // Start new bar if not active
                if (!bar_active) {
                    bar_open = p;
                    bar_high = p;
                    bar_low = p;
                    bar_close = p;
                    bar_volume = 0.0;
                    bar_start_time = t;
                    bar_end_time = t;
                    bar_active = true;
                }

                // Update OHLC
                if (p > bar_high) bar_high = p;
                if (p < bar_low) bar_low = p;
                bar_close = p;
                bar_end_time = t;

                // Accumulate volume
                cumulative_volume += v;
                bar_volume += v;

                // Check if bar is complete (volume threshold reached)
                if (cumulative_volume >= param.volume_per_bar) {
                    // Output completed bar (7 values per bar)
                    if (bar_count < MAX_BARS) {
                        int offset = bar_count * 7;
                        output[offset + 0] = bar_open;
                        output[offset + 1] = bar_high;
                        output[offset + 2] = bar_low;
                        output[offset + 3] = bar_close;
                        output[offset + 4] = bar_volume;
                        output[offset + 5] = bar_start_time;
                        output[offset + 6] = bar_end_time;
                        bar_count++;
                    }

                    // Reset for next bar
                    cumulative_volume = 0.0;
                    bar_active = false;
                }
            }

            // Handle partial final bar (include it)
            if (bar_active && bar_volume > 0.0) {
                if (bar_count < MAX_BARS) {
                    int offset = bar_count * 7;
                    output[offset + 0] = bar_open;
                    output[offset + 1] = bar_high;
                    output[offset + 2] = bar_low;
                    output[offset + 3] = bar_close;
                    output[offset + 4] = bar_volume;
                    output[offset + 5] = bar_start_time;
                    output[offset + 6] = bar_end_time;
                    bar_count++;
                }
            }

            // Fill remaining output with NaN (to indicate end of valid data)
            for (int i = bar_count * 7; i < MAX_BARS * 7; i++) {
                output[i] = CUDART_NAN;
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for VolumeBarAggregator {
    type Params = VolumeBarParams;

    fn kernel_source() -> &'static str {
        VOLUME_BAR_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_volume_bars_kernel"
    }

    fn num_inputs() -> usize {
        3 // Three inputs: timestamp, price, volume
    }

    fn num_outputs() -> usize {
        7 // Seven outputs per bar: O, H, L, C, V, start_time, end_time
    }
}

impl MultiOutputIndicator for VolumeBarAggregator {}

impl CandleAggregator for VolumeBarAggregator {
    type InputData = TradeData;
    type OutputCandle = OHLCVCandle;

    fn supports_streaming() -> bool {
        false // Volume bars require full processing (sequential dependencies)
    }

    fn expected_compression_ratio() -> usize {
        50 // Estimate ~50 trades per volume bar (depends on volume threshold)
    }
}

impl TradeBasedAggregator for VolumeBarAggregator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    fn test_volume_bar_params() {
        let params = VolumeBarParams::new(100.0);
        assert_eq!(params.volume_per_bar, 100.0);

        let standard = VolumeBarParams::standard();
        assert_eq!(standard.volume_per_bar, 100.0);
    }

    #[test]
    #[should_panic(expected = "Volume per bar must be positive")]
    fn test_volume_bar_params_validation() {
        VolumeBarParams::new(0.0);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_volume_bar_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = VolumeBarAggregator::compile_kernel(&device);
        assert!(
            result.is_ok(),
            "Volume bar kernel should compile successfully"
        );
    }

    #[test]
    fn test_volume_bar_trait_properties() {
        assert_eq!(
            VolumeBarAggregator::kernel_name(),
            "persistent_volume_bars_kernel"
        );
        assert_eq!(VolumeBarAggregator::num_inputs(), 3);
        assert_eq!(VolumeBarAggregator::num_outputs(), 7);
    }
}
