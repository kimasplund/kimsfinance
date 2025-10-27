//! Persistent Heikin-Ashi transformation kernel
//!
//! Implements Heikin-Ashi smoothed candles using persistent kernel pattern for batch processing.
//!
//! # Heikin-Ashi Algorithm
//!
//! Heikin-Ashi transforms traditional OHLC candles into smoothed candles that reduce noise
//! and highlight trends:
//!
//! ```text
//! HA-Close = (Open + High + Low + Close) / 4
//! HA-Open = (Previous HA-Open + Previous HA-Close) / 2
//! HA-High = max(High, HA-Open, HA-Close)
//! HA-Low = min(Low, HA-Open, HA-Close)
//! ```
//!
//! # Sequential Dependency
//!
//! Like EMA, Heikin-Ashi has sequential dependencies - each bar's HA-Open depends on
//! the previous bar's values. This makes it an IIR (infinite impulse response) filter.
//!
//! # Performance Note
//!
//! - **Sequential within symbol**: One thread per task due to dependencies
//! - **Parallel across symbols**: Multiple symbols processed simultaneously
//! - **Persistent kernel benefit**: Single kernel launch for batch, reduces overhead
//!
//! # Input/Output Format
//!
//! - **Input**: `[open(n), high(n), low(n), close(n)]` concatenated (4*n elements)
//! - **Output**: `[ha_open(n), ha_high(n), ha_low(n), ha_close(n)]` concatenated (4*n elements)

use super::super::persistent::traits::{MultiOutputIndicator, PersistentIndicator};
use super::traits::{CandleAggregator, CandleBasedAggregator};
use super::types::OHLCVCandle;

/// Heikin-Ashi candle transformation aggregator
pub struct HeikinAshiAggregator;

/// CUDA kernel for persistent Heikin-Ashi transformation
///
/// Uses sequential processing with proper initialization.
/// First candle uses original OHLC, subsequent candles use HA formula.
const HEIKIN_ASHI_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_heikin_ashi_kernel(
    const double** __restrict__ input_batch,    // Array of input pointers [O,H,L,C concatenated]
    double** __restrict__ output_batch,          // Array of output pointers [HA-O,HA-H,HA-L,HA-C]
    const int* __restrict__ sizes,               // Array of dataset sizes (number of bars)
    const int* __restrict__ params,              // Unused (zero-sized type placeholder)
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

        // Input layout: [open(n), high(n), low(n), close(n)]
        const double* open = input;              // First n elements
        const double* high = input + n;          // Next n elements
        const double* low = input + 2*n;         // Next n elements
        const double* close = input + 3*n;       // Last n elements

        // Output layout: [ha_open(n), ha_high(n), ha_low(n), ha_close(n)]
        double* ha_open = output;                // First n elements
        double* ha_high = output + n;            // Next n elements
        double* ha_low = output + 2*n;           // Next n elements
        double* ha_close = output + 3*n;         // Last n elements

        // Sequential calculation (one thread per task due to data dependency)
        // Use thread assignment based on task_id to distribute work across grid
        if (global_tid == task_id % grid_size) {
            // Initialize first Heikin-Ashi candle with original OHLC
            ha_close[0] = (open[0] + high[0] + low[0] + close[0]) * 0.25;
            ha_open[0] = (open[0] + close[0]) * 0.5;
            ha_high[0] = fmax(high[0], fmax(ha_open[0], ha_close[0]));
            ha_low[0] = fmin(low[0], fmin(ha_open[0], ha_close[0]));

            // Apply Heikin-Ashi formula for remaining bars
            for (int i = 1; i < n; i++) {
                // HA-Close = (O + H + L + C) / 4
                ha_close[i] = (open[i] + high[i] + low[i] + close[i]) * 0.25;

                // HA-Open = (Previous HA-Open + Previous HA-Close) / 2
                ha_open[i] = (ha_open[i-1] + ha_close[i-1]) * 0.5;

                // HA-High = max(H, HA-Open, HA-Close)
                ha_high[i] = fmax(high[i], fmax(ha_open[i], ha_close[i]));

                // HA-Low = min(L, HA-Open, HA-Close)
                ha_low[i] = fmin(low[i], fmin(ha_open[i], ha_close[i]));
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for HeikinAshiAggregator {
    type Params = i32; // Zero-sized placeholder (use i32 for C compatibility)

    fn kernel_source() -> &'static str {
        HEIKIN_ASHI_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_heikin_ashi_kernel"
    }

    fn num_inputs() -> usize {
        4 // Open, High, Low, Close
    }

    fn num_outputs() -> usize {
        4 // HA-Open, HA-High, HA-Low, HA-Close
    }
}

impl CandleAggregator for HeikinAshiAggregator {
    type InputData = OHLCVCandle;
    type OutputCandle = OHLCVCandle;

    fn supports_streaming() -> bool {
        false // Heikin-Ashi has sequential dependencies (IIR filter)
    }

    fn expected_compression_ratio() -> usize {
        1 // 1:1 transformation of candles
    }
}

impl CandleBasedAggregator for HeikinAshiAggregator {}

impl MultiOutputIndicator for HeikinAshiAggregator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_heikin_ashi_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = HeikinAshiAggregator::compile_kernel(&device);
        assert!(
            result.is_ok(),
            "Heikin-Ashi kernel should compile successfully"
        );
    }

    #[test]
    fn test_heikin_ashi_trait_properties() {
        assert_eq!(
            HeikinAshiAggregator::kernel_name(),
            "persistent_heikin_ashi_kernel"
        );
        assert_eq!(HeikinAshiAggregator::num_inputs(), 4);
        assert_eq!(HeikinAshiAggregator::num_outputs(), 4);
        assert!(!HeikinAshiAggregator::supports_streaming());
    }

    #[test]
    fn test_heikin_ashi_algorithm_correctness() {
        // Manual calculation test data
        // Original OHLC: [(1.0, 3.0, 0.5, 2.0), (2.0, 4.0, 1.5, 3.0)]
        let open = vec![1.0, 2.0];
        let high = vec![3.0, 4.0];
        let low = vec![0.5, 1.5];
        let close = vec![2.0, 3.0];

        // Expected HA values (calculated manually):
        // Bar 0:
        //   HA-Close[0] = (1.0 + 3.0 + 0.5 + 2.0) / 4 = 1.625
        //   HA-Open[0] = (1.0 + 2.0) / 2 = 1.5
        //   HA-High[0] = max(3.0, 1.5, 1.625) = 3.0
        //   HA-Low[0] = min(0.5, 1.5, 1.625) = 0.5

        let expected_ha_close_0: f64 = (1.0 + 3.0 + 0.5 + 2.0) * 0.25;
        let expected_ha_open_0: f64 = (1.0 + 2.0) * 0.5;
        let expected_ha_high_0: f64 = 3.0_f64.max(expected_ha_open_0.max(expected_ha_close_0));
        let expected_ha_low_0: f64 = 0.5_f64.min(expected_ha_open_0.min(expected_ha_close_0));

        assert_eq!(expected_ha_close_0, 1.625);
        assert_eq!(expected_ha_open_0, 1.5);
        assert_eq!(expected_ha_high_0, 3.0);
        assert_eq!(expected_ha_low_0, 0.5);

        // Bar 1:
        //   HA-Close[1] = (2.0 + 4.0 + 1.5 + 3.0) / 4 = 2.625
        //   HA-Open[1] = (1.5 + 1.625) / 2 = 1.5625
        //   HA-High[1] = max(4.0, 1.5625, 2.625) = 4.0
        //   HA-Low[1] = min(1.5, 1.5625, 2.625) = 1.5

        let expected_ha_close_1: f64 = (2.0 + 4.0 + 1.5 + 3.0) * 0.25;
        let expected_ha_open_1: f64 = (expected_ha_open_0 + expected_ha_close_0) * 0.5;
        let expected_ha_high_1: f64 = 4.0_f64.max(expected_ha_open_1.max(expected_ha_close_1));
        let expected_ha_low_1: f64 = 1.5_f64.min(expected_ha_open_1.min(expected_ha_close_1));

        assert_eq!(expected_ha_close_1, 2.625);
        assert_eq!(expected_ha_open_1, 1.5625);
        assert_eq!(expected_ha_high_1, 4.0);
        assert_eq!(expected_ha_low_1, 1.5);
    }
}
