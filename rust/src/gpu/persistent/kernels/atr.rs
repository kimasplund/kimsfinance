//! Persistent ATR (Average True Range) kernel
//!
//! Implements ATR calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! ATR measures volatility by calculating the exponential moving average of True Range.
//!
//! **True Range** = max(high - low, |high - prev_close|, |low - prev_close|)
//!
//! **ATR** = EMA of True Range over period
//!
//! # Calculation Steps
//!
//! 1. Calculate True Range for each candle:
//!    - TR = max(high[i] - low[i], |high[i] - close[i-1]|, |low[i] - close[i-1]|)
//! 2. Calculate initial ATR using SMA of first `period` True Range values
//! 3. Calculate subsequent ATR using EMA:
//!    - ATR[i] = (TR[i] * alpha) + (ATR[i-1] * (1 - alpha))
//!    - alpha = 1 / period (Wilder's smoothing)
//!
//! # Performance
//!
//! Persistent kernel execution reduces overhead:
//! - Traditional: 3 arrays (high, low, close) × N tasks × 10μs = 30N μs
//! - Persistent: 1 launch × 10μs = 10μs (96% reduction for N=10)

use super::super::traits::{PersistentIndicator, SingleOutputIndicator};

/// ATR indicator for persistent kernel execution
pub struct AtrIndicator;

/// CUDA kernel for persistent ATR calculation
///
/// Input buffer layout: [high(n), low(n), close(n)] - concatenated
const ATR_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_atr_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (high+low+close concatenated)
    double** __restrict__ output_batch,          // Array of output pointers (ATR values)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of ATR periods
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

        double* atr = output_batch[task_id];

        // Sequential calculation (one thread per task for dependency handling)
        if (global_tid == task_id % grid_size) {
            // First value is NaN (no previous close)
            atr[0] = CUDART_NAN;

            if (n <= period) {
                // Not enough data for ATR
                for (int i = 1; i < n; i++) {
                    atr[i] = CUDART_NAN;
                }
            } else {
                // Calculate True Range and accumulate for initial SMA
                double sum_tr = 0.0;

                for (int i = 1; i <= period; i++) {
                    // True Range = max(high - low, |high - prev_close|, |low - prev_close|)
                    double hl = high[i] - low[i];
                    double hc = fabs(high[i] - close[i - 1]);
                    double lc = fabs(low[i] - close[i - 1]);

                    double tr = fmax(hl, fmax(hc, lc));
                    sum_tr += tr;

                    // First `period` values are NaN (not enough data)
                    if (i < period) {
                        atr[i] = CUDART_NAN;
                    }
                }

                // Initial ATR is SMA of True Range
                double prev_atr = sum_tr / period;
                atr[period] = prev_atr;

                // Calculate subsequent ATR using EMA (Wilder's smoothing)
                double alpha = 1.0 / period;

                for (int i = period + 1; i < n; i++) {
                    // Calculate True Range
                    double hl = high[i] - low[i];
                    double hc = fabs(high[i] - close[i - 1]);
                    double lc = fabs(low[i] - close[i - 1]);

                    double tr = fmax(hl, fmax(hc, lc));

                    // EMA update: ATR = alpha * TR + (1 - alpha) * prev_ATR
                    prev_atr = alpha * tr + (1.0 - alpha) * prev_atr;
                    atr[i] = prev_atr;
                }
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for AtrIndicator {
    type Params = i32; // ATR period (typically 14)

    fn kernel_source() -> &'static str {
        ATR_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_atr_kernel"
    }

    fn num_inputs() -> usize {
        3 // Three inputs: high, low, close
    }

    fn num_outputs() -> usize {
        1 // Single output: ATR values
    }
}

impl SingleOutputIndicator for AtrIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_atr_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = AtrIndicator::compile_kernel(&device);
        assert!(result.is_ok(), "ATR kernel should compile successfully");
    }

    #[test]
    fn test_atr_trait_properties() {
        assert_eq!(AtrIndicator::kernel_name(), "persistent_atr_kernel");
        assert_eq!(AtrIndicator::num_inputs(), 3);
        assert_eq!(AtrIndicator::num_outputs(), 1);
    }
}
