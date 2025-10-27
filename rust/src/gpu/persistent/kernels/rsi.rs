//! Persistent RSI (Relative Strength Index) kernel
//!
//! Implements RSI calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! RSI = 100 - (100 / (1 + RS))
//! where RS = average_gain / average_loss over period
//!
//! # Calculation Steps
//!
//! 1. Calculate price changes: delta[i] = price[i] - price[i-1]
//! 2. Separate gains and losses:
//!    - gain[i] = max(delta[i], 0)
//!    - loss[i] = max(-delta[i], 0)
//! 3. Calculate average gain/loss using SMA for first period, then EMA
//! 4. RS = avg_gain / avg_loss
//! 5. RSI = 100 - (100 / (1 + RS))
//!
//! # Performance
//!
//! Persistent kernel execution eliminates per-task launch overhead:
//! - Traditional: N tasks × 10μs = N×10μs overhead
//! - Persistent: 1 launch × 10μs = 10μs overhead (90% reduction for N=10)

use super::super::traits::{PersistentIndicator, SingleOutputIndicator};

/// RSI indicator for persistent kernel execution
pub struct RsiIndicator;

/// CUDA kernel for persistent RSI calculation
///
/// Uses Wilder's smoothing (EMA with alpha = 1/period) for average gain/loss.
const RSI_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_rsi_kernel(
    const double** __restrict__ input_batch,    // Array of input pointers (close prices)
    double** __restrict__ output_batch,          // Array of output pointers (RSI values)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of RSI periods
    int num_tasks                                // Number of tasks to process
) {
    // Get grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Process each task sequentially (persistent kernel pattern)
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* close = input_batch[task_id];
        double* rsi = output_batch[task_id];
        int n = sizes[task_id];
        int period = periods[task_id];

        // First pass: Calculate initial SMA of gains/losses
        // Grid-stride loop for parallel computation
        for (int idx = global_tid; idx < n; idx += grid_size) {
            if (idx < period) {
                // Not enough data for RSI
                rsi[idx] = CUDART_NAN;
            } else if (idx == period) {
                // First RSI value: use SMA for initial average
                double sum_gain = 0.0;
                double sum_loss = 0.0;

                for (int i = 1; i <= period; i++) {
                    double delta = close[i] - close[i - 1];
                    if (delta > 0.0) {
                        sum_gain += delta;
                    } else {
                        sum_loss += -delta;
                    }
                }

                double avg_gain = sum_gain / period;
                double avg_loss = sum_loss / period;

                // Calculate RSI
                if (avg_loss == 0.0) {
                    rsi[idx] = 100.0;  // No losses, RSI = 100
                } else {
                    double rs = avg_gain / avg_loss;
                    rsi[idx] = 100.0 - (100.0 / (1.0 + rs));
                }
            }
        }

        // Synchronize after first pass
        grid.sync();

        // Second pass: Calculate subsequent RSI values using EMA
        // Only one thread per task to ensure sequential dependency
        if (global_tid == task_id % grid_size) {
            double alpha = 1.0 / period;

            // Get initial averages from first RSI calculation
            double sum_gain = 0.0;
            double sum_loss = 0.0;

            for (int i = 1; i <= period; i++) {
                double delta = close[i] - close[i - 1];
                if (delta > 0.0) {
                    sum_gain += delta;
                } else {
                    sum_loss += -delta;
                }
            }

            double avg_gain = sum_gain / period;
            double avg_loss = sum_loss / period;

            // Calculate subsequent RSI values using EMA (Wilder's smoothing)
            for (int i = period + 1; i < n; i++) {
                double delta = close[i] - close[i - 1];
                double gain = (delta > 0.0) ? delta : 0.0;
                double loss = (delta < 0.0) ? -delta : 0.0;

                // EMA update: new_avg = alpha * current + (1 - alpha) * prev_avg
                avg_gain = alpha * gain + (1.0 - alpha) * avg_gain;
                avg_loss = alpha * loss + (1.0 - alpha) * avg_loss;

                // Calculate RSI
                if (avg_loss == 0.0) {
                    rsi[i] = 100.0;
                } else {
                    double rs = avg_gain / avg_loss;
                    rsi[i] = 100.0 - (100.0 / (1.0 + rs));
                }
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for RsiIndicator {
    type Params = i32; // RSI period (typically 14)

    fn kernel_source() -> &'static str {
        RSI_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_rsi_kernel"
    }

    fn num_outputs() -> usize {
        1 // Single output: RSI values
    }
}

impl SingleOutputIndicator for RsiIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_rsi_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = RsiIndicator::compile_kernel(&device);
        assert!(result.is_ok(), "RSI kernel should compile successfully");
    }

    #[test]
    fn test_rsi_trait_properties() {
        assert_eq!(RsiIndicator::kernel_name(), "persistent_rsi_kernel");
        assert_eq!(RsiIndicator::num_inputs(), 1);
        assert_eq!(RsiIndicator::num_outputs(), 1);
    }
}
