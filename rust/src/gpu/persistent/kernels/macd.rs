//! Persistent MACD (Moving Average Convergence Divergence) kernel
//!
//! Implements MACD calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! MACD has three outputs:
//! 1. **MACD Line**: fast_ema - slow_ema
//! 2. **Signal Line**: EMA of MACD line
//! 3. **Histogram**: MACD line - signal line
//!
//! # Standard Parameters
//!
//! - Fast period: 12
//! - Slow period: 26
//! - Signal period: 9
//!
//! # Calculation Steps
//!
//! 1. Calculate fast EMA (exponential moving average)
//! 2. Calculate slow EMA
//! 3. MACD line = fast_ema - slow_ema
//! 4. Signal line = EMA of MACD line over signal period
//! 5. Histogram = MACD line - signal line

use super::super::traits::{MultiOutputIndicator, PersistentIndicator};

/// MACD indicator for persistent kernel execution
pub struct MacdIndicator;

/// Parameters for MACD calculation
///
/// Standard values: (12, 26, 9)
#[derive(Copy, Clone, Debug)]
pub struct MacdParams {
    pub fast_period: i32,
    pub slow_period: i32,
    pub signal_period: i32,
}

impl MacdParams {
    /// Create standard MACD parameters (12, 26, 9)
    pub fn standard() -> Self {
        Self {
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
        }
    }
}

/// CUDA kernel for persistent MACD calculation
///
/// Uses standard EMA formula: EMA = alpha * price + (1 - alpha) * prev_ema
/// where alpha = 2 / (period + 1)
const MACD_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

// MACD parameters struct (must match Rust layout)
struct MacdParams {
    int fast_period;
    int slow_period;
    int signal_period;
};

extern "C" __global__ void persistent_macd_kernel(
    const double** __restrict__ input_batch,      // Array of input pointers (close prices)
    double** __restrict__ macd_batch,             // Array of MACD line output pointers
    double** __restrict__ signal_batch,           // Array of signal line output pointers
    double** __restrict__ histogram_batch,        // Array of histogram output pointers
    const int* __restrict__ sizes,                // Array of dataset sizes
    const MacdParams* __restrict__ params,        // Array of MACD parameters
    int num_tasks                                 // Number of tasks to process
) {
    // Get grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Process each task sequentially (persistent kernel pattern)
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* close = input_batch[task_id];
        double* macd_line = macd_batch[task_id];
        double* signal_line = signal_batch[task_id];
        double* histogram = histogram_batch[task_id];
        int n = sizes[task_id];

        MacdParams p = params[task_id];
        int fast_period = p.fast_period;
        int slow_period = p.slow_period;
        int signal_period = p.signal_period;

        // Calculate EMA smoothing factors
        double fast_alpha = 2.0 / (fast_period + 1);
        double slow_alpha = 2.0 / (slow_period + 1);
        double signal_alpha = 2.0 / (signal_period + 1);

        // Sequential calculation (one thread per task for dependency handling)
        if (global_tid == task_id % grid_size) {
            // Initialize EMAs with SMA
            double fast_ema = 0.0;
            double slow_ema = 0.0;

            // Calculate initial SMA for fast period
            for (int i = 0; i < fast_period && i < n; i++) {
                fast_ema += close[i];
            }
            fast_ema /= fast_period;

            // Calculate initial SMA for slow period
            for (int i = 0; i < slow_period && i < n; i++) {
                slow_ema += close[i];
            }
            slow_ema /= slow_period;

            // Calculate MACD line (EMA difference)
            for (int i = 0; i < n; i++) {
                if (i < slow_period - 1) {
                    // Not enough data for MACD
                    macd_line[i] = CUDART_NAN;
                    signal_line[i] = CUDART_NAN;
                    histogram[i] = CUDART_NAN;
                    continue;
                }

                // Update fast EMA
                if (i >= fast_period - 1) {
                    if (i == fast_period - 1) {
                        // First EMA value is SMA
                        // Already calculated above
                    } else {
                        fast_ema = fast_alpha * close[i] + (1.0 - fast_alpha) * fast_ema;
                    }
                }

                // Update slow EMA
                if (i == slow_period - 1) {
                    // First EMA value is SMA (already calculated)
                } else {
                    slow_ema = slow_alpha * close[i] + (1.0 - slow_alpha) * slow_ema;
                }

                // Calculate MACD line
                macd_line[i] = fast_ema - slow_ema;
            }

            // Calculate signal line (EMA of MACD line)
            double signal_ema = 0.0;
            int signal_start = slow_period - 1 + signal_period - 1;

            // Initial SMA for signal line
            for (int i = slow_period - 1; i < signal_start && i < n; i++) {
                signal_ema += macd_line[i];
            }
            signal_ema /= signal_period;

            for (int i = 0; i < n; i++) {
                if (i < signal_start) {
                    signal_line[i] = CUDART_NAN;
                    histogram[i] = CUDART_NAN;
                } else if (i == signal_start) {
                    // First signal value is SMA
                    signal_line[i] = signal_ema;
                    histogram[i] = macd_line[i] - signal_line[i];
                } else {
                    // EMA of MACD line
                    signal_ema = signal_alpha * macd_line[i] + (1.0 - signal_alpha) * signal_ema;
                    signal_line[i] = signal_ema;
                    histogram[i] = macd_line[i] - signal_line[i];
                }
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for MacdIndicator {
    type Params = MacdParams;

    fn kernel_source() -> &'static str {
        MACD_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_macd_kernel"
    }

    fn num_outputs() -> usize {
        3 // Three outputs: MACD line, signal line, histogram
    }
}

impl MultiOutputIndicator for MacdIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_macd_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = MacdIndicator::compile_kernel(&device);
        assert!(result.is_ok(), "MACD kernel should compile successfully");
    }

    #[test]
    fn test_macd_trait_properties() {
        assert_eq!(MacdIndicator::kernel_name(), "persistent_macd_kernel");
        assert_eq!(MacdIndicator::num_inputs(), 1);
        assert_eq!(MacdIndicator::num_outputs(), 3);
    }

    #[test]
    fn test_macd_params() {
        let params = MacdParams::standard();
        assert_eq!(params.fast_period, 12);
        assert_eq!(params.slow_period, 26);
        assert_eq!(params.signal_period, 9);
    }
}
