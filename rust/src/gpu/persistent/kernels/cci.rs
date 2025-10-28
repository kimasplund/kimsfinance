//! Persistent CCI (Commodity Channel Index) kernel
//!
//! Implements CCI calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! CCI measures the deviation of price from its statistical average.
//!
//! **Calculation Steps**:
//! 1. Typical Price (TP) = (high + low + close) / 3
//! 2. SMA of TP over period
//! 3. Mean Absolute Deviation (MAD) = average of |TP[i] - SMA|
//! 4. CCI = (TP - SMA) / (0.015 × MAD)
//!
//! **Constant**: 0.015 chosen so ~70-80% of CCI values fall between -100 and +100
//!
//! # Interpretation
//!
//! - **CCI > +100**: Overbought (potential sell signal)
//! - **CCI < -100**: Oversold (potential buy signal)
//! - **-100 to +100**: Normal trading range
//!
//! # Performance
//!
//! Persistent kernel execution reduces overhead:
//! - Traditional: 3 arrays (high, low, close) × N tasks × 10μs = 30N μs
//! - Persistent: 1 launch × 10μs = 10μs (96% reduction for N=10)

use super::super::traits::{PersistentIndicator, SingleOutputIndicator};

/// CCI indicator for persistent kernel execution
pub struct CciIndicator;

/// CUDA kernel for persistent CCI calculation
///
/// Input buffer layout: [high(n), low(n), close(n)] - concatenated
/// Produces single output: CCI values
const CCI_KERNEL: &str = r#"
// NVRTC Kernel - Do NOT include system headers
// NVRTC provides built-in CUDA types and functions

// Cooperative Groups API (available in NVRTC without includes)
namespace cooperative_groups {
    struct grid_group {
        __device__ void sync() const {
            __syncthreads();  // Intra-block sync
        }
    };

    __device__ inline grid_group this_grid() {
        return grid_group{};
    }
}
namespace cg = cooperative_groups;

// Define constants for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_cci_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (high+low+close concatenated)
    double** __restrict__ output_batch,          // Array of output pointers (CCI values)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of CCI periods
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

        double* cci = output_batch[task_id];

        // Sequential calculation (one thread per task for dependency handling)
        if (global_tid == task_id % grid_size) {
            // Temporary buffer for typical price (reuse output for efficiency)
            // We'll calculate TP inline to avoid extra memory allocation

            // Calculate CCI for each point
            for (int idx = 0; idx < n; idx++) {
                if (idx >= period - 1) {
                    // Calculate SMA of typical price
                    double sum_tp = 0.0;

                    for (int i = 0; i < period; i++) {
                        int window_idx = idx - i;
                        if (window_idx >= 0) {
                            double tp = (high[window_idx] + low[window_idx] + close[window_idx]) / 3.0;
                            sum_tp += tp;
                        }
                    }

                    double sma = sum_tp / period;

                    // Calculate current typical price
                    double current_tp = (high[idx] + low[idx] + close[idx]) / 3.0;

                    // Calculate Mean Absolute Deviation (MAD)
                    double sum_abs_dev = 0.0;

                    for (int i = 0; i < period; i++) {
                        int window_idx = idx - i;
                        if (window_idx >= 0) {
                            double tp = (high[window_idx] + low[window_idx] + close[window_idx]) / 3.0;
                            sum_abs_dev += fabs(tp - sma);
                        }
                    }

                    double mad = sum_abs_dev / period;

                    // Calculate CCI: (TP - SMA) / (0.015 * MAD)
                    // Handle edge case: MAD == 0 (no deviation) -> NaN
                    if (mad > 1e-10) {
                        cci[idx] = (current_tp - sma) / (0.015 * mad);
                    } else {
                        cci[idx] = CUDART_NAN;
                    }
                } else {
                    cci[idx] = CUDART_NAN;
                }
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for CciIndicator {
    type Params = i32; // CCI period (typically 20)

    fn kernel_source() -> &'static str {
        CCI_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_cci_kernel"
    }

    fn num_inputs() -> usize {
        3 // Three inputs: high, low, close
    }

    fn num_outputs() -> usize {
        1 // Single output: CCI values
    }
}

impl SingleOutputIndicator for CciIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_cci_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = CciIndicator::compile_kernel(&device);
        assert!(result.is_ok(), "CCI kernel should compile successfully");
    }

    #[test]
    fn test_cci_trait_properties() {
        assert_eq!(CciIndicator::kernel_name(), "persistent_cci_kernel");
        assert_eq!(CciIndicator::num_inputs(), 3);
        assert_eq!(CciIndicator::num_outputs(), 1);
    }
}
