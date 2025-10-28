//! Persistent CMF (Chaikin Money Flow) kernel
//!
//! Implements CMF calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! CMF measures accumulation/distribution over a rolling window:
//! 1. Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
//! 2. Money Flow Volume = Money Flow Multiplier * volume
//! 3. CMF = sum(Money Flow Volume, period) / sum(volume, period)
//!
//! # Interpretation
//!
//! - CMF > 0: Accumulation (buying pressure)
//! - CMF < 0: Distribution (selling pressure)
//! - Range: -1.0 to +1.0
//!
//! # Performance
//!
//! This is a FAST indicator with embarrassingly parallel rolling window operations.
//! Each thread independently calculates one CMF value.

use super::super::traits::{PersistentIndicator, SingleOutputIndicator};

/// CMF indicator for persistent kernel execution
pub struct CmfIndicator;

/// CUDA kernel for persistent CMF calculation
///
/// Input buffer layout: [high(n), low(n), close(n), volume(n)] - concatenated
const CMF_KERNEL: &str = r#"
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

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_cmf_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (high+low+close+volume concatenated)
    double** __restrict__ output_batch,          // Array of output pointers (CMF)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of CMF periods
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

        // Split input buffer: [high(n), low(n), close(n), volume(n)]
        const double* high = input;           // First n elements
        const double* low = input + n;        // Next n elements
        const double* close = input + 2*n;    // Next n elements
        const double* volume = input + 3*n;   // Last n elements

        double* cmf = output_batch[task_id];

        // Grid-stride loop for this task's data
        for (int idx = global_tid; idx < n; idx += grid_size) {
            if (idx < period - 1) {
                // Not enough history - set to NAN
                cmf[idx] = CUDART_NAN;
            } else {
                // Calculate CMF for this index
                double mfv_sum = 0.0;
                double vol_sum = 0.0;

                // Rolling window: sum over [idx - period + 1, idx]
                for (int j = 0; j < period; j++) {
                    int pos = idx - j;
                    double range = high[pos] - low[pos];

                    // Calculate Money Flow Multiplier only if range > 0
                    if (range > 1e-10) {
                        // MF Multiplier = ((close - low) - (high - close)) / range
                        double mf_mult = ((close[pos] - low[pos]) - (high[pos] - close[pos])) / range;

                        // Money Flow Volume = MF Multiplier * volume
                        mfv_sum += mf_mult * volume[pos];
                        vol_sum += volume[pos];
                    }
                    // If range is 0, skip this candle (doji - no price movement)
                }

                // Calculate CMF: sum(MF Volume) / sum(Volume)
                if (vol_sum > 1e-10) {
                    cmf[idx] = mfv_sum / vol_sum;
                } else {
                    // No volume in period - undefined
                    cmf[idx] = CUDART_NAN;
                }
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for CmfIndicator {
    type Params = i32; // CMF period (typically 20-21)

    fn kernel_source() -> &'static str {
        CMF_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_cmf_kernel"
    }

    fn num_inputs() -> usize {
        4 // Four inputs: high, low, close, volume
    }

    fn num_outputs() -> usize {
        1 // Single output: CMF values
    }
}

impl SingleOutputIndicator for CmfIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_cmf_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = CmfIndicator::compile_kernel(&device);
        assert!(result.is_ok(), "CMF kernel should compile successfully");
    }

    #[test]
    fn test_cmf_trait_properties() {
        assert_eq!(CmfIndicator::kernel_name(), "persistent_cmf_kernel");
        assert_eq!(CmfIndicator::num_inputs(), 4);
        assert_eq!(CmfIndicator::num_outputs(), 1);
    }
}
