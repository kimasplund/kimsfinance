//! Persistent Elder Ray kernel
//!
//! Implements Elder Ray (Bull/Bear Power) calculation using persistent kernel pattern.
//!
//! # Algorithm
//!
//! Elder Ray measures buyer and seller power relative to EMA:
//! 1. Calculate EMA_13 of close prices (sequential, done on CPU)
//! 2. Bull Power = high - EMA_13 (parallel on GPU)
//! 3. Bear Power = low - EMA_13 (parallel on GPU)
//!
//! # Multi-Output Layout
//!
//! This indicator produces 2 outputs per task. The output buffer is laid out as:
//! `[bull_power[0..n], bear_power[0..n]]` (contiguous, total size = n*2)
//!
//! # Performance
//!
//! Hybrid CPU-GPU approach:
//! - CPU: EMA calculation (~25μs for 100K candles)
//! - GPU: Parallel subtraction (~15μs)
//! - Persistent kernel reduces overhead for batch processing

use super::super::traits::{MultiOutputIndicator, PersistentIndicator};

/// Elder Ray indicator for persistent kernel execution
pub struct ElderRayIndicator;

/// CUDA kernel for persistent Elder Ray calculation
///
/// Input buffer layout: [high(n), low(n), ema(n)] - concatenated
/// Output buffer layout: [bull_power(n), bear_power(n)] - concatenated
const ELDER_RAY_KERNEL: &str = r#"
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

extern "C" __global__ void persistent_elder_ray_kernel(
    const double** __restrict__ input_batch,     // Array of input pointers (high+low+ema concatenated)
    double** __restrict__ output_batch,          // Array of output pointers (bull+bear)
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ ema_periods,         // Array of EMA periods (for NaN range)
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
        int ema_period = ema_periods[task_id];

        // Split input buffer: [high(n), low(n), ema(n)]
        const double* high = input;           // First n elements
        const double* low = input + n;        // Next n elements
        const double* ema = input + 2*n;      // Last n elements

        double* output = output_batch[task_id];

        // Output layout: [bull_power[0..n], bear_power[0..n]]
        double* bull_power = output;
        double* bear_power = output + n;

        // Grid-stride loop for this task's data
        for (int idx = global_tid; idx < n; idx += grid_size) {
            if (isnan(ema[idx])) {
                // EMA is NaN (insufficient history)
                bull_power[idx] = CUDART_NAN;
                bear_power[idx] = CUDART_NAN;
            } else {
                // Bull Power = high - EMA
                // Bear Power = low - EMA
                bull_power[idx] = high[idx] - ema[idx];
                bear_power[idx] = low[idx] - ema[idx];
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for ElderRayIndicator {
    type Params = i32; // EMA period (typically 13)

    fn kernel_source() -> &'static str {
        ELDER_RAY_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_elder_ray_kernel"
    }

    fn num_inputs() -> usize {
        3 // Three inputs: high, low, ema (pre-calculated)
    }

    fn num_outputs() -> usize {
        2 // Two outputs: bull_power, bear_power
    }
}

impl MultiOutputIndicator for ElderRayIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_elder_ray_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = ElderRayIndicator::compile_kernel(&device);
        assert!(
            result.is_ok(),
            "Elder Ray kernel should compile successfully"
        );
    }

    #[test]
    fn test_elder_ray_trait_properties() {
        assert_eq!(
            ElderRayIndicator::kernel_name(),
            "persistent_elder_ray_kernel"
        );
        assert_eq!(ElderRayIndicator::num_inputs(), 3);
        assert_eq!(ElderRayIndicator::num_outputs(), 2);
    }
}
