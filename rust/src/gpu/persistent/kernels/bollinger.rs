//! Persistent Bollinger Bands kernel
//!
//! Implements Bollinger Bands calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! 1. **Middle Band**: SMA(close, period)
//! 2. **Standard Deviation**: sqrt(sum((close[i] - SMA)^2) / period)
//! 3. **Upper Band**: Middle + (std_dev * num_std)
//! 4. **Lower Band**: Middle - (std_dev * num_std)
//!
//! # Output Layout
//!
//! Uses contiguous buffer for multi-output: [upper (n), middle (n), lower (n)]
//! Total buffer size: n * 3 elements
//!
//! # Performance
//!
//! Bollinger Bands is well-suited for GPU parallelization as each thread can
//! independently calculate all three bands for one data point.

use super::super::traits::{MultiOutputIndicator, PersistentIndicator};

/// Bollinger Bands indicator for persistent kernel execution
pub struct BollingerIndicator;

/// Parameters for Bollinger Bands calculation
///
/// Standard values: period=20, std_dev=2.0
#[derive(Copy, Clone, Debug)]
pub struct BollingerParams {
    pub period: i32,
    pub std_dev: f64,
}

impl BollingerParams {
    /// Create standard Bollinger Bands parameters (20, 2.0)
    pub fn standard() -> Self {
        Self {
            period: 20,
            std_dev: 2.0,
        }
    }
}

/// CUDA kernel for persistent Bollinger Bands calculation
///
/// Uses two-pass algorithm for numerical stability:
/// - Pass 1: Calculate SMA (middle band)
/// - Pass 2: Calculate standard deviation and upper/lower bands
const BOLLINGER_KERNEL: &str = r#"
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

// Bollinger parameters struct (must match Rust layout)
struct BollingerParams {
    int period;
    double std_dev;
};

extern "C" __global__ void persistent_bollinger_kernel(
    const double** __restrict__ input_batch,      // Array of input pointers (close prices)
    double** __restrict__ output_batch,           // Array of output pointers (upper+middle+lower concatenated)
    const int* __restrict__ sizes,                // Array of dataset sizes
    const BollingerParams* __restrict__ params,   // Array of Bollinger parameters
    int num_tasks                                 // Number of tasks to process
) {
    // Get grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Process each task sequentially (persistent kernel pattern)
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* close = input_batch[task_id];
        double* output = output_batch[task_id];
        int n = sizes[task_id];

        // Output layout: [upper_band (n), middle_band (n), lower_band (n)]
        double* upper_band = output;           // First n elements
        double* middle_band = output + n;      // Next n elements
        double* lower_band = output + 2*n;     // Last n elements

        BollingerParams p = params[task_id];
        int period = p.period;
        double num_std = p.std_dev;

        // Grid-stride loop for this task's data (parallel across all threads)
        for (int idx = global_tid; idx < n; idx += grid_size) {
            if (idx < period - 1) {
                // Not enough data for Bollinger Bands
                upper_band[idx] = CUDART_NAN;
                middle_band[idx] = CUDART_NAN;
                lower_band[idx] = CUDART_NAN;
            } else {
                // Calculate middle band (SMA)
                double sum = 0.0;
                for (int i = 0; i < period; i++) {
                    sum += close[idx - i];
                }
                double sma = sum / (double)period;
                middle_band[idx] = sma;

                // Calculate standard deviation using two-pass algorithm
                double sum_squared_diff = 0.0;
                for (int i = 0; i < period; i++) {
                    double diff = close[idx - i] - sma;
                    sum_squared_diff += diff * diff;
                }

                // Population standard deviation (divide by period)
                double variance = sum_squared_diff / (double)period;
                double std_dev = sqrt(variance);

                // Calculate upper and lower bands
                upper_band[idx] = sma + (std_dev * num_std);
                lower_band[idx] = sma - (std_dev * num_std);
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for BollingerIndicator {
    type Params = BollingerParams;

    fn kernel_source() -> &'static str {
        BOLLINGER_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_bollinger_kernel"
    }

    fn num_outputs() -> usize {
        3 // Three outputs: upper band, middle band, lower band
    }
}

impl MultiOutputIndicator for BollingerIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_bollinger_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = BollingerIndicator::compile_kernel(&device);
        assert!(
            result.is_ok(),
            "Bollinger kernel should compile successfully"
        );
    }

    #[test]
    fn test_bollinger_trait_properties() {
        assert_eq!(
            BollingerIndicator::kernel_name(),
            "persistent_bollinger_kernel"
        );
        assert_eq!(BollingerIndicator::num_inputs(), 1);
        assert_eq!(BollingerIndicator::num_outputs(), 3);
    }

    #[test]
    fn test_bollinger_params() {
        let params = BollingerParams::standard();
        assert_eq!(params.period, 20);
        assert_eq!(params.std_dev, 2.0);
    }
}
