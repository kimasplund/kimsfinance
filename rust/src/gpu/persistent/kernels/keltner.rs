//! Persistent Keltner Channels kernel
//!
//! Implements Keltner Channels calculation using persistent kernel pattern for batch processing.
//!
//! # Algorithm
//!
//! Keltner Channels are volatility-based envelopes around an EMA:
//!
//! 1. **Middle Line** = EMA(close, ema_period)
//! 2. **ATR** = Average True Range(high, low, close, atr_period)
//! 3. **Upper Band** = Middle + (ATR × multiplier)
//! 4. **Lower Band** = Middle - (ATR × multiplier)
//!
//! # Implementation Strategy
//!
//! This kernel handles the **parallel band calculation** only.
//! Sequential components (EMA, ATR) are calculated separately using existing kernels,
//! then combined in this kernel for optimal performance.
//!
//! # Calculation Steps
//!
//! 1. Receive pre-calculated EMA and ATR arrays
//! 2. For each candle (parallel):
//!    - middle = ema[i]
//!    - offset = atr[i] × multiplier
//!    - upper = middle + offset
//!    - lower = middle - offset
//!
//! # Performance
//!
//! Persistent kernel execution reduces overhead:
//! - Traditional: 3 arrays (high, low, close) × N tasks × 10μs = 30N μs
//! - Persistent: 1 launch × 10μs = 10μs (97% reduction for N=10)

use super::super::traits::{MultiOutputIndicator, PersistentIndicator};

/// Keltner Channels indicator for persistent kernel execution
pub struct KeltnerIndicator;

/// Parameters for Keltner Channels calculation
///
/// Standard values: (20, 10, 2.0)
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct KeltnerParams {
    pub ema_period: i32,
    pub atr_period: i32,
    pub multiplier: f64,
}

impl KeltnerParams {
    /// Create standard Keltner parameters (20, 10, 2.0)
    pub fn standard() -> Self {
        Self {
            ema_period: 20,
            atr_period: 10,
            multiplier: 2.0,
        }
    }
}

/// CUDA kernel for persistent Keltner Channels calculation
///
/// NOTE: This kernel assumes EMA and ATR are pre-calculated and passed as inputs.
/// The full pipeline would be:
/// 1. Calculate EMA(close) on CPU or separate GPU kernel
/// 2. Calculate ATR(high, low, close) using ATR persistent kernel
/// 3. Launch this kernel to compute bands
///
/// Input layout: [ema (n), atr (n)] - concatenated in single input buffer
/// Output layout: [upper (n), middle (n), lower (n)] - concatenated
const KELTNER_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define constants for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

// Keltner parameters struct (must match Rust layout)
struct KeltnerParams {
    int ema_period;
    int atr_period;
    double multiplier;
};

extern "C" __global__ void persistent_keltner_kernel(
    const double** __restrict__ input_batch,      // Array of input pointers [ema+atr concatenated]
    double** __restrict__ output_batch,           // Array of output pointers (upper+middle+lower concatenated)
    const int* __restrict__ sizes,                // Array of dataset sizes
    const KeltnerParams* __restrict__ params,     // Array of Keltner parameters
    int num_tasks                                 // Number of tasks to process
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
        KeltnerParams p = params[task_id];

        // Input layout: [ema (n), atr (n)]
        const double* ema = input;          // First n elements
        const double* atr = input + n;      // Next n elements

        // Output layout: [upper (n), middle (n), lower (n)]
        double* upper = output;             // First n elements
        double* middle = output + n;        // Next n elements
        double* lower = output + 2*n;       // Last n elements

        // Parallel calculation: each thread handles multiple indices
        for (int idx = global_tid; idx < n; idx += grid_size) {
            // Check if both EMA and ATR are valid
            if (!isnan(ema[idx]) && !isnan(atr[idx])) {
                middle[idx] = ema[idx];
                double offset = atr[idx] * p.multiplier;
                upper[idx] = ema[idx] + offset;
                lower[idx] = ema[idx] - offset;
            } else {
                // Not enough data yet
                upper[idx] = CUDART_NAN;
                middle[idx] = CUDART_NAN;
                lower[idx] = CUDART_NAN;
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for KeltnerIndicator {
    type Params = KeltnerParams;

    fn kernel_source() -> &'static str {
        KELTNER_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_keltner_kernel"
    }

    fn num_inputs() -> usize {
        1 // Single input: [ema+atr concatenated]
    }

    fn num_outputs() -> usize {
        3 // Three outputs: upper, middle, lower
    }
}

impl MultiOutputIndicator for KeltnerIndicator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_keltner_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = KeltnerIndicator::compile_kernel(&device);
        assert!(
            result.is_ok(),
            "Keltner kernel should compile successfully"
        );
    }

    #[test]
    fn test_keltner_trait_properties() {
        assert_eq!(
            KeltnerIndicator::kernel_name(),
            "persistent_keltner_kernel"
        );
        assert_eq!(KeltnerIndicator::num_inputs(), 1);
        assert_eq!(KeltnerIndicator::num_outputs(), 3);
    }

    #[test]
    fn test_keltner_params() {
        let params = KeltnerParams::standard();
        assert_eq!(params.ema_period, 20);
        assert_eq!(params.atr_period, 10);
        assert_eq!(params.multiplier, 2.0);
    }
}
