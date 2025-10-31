//! Vega Calculation Kernel
//!
//! Computes option vega using central finite difference:
//! Vega = (price(v+Δv) - price(v-Δv)) / (2Δv)
//!
//! # Performance Target
//!
//! - 1000 options: <0.5ms
//!
//! # Numerical Stability
//!
//! - Vega is always non-negative for both calls and puts
//! - Uses Δv = 0.01 (1% volatility bump)
//! - Validates output: vega ≥ 0

/// Calculate vega for batch of options
///
/// Grid: 1D (n_options threads)
/// Block: 256 threads
///
/// # Arguments
///
/// - prices_vol_up: Option prices at v+Δv [n_options]
/// - prices_vol_down: Option prices at v-Δv [n_options]
/// - vegas: Output vega values [n_options]
/// - n_options: Number of options
///
/// Note: Uses fixed Δv = 0.01 (hardcoded in kernel)
extern "C" __global__ void calculate_vega_kernel(
    const double* __restrict__ prices_vol_up,
    const double* __restrict__ prices_vol_down,
    double* __restrict__ vegas,
    int n_options
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_options) return;

    // Fixed volatility bump (0.01 variance = ~3.2% vol change for v=0.04)
    const double dv = 0.01;

    // Central finite difference
    double price_vol_up = prices_vol_up[idx];
    double price_vol_down = prices_vol_down[idx];

    double vega = (price_vol_up - price_vol_down) / (2.0 * dv);

    // Vega must be non-negative
    vega = fmax(0.0, vega);

    vegas[idx] = vega;
}
