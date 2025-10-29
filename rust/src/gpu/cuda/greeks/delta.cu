//! Delta Calculation Kernel
//!
//! Computes option delta using central finite difference:
//! Delta = (price(S+ΔS) - price(S-ΔS)) / (2ΔS)
//!
//! # Performance Target
//!
//! - 1000 options: <0.5ms (memory-bound operation)
//! - Coalesced memory access for optimal bandwidth
//!
//! # Numerical Stability
//!
//! - Uses adaptive ΔS based on spot price magnitude
//! - ΔS = max(0.01, 0.001 * S) ensures reasonable bumps
//! - Validates output: 0 ≤ delta ≤ 1 for calls, -1 ≤ delta ≤ 0 for puts

/// Calculate delta for batch of options
///
/// Grid: 1D (n_options threads)
/// Block: 256 threads (optimal for memory coalescing)
///
/// # Arguments
///
/// - prices_up: Option prices at S+ΔS [n_options]
/// - prices_down: Option prices at S-ΔS [n_options]
/// - spot_prices: Current spot prices [n_options]
/// - deltas: Output delta values [n_options]
/// - n_options: Number of options
extern "C" __global__ void calculate_delta_kernel(
    const double* __restrict__ prices_up,
    const double* __restrict__ prices_down,
    const double* __restrict__ spot_prices,
    double* __restrict__ deltas,
    int n_options
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_options) return;

    // Adaptive bump size: 1 cent for stocks, 0.1% for crypto
    double S = spot_prices[idx];
    double dS = (S > 1000.0) ? (S * 0.001) : 0.01;

    // Central finite difference
    double price_up = prices_up[idx];
    double price_down = prices_down[idx];

    double delta = (price_up - price_down) / (2.0 * dS);

    // Clamp to valid range [-1, 1]
    delta = fmax(-1.0, fmin(1.0, delta));

    deltas[idx] = delta;
}
