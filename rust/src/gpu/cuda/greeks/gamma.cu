//! Gamma Calculation Kernel
//!
//! Computes option gamma using central finite difference:
//! Gamma = (price(S+ΔS) - 2*price(S) + price(S-ΔS)) / (ΔS)²
//!
//! # Performance Target
//!
//! - 1000 options: <0.5ms
//! - Memory-bound operation with coalesced access
//!
//! # Numerical Stability
//!
//! - Gamma is always non-negative for both calls and puts
//! - Largest near ATM, approaches 0 for deep ITM/OTM
//! - Validates output: gamma ≥ 0

/// Calculate gamma for batch of options
///
/// Grid: 1D (n_options threads)
/// Block: 256 threads
///
/// # Arguments
///
/// - prices_up: Option prices at S+ΔS [n_options]
/// - prices_mid: Option prices at S [n_options]
/// - prices_down: Option prices at S-ΔS [n_options]
/// - spot_prices: Current spot prices [n_options]
/// - gammas: Output gamma values [n_options]
/// - n_options: Number of options
extern "C" __global__ void calculate_gamma_kernel(
    const double* __restrict__ prices_up,
    const double* __restrict__ prices_mid,
    const double* __restrict__ prices_down,
    const double* __restrict__ spot_prices,
    double* __restrict__ gammas,
    int n_options
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_options) return;

    // Adaptive bump size
    double S = spot_prices[idx];
    double dS = (S > 1000.0) ? (S * 0.001) : 0.01;

    // Second derivative via finite difference
    double price_up = prices_up[idx];
    double price_mid = prices_mid[idx];
    double price_down = prices_down[idx];

    double gamma = (price_up - 2.0 * price_mid + price_down) / (dS * dS);

    // Gamma must be non-negative
    gamma = fmax(0.0, gamma);

    gammas[idx] = gamma;
}
