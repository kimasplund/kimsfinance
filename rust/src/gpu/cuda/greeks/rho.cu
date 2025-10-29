//! Rho Calculation Kernel
//!
//! Computes option rho (interest rate sensitivity) using central finite difference:
//! Rho = (price(r+Δr) - price(r-Δr)) / (2Δr)
//!
//! # Performance Target
//!
//! - 1000 options: <0.5ms
//!
//! # Numerical Stability
//!
//! - Rho is typically positive for calls, negative for puts
//! - Uses Δr = 0.01 (1% rate bump)
//! - No bounds checking (rho can be any value)

/// Calculate rho for batch of options
///
/// Grid: 1D (n_options threads)
/// Block: 256 threads
///
/// # Arguments
///
/// - prices_rate_up: Option prices at r+Δr [n_options]
/// - prices_rate_down: Option prices at r-Δr [n_options]
/// - rhos: Output rho values [n_options]
/// - n_options: Number of options
extern "C" __global__ void calculate_rho_kernel(
    const double* __restrict__ prices_rate_up,
    const double* __restrict__ prices_rate_down,
    double* __restrict__ rhos,
    int n_options
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_options) return;

    // Fixed interest rate bump (1%)
    const double dr = 0.01;

    // Central finite difference
    double price_rate_up = prices_rate_up[idx];
    double price_rate_down = prices_rate_down[idx];

    double rho = (price_rate_up - price_rate_down) / (2.0 * dr);

    rhos[idx] = rho;
}
