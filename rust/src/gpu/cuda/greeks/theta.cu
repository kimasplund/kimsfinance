//! Theta Calculation Kernel
//!
//! Computes option theta (time decay) using forward finite difference:
//! Theta = -(price(t+Δt) - price(t)) / Δt
//!
//! # Performance Target
//!
//! - 1000 options: <0.5ms
//!
//! # Numerical Stability
//!
//! - Theta is typically negative for long options (time decay)
//! - Uses Δt = 1 day (1/365 year)
//! - No bounds checking (theta can be positive for deep ITM puts)

/// Calculate theta for batch of options
///
/// Grid: 1D (n_options threads)
/// Block: 256 threads
///
/// # Arguments
///
/// - prices_now: Current option prices [n_options]
/// - prices_tomorrow: Prices at t+1 day [n_options]
/// - thetas: Output theta values (per day) [n_options]
/// - n_options: Number of options
extern "C" __global__ void calculate_theta_kernel(
    const double* __restrict__ prices_now,
    const double* __restrict__ prices_tomorrow,
    double* __restrict__ thetas,
    int n_options
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_options) return;

    // Time step: 1 day
    const double dt = 1.0;

    // Forward difference (negative because price decays with time)
    double price_now = prices_now[idx];
    double price_tomorrow = prices_tomorrow[idx];

    // Theta = -(dP/dt) = -(P(t+dt) - P(t)) / dt
    double theta = -(price_tomorrow - price_now) / dt;

    thetas[idx] = theta;
}
