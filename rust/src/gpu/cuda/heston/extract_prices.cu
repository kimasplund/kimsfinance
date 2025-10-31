//! Price Extraction CUDA Kernel
//!
//! GPU-accelerated extraction of option prices from FFT output.
//!
//! # Mathematical Background
//!
//! After applying FFT to the weighted characteristic function, we extract prices using:
//!
//! Call price: C(K) = S · exp(-α·k) / π · Re[FFT(k)]
//!
//! where:
//! - S = spot price
//! - K = strike price
//! - k = ln(K/S) = log-moneyness
//! - α = damping parameter (1.5)
//! - FFT(k) = FFT output at the log-strike bin closest to k
//!
//! For puts, we use put-call parity:
//! P = C - S + K·exp(-r·T)
//!
//! # Performance Target
//!
//! - 100 options: <0.1 ms (vs 0.6 ms on CPU)
//! - Each thread processes one option independently
//!
//! # Integration
//!
//! This kernel is called after cuFFT to produce final option prices.
//! Eliminates need to download FFT results to CPU for post-processing.

#define PI 3.14159265358979323846

/// Extract option prices from FFT output
///
/// For each option:
/// 1. Compute log-moneyness k = ln(K/S)
/// 2. Find closest FFT bin to k using log-strike grid
/// 3. Extract call price: S·exp(-α·k) / π · Re[FFT[bin]]
/// 4. Convert to put if needed via put-call parity
/// 5. Ensure non-negative price
///
/// # Arguments
///
/// * fft_output_real, fft_output_imag: FFT results [n_options × n_fft]
/// * prices: Output option prices [n_options]
/// * strikes: Strike prices [n_options]
/// * spot_prices: Spot prices [n_options]
/// * expirations: Time to expiry in years [n_options]
/// * risk_free_rates: Risk-free rates [n_options]
/// * option_types: 0=Call, 1=Put [n_options]
/// * alpha: Carr-Madan damping (typically 1.5)
/// * eta: Grid spacing (typically 0.25)
/// * n_fft: FFT size (e.g., 4096)
/// * n_options: Number of options
///
/// # Thread Organization
///
/// Uses 1D thread indexing with one thread per option.
/// Each thread:
/// - Searches through n_fft FFT bins to find closest match
/// - Performs one price calculation
/// - Writes one output value
extern "C" __global__ void extract_prices(
    const double* __restrict__ fft_output_real,
    const double* __restrict__ fft_output_imag,
    double* __restrict__ prices,
    const double* __restrict__ strikes,
    const double* __restrict__ spot_prices,
    const double* __restrict__ expirations,
    const double* __restrict__ risk_free_rates,
    const int* __restrict__ option_types,
    const double alpha,
    const double eta,
    const int n_fft,
    const int n_options
) {
    // One thread per option (1D indexing)
    int option_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (option_idx >= n_options) return;

    // Load option parameters
    double K = strikes[option_idx];
    double S = spot_prices[option_idx];
    double tau = expirations[option_idx];
    double r = risk_free_rates[option_idx];
    int opt_type = option_types[option_idx];

    // Compute log-moneyness
    // k = ln(K/S)
    double k = ::log(K / S);

    // Compute log-strike grid parameters
    // The FFT outputs correspond to log-strikes: k_u = -b + lambda * u / N
    // where lambda = 2π / (η·N) and b = lambda / 2
    double lambda = 2.0 * PI / (eta * n_fft);
    double b = lambda / 2.0;

    // Find closest FFT bin to our log-strike k
    // We search for u such that k_u is closest to k
    int best_idx = n_fft / 2;  // Default to middle
    double min_distance = 1e308;  // Large initial value

    for (int u = 0; u < n_fft; u++) {
        double k_u = -b + lambda * (double)u / (double)n_fft;
        double distance = ::fabs(k_u - k);

        if (distance < min_distance) {
            min_distance = distance;
            best_idx = u;
        }
    }

    // Extract FFT value at best bin
    int fft_idx = option_idx * n_fft + best_idx;
    double fft_real = fft_output_real[fft_idx];
    double fft_imag = fft_output_imag[fft_idx];

    // Compute call price using Carr-Madan formula
    // C = S · exp(-α·k) / π · Re[FFT(k)]
    double exp_term = ::exp(-alpha * k);
    double call_price = S * exp_term / PI * fft_real;

    // Ensure non-negative
    call_price = ::fmax(0.0, call_price);

    // Convert to put if needed via put-call parity
    // P = C - S + K·exp(-r·T)
    double final_price;
    if (opt_type == 0) {
        // Call option
        final_price = call_price;
    } else {
        // Put option - apply put-call parity
        double discount_strike = K * ::exp(-r * tau);
        final_price = call_price - S + discount_strike;
        final_price = ::fmax(0.0, final_price);  // Ensure non-negative
    }

    // Store result
    prices[option_idx] = final_price;

    // DEBUG: Print first few results
    if (option_idx < 2) {
        double k_best = -b + lambda * (double)best_idx / (double)n_fft;
        printf("PRICE_EXTRACT_DEBUG [opt=%d]: K=%.2f, S=%.2f, k=%.6f, best_bin=%d, k_best=%.6f, fft=(%.6f,%.6f), call=%.4f, type=%d, final=%.4f\n",
               option_idx, K, S, k, best_idx, k_best, fft_real, fft_imag, call_price, opt_type, final_price);
    }
}
