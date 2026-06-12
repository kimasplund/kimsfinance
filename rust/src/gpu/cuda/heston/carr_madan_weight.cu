//! Carr-Madan Weighting CUDA Kernel
//!
//! GPU-accelerated weighting of characteristic function for FFT-based option pricing.
//!
//! # Mathematical Background
//!
//! The Carr-Madan FFT method requires weighting the characteristic function:
//!
//! ψ(φ) = exp(-r·T) · φ₁(φ - (α+1)i) / (α² + α - φ² + i(2α+1)φ)
//!
//! where:
//! - φ₁ = Heston characteristic function (already computed by GPU)
//! - α = damping parameter (typically 1.5)
//! - r = risk-free rate
//! - T = time to expiry
//! - φ = frequency variable (0, η, 2η, ..., where η ≈ 0.25)
//!
//! Additionally, we apply Simpson's rule weighting for numerical integration.
//!
//! # Performance Target
//!
//! - 100 options × 4096 FFT points: <0.1 ms (vs 0.6 ms on CPU)
//! - Eliminates need to download 6.5 MB of CF data to CPU
//!
//! # Integration
//!
//! This kernel is called after characteristic_function kernel and before cuFFT.
//! It transforms raw CF values into FFT-ready input.

// Complex number structure (matches characteristic_function.cu)
struct Complex {
    double real;
    double imag;

    __device__ __forceinline__ Complex(double r = 0.0, double i = 0.0) : real(r), imag(i) {}

    // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    __device__ __forceinline__ Complex operator*(const Complex& other) const {
        return Complex(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }

    // Complex multiplication by scalar
    __device__ __forceinline__ Complex operator*(double scalar) const {
        return Complex(real * scalar, imag * scalar);
    }

    // Complex division: (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
    __device__ __forceinline__ Complex operator/(const Complex& other) const {
        double denom = other.real * other.real + other.imag * other.imag;
        return Complex(
            (real * other.real + imag * other.imag) / denom,
            (imag * other.real - real * other.imag) / denom
        );
    }
};

/// Apply Carr-Madan weighting to characteristic function
///
/// Transforms raw CF values into FFT input by applying:
/// 1. Discount factor: exp(-r·T)
/// 2. Denominator division: α² + α - φ² + i(2α+1)φ
/// 3. Simpson's rule weighting: (η/3) × [0.5, 4, 2, 4, 2, ..., 4, 2, 0.5]
/// 4. Clamp Inf/NaN values to zero (numerical stability)
///
/// # Arguments
///
/// * char_func_real, char_func_imag: Input CF [n_options × n_fft]
/// * weighted_real, weighted_imag: Output weighted CF [n_options × n_fft]
/// * risk_free_rates: Risk-free rates [n_options]
/// * expirations: Time to expiry in years [n_options]
/// * alpha: Carr-Madan damping parameter (typically 1.5)
/// * eta: Grid spacing in frequency space (typically 0.25)
/// * n_fft: FFT size (e.g., 4096)
/// * n_options: Number of options to price
///
/// # Thread Organization
///
/// Uses 2D thread indexing:
/// - X dimension: FFT frequency index (phi_idx)
/// - Y dimension: Option index (option_idx)
///
/// Each thread computes one weighted CF value.
extern "C" __global__ void carr_madan_weight(
    const double* __restrict__ char_func_real,
    const double* __restrict__ char_func_imag,
    double* __restrict__ weighted_real,
    double* __restrict__ weighted_imag,
    const double* __restrict__ risk_free_rates,
    const double* __restrict__ expirations,
    const double alpha,
    const double eta,
    const int n_fft,
    const int n_options
) {
    // 2D thread indexing (matches characteristic_function kernel)
    int option_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int phi_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (option_idx >= n_options || phi_idx >= n_fft) return;

    // Linear index for array access
    int idx = option_idx * n_fft + phi_idx;

    // Load option parameters
    double r = risk_free_rates[option_idx];
    double tau = expirations[option_idx];

    // Compute frequency: φ = phi_idx × η
    double phi = phi_idx * eta;

    // Load characteristic function value
    Complex cf(char_func_real[idx], char_func_imag[idx]);

    // Step 1: Apply discount factor
    // discount = exp(-r·T)
    double discount = ::exp(-r * tau);

    // Step 2: Compute denominator for Carr-Madan formula
    // denominator = α² + α - φ² + i(2α+1)φ
    double denom_real = alpha * alpha + alpha - phi * phi;
    double denom_imag = (2.0 * alpha + 1.0) * phi;
    Complex denominator(denom_real, denom_imag);

    // Step 3: Apply Carr-Madan transformation
    // psi = discount · cf / denominator
    Complex psi = (cf * discount) / denominator;

    // Step 4: Apply Simpson's rule weighting
    // weight = 0.5 (endpoints), 4 (odd indices), 2 (even indices)
    double simpson_weight;
    if (phi_idx == 0 || phi_idx == n_fft - 1) {
        simpson_weight = 0.5;
    } else if (phi_idx % 2 == 1) {
        simpson_weight = 4.0;
    } else {
        simpson_weight = 2.0;
    }

    // Apply Simpson's weight and grid spacing
    // weighted_psi = psi × simpson_weight × (η/3)
    Complex weighted_psi = psi * simpson_weight * eta / 3.0;

    // Step 5: Numerical stability - clamp Inf/NaN to zero
    // This prevents FFT overflow from high-frequency instabilities
    double out_real, out_imag;

    if (::isinf(weighted_psi.real) || ::isnan(weighted_psi.real)) {
        out_real = 0.0;
    } else {
        out_real = weighted_psi.real;
    }

    if (::isinf(weighted_psi.imag) || ::isnan(weighted_psi.imag)) {
        out_imag = 0.0;
    } else {
        out_imag = weighted_psi.imag;
    }

    // Store results
    weighted_real[idx] = out_real;
    weighted_imag[idx] = out_imag;
}
