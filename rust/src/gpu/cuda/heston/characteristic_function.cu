//! Heston Characteristic Function CUDA Kernel
//!
//! GPU-accelerated computation of the Heston characteristic function for option pricing.
//!
//! # Mathematical Background
//!
//! The Heston characteristic function φ(u) is:
//!
//! φ(u) = exp(C(τ,u) + D(τ,u)v₀ + iu·ln(S₀))
//!
//! Where:
//! - D(τ,u) = (b - ρσu*i - d) / σ² · (1 - e^(-dτ)) / (1 - ge^(-dτ))
//! - C(τ,u) = r·u·i·τ + (κθ/σ²)[(b - ρσu*i - d)τ - 2ln((1 - ge^(-dτ))/(1 - g))]
//! - d = √[(ρσu*i - b)² - σ²(2u*i·a - u²)]
//! - g = (b - ρσu*i - d) / (b - ρσu*i + d)
//!
//! # Performance Target
//!
//! - Batch size 100 options, 4096 FFT points: <3ms
//! - 100-500x speedup vs CPU for calibration workloads

// CUDA built-in math functions (no header needed with NVRTC)

// Complex number structure for manual complex arithmetic
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
    
    // Complex addition
    __device__ __forceinline__ Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    // Complex subtraction
    __device__ __forceinline__ Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }
    
    // Complex division: (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
    __device__ __forceinline__ Complex operator/(const Complex& other) const {
        double denom = other.real * other.real + other.imag * other.imag;
        return Complex(
            (real * other.real + imag * other.imag) / denom,
            (imag * other.real - real * other.imag) / denom
        );
    }
    
    // Complex square root (principal branch)
    __device__ __forceinline__ Complex sqrt() const {
        double r = ::sqrt(real * real + imag * imag);
        double theta = atan2(imag, real);
        double sqrt_r = ::sqrt(r);
        return Complex(sqrt_r * cos(theta / 2.0), sqrt_r * sin(theta / 2.0));
    }
    
    // Complex exponential: e^(a + bi) = e^a(cos(b) + i·sin(b))
    __device__ __forceinline__ Complex exp() const {
        double exp_real = ::exp(real);
        return Complex(exp_real * cos(imag), exp_real * sin(imag));
    }
    
    // Complex natural logarithm (principal branch)
    __device__ __forceinline__ Complex log() const {
        double r = ::sqrt(real * real + imag * imag);
        double theta = atan2(imag, real);
        return Complex(::log(r), theta);
    }
};

// Scalar * Complex
__device__ __forceinline__ Complex operator*(double scalar, const Complex& c) {
    return Complex(scalar * c.real, scalar * c.imag);
}

// Complex * Scalar
__device__ __forceinline__ Complex operator*(const Complex& c, double scalar) {
    return Complex(c.real * scalar, c.imag * scalar);
}

/// Heston characteristic function kernel (batched for multiple options)
///
/// Computes φ(u) for each (option, frequency) pair in parallel.
///
/// # Arguments
///
/// - kappa: Mean reversion speed
/// - theta: Long-term variance
/// - sigma: Volatility of volatility
/// - rho: Correlation
/// - v0: Initial variance
/// - strikes: Option strikes [n_options]
/// - expirations: Time to expiry in years [n_options]
/// - spot_prices: Current spot prices [n_options]
/// - risk_free_rates: Risk-free rates [n_options]
/// - n_fft: Number of FFT points (power of 2, e.g., 4096)
/// - phi_values: Integration points [n_fft]
/// - char_func_real: Output real part [n_options * n_fft]
/// - char_func_imag: Output imaginary part [n_options * n_fft]
/// - n_options: Number of options
extern "C" __global__ void heston_characteristic_function(
    const double kappa,
    const double theta,
    const double sigma,
    const double rho,
    const double v0,
    const double* __restrict__ strikes,
    const double* __restrict__ expirations,
    const double* __restrict__ spot_prices,
    const double* __restrict__ risk_free_rates,
    const int n_fft,
    const double* __restrict__ phi_values,
    double* __restrict__ char_func_real,
    double* __restrict__ char_func_imag,
    const int n_options
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options * n_fft) return;
    
    int option_idx = idx / n_fft;
    int phi_idx = idx % n_fft;
    
    double K = strikes[option_idx];
    double T = expirations[option_idx];
    double S = spot_prices[option_idx];
    double r = risk_free_rates[option_idx];
    double u = phi_values[phi_idx];
    
    // Heston characteristic function computation
    // φ(u) = exp(C(T,u) + D(T,u)v₀ + iu·ln(S))
    
    // Define constants for Heston formula
    const double a = 0.5; // Parameter for characteristic function form
    
    // Complex arithmetic: u → iu (multiply by i)
    Complex iu = Complex(0.0, u);
    
    // b = kappa + λ - ρσ (for risk-neutral measure, λ = 0)
    double b = kappa - rho * sigma * u;
    
    // Compute d = √[(ρσu*i - b)² - σ²(2u*i·a - u²)]
    // First compute the terms inside the square root
    Complex rho_sigma_u_i = Complex(0.0, rho * sigma * u);
    Complex term1 = rho_sigma_u_i - Complex(b, 0.0);
    Complex term1_sq = term1 * term1;
    
    Complex two_u_i_a = Complex(0.0, 2.0 * u * a);
    Complex u_sq = Complex(u * u, 0.0);
    Complex term2 = two_u_i_a - u_sq;
    Complex term2_scaled = Complex(sigma * sigma, 0.0) * term2;
    
    Complex d_squared = term1_sq - term2_scaled;
    Complex d = d_squared.sqrt();
    
    // Compute g = (b - ρσu*i - d) / (b - ρσu*i + d)
    Complex b_complex = Complex(b, 0.0);
    Complex numerator_g = b_complex - rho_sigma_u_i - d;
    Complex denominator_g = b_complex - rho_sigma_u_i + d;
    Complex g = numerator_g / denominator_g;
    
    // Compute e^(-d·T)
    Complex exp_neg_d_T = (d * (-T)).exp();
    
    // Compute D(T,u) = (b - ρσu*i - d) / σ² · (1 - e^(-dT)) / (1 - g·e^(-dT))
    Complex one = Complex(1.0, 0.0);
    Complex numerator_D_frac = one - exp_neg_d_T;
    Complex denominator_D_frac = one - g * exp_neg_d_T;
    Complex D_frac = numerator_D_frac / denominator_D_frac;
    
    double sigma_sq = sigma * sigma;
    Complex D = numerator_g / Complex(sigma_sq, 0.0) * D_frac;
    
    // Compute C(T,u) = r·u·i·T + (κθ/σ²)[(b - ρσu*i - d)T - 2ln((1 - g·e^(-dT))/(1 - g))]
    Complex r_u_i_T = Complex(0.0, r * u * T);
    
    double kappa_theta_over_sigma_sq = kappa * theta / sigma_sq;
    Complex term_C1 = numerator_g * T;
    
    Complex one_minus_g = one - g;
    Complex ln_numerator = one - g * exp_neg_d_T;
    Complex ln_term = (ln_numerator / one_minus_g).log();
    Complex term_C2 = ln_term * 2.0;
    
    Complex C = r_u_i_T + Complex(kappa_theta_over_sigma_sq, 0.0) * (term_C1 - term_C2);
    
    // Compute φ(u) = exp(C + D·v₀ + iu·ln(S))
    Complex D_v0 = D * v0;
    Complex iu_ln_S = Complex(0.0, u * ::log(S));
    
    Complex exponent = C + D_v0 + iu_ln_S;
    Complex phi = exponent.exp();
    
    // Store results
    char_func_real[idx] = phi.real;
    char_func_imag[idx] = phi.imag;
}
