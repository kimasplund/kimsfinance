//! Heston Characteristic Function CUDA Kernel
//!
//! GPU-accelerated computation of the Heston characteristic function for option pricing.
//!
//! # Mathematical Background
//!
//! The Heston characteristic function φ(z) for COMPLEX argument z is:
//!
//! φ(z) = exp(C(τ,z) + D(τ,z)v₀ + iz·ln(S₀))
//!
//! Where:
//! - D(τ,z) = (b - ρσz*i - d) / σ² · (1 - e^(-dτ)) / (1 - ge^(-dτ))
//! - C(τ,z) = r·z·i·τ + (κθ/σ²)[(b - ρσz*i - d)τ - 2ln((1 - ge^(-dτ))/(1 - g))]
//! - d = √[(ρσz*i - b)² - σ²(2z*i - z²)]
//! - g = (b - ρσz*i - d) / (b - ρσz*i + d)
//!
//! # Carr-Madan FFT Integration
//!
//! For Carr-Madan FFT, we evaluate φ at COMPLEX argument:
//! z = u - (α+1)i
//!
//! where:
//! - u = real frequency variable (0, η, 2η, ...)
//! - α = 1.5 (damping parameter)
//! - Thus z = u - 2.5i
//!
//! This is CRITICAL for getting non-zero imaginary parts!
//!
//! # Performance Target
//!
//! - Batch size 100 options, 4096 FFT points: <3ms
//! - 100-500x speedup vs CPU for calibration workloads
//!
//! # Debug Mode
//!
//! This kernel includes comprehensive printf debugging (thread 0 only).
//! All intermediate complex values are printed with "CUDA_DEBUG:" prefix.
//! Use this to trace where imaginary parts become zero.

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
/// Computes φ(u - (α+1)i) for each (option, frequency) pair in parallel.
///
/// CRITICAL: This evaluates the characteristic function at COMPLEX arguments
/// as required by the Carr-Madan FFT formula. The imaginary part -(α+1) ensures
/// the Fourier transform converges and produces meaningful option prices.
///
/// # Arguments
///
/// - kappa: Mean reversion speed
/// - theta: Long-term variance
/// - sigma: Volatility of volatility
/// - rho: Correlation
/// - v0: Initial variance
/// - alpha: Carr-Madan damping parameter (typically 1.5)
/// - strikes: Option strikes [n_options]
/// - expirations: Time to expiry in years [n_options]
/// - spot_prices: Current spot prices [n_options]
/// - risk_free_rates: Risk-free rates [n_options]
/// - n_fft: Number of FFT points (power of 2, e.g., 4096)
/// - phi_values: REAL frequency points [n_fft]
/// - char_func_real: Output real part [n_options * n_fft]
/// - char_func_imag: Output imaginary part [n_options * n_fft]
/// - n_options: Number of options
extern "C" __global__ void heston_characteristic_function(
    const double kappa,
    const double theta,
    const double sigma,
    const double rho,
    const double v0,
    const double alpha,
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
    // 2D thread indexing (zero overhead)
    int option_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int phi_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Early exit with 2D bounds check
    if (option_idx >= n_options || phi_idx >= n_fft) return;

    // Compute linear index for debug prints and output
    int idx = option_idx * n_fft + phi_idx;

    double K = strikes[option_idx];
    double T = expirations[option_idx];
    double S = spot_prices[option_idx];
    double r = risk_free_rates[option_idx];
    
    // CRITICAL FIX: Construct COMPLEX argument z = u - (α+1)i
    // This is required by Carr-Madan FFT formula!
    double u_real = phi_values[phi_idx];
    double u_imag = -(alpha + 1.0);  // Typically -2.5 for α=1.5
    Complex z = Complex(u_real, u_imag);

    // Print debug for both u=0 (idx=0) and u≠0 (idx=1) to see difference
    bool debug_print = (idx == 0 || idx == 1);

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d, phi_idx=%d]: Initial z = (%f, %f)\n", idx, phi_idx, z.real, z.imag);
        printf("CUDA_DEBUG [idx=%d]: u_real=%f (from phi_values), u_imag=%f (=-(alpha+1))\n",
               idx, u_real, u_imag);
        if (idx == 0) {
            printf("CUDA_DEBUG [idx=%d]: Parameters: kappa=%f, theta=%f, sigma=%f, rho=%f, v0=%f, alpha=%f\n",
                   idx, kappa, theta, sigma, rho, v0, alpha);
            printf("CUDA_DEBUG [idx=%d]: Option params: S=%f, K=%f, T=%f, r=%f\n", idx, S, K, T, r);
        }
    }

    // Heston characteristic function computation for COMPLEX z
    // φ(z) = exp(C(T,z) + D(T,z)v₀ + iz·ln(S))
    // Standard Heston formula (Heston 1993, Carr-Madan 1999)

    double sigma_sq = sigma * sigma;

    // For risk-neutral measure: b = kappa - λ = kappa (with λ=0)
    double b = kappa;
    Complex b_complex = Complex(b, 0.0);

    // Compute i (imaginary unit)
    Complex i_unit = Complex(0.0, 1.0);

    // Compute iz (i times z)
    Complex i_z = i_unit * z;

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d]: i_z = i * z = (%f, %f)\n", idx, i_z.real, i_z.imag);
    }

    // Compute ρσiz
    Complex rho_sigma_i_z = Complex(rho * sigma, 0.0) * i_z;

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d]: rho_sigma_i_z = rho*sigma*i_z = (%f, %f)\n",
               idx, rho_sigma_i_z.real, rho_sigma_i_z.imag);
    }

    // Compute z² (z squared)
    Complex z_squared = z * z;

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d]: z_squared = (%f, %f)\n", idx, z_squared.real, z_squared.imag);
    }

    // GATHERAL (2005) NUMERICALLY STABLE FORMULATION
    // Critical fix for "Little Heston Trap" (Albrecher et al. 2007)
    //
    // Compute d² = (ρσiz - b)² - σ²(z² - 2iz)
    // Note: Standard formula rearranged for numerical stability
    Complex term1_base = rho_sigma_i_z - b_complex;
    Complex term1 = term1_base * term1_base;

    // σ²(z² - 2iz) - note sign change for stability
    Complex two_i_z = i_z * 2.0;
    Complex inner = z_squared - two_i_z;  // z² - 2iz (note order!)
    Complex term2 = Complex(sigma_sq, 0.0) * inner;

    Complex d_squared = term1 - term2;  // Subtract for Gatheral formulation

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d]: d_squared = (%f, %f)\n", idx, d_squared.real, d_squared.imag);
    }

    // Gatheral branch cut selection: Choose branch with Re(d) > 0
    Complex d_raw = d_squared.sqrt();

    // If Re(d) < 0, flip the sign (choose the other branch)
    Complex d = (d_raw.real < 0.0) ? Complex(-d_raw.real, -d_raw.imag) : d_raw;

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d]: d = sqrt(d_squared) = (%f, %f)\n", idx, d.real, d.imag);
    }

    // g = (b - ρσiz - d) / (b - ρσiz + d)
    Complex b_minus_rho_sigma_iz = b_complex - rho_sigma_i_z;
    Complex numerator_g = b_minus_rho_sigma_iz - d;
    Complex denominator_g = b_minus_rho_sigma_iz + d;
    Complex g = numerator_g / denominator_g;

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d]: g = (%f, %f)\n", idx, g.real, g.imag);
    }

    // e^(-d·T)
    Complex exp_neg_d_T = (d * (-T)).exp();

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d]: exp_neg_d_T = exp(-d*T) = (%f, %f)\n",
               idx, exp_neg_d_T.real, exp_neg_d_T.imag);
    }

    // D(T,z) = (b - ρσiz - d) / σ² · (1 - e^(-dT)) / (1 - g·e^(-dT))
    Complex one = Complex(1.0, 0.0);
    Complex numerator_D_frac = one - exp_neg_d_T;
    Complex denominator_D_frac = one - g * exp_neg_d_T;
    Complex D_frac = numerator_D_frac / denominator_D_frac;
    Complex D = numerator_g / Complex(sigma_sq, 0.0) * D_frac;

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d]: D = (%f, %f)\n", idx, D.real, D.imag);
    }

    // C(T,z) = r·iz·T + (κθ/σ²)[(b - ρσiz - d)T - 2ln((1 - g·e^(-dT))/(1 - g))]
    Complex r_iz_T = Complex(r * T, 0.0) * i_z;

    double kappa_theta_over_sigma_sq = kappa * theta / sigma_sq;
    Complex term_C1 = numerator_g * T;

    Complex one_minus_g = one - g;
    Complex ln_numerator = one - g * exp_neg_d_T;
    Complex ln_term = (ln_numerator / one_minus_g).log();
    Complex term_C2 = ln_term * 2.0;

    Complex C = r_iz_T + Complex(kappa_theta_over_sigma_sq, 0.0) * (term_C1 - term_C2);

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d]: C = (%f, %f)\n", idx, C.real, C.imag);
    }

    // φ(z) = exp(C + D·v₀ + iz·ln(S))
    Complex D_v0 = D * v0;
    Complex iz_ln_S = i_z * ::log(S);

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d]: D_v0 = (%f, %f)\n", idx, D_v0.real, D_v0.imag);
        printf("CUDA_DEBUG [idx=%d]: iz_ln_S = (%f, %f)\n", idx, iz_ln_S.real, iz_ln_S.imag);
    }

    Complex exponent = C + D_v0 + iz_ln_S;

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d]: exponent = C + D*v0 + iz*ln(S) = (%f, %f)\n",
               idx, exponent.real, exponent.imag);
    }

    Complex phi = exponent.exp();

    if (debug_print) {
        printf("CUDA_DEBUG [idx=%d]: phi = exp(exponent) = (%f, %f)\n", idx, phi.real, phi.imag);
    }
    
    // Store results
    char_func_real[idx] = phi.real;
    char_func_imag[idx] = phi.imag;

    // DEBUG: Verify write (only first few threads)
    if (idx < 2) {
        printf("CUDA_DEBUG [idx=%d]: AFTER WRITE - char_func_real[%d] = %.6f, char_func_imag[%d] = %.6f\n",
               idx, idx, char_func_real[idx], idx, char_func_imag[idx]);

        // IMMEDIATE readback without syncthreads to avoid race
        double immediate_readback_real = char_func_real[idx];
        double immediate_readback_imag = char_func_imag[idx];
        printf("CUDA_DEBUG [idx=%d]: IMMEDIATE READBACK - real=%.6f, imag=%.6f\n",
               idx, immediate_readback_real, immediate_readback_imag);
    }
}
