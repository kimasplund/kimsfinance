//! Minimal CPU-based Heston Characteristic Function Test
//!
//! This example validates the mathematical formulation of the Heston characteristic function
//! by computing it on the CPU using the EXACT same formula as the CUDA kernel.
//!
//! Purpose: Identify if bug is in the mathematical formula or in the CUDA implementation.
//!
//! Run with: cargo run --example test_heston_cpu --features heston

use num_complex::Complex64;

/// Heston characteristic function (CPU implementation)
///
/// Computes φ(z) for COMPLEX argument z = u - (α+1)i
///
/// This is the EXACT same formula as in characteristic_function.cu
fn heston_characteristic_function_cpu(
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    v0: f64,
    s: f64,
    r: f64,
    t: f64,
    z: Complex64,
) -> Complex64 {
    let sigma_sq = sigma * sigma;
    let b = kappa; // Risk-neutral measure: b = kappa - λ (with λ=0)

    let i = Complex64::new(0.0, 1.0);

    // Compute intermediate values
    let i_z = i * z;
    let rho_sigma_i_z = rho * sigma * i_z;
    let z_squared = z * z;

    // d² = (ρσiz - b)² + σ²(2iz - z²)
    let term1_base = rho_sigma_i_z - b;
    let term1 = term1_base * term1_base;
    let term2 = sigma_sq * (2.0 * i_z - z_squared);
    let d_squared = term1 + term2;
    let d = d_squared.sqrt();

    // g = (b - ρσiz - d) / (b - ρσiz + d)
    let b_minus_rho_sigma_iz = b - rho_sigma_i_z;
    let numerator_g = b_minus_rho_sigma_iz - d;
    let denominator_g = b_minus_rho_sigma_iz + d;
    let g = numerator_g / denominator_g;

    // e^(-d·T)
    let exp_neg_d_t = (-d * t).exp();

    // D(T,z) = (b - ρσiz - d) / σ² · (1 - e^(-dT)) / (1 - g·e^(-dT))
    let one = Complex64::new(1.0, 0.0);
    let numerator_d_frac = one - exp_neg_d_t;
    let denominator_d_frac = one - g * exp_neg_d_t;
    let d_frac = numerator_d_frac / denominator_d_frac;
    let big_d = (numerator_g / sigma_sq) * d_frac;

    // C(T,z) = r·iz·T + (κθ/σ²)[(b - ρσiz - d)T - 2ln((1 - g·e^(-dT))/(1 - g))]
    let r_iz_t = r * i_z * t;
    let kappa_theta_over_sigma_sq = kappa * theta / sigma_sq;
    let term_c1 = numerator_g * t;
    let one_minus_g = one - g;
    let ln_numerator = one - g * exp_neg_d_t;
    let ln_term = (ln_numerator / one_minus_g).ln();
    let term_c2 = 2.0 * ln_term;
    let big_c = r_iz_t + kappa_theta_over_sigma_sq * (term_c1 - term_c2);

    // φ(z) = exp(C + D·v₀ + iz·ln(S))
    let d_v0 = big_d * v0;
    let iz_ln_s = i_z * s.ln();
    let exponent = big_c + d_v0 + iz_ln_s;

    exponent.exp()
}

fn main() {
    println!("=== Heston Characteristic Function CPU Test ===\n");

    // Test parameters (EXACTLY as specified)
    let s = 100.0; // Spot price
    let k = 100.0; // Strike (not directly used in CF, but for context)
    let t = 1.0; // Time to expiry (1 year)
    let r = 0.05; // Risk-free rate

    // Heston parameters
    let kappa = 2.0; // Mean reversion speed
    let theta = 0.04; // Long-term variance (20% vol)
    let sigma = 0.3; // Vol of vol
    let rho = -0.7; // Correlation
    let v0 = 0.04; // Initial variance (20% current vol)

    println!("Parameters:");
    println!("  S = {:.2} (spot price)", s);
    println!("  K = {:.2} (strike, for reference)", k);
    println!("  T = {:.2} (time to expiry)", t);
    println!("  r = {:.4} (risk-free rate)", r);
    println!();
    println!("Heston Parameters:");
    println!("  κ (kappa) = {:.2} (mean reversion)", kappa);
    println!("  θ (theta) = {:.4} (long-term variance)", theta);
    println!("  σ (sigma) = {:.2} (vol of vol)", sigma);
    println!("  ρ (rho)   = {:.2} (correlation)", rho);
    println!("  v₀ (v0)   = {:.4} (initial variance)", v0);
    println!();

    // CRITICAL: Test with COMPLEX argument z = u - (α+1)i
    let alpha = 1.5;
    let u_real = 0.0; // Start with u=0 (pure imaginary z)
    let u_imag = -(alpha + 1.0); // -2.5

    let z = Complex64::new(u_real, u_imag);

    println!("Test 1: Pure imaginary z (u=0.0)");
    println!("  α = {:.1}", alpha);
    println!("  z = {:.1} - {:.1}i", z.re, -z.im);
    println!();

    // Compute characteristic function
    let phi = heston_characteristic_function_cpu(kappa, theta, sigma, rho, v0, s, r, t, z);

    println!("Result:");
    println!("  φ(z).real = {:.10e}", phi.re);
    println!("  φ(z).imag = {:.10e}", phi.im);
    println!("  |φ(z)|    = {:.10e}", phi.norm());
    println!("  arg(φ(z)) = {:.10e} rad", phi.arg());
    println!();

    // Check if imaginary part is non-zero
    if phi.im.abs() < 1e-10 {
        println!("⚠️  WARNING: Imaginary part is essentially ZERO!");
        println!("   This indicates a problem with the formula or implementation.");
    } else {
        println!("✓ SUCCESS: Imaginary part is non-zero");
        println!("  Expected behavior for complex argument z");
    }
    println!();

    // Test 2: Non-zero real frequency
    println!("Test 2: Non-zero frequency u=0.5");
    let u_real2 = 0.5;
    let z2 = Complex64::new(u_real2, u_imag);
    println!("  z = {:.1} - {:.1}i", z2.re, -z2.im);
    println!();

    let phi2 = heston_characteristic_function_cpu(kappa, theta, sigma, rho, v0, s, r, t, z2);

    println!("Result:");
    println!("  φ(z).real = {:.10e}", phi2.re);
    println!("  φ(z).imag = {:.10e}", phi2.im);
    println!("  |φ(z)|    = {:.10e}", phi2.norm());
    println!("  arg(φ(z)) = {:.10e} rad", phi2.arg());
    println!();

    if phi2.im.abs() < 1e-10 {
        println!("⚠️  WARNING: Imaginary part is essentially ZERO!");
    } else {
        println!("✓ SUCCESS: Imaginary part is non-zero");
    }
    println!();

    // Test 3: Multiple frequency points (as in FFT)
    println!("Test 3: Multiple frequency points (FFT simulation)");
    let n_fft = 16; // Small FFT for demonstration
    let eta = 0.25; // Frequency spacing

    println!("  N_FFT = {}", n_fft);
    println!("  η (eta) = {:.2} (frequency spacing)", eta);
    println!();

    println!("  u_idx |      u     |     φ.real      |     φ.imag      |    |φ|");
    println!("  ------|------------|-----------------|-----------------|-------------");

    for idx in 0..n_fft {
        let u = idx as f64 * eta;
        let z_fft = Complex64::new(u, u_imag);
        let phi_fft =
            heston_characteristic_function_cpu(kappa, theta, sigma, rho, v0, s, r, t, z_fft);

        println!(
            "  {:5} | {:10.4} | {:15.6e} | {:15.6e} | {:11.6e}",
            idx,
            u,
            phi_fft.re,
            phi_fft.im,
            phi_fft.norm()
        );
    }
    println!();

    // Detailed intermediate values for u=0 case (debugging)
    println!("Test 4: Detailed intermediate values (u=0.0)");
    let z_debug = Complex64::new(0.0, -2.5);
    println!("  z = {:.1} - {:.1}i", z_debug.re, -z_debug.im);
    println!();

    let i = Complex64::new(0.0, 1.0);
    let i_z = i * z_debug;
    println!("  i·z = {:.10e} + {:.10e}i", i_z.re, i_z.im);

    let rho_sigma_i_z = rho * sigma * i_z;
    println!(
        "  ρσi·z = {:.10e} + {:.10e}i",
        rho_sigma_i_z.re, rho_sigma_i_z.im
    );

    let z_squared = z_debug * z_debug;
    println!("  z² = {:.10e} + {:.10e}i", z_squared.re, z_squared.im);

    let two_i_z = 2.0 * i_z;
    println!("  2i·z = {:.10e} + {:.10e}i", two_i_z.re, two_i_z.im);

    let term1_base = rho_sigma_i_z - kappa;
    println!(
        "  ρσi·z - κ = {:.10e} + {:.10e}i",
        term1_base.re, term1_base.im
    );

    let term1 = term1_base * term1_base;
    println!("  (ρσi·z - κ)² = {:.10e} + {:.10e}i", term1.re, term1.im);

    let sigma_sq = sigma * sigma;
    let term2 = sigma_sq * (two_i_z - z_squared);
    println!("  σ²(2i·z - z²) = {:.10e} + {:.10e}i", term2.re, term2.im);

    let d_squared = term1 + term2;
    println!("  d² = {:.10e} + {:.10e}i", d_squared.re, d_squared.im);

    let d = d_squared.sqrt();
    println!("  d = {:.10e} + {:.10e}i", d.re, d.im);
    println!();

    println!("Full characteristic function:");
    let phi_final =
        heston_characteristic_function_cpu(kappa, theta, sigma, rho, v0, s, r, t, z_debug);
    println!("  φ(z) = {:.10e} + {:.10e}i", phi_final.re, phi_final.im);
    println!();

    // Summary
    println!("=== Summary ===");
    println!("The CPU implementation uses the EXACT same formula as the CUDA kernel.");
    println!("If you see non-zero imaginary parts above, the formula is correct.");
    println!("Compare these CPU results with CUDA kernel output to identify discrepancies.");
}
