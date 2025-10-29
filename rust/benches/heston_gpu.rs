//! GPU vs CPU Benchmark: Heston Option Pricing
//!
//! Validates 100-500x speedup target for batch option pricing.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
use kimsfinance_core::quantitative::heston::{HestonParams, OptionQuote, OptionType};
use std::sync::Arc;

/// Generate test options with various strikes
fn generate_test_options(n: usize) -> Vec<OptionQuote> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let expiration = now + (90 * 24 * 3600); // 3 months

    (0..n)
        .map(|i| {
            let strike = 40000.0 + (i as f64 * 100.0);
            OptionQuote {
                underlying: "BTC".to_string(),
                strike,
                expiration,
                option_type: OptionType::Call,
                spot_price: 42000.0,
                risk_free_rate: 0.05,
                bid: Some(2000.0),
                ask: Some(2100.0),
                last: Some(2050.0),
                implied_vol: Some(0.8),
                volume: 100.0,
                open_interest: 50.0,
                greeks: None,
            }
        })
        .collect()
}

/// CPU-based Heston pricing using FFT (same algorithm as GPU)
fn cpu_price_options(params: &HestonParams, options: &[OptionQuote]) -> Vec<f64> {
    use num_complex::Complex64;
    use std::f64::consts::PI;

    const FFT_SIZE: usize = 4096;
    const ALPHA: f64 = -1.0; // Lewis (2001) damping parameter

    let mut prices = Vec::with_capacity(options.len());

    for opt in options {
        // Time to expiration in years
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let tau = (opt.expiration - now) as f64 / (365.25 * 24.0 * 3600.0);

        if tau <= 0.0 {
            prices.push(0.0);
            continue;
        }

        // Compute characteristic function
        let mut char_func = vec![Complex64::new(0.0, 0.0); FFT_SIZE];

        // Grid spacing for phi
        let du = 0.25;

        for j in 0..FFT_SIZE {
            let u = j as f64 * du;
            let z = Complex64::new(u, -(ALPHA + 1.0));

            // Heston characteristic function computation
            let i_z = Complex64::i() * z;
            let rho_sigma_i_z = params.rho * params.sigma * i_z;
            let z_squared = z * z;

            let d_squared = rho_sigma_i_z.powi(2)
                - params.sigma.powi(2) * (2.0 * i_z - z_squared);
            let d = d_squared.sqrt();

            let g_minus = params.kappa - rho_sigma_i_z - d;
            let g_plus = params.kappa - rho_sigma_i_z + d;
            let g = g_minus / g_plus;

            let exp_neg_d_T = (-d * tau).exp();
            let one_minus_g_exp = 1.0 - g * exp_neg_d_T;
            let one_minus_exp = 1.0 - exp_neg_d_T;

            let D = g_minus / params.sigma.powi(2) *
                ((one_minus_exp) / one_minus_g_exp);

            let C = params.kappa * params.theta / params.sigma.powi(2) *
                (g_minus * tau - 2.0 * (one_minus_g_exp / g).ln());

            let iz_ln_S = i_z * opt.spot_price.ln();
            let exponent = C + D * params.v0 + iz_ln_S;

            char_func[j] = exponent.exp();
        }

        // Lewis (2001) method: integrate using cosine transform
        let k = (opt.strike / opt.spot_price).ln();
        let discount = (-opt.risk_free_rate * tau).exp();

        let mut sum = 0.0;
        for j in 0..FFT_SIZE {
            let phi_j = j as f64 * du;
            let cf_j = char_func[j];

            let denom_real = ALPHA.powi(2) + ALPHA - phi_j.powi(2);
            let denom_imag = (2.0 * ALPHA + 1.0) * phi_j;
            let denom_sq = denom_real.powi(2) + denom_imag.powi(2);

            if denom_sq > 1e-10 {
                let psi_real = (cf_j.re * denom_real + cf_j.im * denom_imag) / denom_sq;
                let cos_term = (phi_j * k).cos();
                sum += psi_real * cos_term;
            }
        }

        let call_price = opt.spot_price - discount * opt.strike * (0.5 + sum * du / PI);

        // Convert to put if needed
        let final_price = match opt.option_type {
            OptionType::Call => call_price.max(0.0),
            OptionType::Put => {
                // Put-call parity: P = C - S + K*exp(-rT)
                (call_price - opt.spot_price + opt.strike * discount).max(0.0)
            }
        };

        prices.push(final_price);
    }

    prices
}

fn bench_heston_pricing(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let mut pricer = HestonGpuPricer::new(device, 4096, 1000).expect("Failed to create HestonGpuPricer");

    let params = HestonParams::new(
        2.0,  // kappa
        0.04, // theta
        0.3,  // sigma
        -0.7, // rho
        0.04, // v0
    )
    .expect("Invalid Heston parameters");

    let mut group = c.benchmark_group("heston_pricing");

    // Benchmark different batch sizes
    for n_options in [10, 50, 100, 500, 1000].iter() {
        let options = generate_test_options(*n_options);

        group.throughput(Throughput::Elements(*n_options as u64));

        // GPU pricing
        group.bench_with_input(BenchmarkId::new("gpu", n_options), n_options, |b, _| {
            b.iter(|| {
                pricer
                    .price_options(black_box(&params), black_box(&options))
                    .expect("GPU pricing failed")
            })
        });

        // CPU pricing (baseline)
        group.bench_with_input(BenchmarkId::new("cpu", n_options), n_options, |b, _| {
            b.iter(|| cpu_price_options(black_box(&params), black_box(&options)))
        });
    }

    group.finish();
}

fn bench_heston_kernel_compilation(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    c.bench_function("heston_kernel_compile_cold", |b| {
        b.iter(|| {
            // This will hit cache after first call, so we measure warm startup
            HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Failed to create pricer")
        })
    });
}

criterion_group!(
    benches,
    bench_heston_pricing,
    bench_heston_kernel_compilation
);
criterion_main!(benches);
