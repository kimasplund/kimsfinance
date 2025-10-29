//! GPU Greeks Accuracy Tests
//!
//! Validates GPU Greeks calculation against CPU reference implementation.
//!
//! # Accuracy Requirements
//!
//! - Delta: <1% error vs CPU
//! - Gamma: <2% error vs CPU (second derivative has larger error)
//! - Vega: <1% error vs CPU
//! - Theta: <2% error vs CPU
//! - Rho: <1% error vs CPU
//!
//! # Run
//!
//! ```bash
//! cargo test --test greeks_gpu_test --features gpu -- --test-threads=1 --nocapture
//! ```

#[cfg(feature = "gpu")]
mod gpu_tests {
    use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
    use kimsfinance_core::quantitative::heston::{
        GreeksGpuCalculator, HestonGreeksCalculator, HestonParams, OptionQuote, OptionType,
    };
    use parking_lot::Mutex;
    use std::sync::Arc;

    fn create_test_option(strike: f64) -> OptionQuote {
        let now = chrono::Utc::now().timestamp();
        let expiry_3months = now + (90 * 24 * 3600);

        OptionQuote {
            underlying: "BTC".to_string(),
            strike,
            expiration: expiry_3months,
            option_type: OptionType::Call,
            spot_price: 48000.0,
            risk_free_rate: 0.05,
            bid: Some(2000.0),
            ask: Some(2100.0),
            last: Some(2050.0),
            implied_vol: Some(0.8),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        }
    }

    fn relative_error(actual: f64, expected: f64) -> f64 {
        if expected.abs() < 1e-6 {
            actual.abs() // Absolute error for near-zero values
        } else {
            ((actual - expected) / expected).abs() * 100.0
        }
    }

    #[test]
    fn test_greeks_gpu_accuracy_single_option() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let pricer_cpu = HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Pricer creation failed");
        let pricer_gpu = HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Pricer creation failed");

        let calculator_cpu = HestonGreeksCalculator::new(Arc::new(Mutex::new(pricer_cpu)));
        let mut calculator_gpu =
            GreeksGpuCalculator::new(device, Arc::new(Mutex::new(pricer_gpu))).expect("Calculator creation failed");

        let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).unwrap();
        let option = create_test_option(50000.0); // OTM call

        // Calculate Greeks with both methods
        let greeks_cpu = calculator_cpu
            .calculate_greeks(&params, &option)
            .expect("CPU Greeks failed");

        let greeks_gpu = calculator_gpu
            .calculate_greeks_batch(&params, &[option])
            .expect("GPU Greeks failed");
        let greeks_gpu = &greeks_gpu[0];

        // Compare results
        println!("\nCPU Greeks:");
        println!("  Delta: {:.6}", greeks_cpu.delta.unwrap());
        println!("  Gamma: {:.6}", greeks_cpu.gamma.unwrap());
        println!("  Vega:  {:.6}", greeks_cpu.vega.unwrap());
        println!("  Theta: {:.6}", greeks_cpu.theta.unwrap());
        println!("  Rho:   {:.6}", greeks_cpu.rho_greek.unwrap());

        println!("\nGPU Greeks:");
        println!("  Delta: {:.6}", greeks_gpu.delta.unwrap());
        println!("  Gamma: {:.6}", greeks_gpu.gamma.unwrap());
        println!("  Vega:  {:.6}", greeks_gpu.vega.unwrap());
        println!("  Theta: {:.6}", greeks_gpu.theta.unwrap());
        println!("  Rho:   {:.6}", greeks_gpu.rho_greek.unwrap());

        // Validate accuracy (allow for finite difference error)
        let delta_err = relative_error(greeks_gpu.delta.unwrap(), greeks_cpu.delta.unwrap());
        assert!(
            delta_err < 1.0,
            "Delta error {:.2}% exceeds 1% tolerance",
            delta_err
        );

        let gamma_err = relative_error(greeks_gpu.gamma.unwrap(), greeks_cpu.gamma.unwrap());
        assert!(
            gamma_err < 2.0,
            "Gamma error {:.2}% exceeds 2% tolerance",
            gamma_err
        );

        let vega_err = relative_error(greeks_gpu.vega.unwrap(), greeks_cpu.vega.unwrap());
        assert!(
            vega_err < 1.0,
            "Vega error {:.2}% exceeds 1% tolerance",
            vega_err
        );

        let theta_err = relative_error(greeks_gpu.theta.unwrap(), greeks_cpu.theta.unwrap());
        assert!(
            theta_err < 2.0,
            "Theta error {:.2}% exceeds 2% tolerance",
            theta_err
        );

        let rho_err = relative_error(greeks_gpu.rho_greek.unwrap(), greeks_cpu.rho_greek.unwrap());
        assert!(
            rho_err < 1.0,
            "Rho error {:.2}% exceeds 1% tolerance",
            rho_err
        );

        println!("\n✅ All Greeks within tolerance");
        println!("  Delta: {:.2}%", delta_err);
        println!("  Gamma: {:.2}%", gamma_err);
        println!("  Vega:  {:.2}%", vega_err);
        println!("  Theta: {:.2}%", theta_err);
        println!("  Rho:   {:.2}%", rho_err);
    }

    #[test]
    fn test_greeks_gpu_batch_accuracy() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let pricer_cpu = HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Pricer creation failed");
        let pricer_gpu = HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Pricer creation failed");

        let calculator_cpu = HestonGreeksCalculator::new(Arc::new(Mutex::new(pricer_cpu)));
        let mut calculator_gpu =
            GreeksGpuCalculator::new(device, Arc::new(Mutex::new(pricer_gpu))).expect("Calculator creation failed");

        let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).unwrap();

        // Test multiple strike prices
        let strikes = vec![45000.0, 47000.0, 48000.0, 49000.0, 51000.0];
        let options: Vec<OptionQuote> = strikes.iter().map(|&s| create_test_option(s)).collect();

        let greeks_cpu = calculator_cpu
            .calculate_greeks_batch(&params, &options)
            .expect("CPU Greeks failed");

        let greeks_gpu = calculator_gpu
            .calculate_greeks_batch(&params, &options)
            .expect("GPU Greeks failed");

        assert_eq!(greeks_cpu.len(), greeks_gpu.len());

        for (i, (cpu, gpu)) in greeks_cpu.iter().zip(greeks_gpu.iter()).enumerate() {
            println!("\nOption {} (K={})", i, strikes[i]);

            let delta_err = relative_error(gpu.delta.unwrap(), cpu.delta.unwrap());
            println!("  Delta error: {:.2}%", delta_err);
            assert!(delta_err < 1.0, "Delta error exceeds tolerance");

            let gamma_err = relative_error(gpu.gamma.unwrap(), cpu.gamma.unwrap());
            println!("  Gamma error: {:.2}%", gamma_err);
            assert!(gamma_err < 2.0, "Gamma error exceeds tolerance");

            let vega_err = relative_error(gpu.vega.unwrap(), cpu.vega.unwrap());
            println!("  Vega error: {:.2}%", vega_err);
            assert!(vega_err < 1.0, "Vega error exceeds tolerance");
        }

        println!("\n✅ All batch Greeks within tolerance");
    }

    #[test]
    fn test_greeks_gpu_stability() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let pricer = HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Pricer creation failed");
        let mut calculator =
            GreeksGpuCalculator::new(device, Arc::new(Mutex::new(pricer))).expect("Calculator creation failed");

        let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).unwrap();
        let option = create_test_option(48000.0); // ATM

        // Run multiple times to check for consistency
        let mut results = Vec::new();
        for _ in 0..5 {
            let greeks = calculator
                .calculate_greeks_batch(&params, &[option.clone()])
                .expect("Greeks calculation failed");
            results.push(greeks[0]);
        }

        // All results should be identical (deterministic)
        for i in 1..results.len() {
            let delta_diff = (results[i].delta.unwrap() - results[0].delta.unwrap()).abs();
            assert!(
                delta_diff < 1e-10,
                "Delta not deterministic: run 0={}, run {}={}",
                results[0].delta.unwrap(),
                i,
                results[i].delta.unwrap()
            );

            let gamma_diff = (results[i].gamma.unwrap() - results[0].gamma.unwrap()).abs();
            assert!(
                gamma_diff < 1e-10,
                "Gamma not deterministic: run 0={}, run {}={}",
                results[0].gamma.unwrap(),
                i,
                results[i].gamma.unwrap()
            );
        }

        println!("✅ GPU Greeks are deterministic across 5 runs");
    }
}
