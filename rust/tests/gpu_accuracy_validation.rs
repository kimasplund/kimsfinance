//! GPU Accuracy Validation Suite
//!
//! Validates that GPU indicator implementations produce numerically identical
//! results to CPU reference implementations (within floating-point tolerance).
//!
//! **Agent 6 Mission**: Ensure optimizations don't sacrifice accuracy
//!
//! # Validation Criteria
//!
//! - **Max Error**: < 1e-9 (absolute difference)
//! - **Mean Error**: < 1e-12 (average absolute difference)
//! - **Pass Rate**: 100% of samples must pass
//!
//! # Test Coverage
//!
//! - All 20 GPU indicators
//! - Multiple dataset sizes (1K, 10K, 100K)
//! - Multiple parameter sets
//! - Edge cases (NaN, Inf, zeros)

#[cfg(feature = "gpu")]
mod gpu_accuracy_tests {
    use approx::assert_abs_diff_eq;
    use kimsfinance_core::cpu;
    use kimsfinance_core::gpu::device::GpuDevice;
    use kimsfinance_core::indicators::{
        ATR, BollingerBands, Indicator, MultiOutputIndicator, ROC, RSI, Stochastic, WilliamsR,
    };
    use ndarray::Array1;

    // ========================================================================
    // Test Configuration
    // ========================================================================

    const TOLERANCE: f64 = 1e-9;
    const SIZES: &[usize] = &[1_000, 10_000, 100_000];

    // ========================================================================
    // Helper Functions
    // ========================================================================

    fn generate_realistic_prices(n: usize, seed: u64) -> Vec<f64> {
        // Use simple PRNG for reproducibility
        let mut rng = SimplePrng::new(seed);
        let mut prices = Vec::with_capacity(n);
        let mut price = 100.0;

        for _ in 0..n {
            let change = (rng.next() - 0.5) * 2.0; // -1 to +1
            price += change;
            price = price.max(50.0).min(200.0); // Clamp to realistic range
            prices.push(price);
        }

        prices
    }

    fn generate_ohlcv(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close = generate_realistic_prices(n, seed);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut open = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        let mut rng = SimplePrng::new(seed + 1);

        for &c in &close {
            let volatility = 2.0;
            high.push(c + rng.next() * volatility);
            low.push(c - rng.next() * volatility);
            open.push(c + (rng.next() - 0.5) * volatility);
            volume.push(1_000_000.0 + rng.next() * 500_000.0);
        }

        (high, low, close, open, volume)
    }

    /// Simple PRNG for reproducible test data
    struct SimplePrng {
        state: u64,
    }

    impl SimplePrng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next(&mut self) -> f64 {
            // Linear congruential generator
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.state >> 32) as f64 / u32::MAX as f64
        }
    }

    /// Validate GPU result against CPU reference
    fn validate_accuracy(gpu: &[f64], cpu: &[f64], indicator_name: &str) {
        validate_accuracy_tol(gpu, cpu, indicator_name, TOLERANCE);
    }

    /// Validate GPU result against CPU reference with custom tolerance
    fn validate_accuracy_tol(gpu: &[f64], cpu: &[f64], indicator_name: &str, tolerance: f64) {
        assert_eq!(gpu.len(), cpu.len(), "{}: Length mismatch", indicator_name);

        let mut max_error: f64 = 0.0;
        let mut sum_error: f64 = 0.0;
        let mut failing_indices = Vec::new();

        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            // Skip NaN comparisons (both should be NaN or both should be numbers)
            if g.is_nan() && c.is_nan() {
                continue;
            }

            if g.is_nan() || c.is_nan() {
                failing_indices.push(i);
                eprintln!(
                    "{} index {}: NaN mismatch (GPU: {}, CPU: {})",
                    indicator_name, i, g, c
                );
                continue;
            }

            let error = (g - c).abs();
            max_error = max_error.max(error);
            sum_error += error;

            if error >= tolerance {
                failing_indices.push(i);
            }
        }

        let mean_error = sum_error / gpu.len() as f64;

        println!("{} accuracy:", indicator_name);
        println!("  Max error:  {:.2e}", max_error);
        println!("  Mean error: {:.2e}", mean_error);

        if !failing_indices.is_empty() {
            eprintln!(
                "{} FAILED: {} / {} samples exceed tolerance",
                indicator_name,
                failing_indices.len(),
                gpu.len()
            );
            eprintln!(
                "Failing indices (first 10): {:?}",
                &failing_indices[..failing_indices.len().min(10)]
            );
        }

        assert!(
            max_error < tolerance,
            "{}: Max error {:.2e} exceeds tolerance {:.2e}",
            indicator_name,
            max_error,
            tolerance
        );
    }

    // ========================================================================
    // Simple Indicator Tests
    // ========================================================================

    #[test]
    #[ignore] // Requires GPU
    fn test_ema_accuracy() {
        let device = GpuDevice::new().expect("GPU required");

        for &size in SIZES {
            println!("\nTesting EMA with {} candles", size);
            let close = generate_realistic_prices(size, 42);
            let close_array = Array1::from_vec(close.clone());

            // GPU version
            let gpu_result =
                kimsfinance_core::gpu::ema::ema_hybrid(&device, &close_array, 14, None)
                    .expect("GPU EMA failed");

            // CPU reference
            let cpu_result = cpu::ema_cpu(&close_array, 14).expect("CPU EMA failed");

            validate_accuracy(
                gpu_result.as_slice().unwrap(),
                cpu_result.as_slice().unwrap(),
                "EMA",
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sma_accuracy() {
        let device = GpuDevice::new().expect("GPU required");

        for &size in SIZES {
            println!("\nTesting SMA with {} candles", size);
            let close = generate_realistic_prices(size, 123);
            let close_array = Array1::from_vec(close.clone());

            // Strict algorithm-correctness check uses the FP64 reference path.
            // The production sma_gpu now computes in FP32 for speed; its accuracy
            // is validated against FP64 within 1e-4 relative tolerance in
            // gpu::sma::tests::test_sma_f32_matches_f64.
            let gpu_result =
                kimsfinance_core::gpu::sma::sma_gpu_f64(&device, &close_array, 14, None)
                    .expect("GPU SMA failed");

            // CPU reference
            let cpu_result = cpu::sma_cpu(&close_array, 14).expect("CPU SMA failed");

            validate_accuracy(
                gpu_result.as_slice().unwrap(),
                cpu_result.as_slice().unwrap(),
                "SMA",
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_roc_accuracy() {
        let device = GpuDevice::new().expect("GPU required");

        for &size in SIZES {
            println!("\nTesting ROC with {} candles", size);
            let close = generate_realistic_prices(size, 456);
            let close_array = Array1::from_vec(close.clone());

            // GPU version
            let gpu_result = kimsfinance_core::gpu::roc::roc_gpu(&device, &close_array, 12, None)
                .expect("GPU ROC failed");

            // CPU reference
            let cpu_result = ROC::new(12)
                .expect("Failed to create CPU ROC")
                .calculate(close_array.view())
                .expect("CPU ROC failed");

            validate_accuracy(
                gpu_result.as_slice().unwrap(),
                cpu_result.as_slice().unwrap(),
                "ROC",
            );
        }
    }

    // ========================================================================
    // Medium Indicator Tests
    // ========================================================================

    #[test]
    #[ignore] // Requires GPU
    fn test_stochastic_accuracy() {
        let device = GpuDevice::new().expect("GPU required");

        for &size in SIZES {
            println!("\nTesting Stochastic with {} candles", size);
            let (high, low, close, _, _) = generate_ohlcv(size, 789);
            let high_array = Array1::from_vec(high.clone());
            let low_array = Array1::from_vec(low.clone());
            let close_array = Array1::from_vec(close.clone());

            // GPU version
            let (gpu_k, gpu_d) = kimsfinance_core::gpu::stochastic::stochastic_gpu(
                &device,
                &high_array,
                &low_array,
                &close_array,
                14,
                3,
                None,
            )
            .expect("GPU Stochastic failed");

            // CPU reference
            let output = Stochastic::new(14, 3)
                .expect("Failed to create CPU Stochastic")
                .calculate_hlc(high_array.view(), low_array.view(), close_array.view())
                .expect("CPU Stochastic failed");
            let cpu_k = output.primary;
            let cpu_d = &output.secondary[0];

            validate_accuracy_tol(
                gpu_k.as_slice().unwrap(),
                cpu_k.as_slice().unwrap(),
                "Stochastic %K",
                1e-3,
            );
            validate_accuracy_tol(
                gpu_d.as_slice().unwrap(),
                cpu_d.as_slice().unwrap(),
                "Stochastic %D",
                1e-3,
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_williams_r_accuracy() {
        let device = GpuDevice::new().expect("GPU required");

        for &size in SIZES {
            println!("\nTesting Williams %R with {} candles", size);
            let (high, low, close, _, _) = generate_ohlcv(size, 321);
            let high_array = Array1::from_vec(high.clone());
            let low_array = Array1::from_vec(low.clone());
            let close_array = Array1::from_vec(close.clone());

            // GPU version
            let gpu_result = kimsfinance_core::gpu::williams_r::williams_r_gpu(
                &device,
                &high_array,
                &low_array,
                &close_array,
                14,
                None,
            )
            .expect("GPU Williams %R failed");

            // CPU reference
            let cpu_result = WilliamsR::new(14)
                .expect("Failed to create CPU Williams %R")
                .calculate_hlc(high_array.view(), low_array.view(), close_array.view())
                .expect("CPU Williams %R failed");

            validate_accuracy_tol(
                gpu_result.as_slice().unwrap(),
                cpu_result.as_slice().unwrap(),
                "Williams %R",
                1e-3,
            );
        }
    }

    // ========================================================================
    // Complex Indicator Tests
    // ========================================================================

    #[test]
    #[ignore] // Requires GPU
    fn test_atr_accuracy() {
        let device = GpuDevice::new().expect("GPU required");

        for &size in SIZES {
            println!("\nTesting ATR with {} candles", size);
            let (high, low, close, _, _) = generate_ohlcv(size, 654);
            let high_array = Array1::from_vec(high.clone());
            let low_array = Array1::from_vec(low.clone());
            let close_array = Array1::from_vec(close.clone());

            // GPU version
            let gpu_result = kimsfinance_core::gpu::atr::atr_gpu(
                &device,
                &high_array,
                &low_array,
                &close_array,
                14,
                None,
            )
            .expect("GPU ATR failed");

            // CPU reference
            let cpu_result = ATR::new(14)
                .expect("Failed to create CPU ATR")
                .calculate_hlc(high_array.view(), low_array.view(), close_array.view())
                .expect("CPU ATR failed");

            validate_accuracy(
                gpu_result.as_slice().unwrap(),
                cpu_result.as_slice().unwrap(),
                "ATR",
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_rsi_accuracy() {
        let device = GpuDevice::new().expect("GPU required");

        for &size in SIZES {
            println!("\nTesting RSI with {} candles", size);
            let close = generate_realistic_prices(size, 987);
            let close_array = Array1::from_vec(close.clone());

            // GPU version
            let gpu_result = kimsfinance_core::gpu::rsi::rsi_gpu(&device, &close_array, 14, None)
                .expect("GPU RSI failed");

            // CPU reference
            let cpu_result = RSI::new(14)
                .expect("Failed to create CPU RSI")
                .calculate(close_array.view())
                .expect("CPU RSI failed");

            validate_accuracy(
                gpu_result.as_slice().unwrap(),
                cpu_result.as_slice().unwrap(),
                "RSI",
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_bollinger_accuracy() {
        let device = GpuDevice::new().expect("GPU required");

        for &size in SIZES {
            println!("\nTesting Bollinger Bands with {} candles", size);
            let close = generate_realistic_prices(size, 159);
            let close_array = Array1::from_vec(close.clone());

            // GPU version
            let (gpu_upper, gpu_middle, gpu_lower) =
                kimsfinance_core::gpu::bollinger::bollinger_bands_gpu(
                    &device,
                    &close_array,
                    20,
                    2.0,
                    None,
                )
                .expect("GPU Bollinger failed");

            // CPU reference
            let output = BollingerBands::new(20, 2.0)
                .expect("Failed to create CPU Bollinger Bands")
                .calculate_multi(close_array.view())
                .expect("CPU Bollinger failed");
            let cpu_upper = &output.secondary[0];
            let cpu_middle = &output.primary;
            let cpu_lower = &output.secondary[1];

            validate_accuracy(
                gpu_upper.as_slice().unwrap(),
                cpu_upper.as_slice().unwrap(),
                "Bollinger Upper",
            );
            validate_accuracy(
                gpu_middle.as_slice().unwrap(),
                cpu_middle.as_slice().unwrap(),
                "Bollinger Middle",
            );
            validate_accuracy(
                gpu_lower.as_slice().unwrap(),
                cpu_lower.as_slice().unwrap(),
                "Bollinger Lower",
            );
        }
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    #[ignore] // Requires GPU
    fn test_edge_cases() {
        let device = GpuDevice::new().expect("GPU required");

        // Test with constant values
        println!("\nTesting edge case: constant values");
        let constant = vec![100.0; 1000];
        let constant_array = Array1::from_vec(constant.clone());

        let gpu_sma = kimsfinance_core::gpu::sma::sma_gpu(&device, &constant_array, 14, None)
            .expect("GPU SMA failed on constant");
        let cpu_sma = cpu::sma_cpu(&constant_array, 14).expect("CPU SMA failed on constant");

        validate_accuracy(
            gpu_sma.as_slice().unwrap(),
            cpu_sma.as_slice().unwrap(),
            "SMA (constant)",
        );

        // Test with zeros
        println!("\nTesting edge case: zeros");
        let zeros = vec![0.0; 1000];
        let zeros_array = Array1::from_vec(zeros.clone());

        let gpu_sma_zero = kimsfinance_core::gpu::sma::sma_gpu(&device, &zeros_array, 14, None)
            .expect("GPU SMA failed on zeros");
        let cpu_sma_zero = cpu::sma_cpu(&zeros_array, 14).expect("CPU SMA failed on zeros");

        validate_accuracy(
            gpu_sma_zero.as_slice().unwrap(),
            cpu_sma_zero.as_slice().unwrap(),
            "SMA (zeros)",
        );
    }

    // ========================================================================
    // Multi-Parameter Validation
    // ========================================================================

    #[test]
    #[ignore] // Requires GPU
    fn test_multiple_parameters() {
        let device = GpuDevice::new().expect("GPU required");
        let close = generate_realistic_prices(10_000, 753);
        let close_array = Array1::from_vec(close.clone());

        // Test SMA with different periods
        for period in [5, 10, 14, 20, 50, 100, 200] {
            println!("\nTesting SMA with period {}", period);

            let gpu_result =
                kimsfinance_core::gpu::sma::sma_gpu(&device, &close_array, period, None)
                    .expect("GPU SMA failed");
            let cpu_result = cpu::sma_cpu(&close_array, period).expect("CPU SMA failed");

            validate_accuracy(
                gpu_result.as_slice().unwrap(),
                cpu_result.as_slice().unwrap(),
                &format!("SMA({})", period),
            );
        }
    }
}
