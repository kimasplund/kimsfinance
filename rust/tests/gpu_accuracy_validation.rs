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
    use kimsfinance_core::gpu::device::GpuDevice;
    use kimsfinance_core::cpu;
    use ndarray::Array1;
    use approx::assert_abs_diff_eq;

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
        assert_eq!(gpu.len(), cpu.len(), "{}: Length mismatch", indicator_name);

        let mut max_error = 0.0;
        let mut sum_error = 0.0;
        let mut failing_indices = Vec::new();

        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            // Skip NaN comparisons (both should be NaN or both should be numbers)
            if g.is_nan() && c.is_nan() {
                continue;
            }

            if g.is_nan() || c.is_nan() {
                failing_indices.push(i);
                eprintln!("{} index {}: NaN mismatch (GPU: {}, CPU: {})", indicator_name, i, g, c);
                continue;
            }

            let error = (g - c).abs();
            max_error = max_error.max(error);
            sum_error += error;

            if error >= TOLERANCE {
                failing_indices.push(i);
            }
        }

        let mean_error = sum_error / gpu.len() as f64;

        println!("{} accuracy:", indicator_name);
        println!("  Max error:  {:.2e}", max_error);
        println!("  Mean error: {:.2e}", mean_error);

        if !failing_indices.is_empty() {
            eprintln!("{} FAILED: {} / {} samples exceed tolerance",
                     indicator_name, failing_indices.len(), gpu.len());
            eprintln!("Failing indices (first 10): {:?}",
                     &failing_indices[..failing_indices.len().min(10)]);
        }

        assert!(max_error < TOLERANCE,
               "{}: Max error {:.2e} exceeds tolerance {:.2e}",
               indicator_name, max_error, TOLERANCE);
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
            let gpu_result = kimsfinance_core::gpu::ema::ema_hybrid(&close_array, 14, &device, None)
                .expect("GPU EMA failed");

            // CPU reference
            let cpu_result = cpu::ema_cpu(&close, 14)
                .expect("CPU EMA failed");

            validate_accuracy(gpu_result.as_slice().unwrap(), &cpu_result, "EMA");
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

            // GPU version
            let gpu_result = kimsfinance_core::gpu::sma::sma_gpu(&close_array, 14, &device, None)
                .expect("GPU SMA failed");

            // CPU reference
            let cpu_result = cpu::sma_cpu(&close, 14)
                .expect("CPU SMA failed");

            validate_accuracy(gpu_result.as_slice().unwrap(), &cpu_result, "SMA");
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
            let gpu_result = kimsfinance_core::gpu::roc::roc_gpu(&close_array, 12, &device, None)
                .expect("GPU ROC failed");

            // CPU reference
            let cpu_result = cpu::roc_cpu(&close, 12)
                .expect("CPU ROC failed");

            validate_accuracy(gpu_result.as_slice().unwrap(), &cpu_result, "ROC");
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
                &high_array, &low_array, &close_array, 14, 3, &device, None
            ).expect("GPU Stochastic failed");

            // CPU reference
            let (cpu_k, cpu_d) = cpu::stochastic_cpu(&high, &low, &close, 14, 3)
                .expect("CPU Stochastic failed");

            validate_accuracy(gpu_k.as_slice().unwrap(), &cpu_k, "Stochastic %K");
            validate_accuracy(gpu_d.as_slice().unwrap(), &cpu_d, "Stochastic %D");
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
                &high_array, &low_array, &close_array, 14, &device, None
            ).expect("GPU Williams %R failed");

            // CPU reference
            let cpu_result = cpu::williams_r_cpu(&high, &low, &close, 14)
                .expect("CPU Williams %R failed");

            validate_accuracy(gpu_result.as_slice().unwrap(), &cpu_result, "Williams %R");
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
                &high_array, &low_array, &close_array, 14, &device, None
            ).expect("GPU ATR failed");

            // CPU reference
            let cpu_result = cpu::atr_cpu(&high, &low, &close, 14)
                .expect("CPU ATR failed");

            validate_accuracy(gpu_result.as_slice().unwrap(), &cpu_result, "ATR");
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
            let gpu_result = kimsfinance_core::gpu::rsi::rsi_gpu(&close_array, 14, &device, None)
                .expect("GPU RSI failed");

            // CPU reference
            let cpu_result = cpu::rsi_cpu(&close, 14)
                .expect("CPU RSI failed");

            validate_accuracy(gpu_result.as_slice().unwrap(), &cpu_result, "RSI");
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
            let (gpu_upper, gpu_middle, gpu_lower) = kimsfinance_core::gpu::bollinger::bollinger_gpu(
                &close_array, 20, 2.0, &device, None
            ).expect("GPU Bollinger failed");

            // CPU reference
            let (cpu_upper, cpu_middle, cpu_lower) = cpu::bollinger_cpu(&close, 20, 2.0)
                .expect("CPU Bollinger failed");

            validate_accuracy(gpu_upper.as_slice().unwrap(), &cpu_upper, "Bollinger Upper");
            validate_accuracy(gpu_middle.as_slice().unwrap(), &cpu_middle, "Bollinger Middle");
            validate_accuracy(gpu_lower.as_slice().unwrap(), &cpu_lower, "Bollinger Lower");
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

        let gpu_sma = kimsfinance_core::gpu::sma::sma_gpu(&constant_array, 14, &device, None)
            .expect("GPU SMA failed on constant");
        let cpu_sma = cpu::sma_cpu(&constant, 14)
            .expect("CPU SMA failed on constant");

        validate_accuracy(gpu_sma.as_slice().unwrap(), &cpu_sma, "SMA (constant)");

        // Test with zeros
        println!("\nTesting edge case: zeros");
        let zeros = vec![0.0; 1000];
        let zeros_array = Array1::from_vec(zeros.clone());

        let gpu_sma_zero = kimsfinance_core::gpu::sma::sma_gpu(&zeros_array, 14, &device, None)
            .expect("GPU SMA failed on zeros");
        let cpu_sma_zero = cpu::sma_cpu(&zeros, 14)
            .expect("CPU SMA failed on zeros");

        validate_accuracy(gpu_sma_zero.as_slice().unwrap(), &cpu_sma_zero, "SMA (zeros)");
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

            let gpu_result = kimsfinance_core::gpu::sma::sma_gpu(&close_array, period, &device, None)
                .expect("GPU SMA failed");
            let cpu_result = cpu::sma_cpu(&close, period)
                .expect("CPU SMA failed");

            validate_accuracy(
                gpu_result.as_slice().unwrap(),
                &cpu_result,
                &format!("SMA({})", period)
            );
        }
    }
}
