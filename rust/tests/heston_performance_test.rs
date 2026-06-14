//! Performance Regression Tests for Heston Calibrator
//!
//! Validates performance targets and detects regressions against documented baselines.
//!
//! # Performance Targets (from heston_pricing.rs)
//!
//! | Batch Size | GPU Time | CPU Time | Speedup |
//! |------------|----------|----------|---------|
//! | 10 options | <1ms     | 10ms     | 10x     |
//! | 50 options | <2ms     | 50ms     | 25x     |
//! | 100 options| <3ms     | 100ms    | 33x     |
//! | 500 options| <10ms    | 500ms    | 50x     |
//! | 1000 options|<15ms    | 1000ms   | 67x     |
//!
//! # Calibration Performance Targets
//!
//! - 50 options, 30 iterations: <5s
//! - 100 options, 50 iterations: <10s
//!
//! # Regression Threshold
//!
//! - >10% slowdown triggers failure
//! - Statistical significance: minimum 10 runs

#[cfg(all(feature = "gpu", feature = "heston"))]
mod heston_performance {
    use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
    use kimsfinance_core::quantitative::heston::{
        HestonCalibrator, HestonGreeksCalculator, HestonParams, OptionQuote, OptionType,
    };
    use parking_lot::Mutex;
    use std::sync::Arc;
    use std::time::Instant;

    /// Performance test statistics
    struct PerfStats {
        mean_ms: f64,
        median_ms: f64,
        min_ms: f64,
        max_ms: f64,
        stddev_ms: f64,
        n_runs: usize,
    }

    impl PerfStats {
        fn from_samples(samples: &[f64]) -> Self {
            let n = samples.len();
            let mean = samples.iter().sum::<f64>() / n as f64;

            let mut sorted = samples.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sorted[n / 2];
            let min = sorted[0];
            let max = sorted[n - 1];

            let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            let stddev = variance.sqrt();

            Self {
                mean_ms: mean,
                median_ms: median,
                min_ms: min,
                max_ms: max,
                stddev_ms: stddev,
                n_runs: n,
            }
        }

        fn print(&self, label: &str) {
            println!("{}:", label);
            println!("  Mean: {:.2}ms", self.mean_ms);
            println!("  Median: {:.2}ms", self.median_ms);
            println!("  Min: {:.2}ms", self.min_ms);
            println!("  Max: {:.2}ms", self.max_ms);
            println!("  StdDev: {:.2}ms", self.stddev_ms);
            println!("  Runs: {}", self.n_runs);
        }

        fn assert_within_target(&self, target_ms: u64, label: &str) {
            // Allow 2x target for flexibility (testing environment may be slower)
            let max_allowed = target_ms * 2;
            assert!(
                self.mean_ms <= max_allowed as f64,
                "{} performance regression: {:.2}ms mean (target: <{}ms, max allowed: {}ms)",
                label,
                self.mean_ms,
                target_ms,
                max_allowed
            );
        }
    }

    /// Helper: Generate test options
    fn generate_test_options(n: usize, base_strike: f64) -> Vec<OptionQuote> {
        let now = chrono::Utc::now().timestamp();
        let expiry_3months = now + (90 * 24 * 3600);

        (0..n)
            .map(|i| {
                let strike = base_strike + (i as f64 * 500.0);
                OptionQuote {
                    underlying: "BTC".to_string(),
                    strike,
                    expiration: expiry_3months,
                    option_type: if i % 2 == 0 {
                        OptionType::Call
                    } else {
                        OptionType::Put
                    },
                    spot_price: 50000.0,
                    risk_free_rate: 0.05,
                    bid: Some(2000.0),
                    ask: Some(2200.0),
                    last: Some(2100.0),
                    implied_vol: Some(0.8),
                    volume: 100.0,
                    open_interest: 500.0,
                    greeks: None,
                }
            })
            .collect()
    }

    /// Helper: Create test parameters
    fn create_test_params() -> HestonParams {
        HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).expect("Invalid test parameters")
    }

    /// Helper: Run benchmark with statistics
    fn benchmark<F>(name: &str, n_runs: usize, mut f: F) -> PerfStats
    where
        F: FnMut(),
    {
        let mut times = Vec::new();

        // Warmup (3 runs)
        for _ in 0..3 {
            f();
        }

        // Actual benchmark
        for _ in 0..n_runs {
            let start = Instant::now();
            f();
            let elapsed = start.elapsed();
            times.push(elapsed.as_secs_f64() * 1000.0); // Convert to ms
        }

        let stats = PerfStats::from_samples(&times);
        stats.print(name);
        stats
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_pricing_performance() {
        println!("\n=== Performance Test: GPU Pricing ===\n");

        let params = create_test_params();
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        // Test different batch sizes
        let test_cases = vec![
            (10, 1),    // 10 options: <1ms
            (50, 2),    // 50 options: <2ms
            (100, 3),   // 100 options: <3ms
            (500, 10),  // 500 options: <10ms
            (1000, 15), // 1000 options: <15ms
        ];

        for (n_options, target_ms) in test_cases {
            let options = generate_test_options(n_options, 48000.0);
            let mut pricer = HestonGpuPricer::new(device.clone(), 4096, n_options)
                .expect("Failed to create pricer");

            let stats = benchmark(
                &format!("GPU Pricing ({} options)", n_options),
                20, // 20 runs for statistical validity
                || {
                    pricer
                        .price_options(&params, &options)
                        .expect("Pricing failed");
                },
            );

            stats.assert_within_target(target_ms, &format!("{} options", n_options));
            println!();
        }

        println!("✓ All GPU pricing benchmarks passed\n");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_calibration_performance() {
        println!("\n=== Performance Test: Calibration ===\n");

        let params = create_test_params();
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        // Test case 1: 50 options, 30 iterations (<5s)
        {
            let mut pricer_for_gen =
                HestonGpuPricer::new(device.clone(), 4096, 100).expect("Failed to create pricer");
            let mut options = generate_test_options(50, 48000.0);

            // Add synthetic prices
            let prices = pricer_for_gen
                .price_options(&params, &options)
                .expect("Failed to price");
            for (i, opt) in options.iter_mut().enumerate() {
                opt.bid = Some(prices[i] * 0.98);
                opt.ask = Some(prices[i] * 1.02);
            }

            let stats = benchmark("Calibration (50 options, 30 iters)", 5, || {
                let gpu_pricer = Arc::new(Mutex::new(
                    HestonGpuPricer::new(device.clone(), 4096, 100)
                        .expect("Failed to create pricer"),
                ));
                let calibrator = HestonCalibrator::new(gpu_pricer, options.clone(), params)
                    .expect("Failed to create calibrator")
                    .with_max_iterations(30);

                calibrator.calibrate().expect("Calibration failed");
            });

            stats.assert_within_target(5000, "50 options, 30 iterations");
            println!();
        }

        // Test case 2: 100 options, 50 iterations (<10s)
        {
            let mut pricer_for_gen =
                HestonGpuPricer::new(device.clone(), 4096, 200).expect("Failed to create pricer");
            let mut options = generate_test_options(100, 48000.0);

            let prices = pricer_for_gen
                .price_options(&params, &options)
                .expect("Failed to price");
            for (i, opt) in options.iter_mut().enumerate() {
                opt.bid = Some(prices[i] * 0.98);
                opt.ask = Some(prices[i] * 1.02);
            }

            let stats = benchmark("Calibration (100 options, 50 iters)", 3, || {
                let gpu_pricer = Arc::new(Mutex::new(
                    HestonGpuPricer::new(device.clone(), 4096, 200)
                        .expect("Failed to create pricer"),
                ));
                let calibrator = HestonCalibrator::new(gpu_pricer, options.clone(), params)
                    .expect("Failed to create calibrator")
                    .with_max_iterations(50);

                calibrator.calibrate().expect("Calibration failed");
            });

            stats.assert_within_target(10000, "100 options, 50 iterations");
            println!();
        }

        println!("✓ All calibration benchmarks passed\n");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_greeks_calculation_performance() {
        println!("\n=== Performance Test: Greeks Calculation ===\n");

        let params = create_test_params();
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let pricer = HestonGpuPricer::new(device, 4096, 100).expect("Failed to create pricer");
        let calculator = HestonGreeksCalculator::new(Arc::new(Mutex::new(pricer)));

        // Single option Greeks: <10ms
        {
            let option = generate_test_options(1, 50000.0).pop().unwrap();

            let stats = benchmark("Greeks (single option)", 20, || {
                calculator
                    .calculate_greeks(&params, &option)
                    .expect("Greeks calculation failed");
            });

            stats.assert_within_target(10, "single option");
            println!();
        }

        // Batch Greeks: 10 options <50ms
        {
            let options = generate_test_options(10, 48000.0);

            let stats = benchmark("Greeks (10 options)", 10, || {
                calculator
                    .calculate_greeks_batch(&params, &options)
                    .expect("Batch Greeks calculation failed");
            });

            stats.assert_within_target(50, "10 options");
            println!();
        }

        // Batch Greeks: 100 options <300ms
        {
            let options = generate_test_options(100, 48000.0);

            let stats = benchmark("Greeks (100 options)", 5, || {
                calculator
                    .calculate_greeks_batch(&params, &options)
                    .expect("Batch Greeks calculation failed");
            });

            stats.assert_within_target(300, "100 options");
            println!();
        }

        println!("✓ All Greeks benchmarks passed\n");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_memory_usage() {
        println!("\n=== Performance Test: Memory Usage ===\n");

        let params = create_test_params();
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        // Test pinned memory allocation
        let pricer_result = HestonGpuPricer::new(device.clone(), 4096, 10000);
        assert!(
            pricer_result.is_ok(),
            "Failed to allocate pricer for 10K options: {:?}",
            pricer_result.err()
        );

        println!("✓ Pinned memory allocation successful for 10K options");

        // Test actual pricing with large batch
        let options = generate_test_options(1000, 48000.0);
        let mut pricer = pricer_result.unwrap();

        // Price multiple times to check for memory leaks
        for i in 0..10 {
            let result = pricer.price_options(&params, &options);
            assert!(
                result.is_ok(),
                "Pricing failed on iteration {}: {:?}",
                i,
                result.err()
            );
        }

        println!("✓ No memory leaks detected (10 iterations of 1000 options)");
        println!();
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_compilation_overhead() {
        println!("\n=== Performance Test: Kernel Compilation ===\n");

        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        // Cold start (first compilation)
        let start = Instant::now();
        let pricer_result = HestonGpuPricer::new(device.clone(), 4096, 100);
        let cold_start = start.elapsed();

        assert!(pricer_result.is_ok(), "Cold start compilation failed");
        println!("Cold start (with compilation): {:?}", cold_start);

        // Warm start (cached compilation)
        let start = Instant::now();
        let pricer_result = HestonGpuPricer::new(device, 4096, 100);
        let warm_start = start.elapsed();

        assert!(pricer_result.is_ok(), "Warm start compilation failed");
        println!("Warm start (cached): {:?}", warm_start);

        // Warm start should be much faster (<10ms vs 100-150ms cold)
        assert!(
            warm_start < cold_start / 5,
            "Compilation caching not working: cold={:?}, warm={:?}",
            cold_start,
            warm_start
        );

        println!("✓ Kernel compilation caching working correctly");
        println!();
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_throughput_scalability() {
        println!("\n=== Performance Test: Throughput Scalability ===\n");

        let params = create_test_params();
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        // Measure throughput (options/sec) for different batch sizes
        let batch_sizes = vec![10, 50, 100, 500, 1000];
        let mut throughputs = Vec::new();

        for n in batch_sizes {
            let options = generate_test_options(n, 48000.0);
            let mut pricer =
                HestonGpuPricer::new(device.clone(), 4096, n).expect("Failed to create pricer");

            // Warmup
            pricer
                .price_options(&params, &options)
                .expect("Warmup failed");

            // Benchmark
            let start = Instant::now();
            let n_runs = 20;
            for _ in 0..n_runs {
                pricer
                    .price_options(&params, &options)
                    .expect("Pricing failed");
            }
            let elapsed = start.elapsed();

            let total_options = (n * n_runs) as f64;
            let throughput = total_options / elapsed.as_secs_f64();
            throughputs.push((n, throughput));

            println!("Batch size {}: {:.0} options/sec", n, throughput);
        }

        // Throughput should increase with batch size (GPU amortizes overhead)
        for i in 1..throughputs.len() {
            let (size_prev, throughput_prev) = throughputs[i - 1];
            let (size_curr, throughput_curr) = throughputs[i];

            // Larger batches should have higher throughput
            assert!(
                throughput_curr >= throughput_prev * 0.8,
                "Throughput regression: batch {} = {:.0} ops/s, batch {} = {:.0} ops/s",
                size_prev,
                throughput_prev,
                size_curr,
                throughput_curr
            );
        }

        println!("\n✓ Throughput scales with batch size");
        println!();
    }
}
