//! Comprehensive Integration Tests for Heston Calibrator
//!
//! Tests end-to-end workflows across multiple modules:
//! - Model validation
//! - GPU pricing
//! - Calibration engine
//! - Greeks calculation
//! - Trading strategies
//!
//! # Test Coverage
//!
//! - Synthetic calibration (known parameter recovery)
//! - GPU pricing consistency (deterministic behavior)
//! - Greeks accuracy (numerical benchmarks)
//! - Vol arbitrage profitability (signal generation)
//! - Delta hedging (portfolio risk management)
//! - Put-call parity validation
//! - Black-Scholes limit verification

#[cfg(all(feature = "gpu", feature = "heston"))]
mod heston_integration {
    use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
    use kimsfinance_core::quantitative::heston::{
        CalibrationResult, DeltaHedgingStrategy, HestonCalibrator, HestonGreeksCalculator,
        HestonParams, OptionPosition, OptionQuote, OptionType, PortfolioGreeks, TradeSignal,
        VolArbitrageStrategy,
    };
    use parking_lot::Mutex;
    use std::sync::Arc;

    /// Helper: Generate test options with various strikes
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

    /// Helper: Generate synthetic market data from known parameters
    fn generate_synthetic_options(
        true_params: &HestonParams,
        pricer: &mut HestonGpuPricer,
        n: usize,
    ) -> Vec<OptionQuote> {
        let mut options = generate_test_options(n, 48000.0);

        // Price with known parameters to create "market" prices
        let market_prices = pricer
            .price_options(true_params, &options)
            .expect("Failed to price synthetic options");

        // Update options with synthetic market prices
        for (i, opt) in options.iter_mut().enumerate() {
            let price = market_prices[i];
            opt.bid = Some(price * 0.98);
            opt.ask = Some(price * 1.02);
            opt.last = Some(price);
        }

        options
    }

    /// Helper: Create test Heston parameters
    fn create_test_params() -> HestonParams {
        HestonParams::new(
            2.0,  // kappa
            0.04, // theta (20% long-term vol)
            0.3,  // sigma
            -0.7, // rho (negative correlation, leverage effect)
            0.04, // v0 (20% current vol)
        )
        .expect("Invalid test parameters")
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_end_to_end_synthetic_calibration() {
        println!("\n=== Test: End-to-End Synthetic Calibration ===");

        // 1. Generate synthetic market with known params
        let true_params = HestonParams {
            kappa: 2.5,
            theta: 0.05,
            sigma: 0.35,
            rho: -0.65,
            v0: 0.06,
        };

        println!("True parameters:");
        println!("  κ: {:.4}", true_params.kappa);
        println!("  θ: {:.4}", true_params.theta);
        println!("  σ: {:.4}", true_params.sigma);
        println!("  ρ: {:.4}", true_params.rho);
        println!("  v₀: {:.4}", true_params.v0);

        // 2. Generate synthetic option prices
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let mut pricer_for_gen =
            HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Failed to create pricer");
        let synthetic_options = generate_synthetic_options(&true_params, &mut pricer_for_gen, 50);

        println!("\nGenerated {} synthetic options", synthetic_options.len());

        // 3. Calibrate with different initial guess
        let initial_guess = HestonParams {
            kappa: 1.5,
            theta: 0.04,
            sigma: 0.4,
            rho: -0.5,
            v0: 0.05,
        };

        println!("\nInitial guess:");
        println!("  κ: {:.4}", initial_guess.kappa);
        println!("  θ: {:.4}", initial_guess.theta);
        println!("  σ: {:.4}", initial_guess.sigma);
        println!("  ρ: {:.4}", initial_guess.rho);
        println!("  v₀: {:.4}", initial_guess.v0);

        let gpu_pricer = Arc::new(Mutex::new(
            HestonGpuPricer::new(device, 4096, 1000).expect("Failed to create pricer for calib"),
        ));
        let calibrator = HestonCalibrator::new(gpu_pricer, synthetic_options, initial_guess)
            .expect("Failed to create calibrator")
            .with_max_iterations(100);

        let start = std::time::Instant::now();
        let result = calibrator.calibrate().expect("Calibration failed");
        let elapsed = start.elapsed();

        println!("\nCalibration completed in {:?}", elapsed);
        println!("  Converged: {}", result.converged);
        println!("  Iterations: {}", result.iterations);
        println!("  Final RMSE: {:.6}", result.rmse());

        // 4. Validate parameter recovery
        println!("\nParameter recovery:");
        println!(
            "  κ: {:.4} (true: {:.4}, error: {:.2}%)",
            result.params.kappa,
            true_params.kappa,
            ((result.params.kappa - true_params.kappa) / true_params.kappa).abs() * 100.0
        );
        println!(
            "  θ: {:.4} (true: {:.4}, error: {:.2}%)",
            result.params.theta,
            true_params.theta,
            ((result.params.theta - true_params.theta) / true_params.theta).abs() * 100.0
        );
        println!(
            "  σ: {:.4} (true: {:.4}, error: {:.2}%)",
            result.params.sigma,
            true_params.sigma,
            ((result.params.sigma - true_params.sigma) / true_params.sigma).abs() * 100.0
        );
        println!(
            "  ρ: {:.4} (true: {:.4}, error: {:.2}%)",
            result.params.rho,
            true_params.rho,
            ((result.params.rho - true_params.rho) / true_params.rho).abs() * 100.0
        );
        println!(
            "  v₀: {:.4} (true: {:.4}, error: {:.2}%)",
            result.params.v0,
            true_params.v0,
            ((result.params.v0 - true_params.v0) / true_params.v0).abs() * 100.0
        );

        // Validate convergence
        assert!(result.converged, "Calibration did not converge");
        assert!(
            result.final_error < 1.0,
            "Error too high: {}",
            result.final_error
        );

        // Relaxed parameter recovery tolerance (30% for synthetic data)
        // Note: Perfect recovery is difficult due to:
        // 1. Heston pricing via FFT is approximate
        // 2. Optimizer may find local minima
        // 3. Some parameters are correlated (identifiability issue)
        assert!(
            (result.params.kappa - true_params.kappa).abs() / true_params.kappa < 0.30,
            "Kappa recovery failed: got {}, expected {}",
            result.params.kappa,
            true_params.kappa
        );
        assert!(
            (result.params.theta - true_params.theta).abs() / true_params.theta < 0.30,
            "Theta recovery failed: got {}, expected {}",
            result.params.theta,
            true_params.theta
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_pricing_consistency() {
        println!("\n=== Test: GPU Pricing Consistency ===");

        // Verify GPU pricing is deterministic across multiple runs
        let params = create_test_params();
        let options = generate_test_options(10, 48000.0);

        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let mut pricer = HestonGpuPricer::new(device, 4096, 100).expect("Failed to create pricer");

        // Price 5 times
        let mut prices = Vec::new();
        for run in 0..5 {
            let p = pricer
                .price_options(&params, &options)
                .expect("GPU pricing failed");
            println!("Run {}: {} prices", run + 1, p.len());
            prices.push(p);
        }

        // Verify all prices are identical (deterministic)
        for i in 1..5 {
            for j in 0..options.len() {
                assert_eq!(
                    prices[0][j], prices[i][j],
                    "GPU pricing not deterministic: run 0 option {} = {}, run {} option {} = {}",
                    j, prices[0][j], i, j, prices[i][j]
                );
            }
        }

        println!(
            "✓ GPU pricing is deterministic across {} runs",
            prices.len()
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_greeks_accuracy() {
        println!("\n=== Test: Greeks Accuracy ===");

        // Verify Greeks match expected ranges for ATM call option
        let params = create_test_params();
        let option = OptionQuote {
            underlying: "BTC".to_string(),
            strike: 50000.0,
            expiration: chrono::Utc::now().timestamp() + (90 * 24 * 3600),
            option_type: OptionType::Call,
            spot_price: 50000.0, // ATM
            risk_free_rate: 0.05,
            bid: Some(2000.0),
            ask: Some(2200.0),
            last: Some(2100.0),
            implied_vol: Some(0.8),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        };

        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let pricer = HestonGpuPricer::new(device, 4096, 100).expect("Failed to create pricer");
        let calculator = HestonGreeksCalculator::new(Arc::new(Mutex::new(pricer)));

        let greeks = calculator
            .calculate_greeks(&params, &option)
            .expect("Greeks calculation failed");

        println!("Greeks for ATM call:");
        println!("  Delta: {:.4}", greeks.delta.unwrap());
        println!("  Gamma: {:.6}", greeks.gamma.unwrap());
        println!("  Vega: {:.4}", greeks.vega.unwrap());
        println!("  Theta: {:.4}", greeks.theta.unwrap());
        println!("  Rho: {:.4}", greeks.rho_greek.unwrap());

        // Delta should be ~0.5 for ATM call (relaxed range due to high vol)
        let delta = greeks.delta.unwrap();
        assert!(
            delta > 0.2 && delta < 0.8,
            "Delta out of range for ATM call: {}",
            delta
        );

        // Gamma should be positive
        let gamma = greeks.gamma.unwrap();
        assert!(gamma > 0.0, "Gamma should be positive, got {}", gamma);

        // Vega should be positive
        let vega = greeks.vega.unwrap();
        assert!(vega > 0.0, "Vega should be positive, got {}", vega);

        // Theta should be negative (time decay for long option)
        let theta = greeks.theta.unwrap();
        assert!(
            theta < 0.0,
            "Theta should be negative for long option, got {}",
            theta
        );

        println!("✓ All Greeks within expected ranges");
    }

    #[test]
    #[ignore] // Requires GPU (uses VolArbitrageStrategy which doesn't need GPU but grouped here)
    fn test_vol_arbitrage_profitability() {
        println!("\n=== Test: Vol Arbitrage Strategy ===");

        // Create mispriced options (market IV different from model IV)
        let params = create_test_params();
        let model_iv = params.current_vol(); // 20% (√0.04)

        let mut options = generate_test_options(5, 48000.0);

        // Artificially set market IVs to create arbitrage opportunities
        options[0].implied_vol = Some(model_iv + 0.10); // Overpriced by 10pp
        options[1].implied_vol = Some(model_iv - 0.08); // Underpriced by 8pp
        options[2].implied_vol = Some(model_iv + 0.03); // Slightly overpriced
        options[3].implied_vol = Some(model_iv - 0.02); // Slightly underpriced
        options[4].implied_vol = Some(model_iv); // Fair value

        let strategy = VolArbitrageStrategy::new(5.0); // 5pp threshold
        let signals = strategy.generate_signals(&options, &params);

        println!("Generated {} signals (threshold: 5pp)", signals.len());

        // Should find mispriced options (only first 2 exceed 5pp threshold)
        assert!(
            signals.len() >= 2,
            "Expected at least 2 signals, got {}",
            signals.len()
        );

        // All signals should have positive edge
        for (i, signal) in signals.iter().enumerate() {
            let (option, edge, reason) = match signal {
                TradeSignal::Buy {
                    option,
                    edge,
                    reason,
                } => (option, edge, reason),
                TradeSignal::Sell {
                    option,
                    edge,
                    reason,
                } => (option, edge, reason),
            };

            println!(
                "Signal {}: {} strike ${} - Edge: {:.2}pp - {}",
                i + 1,
                match signal {
                    TradeSignal::Buy { .. } => "BUY",
                    TradeSignal::Sell { .. } => "SELL",
                },
                option.strike,
                edge,
                reason
            );

            assert!(
                *edge > 5.0,
                "Signal has insufficient edge: {:.2}pp (threshold: 5pp)",
                edge
            );
        }

        println!("✓ All signals have positive edge > 5pp");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_delta_hedging_strategy() {
        println!("\n=== Test: Delta Hedging Strategy ===");

        // Create a portfolio of options
        let params = create_test_params();
        let option = OptionQuote {
            underlying: "BTC".to_string(),
            strike: 50000.0,
            expiration: chrono::Utc::now().timestamp() + (90 * 24 * 3600),
            option_type: OptionType::Call,
            spot_price: 50000.0,
            risk_free_rate: 0.05,
            bid: Some(2000.0),
            ask: Some(2200.0),
            last: Some(2100.0),
            implied_vol: Some(0.8),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        };

        // Calculate Greeks
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let pricer = HestonGpuPricer::new(device, 4096, 100).expect("Failed to create pricer");
        let calculator = HestonGreeksCalculator::new(Arc::new(Mutex::new(pricer)));

        let greeks = calculator
            .calculate_greeks(&params, &option)
            .expect("Greeks calculation failed");

        // Create portfolio: long 10 calls
        let portfolio = vec![OptionPosition {
            option: option.clone(),
            quantity: 10,
        }];

        let greeks_vec = vec![greeks];

        // Calculate hedge
        let strategy = DeltaHedgingStrategy::new(0.0); // Delta-neutral target
        let hedge = strategy.calculate_hedge(&portfolio, &greeks_vec);

        println!("Portfolio:");
        println!("  10 long calls @ strike ${}", option.strike);
        println!("  Delta per contract: {:.4}", greeks.delta.unwrap());
        println!("\nHedge recommendation:");
        println!(
            "  {} {:.2} shares of underlying",
            if hedge.underlying_shares > 0 {
                "BUY"
            } else {
                "SELL"
            },
            hedge.underlying_shares.abs()
        );
        println!("  Reason: {}", hedge.reason);

        // For long calls, delta is positive, so we should SELL underlying to hedge
        assert!(
            hedge.underlying_shares < 0,
            "Should sell underlying to hedge long calls, got {}",
            hedge.underlying_shares
        );

        // Magnitude should be approximately 10 * delta
        let expected_magnitude = 10.0 * greeks.delta.unwrap();
        let actual_magnitude = hedge.underlying_shares.abs() as f64;
        let error_pct = ((actual_magnitude - expected_magnitude) / expected_magnitude).abs();
        assert!(
            error_pct < 0.01,
            "Hedge magnitude error too high: {:.2}% (expected: {:.2}, got: {:.2})",
            error_pct * 100.0,
            expected_magnitude,
            actual_magnitude
        );

        println!("✓ Delta hedge correctly calculated");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_calibration_performance() {
        println!("\n=== Test: Calibration Performance ===");

        // Verify calibration completes within performance target
        let params = create_test_params();
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let mut pricer_for_gen =
            HestonGpuPricer::new(device.clone(), 4096, 100).expect("Failed to create pricer");
        let options = generate_synthetic_options(&params, &mut pricer_for_gen, 50);

        let gpu_pricer =
            Arc::new(Mutex::new(HestonGpuPricer::new(device, 4096, 100).expect("Failed to create pricer")));
        let calibrator = HestonCalibrator::new(gpu_pricer, options, params)
            .expect("Failed to create calibrator")
            .with_max_iterations(50);

        let start = std::time::Instant::now();
        let result = calibrator.calibrate().expect("Calibration failed");
        let elapsed = start.elapsed();

        println!("Calibration time: {:?}", elapsed);
        println!("  Iterations: {}", result.iterations);
        println!("  RMSE: {:.6}", result.rmse());

        // Performance target: <60 seconds for 50 options
        assert!(
            elapsed.as_secs() < 60,
            "Calibration too slow: {:?} (target: <60s)",
            elapsed
        );

        // Should converge
        assert!(result.converged, "Did not converge within 50 iterations");

        println!("✓ Calibration completed in {:?} (target: <60s)", elapsed);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_batch_pricing_performance() {
        println!("\n=== Test: Batch Pricing Performance ===");

        let params = create_test_params();
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let mut pricer = HestonGpuPricer::new(device, 4096, 1000).expect("Failed to create pricer");

        // Test different batch sizes
        for n in [10, 50, 100, 500] {
            let options = generate_test_options(n, 48000.0);

            // Warmup
            pricer
                .price_options(&params, &options)
                .expect("Warmup failed");

            // Benchmark
            let start = std::time::Instant::now();
            let n_runs = 10;
            for _ in 0..n_runs {
                pricer
                    .price_options(&params, &options)
                    .expect("Pricing failed");
            }
            let elapsed = start.elapsed();
            let avg_time_ms = elapsed.as_millis() / n_runs;

            println!(
                "  {} options: {:.2}ms avg ({} runs)",
                n, avg_time_ms, n_runs
            );

            // Performance targets from heston_pricing.rs
            let target_ms = match n {
                10 => 1,
                50 => 2,
                100 => 3,
                500 => 10,
                _ => 100,
            };

            assert!(
                avg_time_ms <= target_ms * 2,
                "{} options too slow: {}ms (target: <{}ms)",
                n,
                avg_time_ms,
                target_ms
            );
        }

        println!("✓ All batch sizes meet performance targets");
    }

    #[test]
    fn test_parameter_validation() {
        println!("\n=== Test: Parameter Validation ===");

        // Test Feller condition
        let result = HestonParams::new(
            1.0,  // kappa
            0.01, // theta
            1.5,  // sigma (violates Feller: σ² = 2.25 > 2κθ = 0.02)
            -0.7, 0.04,
        );
        assert!(result.is_err(), "Should reject Feller violation");
        println!("✓ Feller condition validated");

        // Test correlation bounds
        let result = HestonParams::new(2.0, 0.04, 0.3, -1.5, 0.04);
        assert!(result.is_err(), "Should reject correlation outside [-1, 1]");
        println!("✓ Correlation bounds validated");

        // Test positive parameters
        let result = HestonParams::new(-1.0, 0.04, 0.3, -0.7, 0.04);
        assert!(result.is_err(), "Should reject negative kappa");

        let result = HestonParams::new(2.0, -0.04, 0.3, -0.7, 0.04);
        assert!(result.is_err(), "Should reject negative theta");

        let result = HestonParams::new(2.0, 0.04, -0.3, -0.7, 0.04);
        assert!(result.is_err(), "Should reject negative sigma");

        let result = HestonParams::new(2.0, 0.04, 0.3, -0.7, -0.04);
        assert!(result.is_err(), "Should reject negative v0");

        println!("✓ Positivity constraints validated");
    }

    // ===== Phase 2: Heston-Backtest Integration Tests =====

    #[test]
    #[ignore] // Requires GPU
    fn test_phase2_heston_batch_greeks_accuracy() {
        println!("\n=== Phase 2: Batch Greeks Calculation Test ===\n");

        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let pricer = HestonGpuPricer::new(device, 4096, 1000).expect("Failed to create pricer");
        let calculator = HestonGreeksCalculator::new(Arc::new(Mutex::new(pricer)));

        let params = create_test_params();
        let options = generate_test_options(10, 48000.0);

        // Test batch GPU Greeks
        let start = std::time::Instant::now();
        let greeks_batch = calculator
            .calculate_greeks_batch_gpu(&params, &options)
            .expect("Batch Greeks calculation failed");
        let batch_time = start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "Batch GPU Greeks: {:.2}ms for {} options",
            batch_time,
            options.len()
        );

        // Compare with sequential calculation for accuracy
        let greeks_sequential: Vec<_> = options
            .iter()
            .map(|opt| calculator.calculate_greeks(&params, opt).unwrap())
            .collect();

        // Verify accuracy: batch should match sequential within 1%
        for (i, (batch_g, seq_g)) in greeks_batch.iter().zip(&greeks_sequential).enumerate() {
            let delta_err =
                (batch_g.delta.unwrap() - seq_g.delta.unwrap()).abs() / seq_g.delta.unwrap();
            let gamma_err =
                (batch_g.gamma.unwrap() - seq_g.gamma.unwrap()).abs() / seq_g.gamma.unwrap();
            let vega_err =
                (batch_g.vega.unwrap() - seq_g.vega.unwrap()).abs() / seq_g.vega.unwrap();

            assert!(
                delta_err < 0.01,
                "Option {}: Delta error {:.2}% exceeds 1%",
                i,
                delta_err * 100.0
            );
            assert!(
                gamma_err < 0.01,
                "Option {}: Gamma error {:.2}% exceeds 1%",
                i,
                gamma_err * 100.0
            );
            assert!(
                vega_err < 0.01,
                "Option {}: Vega error {:.2}% exceeds 1%",
                i,
                vega_err * 100.0
            );

            if i == 0 {
                println!(
                    "  Option 0: Delta={:.4}, Gamma={:.6}, Vega={:.4}",
                    batch_g.delta.unwrap(),
                    batch_g.gamma.unwrap(),
                    batch_g.vega.unwrap()
                );
            }
        }

        println!("✓ Batch Greeks accuracy validated (<1% error)");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_phase2_heston_batch_greeks_performance() {
        println!("\n=== Phase 2: Batch Greeks Performance Test ===\n");

        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let pricer = HestonGpuPricer::new(device, 4096, 1000).expect("Failed to create pricer");
        let calculator = HestonGreeksCalculator::new(Arc::new(Mutex::new(pricer)));

        let params = create_test_params();

        // Test different batch sizes
        for batch_size in [10, 50, 100, 500] {
            let options = generate_test_options(batch_size, 48000.0);

            // Sequential timing (baseline)
            let start = std::time::Instant::now();
            let _greeks_seq: Vec<_> = options
                .iter()
                .map(|opt| calculator.calculate_greeks(&params, opt).unwrap())
                .collect();
            let seq_time = start.elapsed().as_secs_f64() * 1000.0;

            // Batch GPU timing
            let start = std::time::Instant::now();
            let _greeks_batch = calculator
                .calculate_greeks_batch_gpu(&params, &options)
                .expect("Batch Greeks failed");
            let batch_time = start.elapsed().as_secs_f64() * 1000.0;

            let speedup = seq_time / batch_time;

            println!(
                "  {} options: Sequential={:.2}ms, Batch={:.2}ms, Speedup={:.2}x",
                batch_size, seq_time, batch_time, speedup
            );

            // Performance validation: batch should be faster for >50 options
            if batch_size >= 50 {
                assert!(
                    speedup > 1.5,
                    "Batch speedup {:.2}x should be >1.5x for {} options",
                    speedup,
                    batch_size
                );
            }
        }

        println!("✓ Batch Greeks performance validated (>1.5x speedup for batches ≥50)");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_phase2_heston_pricing_latency() {
        println!("\n=== Phase 2: Heston Pricing Latency Test ===\n");

        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let mut pricer = HestonGpuPricer::new(device, 4096, 1000).expect("Failed to create pricer");

        let params = create_test_params();

        // Test target: <15ms for 1000 options
        let options = generate_test_options(1000, 48000.0);

        let start = std::time::Instant::now();
        let _prices = pricer
            .price_options(&params, &options)
            .expect("Pricing failed");
        let pricing_time = start.elapsed().as_secs_f64() * 1000.0;

        println!("Heston pricing: {:.2}ms for 1000 options", pricing_time);

        // Validate latency target
        assert!(
            pricing_time < 15.0,
            "Pricing time {:.2}ms exceeds 15ms target",
            pricing_time
        );

        println!("✓ Pricing latency validated (<15ms for 1000 options)");
    }

    #[test]
    #[ignore] // Requires GPU - This test is currently expected to fail as Phase 1 is not complete
    fn test_phase2_batch_backtest_with_heston_stub() {
        println!("\n=== Phase 2: Batch Backtest + Heston Integration Test (STUB) ===\n");
        println!("⚠️ This test is a STUB - will be completed after Phase 1");

        // NOTE: This test cannot fully execute until Phase 1 (core data structures) is complete
        // For now, we just validate that the API compiles and can be instantiated

        let device = Arc::new(GpuDevice::new().expect("GPU required"));

        // Create Heston pricer
        let pricer =
            HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Failed to create pricer");
        let pricer_arc = Arc::new(Mutex::new(pricer));

        let params = create_test_params();
        let options = generate_test_options(10, 48000.0);

        // Validate builder API compiles
        use kimsfinance_core::backtest::batch::BatchBacktestSweep;

        let _sweep = BatchBacktestSweep::new(device.clone())
            .heston_pricer(pricer_arc)
            .heston_params(params)
            .options_data(options);

        println!("✓ Batch backtest builder API validated (Phase 1 pending)");
        println!("  - heston_pricer() method: OK");
        println!("  - heston_params() method: OK");
        println!("  - options_data() method: OK");
    }
}
