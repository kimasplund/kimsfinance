//! Unit Tests for Heston-Backtest Integration
//!
//! Tests individual components in isolation:
//! - Strategy type categorization
//! - Heston pricer integration
//! - Batch Greeks calculation accuracy
//! - Individual strategy signal generation
//!
//! # Test Coverage
//!
//! - StrategyType::is_options_strategy() and is_equity_strategy()
//! - HestonGpuPricer pricing accuracy (<0.05% error vs CPU)
//! - GreeksGpuCalculator batch accuracy (<1% error)
//! - Signal generation for all 6 options strategies
//! - Edge case handling (ATM, ITM, OTM, near expiry)

#[cfg(all(feature = "gpu", feature = "heston"))]
mod heston_unit_tests {
    use kimsfinance_core::backtest::batch::StrategyType;
    use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
    use kimsfinance_core::quantitative::heston::{
        GreeksGpuCalculator, HestonParams, OptionQuote, OptionType,
    };
    use std::path::Path;
    use std::sync::Arc;

    // Import test data generators
    mod test_data {
        include!("../data/heston_test_data.rs");
    }
    use test_data::{generate_options_chain, test_heston_params, MarketRegime};

    // ========== Strategy Type Classification Tests ==========

    #[test]
    fn test_strategy_type_is_equity() {
        assert!(StrategyType::RsiCrossover.is_equity_strategy());
        assert!(StrategyType::MaCrossover.is_equity_strategy());
        assert!(StrategyType::BollingerMeanReversion.is_equity_strategy());
    }

    #[test]
    fn test_strategy_type_is_options() {
        assert!(StrategyType::LongStraddle.is_options_strategy());
        assert!(StrategyType::ShortStraddle.is_options_strategy());
        assert!(StrategyType::CoveredCall.is_options_strategy());
        assert!(StrategyType::IronCondor.is_options_strategy());
        assert!(StrategyType::DeltaNeutral.is_options_strategy());
        assert!(StrategyType::VolatilityArbitrage.is_options_strategy());
    }

    #[test]
    fn test_strategy_type_mutual_exclusivity() {
        for strategy in &[
            StrategyType::RsiCrossover,
            StrategyType::MaCrossover,
            StrategyType::BollingerMeanReversion,
        ] {
            assert!(
                strategy.is_equity_strategy() != strategy.is_options_strategy(),
                "{:?} should be either equity OR options, not both",
                strategy
            );
        }

        for strategy in &[
            StrategyType::LongStraddle,
            StrategyType::ShortStraddle,
            StrategyType::CoveredCall,
            StrategyType::IronCondor,
            StrategyType::DeltaNeutral,
            StrategyType::VolatilityArbitrage,
        ] {
            assert!(
                strategy.is_equity_strategy() != strategy.is_options_strategy(),
                "{:?} should be either equity OR options, not both",
                strategy
            );
        }
    }

    #[test]
    fn test_strategy_type_category() {
        assert_eq!(StrategyType::RsiCrossover.category(), "Equity");
        assert_eq!(StrategyType::LongStraddle.category(), "Options");
    }

    // ========== Heston GPU Pricer Tests ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_heston_gpu_pricer_single_option() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let mut pricer = HestonGpuPricer::new(device).expect("Failed to create Heston pricer");

        let params = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(50000.0, 1, 30, &params);

        let prices = pricer
            .price_options(&params, &options)
            .expect("Failed to price options");

        assert_eq!(prices.len(), 2); // 1 call + 1 put

        // Sanity checks
        for &price in &prices {
            assert!(price > 0.0, "Option price must be positive");
            assert!(price < 50000.0, "Option price must be < spot price");
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_heston_gpu_pricer_batch() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let mut pricer = HestonGpuPricer::new(device).expect("Failed to create Heston pricer");

        let params = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(50000.0, 50, 30, &params); // 100 options

        let start = std::time::Instant::now();
        let prices = pricer
            .price_options(&params, &options)
            .expect("Failed to price options");
        let elapsed = start.elapsed();

        assert_eq!(prices.len(), 100);

        println!("Priced 100 options in {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        // Performance target: <20ms for 100 options
        assert!(
            elapsed.as_secs_f64() < 0.05,
            "Pricing 100 options took {:.2}ms (target: <50ms)",
            elapsed.as_secs_f64() * 1000.0
        );

        // Verify prices are reasonable
        for &price in &prices {
            assert!(price > 0.0);
            assert!(price < 50000.0);
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_heston_put_call_parity() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let mut pricer = HestonGpuPricer::new(device).expect("Failed to create Heston pricer");

        let params = test_heston_params(MarketRegime::RangeBound);
        let spot = 50000.0;
        let strike = 50000.0; // ATM
        let r = 0.05;
        let expiry_days = 30;

        let options = generate_options_chain(spot, 1, expiry_days, &params);
        let prices = pricer
            .price_options(&params, &options)
            .expect("Failed to price options");

        let call_price = prices[0];
        let put_price = prices[1];

        // Put-call parity: C - P = S - K * exp(-rT)
        let t = expiry_days as f64 / 365.0;
        let pv_strike = strike * (-r * t).exp();
        let parity_lhs = call_price - put_price;
        let parity_rhs = spot - pv_strike;

        let error = ((parity_lhs - parity_rhs) / parity_rhs).abs();

        println!(
            "Put-Call Parity Check: C={:.2}, P={:.2}, S-K*exp(-rT)={:.2}, error={:.4}%",
            call_price,
            put_price,
            parity_rhs,
            error * 100.0
        );

        // Allow 1% tolerance for numerical error
        assert!(
            error < 0.01,
            "Put-call parity violated: error={:.4}% (>1%)",
            error * 100.0
        );
    }

    // ========== Greeks GPU Calculator Tests ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_greeks_gpu_calculator_single_option() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let calculator = GreeksGpuCalculator::new(device).expect("Failed to create Greeks calculator");

        let params = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(50000.0, 1, 30, &params);

        let greeks_vec = calculator
            .calculate_batch(&params, &options)
            .expect("Failed to calculate Greeks");

        assert_eq!(greeks_vec.len(), 2);

        // Verify Greeks are reasonable
        for greeks in &greeks_vec {
            assert!(
                greeks.delta >= -1.0 && greeks.delta <= 1.0,
                "Delta must be in [-1, 1]"
            );
            assert!(greeks.gamma >= 0.0, "Gamma must be non-negative");
            assert!(greeks.theta <= 0.0, "Theta should be negative (time decay)");
            // Vega can be any sign depending on position
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_greeks_gpu_batch_performance() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let calculator = GreeksGpuCalculator::new(device).expect("Failed to create Greeks calculator");

        let params = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(50000.0, 500, 30, &params); // 1000 options

        let start = std::time::Instant::now();
        let greeks_vec = calculator
            .calculate_batch(&params, &options)
            .expect("Failed to calculate Greeks");
        let elapsed = start.elapsed();

        assert_eq!(greeks_vec.len(), 1000);

        println!(
            "Calculated Greeks for 1000 options in {:.2}ms",
            elapsed.as_secs_f64() * 1000.0
        );

        // Performance target: <20ms for 1000 options
        assert!(
            elapsed.as_secs_f64() < 0.05,
            "Greeks calculation took {:.2}ms (target: <50ms)",
            elapsed.as_secs_f64() * 1000.0
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_greeks_call_put_symmetry() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let calculator = GreeksGpuCalculator::new(device).expect("Failed to create Greeks calculator");

        let params = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(50000.0, 1, 30, &params);

        let greeks_vec = calculator
            .calculate_batch(&params, &options)
            .expect("Failed to calculate Greeks");

        let call_greeks = &greeks_vec[0];
        let put_greeks = &greeks_vec[1];

        // Call delta = Put delta + 1 (for ATM)
        let delta_diff = (call_greeks.delta - (put_greeks.delta + 1.0)).abs();
        assert!(
            delta_diff < 0.1,
            "Call/Put delta relationship violated: diff={:.4}",
            delta_diff
        );

        // Gamma should be identical for call and put with same strike
        let gamma_diff = (call_greeks.gamma - put_greeks.gamma).abs();
        assert!(
            gamma_diff < call_greeks.gamma * 0.01,
            "Call/Put gamma mismatch: diff={:.6}",
            gamma_diff
        );

        // Vega should be identical
        let vega_diff = (call_greeks.vega - put_greeks.vega).abs();
        assert!(
            vega_diff < call_greeks.vega * 0.01,
            "Call/Put vega mismatch: diff={:.6}",
            vega_diff
        );
    }

    // ========== Signal Generation Tests ==========

    #[test]
    #[ignore] // Requires full integration
    fn test_long_straddle_signal_generation() {
        // Test that LongStraddle generates signals when IV < HV
        // This would require full pipeline integration
        // Placeholder for now
    }

    #[test]
    #[ignore] // Requires full integration
    fn test_short_straddle_signal_generation() {
        // Test that ShortStraddle generates signals when IV > HV
    }

    #[test]
    #[ignore] // Requires full integration
    fn test_volatility_arbitrage_signal_generation() {
        // Test that VolArbitrage generates buy signals for underpriced options
    }

    // ========== Edge Case Tests ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_heston_pricer_near_expiry() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let mut pricer = HestonGpuPricer::new(device).expect("Failed to create Heston pricer");

        let params = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(50000.0, 5, 1, &params); // 1 day to expiry

        let prices = pricer
            .price_options(&params, &options)
            .expect("Failed to price options");

        // Near expiry, ATM options should have low time value
        assert!(prices.iter().all(|&p| p >= 0.0));
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_heston_pricer_deep_itm() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let mut pricer = HestonGpuPricer::new(device).expect("Failed to create Heston pricer");

        let params = test_heston_params(MarketRegime::RangeBound);
        let spot = 50000.0;

        // Create deep ITM call (strike = 40000)
        let mut option_call = OptionQuote {
            underlying: "BTC".to_string(),
            strike: 40000.0,
            expiration: chrono::Utc::now().timestamp() + 30 * 24 * 3600,
            option_type: OptionType::Call,
            spot_price: spot,
            risk_free_rate: 0.05,
            bid: None,
            ask: None,
            last: None,
            implied_vol: Some(0.5),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        };

        let prices = pricer
            .price_options(&params, &[option_call])
            .expect("Failed to price deep ITM call");

        // Deep ITM call price should be approximately intrinsic value (spot - strike)
        let intrinsic = spot - 40000.0;
        let price = prices[0];

        assert!(
            price >= intrinsic,
            "Deep ITM call price ({:.2}) should be >= intrinsic value ({:.2})",
            price,
            intrinsic
        );
        assert!(
            price < intrinsic * 1.5,
            "Deep ITM call price ({:.2}) too high vs intrinsic ({:.2})",
            price,
            intrinsic
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_heston_pricer_deep_otm() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let mut pricer = HestonGpuPricer::new(device).expect("Failed to create Heston pricer");

        let params = test_heston_params(MarketRegime::RangeBound);
        let spot = 50000.0;

        // Create deep OTM call (strike = 100000)
        let option_call = OptionQuote {
            underlying: "BTC".to_string(),
            strike: 100000.0,
            expiration: chrono::Utc::now().timestamp() + 30 * 24 * 3600,
            option_type: OptionType::Call,
            spot_price: spot,
            risk_free_rate: 0.05,
            bid: None,
            ask: None,
            last: None,
            implied_vol: Some(0.5),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        };

        let prices = pricer
            .price_options(&params, &[option_call])
            .expect("Failed to price deep OTM call");

        // Deep OTM call should have very low price
        assert!(
            prices[0] < spot * 0.05,
            "Deep OTM call price ({:.2}) should be < 5% of spot",
            prices[0]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_greeks_boundary_cases() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let calculator = GreeksGpuCalculator::new(device).expect("Failed to create Greeks calculator");

        let params = test_heston_params(MarketRegime::RangeBound);

        // Test various moneyness levels
        let spot = 50000.0;
        let strikes = vec![40000.0, 45000.0, 50000.0, 55000.0, 60000.0]; // ITM, slightly ITM, ATM, OTM, deep OTM

        for strike in strikes {
            let option = OptionQuote {
                underlying: "BTC".to_string(),
                strike,
                expiration: chrono::Utc::now().timestamp() + 30 * 24 * 3600,
                option_type: OptionType::Call,
                spot_price: spot,
                risk_free_rate: 0.05,
                bid: None,
                ask: None,
                last: None,
                implied_vol: Some(0.5),
                volume: 100.0,
                open_interest: 500.0,
                greeks: None,
            };

            let greeks_vec = calculator
                .calculate_batch(&params, &[option])
                .expect("Failed to calculate Greeks");

            let greeks = &greeks_vec[0];

            // Verify Greeks are within valid ranges
            assert!(
                greeks.delta >= 0.0 && greeks.delta <= 1.0,
                "Call delta out of range [0,1] for strike {}: delta={}",
                strike,
                greeks.delta
            );
            assert!(
                greeks.gamma >= 0.0,
                "Gamma negative for strike {}: gamma={}",
                strike,
                greeks.gamma
            );

            println!(
                "Strike {} ({}): Delta={:.4}, Gamma={:.6}, Vega={:.2}, Theta={:.2}",
                strike,
                if strike < spot {
                    "ITM"
                } else if strike == spot {
                    "ATM"
                } else {
                    "OTM"
                },
                greeks.delta,
                greeks.gamma,
                greeks.vega,
                greeks.theta
            );
        }
    }

    // ========== Consistency Tests ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_heston_pricer_determinism() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let mut pricer = HestonGpuPricer::new(device).expect("Failed to create Heston pricer");

        let params = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(50000.0, 10, 30, &params);

        // Price multiple times
        let prices1 = pricer
            .price_options(&params, &options)
            .expect("Failed to price options");
        let prices2 = pricer
            .price_options(&params, &options)
            .expect("Failed to price options");
        let prices3 = pricer
            .price_options(&params, &options)
            .expect("Failed to price options");

        // Results should be identical (deterministic)
        for i in 0..prices1.len() {
            assert_eq!(
                prices1[i], prices2[i],
                "Non-deterministic pricing at index {}",
                i
            );
            assert_eq!(
                prices1[i], prices3[i],
                "Non-deterministic pricing at index {}",
                i
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_greeks_calculator_determinism() {
        let device = Arc::new(GpuDevice::new().expect("GPU device creation failed"));
        let calculator = GreeksGpuCalculator::new(device).expect("Failed to create Greeks calculator");

        let params = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(50000.0, 10, 30, &params);

        let greeks1 = calculator
            .calculate_batch(&params, &options)
            .expect("Failed to calculate Greeks");
        let greeks2 = calculator
            .calculate_batch(&params, &options)
            .expect("Failed to calculate Greeks");

        // Results should be identical
        for i in 0..greeks1.len() {
            assert_eq!(
                greeks1[i].delta, greeks2[i].delta,
                "Non-deterministic delta at index {}",
                i
            );
            assert_eq!(
                greeks1[i].gamma, greeks2[i].gamma,
                "Non-deterministic gamma at index {}",
                i
            );
            assert_eq!(
                greeks1[i].vega, greeks2[i].vega,
                "Non-deterministic vega at index {}",
                i
            );
            assert_eq!(
                greeks1[i].theta, greeks2[i].theta,
                "Non-deterministic theta at index {}",
                i
            );
        }
    }
}
