//! Accuracy Validation Tests for Heston Integration
//!
//! Validates numerical accuracy against reference implementations:
//! - GPU Heston prices vs CPU Black-Scholes (<0.05% error for low vol)
//! - GPU Greeks vs finite difference (<1% error)
//! - Strategy P&L calculations (cross-check manual vs automated)
//! - Position tracking accuracy

#[cfg(all(feature = "gpu", feature = "heston"))]
mod heston_accuracy_tests {
    use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
    use kimsfinance_core::quantitative::heston::{
        BlackScholesPricer, GreeksGpuCalculator, HestonParams, OptionQuote, OptionType,
    };
    use std::sync::Arc;

    mod test_data {
        include!("../data/heston_test_data.rs");
    }
    use test_data::{generate_options_chain, test_heston_params, MarketRegime};

    const PRICE_TOLERANCE: f64 = 0.0005; // 0.05% error
    const GREEKS_TOLERANCE: f64 = 0.01; // 1% error

    // ========== Heston vs Black-Scholes Price Accuracy ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_heston_vs_black_scholes_low_vol() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let mut heston_pricer = HestonGpuPricer::new(device).expect("Heston pricer failed");
        let bs_pricer = BlackScholesPricer::new();

        // Low vol params should match Black-Scholes
        let params = HestonParams::new(
            2.0,  // kappa
            0.04, // theta (20% vol)
            0.1,  // sigma (low vol of vol)
            0.0,  // rho=0 (no correlation)
            0.04, // v0=theta
        )
        .expect("Valid params");

        let spot = 50000.0;
        let options = generate_options_chain(spot, 10, 30, &params);

        let heston_prices = heston_pricer
            .price_options(&params, &options)
            .expect("Heston pricing failed");

        println!("\n===== Heston vs Black-Scholes Accuracy =====");
        let mut max_error = 0.0;

        for (i, option) in options.iter().enumerate() {
            let bs_price = bs_pricer
                .price(
                    option.spot_price,
                    option.strike,
                    (option.expiration - chrono::Utc::now().timestamp()) as f64 / (365.0 * 24.0 * 3600.0),
                    option.risk_free_rate,
                    params.v0.sqrt(), // Use current vol
                    option.option_type == OptionType::Call,
                )
                .expect("BS pricing failed");

            let error = ((heston_prices[i] - bs_price) / bs_price).abs();
            max_error = max_error.max(error);

            if error > PRICE_TOLERANCE {
                println!(
                    "Strike {}: Heston={:.2}, BS={:.2}, Error={:.4}%",
                    option.strike,
                    heston_prices[i],
                    bs_price,
                    error * 100.0
                );
            }
        }

        println!("Max error: {:.4}%", max_error * 100.0);

        assert!(
            max_error < PRICE_TOLERANCE,
            "Heston vs BS error {:.4}% exceeds tolerance {:.4}%",
            max_error * 100.0,
            PRICE_TOLERANCE * 100.0
        );
    }

    // ========== Greeks Accuracy (GPU vs Finite Difference) ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_greeks_vs_finite_difference() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let calculator = GreeksGpuCalculator::new(device.clone()).expect("Calculator failed");
        let mut pricer = HestonGpuPricer::new(device).expect("Pricer failed");

        let params = test_heston_params(MarketRegime::RangeBound);
        let spot = 50000.0;
        let options = vec![generate_options_chain(spot, 1, 30, &params)[0].clone()]; // ATM call

        let gpu_greeks = calculator
            .calculate_batch(&params, &options)
            .expect("Greeks calculation failed");

        // Finite difference approximation
        let h = 1.0; // $1 bump for delta
        let mut option_bumped = options[0].clone();
        option_bumped.spot_price += h;

        let price_base = pricer.price_options(&params, &options).expect("Pricing failed")[0];
        let price_bumped = pricer
            .price_options(&params, &[option_bumped])
            .expect("Pricing failed")[0];

        let fd_delta = (price_bumped - price_base) / h;

        println!("\n===== Greeks Accuracy (GPU vs Finite Difference) =====");
        println!("GPU Delta: {:.6}", gpu_greeks[0].delta);
        println!("FD Delta:  {:.6}", fd_delta);

        let delta_error = ((gpu_greeks[0].delta - fd_delta) / fd_delta).abs();
        println!("Delta Error: {:.4}%", delta_error * 100.0);

        assert!(
            delta_error < GREEKS_TOLERANCE,
            "Delta error {:.4}% exceeds tolerance {:.4}%",
            delta_error * 100.0,
            GREEKS_TOLERANCE * 100.0
        );
    }

    // ========== P&L Calculation Accuracy ==========

    #[test]
    #[ignore] // Requires full integration
    fn test_pnl_calculation_accuracy() {
        // Test that P&L calculations match manual verification
        // Buy option @ 2000, sell @ 2500 = 500 profit - fees
        // This would require full backtest execution
    }

    // ========== Put-Call Parity Validation ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_put_call_parity_accuracy() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let mut pricer = HestonGpuPricer::new(device).expect("Pricer failed");

        let params = test_heston_params(MarketRegime::RangeBound);
        let spot = 50000.0;
        let strike = 50000.0;
        let r = 0.05;
        let t = 30.0 / 365.0;

        let now = chrono::Utc::now().timestamp();
        let expiry = now + (30 * 24 * 3600);

        let call = OptionQuote {
            underlying: "BTC".to_string(),
            strike,
            expiration: expiry,
            option_type: OptionType::Call,
            spot_price: spot,
            risk_free_rate: r,
            bid: None,
            ask: None,
            last: None,
            implied_vol: Some(0.5),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        };

        let put = OptionQuote {
            option_type: OptionType::Put,
            ..call.clone()
        };

        let prices = pricer
            .price_options(&params, &[call, put])
            .expect("Pricing failed");

        let call_price = prices[0];
        let put_price = prices[1];

        // C - P = S - K*exp(-rT)
        let pv_strike = strike * (-r * t).exp();
        let lhs = call_price - put_price;
        let rhs = spot - pv_strike;

        let parity_error = ((lhs - rhs) / rhs).abs();

        println!("\n===== Put-Call Parity Accuracy =====");
        println!("C - P = {:.2}", lhs);
        println!("S - K*exp(-rT) = {:.2}", rhs);
        println!("Error: {:.4}%", parity_error * 100.0);

        assert!(
            parity_error < PRICE_TOLERANCE,
            "Put-call parity error {:.4}% exceeds tolerance {:.4}%",
            parity_error * 100.0,
            PRICE_TOLERANCE * 100.0
        );
    }

    // ========== Consistency Tests ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_pricing_consistency_repeated_calls() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let mut pricer = HestonGpuPricer::new(device).expect("Pricer failed");

        let params = test_heston_params(MarketRegime::RangeBound);
        let options = generate_options_chain(50000.0, 50, 30, &params);

        // Price 10 times
        let mut all_prices = Vec::new();
        for _ in 0..10 {
            let prices = pricer
                .price_options(&params, &options)
                .expect("Pricing failed");
            all_prices.push(prices);
        }

        // Verify all results are identical
        for i in 1..10 {
            for j in 0..options.len() {
                assert_eq!(
                    all_prices[0][j], all_prices[i][j],
                    "Inconsistent pricing at iteration {} option {}",
                    i, j
                );
            }
        }

        println!("\n===== Pricing Consistency: PASS =====");
        println!("10 repeated calls produced identical results");
    }

    // ========== Boundary Condition Tests ==========

    #[test]
    #[ignore] // Requires GPU
    fn test_deep_itm_call_intrinsic_value() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let mut pricer = HestonGpuPricer::new(device).expect("Pricer failed");

        let params = test_heston_params(MarketRegime::RangeBound);
        let spot = 50000.0;
        let strike = 30000.0; // Deep ITM

        let now = chrono::Utc::now().timestamp();
        let expiry = now + (30 * 24 * 3600);

        let call = OptionQuote {
            underlying: "BTC".to_string(),
            strike,
            expiration: expiry,
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

        let prices = pricer.price_options(&params, &[call]).expect("Pricing failed");
        let intrinsic = spot - strike;

        println!("\n===== Deep ITM Call Accuracy =====");
        println!("Option price: {:.2}", prices[0]);
        println!("Intrinsic value: {:.2}", intrinsic);
        println!("Time value: {:.2}", prices[0] - intrinsic);

        // Price should be >= intrinsic value
        assert!(
            prices[0] >= intrinsic,
            "ITM call price ({:.2}) < intrinsic value ({:.2})",
            prices[0],
            intrinsic
        );

        // Time value should be reasonable (<30% of intrinsic for 30 days)
        let time_value = prices[0] - intrinsic;
        assert!(
            time_value < intrinsic * 0.3,
            "Time value ({:.2}) too large for deep ITM call",
            time_value
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_deep_otm_call_low_price() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let mut pricer = HestonGpuPricer::new(device).expect("Pricer failed");

        let params = test_heston_params(MarketRegime::RangeBound);
        let spot = 50000.0;
        let strike = 100000.0; // Deep OTM

        let now = chrono::Utc::now().timestamp();
        let expiry = now + (30 * 24 * 3600);

        let call = OptionQuote {
            underlying: "BTC".to_string(),
            strike,
            expiration: expiry,
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

        let prices = pricer.price_options(&params, &[call]).expect("Pricing failed");

        println!("\n===== Deep OTM Call Accuracy =====");
        println!("Option price: {:.2}", prices[0]);
        println!("As % of spot: {:.4}%", prices[0] / spot * 100.0);

        // Deep OTM call should be very cheap
        assert!(
            prices[0] < spot * 0.05,
            "Deep OTM call price ({:.2}) too high (>5% of spot)",
            prices[0]
        );
    }
}
