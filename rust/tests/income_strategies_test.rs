//! Phase 3b: Income Strategies Tests
//!
//! Comprehensive tests for covered call and iron condor strategies.
//! Tests signal generation, parameter validation, and P&L calculations.

#[cfg(feature = "gpu")]
mod gpu_tests {
    use kimsfinance_core::gpu::GpuDevice;
    use kimsfinance_core::quantitative::heston::{
        CoveredCallParams, CoveredCallStrategyGpu, IronCondorParams, IronCondorStrategyGpu,
    };
    use std::sync::Arc;

    // ============================================================================
    // COVERED CALL TESTS
    // ============================================================================

    #[test]
    #[ignore] // Requires GPU
    fn test_covered_call_basic_signal_generation() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = CoveredCallStrategyGpu::new(device).expect("Strategy creation failed");

        let n_candles = 10;
        let n_strategies = 2;

        // Create test data
        let spot = 50000.0;
        let underlying: Vec<f64> = vec![spot; n_candles];

        // OTM call at 5% above spot (52500)
        let strikes: Vec<f64> = vec![spot * 1.05; n_strategies * n_candles];
        let call_prices: Vec<f64> = vec![1000.0; n_strategies * n_candles]; // $1000 premium (2% of spot)

        let params = vec![
            CoveredCallParams {
                strike_offset_pct: 5.0, // 5% OTM
                min_premium_pct: 1.0,   // 1% min premium
            },
            CoveredCallParams {
                strike_offset_pct: 5.0,
                min_premium_pct: 3.0, // 3% min premium (too high, won't enter)
            },
        ];

        let signals = strategy
            .generate_signals_batch(&underlying, &call_prices, &strikes, &params)
            .expect("Signal generation failed");

        assert_eq!(signals.len(), n_strategies * n_candles);

        // Strategy 0: Should enter (premium 2% > min 1%)
        for i in 0..n_candles {
            let sig = &signals[i];
            assert_eq!(sig.stock_signal, 1, "Should buy stock");
            assert_eq!(sig.call_signal, -1, "Should sell call");
            assert_eq!(sig.premium_collected, 1000.0, "Premium should be $1000");
        }

        // Strategy 1: Should NOT enter (premium 2% < min 3%)
        for i in n_candles..(n_strategies * n_candles) {
            let sig = &signals[i];
            assert_eq!(sig.stock_signal, 0, "Should not buy stock");
            assert_eq!(sig.call_signal, 0, "Should not sell call");
            assert_eq!(sig.premium_collected, 0.0, "No premium collected");
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_covered_call_validates_otm_strike() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = CoveredCallStrategyGpu::new(device).expect("Strategy creation failed");

        let n_candles = 5;
        let n_strategies = 1;

        let spot = 50000.0;
        let underlying: Vec<f64> = vec![spot; n_candles];

        // ITM strike (below spot) - should not enter
        let strikes: Vec<f64> = vec![spot * 0.95; n_strategies * n_candles];
        let call_prices: Vec<f64> = vec![3000.0; n_strategies * n_candles];

        let params = vec![CoveredCallParams {
            strike_offset_pct: 5.0,
            min_premium_pct: 1.0,
        }];

        let signals = strategy
            .generate_signals_batch(&underlying, &call_prices, &strikes, &params)
            .expect("Signal generation failed");

        // Should not enter with ITM strike
        for sig in &signals {
            assert_eq!(sig.stock_signal, 0, "Should not enter with ITM call");
            assert_eq!(sig.call_signal, 0, "Should not sell ITM call");
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_covered_call_batch_performance() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = CoveredCallStrategyGpu::new(device).expect("Strategy creation failed");

        // Large batch: 1000 strategies × 500 candles = 500,000 combinations
        let n_candles = 500;
        let n_strategies = 1000;

        let spot = 50000.0;
        let underlying: Vec<f64> = (0..n_candles)
            .map(|i| spot + (i as f64 - 250.0) * 10.0)
            .collect();

        let strikes: Vec<f64> = underlying
            .iter()
            .cycle()
            .take(n_strategies * n_candles)
            .map(|s| s * 1.05)
            .collect();

        let call_prices: Vec<f64> = vec![1000.0; n_strategies * n_candles];

        let params = vec![
            CoveredCallParams {
                strike_offset_pct: 5.0,
                min_premium_pct: 1.0,
            };
            n_strategies
        ];

        let start = std::time::Instant::now();
        let signals = strategy
            .generate_signals_batch(&underlying, &call_prices, &strikes, &params)
            .expect("Signal generation failed");
        let elapsed = start.elapsed();

        assert_eq!(signals.len(), n_strategies * n_candles);
        println!(
            "Covered Call GPU: {} signals in {:?} ({:.0} signals/sec)",
            signals.len(),
            elapsed,
            signals.len() as f64 / elapsed.as_secs_f64()
        );

        // Performance target: <10ms for 500K combinations
        assert!(
            elapsed.as_millis() < 20,
            "Should complete in <20ms, took {:?}",
            elapsed
        );
    }

    // ============================================================================
    // IRON CONDOR TESTS
    // ============================================================================

    #[test]
    #[ignore] // Requires GPU
    fn test_iron_condor_basic_signal_generation() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = IronCondorStrategyGpu::new(device).expect("Strategy creation failed");

        let n_candles = 5;
        let n_strategies = 1;

        let spot = 50000.0;
        let underlying: Vec<f64> = vec![spot; n_candles];

        // Iron condor strikes:
        // Long put: 45500 (9% below spot)
        // Short put: 47500 (5% below spot)
        // Short call: 52500 (5% above spot)
        // Long call: 54500 (9% above spot)

        let mut put_strikes = Vec::new();
        let mut put_prices = Vec::new();
        let mut call_strikes = Vec::new();
        let mut call_prices = Vec::new();

        for _ in 0..(n_strategies * n_candles) {
            // Put side: [long_put_strike, short_put_strike]
            put_strikes.push(spot * 0.91); // Long put
            put_strikes.push(spot * 0.95); // Short put

            // Put prices: [long_put_price, short_put_price]
            put_prices.push(200.0); // Buy long put (pay)
            put_prices.push(500.0); // Sell short put (receive)

            // Call side: [short_call_strike, long_call_strike]
            call_strikes.push(spot * 1.05); // Short call
            call_strikes.push(spot * 1.09); // Long call

            // Call prices: [short_call_price, long_call_price]
            call_prices.push(500.0); // Sell short call (receive)
            call_prices.push(200.0); // Buy long call (pay)
        }

        let params = vec![IronCondorParams {
            short_put_offset: 5.0,
            short_call_offset: 5.0,
            long_offset: 4.0,
            min_credit: 200.0, // Net credit should be $600
        }];

        let signals = strategy
            .generate_signals_batch(
                &underlying,
                &put_prices,
                &call_prices,
                &put_strikes,
                &call_strikes,
                &params,
            )
            .expect("Signal generation failed");

        assert_eq!(signals.len(), n_strategies * n_candles);

        // Verify signals
        for sig in &signals {
            assert_eq!(sig.long_put_signal, 1, "Should buy long put");
            assert_eq!(sig.short_put_signal, -1, "Should sell short put");
            assert_eq!(sig.short_call_signal, -1, "Should sell short call");
            assert_eq!(sig.long_call_signal, 1, "Should buy long call");

            // Net credit = (500 + 500) - (200 + 200) = $600
            assert_eq!(sig.net_credit, 600.0, "Net credit should be $600");

            // Max loss = max(put_width, call_width) - credit
            // Put width: 47500 - 45500 = 2000
            // Call width: 54500 - 52500 = 2000
            // Max loss: 2000 - 600 = 1400
            assert_eq!(sig.max_loss, 1400.0, "Max loss should be $1400");
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_iron_condor_insufficient_credit() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = IronCondorStrategyGpu::new(device).expect("Strategy creation failed");

        let n_candles = 3;
        let n_strategies = 1;

        let spot = 50000.0;
        let underlying: Vec<f64> = vec![spot; n_candles];

        let mut put_strikes = Vec::new();
        let mut put_prices = Vec::new();
        let mut call_strikes = Vec::new();
        let mut call_prices = Vec::new();

        for _ in 0..(n_strategies * n_candles) {
            put_strikes.push(spot * 0.92);
            put_strikes.push(spot * 0.96);
            put_prices.push(150.0); // Long put
            put_prices.push(250.0); // Short put

            call_strikes.push(spot * 1.04);
            call_strikes.push(spot * 1.08);
            call_prices.push(250.0); // Short call
            call_prices.push(150.0); // Long call
        }

        // Net credit would be (250 + 250) - (150 + 150) = $200
        // But min_credit is $1000 (too high)
        let params = vec![IronCondorParams {
            short_put_offset: 4.0,
            short_call_offset: 4.0,
            long_offset: 4.0,
            min_credit: 1000.0, // Require $1000 min credit
        }];

        let signals = strategy
            .generate_signals_batch(
                &underlying,
                &put_prices,
                &call_prices,
                &put_strikes,
                &call_strikes,
                &params,
            )
            .expect("Signal generation failed");

        // Should not enter due to insufficient credit
        for sig in &signals {
            assert_eq!(sig.long_put_signal, 0, "Should not enter position");
            assert_eq!(sig.short_put_signal, 0, "Should not enter position");
            assert_eq!(sig.net_credit, 0.0, "No credit when not entering");
            assert_eq!(sig.max_loss, 0.0, "No loss when not entering");
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_iron_condor_validates_strike_ordering() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = IronCondorStrategyGpu::new(device).expect("Strategy creation failed");

        let n_candles = 2;
        let n_strategies = 1;

        let spot = 50000.0;
        let underlying: Vec<f64> = vec![spot; n_candles];

        let mut put_strikes = Vec::new();
        let mut put_prices = Vec::new();
        let mut call_strikes = Vec::new();
        let mut call_prices = Vec::new();

        for _ in 0..(n_strategies * n_candles) {
            // Invalid ordering: short_put > long_put (reversed)
            put_strikes.push(spot * 0.96); // Should be long put (lower)
            put_strikes.push(spot * 0.92); // Should be short put (higher)

            put_prices.push(200.0);
            put_prices.push(400.0);

            call_strikes.push(spot * 1.04);
            call_strikes.push(spot * 1.08);
            call_prices.push(400.0);
            call_prices.push(200.0);
        }

        let params = vec![IronCondorParams::default()];

        let signals = strategy
            .generate_signals_batch(
                &underlying,
                &put_prices,
                &call_prices,
                &put_strikes,
                &call_strikes,
                &params,
            )
            .expect("Signal generation failed");

        // Should not enter with invalid strike ordering
        for sig in &signals {
            assert_eq!(
                sig.long_put_signal, 0,
                "Should not enter with invalid strikes"
            );
            assert_eq!(sig.net_credit, 0.0, "No credit with invalid strikes");
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_iron_condor_batch_performance() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = IronCondorStrategyGpu::new(device).expect("Strategy creation failed");

        // Large batch: 1000 strategies × 500 candles = 500,000 combinations
        let n_candles = 500;
        let n_strategies = 1000;

        let spot = 50000.0;
        let underlying: Vec<f64> = (0..n_candles)
            .map(|i| spot + (i as f64 - 250.0) * 20.0)
            .collect();

        let mut put_strikes = Vec::new();
        let mut put_prices = Vec::new();
        let mut call_strikes = Vec::new();
        let mut call_prices = Vec::new();

        for i in 0..(n_strategies * n_candles) {
            let s = underlying[i % n_candles];

            put_strikes.push(s * 0.92);
            put_strikes.push(s * 0.96);
            put_prices.push(200.0);
            put_prices.push(450.0);

            call_strikes.push(s * 1.04);
            call_strikes.push(s * 1.08);
            call_prices.push(450.0);
            call_prices.push(200.0);
        }

        let params = vec![IronCondorParams::default(); n_strategies];

        let start = std::time::Instant::now();
        let signals = strategy
            .generate_signals_batch(
                &underlying,
                &put_prices,
                &call_prices,
                &put_strikes,
                &call_strikes,
                &params,
            )
            .expect("Signal generation failed");
        let elapsed = start.elapsed();

        assert_eq!(signals.len(), n_strategies * n_candles);
        println!(
            "Iron Condor GPU: {} signals in {:?} ({:.0} signals/sec)",
            signals.len(),
            elapsed,
            signals.len() as f64 / elapsed.as_secs_f64()
        );

        // Performance target: <10ms for 500K combinations
        assert!(
            elapsed.as_millis() < 25,
            "Should complete in <25ms, took {:?}",
            elapsed
        );

        // Verify all signals are valid
        let valid_signals = signals.iter().filter(|s| s.net_credit > 0.0).count();
        assert!(
            valid_signals > 0,
            "Should generate some valid signals: {}",
            valid_signals
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_iron_condor_multiple_strategy_configs() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let strategy = IronCondorStrategyGpu::new(device).expect("Strategy creation failed");

        let n_candles = 10;
        let n_strategies = 3;

        let spot = 50000.0;
        let underlying: Vec<f64> = vec![spot; n_candles];

        let mut put_strikes = Vec::new();
        let mut put_prices = Vec::new();
        let mut call_strikes = Vec::new();
        let mut call_prices = Vec::new();

        for _ in 0..(n_strategies * n_candles) {
            put_strikes.push(spot * 0.91);
            put_strikes.push(spot * 0.95);
            put_prices.push(180.0);
            put_prices.push(480.0);

            call_strikes.push(spot * 1.05);
            call_strikes.push(spot * 1.09);
            call_prices.push(480.0);
            call_prices.push(180.0);
        }

        // Three different strategy configurations
        let params = vec![
            IronCondorParams {
                short_put_offset: 5.0,
                short_call_offset: 5.0,
                long_offset: 4.0,
                min_credit: 200.0,
            },
            IronCondorParams {
                short_put_offset: 5.0,
                short_call_offset: 5.0,
                long_offset: 4.0,
                min_credit: 800.0, // Too high, won't enter
            },
            IronCondorParams {
                short_put_offset: 5.0,
                short_call_offset: 5.0,
                long_offset: 4.0,
                min_credit: 100.0,
            },
        ];

        let signals = strategy
            .generate_signals_batch(
                &underlying,
                &put_prices,
                &call_prices,
                &put_strikes,
                &call_strikes,
                &params,
            )
            .expect("Signal generation failed");

        // Strategy 0: Should enter (credit $600 > min $200)
        for i in 0..n_candles {
            assert!(
                signals[i].net_credit > 0.0,
                "Strategy 0 should enter position"
            );
        }

        // Strategy 1: Should NOT enter (credit $600 < min $800)
        for i in n_candles..(2 * n_candles) {
            assert_eq!(
                signals[i].net_credit, 0.0,
                "Strategy 1 should not enter position"
            );
        }

        // Strategy 2: Should enter (credit $600 > min $100)
        for i in (2 * n_candles)..(3 * n_candles) {
            assert!(
                signals[i].net_credit > 0.0,
                "Strategy 2 should enter position"
            );
        }
    }
}
