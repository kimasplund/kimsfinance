//! Integration tests for Delta-Neutral GPU Strategy
//!
//! Tests GPU-accelerated delta-neutral volatility trading strategy.

use kimsfinance_core::gpu::GpuDevice;
use kimsfinance_core::quantitative::heston::{DeltaNeutralParams, DeltaNeutralStrategyGpu};
use std::sync::Arc;

#[test]
#[ignore] // Requires GPU
fn test_delta_neutral_entry_signal_cheap_vol() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = DeltaNeutralStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 100;
    let n_strategies = 10;

    // Market data: IV < HV (cheap volatility)
    let underlying: Vec<f64> = (0..n_candles).map(|i| 48000.0 + i as f64 * 10.0).collect();
    let option_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
    let option_deltas: Vec<f64> = vec![0.6; n_strategies * n_candles]; // ATM call delta
    let implied_vols: Vec<f64> = vec![0.50; n_strategies * n_candles]; // 50% IV
    let historical_vols: Vec<f64> = vec![0.62; n_strategies * n_candles]; // 62% HV (12pp spread)

    let params = vec![
        DeltaNeutralParams {
            delta_threshold: 0.05,
            rebalance_threshold: 0.10,
            vol_threshold: 5.0, // 5pp threshold
        };
        n_strategies
    ];

    // Execute
    let signals = strategy
        .generate_signals_batch(
            &underlying,
            &option_prices,
            &option_deltas,
            &implied_vols,
            &historical_vols,
            &params,
        )
        .expect("Signal generation failed");

    // Verify
    assert_eq!(signals.len(), n_strategies * n_candles);

    // With 12pp IV-HV spread > 5pp threshold, should generate buy signals
    let buy_signals = signals.iter().filter(|s| s.option_signal == 1).count();
    assert!(
        buy_signals > (signals.len() / 2),
        "Expected majority buy signals, got {} / {}",
        buy_signals,
        signals.len()
    );

    // Verify hedging is applied
    for sig in &signals {
        if sig.option_signal == 1 {
            // For long option with positive delta, hedge should be negative
            assert!(
                sig.hedge_signal < 0.0,
                "Expected negative hedge for positive delta"
            );
            // Portfolio delta should be near zero (delta + hedge ≈ 0)
            assert!(
                sig.portfolio_delta.abs() < 0.2,
                "Portfolio delta too large: {}",
                sig.portfolio_delta
            );
        }
    }
}

#[test]
#[ignore] // Requires GPU
fn test_delta_neutral_no_signal_fair_vol() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = DeltaNeutralStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 50;
    let n_strategies = 5;

    // Market data: IV ≈ HV (fair volatility)
    let underlying: Vec<f64> = vec![48000.0; n_candles];
    let option_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
    let option_deltas: Vec<f64> = vec![0.5; n_strategies * n_candles];
    let implied_vols: Vec<f64> = vec![0.60; n_strategies * n_candles]; // 60% IV
    let historical_vols: Vec<f64> = vec![0.62; n_strategies * n_candles]; // 62% HV (2pp spread)

    let params = vec![
        DeltaNeutralParams {
            delta_threshold: 0.05,
            rebalance_threshold: 0.10,
            vol_threshold: 5.0, // 5pp threshold
        };
        n_strategies
    ];

    // Execute
    let signals = strategy
        .generate_signals_batch(
            &underlying,
            &option_prices,
            &option_deltas,
            &implied_vols,
            &historical_vols,
            &params,
        )
        .expect("Signal generation failed");

    // Verify: 2pp spread < 5pp threshold, should not generate signals
    for sig in &signals {
        assert_eq!(sig.option_signal, 0, "Expected no option signal");
        assert_eq!(sig.hedge_signal, 0.0, "Expected no hedge signal");
        assert_eq!(sig.portfolio_delta, 0.0, "Expected zero portfolio delta");
    }
}

#[test]
#[ignore] // Requires GPU
fn test_delta_neutral_rebalancing() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = DeltaNeutralStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 20;
    let n_strategies = 3;

    // Current positions: long 1 option with increasing delta
    let option_positions: Vec<f64> = vec![1.0; n_strategies * n_candles];

    // Initial hedge was for delta 0.5, but delta has increased to 0.7
    let hedge_positions: Vec<f64> = vec![-0.5; n_strategies * n_candles];
    let option_deltas: Vec<f64> = vec![0.7; n_strategies * n_candles]; // Delta increased

    let params = vec![
        DeltaNeutralParams {
            delta_threshold: 0.05,
            rebalance_threshold: 0.15, // 15% rebalance threshold
            vol_threshold: 5.0,
        };
        n_strategies
    ];

    // Execute
    let rebalance_signals = strategy
        .generate_rebalance_signals(&option_positions, &hedge_positions, &option_deltas, &params)
        .expect("Rebalance signal generation failed");

    // Verify
    assert_eq!(rebalance_signals.len(), n_strategies * n_candles);

    // Portfolio delta = 1.0 × 0.7 + (-0.5) × 1.0 = 0.2
    // This exceeds rebalance_threshold (0.15), so rebalancing should be triggered
    for sig in &rebalance_signals {
        // Should suggest additional short position to bring delta to zero
        // Need to short 0.2 more underlying: hedge_adjustment = -0.2
        assert!(
            (sig.hedge_adjustment + 0.2).abs() < 0.01,
            "Expected hedge adjustment of -0.2, got {}",
            sig.hedge_adjustment
        );

        // After rebalancing, portfolio delta should be near zero
        assert!(
            sig.new_portfolio_delta.abs() < 0.01,
            "Expected near-zero portfolio delta, got {}",
            sig.new_portfolio_delta
        );
    }
}

#[test]
#[ignore] // Requires GPU
fn test_delta_neutral_no_rebalance_needed() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = DeltaNeutralStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 15;
    let n_strategies = 2;

    // Positions already well-hedged
    let option_positions: Vec<f64> = vec![1.0; n_strategies * n_candles];
    let hedge_positions: Vec<f64> = vec![-0.52; n_strategies * n_candles]; // Good hedge
    let option_deltas: Vec<f64> = vec![0.53; n_strategies * n_candles]; // Slight delta change

    let params = vec![
        DeltaNeutralParams {
            delta_threshold: 0.05,
            rebalance_threshold: 0.10, // 10% threshold
            vol_threshold: 5.0,
        };
        n_strategies
    ];

    // Execute
    let rebalance_signals = strategy
        .generate_rebalance_signals(&option_positions, &hedge_positions, &option_deltas, &params)
        .expect("Rebalance signal generation failed");

    // Verify
    // Portfolio delta = 1.0 × 0.53 + (-0.52) × 1.0 = 0.01
    // This is < rebalance_threshold (0.10), so no rebalancing needed
    for sig in &rebalance_signals {
        assert!(
            sig.hedge_adjustment.abs() < 0.01,
            "Expected no hedge adjustment, got {}",
            sig.hedge_adjustment
        );
        // Portfolio delta should remain unchanged (not exactly zero, but within threshold)
        assert!(
            sig.new_portfolio_delta.abs() < 0.05,
            "Portfolio delta changed unexpectedly: {}",
            sig.new_portfolio_delta
        );
    }
}

#[test]
#[ignore] // Requires GPU
fn test_delta_neutral_batch_performance() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = DeltaNeutralStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 500;
    let n_strategies = 1000; // Large batch

    // Create large dataset
    let underlying: Vec<f64> = (0..n_candles).map(|i| 48000.0 + i as f64 * 10.0).collect();
    let option_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
    let option_deltas: Vec<f64> = vec![0.5; n_strategies * n_candles];
    let implied_vols: Vec<f64> = vec![0.50; n_strategies * n_candles];
    let historical_vols: Vec<f64> = vec![0.60; n_strategies * n_candles];

    let params = vec![DeltaNeutralParams::default(); n_strategies];

    // Execute with timing
    let start = std::time::Instant::now();
    let signals = strategy
        .generate_signals_batch(
            &underlying,
            &option_prices,
            &option_deltas,
            &implied_vols,
            &historical_vols,
            &params,
        )
        .expect("Signal generation failed");
    let elapsed = start.elapsed();

    // Verify performance
    assert_eq!(signals.len(), n_strategies * n_candles);
    println!(
        "Delta-Neutral GPU: {} strategies × {} candles in {:.2}ms",
        n_strategies,
        n_candles,
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "Throughput: {:.0} signals/sec",
        signals.len() as f64 / elapsed.as_secs_f64()
    );

    // Performance target: <50ms for 1000 strategies × 500 candles
    assert!(
        elapsed.as_millis() < 50,
        "Performance target missed: {:.2}ms > 50ms",
        elapsed.as_secs_f64() * 1000.0
    );
}

#[test]
#[ignore] // Requires GPU
fn test_delta_neutral_negative_delta_options() {
    // Setup: Test with put options (negative delta)
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = DeltaNeutralStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 50;
    let n_strategies = 5;

    let underlying: Vec<f64> = vec![48000.0; n_candles];
    let option_prices: Vec<f64> = vec![1800.0; n_strategies * n_candles];
    let option_deltas: Vec<f64> = vec![-0.4; n_strategies * n_candles]; // Put option delta
    let implied_vols: Vec<f64> = vec![0.50; n_strategies * n_candles];
    let historical_vols: Vec<f64> = vec![0.62; n_strategies * n_candles]; // Cheap vol

    let params = vec![DeltaNeutralParams::default(); n_strategies];

    // Execute
    let signals = strategy
        .generate_signals_batch(
            &underlying,
            &option_prices,
            &option_deltas,
            &implied_vols,
            &historical_vols,
            &params,
        )
        .expect("Signal generation failed");

    // Verify
    for sig in &signals {
        if sig.option_signal == 1 {
            // Long put option with delta = -0.4
            // Hedge should be positive (buy underlying) to offset negative delta
            assert!(
                sig.hedge_signal > 0.0,
                "Expected positive hedge for negative delta, got {}",
                sig.hedge_signal
            );
            // Portfolio delta should be near zero
            assert!(
                sig.portfolio_delta.abs() < 0.2,
                "Portfolio delta too large: {}",
                sig.portfolio_delta
            );
        }
    }
}

#[test]
#[ignore] // Requires GPU
fn test_delta_neutral_input_validation() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = DeltaNeutralStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 10;
    let n_strategies = 2;

    let underlying: Vec<f64> = vec![48000.0; n_candles];
    let option_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
    let option_deltas: Vec<f64> = vec![0.5; n_strategies * n_candles];
    let implied_vols: Vec<f64> = vec![0.50; n_strategies * n_candles];

    // Mismatched dimensions (should error)
    let historical_vols_wrong: Vec<f64> = vec![0.60; (n_strategies * n_candles) - 5];

    let params = vec![DeltaNeutralParams::default(); n_strategies];

    // Execute
    let result = strategy.generate_signals_batch(
        &underlying,
        &option_prices,
        &option_deltas,
        &implied_vols,
        &historical_vols_wrong,
        &params,
    );

    // Verify: should return error for dimension mismatch
    assert!(result.is_err(), "Expected error for mismatched dimensions");
}
