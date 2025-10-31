//! Integration tests for Volatility Arbitrage GPU Strategy
//!
//! Tests GPU-accelerated volatility arbitrage trading strategy.

use kimsfinance_core::gpu::GpuDevice;
use kimsfinance_core::quantitative::heston::{VolArbitrageParams, VolArbitrageStrategyGpu};
use std::sync::Arc;

#[test]
#[ignore] // Requires GPU
fn test_vol_arbitrage_long_vol_signal() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = VolArbitrageStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 100;
    let n_strategies = 10;

    // Market data: IV < HV (cheap volatility - buy signal)
    let underlying: Vec<f64> = (0..n_candles).map(|i| 48000.0 + i as f64 * 10.0).collect();
    let option_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
    let option_deltas: Vec<f64> = vec![0.5; n_strategies * n_candles];
    let option_vegas: Vec<f64> = vec![100.0; n_strategies * n_candles]; // High vega
    let implied_vols: Vec<f64> = vec![0.50; n_strategies * n_candles]; // 50% IV
    let historical_vols: Vec<f64> = vec![0.65; n_strategies * n_candles]; // 65% HV (15pp edge)

    let params = vec![
        VolArbitrageParams {
            vol_threshold: 5.0, // 5pp threshold
            hedge_delta: 1.0,   // Enable hedging
            min_edge: 2.0,      // 2% minimum edge
        };
        n_strategies
    ];

    // Execute
    let signals = strategy
        .generate_signals_batch(
            &underlying,
            &option_prices,
            &option_deltas,
            &option_vegas,
            &implied_vols,
            &historical_vols,
            &params,
        )
        .expect("Signal generation failed");

    // Verify
    assert_eq!(signals.len(), n_strategies * n_candles);

    // With 15pp IV-HV spread > 5pp threshold, should generate buy signals
    let buy_signals = signals.iter().filter(|s| s.option_signal == 1).count();
    assert!(
        buy_signals > (signals.len() / 2),
        "Expected majority buy signals (long vol), got {} / {}",
        buy_signals,
        signals.len()
    );

    // Verify signal properties
    for sig in &signals {
        if sig.option_signal == 1 {
            // Vol edge should be positive (HV > IV)
            assert!(sig.vol_edge > 0.0, "Expected positive vol edge");
            // Expected profit should be positive
            assert!(sig.expected_profit > 0.0, "Expected positive profit");
            // Hedge should be applied (negative for long call)
            assert!(
                sig.hedge_signal < 0.0,
                "Expected negative hedge for long call"
            );
        }
    }
}

#[test]
#[ignore] // Requires GPU
fn test_vol_arbitrage_short_vol_signal() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = VolArbitrageStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 80;
    let n_strategies = 8;

    // Market data: IV > HV (expensive volatility - sell signal)
    let underlying: Vec<f64> = (0..n_candles).map(|i| 48000.0 + i as f64 * 10.0).collect();
    let option_prices: Vec<f64> = vec![2500.0; n_strategies * n_candles];
    let option_deltas: Vec<f64> = vec![0.5; n_strategies * n_candles];
    let option_vegas: Vec<f64> = vec![80.0; n_strategies * n_candles];
    let implied_vols: Vec<f64> = vec![0.70; n_strategies * n_candles]; // 70% IV
    let historical_vols: Vec<f64> = vec![0.52; n_strategies * n_candles]; // 52% HV (18pp edge)

    let params = vec![
        VolArbitrageParams {
            vol_threshold: 5.0,
            hedge_delta: 1.0,
            min_edge: 2.0,
        };
        n_strategies
    ];

    // Execute
    let signals = strategy
        .generate_signals_batch(
            &underlying,
            &option_prices,
            &option_deltas,
            &option_vegas,
            &implied_vols,
            &historical_vols,
            &params,
        )
        .expect("Signal generation failed");

    // Verify
    assert_eq!(signals.len(), n_strategies * n_candles);

    // With 18pp IV-HV spread > 5pp threshold, should generate sell signals
    let sell_signals = signals.iter().filter(|s| s.option_signal == -1).count();
    assert!(
        sell_signals > (signals.len() / 2),
        "Expected majority sell signals (short vol), got {} / {}",
        sell_signals,
        signals.len()
    );

    // Verify signal properties
    for sig in &signals {
        if sig.option_signal == -1 {
            // Vol edge should be negative (IV > HV)
            assert!(sig.vol_edge < 0.0, "Expected negative vol edge");
            // Expected profit should be positive (from selling expensive vol)
            assert!(sig.expected_profit > 0.0, "Expected positive profit");
            // Hedge should be applied (positive for short call)
            assert!(
                sig.hedge_signal > 0.0,
                "Expected positive hedge for short call"
            );
        }
    }
}

#[test]
#[ignore] // Requires GPU
fn test_vol_arbitrage_no_edge() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = VolArbitrageStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 50;
    let n_strategies = 5;

    // Market data: IV ≈ HV (no edge)
    let underlying: Vec<f64> = vec![48000.0; n_candles];
    let option_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
    let option_deltas: Vec<f64> = vec![0.5; n_strategies * n_candles];
    let option_vegas: Vec<f64> = vec![100.0; n_strategies * n_candles];
    let implied_vols: Vec<f64> = vec![0.60; n_strategies * n_candles]; // 60% IV
    let historical_vols: Vec<f64> = vec![0.61; n_strategies * n_candles]; // 61% HV (1pp edge)

    let params = vec![
        VolArbitrageParams {
            vol_threshold: 5.0,
            hedge_delta: 1.0,
            min_edge: 2.0, // 2% minimum edge
        };
        n_strategies
    ];

    // Execute
    let signals = strategy
        .generate_signals_batch(
            &underlying,
            &option_prices,
            &option_deltas,
            &option_vegas,
            &implied_vols,
            &historical_vols,
            &params,
        )
        .expect("Signal generation failed");

    // Verify: 1pp edge < 5pp threshold, should not generate signals
    for sig in &signals {
        assert_eq!(sig.option_signal, 0, "Expected no option signal");
        assert_eq!(sig.expected_profit, 0.0, "Expected no profit");
        // Vol edge should be small
        assert!(
            sig.vol_edge.abs() < 0.05,
            "Vol edge too large: {}",
            sig.vol_edge
        );
    }
}

#[test]
#[ignore] // Requires GPU
fn test_vol_arbitrage_without_hedging() {
    // Setup: Test with delta hedging disabled
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = VolArbitrageStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 50;
    let n_strategies = 5;

    let underlying: Vec<f64> = vec![48000.0; n_candles];
    let option_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
    let option_deltas: Vec<f64> = vec![0.5; n_strategies * n_candles];
    let option_vegas: Vec<f64> = vec![100.0; n_strategies * n_candles];
    let implied_vols: Vec<f64> = vec![0.50; n_strategies * n_candles];
    let historical_vols: Vec<f64> = vec![0.62; n_strategies * n_candles]; // Cheap vol

    let params = vec![
        VolArbitrageParams {
            vol_threshold: 5.0,
            hedge_delta: 0.0, // Disable hedging
            min_edge: 2.0,
        };
        n_strategies
    ];

    // Execute
    let signals = strategy
        .generate_signals_batch(
            &underlying,
            &option_prices,
            &option_deltas,
            &option_vegas,
            &implied_vols,
            &historical_vols,
            &params,
        )
        .expect("Signal generation failed");

    // Verify: signals generated but no hedging
    for sig in &signals {
        if sig.option_signal == 1 {
            // Buy signal should exist
            assert_eq!(
                sig.hedge_signal, 0.0,
                "Expected no hedge signal when hedging disabled"
            );
        }
    }
}

#[test]
#[ignore] // Requires GPU
fn test_vol_arbitrage_edge_monitoring() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = VolArbitrageStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 100;
    let n_strategies = 10;
    let expected_len = n_strategies * n_candles;

    // Create varying vol edges
    let implied_vols: Vec<f64> = (0..expected_len)
        .map(|i| 0.50 + (i as f64 / expected_len as f64) * 0.20)
        .collect(); // 50-70%
    let historical_vols: Vec<f64> = vec![0.60; expected_len]; // Constant 60%
    let option_prices: Vec<f64> = vec![2000.0; expected_len];
    let option_vegas: Vec<f64> = vec![100.0; expected_len];

    // Execute
    let edge_monitors = strategy
        .monitor_edge_batch(
            &implied_vols,
            &historical_vols,
            &option_prices,
            &option_vegas,
        )
        .expect("Edge monitoring failed");

    // Verify
    assert_eq!(edge_monitors.len(), expected_len);

    // Check vol edge calculation
    for (i, edge) in edge_monitors.iter().enumerate() {
        let expected_edge = historical_vols[i] - implied_vols[i];
        assert!(
            (edge.vol_edge - expected_edge).abs() < 0.001,
            "Vol edge mismatch at {}: expected {}, got {}",
            i,
            expected_edge,
            edge.vol_edge
        );

        // Edge quality should be |edge| × vega × 100
        let expected_quality = edge.vol_edge.abs() * option_vegas[i] * 100.0;
        assert!(
            (edge.edge_quality - expected_quality).abs() < 0.1,
            "Edge quality mismatch at {}: expected {}, got {}",
            i,
            expected_quality,
            edge.edge_quality
        );
    }
}

#[test]
#[ignore] // Requires GPU
fn test_vol_arbitrage_pnl_calculation() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = VolArbitrageStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 50;
    let n_strategies = 5;
    let expected_len = n_strategies * n_candles;

    // Simulate position P&L
    let entry_prices: Vec<f64> = vec![2000.0; expected_len];
    let current_prices: Vec<f64> = vec![2200.0; expected_len]; // Profit of 200
    let entry_iv: Vec<f64> = vec![0.50; expected_len]; // Entered at 50% IV
    let current_iv: Vec<f64> = vec![0.55; expected_len]; // IV increased to 55%
    let option_positions: Vec<f64> = vec![1.0; expected_len]; // Long 1 option
    let option_vegas: Vec<f64> = vec![100.0; expected_len]; // Vega = 100

    // Execute
    let pnls = strategy
        .calculate_pnl_batch(
            &entry_prices,
            &current_prices,
            &entry_iv,
            &current_iv,
            &option_positions,
            &option_vegas,
        )
        .expect("PnL calculation failed");

    // Verify
    assert_eq!(pnls.len(), expected_len);

    for pnl in &pnls {
        // Total P&L = 1.0 × (2200 - 2000) = 200
        assert!(
            (pnl.total_pnl - 200.0).abs() < 0.1,
            "Total P&L mismatch: expected 200, got {}",
            pnl.total_pnl
        );

        // Vol P&L = 1.0 × 100 × (0.55 - 0.50) × 100 = 500
        // (Vega per 1% vol change, so 5pp change = 500)
        assert!(
            (pnl.vol_pnl - 500.0).abs() < 1.0,
            "Vol P&L mismatch: expected 500, got {}",
            pnl.vol_pnl
        );
    }
}

#[test]
#[ignore] // Requires GPU
fn test_vol_arbitrage_pnl_short_position() {
    // Setup: Test P&L for short position
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = VolArbitrageStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 30;
    let n_strategies = 3;
    let expected_len = n_strategies * n_candles;

    // Short position that's profitable
    let entry_prices: Vec<f64> = vec![2500.0; expected_len];
    let current_prices: Vec<f64> = vec![2200.0; expected_len]; // Price decreased
    let entry_iv: Vec<f64> = vec![0.70; expected_len]; // High IV at entry
    let current_iv: Vec<f64> = vec![0.60; expected_len]; // IV decreased
    let option_positions: Vec<f64> = vec![-1.0; expected_len]; // Short 1 option
    let option_vegas: Vec<f64> = vec![80.0; expected_len];

    // Execute
    let pnls = strategy
        .calculate_pnl_batch(
            &entry_prices,
            &current_prices,
            &entry_iv,
            &current_iv,
            &option_positions,
            &option_vegas,
        )
        .expect("PnL calculation failed");

    // Verify
    for pnl in &pnls {
        // Total P&L = -1.0 × (2200 - 2500) = 300 (profit from short)
        assert!(
            (pnl.total_pnl - 300.0).abs() < 0.1,
            "Total P&L mismatch: expected 300, got {}",
            pnl.total_pnl
        );

        // Vol P&L = -1.0 × 80 × (0.60 - 0.70) × 100 = 800 (profit from IV decline)
        assert!(
            (pnl.vol_pnl - 800.0).abs() < 1.0,
            "Vol P&L mismatch: expected 800, got {}",
            pnl.vol_pnl
        );
    }
}

#[test]
#[ignore] // Requires GPU
fn test_vol_arbitrage_batch_performance() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = VolArbitrageStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 500;
    let n_strategies = 1000; // Large batch

    // Create large dataset
    let underlying: Vec<f64> = (0..n_candles).map(|i| 48000.0 + i as f64 * 10.0).collect();
    let option_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
    let option_deltas: Vec<f64> = vec![0.5; n_strategies * n_candles];
    let option_vegas: Vec<f64> = vec![100.0; n_strategies * n_candles];
    let implied_vols: Vec<f64> = vec![0.50; n_strategies * n_candles];
    let historical_vols: Vec<f64> = vec![0.62; n_strategies * n_candles];

    let params = vec![VolArbitrageParams::default(); n_strategies];

    // Execute with timing
    let start = std::time::Instant::now();
    let signals = strategy
        .generate_signals_batch(
            &underlying,
            &option_prices,
            &option_deltas,
            &option_vegas,
            &implied_vols,
            &historical_vols,
            &params,
        )
        .expect("Signal generation failed");
    let elapsed = start.elapsed();

    // Verify performance
    assert_eq!(signals.len(), n_strategies * n_candles);
    println!(
        "Vol Arbitrage GPU: {} strategies × {} candles in {:.2}ms",
        n_strategies,
        n_candles,
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "Throughput: {:.0} signals/sec",
        signals.len() as f64 / elapsed.as_secs_f64()
    );

    // Performance target: <45ms for 1000 strategies × 500 candles
    assert!(
        elapsed.as_millis() < 45,
        "Performance target missed: {:.2}ms > 45ms",
        elapsed.as_secs_f64() * 1000.0
    );
}

#[test]
#[ignore] // Requires GPU
fn test_vol_arbitrage_edge_quality_ranking() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = VolArbitrageStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 20;
    let n_strategies = 5;
    let expected_len = n_strategies * n_candles;

    // Create varying edge qualities
    let implied_vols: Vec<f64> = vec![0.50; expected_len];
    let historical_vols: Vec<f64> = vec![0.60; expected_len]; // 10pp edge
    let option_prices: Vec<f64> = vec![2000.0; expected_len];
    // Varying vegas: higher vega = higher edge quality
    let option_vegas: Vec<f64> = (0..expected_len)
        .map(|i| 50.0 + (i % 5) as f64 * 20.0)
        .collect(); // 50, 70, 90, 110, 130

    // Execute
    let edge_monitors = strategy
        .monitor_edge_batch(
            &implied_vols,
            &historical_vols,
            &option_prices,
            &option_vegas,
        )
        .expect("Edge monitoring failed");

    // Verify: higher vega should result in higher edge quality
    let mut prev_vega = 0.0;
    for (i, edge) in edge_monitors.iter().enumerate() {
        let current_vega = option_vegas[i];
        if i % n_candles == 0 {
            // New strategy group
            prev_vega = 0.0;
        }

        if current_vega > prev_vega + 1.0 {
            // Edge quality should increase with vega (same vol edge)
            let expected_quality = edge.vol_edge.abs() * current_vega * 100.0;
            assert!(
                (edge.edge_quality - expected_quality).abs() < 1.0,
                "Edge quality mismatch: expected {}, got {}",
                expected_quality,
                edge.edge_quality
            );
        }
        prev_vega = current_vega;
    }
}

#[test]
#[ignore] // Requires GPU
fn test_vol_arbitrage_input_validation() {
    // Setup
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let strategy = VolArbitrageStrategyGpu::new(device).expect("Strategy creation failed");

    let n_candles = 10;
    let n_strategies = 2;

    let underlying: Vec<f64> = vec![48000.0; n_candles];
    let option_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
    let option_deltas: Vec<f64> = vec![0.5; n_strategies * n_candles];
    let option_vegas: Vec<f64> = vec![100.0; n_strategies * n_candles];
    let implied_vols: Vec<f64> = vec![0.50; n_strategies * n_candles];

    // Mismatched dimensions (should error)
    let historical_vols_wrong: Vec<f64> = vec![0.60; (n_strategies * n_candles) - 3];

    let params = vec![VolArbitrageParams::default(); n_strategies];

    // Execute
    let result = strategy.generate_signals_batch(
        &underlying,
        &option_prices,
        &option_deltas,
        &option_vegas,
        &implied_vols,
        &historical_vols_wrong,
        &params,
    );

    // Verify: should return error for dimension mismatch
    assert!(result.is_err(), "Expected error for mismatched dimensions");
}
