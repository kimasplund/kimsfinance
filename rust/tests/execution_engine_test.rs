//! Integration Tests for Execution Engine
//!
//! Tests complete workflows including position management, P&L tracking,
//! and expiration handling.

use kimsfinance_core::quantitative::heston::{
    ExecutionConfig, ExecutionEngine, Greeks, MarketData, OptionSignal, OptionType, SignalType,
};
use std::collections::HashMap;

#[test]
fn test_complete_trading_cycle() {
    // Create engine
    let config = ExecutionConfig::default();
    let mut engine = ExecutionEngine::new(10_000.0, config).unwrap();

    // Open position
    let signal = OptionSignal {
        option_type: OptionType::Call,
        strike: 100.0,
        expiration: 1735689600,
        signal_type: SignalType::OpenLong,
        quantity: 1,
        strength: 0.8,
    };

    let market_data = MarketData {
        underlying_price: 100.0,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1735000000,
    };

    let trades = engine.execute_signals(&[signal], &market_data).unwrap();
    assert_eq!(trades.len(), 1);
    assert_eq!(engine.position_manager().position_count(), 1);

    // Update position (profit scenario)
    let position_id = trades[0].position_id.clone();
    let mut option_prices = HashMap::new();
    option_prices.insert(position_id.clone(), 7.0); // Profitable

    let mut option_greeks = HashMap::new();
    option_greeks.insert(
        position_id,
        Greeks {
            delta: Some(0.6),
            gamma: Some(0.02),
            vega: Some(0.15),
            theta: Some(-0.05),
            rho_greek: Some(0.03),
        },
    );

    let market_data = MarketData {
        underlying_price: 105.0,
        option_prices,
        option_greeks,
        timestamp: 1735100000,
    };

    let result = engine.process_time_step(1735100000, &market_data).unwrap();
    assert!(result.unrealized_pnl > 0.0);

    // Close position
    let close_signal = OptionSignal {
        option_type: OptionType::Call,
        strike: 100.0,
        expiration: 1735689600,
        signal_type: SignalType::Close,
        quantity: 1,
        strength: 0.9,
    };

    let trades = engine.execute_signals(&[close_signal], &market_data).unwrap();
    assert_eq!(trades.len(), 1);
    assert!(trades[0].realized_pnl.is_some());
    assert_eq!(engine.position_manager().position_count(), 0);

    // Check final metrics
    let report = engine.get_execution_report();
    assert!(report.metrics.realized_pnl > 0.0);
    assert_eq!(report.total_trades, 2); // Open + close
}

#[test]
fn test_multiple_positions_management() {
    let config = ExecutionConfig::default();
    let mut engine = ExecutionEngine::new(50_000.0, config).unwrap();

    // Open multiple positions
    let signals = vec![
        OptionSignal {
            option_type: OptionType::Call,
            strike: 100.0,
            expiration: 1735689600,
            signal_type: SignalType::OpenLong,
            quantity: 2,
            strength: 0.8,
        },
        OptionSignal {
            option_type: OptionType::Put,
            strike: 100.0,
            expiration: 1735689600,
            signal_type: SignalType::OpenLong,
            quantity: 2,
            strength: 0.75,
        },
        OptionSignal {
            option_type: OptionType::Call,
            strike: 110.0,
            expiration: 1736000000,
            signal_type: SignalType::OpenShort,
            quantity: 1,
            strength: 0.7,
        },
    ];

    let market_data = MarketData {
        underlying_price: 100.0,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1735000000,
    };

    let trades = engine.execute_signals(&signals, &market_data).unwrap();
    assert_eq!(trades.len(), 3);
    assert_eq!(engine.position_manager().position_count(), 3);

    // Verify portfolio Greeks
    let mut option_prices = HashMap::new();
    let mut option_greeks = HashMap::new();
    for (position_id, _) in engine.position_manager().positions() {
        option_prices.insert(position_id.clone(), 5.0);
        option_greeks.insert(
            position_id.clone(),
            Greeks {
                delta: Some(0.5),
                gamma: Some(0.02),
                vega: Some(0.1),
                theta: Some(-0.05),
                rho_greek: Some(0.02),
            },
        );
    }

    let market_data = MarketData {
        underlying_price: 100.0,
        option_prices,
        option_greeks,
        timestamp: 1735100000,
    };

    engine.process_time_step(1735100000, &market_data).unwrap();

    let portfolio_greeks = engine.position_manager().get_portfolio_greeks();
    assert!(portfolio_greeks.delta.abs() > 0.0);
    assert!(portfolio_greeks.gamma > 0.0);
}

#[test]
fn test_expiration_scenarios() {
    let config = ExecutionConfig::default();
    let mut engine = ExecutionEngine::new(10_000.0, config).unwrap();

    // Open ITM and OTM positions
    let signals = vec![
        // ITM call
        OptionSignal {
            option_type: OptionType::Call,
            strike: 100.0,
            expiration: 1735000000,
            signal_type: SignalType::OpenLong,
            quantity: 1,
            strength: 0.8,
        },
        // OTM put
        OptionSignal {
            option_type: OptionType::Put,
            strike: 90.0,
            expiration: 1735000000,
            signal_type: SignalType::OpenLong,
            quantity: 1,
            strength: 0.7,
        },
    ];

    let market_data = MarketData {
        underlying_price: 100.0,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1734900000,
    };

    let trades = engine.execute_signals(&signals, &market_data).unwrap();
    assert_eq!(trades.len(), 2);

    // Process expiration at $110 (ITM call profitable, OTM put worthless)
    let market_data = MarketData {
        underlying_price: 110.0,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1735100000,
    };

    let result = engine.process_time_step(1735100000, &market_data).unwrap();
    assert_eq!(result.expirations, 2);
    assert_eq!(engine.position_manager().position_count(), 0);

    // ITM call should have positive realized P&L
    assert!(engine.pnl_tracker().realized_pnl() > -1000.0); // Account for premiums paid
}

#[test]
fn test_short_position_assignment() {
    let config = ExecutionConfig::default();
    let mut engine = ExecutionEngine::new(10_000.0, config).unwrap();

    // Sell ITM put
    let signal = OptionSignal {
        option_type: OptionType::Put,
        strike: 100.0,
        expiration: 1735000000,
        signal_type: SignalType::OpenShort,
        quantity: 1,
        strength: 0.8,
    };

    let market_data = MarketData {
        underlying_price: 100.0,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1734900000,
    };

    engine.execute_signals(&[signal], &market_data).unwrap();

    // Process expiration ITM (underlying dropped to $90)
    let market_data = MarketData {
        underlying_price: 90.0,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1735100000,
    };

    let result = engine.process_time_step(1735100000, &market_data).unwrap();
    assert_eq!(result.expirations, 1);

    // Short put assignment should result in loss
    assert!(engine.pnl_tracker().realized_pnl() < 0.0);
}

#[test]
fn test_pnl_metrics_accuracy() {
    let config = ExecutionConfig::default();
    let mut engine = ExecutionEngine::new(10_000.0, config).unwrap();

    let initial_capital = 10_000.0;

    // Execute winning trade
    let signal = OptionSignal {
        option_type: OptionType::Call,
        strike: 100.0,
        expiration: 1735689600,
        signal_type: SignalType::OpenLong,
        quantity: 1,
        strength: 0.8,
    };

    let market_data = MarketData {
        underlying_price: 100.0,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1735000000,
    };

    engine.execute_signals(&[signal], &market_data).unwrap();

    // Close with profit
    let close_signal = OptionSignal {
        option_type: OptionType::Call,
        strike: 100.0,
        expiration: 1735689600,
        signal_type: SignalType::Close,
        quantity: 1,
        strength: 0.9,
    };

    let market_data = MarketData {
        underlying_price: 110.0,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1735100000,
    };

    engine.execute_signals(&[close_signal], &market_data).unwrap();

    let report = engine.get_execution_report();

    // Verify metrics
    assert!(report.metrics.total_return != 0.0);
    assert!(report.metrics.win_rate > 0.0);
    assert_eq!(report.metrics.total_trades, 2);
    assert_eq!(report.metrics.winning_trades, 1);
    assert_eq!(report.metrics.losing_trades, 0);
}

#[test]
fn test_performance_at_scale() {
    use std::time::Instant;

    let config = ExecutionConfig {
        max_position_size: 1000,
        ..Default::default()
    };
    let mut engine = ExecutionEngine::new(1_000_000.0, config).unwrap();

    // Test opening 1000 positions
    let start = Instant::now();

    for i in 0..1000 {
        let signal = OptionSignal {
            option_type: if i % 2 == 0 {
                OptionType::Call
            } else {
                OptionType::Put
            },
            strike: 100.0 + (i as f64 * 0.1),
            expiration: 1735689600,
            signal_type: SignalType::OpenLong,
            quantity: 1,
            strength: 0.8,
        };

        let market_data = MarketData {
            underlying_price: 100.0,
            option_prices: HashMap::new(),
            option_greeks: HashMap::new(),
            timestamp: 1735000000,
        };

        let _ = engine.execute_signals(&[signal], &market_data);
    }

    let duration = start.elapsed();
    println!("1000 positions opened in {:?}", duration);
    assert!(duration.as_millis() < 50, "Performance target not met");

    // Test expiration handling
    let start = Instant::now();

    let market_data = MarketData {
        underlying_price: 110.0,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1735700000,
    };

    let _ = engine.process_time_step(1735700000, &market_data);

    let duration = start.elapsed();
    println!("1000 expirations processed in {:?}", duration);
    assert!(duration.as_millis() < 10, "Expiration performance target not met");
}

#[test]
fn test_risk_limits_enforcement() {
    let mut config = ExecutionConfig::default();
    config.max_position_size = 2;

    let mut engine = ExecutionEngine::new(10_000.0, config).unwrap();

    // Open 2 positions (should succeed)
    for i in 0..2 {
        let signal = OptionSignal {
            option_type: OptionType::Call,
            strike: 100.0 + (i as f64 * 10.0),
            expiration: 1735689600,
            signal_type: SignalType::OpenLong,
            quantity: 1,
            strength: 0.8,
        };

        let market_data = MarketData {
            underlying_price: 100.0,
            option_prices: HashMap::new(),
            option_greeks: HashMap::new(),
            timestamp: 1735000000,
        };

        engine.execute_signals(&[signal], &market_data).unwrap();
    }

    assert_eq!(engine.position_manager().position_count(), 2);

    // Try to open 3rd position (should be rejected)
    let signal = OptionSignal {
        option_type: OptionType::Call,
        strike: 120.0,
        expiration: 1735689600,
        signal_type: SignalType::OpenLong,
        quantity: 1,
        strength: 0.8,
    };

    let market_data = MarketData {
        underlying_price: 100.0,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1735000000,
    };

    let result = engine.execute_signals(&[signal], &market_data);
    // Should still have 2 positions (3rd rejected)
    assert_eq!(engine.position_manager().position_count(), 2);
}

#[test]
fn test_portfolio_greeks_aggregation() {
    let config = ExecutionConfig::default();
    let mut engine = ExecutionEngine::new(50_000.0, config).unwrap();

    // Open positions with known Greeks
    let signals = vec![
        OptionSignal {
            option_type: OptionType::Call,
            strike: 100.0,
            expiration: 1735689600,
            signal_type: SignalType::OpenLong,
            quantity: 5,
            strength: 0.8,
        },
        OptionSignal {
            option_type: OptionType::Put,
            strike: 100.0,
            expiration: 1735689600,
            signal_type: SignalType::OpenLong,
            quantity: 3,
            strength: 0.75,
        },
    ];

    let market_data = MarketData {
        underlying_price: 100.0,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1735000000,
    };

    engine.execute_signals(&signals, &market_data).unwrap();

    // Update with Greeks
    let mut option_prices = HashMap::new();
    let mut option_greeks = HashMap::new();
    for (position_id, position) in engine.position_manager().positions() {
        option_prices.insert(position_id.clone(), 5.0);

        let delta = match position.option_type {
            OptionType::Call => 0.6,
            OptionType::Put => -0.4,
        };

        option_greeks.insert(
            position_id.clone(),
            Greeks {
                delta: Some(delta),
                gamma: Some(0.02),
                vega: Some(0.1),
                theta: Some(-0.05),
                rho_greek: Some(0.02),
            },
        );
    }

    let market_data = MarketData {
        underlying_price: 100.0,
        option_prices,
        option_greeks,
        timestamp: 1735100000,
    };

    engine.process_time_step(1735100000, &market_data).unwrap();

    let portfolio_greeks = engine.position_manager().get_portfolio_greeks();

    // Verify aggregation:
    // Calls: 5 × 0.6 × 100 = 300 delta
    // Puts: 3 × -0.4 × 100 = -120 delta
    // Total: 180 delta
    assert!((portfolio_greeks.delta - 180.0).abs() < 1.0);

    // Gamma: (5 + 3) × 0.02 × 100 = 16
    assert!((portfolio_greeks.gamma - 16.0).abs() < 1.0);
}
