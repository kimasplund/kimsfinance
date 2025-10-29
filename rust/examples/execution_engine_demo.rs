//! Execution Engine Demo
//!
//! Demonstrates the complete options execution engine with:
//! - Position management
//! - P&L tracking
//! - Expiration handling
//! - Performance metrics

use kimsfinance_core::quantitative::heston::{
    ExecutionConfig, ExecutionEngine, Greeks, MarketData, OptionSignal, OptionType, SignalType,
};
use std::collections::HashMap;

fn main() {
    println!("=== Options Execution Engine Demo ===\n");

    // Initialize engine with $100,000 capital
    let config = ExecutionConfig {
        trading_fee: 1.0,     // $1 per contract
        slippage: 0.0005,     // 0.05%
        max_position_size: 50,
        margin_requirement: 0.2,
    };

    let mut engine = ExecutionEngine::new(100_000.0, config).expect("Failed to create engine");

    println!("Initial Capital: ${:.2}\n", 100_000.0);

    // Scenario: Trading options on SPX at $4,500
    let underlying_price = 4500.0;

    // Generate signals for various strategies
    println!("=== Phase 1: Open Positions ===");

    let signals = vec![
        // Long call spread (bullish)
        OptionSignal {
            option_type: OptionType::Call,
            strike: 4500.0,
            expiration: 1735689600, // 30 days out
            signal_type: SignalType::OpenLong,
            quantity: 5,
            strength: 0.85,
        },
        OptionSignal {
            option_type: OptionType::Call,
            strike: 4550.0,
            expiration: 1735689600,
            signal_type: SignalType::OpenShort,
            quantity: 5,
            strength: 0.85,
        },
        // Long straddle (volatility play)
        OptionSignal {
            option_type: OptionType::Call,
            strike: 4500.0,
            expiration: 1736000000, // 60 days out
            signal_type: SignalType::OpenLong,
            quantity: 3,
            strength: 0.75,
        },
        OptionSignal {
            option_type: OptionType::Put,
            strike: 4500.0,
            expiration: 1736000000,
            signal_type: SignalType::OpenLong,
            quantity: 3,
            strength: 0.75,
        },
    ];

    let market_data = MarketData {
        underlying_price,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1735000000,
    };

    let trades = engine
        .execute_signals(&signals, &market_data)
        .expect("Failed to execute signals");

    println!("Executed {} trades:", trades.len());
    for trade in &trades {
        println!(
            "  - {:?} {} @ ${} (qty: {})",
            trade.option_type, trade.strike, trade.price, trade.quantity
        );
    }

    println!(
        "\nActive Positions: {}",
        engine.position_manager().position_count()
    );
    println!("Cash Remaining: ${:.2}\n", engine.position_manager().cash());

    // Update positions after price movement
    println!("=== Phase 2: Market Update (Price Up 2%) ===");

    let new_underlying_price = 4590.0; // 2% up
    let mut option_prices = HashMap::new();
    let mut option_greeks = HashMap::new();

    // Simulate option prices and Greeks for open positions
    for (position_id, position) in engine.position_manager().positions() {
        // Simplified pricing: calls up, puts down
        let price_change = match position.option_type {
            OptionType::Call if position.strike <= new_underlying_price => 15.0,
            OptionType::Call => 2.0,
            OptionType::Put if position.strike >= new_underlying_price => 2.0,
            OptionType::Put => 8.0,
        };

        option_prices.insert(position_id.clone(), position.entry_price + price_change);

        option_greeks.insert(
            position_id.clone(),
            Greeks {
                delta: Some(0.6),
                gamma: Some(0.02),
                vega: Some(0.15),
                theta: Some(-0.05),
                rho_greek: Some(0.03),
            },
        );
    }

    let market_data = MarketData {
        underlying_price: new_underlying_price,
        option_prices,
        option_greeks,
        timestamp: 1735086400, // 1 day later
    };

    let result = engine
        .process_time_step(1735086400, &market_data)
        .expect("Failed to process time step");

    println!("Current Equity: ${:.2}", result.current_equity);
    println!("Unrealized P&L: ${:.2}", result.unrealized_pnl);
    println!("Portfolio Greeks:");
    let portfolio_greeks = engine.position_manager().get_portfolio_greeks();
    println!("  Delta: {:.2}", portfolio_greeks.delta);
    println!("  Gamma: {:.2}", portfolio_greeks.gamma);
    println!("  Vega: {:.2}", portfolio_greeks.vega);
    println!("  Theta: {:.2}", portfolio_greeks.theta);

    // Close some positions
    println!("\n=== Phase 3: Close Profitable Positions ===");

    let close_signals = vec![OptionSignal {
        option_type: OptionType::Call,
        strike: 4500.0,
        expiration: 1735689600,
        signal_type: SignalType::Close,
        quantity: 5,
        strength: 0.9,
    }];

    let trades = engine
        .execute_signals(&close_signals, &market_data)
        .expect("Failed to close positions");

    for trade in &trades {
        if let Some(pnl) = trade.realized_pnl {
            println!(
                "Closed {:?} {} @ ${}: Realized P&L = ${:.2}",
                trade.option_type, trade.strike, trade.price, pnl
            );
        }
    }

    // Handle expirations
    println!("\n=== Phase 4: Expiration (30 days later) ===");

    let expiration_time = 1735700000;
    let final_underlying = 4520.0;

    let market_data = MarketData {
        underlying_price: final_underlying,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: expiration_time,
    };

    let result = engine
        .process_time_step(expiration_time, &market_data)
        .expect("Failed to process expirations");

    println!("Processed {} expirations", result.expirations);
    println!(
        "Remaining Positions: {}",
        engine.position_manager().position_count()
    );

    // Final report
    println!("\n=== Final Performance Report ===\n");

    let report = engine.get_execution_report();
    println!("{}", report.to_string());

    // Performance benchmarks
    println!("\n=== Performance Validation ===");

    // Simulate 1000 positions for performance test
    let start = std::time::Instant::now();
    let mut test_engine = ExecutionEngine::new(1_000_000.0, ExecutionConfig::default()).unwrap();

    // Open 1000 positions
    for i in 0..1000 {
        let signal = OptionSignal {
            option_type: if i % 2 == 0 {
                OptionType::Call
            } else {
                OptionType::Put
            },
            strike: 4500.0 + (i as f64 * 10.0),
            expiration: 1735689600,
            signal_type: SignalType::OpenLong,
            quantity: 1,
            strength: 0.8,
        };

        let market_data = MarketData {
            underlying_price: 4500.0,
            option_prices: HashMap::new(),
            option_greeks: HashMap::new(),
            timestamp: 1735000000,
        };

        let _ = test_engine.execute_signals(&[signal], &market_data);
    }

    let open_duration = start.elapsed();
    println!(
        "1000 positions opened in {:.2}ms (target: <50ms)",
        open_duration.as_micros() as f64 / 1000.0
    );

    // Check expirations
    let start = std::time::Instant::now();
    let market_data = MarketData {
        underlying_price: 4500.0,
        option_prices: HashMap::new(),
        option_greeks: HashMap::new(),
        timestamp: 1735700000,
    };
    let _ = test_engine.process_time_step(1735700000, &market_data);
    let expiration_duration = start.elapsed();
    println!(
        "1000 expirations processed in {:.2}ms (target: <10ms)",
        expiration_duration.as_micros() as f64 / 1000.0
    );

    // Validate targets
    println!("\nPerformance Targets:");
    println!(
        "  ✓ 1000 positions: {} (target: <50ms)",
        if open_duration.as_millis() < 50 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!(
        "  ✓ Expiration checks: {} (target: <10ms)",
        if expiration_duration.as_millis() < 10 {
            "PASS"
        } else {
            "FAIL"
        }
    );

    println!("\n=== Demo Complete ===");
}
