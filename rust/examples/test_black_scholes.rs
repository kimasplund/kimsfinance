//! Test Black-Scholes implementation
//!
//! Run with: cargo run --example test_black_scholes

use kimsfinance_core::strategy::BlackScholesPutPricer;

fn main() {
    println!("=== Black-Scholes Put Pricer Tests ===\n");

    // Test 1: ATM Put
    println!("Test 1: ATM Put");
    let price = BlackScholesPutPricer::price(100.0, 100.0, 1.0, 0.05, 0.20);
    println!("  Price: ${:.4}", price);
    assert!(price > 7.0 && price < 9.0, "ATM put price should be 7-9");
    println!("  ✓ Pass\n");

    // Test 2: ITM Put
    println!("Test 2: ITM Put");
    let price = BlackScholesPutPricer::price(90.0, 100.0, 1.0, 0.05, 0.20);
    println!("  Price: ${:.4}", price);
    assert!(price > 10.0, "ITM put should be > intrinsic (10)");
    println!("  ✓ Pass\n");

    // Test 3: OTM Put
    println!("Test 3: OTM Put");
    let price = BlackScholesPutPricer::price(110.0, 100.0, 1.0, 0.05, 0.20);
    println!("  Price: ${:.4}", price);
    assert!(
        price > 0.0 && price < 5.0,
        "OTM put should be positive but small"
    );
    println!("  ✓ Pass\n");

    // Test 4: Vega Calculation
    println!("Test 4: Vega Calculation");
    let vega = BlackScholesPutPricer::vega(100.0, 100.0, 1.0, 0.05, 0.20);
    println!("  Vega: {:.4}", vega);
    assert!(vega > 0.0, "Vega should be positive");
    println!("  ✓ Pass\n");

    // Test 5: Implied Volatility Solver
    println!("Test 5: Implied Volatility Solver");
    let known_vol = 0.20;
    let market_price = BlackScholesPutPricer::price(100.0, 100.0, 0.5, 0.05, known_vol);
    println!("  Market price: ${:.4}", market_price);

    let iv = BlackScholesPutPricer::implied_volatility(market_price, 100.0, 100.0, 0.5, 0.05);
    assert!(iv.is_some(), "IV solver should converge");

    let recovered_iv = iv.unwrap();
    println!("  Known vol: {:.4}", known_vol);
    println!("  Recovered IV: {:.4}", recovered_iv);
    println!("  Error: {:.6}", (recovered_iv - known_vol).abs());

    assert!(
        (recovered_iv - known_vol).abs() < 0.001,
        "IV should be accurate"
    );
    println!("  ✓ Pass\n");

    // Test 6: IV Solver for ITM
    println!("Test 6: IV Solver for ITM Option");
    let known_vol = 0.25;
    let market_price = BlackScholesPutPricer::price(90.0, 100.0, 0.5, 0.05, known_vol);
    let iv = BlackScholesPutPricer::implied_volatility(market_price, 90.0, 100.0, 0.5, 0.05);

    assert!(iv.is_some(), "IV solver should converge for ITM");
    let recovered_iv = iv.unwrap();
    println!("  Known vol: {:.4}", known_vol);
    println!("  Recovered IV: {:.4}", recovered_iv);
    assert!(
        (recovered_iv - known_vol).abs() < 0.001,
        "ITM IV should be accurate"
    );
    println!("  ✓ Pass\n");

    // Test 7: IV Solver for OTM
    println!("Test 7: IV Solver for OTM Option");
    let known_vol = 0.30;
    let market_price = BlackScholesPutPricer::price(110.0, 100.0, 0.5, 0.05, known_vol);
    let iv = BlackScholesPutPricer::implied_volatility(market_price, 110.0, 100.0, 0.5, 0.05);

    assert!(iv.is_some(), "IV solver should converge for OTM");
    let recovered_iv = iv.unwrap();
    println!("  Known vol: {:.4}", known_vol);
    println!("  Recovered IV: {:.4}", recovered_iv);
    assert!(
        (recovered_iv - known_vol).abs() < 0.001,
        "OTM IV should be accurate"
    );
    println!("  ✓ Pass\n");

    // Test 8: IV Rank Calculation
    println!("Test 8: IV Rank Calculation");
    let history = vec![0.15, 0.18, 0.20, 0.25, 0.30, 0.22, 0.19];

    let rank_low = BlackScholesPutPricer::iv_rank(0.16, &history);
    println!("  IV 0.16 rank: {:.2}% (should be low)", rank_low);
    assert!(rank_low < 30.0, "Low IV should give low rank");

    let rank_high = BlackScholesPutPricer::iv_rank(0.28, &history);
    println!("  IV 0.28 rank: {:.2}% (should be high)", rank_high);
    assert!(rank_high > 70.0, "High IV should give high rank");

    let rank_mid = BlackScholesPutPricer::iv_rank(0.225, &history);
    println!("  IV 0.225 rank: {:.2}% (should be ~50%)", rank_mid);
    assert!(
        rank_mid > 40.0 && rank_mid < 60.0,
        "Mid IV should give ~50% rank"
    );
    println!("  ✓ Pass\n");

    // Test 9: IV Rank with 52-week history
    println!("Test 9: IV Rank with 52-Week History");
    let mut history_52w = Vec::new();
    for week in 0..52 {
        let iv = 0.15 + 0.20 * (week as f64 / 52.0);
        history_52w.push(iv);
    }

    let rank_high = BlackScholesPutPricer::iv_rank(0.32, &history_52w);
    println!("  IV 0.32 rank: {:.2}% (should be high)", rank_high);
    assert!(rank_high > 80.0, "High IV in 52w window");

    let rank_low = BlackScholesPutPricer::iv_rank(0.18, &history_52w);
    println!("  IV 0.18 rank: {:.2}% (should be low)", rank_low);
    assert!(rank_low < 20.0, "Low IV in 52w window");
    println!("  ✓ Pass\n");

    // Test 10: Edge Cases
    println!("Test 10: Edge Cases");

    // Zero time to expiration
    let price = BlackScholesPutPricer::price(90.0, 100.0, 0.0, 0.05, 0.20);
    println!("  Expired ITM put: ${:.4} (intrinsic: $10.00)", price);
    assert!(
        (price - 10.0).abs() < 1e-6,
        "Expired ITM should be intrinsic"
    );

    // Zero volatility
    let price = BlackScholesPutPricer::price(90.0, 100.0, 1.0, 0.05, 0.0);
    println!("  Zero vol ITM put: ${:.4} (intrinsic: $10.00)", price);
    assert!((price - 10.0).abs() < 1e-6, "Zero vol should be intrinsic");

    // Invalid IV solver input
    let iv = BlackScholesPutPricer::implied_volatility(-5.0, 100.0, 100.0, 0.5, 0.05);
    assert!(iv.is_none(), "Should reject negative price");
    println!("  Negative price rejected: ✓");

    let iv = BlackScholesPutPricer::implied_volatility(5.0, 90.0, 100.0, 0.5, 0.05);
    assert!(iv.is_none(), "Should reject price below intrinsic");
    println!("  Below-intrinsic price rejected: ✓");

    println!("  ✓ Pass\n");

    println!("=== All Tests Passed! ===");
    println!("\nSummary:");
    println!("  ✓ Black-Scholes put pricing");
    println!("  ✓ Vega calculation");
    println!("  ✓ Implied volatility solver (ATM, ITM, OTM)");
    println!("  ✓ IV rank percentile calculation");
    println!("  ✓ Edge case handling");
}
