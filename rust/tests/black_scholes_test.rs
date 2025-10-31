//! Black-Scholes Implied Volatility Tests
//!
//! Tests for the Black-Scholes put pricer and IV solver

// Only compile this test when data-downloaders feature is enabled (which enables strategy module)
#![cfg(feature = "data-downloaders")]

use kimsfinance_core::strategy::BlackScholesPutPricer;

#[test]
fn test_bs_put_atm() {
    // ATM put (spot = strike)
    let price = BlackScholesPutPricer::price(100.0, 100.0, 1.0, 0.05, 0.20);
    // ATM put should cost roughly 7-8 for these parameters
    assert!(price > 7.0 && price < 9.0, "ATM put price: {}", price);
}

#[test]
fn test_bs_put_itm() {
    // ITM put (spot < strike)
    let price = BlackScholesPutPricer::price(90.0, 100.0, 1.0, 0.05, 0.20);
    // ITM put should be worth at least intrinsic value (10)
    assert!(price > 10.0, "ITM put price: {}", price);
}

#[test]
fn test_bs_put_otm() {
    // OTM put (spot > strike)
    let price = BlackScholesPutPricer::price(110.0, 100.0, 1.0, 0.05, 0.20);
    // OTM put should be less than ATM
    assert!(price > 0.0 && price < 5.0, "OTM put price: {}", price);
}

#[test]
fn test_bs_put_zero_tte() {
    // At expiration - should return intrinsic value
    let price = BlackScholesPutPricer::price(90.0, 100.0, 0.0, 0.05, 0.20);
    assert!(
        (price - 10.0).abs() < 1e-10,
        "Expired ITM put should be intrinsic: {}",
        price
    );

    let price_otm = BlackScholesPutPricer::price(110.0, 100.0, 0.0, 0.05, 0.20);
    assert!(
        price_otm < 1e-10,
        "Expired OTM put should be zero: {}",
        price_otm
    );
}

#[test]
fn test_bs_put_zero_vol() {
    // Zero volatility - should return intrinsic value
    let price = BlackScholesPutPricer::price(90.0, 100.0, 1.0, 0.05, 0.0);
    assert!(
        (price - 10.0).abs() < 1e-10,
        "Zero vol ITM put should be intrinsic: {}",
        price
    );
}

#[test]
fn test_vega_positive() {
    // Vega should always be positive
    let vega = BlackScholesPutPricer::vega(100.0, 100.0, 1.0, 0.05, 0.20);
    assert!(vega > 0.0, "Vega should be positive: {}", vega);
}

#[test]
fn test_vega_atm_highest() {
    // ATM options have highest vega
    let vega_atm = BlackScholesPutPricer::vega(100.0, 100.0, 1.0, 0.05, 0.20);
    let vega_itm = BlackScholesPutPricer::vega(90.0, 100.0, 1.0, 0.05, 0.20);
    let vega_otm = BlackScholesPutPricer::vega(110.0, 100.0, 1.0, 0.05, 0.20);

    assert!(vega_atm > vega_itm, "ATM vega > ITM vega");
    assert!(vega_atm > vega_otm, "ATM vega > OTM vega");
}

#[test]
fn test_vega_zero_tte() {
    // Vega should be zero at expiration
    let vega = BlackScholesPutPricer::vega(100.0, 100.0, 0.0, 0.05, 0.20);
    assert!(
        vega.abs() < 1e-10,
        "Vega at expiration should be zero: {}",
        vega
    );
}

#[test]
fn test_iv_solver_convergence() {
    // Test that IV solver recovers known volatility
    let known_vol = 0.20;
    let market_price = BlackScholesPutPricer::price(100.0, 100.0, 0.5, 0.05, known_vol);

    let iv = BlackScholesPutPricer::implied_volatility(market_price, 100.0, 100.0, 0.5, 0.05);

    assert!(iv.is_some(), "IV solver should converge");
    assert!(
        (iv.unwrap() - known_vol).abs() < 0.001,
        "Recovered IV: {}, Expected: {}",
        iv.unwrap(),
        known_vol
    );
}

#[test]
fn test_iv_solver_high_vol() {
    // Test with high volatility (50%)
    let known_vol = 0.50;
    let market_price = BlackScholesPutPricer::price(100.0, 100.0, 0.5, 0.05, known_vol);

    let iv = BlackScholesPutPricer::implied_volatility(market_price, 100.0, 100.0, 0.5, 0.05);

    assert!(iv.is_some(), "IV solver should converge for high vol");
    assert!(
        (iv.unwrap() - known_vol).abs() < 0.001,
        "Recovered IV: {}, Expected: {}",
        iv.unwrap(),
        known_vol
    );
}

#[test]
fn test_iv_solver_low_vol() {
    // Test with low volatility (5%)
    let known_vol = 0.05;
    let market_price = BlackScholesPutPricer::price(100.0, 100.0, 0.5, 0.05, known_vol);

    let iv = BlackScholesPutPricer::implied_volatility(market_price, 100.0, 100.0, 0.5, 0.05);

    assert!(iv.is_some(), "IV solver should converge for low vol");
    assert!(
        (iv.unwrap() - known_vol).abs() < 0.001,
        "Recovered IV: {}, Expected: {}",
        iv.unwrap(),
        known_vol
    );
}

#[test]
fn test_iv_solver_itm() {
    // Test IV solver for ITM option
    let known_vol = 0.25;
    let market_price = BlackScholesPutPricer::price(90.0, 100.0, 0.5, 0.05, known_vol);

    let iv = BlackScholesPutPricer::implied_volatility(market_price, 90.0, 100.0, 0.5, 0.05);

    assert!(iv.is_some(), "IV solver should converge for ITM");
    assert!(
        (iv.unwrap() - known_vol).abs() < 0.001,
        "ITM recovered IV: {}, Expected: {}",
        iv.unwrap(),
        known_vol
    );
}

#[test]
fn test_iv_solver_otm() {
    // Test IV solver for OTM option
    let known_vol = 0.30;
    let market_price = BlackScholesPutPricer::price(110.0, 100.0, 0.5, 0.05, known_vol);

    let iv = BlackScholesPutPricer::implied_volatility(market_price, 110.0, 100.0, 0.5, 0.05);

    assert!(iv.is_some(), "IV solver should converge for OTM");
    assert!(
        (iv.unwrap() - known_vol).abs() < 0.001,
        "OTM recovered IV: {}, Expected: {}",
        iv.unwrap(),
        known_vol
    );
}

#[test]
fn test_iv_solver_invalid_price() {
    // Test that solver rejects invalid prices

    // Negative price
    let iv = BlackScholesPutPricer::implied_volatility(-5.0, 100.0, 100.0, 0.5, 0.05);
    assert!(iv.is_none(), "Should reject negative price");

    // Zero price
    let iv = BlackScholesPutPricer::implied_volatility(0.0, 100.0, 100.0, 0.5, 0.05);
    assert!(iv.is_none(), "Should reject zero price");

    // Below intrinsic value (intrinsic = 10 for spot=90, strike=100)
    let iv = BlackScholesPutPricer::implied_volatility(5.0, 90.0, 100.0, 0.5, 0.05);
    assert!(iv.is_none(), "Should reject price below intrinsic");
}

#[test]
fn test_iv_solver_deep_itm() {
    // Deep ITM option - should still converge
    let known_vol = 0.15;
    let market_price = BlackScholesPutPricer::price(50.0, 100.0, 0.5, 0.05, known_vol);

    let iv = BlackScholesPutPricer::implied_volatility(market_price, 50.0, 100.0, 0.5, 0.05);

    // Deep ITM may be harder to solve accurately, allow more tolerance
    assert!(iv.is_some(), "IV solver should converge for deep ITM");
    if let Some(recovered_iv) = iv {
        assert!(
            (recovered_iv - known_vol).abs() < 0.01,
            "Deep ITM recovered IV: {}, Expected: {}",
            recovered_iv,
            known_vol
        );
    }
}

#[test]
fn test_iv_solver_deep_otm() {
    // Deep OTM option - should still converge
    let known_vol = 0.35;
    let market_price = BlackScholesPutPricer::price(150.0, 100.0, 0.5, 0.05, known_vol);

    let iv = BlackScholesPutPricer::implied_volatility(market_price, 150.0, 100.0, 0.5, 0.05);

    // Deep OTM may be harder to solve, allow more tolerance
    assert!(iv.is_some(), "IV solver should converge for deep OTM");
    if let Some(recovered_iv) = iv {
        assert!(
            (recovered_iv - known_vol).abs() < 0.01,
            "Deep OTM recovered IV: {}, Expected: {}",
            recovered_iv,
            known_vol
        );
    }
}

#[test]
fn test_iv_rank_basic() {
    // Current IV at min should give 0%
    let history = vec![0.10, 0.15, 0.20, 0.25, 0.30];
    let rank = BlackScholesPutPricer::iv_rank(0.10, &history);
    assert!(
        (rank - 0.0).abs() < 1e-6,
        "Min IV should give 0% rank: {}",
        rank
    );

    // Current IV at max should give 100%
    let rank = BlackScholesPutPricer::iv_rank(0.30, &history);
    assert!(
        (rank - 100.0).abs() < 1e-6,
        "Max IV should give 100% rank: {}",
        rank
    );

    // Current IV at midpoint should give 50%
    let rank = BlackScholesPutPricer::iv_rank(0.20, &history);
    assert!(
        (rank - 50.0).abs() < 1e-6,
        "Mid IV should give 50% rank: {}",
        rank
    );
}

#[test]
fn test_iv_rank_52_week_simulation() {
    // Simulate 52-week IV history
    let mut history = Vec::new();
    for week in 0..52 {
        // Simulate varying IV between 15% and 35%
        let iv = 0.15 + 0.20 * (week as f64 / 52.0);
        history.push(iv);
    }

    // Current IV at 30% should be high percentile
    let rank = BlackScholesPutPricer::iv_rank(0.30, &history);
    assert!(rank > 70.0, "High IV should give high rank: {}", rank);

    // Current IV at 18% should be low percentile
    let rank = BlackScholesPutPricer::iv_rank(0.18, &history);
    assert!(rank < 30.0, "Low IV should give low rank: {}", rank);
}

#[test]
fn test_iv_rank_empty_history() {
    // Empty history should default to 50%
    let rank = BlackScholesPutPricer::iv_rank(0.20, &[]);
    assert!(
        (rank - 50.0).abs() < 1e-6,
        "Empty history should default to 50%: {}",
        rank
    );
}

#[test]
fn test_iv_rank_constant_history() {
    // All same values should give 50%
    let history = vec![0.20; 52];
    let rank = BlackScholesPutPricer::iv_rank(0.20, &history);
    assert!(
        (rank - 50.0).abs() < 1e-6,
        "Constant history should give 50%: {}",
        rank
    );
}

#[test]
fn test_iv_rank_with_invalid_values() {
    // History with NaN and infinity should be filtered
    let history = vec![0.10, f64::NAN, 0.20, f64::INFINITY, 0.30];
    let rank = BlackScholesPutPricer::iv_rank(0.20, &history);

    // Should use only valid values (0.10, 0.20, 0.30)
    assert!(
        (rank - 50.0).abs() < 1e-6,
        "Should filter invalid values: {}",
        rank
    );
}

#[test]
fn test_iv_rank_outside_range() {
    // Current IV above max should clamp to 100%
    let history = vec![0.10, 0.15, 0.20];
    let rank = BlackScholesPutPricer::iv_rank(0.50, &history);
    assert!(
        (rank - 100.0).abs() < 1e-6,
        "Above max should clamp to 100%: {}",
        rank
    );

    // Current IV below min should clamp to 0%
    let rank = BlackScholesPutPricer::iv_rank(0.05, &history);
    assert!(
        (rank - 0.0).abs() < 1e-6,
        "Below min should clamp to 0%: {}",
        rank
    );
}
