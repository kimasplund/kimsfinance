//! Black-Scholes Option Pricing and Implied Volatility Calculation
//!
//! This module provides Black-Scholes pricing for put options and Newton-Raphson
//! implied volatility solver for options strategy backtesting.
//!
//! ## Features
//! - Black-Scholes formula for put options
//! - Newton-Raphson IV solver with configurable tolerance
//! - Vega calculation for IV solver
//! - IV percentile (IV rank) calculation over rolling windows
//! - Edge case handling (deep ITM/OTM, near expiry)
//!
//! ## Performance
//! - Pure analytical formulas (no numerical integration)
//! - Typically converges in 3-5 iterations
//! - Handles edge cases gracefully

use std::f64::consts::PI;

/// Black-Scholes pricer for put options
pub struct BlackScholesPutPricer;

impl BlackScholesPutPricer {
    /// Calculate put option price using Black-Scholes formula
    ///
    /// # Arguments
    /// * `spot_price` - Current price of underlying asset
    /// * `strike` - Strike price of the option
    /// * `time_to_exp` - Time to expiration in years
    /// * `rate` - Risk-free interest rate (annual, as decimal, e.g., 0.05 for 5%)
    /// * `volatility` - Volatility (annual, as decimal, e.g., 0.25 for 25%)
    ///
    /// # Returns
    /// Option price
    ///
    /// # Examples
    /// ```
    /// use kimsfinance_core::strategy::BlackScholesPutPricer;
    ///
    /// let price = BlackScholesPutPricer::price(100.0, 100.0, 0.5, 0.05, 0.20);
    /// assert!(price > 0.0);
    /// ```
    pub fn price(
        spot_price: f64,
        strike: f64,
        time_to_exp: f64,
        rate: f64,
        volatility: f64,
    ) -> f64 {
        // Handle edge cases
        if time_to_exp <= 0.0 {
            // Option expired - return intrinsic value
            return (strike - spot_price).max(0.0);
        }

        if volatility <= 0.0 {
            // Zero volatility - return intrinsic value
            return (strike - spot_price).max(0.0);
        }

        // Compute d1 and d2
        let (d1, d2) = Self::compute_d1_d2(spot_price, strike, time_to_exp, rate, volatility);

        // Black-Scholes formula for put option
        // P = K * e^(-rT) * N(-d2) - S * N(-d1)
        let discount_factor = (-rate * time_to_exp).exp();
        let price =
            strike * discount_factor * Self::norm_cdf(-d2) - spot_price * Self::norm_cdf(-d1);

        // Ensure non-negative price
        price.max(0.0)
    }

    /// Calculate vega (sensitivity to volatility) for a put option
    ///
    /// Vega is the same for calls and puts (put-call parity derivative)
    ///
    /// # Arguments
    /// * `spot` - Current price of underlying
    /// * `strike` - Strike price
    /// * `tte` - Time to expiration in years
    /// * `rate` - Risk-free rate (annual decimal)
    /// * `vol` - Volatility (annual decimal)
    ///
    /// # Returns
    /// Vega (price change per 1% volatility change)
    pub fn vega(spot: f64, strike: f64, tte: f64, rate: f64, vol: f64) -> f64 {
        if tte <= 0.0 || vol <= 0.0 {
            return 0.0;
        }

        let (d1, _) = Self::compute_d1_d2(spot, strike, tte, rate, vol);
        let sqrt_t = tte.sqrt();

        // Vega = S * N'(d1) * sqrt(T)
        // Divide by 100 to get change per 1% vol change
        spot * Self::norm_pdf(d1) * sqrt_t / 100.0
    }

    /// Calculate implied volatility using Newton-Raphson method
    ///
    /// # Arguments
    /// * `option_market_price` - Observed market price of the option
    /// * `spot` - Current price of underlying
    /// * `strike` - Strike price
    /// * `tte` - Time to expiration in years
    /// * `rate` - Risk-free rate (annual decimal)
    ///
    /// # Returns
    /// * `Some(iv)` - Implied volatility if solver converges
    /// * `None` - If solver fails to converge or inputs are invalid
    ///
    /// # Examples
    /// ```
    /// use kimsfinance_core::strategy::BlackScholesPutPricer;
    ///
    /// // Market price of ATM put with 20% vol should recover ~0.20 IV
    /// let market_price = BlackScholesPutPricer::price(100.0, 100.0, 0.5, 0.05, 0.20);
    /// let iv = BlackScholesPutPricer::implied_volatility(market_price, 100.0, 100.0, 0.5, 0.05);
    /// assert!(iv.is_some());
    /// assert!((iv.unwrap() - 0.20).abs() < 0.001);
    /// ```
    pub fn implied_volatility(
        option_market_price: f64,
        spot: f64,
        strike: f64,
        tte: f64,
        rate: f64,
    ) -> Option<f64> {
        // Input validation
        if option_market_price <= 0.0 || spot <= 0.0 || strike <= 0.0 || tte <= 0.0 {
            return None;
        }

        // Check if price is below intrinsic value
        let intrinsic = (strike - spot).max(0.0);
        if option_market_price < intrinsic - 1e-6 {
            return None; // Invalid price - below intrinsic value
        }

        // Newton-Raphson parameters
        const MAX_ITERATIONS: usize = 100;
        const TOLERANCE: f64 = 0.0001;
        const MIN_VOL: f64 = 0.0001; // 0.01%
        const MAX_VOL: f64 = 5.0; // 500%

        // Initial guess: 25% IV
        let mut vol = 0.25;

        for _ in 0..MAX_ITERATIONS {
            // Calculate theoretical price and vega at current vol estimate
            let theo_price = Self::price(spot, strike, tte, rate, vol);
            let vega_val = Self::vega(spot, strike, tte, rate, vol);

            // Price difference
            let price_diff = theo_price - option_market_price;

            // Check convergence
            if price_diff.abs() < TOLERANCE {
                return Some(vol);
            }

            // Check if vega is too small (can't improve)
            if vega_val.abs() < 1e-10 {
                return None;
            }

            // Newton-Raphson update: vol_new = vol_old - f(vol) / f'(vol)
            // f(vol) = BS_price(vol) - market_price
            // f'(vol) = vega
            vol -= price_diff / vega_val;

            // Clamp to reasonable range
            vol = vol.clamp(MIN_VOL, MAX_VOL);
        }

        // Failed to converge
        None
    }

    /// Calculate IV rank (percentile) over a rolling window
    ///
    /// IV rank shows where current IV stands relative to its historical range.
    /// Higher values indicate higher relative volatility.
    ///
    /// # Arguments
    /// * `current_iv` - Current implied volatility
    /// * `iv_history` - Historical IV values (e.g., 52-week window)
    ///
    /// # Returns
    /// IV rank as percentage (0-100)
    ///
    /// # Formula
    /// IV Rank = (current_iv - min_iv) / (max_iv - min_iv) * 100
    ///
    /// # Examples
    /// ```
    /// use kimsfinance_core::strategy::BlackScholesPutPricer;
    ///
    /// let history = vec![0.15, 0.18, 0.20, 0.25, 0.30, 0.22, 0.19];
    /// let rank = BlackScholesPutPricer::iv_rank(0.28, &history);
    /// assert!(rank > 80.0); // 0.28 is high in this range
    /// ```
    pub fn iv_rank(current_iv: f64, iv_history: &[f64]) -> f64 {
        if iv_history.is_empty() {
            return 50.0; // Default to middle if no history
        }

        // Find min and max IV in history
        let mut min_iv = f64::INFINITY;
        let mut max_iv = f64::NEG_INFINITY;

        for &iv in iv_history {
            if iv.is_finite() {
                min_iv = min_iv.min(iv);
                max_iv = max_iv.max(iv);
            }
        }

        // Handle edge cases
        if !min_iv.is_finite() || !max_iv.is_finite() {
            return 50.0;
        }

        let range = max_iv - min_iv;
        if range <= 0.0 {
            return 50.0; // All values the same
        }

        // Calculate percentile
        let rank = (current_iv - min_iv) / range * 100.0;

        // Clamp to [0, 100]
        rank.clamp(0.0, 100.0)
    }

    /// Helper: Compute d1 and d2 for Black-Scholes
    fn compute_d1_d2(spot: f64, strike: f64, tte: f64, rate: f64, vol: f64) -> (f64, f64) {
        let sqrt_t = tte.sqrt();
        let vol_sqrt_t = vol * sqrt_t;

        let d1 = ((spot / strike).ln() + (rate + 0.5 * vol * vol) * tte) / vol_sqrt_t;
        let d2 = d1 - vol_sqrt_t;

        (d1, d2)
    }

    /// Cumulative distribution function for standard normal distribution
    ///
    /// Uses Abramowitz and Stegun approximation (accurate to ~7 decimal places)
    fn norm_cdf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let prob = d
            * t
            * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

        if x >= 0.0 { 1.0 - prob } else { prob }
    }

    /// Probability density function for standard normal distribution
    fn norm_pdf(x: f64) -> f64 {
        (1.0 / (2.0 * PI).sqrt()) * (-0.5 * x * x).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_put_call_parity_consistency() {
        // While this is a put pricer, let's verify internal consistency
        // by checking that intrinsic value bounds are respected

        let spot = 100.0;
        let strike = 100.0;
        let tte = 1.0;
        let rate = 0.05;
        let vol = 0.20;

        let put_price = BlackScholesPutPricer::price(spot, strike, tte, rate, vol);
        let intrinsic = (strike - spot).max(0.0);

        // Put price should be >= intrinsic value
        assert!(
            put_price >= intrinsic - 1e-6,
            "Put price {} should be >= intrinsic {}",
            put_price,
            intrinsic
        );
    }

    #[test]
    fn test_edge_case_near_zero_time() {
        // Very small time to expiration (1 hour)
        let tte = 1.0 / (365.0 * 24.0); // 1 hour in years
        let price = BlackScholesPutPricer::price(100.0, 100.0, tte, 0.05, 0.20);

        // Price should be very small but positive
        assert!(price >= 0.0, "Price should be non-negative");
        assert!(price < 1.0, "Price should be small for near-expiry ATM");
    }

    #[test]
    fn test_edge_case_very_high_vol() {
        // Test with very high volatility (200%)
        let price = BlackScholesPutPricer::price(100.0, 100.0, 1.0, 0.05, 2.0);

        // Should still produce valid price
        assert!(
            price > 0.0 && price < 150.0,
            "High vol price should be valid: {}",
            price
        );
    }
}
