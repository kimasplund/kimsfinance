//! Black-Scholes Option Pricing
//!
//! CPU-based analytical option pricing using the Black-Scholes formula.
//! Used as a fallback when GPU FFT pricing fails or produces invalid results.

use crate::quantitative::heston::{Greeks, OptionQuote, OptionType};
use std::f64::consts::PI;

/// Black-Scholes pricer with analytical Greeks
pub struct BlackScholesPricer;

impl BlackScholesPricer {
    /// Price a single option using Black-Scholes formula
    pub fn price(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        volatility: f64,
        option_type: OptionType,
    ) -> f64 {
        // Handle edge cases
        if time_to_expiry <= 0.0 || volatility <= 0.0 {
            return Self::intrinsic_value(spot, strike, option_type);
        }

        // Compute d1 and d2
        let (d1, d2) =
            Self::compute_d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility);

        // Compute price based on option type
        let price = match option_type {
            OptionType::Call => {
                spot * Self::norm_cdf(d1)
                    - strike * (-risk_free_rate * time_to_expiry).exp() * Self::norm_cdf(d2)
            }
            OptionType::Put => {
                strike * (-risk_free_rate * time_to_expiry).exp() * Self::norm_cdf(-d2)
                    - spot * Self::norm_cdf(-d1)
            }
        };

        price.max(0.0)
    }

    /// Compute Greeks for a single option
    pub fn greeks(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        volatility: f64,
        option_type: OptionType,
    ) -> Greeks {
        if time_to_expiry <= 0.0 || volatility <= 0.0 {
            return Greeks::default();
        }

        let (d1, d2) =
            Self::compute_d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility);
        let sqrt_t = time_to_expiry.sqrt();
        let exp_rt = (-risk_free_rate * time_to_expiry).exp();
        let n_d1 = Self::norm_pdf(d1);

        let gamma = n_d1 / (spot * volatility * sqrt_t);
        let vega = spot * n_d1 * sqrt_t / 100.0;

        match option_type {
            OptionType::Call => {
                let delta = Self::norm_cdf(d1);
                let theta = -(spot * n_d1 * volatility) / (2.0 * sqrt_t)
                    - risk_free_rate * strike * exp_rt * Self::norm_cdf(d2);
                let theta = theta / 365.0;
                let rho = strike * time_to_expiry * exp_rt * Self::norm_cdf(d2) / 100.0;

                Greeks {
                    delta: Some(delta),
                    gamma: Some(gamma),
                    vega: Some(vega),
                    theta: Some(theta),
                    rho_greek: Some(rho),
                }
            }
            OptionType::Put => {
                let delta = Self::norm_cdf(d1) - 1.0;
                let theta = -(spot * n_d1 * volatility) / (2.0 * sqrt_t)
                    + risk_free_rate * strike * exp_rt * Self::norm_cdf(-d2);
                let theta = theta / 365.0;
                let rho = -strike * time_to_expiry * exp_rt * Self::norm_cdf(-d2) / 100.0;

                Greeks {
                    delta: Some(delta),
                    gamma: Some(gamma),
                    vega: Some(vega),
                    theta: Some(theta),
                    rho_greek: Some(rho),
                }
            }
        }
    }

    /// Validate option price is reasonable
    pub fn is_valid_price(
        price: f64,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        option_type: OptionType,
    ) -> bool {
        if !price.is_finite() || price < 0.0 {
            return false;
        }

        let intrinsic = Self::intrinsic_value(spot, strike, option_type);
        if price < intrinsic - 1e-6 {
            return false;
        }

        match option_type {
            OptionType::Call => price <= spot,
            OptionType::Put => {
                let discounted_strike = strike * (-risk_free_rate * time_to_expiry).exp();
                price <= discounted_strike
            }
        }
    }

    // Helper functions
    fn compute_d1_d2(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        volatility: f64,
    ) -> (f64, f64) {
        let sqrt_t = time_to_expiry.sqrt();
        let d1 = ((spot / strike).ln()
            + (risk_free_rate + 0.5 * volatility * volatility) * time_to_expiry)
            / (volatility * sqrt_t);
        let d2 = d1 - volatility * sqrt_t;
        (d1, d2)
    }

    fn intrinsic_value(spot: f64, strike: f64, option_type: OptionType) -> f64 {
        match option_type {
            OptionType::Call => (spot - strike).max(0.0),
            OptionType::Put => (strike - spot).max(0.0),
        }
    }

    fn norm_cdf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let prob = d
            * t
            * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

        if x >= 0.0 { 1.0 - prob } else { prob }
    }

    fn norm_pdf(x: f64) -> f64 {
        (1.0 / (2.0 * PI).sqrt()) * (-0.5 * x * x).exp()
    }

    pub fn compute_time_to_expiry(expiration_timestamp: i64) -> Option<f64> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs() as i64;
        let seconds_to_expiry = expiration_timestamp - now;
        if seconds_to_expiry <= 0 {
            Some(0.0)
        } else {
            Some(seconds_to_expiry as f64 / (365.25 * 24.0 * 3600.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atm_call() {
        let price = BlackScholesPricer::price(100.0, 100.0, 1.0, 0.05, 0.20, OptionType::Call);
        assert!((price - 10.45).abs() < 0.1, "ATM call price: {}", price);
    }

    #[test]
    fn test_put_call_parity() {
        let call = BlackScholesPricer::price(100.0, 100.0, 1.0, 0.05, 0.20, OptionType::Call);
        let put = BlackScholesPricer::price(100.0, 100.0, 1.0, 0.05, 0.20, OptionType::Put);
        let lhs = call - put;
        let rhs = 100.0 - 100.0 * (-0.05f64).exp();
        assert!((lhs - rhs).abs() < 1e-6);
    }

    #[test]
    fn test_price_validation() {
        assert!(BlackScholesPricer::is_valid_price(
            10.0,
            100.0,
            100.0,
            1.0,
            0.05,
            OptionType::Call
        ));
        assert!(!BlackScholesPricer::is_valid_price(
            -1.0,
            100.0,
            100.0,
            1.0,
            0.05,
            OptionType::Call
        ));
        assert!(!BlackScholesPricer::is_valid_price(
            f64::NAN,
            100.0,
            100.0,
            1.0,
            0.05,
            OptionType::Call
        ));
        assert!(!BlackScholesPricer::is_valid_price(
            150.0,
            100.0,
            100.0,
            1.0,
            0.05,
            OptionType::Call
        ));
    }
}
