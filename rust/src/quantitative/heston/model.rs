//! Heston Stochastic Volatility Model
//!
//! Implements the Heston model for option pricing with GPU acceleration.
//!
//! # Model Equations
//!
//! - Asset price: dS_t = μS_t dt + √v_t S_t dW_t^S
//! - Variance:    dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
//! - Correlation: Corr(dW_t^S, dW_t^v) = ρ dt
//!
//! # Parameters
//!
//! - κ (kappa): Mean reversion speed (typical: 0.5 - 5.0)
//! - θ (theta): Long-term variance (typical: 0.01 - 0.1)
//! - σ (sigma): Volatility of volatility (typical: 0.1 - 1.0)
//! - ρ (rho): Correlation between asset and variance (-1.0 to +1.0)
//! - v₀ (v0): Initial variance (current market vol²)

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Heston model parameters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HestonParams {
    /// Mean reversion speed (typical: 0.5 - 5.0)
    /// Higher values → faster return to long-term variance
    pub kappa: f64,

    /// Long-term variance (typical: 0.01 - 0.1)
    /// This is the variance level that volatility reverts to
    /// Note: √θ is the long-term volatility (annualized)
    pub theta: f64,

    /// Volatility of volatility (typical: 0.1 - 1.0)
    /// Higher values → more volatile volatility (vol clustering)
    pub sigma: f64,

    /// Correlation between asset and variance (-1.0 to +1.0)
    /// Negative = leverage effect (vol increases when price drops)
    /// Typical for equities: -0.5 to -0.9
    pub rho: f64,

    /// Initial variance (current market vol²)
    /// For example: if current IV = 20%, then v₀ = 0.04
    pub v0: f64,
}

impl HestonParams {
    /// Create new Heston parameters with validation
    pub fn new(
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
        v0: f64,
    ) -> Result<Self, ValidationError> {
        let params = Self {
            kappa,
            theta,
            sigma,
            rho,
            v0,
        };
        params.validate()?;
        Ok(params)
    }

    /// Validate parameters satisfy Feller condition and bounds
    ///
    /// # Feller Condition
    ///
    /// 2κθ > σ² ensures variance stays positive
    ///
    /// # Returns
    ///
    /// - `Ok(())` if valid
    /// - `Err(ValidationError)` if invalid
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Check parameter positivity first
        if self.kappa <= 0.0 {
            return Err(ValidationError::MustBePositive {
                param: "kappa".to_string(),
                value: self.kappa,
            });
        }
        if self.theta <= 0.0 {
            return Err(ValidationError::MustBePositive {
                param: "theta".to_string(),
                value: self.theta,
            });
        }
        if self.sigma <= 0.0 {
            return Err(ValidationError::MustBePositive {
                param: "sigma".to_string(),
                value: self.sigma,
            });
        }
        if self.v0 <= 0.0 {
            return Err(ValidationError::MustBePositive {
                param: "v0".to_string(),
                value: self.v0,
            });
        }

        // Check correlation bounds
        if !(-1.0..=1.0).contains(&self.rho) {
            return Err(ValidationError::InvalidCorrelation { rho: self.rho });
        }

        // Check Feller condition: 2κθ > σ²
        let kappa_theta = 2.0 * self.kappa * self.theta;
        let sigma_sq = self.sigma.powi(2);

        if kappa_theta <= sigma_sq {
            return Err(ValidationError::FellerCondition {
                kappa_theta,
                sigma_sq,
            });
        }

        Ok(())
    }

    /// Forecast variance at time t
    ///
    /// E[v_t] = v₀e^(-κt) + θ(1 - e^(-κt))
    pub fn forecast_variance(&self, t: f64) -> f64 {
        let exp_term = (-self.kappa * t).exp();
        self.v0 * exp_term + self.theta * (1.0 - exp_term)
    }

    /// Long-term volatility (annualized)
    ///
    /// Returns √θ as percentage (e.g., 0.2 = 20% annualized vol)
    pub fn long_term_vol(&self) -> f64 {
        self.theta.sqrt()
    }

    /// Current volatility (annualized)
    ///
    /// Returns √v₀ as percentage
    pub fn current_vol(&self) -> f64 {
        self.v0.sqrt()
    }
}

/// Option type (Call or Put)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

/// Option Greeks
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Greeks {
    /// Delta: ∂V/∂S (sensitivity to underlying price)
    pub delta: Option<f64>,
    /// Gamma: ∂²V/∂S² (rate of change of delta)
    pub gamma: Option<f64>,
    /// Vega: ∂V/∂σ (sensitivity to volatility)
    pub vega: Option<f64>,
    /// Theta: ∂V/∂t (time decay)
    pub theta: Option<f64>,
    /// Rho: ∂V/∂r (sensitivity to interest rate)
    pub rho_greek: Option<f64>,
}

/// Option quote from market
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptionQuote {
    pub underlying: String,
    pub strike: f64,
    pub expiration: i64, // Unix timestamp
    pub option_type: OptionType,
    pub spot_price: f64,
    pub risk_free_rate: f64,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub last: Option<f64>,
    pub implied_vol: Option<f64>,
    pub volume: f64,
    pub open_interest: f64,
    pub greeks: Option<Greeks>,
}

impl OptionQuote {
    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        if let (Some(bid), Some(ask)) = (self.bid, self.ask) {
            Some((bid + ask) / 2.0)
        } else {
            self.last
        }
    }

    /// Calculate time to expiry in years
    pub fn time_to_expiry(&self, now: i64) -> f64 {
        let seconds_remaining = (self.expiration - now).max(0) as f64;
        seconds_remaining / (365.25 * 24.0 * 3600.0)
    }

    /// Check if in-the-money
    pub fn is_itm(&self) -> bool {
        match self.option_type {
            OptionType::Call => self.spot_price > self.strike,
            OptionType::Put => self.spot_price < self.strike,
        }
    }
}

/// Validation errors for Heston parameters
#[derive(Debug, Clone, Error)]
pub enum ValidationError {
    /// Feller condition violated: 2κθ ≤ σ²
    #[error(
        "Feller condition violated: 2κθ = {kappa_theta:.6} ≤ σ² = {sigma_sq:.6} (variance may become negative)"
    )]
    FellerCondition { kappa_theta: f64, sigma_sq: f64 },

    /// Parameter out of bounds
    #[error("Parameter {param} out of bounds: {value} not in [{min}, {max}]")]
    OutOfBounds {
        param: String,
        value: f64,
        min: f64,
        max: f64,
    },

    /// Invalid correlation
    #[error("Correlation must be in [-1, 1], got {rho}")]
    InvalidCorrelation { rho: f64 },

    /// Must be positive
    #[error("Parameter {param} must be positive, got {value}")]
    MustBePositive { param: String, value: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heston_params_valid() {
        let params = HestonParams::new(
            2.0,  // kappa
            0.04, // theta (20% long-term vol)
            0.3,  // sigma
            -0.7, // rho
            0.04, // v0 (20% current vol)
        );
        assert!(params.is_ok());

        let p = params.unwrap();
        assert_eq!(p.long_term_vol(), 0.2); // √0.04 = 0.2
        assert_eq!(p.current_vol(), 0.2);
    }

    #[test]
    fn test_feller_condition_violation() {
        let params = HestonParams::new(
            1.0,  // kappa
            0.01, // theta
            1.0,  // sigma (too high: σ² = 1.0 > 2κθ = 0.02)
            -0.7, 0.04,
        );
        assert!(matches!(
            params,
            Err(ValidationError::FellerCondition { .. })
        ));
    }

    #[test]
    fn test_forecast_variance() {
        let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.09).unwrap();

        // At t=0, variance should be v0
        assert!((params.forecast_variance(0.0) - 0.09).abs() < 1e-10);

        // At t→∞, variance should approach theta
        assert!((params.forecast_variance(10.0) - 0.04).abs() < 0.001);
    }

    #[test]
    fn test_invalid_rho() {
        let params = HestonParams::new(2.0, 0.04, 0.3, -1.5, 0.04);
        assert!(matches!(
            params,
            Err(ValidationError::InvalidCorrelation { .. })
        ));
    }

    #[test]
    fn test_negative_kappa() {
        let params = HestonParams::new(-1.0, 0.04, 0.3, -0.7, 0.04);
        assert!(matches!(
            params,
            Err(ValidationError::MustBePositive { .. })
        ));
    }

    #[test]
    fn test_negative_theta() {
        let params = HestonParams::new(2.0, -0.04, 0.3, -0.7, 0.04);
        assert!(matches!(
            params,
            Err(ValidationError::MustBePositive { .. })
        ));
    }

    #[test]
    fn test_negative_sigma() {
        let params = HestonParams::new(2.0, 0.04, -0.3, -0.7, 0.04);
        assert!(matches!(
            params,
            Err(ValidationError::MustBePositive { .. })
        ));
    }

    #[test]
    fn test_negative_v0() {
        let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, -0.04);
        assert!(matches!(
            params,
            Err(ValidationError::MustBePositive { .. })
        ));
    }

    #[test]
    fn test_rho_positive_boundary() {
        let params = HestonParams::new(2.0, 0.04, 0.3, 1.0, 0.04);
        assert!(params.is_ok());
    }

    #[test]
    fn test_rho_negative_boundary() {
        let params = HestonParams::new(2.0, 0.04, 0.3, -1.0, 0.04);
        assert!(params.is_ok());
    }

    #[test]
    fn test_rho_out_of_bounds_high() {
        let params = HestonParams::new(2.0, 0.04, 0.3, 1.5, 0.04);
        assert!(matches!(
            params,
            Err(ValidationError::InvalidCorrelation { .. })
        ));
    }

    #[test]
    fn test_serialization() {
        let params = HestonParams {
            kappa: 2.0,
            theta: 0.04,
            sigma: 0.3,
            rho: -0.7,
            v0: 0.04,
        };
        let json = serde_json::to_string(&params).unwrap();
        let deserialized: HestonParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, deserialized);
    }

    #[test]
    fn test_option_quote_mid_price() {
        let quote = OptionQuote {
            underlying: "BTC".to_string(),
            strike: 50000.0,
            expiration: 1735689600,
            option_type: OptionType::Call,
            spot_price: 48000.0,
            risk_free_rate: 0.05,
            bid: Some(1200.0),
            ask: Some(1300.0),
            last: None,
            implied_vol: None,
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        };
        assert_eq!(quote.mid_price(), Some(1250.0));
    }

    #[test]
    fn test_option_quote_mid_price_fallback() {
        let quote = OptionQuote {
            underlying: "BTC".to_string(),
            strike: 50000.0,
            expiration: 1735689600,
            option_type: OptionType::Call,
            spot_price: 48000.0,
            risk_free_rate: 0.05,
            bid: None,
            ask: None,
            last: Some(1225.0),
            implied_vol: None,
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        };
        assert_eq!(quote.mid_price(), Some(1225.0));
    }

    #[test]
    fn test_option_quote_time_to_expiry() {
        let quote = OptionQuote {
            underlying: "BTC".to_string(),
            strike: 50000.0,
            expiration: 1735689600,
            option_type: OptionType::Call,
            spot_price: 48000.0,
            risk_free_rate: 0.05,
            bid: None,
            ask: None,
            last: None,
            implied_vol: None,
            volume: 0.0,
            open_interest: 0.0,
            greeks: None,
        };
        let now = 1735689600 - (30 * 24 * 3600);
        let tte = quote.time_to_expiry(now);
        assert!((tte - 30.0 / 365.25).abs() < 1e-6);
    }

    #[test]
    fn test_option_is_itm_call() {
        let quote = OptionQuote {
            underlying: "BTC".to_string(),
            strike: 50000.0,
            expiration: 1735689600,
            option_type: OptionType::Call,
            spot_price: 51000.0,
            risk_free_rate: 0.05,
            bid: None,
            ask: None,
            last: None,
            implied_vol: None,
            volume: 0.0,
            open_interest: 0.0,
            greeks: None,
        };
        assert!(quote.is_itm());
    }

    #[test]
    fn test_option_is_otm_call() {
        let quote = OptionQuote {
            underlying: "BTC".to_string(),
            strike: 50000.0,
            expiration: 1735689600,
            option_type: OptionType::Call,
            spot_price: 49000.0,
            risk_free_rate: 0.05,
            bid: None,
            ask: None,
            last: None,
            implied_vol: None,
            volume: 0.0,
            open_interest: 0.0,
            greeks: None,
        };
        assert!(!quote.is_itm());
    }

    #[test]
    fn test_option_is_itm_put() {
        let quote = OptionQuote {
            underlying: "BTC".to_string(),
            strike: 50000.0,
            expiration: 1735689600,
            option_type: OptionType::Put,
            spot_price: 49000.0,
            risk_free_rate: 0.05,
            bid: None,
            ask: None,
            last: None,
            implied_vol: None,
            volume: 0.0,
            open_interest: 0.0,
            greeks: None,
        };
        assert!(quote.is_itm());
    }

    #[test]
    fn test_greeks_default() {
        let greeks = Greeks::default();
        assert_eq!(greeks.delta, None);
        assert_eq!(greeks.gamma, None);
        assert_eq!(greeks.vega, None);
    }

    #[test]
    fn test_option_quote_serialization() {
        let quote = OptionQuote {
            underlying: "BTC".to_string(),
            strike: 50000.0,
            expiration: 1735689600,
            option_type: OptionType::Call,
            spot_price: 48000.0,
            risk_free_rate: 0.05,
            bid: Some(1200.0),
            ask: Some(1300.0),
            last: Some(1250.0),
            implied_vol: Some(0.6),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        };
        let json = serde_json::to_string(&quote).unwrap();
        let deserialized: OptionQuote = serde_json::from_str(&json).unwrap();
        assert_eq!(quote, deserialized);
    }
}
