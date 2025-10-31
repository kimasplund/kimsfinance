//! Parameter constraints for Heston calibration
//!
//! Defines box constraints for L-BFGS-B optimization to ensure
//! parameter values stay within valid ranges.

use serde::{Deserialize, Serialize};

/// Parameter bounds for Heston model calibration
///
/// These bounds ensure parameters stay within physically meaningful
/// and numerically stable ranges during optimization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParameterBounds {
    /// Mean reversion speed bounds (typical: 0.1 - 10.0)
    pub kappa: (f64, f64),

    /// Long-term variance bounds (typical: 0.001 - 0.5)
    /// Corresponds to long-term vol of √θ = 3.2% to 70.7%
    pub theta: (f64, f64),

    /// Volatility of volatility bounds (typical: 0.01 - 2.0)
    pub sigma: (f64, f64),

    /// Correlation bounds (must be in [-1, 1])
    pub rho: (f64, f64),

    /// Initial variance bounds (typical: 0.001 - 0.5)
    pub v0: (f64, f64),
}

impl Default for ParameterBounds {
    /// Default parameter bounds based on market observations
    ///
    /// These are conservative bounds that work well for most assets:
    /// - Equities: Higher vol of vol, negative correlation
    /// - Crypto: Higher variance levels, less correlation
    /// - FX: Lower variance, moderate mean reversion
    fn default() -> Self {
        Self {
            kappa: (0.1, 10.0),  // Mean reversion: slow to very fast
            theta: (0.001, 0.5), // Long-term vol: 3.2% to 70.7%
            sigma: (0.01, 2.0),  // Vol of vol: low to high
            rho: (-1.0, 1.0),    // Full correlation range
            v0: (0.001, 0.5),    // Initial vol: 3.2% to 70.7%
        }
    }
}

impl ParameterBounds {
    /// Create custom parameter bounds
    pub fn new(
        kappa: (f64, f64),
        theta: (f64, f64),
        sigma: (f64, f64),
        rho: (f64, f64),
        v0: (f64, f64),
    ) -> Self {
        Self {
            kappa,
            theta,
            sigma,
            rho,
            v0,
        }
    }

    /// Conservative bounds for equity options
    ///
    /// - Moderate mean reversion (0.5 - 5.0)
    /// - Lower variance levels (1% - 25% vol)
    /// - Strong negative correlation (leverage effect)
    pub fn equity() -> Self {
        Self {
            kappa: (0.5, 5.0),
            theta: (0.0001, 0.0625), // 1% - 25% vol
            sigma: (0.1, 1.0),
            rho: (-0.95, -0.3), // Leverage effect
            v0: (0.0001, 0.0625),
        }
    }

    /// Relaxed bounds for cryptocurrency options
    ///
    /// - Wide mean reversion range
    /// - High variance levels (up to 100% vol)
    /// - Less constrained correlation
    pub fn crypto() -> Self {
        Self {
            kappa: (0.1, 8.0),
            theta: (0.01, 1.0), // 10% - 100% vol
            sigma: (0.1, 3.0),  // Very high vol of vol
            rho: (-0.9, 0.5),   // Less leverage effect
            v0: (0.01, 1.0),
        }
    }

    /// Convert bounds to flat vector format for L-BFGS-B
    ///
    /// Returns (lower_bounds, upper_bounds) where each is a 5-element vector
    /// in the order: [kappa, theta, sigma, rho, v0]
    pub fn to_vectors(&self) -> (Vec<f64>, Vec<f64>) {
        let lower = vec![
            self.kappa.0,
            self.theta.0,
            self.sigma.0,
            self.rho.0,
            self.v0.0,
        ];

        let upper = vec![
            self.kappa.1,
            self.theta.1,
            self.sigma.1,
            self.rho.1,
            self.v0.1,
        ];

        (lower, upper)
    }

    /// Validate that bounds are consistent
    ///
    /// Ensures lower < upper for all parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.kappa.0 >= self.kappa.1 {
            return Err(format!("Invalid kappa bounds: {:?}", self.kappa));
        }
        if self.theta.0 >= self.theta.1 {
            return Err(format!("Invalid theta bounds: {:?}", self.theta));
        }
        if self.sigma.0 >= self.sigma.1 {
            return Err(format!("Invalid sigma bounds: {:?}", self.sigma));
        }
        if self.rho.0 >= self.rho.1 {
            return Err(format!("Invalid rho bounds: {:?}", self.rho));
        }
        if self.v0.0 >= self.v0.1 {
            return Err(format!("Invalid v0 bounds: {:?}", self.v0));
        }

        // Validate correlation bounds
        if self.rho.0 < -1.0 || self.rho.1 > 1.0 {
            return Err(format!("Rho bounds must be in [-1, 1], got {:?}", self.rho));
        }

        // Validate positive parameters
        if self.kappa.0 <= 0.0 || self.theta.0 <= 0.0 || self.sigma.0 <= 0.0 || self.v0.0 <= 0.0 {
            return Err("All parameters except rho must have positive lower bounds".to_string());
        }

        Ok(())
    }

    /// Clamp parameter vector to bounds
    ///
    /// Ensures all parameters are within their respective bounds
    pub fn clamp(&self, params: &[f64; 5]) -> [f64; 5] {
        [
            params[0].clamp(self.kappa.0, self.kappa.1),
            params[1].clamp(self.theta.0, self.theta.1),
            params[2].clamp(self.sigma.0, self.sigma.1),
            params[3].clamp(self.rho.0, self.rho.1),
            params[4].clamp(self.v0.0, self.v0.1),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_bounds() {
        let bounds = ParameterBounds::default();
        assert_eq!(bounds.kappa, (0.1, 10.0));
        assert_eq!(bounds.theta, (0.001, 0.5));
        assert_eq!(bounds.sigma, (0.01, 2.0));
        assert_eq!(bounds.rho, (-1.0, 1.0));
        assert_eq!(bounds.v0, (0.001, 0.5));
        assert!(bounds.validate().is_ok());
    }

    #[test]
    fn test_equity_bounds() {
        let bounds = ParameterBounds::equity();
        assert!(bounds.validate().is_ok());

        // Verify negative correlation (leverage effect)
        assert!(bounds.rho.0 < 0.0);
        assert!(bounds.rho.1 < 0.0);
    }

    #[test]
    fn test_crypto_bounds() {
        let bounds = ParameterBounds::crypto();
        assert!(bounds.validate().is_ok());

        // Verify higher variance levels
        assert!(bounds.theta.1 >= 1.0);
        assert!(bounds.v0.1 >= 1.0);
    }

    #[test]
    fn test_to_vectors() {
        let bounds = ParameterBounds::default();
        let (lower, upper) = bounds.to_vectors();

        assert_eq!(lower.len(), 5);
        assert_eq!(upper.len(), 5);

        assert_eq!(lower[0], bounds.kappa.0);
        assert_eq!(upper[0], bounds.kappa.1);
        assert_eq!(lower[3], bounds.rho.0);
        assert_eq!(upper[3], bounds.rho.1);
    }

    #[test]
    fn test_invalid_bounds() {
        let bounds = ParameterBounds::new(
            (10.0, 1.0), // Invalid: lower > upper
            (0.001, 0.5),
            (0.01, 2.0),
            (-1.0, 1.0),
            (0.001, 0.5),
        );
        assert!(bounds.validate().is_err());
    }

    #[test]
    fn test_clamp() {
        let bounds = ParameterBounds::default();

        // Test clamping values outside bounds
        let params = [15.0, 0.0005, 3.0, -1.5, 0.8];
        let clamped = bounds.clamp(&params);

        assert_eq!(clamped[0], 10.0); // kappa clamped to upper
        assert_eq!(clamped[1], 0.001); // theta clamped to lower
        assert_eq!(clamped[2], 2.0); // sigma clamped to upper
        assert_eq!(clamped[3], -1.0); // rho clamped to lower
        assert_eq!(clamped[4], 0.5); // v0 clamped to upper
    }

    #[test]
    fn test_clamp_within_bounds() {
        let bounds = ParameterBounds::default();

        // Test values already within bounds
        let params = [2.0, 0.04, 0.3, -0.7, 0.04];
        let clamped = bounds.clamp(&params);

        assert_eq!(clamped, params); // Should be unchanged
    }

    #[test]
    fn test_serialization() {
        let bounds = ParameterBounds::default();
        let json = serde_json::to_string(&bounds).unwrap();
        let deserialized: ParameterBounds = serde_json::from_str(&json).unwrap();
        assert_eq!(bounds, deserialized);
    }
}
