//! Validation errors for Heston model parameters

use thiserror::Error;

/// Errors that can occur during Heston parameter validation
#[derive(Debug, Clone, Error)]
pub enum ValidationError {
    /// Feller condition violated: 2κθ ≤ σ²
    ///
    /// This condition ensures that variance stays positive.
    /// When violated, the variance process can become negative.
    #[error(
        "Feller condition violated: 2κθ = {kappa_theta:.6} ≤ σ² = {sigma_sq:.6} (variance may become negative)"
    )]
    FellerCondition { kappa_theta: f64, sigma_sq: f64 },

    /// Parameter out of valid bounds
    #[error("Parameter {param} out of bounds: {value} not in [{min}, {max}]")]
    OutOfBounds {
        param: String,
        value: f64,
        min: f64,
        max: f64,
    },

    /// Invalid correlation value
    #[error("Correlation must be in [-1, 1], got {rho}")]
    InvalidCorrelation { rho: f64 },

    /// Parameter must be positive
    #[error("Parameter {param} must be positive, got {value}")]
    MustBePositive { param: String, value: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feller_error_display() {
        let err = ValidationError::FellerCondition {
            kappa_theta: 0.02,
            sigma_sq: 1.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Feller condition"));
        assert!(msg.contains("0.02"));
        assert!(msg.contains("1.0"));
    }

    #[test]
    fn test_out_of_bounds_error() {
        let err = ValidationError::OutOfBounds {
            param: "kappa".to_string(),
            value: 10.0,
            min: 0.1,
            max: 5.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("kappa"));
        assert!(msg.contains("10"));
        assert!(msg.contains("0.1"));
        assert!(msg.contains("5"));
    }
}
