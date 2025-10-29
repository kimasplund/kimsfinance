//! Extended Heston model implementation with full features
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Validation errors for Heston parameters
#[derive(Debug, Clone, Error)]
pub enum ValidationError {
    /// Feller condition violated: 2κθ ≤ σ²
    #[error("Feller condition violated: 2κθ = {kappa_theta:.6} ≤ σ² = {sigma_sq:.6}")]
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

/// Greeks for option pricing
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Greeks {
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub vega: Option<f64>,
    pub theta: Option<f64>,
    pub rho: Option<f64>,
}
