//! Objective function for Heston calibration
//!
//! Implements the cost function that measures the error between
//! model prices and market prices, optimized for GPU execution.

#[cfg(feature = "heston")]
use crate::gpu::heston_pricing::HestonGpuPricer;
#[cfg(feature = "heston")]
use crate::quantitative::heston::{HestonParams, OptionQuote};
#[cfg(feature = "heston")]
use argmin::core::{CostFunction, Error, Gradient};
#[cfg(feature = "heston")]
use ndarray::Array1;
#[cfg(feature = "heston")]
use parking_lot::Mutex;
#[cfg(feature = "heston")]
use std::sync::Arc;

/// Objective function for Heston model calibration
///
/// This function calculates the sum of squared errors (SSE) between
/// market option prices and model prices computed using the GPU-accelerated
/// Heston pricer.
///
/// # Cost Function
///
/// SSE = Σ(market_price - model_price)²
///
/// # Parameters
///
/// The parameter vector contains 5 elements in this order:
/// - [0]: κ (kappa) - mean reversion speed
/// - [1]: θ (theta) - long-term variance
/// - [2]: σ (sigma) - volatility of volatility
/// - [3]: ρ (rho) - correlation
/// - [4]: v₀ (v0) - initial variance
///
/// # Invalid Parameters
///
/// If parameters violate the Feller condition or are out of bounds,
/// the function returns `f64::INFINITY` to guide the optimizer away.
#[cfg(feature = "heston")]
pub struct HestonObjective {
    /// GPU pricer for fast batch pricing (wrapped in Mutex for interior mutability)
    pub gpu_pricer: Arc<Mutex<HestonGpuPricer>>,

    /// Market option quotes with observed prices
    pub market_options: Vec<OptionQuote>,

    /// Weighting scheme for different options
    pub weights: Option<Vec<f64>>,
}

#[cfg(feature = "heston")]
impl HestonObjective {
    /// Create new objective function
    ///
    /// # Arguments
    ///
    /// * `gpu_pricer` - GPU-accelerated Heston pricer (wrapped in Mutex)
    /// * `market_options` - Market option quotes (must have bid/ask or last price)
    pub fn new(gpu_pricer: Arc<Mutex<HestonGpuPricer>>, market_options: Vec<OptionQuote>) -> Self {
        Self {
            gpu_pricer,
            market_options,
            weights: None,
        }
    }

    /// Create objective function with custom weights
    ///
    /// Weights can be used to:
    /// - Emphasize near-the-money options
    /// - Down-weight illiquid options
    /// - Prioritize specific maturities
    ///
    /// # Panics
    ///
    /// Panics if weights.len() != market_options.len()
    pub fn with_weights(
        gpu_pricer: Arc<Mutex<HestonGpuPricer>>,
        market_options: Vec<OptionQuote>,
        weights: Vec<f64>,
    ) -> Self {
        assert_eq!(
            weights.len(),
            market_options.len(),
            "Weights must match number of options"
        );
        Self {
            gpu_pricer,
            market_options,
            weights: Some(weights),
        }
    }

    /// Convert parameter vector to HestonParams
    fn vec_to_params(&self, param: &Array1<f64>) -> HestonParams {
        HestonParams {
            kappa: param[0],
            theta: param[1],
            sigma: param[2],
            rho: param[3],
            v0: param[4],
        }
    }

    /// Calculate weighted sum of squared errors
    fn calculate_sse(&self, model_prices: &[f64]) -> f64 {
        let mut sse = 0.0;

        for (i, option) in self.market_options.iter().enumerate() {
            if let Some(market_price) = option.mid_price() {
                let model_price = model_prices[i];
                let error = market_price - model_price;
                let squared_error = error * error;

                // Apply weight if provided
                let weighted_error = if let Some(ref weights) = self.weights {
                    squared_error * weights[i]
                } else {
                    squared_error
                };

                sse += weighted_error;
            }
        }

        sse
    }
}

#[cfg(feature = "heston")]
impl CostFunction for HestonObjective {
    type Param = Array1<f64>;
    type Output = f64;

    /// Evaluate the cost function for given parameters
    ///
    /// # Returns
    ///
    /// - Sum of squared errors if parameters are valid
    /// - `f64::INFINITY` if parameters are invalid (guides optimizer away)
    ///
    /// # Errors
    ///
    /// Returns error if GPU pricing fails (rare)
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        // Convert parameter vector to HestonParams
        let params = self.vec_to_params(param);

        // Validate parameters (Feller condition, bounds)
        if params.validate().is_err() {
            // Invalid parameters: return infinity to guide optimizer
            return Ok(f64::INFINITY);
        }

        // Price options with GPU (lock the pricer for mutable access)
        let mut pricer = self.gpu_pricer.lock();
        let model_prices = pricer
            .price_options(&params, &self.market_options)
            .map_err(|e| Error::msg(format!("GPU pricing failed: {}", e)))?;

        // Calculate sum of squared errors
        let sse = self.calculate_sse(&model_prices);

        Ok(sse)
    }
}

#[cfg(feature = "heston")]
impl Gradient for HestonObjective {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    /// Compute numerical gradient via finite differences
    ///
    /// Uses central differences for better accuracy:
    /// ∂f/∂xᵢ ≈ (f(x + εeᵢ) - f(x - εeᵢ)) / (2ε)
    ///
    /// # Performance
    ///
    /// - Requires 2×n function evaluations (n = number of parameters)
    /// - Each evaluation uses GPU pricing (fast for batch)
    /// - Total time: ~10-20ms for typical cases
    ///
    /// # Errors
    ///
    /// Returns error if cost function evaluation fails
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        let epsilon = 1e-5;
        let n = param.len();
        let mut grad = Array1::zeros(n);

        // Calculate gradient for each parameter
        for i in 0..n {
            // Forward step: x + ε·eᵢ
            let mut param_plus = param.clone();
            param_plus[i] += epsilon;
            let cost_plus = self.cost(&param_plus)?;

            // Backward step: x - ε·eᵢ
            let mut param_minus = param.clone();
            param_minus[i] -= epsilon;
            let cost_minus = self.cost(&param_minus)?;

            // Central difference
            grad[i] = (cost_plus - cost_minus) / (2.0 * epsilon);
        }

        Ok(grad)
    }
}

/// Generate uniform weights for all options
#[cfg(feature = "heston")]
pub fn uniform_weights(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

/// Generate inverse-variance weights (down-weight noisy options)
///
/// Uses bid-ask spread as proxy for noise:
/// weight = 1 / (bid_ask_spread)²
#[cfg(feature = "heston")]
pub fn inverse_variance_weights(options: &[OptionQuote]) -> Vec<f64> {
    options
        .iter()
        .map(|opt| {
            if let (Some(bid), Some(ask)) = (opt.bid, opt.ask) {
                let spread = ask - bid;
                if spread > 0.0 {
                    1.0 / (spread * spread)
                } else {
                    1.0
                }
            } else {
                1.0 // No spread info, use uniform weight
            }
        })
        .collect()
}

/// Generate moneyness-based weights (emphasize ATM options)
///
/// weight = exp(-k²) where k = log(K/S)
/// ATM options (K ≈ S) get weight ≈ 1.0
/// Far OTM/ITM options get lower weights
#[cfg(feature = "heston")]
pub fn moneyness_weights(options: &[OptionQuote]) -> Vec<f64> {
    options
        .iter()
        .map(|opt| {
            let moneyness = (opt.strike / opt.spot_price).ln();
            (-moneyness * moneyness).exp()
        })
        .collect()
}

#[cfg(all(test, feature = "heston"))]
mod tests {
    use super::*;
    use crate::gpu::GpuDevice;
    use crate::quantitative::heston::OptionType;

    fn create_test_options() -> Vec<OptionQuote> {
        vec![
            OptionQuote {
                underlying: "BTC".to_string(),
                strike: 48000.0,
                expiration: 1735689600,
                option_type: OptionType::Call,
                spot_price: 50000.0,
                risk_free_rate: 0.05,
                bid: Some(2400.0),
                ask: Some(2600.0),
                last: None,
                implied_vol: Some(0.8),
                volume: 100.0,
                open_interest: 500.0,
                greeks: None,
            },
            OptionQuote {
                underlying: "BTC".to_string(),
                strike: 50000.0,
                expiration: 1735689600,
                option_type: OptionType::Call,
                spot_price: 50000.0,
                risk_free_rate: 0.05,
                bid: Some(1950.0),
                ask: Some(2050.0),
                last: None,
                implied_vol: Some(0.75),
                volume: 200.0,
                open_interest: 1000.0,
                greeks: None,
            },
        ]
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_objective_valid_params() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let pricer = Arc::new(Mutex::new(HestonGpuPricer::new(device, 4096, 4096).unwrap()));
        let options = create_test_options();

        let objective = HestonObjective::new(pricer, options);

        // Test with valid parameters
        let params = Array1::from_vec(vec![2.0, 0.04, 0.3, -0.7, 0.04]);
        let cost = objective.cost(&params).unwrap();

        assert!(cost.is_finite(), "Cost should be finite for valid params");
        assert!(cost >= 0.0, "Cost should be non-negative");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_objective_invalid_params() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let pricer = Arc::new(Mutex::new(HestonGpuPricer::new(device, 4096, 4096).unwrap()));
        let options = create_test_options();

        let objective = HestonObjective::new(pricer, options);

        // Test with invalid parameters (violates Feller condition)
        let params = Array1::from_vec(vec![1.0, 0.01, 1.5, -0.7, 0.04]); // 2κθ = 0.02 < σ² = 2.25
        let cost = objective.cost(&params).unwrap();

        assert_eq!(cost, f64::INFINITY, "Invalid params should return infinity");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gradient_computation() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let pricer = Arc::new(Mutex::new(HestonGpuPricer::new(device, 4096, 4096).unwrap()));
        let options = create_test_options();

        let objective = HestonObjective::new(pricer, options);

        let params = Array1::from_vec(vec![2.0, 0.04, 0.3, -0.7, 0.04]);
        let grad = objective.gradient(&params).unwrap();

        assert_eq!(grad.len(), 5, "Gradient should have 5 elements");
        assert!(
            grad.iter().all(|&g| g.is_finite()),
            "All gradient components should be finite"
        );
    }

    #[test]
    fn test_uniform_weights() {
        let weights = uniform_weights(10);
        assert_eq!(weights.len(), 10);
        assert!(weights.iter().all(|&w| w == 1.0));
    }

    #[test]
    fn test_inverse_variance_weights() {
        let options = create_test_options();
        let weights = inverse_variance_weights(&options);

        assert_eq!(weights.len(), 2);
        assert!(weights.iter().all(|&w| w > 0.0));

        // Option with narrower spread should have higher weight
        assert!(weights[1] > weights[0]); // 200 spread vs 200 spread
    }

    #[test]
    fn test_moneyness_weights() {
        let options = create_test_options();
        let weights = moneyness_weights(&options);

        assert_eq!(weights.len(), 2);
        assert!(weights.iter().all(|&w| w > 0.0 && w <= 1.0));

        // ATM option should have higher weight
        assert!(weights[1] > weights[0]); // 50k strike at 50k spot vs 48k
    }

    #[test]
    fn test_vec_to_params() {
        let device = Arc::new(GpuDevice::new().ok().unwrap());
        let pricer = Arc::new(Mutex::new(HestonGpuPricer::new(device, 4096, 4096).unwrap()));
        let options = create_test_options();
        let objective = HestonObjective::new(pricer, options);

        let vec = Array1::from_vec(vec![2.0, 0.04, 0.3, -0.7, 0.05]);
        let params = objective.vec_to_params(&vec);

        assert_eq!(params.kappa, 2.0);
        assert_eq!(params.theta, 0.04);
        assert_eq!(params.sigma, 0.3);
        assert_eq!(params.rho, -0.7);
        assert_eq!(params.v0, 0.05);
    }
}
