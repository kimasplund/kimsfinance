//! Trading Strategies Using Heston Model
//!
//! Implements example trading strategies leveraging the calibrated Heston model:
//!
//! 1. **Vol Arbitrage**: Trade when model IV differs from market IV
//! 2. **Delta Hedging**: Maintain delta-neutral portfolio
//!
//! # Usage
//!
//! ```no_run
//! use kimsfinance_core::quantitative::heston::{HestonParams, OptionQuote};
//! use kimsfinance_core::quantitative::heston::strategies::{VolArbitrageStrategy, TradeSignal};
//!
//! let strategy = VolArbitrageStrategy::new(5.0); // 5% threshold
//! let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).unwrap();
//! let signals = strategy.generate_signals(&options, &params);
//!
//! for signal in signals {
//!     match signal {
//!         TradeSignal::Buy { option, edge, .. } => {
//!             println!("BUY {} @ ${} - Edge: {:.2}%", option.underlying, option.strike, edge);
//!         }
//!         TradeSignal::Sell { option, edge, .. } => {
//!             println!("SELL {} @ ${} - Edge: {:.2}%", option.underlying, option.strike, edge);
//!         }
//!     }
//! }
//! ```

use crate::quantitative::heston::{Greeks, HestonParams, OptionQuote};
use serde::{Deserialize, Serialize};

/// Vol arbitrage strategy: trade when model IV differs from market IV
///
/// # Strategy Logic
///
/// 1. Calculate model IV from calibrated Heston parameters
/// 2. Compare with market implied volatility
/// 3. Generate BUY signal if market IV < model IV (underpriced)
/// 4. Generate SELL signal if market IV > model IV (overpriced)
///
/// # Parameters
///
/// - `threshold`: Minimum % difference to trigger signal (e.g., 5% = 5 percentage points)
///
/// # Example
///
/// ```no_run
/// let strategy = VolArbitrageStrategy::new(5.0); // 5% threshold
/// let signals = strategy.generate_signals(&options, &params);
/// ```
pub struct VolArbitrageStrategy {
    /// Minimum % IV difference to trigger signal (percentage points)
    threshold: f64,
}

impl VolArbitrageStrategy {
    /// Create new vol arbitrage strategy
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum % IV difference (e.g., 5.0 = 5 percentage points)
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Generate trade signals from options chain
    ///
    /// # Arguments
    ///
    /// * `options` - Options to analyze
    /// * `model_params` - Calibrated Heston parameters
    ///
    /// # Returns
    ///
    /// Vector of trade signals (Buy/Sell) sorted by edge (best opportunities first)
    pub fn generate_signals(
        &self,
        options: &[OptionQuote],
        model_params: &HestonParams,
    ) -> Vec<TradeSignal> {
        let mut signals = Vec::new();

        for option in options {
            // Skip options without market IV
            if let Some(market_iv) = option.implied_vol {
                // Calculate model IV from Heston parameters
                let model_iv = self.calculate_model_iv(option, model_params);

                // Calculate percentage point difference
                let diff_pct = (market_iv - model_iv) * 100.0; // Convert to percentage points

                if diff_pct.abs() > self.threshold {
                    let signal = if diff_pct > 0.0 {
                        // Market overpricing - sell option
                        TradeSignal::Sell {
                            option: option.clone(),
                            reason: format!(
                                "Market IV {:.1}% > Model IV {:.1}% (diff: +{:.1}pp)",
                                market_iv * 100.0,
                                model_iv * 100.0,
                                diff_pct
                            ),
                            edge: diff_pct,
                        }
                    } else {
                        // Market underpricing - buy option
                        TradeSignal::Buy {
                            option: option.clone(),
                            reason: format!(
                                "Market IV {:.1}% < Model IV {:.1}% (diff: {:.1}pp)",
                                market_iv * 100.0,
                                model_iv * 100.0,
                                diff_pct
                            ),
                            edge: -diff_pct,
                        }
                    };
                    signals.push(signal);
                }
            }
        }

        // Sort by edge (best opportunities first)
        signals.sort_by(|a, b| {
            let edge_a = match a {
                TradeSignal::Buy { edge, .. } | TradeSignal::Sell { edge, .. } => *edge,
            };
            let edge_b = match b {
                TradeSignal::Buy { edge, .. } | TradeSignal::Sell { edge, .. } => *edge,
            };
            edge_b
                .partial_cmp(&edge_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        signals
    }

    /// Calculate model IV from Heston parameters
    ///
    /// For simplicity, uses √v₀ as the model IV.
    /// More sophisticated: could use implied vol from Heston pricing.
    fn calculate_model_iv(&self, _option: &OptionQuote, params: &HestonParams) -> f64 {
        // Simple approach: current volatility from Heston model
        // TODO: More accurate approach would be to:
        // 1. Price option with Heston model
        // 2. Invert Black-Scholes to get implied vol
        params.current_vol()
    }
}

/// Delta hedging strategy: maintain delta-neutral portfolio
///
/// # Strategy Logic
///
/// 1. Calculate portfolio delta (sum of individual position deltas)
/// 2. Recommend underlying shares to hedge to target delta (typically 0)
///
/// # Example
///
/// ```no_run
/// let strategy = DeltaHedgingStrategy::new(0.0); // Delta-neutral
/// let hedge = strategy.calculate_hedge(&portfolio, &greeks);
/// println!("Buy {} shares to hedge", hedge.underlying_shares);
/// ```
pub struct DeltaHedgingStrategy {
    /// Target portfolio delta (typically 0.0 for delta-neutral)
    target_delta: f64,
}

impl DeltaHedgingStrategy {
    /// Create new delta hedging strategy
    ///
    /// # Arguments
    ///
    /// * `target_delta` - Target portfolio delta (0.0 = delta-neutral)
    pub fn new(target_delta: f64) -> Self {
        Self { target_delta }
    }

    /// Calculate hedge recommendation
    ///
    /// # Arguments
    ///
    /// * `portfolio` - Current option positions
    /// * `greeks` - Greeks for each position (must match portfolio order)
    ///
    /// # Returns
    ///
    /// Hedge recommendation with number of underlying shares to buy/sell
    ///
    /// # Panics
    ///
    /// Panics if portfolio and greeks lengths don't match
    pub fn calculate_hedge(
        &self,
        portfolio: &[OptionPosition],
        greeks: &[Greeks],
    ) -> HedgeRecommendation {
        assert_eq!(
            portfolio.len(),
            greeks.len(),
            "Portfolio and greeks must have same length"
        );

        // Calculate portfolio delta
        let portfolio_delta: f64 = portfolio
            .iter()
            .zip(greeks.iter())
            .map(|(pos, greek)| {
                let delta = greek.delta.unwrap_or(0.0);
                pos.quantity as f64 * delta
            })
            .sum();

        // Calculate required hedge to reach target delta
        let hedge_delta = self.target_delta - portfolio_delta;

        HedgeRecommendation {
            underlying_shares: hedge_delta.round() as i32,
            current_delta: portfolio_delta,
            target_delta: self.target_delta,
            reason: format!(
                "Portfolio delta: {:.2}, Target: {:.2}, Hedge: {} shares",
                portfolio_delta,
                self.target_delta,
                hedge_delta.round() as i32
            ),
        }
    }

    /// Calculate portfolio Greeks (aggregated)
    ///
    /// Useful for monitoring overall portfolio risk
    pub fn calculate_portfolio_greeks(
        portfolio: &[OptionPosition],
        greeks: &[Greeks],
    ) -> PortfolioGreeks {
        assert_eq!(
            portfolio.len(),
            greeks.len(),
            "Portfolio and greeks must have same length"
        );

        let mut total_delta = 0.0;
        let mut total_gamma = 0.0;
        let mut total_vega = 0.0;
        let mut total_theta = 0.0;
        let mut total_rho = 0.0;

        for (pos, greek) in portfolio.iter().zip(greeks.iter()) {
            let qty = pos.quantity as f64;
            total_delta += qty * greek.delta.unwrap_or(0.0);
            total_gamma += qty * greek.gamma.unwrap_or(0.0);
            total_vega += qty * greek.vega.unwrap_or(0.0);
            total_theta += qty * greek.theta.unwrap_or(0.0);
            total_rho += qty * greek.rho_greek.unwrap_or(0.0);
        }

        PortfolioGreeks {
            delta: total_delta,
            gamma: total_gamma,
            vega: total_vega,
            theta: total_theta,
            rho: total_rho,
        }
    }
}

/// Trade signal (Buy or Sell)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSignal {
    /// Buy signal (market underpriced)
    Buy {
        option: OptionQuote,
        reason: String,
        edge: f64, // Expected profit % (percentage points)
    },
    /// Sell signal (market overpriced)
    Sell {
        option: OptionQuote,
        reason: String,
        edge: f64, // Expected profit % (percentage points)
    },
}

/// Option position in portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionPosition {
    pub option: OptionQuote,
    /// Quantity: positive = long, negative = short
    pub quantity: i32,
}

/// Hedge recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgeRecommendation {
    /// Number of underlying shares to buy (positive) or sell (negative)
    pub underlying_shares: i32,
    /// Current portfolio delta
    pub current_delta: f64,
    /// Target portfolio delta
    pub target_delta: f64,
    /// Explanation
    pub reason: String,
}

/// Aggregated portfolio Greeks
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PortfolioGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantitative::heston::{OptionType, ValidationError};
    use chrono::Utc;

    fn create_test_option_with_iv(strike: f64, iv: f64) -> OptionQuote {
        let now = Utc::now().timestamp();
        let expiry_3months = now + (90 * 24 * 3600);

        OptionQuote {
            underlying: "BTC".to_string(),
            strike,
            expiration: expiry_3months,
            option_type: OptionType::Call,
            spot_price: 48000.0,
            risk_free_rate: 0.05,
            bid: Some(2000.0),
            ask: Some(2100.0),
            last: Some(2050.0),
            implied_vol: Some(iv),
            volume: 100.0,
            open_interest: 500.0,
            greeks: None,
        }
    }

    #[test]
    fn test_vol_arbitrage_buy_signal() {
        let strategy = VolArbitrageStrategy::new(5.0);
        let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).unwrap();

        // Market IV = 15%, Model IV = 20% (from √v₀ = √0.04 = 0.2)
        // Difference = -5pp → BUY signal
        let option = create_test_option_with_iv(50000.0, 0.15);
        let signals = strategy.generate_signals(&[option.clone()], &params);

        assert_eq!(signals.len(), 1);
        match &signals[0] {
            TradeSignal::Buy { edge, .. } => {
                assert!(
                    *edge > 4.0 && *edge < 6.0,
                    "Expected edge ~5%, got {}%",
                    edge
                );
            }
            TradeSignal::Sell { .. } => panic!("Expected Buy signal, got Sell"),
        }
    }

    #[test]
    fn test_vol_arbitrage_sell_signal() {
        let strategy = VolArbitrageStrategy::new(5.0);
        let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).unwrap();

        // Market IV = 26%, Model IV = 20% (from √v₀ = √0.04 = 0.2)
        // Difference = +6pp → SELL signal (above 5pp threshold)
        let option = create_test_option_with_iv(50000.0, 0.26);
        let signals = strategy.generate_signals(&[option.clone()], &params);

        assert_eq!(signals.len(), 1);
        match &signals[0] {
            TradeSignal::Sell { edge, .. } => {
                assert!(
                    *edge > 5.0 && *edge < 7.0,
                    "Expected edge ~6%, got {}%",
                    edge
                );
            }
            TradeSignal::Buy { .. } => panic!("Expected Sell signal, got Buy"),
        }
    }

    #[test]
    fn test_vol_arbitrage_no_signal() {
        let strategy = VolArbitrageStrategy::new(5.0);
        let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).unwrap();

        // Market IV = 21%, Model IV = 20%
        // Difference = 1pp < threshold → NO signal
        let option = create_test_option_with_iv(50000.0, 0.21);
        let signals = strategy.generate_signals(&[option.clone()], &params);

        assert_eq!(signals.len(), 0);
    }

    #[test]
    fn test_vol_arbitrage_sorting() {
        let strategy = VolArbitrageStrategy::new(3.0);
        let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).unwrap();

        let options = vec![
            create_test_option_with_iv(48000.0, 0.15), // -5pp edge
            create_test_option_with_iv(49000.0, 0.28), // +8pp edge
            create_test_option_with_iv(50000.0, 0.24), // +4pp edge
        ];

        let signals = strategy.generate_signals(&options, &params);

        // Should be sorted by edge: 8pp, 5pp, 4pp
        assert_eq!(signals.len(), 3);
        assert!(matches!(signals[0], TradeSignal::Sell { edge, .. } if edge > 7.0));
        assert!(matches!(signals[1], TradeSignal::Buy { edge, .. } if edge > 4.0));
        assert!(matches!(signals[2], TradeSignal::Sell { edge, .. } if edge > 3.0));
    }

    #[test]
    fn test_delta_hedging_neutral() {
        let strategy = DeltaHedgingStrategy::new(0.0);

        let option = create_test_option_with_iv(50000.0, 0.2);
        let portfolio = vec![OptionPosition {
            option: option.clone(),
            quantity: 10, // Long 10 calls
        }];

        let greeks = vec![Greeks {
            delta: Some(0.5),
            gamma: Some(0.01),
            vega: Some(100.0),
            theta: Some(-50.0),
            rho_greek: Some(25.0),
        }];

        let hedge = strategy.calculate_hedge(&portfolio, &greeks);

        // Portfolio delta = 10 * 0.5 = 5.0
        // To reach 0, need to sell 5 shares
        assert_eq!(hedge.underlying_shares, -5);
        assert!((hedge.current_delta - 5.0).abs() < 1e-6);
        assert_eq!(hedge.target_delta, 0.0);
    }

    #[test]
    fn test_delta_hedging_multiple_positions() {
        let strategy = DeltaHedgingStrategy::new(0.0);

        let option = create_test_option_with_iv(50000.0, 0.2);
        let portfolio = vec![
            OptionPosition {
                option: option.clone(),
                quantity: 10, // Long 10 calls (delta +0.5 each)
            },
            OptionPosition {
                option: option.clone(),
                quantity: -5, // Short 5 calls (delta +0.5 each)
            },
        ];

        let greeks = vec![
            Greeks {
                delta: Some(0.5),
                gamma: Some(0.01),
                vega: Some(100.0),
                theta: Some(-50.0),
                rho_greek: Some(25.0),
            },
            Greeks {
                delta: Some(0.5),
                gamma: Some(0.01),
                vega: Some(100.0),
                theta: Some(-50.0),
                rho_greek: Some(25.0),
            },
        ];

        let hedge = strategy.calculate_hedge(&portfolio, &greeks);

        // Portfolio delta = (10 * 0.5) + (-5 * 0.5) = 5.0 - 2.5 = 2.5
        // To reach 0, need to sell 2.5 ≈ 3 shares
        assert!(
            hedge.underlying_shares >= -3 && hedge.underlying_shares <= -2,
            "Expected -3 or -2, got {}",
            hedge.underlying_shares
        );
    }

    #[test]
    fn test_portfolio_greeks() {
        let option = create_test_option_with_iv(50000.0, 0.2);
        let portfolio = vec![
            OptionPosition {
                option: option.clone(),
                quantity: 10,
            },
            OptionPosition {
                option: option.clone(),
                quantity: -5,
            },
        ];

        let greeks = vec![
            Greeks {
                delta: Some(0.5),
                gamma: Some(0.01),
                vega: Some(100.0),
                theta: Some(-50.0),
                rho_greek: Some(25.0),
            },
            Greeks {
                delta: Some(0.6),
                gamma: Some(0.02),
                vega: Some(120.0),
                theta: Some(-60.0),
                rho_greek: Some(30.0),
            },
        ];

        let port_greeks = DeltaHedgingStrategy::calculate_portfolio_greeks(&portfolio, &greeks);

        // Delta: (10 * 0.5) + (-5 * 0.6) = 5.0 - 3.0 = 2.0
        assert!((port_greeks.delta - 2.0).abs() < 1e-6);

        // Gamma: (10 * 0.01) + (-5 * 0.02) = 0.1 - 0.1 = 0.0
        assert!(port_greeks.gamma.abs() < 1e-6);

        // Vega: (10 * 100) + (-5 * 120) = 1000 - 600 = 400
        assert!((port_greeks.vega - 400.0).abs() < 1e-6);

        // Theta: (10 * -50) + (-5 * -60) = -500 + 300 = -200
        assert!((port_greeks.theta - (-200.0)).abs() < 1e-6);

        // Rho: (10 * 25) + (-5 * 30) = 250 - 150 = 100
        assert!((port_greeks.rho - 100.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Portfolio and greeks must have same length")]
    fn test_hedge_length_mismatch() {
        let strategy = DeltaHedgingStrategy::new(0.0);
        let option = create_test_option_with_iv(50000.0, 0.2);
        let portfolio = vec![OptionPosition {
            option: option.clone(),
            quantity: 10,
        }];
        let greeks = vec![];

        strategy.calculate_hedge(&portfolio, &greeks);
    }
}
