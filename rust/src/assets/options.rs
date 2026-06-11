//! Options Contract Implementation
//!
//! Handles options-specific features:
//! - Greeks calculation (delta, gamma, theta, vega, rho)
//! - Strike price chains
//! - Expiration dates
//! - American vs European style
//! - Implied volatility
//! - Black-Scholes pricing

use super::specs::{Currency, SettlementType, us_equity_sessions};
use super::{Asset, AssetResult, AssetSpec, AssetType, Exchange};
use chrono::{DateTime, Datelike, Utc};
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, SQRT_2};
use std::fmt;

/// Options contract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionsContract {
    /// Asset specification
    spec: AssetSpec,

    /// Option type (Call or Put)
    option_type: OptionType,

    /// Strike price
    strike: f64,

    /// Expiration date
    expiration: DateTime<Utc>,

    /// Option style (American or European)
    style: OptionStyle,

    /// Underlying symbol
    underlying: String,

    /// Cached Greeks
    greeks: Option<Greeks>,
}

impl OptionsContract {
    /// Create new options contract
    pub fn new(
        underlying: &str,
        option_type: OptionType,
        strike: f64,
        expiration: DateTime<Utc>,
        exchange: Exchange,
    ) -> Self {
        let symbol = Self::format_occ_symbol(underlying, option_type, strike, expiration);

        let spec = AssetSpec::new(
            AssetType::Options,
            symbol,
            exchange,
            format!("{} {} {}", underlying, option_type, strike),
        )
        .with_tick_spec(0.01, 0.01) // Penny tick for options
        .with_multiplier(100.0) // Standard option contract
        .with_quantity_increment(1.0)
        .with_expiration(expiration)
        .with_underlying(underlying.to_string())
        .with_currency(Currency::USD)
        .with_settlement(SettlementType::Physical);

        // Add US equity trading sessions
        let spec = us_equity_sessions()
            .into_iter()
            .fold(spec, |s, session| s.with_session(session));

        Self {
            spec,
            option_type,
            strike,
            expiration,
            style: OptionStyle::American, // US equity options are American
            underlying: underlying.to_string(),
            greeks: None,
        }
    }

    /// Format OCC symbol (Options Clearing Corporation standard)
    /// Format: SYMBOL[6]YYMMDD[C/P][STRIKE*1000, 8 digits]
    /// Example: AAPL250117C00150000 = AAPL Jan 17, 2025 $150 Call
    pub fn format_occ_symbol(
        underlying: &str,
        option_type: OptionType,
        strike: f64,
        expiration: DateTime<Utc>,
    ) -> String {
        let symbol_padded = format!("{:6}", underlying);
        let year = expiration.year() % 100;
        let month = expiration.month();
        let day = expiration.day();
        let type_char = match option_type {
            OptionType::Call => 'C',
            OptionType::Put => 'P',
        };
        let strike_int = (strike * 1000.0).round() as i64;

        format!(
            "{}{:02}{:02}{:02}{}{:08}",
            symbol_padded, year, month, day, type_char, strike_int
        )
    }

    /// Builder: Set option style
    pub fn with_style(mut self, style: OptionStyle) -> Self {
        self.style = style;
        self
    }

    /// Calculate Black-Scholes price
    pub fn black_scholes_price(
        &self,
        spot_price: f64,
        volatility: f64,
        risk_free_rate: f64,
        time_to_expiry: f64,
    ) -> f64 {
        if time_to_expiry <= 0.0 {
            return self.intrinsic_value(spot_price);
        }

        let d1 = ((spot_price / self.strike).ln()
            + (risk_free_rate + 0.5 * volatility.powi(2)) * time_to_expiry)
            / (volatility * time_to_expiry.sqrt());

        let d2 = d1 - volatility * time_to_expiry.sqrt();

        match self.option_type {
            OptionType::Call => {
                spot_price * Self::norm_cdf(d1)
                    - self.strike * (-risk_free_rate * time_to_expiry).exp() * Self::norm_cdf(d2)
            }
            OptionType::Put => {
                self.strike * (-risk_free_rate * time_to_expiry).exp() * Self::norm_cdf(-d2)
                    - spot_price * Self::norm_cdf(-d1)
            }
        }
    }

    /// Calculate Greeks
    pub fn calculate_greeks(
        &mut self,
        spot_price: f64,
        volatility: f64,
        risk_free_rate: f64,
        time_to_expiry: f64,
    ) -> Greeks {
        let greeks = Greeks::calculate(
            self.option_type,
            spot_price,
            self.strike,
            time_to_expiry,
            volatility,
            risk_free_rate,
        );

        self.greeks = Some(greeks.clone());
        greeks
    }

    /// Get cached Greeks (if available)
    pub fn greeks(&self) -> Option<&Greeks> {
        self.greeks.as_ref()
    }

    /// Get intrinsic value
    pub fn intrinsic_value(&self, spot_price: f64) -> f64 {
        match self.option_type {
            OptionType::Call => (spot_price - self.strike).max(0.0),
            OptionType::Put => (self.strike - spot_price).max(0.0),
        }
    }

    /// Get time value (extrinsic value)
    pub fn time_value(&self, spot_price: f64, option_price: f64) -> f64 {
        option_price - self.intrinsic_value(spot_price)
    }

    /// Check if option is in-the-money
    pub fn is_itm(&self, spot_price: f64) -> bool {
        match self.option_type {
            OptionType::Call => spot_price > self.strike,
            OptionType::Put => spot_price < self.strike,
        }
    }

    /// Check if option is at-the-money
    pub fn is_atm(&self, spot_price: f64, tolerance: f64) -> bool {
        (spot_price - self.strike).abs() / self.strike < tolerance
    }

    /// Check if option is out-of-the-money
    pub fn is_otm(&self, spot_price: f64) -> bool {
        !self.is_itm(spot_price)
    }

    /// Get moneyness (spot / strike for calls, strike / spot for puts)
    pub fn moneyness(&self, spot_price: f64) -> f64 {
        match self.option_type {
            OptionType::Call => spot_price / self.strike,
            OptionType::Put => self.strike / spot_price,
        }
    }

    /// Calculate implied volatility using Newton-Raphson method
    pub fn implied_volatility(
        &self,
        spot_price: f64,
        option_price: f64,
        risk_free_rate: f64,
        time_to_expiry: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Option<f64> {
        if time_to_expiry <= 0.0 || option_price <= 0.0 {
            return None;
        }

        let mut sigma = 0.3; // Initial guess: 30% volatility

        for _ in 0..max_iterations {
            let price = self.black_scholes_price(spot_price, sigma, risk_free_rate, time_to_expiry);
            let diff = price - option_price;

            if diff.abs() < tolerance {
                return Some(sigma);
            }

            // Vega (derivative of price w.r.t. volatility)
            let vega = Greeks::calculate_vega(
                spot_price,
                self.strike,
                time_to_expiry,
                sigma,
                risk_free_rate,
            );

            if vega < 1e-10 {
                break; // Vega too small, can't converge
            }

            sigma -= diff / vega;

            if sigma <= 0.0 {
                sigma = 0.01; // Minimum volatility
            }
        }

        None // Failed to converge
    }

    /// Standard normal cumulative distribution function
    fn norm_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / SQRT_2))
    }

    /// Error function approximation (Abramowitz and Stegun)
    fn erf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Get option type
    pub fn option_type(&self) -> OptionType {
        self.option_type
    }

    /// Get strike price
    pub fn strike(&self) -> f64 {
        self.strike
    }

    /// Get expiration
    pub fn expiration(&self) -> DateTime<Utc> {
        self.expiration
    }

    /// Get option style
    pub fn style(&self) -> OptionStyle {
        self.style
    }

    /// Get underlying symbol
    pub fn underlying(&self) -> &str {
        &self.underlying
    }
}

impl Asset for OptionsContract {
    fn asset_type(&self) -> AssetType {
        AssetType::Options
    }

    fn symbol(&self) -> &str {
        &self.spec.symbol
    }

    fn validate_price(&self, price: f64) -> AssetResult<f64> {
        self.spec.validate_price(price)
    }

    fn normalize_symbol(&self, symbol: &str) -> AssetResult<String> {
        // OCC format validation would go here
        Ok(symbol.to_uppercase())
    }

    fn calculate_value(&self, price: f64, quantity: f64) -> AssetResult<f64> {
        self.spec.calculate_value(price, quantity)
    }

    fn is_market_open(&self, timestamp: DateTime<Utc>) -> bool {
        !self.spec.is_expired(timestamp) && self.spec.is_market_open(timestamp)
    }

    fn tick_size(&self) -> f64 {
        self.spec.tick_size
    }

    fn quantity_increment(&self) -> f64 {
        1.0 // Contracts are whole numbers
    }

    fn contract_multiplier(&self) -> f64 {
        100.0 // Standard option contract
    }

    fn specification(&self) -> &AssetSpec {
        &self.spec
    }
}

/// Option type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

impl fmt::Display for OptionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptionType::Call => write!(f, "CALL"),
            OptionType::Put => write!(f, "PUT"),
        }
    }
}

/// Option style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptionStyle {
    /// American: can be exercised any time before expiration
    American,
    /// European: can only be exercised at expiration
    European,
}

/// Greeks (option sensitivities)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Greeks {
    /// Delta: rate of change of option price w.r.t. underlying price
    pub delta: f64,

    /// Gamma: rate of change of delta w.r.t. underlying price
    pub gamma: f64,

    /// Theta: rate of change of option price w.r.t. time (per day)
    pub theta: f64,

    /// Vega: rate of change of option price w.r.t. volatility (per 1% change)
    pub vega: f64,

    /// Rho: rate of change of option price w.r.t. risk-free rate (per 1% change)
    pub rho: f64,
}

impl Greeks {
    /// Calculate all Greeks using Black-Scholes model
    pub fn calculate(
        option_type: OptionType,
        spot_price: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
        risk_free_rate: f64,
    ) -> Self {
        if time_to_expiry <= 0.0 {
            return Self::zero();
        }

        let d1 = ((spot_price / strike).ln()
            + (risk_free_rate + 0.5 * volatility.powi(2)) * time_to_expiry)
            / (volatility * time_to_expiry.sqrt());

        let d2 = d1 - volatility * time_to_expiry.sqrt();

        let nd1 = OptionsContract::norm_cdf(d1);
        let nd2 = OptionsContract::norm_cdf(d2);
        let phi_d1 = Self::norm_pdf(d1);

        // Delta
        let delta = match option_type {
            OptionType::Call => nd1,
            OptionType::Put => nd1 - 1.0,
        };

        // Gamma (same for calls and puts)
        let gamma = phi_d1 / (spot_price * volatility * time_to_expiry.sqrt());

        // Theta (per day, divide by 365)
        let theta = match option_type {
            OptionType::Call => {
                (-spot_price * phi_d1 * volatility / (2.0 * time_to_expiry.sqrt())
                    - risk_free_rate * strike * (-risk_free_rate * time_to_expiry).exp() * nd2)
                    / 365.0
            }
            OptionType::Put => {
                (-spot_price * phi_d1 * volatility / (2.0 * time_to_expiry.sqrt())
                    + risk_free_rate
                        * strike
                        * (-risk_free_rate * time_to_expiry).exp()
                        * OptionsContract::norm_cdf(-d2))
                    / 365.0
            }
        };

        // Vega (same for calls and puts, per 1% change in volatility)
        let vega = spot_price * phi_d1 * time_to_expiry.sqrt() / 100.0;

        // Rho (per 1% change in risk-free rate)
        let rho = match option_type {
            OptionType::Call => {
                strike * time_to_expiry * (-risk_free_rate * time_to_expiry).exp() * nd2 / 100.0
            }
            OptionType::Put => {
                -strike
                    * time_to_expiry
                    * (-risk_free_rate * time_to_expiry).exp()
                    * OptionsContract::norm_cdf(-d2)
                    / 100.0
            }
        };

        Self {
            delta,
            gamma,
            theta,
            vega,
            rho,
        }
    }

    /// Calculate vega only (for implied volatility calculation)
    pub fn calculate_vega(
        spot_price: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
        risk_free_rate: f64,
    ) -> f64 {
        if time_to_expiry <= 0.0 {
            return 0.0;
        }

        let d1 = ((spot_price / strike).ln()
            + (risk_free_rate + 0.5 * volatility.powi(2)) * time_to_expiry)
            / (volatility * time_to_expiry.sqrt());

        spot_price * Self::norm_pdf(d1) * time_to_expiry.sqrt()
    }

    /// Standard normal probability density function
    fn norm_pdf(x: f64) -> f64 {
        (1.0 / (2.0 * PI).sqrt()) * (-0.5 * x.powi(2)).exp()
    }

    /// Zero Greeks (for expired options)
    fn zero() -> Self {
        Self {
            delta: 0.0,
            gamma: 0.0,
            theta: 0.0,
            vega: 0.0,
            rho: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_occ_symbol_formatting() {
        let expiration = DateTime::from_timestamp(1737072000, 0).unwrap(); // Jan 17, 2025
        let symbol =
            OptionsContract::format_occ_symbol("AAPL", OptionType::Call, 150.0, expiration);
        assert_eq!(symbol, "AAPL  250117C00150000");
    }

    #[test]
    fn test_options_creation() {
        let expiration = DateTime::from_timestamp(1737072000, 0).unwrap();
        let opt = OptionsContract::new("AAPL", OptionType::Call, 150.0, expiration, Exchange::CBOE);

        assert_eq!(opt.option_type(), OptionType::Call);
        assert_eq!(opt.strike(), 150.0);
        assert_eq!(opt.underlying(), "AAPL");
    }

    #[test]
    fn test_intrinsic_value() {
        let expiration = DateTime::from_timestamp(1737072000, 0).unwrap();
        let call =
            OptionsContract::new("AAPL", OptionType::Call, 150.0, expiration, Exchange::CBOE);
        let put = OptionsContract::new("AAPL", OptionType::Put, 150.0, expiration, Exchange::CBOE);

        // Call intrinsic value
        assert_eq!(call.intrinsic_value(160.0), 10.0);
        assert_eq!(call.intrinsic_value(150.0), 0.0);
        assert_eq!(call.intrinsic_value(140.0), 0.0);

        // Put intrinsic value
        assert_eq!(put.intrinsic_value(140.0), 10.0);
        assert_eq!(put.intrinsic_value(150.0), 0.0);
        assert_eq!(put.intrinsic_value(160.0), 0.0);
    }

    #[test]
    fn test_moneyness() {
        let expiration = DateTime::from_timestamp(1737072000, 0).unwrap();
        let call =
            OptionsContract::new("AAPL", OptionType::Call, 150.0, expiration, Exchange::CBOE);

        assert!(call.is_itm(160.0));
        assert!(call.is_atm(150.0, 0.01));
        assert!(call.is_otm(140.0));
    }

    #[test]
    fn test_black_scholes_call() {
        let expiration = DateTime::from_timestamp(1737072000, 0).unwrap();
        let call =
            OptionsContract::new("AAPL", OptionType::Call, 150.0, expiration, Exchange::CBOE);

        let price = call.black_scholes_price(
            150.0, // spot
            0.25,  // 25% volatility
            0.05,  // 5% risk-free rate
            0.5,   // 6 months
        );

        // Call should have positive value
        assert!(price > 0.0);
        // ATM call should be worth approximately 10-15 for these parameters
        assert!(price > 5.0 && price < 20.0);
    }

    #[test]
    fn test_greeks_calculation() {
        let expiration = DateTime::from_timestamp(1737072000, 0).unwrap();
        let mut call =
            OptionsContract::new("AAPL", OptionType::Call, 150.0, expiration, Exchange::CBOE);

        let greeks = call.calculate_greeks(
            150.0, // spot
            0.25,  // volatility
            0.05,  // risk-free rate
            0.5,   // time to expiry
        );

        // ATM call delta should be around 0.5
        assert!(greeks.delta > 0.4 && greeks.delta < 0.6);

        // Gamma should be positive
        assert!(greeks.gamma > 0.0);

        // Theta should be negative (time decay)
        assert!(greeks.theta < 0.0);

        // Vega should be positive
        assert!(greeks.vega > 0.0);
    }

    #[test]
    fn test_norm_cdf() {
        // Standard normal CDF at 0 should be 0.5
        assert!((OptionsContract::norm_cdf(0.0) - 0.5).abs() < 1e-6);

        // CDF(-∞) ≈ 0, CDF(∞) ≈ 1
        assert!(OptionsContract::norm_cdf(-5.0) < 0.001);
        assert!(OptionsContract::norm_cdf(5.0) > 0.999);
    }

    #[test]
    fn test_contract_value() {
        let expiration = DateTime::from_timestamp(1737072000, 0).unwrap();
        let call =
            OptionsContract::new("AAPL", OptionType::Call, 150.0, expiration, Exchange::CBOE);

        // 1 contract at $5.00 = 1 * 5.00 * 100 = $500
        let value = call.calculate_value(5.00, 1.0).unwrap();
        assert_eq!(value, 500.0);
    }
}
