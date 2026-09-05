//! Asset Specifications and Contract Details
//!
//! Defines comprehensive specifications for different asset types including:
//! - Tick sizes and price increments
//! - Contract multipliers
//! - Trading hours
//! - Margin requirements
//! - Settlement types

use super::{AssetError, AssetResult, AssetType, Exchange, TradingSession};
use chrono::{DateTime, NaiveTime, Utc, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete asset specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetSpec {
    /// Asset type
    pub asset_type: AssetType,

    /// Primary symbol
    pub symbol: String,

    /// Exchange/venue
    pub exchange: Exchange,

    /// Display name
    pub name: String,

    /// Tick size (minimum price increment)
    pub tick_size: f64,

    /// Tick value (dollar value of one tick)
    pub tick_value: f64,

    /// Contract multiplier (futures/options)
    pub contract_multiplier: f64,

    /// Minimum quantity increment
    pub quantity_increment: f64,

    /// Trading sessions
    pub sessions: Vec<TradingSession>,

    /// Currency
    pub currency: Currency,

    /// Settlement type
    pub settlement_type: SettlementType,

    /// Margin requirements (as decimal, e.g., 0.5 = 50%)
    pub initial_margin_rate: Option<f64>,
    pub maintenance_margin_rate: Option<f64>,

    /// Expiration date (for derivatives)
    pub expiration: Option<DateTime<Utc>>,

    /// Underlying symbol (for derivatives)
    pub underlying: Option<String>,

    /// Asset-specific metadata
    pub metadata: HashMap<String, String>,
}

impl AssetSpec {
    /// Create new asset specification
    pub fn new(asset_type: AssetType, symbol: String, exchange: Exchange, name: String) -> Self {
        Self {
            asset_type,
            symbol,
            exchange,
            name,
            tick_size: 0.01,
            tick_value: 0.01,
            contract_multiplier: 1.0,
            quantity_increment: 1.0,
            sessions: Vec::new(),
            currency: Currency::USD,
            settlement_type: SettlementType::Cash,
            initial_margin_rate: None,
            maintenance_margin_rate: None,
            expiration: None,
            underlying: None,
            metadata: HashMap::new(),
        }
    }

    /// Builder: Set tick specifications
    pub fn with_tick_spec(mut self, tick_size: f64, tick_value: f64) -> Self {
        self.tick_size = tick_size;
        self.tick_value = tick_value;
        self
    }

    /// Builder: Set contract multiplier
    pub fn with_multiplier(mut self, multiplier: f64) -> Self {
        self.contract_multiplier = multiplier;
        self
    }

    /// Builder: Set quantity increment
    pub fn with_quantity_increment(mut self, increment: f64) -> Self {
        self.quantity_increment = increment;
        self
    }

    /// Builder: Add trading session
    pub fn with_session(mut self, session: TradingSession) -> Self {
        self.sessions.push(session);
        self
    }

    /// Builder: Set currency
    pub fn with_currency(mut self, currency: Currency) -> Self {
        self.currency = currency;
        self
    }

    /// Builder: Set settlement type
    pub fn with_settlement(mut self, settlement_type: SettlementType) -> Self {
        self.settlement_type = settlement_type;
        self
    }

    /// Builder: Set margin requirements
    pub fn with_margin(mut self, initial: f64, maintenance: f64) -> Self {
        self.initial_margin_rate = Some(initial);
        self.maintenance_margin_rate = Some(maintenance);
        self
    }

    /// Builder: Set expiration
    pub fn with_expiration(mut self, expiration: DateTime<Utc>) -> Self {
        self.expiration = Some(expiration);
        self
    }

    /// Builder: Set underlying
    pub fn with_underlying(mut self, underlying: String) -> Self {
        self.underlying = Some(underlying);
        self
    }

    /// Builder: Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Validate price according to tick size rules
    pub fn validate_price(&self, price: f64) -> AssetResult<f64> {
        if price <= 0.0 {
            return Err(AssetError::InvalidPrice(
                "Price must be positive".to_string(),
            ));
        }

        // Check if price is multiple of tick size
        let remainder = (price / self.tick_size).fract().abs();
        if remainder > 1e-10 {
            return Err(AssetError::InvalidTickSize(format!(
                "Price {} is not a multiple of tick size {}",
                price, self.tick_size
            )));
        }

        Ok(price)
    }

    /// Round price to nearest valid tick
    pub fn round_to_tick(&self, price: f64) -> f64 {
        (price / self.tick_size).round() * self.tick_size
    }

    /// Check if market is currently open
    pub fn is_market_open(&self, timestamp: DateTime<Utc>) -> bool {
        self.sessions
            .iter()
            .any(|session| session.is_active(timestamp))
    }

    /// Calculate contract value
    pub fn calculate_value(&self, price: f64, quantity: f64) -> AssetResult<f64> {
        let validated_price = self.validate_price(price)?;
        Ok(validated_price * quantity * self.contract_multiplier)
    }

    /// Check if contract is expired
    pub fn is_expired(&self, timestamp: DateTime<Utc>) -> bool {
        self.expiration.is_some_and(|exp| timestamp >= exp)
    }
}

/// Currency enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Currency {
    USD,
    EUR,
    GBP,
    JPY,
    CHF,
    CAD,
    AUD,
    NZD,
    CNY,
    HKD,
    BTC,
    ETH,
    USDT,
}

impl Currency {
    /// Get currency symbol
    pub fn symbol(&self) -> &str {
        match self {
            Currency::USD => "$",
            Currency::EUR => "€",
            Currency::GBP => "£",
            Currency::JPY => "¥",
            Currency::CHF => "Fr",
            Currency::CAD => "C$",
            Currency::AUD => "A$",
            Currency::NZD => "NZ$",
            Currency::CNY => "¥",
            Currency::HKD => "HK$",
            Currency::BTC => "₿",
            Currency::ETH => "Ξ",
            Currency::USDT => "₮",
        }
    }

    /// Get currency code
    pub fn code(&self) -> &str {
        match self {
            Currency::USD => "USD",
            Currency::EUR => "EUR",
            Currency::GBP => "GBP",
            Currency::JPY => "JPY",
            Currency::CHF => "CHF",
            Currency::CAD => "CAD",
            Currency::AUD => "AUD",
            Currency::NZD => "NZD",
            Currency::CNY => "CNY",
            Currency::HKD => "HKD",
            Currency::BTC => "BTC",
            Currency::ETH => "ETH",
            Currency::USDT => "USDT",
        }
    }
}

/// Settlement type for derivatives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SettlementType {
    /// Cash settlement (no physical delivery)
    Cash,
    /// Physical delivery of underlying
    Physical,
}

/// Standard US equity trading hours (NYSE/Nasdaq)
pub fn us_equity_sessions() -> Vec<TradingSession> {
    use Weekday::*;

    vec![
        TradingSession {
            name: "Pre-Market".to_string(),
            days: vec![Mon, Tue, Wed, Thu, Fri],
            start_time: NaiveTime::from_hms_opt(4, 0, 0).unwrap(),
            end_time: NaiveTime::from_hms_opt(9, 30, 0).unwrap(),
        },
        TradingSession {
            name: "Regular".to_string(),
            days: vec![Mon, Tue, Wed, Thu, Fri],
            start_time: NaiveTime::from_hms_opt(9, 30, 0).unwrap(),
            end_time: NaiveTime::from_hms_opt(16, 0, 0).unwrap(),
        },
        TradingSession {
            name: "After-Hours".to_string(),
            days: vec![Mon, Tue, Wed, Thu, Fri],
            start_time: NaiveTime::from_hms_opt(16, 0, 0).unwrap(),
            end_time: NaiveTime::from_hms_opt(20, 0, 0).unwrap(),
        },
    ]
}

/// CME futures trading hours (nearly 24/5)
pub fn cme_futures_sessions() -> Vec<TradingSession> {
    use Weekday::*;

    vec![TradingSession {
        name: "Regular".to_string(),
        days: vec![Mon, Tue, Wed, Thu, Fri],
        start_time: NaiveTime::from_hms_opt(18, 0, 0).unwrap(), // Sunday 6 PM ET
        end_time: NaiveTime::from_hms_opt(17, 0, 0).unwrap(),   // Friday 5 PM ET
    }]
}

/// 24/7 crypto trading
pub fn crypto_sessions() -> Vec<TradingSession> {
    use Weekday::*;

    vec![TradingSession {
        name: "24/7".to_string(),
        days: vec![Mon, Tue, Wed, Thu, Fri, Sat, Sun],
        start_time: NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
        end_time: NaiveTime::from_hms_opt(23, 59, 59).unwrap(),
    }]
}

/// Forex trading sessions (Tokyo, London, New York)
pub fn forex_sessions() -> Vec<TradingSession> {
    use Weekday::*;

    vec![
        TradingSession {
            name: "Tokyo".to_string(),
            days: vec![Mon, Tue, Wed, Thu, Fri],
            start_time: NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
            end_time: NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
        },
        TradingSession {
            name: "London".to_string(),
            days: vec![Mon, Tue, Wed, Thu, Fri],
            start_time: NaiveTime::from_hms_opt(8, 0, 0).unwrap(),
            end_time: NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
        },
        TradingSession {
            name: "New York".to_string(),
            days: vec![Mon, Tue, Wed, Thu, Fri],
            start_time: NaiveTime::from_hms_opt(13, 0, 0).unwrap(),
            end_time: NaiveTime::from_hms_opt(22, 0, 0).unwrap(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_spec_builder() {
        let spec = AssetSpec::new(
            AssetType::Equity,
            "AAPL".to_string(),
            Exchange::Nasdaq,
            "Apple Inc.".to_string(),
        )
        .with_tick_spec(0.01, 0.01)
        .with_multiplier(1.0)
        .with_currency(Currency::USD);

        assert_eq!(spec.symbol, "AAPL");
        assert_eq!(spec.tick_size, 0.01);
        assert_eq!(spec.currency, Currency::USD);
    }

    #[test]
    fn test_price_validation() {
        let spec = AssetSpec::new(
            AssetType::Equity,
            "AAPL".to_string(),
            Exchange::Nasdaq,
            "Apple Inc.".to_string(),
        )
        .with_tick_spec(0.01, 0.01);

        // Valid price
        assert!(spec.validate_price(150.05).is_ok());

        // Invalid price (not multiple of tick)
        assert!(spec.validate_price(150.001).is_err());

        // Invalid price (negative)
        assert!(spec.validate_price(-150.0).is_err());
    }

    #[test]
    fn test_round_to_tick() {
        let spec = AssetSpec::new(
            AssetType::Futures,
            "ES".to_string(),
            Exchange::CME,
            "E-mini S&P 500".to_string(),
        )
        .with_tick_spec(0.25, 12.50);

        assert_eq!(spec.round_to_tick(5000.12), 5000.00);
        assert_eq!(spec.round_to_tick(5000.15), 5000.25);
        assert_eq!(spec.round_to_tick(5000.38), 5000.50);
    }

    #[test]
    fn test_contract_value() {
        let spec = AssetSpec::new(
            AssetType::Futures,
            "ES".to_string(),
            Exchange::CME,
            "E-mini S&P 500".to_string(),
        )
        .with_tick_spec(0.25, 12.50)
        .with_multiplier(50.0);

        // 1 contract at 5000.00 = 5000 * 50 = $250,000
        let value = spec.calculate_value(5000.00, 1.0).unwrap();
        assert_eq!(value, 250_000.0);
    }

    #[test]
    fn test_currency_symbols() {
        assert_eq!(Currency::USD.symbol(), "$");
        assert_eq!(Currency::EUR.symbol(), "€");
        assert_eq!(Currency::BTC.symbol(), "₿");
    }
}
