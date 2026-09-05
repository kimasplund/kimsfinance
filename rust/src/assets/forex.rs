//! Forex (Foreign Exchange) Asset Implementation
//!
//! Handles forex-specific features:
//! - Currency pairs
//! - Pip value calculations
//! - Cross rates
//! - Session times (Tokyo, London, New York)

use super::specs::{Currency, forex_sessions};
use super::{Asset, AssetError, AssetResult, AssetSpec, AssetType, Exchange};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Forex currency pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForexPair {
    /// Asset specification
    spec: AssetSpec,

    /// Base currency
    base_currency: Currency,

    /// Quote currency
    quote_currency: Currency,

    /// Pip size (usually 0.0001 for most pairs, 0.01 for JPY pairs)
    pip_size: f64,

    /// Pip value (in quote currency)
    pip_value: f64,

    /// Lot size (standard = 100,000 units)
    lot_size: f64,
}

impl ForexPair {
    /// Create new forex pair
    pub fn new(base: Currency, quote: Currency) -> Self {
        let symbol = format!("{}/{}", base.code(), quote.code());

        // Determine pip size (JPY pairs use 0.01, others use 0.0001)
        let pip_size = if matches!(quote, Currency::JPY) {
            0.01
        } else {
            0.0001
        };

        let spec = AssetSpec::new(AssetType::Forex, symbol.clone(), Exchange::Forex, symbol)
            .with_tick_spec(pip_size, pip_size)
            .with_multiplier(1.0)
            .with_quantity_increment(0.01); // Micro lots (0.01)

        // Add forex trading sessions
        let spec = forex_sessions()
            .into_iter()
            .fold(spec, |s, session| s.with_session(session));

        Self {
            spec,
            base_currency: base,
            quote_currency: quote,
            pip_size,
            pip_value: pip_size,
            lot_size: 100_000.0, // Standard lot
        }
    }

    /// Builder: Set lot size
    pub fn with_lot_size(mut self, lot_size: f64) -> Self {
        self.lot_size = lot_size;
        self
    }

    /// Calculate pip value in account currency
    pub fn calculate_pip_value(&self, exchange_rate: f64, lots: f64) -> f64 {
        self.pip_value * self.lot_size * lots / exchange_rate
    }

    /// Calculate position value
    pub fn calculate_position_value(&self, rate: f64, lots: f64) -> f64 {
        rate * self.lot_size * lots
    }

    /// Calculate profit/loss in pips
    pub fn calculate_pips(&self, entry_rate: f64, exit_rate: f64, is_long: bool) -> f64 {
        if is_long {
            (exit_rate - entry_rate) / self.pip_size
        } else {
            (entry_rate - exit_rate) / self.pip_size
        }
    }

    /// Get base currency
    pub fn base_currency(&self) -> Currency {
        self.base_currency
    }

    /// Get quote currency
    pub fn quote_currency(&self) -> Currency {
        self.quote_currency
    }

    /// Get pip size
    pub fn pip_size(&self) -> f64 {
        self.pip_size
    }

    /// Get lot size
    pub fn lot_size(&self) -> f64 {
        self.lot_size
    }
}

impl Asset for ForexPair {
    fn asset_type(&self) -> AssetType {
        AssetType::Forex
    }

    fn symbol(&self) -> &str {
        &self.spec.symbol
    }

    fn validate_price(&self, price: f64) -> AssetResult<f64> {
        if price <= 0.0 {
            return Err(AssetError::InvalidPrice(
                "Exchange rate must be positive".to_string(),
            ));
        }
        Ok(price)
    }

    fn normalize_symbol(&self, symbol: &str) -> AssetResult<String> {
        // Normalize: EURUSD, EUR/USD, EUR-USD -> EUR/USD
        let cleaned = symbol.replace(&['-', '_'][..], "/").to_uppercase();
        if cleaned.contains('/') {
            Ok(cleaned)
        } else if cleaned.len() == 6 {
            // EURUSD -> EUR/USD
            Ok(format!("{}/{}", &cleaned[0..3], &cleaned[3..6]))
        } else {
            Err(AssetError::InvalidSymbol(format!(
                "Invalid forex pair format: {}",
                symbol
            )))
        }
    }

    fn calculate_value(&self, rate: f64, lots: f64) -> AssetResult<f64> {
        Ok(self.calculate_position_value(rate, lots))
    }

    fn is_market_open(&self, timestamp: DateTime<Utc>) -> bool {
        // Forex markets are open 24/5 (Sunday 5 PM ET - Friday 5 PM ET)
        self.spec.is_market_open(timestamp)
    }

    fn tick_size(&self) -> f64 {
        self.pip_size
    }

    fn quantity_increment(&self) -> f64 {
        0.01 // Micro lots
    }

    fn specification(&self) -> &AssetSpec {
        &self.spec
    }
}

/// Standard forex pairs
pub struct StandardForexPairs;

impl StandardForexPairs {
    /// EUR/USD (most liquid pair)
    pub fn eurusd() -> ForexPair {
        ForexPair::new(Currency::USD, Currency::EUR)
    }

    /// GBP/USD (cable)
    pub fn gbpusd() -> ForexPair {
        ForexPair::new(Currency::GBP, Currency::USD)
    }

    /// USD/JPY
    pub fn usdjpy() -> ForexPair {
        ForexPair::new(Currency::USD, Currency::JPY)
    }

    /// USD/CHF (swissie)
    pub fn usdchf() -> ForexPair {
        ForexPair::new(Currency::USD, Currency::CHF)
    }

    /// AUD/USD (aussie)
    pub fn audusd() -> ForexPair {
        ForexPair::new(Currency::AUD, Currency::USD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forex_pair_creation() {
        let eurusd = ForexPair::new(Currency::EUR, Currency::USD);
        assert_eq!(eurusd.symbol(), "EUR/USD");
        assert_eq!(eurusd.pip_size(), 0.0001);
    }

    #[test]
    fn test_jpy_pip_size() {
        let usdjpy = ForexPair::new(Currency::USD, Currency::JPY);
        assert_eq!(usdjpy.pip_size(), 0.01); // JPY pairs use 0.01
    }

    #[test]
    fn test_pip_calculation() {
        let eurusd = ForexPair::new(Currency::EUR, Currency::USD);

        // Long: buy at 1.1000, sell at 1.1050 = 50 pips profit
        let pips = eurusd.calculate_pips(1.1000, 1.1050, true);
        assert!((pips - 50.0).abs() < 0.01);

        // Short: sell at 1.1050, buy at 1.1000 = 50 pips profit
        let pips = eurusd.calculate_pips(1.1050, 1.1000, false);
        assert!((pips - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_position_value() {
        let eurusd = ForexPair::new(Currency::EUR, Currency::USD);

        // 1 standard lot at 1.1000 = 110,000 USD
        let value = eurusd.calculate_position_value(1.1000, 1.0);
        assert!((value - 110_000.0).abs() < 0.01);
    }

    #[test]
    fn test_symbol_normalization() {
        let eurusd = ForexPair::new(Currency::EUR, Currency::USD);

        assert_eq!(eurusd.normalize_symbol("EURUSD").unwrap(), "EUR/USD");
        assert_eq!(eurusd.normalize_symbol("EUR-USD").unwrap(), "EUR/USD");
        assert_eq!(eurusd.normalize_symbol("EUR/USD").unwrap(), "EUR/USD");
    }
}
