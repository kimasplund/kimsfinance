//! Cryptocurrency Asset Implementation
//!
//! Handles crypto-specific features:
//! - 24/7 trading
//! - Exchange-specific conventions
//! - Satoshi/wei precision
//! - Multiple quote currencies (USD, USDT, BTC, ETH)

use super::specs::{Currency, crypto_sessions};
use super::{Asset, AssetError, AssetResult, AssetSpec, AssetType, Exchange};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Cryptocurrency asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoAsset {
    /// Asset specification
    spec: AssetSpec,

    /// Base currency
    base_currency: Currency,

    /// Quote currency
    quote_currency: Currency,

    /// Minimum precision (e.g., 8 decimals for BTC)
    precision: u8,

    /// Maker fee (as decimal, e.g., 0.001 = 0.1%)
    maker_fee: f64,

    /// Taker fee (as decimal)
    taker_fee: f64,
}

impl CryptoAsset {
    /// Create new crypto asset
    pub fn new(base: Currency, quote: Currency, exchange: Exchange) -> Self {
        let symbol = format!("{}/{}", base.code(), quote.code());

        // Determine precision based on base currency
        let precision = match base {
            Currency::BTC => 8,
            Currency::ETH => 18,
            _ => 8,
        };

        let tick_size = 10f64.powi(-(precision as i32));

        let spec = AssetSpec::new(AssetType::Crypto, symbol.clone(), exchange, symbol)
            .with_tick_spec(tick_size, tick_size)
            .with_multiplier(1.0)
            .with_quantity_increment(tick_size)
            .with_currency(quote);

        // Add 24/7 crypto trading sessions
        let spec = crypto_sessions()
            .into_iter()
            .fold(spec, |s, session| s.with_session(session));

        Self {
            spec,
            base_currency: base,
            quote_currency: quote,
            precision,
            maker_fee: 0.001, // Default 0.1%
            taker_fee: 0.001,
        }
    }

    /// Builder: Set fees
    pub fn with_fees(mut self, maker_fee: f64, taker_fee: f64) -> Self {
        self.maker_fee = maker_fee;
        self.taker_fee = taker_fee;
        self
    }

    /// Builder: Set precision
    pub fn with_precision(mut self, precision: u8) -> Self {
        self.precision = precision;
        let tick_size = 10f64.powi(-(precision as i32));
        self.spec.tick_size = tick_size;
        self.spec.tick_value = tick_size;
        self.spec.quantity_increment = tick_size;
        self
    }

    /// Convert to satoshis (for BTC)
    pub fn to_satoshis(&self, amount: f64) -> i64 {
        if self.base_currency == Currency::BTC {
            (amount * 100_000_000.0).round() as i64
        } else {
            0
        }
    }

    /// Convert from satoshis (for BTC)
    pub fn from_satoshis(&self, satoshis: i64) -> f64 {
        if self.base_currency == Currency::BTC {
            satoshis as f64 / 100_000_000.0
        } else {
            0.0
        }
    }

    /// Calculate trading fee
    pub fn calculate_fee(&self, value: f64, is_maker: bool) -> f64 {
        let fee_rate = if is_maker {
            self.maker_fee
        } else {
            self.taker_fee
        };
        value * fee_rate
    }

    /// Get base currency
    pub fn base_currency(&self) -> Currency {
        self.base_currency
    }

    /// Get quote currency
    pub fn quote_currency(&self) -> Currency {
        self.quote_currency
    }

    /// Get precision
    pub fn precision(&self) -> u8 {
        self.precision
    }
}

impl Asset for CryptoAsset {
    fn asset_type(&self) -> AssetType {
        AssetType::Crypto
    }

    fn symbol(&self) -> &str {
        &self.spec.symbol
    }

    fn validate_price(&self, price: f64) -> AssetResult<f64> {
        if price <= 0.0 {
            return Err(AssetError::InvalidPrice(
                "Price must be positive".to_string(),
            ));
        }
        Ok(price)
    }

    fn normalize_symbol(&self, symbol: &str) -> AssetResult<String> {
        // Normalize: BTCUSD, BTC/USD, BTC-USD -> BTC/USD
        let cleaned = symbol.replace(&['-', '_'][..], "/").to_uppercase();
        if cleaned.contains('/') {
            Ok(cleaned)
        } else {
            // Try to split common pairs
            for base in ["BTC", "ETH", "USDT", "BNB", "SOL"] {
                if cleaned.starts_with(base) {
                    let quote = &cleaned[base.len()..];
                    return Ok(format!("{}/{}", base, quote));
                }
            }
            Err(AssetError::InvalidSymbol(format!(
                "Invalid crypto pair format: {}",
                symbol
            )))
        }
    }

    fn calculate_value(&self, price: f64, quantity: f64) -> AssetResult<f64> {
        Ok(price * quantity)
    }

    fn is_market_open(&self, _timestamp: DateTime<Utc>) -> bool {
        // Crypto markets are always open (24/7)
        true
    }

    fn tick_size(&self) -> f64 {
        self.spec.tick_size
    }

    fn quantity_increment(&self) -> f64 {
        self.spec.quantity_increment
    }

    fn specification(&self) -> &AssetSpec {
        &self.spec
    }
}

/// Standard crypto pairs
pub struct StandardCryptoPairs;

impl StandardCryptoPairs {
    /// BTC/USD
    pub fn btcusd(exchange: Exchange) -> CryptoAsset {
        CryptoAsset::new(Currency::BTC, Currency::USD, exchange).with_fees(0.001, 0.001) // 0.1% maker/taker
    }

    /// BTC/USDT
    pub fn btcusdt(exchange: Exchange) -> CryptoAsset {
        CryptoAsset::new(Currency::BTC, Currency::USDT, exchange).with_fees(0.001, 0.001)
    }

    /// ETH/USD
    pub fn ethusd(exchange: Exchange) -> CryptoAsset {
        CryptoAsset::new(Currency::ETH, Currency::USD, exchange).with_fees(0.001, 0.001)
    }

    /// ETH/BTC
    pub fn ethbtc(exchange: Exchange) -> CryptoAsset {
        CryptoAsset::new(Currency::ETH, Currency::BTC, exchange).with_fees(0.001, 0.001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crypto_asset_creation() {
        let btcusd = CryptoAsset::new(Currency::BTC, Currency::USD, Exchange::Binance);
        assert_eq!(btcusd.symbol(), "BTC/USD");
        assert_eq!(btcusd.precision(), 8);
    }

    #[test]
    fn test_satoshi_conversion() {
        let btcusd = CryptoAsset::new(Currency::BTC, Currency::USD, Exchange::Binance);

        // 1 BTC = 100,000,000 satoshis
        assert_eq!(btcusd.to_satoshis(1.0), 100_000_000);

        // 0.00000001 BTC = 1 satoshi
        assert_eq!(btcusd.to_satoshis(0.00000001), 1);

        // Round trip
        let sats = btcusd.to_satoshis(0.12345678);
        assert_eq!(btcusd.from_satoshis(sats), 0.12345678);
    }

    #[test]
    fn test_fee_calculation() {
        let btcusd = CryptoAsset::new(Currency::BTC, Currency::USD, Exchange::Binance)
            .with_fees(0.001, 0.0015); // 0.1% maker, 0.15% taker

        let value = 10_000.0;

        // Maker fee: 0.1% of $10,000 = $10
        assert_eq!(btcusd.calculate_fee(value, true), 10.0);

        // Taker fee: 0.15% of $10,000 = $15
        assert_eq!(btcusd.calculate_fee(value, false), 15.0);
    }

    #[test]
    fn test_always_open() {
        let btcusd = CryptoAsset::new(Currency::BTC, Currency::USD, Exchange::Binance);
        let now = Utc::now();

        // Crypto markets are always open
        assert!(btcusd.is_market_open(now));
    }

    #[test]
    fn test_symbol_normalization() {
        let btcusd = CryptoAsset::new(Currency::BTC, Currency::USD, Exchange::Binance);

        assert_eq!(btcusd.normalize_symbol("BTCUSD").unwrap(), "BTC/USD");
        assert_eq!(btcusd.normalize_symbol("BTC-USD").unwrap(), "BTC/USD");
        assert_eq!(btcusd.normalize_symbol("BTC/USD").unwrap(), "BTC/USD");
    }
}
