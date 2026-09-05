//! Equity (Stock) Asset Implementation
//!
//! Handles stock-specific features:
//! - Corporate actions (splits, dividends)
//! - Tick size rules (penny pilot, sub-penny)
//! - Market hours (regular, pre-market, after-hours)
//! - Symbol normalization

use super::specs::{Currency, us_equity_sessions};
use super::{Asset, AssetError, AssetResult, AssetSpec, AssetType, Exchange};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Equity asset (stock)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityAsset {
    /// Asset specification
    spec: AssetSpec,

    /// Corporate actions history
    corporate_actions: Vec<CorporateAction>,

    /// Penny pilot program eligibility
    is_penny_pilot: bool,

    /// Current price (for tick size rules)
    cached_price: Option<f64>,
}

impl EquityAsset {
    /// Create new equity asset
    pub fn new(symbol: &str, exchange: Exchange) -> Self {
        let spec = AssetSpec::new(
            AssetType::Equity,
            symbol.to_string(),
            exchange,
            symbol.to_string(),
        )
        .with_tick_spec(0.01, 0.01) // Default penny tick
        .with_quantity_increment(1.0) // Shares are whole numbers
        .with_currency(Currency::USD);

        // Add US equity trading sessions
        let spec = us_equity_sessions()
            .into_iter()
            .fold(spec, |s, session| s.with_session(session));

        Self {
            spec,
            corporate_actions: Vec::new(),
            is_penny_pilot: true, // Most stocks are penny pilot now
            cached_price: None,
        }
    }

    /// Set penny pilot eligibility
    pub fn with_penny_pilot(mut self, is_penny_pilot: bool) -> Self {
        self.is_penny_pilot = is_penny_pilot;
        self
    }

    /// Add corporate action
    pub fn add_corporate_action(&mut self, action: CorporateAction) {
        self.corporate_actions.push(action);
        // Sort by date
        self.corporate_actions.sort_by_key(|a| a.ex_date);
    }

    /// Get applicable tick size for given price
    pub fn get_tick_size(&self, price: f64) -> f64 {
        if self.is_penny_pilot {
            // Penny pilot: $0.01 tick for prices >= $1.00
            if price >= 1.0 {
                0.01
            } else {
                // Sub-penny: $0.0001 tick for prices < $1.00
                0.0001
            }
        } else {
            // Non-penny pilot: $0.05 tick
            0.05
        }
    }

    /// Adjust price for corporate actions
    pub fn adjust_price(&self, price: f64, as_of_date: NaiveDate) -> f64 {
        let mut adjusted = price;

        for action in &self.corporate_actions {
            if action.ex_date <= as_of_date {
                adjusted = action.adjust_price(adjusted);
            }
        }

        adjusted
    }

    /// Get dividend history
    pub fn get_dividends(
        &self,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<&CorporateAction> {
        self.corporate_actions
            .iter()
            .filter(|action| {
                matches!(action.action_type, CorporateActionType::Dividend { .. })
                    && action.ex_date >= start_date
                    && action.ex_date <= end_date
            })
            .collect()
    }

    /// Get split history
    pub fn get_splits(&self, start_date: NaiveDate, end_date: NaiveDate) -> Vec<&CorporateAction> {
        self.corporate_actions
            .iter()
            .filter(|action| {
                matches!(action.action_type, CorporateActionType::Split { .. })
                    && action.ex_date >= start_date
                    && action.ex_date <= end_date
            })
            .collect()
    }

    /// Normalize symbol (remove exchange suffix, convert to uppercase)
    pub fn normalize_symbol_str(symbol: &str) -> String {
        symbol
            .split('.')
            .next()
            .unwrap_or(symbol)
            .to_uppercase()
            .trim()
            .to_string()
    }
}

impl Asset for EquityAsset {
    fn asset_type(&self) -> AssetType {
        AssetType::Equity
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

        let tick = self.get_tick_size(price);
        let remainder = (price / tick).fract().abs();

        if remainder > 1e-10 {
            return Err(AssetError::InvalidTickSize(format!(
                "Price {} is not a multiple of tick size {}",
                price, tick
            )));
        }

        Ok(price)
    }

    fn normalize_symbol(&self, symbol: &str) -> AssetResult<String> {
        Ok(Self::normalize_symbol_str(symbol))
    }

    fn calculate_value(&self, price: f64, quantity: f64) -> AssetResult<f64> {
        let validated_price = self.validate_price(price)?;
        Ok(validated_price * quantity)
    }

    fn is_market_open(&self, timestamp: DateTime<Utc>) -> bool {
        self.spec.is_market_open(timestamp)
    }

    fn tick_size(&self) -> f64 {
        self.cached_price
            .map(|p| self.get_tick_size(p))
            .unwrap_or(0.01)
    }

    fn quantity_increment(&self) -> f64 {
        1.0 // Shares are whole numbers
    }

    fn specification(&self) -> &AssetSpec {
        &self.spec
    }
}

/// Corporate action type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorporateActionType {
    /// Stock split (numerator, denominator)
    /// e.g., 2-for-1 split = Split { numerator: 2, denominator: 1 }
    Split { numerator: i32, denominator: i32 },

    /// Cash dividend (amount per share)
    Dividend { amount: f64 },

    /// Stock dividend (shares per share)
    StockDividend { shares_per_share: f64 },

    /// Merger/acquisition
    Merger {
        new_symbol: String,
        conversion_ratio: f64,
    },

    /// Spinoff
    Spinoff {
        new_symbol: String,
        allocation_ratio: f64,
    },

    /// Rights offering
    Rights { strike_price: f64, ratio: f64 },
}

/// Corporate action record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorporateAction {
    /// Action type
    pub action_type: CorporateActionType,

    /// Ex-dividend date (date after which new buyers don't get dividend)
    pub ex_date: NaiveDate,

    /// Record date
    pub record_date: NaiveDate,

    /// Payment date
    pub payment_date: Option<NaiveDate>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl CorporateAction {
    /// Create new split action
    pub fn new_split(ex_date: NaiveDate, numerator: i32, denominator: i32) -> Self {
        Self {
            action_type: CorporateActionType::Split {
                numerator,
                denominator,
            },
            ex_date,
            record_date: ex_date,
            payment_date: None,
            metadata: HashMap::new(),
        }
    }

    /// Create new dividend action
    pub fn new_dividend(
        ex_date: NaiveDate,
        record_date: NaiveDate,
        payment_date: NaiveDate,
        amount: f64,
    ) -> Self {
        Self {
            action_type: CorporateActionType::Dividend { amount },
            ex_date,
            record_date,
            payment_date: Some(payment_date),
            metadata: HashMap::new(),
        }
    }

    /// Adjust historical price for this corporate action
    pub fn adjust_price(&self, price: f64) -> f64 {
        match &self.action_type {
            CorporateActionType::Split {
                numerator,
                denominator,
            } => {
                // Price adjustment for split
                price * (*denominator as f64) / (*numerator as f64)
            }
            CorporateActionType::Dividend { amount } => {
                // Price adjustment for cash dividend
                price - amount
            }
            CorporateActionType::StockDividend { shares_per_share } => {
                // Price adjustment for stock dividend
                price / (1.0 + shares_per_share)
            }
            _ => price, // Other actions don't affect price directly
        }
    }

    /// Adjust historical quantity for this corporate action
    pub fn adjust_quantity(&self, quantity: f64) -> f64 {
        match &self.action_type {
            CorporateActionType::Split {
                numerator,
                denominator,
            } => {
                // Quantity adjustment for split
                quantity * (*numerator as f64) / (*denominator as f64)
            }
            CorporateActionType::StockDividend { shares_per_share } => {
                // Quantity adjustment for stock dividend
                quantity * (1.0 + shares_per_share)
            }
            _ => quantity, // Other actions don't affect quantity
        }
    }
}

/// Market capitalization tier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketCapTier {
    MegaCap,  // > $200B
    LargeCap, // $10B - $200B
    MidCap,   // $2B - $10B
    SmallCap, // $300M - $2B
    MicroCap, // $50M - $300M
    NanoCap,  // < $50M
}

impl MarketCapTier {
    /// Classify market cap tier
    pub fn from_market_cap(market_cap: f64) -> Self {
        if market_cap >= 200_000_000_000.0 {
            MarketCapTier::MegaCap
        } else if market_cap >= 10_000_000_000.0 {
            MarketCapTier::LargeCap
        } else if market_cap >= 2_000_000_000.0 {
            MarketCapTier::MidCap
        } else if market_cap >= 300_000_000.0 {
            MarketCapTier::SmallCap
        } else if market_cap >= 50_000_000.0 {
            MarketCapTier::MicroCap
        } else {
            MarketCapTier::NanoCap
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equity_creation() {
        let aapl = EquityAsset::new("AAPL", Exchange::Nasdaq);
        assert_eq!(aapl.symbol(), "AAPL");
        assert_eq!(aapl.asset_type(), AssetType::Equity);
    }

    #[test]
    fn test_tick_size_penny_pilot() {
        let asset = EquityAsset::new("AAPL", Exchange::Nasdaq);

        // >= $1.00: penny tick
        assert_eq!(asset.get_tick_size(150.0), 0.01);

        // < $1.00: sub-penny tick
        assert_eq!(asset.get_tick_size(0.50), 0.0001);
    }

    #[test]
    fn test_price_validation() {
        let asset = EquityAsset::new("AAPL", Exchange::Nasdaq);

        // Valid penny price
        assert!(asset.validate_price(150.05).is_ok());

        // Invalid sub-cent price
        assert!(asset.validate_price(150.001).is_err());
    }

    #[test]
    fn test_split_adjustment() {
        let split = CorporateAction::new_split(
            NaiveDate::from_ymd_opt(2024, 6, 10).unwrap(),
            2, // 2-for-1 split
            1,
        );

        // Pre-split price $100 becomes $50 post-split
        assert_eq!(split.adjust_price(100.0), 50.0);

        // 100 shares become 200 shares
        assert_eq!(split.adjust_quantity(100.0), 200.0);
    }

    #[test]
    fn test_dividend_adjustment() {
        let dividend = CorporateAction::new_dividend(
            NaiveDate::from_ymd_opt(2024, 6, 10).unwrap(),
            NaiveDate::from_ymd_opt(2024, 6, 12).unwrap(),
            NaiveDate::from_ymd_opt(2024, 6, 15).unwrap(),
            0.50, // $0.50 per share
        );

        // Price adjusts down by dividend amount
        assert_eq!(dividend.adjust_price(100.0), 99.5);
    }

    #[test]
    fn test_symbol_normalization() {
        assert_eq!(EquityAsset::normalize_symbol_str("aapl"), "AAPL");
        assert_eq!(EquityAsset::normalize_symbol_str("AAPL.O"), "AAPL");
        assert_eq!(EquityAsset::normalize_symbol_str(" BRK.B "), "BRK");
    }

    #[test]
    fn test_market_cap_tiers() {
        assert_eq!(
            MarketCapTier::from_market_cap(500_000_000_000.0),
            MarketCapTier::MegaCap
        );
        assert_eq!(
            MarketCapTier::from_market_cap(50_000_000_000.0),
            MarketCapTier::LargeCap
        );
        assert_eq!(
            MarketCapTier::from_market_cap(5_000_000_000.0),
            MarketCapTier::MidCap
        );
    }
}
