//! Index Asset Implementation
//!
//! Handles index-specific features:
//! - Composite instruments (not directly tradable)
//! - Constituent weighting
//! - Rebalancing events

use super::specs::{Currency, SettlementType};
use super::{Asset, AssetError, AssetResult, AssetSpec, AssetType, Exchange};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Market index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketIndex {
    /// Asset specification
    spec: AssetSpec,

    /// Index constituents (symbol -> weight)
    constituents: HashMap<String, f64>,

    /// Index methodology (price-weighted, market-cap-weighted, etc.)
    methodology: IndexMethodology,

    /// Divisor (for price-weighted indices)
    divisor: f64,

    /// Rebalancing schedule
    rebalancing_schedule: RebalancingSchedule,
}

impl MarketIndex {
    /// Create new market index
    pub fn new(symbol: &str, name: &str, methodology: IndexMethodology) -> Self {
        let spec = AssetSpec::new(
            AssetType::Index,
            symbol.to_string(),
            Exchange::Custom(0),
            name.to_string(),
        )
        .with_tick_spec(0.01, 0.01)
        .with_currency(Currency::USD)
        .with_settlement(SettlementType::Cash);

        Self {
            spec,
            constituents: HashMap::new(),
            methodology,
            divisor: 1.0,
            rebalancing_schedule: RebalancingSchedule::Quarterly,
        }
    }

    /// Add constituent to index
    pub fn add_constituent(&mut self, symbol: String, weight: f64) {
        self.constituents.insert(symbol, weight);
    }

    /// Get constituent weight
    pub fn get_weight(&self, symbol: &str) -> Option<f64> {
        self.constituents.get(symbol).copied()
    }

    /// Calculate index value from constituent prices
    pub fn calculate_value(&self, prices: &HashMap<String, f64>) -> AssetResult<f64> {
        match self.methodology {
            IndexMethodology::PriceWeighted => {
                let sum: f64 = self
                    .constituents
                    .keys()
                    .filter_map(|symbol| prices.get(symbol))
                    .sum();
                Ok(sum / self.divisor)
            }
            IndexMethodology::MarketCapWeighted => {
                let weighted_sum: f64 = self
                    .constituents
                    .iter()
                    .filter_map(|(symbol, weight)| prices.get(symbol).map(|price| price * weight))
                    .sum();
                Ok(weighted_sum)
            }
            IndexMethodology::EqualWeighted => {
                let count = self.constituents.len() as f64;
                let sum: f64 = self
                    .constituents
                    .keys()
                    .filter_map(|symbol| prices.get(symbol))
                    .sum();
                Ok(sum / count)
            }
        }
    }

    /// Get constituents
    pub fn constituents(&self) -> &HashMap<String, f64> {
        &self.constituents
    }

    /// Get methodology
    pub fn methodology(&self) -> IndexMethodology {
        self.methodology
    }

    /// Get divisor
    pub fn divisor(&self) -> f64 {
        self.divisor
    }

    /// Set divisor (for price-weighted indices)
    pub fn set_divisor(&mut self, divisor: f64) {
        self.divisor = divisor;
    }
}

impl Asset for MarketIndex {
    fn asset_type(&self) -> AssetType {
        AssetType::Index
    }

    fn symbol(&self) -> &str {
        &self.spec.symbol
    }

    fn validate_price(&self, price: f64) -> AssetResult<f64> {
        if price <= 0.0 {
            return Err(AssetError::InvalidPrice(
                "Index value must be positive".to_string(),
            ));
        }
        Ok(price)
    }

    fn normalize_symbol(&self, symbol: &str) -> AssetResult<String> {
        Ok(symbol.to_uppercase())
    }

    fn calculate_value(&self, price: f64, quantity: f64) -> AssetResult<f64> {
        // Indices themselves aren't traded, but index derivatives are
        Ok(price * quantity)
    }

    fn is_market_open(&self, timestamp: DateTime<Utc>) -> bool {
        self.spec.is_market_open(timestamp)
    }

    fn tick_size(&self) -> f64 {
        self.spec.tick_size
    }

    fn quantity_increment(&self) -> f64 {
        1.0
    }

    fn specification(&self) -> &AssetSpec {
        &self.spec
    }
}

/// Index methodology
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexMethodology {
    /// Price-weighted (e.g., DJIA)
    PriceWeighted,
    /// Market capitalization weighted (e.g., S&P 500)
    MarketCapWeighted,
    /// Equal-weighted (all constituents have equal weight)
    EqualWeighted,
}

/// Rebalancing schedule
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RebalancingSchedule {
    /// Quarterly rebalancing
    Quarterly,
    /// Annual rebalancing
    Annual,
    /// Semi-annual rebalancing
    SemiAnnual,
    /// Monthly rebalancing
    Monthly,
    /// No regular rebalancing
    None,
}

/// Standard indices
pub struct StandardIndices;

impl StandardIndices {
    /// S&P 500
    pub fn sp500() -> MarketIndex {
        MarketIndex::new("SPX", "S&P 500 Index", IndexMethodology::MarketCapWeighted)
    }

    /// Dow Jones Industrial Average
    pub fn djia() -> MarketIndex {
        let mut index = MarketIndex::new(
            "DJI",
            "Dow Jones Industrial Average",
            IndexMethodology::PriceWeighted,
        );
        index.set_divisor(0.152); // Approximate current divisor
        index
    }

    /// Nasdaq 100
    pub fn nasdaq100() -> MarketIndex {
        MarketIndex::new(
            "NDX",
            "Nasdaq 100 Index",
            IndexMethodology::MarketCapWeighted,
        )
    }

    /// Russell 2000
    pub fn russell2000() -> MarketIndex {
        MarketIndex::new(
            "RUT",
            "Russell 2000 Index",
            IndexMethodology::MarketCapWeighted,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_creation() {
        let sp500 = StandardIndices::sp500();
        assert_eq!(sp500.symbol(), "SPX");
        assert_eq!(sp500.methodology(), IndexMethodology::MarketCapWeighted);
    }

    #[test]
    fn test_constituent_management() {
        let mut index = MarketIndex::new("TEST", "Test Index", IndexMethodology::MarketCapWeighted);

        index.add_constituent("AAPL".to_string(), 0.07);
        index.add_constituent("MSFT".to_string(), 0.06);

        assert_eq!(index.get_weight("AAPL"), Some(0.07));
        assert_eq!(index.get_weight("MSFT"), Some(0.06));
        assert_eq!(index.get_weight("GOOGL"), None);
    }

    #[test]
    fn test_market_cap_weighted_calculation() {
        let mut index = MarketIndex::new("TEST", "Test Index", IndexMethodology::MarketCapWeighted);

        index.add_constituent("AAPL".to_string(), 0.07);
        index.add_constituent("MSFT".to_string(), 0.06);

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 150.0);
        prices.insert("MSFT".to_string(), 300.0);

        // Value = (150 * 0.07) + (300 * 0.06) = 10.5 + 18 = 28.5
        let value = index.calculate_value(&prices).unwrap();
        assert!((value - 28.5).abs() < 0.01);
    }

    #[test]
    fn test_price_weighted_calculation() {
        let mut index = MarketIndex::new("TEST", "Test Index", IndexMethodology::PriceWeighted);
        index.set_divisor(2.0);

        index.add_constituent("AAPL".to_string(), 1.0);
        index.add_constituent("MSFT".to_string(), 1.0);

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 150.0);
        prices.insert("MSFT".to_string(), 300.0);

        // Value = (150 + 300) / 2 = 225
        let value = index.calculate_value(&prices).unwrap();
        assert!((value - 225.0).abs() < 0.01);
    }
}
