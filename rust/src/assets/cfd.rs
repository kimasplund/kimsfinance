//! CFD (Contract for Difference) Asset Implementation
//!
//! Handles CFD-specific features:
//! - Leverage and margin
//! - Overnight financing charges
//! - No ownership of underlying asset

use super::specs::{Currency, SettlementType};
use super::{Asset, AssetResult, AssetSpec, AssetType, Exchange};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// CFD contract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CFDContract {
    /// Asset specification
    spec: AssetSpec,

    /// Underlying asset symbol
    underlying: String,

    /// Maximum leverage
    max_leverage: f64,

    /// Overnight financing rate (annual, as decimal)
    financing_rate: f64,

    /// Minimum margin requirement (as decimal, e.g., 0.01 = 1%)
    margin_requirement: f64,
}

impl CFDContract {
    /// Create new CFD contract
    pub fn new(underlying: &str, exchange: Exchange, max_leverage: f64, tick_size: f64) -> Self {
        let symbol = format!("{}.CFD", underlying);

        let spec = AssetSpec::new(AssetType::CFD, symbol, exchange, underlying.to_string())
            .with_tick_spec(tick_size, tick_size)
            .with_multiplier(1.0)
            .with_quantity_increment(0.01) // Fractional contracts
            .with_currency(Currency::USD)
            .with_settlement(SettlementType::Cash)
            .with_underlying(underlying.to_string());

        let margin_requirement = 1.0 / max_leverage;

        Self {
            spec,
            underlying: underlying.to_string(),
            max_leverage,
            financing_rate: 0.05, // Default 5% annual
            margin_requirement,
        }
    }

    /// Builder: Set financing rate
    pub fn with_financing_rate(mut self, rate: f64) -> Self {
        self.financing_rate = rate;
        self
    }

    /// Calculate required margin for position
    pub fn calculate_margin(&self, price: f64, quantity: f64) -> f64 {
        let position_value = price * quantity.abs();
        position_value * self.margin_requirement
    }

    /// Calculate overnight financing charge
    pub fn calculate_overnight_financing(&self, position_value: f64, days: f64) -> f64 {
        position_value * self.financing_rate * days / 365.0
    }

    /// Get maximum position size for given capital
    pub fn max_position_size(&self, capital: f64, price: f64) -> f64 {
        (capital * self.max_leverage) / price
    }

    /// Get underlying symbol
    pub fn underlying(&self) -> &str {
        &self.underlying
    }

    /// Get maximum leverage
    pub fn max_leverage(&self) -> f64 {
        self.max_leverage
    }

    /// Get margin requirement
    pub fn margin_requirement(&self) -> f64 {
        self.margin_requirement
    }
}

impl Asset for CFDContract {
    fn asset_type(&self) -> AssetType {
        AssetType::CFD
    }

    fn symbol(&self) -> &str {
        &self.spec.symbol
    }

    fn validate_price(&self, price: f64) -> AssetResult<f64> {
        self.spec.validate_price(price)
    }

    fn normalize_symbol(&self, symbol: &str) -> AssetResult<String> {
        // Remove .CFD suffix if present
        Ok(symbol.replace(".CFD", "").to_uppercase())
    }

    fn calculate_value(&self, price: f64, quantity: f64) -> AssetResult<f64> {
        Ok(price * quantity)
    }

    fn is_market_open(&self, timestamp: DateTime<Utc>) -> bool {
        self.spec.is_market_open(timestamp)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfd_creation() {
        let cfd = CFDContract::new("AAPL", Exchange::OTC, 10.0, 0.01);
        assert_eq!(cfd.symbol(), "AAPL.CFD");
        assert_eq!(cfd.max_leverage(), 10.0);
        assert_eq!(cfd.margin_requirement(), 0.1); // 1/10 = 10%
    }

    #[test]
    fn test_margin_calculation() {
        let cfd = CFDContract::new("AAPL", Exchange::OTC, 10.0, 0.01);

        // 100 shares at $150 = $15,000 position
        // With 10x leverage: $1,500 margin required
        let margin = cfd.calculate_margin(150.0, 100.0);
        assert_eq!(margin, 1_500.0);
    }

    #[test]
    fn test_max_position_size() {
        let cfd = CFDContract::new("AAPL", Exchange::OTC, 10.0, 0.01);

        // $10,000 capital, $150 price, 10x leverage
        // Max position: ($10,000 * 10) / $150 = 666.67 shares
        let max_size = cfd.max_position_size(10_000.0, 150.0);
        assert!((max_size - 666.67).abs() < 0.1);
    }

    #[test]
    fn test_overnight_financing() {
        let cfd = CFDContract::new("AAPL", Exchange::OTC, 10.0, 0.01).with_financing_rate(0.05); // 5% annual

        // $10,000 position for 1 day
        // Financing: $10,000 * 0.05 / 365 = $1.37
        let financing = cfd.calculate_overnight_financing(10_000.0, 1.0);
        assert!((financing - 1.37).abs() < 0.01);
    }
}
