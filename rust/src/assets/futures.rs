//! Futures Contract Implementation
//!
//! Handles futures-specific features:
//! - Contract specifications (multiplier, tick size, tick value)
//! - Expiration handling and roll-over logic
//! - Margin requirements
//! - Settlement types (cash vs physical)

use super::specs::{Currency, SettlementType, cme_futures_sessions};
use super::{Asset, AssetResult, AssetSpec, AssetType, Exchange};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Futures contract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuturesContract {
    /// Asset specification
    spec: AssetSpec,

    /// Contract month code
    month_code: FuturesMonthCode,

    /// Contract year
    year: i32,

    /// Expiration date
    expiration: DateTime<Utc>,

    /// First notice date (for physical delivery)
    first_notice_date: Option<DateTime<Utc>>,

    /// Last trading date
    last_trading_date: DateTime<Utc>,

    /// Settlement type
    settlement_type: SettlementType,

    /// Initial margin (per contract, in dollars)
    initial_margin: f64,

    /// Maintenance margin (per contract, in dollars)
    maintenance_margin: f64,
}

impl FuturesContract {
    /// Create new futures contract
    #[allow(clippy::too_many_arguments)] // public API: signature is documented and used by callers
    pub fn new(
        symbol: &str,
        exchange: Exchange,
        month_code: FuturesMonthCode,
        year: i32,
        expiration: DateTime<Utc>,
        tick_size: f64,
        tick_value: f64,
        contract_multiplier: f64,
    ) -> Self {
        let full_symbol = format!("{}{}{}", symbol, month_code.to_char(), year % 100);

        let spec = AssetSpec::new(
            AssetType::Futures,
            full_symbol,
            exchange,
            symbol.to_string(),
        )
        .with_tick_spec(tick_size, tick_value)
        .with_multiplier(contract_multiplier)
        .with_quantity_increment(1.0)
        .with_expiration(expiration)
        .with_currency(Currency::USD);

        // Add CME futures trading sessions
        let spec = cme_futures_sessions()
            .into_iter()
            .fold(spec, |s, session| s.with_session(session));

        Self {
            spec,
            month_code,
            year,
            expiration,
            first_notice_date: None,
            last_trading_date: expiration,
            settlement_type: SettlementType::Cash,
            initial_margin: 0.0,
            maintenance_margin: 0.0,
        }
    }

    /// Builder: Set settlement type
    pub fn with_settlement(mut self, settlement_type: SettlementType) -> Self {
        self.settlement_type = settlement_type;
        self.spec.settlement_type = settlement_type;
        self
    }

    /// Builder: Set margin requirements
    pub fn with_margins(mut self, initial: f64, maintenance: f64) -> Self {
        self.initial_margin = initial;
        self.maintenance_margin = maintenance;
        self
    }

    /// Builder: Set first notice date
    pub fn with_first_notice_date(mut self, date: DateTime<Utc>) -> Self {
        self.first_notice_date = Some(date);
        self
    }

    /// Builder: Set last trading date
    pub fn with_last_trading_date(mut self, date: DateTime<Utc>) -> Self {
        self.last_trading_date = date;
        self
    }

    /// Check if contract is expired
    pub fn is_expired(&self, timestamp: DateTime<Utc>) -> bool {
        timestamp >= self.expiration
    }

    /// Check if contract is near expiration (within days threshold)
    pub fn is_near_expiration(&self, timestamp: DateTime<Utc>, days_threshold: i64) -> bool {
        let days_to_expiry = (self.expiration - timestamp).num_days();
        days_to_expiry <= days_threshold && days_to_expiry >= 0
    }

    /// Get days to expiration
    pub fn days_to_expiration(&self, timestamp: DateTime<Utc>) -> i64 {
        (self.expiration - timestamp).num_days()
    }

    /// Get next contract in the series
    pub fn next_contract(&self) -> (FuturesMonthCode, i32) {
        let next_month = self.month_code.next();
        let next_year = if next_month < self.month_code {
            self.year + 1
        } else {
            self.year
        };
        (next_month, next_year)
    }

    /// Calculate point value (contract multiplier * price change)
    pub fn point_value(&self, price_change: f64) -> f64 {
        price_change * self.spec.contract_multiplier
    }

    /// Calculate tick profit/loss
    pub fn tick_pnl(&self, num_ticks: f64) -> f64 {
        num_ticks * self.spec.tick_value
    }

    /// Calculate required margin for positions
    pub fn required_margin(&self, num_contracts: i32) -> (f64, f64) {
        let initial = self.initial_margin * num_contracts.abs() as f64;
        let maintenance = self.maintenance_margin * num_contracts.abs() as f64;
        (initial, maintenance)
    }

    /// Get settlement type
    pub fn settlement_type(&self) -> SettlementType {
        self.settlement_type
    }

    /// Get expiration date
    pub fn expiration(&self) -> DateTime<Utc> {
        self.expiration
    }

    /// Get month code
    pub fn month_code(&self) -> FuturesMonthCode {
        self.month_code
    }

    /// Get year
    pub fn year(&self) -> i32 {
        self.year
    }
}

impl Asset for FuturesContract {
    fn asset_type(&self) -> AssetType {
        AssetType::Futures
    }

    fn symbol(&self) -> &str {
        &self.spec.symbol
    }

    fn validate_price(&self, price: f64) -> AssetResult<f64> {
        self.spec.validate_price(price)
    }

    fn normalize_symbol(&self, symbol: &str) -> AssetResult<String> {
        // Futures symbols: ESH24, NQM24, etc.
        Ok(symbol.to_uppercase())
    }

    fn calculate_value(&self, price: f64, quantity: f64) -> AssetResult<f64> {
        self.spec.calculate_value(price, quantity)
    }

    fn is_market_open(&self, timestamp: DateTime<Utc>) -> bool {
        !self.is_expired(timestamp) && self.spec.is_market_open(timestamp)
    }

    fn tick_size(&self) -> f64 {
        self.spec.tick_size
    }

    fn quantity_increment(&self) -> f64 {
        1.0 // Contracts are whole numbers
    }

    fn contract_multiplier(&self) -> f64 {
        self.spec.contract_multiplier
    }

    fn specification(&self) -> &AssetSpec {
        &self.spec
    }
}

/// Futures month codes (standard across all exchanges)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FuturesMonthCode {
    January,
    February,
    March,
    April,
    May,
    June,
    July,
    August,
    September,
    October,
    November,
    December,
}

impl FuturesMonthCode {
    /// Convert to single-letter code
    pub fn to_char(self) -> char {
        match self {
            FuturesMonthCode::January => 'F',
            FuturesMonthCode::February => 'G',
            FuturesMonthCode::March => 'H',
            FuturesMonthCode::April => 'J',
            FuturesMonthCode::May => 'K',
            FuturesMonthCode::June => 'M',
            FuturesMonthCode::July => 'N',
            FuturesMonthCode::August => 'Q',
            FuturesMonthCode::September => 'U',
            FuturesMonthCode::October => 'V',
            FuturesMonthCode::November => 'X',
            FuturesMonthCode::December => 'Z',
        }
    }

    /// Parse from single-letter code
    pub fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_uppercase() {
            'F' => Some(FuturesMonthCode::January),
            'G' => Some(FuturesMonthCode::February),
            'H' => Some(FuturesMonthCode::March),
            'J' => Some(FuturesMonthCode::April),
            'K' => Some(FuturesMonthCode::May),
            'M' => Some(FuturesMonthCode::June),
            'N' => Some(FuturesMonthCode::July),
            'Q' => Some(FuturesMonthCode::August),
            'U' => Some(FuturesMonthCode::September),
            'V' => Some(FuturesMonthCode::October),
            'X' => Some(FuturesMonthCode::November),
            'Z' => Some(FuturesMonthCode::December),
            _ => None,
        }
    }

    /// Get numeric month (1-12)
    pub fn to_month(self) -> u32 {
        match self {
            FuturesMonthCode::January => 1,
            FuturesMonthCode::February => 2,
            FuturesMonthCode::March => 3,
            FuturesMonthCode::April => 4,
            FuturesMonthCode::May => 5,
            FuturesMonthCode::June => 6,
            FuturesMonthCode::July => 7,
            FuturesMonthCode::August => 8,
            FuturesMonthCode::September => 9,
            FuturesMonthCode::October => 10,
            FuturesMonthCode::November => 11,
            FuturesMonthCode::December => 12,
        }
    }

    /// Create from numeric month (1-12)
    pub fn from_month(month: u32) -> Option<Self> {
        match month {
            1 => Some(FuturesMonthCode::January),
            2 => Some(FuturesMonthCode::February),
            3 => Some(FuturesMonthCode::March),
            4 => Some(FuturesMonthCode::April),
            5 => Some(FuturesMonthCode::May),
            6 => Some(FuturesMonthCode::June),
            7 => Some(FuturesMonthCode::July),
            8 => Some(FuturesMonthCode::August),
            9 => Some(FuturesMonthCode::September),
            10 => Some(FuturesMonthCode::October),
            11 => Some(FuturesMonthCode::November),
            12 => Some(FuturesMonthCode::December),
            _ => None,
        }
    }

    /// Get next month in series
    pub fn next(self) -> Self {
        match self {
            FuturesMonthCode::January => FuturesMonthCode::February,
            FuturesMonthCode::February => FuturesMonthCode::March,
            FuturesMonthCode::March => FuturesMonthCode::April,
            FuturesMonthCode::April => FuturesMonthCode::May,
            FuturesMonthCode::May => FuturesMonthCode::June,
            FuturesMonthCode::June => FuturesMonthCode::July,
            FuturesMonthCode::July => FuturesMonthCode::August,
            FuturesMonthCode::August => FuturesMonthCode::September,
            FuturesMonthCode::September => FuturesMonthCode::October,
            FuturesMonthCode::October => FuturesMonthCode::November,
            FuturesMonthCode::November => FuturesMonthCode::December,
            FuturesMonthCode::December => FuturesMonthCode::January,
        }
    }
}

impl fmt::Display for FuturesMonthCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// Standard futures contracts
pub struct StandardFutures;

impl StandardFutures {
    /// E-mini S&P 500 (ES)
    pub fn es(month: FuturesMonthCode, year: i32, expiration: DateTime<Utc>) -> FuturesContract {
        FuturesContract::new(
            "ES",
            Exchange::CME,
            month,
            year,
            expiration,
            0.25,  // $0.25 tick size
            12.50, // $12.50 tick value
            50.0,  // $50 multiplier
        )
        .with_settlement(SettlementType::Cash)
        .with_margins(13_200.0, 12_000.0) // Approximate margins (2024)
    }

    /// E-mini Nasdaq 100 (NQ)
    pub fn nq(month: FuturesMonthCode, year: i32, expiration: DateTime<Utc>) -> FuturesContract {
        FuturesContract::new(
            "NQ",
            Exchange::CME,
            month,
            year,
            expiration,
            0.25, // $0.25 tick size
            5.00, // $5.00 tick value
            20.0, // $20 multiplier
        )
        .with_settlement(SettlementType::Cash)
        .with_margins(18_700.0, 17_000.0) // Approximate margins (2024)
    }

    /// E-mini Dow (YM)
    pub fn ym(month: FuturesMonthCode, year: i32, expiration: DateTime<Utc>) -> FuturesContract {
        FuturesContract::new(
            "YM",
            Exchange::CBOT,
            month,
            year,
            expiration,
            1.0,  // $1.00 tick size
            5.00, // $5.00 tick value
            5.0,  // $5 multiplier
        )
        .with_settlement(SettlementType::Cash)
        .with_margins(8_800.0, 8_000.0) // Approximate margins (2024)
    }

    /// Crude Oil (CL)
    pub fn cl(month: FuturesMonthCode, year: i32, expiration: DateTime<Utc>) -> FuturesContract {
        FuturesContract::new(
            "CL",
            Exchange::NYMEX,
            month,
            year,
            expiration,
            0.01,   // $0.01 tick size
            10.00,  // $10.00 tick value
            1000.0, // 1000 barrels
        )
        .with_settlement(SettlementType::Physical)
        .with_margins(5_500.0, 5_000.0) // Approximate margins (2024)
    }

    /// Gold (GC)
    pub fn gc(month: FuturesMonthCode, year: i32, expiration: DateTime<Utc>) -> FuturesContract {
        FuturesContract::new(
            "GC",
            Exchange::COMEX,
            month,
            year,
            expiration,
            0.10,  // $0.10 tick size
            10.00, // $10.00 tick value
            100.0, // 100 troy ounces
        )
        .with_settlement(SettlementType::Physical)
        .with_margins(11_000.0, 10_000.0) // Approximate margins (2024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_futures_month_codes() {
        assert_eq!(FuturesMonthCode::March.to_char(), 'H');
        assert_eq!(
            FuturesMonthCode::from_char('H'),
            Some(FuturesMonthCode::March)
        );
        assert_eq!(FuturesMonthCode::March.to_month(), 3);
        assert_eq!(
            FuturesMonthCode::from_month(3),
            Some(FuturesMonthCode::March)
        );
    }

    #[test]
    fn test_futures_month_next() {
        assert_eq!(FuturesMonthCode::March.next(), FuturesMonthCode::April);
        assert_eq!(FuturesMonthCode::December.next(), FuturesMonthCode::January);
    }

    #[test]
    fn test_es_contract() {
        let expiration = DateTime::from_timestamp(1711670400, 0).unwrap(); // March 2024
        let es = StandardFutures::es(FuturesMonthCode::March, 2024, expiration);

        assert_eq!(es.symbol(), "ESH24");
        assert_eq!(es.asset_type(), AssetType::Futures);
        assert_eq!(es.tick_size(), 0.25);
        assert_eq!(es.contract_multiplier(), 50.0);
    }

    #[test]
    fn test_futures_value_calculation() {
        let expiration = DateTime::from_timestamp(1711670400, 0).unwrap();
        let es = StandardFutures::es(FuturesMonthCode::March, 2024, expiration);

        // 1 contract at 5000.00 = 5000 * 50 = $250,000
        let value = es.calculate_value(5000.00, 1.0).unwrap();
        assert_eq!(value, 250_000.0);
    }

    #[test]
    fn test_futures_tick_pnl() {
        let expiration = DateTime::from_timestamp(1711670400, 0).unwrap();
        let es = StandardFutures::es(FuturesMonthCode::March, 2024, expiration);

        // 4 ticks profit = 4 * $12.50 = $50
        assert_eq!(es.tick_pnl(4.0), 50.0);
    }

    #[test]
    fn test_futures_margin() {
        let expiration = DateTime::from_timestamp(1711670400, 0).unwrap();
        let es = StandardFutures::es(FuturesMonthCode::March, 2024, expiration);

        let (initial, maintenance) = es.required_margin(2);
        assert_eq!(initial, 26_400.0); // 2 * 13_200
        assert_eq!(maintenance, 24_000.0); // 2 * 12_000
    }

    #[test]
    fn test_expiration_checks() {
        let expiration = DateTime::from_timestamp(1711670400, 0).unwrap();
        let es = StandardFutures::es(FuturesMonthCode::March, 2024, expiration);

        let before = DateTime::from_timestamp(1711670399, 0).unwrap();
        let after = DateTime::from_timestamp(1711670401, 0).unwrap();

        assert!(!es.is_expired(before));
        assert!(es.is_expired(after));
    }

    #[test]
    fn test_next_contract() {
        let expiration = DateTime::from_timestamp(1711670400, 0).unwrap();
        let es = StandardFutures::es(FuturesMonthCode::March, 2024, expiration);

        let (next_month, next_year) = es.next_contract();
        assert_eq!(next_month, FuturesMonthCode::April);
        assert_eq!(next_year, 2024);

        let dec = StandardFutures::es(FuturesMonthCode::December, 2024, expiration);
        let (next_month, next_year) = dec.next_contract();
        assert_eq!(next_month, FuturesMonthCode::January);
        assert_eq!(next_year, 2025);
    }
}
