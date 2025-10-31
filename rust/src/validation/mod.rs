//! Data quality validation for trade data
//!
//! This module provides comprehensive validation tools for Binance trade data:
//! - Gap detection: Identify timestamp discontinuities
//! - Outlier detection: Statistical anomaly identification
//! - Checksum verification: File integrity validation
//!
//! # Example
//! ```no_run
//! use kimsfinance_core::validation::DataQualityReport;
//! use kimsfinance_core::binance::Trade;
//!
//! let trades = vec![/* ... */];
//! let report = DataQualityReport::generate(&trades);
//! report.print_summary();
//! println!("Quality score: {:.2}", report.quality_score());
//! ```

pub mod checksum;
pub mod gap_detector;
pub mod outlier_detector;

pub use checksum::{calculate_checksum, verify_checksum};
pub use gap_detector::{Gap, GapDetector};
pub use outlier_detector::{Outlier, OutlierDetector};

use crate::binance::Trade;
use std::collections::HashSet;

/// Comprehensive data quality report
///
/// Aggregates all validation results into a single report with:
/// - Gap detection
/// - Outlier detection
/// - Duplicate trade ID detection
/// - Negative quantity detection
/// - Zero price detection
#[derive(Debug, Clone)]
pub struct DataQualityReport {
    pub total_trades: usize,
    pub date_range: (i64, i64),
    pub gaps: Vec<Gap>,
    pub outliers: Vec<Outlier>,
    pub duplicate_trade_ids: Vec<u64>,
    pub negative_quantities: Vec<Trade>,
    pub zero_prices: Vec<Trade>,
}

impl DataQualityReport {
    /// Generate comprehensive quality report from trade data
    ///
    /// Runs all validators and aggregates results.
    ///
    /// # Arguments
    /// * `trades` - Slice of trades (must be sorted by timestamp)
    ///
    /// # Returns
    /// Complete data quality report
    ///
    /// # Example
    /// ```no_run
    /// # use kimsfinance_core::validation::DataQualityReport;
    /// # use kimsfinance_core::binance::Trade;
    /// let trades = vec![/* ... */];
    /// let report = DataQualityReport::generate(&trades);
    /// ```
    pub fn generate(trades: &[Trade]) -> Self {
        if trades.is_empty() {
            return Self {
                total_trades: 0,
                date_range: (0, 0),
                gaps: Vec::new(),
                outliers: Vec::new(),
                duplicate_trade_ids: Vec::new(),
                negative_quantities: Vec::new(),
                zero_prices: Vec::new(),
            };
        }

        // Find date range
        let date_range = (
            trades.first().map(|t| t.timestamp_ms).unwrap_or(0),
            trades.last().map(|t| t.timestamp_ms).unwrap_or(0),
        );

        // Gap detection (10 minutes threshold)
        let gap_detector = GapDetector::new(600_000);
        let gaps = gap_detector.find_gaps(trades);

        // Outlier detection (5 standard deviations)
        let outlier_detector = OutlierDetector::new(5.0);
        let outliers = outlier_detector.find_outliers(trades);

        // Duplicate trade IDs
        let mut seen_ids = HashSet::new();
        let duplicate_trade_ids: Vec<u64> = trades
            .iter()
            .filter_map(|trade| {
                if !seen_ids.insert(trade.trade_id) {
                    Some(trade.trade_id)
                } else {
                    None
                }
            })
            .collect();

        // Negative quantities
        let negative_quantities: Vec<Trade> = trades
            .iter()
            .filter(|t| t.quantity < 0.0)
            .cloned()
            .collect();

        // Zero prices
        let zero_prices: Vec<Trade> = trades
            .iter()
            .filter(|t| t.price == 0.0 || t.price.is_nan())
            .cloned()
            .collect();

        Self {
            total_trades: trades.len(),
            date_range,
            gaps,
            outliers,
            duplicate_trade_ids,
            negative_quantities,
            zero_prices,
        }
    }

    /// Calculate quality score (0-100)
    ///
    /// Score is calculated as:
    /// - Start at 100
    /// - Subtract 10 points per gap
    /// - Subtract 5 points per outlier
    /// - Subtract 15 points per duplicate
    /// - Subtract 20 points per negative quantity
    /// - Subtract 20 points per zero price
    /// - Minimum score: 0
    ///
    /// # Returns
    /// Quality score from 0.0 (worst) to 100.0 (perfect)
    pub fn quality_score(&self) -> f64 {
        if self.total_trades == 0 {
            return 100.0;
        }

        let mut score = 100.0;

        // Gaps penalty (10 points each, max 50 points)
        score -= (self.gaps.len() as f64 * 10.0).min(50.0);

        // Outliers penalty (5 points each, max 20 points)
        score -= (self.outliers.len() as f64 * 5.0).min(20.0);

        // Duplicates penalty (15 points each, max 30 points)
        score -= (self.duplicate_trade_ids.len() as f64 * 15.0).min(30.0);

        // Negative quantities penalty (20 points each, max 40 points)
        score -= (self.negative_quantities.len() as f64 * 20.0).min(40.0);

        // Zero prices penalty (20 points each, max 40 points)
        score -= (self.zero_prices.len() as f64 * 20.0).min(40.0);

        score.max(0.0)
    }

    /// Print human-readable summary to stdout
    ///
    /// Displays:
    /// - Total trades
    /// - Date range
    /// - Issue counts
    /// - Quality score
    pub fn print_summary(&self) {
        println!("=== Data Quality Report ===");
        println!("Total trades: {}", self.total_trades);
        println!(
            "Date range: {} to {} ({} ms)",
            self.date_range.0,
            self.date_range.1,
            self.date_range.1 - self.date_range.0
        );
        println!();
        println!("Issues found:");
        println!("  - Gaps (>10 min): {}", self.gaps.len());
        println!("  - Outliers (>5σ): {}", self.outliers.len());
        println!("  - Duplicate IDs: {}", self.duplicate_trade_ids.len());
        println!(
            "  - Negative quantities: {}",
            self.negative_quantities.len()
        );
        println!("  - Zero prices: {}", self.zero_prices.len());
        println!();
        println!("Quality score: {:.2}/100", self.quality_score());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(id: u64, price: f64, quantity: f64, timestamp_ms: i64) -> Trade {
        Trade {
            trade_id: id,
            price,
            quantity,
            quote_quantity: price * quantity,
            timestamp_ms,
            is_buyer_maker: false,
        }
    }

    #[test]
    fn test_empty_report() {
        let report = DataQualityReport::generate(&[]);
        assert_eq!(report.total_trades, 0);
        assert_eq!(report.quality_score(), 100.0);
    }

    #[test]
    fn test_perfect_data() {
        let trades = vec![
            make_trade(1, 100.0, 1.0, 1000),
            make_trade(2, 101.0, 1.0, 2000),
            make_trade(3, 102.0, 1.0, 3000),
        ];

        let report = DataQualityReport::generate(&trades);
        assert_eq!(report.total_trades, 3);
        assert_eq!(report.gaps.len(), 0);
        assert_eq!(report.outliers.len(), 0);
        assert_eq!(report.duplicate_trade_ids.len(), 0);
        assert_eq!(report.quality_score(), 100.0);
    }

    #[test]
    fn test_duplicate_detection() {
        let trades = vec![
            make_trade(1, 100.0, 1.0, 1000),
            make_trade(1, 101.0, 1.0, 2000), // Duplicate ID
            make_trade(3, 102.0, 1.0, 3000),
        ];

        let report = DataQualityReport::generate(&trades);
        assert_eq!(report.duplicate_trade_ids.len(), 1);
        assert!(report.duplicate_trade_ids.contains(&1));
        assert!(report.quality_score() < 100.0);
    }

    #[test]
    fn test_negative_quantity_detection() {
        let trades = vec![
            make_trade(1, 100.0, 1.0, 1000),
            make_trade(2, 101.0, -1.0, 2000), // Negative quantity
            make_trade(3, 102.0, 1.0, 3000),
        ];

        let report = DataQualityReport::generate(&trades);
        assert_eq!(report.negative_quantities.len(), 1);
        assert!(report.quality_score() < 100.0);
    }

    #[test]
    fn test_zero_price_detection() {
        let trades = vec![
            make_trade(1, 100.0, 1.0, 1000),
            make_trade(2, 0.0, 1.0, 2000), // Zero price
            make_trade(3, 102.0, 1.0, 3000),
        ];

        let report = DataQualityReport::generate(&trades);
        assert_eq!(report.zero_prices.len(), 1);
        assert!(report.quality_score() < 100.0);
    }
}
