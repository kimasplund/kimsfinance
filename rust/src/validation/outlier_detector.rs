//! Statistical outlier detection for trade prices
//!
//! Identifies price anomalies using z-score analysis (standard deviations from mean).
//! Useful for detecting:
//! - Flash crashes / flash spikes
//! - Data corruption
//! - Market manipulation events

use crate::binance::Trade;

/// Represents a price outlier in trade data
///
/// An outlier is a trade whose price deviates significantly from the mean,
/// detected using z-score analysis (standard deviations from mean).
#[derive(Debug, Clone, PartialEq)]
pub struct Outlier {
    /// Index of the outlier in the original trade array
    pub index: usize,
    /// The outlier trade
    pub trade: Trade,
    /// Z-score (standard deviations from mean)
    pub z_score: f64,
}

impl std::fmt::Display for Outlier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Outlier at index {}: price={} (z-score: {:.2}σ, trade_id: {})",
            self.index, self.trade.price, self.z_score, self.trade.trade_id
        )
    }
}

impl Outlier {
    /// Check if outlier is extremely significant (>10σ)
    ///
    /// Typically indicates data corruption rather than genuine market events.
    pub fn is_extreme(&self) -> bool {
        self.z_score > 10.0
    }

    /// Check if outlier is a flash spike (unusually high)
    ///
    /// Returns true if price is more than threshold σ above mean.
    pub fn is_spike(&self, mean: f64) -> bool {
        self.trade.price > mean && self.z_score > 5.0
    }

    /// Check if outlier is a flash crash (unusually low)
    ///
    /// Returns true if price is more than threshold σ below mean.
    pub fn is_crash(&self, mean: f64) -> bool {
        self.trade.price < mean && self.z_score > 5.0
    }
}

/// Outlier detector using z-score analysis
///
/// Identifies price anomalies by calculating how many standard deviations
/// each price is from the mean. Configurable threshold (default: 5σ).
///
/// # Statistical Background
/// - Z-score = (price - mean) / std_dev
/// - Normal distribution: 99.7% of data within 3σ
/// - 5σ events: ~1 in 3.5 million occurrences
///
/// # Example
/// ```no_run
/// # use kimsfinance_core::validation::OutlierDetector;
/// # use kimsfinance_core::binance::Trade;
/// let detector = OutlierDetector::new(5.0); // 5 standard deviations
/// let trades = vec![/* ... */];
/// let outliers = detector.find_outliers(&trades);
/// for outlier in outliers {
///     println!("{}", outlier.to_string());
/// }
/// ```
pub struct OutlierDetector {
    /// Number of standard deviations to classify as outlier
    std_dev_threshold: f64,
}

impl OutlierDetector {
    /// Create new outlier detector
    ///
    /// # Arguments
    /// * `std_dev_threshold` - Number of standard deviations (e.g., 5.0)
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::validation::OutlierDetector;
    /// let detector = OutlierDetector::new(5.0); // 5σ threshold
    /// ```
    pub fn new(std_dev_threshold: f64) -> Self {
        Self { std_dev_threshold }
    }

    /// Create detector with default threshold (5σ)
    ///
    /// Suitable for financial data where 5σ events are extremely rare.
    #[must_use]
    pub fn with_default_threshold() -> Self {
        Self::new(5.0)
    }

    /// Find all outliers in trade data
    ///
    /// Calculates mean and standard deviation of prices, then identifies
    /// trades whose prices deviate by more than the configured threshold.
    ///
    /// # Arguments
    /// * `trades` - Slice of trades
    ///
    /// # Returns
    /// Vector of detected outliers, ordered by index
    ///
    /// # Performance
    /// - Time complexity: O(n) - two passes (mean, std dev)
    /// - Space complexity: O(k) where k is number of outliers
    /// - Typical throughput: 1M+ trades/second
    ///
    /// # Example
    /// ```no_run
    /// # use kimsfinance_core::validation::OutlierDetector;
    /// # use kimsfinance_core::binance::Trade;
    /// let detector = OutlierDetector::new(3.0);
    /// let mut trades = vec![];
    /// for _ in 0..100 {
    ///     trades.push(Trade { price: 100.0, ..Default::default() });
    /// }
    /// trades.push(Trade { price: 1000.0, ..Default::default() }); // Outlier!
    /// let outliers = detector.find_outliers(&trades);
    /// assert!(outliers.len() > 0);
    /// ```
    pub fn find_outliers(&self, trades: &[Trade]) -> Vec<Outlier> {
        if trades.is_empty() {
            return Vec::new();
        }

        // Calculate mean price
        let prices: Vec<f64> = trades.iter().map(|t| t.price).collect();
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;

        // Calculate standard deviation
        let variance = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();

        // Avoid division by zero for constant prices
        if std_dev == 0.0 {
            return Vec::new();
        }

        // Find outliers
        trades
            .iter()
            .enumerate()
            .filter_map(|(i, trade)| {
                let z_score = (trade.price - mean).abs() / std_dev;
                if z_score > self.std_dev_threshold {
                    Some(Outlier {
                        index: i,
                        trade: trade.clone(),
                        z_score,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find extreme outliers (>10σ)
    ///
    /// Convenience method for finding likely data corruption.
    ///
    /// # Arguments
    /// * `trades` - Slice of trades
    ///
    /// # Returns
    /// Vector of extreme outliers (>10 standard deviations)
    pub fn find_extreme_outliers(&self, trades: &[Trade]) -> Vec<Outlier> {
        let all_outliers = self.find_outliers(trades);
        all_outliers
            .into_iter()
            .filter(|outlier| outlier.z_score > 10.0)
            .collect()
    }

    /// Calculate outlier statistics
    ///
    /// # Arguments
    /// * `outliers` - Slice of outliers
    ///
    /// # Returns
    /// Tuple of (count, max_z_score, avg_z_score)
    pub fn outlier_statistics(&self, outliers: &[Outlier]) -> (usize, f64, f64) {
        if outliers.is_empty() {
            return (0, 0.0, 0.0);
        }

        let count = outliers.len();
        let max_z = outliers
            .iter()
            .map(|o| o.z_score)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        let avg_z = outliers.iter().map(|o| o.z_score).sum::<f64>() / count as f64;

        (count, max_z, avg_z)
    }

    /// Calculate mean and standard deviation
    ///
    /// Helper method for external analysis.
    ///
    /// # Arguments
    /// * `trades` - Slice of trades
    ///
    /// # Returns
    /// Tuple of (mean_price, std_dev_price)
    pub fn price_statistics(&self, trades: &[Trade]) -> (f64, f64) {
        if trades.is_empty() {
            return (0.0, 0.0);
        }

        let prices: Vec<f64> = trades.iter().map(|t| t.price).collect();
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();

        (mean, std_dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(price: f64) -> Trade {
        Trade {
            trade_id: 0,
            price,
            quantity: 1.0,
            quote_quantity: price,
            timestamp_ms: 0,
            is_buyer_maker: false,
        }
    }

    #[test]
    fn test_detect_outlier() {
        let mut trades = vec![];
        // Normal prices around 100
        for _ in 0..100 {
            trades.push(make_trade(100.0));
        }
        // One outlier at 1000
        trades.push(make_trade(1000.0));

        let detector = OutlierDetector::new(3.0);
        let outliers = detector.find_outliers(&trades);

        assert_eq!(outliers.len(), 1);
        assert_eq!(outliers[0].trade.price, 1000.0);
        assert!(outliers[0].z_score > 3.0);
    }

    #[test]
    fn test_no_outliers() {
        let trades = vec![
            make_trade(100.0),
            make_trade(101.0),
            make_trade(99.0),
            make_trade(100.5),
        ];

        let detector = OutlierDetector::new(3.0);
        let outliers = detector.find_outliers(&trades);

        assert_eq!(outliers.len(), 0);
    }

    #[test]
    fn test_empty_trades() {
        let detector = OutlierDetector::new(5.0);
        let outliers = detector.find_outliers(&[]);
        assert_eq!(outliers.len(), 0);
    }

    #[test]
    fn test_constant_prices() {
        // All same price -> std_dev = 0 -> no outliers
        let trades = vec![make_trade(100.0), make_trade(100.0), make_trade(100.0)];

        let detector = OutlierDetector::new(3.0);
        let outliers = detector.find_outliers(&trades);
        assert_eq!(outliers.len(), 0);
    }

    #[test]
    fn test_multiple_outliers() {
        let mut trades = vec![];
        // Create tightly clustered prices around 100.0 (± 1.0)
        for i in 0..50 {
            trades.push(make_trade(99.0 + (i % 2) as f64));
        }
        for i in 0..50 {
            trades.push(make_trade(100.0 + (i % 2) as f64));
        }

        // Add clear outliers: mean ~100, std dev ~0.5
        // 3σ threshold = ~1.5, so 150 and 50 are way beyond
        trades.push(make_trade(150.0)); // Outlier 1 (~100σ away)
        trades.push(make_trade(50.0)); // Outlier 2 (~100σ away)

        let detector = OutlierDetector::new(3.0);
        let outliers = detector.find_outliers(&trades);

        // Both 150 and 50 should be detected as outliers
        assert!(
            outliers.len() >= 2,
            "Expected at least 2 outliers, found {}",
            outliers.len()
        );
    }

    #[test]
    fn test_outlier_display() {
        let outlier = Outlier {
            index: 42,
            trade: make_trade(1000.0),
            z_score: 7.5,
        };

        let s = format!("{}", outlier);
        assert!(s.contains("42"));
        assert!(s.contains("1000"));
        assert!(s.contains("7.5"));
    }

    #[test]
    fn test_extreme_outliers() {
        let mut trades = vec![];
        // Create many tightly clustered prices around 100.0 (± 0.01)
        for i in 0..1000 {
            trades.push(make_trade(100.0 + (i % 2) as f64 * 0.01));
        }

        // Add extreme outlier: with 1000 points at mean ~100.0, std dev ~0.005
        // A value at 1000.0 is ~180,000σ away (extremely significant)
        trades.push(make_trade(1000.0)); // Extreme outlier

        let detector = OutlierDetector::new(3.0);
        let extreme = detector.find_extreme_outliers(&trades);

        assert!(extreme.len() > 0, "Expected at least one extreme outlier");
        if extreme.len() > 0 {
            assert!(
                extreme[0].z_score > 10.0,
                "Z-score should be >10, got {}",
                extreme[0].z_score
            );
            assert!(extreme[0].is_extreme());
        }
    }

    #[test]
    fn test_price_statistics() {
        let trades = vec![make_trade(100.0), make_trade(110.0), make_trade(90.0)];

        let detector = OutlierDetector::new(5.0);
        let (mean, std_dev) = detector.price_statistics(&trades);

        assert!((mean - 100.0).abs() < 0.01);
        assert!(std_dev > 0.0);
    }

    #[test]
    fn test_outlier_statistics() {
        let outliers = vec![
            Outlier {
                index: 0,
                trade: make_trade(1000.0),
                z_score: 5.5,
            },
            Outlier {
                index: 1,
                trade: make_trade(2000.0),
                z_score: 10.0,
            },
        ];

        let detector = OutlierDetector::new(5.0);
        let (count, max_z, avg_z) = detector.outlier_statistics(&outliers);

        assert_eq!(count, 2);
        assert_eq!(max_z, 10.0);
        assert_eq!(avg_z, 7.75);
    }

    #[test]
    fn test_spike_detection() {
        let outlier = Outlier {
            index: 0,
            trade: make_trade(1000.0),
            z_score: 7.0,
        };

        assert!(outlier.is_spike(100.0)); // Price > mean
        assert!(!outlier.is_crash(100.0)); // Price > mean, not crash
    }

    #[test]
    fn test_crash_detection() {
        let outlier = Outlier {
            index: 0,
            trade: make_trade(10.0),
            z_score: 7.0,
        };

        assert!(outlier.is_crash(100.0)); // Price < mean
        assert!(!outlier.is_spike(100.0)); // Price < mean, not spike
    }
}
