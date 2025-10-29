//! Timestamp gap detection for trade data
//!
//! Identifies discontinuities in trade data where the time between trades
//! exceeds a configured threshold (e.g., 10 minutes for 24/7 crypto markets).

use crate::binance::Trade;

/// Represents a gap in trade data
///
/// A gap is a period where no trades occurred, detected when the time
/// difference between consecutive trades exceeds the configured threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct Gap {
    /// Timestamp of the last trade before the gap (milliseconds)
    pub start_timestamp: i64,
    /// Timestamp of the first trade after the gap (milliseconds)
    pub end_timestamp: i64,
    /// Duration of the gap in milliseconds
    pub duration_ms: i64,
}

impl std::fmt::Display for Gap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gap: {} → {} ({} ms / {:.2} minutes)",
            self.start_timestamp,
            self.end_timestamp,
            self.duration_ms,
            self.duration_ms as f64 / 60_000.0
        )
    }
}

impl Gap {

    /// Convert gap duration to minutes
    ///
    /// # Returns
    /// Duration in minutes (fractional)
    pub fn duration_minutes(&self) -> f64 {
        self.duration_ms as f64 / 60_000.0
    }

    /// Convert gap duration to hours
    ///
    /// # Returns
    /// Duration in hours (fractional)
    pub fn duration_hours(&self) -> f64 {
        self.duration_ms as f64 / 3_600_000.0
    }
}

/// Gap detector for identifying data discontinuities
///
/// Scans trade data for time gaps exceeding a threshold, useful for:
/// - Detecting exchange downtime
/// - Identifying data collection failures
/// - Validating data completeness
///
/// # Example
/// ```no_run
/// # use kimsfinance_core::validation::GapDetector;
/// # use kimsfinance_core::binance::Trade;
/// let detector = GapDetector::new(600_000); // 10 minutes
/// let trades = vec![/* ... */];
/// let gaps = detector.find_gaps(&trades);
/// for gap in gaps {
///     println!("{}", gap.to_string());
/// }
/// ```
pub struct GapDetector {
    /// Maximum allowed time gap between trades (milliseconds)
    max_gap_ms: i64,
}

impl GapDetector {
    /// Create new gap detector
    ///
    /// # Arguments
    /// * `max_gap_ms` - Maximum allowed gap in milliseconds
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::validation::GapDetector;
    /// let detector = GapDetector::new(600_000); // 10 minutes
    /// ```
    pub fn new(max_gap_ms: i64) -> Self {
        Self { max_gap_ms }
    }

    /// Create detector with default threshold (10 minutes)
    ///
    /// Suitable for 24/7 cryptocurrency markets where trades should be
    /// near-continuous during active market hours.
    #[must_use]
    pub fn with_default_threshold() -> Self {
        Self::new(600_000) // 10 minutes
    }

    /// Find all gaps in trade data
    ///
    /// Scans consecutive trades and identifies gaps where the time difference
    /// exceeds the configured threshold.
    ///
    /// # Arguments
    /// * `trades` - Slice of trades (must be sorted by timestamp)
    ///
    /// # Returns
    /// Vector of detected gaps, sorted by start timestamp
    ///
    /// # Performance
    /// - Time complexity: O(n) where n is number of trades
    /// - Space complexity: O(g) where g is number of gaps found
    /// - Typical throughput: 1M+ trades/second
    ///
    /// # Example
    /// ```no_run
    /// # use kimsfinance_core::validation::GapDetector;
    /// # use kimsfinance_core::binance::Trade;
    /// let detector = GapDetector::new(600_000);
    /// let trades = vec![
    ///     Trade { timestamp_ms: 1000, ..Default::default() },
    ///     Trade { timestamp_ms: 2000, ..Default::default() },
    ///     Trade { timestamp_ms: 1_000_000, ..Default::default() }, // Big gap!
    /// ];
    /// let gaps = detector.find_gaps(&trades);
    /// assert_eq!(gaps.len(), 1);
    /// ```
    pub fn find_gaps(&self, trades: &[Trade]) -> Vec<Gap> {
        if trades.len() < 2 {
            return Vec::new();
        }

        trades
            .windows(2)
            .filter_map(|w| {
                let gap_ms = w[1].timestamp_ms - w[0].timestamp_ms;
                if gap_ms > self.max_gap_ms {
                    Some(Gap {
                        start_timestamp: w[0].timestamp_ms,
                        end_timestamp: w[1].timestamp_ms,
                        duration_ms: gap_ms,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find significant gaps (>1 hour)
    ///
    /// Convenience method for finding major data outages.
    ///
    /// # Arguments
    /// * `trades` - Slice of trades (must be sorted by timestamp)
    ///
    /// # Returns
    /// Vector of gaps longer than 1 hour
    pub fn find_significant_gaps(&self, trades: &[Trade]) -> Vec<Gap> {
        let all_gaps = self.find_gaps(trades);
        all_gaps
            .into_iter()
            .filter(|gap| gap.duration_ms > 3_600_000) // 1 hour
            .collect()
    }

    /// Calculate gap statistics
    ///
    /// # Arguments
    /// * `gaps` - Slice of gaps
    ///
    /// # Returns
    /// Tuple of (total_gap_time_ms, max_gap_ms, avg_gap_ms)
    pub fn gap_statistics(&self, gaps: &[Gap]) -> (i64, i64, f64) {
        if gaps.is_empty() {
            return (0, 0, 0.0);
        }

        let total_gap_time: i64 = gaps.iter().map(|g| g.duration_ms).sum();
        let max_gap = gaps.iter().map(|g| g.duration_ms).max().unwrap_or(0);
        let avg_gap = total_gap_time as f64 / gaps.len() as f64;

        (total_gap_time, max_gap, avg_gap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(timestamp_ms: i64) -> Trade {
        Trade {
            trade_id: 0,
            price: 100.0,
            quantity: 1.0,
            quote_quantity: 100.0,
            timestamp_ms,
            is_buyer_maker: false,
        }
    }

    #[test]
    fn test_detect_gap() {
        let trades = vec![
            make_trade(1000),
            make_trade(2000),
            make_trade(1_000_000), // Big gap!
        ];

        let detector = GapDetector::new(10_000);
        let gaps = detector.find_gaps(&trades);

        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_timestamp, 2000);
        assert_eq!(gaps[0].end_timestamp, 1_000_000);
        assert_eq!(gaps[0].duration_ms, 998_000);
    }

    #[test]
    fn test_no_gaps() {
        let trades = vec![make_trade(1000), make_trade(2000), make_trade(3000)];

        let detector = GapDetector::new(5_000);
        let gaps = detector.find_gaps(&trades);

        assert_eq!(gaps.len(), 0);
    }

    #[test]
    fn test_multiple_gaps() {
        let trades = vec![
            make_trade(1000),
            make_trade(2000),
            make_trade(100_000), // Gap 1
            make_trade(101_000),
            make_trade(200_000), // Gap 2
            make_trade(201_000),
        ];

        let detector = GapDetector::new(10_000);
        let gaps = detector.find_gaps(&trades);

        assert_eq!(gaps.len(), 2);
        assert_eq!(gaps[0].duration_ms, 98_000);
        assert_eq!(gaps[1].duration_ms, 99_000);
    }

    #[test]
    fn test_empty_trades() {
        let detector = GapDetector::new(10_000);
        let gaps = detector.find_gaps(&[]);
        assert_eq!(gaps.len(), 0);
    }

    #[test]
    fn test_single_trade() {
        let trades = vec![make_trade(1000)];
        let detector = GapDetector::new(10_000);
        let gaps = detector.find_gaps(&trades);
        assert_eq!(gaps.len(), 0);
    }

    #[test]
    fn test_gap_duration_conversions() {
        let gap = Gap {
            start_timestamp: 0,
            end_timestamp: 3_600_000,
            duration_ms: 3_600_000,
        };

        assert_eq!(gap.duration_minutes(), 60.0);
        assert_eq!(gap.duration_hours(), 1.0);
    }

    #[test]
    fn test_gap_display() {
        let gap = Gap {
            start_timestamp: 1000,
            end_timestamp: 1_000_000,
            duration_ms: 999_000,
        };
        let s = format!("{}", gap);
        assert!(s.contains("999000 ms"));
    }

    #[test]
    fn test_significant_gaps() {
        let trades = vec![
            make_trade(1000),
            make_trade(2000),
            make_trade(3_700_000), // 1+ hour gap
            make_trade(3_800_000),
            make_trade(4_000_000), // Small gap
        ];

        let detector = GapDetector::new(10_000);
        let significant = detector.find_significant_gaps(&trades);

        assert_eq!(significant.len(), 1);
        assert!(significant[0].duration_ms > 3_600_000);
    }

    #[test]
    fn test_gap_statistics() {
        let gaps = vec![
            Gap {
                start_timestamp: 0,
                end_timestamp: 100_000,
                duration_ms: 100_000,
            },
            Gap {
                start_timestamp: 200_000,
                end_timestamp: 500_000,
                duration_ms: 300_000,
            },
        ];

        let detector = GapDetector::new(10_000);
        let (total, max, avg) = detector.gap_statistics(&gaps);

        assert_eq!(total, 400_000);
        assert_eq!(max, 300_000);
        assert_eq!(avg, 200_000.0);
    }
}
