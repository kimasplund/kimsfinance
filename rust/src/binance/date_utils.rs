//! Date range utilities for multi-month Binance data processing
//!
//! Provides date range parsing and generation for discovering Binance trade data files
//! across multiple months or days.
//!
//! # Features
//! - Parse date ranges from "YYYY-MM-DD" strings
//! - Generate month strings for file discovery
//! - Generate day strings for daily file patterns
//! - Robust error handling for invalid dates
//!
//! # Example
//! ```
//! use kimsfinance_core::binance::DateRange;
//!
//! let range = DateRange::parse("2021-01-01", "2021-03-31")?;
//! let months = range.months();
//! assert_eq!(months, vec!["2021-01", "2021-02", "2021-03"]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use chrono::{Datelike, NaiveDate};

/// Date range for Binance data file discovery
///
/// Represents a time range from start to end date (inclusive).
/// Used to generate month/day strings for file pattern matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DateRange {
    pub start: NaiveDate,
    pub end: NaiveDate,
}

impl DateRange {
    /// Parse date range from "YYYY-MM-DD" formatted strings
    ///
    /// # Arguments
    /// * `start` - Start date in "YYYY-MM-DD" format
    /// * `end` - End date in "YYYY-MM-DD" format
    ///
    /// # Errors
    /// Returns `ParseError` if:
    /// - Either date string is malformed
    /// - End date is before start date
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::binance::DateRange;
    /// let range = DateRange::parse("2021-01-01", "2021-12-31")?;
    /// assert_eq!(range.start.to_string(), "2021-01-01");
    /// assert_eq!(range.end.to_string(), "2021-12-31");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn parse(start: &str, end: &str) -> Result<Self, ParseError> {
        let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
            .map_err(|e| ParseError(format!("Invalid start date '{}': {}", start, e)))?;
        let end_date = NaiveDate::parse_from_str(end, "%Y-%m-%d")
            .map_err(|e| ParseError(format!("Invalid end date '{}': {}", end, e)))?;

        if end_date < start_date {
            return Err(ParseError(format!(
                "End date {} is before start date {}",
                end, start
            )));
        }

        Ok(Self {
            start: start_date,
            end: end_date,
        })
    }

    /// Generate month strings in "YYYY-MM" format
    ///
    /// Returns all months from start to end (inclusive).
    /// Useful for discovering Binance monthly trade data files.
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::binance::DateRange;
    /// let range = DateRange::parse("2021-01-15", "2021-03-20")?;
    /// let months = range.months();
    /// assert_eq!(months, vec!["2021-01", "2021-02", "2021-03"]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn months(&self) -> Vec<String> {
        let mut months = Vec::new();
        let mut current = NaiveDate::from_ymd_opt(self.start.year(), self.start.month(), 1)
            .expect("valid first day of month");

        // Get first day of end month for comparison
        let end_month =
            NaiveDate::from_ymd_opt(self.end.year(), self.end.month(), 1)
                .expect("valid first day of end month");

        while current <= end_month {
            months.push(format!("{:04}-{:02}", current.year(), current.month()));

            // Move to first day of next month
            current = if current.month() == 12 {
                NaiveDate::from_ymd_opt(current.year() + 1, 1, 1)
                    .expect("valid year increment")
            } else {
                NaiveDate::from_ymd_opt(current.year(), current.month() + 1, 1)
                    .expect("valid month increment")
            };
        }

        months
    }

    /// Generate day strings in "YYYY-MM-DD" format
    ///
    /// Returns all days from start to end (inclusive).
    /// Useful for discovering Binance daily trade data files.
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::binance::DateRange;
    /// let range = DateRange::parse("2021-01-01", "2021-01-03")?;
    /// let days = range.days();
    /// assert_eq!(days, vec!["2021-01-01", "2021-01-02", "2021-01-03"]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn days(&self) -> Vec<String> {
        let mut days = Vec::new();
        let mut current = self.start;

        while current <= self.end {
            days.push(current.format("%Y-%m-%d").to_string());
            current = current
                .succ_opt()
                .expect("date increment within valid range");
        }

        days
    }

    /// Get the number of days in this range (inclusive)
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::binance::DateRange;
    /// let range = DateRange::parse("2021-01-01", "2021-01-31")?;
    /// assert_eq!(range.num_days(), 31);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn num_days(&self) -> i64 {
        (self.end - self.start).num_days() + 1
    }

    /// Get the number of months in this range (inclusive)
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::binance::DateRange;
    /// let range = DateRange::parse("2021-01-01", "2021-03-31")?;
    /// assert_eq!(range.num_months(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn num_months(&self) -> usize {
        self.months().len()
    }
}

/// Error when parsing date ranges
#[derive(Debug, Clone)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Date parse error: {}", self.0)
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_date_range() {
        let range = DateRange::parse("2021-01-01", "2021-03-31").unwrap();
        assert_eq!(
            range.start,
            NaiveDate::from_ymd_opt(2021, 1, 1).unwrap()
        );
        assert_eq!(range.end, NaiveDate::from_ymd_opt(2021, 3, 31).unwrap());
    }

    #[test]
    fn test_parse_single_day() {
        let range = DateRange::parse("2021-01-01", "2021-01-01").unwrap();
        assert_eq!(range.start, range.end);
    }

    #[test]
    fn test_months_generation() {
        let range = DateRange::parse("2021-01-01", "2021-03-31").unwrap();
        let months = range.months();

        assert_eq!(months, vec!["2021-01", "2021-02", "2021-03"]);
    }

    #[test]
    fn test_months_partial_range() {
        // Range starting mid-month
        let range = DateRange::parse("2021-01-15", "2021-03-10").unwrap();
        let months = range.months();

        assert_eq!(months, vec!["2021-01", "2021-02", "2021-03"]);
    }

    #[test]
    fn test_months_single_month() {
        let range = DateRange::parse("2021-01-01", "2021-01-31").unwrap();
        let months = range.months();

        assert_eq!(months, vec!["2021-01"]);
    }

    #[test]
    fn test_months_year_boundary() {
        let range = DateRange::parse("2020-12-01", "2021-02-28").unwrap();
        let months = range.months();

        assert_eq!(months, vec!["2020-12", "2021-01", "2021-02"]);
    }

    #[test]
    fn test_days_generation() {
        let range = DateRange::parse("2021-01-01", "2021-01-03").unwrap();
        let days = range.days();

        assert_eq!(days, vec!["2021-01-01", "2021-01-02", "2021-01-03"]);
    }

    #[test]
    fn test_days_single_day() {
        let range = DateRange::parse("2021-01-01", "2021-01-01").unwrap();
        let days = range.days();

        assert_eq!(days, vec!["2021-01-01"]);
    }

    #[test]
    fn test_days_month_boundary() {
        let range = DateRange::parse("2021-01-30", "2021-02-02").unwrap();
        let days = range.days();

        assert_eq!(
            days,
            vec!["2021-01-30", "2021-01-31", "2021-02-01", "2021-02-02"]
        );
    }

    #[test]
    fn test_days_leap_year() {
        let range = DateRange::parse("2020-02-28", "2020-03-01").unwrap();
        let days = range.days();

        assert_eq!(
            days,
            vec!["2020-02-28", "2020-02-29", "2020-03-01"]
        );
    }

    #[test]
    fn test_invalid_date_format() {
        assert!(DateRange::parse("2021/01/01", "2021-03-31").is_err());
        assert!(DateRange::parse("2021-01-01", "2021/03/31").is_err());
        assert!(DateRange::parse("not-a-date", "2021-03-31").is_err());
    }

    #[test]
    fn test_invalid_date_range() {
        // End before start
        let result = DateRange::parse("2021-12-31", "2021-01-01");
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("before"));
    }

    #[test]
    fn test_invalid_dates() {
        // Invalid month
        assert!(DateRange::parse("2021-13-01", "2021-12-31").is_err());
        // Invalid day
        assert!(DateRange::parse("2021-02-30", "2021-12-31").is_err());
    }

    #[test]
    fn test_num_days() {
        let range = DateRange::parse("2021-01-01", "2021-01-31").unwrap();
        assert_eq!(range.num_days(), 31);

        let range = DateRange::parse("2021-01-01", "2021-01-01").unwrap();
        assert_eq!(range.num_days(), 1);

        let range = DateRange::parse("2020-02-28", "2020-03-01").unwrap();
        assert_eq!(range.num_days(), 3); // Includes leap day
    }

    #[test]
    fn test_num_months() {
        let range = DateRange::parse("2021-01-01", "2021-03-31").unwrap();
        assert_eq!(range.num_months(), 3);

        let range = DateRange::parse("2021-01-01", "2021-01-31").unwrap();
        assert_eq!(range.num_months(), 1);

        let range = DateRange::parse("2020-12-01", "2021-02-28").unwrap();
        assert_eq!(range.num_months(), 3);
    }

    #[test]
    fn test_long_range() {
        // Test a full year
        let range = DateRange::parse("2021-01-01", "2021-12-31").unwrap();
        assert_eq!(range.num_months(), 12);
        assert_eq!(range.num_days(), 365);

        let months = range.months();
        assert_eq!(months.len(), 12);
        assert_eq!(months[0], "2021-01");
        assert_eq!(months[11], "2021-12");
    }
}
