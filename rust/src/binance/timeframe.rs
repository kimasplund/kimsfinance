//! Flexible timeframe system supporting arbitrary durations
//!
//! This module provides a Duration-based timeframe system that can parse
//! any duration string ("5m", "3m", "45s", "2h") while maintaining full
//! backward compatibility with the original enum-based system.
//!
//! # Example
//! ```
//! use kimsfinance_core::binance::Timeframe;
//!
//! // Parse from string
//! let tf = Timeframe::parse("5m").unwrap();
//! assert_eq!(tf.to_ms(), 300_000);
//!
//! // Construct directly
//! let tf = Timeframe::minutes(5);
//! assert_eq!(tf.to_ms(), 300_000);
//!
//! // Backward compatibility with old enum
//! use kimsfinance_core::binance::TimeframeEnum;
//! let old_tf = TimeframeEnum::FiveMinutes;
//! let new_tf: Timeframe = old_tf.into();
//! assert_eq!(new_tf.to_ms(), 300_000);
//! ```

use std::time::Duration;

/// Original hardcoded timeframe enum (kept for backward compatibility)
///
/// Use `Timeframe` struct for new code. This enum is deprecated and will
/// be removed in a future version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[deprecated(
    since = "0.2.0",
    note = "Use Timeframe::parse() or Timeframe::minutes() instead"
)]
pub enum TimeframeEnum {
    OneMinute,
    FiveMinutes,
    FifteenMinutes,
    OneHour,
    FourHours,
    OneDay,
}

#[allow(deprecated)]
impl TimeframeEnum {
    /// Convert timeframe to milliseconds
    #[inline]
    pub const fn to_ms(&self) -> i64 {
        match self {
            TimeframeEnum::OneMinute => 60_000,
            TimeframeEnum::FiveMinutes => 300_000,
            TimeframeEnum::FifteenMinutes => 900_000,
            TimeframeEnum::OneHour => 3_600_000,
            TimeframeEnum::FourHours => 14_400_000,
            TimeframeEnum::OneDay => 86_400_000,
        }
    }
}

/// Flexible timeframe supporting any duration
///
/// This is a zero-cost abstraction over `std::time::Duration` that provides
/// convenient constructors and string parsing for trading timeframes.
///
/// # Performance
/// - Zero overhead: Same size as `Duration` (16 bytes)
/// - All conversions are `#[inline]` for optimal performance
/// - Parsing is allocation-free (except for error messages)
///
/// # Supported Units
/// - `s` or `S`: seconds (e.g., "45s", "30S")
/// - `m` or `M`: minutes (e.g., "5m", "15M")
/// - `h` or `H`: hours (e.g., "1h", "4H")
/// - `d` or `D`: days (e.g., "1d", "7D")
///
/// # Examples
/// ```
/// use kimsfinance_core::binance::Timeframe;
///
/// // String parsing
/// assert_eq!(Timeframe::parse("5m").unwrap().to_ms(), 300_000);
/// assert_eq!(Timeframe::parse("1h").unwrap().to_ms(), 3_600_000);
/// assert_eq!(Timeframe::parse("45s").unwrap().to_ms(), 45_000);
/// assert_eq!(Timeframe::parse("2d").unwrap().to_ms(), 172_800_000);
///
/// // Direct construction
/// let tf = Timeframe::minutes(5);
/// assert_eq!(tf.to_ms(), 300_000);
///
/// // Access underlying Duration
/// let duration = tf.as_duration();
/// assert_eq!(duration.as_secs(), 300);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Timeframe {
    duration: Duration,
}

impl Timeframe {
    /// Create a timeframe from seconds
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::binance::Timeframe;
    /// let tf = Timeframe::seconds(45);
    /// assert_eq!(tf.to_ms(), 45_000);
    /// ```
    #[inline]
    pub const fn seconds(s: u64) -> Self {
        Self {
            duration: Duration::from_secs(s),
        }
    }

    /// Create a timeframe from minutes
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::binance::Timeframe;
    /// let tf = Timeframe::minutes(5);
    /// assert_eq!(tf.to_ms(), 300_000);
    /// ```
    #[inline]
    pub const fn minutes(m: u64) -> Self {
        Self {
            duration: Duration::from_secs(m * 60),
        }
    }

    /// Create a timeframe from hours
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::binance::Timeframe;
    /// let tf = Timeframe::hours(1);
    /// assert_eq!(tf.to_ms(), 3_600_000);
    /// ```
    #[inline]
    pub const fn hours(h: u64) -> Self {
        Self {
            duration: Duration::from_secs(h * 3600),
        }
    }

    /// Create a timeframe from days
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::binance::Timeframe;
    /// let tf = Timeframe::days(1);
    /// assert_eq!(tf.to_ms(), 86_400_000);
    /// ```
    #[inline]
    pub const fn days(d: u64) -> Self {
        Self {
            duration: Duration::from_secs(d * 86400),
        }
    }

    /// Parse a timeframe from a string
    ///
    /// # Format
    /// The string must be in the format `<number><unit>` where:
    /// - `<number>` is a positive integer
    /// - `<unit>` is one of: `s`, `S`, `m`, `M`, `h`, `H`, `d`, `D`
    ///
    /// # Examples
    /// ```
    /// # use kimsfinance_core::binance::Timeframe;
    /// assert!(Timeframe::parse("5m").is_ok());
    /// assert!(Timeframe::parse("1h").is_ok());
    /// assert!(Timeframe::parse("45s").is_ok());
    /// assert!(Timeframe::parse("2D").is_ok());
    ///
    /// // Invalid formats
    /// assert!(Timeframe::parse("invalid").is_err());
    /// assert!(Timeframe::parse("5x").is_err());
    /// assert!(Timeframe::parse("m5").is_err());
    /// assert!(Timeframe::parse("").is_err());
    /// ```
    ///
    /// # Errors
    /// Returns `ParseError` if:
    /// - The string is empty
    /// - The format is invalid (missing number or unit)
    /// - The number is invalid or overflows
    /// - The unit is not recognized
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        if s.is_empty() {
            return Err(ParseError::EmptyString);
        }

        // Find where the number ends and unit begins
        let split_pos = s
            .chars()
            .position(|c| !c.is_ascii_digit())
            .ok_or_else(|| ParseError::MissingUnit(s.to_string()))?;

        if split_pos == 0 {
            return Err(ParseError::MissingNumber(s.to_string()));
        }

        let (number_str, unit_str) = s.split_at(split_pos);

        // Parse the number
        let number = number_str
            .parse::<u64>()
            .map_err(|_| ParseError::InvalidNumber(number_str.to_string()))?;

        if number == 0 {
            return Err(ParseError::ZeroDuration);
        }

        // Parse the unit
        let timeframe = match unit_str {
            "s" | "S" => Self::seconds(number),
            "m" | "M" => Self::minutes(number),
            "h" | "H" => Self::hours(number),
            "d" | "D" => Self::days(number),
            _ => return Err(ParseError::InvalidUnit(unit_str.to_string())),
        };

        Ok(timeframe)
    }

    /// Convert timeframe to milliseconds
    ///
    /// This is the primary method for interfacing with existing code that
    /// expects millisecond timestamps.
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::binance::Timeframe;
    /// let tf = Timeframe::minutes(5);
    /// assert_eq!(tf.to_ms(), 300_000);
    /// ```
    #[inline]
    pub const fn to_ms(&self) -> i64 {
        self.duration.as_millis() as i64
    }

    /// Get the underlying `Duration`
    ///
    /// Useful for interoperability with standard library time APIs.
    ///
    /// # Example
    /// ```
    /// # use kimsfinance_core::binance::Timeframe;
    /// let tf = Timeframe::minutes(5);
    /// let duration = tf.as_duration();
    /// assert_eq!(duration.as_secs(), 300);
    /// ```
    #[inline]
    pub const fn as_duration(&self) -> Duration {
        self.duration
    }
}

/// Error type for timeframe parsing
///
/// Provides detailed error messages for debugging invalid timeframe strings.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// The input string was empty
    EmptyString,
    /// The string was missing a number prefix
    MissingNumber(String),
    /// The string was missing a unit suffix
    MissingUnit(String),
    /// The number could not be parsed
    InvalidNumber(String),
    /// The unit was not recognized (must be s/m/h/d)
    InvalidUnit(String),
    /// The duration was zero (not allowed)
    ZeroDuration,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::EmptyString => write!(f, "Timeframe string cannot be empty"),
            ParseError::MissingNumber(s) => {
                write!(f, "Missing number in timeframe string: '{}'", s)
            }
            ParseError::MissingUnit(s) => {
                write!(f, "Missing unit in timeframe string: '{}'", s)
            }
            ParseError::InvalidNumber(s) => write!(f, "Invalid number: '{}'", s),
            ParseError::InvalidUnit(s) => {
                write!(f, "Invalid unit: '{}' (must be one of: s, m, h, d)", s)
            }
            ParseError::ZeroDuration => write!(f, "Timeframe duration cannot be zero"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Backward compatibility: Convert old enum to new flexible timeframe
///
/// This allows existing code using `TimeframeEnum` to work unchanged
/// with the new `Timeframe` system.
///
/// # Example
/// ```
/// # use kimsfinance_core::binance::{Timeframe, TimeframeEnum};
/// # #[allow(deprecated)]
/// let old_tf = TimeframeEnum::FiveMinutes;
/// let new_tf: Timeframe = old_tf.into();
/// assert_eq!(new_tf.to_ms(), 300_000);
/// ```
#[allow(deprecated)]
impl From<TimeframeEnum> for Timeframe {
    #[inline]
    fn from(tf: TimeframeEnum) -> Self {
        match tf {
            TimeframeEnum::OneMinute => Timeframe::minutes(1),
            TimeframeEnum::FiveMinutes => Timeframe::minutes(5),
            TimeframeEnum::FifteenMinutes => Timeframe::minutes(15),
            TimeframeEnum::OneHour => Timeframe::hours(1),
            TimeframeEnum::FourHours => Timeframe::hours(4),
            TimeframeEnum::OneDay => Timeframe::days(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Parsing Tests =====

    #[test]
    fn test_parse_minutes() {
        assert_eq!(Timeframe::parse("1m").unwrap().to_ms(), 60_000);
        assert_eq!(Timeframe::parse("5m").unwrap().to_ms(), 300_000);
        assert_eq!(Timeframe::parse("15m").unwrap().to_ms(), 900_000);
        assert_eq!(Timeframe::parse("7m").unwrap().to_ms(), 420_000);
        assert_eq!(Timeframe::parse("33m").unwrap().to_ms(), 1_980_000);
    }

    #[test]
    fn test_parse_hours() {
        assert_eq!(Timeframe::parse("1h").unwrap().to_ms(), 3_600_000);
        assert_eq!(Timeframe::parse("4h").unwrap().to_ms(), 14_400_000);
        assert_eq!(Timeframe::parse("2h").unwrap().to_ms(), 7_200_000);
        assert_eq!(Timeframe::parse("24h").unwrap().to_ms(), 86_400_000);
    }

    #[test]
    fn test_parse_seconds() {
        assert_eq!(Timeframe::parse("1s").unwrap().to_ms(), 1_000);
        assert_eq!(Timeframe::parse("30s").unwrap().to_ms(), 30_000);
        assert_eq!(Timeframe::parse("45s").unwrap().to_ms(), 45_000);
        assert_eq!(Timeframe::parse("90s").unwrap().to_ms(), 90_000);
    }

    #[test]
    fn test_parse_days() {
        assert_eq!(Timeframe::parse("1d").unwrap().to_ms(), 86_400_000);
        assert_eq!(Timeframe::parse("2d").unwrap().to_ms(), 172_800_000);
        assert_eq!(Timeframe::parse("7d").unwrap().to_ms(), 604_800_000);
    }

    #[test]
    fn test_parse_case_insensitive() {
        // Lowercase
        assert_eq!(Timeframe::parse("5m").unwrap().to_ms(), 300_000);
        assert_eq!(Timeframe::parse("1h").unwrap().to_ms(), 3_600_000);

        // Uppercase
        assert_eq!(Timeframe::parse("5M").unwrap().to_ms(), 300_000);
        assert_eq!(Timeframe::parse("1H").unwrap().to_ms(), 3_600_000);
    }

    #[test]
    fn test_parse_invalid_empty() {
        let result = Timeframe::parse("");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ParseError::EmptyString);
    }

    #[test]
    fn test_parse_invalid_unit() {
        assert!(Timeframe::parse("5x").is_err());
        assert!(Timeframe::parse("5y").is_err());
        assert!(Timeframe::parse("5w").is_err());

        let result = Timeframe::parse("5x");
        assert!(matches!(result, Err(ParseError::InvalidUnit(_))));
    }

    #[test]
    fn test_parse_invalid_number() {
        assert!(Timeframe::parse("invalid").is_err());
        assert!(Timeframe::parse("m5").is_err());

        let result = Timeframe::parse("m5");
        assert!(matches!(result, Err(ParseError::MissingNumber(_))));
    }

    #[test]
    fn test_parse_missing_unit() {
        let result = Timeframe::parse("123");
        assert!(result.is_err());
        assert!(matches!(result, Err(ParseError::MissingUnit(_))));
    }

    #[test]
    fn test_parse_zero_duration() {
        let result = Timeframe::parse("0m");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ParseError::ZeroDuration);
    }

    #[test]
    fn test_parse_large_numbers() {
        // Should handle large but valid durations
        assert_eq!(Timeframe::parse("1000m").unwrap().to_ms(), 60_000_000);
        assert_eq!(Timeframe::parse("100h").unwrap().to_ms(), 360_000_000);
    }

    // ===== Constructor Tests =====

    #[test]
    fn test_constructor_seconds() {
        let tf = Timeframe::seconds(45);
        assert_eq!(tf.to_ms(), 45_000);
        assert_eq!(tf.as_duration().as_secs(), 45);
    }

    #[test]
    fn test_constructor_minutes() {
        let tf = Timeframe::minutes(5);
        assert_eq!(tf.to_ms(), 300_000);
        assert_eq!(tf.as_duration().as_secs(), 300);
    }

    #[test]
    fn test_constructor_hours() {
        let tf = Timeframe::hours(1);
        assert_eq!(tf.to_ms(), 3_600_000);
        assert_eq!(tf.as_duration().as_secs(), 3600);
    }

    #[test]
    fn test_constructor_days() {
        let tf = Timeframe::days(1);
        assert_eq!(tf.to_ms(), 86_400_000);
        assert_eq!(tf.as_duration().as_secs(), 86400);
    }

    // ===== Backward Compatibility Tests =====

    #[test]
    #[allow(deprecated)]
    fn test_backward_compat_one_minute() {
        let old = TimeframeEnum::OneMinute;
        let new: Timeframe = old.into();
        assert_eq!(new.to_ms(), 60_000);
        assert_eq!(new.to_ms(), old.to_ms());
    }

    #[test]
    #[allow(deprecated)]
    fn test_backward_compat_five_minutes() {
        let old = TimeframeEnum::FiveMinutes;
        let new: Timeframe = old.into();
        assert_eq!(new.to_ms(), 300_000);
        assert_eq!(new.to_ms(), old.to_ms());
    }

    #[test]
    #[allow(deprecated)]
    fn test_backward_compat_fifteen_minutes() {
        let old = TimeframeEnum::FifteenMinutes;
        let new: Timeframe = old.into();
        assert_eq!(new.to_ms(), 900_000);
        assert_eq!(new.to_ms(), old.to_ms());
    }

    #[test]
    #[allow(deprecated)]
    fn test_backward_compat_one_hour() {
        let old = TimeframeEnum::OneHour;
        let new: Timeframe = old.into();
        assert_eq!(new.to_ms(), 3_600_000);
        assert_eq!(new.to_ms(), old.to_ms());
    }

    #[test]
    #[allow(deprecated)]
    fn test_backward_compat_four_hours() {
        let old = TimeframeEnum::FourHours;
        let new: Timeframe = old.into();
        assert_eq!(new.to_ms(), 14_400_000);
        assert_eq!(new.to_ms(), old.to_ms());
    }

    #[test]
    #[allow(deprecated)]
    fn test_backward_compat_one_day() {
        let old = TimeframeEnum::OneDay;
        let new: Timeframe = old.into();
        assert_eq!(new.to_ms(), 86_400_000);
        assert_eq!(new.to_ms(), old.to_ms());
    }

    // ===== Equivalence Tests =====

    #[test]
    fn test_parse_equals_constructor() {
        assert_eq!(Timeframe::parse("5m").unwrap(), Timeframe::minutes(5));
        assert_eq!(Timeframe::parse("1h").unwrap(), Timeframe::hours(1));
        assert_eq!(Timeframe::parse("45s").unwrap(), Timeframe::seconds(45));
        assert_eq!(Timeframe::parse("2d").unwrap(), Timeframe::days(2));
    }

    #[test]
    fn test_equality() {
        let tf1 = Timeframe::minutes(5);
        let tf2 = Timeframe::parse("5m").unwrap();
        assert_eq!(tf1, tf2);

        let tf3 = Timeframe::seconds(300);
        assert_eq!(tf1, tf3);
    }

    // ===== Error Display Tests =====

    #[test]
    fn test_error_display() {
        let err = ParseError::EmptyString;
        assert_eq!(err.to_string(), "Timeframe string cannot be empty");

        let err = ParseError::InvalidUnit("x".to_string());
        assert!(err.to_string().contains("Invalid unit"));
        assert!(err.to_string().contains("x"));

        let err = ParseError::ZeroDuration;
        assert!(err.to_string().contains("cannot be zero"));
    }

    // ===== Hash and Copy Tests =====

    #[test]
    fn test_hash() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        let tf = Timeframe::minutes(5);
        map.insert(tf, "5m candles");

        assert_eq!(map.get(&Timeframe::minutes(5)), Some(&"5m candles"));
    }

    #[test]
    fn test_copy_semantics() {
        let tf1 = Timeframe::minutes(5);
        let tf2 = tf1; // Copy, not move
        assert_eq!(tf1, tf2);
        assert_eq!(tf1.to_ms(), tf2.to_ms());
    }

    // ===== Duration Conversions =====

    #[test]
    fn test_duration_conversion() {
        let tf = Timeframe::minutes(5);
        let duration = tf.as_duration();
        assert_eq!(duration.as_secs(), 300);
        assert_eq!(duration.as_millis(), 300_000);
    }

    #[test]
    fn test_complex_durations() {
        // 90 minutes = 1.5 hours
        let tf = Timeframe::minutes(90);
        assert_eq!(tf.to_ms(), 5_400_000);

        // 7200 seconds = 2 hours = 120 minutes
        let tf = Timeframe::seconds(7200);
        assert_eq!(tf.to_ms(), 7_200_000);
    }
}
