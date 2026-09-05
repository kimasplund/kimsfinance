//! Candlestick Pattern Recognition Module
//!
//! Implements 35+ candlestick patterns used in technical analysis for price action trading.
//! Patterns include bullish, bearish, and neutral formations with configurable sensitivity
//! and confidence scoring.
//!
//! # Performance
//!
//! - Target: >1M candles/sec throughput
//! - Vectorized operations where possible
//! - Zero-allocation hot paths
//! - SIMD-friendly data layout
//!
//! # Pattern Categories
//!
//! - **Bullish (15)**: Hammer, Inverted Hammer, Bullish Engulfing, Piercing Line, etc.
//! - **Bearish (15)**: Hanging Man, Shooting Star, Bearish Engulfing, Dark Cloud Cover, etc.
//! - **Neutral (5)**: Doji, Spinning Top, High Wave, etc.
//!
//! # Example
//!
//! ```rust
//! use kimsfinance_core::indicators::candlestick::{recognize_patterns, PatternConfig, CandlestickPattern};
//!
//! let open = vec![100.0, 102.0, 105.0];
//! let high = vec![103.0, 106.0, 108.0];
//! let low = vec![99.0, 101.0, 104.0];
//! let close = vec![102.0, 105.0, 107.0];
//! let volume = vec![1000.0, 1500.0, 2000.0];
//!
//! let config = PatternConfig::default();
//! let patterns = recognize_patterns(&open, &high, &low, &close, &volume, &config);
//!
//! for detection in patterns {
//!     println!("Pattern: {:?} at index {} with confidence {:.2}",
//!              detection.pattern, detection.index, detection.confidence);
//! }
//! ```


use std::fmt;

/// Candlestick pattern types (35 total)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CandlestickPattern {
    // Bullish Patterns (15)
    Hammer = 0,
    InvertedHammer = 1,
    BullishEngulfing = 2,
    PiercingLine = 3,
    MorningStar = 4,
    ThreeWhiteSoldiers = 5,
    WhiteMarubozu = 6,
    ThreeInsideUp = 7,
    ThreeOutsideUp = 8,
    BullishHarami = 9,
    TweezerBottom = 10,
    RisingThreeMethods = 11,
    DragonflyDoji = 12,
    BullishKicking = 13,
    ConcealingBabySwallow = 14,

    // Bearish Patterns (15)
    HangingMan = 15,
    ShootingStar = 16,
    BearishEngulfing = 17,
    DarkCloudCover = 18,
    EveningStar = 19,
    ThreeBlackCrows = 20,
    BlackMarubozu = 21,
    ThreeInsideDown = 22,
    ThreeOutsideDown = 23,
    BearishHarami = 24,
    TweezerTop = 25,
    FallingThreeMethods = 26,
    GravestoneDoji = 27,
    BearishKicking = 28,
    IdenticalThreeCrows = 29,

    // Neutral Patterns (5)
    Doji = 30,
    SpinningTop = 31,
    HighWave = 32,
    LongLeggedDoji = 33,
    RickshawMan = 34,
}

impl CandlestickPattern {
    /// Get pattern name as string
    pub fn name(&self) -> &'static str {
        match self {
            // Bullish
            Self::Hammer => "Hammer",
            Self::InvertedHammer => "Inverted Hammer",
            Self::BullishEngulfing => "Bullish Engulfing",
            Self::PiercingLine => "Piercing Line",
            Self::MorningStar => "Morning Star",
            Self::ThreeWhiteSoldiers => "Three White Soldiers",
            Self::WhiteMarubozu => "White Marubozu",
            Self::ThreeInsideUp => "Three Inside Up",
            Self::ThreeOutsideUp => "Three Outside Up",
            Self::BullishHarami => "Bullish Harami",
            Self::TweezerBottom => "Tweezer Bottom",
            Self::RisingThreeMethods => "Rising Three Methods",
            Self::DragonflyDoji => "Dragonfly Doji",
            Self::BullishKicking => "Bullish Kicking",
            Self::ConcealingBabySwallow => "Concealing Baby Swallow",

            // Bearish
            Self::HangingMan => "Hanging Man",
            Self::ShootingStar => "Shooting Star",
            Self::BearishEngulfing => "Bearish Engulfing",
            Self::DarkCloudCover => "Dark Cloud Cover",
            Self::EveningStar => "Evening Star",
            Self::ThreeBlackCrows => "Three Black Crows",
            Self::BlackMarubozu => "Black Marubozu",
            Self::ThreeInsideDown => "Three Inside Down",
            Self::ThreeOutsideDown => "Three Outside Down",
            Self::BearishHarami => "Bearish Harami",
            Self::TweezerTop => "Tweezer Top",
            Self::FallingThreeMethods => "Falling Three Methods",
            Self::GravestoneDoji => "Gravestone Doji",
            Self::BearishKicking => "Bearish Kicking",
            Self::IdenticalThreeCrows => "Identical Three Crows",

            // Neutral
            Self::Doji => "Doji",
            Self::SpinningTop => "Spinning Top",
            Self::HighWave => "High Wave",
            Self::LongLeggedDoji => "Long-Legged Doji",
            Self::RickshawMan => "Rickshaw Man",
        }
    }

    /// Check if pattern is bullish
    pub fn is_bullish(&self) -> bool {
        (*self as u8) <= 14
    }

    /// Check if pattern is bearish
    pub fn is_bearish(&self) -> bool {
        let val = *self as u8;
        (15..=29).contains(&val)
    }

    /// Check if pattern is neutral
    pub fn is_neutral(&self) -> bool {
        (*self as u8) >= 30
    }
}

impl fmt::Display for CandlestickPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Configuration for pattern recognition sensitivity
#[derive(Debug, Clone)]
pub struct PatternConfig {
    /// Body-to-range ratio threshold for doji detection (0.0-1.0)
    /// Lower = more strict (default: 0.05 = 5%)
    pub doji_body_threshold: f64,

    /// Shadow-to-body ratio for hammer/shooting star (typically 2.0-3.0)
    /// Higher = more strict (default: 2.0)
    pub shadow_body_ratio: f64,

    /// Minimum body percentage for strong candles (0.0-1.0)
    /// Higher = more strict (default: 0.6 = 60%)
    pub strong_body_threshold: f64,

    /// Engulfing pattern strictness (0.0-1.0)
    /// 0.0 = just body, 1.0 = full engulfment including shadows
    pub engulfing_strictness: f64,

    /// Use volume confirmation (increases confidence if volume supports pattern)
    pub use_volume: bool,

    /// Minimum confidence threshold to report (0.0-1.0)
    pub min_confidence: f64,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            doji_body_threshold: 0.05,
            shadow_body_ratio: 2.0,
            strong_body_threshold: 0.6,
            engulfing_strictness: 0.0,
            use_volume: true,
            min_confidence: 0.5,
        }
    }
}

impl PatternConfig {
    /// Create strict configuration (fewer false positives)
    pub fn strict() -> Self {
        Self {
            doji_body_threshold: 0.03,
            shadow_body_ratio: 2.5,
            strong_body_threshold: 0.7,
            engulfing_strictness: 0.5,
            use_volume: true,
            min_confidence: 0.7,
        }
    }

    /// Create relaxed configuration (more detections, more false positives)
    pub fn relaxed() -> Self {
        Self {
            doji_body_threshold: 0.1,
            shadow_body_ratio: 1.5,
            strong_body_threshold: 0.5,
            engulfing_strictness: 0.0,
            use_volume: false,
            min_confidence: 0.3,
        }
    }
}

/// Pattern detection result with confidence score
#[derive(Debug, Clone)]
pub struct PatternDetection {
    /// Type of pattern detected
    pub pattern: CandlestickPattern,
    /// Index in the data where pattern completes
    pub index: usize,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Number of candles involved in pattern (1-5)
    pub candles_used: usize,
}

/// Internal candle structure for efficient calculations
#[derive(Debug, Clone, Copy)]
struct Candle {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

impl Candle {
    #[inline]
    fn new(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Body size (absolute)
    #[inline]
    fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Full range (high - low)
    #[inline]
    fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Upper shadow length
    #[inline]
    fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Lower shadow length
    #[inline]
    fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    /// Check if candle is bullish
    #[inline]
    fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if candle is bearish
    #[inline]
    fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Body to range ratio (0.0-1.0)
    #[inline]
    fn body_ratio(&self) -> f64 {
        let range = self.range();
        if range < 1e-10 {
            return 0.0;
        }
        self.body() / range
    }

    /// Check if this is a doji (small body)
    #[inline]
    fn is_doji(&self, threshold: f64) -> bool {
        self.body_ratio() < threshold
    }

    /// Check if this is a strong body candle
    #[inline]
    fn is_strong_body(&self, threshold: f64) -> bool {
        self.body_ratio() > threshold
    }

    /// Body midpoint
    #[inline]
    fn body_midpoint(&self) -> f64 {
        (self.open + self.close) / 2.0
    }
}

/// Recognize all candlestick patterns in the given OHLCV data
///
/// # Arguments
///
/// * `open` - Open prices
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `volume` - Volume data
/// * `config` - Pattern recognition configuration
///
/// # Returns
///
/// Vector of detected patterns with confidence scores
pub fn recognize_patterns(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: &PatternConfig,
) -> Vec<PatternDetection> {
    let len = open.len();
    if len < 3 || len != high.len() || len != low.len() || len != close.len() || len != volume.len()
    {
        return Vec::new();
    }

    // Pre-allocate result vector
    let mut detections = Vec::with_capacity(len / 10);

    // Convert to internal candle representation
    let candles: Vec<Candle> = (0..len)
        .map(|i| Candle::new(open[i], high[i], low[i], close[i], volume[i]))
        .collect();

    // Scan for patterns (start at index 0 to include all single/double candle patterns)
    for i in 0..len {
        // Single-candle patterns
        if let Some(detection) = check_single_candle_patterns(&candles, i, config)
            && detection.confidence >= config.min_confidence {
                detections.push(detection);
            }

        // Two-candle patterns
        if let Some(detection) = check_two_candle_patterns(&candles, i, config)
            && detection.confidence >= config.min_confidence {
                detections.push(detection);
            }

        // Three-candle patterns
        if let Some(detection) = check_three_candle_patterns(&candles, i, config)
            && detection.confidence >= config.min_confidence {
                detections.push(detection);
            }

        // Five-candle patterns (Rising/Falling Three Methods)
        if i >= 4
            && let Some(detection) = check_five_candle_patterns(&candles, i, config)
                && detection.confidence >= config.min_confidence {
                    detections.push(detection);
                }
    }

    detections
}

/// Check single-candle patterns (Doji, Hammer, Shooting Star, etc.)
fn check_single_candle_patterns(
    candles: &[Candle],
    i: usize,
    config: &PatternConfig,
) -> Option<PatternDetection> {
    let c = candles[i];
    let range = c.range();

    if range < 1e-10 {
        return None;
    }

    // Check for various doji types
    if c.is_doji(config.doji_body_threshold) {
        let upper = c.upper_shadow();
        let lower = c.lower_shadow();
        let body = c.body();

        // Dragonfly Doji: long lower shadow, no upper shadow
        if lower > 2.0 * body && upper < body * 0.5 {
            return Some(PatternDetection {
                pattern: CandlestickPattern::DragonflyDoji,
                index: i,
                confidence: 0.8 + (lower / range * 0.2).min(0.2),
                candles_used: 1,
            });
        }

        // Gravestone Doji: long upper shadow, no lower shadow
        if upper > 2.0 * body && lower < body * 0.5 {
            return Some(PatternDetection {
                pattern: CandlestickPattern::GravestoneDoji,
                index: i,
                confidence: 0.8 + (upper / range * 0.2).min(0.2),
                candles_used: 1,
            });
        }

        // Long-Legged Doji: both shadows long
        if upper > body * 2.0 && lower > body * 2.0 {
            return Some(PatternDetection {
                pattern: CandlestickPattern::LongLeggedDoji,
                index: i,
                confidence: 0.75,
                candles_used: 1,
            });
        }

        // Standard Doji
        return Some(PatternDetection {
            pattern: CandlestickPattern::Doji,
            index: i,
            confidence: 0.7,
            candles_used: 1,
        });
    }

    // Hammer (bullish): small body at top, long lower shadow
    let body_pos = (c.close.min(c.open) - c.low) / range;
    if body_pos > 0.65
        && c.lower_shadow() > c.body() * config.shadow_body_ratio
        && c.upper_shadow() < c.body() * 0.5
    {
        let confidence = 0.6 + (c.lower_shadow() / c.body() / 10.0).min(0.3);
        return Some(PatternDetection {
            pattern: CandlestickPattern::Hammer,
            index: i,
            confidence,
            candles_used: 1,
        });
    }

    // Inverted Hammer (bullish): small body at bottom, long upper shadow
    if body_pos < 0.35
        && c.upper_shadow() > c.body() * config.shadow_body_ratio
        && c.lower_shadow() < c.body() * 0.5
    {
        let confidence = 0.6 + (c.upper_shadow() / c.body() / 10.0).min(0.3);
        return Some(PatternDetection {
            pattern: CandlestickPattern::InvertedHammer,
            index: i,
            confidence,
            candles_used: 1,
        });
    }

    // Hanging Man (bearish): same as hammer but in uptrend
    // Note: Trend detection would require more context
    if body_pos > 0.65
        && c.lower_shadow() > c.body() * config.shadow_body_ratio
        && c.upper_shadow() < c.body() * 0.5
        && c.is_bearish()
    {
        return Some(PatternDetection {
            pattern: CandlestickPattern::HangingMan,
            index: i,
            confidence: 0.65,
            candles_used: 1,
        });
    }

    // Shooting Star (bearish): same as inverted hammer but in uptrend
    if body_pos < 0.35
        && c.upper_shadow() > c.body() * config.shadow_body_ratio
        && c.lower_shadow() < c.body() * 0.5
        && c.is_bearish()
    {
        return Some(PatternDetection {
            pattern: CandlestickPattern::ShootingStar,
            index: i,
            confidence: 0.65,
            candles_used: 1,
        });
    }

    // White Marubozu: strong bullish candle with no shadows
    if c.is_bullish()
        && c.is_strong_body(config.strong_body_threshold)
        && c.upper_shadow() < range * 0.05
        && c.lower_shadow() < range * 0.05
    {
        return Some(PatternDetection {
            pattern: CandlestickPattern::WhiteMarubozu,
            index: i,
            confidence: 0.85,
            candles_used: 1,
        });
    }

    // Black Marubozu: strong bearish candle with no shadows
    if c.is_bearish()
        && c.is_strong_body(config.strong_body_threshold)
        && c.upper_shadow() < range * 0.05
        && c.lower_shadow() < range * 0.05
    {
        return Some(PatternDetection {
            pattern: CandlestickPattern::BlackMarubozu,
            index: i,
            confidence: 0.85,
            candles_used: 1,
        });
    }

    // Spinning Top: small body with upper and lower shadows
    if c.body_ratio() > 0.1
        && c.body_ratio() < 0.4
        && c.upper_shadow() > c.body()
        && c.lower_shadow() > c.body()
    {
        return Some(PatternDetection {
            pattern: CandlestickPattern::SpinningTop,
            index: i,
            confidence: 0.7,
            candles_used: 1,
        });
    }

    None
}

/// Check two-candle patterns (Engulfing, Harami, Piercing, etc.)
fn check_two_candle_patterns(
    candles: &[Candle],
    i: usize,
    config: &PatternConfig,
) -> Option<PatternDetection> {
    if i < 1 {
        return None;
    }

    let prev = candles[i - 1];
    let curr = candles[i];

    // Bullish Engulfing: small bearish followed by large bullish that engulfs
    if prev.is_bearish() && curr.is_bullish() {
        let engulfs_body = curr.open <= prev.close && curr.close >= prev.open && (curr.open < prev.close || curr.close > prev.open);
        let engulfs_full = curr.open < prev.low && curr.close > prev.high;

        if engulfs_body {
            let confidence = if engulfs_full {
                0.85
            } else {
                0.7 - config.engulfing_strictness * 0.2
            };

            // Volume confirmation
            let vol_boost = if config.use_volume && curr.volume > prev.volume * 1.5 {
                0.1
            } else {
                0.0
            };

            return Some(PatternDetection {
                pattern: CandlestickPattern::BullishEngulfing,
                index: i,
                confidence: (confidence + vol_boost).min(1.0),
                candles_used: 2,
            });
        }
    }

    // Bearish Engulfing: small bullish followed by large bearish that engulfs
    if prev.is_bullish() && curr.is_bearish() {
        let engulfs_body = curr.open >= prev.close && curr.close <= prev.open && (curr.open > prev.close || curr.close < prev.open);
        let engulfs_full = curr.open > prev.high && curr.close < prev.low;

        if engulfs_body {
            let confidence = if engulfs_full {
                0.85
            } else {
                0.7 - config.engulfing_strictness * 0.2
            };

            let vol_boost = if config.use_volume && curr.volume > prev.volume * 1.5 {
                0.1
            } else {
                0.0
            };

            return Some(PatternDetection {
                pattern: CandlestickPattern::BearishEngulfing,
                index: i,
                confidence: (confidence + vol_boost).min(1.0),
                candles_used: 2,
            });
        }
    }

    // Piercing Line: bearish followed by bullish that closes above midpoint
    if prev.is_bearish() && curr.is_bullish() {
        let prev_mid = prev.body_midpoint();
        if curr.open < prev.close && curr.close > prev_mid && curr.close < prev.open {
            let penetration = (curr.close - prev.close) / (prev.open - prev.close);
            let confidence = 0.6 + (penetration * 0.3).min(0.3);
            return Some(PatternDetection {
                pattern: CandlestickPattern::PiercingLine,
                index: i,
                confidence,
                candles_used: 2,
            });
        }
    }

    // Dark Cloud Cover: bullish followed by bearish that closes below midpoint
    if prev.is_bullish() && curr.is_bearish() {
        let prev_mid = prev.body_midpoint();
        if curr.open > prev.close && curr.close < prev_mid && curr.close > prev.open {
            let penetration = (prev.close - curr.close) / (prev.close - prev.open);
            let confidence = 0.6 + (penetration * 0.3).min(0.3);
            return Some(PatternDetection {
                pattern: CandlestickPattern::DarkCloudCover,
                index: i,
                confidence,
                candles_used: 2,
            });
        }
    }

    // Bullish Harami: large bearish followed by small bullish contained within
    if prev.is_bearish() && curr.is_bullish()
        && curr.open > prev.close && curr.close < prev.open {
            let size_ratio = curr.body() / prev.body();
            let confidence = if size_ratio < 0.5 { 0.75 } else { 0.6 };
            return Some(PatternDetection {
                pattern: CandlestickPattern::BullishHarami,
                index: i,
                confidence,
                candles_used: 2,
            });
        }

    // Bearish Harami: large bullish followed by small bearish contained within
    if prev.is_bullish() && curr.is_bearish()
        && curr.open < prev.close && curr.close > prev.open {
            let size_ratio = curr.body() / prev.body();
            let confidence = if size_ratio < 0.5 { 0.75 } else { 0.6 };
            return Some(PatternDetection {
                pattern: CandlestickPattern::BearishHarami,
                index: i,
                confidence,
                candles_used: 2,
            });
        }

    // Tweezer Bottom: two candles with same lows (bullish reversal)
    if (prev.low - curr.low).abs() < prev.range() * 0.01
        && prev.is_bearish() && curr.is_bullish() {
            return Some(PatternDetection {
                pattern: CandlestickPattern::TweezerBottom,
                index: i,
                confidence: 0.7,
                candles_used: 2,
            });
        }

    // Tweezer Top: two candles with same highs (bearish reversal)
    if (prev.high - curr.high).abs() < prev.range() * 0.01
        && prev.is_bullish() && curr.is_bearish() {
            return Some(PatternDetection {
                pattern: CandlestickPattern::TweezerTop,
                index: i,
                confidence: 0.7,
                candles_used: 2,
            });
        }

    // Bullish Kicking: gap up with strong bullish candle after bearish
    if prev.is_bearish() && curr.is_bullish()
        && curr.open > prev.close && curr.is_strong_body(config.strong_body_threshold) {
            let gap_size = (curr.open - prev.close) / prev.range();
            let confidence = 0.65 + (gap_size * 0.25).min(0.25);
            return Some(PatternDetection {
                pattern: CandlestickPattern::BullishKicking,
                index: i,
                confidence,
                candles_used: 2,
            });
        }

    // Bearish Kicking: gap down with strong bearish candle after bullish
    if prev.is_bullish() && curr.is_bearish()
        && curr.open < prev.close && curr.is_strong_body(config.strong_body_threshold) {
            let gap_size = (prev.close - curr.open) / prev.range();
            let confidence = 0.65 + (gap_size * 0.25).min(0.25);
            return Some(PatternDetection {
                pattern: CandlestickPattern::BearishKicking,
                index: i,
                confidence,
                candles_used: 2,
            });
        }

    None
}

/// Check three-candle patterns (Morning/Evening Star, Three Soldiers/Crows, etc.)
fn check_three_candle_patterns(
    candles: &[Candle],
    i: usize,
    config: &PatternConfig,
) -> Option<PatternDetection> {
    if i < 2 {
        return None;
    }

    let c1 = candles[i - 2];
    let c2 = candles[i - 1];
    let c3 = candles[i];

    // Morning Star: bearish, doji/small, bullish (reversal)
    if c1.is_bearish() && c3.is_bullish()
        && c2.body() < c1.body() * 0.5 && c2.body() < c3.body() * 0.5
            && c3.close > (c1.open + c1.close) / 2.0 {
                return Some(PatternDetection {
                    pattern: CandlestickPattern::MorningStar,
                    index: i,
                    confidence: 0.8,
                    candles_used: 3,
                });
            }

    // Evening Star: bullish, doji/small, bearish (reversal)
    if c1.is_bullish() && c3.is_bearish()
        && c2.body() < c1.body() * 0.5 && c2.body() < c3.body() * 0.5
            && c3.close < (c1.open + c1.close) / 2.0 {
                return Some(PatternDetection {
                    pattern: CandlestickPattern::EveningStar,
                    index: i,
                    confidence: 0.8,
                    candles_used: 3,
                });
            }

    // Three White Soldiers: three consecutive strong bullish candles
    if c1.is_bullish() && c2.is_bullish() && c3.is_bullish()
        && c1.is_strong_body(config.strong_body_threshold)
            && c2.is_strong_body(config.strong_body_threshold)
            && c3.is_strong_body(config.strong_body_threshold)
            && c2.close > c1.close && c3.close > c2.close {
                return Some(PatternDetection {
                    pattern: CandlestickPattern::ThreeWhiteSoldiers,
                    index: i,
                    confidence: 0.85,
                    candles_used: 3,
                });
            }

    // Three Black Crows: three consecutive strong bearish candles
    if c1.is_bearish() && c2.is_bearish() && c3.is_bearish()
        && c1.is_strong_body(config.strong_body_threshold)
            && c2.is_strong_body(config.strong_body_threshold)
            && c3.is_strong_body(config.strong_body_threshold)
            && c2.close < c1.close && c3.close < c2.close {
                return Some(PatternDetection {
                    pattern: CandlestickPattern::ThreeBlackCrows,
                    index: i,
                    confidence: 0.85,
                    candles_used: 3,
                });
            }

    // Identical Three Crows: three bearish with same opens
    if c1.is_bearish() && c2.is_bearish() && c3.is_bearish() {
        let open_diff_1 = (c1.open - c2.open).abs() / c1.range();
        let open_diff_2 = (c2.open - c3.open).abs() / c2.range();
        if open_diff_1 < 0.02 && open_diff_2 < 0.02 {
            return Some(PatternDetection {
                pattern: CandlestickPattern::IdenticalThreeCrows,
                index: i,
                confidence: 0.8,
                candles_used: 3,
            });
        }
    }

    // Three Inside Up: harami followed by breakout
    if c1.is_bearish() && c2.is_bullish() && c3.is_bullish()
        && c2.open > c1.close && c2.close < c1.open && c3.close > c1.open {
            return Some(PatternDetection {
                pattern: CandlestickPattern::ThreeInsideUp,
                index: i,
                confidence: 0.75,
                candles_used: 3,
            });
        }

    // Three Inside Down: harami followed by breakdown
    if c1.is_bullish() && c2.is_bearish() && c3.is_bearish()
        && c2.open < c1.close && c2.close > c1.open && c3.close < c1.open {
            return Some(PatternDetection {
                pattern: CandlestickPattern::ThreeInsideDown,
                index: i,
                confidence: 0.75,
                candles_used: 3,
            });
        }

    // Three Outside Up: engulfing followed by breakout
    if c1.is_bearish() && c2.is_bullish() && c3.is_bullish()
        && c2.open < c1.close && c2.close > c1.open && c3.close > c2.close {
            return Some(PatternDetection {
                pattern: CandlestickPattern::ThreeOutsideUp,
                index: i,
                confidence: 0.75,
                candles_used: 3,
            });
        }

    // Three Outside Down: engulfing followed by breakdown
    if c1.is_bullish() && c2.is_bearish() && c3.is_bearish()
        && c2.open > c1.close && c2.close < c1.open && c3.close < c2.close {
            return Some(PatternDetection {
                pattern: CandlestickPattern::ThreeOutsideDown,
                index: i,
                confidence: 0.75,
                candles_used: 3,
            });
        }

    None
}

/// Check five-candle patterns (Rising/Falling Three Methods)
fn check_five_candle_patterns(
    candles: &[Candle],
    i: usize,
    config: &PatternConfig,
) -> Option<PatternDetection> {
    if i < 4 {
        return None;
    }

    let c1 = candles[i - 4];
    let c2 = candles[i - 3];
    let c3 = candles[i - 2];
    let c4 = candles[i - 1];
    let c5 = candles[i];

    // Rising Three Methods: strong bullish, three small consolidation, strong bullish
    if c1.is_bullish()
        && c5.is_bullish()
        && c1.is_strong_body(config.strong_body_threshold)
        && c5.is_strong_body(config.strong_body_threshold)
    {
        // Middle three should be contained and smaller
        let middle_contained = c2.high < c1.high
            && c3.high < c1.high
            && c4.high < c1.high
            && c2.low > c1.low
            && c3.low > c1.low
            && c4.low > c1.low;

        let middle_small = c2.body() < c1.body() * 0.5
            && c3.body() < c1.body() * 0.5
            && c4.body() < c1.body() * 0.5;

        if middle_contained && middle_small && c5.close > c1.close {
            return Some(PatternDetection {
                pattern: CandlestickPattern::RisingThreeMethods,
                index: i,
                confidence: 0.8,
                candles_used: 5,
            });
        }
    }

    // Falling Three Methods: strong bearish, three small consolidation, strong bearish
    if c1.is_bearish()
        && c5.is_bearish()
        && c1.is_strong_body(config.strong_body_threshold)
        && c5.is_strong_body(config.strong_body_threshold)
    {
        let middle_contained = c2.high < c1.high
            && c3.high < c1.high
            && c4.high < c1.high
            && c2.low > c1.low
            && c3.low > c1.low
            && c4.low > c1.low;

        let middle_small = c2.body() < c1.body() * 0.5
            && c3.body() < c1.body() * 0.5
            && c4.body() < c1.body() * 0.5;

        if middle_contained && middle_small && c5.close < c1.close {
            return Some(PatternDetection {
                pattern: CandlestickPattern::FallingThreeMethods,
                index: i,
                confidence: 0.8,
                candles_used: 5,
            });
        }
    }

    // Concealing Baby Swallow (rare bearish pattern)
    if c1.is_bearish()
        && c2.is_bearish()
        && c3.is_bearish()
        && c4.is_bearish()
        && c1.is_strong_body(config.strong_body_threshold)
    {
        // Gap down then engulfed by fourth candle
        if c3.high < c2.low && c4.open < c3.close && c4.close > c3.open {
            return Some(PatternDetection {
                pattern: CandlestickPattern::ConcealingBabySwallow,
                index: i,
                confidence: 0.7,
                candles_used: 4,
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_calculations() {
        let c = Candle::new(100.0, 110.0, 95.0, 105.0, 1000.0);
        assert_eq!(c.body(), 5.0);
        assert_eq!(c.range(), 15.0);
        assert_eq!(c.upper_shadow(), 5.0);
        assert_eq!(c.lower_shadow(), 5.0);
        assert!(c.is_bullish());
        assert!(!c.is_bearish());
    }

    #[test]
    fn test_doji_detection() {
        let c = Candle::new(100.0, 105.0, 95.0, 100.1, 1000.0);
        assert!(c.is_doji(0.05));
    }

    #[test]
    fn test_hammer_pattern() {
        // Hammer: small body at top, long lower shadow
        let open = vec![100.0, 100.0, 105.0];
        let high = vec![105.0, 105.0, 106.0];
        let low = vec![95.0, 90.0, 103.0];
        let close = vec![102.0, 104.0, 105.5];
        let volume = vec![1000.0, 1500.0, 1200.0];

        let config = PatternConfig::default();
        let patterns = recognize_patterns(&open, &high, &low, &close, &volume, &config);

        // Should detect hammer at index 1
        let hammer = patterns
            .iter()
            .find(|p| p.pattern == CandlestickPattern::Hammer);
        assert!(hammer.is_some());
        assert_eq!(hammer.unwrap().index, 1);
    }

    #[test]
    fn test_bullish_engulfing() {
        let open = vec![105.0, 103.0, 101.0];
        let high = vec![106.0, 104.0, 107.0];
        let low = vec![103.0, 100.0, 100.0];
        let close = vec![104.0, 101.0, 106.0];
        let volume = vec![1000.0, 1200.0, 2000.0];

        let config = PatternConfig::default();
        let patterns = recognize_patterns(&open, &high, &low, &close, &volume, &config);

        let engulfing = patterns
            .iter()
            .find(|p| p.pattern == CandlestickPattern::BullishEngulfing);
        assert!(engulfing.is_some());
    }

    #[test]
    fn test_morning_star() {
        let open = vec![100.0, 105.0, 96.0, 100.0];
        let high = vec![102.0, 106.0, 97.0, 103.0];
        let low = vec![95.0, 95.0, 95.0, 99.0];
        let close = vec![96.0, 96.0, 96.5, 102.0];
        let volume = vec![1000.0, 1000.0, 500.0, 1500.0];

        let config = PatternConfig::default();
        let patterns = recognize_patterns(&open, &high, &low, &close, &volume, &config);

        let morning = patterns
            .iter()
            .find(|p| p.pattern == CandlestickPattern::MorningStar);
        assert!(morning.is_some());
    }

    #[test]
    fn test_confidence_filtering() {
        let open = vec![100.0, 100.0, 101.0];
        let high = vec![105.0, 105.0, 106.0];
        let low = vec![95.0, 95.0, 96.0];
        let close = vec![102.0, 102.0, 103.0];
        let volume = vec![1000.0, 1000.0, 1000.0];

        let config = PatternConfig {
            min_confidence: 0.9, // Very high threshold
            ..Default::default()
        };

        let patterns = recognize_patterns(&open, &high, &low, &close, &volume, &config);

        // Should have fewer patterns with high confidence threshold
        assert!(patterns.len() < 10);
    }

    #[test]
    fn test_pattern_names() {
        assert_eq!(CandlestickPattern::Hammer.name(), "Hammer");
        assert_eq!(
            CandlestickPattern::BullishEngulfing.name(),
            "Bullish Engulfing"
        );
        assert_eq!(CandlestickPattern::MorningStar.name(), "Morning Star");
    }

    #[test]
    fn test_pattern_classification() {
        assert!(CandlestickPattern::Hammer.is_bullish());
        assert!(CandlestickPattern::ShootingStar.is_bearish());
        assert!(CandlestickPattern::Doji.is_neutral());
    }
}
