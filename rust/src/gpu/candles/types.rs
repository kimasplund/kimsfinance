//! Core data structures for custom candle generation
//!
//! Defines trade data and candle output formats for GPU-accelerated aggregation.

use std::io::BufRead;

/// Trade side indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    /// Buy order (taker was buyer)
    Buy,
    /// Sell order (taker was seller)
    Sell,
    /// Unknown or unspecified
    Unknown,
}

impl TradeSide {
    /// Convert to i8 representation (1=buy, -1=sell, 0=unknown)
    #[must_use]
    pub fn to_i8(self) -> i8 {
        match self {
            Self::Buy => 1,
            Self::Sell => -1,
            Self::Unknown => 0,
        }
    }

    /// Create from i8 representation
    #[must_use]
    pub fn from_i8(value: i8) -> Self {
        match value {
            1 => Self::Buy,
            -1 => Self::Sell,
            _ => Self::Unknown,
        }
    }
}

/// Raw trade data from CSV or other sources
///
/// Contains tick-level price, volume, and timestamp information for aggregation
/// into various candle types.
///
/// # Memory Layout
///
/// Data is stored in separate vectors for efficient GPU transfer:
/// - `timestamps`: Unix timestamps (i64) in nanoseconds or milliseconds
/// - `prices`: Trade execution prices (f64)
/// - `volumes`: Trade volumes (f64)
/// - `sides`: Buy (+1), Sell (-1), or Unknown (0) indicator
/// - `symbols`: Trading pair symbols (e.g., "BTCUSDT")
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::candles::TradeData;
///
/// // Create from vectors
/// let trades = TradeData {
///     timestamps: vec![1234567890, 1234567891, 1234567892],
///     prices: vec![50000.0, 50001.0, 49999.0],
///     volumes: vec![0.5, 0.3, 0.7],
///     sides: vec![1, 1, -1],  // buy, buy, sell
///     symbols: vec!["BTCUSDT".to_string(); 3],
/// };
///
/// // Load from CSV
/// let trades = TradeData::from_csv("btc_trades.csv")?;
///
/// // Concatenate for GPU transfer
/// let buffer = trades.concat_buffers();  // [timestamps..., prices..., volumes...]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TradeData {
    /// Unix timestamps (nanoseconds or milliseconds)
    pub timestamps: Vec<i64>,
    /// Trading pair symbols (e.g., "BTCUSDT", "ETHUSDT")
    pub symbols: Vec<String>,
    /// Trade execution prices
    pub prices: Vec<f64>,
    /// Trade volumes
    pub volumes: Vec<f64>,
    /// Trade sides: 1=buy, -1=sell, 0=unknown
    pub sides: Vec<i8>,
}

impl TradeData {
    /// Create empty trade data
    #[must_use]
    pub fn new() -> Self {
        Self {
            timestamps: Vec::new(),
            symbols: Vec::new(),
            prices: Vec::new(),
            volumes: Vec::new(),
            sides: Vec::new(),
        }
    }

    /// Create trade data with pre-allocated capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            timestamps: Vec::with_capacity(capacity),
            symbols: Vec::with_capacity(capacity),
            prices: Vec::with_capacity(capacity),
            volumes: Vec::with_capacity(capacity),
            sides: Vec::with_capacity(capacity),
        }
    }

    /// Get number of trades
    #[must_use]
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Concatenate all buffers for GPU transfer
    ///
    /// Produces a single contiguous buffer: `[timestamps..., prices..., volumes...]`
    ///
    /// Note: `sides` are currently not transferred (future enhancement for volume classification)
    ///
    /// # Returns
    ///
    /// Vector of length `3 * self.len()` with concatenated data
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let trades = TradeData {
    ///     timestamps: vec![1, 2, 3],
    ///     prices: vec![100.0, 101.0, 102.0],
    ///     volumes: vec![0.5, 0.6, 0.7],
    ///     sides: vec![1, 1, -1],
    /// };
    ///
    /// let buffer = trades.concat_buffers();
    /// assert_eq!(buffer.len(), 9);  // 3 arrays × 3 elements
    /// // buffer = [1, 2, 3, 100.0, 101.0, 102.0, 0.5, 0.6, 0.7]
    /// ```
    #[must_use]
    pub fn concat_buffers(&self) -> Vec<f64> {
        let n = self.len();
        let mut buffer = Vec::with_capacity(n * 3);

        // Convert timestamps to f64 for GPU transfer
        buffer.extend(self.timestamps.iter().map(|&t| t as f64));
        buffer.extend_from_slice(&self.prices);
        buffer.extend_from_slice(&self.volumes);

        buffer
    }

    /// Load trade data from CSV file
    ///
    /// Expected CSV format:
    /// ```csv
    /// timestamp,symbol,price,volume,side
    /// 1234567890,BTC,50000.0,0.5,buy
    /// 1234567891,BTC,50001.0,0.3,buy
    /// 1234567892,BTC,49999.0,0.7,sell
    /// ```
    ///
    /// # Arguments
    ///
    /// * `path` - Path to CSV file
    ///
    /// # Returns
    ///
    /// Parsed trade data or error if file cannot be read
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - File does not exist or cannot be opened
    /// - CSV parsing fails
    /// - Invalid numeric values in columns
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let trades = TradeData::from_csv("btc_trades.csv")?;
    /// println!("Loaded {} trades", trades.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_csv(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);

        let mut trades = Self::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;

            // Skip header
            if line_num == 0 && line.starts_with("timestamp") {
                continue;
            }

            // Parse CSV line
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 5 {
                eprintln!("Warning: Skipping malformed line {}: {}", line_num + 1, line);
                continue;
            }

            // Parse fields
            let timestamp: i64 = parts[0].trim().parse()?;
            let price: f64 = parts[2].trim().parse()?;
            let volume: f64 = parts[3].trim().parse()?;
            let side_str = parts[4].trim();
            let side: i8 = match side_str {
                "buy" => 1,
                "sell" => -1,
                _ => 0,
            };

            trades.timestamps.push(timestamp);
            trades.prices.push(price);
            trades.volumes.push(volume);
            trades.sides.push(side);
        }

        Ok(trades)
    }

    /// Validate data consistency
    ///
    /// Checks that all arrays have the same length.
    ///
    /// # Returns
    ///
    /// `true` if valid, `false` if array lengths mismatch
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let n = self.timestamps.len();
        n == self.symbols.len() && n == self.prices.len() && n == self.volumes.len() && n == self.sides.len()
    }
}

impl Default for TradeData {
    fn default() -> Self {
        Self::new()
    }
}

/// OHLCV candle output format
///
/// Standard candlestick representation used in technical analysis.
///
/// # Fields
///
/// - `timestamp`: Candle start time (Unix timestamp)
/// - `open`: First trade price in the candle
/// - `high`: Highest trade price in the candle
/// - `low`: Lowest trade price in the candle
/// - `close`: Last trade price in the candle
/// - `volume`: Total volume traded during the candle
///
/// # Example
///
/// ```rust
/// use kimsfinance_core::gpu::candles::OHLCVCandle;
///
/// let candle = OHLCVCandle {
///     timestamp: 1234567890,
///     open: 50000.0,
///     high: 50100.0,
///     low: 49900.0,
///     close: 50050.0,
///     volume: 15.5,
/// };
///
/// assert_eq!(candle.range(), 200.0);  // high - low
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OHLCVCandle {
    /// Candle start time (Unix timestamp)
    pub timestamp: i64,
    /// First trade price
    pub open: f64,
    /// Highest trade price
    pub high: f64,
    /// Lowest trade price
    pub low: f64,
    /// Last trade price
    pub close: f64,
    /// Total volume traded
    pub volume: f64,
}

impl OHLCVCandle {
    /// Calculate candle price range (high - low)
    #[must_use]
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate candle body (close - open, signed)
    #[must_use]
    pub fn body(&self) -> f64 {
        self.close - self.open
    }

    /// Check if candle is bullish (close > open)
    #[must_use]
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if candle is bearish (close < open)
    #[must_use]
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Check if candle is doji (close ≈ open, within 0.1% tolerance)
    #[must_use]
    pub fn is_doji(&self) -> bool {
        let body_pct = (self.close - self.open).abs() / self.open * 100.0;
        body_pct < 0.1
    }
}

/// Helper function to concatenate OHLC buffers for GPU transfer
///
/// Produces a single contiguous buffer: `[open..., high..., low..., close...]`
///
/// # Arguments
///
/// * `open` - Open prices
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
///
/// # Returns
///
/// Vector of length `4 * open.len()` with concatenated OHLC data
///
/// # Panics
///
/// Panics if input slices have different lengths
///
/// # Example
///
/// ```rust,ignore
/// let open = vec![100.0, 101.0];
/// let high = vec![102.0, 103.0];
/// let low = vec![99.0, 100.0];
/// let close = vec![101.5, 102.5];
///
/// let buffer = concat_ohlcv(&open, &high, &low, &close);
/// assert_eq!(buffer.len(), 8);  // 4 arrays × 2 elements
/// // buffer = [100.0, 101.0, 102.0, 103.0, 99.0, 100.0, 101.5, 102.5]
/// ```
#[must_use]
pub fn concat_ohlcv(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    assert_eq!(open.len(), high.len());
    assert_eq!(open.len(), low.len());
    assert_eq!(open.len(), close.len());

    let n = open.len();
    let mut buffer = Vec::with_capacity(n * 4);

    buffer.extend_from_slice(open);
    buffer.extend_from_slice(high);
    buffer.extend_from_slice(low);
    buffer.extend_from_slice(close);

    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_data_creation() {
        let trades = TradeData::new();
        assert!(trades.is_empty());
        assert_eq!(trades.len(), 0);
    }

    #[test]
    fn test_trade_data_with_capacity() {
        let trades = TradeData::with_capacity(100);
        assert!(trades.is_empty());
        assert!(trades.timestamps.capacity() >= 100);
    }

    #[test]
    fn test_concat_buffers() {
        let trades = TradeData {
            timestamps: vec![1, 2, 3],
            symbols: vec!["BTC".to_string(), "BTC".to_string(), "BTC".to_string()],
            prices: vec![100.0, 101.0, 102.0],
            volumes: vec![0.5, 0.6, 0.7],
            sides: vec![1, 1, -1],
        };

        let buffer = trades.concat_buffers();
        assert_eq!(buffer.len(), 9);

        // Check timestamps (converted to f64)
        assert_eq!(buffer[0], 1.0);
        assert_eq!(buffer[1], 2.0);
        assert_eq!(buffer[2], 3.0);

        // Check prices
        assert_eq!(buffer[3], 100.0);
        assert_eq!(buffer[4], 101.0);
        assert_eq!(buffer[5], 102.0);

        // Check volumes
        assert_eq!(buffer[6], 0.5);
        assert_eq!(buffer[7], 0.6);
        assert_eq!(buffer[8], 0.7);
    }

    #[test]
    fn test_is_valid() {
        let valid = TradeData {
            timestamps: vec![1, 2],
            symbols: vec!["BTC".to_string(), "BTC".to_string()],
            prices: vec![100.0, 101.0],
            volumes: vec![0.5, 0.6],
            sides: vec![1, -1],
        };
        assert!(valid.is_valid());

        let invalid = TradeData {
            timestamps: vec![1, 2],
            symbols: vec!["BTC".to_string(), "BTC".to_string()],
            prices: vec![100.0], // Wrong length!
            volumes: vec![0.5, 0.6],
            sides: vec![1, -1],
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_ohlcv_candle_helpers() {
        let candle = OHLCVCandle {
            timestamp: 1234567890,
            open: 50000.0,
            high: 50100.0,
            low: 49900.0,
            close: 50050.0,
            volume: 15.5,
        };

        assert_eq!(candle.range(), 200.0);
        assert_eq!(candle.body(), 50.0);
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
        assert!(!candle.is_doji());
    }

    #[test]
    fn test_bearish_candle() {
        let candle = OHLCVCandle {
            timestamp: 1234567890,
            open: 50000.0,
            high: 50100.0,
            low: 49900.0,
            close: 49950.0, // Close below open
            volume: 15.5,
        };

        assert_eq!(candle.body(), -50.0);
        assert!(!candle.is_bullish());
        assert!(candle.is_bearish());
    }

    #[test]
    fn test_doji_candle() {
        let candle = OHLCVCandle {
            timestamp: 1234567890,
            open: 50000.0,
            high: 50100.0,
            low: 49900.0,
            close: 50003.0, // Very close to open (0.006%)
            volume: 15.5,
        };

        assert!(candle.is_doji());
    }

    #[test]
    fn test_concat_ohlcv() {
        let open = vec![100.0, 101.0];
        let high = vec![102.0, 103.0];
        let low = vec![99.0, 100.0];
        let close = vec![101.5, 102.5];

        let buffer = concat_ohlcv(&open, &high, &low, &close);
        assert_eq!(buffer.len(), 8);

        // Check concatenation order
        assert_eq!(&buffer[0..2], &[100.0, 101.0]); // open
        assert_eq!(&buffer[2..4], &[102.0, 103.0]); // high
        assert_eq!(&buffer[4..6], &[99.0, 100.0]); // low
        assert_eq!(&buffer[6..8], &[101.5, 102.5]); // close
    }

    #[test]
    #[should_panic]
    fn test_concat_ohlcv_mismatched_lengths() {
        let open = vec![100.0, 101.0];
        let high = vec![102.0]; // Wrong length!
        let low = vec![99.0, 100.0];
        let close = vec![101.5, 102.5];

        concat_ohlcv(&open, &high, &low, &close);
    }
}
