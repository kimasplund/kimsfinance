/// Shared types for kimsfinance coordinate calculations
///
/// This module defines the core types used across the Rust implementation,
/// optimized for zero-copy interop with Python NumPy arrays.
use ndarray::{Array1, ArrayView1};

/// Coordinate calculation result containing all pre-computed coordinates
/// for candlestick chart rendering.
///
/// All arrays are i32 for direct compatibility with PIL drawing operations.
/// This struct is designed to minimize allocations and enable zero-copy
/// transfer to Python when possible.
#[derive(Debug, Clone)]
pub struct CandlestickCoordinates {
    /// X coordinate of candle start (left edge)
    pub x_start: Array1<i32>,

    /// X coordinate of candle end (right edge)
    pub x_end: Array1<i32>,

    /// X coordinate of candle center (for wick)
    pub x_center: Array1<i32>,

    /// Y coordinate of high price
    pub y_high: Array1<i32>,

    /// Y coordinate of low price
    pub y_low: Array1<i32>,

    /// Y coordinate of open price
    pub y_open: Array1<i32>,

    /// Y coordinate of close price
    pub y_close: Array1<i32>,

    /// Volume bar heights
    pub vol_heights: Array1<i32>,

    /// Y coordinate of candle body top
    pub body_top: Array1<i32>,

    /// Y coordinate of candle body bottom
    pub body_bottom: Array1<i32>,

    /// Boolean array indicating bullish (true) vs bearish (false) candles
    pub is_bullish: Array1<bool>,
}

/// Chart rendering parameters
///
/// Contains all the geometric parameters needed for coordinate calculations.
/// These match the Python implementation exactly for compatibility.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ChartParams {
    /// Number of candles to render
    pub num_candles: usize,

    /// Width of each candle in pixels (including spacing)
    pub candle_width: f64,

    /// Spacing between candles in pixels
    pub spacing: f64,

    /// Width of candle body in pixels
    pub bar_width: f64,

    /// Minimum price value for scaling
    pub price_min: f64,

    /// Price range (max - min) for scaling
    pub price_range: f64,

    /// Maximum volume value for scaling
    pub volume_range: f64,

    /// Height of chart area in pixels
    pub chart_height: i32,

    /// Height of volume area in pixels
    pub volume_height: i32,

    /// Total image height in pixels
    pub height: i32,
}

impl ChartParams {
    /// Create new chart parameters with validation
    ///
    /// # Panics
    /// Panics if price_range or volume_range is zero (would cause division by zero)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_candles: usize,
        candle_width: f64,
        spacing: f64,
        bar_width: f64,
        price_min: f64,
        price_range: f64,
        volume_range: f64,
        chart_height: i32,
        volume_height: i32,
        height: i32,
    ) -> Self {
        assert!(price_range > 0.0, "price_range must be positive");
        assert!(volume_range > 0.0, "volume_range must be positive");

        Self {
            num_candles,
            candle_width,
            spacing,
            bar_width,
            price_min,
            price_range,
            volume_range,
            chart_height,
            volume_height,
            height,
        }
    }
}

/// Input price and volume data
///
/// Provides a zero-copy view into NumPy arrays from Python.
/// Uses ArrayView1 to avoid copying data across the FFI boundary.
#[derive(Debug)]
pub struct OHLCVData<'a> {
    pub high_prices: ArrayView1<'a, f64>,
    pub low_prices: ArrayView1<'a, f64>,
    pub open_prices: ArrayView1<'a, f64>,
    pub close_prices: ArrayView1<'a, f64>,
    pub volume_data: ArrayView1<'a, f64>,
}

impl<'a> OHLCVData<'a> {
    /// Create new OHLCV data with validation
    ///
    /// # Panics
    /// Panics if arrays have different lengths
    pub fn new(
        high_prices: ArrayView1<'a, f64>,
        low_prices: ArrayView1<'a, f64>,
        open_prices: ArrayView1<'a, f64>,
        close_prices: ArrayView1<'a, f64>,
        volume_data: ArrayView1<'a, f64>,
    ) -> Self {
        let len = high_prices.len();
        assert_eq!(low_prices.len(), len, "low_prices length mismatch");
        assert_eq!(open_prices.len(), len, "open_prices length mismatch");
        assert_eq!(close_prices.len(), len, "close_prices length mismatch");
        assert_eq!(volume_data.len(), len, "volume_data length mismatch");

        Self {
            high_prices,
            low_prices,
            open_prices,
            close_prices,
            volume_data,
        }
    }

    /// Get the number of candles
    #[inline]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.high_prices.len()
    }

    /// Check if the data is empty
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.high_prices.is_empty()
    }
}
