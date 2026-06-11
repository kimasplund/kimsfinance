//! Python bindings for candlestick pattern recognition
//!
//! Exposes Rust-accelerated pattern recognition to Python with NumPy integration.

use crate::indicators::candlestick::{
    CandlestickPattern, PatternConfig, recognize_patterns,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Recognize candlestick patterns from OHLCV data
///
/// # Arguments
///
/// * `open` - NumPy array of open prices
/// * `high` - NumPy array of high prices
/// * `low` - NumPy array of low prices
/// * `close` - NumPy array of close prices
/// * `volume` - NumPy array of volume data
/// * `doji_threshold` - Body-to-range ratio for doji (default: 0.05)
/// * `shadow_ratio` - Shadow-to-body ratio for hammer/star (default: 2.0)
/// * `body_threshold` - Minimum body ratio for strong candles (default: 0.6)
/// * `use_volume` - Use volume confirmation (default: True)
/// * `min_confidence` - Minimum confidence to report (default: 0.5)
///
/// # Returns
///
/// List of dictionaries with pattern detections:
/// ```python
/// [
///     {
///         'pattern': 'Hammer',
///         'index': 42,
///         'confidence': 0.85,
///         'candles_used': 1,
///         'type': 'bullish'  # or 'bearish', 'neutral'
///     },
///     ...
/// ]
/// ```
///
/// # Example
///
/// ```python
/// import kimsfinance_core
/// import numpy as np
///
/// # Historical OHLCV data
/// open = np.array([100.0, 102.0, 105.0, 103.0, 107.0])
/// high = np.array([103.0, 106.0, 108.0, 106.0, 110.0])
/// low = np.array([99.0, 101.0, 104.0, 101.0, 105.0])
/// close = np.array([102.0, 105.0, 107.0, 102.0, 109.0])
/// volume = np.array([1000.0, 1500.0, 2000.0, 1200.0, 1800.0])
///
/// # Detect patterns with default settings
/// patterns = kimsfinance_core.recognize_candlestick_patterns(
///     open, high, low, close, volume
/// )
///
/// # Print detections
/// for p in patterns:
///     print(f"{p['pattern']} at index {p['index']} (confidence: {p['confidence']:.2f})")
///
/// # Custom configuration (strict)
/// patterns_strict = kimsfinance_core.recognize_candlestick_patterns(
///     open, high, low, close, volume,
///     doji_threshold=0.03,
///     shadow_ratio=2.5,
///     body_threshold=0.7,
///     min_confidence=0.7
/// )
/// ```
#[pyfunction]
#[pyo3(signature = (
    open,
    high,
    low,
    close,
    volume,
    doji_threshold=0.05,
    shadow_ratio=2.0,
    body_threshold=0.6,
    use_volume=true,
    min_confidence=0.5
))]
pub fn recognize_candlestick_patterns(
    py: Python,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
    doji_threshold: f64,
    shadow_ratio: f64,
    body_threshold: f64,
    use_volume: bool,
    min_confidence: f64,
) -> PyResult<Py<PyAny>> {
    // Convert to slices
    let open_slice = open.as_slice()?;
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;

    // Create config
    let config = PatternConfig {
        doji_body_threshold: doji_threshold,
        shadow_body_ratio: shadow_ratio,
        strong_body_threshold: body_threshold,
        engulfing_strictness: 0.0,
        use_volume,
        min_confidence,
    };

    // Recognize patterns
    let detections = recognize_patterns(
        open_slice,
        high_slice,
        low_slice,
        close_slice,
        volume_slice,
        &config,
    );

    // Convert to Python list of dicts
    let result_list = PyList::empty(py);

    for detection in detections {
        let dict = PyDict::new(py);
        dict.set_item("pattern", detection.pattern.name())?;
        dict.set_item("index", detection.index)?;
        dict.set_item("confidence", detection.confidence)?;
        dict.set_item("candles_used", detection.candles_used)?;

        // Add pattern type
        let pattern_type = if detection.pattern.is_bullish() {
            "bullish"
        } else if detection.pattern.is_bearish() {
            "bearish"
        } else {
            "neutral"
        };
        dict.set_item("type", pattern_type)?;

        result_list.append(dict)?;
    }

    Ok(result_list.into())
}

/// Get list of all available candlestick patterns
///
/// # Returns
///
/// Dictionary mapping pattern names to their types (bullish/bearish/neutral)
///
/// # Example
///
/// ```python
/// import kimsfinance_core
///
/// patterns = kimsfinance_core.get_candlestick_patterns()
/// print(f"Total patterns: {len(patterns)}")
/// print(f"Bullish patterns: {sum(1 for t in patterns.values() if t == 'bullish')}")
/// ```
#[pyfunction]
pub fn get_candlestick_patterns(py: Python) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    let all_patterns = [
        // Bullish
        CandlestickPattern::Hammer,
        CandlestickPattern::InvertedHammer,
        CandlestickPattern::BullishEngulfing,
        CandlestickPattern::PiercingLine,
        CandlestickPattern::MorningStar,
        CandlestickPattern::ThreeWhiteSoldiers,
        CandlestickPattern::WhiteMarubozu,
        CandlestickPattern::ThreeInsideUp,
        CandlestickPattern::ThreeOutsideUp,
        CandlestickPattern::BullishHarami,
        CandlestickPattern::TweezerBottom,
        CandlestickPattern::RisingThreeMethods,
        CandlestickPattern::DragonflyDoji,
        CandlestickPattern::BullishKicking,
        CandlestickPattern::ConcealingBabySwallow,
        // Bearish
        CandlestickPattern::HangingMan,
        CandlestickPattern::ShootingStar,
        CandlestickPattern::BearishEngulfing,
        CandlestickPattern::DarkCloudCover,
        CandlestickPattern::EveningStar,
        CandlestickPattern::ThreeBlackCrows,
        CandlestickPattern::BlackMarubozu,
        CandlestickPattern::ThreeInsideDown,
        CandlestickPattern::ThreeOutsideDown,
        CandlestickPattern::BearishHarami,
        CandlestickPattern::TweezerTop,
        CandlestickPattern::FallingThreeMethods,
        CandlestickPattern::GravestoneDoji,
        CandlestickPattern::BearishKicking,
        CandlestickPattern::IdenticalThreeCrows,
        // Neutral
        CandlestickPattern::Doji,
        CandlestickPattern::SpinningTop,
        CandlestickPattern::HighWave,
        CandlestickPattern::LongLeggedDoji,
        CandlestickPattern::RickshawMan,
    ];

    for pattern in all_patterns {
        let pattern_type = if pattern.is_bullish() {
            "bullish"
        } else if pattern.is_bearish() {
            "bearish"
        } else {
            "neutral"
        };
        dict.set_item(pattern.name(), pattern_type)?;
    }

    Ok(dict.into())
}

/// Batch recognize patterns for multiple securities
///
/// # Arguments
///
/// * `open_batch` - List of NumPy arrays of open prices
/// * `high_batch` - List of NumPy arrays of high prices
/// * `low_batch` - List of NumPy arrays of low prices
/// * `close_batch` - List of NumPy arrays of close prices
/// * `volume_batch` - List of NumPy arrays of volume data
/// * `config_dict` - Optional configuration dictionary
///
/// # Returns
///
/// List of pattern detection lists (one per security)
///
/// # Example
///
/// ```python
/// import kimsfinance_core
/// import numpy as np
///
/// # Multiple securities
/// opens = [np.array([100.0, 102.0]), np.array([50.0, 52.0])]
/// highs = [np.array([103.0, 105.0]), np.array([53.0, 55.0])]
/// lows = [np.array([99.0, 101.0]), np.array([49.0, 51.0])]
/// closes = [np.array([102.0, 104.0]), np.array([52.0, 54.0])]
/// volumes = [np.array([1000.0, 1500.0]), np.array([2000.0, 2500.0])]
///
/// # Batch processing
/// results = kimsfinance_core.recognize_candlestick_patterns_batch(
///     opens, highs, lows, closes, volumes
/// )
///
/// # Results is a list of pattern lists
/// for i, patterns in enumerate(results):
///     print(f"Security {i}: {len(patterns)} patterns detected")
/// ```
#[pyfunction]
#[pyo3(signature = (
    open_batch,
    high_batch,
    low_batch,
    close_batch,
    volume_batch,
    doji_threshold=0.05,
    shadow_ratio=2.0,
    body_threshold=0.6,
    use_volume=true,
    min_confidence=0.5
))]
pub fn recognize_candlestick_patterns_batch(
    py: Python,
    open_batch: Vec<PyReadonlyArray1<f64>>,
    high_batch: Vec<PyReadonlyArray1<f64>>,
    low_batch: Vec<PyReadonlyArray1<f64>>,
    close_batch: Vec<PyReadonlyArray1<f64>>,
    volume_batch: Vec<PyReadonlyArray1<f64>>,
    doji_threshold: f64,
    shadow_ratio: f64,
    body_threshold: f64,
    use_volume: bool,
    min_confidence: f64,
) -> PyResult<Py<PyAny>> {
    let n_securities = open_batch.len();

    if n_securities != high_batch.len()
        || n_securities != low_batch.len()
        || n_securities != close_batch.len()
        || n_securities != volume_batch.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All batch arrays must have the same length",
        ));
    }

    let config = PatternConfig {
        doji_body_threshold: doji_threshold,
        shadow_body_ratio: shadow_ratio,
        strong_body_threshold: body_threshold,
        engulfing_strictness: 0.0,
        use_volume,
        min_confidence,
    };

    let result_list = PyList::empty(py);

    for i in 0..n_securities {
        let open_slice = open_batch[i].as_slice()?;
        let high_slice = high_batch[i].as_slice()?;
        let low_slice = low_batch[i].as_slice()?;
        let close_slice = close_batch[i].as_slice()?;
        let volume_slice = volume_batch[i].as_slice()?;

        let detections = recognize_patterns(
            open_slice,
            high_slice,
            low_slice,
            close_slice,
            volume_slice,
            &config,
        );

        // Convert to Python list
        let security_list = PyList::empty(py);
        for detection in detections {
            let dict = PyDict::new(py);
            dict.set_item("pattern", detection.pattern.name())?;
            dict.set_item("index", detection.index)?;
            dict.set_item("confidence", detection.confidence)?;
            dict.set_item("candles_used", detection.candles_used)?;

            let pattern_type = if detection.pattern.is_bullish() {
                "bullish"
            } else if detection.pattern.is_bearish() {
                "bearish"
            } else {
                "neutral"
            };
            dict.set_item("type", pattern_type)?;

            security_list.append(dict)?;
        }

        result_list.append(security_list)?;
    }

    Ok(result_list.into())
}

/// Filter patterns by type (bullish/bearish/neutral)
///
/// # Arguments
///
/// * `patterns` - List of pattern detections from recognize_candlestick_patterns
/// * `pattern_type` - 'bullish', 'bearish', or 'neutral'
///
/// # Returns
///
/// Filtered list of patterns
///
/// # Example
///
/// ```python
/// import kimsfinance_core
///
/// all_patterns = kimsfinance_core.recognize_candlestick_patterns(
///     open, high, low, close, volume
/// )
/// bullish_only = kimsfinance_core.filter_patterns_by_type(all_patterns, 'bullish')
/// ```
#[pyfunction]
pub fn filter_patterns_by_type(
    py: Python,
    patterns: Vec<Bound<PyDict>>,
    pattern_type: &str,
) -> PyResult<Py<PyAny>> {
    let result_list = PyList::empty(py);

    for pattern_dict in patterns {
        if let Ok(ptype) = pattern_dict.get_item("type") {
            if let Some(ptype_str) = ptype.and_then(|v| v.extract::<String>().ok()) {
                if ptype_str == pattern_type {
                    result_list.append(pattern_dict)?;
                }
            }
        }
    }

    Ok(result_list.into())
}

/// Get pattern statistics summary
///
/// # Arguments
///
/// * `patterns` - List of pattern detections
///
/// # Returns
///
/// Dictionary with statistics:
/// ```python
/// {
///     'total': 42,
///     'bullish': 20,
///     'bearish': 18,
///     'neutral': 4,
///     'avg_confidence': 0.72,
///     'pattern_counts': {'Hammer': 5, 'Doji': 8, ...}
/// }
/// ```
#[pyfunction]
pub fn get_pattern_statistics(py: Python, patterns: Vec<Bound<PyDict>>) -> PyResult<Py<PyAny>> {
    let mut total = 0;
    let mut bullish = 0;
    let mut bearish = 0;
    let mut neutral = 0;
    let mut confidence_sum = 0.0;
    let mut pattern_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for pattern_dict in &patterns {
        total += 1;

        if let Ok(Some(ptype)) = pattern_dict.get_item("type") {
            if let Ok(ptype_str) = ptype.extract::<String>() {
                match ptype_str.as_str() {
                    "bullish" => bullish += 1,
                    "bearish" => bearish += 1,
                    "neutral" => neutral += 1,
                    _ => {}
                }
            }
        }

        if let Ok(Some(conf)) = pattern_dict.get_item("confidence") {
            if let Ok(conf_val) = conf.extract::<f64>() {
                confidence_sum += conf_val;
            }
        }

        if let Ok(Some(name)) = pattern_dict.get_item("pattern") {
            if let Ok(name_str) = name.extract::<String>() {
                *pattern_counts.entry(name_str).or_insert(0) += 1;
            }
        }
    }

    let dict = PyDict::new(py);
    dict.set_item("total", total)?;
    dict.set_item("bullish", bullish)?;
    dict.set_item("bearish", bearish)?;
    dict.set_item("neutral", neutral)?;
    dict.set_item(
        "avg_confidence",
        if total > 0 {
            confidence_sum / total as f64
        } else {
            0.0
        },
    )?;

    // Convert pattern_counts to Python dict
    let counts_dict = PyDict::new(py);
    for (name, count) in pattern_counts {
        counts_dict.set_item(name, count)?;
    }
    dict.set_item("pattern_counts", counts_dict)?;

    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_bindings_compile() {
        // Just ensure Python bindings compile
        // Actual Python integration tests in Python test suite
    }
}
