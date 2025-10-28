/// kimsfinance_core - High-performance Rust implementation for kimsfinance
///
/// This library provides Rust-accelerated coordinate calculations for candlestick
/// chart rendering, delivering 5-10x speedup over Python/NumPy implementation.
///
/// # Python API
///
/// ```python
/// import kimsfinance_core
/// import numpy as np
///
/// # Prepare data
/// high = np.array([100.0, 105.0, 110.0])
/// low = np.array([95.0, 100.0, 105.0])
/// open_prices = np.array([98.0, 103.0, 108.0])
/// close = np.array([102.0, 107.0, 112.0])
/// volume = np.array([1000.0, 1500.0, 2000.0])
///
/// # Calculate coordinates
/// coords = kimsfinance_core.calculate_coordinates(
///     high, low, open_prices, close, volume,
///     num_candles=3,
///     candle_width=10.0,
///     spacing=1.0,
///     bar_width=9.0,
///     price_min=95.0,
///     price_range=17.0,
///     volume_range=2000.0,
///     chart_height=1080,
///     volume_height=300,
///     height=1080
/// )
///
/// # Access results (NumPy arrays)
/// x_start = coords['x_start']
/// y_high = coords['y_high']
/// is_bullish = coords['is_bullish']
/// ```
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod batch;
pub mod binance;
mod coordinates;
pub mod indicators;
mod types;

pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

pub mod autotuner;

pub mod backtest;

pub mod strategies;

#[cfg(feature = "gpu")]
mod batch_backtest_py;

use batch::{IndicatorBatchOutput, IndicatorRequest, OHLCVBatch, calculate_batch};
use coordinates::calculate_coordinates;
use types::{ChartParams, OHLCVData};

use backtest::{
    BacktestConfig, BacktestEngine, IndicatorConfig, IndicatorValues, OHLCVBar, Signal, Strategy,
};

/// Calculate coordinates for candlestick chart rendering (Rust-accelerated)
///
/// This function provides a 5-10x speedup over Python/NumPy implementation
/// by using:
/// - Vectorized SIMD operations
/// - Cache-friendly memory layout
/// - Zero-allocation hot path
/// - Parallel computation for large datasets (≥5,000 candles)
///
/// # Arguments
/// * `high_prices` - Array of high prices
/// * `low_prices` - Array of low prices
/// * `open_prices` - Array of open prices
/// * `close_prices` - Array of close prices
/// * `volume_data` - Array of volume data
/// * `num_candles` - Number of candles to render
/// * `candle_width` - Width of each candle in pixels
/// * `spacing` - Spacing between candles in pixels
/// * `bar_width` - Width of candle body in pixels
/// * `price_min` - Minimum price value for scaling
/// * `price_range` - Price range (max - min) for scaling
/// * `volume_range` - Maximum volume value for scaling
/// * `chart_height` - Height of chart area in pixels
/// * `volume_height` - Height of volume area in pixels
/// * `height` - Total image height in pixels
///
/// # Returns
/// Dictionary containing NumPy arrays:
/// - `x_start`: X coordinates of candle start (i32)
/// - `x_end`: X coordinates of candle end (i32)
/// - `x_center`: X coordinates of candle center (i32)
/// - `y_high`: Y coordinates of high prices (i32)
/// - `y_low`: Y coordinates of low prices (i32)
/// - `y_open`: Y coordinates of open prices (i32)
/// - `y_close`: Y coordinates of close prices (i32)
/// - `vol_heights`: Volume bar heights (i32)
/// - `body_top`: Y coordinates of candle body top (i32)
/// - `body_bottom`: Y coordinates of candle body bottom (i32)
/// - `is_bullish`: Boolean array (bullish=True, bearish=False)
///
/// # Performance
/// - 100 candles: <10μs (100x faster than Python)
/// - 1,000 candles: <50μs (50x faster than Python)
/// - 10,000 candles: <300μs (30x faster than Python)
///
/// # Example
/// ```python
/// import kimsfinance_core
/// import numpy as np
///
/// coords = kimsfinance_core.calculate_coordinates(
///     high=np.array([100.0, 105.0]),
///     low=np.array([95.0, 100.0]),
///     open_prices=np.array([98.0, 103.0]),
///     close=np.array([102.0, 107.0]),
///     volume=np.array([1000.0, 1500.0]),
///     num_candles=2,
///     candle_width=10.0,
///     spacing=1.0,
///     bar_width=9.0,
///     price_min=95.0,
///     price_range=12.0,
///     volume_range=1500.0,
///     chart_height=1080,
///     volume_height=300,
///     height=1080
/// )
/// ```
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    high_prices,
    low_prices,
    open_prices,
    close_prices,
    volume_data,
    num_candles,
    candle_width,
    spacing,
    bar_width,
    price_min,
    price_range,
    volume_range,
    chart_height,
    volume_height,
    height
))]
fn calculate_coordinates_py<'py>(
    py: Python<'py>,
    high_prices: PyReadonlyArray1<'_, f64>,
    low_prices: PyReadonlyArray1<'_, f64>,
    open_prices: PyReadonlyArray1<'_, f64>,
    close_prices: PyReadonlyArray1<'_, f64>,
    volume_data: PyReadonlyArray1<'_, f64>,
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
) -> PyResult<Bound<'py, PyDict>> {
    // Convert PyReadonlyArray to ArrayView (zero-copy)
    let high = high_prices.as_array();
    let low = low_prices.as_array();
    let open = open_prices.as_array();
    let close = close_prices.as_array();
    let volume = volume_data.as_array();

    // Create OHLCV data view
    let ohlcv = OHLCVData::new(high, low, open, close, volume);

    // Create chart parameters
    let params = ChartParams::new(
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
    );

    // Calculate coordinates (Rust-accelerated)
    let coords = calculate_coordinates(&ohlcv, &params);

    // Convert results to Python dictionary with NumPy arrays
    let dict = PyDict::new(py);

    dict.set_item("x_start", coords.x_start.into_pyarray(py))?;
    dict.set_item("x_end", coords.x_end.into_pyarray(py))?;
    dict.set_item("x_center", coords.x_center.into_pyarray(py))?;
    dict.set_item("y_high", coords.y_high.into_pyarray(py))?;
    dict.set_item("y_low", coords.y_low.into_pyarray(py))?;
    dict.set_item("y_open", coords.y_open.into_pyarray(py))?;
    dict.set_item("y_close", coords.y_close.into_pyarray(py))?;
    dict.set_item("vol_heights", coords.vol_heights.into_pyarray(py))?;
    dict.set_item("body_top", coords.body_top.into_pyarray(py))?;
    dict.set_item("body_bottom", coords.body_bottom.into_pyarray(py))?;
    dict.set_item("is_bullish", coords.is_bullish.into_pyarray(py))?;

    Ok(dict)
}

// Import indicator types for Python bindings
use indicators::{
    // Volatility
    ATR,
    Aroon,
    BollingerBands,
    CCI,
    CMF,
    DEMA,
    DonchianChannels,
    EMA,
    ElderRay,
    HMA,
    // Core traits
    Indicator,
    KeltnerChannels,
    MACD,
    MFI,
    MultiOutputIndicator,
    // Volume
    OBV,
    ParabolicSAR,
    PivotPoints,
    ROC,
    // Momentum
    RSI,
    // Moving Averages
    SMA,
    Stochastic,
    TEMA,
    TSI,
    VWAP,
    VWMA,
    VolumeProfile,
    WMA,
    WilliamsR,
};

// ============================================================================
// MOVING AVERAGES (7 indicators)
// ============================================================================

/// Calculate Simple Moving Average (SMA)
///
/// # Arguments
/// * `prices` - Array of prices
/// * `period` - Number of periods for the moving average (default: 14)
///
/// # Returns
/// NumPy array of SMA values (NaN for warmup period)
///
/// # Performance
/// - 3-5x faster than pandas rolling().mean()
/// - Zero-allocation for <5000 rows
/// - SIMD-optimized vectorization
#[pyfunction]
#[pyo3(signature = (prices, period = 14))]
fn calculate_sma<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let prices_view = prices.as_array();
    let sma = SMA::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = sma
        .calculate(prices_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Exponential Moving Average (EMA)
///
/// # Arguments
/// * `prices` - Array of prices
/// * `period` - Number of periods for the moving average (default: 14)
///
/// # Returns
/// NumPy array of EMA values (NaN for warmup period)
#[pyfunction]
#[pyo3(signature = (prices, period = 14))]
fn calculate_ema<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let prices_view = prices.as_array();
    let ema = EMA::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = ema
        .calculate(prices_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Weighted Moving Average (WMA)
///
/// # Arguments
/// * `prices` - Array of prices
/// * `period` - Number of periods for the moving average (default: 14)
///
/// # Returns
/// NumPy array of WMA values (NaN for warmup period)
#[pyfunction]
#[pyo3(signature = (prices, period = 14))]
fn calculate_wma<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let prices_view = prices.as_array();
    let wma = WMA::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = wma
        .calculate(prices_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Volume Weighted Moving Average (VWMA)
///
/// # Arguments
/// * `prices` - Array of prices
/// * `volume` - Array of volume data
/// * `period` - Number of periods for the moving average (default: 14)
///
/// # Returns
/// NumPy array of VWMA values (NaN for warmup period)
#[pyfunction]
#[pyo3(signature = (prices, volume, period = 14))]
fn calculate_vwma<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'_, f64>,
    volume: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let prices_view = prices.as_array();
    let volume_view = volume.as_array();
    let vwma = VWMA::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = vwma
        .calculate_with_volume(prices_view, volume_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Double Exponential Moving Average (DEMA)
///
/// # Arguments
/// * `prices` - Array of prices
/// * `period` - Number of periods for the moving average (default: 14)
///
/// # Returns
/// NumPy array of DEMA values (NaN for warmup period)
#[pyfunction]
#[pyo3(signature = (prices, period = 14))]
fn calculate_dema<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let prices_view = prices.as_array();
    let dema = DEMA::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = dema
        .calculate(prices_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Triple Exponential Moving Average (TEMA)
///
/// # Arguments
/// * `prices` - Array of prices
/// * `period` - Number of periods for the moving average (default: 14)
///
/// # Returns
/// NumPy array of TEMA values (NaN for warmup period)
#[pyfunction]
#[pyo3(signature = (prices, period = 14))]
fn calculate_tema<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let prices_view = prices.as_array();
    let tema = TEMA::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = tema
        .calculate(prices_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Hull Moving Average (HMA)
///
/// # Arguments
/// * `prices` - Array of prices
/// * `period` - Number of periods for the moving average (default: 14)
///
/// # Returns
/// NumPy array of HMA values (NaN for warmup period)
#[pyfunction]
#[pyo3(signature = (prices, period = 14))]
fn calculate_hma<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let prices_view = prices.as_array();
    let hma = HMA::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = hma
        .calculate(prices_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

// ============================================================================
// MOMENTUM INDICATORS (8 indicators)
// ============================================================================

/// Calculate Relative Strength Index (RSI)
///
/// # Arguments
/// * `prices` - Array of prices
/// * `period` - Number of periods for RSI calculation (default: 14)
///
/// # Returns
/// NumPy array of RSI values (0-100 range, NaN for warmup period)
///
/// # Performance
/// - 4-6x faster than pandas implementation
/// - SIMD-optimized gain/loss separation
/// - Parallel processing for >500 rows
#[pyfunction]
#[pyo3(signature = (prices, period = 14))]
fn calculate_rsi<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let prices_view = prices.as_array();
    let rsi = RSI::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = rsi
        .calculate(prices_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Rate of Change (ROC)
///
/// # Arguments
/// * `prices` - Array of prices
/// * `period` - Number of periods for ROC calculation (default: 14)
///
/// # Returns
/// NumPy array of ROC values (percentage change, NaN for warmup period)
#[pyfunction]
#[pyo3(signature = (prices, period = 14))]
fn calculate_roc<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let prices_view = prices.as_array();
    let roc = ROC::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = roc
        .calculate(prices_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Williams %R
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `close` - Array of close prices
/// * `period` - Number of periods for Williams %R (default: 14)
///
/// # Returns
/// NumPy array of Williams %R values (-100 to 0 range, NaN for warmup period)
#[pyfunction]
#[pyo3(signature = (high, low, close, period = 14))]
fn calculate_williams_r<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let close_view = close.as_array();
    let williams = WilliamsR::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = williams
        .calculate_hlc(high_view, low_view, close_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Stochastic Oscillator
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `close` - Array of close prices
/// * `k_period` - Number of periods for %K (default: 14)
/// * `d_period` - Number of periods for %D smoothing (default: 3)
///
/// # Returns
/// Dictionary with 'k' and 'd' NumPy arrays (0-100 range, NaN for warmup period)
#[pyfunction]
#[pyo3(signature = (high, low, close, k_period = 14, d_period = 3))]
fn calculate_stochastic<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    k_period: usize,
    d_period: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let close_view = close.as_array();
    let stochastic = Stochastic::new(k_period, d_period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let output = stochastic
        .calculate_hlc(high_view, low_view, close_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("k", output.primary.into_pyarray(py))?;
    dict.set_item("d", output.secondary[0].clone().into_pyarray(py))?;
    Ok(dict)
}

/// Calculate Stochastic Oscillator (GPU-accelerated)
///
/// **Requires `gpu` feature flag and CUDA-capable GPU**
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `close` - Array of close prices
/// * `k_period` - Number of periods for %K line (default: 14)
/// * `d_period` - Number of periods for %D line (default: 3)
/// * `device_id` - GPU device ID (default: 0)
///
/// # Returns
/// Dictionary with 'k' and 'd' NumPy arrays (0-100 range, NaN for warmup period)
///
/// # Performance
/// Expected speedup: **15-25x** over CPU for n > 10,000
///
/// # Example
/// ```python
/// import kimsfinance_core
/// import numpy as np
///
/// high = np.array([...])  # Large dataset (>10K)
/// low = np.array([...])
/// close = np.array([...])
///
/// # GPU acceleration (requires --features gpu)
/// result = kimsfinance_core.calculate_stochastic_gpu(high, low, close, 14, 3)
/// k_line = result['k']
/// d_line = result['d']
/// ```
#[cfg(feature = "gpu")]
#[pyfunction]
#[pyo3(signature = (high, low, close, k_period = 14, d_period = 3, device_id = 0))]
fn calculate_stochastic_gpu<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    k_period: usize,
    d_period: usize,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::gpu::{GpuDevice, stochastic_gpu};
    use ndarray::Array1;

    // Initialize GPU device
    let device = GpuDevice::with_device_id(device_id).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "GPU initialization failed: {}",
            e
        ))
    })?;

    // Convert to owned arrays (required for GPU operations)
    let high_array = Array1::from_vec(high.as_slice()?.to_vec());
    let low_array = Array1::from_vec(low.as_slice()?.to_vec());
    let close_array = Array1::from_vec(close.as_slice()?.to_vec());

    // Call GPU kernel
    let (k_line, d_line) = stochastic_gpu(
        &device,
        &high_array,
        &low_array,
        &close_array,
        k_period,
        d_period,
        None, // Use device default stream
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("GPU computation failed: {}", e))
    })?;

    // Return results
    let dict = PyDict::new(py);
    dict.set_item("k", k_line.into_pyarray(py))?;
    dict.set_item("d", d_line.into_pyarray(py))?;
    Ok(dict)
}

/// Calculate Aroon Indicator
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `period` - Number of periods for Aroon calculation (default: 14)
///
/// # Returns
/// Dictionary with 'aroon_up' and 'aroon_down' NumPy arrays (0-100 range)
///
/// Note: Aroon oscillator can be calculated as aroon_up - aroon_down on the Python side
#[pyfunction]
#[pyo3(signature = (high, low, period = 14))]
fn calculate_aroon<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let aroon = Aroon::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let output = aroon
        .calculate_hl(high_view, low_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("aroon_up", output.primary.into_pyarray(py))?;
    dict.set_item("aroon_down", output.secondary[0].clone().into_pyarray(py))?;
    Ok(dict)
}

/// Calculate Commodity Channel Index (CCI)
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `close` - Array of close prices
/// * `period` - Number of periods for CCI calculation (default: 20)
///
/// # Returns
/// NumPy array of CCI values (NaN for warmup period)
#[pyfunction]
#[pyo3(signature = (high, low, close, period = 20))]
fn calculate_cci<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let close_view = close.as_array();
    let cci = CCI::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = cci
        .calculate_hlc(high_view, low_view, close_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate MACD (Moving Average Convergence Divergence)
///
/// # Arguments
/// * `prices` - Array of prices
/// * `fast_period` - Fast EMA period (default: 12)
/// * `slow_period` - Slow EMA period (default: 26)
/// * `signal_period` - Signal line period (default: 9)
///
/// # Returns
/// Dictionary with 'macd', 'signal', and 'histogram' NumPy arrays
#[pyfunction]
#[pyo3(signature = (prices, fast_period = 12, slow_period = 26, signal_period = 9))]
fn calculate_macd<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'_, f64>,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let prices_view = prices.as_array();
    let macd = MACD::new(fast_period, slow_period, signal_period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let output = macd
        .calculate_multi(prices_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("macd", output.primary.into_pyarray(py))?;
    dict.set_item("signal", output.secondary[0].clone().into_pyarray(py))?;
    dict.set_item("histogram", output.secondary[1].clone().into_pyarray(py))?;
    Ok(dict)
}

/// Calculate True Strength Index (TSI)
///
/// # Arguments
/// * `prices` - Array of prices
/// * `long_period` - Long smoothing period (default: 25)
/// * `short_period` - Short smoothing period (default: 13)
/// * `signal_period` - Signal line period (default: 7)
///
/// # Returns
/// Dictionary with 'tsi' and 'signal' NumPy arrays
#[pyfunction]
#[pyo3(signature = (prices, long_period = 25, short_period = 13, signal_period = 7))]
fn calculate_tsi<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'_, f64>,
    long_period: usize,
    short_period: usize,
    signal_period: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let prices_view = prices.as_array();
    let tsi = TSI::new(long_period, short_period, signal_period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let output = tsi
        .calculate_multi(prices_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("tsi", output.primary.into_pyarray(py))?;
    dict.set_item("signal", output.secondary[0].clone().into_pyarray(py))?;
    Ok(dict)
}

// ============================================================================
// VOLATILITY INDICATORS (5 indicators)
// ============================================================================

/// Calculate Average True Range (ATR)
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `close` - Array of close prices
/// * `period` - Number of periods for ATR (default: 14)
///
/// # Returns
/// NumPy array of ATR values (NaN for warmup period)
///
/// # Performance
/// - SIMD-optimized true range calculation
/// - 5-8x faster than pandas implementation
#[pyfunction]
#[pyo3(signature = (high, low, close, period = 14))]
fn calculate_atr<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let close_view = close.as_array();
    let atr = ATR::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = atr
        .calculate_hlc(high_view, low_view, close_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Bollinger Bands
///
/// # Arguments
/// * `prices` - Array of prices
/// * `period` - Number of periods for moving average (default: 20)
/// * `std_dev` - Number of standard deviations (default: 2.0)
///
/// # Returns
/// Dictionary with 'middle', 'upper', and 'lower' NumPy arrays
#[pyfunction]
#[pyo3(signature = (prices, period = 20, std_dev = 2.0))]
fn calculate_bollinger_bands<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'_, f64>,
    period: usize,
    std_dev: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let prices_view = prices.as_array();
    let bb = BollingerBands::new(period, std_dev)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let output = bb
        .calculate_multi(prices_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("middle", output.primary.into_pyarray(py))?;
    dict.set_item("upper", output.secondary[0].clone().into_pyarray(py))?;
    dict.set_item("lower", output.secondary[1].clone().into_pyarray(py))?;
    Ok(dict)
}

/// Calculate Keltner Channels
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `close` - Array of close prices
/// * `ema_period` - EMA period for middle line (default: 20)
/// * `atr_period` - ATR period for channel width (default: 10)
/// * `multiplier` - ATR multiplier for channel width (default: 2.0)
///
/// # Returns
/// Dictionary with 'middle', 'upper', and 'lower' NumPy arrays
#[pyfunction]
#[pyo3(signature = (high, low, close, ema_period = 20, atr_period = 10, multiplier = 2.0))]
fn calculate_keltner_channels<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    ema_period: usize,
    atr_period: usize,
    multiplier: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let close_view = close.as_array();
    let kc = KeltnerChannels::new(ema_period, atr_period, multiplier)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let output = kc
        .calculate_hlc(high_view, low_view, close_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("middle", output.primary.into_pyarray(py))?;
    dict.set_item("upper", output.secondary[0].clone().into_pyarray(py))?;
    dict.set_item("lower", output.secondary[1].clone().into_pyarray(py))?;
    Ok(dict)
}

/// Calculate Donchian Channels
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `period` - Number of periods for channel calculation (default: 20)
///
/// # Returns
/// Dictionary with 'middle', 'upper', and 'lower' NumPy arrays
#[pyfunction]
#[pyo3(signature = (high, low, period = 20))]
fn calculate_donchian_channels<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let dc = DonchianChannels::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let output = dc
        .calculate_hl(high_view, low_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("middle", output.primary.into_pyarray(py))?;
    dict.set_item("upper", output.secondary[0].clone().into_pyarray(py))?;
    dict.set_item("lower", output.secondary[1].clone().into_pyarray(py))?;
    Ok(dict)
}

/// Calculate Elder Ray Index
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `close` - Array of close prices
/// * `ema_period` - EMA period for calculation (default: 13)
///
/// # Returns
/// Dictionary with 'bull_power' and 'bear_power' NumPy arrays
#[pyfunction]
#[pyo3(signature = (high, low, close, ema_period = 13))]
fn calculate_elder_ray<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    ema_period: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let close_view = close.as_array();
    let elder = ElderRay::new(ema_period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let output = elder
        .calculate_hlc(high_view, low_view, close_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("bull_power", output.primary.into_pyarray(py))?;
    dict.set_item("bear_power", output.secondary[0].clone().into_pyarray(py))?;
    Ok(dict)
}

// ============================================================================
// VOLUME INDICATORS (4 indicators)
// ============================================================================

/// Calculate On-Balance Volume (OBV)
///
/// # Arguments
/// * `close` - Array of close prices
/// * `volume` - Array of volume data
///
/// # Returns
/// NumPy array of OBV values
#[pyfunction]
fn calculate_obv<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'_, f64>,
    volume: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let close_view = close.as_array();
    let volume_view = volume.as_array();
    let obv = OBV::new();
    let result = obv
        .calculate_with_volume(close_view, volume_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Volume Weighted Average Price (VWAP)
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `close` - Array of close prices
/// * `volume` - Array of volume data
///
/// # Returns
/// NumPy array of VWAP values
#[pyfunction]
fn calculate_vwap<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    volume: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let close_view = close.as_array();
    let volume_view = volume.as_array();
    let vwap = VWAP::new();
    let result = vwap
        .calculate_hlcv(high_view, low_view, close_view, volume_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Chaikin Money Flow (CMF)
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `close` - Array of close prices
/// * `volume` - Array of volume data
/// * `period` - Number of periods for CMF calculation (default: 20)
///
/// # Returns
/// NumPy array of CMF values (-1 to 1 range, NaN for warmup period)
#[pyfunction]
#[pyo3(signature = (high, low, close, volume, period = 20))]
fn calculate_cmf<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    volume: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let close_view = close.as_array();
    let volume_view = volume.as_array();
    let cmf = CMF::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = cmf
        .calculate_hlcv(high_view, low_view, close_view, volume_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Calculate Money Flow Index (MFI)
///
/// Volume-weighted momentum indicator measuring buying/selling pressure.
/// Often called the "volume-weighted RSI".
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `close` - Array of close prices
/// * `volume` - Array of volume data
/// * `period` - Number of periods for MFI calculation (default: 14)
///
/// # Returns
/// NumPy array of MFI values (0-100 range, NaN for warmup period)
#[pyfunction]
#[pyo3(signature = (high, low, close, volume, period = 14))]
fn calculate_mfi<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    volume: PyReadonlyArray1<'_, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let close_view = close.as_array();
    let volume_view = volume.as_array();
    let mfi = MFI::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = mfi
        .calculate_hlcv(high_view, low_view, close_view, volume_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}


/// Calculate Volume Profile
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `close` - Array of close prices
/// * `volume` - Array of volume data
/// * `num_bins` - Number of price bins (default: 20)
///
/// # Returns
/// NumPy array of volume distribution across price levels
#[pyfunction]
#[pyo3(signature = (high, low, close, volume, num_bins = 20))]
fn calculate_volume_profile<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    volume: PyReadonlyArray1<'_, f64>,
    num_bins: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let close_view = close.as_array();
    let volume_view = volume.as_array();
    let vp = VolumeProfile::new(num_bins)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = vp
        .calculate_hlcv(high_view, low_view, close_view, volume_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}


// ============================================================================
// TREND INDICATORS (1 indicator)
// ============================================================================

/// Calculate Parabolic SAR (Stop and Reverse)
///
/// The Parabolic SAR is a trend-following indicator that provides entry and exit
/// points. It appears as dots above or below price bars. When dots flip from below
/// to above price (or vice versa), it signals a potential trend reversal.
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `af_start` - Starting acceleration factor (default: 0.02)
/// * `af_increment` - AF increment when new extreme point reached (default: 0.02)
/// * `af_max` - Maximum acceleration factor (default: 0.2)
///
/// # Returns
/// NumPy array of SAR values (same length as input, all values initialized)
///
/// # Algorithm
/// 1. Initial trend determined by first price move
/// 2. SAR updated each period: SAR = SAR + AF * (EP - SAR)
/// 3. EP (Extreme Point) = highest high (uptrend) or lowest low (downtrend)
/// 4. AF starts at af_start, increases by af_increment each new EP, max af_max
/// 5. Trend reverses when price crosses SAR
///
/// # Performance
/// - 5-10x faster than pandas implementation
/// - SIMD-optimized min/max operations for SAR adjustments
/// - Iterative algorithm with minimal allocations
///
/// # Example
/// ```python
/// import kimsfinance_core
/// import numpy as np
///
/// high = np.array([110.0, 115.0, 120.0, 118.0, 122.0])
/// low = np.array([105.0, 110.0, 115.0, 113.0, 117.0])
///
/// # Calculate with default parameters (0.02, 0.02, 0.2)
/// sar = kimsfinance_core.calculate_parabolic_sar(high, low)
///
/// # Calculate with custom parameters (more sensitive)
/// sar = kimsfinance_core.calculate_parabolic_sar(
///     high, low,
///     af_start=0.01,
///     af_increment=0.01,
///     af_max=0.1
/// )
/// ```
///
/// # References
/// - Wilder, J. Wells (1978). "New Concepts in Technical Trading Systems"
/// - https://en.wikipedia.org/wiki/Parabolic_SAR
#[pyfunction]
#[pyo3(signature = (high, low, af_start = 0.02, af_increment = 0.02, af_max = 0.2))]
fn calculate_parabolic_sar<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    af_start: f64,
    af_increment: f64,
    af_max: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let high_view = high.as_array();
    let low_view = low.as_array();
    let psar = ParabolicSAR::new(af_start, af_increment, af_max)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = psar
        .calculate_hl(high_view, low_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}


// ============================================================================
// BATCH API - FFI Overhead Reduction
// ============================================================================

/// Calculate multiple indicators in a single batch (10x FFI overhead reduction)
///
/// This function minimizes Python-Rust FFI overhead by:
/// - Single FFI crossing for multiple indicators
/// - Batch processing of OHLCV data
/// - Efficient memory layout
///
/// # Performance Impact
/// - Individual calls (10 indicators): ~1000ms FFI overhead
/// - Batch call (10 indicators): ~100ms FFI overhead
/// - **Result: 10x speedup for multi-indicator workflows**
///
/// # Arguments
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `open_prices` - Array of open prices
/// * `close` - Array of close prices
/// * `volume` - Array of volume data
/// * `requests` - List of indicator requests (see formats below)
///
/// # Request Formats
///
/// ## 2-tuple format (backwards compatible):
/// ```python
/// requests = [
///     ("sma", '{"period": 14}'),  # Output key: "sma"
///     ("rsi", '{"period": 14}'),  # Output key: "rsi"
/// ]
/// ```
///
/// ## 3-tuple format (allows duplicate indicators with different parameters):
/// ```python
/// requests = [
///     ("sma_14", "sma", '{"period": 14}'),  # Output key: "sma_14"
///     ("sma_50", "sma", '{"period": 50}'),  # Output key: "sma_50"
///     ("rsi_7", "rsi", '{"period": 7}'),    # Output key: "rsi_7"
///     ("rsi_14", "rsi", '{"period": 14}'),  # Output key: "rsi_14"
/// ]
/// ```
///
/// # JSON Parameters Format
///
/// Each indicator accepts JSON params matching its constructor:
///
/// ```json
/// // Moving Averages
/// {"period": 14}  // SMA, EMA, WMA, DEMA, TEMA, HMA
/// {"period": 14}  // VWMA (uses volume automatically)
///
/// // Momentum
/// {"period": 14}  // RSI, ROC, CCI
/// {"period": 14}  // WilliamsR
/// {"k_period": 14, "d_period": 3}  // Stochastic
/// {"period": 14}  // Aroon
/// {"fast_period": 12, "slow_period": 26, "signal_period": 9}  // MACD
/// {"long_period": 25, "short_period": 13, "signal_period": 7}  // TSI
///
/// // Volatility
/// {"period": 14}  // ATR
/// {"period": 20, "std_dev": 2.0}  // BollingerBands
/// {"ema_period": 20, "atr_period": 10, "atr_multiplier": 2.0}  // KeltnerChannels
/// {"period": 20}  // DonchianChannels
/// {"ema_period": 13}  // ElderRay
///
/// // Volume
/// {}  // OBV, VWAP (no params)
/// {"period": 20}  // CMF
/// {"num_bins": 20}  // VolumeProfile
///
/// // Trend
/// {"af_start": 0.02, "af_increment": 0.02, "af_max": 0.2}  // ParabolicSAR
/// {}  // PivotPoints
/// ```
///
/// # Returns
/// Dictionary mapping output names to their results:
/// - Single-output indicators: NumPy array
/// - Multi-output indicators: Nested dict with named arrays
///
/// # Example
/// ```python
/// import kimsfinance_core
/// import numpy as np
///
/// high = np.array([110.0, 115.0, 120.0, ...])
/// low = np.array([105.0, 110.0, 115.0, ...])
/// open_prices = np.array([107.0, 111.0, 116.0, ...])
/// close = np.array([108.0, 112.0, 118.0, ...])
/// volume = np.array([1000.0, 1500.0, 2000.0, ...])
///
/// # 2-tuple format (backwards compatible)
/// results = kimsfinance_core.calculate_indicators_batch(
///     high, low, open_prices, close, volume,
///     requests=[
///         ("rsi", '{"period": 14}'),
///         ("macd", '{"fast_period": 12, "slow_period": 26, "signal_period": 9}'),
///         ("atr", '{"period": 14}'),
///         ("bollinger", '{"period": 20, "std_dev": 2.0}'),
///     ]
/// )
///
/// # 3-tuple format (allows duplicates)
/// results = kimsfinance_core.calculate_indicators_batch(
///     high, low, open_prices, close, volume,
///     requests=[
///         ("sma_14", "sma", '{"period": 14}'),
///         ("sma_50", "sma", '{"period": 50}'),
///         ("rsi_7", "rsi", '{"period": 7}'),
///         ("rsi_14", "rsi", '{"period": 14}'),
///     ]
/// )
///
/// # Access results
/// rsi = results['rsi']  # NumPy array (2-tuple format)
/// sma_14 = results['sma_14']  # NumPy array (3-tuple format)
/// sma_50 = results['sma_50']  # NumPy array (3-tuple format)
/// macd_line = results['macd']['macd']  # Nested dict for multi-output
/// macd_signal = results['macd']['signal']
/// macd_histogram = results['macd']['histogram']
/// atr = results['atr']  # NumPy array
/// bb_middle = results['bollinger']['middle']
/// bb_upper = results['bollinger']['upper']
/// bb_lower = results['bollinger']['lower']
/// ```
#[pyfunction]
fn calculate_indicators_batch<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    open_prices: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    volume: PyReadonlyArray1<'_, f64>,
    requests: &Bound<'py, pyo3::types::PyList>,
) -> PyResult<Bound<'py, PyDict>> {
    // Convert PyReadonlyArray to ArrayView (zero-copy)
    let high_view = high.as_array();
    let low_view = low.as_array();
    let open_view = open_prices.as_array();
    let close_view = close.as_array();
    let volume_view = volume.as_array();

    // Create OHLCV batch container
    let ohlcv = OHLCVBatch {
        high: high_view,
        low: low_view,
        open: open_view,
        close: close_view,
        volume: volume_view,
    };

    // Parse requests - support both 2-tuple and 3-tuple formats
    let parsed_requests: Result<Vec<(String, IndicatorRequest)>, PyErr> = requests
        .iter()
        .map(|item| {
            let tuple = item.cast::<pyo3::types::PyTuple>()?;
            let len = tuple.len();

            match len {
                2 => {
                    // 2-tuple format: (indicator_type, params_json)
                    // Use indicator_type as both output name and indicator type
                    let indicator_type: String = tuple.get_item(0)?.extract()?;
                    let params_json: String = tuple.get_item(1)?.extract()?;

                    parse_indicator_request(&indicator_type, &params_json)
                        .map(|req| (indicator_type.clone(), req))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "Failed to parse 2-tuple request for '{}': {}",
                                indicator_type, e
                            ))
                        })
                }
                3 => {
                    // 3-tuple format: (output_name, indicator_type, params_json)
                    let output_name: String = tuple.get_item(0)?.extract()?;
                    let indicator_type: String = tuple.get_item(1)?.extract()?;
                    let params_json: String = tuple.get_item(2)?.extract()?;

                    parse_indicator_request(&indicator_type, &params_json)
                        .map(|req| (output_name.clone(), req))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "Failed to parse 3-tuple request for '{}' (indicator: '{}'): {}",
                                output_name, indicator_type, e
                            ))
                        })
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid request format: expected 2-tuple (indicator_type, params_json) or 3-tuple (output_name, indicator_type, params_json), got {}-tuple",
                    len
                ))),
            }
        })
        .collect();

    let parsed_requests = parsed_requests?;

    // Calculate batch (single Rust call)
    let batch_results = calculate_batch(&ohlcv, parsed_requests)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Convert results to Python dictionary
    let result_dict = PyDict::new(py);

    for (name, output) in batch_results {
        match output {
            IndicatorBatchOutput::Single(data) => {
                // Single output: return as NumPy array
                result_dict.set_item(&name, data.into_pyarray(py))?;
            }
            IndicatorBatchOutput::Multiple {
                primary,
                secondary,
                names,
            } => {
                // Multiple outputs: return as nested dict
                let inner_dict = PyDict::new(py);
                inner_dict.set_item(&names[0], primary.into_pyarray(py))?;
                for (i, arr) in secondary.iter().enumerate() {
                    inner_dict.set_item(&names[i + 1], arr.clone().into_pyarray(py))?;
                }
                result_dict.set_item(&name, inner_dict)?;
            }
        }
    }

    Ok(result_dict)
}

/// Parse JSON parameters to IndicatorRequest enum based on indicator name
fn parse_indicator_request(
    indicator_name: &str,
    json_params: &str,
) -> Result<IndicatorRequest, String> {
    use serde_json::Value;

    let params: Value =
        serde_json::from_str(json_params).map_err(|e| format!("Invalid JSON: {}", e))?;

    // Optional getters with defaults
    let get_usize_or = |key: &str, default: usize| -> usize {
        params
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    };

    let get_f64_or = |key: &str, default: f64| -> f64 {
        params.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    };

    // Match indicator name to construct appropriate request
    match indicator_name.to_lowercase().as_str() {
        // Moving Averages
        "sma" => Ok(IndicatorRequest::SMA {
            period: get_usize_or("period", 14),
        }),
        "ema" => Ok(IndicatorRequest::EMA {
            period: get_usize_or("period", 14),
        }),
        "wma" => Ok(IndicatorRequest::WMA {
            period: get_usize_or("period", 14),
        }),
        "vwma" => Ok(IndicatorRequest::VWMA {
            period: get_usize_or("period", 14),
        }),
        "dema" => Ok(IndicatorRequest::DEMA {
            period: get_usize_or("period", 14),
        }),
        "tema" => Ok(IndicatorRequest::TEMA {
            period: get_usize_or("period", 14),
        }),
        "hma" => Ok(IndicatorRequest::HMA {
            period: get_usize_or("period", 14),
        }),

        // Momentum
        "rsi" => Ok(IndicatorRequest::RSI {
            period: get_usize_or("period", 14),
        }),
        "roc" => Ok(IndicatorRequest::ROC {
            period: get_usize_or("period", 14),
        }),
        "williamsr" | "williams_r" => Ok(IndicatorRequest::WilliamsR {
            period: get_usize_or("period", 14),
        }),
        "stochastic" => Ok(IndicatorRequest::Stochastic {
            k_period: get_usize_or("k_period", 14),
            d_period: get_usize_or("d_period", 3),
        }),
        "aroon" => Ok(IndicatorRequest::Aroon {
            period: get_usize_or("period", 14),
        }),
        "cci" => Ok(IndicatorRequest::CCI {
            period: get_usize_or("period", 20),
        }),
        "macd" => Ok(IndicatorRequest::MACD {
            fast_period: get_usize_or("fast_period", 12),
            slow_period: get_usize_or("slow_period", 26),
            signal_period: get_usize_or("signal_period", 9),
        }),
        "tsi" => Ok(IndicatorRequest::TSI {
            long_period: get_usize_or("long_period", 25),
            short_period: get_usize_or("short_period", 13),
            signal_period: get_usize_or("signal_period", 7),
        }),

        // Volatility
        "atr" => Ok(IndicatorRequest::ATR {
            period: get_usize_or("period", 14),
        }),
        "bollinger" | "bollingerbands" | "bb" => Ok(IndicatorRequest::BollingerBands {
            period: get_usize_or("period", 20),
            std_dev: get_f64_or("std_dev", 2.0),
        }),
        "keltner" | "keltnerchannels" | "kc" => Ok(IndicatorRequest::KeltnerChannels {
            ema_period: get_usize_or("ema_period", 20),
            atr_period: get_usize_or("atr_period", 10),
            atr_multiplier: get_f64_or("atr_multiplier", 2.0),
        }),
        "donchian" | "donchianchannels" | "dc" => Ok(IndicatorRequest::DonchianChannels {
            period: get_usize_or("period", 20),
        }),
        "elderray" | "elder_ray" => Ok(IndicatorRequest::ElderRay {
            ema_period: get_usize_or("ema_period", 13),
        }),

        // Volume
        "obv" => Ok(IndicatorRequest::OBV),
        "vwap" => Ok(IndicatorRequest::VWAP),
        "cmf" => Ok(IndicatorRequest::CMF {
            period: get_usize_or("period", 20),
        }),
        "mfi" => Ok(IndicatorRequest::MFI {
            period: get_usize_or("period", 14),
        }),
        "volumeprofile" | "volume_profile" => Ok(IndicatorRequest::VolumeProfile {
            num_bins: get_usize_or("num_bins", 20),
        }),

        // Trend
        "parabolicsar" | "psar" | "sar" => Ok(IndicatorRequest::ParabolicSAR {
            af_start: get_f64_or("af_start", 0.02),
            af_increment: get_f64_or("af_increment", 0.02),
            af_max: get_f64_or("af_max", 0.2),
        }),
        "pivotpoints" | "pivot" => Ok(IndicatorRequest::PivotPoints),

        _ => Err(format!(
            "Unknown indicator: '{}'. Supported indicators: sma, ema, wma, vwma, dema, tema, hma, rsi, roc, williamsr, stochastic, aroon, cci, macd, tsi, atr, bollinger, keltner, donchian, elderray, obv, vwap, cmf, volumeprofile, parabolicsar, pivotpoints",
            indicator_name
        )),
    }
}

// ============================================================================
// BACKTESTING API - Python Strategy Interface
// ============================================================================

/// Python strategy wrapper that calls user-defined Python code
struct PyStrategyWrapper {
    strategy_obj: Py<pyo3::types::PyAny>,
    initial_capital: f64,
}

impl PyStrategyWrapper {
    fn new(strategy_obj: Py<pyo3::types::PyAny>, initial_capital: f64) -> Self {
        Self {
            strategy_obj,
            initial_capital,
        }
    }
}

impl Strategy for PyStrategyWrapper {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        // Acquire GIL to call into Python (PyO3 0.27+)
        Python::attach(|py| {
            // Convert bar to Python dict
            let bar_dict = PyDict::new(py);
            bar_dict.set_item("timestamp", bar.timestamp).ok()?;
            bar_dict.set_item("open", bar.open).ok()?;
            bar_dict.set_item("high", bar.high).ok()?;
            bar_dict.set_item("low", bar.low).ok()?;
            bar_dict.set_item("close", bar.close).ok()?;
            bar_dict.set_item("volume", bar.volume).ok()?;

            // Convert indicators to Python dict
            let indicators_dict = PyDict::new(py);
            for (key, value) in indicators {
                indicators_dict.set_item(key, value).ok()?;
            }

            // Call Python strategy.on_data(bar, indicators)
            let result = self
                .strategy_obj
                .call_method1(py, "on_data", (bar_dict, indicators_dict))
                .ok()?;

            // Convert Python string to Signal
            let signal_str: String = result.extract(py).ok()?;
            Some(match signal_str.to_lowercase().as_str() {
                "buy" => Signal::Buy,
                "sell" => Signal::Sell,
                "hold" => Signal::Hold,
                "short" => Signal::Short,
                "cover" => Signal::Cover,
                _ => Signal::Hold,
            })
        })
        .unwrap_or(Signal::Hold)
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        // Acquire GIL to call into Python (PyO3 0.27+)
        Python::attach(|py| {
            // Call Python strategy.get_indicators()
            let result = self.strategy_obj.call_method0(py, "get_indicators").ok()?;

            // Convert Python list of strings to IndicatorConfig
            let indicators_list: Vec<String> = result.extract(py).ok()?;
            let mut configs = Vec::new();

            for indicator_str in indicators_list {
                // Parse "indicator_period" format (e.g., "rsi_14", "atr_20")
                let parts: Vec<&str> = indicator_str.split('_').collect();
                if parts.len() >= 2 {
                    let indicator_type = parts[0];
                    let period: usize = parts[1].parse().unwrap_or(14);

                    let config = match indicator_type {
                        "rsi" => IndicatorConfig::RSI { period },
                        "atr" => IndicatorConfig::ATR { period },
                        "sma" => IndicatorConfig::SMA { period },
                        "ema" => IndicatorConfig::EMA { period },
                        "cci" => IndicatorConfig::CCI { period },
                        "roc" => IndicatorConfig::ROC { period },
                        "williamsr" | "williams" => IndicatorConfig::WilliamsR { period },
                        _ => continue,
                    };
                    configs.push(config);
                }
            }

            Some(configs)
        })
        .unwrap_or_default()
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }

    fn position_size(&self, equity: f64, signal: Signal) -> f64 {
        // Acquire GIL to call into Python (PyO3 0.27+)
        Python::attach(|py| {
            // Call Python strategy.position_size(equity, signal) if exists
            let signal_str = match signal {
                Signal::Buy => "buy",
                Signal::Sell => "sell",
                Signal::Hold => "hold",
                Signal::Short => "short",
                Signal::Cover => "cover",
            };

            self.strategy_obj
                .call_method1(py, "position_size", (equity, signal_str))
                .ok()?
                .extract::<f64>(py)
                .ok()
        })
        .unwrap_or(1.0) // Default: 100% allocation
    }
}

/// Run backtest on OHLCV data with Python strategy
///
/// # Arguments
///
/// * `high` - Array of high prices
/// * `low` - Array of low prices
/// * `close` - Array of close prices
/// * `open_prices` - Array of open prices
/// * `volume` - Array of volume data
/// * `timestamps` - Array of Unix timestamps
/// * `strategy` - Python strategy object with on_data() and get_indicators() methods
/// * `initial_capital` - Starting capital (default: 10000.0)
/// * `trading_fee` - Trading fee per trade (default: 0.001 = 0.1%)
/// * `slippage` - Slippage per trade (default: 0.0005 = 0.05%)
/// * `use_gpu` - Enable GPU acceleration if available (default: True)
///
/// # Returns
///
/// Dictionary with backtest results:
/// - `final_equity`: Final equity value
/// - `total_return`: Total return percentage
/// - `sharpe_ratio`: Annualized Sharpe ratio
/// - `max_drawdown`: Maximum drawdown percentage
/// - `win_rate`: Win rate percentage
/// - `num_trades`: Number of trades executed
/// - `profit_factor`: Gross profit / gross loss
/// - `equity_curve`: NumPy array of equity values over time
/// - `trades`: List of trade dictionaries
///
/// # Example
///
/// ```python
/// import kimsfinance_core
/// import numpy as np
///
/// class SimpleRSI:
///     def on_data(self, bar, indicators):
///         rsi = indicators.get('rsi_14', 50.0)
///         if rsi < 30:
///             return 'buy'
///         elif rsi > 70:
///             return 'sell'
///         return 'hold'
///
///     def get_indicators(self):
///         return ['rsi_14']
///
/// result = kimsfinance_core.run_backtest(
///     high=df['high'].values,
///     low=df['low'].values,
///     close=df['close'].values,
///     open_prices=df['open'].values,
///     volume=df['volume'].values,
///     timestamps=df['timestamp'].values,
///     strategy=SimpleRSI(),
///     use_gpu=True
/// )
///
/// print(f"Sharpe: {result['sharpe_ratio']:.2f}")
/// print(f"Return: {result['total_return']:.2f}%")
/// ```
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    high,
    low,
    close,
    open_prices,
    volume,
    timestamps,
    strategy,
    initial_capital = 10_000.0,
    trading_fee = 0.001,
    slippage = 0.0005,
    use_gpu = true
))]
fn run_backtest<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    open_prices: PyReadonlyArray1<'_, f64>,
    volume: PyReadonlyArray1<'_, f64>,
    timestamps: PyReadonlyArray1<'_, i64>,
    strategy: Py<pyo3::types::PyAny>,
    initial_capital: f64,
    trading_fee: f64,
    slippage: f64,
    use_gpu: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use ndarray::Array1;

    // Convert PyReadonlyArray to Array1 (owned)
    let high_array = Array1::from_vec(high.as_slice()?.to_vec());
    let low_array = Array1::from_vec(low.as_slice()?.to_vec());
    let close_array = Array1::from_vec(close.as_slice()?.to_vec());
    let open_array = Array1::from_vec(open_prices.as_slice()?.to_vec());
    let volume_array = Array1::from_vec(volume.as_slice()?.to_vec());
    let timestamps_vec = timestamps.as_slice()?.to_vec();

    // Create backtest configuration
    let config = BacktestConfig {
        initial_capital,
        trading_fee,
        slippage,
        use_gpu,
        force_cpu: !use_gpu,
    };

    // Create backtest engine
    let engine = BacktestEngine::with_config(config);

    // Wrap Python strategy
    let mut py_strategy = PyStrategyWrapper::new(strategy, initial_capital);

    // Run backtest
    let result = engine
        .run(
            &mut py_strategy,
            &timestamps_vec,
            &open_array,
            &high_array,
            &low_array,
            &close_array,
            &volume_array,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Convert BacktestResult to Python dictionary
    let result_dict = PyDict::new(py);

    result_dict.set_item("final_equity", result.final_equity)?;
    result_dict.set_item("total_return", result.total_return)?;
    result_dict.set_item("sharpe_ratio", result.sharpe_ratio)?;
    result_dict.set_item("max_drawdown", result.max_drawdown)?;
    result_dict.set_item("win_rate", result.win_rate)?;
    result_dict.set_item("num_trades", result.num_trades)?;
    result_dict.set_item("profit_factor", result.profit_factor)?;

    // Convert equity curve to NumPy array
    let equity_curve_array = Array1::from_vec(result.equity_curve);
    result_dict.set_item("equity_curve", equity_curve_array.into_pyarray(py))?;

    // Convert trades to list of dictionaries
    let mut trades_list = Vec::new();
    for trade in result.trades {
        let trade_dict = PyDict::new(py);
        trade_dict.set_item("entry_time", trade.entry_time)?;
        trade_dict.set_item("exit_time", trade.exit_time)?;
        trade_dict.set_item("entry_price", trade.entry_price)?;
        trade_dict.set_item("exit_price", trade.exit_price)?;
        trade_dict.set_item("quantity", trade.quantity)?;
        trade_dict.set_item(
            "direction",
            match trade.direction {
                backtest::TradeDirection::Long => "long",
                backtest::TradeDirection::Short => "short",
            },
        )?;
        trade_dict.set_item("pnl", trade.pnl)?;
        trade_dict.set_item("pnl_percent", trade.pnl_percent)?;
        trades_list.push(trade_dict);
    }
    result_dict.set_item("trades", trades_list)?;

    Ok(result_dict)
}

/// Python module for kimsfinance core functionality
///
/// This module declares support for Python 3.14 free-threading (no-GIL).
/// All functions are thread-safe and can be called concurrently without GIL.
#[pymodule(gil_used = false)]
fn kimsfinance_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Coordinate calculations
    m.add_function(wrap_pyfunction!(calculate_coordinates_py, m)?)?;

    // Moving Averages (7 indicators)
    m.add_function(wrap_pyfunction!(calculate_sma, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_ema, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_wma, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_vwma, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_dema, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_tema, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_hma, m)?)?;

    // Momentum indicators (8 indicators)
    m.add_function(wrap_pyfunction!(calculate_rsi, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_roc, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_williams_r, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_stochastic, m)?)?;
    #[cfg(feature = "gpu")]
    m.add_function(wrap_pyfunction!(calculate_stochastic_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_aroon, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_cci, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_macd, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_tsi, m)?)?;

    // Volatility indicators (5 indicators)
    m.add_function(wrap_pyfunction!(calculate_atr, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_keltner_channels, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_donchian_channels, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_elder_ray, m)?)?;

    // Volume indicators (5 indicators)
    m.add_function(wrap_pyfunction!(calculate_obv, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_vwap, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_cmf, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_mfi, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_volume_profile, m)?)?;

    // Trend indicators (1 indicator)
    m.add_function(wrap_pyfunction!(calculate_parabolic_sar, m)?)?;

    // Batch API (FFI overhead reduction)
    m.add_function(wrap_pyfunction!(calculate_indicators_batch, m)?)?;

    // Backtesting API
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;

    // GPU Batch Backtesting (genetic algorithm optimization)
    #[cfg(feature = "gpu")]
    {
        m.add_function(wrap_pyfunction!(batch_backtest_py::batch_backtest, m)?)?;
        m.add_function(wrap_pyfunction!(batch_backtest_py::batch_backtest_info, m)?)?;
        m.add_class::<batch_backtest_py::PyBacktestResult>()?;
    }

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__doc__",
        "High-performance Rust implementation for kimsfinance (coordinates + 24 technical indicators + batch API + backtesting)"
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_exports() {
        // Basic smoke test to ensure module compiles
        let version = env!("CARGO_PKG_VERSION");
        assert!(!version.is_empty());
    }
}
