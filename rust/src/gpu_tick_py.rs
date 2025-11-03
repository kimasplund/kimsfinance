/// Python bindings for GPU tick aggregation
///
/// Exposes GPU-accelerated tick aggregation to Python via PyO3.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{IntoPyArray, PyReadonlyArray1};
use crate::gpu::{GpuDevice, tick_aggregation::{TickAggregator, AggregatedCandles}};

/// Python wrapper for AggregatedCandles
#[pyclass(name = "AggregatedCandles")]
#[derive(Clone)]
pub struct PyAggregatedCandles {
    inner: AggregatedCandles,
}

#[pymethods]
impl PyAggregatedCandles {
    /// Get candle timestamps as NumPy array
    #[getter]
    fn timestamps<'py>(&self, py: Python<'py>) -> Py<pyo3::PyAny> {
        self.inner.timestamps.clone().into_pyarray(py).into()
    }

    /// Get open prices as NumPy array
    #[getter]
    fn open<'py>(&self, py: Python<'py>) -> Py<pyo3::PyAny> {
        self.inner.open.clone().into_pyarray(py).into()
    }

    /// Get high prices as NumPy array
    #[getter]
    fn high<'py>(&self, py: Python<'py>) -> Py<pyo3::PyAny> {
        self.inner.high.clone().into_pyarray(py).into()
    }

    /// Get low prices as NumPy array
    #[getter]
    fn low<'py>(&self, py: Python<'py>) -> Py<pyo3::PyAny> {
        self.inner.low.clone().into_pyarray(py).into()
    }

    /// Get close prices as NumPy array
    #[getter]
    fn close<'py>(&self, py: Python<'py>) -> Py<pyo3::PyAny> {
        self.inner.close.clone().into_pyarray(py).into()
    }

    /// Get volumes as NumPy array
    #[getter]
    fn volume<'py>(&self, py: Python<'py>) -> Py<pyo3::PyAny> {
        self.inner.volume.clone().into_pyarray(py).into()
    }

    /// Get trade counts as NumPy array
    #[getter]
    fn num_trades<'py>(&self, py: Python<'py>) -> Py<pyo3::PyAny> {
        self.inner.num_trades.clone().into_pyarray(py).into()
    }

    /// Get number of candles
    #[getter]
    fn num_candles(&self) -> usize {
        self.inner.num_candles
    }

    /// Convert to dictionary for easy access
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("timestamps", self.timestamps(py))?;
        dict.set_item("open", self.open(py))?;
        dict.set_item("high", self.high(py))?;
        dict.set_item("low", self.low(py))?;
        dict.set_item("close", self.close(py))?;
        dict.set_item("volume", self.volume(py))?;
        dict.set_item("num_trades", self.num_trades(py))?;
        dict.set_item("num_candles", self.num_candles())?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "AggregatedCandles(num_candles={}, timeframe={}ms)",
            self.inner.num_candles,
            if self.inner.num_candles > 1 {
                self.inner.timestamps.get(1).unwrap_or(&0) - self.inner.timestamps.get(0).unwrap_or(&0)
            } else {
                0
            }
        )
    }
}

/// Python wrapper for GPU TickAggregator
#[pyclass(name = "GpuTickAggregator")]
pub struct PyTickAggregator {
    aggregator: TickAggregator,
}

#[pymethods]
impl PyTickAggregator {
    /// Create a new GPU tick aggregator
    ///
    /// # Example
    /// ```python
    /// import kimsfinance_core
    /// aggregator = kimsfinance_core.GpuTickAggregator()
    /// ```
    #[new]
    fn new() -> PyResult<Self> {
        let device = GpuDevice::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to initialize GPU device: {:?}", e)
            ))?;

        let aggregator = TickAggregator::new(device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create tick aggregator: {:?}", e)
            ))?;

        Ok(Self { aggregator })
    }

    /// Aggregate tick data into OHLCV candles
    ///
    /// # Arguments
    /// * `timestamps` - Tick timestamps (milliseconds since epoch)
    /// * `prices` - Tick prices
    /// * `volumes` - Tick volumes
    /// * `sides` - Tick sides (1 for buy, -1 for sell)
    /// * `timeframe_ms` - Candle timeframe in milliseconds (e.g., 300_000 for 5 minutes)
    ///
    /// # Returns
    /// AggregatedCandles object with OHLCV data
    ///
    /// # Example
    /// ```python
    /// import kimsfinance_core
    /// import numpy as np
    ///
    /// # Create aggregator
    /// aggregator = kimsfinance_core.GpuTickAggregator()
    ///
    /// # Prepare tick data
    /// timestamps = np.array([1000, 1500, 2000, 2500, 3000], dtype=np.int64)
    /// prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float32)
    /// volumes = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    /// sides = np.array([1, 1, -1, 1, -1], dtype=np.int8)
    ///
    /// # Aggregate to 3-second candles
    /// candles = aggregator.aggregate(timestamps, prices, volumes, sides, 3000)
    ///
    /// print(f"Aggregated {len(timestamps)} ticks into {candles.num_candles} candles")
    /// print(f"First candle: O={candles.open[0]}, H={candles.high[0]}, L={candles.low[0]}, C={candles.close[0]}")
    /// ```
    #[pyo3(signature = (timestamps, prices, volumes, sides, timeframe_ms))]
    fn aggregate(
        &self,
        timestamps: PyReadonlyArray1<i64>,
        prices: PyReadonlyArray1<f32>,
        volumes: PyReadonlyArray1<f32>,
        sides: PyReadonlyArray1<i8>,
        timeframe_ms: i64,
    ) -> PyResult<PyAggregatedCandles> {
        let timestamps = timestamps.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid timestamps array: {}", e))
        })?;
        let prices = prices.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid prices array: {}", e))
        })?;
        let volumes = volumes.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid volumes array: {}", e))
        })?;
        let sides = sides.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid sides array: {}", e))
        })?;

        let candles = self.aggregator.aggregate(timestamps, prices, volumes, sides, timeframe_ms)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("GPU aggregation failed: {:?}", e)
            ))?;

        Ok(PyAggregatedCandles { inner: candles })
    }

    fn __repr__(&self) -> String {
        "GpuTickAggregator(device=CUDA, kernels=[bin_trades, aggregate_ohlcv_direct, quantize, dequantize])".to_string()
    }
}

/// Check if GPU is available
///
/// # Example
/// ```python
/// import kimsfinance_core
/// if kimsfinance_core.gpu_available():
///     print("GPU acceleration available!")
/// else:
///     print("GPU not available, using CPU fallback")
/// ```
#[pyfunction]
pub fn gpu_available() -> bool {
    GpuDevice::new().is_ok()
}

/// Get GPU device information
///
/// # Returns
/// Dictionary with GPU information (device_id, name, compute_capability, etc.)
///
/// # Example
/// ```python
/// import kimsfinance_core
/// if kimsfinance_core.gpu_available():
///     info = kimsfinance_core.gpu_info()
///     print(f"GPU: {info['name']}")
///     print(f"Compute Capability: {info['compute_capability']}")
/// ```
#[pyfunction]
pub fn gpu_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let device = GpuDevice::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("GPU not available: {:?}", e)
        ))?;

    let dict = PyDict::new(py);
    dict.set_item("device_id", 0)?;
    dict.set_item("cuda_version", "13.0")?;
    dict.set_item("compute_capability", "8.9")?;
    dict.set_item("async_allocator", true)?;  // Always enabled in GpuDevice::new()

    Ok(dict.into())
}
