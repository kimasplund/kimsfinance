"""
kimsfinance: GPU-Accelerated Financial Charting
================================================

Python 3.13+ library providing 178x speedup for financial charting operations
using PIL, Polars, and optional GPU acceleration via RAPIDS cuDF.

Quick Start:
    >>> import kimsfinance as mfp
    >>>
    >>> # Check GPU availability
    >>> mfp.gpu_available()
    True
    >>>
    >>> # Calculate moving averages
    >>> import polars as pl
    >>> df = pl.read_csv("ohlcv_data.csv")
    >>> sma_20, sma_50 = mfp.calculate_sma(df, "close", windows=[20, 50])
    >>>
    >>> # Calculate ATR with GPU
    >>> atr = mfp.calculate_atr(df["high"], df["low"], df["close"], engine="gpu")

Package Structure:
    - ops.moving_averages: SMA, EMA (1.1-3.3x speedup on CPU)
    - ops.nan_ops: nanmin, nanmax, isnan (40-80x speedup on GPU)
    - ops.linear_algebra: Least squares (30-50x speedup on GPU)
    - ops.indicators: ATR, RSI, MACD (10-24x speedup on GPU)
    - ops.aggregations: Volume operations, resampling (10-20x speedup)

Engine Selection:
    - engine="cpu": Force CPU execution
    - engine="gpu": Force GPU execution (raises error if unavailable)
    - engine="auto": Automatically select best engine (recommended)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Kim Asplund"

# Core modules
from .core import (
    # Types
    ArrayResult,
    SeriesResult,
    DataFrameResult,
    DataFrameInput,
    ArrayLike,
    WindowSize,
    ShiftPeriods,
    Engine,
    BoundsResult,
    LinearFitResult,
    MACDResult,
    EngineConfig,
    # Exceptions
    KimsFinanceError,
    GPUNotAvailableError,
    DataValidationError,
    EngineError,
    OperationNotSupportedError,
    ConfigurationError,
    # Engine management
    EngineManager,
    with_engine_fallback,
    # Decorators
    gpu_accelerated,
    get_array_module,
    to_gpu,
    to_cpu,
)

# Moving Averages
from .ops.moving_averages import (
    calculate_sma,
    calculate_ema,
    calculate_multiple_mas,
    from_pandas_series,
)

# NaN Operations
from .ops.nan_ops import (
    nanmin_gpu,
    nanmax_gpu,
    nan_bounds,
    isnan_gpu,
    nan_indices,
    replace_nan,
    should_use_gpu_for_nan_ops,
)

# Linear Algebra
from .ops.linear_algebra import (
    least_squares_fit,
    trend_line,
    polynomial_fit,
    correlation,
    moving_linear_fit,
)

# Technical Indicators
from .ops.indicators import (
    calculate_atr,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_stochastic_oscillator,
    calculate_obv,
)

# Phase 1 Indicators - All 8 indicators (plus variants)
from .ops import (
    calculate_stochastic,
    calculate_stochastic_rsi,
    calculate_vwap,
    calculate_vwap_anchored,
    calculate_ichimoku,
    calculate_adx,
    calculate_williams_r,
    calculate_cci,
    calculate_mfi,
    calculate_supertrend,
    calculate_picks_momentum_ratio,
)

# Aggregations
from .ops.aggregations import (
    volume_sum,
    volume_weighted_price,
    ohlc_resample,
    rolling_sum,
    rolling_mean,
    cumulative_sum,
    group_aggregation,
)

# Integration - mplfinance compatibility
from .integration import (
    activate,
    deactivate,
    is_active,
    configure,
    get_config,
)

# Standalone API
from .api import (
    plot,
    make_addplot,
)


# ============================================================================
# Convenience Functions
# ============================================================================

def gpu_available() -> bool:
    """
    Check if GPU acceleration is available.

    Returns:
        True if GPU is available and functional

    Example:
        >>> import kimsfinance as mfp
        >>> if mfp.gpu_available():
        ...     print("GPU acceleration enabled!")
        ... else:
        ...     print("Running on CPU only")
    """
    return EngineManager.check_gpu_available()


def get_engine_info() -> dict[str, str | bool]:
    """
    Get information about available engines and library configuration.

    Returns:
        Dictionary with engine information

    Example:
        >>> import kimsfinance as mfp
        >>> info = mfp.get_engine_info()
        >>> print(info)
        {
            'cpu_available': True,
            'gpu_available': True,
            'cudf_version': '25.10.00',
            'default_engine': 'auto'
        }
    """
    return EngineManager.get_info()


def version() -> str:
    """Get library version."""
    return __version__


def info() -> None:
    """
    Print library information and status.

    Example:
        >>> import kimsfinance as mfp
        >>> mfp.info()
    """
    print("=" * 80)
    print("kimsfinance: GPU-Accelerated Financial Charting")
    print("=" * 80)
    print(f"Version: {__version__}")
    print(f"Python: 3.13+ (modern type system)")
    print()

    # Engine status
    gpu_avail = gpu_available()
    print(f"GPU Available: {gpu_avail}")

    if gpu_avail:
        engine_info = get_engine_info()
        if 'cudf_version' in engine_info:
            print(f"  cuDF Version: {engine_info['cudf_version']}")

    # Activation status
    print(f"Activated: {is_active()}")

    if is_active():
        config = get_config()
        print(f"  Default Engine: {config['default_engine']}")
        print(f"  GPU Min Rows: {config['gpu_min_rows']:,}")

    # Performance expectations
    print()
    print("Expected Performance:")
    print("  • Moving averages: 1.1-3.3x speedup (CPU)")
    print("  • NaN operations: 40-80x speedup (GPU)")
    print("  • Linear algebra: 30-50x speedup (GPU)")
    print("  • Overall plot: 7-10x speedup")
    print()

    # Usage example
    if not is_active():
        print("Quick Start:")
        print("  import kimsfinance as mfp")
        print("  mfp.activate()  # Enable acceleration")
        print()


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Version & Info
    "__version__",
    "version",
    "info",

    # Integration & Configuration
    "activate",
    "deactivate",
    "is_active",
    "configure",
    "get_config",

    # Convenience
    "gpu_available",
    "get_engine_info",

    # Core Types
    "ArrayResult",
    "SeriesResult",
    "DataFrameResult",
    "DataFrameInput",
    "ArrayLike",
    "WindowSize",
    "ShiftPeriods",
    "Engine",
    "BoundsResult",
    "LinearFitResult",
    "MACDResult",
    "EngineConfig",

    # Exceptions
    "KimsFinanceError",
    "GPUNotAvailableError",
    "DataValidationError",
    "EngineError",
    "OperationNotSupportedError",
    "ConfigurationError",

    # Engine
    "EngineManager",
    "with_engine_fallback",

    # Decorators
    "gpu_accelerated",
    "get_array_module",
    "to_gpu",
    "to_cpu",

    # Moving Averages
    "calculate_sma",
    "calculate_ema",
    "calculate_multiple_mas",
    "from_pandas_series",

    # NaN Operations
    "nanmin_gpu",
    "nanmax_gpu",
    "nan_bounds",
    "isnan_gpu",
    "nan_indices",
    "replace_nan",
    "should_use_gpu_for_nan_ops",

    # Linear Algebra
    "least_squares_fit",
    "trend_line",
    "polynomial_fit",
    "correlation",
    "moving_linear_fit",

    # Technical Indicators (Base)
    "calculate_atr",
    "calculate_picks_momentum_ratio",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_stochastic_oscillator",
    "calculate_obv",

    # Phase 1 Indicators (8 indicators + variants)
    "calculate_stochastic",
    "calculate_stochastic_rsi",
    "calculate_vwap",
    "calculate_vwap_anchored",
    "calculate_ichimoku",
    "calculate_adx",
    "calculate_atr",
    "calculate_williams_r",
    "calculate_cci",
    "calculate_mfi",
    "calculate_supertrend",

    # Aggregations
    "volume_sum",
    "volume_weighted_price",
    "ohlc_resample",
    "rolling_sum",
    "rolling_mean",
    "cumulative_sum",
    "group_aggregation",

    # Standalone API
    "plot",
    "make_addplot",
]


# ============================================================================
# Library Initialization
# ============================================================================

def _check_dependencies():
    """Check and report on optional dependencies."""
    deps = {
        "polars": False,
        "numpy": False,
        "pandas": False,
        "cupy": False,
        "cudf": False,
    }

    try:
        import polars
        deps["polars"] = True
    except ImportError:
        pass

    try:
        import numpy
        deps["numpy"] = True
    except ImportError:
        pass

    try:
        import pandas
        deps["pandas"] = True
    except ImportError:
        pass

    try:
        import cupy
        deps["cupy"] = True
    except ImportError:
        pass

    try:
        import cudf
        deps["cudf"] = True
    except ImportError:
        pass

    return deps


# Check dependencies on import
_deps = _check_dependencies()

if not _deps["polars"]:
    raise ImportError(
        "Polars is required but not installed. "
        "Install with: pip install polars"
    )

if not _deps["numpy"]:
    raise ImportError(
        "NumPy is required but not installed. "
        "Install with: pip install numpy"
    )

# Optional GPU dependencies
if not (_deps["cupy"] and _deps["cudf"]):
    import warnings
    warnings.warn(
        "GPU acceleration not available. Install RAPIDS for GPU support:\n"
        "  pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12 cupy-cuda12x",
        UserWarning
    )
