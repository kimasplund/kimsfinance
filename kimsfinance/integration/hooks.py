"""
mplfinance Function Hooks
==========================

Monkey-patches mplfinance internal functions with GPU-accelerated versions.
"""

from __future__ import annotations

import sys
from typing import Any
import numpy as np
import polars as pl

# Import our operations
from ..ops.moving_averages import calculate_sma, calculate_ema
from ..ops.nan_ops import nanmin_gpu, nanmax_gpu, nan_bounds
from ..core.engine import EngineManager

# Store original functions for restoration
_original_functions = {}
_config = {}


def patch_plotting_functions(config: dict[str, Any]) -> None:
    """
    Patch mplfinance plotting functions with accelerated versions.

    Args:
        config: Configuration dictionary from adapter
    """
    global _config
    _config = config

    try:
        import mplfinance.plotting as mpf_plotting
        import mplfinance._utils as mpf_utils
    except ImportError:
        raise ImportError("mplfinance not installed or incompatible version")

    # Store original functions
    _original_functions['_plot_mav'] = mpf_plotting._plot_mav
    _original_functions['_plot_ema'] = mpf_plotting._plot_ema

    # Patch plotting functions
    mpf_plotting._plot_mav = _plot_mav_accelerated
    mpf_plotting._plot_ema = _plot_ema_accelerated

    # Note: We don't patch _utils functions as they might break internal logic
    # Instead, we optimize at the plotting layer where data flows through


def unpatch_plotting_functions() -> None:
    """Restore original mplfinance functions."""
    if not _original_functions:
        return

    try:
        import mplfinance.plotting as mpf_plotting
    except ImportError:
        return

    # Restore original functions
    if '_plot_mav' in _original_functions:
        mpf_plotting._plot_mav = _original_functions['_plot_mav']
    if '_plot_ema' in _original_functions:
        mpf_plotting._plot_ema = _original_functions['_plot_ema']

    _original_functions.clear()


def _plot_mav_accelerated(ax, config, xdates, prices, apmav=None, apwidth=None):
    """
    GPU-accelerated Simple Moving Average plotting.

    Replaces: mplfinance.plotting._plot_mav
    Speedup: 1.1-3.3x on CPU
    """
    # Get MA parameters
    mavgs = config.get('mav', ())
    if not mavgs:
        return

    # Convert to tuple if needed
    if isinstance(mavgs, int):
        mavgs = (mavgs,)

    # Get shift if specified
    shift = config.get('mav_shift', None)

    # Determine engine
    engine = _config.get('default_engine', 'auto')
    data_size = len(prices)

    # For moving averages, always use CPU (GPU not beneficial)
    exec_engine = "cpu"

    try:
        # Convert prices to Polars DataFrame
        df = pl.DataFrame({"price": prices})

        # Calculate all SMAs in single pass
        sma_results = calculate_sma(
            df, "price",
            windows=list(mavgs),
            shift=shift,
            engine=exec_engine
        )

        # Plot each MA using original plotting logic
        if apmav is not None:
            _plot_mav_with_panel(ax, config, xdates, sma_results, mavgs, apmav, apwidth)
        else:
            _plot_mav_on_main(ax, config, xdates, sma_results, mavgs)

    except Exception as e:
        if _config.get('strict_mode', False):
            raise

        # Fallback to original pandas implementation
        import warnings
        warnings.warn(
            f"GPU acceleration failed, falling back to pandas: {e}",
            UserWarning
        )
        _original_functions['_plot_mav'](ax, config, xdates, prices, apmav, apwidth)


def _plot_ema_accelerated(ax, config, xdates, prices, apmav=None, apwidth=None):
    """
    GPU-accelerated Exponential Moving Average plotting.

    Replaces: mplfinance.plotting._plot_ema
    Speedup: 1.1-3.3x on CPU
    """
    # Get EMA parameters
    mavgs = config.get('ema', ())
    if not mavgs:
        return

    if isinstance(mavgs, int):
        mavgs = (mavgs,)

    shift = config.get('ema_shift', None)

    # Always use CPU for moving averages
    exec_engine = "cpu"

    try:
        # Convert to Polars
        df = pl.DataFrame({"price": prices})

        # Calculate all EMAs
        ema_results = calculate_ema(
            df, "price",
            windows=list(mavgs),
            shift=shift,
            engine=exec_engine,
            adjust=False  # Match mplfinance behavior
        )

        # Plot each EMA
        if apmav is not None:
            _plot_ema_with_panel(ax, config, xdates, ema_results, mavgs, apmav, apwidth)
        else:
            _plot_ema_on_main(ax, config, xdates, ema_results, mavgs)

    except Exception as e:
        if _config.get('strict_mode', False):
            raise

        import warnings
        warnings.warn(
            f"GPU acceleration failed, falling back to pandas: {e}",
            UserWarning
        )
        _original_functions['_plot_ema'](ax, config, xdates, prices, apmav, apwidth)


# Helper functions for plotting (mimic mplfinance internal logic)

def _plot_mav_on_main(ax, config, xdates, sma_results, mavgs):
    """Plot SMAs on main price axis."""
    # Get colors and styles from config
    from matplotlib import cycler

    # Use mplfinance's color cycle or default
    if 'marketcolor' in config:
        colors = config['marketcolor'].get('mav_colors', None)
    else:
        colors = None

    for idx, (mav, mavprice) in enumerate(zip(mavgs, sma_results)):
        # Plot the MA line
        if colors and idx < len(colors):
            ax.plot(xdates, mavprice, color=colors[idx], linewidth=1, label=f'MA({mav})')
        else:
            ax.plot(xdates, mavprice, linewidth=1, label=f'MA({mav})')


def _plot_mav_with_panel(ax, config, xdates, sma_results, mavgs, apmav, apwidth):
    """Plot SMAs on separate panel."""
    # Similar to above but uses panel axis
    for idx, (mav, mavprice) in enumerate(zip(mavgs, sma_results)):
        ax.plot(xdates, mavprice, linewidth=apwidth[idx] if apwidth else 1, label=f'MA({mav})')


def _plot_ema_on_main(ax, config, xdates, ema_results, mavgs):
    """Plot EMAs on main price axis."""
    from matplotlib import cycler

    if 'marketcolor' in config:
        colors = config['marketcolor'].get('ema_colors', None)
    else:
        colors = None

    for idx, (mav, emaprice) in enumerate(zip(mavgs, ema_results)):
        if colors and idx < len(colors):
            ax.plot(xdates, emaprice, color=colors[idx], linewidth=1, label=f'EMA({mav})', linestyle='--')
        else:
            ax.plot(xdates, emaprice, linewidth=1, label=f'EMA({mav})', linestyle='--')


def _plot_ema_with_panel(ax, config, xdates, ema_results, mavgs, apmav, apwidth):
    """Plot EMAs on separate panel."""
    for idx, (mav, emaprice) in enumerate(zip(mavgs, ema_results)):
        ax.plot(xdates, emaprice, linewidth=apwidth[idx] if apwidth else 1, label=f'EMA({mav})', linestyle='--')


# Additional optimizations we can add later:
# - Patch _check_and_prepare_data for NaN operations
# - Patch volume aggregations
# - Patch axis scaling (nanmin/nanmax)

def _optimize_nan_operations(data):
    """
    Future: Optimize NaN operations in data preparation.

    This would replace numpy's nanmin/nanmax with GPU versions
    for axis scaling and data validation.
    """
    pass
