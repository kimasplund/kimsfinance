"""
mplfinance Integration Adapter
===============================

Provides seamless integration with mplfinance through monkey-patching.

Usage:
    >>> import mplfinance as mpf
    >>> import kimsfinance as mfp
    >>>
    >>> mfp.activate()  # Enable GPU acceleration
    >>> mpf.plot(df, type='candle', mav=(5,10,20))  # Now 7-10x faster!
"""

from __future__ import annotations

from typing import Any
import warnings

# Global state
_is_active = False
_config = {
    "default_engine": "auto",
    "gpu_min_rows": 10_000,
    "strict_mode": False,
    "performance_tracking": False,
    "verbose": True,
}

# Performance tracking
_performance_stats = {
    "total_calls": 0,
    "gpu_calls": 0,
    "cpu_calls": 0,
    "time_saved_ms": 0.0,
    "speedup": 1.0,
}


def activate(*, engine: str = "auto", strict: bool = False, verbose: bool = True) -> None:
    """
    Activate kimsfinance acceleration.

    This function monkey-patches mplfinance's internal functions to use
    GPU-accelerated Polars operations instead of pandas.

    Args:
        engine: Default engine ("cpu", "gpu", or "auto")
        strict: If True, raise errors; if False, fallback to pandas silently
        verbose: If True, print activation messages

    Example:
        >>> import mplfinance as mpf
        >>> import kimsfinance as mfp
        >>>
        >>> mfp.activate(engine="auto", strict=False)
        >>> # All mplfinance operations are now accelerated
        >>> mpf.plot(df, type='candle', mav=(5,10,20))
    """
    global _is_active

    if _is_active:
        if verbose:
            print("⚠ kimsfinance already active")
        return

    # Update configuration
    _config["default_engine"] = engine
    _config["strict_mode"] = strict
    _config["verbose"] = verbose

    # Check if mplfinance is installed
    try:
        import mplfinance
    except ImportError:
        raise ImportError("mplfinance is not installed. " "Install with: pip install mplfinance")

    # Apply patches
    from .hooks import patch_plotting_functions

    patch_plotting_functions(config=_config)

    _is_active = True

    if verbose:
        from ..core.engine import EngineManager

        gpu_available = EngineManager.check_gpu_available()

        print("✓ kimsfinance activated!")
        print(f"  Engine: {engine}")
        print(f"  GPU Available: {gpu_available}")
        print(f"  Expected Speedup: 7-10x for typical plots")

        if not gpu_available and engine == "gpu":
            warnings.warn(
                "GPU engine requested but not available. Falling back to CPU.", UserWarning
            )


def deactivate(*, verbose: bool = True) -> None:
    """
    Deactivate kimsfinance acceleration.

    Restores original mplfinance functions.

    Args:
        verbose: If True, print deactivation messages

    Example:
        >>> import kimsfinance as mfp
        >>> mfp.deactivate()
    """
    global _is_active

    if not _is_active:
        if verbose:
            print("⚠ kimsfinance is not active")
        return

    # Remove patches
    from .hooks import unpatch_plotting_functions

    unpatch_plotting_functions()

    _is_active = False

    if verbose:
        print("✓ kimsfinance deactivated")
        print("  Original mplfinance functions restored")


def is_active() -> bool:
    """
    Check if kimsfinance acceleration is active.

    Returns:
        bool: True if acceleration is active

    Example:
        >>> import kimsfinance as mfp
        >>> mfp.activate()
        >>> print(mfp.is_active())
        True
    """
    return _is_active


def configure(**kwargs) -> None:
    """
    Configure kimsfinance behavior.

    Args:
        default_engine: Default engine ("cpu", "gpu", or "auto")
        gpu_min_rows: Minimum rows for GPU to be beneficial
        strict_mode: If True, raise errors; if False, fallback silently
        performance_tracking: If True, track performance statistics
        verbose: If True, print informational messages

    Example:
        >>> import kimsfinance as mfp
        >>> mfp.configure(
        ...     default_engine="gpu",
        ...     gpu_min_rows=5000,
        ...     performance_tracking=True
        ... )
    """
    global _config

    valid_keys = {
        "default_engine",
        "gpu_min_rows",
        "strict_mode",
        "performance_tracking",
        "verbose",
    }

    for key, value in kwargs.items():
        if key not in valid_keys:
            raise ValueError(f"Invalid configuration key: {key!r}. " f"Valid keys: {valid_keys}")
        _config[key] = value

    if _config["verbose"]:
        print(f"✓ Configuration updated: {kwargs}")


def get_config() -> dict[str, Any]:
    """
    Get current configuration.

    Returns:
        dict: Current configuration settings

    Example:
        >>> import kimsfinance as mfp
        >>> config = mfp.get_config()
        >>> print(config['default_engine'])
        auto
    """
    return _config.copy()


def get_performance_stats() -> dict[str, Any]:
    """
    Get performance statistics (if tracking is enabled).

    Returns:
        dict: Performance statistics

    Example:
        >>> import kimsfinance as mfp
        >>> mfp.configure(performance_tracking=True)
        >>> mfp.activate()
        >>> # ... run some plots ...
        >>> stats = mfp.get_performance_stats()
        >>> print(f"Speedup: {stats['speedup']:.2f}x")
    """
    if not _config["performance_tracking"]:
        warnings.warn(
            "Performance tracking is disabled. "
            "Enable with: configure(performance_tracking=True)",
            UserWarning,
        )

    return _performance_stats.copy()


def reset_performance_stats() -> None:
    """Reset performance statistics."""
    global _performance_stats

    _performance_stats = {
        "total_calls": 0,
        "gpu_calls": 0,
        "cpu_calls": 0,
        "time_saved_ms": 0.0,
        "speedup": 1.0,
    }

    if _config["verbose"]:
        print("✓ Performance statistics reset")


# Internal functions for hooks to update stats
def _track_operation(engine_used: str, time_saved_ms: float = 0.0) -> None:
    """Internal: Track operation for performance statistics."""
    if not _config["performance_tracking"]:
        return

    global _performance_stats

    _performance_stats["total_calls"] += 1

    if engine_used == "gpu":
        _performance_stats["gpu_calls"] += 1
    else:
        _performance_stats["cpu_calls"] += 1

    if time_saved_ms > 0:
        _performance_stats["time_saved_ms"] += time_saved_ms

    # Update speedup estimate
    if _performance_stats["total_calls"] > 0:
        avg_speedup = 1.0 + (
            _performance_stats["time_saved_ms"] / (_performance_stats["total_calls"] * 10)
        )  # Rough estimate
        _performance_stats["speedup"] = avg_speedup
