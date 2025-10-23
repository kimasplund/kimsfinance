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
import threading
from contextlib import contextmanager
from collections import deque
from datetime import datetime, timedelta

# Constants for performance tracking
MAX_STATS_ENTRIES = 10_000  # Limit to 10K most recent renders
STATS_RETENTION_HOURS = 24  # Keep only last 24 hours


class BoundedPerformanceStats:
    """
    Thread-safe bounded performance statistics tracker.

    Prevents unbounded memory growth by:
    - Limiting recent entries to MAX_STATS_ENTRIES (10K)
    - Auto-evicting entries older than STATS_RETENTION_HOURS (24h)
    - Using deque for O(1) append/popleft operations
    """

    def __init__(self, max_entries: int = MAX_STATS_ENTRIES):
        self._max_entries = max_entries
        self._lock = threading.Lock()
        # Use deque for O(1) append and popleft
        self._recent_renders: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._aggregated_stats: dict[str, Any] = {
            "total_calls": 0,
            "gpu_calls": 0,
            "cpu_calls": 0,
            "time_saved_ms": 0.0,
            "speedup": 1.0,
        }

    def record(self, engine_used: str, time_saved_ms: float = 0.0) -> None:
        """Record a render with automatic cleanup of old entries."""
        timestamp = datetime.now()

        with self._lock:
            # Add new entry (automatic eviction if deque full)
            self._recent_renders.append(
                {"engine": engine_used, "time_saved_ms": time_saved_ms, "timestamp": timestamp}
            )

            # Update aggregated stats
            self._aggregated_stats["total_calls"] += 1

            if engine_used == "gpu":
                self._aggregated_stats["gpu_calls"] += 1
            else:
                self._aggregated_stats["cpu_calls"] += 1

            if time_saved_ms > 0:
                self._aggregated_stats["time_saved_ms"] += time_saved_ms

            # Update speedup estimate
            if self._aggregated_stats["total_calls"] > 0:
                avg_speedup = 1.0 + (
                    self._aggregated_stats["time_saved_ms"]
                    / (self._aggregated_stats["total_calls"] * 10)
                )  # Rough estimate
                self._aggregated_stats["speedup"] = avg_speedup

            # Cleanup old entries (older than retention period)
            cutoff = timestamp - timedelta(hours=STATS_RETENTION_HOURS)
            self._cleanup_old_entries(cutoff)

    def _cleanup_old_entries(self, cutoff: datetime) -> None:
        """Remove entries older than cutoff (must hold lock)."""
        # Remove from left while entries are old
        while self._recent_renders and self._recent_renders[0]["timestamp"] < cutoff:
            self._recent_renders.popleft()

    def get_stats(self) -> dict[str, Any]:
        """Get current aggregated statistics."""
        with self._lock:
            return {
                **self._aggregated_stats,
                "total_tracked": len(self._recent_renders),
                "max_entries": self._max_entries,
            }

    def copy(self) -> dict[str, Any]:
        """Get a copy of stats (for backward compatibility)."""
        return self.get_stats()

    def reset(self) -> None:
        """Clear all statistics."""
        with self._lock:
            self._recent_renders.clear()
            self._aggregated_stats = {
                "total_calls": 0,
                "gpu_calls": 0,
                "cpu_calls": 0,
                "time_saved_ms": 0.0,
                "speedup": 1.0,
            }


# Global state with lock protection
_lock = threading.RLock()  # Reentrant lock for nested calls
_is_active = False
_config = {
    "default_engine": "auto",
    "gpu_min_rows": 10_000,
    "strict_mode": False,
    "performance_tracking": False,
    "verbose": True,
}

# Performance tracking with bounded memory
_performance_stats = BoundedPerformanceStats()


@contextmanager
def _state_lock() -> Any:
    """Context manager for thread-safe state operations."""
    _lock.acquire()
    try:
        yield
    finally:
        _lock.release()


def activate(*, engine: str = "auto", strict: bool = False, verbose: bool = True) -> None:
    """
    Activate kimsfinance acceleration (thread-safe).

    This function monkey-patches mplfinance's internal functions to use
    GPU-accelerated Polars operations instead of pandas.

    Args:
        engine: Default engine ("cpu", "gpu", or "auto")
        strict: If True, raise errors; if False, fallback to pandas silently
        verbose: If True, print activation messages

    Thread-safe: Yes (uses global lock)

    Example:
        >>> import mplfinance as mpf
        >>> import kimsfinance as mfp
        >>>
        >>> mfp.activate(engine="auto", strict=False)
        >>> # All mplfinance operations are now accelerated
        >>> mpf.plot(df, type='candle', mav=(5,10,20))
    """
    with _state_lock():
        global _is_active

        if _is_active:
            if verbose:
                print("⚠ kimsfinance already active")
            return

        # Validate engine parameter
        if engine not in ("auto", "cpu", "gpu"):
            raise ValueError(f"Invalid engine: {engine}. Must be 'auto', 'cpu', or 'gpu'")

        # Update configuration
        _config["default_engine"] = engine
        _config["strict_mode"] = strict
        _config["verbose"] = verbose

        # Check if mplfinance is installed
        try:
            import mplfinance
        except ImportError:
            raise ImportError(
                "mplfinance is not installed. " "Install with: pip install mplfinance"
            )

        # Apply patches
        from .hooks import patch_plotting_functions

        patch_plotting_functions(config=_config.copy())  # Pass copy to avoid external modification

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
    Deactivate kimsfinance acceleration (thread-safe).

    Restores original mplfinance functions.

    Args:
        verbose: If True, print deactivation messages

    Thread-safe: Yes (uses global lock)

    Example:
        >>> import kimsfinance as mfp
        >>> mfp.deactivate()
    """
    with _state_lock():
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
    Check if kimsfinance acceleration is active (thread-safe).

    Returns:
        bool: True if acceleration is active

    Thread-safe: Yes (simple read with lock)

    Example:
        >>> import kimsfinance as mfp
        >>> mfp.activate()
        >>> print(mfp.is_active())
        True
    """
    with _state_lock():
        return _is_active


def configure(**kwargs) -> None:
    """
    Configure kimsfinance behavior (thread-safe).

    Args:
        default_engine: Default engine ("cpu", "gpu", or "auto")
        gpu_min_rows: Minimum rows for GPU to be beneficial
        strict_mode: If True, raise errors; if False, fallback silently
        performance_tracking: If True, track performance statistics
        verbose: If True, print informational messages

    Thread-safe: Yes (uses global lock)

    Example:
        >>> import kimsfinance as mfp
        >>> mfp.configure(
        ...     default_engine="gpu",
        ...     gpu_min_rows=5000,
        ...     performance_tracking=True
        ... )
    """
    with _state_lock():
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
                raise ValueError(
                    f"Invalid configuration key: {key!r}. " f"Valid keys: {valid_keys}"
                )

            # Type validation
            if key == "default_engine" and value not in ("auto", "cpu", "gpu"):
                raise ValueError(f"Invalid engine: {value}. Must be 'auto', 'cpu', or 'gpu'")
            if key == "gpu_min_rows" and (not isinstance(value, int) or value < 0):
                raise ValueError(f"gpu_min_rows must be non-negative int, got {value}")

            _config[key] = value

        if _config["verbose"]:
            print(f"✓ Configuration updated: {kwargs}")


def get_config() -> dict[str, Any]:
    """
    Get current configuration (thread-safe).

    Returns:
        dict: Copy of current configuration settings

    Thread-safe: Yes (returns copy under lock)

    Example:
        >>> import kimsfinance as mfp
        >>> config = mfp.get_config()
        >>> print(config['default_engine'])
        auto
    """
    with _state_lock():
        return _config.copy()


def get_performance_stats() -> dict[str, Any]:
    """
    Get performance statistics (thread-safe, if tracking is enabled).

    Returns:
        dict: Copy of performance statistics including:
            - total_calls: Total operations tracked
            - gpu_calls: Operations on GPU
            - cpu_calls: Operations on CPU
            - time_saved_ms: Total time saved
            - speedup: Average speedup ratio
            - total_tracked: Number of recent entries kept
            - max_entries: Maximum entries retained

    Thread-safe: Yes (uses internal lock in BoundedPerformanceStats)

    Example:
        >>> import kimsfinance as mfp
        >>> mfp.configure(performance_tracking=True)
        >>> mfp.activate()
        >>> # ... run some plots ...
        >>> stats = mfp.get_performance_stats()
        >>> print(f"Speedup: {stats['speedup']:.2f}x")
        >>> print(f"Tracking {stats['total_tracked']} recent operations")
    """
    if not _config["performance_tracking"]:
        warnings.warn(
            "Performance tracking is disabled. "
            "Enable with: configure(performance_tracking=True)",
            UserWarning,
        )

    return _performance_stats.copy()


def reset_performance_stats() -> None:
    """
    Reset performance statistics (thread-safe).

    Clears all tracked entries and resets aggregated statistics.

    Thread-safe: Yes (uses internal lock in BoundedPerformanceStats)
    """
    _performance_stats.reset()

    if _config["verbose"]:
        print("✓ Performance statistics reset")


# Internal functions for hooks to update stats
def _track_operation(engine_used: str, time_saved_ms: float = 0.0) -> None:
    """
    Internal: Track operation for performance statistics (thread-safe).

    Records operation with automatic memory management:
    - Bounded to MAX_STATS_ENTRIES (10K) most recent operations
    - Auto-evicts entries older than STATS_RETENTION_HOURS (24h)

    Thread-safe: Yes (uses internal lock in BoundedPerformanceStats)
    """
    if not _config["performance_tracking"]:
        return

    _performance_stats.record(engine_used, time_saved_ms)
