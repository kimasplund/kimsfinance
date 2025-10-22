"""Configuration module for kimsfinance."""

from __future__ import annotations

from .gpu_thresholds import (
    GPU_THRESHOLDS,
    get_threshold,
    should_use_gpu_complex,
    should_use_gpu_histogram,
    should_use_gpu_iterative,
    should_use_gpu_rolling,
    should_use_gpu_simple,
)

__all__ = [
    "GPU_THRESHOLDS",
    "get_threshold",
    "should_use_gpu_simple",
    "should_use_gpu_complex",
    "should_use_gpu_iterative",
    "should_use_gpu_histogram",
    "should_use_gpu_rolling",
]
