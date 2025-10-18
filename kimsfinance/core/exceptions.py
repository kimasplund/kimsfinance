"""
Exception Hierarchy for mplfinance-polars
==========================================

Custom exceptions for error handling and debugging.
"""

from __future__ import annotations


class MplfinancePolarsError(Exception):
    """Base exception for all mplfinance-polars errors."""
    pass


class GPUNotAvailableError(MplfinancePolarsError):
    """
    Raised when GPU engine is explicitly requested but not available.

    This typically occurs when:
    - RAPIDS cuDF is not installed
    - No CUDA-capable GPU is present
    - GPU drivers are not properly configured
    """

    def __init__(self, message: str | None = None):
        if message is None:
            message = (
                "GPU engine requested but not available. "
                "Ensure RAPIDS cuDF is installed and a CUDA-capable GPU is accessible. "
                "Install with: pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12"
            )
        super().__init__(message)


class DataValidationError(MplfinancePolarsError):
    """
    Raised when input data fails validation.

    Examples:
    - Missing required columns (open, high, low, close)
    - Invalid data types
    - Empty DataFrames
    - NaN values in critical columns
    """
    pass


class EngineError(MplfinancePolarsError):
    """
    Raised when an engine-specific operation fails.

    This is typically caught internally for fallback to CPU.
    """
    pass


class OperationNotSupportedError(MplfinancePolarsError):
    """
    Raised when an operation is not supported by the selected engine.

    Example: Certain complex operations may not have GPU implementations yet.
    """
    pass


class ConfigurationError(MplfinancePolarsError):
    """
    Raised when library configuration is invalid.

    Examples:
    - Invalid engine parameter
    - Conflicting settings
    """
    pass
