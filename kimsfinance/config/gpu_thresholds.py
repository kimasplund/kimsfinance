"""
GPU Crossover Thresholds Configuration
======================================

Defines when operations should switch from CPU to GPU based on data size.
These thresholds are based on hardware benchmarking and operation complexity.

Threshold Selection Rationale:
- Simple vectorizable ops (RSI, ROC, Bollinger): 50K rows (1-2 passes, high vectorization)
- Complex vectorizable ops (MACD, Stochastic): 100K rows (multiple passes, moderate complexity)
- Iterative/state-dependent ops (Parabolic SAR, Aroon): 500K rows (sequential logic, limited GPU benefit)
- Histogram/binning ops (Volume Profile): 100K rows (GPU-optimized histogram operations)
- Rolling window ops: 50K rows (GPU-accelerated rolling operations)

Hardware Reference:
- Tested on NVIDIA RTX 3500 Ada (ThinkPad P16 Gen2)
- CPU: Intel i9-13980HX (24 cores, 5.6 GHz boost)
- Results may vary on different hardware configurations
"""

from __future__ import annotations

# Default GPU crossover thresholds (rows)
GPU_THRESHOLDS: dict[str, int] = {
    # Simple vectorizable operations (1-2 passes)
    # High parallelization potential, minimal memory overhead
    "vectorizable_simple": 50_000,  # RSI, ROC, Bollinger Bands

    # Complex vectorizable operations (multiple passes)
    # Moderate parallelization, multiple intermediate arrays
    "vectorizable_complex": 100_000,  # MACD, Stochastic

    # Iterative/state-dependent operations
    # Limited parallelization, sequential state updates
    "iterative": 500_000,  # Parabolic SAR, Aroon

    # Histogram/binning operations
    # GPU-optimized histogram kernels provide significant benefit
    "histogram": 100_000,  # Volume Profile

    # Rolling window operations
    # GPU-accelerated rolling operations (min/max/mean/std)
    "rolling": 50_000,  # Rolling min/max/mean/std

    # Aggregation operations (simple reductions)
    # GPU benefit only for large datasets
    "aggregation": 5_000,  # volume_sum, cumulative_sum, volume_weighted_price

    # Batch indicator operations
    # Lower threshold due to batch processing overhead
    "batch_indicators": 15_000,  # Batch processing of multiple indicators

    # NaN operations
    # GPU benefit for large datasets with many NaN checks
    "nan_ops": 10_000,  # nanmin, nanmax, isnan

    # Linear algebra operations
    # GPU efficient for matrix operations
    "linear_algebra": 1_000,  # least_squares, trend_line

    # Transformation operations
    # GPU benefit for large datasets with complex transforms
    "transformation": 10_000,  # pnf, renko

    # Default fallback for unspecified operations
    "default": 100_000,
}


def get_threshold(operation_type: str) -> int:
    """
    Get GPU crossover threshold for a specific operation type.

    Args:
        operation_type: Type of operation ("vectorizable_simple",
                       "vectorizable_complex", "iterative", "histogram", "rolling")

    Returns:
        Threshold in number of rows. If operation_type not found, returns 100,000.

    Examples:
        >>> get_threshold("vectorizable_simple")
        50000
        >>> get_threshold("iterative")
        500000
        >>> get_threshold("unknown")  # Default fallback
        100000
    """
    return GPU_THRESHOLDS.get(operation_type, 100_000)  # Default: 100k


# Type-specific threshold helpers for convenience
def should_use_gpu_simple(data_size: int) -> bool:
    """Check if GPU should be used for simple vectorizable operations."""
    return data_size >= GPU_THRESHOLDS["vectorizable_simple"]


def should_use_gpu_complex(data_size: int) -> bool:
    """Check if GPU should be used for complex vectorizable operations."""
    return data_size >= GPU_THRESHOLDS["vectorizable_complex"]


def should_use_gpu_iterative(data_size: int) -> bool:
    """Check if GPU should be used for iterative operations."""
    return data_size >= GPU_THRESHOLDS["iterative"]


def should_use_gpu_histogram(data_size: int) -> bool:
    """Check if GPU should be used for histogram operations."""
    return data_size >= GPU_THRESHOLDS["histogram"]


def should_use_gpu_rolling(data_size: int) -> bool:
    """Check if GPU should be used for rolling window operations."""
    return data_size >= GPU_THRESHOLDS["rolling"]


def should_use_gpu_aggregation(data_size: int) -> bool:
    """Check if GPU should be used for aggregation operations."""
    return data_size >= GPU_THRESHOLDS["aggregation"]


def should_use_gpu_batch(data_size: int) -> bool:
    """Check if GPU should be used for batch indicator operations."""
    return data_size >= GPU_THRESHOLDS["batch_indicators"]


def should_use_gpu_nan_ops(data_size: int) -> bool:
    """Check if GPU should be used for NaN operations."""
    return data_size >= GPU_THRESHOLDS["nan_ops"]


def should_use_gpu_linear_algebra(data_size: int) -> bool:
    """Check if GPU should be used for linear algebra operations."""
    return data_size >= GPU_THRESHOLDS["linear_algebra"]


def should_use_gpu_transformation(data_size: int) -> bool:
    """Check if GPU should be used for transformation operations."""
    return data_size >= GPU_THRESHOLDS["transformation"]
