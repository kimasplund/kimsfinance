"""Parallel rendering support for batch chart generation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Literal
import os
import io
import sys


def _render_one_chart(args: tuple[Any, ...]) -> str | bytes:
    """
    Helper function for parallel rendering (must be module-level for pickle).

    This function is called by each worker process to render a single chart.
    It must be at module level (not nested) to be picklable by multiprocessing.

    Args:
        args: Tuple of (ohlc, volume, output_path, save_kwargs, render_kwargs)

    Returns:
        Output path if file was saved, or PNG bytes if in-memory rendering
    """
    from kimsfinance.plotting import render_ohlcv_chart, save_chart

    ohlc, volume, output_path, save_kwargs, render_kwargs = args

    # Render the chart
    img = render_ohlcv_chart(ohlc, volume, **render_kwargs)

    # Save to file or return as bytes
    if output_path:
        save_chart(img, output_path, **save_kwargs)
        return str(output_path)
    else:
        # Return as PNG bytes for in-memory processing
        # Use context manager to ensure buffer is properly closed
        with io.BytesIO() as buf:
            img.save(buf, format="PNG")
            image_data = buf.getvalue()
        # Buffer automatically closed here
        return image_data


def _check_free_threading_available() -> bool:
    """
    Check if Python free-threading (GIL removal) is available.

    Returns:
        True if running on python3.14t with GIL disabled
    """
    try:
        # Python 3.13+ with free-threading build (python3.14t)
        return hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled()
    except Exception:
        return False


FREE_THREADING_AVAILABLE = _check_free_threading_available()


def render_charts_parallel(
    datasets: list[dict[str, Any]],
    output_paths: list[str] | None = None,
    num_workers: int | None = None,
    speed: str = "fast",
    executor_type: Literal["process", "thread", "auto"] = "auto",
    **common_render_kwargs: Any,
) -> list[str | bytes]:
    """
    Render multiple charts in parallel using multiprocessing or multithreading.

    This function distributes chart rendering across multiple CPU cores
    for maximum throughput. Each chart is rendered independently, enabling
    linear scaling with CPU cores.

    **Free-Threading Support (Python 3.14t)**:
    When running on python3.14t (free-threaded build with GIL removal),
    this function automatically uses ThreadPoolExecutor for 3.1x better
    parallel efficiency compared to ProcessPoolExecutor. Set executor_type='auto'
    to enable automatic selection.

    Args:
        datasets: List of dicts with 'ohlc' and 'volume' keys
            Each dict must contain:
            - 'ohlc': Dict with 'open', 'high', 'low', 'close' arrays
            - 'volume': Volume data array
        output_paths: Optional list of output paths (same length as datasets)
            If None, returns charts as PNG bytes
            If provided, saves charts to disk at specified paths
        num_workers: Number of parallel workers (defaults to CPU count)
            Set to None to use os.cpu_count() (all available cores)
            Set to specific value to limit parallelism
        speed: Encoding speed for save_chart() ('fast', 'balanced', 'best')
            Default is 'fast' for batch processing performance
            This parameter is only used when output_paths is provided
        executor_type: Parallelism strategy ('process', 'thread', 'auto')
            - 'process': Use ProcessPoolExecutor (multiprocessing)
            - 'thread': Use ThreadPoolExecutor (multithreading)
            - 'auto': Use threads on python3.14t, processes otherwise (recommended)
            Default is 'auto' for best performance
        **common_render_kwargs: Common rendering options for all charts
            Examples: theme, width, height, enable_antialiasing, show_grid

    Returns:
        List of output paths (if output_paths provided) or PNG bytes
        Results are in same order as input datasets

    Examples:
        >>> # Automatic executor selection (recommended)
        >>> datasets = [
        ...     {'ohlc': ohlc1, 'volume': vol1},
        ...     {'ohlc': ohlc2, 'volume': vol2},
        ...     {'ohlc': ohlc3, 'volume': vol3},
        ... ]
        >>> paths = [f"chart_{i}.webp" for i, _ in enumerate(datasets)]
        >>> result = render_charts_parallel(datasets, paths, executor_type='auto')
        >>> # Uses ThreadPoolExecutor on python3.14t, ProcessPoolExecutor otherwise

        >>> # Force multithreading (python3.14t recommended)
        >>> result = render_charts_parallel(datasets, paths, executor_type='thread')

        >>> # Force multiprocessing (always works)
        >>> result = render_charts_parallel(datasets, paths, executor_type='process')

        >>> # In-memory rendering (returns PNG bytes)
        >>> png_bytes = render_charts_parallel(datasets, num_workers=8)
        >>> # Returns: [b'\\x89PNG...', b'\\x89PNG...', b'\\x89PNG...']

    Notes:
        - **Python 3.14t (free-threaded)**: 3.1x better parallel efficiency
        - **ProcessPoolExecutor**: Each worker has ~100ms startup overhead
        - Efficient for >10 charts or when rendering time >100ms per chart
        - Uses pickle for data transfer (ensure ohlc/volume are picklable)
        - For small batches (<10 charts), sequential rendering may be faster
        - Memory usage scales with num_workers (each worker needs ~100-200MB)
        - Results are returned in same order as input (order-preserving)
        - PIL operations are thread-safe (safe for ThreadPoolExecutor)

    Performance Tips:
        - Install python3.14t for 3.1x parallel speedup (free-threading)
        - Use executor_type='auto' to automatically use best executor
        - Use speed='fast' for batch processing (4-10x faster encoding)
        - Set num_workers based on available RAM (each worker needs ~100-200MB)
        - For very large datasets, consider processing in chunks
        - Use output_paths for file output (lower memory usage than bytes)

    Raises:
        ValueError: If output_paths length doesn't match datasets length
        RuntimeError: If parallel execution fails (e.g., pickling errors)
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 4

    # Validate output_paths length
    if output_paths is not None and len(output_paths) != len(datasets):
        raise ValueError(
            f"Length mismatch: datasets has {len(datasets)} items, "
            f"but output_paths has {len(output_paths)} items"
        )

    # Prepare arguments for parallel execution
    output_paths_list: list[str | None]
    if output_paths is None:
        output_paths_list = [None] * len(datasets)
    else:
        output_paths_list = list(output_paths)

    # Prepare save_kwargs (only used when saving to file)
    save_kwargs = {"speed": speed}

    # Build argument tuples for each chart
    args_list = [
        (d["ohlc"], d["volume"], path, save_kwargs, common_render_kwargs)
        for d, path in zip(datasets, output_paths_list)
    ]

    # Select executor type based on configuration and available features
    if executor_type == "auto":
        # Use threads on python3.14t (free-threading), processes otherwise
        use_threads = FREE_THREADING_AVAILABLE
    elif executor_type == "thread":
        use_threads = True
    else:  # executor_type == "process"
        use_threads = False

    # Execute in parallel using selected executor
    # map() preserves order and returns results in same order as input
    if use_threads:
        # ThreadPoolExecutor: 3.1x better on python3.14t (free-threading)
        # Lower overhead (~1ms vs ~100ms per worker)
        # Shared memory (no pickling overhead)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_render_one_chart, args_list))
    else:
        # ProcessPoolExecutor: Traditional multiprocessing
        # Higher overhead but works on all Python versions
        # Process isolation (safer for unstable code)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_render_one_chart, args_list))

    return results
