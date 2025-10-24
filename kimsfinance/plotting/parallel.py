"""Parallel rendering support for batch chart generation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Executor
from typing import Any
import os
import io


def _get_optimal_executor() -> type[Executor]:
    """
    Select the optimal executor based on Python version and GIL status.

    Python 3.14+ with free-threading (no-GIL) enabled uses ThreadPoolExecutor
    for 5x better performance (zero-copy data sharing, no pickle overhead).

    For GIL-enabled Python (< 3.14 or GIL not disabled), falls back to
    ProcessPoolExecutor for true parallelism.

    Returns:
        Executor class (ThreadPoolExecutor or ProcessPoolExecutor)

    Notes:
        - Free-threading requires Python 3.14+ built with --disable-gil
        - Check with: python3.14t (t = threads build)
        - ThreadPoolExecutor benefits:
          * No pickle serialization overhead
          * Shared memory (zero-copy)
          * Faster worker startup (<1ms vs ~100ms)
          * Lower memory usage (1x vs Nx process size)
        - ProcessPoolExecutor benefits:
          * Works on all Python versions
          * True parallelism with GIL-enabled Python
          * Isolated memory spaces (safer)
    """
    from kimsfinance.core import EngineManager

    if EngineManager.supports_free_threading():
        return ThreadPoolExecutor
    return ProcessPoolExecutor


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


def render_charts_parallel(
    datasets: list[dict[str, Any]],
    output_paths: list[str] | None = None,
    num_workers: int | None = None,
    speed: str = "fast",
    **common_render_kwargs: Any,
) -> list[str | bytes]:
    """
    Render multiple charts in parallel using optimal executor (threads or processes).

    Automatically selects the best executor:
    - Python 3.14t (free-threading): ThreadPoolExecutor (5x faster)
    - Standard Python: ProcessPoolExecutor (compatible with all versions)

    This function distributes chart rendering across multiple CPU cores
    for maximum throughput. Each chart is rendered independently,
    enabling linear scaling with CPU cores.

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
        **common_render_kwargs: Common rendering options for all charts
            Examples: theme, width, height, enable_antialiasing, show_grid

    Returns:
        List of output paths (if output_paths provided) or PNG bytes
        Results are in same order as input datasets

    Examples:
        >>> # Save to files in parallel
        >>> datasets = [
        ...     {'ohlc': ohlc1, 'volume': vol1},
        ...     {'ohlc': ohlc2, 'volume': vol2},
        ...     {'ohlc': ohlc3, 'volume': vol3},
        ... ]
        >>> paths = [f"chart_{i}.webp" for i, _ in enumerate(datasets)]
        >>> result = render_charts_parallel(datasets, paths, speed='fast')
        >>> # Returns: ['chart_0.webp', 'chart_1.webp', 'chart_2.webp']

        >>> # In-memory rendering (returns PNG bytes)
        >>> png_bytes = render_charts_parallel(datasets, num_workers=8)
        >>> # Returns: [b'\\x89PNG...', b'\\x89PNG...', b'\\x89PNG...']

        >>> # Custom rendering options
        >>> result = render_charts_parallel(
        ...     datasets,
        ...     paths,
        ...     theme='modern',
        ...     width=1920,
        ...     height=1080,
        ...     speed='balanced'
        ... )

    Notes:
        - Executor selection is automatic (no configuration needed):
          * Python 3.14t: Uses ThreadPoolExecutor (zero-copy, <1ms startup)
          * Other Python: Uses ProcessPoolExecutor (pickle, ~100ms startup)
        - ProcessPoolExecutor overhead:
          * Worker startup: ~100ms per process
          * Pickle serialization for data transfer
          * Memory duplication (each process has separate memory)
        - ThreadPoolExecutor benefits (Python 3.14t only):
          * Zero-copy data sharing (no pickle)
          * <1ms worker startup
          * Shared memory (1x memory usage vs Nx for processes)
        - Efficient for >10 charts or when rendering time >100ms per chart
        - For small batches (<10 charts), sequential rendering may be faster
        - Results are returned in same order as input (order-preserving)

    Performance Tips:
        - Use speed='fast' for batch processing (4-10x faster encoding)
        - Set num_workers based on available RAM (each worker needs ~100-200MB)
        - For very large datasets, consider processing in chunks
        - Use output_paths for file output (lower memory usage than bytes)

    Raises:
        ValueError: If output_paths length doesn't match datasets length
        RuntimeError: If multiprocessing fails (e.g., pickling errors)
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

    # Select optimal executor (ThreadPoolExecutor for Python 3.14t, else ProcessPoolExecutor)
    executor_class = _get_optimal_executor()

    # Execute in parallel using optimal executor
    # map() preserves order and returns results in same order as input
    with executor_class(max_workers=num_workers) as executor:
        results = list(executor.map(_render_one_chart, args_list))

    return results
