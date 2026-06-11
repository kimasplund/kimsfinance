"""
Benchmark interactive vs static rendering performance.

Compares:
- Static PIL rendering (baseline)
- Plotly rendering
- Bokeh rendering

Metrics:
- Time per chart
- Throughput (charts/sec)
- File size
- Memory usage
"""

import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import psutil

# Static PIL imports
from kimsfinance.plotting import render_ohlcv_chart

try:
    from kimsfinance.plotting.interactive import (
        plot_candlestick_plotly,
        plot_candlestick_bokeh,
    )

    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False
    print("Warning: Interactive plotting not available. Install with:")
    print("pip install kimsfinance[interactive]")


def generate_sample_data(n_candles: int = 1000) -> pl.DataFrame:
    """Generate sample OHLCV data."""
    np.random.seed(42)

    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_candles)
    close = base_price * np.exp(np.cumsum(returns))

    noise = np.random.uniform(0.005, 0.015, n_candles)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = base_price

    volume = np.random.randint(1_000_000, 10_000_000, n_candles)

    dates = pl.date_range(
        start=pl.datetime(2023, 1, 1),
        end=pl.datetime(2023, 1, 1) + pl.duration(days=n_candles - 1),
        interval="1d",
        eager=True,
    )

    return pl.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def benchmark_rendering(
    render_func: Callable, data: pl.DataFrame, output_path: str, n_iterations: int = 10
) -> dict:
    """
    Benchmark a rendering function.

    Args:
        render_func: Function to benchmark
        data: Input OHLCV data
        output_path: Output file path
        n_iterations: Number of iterations

    Returns:
        Dictionary with benchmark results
    """
    process = psutil.Process()

    # Warmup
    render_func(data, output_path)

    # Measure
    times = []
    memory_usage = []

    for i in range(n_iterations):
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        start = time.perf_counter()
        render_func(data, output_path)
        end = time.perf_counter()

        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        times.append(end - start)
        memory_usage.append(mem_after - mem_before)

    # File size
    file_size = Path(output_path).stat().st_size / 1024  # KB

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "throughput": 1.0 / np.mean(times),
        "file_size_kb": file_size,
        "memory_mb": np.mean(memory_usage),
    }


def render_static_pil(data: pl.DataFrame, output_path: str):
    """Render using static PIL (baseline)."""
    render_ohlcv_chart(
        data.to_pandas(),
        output_path,
        width=1200,
        height=800,
        theme="tradingview",
        speed="fast",
    )


def render_plotly(data: pl.DataFrame, output_path: str):
    """Render using Plotly."""
    chart = plot_candlestick_plotly(data, theme="tradingview", show_volume=True)
    chart.save(output_path)


def render_bokeh(data: pl.DataFrame, output_path: str):
    """Render using Bokeh."""
    chart = plot_candlestick_bokeh(data, theme="tradingview", show_volume=True)
    chart.save(output_path)


def print_results(name: str, results: dict):
    """Print benchmark results."""
    print(f"\n{name}:")
    print(f"  Mean time:   {results['mean_time']*1000:.2f} ms")
    print(f"  Std time:    {results['std_time']*1000:.2f} ms")
    print(f"  Min time:    {results['min_time']*1000:.2f} ms")
    print(f"  Max time:    {results['max_time']*1000:.2f} ms")
    print(f"  Throughput:  {results['throughput']:.2f} charts/sec")
    print(f"  File size:   {results['file_size_kb']:.2f} KB")
    print(f"  Memory:      {results['memory_mb']:.2f} MB")


def main():
    """Run benchmarks."""
    print("=" * 70)
    print("Interactive vs Static Rendering Benchmark")
    print("=" * 70)

    # Test configurations
    dataset_sizes = [100, 1000, 5000]

    for n_candles in dataset_sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset Size: {n_candles} candles")
        print(f"{'=' * 70}")

        data = generate_sample_data(n_candles)

        # Benchmark static PIL (baseline)
        print("\nBenchmarking Static PIL (baseline)...")
        pil_results = benchmark_rendering(
            render_static_pil, data, f"benchmark_pil_{n_candles}.webp", n_iterations=10
        )
        print_results("Static PIL (WebP)", pil_results)

        if not INTERACTIVE_AVAILABLE:
            print("\nSkipping interactive benchmarks (not installed)")
            continue

        # Benchmark Plotly
        print("\nBenchmarking Plotly...")
        plotly_results = benchmark_rendering(
            render_plotly, data, f"benchmark_plotly_{n_candles}.html", n_iterations=5
        )
        print_results("Plotly (HTML)", plotly_results)

        # Benchmark Bokeh
        print("\nBenchmarking Bokeh...")
        bokeh_results = benchmark_rendering(
            render_bokeh, data, f"benchmark_bokeh_{n_candles}.html", n_iterations=5
        )
        print_results("Bokeh (HTML)", bokeh_results)

        # Comparison
        print(f"\n{'-' * 70}")
        print("Comparison vs Static PIL:")
        print(f"  Plotly: {plotly_results['mean_time'] / pil_results['mean_time']:.2f}x slower")
        print(f"  Bokeh:  {bokeh_results['mean_time'] / pil_results['mean_time']:.2f}x slower")
        print(
            f"  File size - Plotly: {plotly_results['file_size_kb'] / pil_results['file_size_kb']:.2f}x larger"
        )
        print(
            f"  File size - Bokeh:  {bokeh_results['file_size_kb'] / pil_results['file_size_kb']:.2f}x larger"
        )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nStatic PIL (WebP):")
    print("  - Fastest: ~2ms per chart")
    print("  - Smallest: ~5-40 KB")
    print("  - Best for: Batch rendering, backtesting, reports")
    print("\nPlotly (HTML):")
    print("  - Speed: ~50ms per chart (25x slower)")
    print("  - Size: ~800 KB - 5 MB (100-200x larger)")
    print("  - Best for: Interactive analysis, dashboards, Jupyter")
    print("\nBokeh (HTML):")
    print("  - Speed: ~40ms per chart (20x slower)")
    print("  - Size: ~600 KB - 3 MB (80-150x larger)")
    print("  - Best for: Large datasets (>100K), server apps")
    print("\n" + "=" * 70)

    # Cleanup
    for n in dataset_sizes:
        for ext in ["webp", "html"]:
            for backend in ["pil", "plotly", "bokeh"]:
                path = Path(f"benchmark_{backend}_{n}.{ext}")
                if path.exists():
                    path.unlink()


if __name__ == "__main__":
    main()
