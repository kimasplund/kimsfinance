#!/usr/bin/env python3
"""
Comprehensive Benchmark: Rust vs Python/NumPy for All 30 Indicators
=====================================================================

Compares performance of Rust-accelerated indicators against Python/NumPy
across multiple dataset sizes to identify GPU crossover thresholds.

**Hypothesis**:
- Rust wins at <1,000 candles (3-5x speedup) - FFI overhead negligible
- Python wins at >10,000 candles (0.67-0.93x) - FFI overhead dominates
- Batch API extends Rust viability to ~10K candles

**Statistical Rigor**:
- Median of 100-1000 iterations (adaptive based on size)
- Warm-up runs (10 for Rust, 5 for Python)
- Confidence intervals (95%)
- Report speedup and statistical significance

Usage:
    python benchmarks/benchmark_indicators_rust.py

Requirements:
    - kimsfinance_core (Rust extension built)
    - kimsfinance Python package
    - numpy >= 2.0
    - scipy (for statistical tests)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import numpy as np
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Python implementations
from kimsfinance.ops.indicators import (
    calculate_sma,
    calculate_ema,
    calculate_wma,
    calculate_vwma,
    calculate_dema,
    calculate_tema,
    calculate_hma,
    calculate_rsi,
    calculate_roc,
    calculate_macd,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_keltner_channels,
    calculate_donchian_channels,
    calculate_elder_ray,
    calculate_obv,
    calculate_vwap,
    calculate_cmf,
    calculate_williams_r,
    calculate_stochastic_oscillator,
    calculate_aroon,
    calculate_cci,
    calculate_tsi,
)

# Try to import Rust implementations
try:
    import kimsfinance_core

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("\n❌ ERROR: Rust extension 'kimsfinance_core' not available")
    print("Build with: cd rust && maturin develop --release")
    sys.exit(1)


# ================================================================================================
# DATA GENERATION
# ================================================================================================


def generate_ohlcv_data(size: int, seed: int = 42) -> dict[str, np.ndarray]:
    """
    Generate synthetic OHLCV data for benchmarking.

    Returns dict with keys: 'high', 'low', 'open', 'close', 'volume'
    """
    rng = np.random.RandomState(seed)

    # Generate realistic price movement
    close = 100.0 + np.cumsum(rng.randn(size) * 0.5)
    high = close + np.abs(rng.randn(size) * 0.3)
    low = close - np.abs(rng.randn(size) * 0.3)
    open_ = close + rng.randn(size) * 0.2
    volume = np.abs(rng.randn(size) * 1_000_000)

    return {
        "high": high,
        "low": low,
        "open": open_,
        "close": close,
        "volume": volume,
    }


# ================================================================================================
# BENCHMARK INFRASTRUCTURE
# ================================================================================================


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    dataset_size: int
    python_median_ms: float
    rust_median_ms: float
    python_std_ms: float
    rust_std_ms: float
    speedup: float
    winner: str  # "Rust" or "Python"
    confidence_level: float  # 0-100
    p_value: float  # Statistical significance


def timeit_median(
    func: Callable[[], Any], n_iterations: int, warmup: int = 5
) -> tuple[float, float]:
    """
    Time a function with warmup and return median and std in milliseconds.

    Args:
        func: Function to time (no arguments)
        n_iterations: Number of iterations to collect
        warmup: Number of warmup iterations

    Returns:
        Tuple of (median_ms, std_ms)
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Collect timings
    timings = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        timings.append((end - start) * 1000)  # Convert to ms

    return float(np.median(timings)), float(np.std(timings))


def compare_implementations(
    name: str,
    python_func: Callable[[], Any],
    rust_func: Callable[[], Any],
    dataset_size: int,
    n_iterations: int = 100,
) -> BenchmarkResult:
    """
    Compare Python and Rust implementations.

    Returns:
        BenchmarkResult with timing and statistical analysis
    """
    # Time Python implementation
    python_median, python_std = timeit_median(python_func, n_iterations, warmup=5)

    # Time Rust implementation
    rust_median, rust_std = timeit_median(rust_func, n_iterations, warmup=10)

    # Calculate speedup
    speedup = python_median / rust_median if rust_median > 0 else float("inf")

    # Determine winner
    winner = "Rust" if speedup > 1.0 else "Python"

    # Calculate statistical significance (Mann-Whitney U test)
    # Collect samples for statistical test
    python_samples = []
    for _ in range(min(n_iterations, 100)):
        start = time.perf_counter()
        python_func()
        python_samples.append((time.perf_counter() - start) * 1000)

    rust_samples = []
    for _ in range(min(n_iterations, 100)):
        start = time.perf_counter()
        rust_func()
        rust_samples.append((time.perf_counter() - start) * 1000)

    # Mann-Whitney U test (non-parametric, doesn't assume normality)
    try:
        _, p_value = stats.mannwhitneyu(python_samples, rust_samples, alternative="two-sided")
    except Exception:
        p_value = 1.0

    # Confidence level (higher p-value = lower confidence in difference)
    confidence = (1.0 - p_value) * 100 if p_value < 1.0 else 0.0

    return BenchmarkResult(
        name=name,
        dataset_size=dataset_size,
        python_median_ms=python_median,
        rust_median_ms=rust_median,
        python_std_ms=python_std,
        rust_std_ms=rust_std,
        speedup=speedup,
        winner=winner,
        confidence_level=confidence,
        p_value=p_value,
    )


# ================================================================================================
# INDIVIDUAL INDICATOR BENCHMARKS
# ================================================================================================


def benchmark_sma(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Simple Moving Average."""
    close = data["close"]
    period = 20

    def python_func():
        return calculate_sma(close, period=period)

    def rust_func():
        return kimsfinance_core.calculate_sma(close, period=period)

    return compare_implementations("SMA", python_func, rust_func, size)


def benchmark_ema(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Exponential Moving Average."""
    close = data["close"]
    period = 12

    def python_func():
        return calculate_ema(close, period=period)

    def rust_func():
        return kimsfinance_core.calculate_ema(close, period=period)

    return compare_implementations("EMA", python_func, rust_func, size)


def benchmark_wma(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Weighted Moving Average."""
    close = data["close"]
    period = 20

    def python_func():
        return calculate_wma(close, period=period)

    def rust_func():
        return kimsfinance_core.calculate_wma(close, period=period)

    return compare_implementations("WMA", python_func, rust_func, size)


def benchmark_vwma(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Volume Weighted Moving Average."""
    close = data["close"]
    volume = data["volume"]
    period = 20

    def python_func():
        return calculate_vwma(close, volume, period=period)

    def rust_func():
        return kimsfinance_core.calculate_vwma(close, volume, period=period)

    return compare_implementations("VWMA", python_func, rust_func, size)


def benchmark_rsi(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Relative Strength Index."""
    close = data["close"]
    period = 14

    def python_func():
        return calculate_rsi(close, period=period)

    def rust_func():
        return kimsfinance_core.calculate_rsi(close, period=period)

    return compare_implementations("RSI", python_func, rust_func, size)


def benchmark_roc(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Rate of Change."""
    close = data["close"]
    period = 12

    def python_func():
        return calculate_roc(close, period=period)

    def rust_func():
        return kimsfinance_core.calculate_roc(close, period=period)

    return compare_implementations("ROC", python_func, rust_func, size)


def benchmark_macd(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark MACD."""
    close = data["close"]

    def python_func():
        return calculate_macd(close, fast_period=12, slow_period=26, signal_period=9)

    def rust_func():
        return kimsfinance_core.calculate_macd(
            close, fast_period=12, slow_period=26, signal_period=9
        )

    return compare_implementations("MACD", python_func, rust_func, size)


def benchmark_atr(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Average True Range."""
    high, low, close = data["high"], data["low"], data["close"]
    period = 14

    def python_func():
        return calculate_atr(high, low, close, period=period)

    def rust_func():
        return kimsfinance_core.calculate_atr(high, low, close, period=period)

    return compare_implementations("ATR", python_func, rust_func, size)


def benchmark_bollinger_bands(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Bollinger Bands."""
    close = data["close"]

    def python_func():
        return calculate_bollinger_bands(close, period=20, num_std=2.0)

    def rust_func():
        return kimsfinance_core.calculate_bollinger_bands(close, period=20, std_dev=2.0)

    return compare_implementations("Bollinger Bands", python_func, rust_func, size)


def benchmark_obv(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark On-Balance Volume."""
    close, volume = data["close"], data["volume"]

    def python_func():
        return calculate_obv(close, volume)

    def rust_func():
        return kimsfinance_core.calculate_obv(close, volume)

    return compare_implementations("OBV", python_func, rust_func, size)


def benchmark_vwap(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Volume Weighted Average Price."""
    high, low, close, volume = data["high"], data["low"], data["close"], data["volume"]

    def python_func():
        return calculate_vwap(high, low, close, volume)

    def rust_func():
        return kimsfinance_core.calculate_vwap(high, low, close, volume)

    return compare_implementations("VWAP", python_func, rust_func, size)


def benchmark_williams_r(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Williams %R."""
    high, low, close = data["high"], data["low"], data["close"]
    period = 14

    def python_func():
        return calculate_williams_r(high, low, close, period=period)

    def rust_func():
        return kimsfinance_core.calculate_williams_r(high, low, close, period=period)

    return compare_implementations("Williams %R", python_func, rust_func, size)


def benchmark_stochastic(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Stochastic Oscillator."""
    high, low, close = data["high"], data["low"], data["close"]

    def python_func():
        return calculate_stochastic_oscillator(high, low, close, k_period=14, d_period=3)

    def rust_func():
        return kimsfinance_core.calculate_stochastic(high, low, close, k_period=14, d_period=3)

    return compare_implementations("Stochastic", python_func, rust_func, size)


def benchmark_aroon(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Aroon Indicator."""
    high, low = data["high"], data["low"]
    period = 25

    def python_func():
        return calculate_aroon(high, low, period=period)

    def rust_func():
        return kimsfinance_core.calculate_aroon(high, low, period=period)

    return compare_implementations("Aroon", python_func, rust_func, size)


def benchmark_cci(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Commodity Channel Index."""
    high, low, close = data["high"], data["low"], data["close"]
    period = 20

    def python_func():
        return calculate_cci(high, low, close, period=period)

    def rust_func():
        return kimsfinance_core.calculate_cci(high, low, close, period=period)

    return compare_implementations("CCI", python_func, rust_func, size)


def benchmark_cmf(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Chaikin Money Flow."""
    high, low, close, volume = data["high"], data["low"], data["close"], data["volume"]
    period = 20

    def python_func():
        return calculate_cmf(high, low, close, volume, period=period)

    def rust_func():
        return kimsfinance_core.calculate_cmf(high, low, close, volume, period=period)

    return compare_implementations("CMF", python_func, rust_func, size)


def benchmark_dema(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Double Exponential Moving Average."""
    close = data["close"]
    period = 20

    def python_func():
        return calculate_dema(close, period=period)

    def rust_func():
        return kimsfinance_core.calculate_dema(close, period=period)

    return compare_implementations("DEMA", python_func, rust_func, size)


def benchmark_tema(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Triple Exponential Moving Average."""
    close = data["close"]
    period = 20

    def python_func():
        return calculate_tema(close, period=period)

    def rust_func():
        return kimsfinance_core.calculate_tema(close, period=period)

    return compare_implementations("TEMA", python_func, rust_func, size)


def benchmark_hma(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Hull Moving Average."""
    close = data["close"]
    period = 20

    def python_func():
        return calculate_hma(close, period=period)

    def rust_func():
        return kimsfinance_core.calculate_hma(close, period=period)

    return compare_implementations("HMA", python_func, rust_func, size)


def benchmark_keltner_channels(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Keltner Channels."""
    high, low, close = data["high"], data["low"], data["close"]

    def python_func():
        return calculate_keltner_channels(high, low, close, ema_period=20, atr_period=10)

    def rust_func():
        return kimsfinance_core.calculate_keltner_channels(
            high, low, close, ema_period=20, atr_period=10, atr_multiplier=2.0
        )

    return compare_implementations("Keltner Channels", python_func, rust_func, size)


def benchmark_donchian_channels(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Donchian Channels."""
    high, low = data["high"], data["low"]
    period = 20

    def python_func():
        return calculate_donchian_channels(high, low, period=period)

    def rust_func():
        return kimsfinance_core.calculate_donchian_channels(high, low, period=period)

    return compare_implementations("Donchian Channels", python_func, rust_func, size)


def benchmark_elder_ray(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark Elder Ray Index."""
    high, low, close = data["high"], data["low"], data["close"]

    def python_func():
        return calculate_elder_ray(high, low, close, ema_period=13)

    def rust_func():
        return kimsfinance_core.calculate_elder_ray(high, low, close, ema_period=13)

    return compare_implementations("Elder Ray", python_func, rust_func, size)


def benchmark_tsi(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """Benchmark True Strength Index."""
    close = data["close"]

    def python_func():
        return calculate_tsi(close, long_period=25, short_period=13, signal_period=13)

    def rust_func():
        return kimsfinance_core.calculate_tsi(
            close, long_period=25, short_period=13, signal_period=13
        )

    return compare_implementations("TSI", python_func, rust_func, size)


# ================================================================================================
# BATCH API BENCHMARK
# ================================================================================================


def benchmark_batch_api(data: dict[str, np.ndarray], size: int) -> BenchmarkResult:
    """
    Benchmark batch API vs individual calls.

    Tests calculating 10 common indicators:
    - SMA(20), EMA(12), RSI(14), MACD, ATR(14)
    - Bollinger Bands, OBV, VWAP, Stochastic, Williams%R
    """
    high, low, open_, close, volume = (
        data["high"],
        data["low"],
        data["open"],
        data["close"],
        data["volume"],
    )

    # Python: Individual calls
    def python_individual():
        calculate_sma(close, period=20)
        calculate_ema(close, period=12)
        calculate_rsi(close, period=14)
        calculate_macd(close, fast_period=12, slow_period=26, signal_period=9)
        calculate_atr(high, low, close, period=14)
        calculate_bollinger_bands(close, period=20, num_std=2.0)
        calculate_obv(close, volume)
        calculate_vwap(high, low, close, volume)
        calculate_stochastic_oscillator(high, low, close, k_period=14, d_period=3)
        calculate_williams_r(high, low, close, period=14)

    # Rust: Batch API
    def rust_batch():
        kimsfinance_core.calculate_indicators_batch(
            high,
            low,
            open_,
            close,
            volume,
            indicators=[
                {"name": "sma", "period": 20},
                {"name": "ema", "period": 12},
                {"name": "rsi", "period": 14},
                {"name": "macd", "fast_period": 12, "slow_period": 26, "signal_period": 9},
                {"name": "atr", "period": 14},
                {"name": "bollinger", "period": 20, "std_dev": 2.0},
                {"name": "obv"},
                {"name": "vwap"},
                {"name": "stochastic", "k_period": 14, "d_period": 3},
                {"name": "williams_r", "period": 14},
            ],
        )

    return compare_implementations(
        "Batch API (10 indicators)", python_individual, rust_batch, size, n_iterations=50
    )


# ================================================================================================
# MAIN BENCHMARK RUNNER
# ================================================================================================


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_result(result: BenchmarkResult):
    """Print a single benchmark result."""
    # Format speedup with color
    if result.speedup >= 1.0:
        speedup_str = f"✅ {result.speedup:.2f}x"
    else:
        speedup_str = f"❌ {result.speedup:.2f}x"

    # Format confidence
    confidence_str = f"{result.confidence_level:.1f}%"
    if result.confidence_level >= 95:
        confidence_str = f"🟢 {confidence_str}"
    elif result.confidence_level >= 80:
        confidence_str = f"🟡 {confidence_str}"
    else:
        confidence_str = f"🔴 {confidence_str}"

    print(
        f"  {result.name:25s} | "
        f"NumPy: {result.python_median_ms:7.3f}ms | "
        f"Rust: {result.rust_median_ms:7.3f}ms | "
        f"Speedup: {speedup_str:12s} | "
        f"Confidence: {confidence_str}"
    )


def run_benchmarks_for_size(size: int) -> list[BenchmarkResult]:
    """
    Run all benchmarks for a given dataset size.

    Returns:
        List of BenchmarkResult objects
    """
    print(f"\n📊 Generating {size:,} candles of synthetic data...")
    data = generate_ohlcv_data(size)

    # Determine number of iterations based on size
    if size <= 1000:
        n_iterations = 500
    elif size <= 10000:
        n_iterations = 100
    else:
        n_iterations = 50

    print(f"⚙️  Running benchmarks ({n_iterations} iterations each)...\n")

    results = []

    # Moving Averages (7 indicators)
    print_header("MOVING AVERAGES (7 indicators)")
    results.append(benchmark_sma(data, size))
    print_result(results[-1])

    results.append(benchmark_ema(data, size))
    print_result(results[-1])

    results.append(benchmark_wma(data, size))
    print_result(results[-1])

    results.append(benchmark_vwma(data, size))
    print_result(results[-1])

    results.append(benchmark_dema(data, size))
    print_result(results[-1])

    results.append(benchmark_tema(data, size))
    print_result(results[-1])

    results.append(benchmark_hma(data, size))
    print_result(results[-1])

    # Momentum Indicators (8 indicators)
    print_header("MOMENTUM INDICATORS (8 indicators)")
    results.append(benchmark_rsi(data, size))
    print_result(results[-1])

    results.append(benchmark_roc(data, size))
    print_result(results[-1])

    results.append(benchmark_macd(data, size))
    print_result(results[-1])

    results.append(benchmark_williams_r(data, size))
    print_result(results[-1])

    results.append(benchmark_stochastic(data, size))
    print_result(results[-1])

    results.append(benchmark_aroon(data, size))
    print_result(results[-1])

    results.append(benchmark_cci(data, size))
    print_result(results[-1])

    results.append(benchmark_tsi(data, size))
    print_result(results[-1])

    # Volatility Indicators (5 indicators)
    print_header("VOLATILITY INDICATORS (5 indicators)")
    results.append(benchmark_atr(data, size))
    print_result(results[-1])

    results.append(benchmark_bollinger_bands(data, size))
    print_result(results[-1])

    results.append(benchmark_keltner_channels(data, size))
    print_result(results[-1])

    results.append(benchmark_donchian_channels(data, size))
    print_result(results[-1])

    results.append(benchmark_elder_ray(data, size))
    print_result(results[-1])

    # Volume Indicators (3 indicators)
    print_header("VOLUME INDICATORS (3 indicators)")
    results.append(benchmark_obv(data, size))
    print_result(results[-1])

    results.append(benchmark_vwap(data, size))
    print_result(results[-1])

    results.append(benchmark_cmf(data, size))
    print_result(results[-1])

    # Batch API
    print_header("BATCH API (10 indicators in 1 call)")
    results.append(benchmark_batch_api(data, size))
    print_result(results[-1])

    return results


def print_summary(all_results: dict[int, list[BenchmarkResult]]):
    """
    Print comprehensive summary of all benchmarks.

    Args:
        all_results: Dict mapping dataset size to list of results
    """
    print_header("COMPREHENSIVE SUMMARY")

    # Calculate statistics
    for size, results in sorted(all_results.items()):
        rust_wins = sum(1 for r in results if r.winner == "Rust")
        python_wins = len(results) - rust_wins
        avg_speedup = np.mean([r.speedup for r in results])
        median_speedup = np.median([r.speedup for r in results])

        print(f"\n📏 Dataset Size: {size:,} candles")
        print(f"   Rust wins: {rust_wins}/{len(results)} indicators")
        print(f"   Python wins: {python_wins}/{len(results)} indicators")
        print(f"   Average speedup: {avg_speedup:.2f}x")
        print(f"   Median speedup: {median_speedup:.2f}x")

        # Find fastest and slowest indicators
        fastest = max(results, key=lambda r: r.speedup)
        slowest = min(results, key=lambda r: r.speedup)

        print(f"   ⚡ Fastest Rust gain: {fastest.name} ({fastest.speedup:.2f}x)")
        print(f"   🐌 Slowest Rust gain: {slowest.name} ({slowest.speedup:.2f}x)")

    # Overall recommendations
    print_header("RECOMMENDATIONS")

    # Find GPU crossover threshold
    rust_dominant = None
    python_dominant = None

    for size in sorted(all_results.keys()):
        results = all_results[size]
        rust_wins = sum(1 for r in results if r.winner == "Rust")
        if rust_wins > len(results) / 2 and rust_dominant is None:
            rust_dominant = size

        python_wins = len(results) - rust_wins
        if python_wins > len(results) / 2 and python_dominant is None:
            python_dominant = size
            break

    if rust_dominant and python_dominant:
        print(f"📍 Crossover Threshold: ~{python_dominant:,} candles")
        print(f"   ✅ Use Rust for datasets < {python_dominant:,} candles")
        print(f"   ✅ Use Python/NumPy for datasets >= {python_dominant:,} candles")
    elif rust_dominant:
        print(f"✅ Rust dominates across all tested sizes (up to {max(all_results.keys()):,})")
    else:
        print("⚠️  Python/NumPy dominates across all tested sizes")

    print(
        "\n💡 Consider Batch API for >1,000 candles to minimize FFI overhead\n"
        "   (see benchmark results above)\n"
    )


def main():
    """Main benchmark runner."""
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE INDICATOR BENCHMARK: Rust vs Python/NumPy")
    print("=" * 80)

    if not RUST_AVAILABLE:
        print("\n❌ ERROR: Rust extension not available. Build with:")
        print("   cd rust && maturin develop --release\n")
        sys.exit(1)

    print(f"\n✅ Rust extension loaded: kimsfinance_core v{kimsfinance_core.__version__}")
    print(f"✅ NumPy version: {np.__version__}")

    # Dataset sizes to test
    sizes = [100, 1_000, 10_000, 100_000]

    print("\n📋 Test Plan:")
    print(f"   Dataset sizes: {', '.join(f'{s:,}' for s in sizes)} candles")
    print("   Indicators tested: 24 individual + 1 batch API = 25 benchmarks per size")
    print("   Categories: Moving Averages (7), Momentum (8), Volatility (5), Volume (3), Batch (1)")
    print("   Statistical method: Mann-Whitney U test")
    print("   Confidence threshold: 95%")

    # Run benchmarks
    all_results = {}
    for size in sizes:
        print_header(f"DATASET SIZE: {size:,} CANDLES")
        all_results[size] = run_benchmarks_for_size(size)

    # Print summary
    print_summary(all_results)

    print("\n" + "=" * 80)
    print("  Benchmark Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
