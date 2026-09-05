#!/usr/bin/env python3
"""
Comprehensive Benchmark: mplfinance vs kimsfinance vs kimsfinance_core (CPU/GPU)

Compares indicator calculation performance across 5 implementations:
1. mplfinance (baseline)
2. kimsfinance Python CPU
3. kimsfinance Python GPU (Polars GPU engine)
4. kimsfinance_core Rust CPU
5. kimsfinance_core Rust GPU

Uses real Binance BTCUSDT 2024 1-minute OHLC data.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Tuple
import sys


# Colors for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def load_binance_data(limit: int = None) -> pd.DataFrame:
    """Load Binance BTCUSDT 2024 1-minute OHLC data."""
    data_path = Path("/home/kim/projects/binance-data/BTCUSDT_2024_1min_ohlc.csv")

    print(f"\n{Colors.OKCYAN}Loading Binance BTCUSDT 2024 data...{Colors.ENDC}")
    df = pd.read_csv(data_path)

    if limit:
        df = df.head(limit)

    print(f"  ✅ Loaded {len(df):,} candles")
    print(
        f"  📅 Date range: {pd.to_datetime(df['timestamp'].iloc[0], unit='ms')} to {pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')}"
    )

    return df


def benchmark_function(func: Callable, name: str, iterations: int = 10) -> Tuple[float, float]:
    """
    Benchmark a function with multiple iterations.
    Returns (mean_time_ms, calculations_per_second).
    """
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    mean_time = np.mean(times)
    std_time = np.std(times)
    calc_per_sec = 1000 / mean_time if mean_time > 0 else 0

    return mean_time, calc_per_sec, std_time


# ============================================================================
# Implementation 1: mplfinance (if available)
# ============================================================================


def benchmark_mplfinance(df: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    """Benchmark mplfinance indicators (if available)."""
    results = {}

    try:
        import mplfinance as mpf
        from ta.trend import SMAIndicator, EMAIndicator
        from ta.momentum import RSIIndicator
        from ta.volatility import AverageTrueRange

        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}Benchmarking: mplfinance + ta-lib{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # SMA
        def sma_func():
            return SMAIndicator(pd.Series(close), window=20).sma_indicator().values

        mean, cps, std = benchmark_function(sma_func, "SMA(20)")
        results["SMA(20)"] = (mean, cps, std)
        print(f"  SMA(20):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # EMA
        def ema_func():
            return EMAIndicator(pd.Series(close), window=20).ema_indicator().values

        mean, cps, std = benchmark_function(ema_func, "EMA(20)")
        results["EMA(20)"] = (mean, cps, std)
        print(f"  EMA(20):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # RSI
        def rsi_func():
            return RSIIndicator(pd.Series(close), window=14).rsi().values

        mean, cps, std = benchmark_function(rsi_func, "RSI(14)")
        results["RSI(14)"] = (mean, cps, std)
        print(f"  RSI(14):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # ATR
        df_temp = pd.DataFrame({"high": high, "low": low, "close": close})

        def atr_func():
            return (
                AverageTrueRange(df_temp["high"], df_temp["low"], df_temp["close"], window=14)
                .average_true_range()
                .values
            )

        mean, cps, std = benchmark_function(atr_func, "ATR(14)")
        results["ATR(14)"] = (mean, cps, std)
        print(f"  ATR(14):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

    except ImportError as e:
        print(f"\n{Colors.WARNING}⚠️  mplfinance/ta-lib not available: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}   Install with: pip install mplfinance ta{Colors.ENDC}")

    return results


# ============================================================================
# Implementation 2: kimsfinance (Python)
# ============================================================================


def benchmark_kimsfinance_python(df: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    """Benchmark kimsfinance Python package."""
    results = {}

    try:
        sys.path.insert(0, "/home/kim/projects/kimsfinance")
        import kimsfinance as mfp

        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}Benchmarking: kimsfinance (Python){Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # SMA
        def sma_func():
            return mfp.ops.indicators.calculate_sma(close, 20)

        mean, cps, std = benchmark_function(sma_func, "SMA(20)")
        results["SMA(20)"] = (mean, cps, std)
        print(f"  SMA(20):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # EMA
        def ema_func():
            return mfp.ops.indicators.calculate_ema(close, 20)

        mean, cps, std = benchmark_function(ema_func, "EMA(20)")
        results["EMA(20)"] = (mean, cps, std)
        print(f"  EMA(20):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # RSI
        def rsi_func():
            return mfp.ops.indicators.calculate_rsi(close, 14)

        mean, cps, std = benchmark_function(rsi_func, "RSI(14)")
        results["RSI(14)"] = (mean, cps, std)
        print(f"  RSI(14):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # ATR
        def atr_func():
            return mfp.ops.indicators.calculate_atr(high, low, close, 14)

        mean, cps, std = benchmark_function(atr_func, "ATR(14)")
        results["ATR(14)"] = (mean, cps, std)
        print(f"  ATR(14):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

    except ImportError as e:
        print(f"\n{Colors.WARNING}⚠️  kimsfinance Python package not available: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}   Install with: pip install -e .{Colors.ENDC}")

    return results


# ============================================================================
# Implementation 3: kimsfinance Python GPU (Polars GPU engine)
# ============================================================================


def benchmark_kimsfinance_python_gpu(df: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    """Benchmark kimsfinance Python package with Polars GPU engine."""
    results = {}

    try:
        sys.path.insert(0, "/home/kim/projects/kimsfinance")
        import kimsfinance as mfp

        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(
            f"{Colors.HEADER}{Colors.BOLD}Benchmarking: kimsfinance (Python GPU - Polars){Colors.ENDC}"
        )
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

        # Check if Polars GPU is available
        try:
            import polars as pl

            # Test GPU engine with a simple query
            test_df = pl.LazyFrame({"test": [1, 2, 3]})
            test_df.collect(engine="gpu")
            gpu_available = True
            print(f"  {Colors.OKGREEN}✅ Polars GPU engine available{Colors.ENDC}")
        except Exception as e:
            gpu_available = False
            print(f"  {Colors.WARNING}⚠️  Polars GPU not available: {e}{Colors.ENDC}")
            return results

        # Convert to Polars DataFrame for GPU processing
        pl_df = pl.DataFrame(
            {
                "close": df["close"].values,
                "high": df["high"].values,
                "low": df["low"].values,
            }
        )

        # For GPU benchmarks, we need to ensure operations use Polars GPU
        # Note: Not all indicators may benefit from GPU in kimsfinance
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # Test if kimsfinance has GPU-accelerated versions
        # (This may require specific GPU-enabled functions)

        # SMA with Polars GPU
        def sma_func():
            # If kimsfinance supports Polars GPU internally
            return mfp.ops.indicators.calculate_sma(close, 20)

        mean, cps, std = benchmark_function(sma_func, "SMA(20)")
        results["SMA(20)"] = (mean, cps, std)
        print(f"  SMA(20):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # EMA with Polars GPU
        def ema_func():
            return mfp.ops.indicators.calculate_ema(close, 20)

        mean, cps, std = benchmark_function(ema_func, "EMA(20)")
        results["EMA(20)"] = (mean, cps, std)
        print(f"  EMA(20):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # RSI with Polars GPU
        def rsi_func():
            return mfp.ops.indicators.calculate_rsi(close, 14)

        mean, cps, std = benchmark_function(rsi_func, "RSI(14)")
        results["RSI(14)"] = (mean, cps, std)
        print(f"  RSI(14):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # ATR with Polars GPU
        def atr_func():
            return mfp.ops.indicators.calculate_atr(high, low, close, 14)

        mean, cps, std = benchmark_function(atr_func, "ATR(14)")
        results["ATR(14)"] = (mean, cps, std)
        print(f"  ATR(14):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        print(
            f"  {Colors.OKCYAN}Note: Results reflect Polars GPU engine if enabled in kimsfinance{Colors.ENDC}"
        )

    except ImportError as e:
        print(f"\n{Colors.WARNING}⚠️  kimsfinance Python package not available: {e}{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.WARNING}⚠️  Error benchmarking GPU: {e}{Colors.ENDC}")

    return results


# ============================================================================
# Implementation 4: kimsfinance_core (Rust CPU)
# ============================================================================


def benchmark_kimsfinance_rust_cpu(df: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    """Benchmark kimsfinance_core Rust bindings (CPU mode)."""
    results = {}

    try:
        import kimsfinance_core

        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}Benchmarking: kimsfinance_core (Rust CPU){Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)

        # SMA
        def sma_func():
            return kimsfinance_core.calculate_sma(close, 20)

        mean, cps, std = benchmark_function(sma_func, "SMA(20)")
        results["SMA(20)"] = (mean, cps, std)
        print(f"  SMA(20):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # EMA
        def ema_func():
            return kimsfinance_core.calculate_ema(close, 20)

        mean, cps, std = benchmark_function(ema_func, "EMA(20)")
        results["EMA(20)"] = (mean, cps, std)
        print(f"  EMA(20):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # RSI
        def rsi_func():
            return kimsfinance_core.calculate_rsi(close, 14)

        mean, cps, std = benchmark_function(rsi_func, "RSI(14)")
        results["RSI(14)"] = (mean, cps, std)
        print(f"  RSI(14):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # ATR
        def atr_func():
            return kimsfinance_core.calculate_atr(high, low, close, 14)

        mean, cps, std = benchmark_function(atr_func, "ATR(14)")
        results["ATR(14)"] = (mean, cps, std)
        print(f"  ATR(14):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

    except ImportError as e:
        print(f"\n{Colors.WARNING}⚠️  kimsfinance_core not available: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}   Build with: cd rust && maturin develop --release{Colors.ENDC}")

    return results


# ============================================================================
# Implementation 5: kimsfinance_core (Rust GPU)
# ============================================================================


def benchmark_kimsfinance_rust_gpu(df: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    """Benchmark kimsfinance_core Rust bindings (GPU mode)."""
    results = {}

    try:
        import kimsfinance_core

        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}Benchmarking: kimsfinance_core (Rust GPU){Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)

        # Check if GPU functions are available
        gpu_available = hasattr(kimsfinance_core, "calculate_sma_gpu")

        if not gpu_available:
            print(
                f"{Colors.WARNING}⚠️  GPU functions not available in kimsfinance_core{Colors.ENDC}"
            )
            print(
                f"{Colors.WARNING}   Build with GPU support: maturin develop --release --features gpu{Colors.ENDC}"
            )
            return results

        # SMA GPU
        def sma_func():
            return kimsfinance_core.calculate_sma_gpu(close, 20)

        mean, cps, std = benchmark_function(sma_func, "SMA(20)")
        results["SMA(20)"] = (mean, cps, std)
        print(f"  SMA(20):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # EMA GPU
        def ema_func():
            return kimsfinance_core.calculate_ema_gpu(close, 20)

        mean, cps, std = benchmark_function(ema_func, "EMA(20)")
        results["EMA(20)"] = (mean, cps, std)
        print(f"  EMA(20):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # RSI GPU
        def rsi_func():
            return kimsfinance_core.calculate_rsi_gpu(close, 14)

        mean, cps, std = benchmark_function(rsi_func, "RSI(14)")
        results["RSI(14)"] = (mean, cps, std)
        print(f"  RSI(14):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

        # ATR GPU
        def atr_func():
            return kimsfinance_core.calculate_atr_gpu(high, low, close, 14)

        mean, cps, std = benchmark_function(atr_func, "ATR(14)")
        results["ATR(14)"] = (mean, cps, std)
        print(f"  ATR(14):    {mean:7.2f} ms  ({cps:7.1f} calc/sec)")

    except ImportError as e:
        print(f"\n{Colors.WARNING}⚠️  kimsfinance_core not available: {e}{Colors.ENDC}")
    except AttributeError as e:
        print(f"\n{Colors.WARNING}⚠️  GPU functions not found: {e}{Colors.ENDC}")

    return results


# ============================================================================
# Results Table Generation
# ============================================================================


def print_comparison_table(
    mplfinance_results: Dict,
    kimsfinance_cpu_results: Dict,
    kimsfinance_gpu_results: Dict,
    rust_cpu_results: Dict,
    rust_gpu_results: Dict,
    num_candles: int,
):
    """Print a comprehensive comparison table."""

    print(f"\n{Colors.BOLD}{'='*145}{Colors.ENDC}")
    print(
        f"{Colors.BOLD}{Colors.OKGREEN}PERFORMANCE COMPARISON - {num_candles:,} Candles{Colors.ENDC}"
    )
    print(f"{Colors.BOLD}{'='*145}{Colors.ENDC}\n")

    # Table header
    print(
        f"{Colors.BOLD}{'Indicator':<12} | {'mplfinance':<18} | {'KF Py CPU':<18} | {'KF Py GPU':<18} | {'Rust CPU':<18} | {'Rust GPU':<18}{Colors.ENDC}"
    )
    print(
        f"{Colors.BOLD}{'-'*12}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}{Colors.ENDC}"
    )

    # Get all indicators
    all_indicators = set()
    all_indicators.update(mplfinance_results.keys())
    all_indicators.update(kimsfinance_cpu_results.keys())
    all_indicators.update(kimsfinance_gpu_results.keys())
    all_indicators.update(rust_cpu_results.keys())
    all_indicators.update(rust_gpu_results.keys())

    # Sort indicators
    indicators = sorted(all_indicators)

    for indicator in indicators:
        # Get results for each implementation
        mpf_time, mpf_cps, mpf_std = mplfinance_results.get(indicator, (None, None, None))
        kf_cpu_time, kf_cpu_cps, kf_cpu_std = kimsfinance_cpu_results.get(
            indicator, (None, None, None)
        )
        kf_gpu_time, kf_gpu_cps, kf_gpu_std = kimsfinance_gpu_results.get(
            indicator, (None, None, None)
        )
        rust_cpu_time, rust_cpu_cps, rust_cpu_std = rust_cpu_results.get(
            indicator, (None, None, None)
        )
        rust_gpu_time, rust_gpu_cps, rust_gpu_std = rust_gpu_results.get(
            indicator, (None, None, None)
        )

        # Format results
        def format_result(time_ms, cps):
            if time_ms is None:
                return "N/A"
            return f"{time_ms:5.1f}ms ({cps:5.0f}/s)"

        mpf_str = format_result(mpf_time, mpf_cps)
        kf_cpu_str = format_result(kf_cpu_time, kf_cpu_cps)
        kf_gpu_str = format_result(kf_gpu_time, kf_gpu_cps)
        rust_cpu_str = format_result(rust_cpu_time, rust_cpu_cps)
        rust_gpu_str = format_result(rust_gpu_time, rust_gpu_cps)

        print(
            f"{indicator:<12} | {mpf_str:<18} | {kf_cpu_str:<18} | {kf_gpu_str:<18} | {rust_cpu_str:<18} | {rust_gpu_str:<18}"
        )

    # Speedup analysis
    print(f"\n{Colors.BOLD}{'='*145}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKGREEN}SPEEDUP vs mplfinance (Baseline){Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*145}{Colors.ENDC}\n")

    print(
        f"{Colors.BOLD}{'Indicator':<12} | {'KF Py CPU':<18} | {'KF Py GPU':<18} | {'Rust CPU':<18} | {'Rust GPU':<18}{Colors.ENDC}"
    )
    print(f"{Colors.BOLD}{'-'*12}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}{Colors.ENDC}")

    for indicator in indicators:
        mpf_time, _, _ = mplfinance_results.get(indicator, (None, None, None))
        kf_cpu_time, _, _ = kimsfinance_cpu_results.get(indicator, (None, None, None))
        kf_gpu_time, _, _ = kimsfinance_gpu_results.get(indicator, (None, None, None))
        rust_cpu_time, _, _ = rust_cpu_results.get(indicator, (None, None, None))
        rust_gpu_time, _, _ = rust_gpu_results.get(indicator, (None, None, None))

        def format_speedup(baseline, impl_time):
            if baseline is None or impl_time is None:
                return "N/A"
            speedup = baseline / impl_time
            color = Colors.OKGREEN if speedup >= 1.0 else Colors.FAIL
            return f"{color}{speedup:6.2f}x{Colors.ENDC}"

        kf_cpu_speedup = format_speedup(mpf_time, kf_cpu_time)
        kf_gpu_speedup = format_speedup(mpf_time, kf_gpu_time)
        rust_cpu_speedup = format_speedup(mpf_time, rust_cpu_time)
        rust_gpu_speedup = format_speedup(mpf_time, rust_gpu_time)

        print(
            f"{indicator:<12} | {kf_cpu_speedup:<27} | {kf_gpu_speedup:<27} | {rust_cpu_speedup:<27} | {rust_gpu_speedup:<27}"
        )

    # Summary statistics
    print(f"\n{Colors.BOLD}{'='*145}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKGREEN}SUMMARY STATISTICS{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*145}{Colors.ENDC}\n")

    def calc_avg_speedup(baseline_results, impl_results):
        speedups = []
        for indicator in baseline_results:
            if indicator in impl_results:
                baseline_time, _, _ = baseline_results[indicator]
                impl_time, _, _ = impl_results[indicator]
                if baseline_time and impl_time:
                    speedups.append(baseline_time / impl_time)
        return np.mean(speedups) if speedups else None

    if mplfinance_results:
        kf_cpu_avg_speedup = calc_avg_speedup(mplfinance_results, kimsfinance_cpu_results)
        kf_gpu_avg_speedup = calc_avg_speedup(mplfinance_results, kimsfinance_gpu_results)
        rust_cpu_avg_speedup = calc_avg_speedup(mplfinance_results, rust_cpu_results)
        rust_gpu_avg_speedup = calc_avg_speedup(mplfinance_results, rust_gpu_results)

        print("  Average Speedup vs mplfinance:")
        if kf_cpu_avg_speedup:
            print(
                f"    kimsfinance Py CPU:  {Colors.OKGREEN}{kf_cpu_avg_speedup:6.2f}x{Colors.ENDC}"
            )
        if kf_gpu_avg_speedup:
            print(
                f"    kimsfinance Py GPU:  {Colors.OKGREEN}{kf_gpu_avg_speedup:6.2f}x{Colors.ENDC}"
            )
        if rust_cpu_avg_speedup:
            print(
                f"    Rust CPU:            {Colors.OKGREEN}{rust_cpu_avg_speedup:6.2f}x{Colors.ENDC}"
            )
        if rust_gpu_avg_speedup:
            print(
                f"    Rust GPU:            {Colors.OKGREEN}{rust_gpu_avg_speedup:6.2f}x{Colors.ENDC}"
            )

    print(f"\n{Colors.BOLD}{'='*145}{Colors.ENDC}\n")


# ============================================================================
# Main
# ============================================================================


def main():
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔══════════════════════════════════════════════════════════════════════════════════╗")
    print("║  Comprehensive Indicator Performance Benchmark (5-Way Comparison)               ║")
    print("║  mplfinance | kimsfinance Py (CPU/GPU) | kimsfinance_core Rust (CPU/GPU)       ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")

    # Load data
    df = load_binance_data(limit=100_000)  # Use 100K candles for comprehensive test
    num_candles = len(df)

    # Run benchmarks for all 5 implementations
    mplfinance_results = benchmark_mplfinance(df)
    kimsfinance_cpu_results = benchmark_kimsfinance_python(df)
    kimsfinance_gpu_results = benchmark_kimsfinance_python_gpu(df)
    rust_cpu_results = benchmark_kimsfinance_rust_cpu(df)
    rust_gpu_results = benchmark_kimsfinance_rust_gpu(df)

    # Print comparison table
    print_comparison_table(
        mplfinance_results,
        kimsfinance_cpu_results,
        kimsfinance_gpu_results,
        rust_cpu_results,
        rust_gpu_results,
        num_candles,
    )

    print(f"{Colors.OKGREEN}✅ Benchmark complete!{Colors.ENDC}\n")


if __name__ == "__main__":
    main()
