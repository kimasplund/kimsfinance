#!/usr/bin/env python3
"""
Benchmark: Rust vs Python Indicator Calculations
Tests performance on 1 million rows of OHLCV data

Compares:
- Python kimsfinance indicators (NumPy/pandas)
- Rust kimsfinance_core indicators (compiled Rust)
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add Python kimsfinance to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import Python indicators
try:
    from kimsfinance.ops.indicators import (
        calculate_rsi,
        calculate_atr,
        calculate_macd,
        calculate_bollinger_bands,
    )
    from kimsfinance.ops import (
        calculate_stochastic,
        calculate_adx,
    )
    PYTHON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Python kimsfinance not available: {e}")
    PYTHON_AVAILABLE = False

# Import Rust library
try:
    import kimsfinance_core
    RUST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Rust kimsfinance_core not available: {e}")
    print("Build it with: cd rust && maturin develop --release")
    RUST_AVAILABLE = False


def generate_synthetic_data(n_rows: int = 1_000_000):
    """Generate synthetic OHLCV data for benchmarking"""
    print(f"Generating {n_rows:,} rows of synthetic OHLCV data...")

    np.random.seed(42)

    # Generate realistic price movement
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_rows)
    close_prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV with realistic relationships
    volatility = np.abs(np.random.normal(0, 0.01, n_rows))
    high = close_prices * (1 + volatility)
    low = close_prices * (1 - volatility)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    # Ensure OHLC relationships are valid
    high = np.maximum(high, np.maximum(open_prices, close_prices))
    low = np.minimum(low, np.minimum(open_prices, close_prices))

    volume = np.random.lognormal(10, 1, n_rows)
    timestamps = np.arange(n_rows, dtype=np.int64) * 60  # 1-minute intervals

    data = {
        'timestamp': timestamps,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_prices,
        'volume': volume
    }

    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df):,} rows")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  Volume range: {df['volume'].min():.0f} - {df['volume'].max():.0f}")

    return df


def benchmark_python_indicators(df: pd.DataFrame):
    """Benchmark Python indicator calculations"""
    if not PYTHON_AVAILABLE:
        return None

    print("\n" + "="*80)
    print("PYTHON INDICATORS (NumPy/pandas)")
    print("="*80)

    results = {}

    # RSI
    print("\n1. RSI (14 period)...")
    start = time.perf_counter()
    rsi = calculate_rsi(df['close'].values, period=14)
    elapsed = time.perf_counter() - start
    results['rsi'] = elapsed
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Result: {len(rsi):,} values, range [{np.nanmin(rsi):.2f}, {np.nanmax(rsi):.2f}]")

    # ATR
    print("\n2. ATR (14 period)...")
    start = time.perf_counter()
    atr = calculate_atr(
        df['high'].values,
        df['low'].values,
        df['close'].values,
        period=14
    )
    elapsed = time.perf_counter() - start
    results['atr'] = elapsed
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Result: {len(atr):,} values, range [{np.nanmin(atr):.4f}, {np.nanmax(atr):.4f}]")

    # MACD
    print("\n3. MACD (12, 26, 9)...")
    start = time.perf_counter()
    macd_line, signal_line, histogram = calculate_macd(
        df['close'].values,
        fast_period=12,
        slow_period=26,
        signal_period=9
    )
    elapsed = time.perf_counter() - start
    results['macd'] = elapsed
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Result: 3 arrays of {len(macd_line):,} values each")

    # Bollinger Bands
    print("\n4. Bollinger Bands (20, 2.0)...")
    start = time.perf_counter()
    upper, middle, lower = calculate_bollinger_bands(
        df['close'].values,
        period=20,
        std_dev=2.0
    )
    elapsed = time.perf_counter() - start
    results['bollinger'] = elapsed
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Result: 3 arrays of {len(upper):,} values each")

    # Stochastic
    print("\n5. Stochastic (14, 3)...")
    start = time.perf_counter()
    k_line, d_line = calculate_stochastic(
        df['high'].values,
        df['low'].values,
        df['close'].values,
        k_period=14,
        d_period=3
    )
    elapsed = time.perf_counter() - start
    results['stochastic'] = elapsed
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Result: 2 arrays of {len(k_line):,} values each")

    # ADX
    print("\n6. ADX (14 period)...")
    start = time.perf_counter()
    adx, plus_di, minus_di = calculate_adx(
        df['high'].values,
        df['low'].values,
        df['close'].values,
        period=14
    )
    elapsed = time.perf_counter() - start
    results['adx'] = elapsed
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Result: 3 arrays of {len(adx):,} values each")

    # Total time
    total_time = sum(results.values())
    print(f"\n{'='*80}")
    print(f"TOTAL PYTHON TIME: {total_time:.4f}s")
    print(f"{'='*80}")

    return results


def benchmark_rust_indicators(df: pd.DataFrame):
    """Benchmark Rust indicator calculations"""
    if not RUST_AVAILABLE:
        return None

    print("\n" + "="*80)
    print("RUST INDICATORS (Compiled Rust + PyO3)")
    print("="*80)

    results = {}

    # Convert to numpy arrays
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # RSI
    print("\n1. RSI (14 period)...")
    start = time.perf_counter()
    rsi = kimsfinance_core.rsi(close, 14)
    elapsed = time.perf_counter() - start
    results['rsi'] = elapsed
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Result: {len(rsi):,} values, range [{np.nanmin(rsi):.2f}, {np.nanmax(rsi):.2f}]")

    # ATR
    print("\n2. ATR (14 period)...")
    start = time.perf_counter()
    atr = kimsfinance_core.atr(high, low, close, 14)
    elapsed = time.perf_counter() - start
    results['atr'] = elapsed
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Result: {len(atr):,} values, range [{np.nanmin(atr):.4f}, {np.nanmax(atr):.4f}]")

    # MACD
    print("\n3. MACD (12, 26, 9)...")
    start = time.perf_counter()
    macd_result = kimsfinance_core.macd(close, 12, 26, 9)
    elapsed = time.perf_counter() - start
    results['macd'] = elapsed
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Result: Dictionary with 3 arrays")

    # Bollinger Bands
    print("\n4. Bollinger Bands (20, 2.0)...")
    start = time.perf_counter()
    bb_result = kimsfinance_core.bollinger_bands(close, 20, 2.0)
    elapsed = time.perf_counter() - start
    results['bollinger'] = elapsed
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Result: Dictionary with 3 arrays")

    # Stochastic
    print("\n5. Stochastic (14, 3)...")
    start = time.perf_counter()
    stoch_result = kimsfinance_core.stochastic(high, low, close, 14, 3)
    elapsed = time.perf_counter() - start
    results['stochastic'] = elapsed
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Result: Dictionary with 2 arrays")

    # ADX (if available)
    if hasattr(kimsfinance_core, 'adx'):
        print("\n6. ADX (14 period)...")
        start = time.perf_counter()
        adx_result = kimsfinance_core.adx(high, low, close, 14)
        elapsed = time.perf_counter() - start
        results['adx'] = elapsed
        print(f"   Time: {elapsed:.4f}s")
        print(f"   Result: Dictionary with 3 arrays")
    else:
        print("\n6. ADX (14 period)... [NOT AVAILABLE IN RUST]")
        results['adx'] = None

    # Total time
    total_time = sum(t for t in results.values() if t is not None)
    print(f"\n{'='*80}")
    print(f"TOTAL RUST TIME: {total_time:.4f}s")
    print(f"{'='*80}")

    return results


def print_comparison(python_results, rust_results, n_rows):
    """Print side-by-side comparison"""
    if python_results is None or rust_results is None:
        print("\n⚠️  Cannot compare - one or both implementations not available")
        return

    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"\nDataset: {n_rows:,} rows")
    print(f"\n{'Indicator':<20} {'Python (s)':<15} {'Rust (s)':<15} {'Speedup':<10} {'Throughput':<15}")
    print("-" * 80)

    speedups = []

    for indicator in python_results.keys():
        py_time = python_results[indicator]
        rust_time = rust_results.get(indicator)

        if rust_time is None or rust_time == 0:
            continue

        speedup = py_time / rust_time
        speedups.append(speedup)
        throughput = n_rows / rust_time  # rows/sec

        print(f"{indicator:<20} {py_time:<15.4f} {rust_time:<15.4f} {speedup:<10.2f}x {throughput:>14,.0f}/s")

    # Summary statistics
    py_total = sum(python_results.values())
    rust_total = sum(t for t in rust_results.values() if t is not None)
    overall_speedup = py_total / rust_total if rust_total > 0 else 0

    print("-" * 80)
    print(f"{'TOTAL':<20} {py_total:<15.4f} {rust_total:<15.4f} {overall_speedup:<10.2f}x")

    if speedups:
        avg_speedup = np.mean(speedups)
        min_speedup = np.min(speedups)
        max_speedup = np.max(speedups)

        print(f"\n{'='*80}")
        print("SPEEDUP STATISTICS")
        print(f"{'='*80}")
        print(f"Average Speedup:  {avg_speedup:.2f}x")
        print(f"Minimum Speedup:  {min_speedup:.2f}x")
        print(f"Maximum Speedup:  {max_speedup:.2f}x")
        print(f"Overall Speedup:  {overall_speedup:.2f}x")

        # Performance gains
        print(f"\n{'='*80}")
        print("TIME SAVINGS")
        print(f"{'='*80}")
        time_saved = py_total - rust_total
        pct_faster = ((py_total - rust_total) / py_total) * 100
        print(f"Time Saved:       {time_saved:.4f}s ({pct_faster:.1f}% faster)")
        print(f"Rust Efficiency:  {(rust_total / py_total * 100):.1f}% of Python time")


def main():
    print("="*80)
    print("RUST VS PYTHON INDICATOR BENCHMARK")
    print("="*80)
    print(f"\nPython Available: {PYTHON_AVAILABLE}")
    print(f"Rust Available:   {RUST_AVAILABLE}")

    if not PYTHON_AVAILABLE and not RUST_AVAILABLE:
        print("\n❌ Neither implementation available. Cannot run benchmark.")
        return 1

    # Generate test data
    n_rows = 1_000_000
    df = generate_synthetic_data(n_rows)

    # Run benchmarks
    python_results = benchmark_python_indicators(df)
    rust_results = benchmark_rust_indicators(df)

    # Compare results
    print_comparison(python_results, rust_results, n_rows)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
