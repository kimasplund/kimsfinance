#!/usr/bin/env python3
"""Simple Rust vs Python indicator benchmark on 1M rows"""

import time
import numpy as np
import sys
from pathlib import Path

# Add Rust build path
sys.path.insert(0, str(Path(__file__).parent.parent / "rust"))

# Python indicators
from kimsfinance.ops.indicators import calculate_rsi, calculate_atr, calculate_macd

# Try to import Rust
try:
    import kimsfinance_core

    RUST_AVAILABLE = True
except Exception:
    print("⚠️  Rust not available. Build with: cd rust && maturin develop --release")
    RUST_AVAILABLE = False


def generate_data(n=1_000_000):
    """Generate synthetic OHLCV data"""
    print(f"Generating {n:,} rows...")
    np.random.seed(42)

    base = 100.0
    returns = np.random.normal(0.0001, 0.02, n)
    close = base * np.exp(np.cumsum(returns))

    vol = np.abs(np.random.normal(0, 0.01, n))
    high = close * (1 + vol)
    low = close * (1 - vol)

    print(f"✓ Generated {n:,} rows\n")
    return high, low, close


def main():
    print("=" * 70)
    print("RUST VS PYTHON - 1 MILLION ROWS")
    print("=" * 70 + "\n")

    # Generate data
    high, low, close = generate_data()

    # ========== PYTHON ==========
    print("PYTHON (NumPy):")
    print("-" * 70)

    py_times = {}

    # RSI
    start = time.perf_counter()
    calculate_rsi(close, 14)
    py_times["RSI"] = time.perf_counter() - start
    print(f"RSI (14):  {py_times['RSI']:.4f}s")

    # ATR
    start = time.perf_counter()
    calculate_atr(high, low, close, 14)
    py_times["ATR"] = time.perf_counter() - start
    print(f"ATR (14):  {py_times['ATR']:.4f}s")

    # MACD
    start = time.perf_counter()
    calculate_macd(close, 12, 26, 9)
    py_times["MACD"] = time.perf_counter() - start
    print(f"MACD:      {py_times['MACD']:.4f}s")

    py_total = sum(py_times.values())
    print(f"{'='*70}")
    print(f"TOTAL:     {py_total:.4f}s\n")

    if not RUST_AVAILABLE:
        return

    # ========== RUST ==========
    print("RUST (Compiled):")
    print("-" * 70)

    rust_times = {}

    # RSI
    start = time.perf_counter()
    kimsfinance_core.calculate_rsi(close, 14)
    rust_times["RSI"] = time.perf_counter() - start
    print(f"RSI (14):  {rust_times['RSI']:.4f}s")

    # ATR
    start = time.perf_counter()
    kimsfinance_core.calculate_atr(high, low, close, 14)
    rust_times["ATR"] = time.perf_counter() - start
    print(f"ATR (14):  {rust_times['ATR']:.4f}s")

    # MACD
    start = time.perf_counter()
    kimsfinance_core.calculate_macd(close, 12, 26, 9)
    rust_times["MACD"] = time.perf_counter() - start
    print(f"MACD:      {rust_times['MACD']:.4f}s")

    rust_total = sum(rust_times.values())
    print(f"{'='*70}")
    print(f"TOTAL:     {rust_total:.4f}s\n")

    # ========== COMPARISON ==========
    print("=" * 70)
    print("SPEEDUP COMPARISON")
    print("=" * 70)
    print(f"{'Indicator':<12} {'Python':<12} {'Rust':<12} {'Speedup':<12}")
    print("-" * 70)

    speedups = []
    for name in py_times:
        speedup = py_times[name] / rust_times[name]
        speedups.append(speedup)
        print(f"{name:<12} {py_times[name]:<12.4f} {rust_times[name]:<12.4f} {speedup:>10.2f}x")

    overall = py_total / rust_total
    print("-" * 70)
    print(f"{'TOTAL':<12} {py_total:<12.4f} {rust_total:<12.4f} {overall:>10.2f}x")

    print(f"\n{'='*70}")
    print(f"AVERAGE SPEEDUP: {np.mean(speedups):.2f}x")
    print(f"TIME SAVED: {py_total - rust_total:.4f}s ({(1-rust_total/py_total)*100:.1f}% faster)")
    print("=" * 70)


if __name__ == "__main__":
    main()
