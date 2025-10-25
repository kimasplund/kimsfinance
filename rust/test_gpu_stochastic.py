#!/usr/bin/env python3
"""
GPU Stochastic Oscillator Validation Test

This script validates that the GPU-accelerated Stochastic Oscillator
implementation works correctly with realistic test data.

Requirements:
- kimsfinance_core module with GPU support
- NVIDIA GPU with CUDA support
- cupy for GPU array operations
"""

import sys
import time
import numpy as np

try:
    import kimsfinance_core
except ImportError:
    print("Error: kimsfinance_core module not found")
    print("Build the module first with: maturin develop --release")
    sys.exit(1)

try:
    import cupy as cp
except ImportError:
    print("Error: cupy not installed")
    print("Install with: pip install cupy-cuda12x")
    sys.exit(1)


def generate_realistic_ohlc_data(n_candles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate realistic OHLC test data with trending and oscillating behavior.

    Args:
        n_candles: Number of candles to generate

    Returns:
        Tuple of (high, low, close) numpy arrays
    """
    # Start with a base price
    base_price = 100.0

    # Generate price movement with trend + noise
    trend = np.linspace(0, 20, n_candles)  # Upward trend
    noise = np.cumsum(np.random.randn(n_candles) * 0.5)  # Random walk

    close = base_price + trend + noise

    # High/Low are +/- some percentage of close
    high_spread = np.random.uniform(0.5, 2.5, n_candles)
    low_spread = np.random.uniform(0.5, 2.5, n_candles)

    high = close + high_spread
    low = close - low_spread

    # Ensure high >= close >= low
    high = np.maximum(high, close)
    low = np.minimum(low, close)

    return high, low, close


def validate_stochastic_output(result: dict, n_candles: int) -> bool:
    """
    Validate that Stochastic Oscillator output is correct.

    Args:
        result: Dictionary with 'k' and 'd' keys
        n_candles: Expected number of candles

    Returns:
        True if validation passes, False otherwise
    """
    # Check result structure
    if not isinstance(result, dict):
        print(f"Error: Result is not a dict, got {type(result)}")
        return False

    if 'k' not in result or 'd' not in result:
        print(f"Error: Result missing 'k' or 'd' keys, got {result.keys()}")
        return False

    k_values = result['k']
    d_values = result['d']

    # Check array lengths
    if len(k_values) != n_candles:
        print(f"Error: %K has {len(k_values)} values, expected {n_candles}")
        return False

    if len(d_values) != n_candles:
        print(f"Error: %D has {len(d_values)} values, expected {n_candles}")
        return False

    # Check value ranges (allowing for NaN in warmup period)
    k_valid = k_values[~np.isnan(k_values)]
    d_valid = d_values[~np.isnan(d_values)]

    if len(k_valid) > 0:
        k_min, k_max = k_valid.min(), k_valid.max()
        if k_min < 0 or k_max > 100:
            print(f"Error: %K values out of range [0, 100]: [{k_min:.2f}, {k_max:.2f}]")
            return False

    if len(d_valid) > 0:
        d_min, d_max = d_valid.min(), d_valid.max()
        if d_min < 0 or d_max > 100:
            print(f"Error: %D values out of range [0, 100]: [{d_min:.2f}, {d_max:.2f}]")
            return False

    return True


def test_gpu_stochastic(n_candles: int = 100_000, k_period: int = 14,
                        d_period: int = 3, d_smooth: int = 0) -> bool:
    """
    Test GPU-accelerated Stochastic Oscillator calculation.

    Args:
        n_candles: Number of candles to test with
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)
        d_smooth: %D smoothing (default: 0)

    Returns:
        True if test passes, False otherwise
    """
    print(f"Testing GPU Stochastic Oscillator with {n_candles:,} candles...")
    print(f"Parameters: k_period={k_period}, d_period={d_period}, d_smooth={d_smooth}")
    print()

    # Generate test data
    print("Generating realistic OHLC test data...")
    high, low, close = generate_realistic_ohlc_data(n_candles)

    # Verify data is reasonable
    print(f"Data ranges:")
    print(f"  High:  [{high.min():.2f}, {high.max():.2f}]")
    print(f"  Low:   [{low.min():.2f}, {low.max():.2f}]")
    print(f"  Close: [{close.min():.2f}, {close.max():.2f}]")
    print()

    # Run GPU calculation
    print("Running GPU Stochastic calculation...")
    try:
        start = time.perf_counter()
        result = kimsfinance_core.calculate_stochastic_gpu(
            high, low, close, k_period, d_period, d_smooth
        )
        elapsed = time.perf_counter() - start

        print(f"✓ GPU calculation completed in {elapsed*1000:.2f}ms")
        print()

    except Exception as e:
        print(f"Error during GPU calculation: {e}")
        return False

    # Validate output
    print("Validating output...")
    if not validate_stochastic_output(result, n_candles):
        return False

    # Print statistics
    k_values = result['k']
    d_values = result['d']

    # Skip NaN values in warmup period
    k_valid = k_values[~np.isnan(k_values)]
    d_valid = d_values[~np.isnan(d_values)]

    print(f"✓ Output validation passed")
    print()
    print(f"Results:")
    print(f"  %K values: {len(k_valid):,} valid (after {len(k_values) - len(k_valid)} warmup)")
    print(f"    Range: [{k_valid.min():.2f}, {k_valid.max():.2f}]")
    print(f"    Mean:  {k_valid.mean():.2f}")
    print(f"    Std:   {k_valid.std():.2f}")
    print()
    print(f"  %D values: {len(d_valid):,} valid (after {len(d_values) - len(d_valid)} warmup)")
    print(f"    Range: [{d_valid.min():.2f}, {d_valid.max():.2f}]")
    print(f"    Mean:  {d_valid.mean():.2f}")
    print(f"    Std:   {d_valid.std():.2f}")
    print()

    # Calculate throughput
    throughput = n_candles / elapsed
    print(f"Performance:")
    print(f"  Throughput: {throughput:,.0f} candles/sec")
    print(f"  Time per candle: {elapsed/n_candles*1e6:.2f} μs")
    print()

    return True


def main():
    """Run GPU Stochastic Oscillator validation tests."""
    print("=" * 70)
    print("GPU Stochastic Oscillator Validation Test")
    print("=" * 70)
    print()

    # Test with different dataset sizes
    test_sizes = [
        (10_000, "Small dataset"),
        (100_000, "Medium dataset"),
        (1_000_000, "Large dataset"),
    ]

    all_passed = True

    for n_candles, description in test_sizes:
        print(f"Test: {description} ({n_candles:,} candles)")
        print("-" * 70)

        passed = test_gpu_stochastic(n_candles)

        if passed:
            print(f"✓ {description} test PASSED")
        else:
            print(f"✗ {description} test FAILED")
            all_passed = False

        print()

    print("=" * 70)
    if all_passed:
        print("✓ All tests PASSED")
        print("GPU Stochastic Oscillator is working correctly!")
        return 0
    else:
        print("✗ Some tests FAILED")
        print("Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
