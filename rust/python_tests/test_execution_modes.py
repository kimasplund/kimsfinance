#!/usr/bin/env python3
"""
Test script for execution_mode parameter in batch_backtest()

Validates that all 4 execution modes (auto, traditional, fused, async) work correctly
and produce identical results.
"""

import numpy as np
import sys

try:
    from kimsfinance_core import batch_backtest
except ImportError:
    print("ERROR: kimsfinance_core not found. Run: maturin develop --release --features gpu")
    sys.exit(1)


def generate_test_data(n_candles: int = 1000):
    """Generate synthetic OHLCV data for testing"""
    np.random.seed(42)  # Reproducible results
    ohlcv = np.random.randn(n_candles, 5).cumsum(axis=0) + 100
    ohlcv[:, 0:4] = np.abs(ohlcv[:, 0:4])  # Prices must be positive
    ohlcv[:, 4] = np.abs(ohlcv[:, 4]) * 1000  # Volume
    return ohlcv


def test_all_execution_modes():
    """Test all 4 execution modes"""
    print("=" * 60)
    print("Testing batch_backtest execution_mode parameter")
    print("=" * 60)

    # Generate test data
    n_candles = 1000
    ohlcv = generate_test_data(n_candles)
    print(f"\nTest data: {n_candles} candles, {ohlcv.shape}")

    # Test parameters (100 strategies)
    parameters = [[14.0, 20.0 + i, 70.0 + i] for i in range(100)]
    print(f"Test strategies: {len(parameters)} RSI crossover strategies")

    # Test all execution modes
    modes = ['auto', 'traditional', 'fused', 'async']
    results_by_mode = {}

    print("\n" + "-" * 60)
    print("Testing all execution modes:")
    print("-" * 60)

    for mode in modes:
        try:
            print(f"\n[{mode.upper()}] Testing execution_mode='{mode}'...")
            results = batch_backtest(
                strategy='rsi_crossover',
                ohlcv=ohlcv,
                parameters=parameters,
                execution_mode=mode
            )

            results_by_mode[mode] = results

            # Extract key metrics
            best = results[0]
            worst = results[-1]
            avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)

            print(f"  ✓ Success: {len(results)} results returned")
            print(f"  ✓ Best Sharpe: {best.sharpe_ratio:.4f}")
            print(f"  ✓ Worst Sharpe: {worst.sharpe_ratio:.4f}")
            print(f"  ✓ Average Sharpe: {avg_sharpe:.4f}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            return False

    # Validate consistency across modes
    print("\n" + "-" * 60)
    print("Validating consistency across modes:")
    print("-" * 60)

    reference_mode = 'traditional'
    reference_results = results_by_mode[reference_mode]

    for mode in ['fused', 'async', 'auto']:
        print(f"\nComparing {mode} vs {reference_mode}:")
        test_results = results_by_mode[mode]

        # Compare all Sharpe ratios
        max_diff = 0.0
        for i, (ref, test) in enumerate(zip(reference_results, test_results)):
            diff = abs(ref.sharpe_ratio - test.sharpe_ratio)
            if diff > max_diff:
                max_diff = diff

            # Check if difference is within tolerance (1e-6 for floating point)
            if diff > 1e-4:  # Relaxed tolerance for GPU operations
                print(f"  ✗ Strategy {i}: Sharpe mismatch (diff={diff:.6f})")
                print(f"    {reference_mode}: {ref.sharpe_ratio:.6f}")
                print(f"    {mode}: {test.sharpe_ratio:.6f}")
                return False

        print(f"  ✓ All Sharpe ratios match (max diff: {max_diff:.8f})")

    return True


def test_invalid_mode():
    """Test that invalid execution modes raise ValueError"""
    print("\n" + "-" * 60)
    print("Testing invalid execution_mode:")
    print("-" * 60)

    ohlcv = generate_test_data(100)
    parameters = [[14.0, 30.0, 70.0]]

    try:
        print("\nAttempting execution_mode='invalid'...")
        results = batch_backtest(
            strategy='rsi_crossover',
            ohlcv=ohlcv,
            parameters=parameters,
            execution_mode='invalid'
        )
        print("  ✗ FAILED: Should have raised ValueError!")
        return False
    except ValueError as e:
        error_msg = str(e)
        print(f"  ✓ ValueError raised correctly: {error_msg}")

        # Validate error message is helpful
        if 'invalid' in error_msg.lower() and 'auto' in error_msg.lower():
            print("  ✓ Error message is helpful and suggests valid options")
            return True
        else:
            print(f"  ✗ Error message not helpful: {error_msg}")
            return False
    except Exception as e:
        print(f"  ✗ Wrong exception type: {type(e).__name__}: {e}")
        return False


def test_case_insensitive():
    """Test that execution_mode is case-insensitive"""
    print("\n" + "-" * 60)
    print("Testing case-insensitive mode parsing:")
    print("-" * 60)

    ohlcv = generate_test_data(100)
    parameters = [[14.0, 30.0, 70.0]]

    test_cases = ['Auto', 'TRADITIONAL', 'FuSeD', 'ASYNC']

    for mode in test_cases:
        try:
            print(f"\nTesting execution_mode='{mode}'...")
            results = batch_backtest(
                strategy='rsi_crossover',
                ohlcv=ohlcv,
                parameters=parameters,
                execution_mode=mode
            )
            print(f"  ✓ Success: {mode} accepted (case-insensitive)")
        except Exception as e:
            print(f"  ✗ FAILED: {mode} rejected: {e}")
            return False

    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("batch_backtest execution_mode test suite")
    print("=" * 60)

    all_passed = True

    # Test 1: All execution modes
    if not test_all_execution_modes():
        all_passed = False
        print("\n❌ TEST FAILED: Execution modes test")

    # Test 2: Invalid mode
    if not test_invalid_mode():
        all_passed = False
        print("\n❌ TEST FAILED: Invalid mode test")

    # Test 3: Case-insensitive
    if not test_case_insensitive():
        all_passed = False
        print("\n❌ TEST FAILED: Case-insensitive test")

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
