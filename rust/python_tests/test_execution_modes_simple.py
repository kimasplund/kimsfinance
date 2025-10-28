#!/usr/bin/env python3
"""
Simplified test for execution_mode parameter validation

Tests that our Python bindings correctly parse and apply the execution_mode
parameter, regardless of backend implementation issues.
"""

import numpy as np
import sys

try:
    from kimsfinance_core import batch_backtest
except ImportError:
    print("ERROR: kimsfinance_core not found")
    sys.exit(1)


def main():
    """Test execution_mode parameter parsing"""
    print("=" * 60)
    print("execution_mode Parameter Validation Test")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    n_candles = 100
    ohlcv = np.random.randn(n_candles, 5).cumsum(axis=0) + 100
    ohlcv[:, 0:4] = np.abs(ohlcv[:, 0:4])
    ohlcv[:, 4] = np.abs(ohlcv[:, 4]) * 1000

    parameters = [[14.0, 30.0, 70.0]]

    tests_passed = 0
    tests_failed = 0

    # Test 1: Valid modes
    print("\n1. Testing valid execution modes...")
    for mode in ['auto', 'traditional', 'fused', 'async']:
        try:
            results = batch_backtest(
                strategy='rsi_crossover',
                ohlcv=ohlcv,
                parameters=parameters,
                execution_mode=mode
            )
            print(f"   ✓ {mode:12s} - {len(results)} results returned")
            tests_passed += 1
        except Exception as e:
            print(f"   ✗ {mode:12s} - FAILED: {e}")
            tests_failed += 1

    # Test 2: Case-insensitive
    print("\n2. Testing case-insensitive parsing...")
    for mode in ['Auto', 'FUSED']:
        try:
            results = batch_backtest(
                strategy='rsi_crossover',
                ohlcv=ohlcv,
                parameters=parameters,
                execution_mode=mode
            )
            print(f"   ✓ {mode:12s} - Accepted (case-insensitive)")
            tests_passed += 1
        except Exception as e:
            print(f"   ✗ {mode:12s} - FAILED: {e}")
            tests_failed += 1

    # Test 3: Invalid mode
    print("\n3. Testing invalid execution mode...")
    try:
        results = batch_backtest(
            strategy='rsi_crossover',
            ohlcv=ohlcv,
            parameters=parameters,
            execution_mode='invalid_mode'
        )
        print("   ✗ Should have raised ValueError")
        tests_failed += 1
    except ValueError as e:
        error_msg = str(e)
        if 'invalid_mode' in error_msg and 'auto' in error_msg:
            print(f"   ✓ ValueError raised with helpful message")
            tests_passed += 1
        else:
            print(f"   ✗ ValueError raised but message not helpful: {error_msg}")
            tests_failed += 1
    except Exception as e:
        print(f"   ✗ Wrong exception: {type(e).__name__}: {e}")
        tests_failed += 1

    # Test 4: Default value
    print("\n4. Testing default execution_mode...")
    try:
        # Call without execution_mode parameter (should default to 'auto')
        results = batch_backtest(
            strategy='rsi_crossover',
            ohlcv=ohlcv,
            parameters=parameters
        )
        print(f"   ✓ Default mode works (returned {len(results)} results)")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Default mode failed: {e}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print("=" * 60)

    if tests_failed == 0:
        print("✅ ALL TESTS PASSED - Python bindings working correctly!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
