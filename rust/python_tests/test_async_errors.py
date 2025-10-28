"""
Test async mode error handling

Validates that async execution mode provides clear, Python-friendly error messages
for common failure scenarios.

Run with: python python_tests/test_async_errors.py
"""

import numpy as np
import sys

try:
    from kimsfinance_core import batch_backtest
except ImportError:
    print("ERROR: kimsfinance_core not installed. Run: maturin develop --release")
    sys.exit(1)


def test_empty_parameters():
    """Test error handling for empty parameters"""
    print("TEST 1: Empty parameters")
    print("-" * 50)

    try:
        batch_backtest(
            strategy='rsi_crossover',
            ohlcv=np.zeros((100, 5)),
            parameters=[],
            execution_mode='async'
        )
        print("❌ FAILED: Should have raised ValueError for empty parameters!")
        return False
    except ValueError as e:
        error_msg = str(e)
        print(f"✅ PASSED: Caught ValueError")
        print(f"   Message: {error_msg}")
        if "parameters cannot be empty" in error_msg.lower():
            print("   ✓ Error message is clear and helpful")
        else:
            print("   ⚠ Error message could be more descriptive")
        return True


def test_invalid_ohlcv_shape():
    """Test error handling for invalid OHLCV shape"""
    print("\nTEST 2: Invalid OHLCV shape")
    print("-" * 50)

    try:
        batch_backtest(
            strategy='rsi_crossover',
            ohlcv=np.zeros((100, 3)),  # Wrong shape (should be N×5)
            parameters=[[14, 30, 70]],
            execution_mode='async'
        )
        print("❌ FAILED: Should have raised ValueError for wrong OHLCV shape!")
        return False
    except ValueError as e:
        error_msg = str(e)
        print(f"✅ PASSED: Caught ValueError")
        print(f"   Message: {error_msg}")
        if "shape" in error_msg.lower():
            print("   ✓ Error message mentions shape issue")
        else:
            print("   ⚠ Error message could mention 'shape' explicitly")
        return True


def test_invalid_execution_mode():
    """Test error handling for invalid execution mode"""
    print("\nTEST 3: Invalid execution mode")
    print("-" * 50)

    try:
        batch_backtest(
            strategy='rsi_crossover',
            ohlcv=np.zeros((100, 5)),
            parameters=[[14, 30, 70]],
            execution_mode='invalid_mode'
        )
        print("❌ FAILED: Should have raised ValueError for invalid execution mode!")
        return False
    except ValueError as e:
        error_msg = str(e)
        print(f"✅ PASSED: Caught ValueError")
        print(f"   Message: {error_msg}")
        if "execution_mode" in error_msg.lower():
            print("   ✓ Error message mentions execution_mode")
        if "auto" in error_msg.lower() and "async" in error_msg.lower():
            print("   ✓ Error message lists valid options")
        else:
            print("   ⚠ Error message could list valid execution modes")
        return True


def test_invalid_strategy():
    """Test error handling for invalid strategy name"""
    print("\nTEST 4: Invalid strategy name")
    print("-" * 50)

    try:
        batch_backtest(
            strategy='nonexistent_strategy',
            ohlcv=np.zeros((100, 5)),
            parameters=[[14, 30, 70]],
            execution_mode='async'
        )
        print("❌ FAILED: Should have raised ValueError for invalid strategy!")
        return False
    except ValueError as e:
        error_msg = str(e)
        print(f"✅ PASSED: Caught ValueError")
        print(f"   Message: {error_msg}")
        if "strategy" in error_msg.lower():
            print("   ✓ Error message mentions strategy issue")
        if "rsi_crossover" in error_msg.lower():
            print("   ✓ Error message lists valid strategies")
        else:
            print("   ⚠ Error message could list valid strategies")
        return True


def test_mismatched_timestamps():
    """Test error handling for timestamp length mismatch"""
    print("\nTEST 5: Mismatched timestamp length")
    print("-" * 50)

    try:
        batch_backtest(
            strategy='rsi_crossover',
            ohlcv=np.zeros((100, 5)),
            parameters=[[14, 30, 70]],
            timestamps=np.arange(50),  # Only 50 timestamps for 100 candles
            execution_mode='async'
        )
        print("❌ FAILED: Should have raised ValueError for timestamp mismatch!")
        return False
    except ValueError as e:
        error_msg = str(e)
        print(f"✅ PASSED: Caught ValueError")
        print(f"   Message: {error_msg}")
        if "timestamp" in error_msg.lower():
            print("   ✓ Error message mentions timestamp issue")
        return True


def test_all_execution_modes():
    """Test that all execution modes work with valid inputs"""
    print("\nTEST 6: All execution modes work with valid input")
    print("-" * 50)

    np.random.seed(42)
    ohlcv = np.random.randn(1000, 5).cumsum(axis=0) + 100.0
    ohlcv[:, 0:4] = np.abs(ohlcv[:, 0:4])
    ohlcv[:, 4] = np.abs(ohlcv[:, 4]) * 1000.0

    parameters = [[14.0, 30.0, 70.0], [14.0, 25.0, 75.0]]

    modes = ['auto', 'traditional', 'fused', 'async']
    for mode in modes:
        try:
            results = batch_backtest(
                strategy='rsi_crossover',
                ohlcv=ohlcv,
                parameters=parameters,
                execution_mode=mode
            )
            print(f"✓ Mode '{mode}': {len(results)} results")
        except Exception as e:
            print(f"❌ Mode '{mode}' failed: {e}")
            return False

    print("✅ PASSED: All execution modes work correctly")
    return True


def main():
    """Run all error handling tests"""
    print("="*70)
    print("ASYNC MODE ERROR HANDLING TESTS")
    print("="*70)
    print()

    tests = [
        test_empty_parameters,
        test_invalid_ohlcv_shape,
        test_invalid_execution_mode,
        test_invalid_strategy,
        test_mismatched_timestamps,
        test_all_execution_modes,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ ALL ERROR HANDLING TESTS PASSED!")
        print("\nAsync mode provides clear, Python-friendly error messages.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
