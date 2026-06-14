#!/usr/bin/env python3
"""
Test VWAP Anchored implementation - Ensure parity between Rust and Python.

This test validates:
1. Rust implementation matches Python implementation exactly
2. Anchored VWAP correctly resets at anchor points
3. Edge cases (single anchor, multiple anchors, no anchors)
4. Performance comparison (Rust should be 5-10x faster)
"""

import sys
import numpy as np
import time

# Import kimsfinance_core (Rust implementation)
try:
    import kimsfinance_core
    RUST_AVAILABLE = True
except ImportError:
    print("ERROR: kimsfinance_core not available. Build with: maturin develop --release")
    sys.exit(1)

# Import Python implementation
try:
    sys.path.insert(0, '/home/kim/projects/kimsfinance')
    from kimsfinance.ops.indicators.vwap import calculate_vwap_anchored
    PYTHON_AVAILABLE = True
except ImportError:
    print("WARNING: Python kimsfinance not available. Skipping parity tests.")
    PYTHON_AVAILABLE = False


def manual_vwap_anchored(high, low, close, volume, anchors):
    """
    Manual reference implementation for validation.

    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3

    Resets cumulative sums at anchor points (anchors[i] == True).
    """
    n = len(high)
    vwap = np.zeros(n)
    cumsum_tp_volume = 0.0
    cumsum_volume = 0.0

    for i in range(n):
        # Reset on anchor points
        if anchors[i]:
            cumsum_tp_volume = 0.0
            cumsum_volume = 0.0

        # Calculate typical price
        typical_price = (high[i] + low[i] + close[i]) / 3.0

        # Accumulate sums
        cumsum_tp_volume += typical_price * volume[i]
        cumsum_volume += volume[i]

        # Calculate VWAP
        if cumsum_volume > 0.0:
            vwap[i] = cumsum_tp_volume / cumsum_volume

    return vwap


def test_basic_anchored_vwap():
    """Test basic anchored VWAP calculation."""
    print("\n=== Test 1: Basic Anchored VWAP ===")

    high = np.array([110.0, 115.0, 120.0, 118.0, 122.0])
    low = np.array([105.0, 110.0, 115.0, 113.0, 117.0])
    close = np.array([108.0, 112.0, 118.0, 115.0, 120.0])
    volume = np.array([100.0, 200.0, 150.0, 120.0, 180.0])
    anchors = np.array([True, False, False, True, False])  # Reset at indices 0 and 3

    # Manual reference
    manual_result = manual_vwap_anchored(high, low, close, volume, anchors)
    print(f"Manual result: {manual_result}")

    # Rust implementation
    rust_result = kimsfinance_core.calculate_vwap_anchored(high, low, close, volume, anchors)
    print(f"Rust result:   {rust_result}")

    # Validate
    np.testing.assert_allclose(rust_result, manual_result, rtol=1e-10, atol=1e-10)
    print("✓ Rust matches manual implementation")

    # Python implementation (if available)
    if PYTHON_AVAILABLE:
        python_result = calculate_vwap_anchored(high, low, close, volume, anchors)
        print(f"Python result: {python_result}")
        np.testing.assert_allclose(rust_result, python_result, rtol=1e-10, atol=1e-10)
        print("✓ Rust matches Python implementation")

    # Validate reset behavior
    # VWAP should reset at index 3 (second anchor)
    assert rust_result[0] > 0, "First value should be non-zero"
    assert rust_result[3] > 0, "Reset point should have non-zero value"
    # VWAP should be cumulative between anchors
    assert rust_result[1] != rust_result[0], "VWAP should change between indices"

    print("✓ Test passed")


def test_single_anchor():
    """Test with a single anchor at the beginning."""
    print("\n=== Test 2: Single Anchor (at start) ===")

    high = np.array([110.0, 115.0, 120.0, 125.0])
    low = np.array([105.0, 110.0, 115.0, 120.0])
    close = np.array([108.0, 112.0, 118.0, 123.0])
    volume = np.array([100.0, 150.0, 200.0, 180.0])
    anchors = np.array([True, False, False, False])  # Single anchor at start

    # Manual reference
    manual_result = manual_vwap_anchored(high, low, close, volume, anchors)

    # Rust implementation
    rust_result = kimsfinance_core.calculate_vwap_anchored(high, low, close, volume, anchors)

    # Validate
    np.testing.assert_allclose(rust_result, manual_result, rtol=1e-10, atol=1e-10)
    print("✓ Rust matches manual implementation")

    # Single anchor at start should behave like regular VWAP
    regular_vwap = kimsfinance_core.calculate_vwap(high, low, close, volume)
    np.testing.assert_allclose(rust_result, regular_vwap, rtol=1e-10, atol=1e-10)
    print("✓ Single anchor at start matches regular VWAP")


def test_multiple_anchors():
    """Test with multiple anchor points."""
    print("\n=== Test 3: Multiple Anchors ===")

    high = np.array([110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 128.0])
    low = np.array([105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 123.0])
    close = np.array([108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 126.0])
    volume = np.array([100.0, 200.0, 150.0, 120.0, 180.0, 220.0, 140.0, 190.0])
    anchors = np.array([True, False, True, False, True, False, False, True])  # Multiple resets

    # Manual reference
    manual_result = manual_vwap_anchored(high, low, close, volume, anchors)

    # Rust implementation
    rust_result = kimsfinance_core.calculate_vwap_anchored(high, low, close, volume, anchors)

    # Validate
    np.testing.assert_allclose(rust_result, manual_result, rtol=1e-10, atol=1e-10)
    print("✓ Rust matches manual implementation")

    # Validate reset behavior at each anchor
    anchor_indices = np.where(anchors)[0]
    print(f"Anchor indices: {anchor_indices}")
    for idx in anchor_indices:
        assert rust_result[idx] > 0, f"Anchor at index {idx} should have non-zero VWAP"

    print("✓ Test passed")


def test_no_anchors():
    """Test with no anchors (all False) - should be all zeros or NaN."""
    print("\n=== Test 4: No Anchors ===")

    high = np.array([110.0, 115.0, 120.0])
    low = np.array([105.0, 110.0, 115.0])
    close = np.array([108.0, 112.0, 118.0])
    volume = np.array([100.0, 150.0, 200.0])
    anchors = np.array([False, False, False])  # No anchors

    # Manual reference
    manual_result = manual_vwap_anchored(high, low, close, volume, anchors)

    # Rust implementation
    rust_result = kimsfinance_core.calculate_vwap_anchored(high, low, close, volume, anchors)

    # Validate
    np.testing.assert_allclose(rust_result, manual_result, rtol=1e-10, atol=1e-10)
    print("✓ Rust matches manual implementation")

    # With no anchors, cumulative sums never reset, so VWAP is cumulative from start
    # This is actually a valid use case (treat entire dataset as one session)
    print(f"Result with no anchors: {rust_result}")
    print("✓ Test passed")


def test_edge_case_single_value():
    """Test with a single data point."""
    print("\n=== Test 5: Edge Case - Single Value ===")

    high = np.array([110.0])
    low = np.array([105.0])
    close = np.array([108.0])
    volume = np.array([100.0])
    anchors = np.array([True])

    # Manual reference
    manual_result = manual_vwap_anchored(high, low, close, volume, anchors)

    # Rust implementation
    rust_result = kimsfinance_core.calculate_vwap_anchored(high, low, close, volume, anchors)

    # Validate
    np.testing.assert_allclose(rust_result, manual_result, rtol=1e-10, atol=1e-10)
    print("✓ Rust matches manual implementation")

    # Single value: VWAP should equal typical price
    expected = (110.0 + 105.0 + 108.0) / 3.0
    np.testing.assert_allclose(rust_result[0], expected, rtol=1e-10)
    print(f"✓ Single value VWAP = {rust_result[0]:.6f} (expected {expected:.6f})")


def test_performance_comparison():
    """Compare performance: Rust vs Python."""
    print("\n=== Test 6: Performance Comparison ===")

    # Large dataset
    n = 100_000
    np.random.seed(42)
    high = 100.0 + np.random.randn(n).cumsum() * 0.5
    low = high - np.random.uniform(1.0, 5.0, n)
    close = low + np.random.uniform(0, high - low)
    volume = np.random.uniform(1000, 10000, n)

    # Anchor every 1000 bars (simulate daily sessions)
    anchors = np.zeros(n, dtype=bool)
    anchors[::1000] = True

    # Warm-up (JIT compilation)
    _ = kimsfinance_core.calculate_vwap_anchored(high[:100], low[:100], close[:100], volume[:100], anchors[:100])

    # Rust benchmark
    rust_times = []
    for _ in range(10):
        start = time.perf_counter()
        rust_result = kimsfinance_core.calculate_vwap_anchored(high, low, close, volume, anchors)
        rust_times.append(time.perf_counter() - start)

    rust_mean = np.mean(rust_times) * 1000  # Convert to ms
    rust_std = np.std(rust_times) * 1000
    print(f"Rust: {rust_mean:.2f} ± {rust_std:.2f} ms (n={n})")

    # Python benchmark (if available)
    if PYTHON_AVAILABLE:
        python_times = []
        for _ in range(10):
            start = time.perf_counter()
            python_result = calculate_vwap_anchored(high, low, close, volume, anchors)
            python_times.append(time.perf_counter() - start)

        python_mean = np.mean(python_times) * 1000
        python_std = np.std(python_times) * 1000
        print(f"Python: {python_mean:.2f} ± {python_std:.2f} ms (n={n})")

        speedup = python_mean / rust_mean
        print(f"Speedup: {speedup:.2f}x")

        # Validate correctness at scale
        np.testing.assert_allclose(rust_result, python_result, rtol=1e-8, atol=1e-8)
        print("✓ Rust matches Python at scale")

        # Performance assertion (Rust should be at least 2x faster)
        assert speedup >= 2.0, f"Expected Rust to be >=2x faster, got {speedup:.2f}x"
        print(f"✓ Performance target met ({speedup:.2f}x >= 2.0x)")
    else:
        print("(Python implementation not available, skipping comparison)")

    print("✓ Test passed")


def test_consistency_with_regular_vwap():
    """Verify that anchored VWAP with single anchor matches regular VWAP."""
    print("\n=== Test 7: Consistency with Regular VWAP ===")

    high = np.array([110.0, 115.0, 120.0, 118.0, 122.0, 125.0])
    low = np.array([105.0, 110.0, 115.0, 113.0, 117.0, 120.0])
    close = np.array([108.0, 112.0, 118.0, 115.0, 120.0, 123.0])
    volume = np.array([100.0, 200.0, 150.0, 120.0, 180.0, 220.0])

    # Regular VWAP
    regular_vwap = kimsfinance_core.calculate_vwap(high, low, close, volume)

    # Anchored VWAP with single anchor at start
    anchors = np.array([True, False, False, False, False, False])
    anchored_vwap = kimsfinance_core.calculate_vwap_anchored(high, low, close, volume, anchors)

    # Should match exactly
    np.testing.assert_allclose(anchored_vwap, regular_vwap, rtol=1e-10, atol=1e-10)
    print("✓ Anchored VWAP (single anchor) matches regular VWAP")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("VWAP Anchored - Rust vs Python Parity Tests")
    print("=" * 70)

    try:
        test_basic_anchored_vwap()
        test_single_anchor()
        test_multiple_anchors()
        test_no_anchors()
        test_edge_case_single_value()
        test_consistency_with_regular_vwap()
        test_performance_comparison()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"TEST FAILED ✗: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
