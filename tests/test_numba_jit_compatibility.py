"""
Test Numba JIT Compatibility
=============================

Validates:
1. JIT compilation succeeds
2. JIT functions produce correct results
3. JIT functions match non-JIT versions
4. Performance improvements
5. Numerical stability
6. Thread safety

Success Criteria:
- All JIT functions compile without errors
- Numerical accuracy matches non-JIT versions (within floating point precision)
- No regressions in functionality
- Performance improvement measured (informational)
"""

from __future__ import annotations

import pytest
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        """Fallback decorator when Numba is not available."""

        def decorator(func):  # type: ignore
            return func

        return decorator


# ============================================================================
# Test Functions: JIT and Non-JIT versions
# ============================================================================


def calculate_sma_python(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate SMA without JIT (reference implementation)."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[: window - 1] = np.nan

    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1 : i + 1])

    return result


@njit(cache=True, fastmath=True)
def calculate_sma_jit(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate SMA with Numba JIT."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[: window - 1] = np.nan

    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1 : i + 1])

    return result


def rolling_max_python(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling max without JIT (reference implementation)."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[: window - 1] = np.nan

    for i in range(window - 1, n):
        result[i] = np.max(arr[max(0, i - window + 1) : i + 1])

    return result


@njit(cache=True, fastmath=True)
def rolling_max_jit(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling max with Numba JIT."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[: window - 1] = np.nan

    for i in range(window - 1, n):
        result[i] = np.max(arr[max(0, i - window + 1) : i + 1])

    return result


def replace_nan_python(arr: np.ndarray, value: float) -> np.ndarray:
    """Replace NaN values without JIT (reference implementation)."""
    result = arr.copy()
    for i in range(len(result)):
        if np.isnan(result[i]):
            result[i] = value
    return result


@njit(cache=True, fastmath=True)
def replace_nan_jit(arr: np.ndarray, value: float) -> np.ndarray:
    """Replace NaN values with Numba JIT."""
    result = arr.copy()
    for i in range(len(result)):
        if np.isnan(result[i]):
            result[i] = value
    return result


# ============================================================================
# Test Suite
# ============================================================================


class TestNumbaJITCompilation:
    """Test that JIT functions compile successfully."""

    def test_sma_jit_compilation(self):
        """Test SMA JIT function compiles without errors."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # This should not raise any compilation errors
        result = calculate_sma_jit(arr, window=3)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(arr)

    def test_rolling_max_jit_compilation(self):
        """Test rolling max JIT function compiles without errors."""
        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0])

        result = rolling_max_jit(arr, window=3)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(arr)

    def test_replace_nan_jit_compilation(self):
        """Test replace NaN JIT function compiles without errors."""
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

        result = replace_nan_jit(arr, value=0.0)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(arr)


class TestNumbaJITCorrectness:
    """Test JIT functions produce correct results."""

    def test_sma_jit_correctness(self):
        """Test SMA JIT function produces correct results."""
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        window = 3

        result = calculate_sma_jit(arr, window=window)

        # First window-1 values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # Remaining values should be correct
        np.testing.assert_almost_equal(result[2], 20.0, decimal=6)  # mean(10, 20, 30)
        np.testing.assert_almost_equal(result[3], 30.0, decimal=6)  # mean(20, 30, 40)
        np.testing.assert_almost_equal(result[4], 40.0, decimal=6)  # mean(30, 40, 50)

    def test_rolling_max_jit_correctness(self):
        """Test rolling max JIT function produces correct results."""
        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        window = 3

        result = rolling_max_jit(arr, window=window)

        # First window-1 values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # Remaining values should be correct
        np.testing.assert_almost_equal(result[2], 3.0, decimal=6)  # max(1, 3, 2)
        np.testing.assert_almost_equal(result[3], 5.0, decimal=6)  # max(3, 2, 5)
        np.testing.assert_almost_equal(result[4], 5.0, decimal=6)  # max(2, 5, 4)

    def test_replace_nan_jit_correctness(self):
        """Test replace NaN JIT function produces correct results."""
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        value = 0.0

        result = replace_nan_jit(arr, value=value)

        # Non-NaN values should be unchanged
        assert result[0] == 1.0
        assert result[2] == 3.0
        assert result[4] == 5.0

        # NaN values should be replaced
        assert result[1] == 0.0
        assert result[3] == 0.0


class TestNumbaJITVsPython:
    """Test JIT versions match pure Python versions."""

    def test_sma_jit_matches_python(self):
        """Test SMA JIT version matches pure Python version."""
        np.random.seed(42)
        arr = np.random.random(100) * 100
        window = 14

        python_result = calculate_sma_python(arr, window=window)
        jit_result = calculate_sma_jit(arr, window=window)

        # Results should be identical (allowing for floating point precision)
        np.testing.assert_array_almost_equal(python_result, jit_result, decimal=10)

    def test_rolling_max_jit_matches_python(self):
        """Test rolling max JIT version matches pure Python version."""
        np.random.seed(42)
        arr = np.random.random(100) * 100
        window = 14

        python_result = rolling_max_python(arr, window=window)
        jit_result = rolling_max_jit(arr, window=window)

        np.testing.assert_array_almost_equal(python_result, jit_result, decimal=10)

    def test_replace_nan_jit_matches_python(self):
        """Test replace NaN JIT version matches pure Python version."""
        np.random.seed(42)
        arr = np.random.random(100) * 100
        # Add some NaN values
        arr[10:20] = np.nan
        arr[50:55] = np.nan
        value = -999.0

        python_result = replace_nan_python(arr, value=value)
        jit_result = replace_nan_jit(arr, value=value)

        np.testing.assert_array_almost_equal(python_result, jit_result, decimal=10)


class TestNumbaJITEdgeCases:
    """Test JIT functions handle edge cases correctly."""

    def test_sma_jit_small_array(self):
        """Test SMA JIT with array smaller than window."""
        arr_small = np.array([1.0, 2.0])

        # Should handle gracefully
        result = calculate_sma_jit(arr_small, window=5)
        assert len(result) == 2
        # All values should be NaN since window > array size
        assert np.all(np.isnan(result))

    def test_sma_jit_window_equals_array_size(self):
        """Test SMA JIT when window equals array size."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calculate_sma_jit(arr, window=5)
        assert len(result) == 5
        # First 4 should be NaN, last should be mean of all
        assert np.all(np.isnan(result[:4]))
        np.testing.assert_almost_equal(result[4], 3.0, decimal=6)  # mean(1,2,3,4,5)

    def test_rolling_max_jit_with_negatives(self):
        """Test rolling max JIT handles negative values correctly."""
        arr = np.array([-5.0, -3.0, -7.0, -2.0, -4.0])
        window = 3

        result = rolling_max_jit(arr, window=window)

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_almost_equal(result[2], -3.0, decimal=6)
        np.testing.assert_almost_equal(result[3], -2.0, decimal=6)
        np.testing.assert_almost_equal(result[4], -2.0, decimal=6)

    def test_replace_nan_jit_all_nan(self):
        """Test replace NaN JIT with all NaN values."""
        arr = np.array([np.nan, np.nan, np.nan, np.nan])
        value = 42.0

        result = replace_nan_jit(arr, value=value)

        # All should be replaced
        assert np.all(result == 42.0)

    def test_replace_nan_jit_no_nan(self):
        """Test replace NaN JIT with no NaN values."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        value = 42.0

        result = replace_nan_jit(arr, value=value)

        # Nothing should change
        np.testing.assert_array_equal(result, arr)


class TestNumbaJITNumericalStability:
    """Test JIT functions maintain numerical stability."""

    def test_sma_jit_large_numbers(self):
        """Test SMA JIT with large numbers."""
        arr_large = np.array([1e10, 1e10, 1e10, 1e10, 1e10])
        result_large = calculate_sma_jit(arr_large, window=3)
        assert not np.any(np.isinf(result_large[2:]))

    def test_sma_jit_small_numbers(self):
        """Test SMA JIT with small numbers."""
        arr_small = np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10])
        result_small = calculate_sma_jit(arr_small, window=3)
        # Check non-NaN values
        assert not np.any(np.isnan(result_small[2:]))
        assert not np.any(np.isinf(result_small[2:]))

    def test_sma_jit_mixed_scale(self):
        """Test SMA JIT with mixed scale numbers."""
        arr_mixed = np.array([1e10, 1e-10, 1e10, 1e-10, 1e10])
        result_mixed = calculate_sma_jit(arr_mixed, window=3)
        assert not np.any(np.isinf(result_mixed[2:]))

    def test_rolling_max_jit_large_numbers(self):
        """Test rolling max JIT with large numbers."""
        arr = np.array([1e15, 2e15, 1.5e15, 3e15, 2.5e15])
        result = rolling_max_jit(arr, window=3)
        assert not np.any(np.isinf(result[2:]))
        assert not np.any(np.isnan(result[2:]))

    def test_replace_nan_jit_large_value(self):
        """Test replace NaN JIT with large replacement value."""
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        value = 1e15

        result = replace_nan_jit(arr, value=value)

        assert result[1] == 1e15
        assert result[3] == 1e15
        assert not np.any(np.isinf(result))


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
class TestNumbaJITPerformance:
    """Test JIT provides performance improvements (informational)."""

    def test_sma_jit_performance(self):
        """Test SMA JIT provides performance improvement."""
        np.random.seed(42)
        arr = np.random.random(10000) * 100
        window = 14
        iterations = 100

        # Warmup JIT
        _ = calculate_sma_jit(arr, window=window)

        # Benchmark Python
        start = time.perf_counter()
        for _ in range(iterations):
            _ = calculate_sma_python(arr, window=window)
        python_time = time.perf_counter() - start

        # Benchmark JIT
        start = time.perf_counter()
        for _ in range(iterations):
            _ = calculate_sma_jit(arr, window=window)
        jit_time = time.perf_counter() - start

        speedup = python_time / jit_time
        print(
            f"\nSMA Numba JIT Speedup: {speedup:.2f}x "
            f"(Python: {python_time:.3f}s, JIT: {jit_time:.3f}s)"
        )

        # JIT should be faster or at least complete
        assert speedup > 0

    def test_rolling_max_jit_performance(self):
        """Test rolling max JIT provides performance improvement."""
        np.random.seed(42)
        arr = np.random.random(10000) * 100
        window = 14
        iterations = 100

        # Warmup JIT
        _ = rolling_max_jit(arr, window=window)

        # Benchmark Python
        start = time.perf_counter()
        for _ in range(iterations):
            _ = rolling_max_python(arr, window=window)
        python_time = time.perf_counter() - start

        # Benchmark JIT
        start = time.perf_counter()
        for _ in range(iterations):
            _ = rolling_max_jit(arr, window=window)
        jit_time = time.perf_counter() - start

        speedup = python_time / jit_time
        print(
            f"\nRolling Max Numba JIT Speedup: {speedup:.2f}x "
            f"(Python: {python_time:.3f}s, JIT: {jit_time:.3f}s)"
        )

        assert speedup > 0

    def test_replace_nan_jit_performance(self):
        """Test replace NaN JIT provides performance improvement."""
        np.random.seed(42)
        arr = np.random.random(10000) * 100
        # Add some NaN values
        arr[100:1000] = np.nan
        value = 0.0
        iterations = 100

        # Warmup JIT
        _ = replace_nan_jit(arr, value=value)

        # Benchmark Python
        start = time.perf_counter()
        for _ in range(iterations):
            _ = replace_nan_python(arr, value=value)
        python_time = time.perf_counter() - start

        # Benchmark JIT
        start = time.perf_counter()
        for _ in range(iterations):
            _ = replace_nan_jit(arr, value=value)
        jit_time = time.perf_counter() - start

        speedup = python_time / jit_time
        print(
            f"\nReplace NaN Numba JIT Speedup: {speedup:.2f}x "
            f"(Python: {python_time:.3f}s, JIT: {jit_time:.3f}s)"
        )

        assert speedup > 0


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
class TestNumbaJITThreadSafety:
    """Test JIT functions are thread-safe."""

    def test_sma_jit_thread_safety(self):
        """Test SMA JIT function is thread-safe."""
        np.random.seed(42)
        arr = np.random.random(1000) * 100
        window = 14

        def worker():
            return calculate_sma_jit(arr, window=window)

        # Run in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(10)]
            results = [f.result() for f in futures]

        # All results should be identical
        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)

    def test_rolling_max_jit_thread_safety(self):
        """Test rolling max JIT function is thread-safe."""
        np.random.seed(42)
        arr = np.random.random(1000) * 100
        window = 14

        def worker():
            return rolling_max_jit(arr, window=window)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(10)]
            results = [f.result() for f in futures]

        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)

    def test_replace_nan_jit_thread_safety(self):
        """Test replace NaN JIT function is thread-safe."""
        np.random.seed(42)
        arr = np.random.random(1000) * 100
        arr[100:200] = np.nan
        value = 0.0

        def worker():
            return replace_nan_jit(arr, value=value)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(10)]
            results = [f.result() for f in futures]

        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)


class TestNumbaAvailability:
    """Test Numba availability detection."""

    def test_numba_available_flag(self):
        """Test NUMBA_AVAILABLE flag is set correctly."""
        # This should match whether import succeeded
        try:
            import numba

            expected = True
        except ImportError:
            expected = False

        assert NUMBA_AVAILABLE == expected

    @pytest.mark.skipif(NUMBA_AVAILABLE, reason="Test only when Numba not installed")
    def test_njit_fallback_when_unavailable(self):
        """Test njit fallback decorator works when Numba unavailable."""

        @njit(cache=True, fastmath=True)
        def test_func(x):
            return x * 2

        # Should still work (as regular function)
        result = test_func(5)
        assert result == 10
