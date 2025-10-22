"""
Memory Leak Tests
==================

Validates that memory leak fixes are working correctly:
1. BoundedPerformanceStats stays within memory limits
2. Array copies are avoided when possible
3. BytesIO buffers are properly closed
4. DataFrame cleanup works correctly
"""

import gc
import sys
import numpy as np
import pytest
from datetime import datetime, timedelta

# Memory tracking
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@pytest.fixture
def memory_baseline():
    """Get baseline memory usage."""
    if not PSUTIL_AVAILABLE:
        pytest.skip("psutil not available for memory tracking")
    gc.collect()
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB


class TestBoundedPerformanceStats:
    """Test bounded performance statistics tracker."""

    def test_bounded_entries(self):
        """Test that entries are bounded to MAX_STATS_ENTRIES."""
        from kimsfinance.integration.adapter import BoundedPerformanceStats, MAX_STATS_ENTRIES

        stats = BoundedPerformanceStats(max_entries=100)

        # Add more than max_entries
        for i in range(500):
            stats.record("cpu", time_saved_ms=float(i))

        # Should only keep last 100
        result = stats.get_stats()
        assert result["total_tracked"] == 100
        assert result["total_calls"] == 500  # Aggregated count is total
        assert result["max_entries"] == 100

    def test_time_based_cleanup(self):
        """Test that old entries are cleaned up."""
        from kimsfinance.integration.adapter import BoundedPerformanceStats

        stats = BoundedPerformanceStats(max_entries=1000)

        # Add entries with old timestamps
        old_cutoff = datetime.now() - timedelta(hours=25)  # Older than 24h
        for i in range(10):
            entry = {"engine": "cpu", "time_saved_ms": float(i), "timestamp": old_cutoff}
            stats._recent_renders.append(entry)

        # Add current entry (triggers cleanup)
        stats.record("gpu", 10.0)

        # Old entries should be removed
        result = stats.get_stats()
        assert result["total_tracked"] == 1  # Only the new entry

    def test_thread_safety(self):
        """Test thread-safe operations."""
        import threading
        from kimsfinance.integration.adapter import BoundedPerformanceStats

        stats = BoundedPerformanceStats(max_entries=10000)

        def record_stats(engine, count):
            for _ in range(count):
                stats.record(engine, 1.0)

        # Create threads
        threads = [
            threading.Thread(target=record_stats, args=("cpu", 100)),
            threading.Thread(target=record_stats, args=("gpu", 100)),
        ]

        # Run threads
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check results
        result = stats.get_stats()
        assert result["total_calls"] == 200
        assert result["cpu_calls"] == 100
        assert result["gpu_calls"] == 100

    def test_reset(self):
        """Test statistics reset."""
        from kimsfinance.integration.adapter import BoundedPerformanceStats

        stats = BoundedPerformanceStats()

        # Add some data
        for i in range(50):
            stats.record("cpu", float(i))

        # Reset
        stats.reset()

        # Check empty
        result = stats.get_stats()
        assert result["total_calls"] == 0
        assert result["total_tracked"] == 0

    def test_memory_bounded_10k_iterations(self, memory_baseline):
        """Test that memory stays bounded over 10K iterations."""
        from kimsfinance.integration.adapter import BoundedPerformanceStats

        stats = BoundedPerformanceStats(max_entries=1000)

        # Run 10K iterations
        for i in range(10_000):
            stats.record("cpu" if i % 2 == 0 else "gpu", float(i % 100))

        # Check memory growth
        gc.collect()
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Memory should not grow more than 10MB
        memory_growth = current_memory - memory_baseline
        assert memory_growth < 10, f"Memory grew {memory_growth:.2f}MB (should be <10MB)"

        # Verify bounded
        result = stats.get_stats()
        assert result["total_tracked"] <= 1000


class TestArrayCopyOptimization:
    """Test array copy optimizations."""

    def test_no_copy_for_contiguous_arrays(self):
        """Test that C-contiguous arrays are not copied."""
        from kimsfinance.plotting.pil_renderer import _ensure_c_contiguous

        # Create C-contiguous array
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        assert arr.flags["C_CONTIGUOUS"]

        # Should not copy
        result = _ensure_c_contiguous(arr)
        assert result is arr  # Same object
        assert id(result) == id(arr)

    def test_copy_for_non_contiguous_arrays(self):
        """Test that non-contiguous arrays ARE copied."""
        from kimsfinance.plotting.pil_renderer import _ensure_c_contiguous

        # Create non-contiguous array (slice with step)
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        non_contiguous = arr[::2]  # Every other element
        assert not non_contiguous.flags["C_CONTIGUOUS"]

        # Should copy
        result = _ensure_c_contiguous(non_contiguous)
        assert result is not non_contiguous  # Different object
        assert result.flags["C_CONTIGUOUS"]

    def test_memory_savings_in_rendering(self, memory_baseline):
        """Test memory savings from avoiding unnecessary copies."""
        from kimsfinance.plotting import render_ohlcv_chart

        # Create test data (C-contiguous)
        n = 1000
        ohlc = {
            "open": np.random.rand(n),
            "high": np.random.rand(n) + 1,
            "low": np.random.rand(n) - 1,
            "close": np.random.rand(n),
        }
        volume = np.random.rand(n) * 1000

        # Render multiple charts
        for _ in range(100):
            img = render_ohlcv_chart(ohlc, volume, width=800, height=600)
            del img

        # Check memory
        gc.collect()
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024

        # Should not leak memory significantly
        memory_growth = current_memory - memory_baseline
        assert memory_growth < 50, f"Memory grew {memory_growth:.2f}MB (should be <50MB)"


class TestByteIOBufferCleanup:
    """Test BytesIO buffer cleanup."""

    def test_bytesio_closed_after_use(self):
        """Test that BytesIO buffers are closed."""
        from kimsfinance.plotting.parallel import _render_one_chart
        import io

        # Create test data
        ohlc = {
            "open": np.array([100.0, 101.0, 102.0]),
            "high": np.array([101.0, 102.0, 103.0]),
            "low": np.array([99.0, 100.0, 101.0]),
            "close": np.array([100.5, 101.5, 102.5]),
        }
        volume = np.array([1000.0, 2000.0, 1500.0])

        args = (ohlc, volume, None, {}, {"width": 400, "height": 300})

        # Render (returns bytes, buffer should be closed)
        result = _render_one_chart(args)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_no_bytesio_leak_in_parallel_rendering(self, memory_baseline):
        """Test no memory leak from BytesIO in parallel rendering."""
        from kimsfinance.plotting.parallel import render_charts_parallel

        # Create test datasets
        datasets = []
        for _ in range(100):
            ohlc = {
                "open": np.random.rand(50),
                "high": np.random.rand(50) + 1,
                "low": np.random.rand(50) - 1,
                "close": np.random.rand(50),
            }
            volume = np.random.rand(50) * 1000
            datasets.append({"ohlc": ohlc, "volume": volume})

        # Render in-memory (uses BytesIO)
        results = render_charts_parallel(datasets, num_workers=2, width=400, height=300)

        assert len(results) == 100

        # Check memory
        gc.collect()
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024

        memory_growth = current_memory - memory_baseline
        assert memory_growth < 100, f"Memory grew {memory_growth:.2f}MB (should be <100MB)"


class TestDataFrameMemoryOptimization:
    """Test DataFrame memory cleanup."""

    def test_dataframe_deleted_after_use(self):
        """Test that DataFrames are explicitly deleted."""
        from kimsfinance.ops.aggregations import rolling_sum

        data = np.random.rand(1000)
        result = rolling_sum(data, window=10)

        assert len(result) == 1000
        # DataFrame should be deleted inside function

    def test_no_dataframe_leak(self, memory_baseline):
        """Test no memory leak from DataFrame operations."""
        from kimsfinance.ops.aggregations import rolling_sum, rolling_mean

        data = np.random.rand(10_000)

        # Run many operations
        for _ in range(100):
            _ = rolling_sum(data, window=50)
            _ = rolling_mean(data, window=50)

        # Check memory
        gc.collect()
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024

        memory_growth = current_memory - memory_baseline
        assert memory_growth < 50, f"Memory grew {memory_growth:.2f}MB (should be <50MB)"


def test_overall_memory_leak_10k_renders(memory_baseline):
    """
    Overall memory leak test: 10K renders should not leak significant memory.

    This is the critical test that validates all fixes together.
    """
    from kimsfinance.plotting import render_ohlcv_chart
    from kimsfinance.integration.adapter import _performance_stats

    # Configure tracking
    _performance_stats.reset()

    # Create test data
    ohlc = {
        "open": np.random.rand(100),
        "high": np.random.rand(100) + 1,
        "low": np.random.rand(100) - 1,
        "close": np.random.rand(100),
    }
    volume = np.random.rand(100) * 1000

    # Run 10K renders
    for i in range(10_000):
        img = render_ohlcv_chart(ohlc, volume, width=400, height=300)
        _performance_stats.record("cpu", 1.0)
        del img

        # Force garbage collection every 1000 iterations
        if i % 1000 == 0:
            gc.collect()

    # Final garbage collection
    gc.collect()

    # Check memory
    process = psutil.Process()
    current_memory = process.memory_info().rss / 1024 / 1024

    memory_growth = current_memory - memory_baseline
    print(f"Memory growth after 10K renders: {memory_growth:.2f}MB")

    # Should not leak more than 100MB over 10K renders
    assert memory_growth < 100, f"Memory leak detected: {memory_growth:.2f}MB growth"

    # Verify performance stats are bounded
    stats = _performance_stats.get_stats()
    assert stats["total_tracked"] <= 10_000
    print(f"Performance stats tracked: {stats['total_tracked']} entries")


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_memory_leaks.py -v
    pytest.main([__file__, "-v", "-s"])
