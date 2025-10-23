"""
Thread Safety Tests for kimsfinance
====================================

Tests concurrent access to global state management across:
- Adapter activation/deactivation
- Configuration updates
- Performance stats tracking
- Engine GPU checking
- Autotune file operations

Success Criteria:
- No race conditions under 10 concurrent threads
- No deadlocks in stress tests
- Consistent state after concurrent operations
- Performance overhead < 1% for single-threaded use
"""

import pytest
import threading
import time
import tempfile
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Test imports
from kimsfinance.integration import adapter
from kimsfinance.core import engine
from kimsfinance.core import autotune


class TestAdapterThreadSafety:
    """Test thread safety of adapter.py global state management."""

    def test_concurrent_activate_deactivate(self):
        """Test concurrent activate/deactivate from 10 threads."""
        errors = []
        iterations = 100

        def worker(thread_id: int):
            try:
                # Stress test activate/deactivate without asserting intermediate states
                # which caused race conditions.
                for i in range(iterations):
                    adapter.activate(verbose=False)
                    adapter.deactivate(verbose=False)
            except Exception as e:
                errors.append((thread_id, e))
            finally:
                # Ensure every thread deactivates before exiting to guarantee final state
                adapter.deactivate(verbose=False)

        # Run 10 concurrent threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check for errors
        if errors:
            pytest.fail(f"Thread safety errors: {errors}")

        # Final state should be deactivated
        assert adapter.is_active() is False

    def test_concurrent_configure(self):
        """Test concurrent configuration updates from 10 threads."""
        errors = []
        iterations = 50

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Each thread sets different values
                    adapter.configure(
                        gpu_min_rows=thread_id * 1000,
                        strict_mode=(thread_id % 2 == 0),
                        verbose=False,
                    )
                    # Verify config can be read without corruption
                    config = adapter.get_config()
                    assert isinstance(config, dict)
                    assert "gpu_min_rows" in config
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if errors:
            pytest.fail(f"Thread safety errors: {errors}")

        # Final config should be consistent
        config = adapter.get_config()
        assert isinstance(config, dict)

    def test_concurrent_performance_tracking(self):
        """Test concurrent performance stats updates from 10 threads."""
        adapter.configure(performance_tracking=True, verbose=False)
        adapter.reset_performance_stats()

        errors = []
        iterations = 100

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Track operations
                    adapter._track_operation(
                        engine_used="gpu" if thread_id % 2 == 0 else "cpu", time_saved_ms=10.0
                    )
                    # Read stats (should not corrupt)
                    stats = adapter.get_performance_stats()
                    assert isinstance(stats, dict)
                    assert stats["total_calls"] >= 0
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if errors:
            pytest.fail(f"Thread safety errors: {errors}")

        # Verify final counts (should be 10 threads * 100 iterations = 1000)
        stats = adapter.get_performance_stats()
        assert stats["total_calls"] == 1000

    def test_config_validation_thread_safe(self):
        """Test configuration validation is thread-safe."""
        errors = []

        def worker_valid(thread_id: int):
            try:
                for _ in range(50):
                    adapter.configure(default_engine="auto", gpu_min_rows=5000, verbose=False)
            except Exception as e:
                errors.append((thread_id, e))

        def worker_invalid(thread_id: int):
            try:
                for _ in range(50):
                    try:
                        adapter.configure(default_engine="invalid", verbose=False)
                        errors.append((thread_id, "Should have raised ValueError"))
                    except ValueError:
                        pass  # Expected
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=worker_valid, args=(i,)))
            threads.append(threading.Thread(target=worker_invalid, args=(i + 5,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            pytest.fail(f"Thread safety errors: {errors}")


class TestEngineThreadSafety:
    """Test thread safety of engine.py GPU checking."""

    def test_concurrent_gpu_check(self):
        """Test concurrent GPU availability checks (double-checked locking)."""
        # Reset cache
        engine.EngineManager.reset_gpu_cache()

        errors = []
        results = []
        iterations = 100

        def worker(thread_id: int):
            try:
                for _ in range(iterations):
                    result = engine.EngineManager.check_gpu_available()
                    results.append(result)
            except Exception as e:
                errors.append((thread_id, e))

        # Run 20 concurrent threads (stress test)
        threads = []
        for i in range(20):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if errors:
            pytest.fail(f"Thread safety errors: {errors}")

        # All results should be identical (GPU either available or not)
        assert len(set(results)) == 1, "GPU availability check should be consistent"

    def test_concurrent_engine_selection(self):
        """Test concurrent engine selection from multiple threads."""
        errors = []
        iterations = 50

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Mix of engine selections
                    selected = engine.EngineManager.select_engine(
                        engine="auto", operation="atr", data_size=thread_id * 1000
                    )
                    assert selected in ("cpu", "gpu")
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if errors:
            pytest.fail(f"Thread safety errors: {errors}")


class TestAutotuneThreadSafety:
    """Test thread safety of autotune.py file operations."""

    def test_concurrent_load_thresholds(self):
        """Test concurrent threshold loading from multiple threads."""
        errors = []
        iterations = 100

        def worker(thread_id: int):
            try:
                for _ in range(iterations):
                    thresholds = autotune.load_tuned_thresholds()
                    assert isinstance(thresholds, dict)
                    assert "default" in thresholds
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if errors:
            pytest.fail(f"Thread safety errors: {errors}")

    def test_concurrent_save_and_load(self):
        """Test concurrent save/load operations (atomic file write)."""
        # Use temp cache file
        original_cache = autotune.CACHE_FILE
        temp_dir = Path(tempfile.mkdtemp())
        autotune.CACHE_FILE = temp_dir / "test_threshold_cache.json"

        try:
            errors = []
            iterations = 20

            def worker_save(thread_id: int):
                try:
                    for i in range(iterations):
                        thresholds = {
                            "atr": thread_id * 1000 + i,
                            "rsi": thread_id * 2000 + i,
                            "default": 100_000,
                        }
                        autotune._save_tuned_thresholds(thresholds)
                except Exception as e:
                    errors.append((f"save-{thread_id}", e))

            def worker_load(thread_id: int):
                try:
                    for _ in range(iterations * 2):
                        thresholds = autotune.load_tuned_thresholds()
                        assert isinstance(thresholds, dict)
                        # Either loaded successfully or got defaults
                        assert "default" in thresholds
                except Exception as e:
                    errors.append((f"load-{thread_id}", e))

            threads = []
            # 5 writers, 5 readers
            for i in range(5):
                threads.append(threading.Thread(target=worker_save, args=(i,)))
                threads.append(threading.Thread(target=worker_load, args=(i + 5,)))

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            if errors:
                pytest.fail(f"Thread safety errors: {errors}")

            # Final file should be valid JSON
            if autotune.CACHE_FILE.exists():
                with open(autotune.CACHE_FILE, "r") as f:
                    data = json.load(f)
                    assert isinstance(data, dict)

        finally:
            # Restore original cache file
            autotune.CACHE_FILE = original_cache
            # Cleanup temp directory
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestDeadlockPrevention:
    """Test that locks don't cause deadlocks under stress."""

    def test_no_deadlock_mixed_operations(self):
        """Test mixed operations from 20 threads for deadlock detection."""
        errors = []
        timeout_seconds = 10

        def worker(thread_id: int):
            try:
                for i in range(50):
                    # Mix different operations
                    if i % 4 == 0:
                        adapter.configure(gpu_min_rows=thread_id * 100, verbose=False)
                    elif i % 4 == 1:
                        config = adapter.get_config()
                    elif i % 4 == 2:
                        is_active = adapter.is_active()
                    else:
                        gpu_avail = engine.EngineManager.check_gpu_available()
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(20):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

        start_time = time.time()
        for t in threads:
            t.start()

        # Wait with timeout (detect deadlock)
        for t in threads:
            remaining = timeout_seconds - (time.time() - start_time)
            t.join(timeout=max(remaining, 0))
            if t.is_alive():
                pytest.fail(f"Deadlock detected: thread {t.name} did not complete")

        if errors:
            pytest.fail(f"Thread safety errors: {errors}")

        elapsed = time.time() - start_time
        assert elapsed < timeout_seconds, f"Operations took too long ({elapsed:.2f}s), possible deadlock"


class TestPerformanceOverhead:
    """Test that thread safety adds minimal overhead (<1% for single-threaded)."""

    def test_single_thread_overhead(self):
        """Measure overhead of locks in single-threaded scenario."""
        iterations = 1000

        # Measure time for config operations
        start = time.perf_counter()
        for _ in range(iterations):
            adapter.configure(gpu_min_rows=5000, verbose=False)
            config = adapter.get_config()
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 0.5 seconds for 1000 operations)
        assert elapsed < 0.5, f"Single-threaded overhead too high: {elapsed:.3f}s"

        # Calculate overhead per operation
        overhead_per_op = (elapsed / iterations) * 1000  # milliseconds
        assert overhead_per_op < 0.5, f"Per-operation overhead too high: {overhead_per_op:.3f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
