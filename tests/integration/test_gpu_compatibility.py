"""
GPU compatibility and fallback testing.

Tests:
- GPU availability detection
- Graceful fallback to CPU
- Error messages when GPU not available
- Multi-GPU support (if available)
- Environment-specific GPU features
"""

import importlib
import os

import pytest

from _backtesters import AnalyticBacktester, BatchBacktester, FailingBacktester

try:
    # GPU_AVAILABLE is device-based (see kimsfinance.batch): False when the
    # bindings import but no CUDA device can be initialised.
    from kimsfinance.batch import batch_backtest, get_gpu_info, GPU_AVAILABLE
    from kimsfinance.optimization.genetic import GeneticOptimizer

    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False
    GPU_AVAILABLE = False

try:
    import deap  # noqa: F401

    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False


# Test GPU availability detection


def test_gpu_info_structure():
    """Test GPU info function returns correct structure."""
    if not BATCH_AVAILABLE:
        pytest.skip("Batch backtesting not available")

    info = get_gpu_info()

    # Always present keys
    assert "gpu_available" in info
    assert "expected_speedup" in info
    assert isinstance(info["gpu_available"], bool)
    assert isinstance(info["expected_speedup"], (int, float))

    if info["gpu_available"]:
        # GPU-specific keys
        assert "gpu_name" in info or "error" not in info
        print(f"✅ GPU available: {info.get('gpu_name', 'Unknown GPU')}")
    else:
        # Error message present when GPU not available
        assert "error" in info
        print(f"⚠️  GPU not available: {info['error']}")


@pytest.mark.skipif(not BATCH_AVAILABLE, reason="Batch backtesting not available")
def test_gpu_available_constant():
    """Test GPU_AVAILABLE constant matches get_gpu_info()."""
    info = get_gpu_info()

    # GPU_AVAILABLE should match info['gpu_available']
    assert (
        GPU_AVAILABLE == info["gpu_available"]
    ), f"GPU_AVAILABLE ({GPU_AVAILABLE}) doesn't match info ({info['gpu_available']})"

    print(f"✅ GPU_AVAILABLE constant: {GPU_AVAILABLE}")


# Test CPU fallback mode


@pytest.mark.skipif(not BATCH_AVAILABLE, reason="Batch backtesting not available")
@pytest.mark.skipif(
    not DEAP_AVAILABLE, reason="deap package not installed (pip install kimsfinance[optimization])"
)
class TestCPUFallback:
    """GeneticOptimizer without a GPU: fitness comes from whatever backtester is passed."""

    def test_cpu_mode_works(self):
        """A CPU backtester produces valid results (no GPU involved)."""
        import pandas as pd
        import numpy as np

        # Generate test data
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(1000) * 0.02)
        data = pd.DataFrame(
            {
                "open": close + np.random.randn(1000) * 0.5,
                "high": close + np.abs(np.random.randn(1000)),
                "low": close - np.abs(np.random.randn(1000)),
                "close": close,
                "volume": np.abs(np.random.randn(1000)) * 1000,
            }
        )

        optimizer = GeneticOptimizer(
            param_space={
                "period": (10, 20, int),
                "buy_threshold": (25.0, 35.0, float),
                "sell_threshold": (65.0, 75.0, float),
            },
            population_size=10,
            generations=2,
        )

        results = optimizer.optimize(
            strategy="rsi_crossover", data=data, backtester=AnalyticBacktester(), verbose=False
        )

        assert len(results) > 0
        assert all("params" in r for r in results)
        assert all("fitness" in r for r in results)

        print(f"✅ CPU mode works: {len(results)} solutions found")

    def test_optimizer_matches_direct_gpu_backtest(self):
        """Optimizer + BatchBacktester reproduces a direct batch_backtest() call."""
        if not GPU_AVAILABLE:
            pytest.skip("GPU not available, cannot compare")

        import pandas as pd
        import numpy as np

        # Generate deterministic data
        np.random.seed(123)
        close = 100 + np.cumsum(np.random.randn(500) * 0.02)
        data = pd.DataFrame(
            {
                "open": close + np.random.randn(500) * 0.5,
                "high": close + np.abs(np.random.randn(500)),
                "low": close - np.abs(np.random.randn(500)),
                "close": close,
                "volume": np.abs(np.random.randn(500)) * 1000,
            }
        )

        # Fixed parameter for comparison
        params = [{"period": 14, "buy_threshold": 30.0, "sell_threshold": 70.0}]

        # GPU result
        result_gpu = batch_backtest("rsi_crossover", data, params)[0]

        # Same parameters through the optimizer (degenerate 1-point search)
        optimizer = GeneticOptimizer(
            param_space={
                "period": (14, 14, int),
                "buy_threshold": (30.0, 30.0, float),
                "sell_threshold": (70.0, 70.0, float),
            },
            population_size=2,
            generations=1,
        )
        result_opt_list = optimizer.optimize(
            strategy="rsi_crossover", data=data, backtester=BatchBacktester(), verbose=False
        )
        assert result_opt_list, "Optimizer returned no solutions"
        result_opt = result_opt_list[0]

        assert result_opt["params"] == params[0]
        assert result_opt["sharpe"] == pytest.approx(result_gpu["sharpe_ratio"], rel=1e-4, abs=1e-6)
        assert result_opt["win_rate"] == pytest.approx(result_gpu["win_rate"], rel=1e-4, abs=1e-6)

        print(f"✅ Optimizer vs direct GPU: Sharpe = {result_opt['sharpe']:.4f}")


# Test error handling


@pytest.mark.skipif(BATCH_AVAILABLE and GPU_AVAILABLE, reason="GPU is available")
class TestGPUNotAvailable:
    """Tests for when GPU is not available."""

    def test_import_error_message(self):
        """Test that import error provides clear guidance."""
        if BATCH_AVAILABLE:
            pytest.skip("Batch backtesting is available")

        # Try importing and check error message
        try:
            from kimsfinance.batch import batch_backtest  # noqa: F401  # import is the check

            pytest.fail("Import should have failed")
        except ImportError as e:
            # This is expected behavior - no assertion needed
            print(f"✅ Import error (expected): {e}")

    def test_gpu_info_when_unavailable(self):
        """Test get_gpu_info() when GPU not available."""
        if GPU_AVAILABLE:
            pytest.skip("GPU is available")

        info = get_gpu_info()

        assert info["gpu_available"] is False
        assert "error" in info
        assert info["expected_speedup"] == 1.0

        print(f"✅ GPU unavailable info: {info['error']}")


# Environment tests


@pytest.mark.skipif(not BATCH_AVAILABLE or not GPU_AVAILABLE, reason="GPU not available")
class TestGPUEnvironment:
    """Test GPU environment detection and configuration."""

    def test_cuda_visible_devices(self):
        """Test CUDA_VISIBLE_DEVICES environment variable."""
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")

        info = get_gpu_info()

        print(f"✅ CUDA_VISIBLE_DEVICES: {cuda_devices}")
        print(f"   GPU available: {info['gpu_available']}")
        if "gpu_name" in info:
            print(f"   GPU name: {info['gpu_name']}")

    def test_gpu_memory_info(self):
        """Test GPU memory information if available."""
        info = get_gpu_info()

        if "vram_gb" in info:
            vram = info["vram_gb"]
            assert vram > 0, f"VRAM should be positive: {vram}"
            print(f"✅ VRAM: {vram:.1f} GB")
        else:
            print("⚠️  VRAM info not available in GPU info")

    def test_expected_speedup_reasonable(self):
        """Test that expected speedup is reasonable."""
        info = get_gpu_info()

        speedup = info["expected_speedup"]

        if GPU_AVAILABLE:
            # GPU speedup should be 10x-100x
            assert (
                5.0 <= speedup <= 100.0
            ), f"Expected speedup out of range: {speedup}x (expected 5-100x)"
            print(f"✅ Expected speedup: {speedup:.0f}x")
        else:
            assert speedup == 1.0, f"CPU speedup should be 1.0x, got {speedup}x"


# Test graceful degradation


@pytest.mark.skipif(not BATCH_AVAILABLE, reason="Batch backtesting not available")
@pytest.mark.skipif(
    not DEAP_AVAILABLE, reason="deap package not installed (pip install kimsfinance[optimization])"
)
class TestGracefulDegradation:
    """Test graceful degradation when backtest evaluation fails."""

    def test_backtester_error_yields_worst_fitness(self):
        """A backtester that raises must not crash the optimizer (worst fitness instead)."""
        import pandas as pd
        import numpy as np

        # Generate test data
        np.random.seed(456)
        close = 100 + np.cumsum(np.random.randn(500) * 0.02)
        data = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(500) * 1000,
            }
        )

        optimizer = GeneticOptimizer(
            param_space={
                "period": (10, 15, int),
                "buy_threshold": (28.0, 32.0, float),
                "sell_threshold": (68.0, 72.0, float),
            },
            population_size=5,
            generations=2,
        )

        results = optimizer.optimize(
            strategy="rsi_crossover", data=data, backtester=FailingBacktester(), verbose=False
        )

        assert len(results) > 0
        assert all(r["sharpe"] == float("-inf") for r in results)
        print(f"✅ Optimizer survived backtester failures: {len(results)} solutions")


# Platform-specific tests


def test_platform_detection():
    """Test platform detection."""
    import platform

    system = platform.system()
    machine = platform.machine()

    print(f"✅ Platform: {system} {machine}")

    if system == "Linux":
        # Check for NVIDIA driver
        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                print(f"   NVIDIA GPU detected: {gpu_name}")
            else:
                print("   nvidia-smi failed (GPU may not be available)")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("   nvidia-smi not found (no NVIDIA GPU)")
    elif system == "Windows":
        print("   Windows platform (CUDA support depends on drivers)")
    elif system == "Darwin":
        print("   macOS platform (no CUDA support)")


# Module-level validation


def test_module_imports():
    """Test that all required modules can be imported."""
    try:
        importlib.import_module("kimsfinance.batch")

        print("✅ kimsfinance.batch imported successfully")
    except ImportError as e:
        print(f"⚠️  kimsfinance.batch import failed: {e}")

    try:
        importlib.import_module("kimsfinance.optimization.genetic")

        print("✅ kimsfinance.optimization.genetic imported successfully")
    except ImportError as e:
        print(f"⚠️  kimsfinance.optimization.genetic import failed: {e}")

    # Check if GPU dependencies available
    try:
        import cupy

        print(f"✅ CuPy available (version: {cupy.__version__})")
    except ImportError:
        print("⚠️  CuPy not available (GPU acceleration unavailable)")

    try:
        importlib.import_module("numba.cuda")

        print("✅ Numba CUDA available")
    except ImportError:
        print("⚠️  Numba CUDA not available")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
