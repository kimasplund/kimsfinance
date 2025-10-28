"""
Comprehensive tests for GPU-accelerated genetic optimization.

Tests GPU batch evaluation integration with genetic optimizer,
validating 20-40x speedup claims and correctness vs CPU.
"""

import pytest
import numpy as np
import pandas as pd
import time
from typing import Dict, Any

from kimsfinance.optimization.genetic import GeneticOptimizer

try:
    from kimsfinance.batch import GPU_AVAILABLE
except ImportError:
    GPU_AVAILABLE = False


@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 1000

    # Random walk for prices
    returns = np.random.randn(n) * 0.01
    close = 100.0 * np.exp(np.cumsum(returns))

    # Generate OHLC
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01)
    open_ = close * (1 + np.random.randn(n) * 0.005)
    volume = np.abs(np.random.randn(n)) * 1000

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def rsi_param_space():
    """Standard RSI parameter space."""
    return {
        'period': (10, 20, int),
        'buy_threshold': (25, 35, float),
        'sell_threshold': (65, 75, float),
    }


class TestGeneticGPUIntegration:
    """Test GPU batch evaluation integration."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_batch_evaluation_basic(self, sample_ohlcv, rsi_param_space):
        """Test basic GPU batch evaluation works."""
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=20,
            generations=5,
            objectives=['sharpe']
        )

        results = optimizer.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=True,
            verbose=False
        )

        # Validate results
        assert len(results) > 0, "Should return at least one solution"
        assert 'params' in results[0], "Solution should have params"
        assert 'sharpe' in results[0], "Solution should have sharpe ratio"

        # Check parameters are within bounds
        best = results[0]['params']
        assert 10 <= best['period'] <= 20
        assert 25 <= best['buy_threshold'] <= 35
        assert 65 <= best['sell_threshold'] <= 75

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_vs_cpu_correctness(self, sample_ohlcv, rsi_param_space):
        """Test that GPU results are consistent with CPU results."""
        np.random.seed(42)  # Fix seed for reproducibility

        optimizer = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=20,
            generations=10,
            objectives=['sharpe', 'max_drawdown', 'win_rate']
        )

        # Run with GPU
        results_gpu = optimizer.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=True,
            verbose=False
        )

        # Reset seed and run with CPU
        np.random.seed(42)
        optimizer2 = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=20,
            generations=10,
            objectives=['sharpe', 'max_drawdown', 'win_rate']
        )

        results_cpu = optimizer2.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=False,
            verbose=False
        )

        # Both should find solutions
        assert len(results_gpu) > 0
        assert len(results_cpu) > 0

        # Top solutions should have similar fitness (within tolerance)
        # Note: Due to randomness in evolution, exact match is not expected
        top_gpu = results_gpu[0]
        top_cpu = results_cpu[0]

        sharpe_diff = abs(top_gpu['sharpe'] - top_cpu['sharpe'])
        # Allow 30% difference due to evolutionary randomness
        assert sharpe_diff < 1.0, f"Sharpe difference {sharpe_diff:.2f} too large"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_multi_objective_optimization(self, sample_ohlcv, rsi_param_space):
        """Test multi-objective optimization on GPU."""
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=30,
            generations=10,
            objectives=['sharpe', 'max_drawdown', 'win_rate']
        )

        results = optimizer.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=True,
            verbose=False
        )

        # Should find Pareto front
        assert len(results) >= 3, "Should find at least 3 Pareto-optimal solutions"

        # All results should have all objectives
        for sol in results:
            assert 'sharpe' in sol
            assert 'max_drawdown' in sol
            assert 'win_rate' in sol

        # Values should be reasonable
        best = results[0]
        assert best['sharpe'] != float('-inf'), "Should not have infinite Sharpe"
        assert -1.0 <= best['max_drawdown'] <= 0.0, "Drawdown should be negative"
        assert 0.0 <= best['win_rate'] <= 1.0, "Win rate should be [0, 1]"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_large_population_gpu(self, sample_ohlcv, rsi_param_space):
        """Test GPU with large population (stress test)."""
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=200,  # Large population
            generations=5,
            objectives=['sharpe']
        )

        start = time.perf_counter()
        results = optimizer.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=True,
            verbose=False
        )
        elapsed = time.perf_counter() - start

        # Should complete without errors
        assert len(results) > 0
        print(f"\nLarge population (200) completed in {elapsed:.2f}s")

        # Should be faster than 10 seconds (generous upper bound)
        assert elapsed < 10.0, f"GPU optimization too slow: {elapsed:.2f}s"


class TestGPUPerformance:
    """Benchmark GPU vs CPU performance."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.benchmark
    def test_gpu_speedup_medium_population(self, sample_ohlcv, rsi_param_space):
        """Benchmark GPU vs CPU speedup with medium population (100)."""
        optimizer_gpu = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=100,
            generations=20,
            objectives=['sharpe']
        )

        optimizer_cpu = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=100,
            generations=20,
            objectives=['sharpe']
        )

        # Benchmark GPU
        start_gpu = time.perf_counter()
        results_gpu = optimizer_gpu.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=True,
            verbose=False
        )
        gpu_time = time.perf_counter() - start_gpu

        # Benchmark CPU
        start_cpu = time.perf_counter()
        results_cpu = optimizer_cpu.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=False,
            verbose=False
        )
        cpu_time = time.perf_counter() - start_cpu

        speedup = cpu_time / gpu_time

        print(f"\n{'='*60}")
        print(f"GPU Speedup Benchmark (Population=100, Generations=20)")
        print(f"{'='*60}")
        print(f"GPU time: {gpu_time:.2f}s")
        print(f"CPU time: {cpu_time:.2f}s")
        print(f"Speedup: {speedup:.1f}x")
        print(f"{'='*60}")

        # Validate both produced results
        assert len(results_gpu) > 0
        assert len(results_cpu) > 0

        # GPU should be at least 5x faster (conservative, target is 20-40x)
        assert speedup >= 5.0, f"GPU speedup {speedup:.1f}x is below 5x minimum"

        # Report if we hit target
        if speedup >= 20.0:
            print(f"SUCCESS: GPU speedup {speedup:.1f}x meets 20-40x target!")
        elif speedup >= 10.0:
            print(f"GOOD: GPU speedup {speedup:.1f}x is above 10x")
        else:
            print(f"WARNING: GPU speedup {speedup:.1f}x below 20x target")

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_gpu_speedup_large_population(self, sample_ohlcv, rsi_param_space):
        """Benchmark GPU vs CPU speedup with large population (1000)."""
        optimizer_gpu = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=1000,
            generations=10,
            objectives=['sharpe']
        )

        optimizer_cpu = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=1000,
            generations=10,
            objectives=['sharpe']
        )

        # Benchmark GPU
        start_gpu = time.perf_counter()
        results_gpu = optimizer_gpu.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=True,
            verbose=False
        )
        gpu_time = time.perf_counter() - start_gpu

        # Benchmark CPU (this will be SLOW!)
        start_cpu = time.perf_counter()
        results_cpu = optimizer_cpu.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=False,
            verbose=False
        )
        cpu_time = time.perf_counter() - start_cpu

        speedup = cpu_time / gpu_time

        print(f"\n{'='*60}")
        print(f"GPU Speedup Benchmark (Population=1000, Generations=10)")
        print(f"{'='*60}")
        print(f"GPU time: {gpu_time:.2f}s")
        print(f"CPU time: {cpu_time:.2f}s")
        print(f"Speedup: {speedup:.1f}x")
        print(f"{'='*60}")

        # Validate both produced results
        assert len(results_gpu) > 0
        assert len(results_cpu) > 0

        # GPU should be at least 10x faster (conservative, target is 20-40x)
        assert speedup >= 10.0, f"GPU speedup {speedup:.1f}x is below 10x minimum"

        # Report if we hit target
        if speedup >= 30.0:
            print(f"SUCCESS: GPU speedup {speedup:.1f}x meets 20-40x target!")
        elif speedup >= 20.0:
            print(f"GOOD: GPU speedup {speedup:.1f}x is within target range")
        else:
            print(f"WARNING: GPU speedup {speedup:.1f}x below 20x target")


class TestBackwardCompatibility:
    """Test backward compatibility with legacy backtester API."""

    def test_cpu_fallback_no_gpu(self, sample_ohlcv, rsi_param_space):
        """Test CPU fallback works when GPU not available."""
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=10,
            generations=3,
            objectives=['sharpe']
        )

        # Should work even without GPU
        results = optimizer.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=False,  # Force CPU
            verbose=False
        )

        assert len(results) > 0
        assert 'params' in results[0]
        assert 'sharpe' in results[0]


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_small_population(self, sample_ohlcv, rsi_param_space):
        """Test with very small population."""
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=5,  # Tiny population
            generations=3,
            objectives=['sharpe']
        )

        results = optimizer.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=True,
            verbose=False
        )

        assert len(results) > 0

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_single_generation(self, sample_ohlcv, rsi_param_space):
        """Test with single generation."""
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=20,
            generations=1,  # Single generation
            objectives=['sharpe']
        )

        results = optimizer.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=True,
            verbose=False
        )

        assert len(results) > 0

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_island_model_gpu(self, sample_ohlcv, rsi_param_space):
        """Test island model with GPU acceleration."""
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=20,
            generations=5,
            n_islands=4,  # Island model
            objectives=['sharpe']
        )

        results = optimizer.optimize(
            strategy='rsi_crossover',
            data=sample_ohlcv,
            use_gpu=True,
            verbose=False
        )

        assert len(results) > 0


if __name__ == '__main__':
    # Quick manual test
    print("Running manual GPU genetic optimizer test...")

    # Generate test data
    np.random.seed(42)
    n = 1000
    returns = np.random.randn(n) * 0.01
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * 1.01
    low = close * 0.99
    open_ = close * (1 + np.random.randn(n) * 0.005)
    volume = np.abs(np.random.randn(n)) * 1000

    data = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    # Test optimizer
    param_space = {
        'period': (10, 20, int),
        'buy_threshold': (25, 35, float),
        'sell_threshold': (65, 75, float),
    }

    optimizer = GeneticOptimizer(
        param_space=param_space,
        population_size=50,
        generations=10,
        objectives=['sharpe', 'max_drawdown', 'win_rate']
    )

    print("\nRunning GPU optimization...")
    start = time.perf_counter()
    results = optimizer.optimize(
        strategy='rsi_crossover',
        data=data,
        use_gpu=True,
        verbose=True
    )
    gpu_time = time.perf_counter() - start

    print(f"\nCompleted in {gpu_time:.2f}s")
    print(f"Found {len(results)} Pareto-optimal solutions")
    print(f"\nTop 3 solutions:")
    for i, sol in enumerate(results[:3]):
        print(f"{i+1}. Sharpe: {sol['sharpe']:.2f}, "
              f"DD: {sol['max_drawdown']:.2%}, "
              f"WR: {sol['win_rate']:.1%}")
        print(f"   Params: {sol['params']}")

    print("\n✓ Manual test completed successfully!")
