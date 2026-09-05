"""
GeneticOptimizer driven by the GPU batch backtester.

``GeneticOptimizer`` has no GPU switch of its own: it evaluates fitness through
whatever ``backtester`` object is passed to ``optimize()``. The GPU path is the
``BatchBacktester`` adapter over ``kimsfinance.batch.batch_backtest``; the CPU
fallback is any other backtester (here the deterministic ``AnalyticBacktester``).

GPU tests skip unless the Rust bindings can reach a usable CUDA device
(``tests/_gpu.py``), and are skipped in CI, which installs no CUDA wheels.
"""

import time

import numpy as np
import pandas as pd
import pytest

# GeneticOptimizer requires the optional 'deap' package (the [optimization] extra).
pytest.importorskip("deap")

from _backtesters import AnalyticBacktester, BatchBacktester  # noqa: E402
from _gpu import requires_core_gpu  # noqa: E402
from kimsfinance.optimization.genetic import GeneticOptimizer  # noqa: E402


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

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


@pytest.fixture
def rsi_param_space():
    """Standard RSI parameter space."""
    return {
        "period": (10, 20, int),
        "buy_threshold": (25, 35, float),
        "sell_threshold": (65, 75, float),
    }


def _assert_in_bounds(params):
    assert 10 <= params["period"] <= 20
    assert 25 <= params["buy_threshold"] <= 35
    assert 65 <= params["sell_threshold"] <= 75


@requires_core_gpu
class TestGeneticGPUIntegration:
    """GeneticOptimizer + GPU batch backtester."""

    def test_gpu_batch_evaluation_basic(self, sample_ohlcv, rsi_param_space):
        """Optimizer runs end-to-end with GPU fitness evaluation."""
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space, population_size=20, generations=5, objectives=["sharpe"]
        )
        backtester = BatchBacktester()

        results = optimizer.optimize(
            strategy="rsi_crossover", data=sample_ohlcv, backtester=backtester, verbose=False
        )

        assert len(results) > 0, "Should return at least one solution"
        assert "params" in results[0], "Solution should have params"
        assert "sharpe" in results[0], "Solution should have sharpe ratio"
        assert backtester.call_count >= 20, "Every initial individual must be evaluated on GPU"
        _assert_in_bounds(results[0]["params"])

    def test_optimizer_matches_direct_batch_backtest(self, sample_ohlcv):
        """A degenerate 1-point search must reproduce the direct batch_backtest metrics."""
        from kimsfinance.batch import batch_backtest

        params = {"period": 14, "buy_threshold": 30.0, "sell_threshold": 70.0}
        direct = batch_backtest("rsi_crossover", sample_ohlcv, [params])[0]

        optimizer = GeneticOptimizer(
            param_space={
                "period": (14, 14, int),
                "buy_threshold": (30.0, 30.0, float),
                "sell_threshold": (70.0, 70.0, float),
            },
            population_size=2,
            generations=1,
            objectives=["sharpe", "max_drawdown", "win_rate"],
        )
        results = optimizer.optimize(
            strategy="rsi_crossover", data=sample_ohlcv, backtester=BatchBacktester(), verbose=False
        )

        assert results, "Optimizer should return the single reachable solution"
        best = results[0]
        assert best["params"] == params
        # Same kernel, same inputs; the loose tolerance only guards against
        # fast-math run-to-run noise in non-strict-fp builds.
        assert best["sharpe"] == pytest.approx(direct["sharpe_ratio"], rel=1e-4, abs=1e-6)
        assert best["max_drawdown"] == pytest.approx(
            abs(direct["max_drawdown"]), rel=1e-4, abs=1e-6
        )
        assert best["win_rate"] == pytest.approx(direct["win_rate"], rel=1e-4, abs=1e-6)

    def test_gpu_multi_objective_optimization(self, sample_ohlcv, rsi_param_space):
        """Multi-objective (NSGA-II) optimization on GPU returns a Pareto front."""
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=30,
            generations=10,
            objectives=["sharpe", "max_drawdown", "win_rate"],
        )

        results = optimizer.optimize(
            strategy="rsi_crossover", data=sample_ohlcv, backtester=BatchBacktester(), verbose=False
        )

        assert len(results) >= 3, "Should find at least 3 Pareto-optimal solutions"

        for sol in results:
            assert "sharpe" in sol
            assert "max_drawdown" in sol
            assert "win_rate" in sol

        best = results[0]
        assert best["sharpe"] != float("-inf"), "GPU evaluation must not fail (worst fitness)"
        assert 0.0 <= best["max_drawdown"] <= 1.0, "Drawdown is reported as a positive fraction"
        assert 0.0 <= best["win_rate"] <= 1.0, "Win rate should be [0, 1]"

    def test_large_population_gpu(self, sample_ohlcv, rsi_param_space):
        """Large population (stress test) completes in bounded time."""
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=200,
            generations=5,
            objectives=["sharpe"],
        )

        start = time.perf_counter()
        results = optimizer.optimize(
            strategy="rsi_crossover", data=sample_ohlcv, backtester=BatchBacktester(), verbose=False
        )
        elapsed = time.perf_counter() - start

        assert len(results) > 0
        print(f"\nLarge population (200) completed in {elapsed:.2f}s")
        assert elapsed < 60.0, f"GPU optimization too slow: {elapsed:.2f}s"

    def test_small_population(self, sample_ohlcv, rsi_param_space):
        """Tiny population still yields a solution."""
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space, population_size=5, generations=3, objectives=["sharpe"]
        )
        results = optimizer.optimize(
            strategy="rsi_crossover", data=sample_ohlcv, backtester=BatchBacktester(), verbose=False
        )
        assert len(results) > 0

    def test_single_generation(self, sample_ohlcv, rsi_param_space):
        """Single generation still yields a solution."""
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space, population_size=20, generations=1, objectives=["sharpe"]
        )
        results = optimizer.optimize(
            strategy="rsi_crossover", data=sample_ohlcv, backtester=BatchBacktester(), verbose=False
        )
        assert len(results) > 0


class TestCPUFallback:
    """The optimizer works without a GPU: any ``run()``-style backtester will do."""

    def test_cpu_backtester_no_gpu(self, sample_ohlcv, rsi_param_space):
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space, population_size=10, generations=3, objectives=["sharpe"]
        )
        backtester = AnalyticBacktester()

        results = optimizer.optimize(
            strategy="rsi_crossover", data=sample_ohlcv, backtester=backtester, verbose=False
        )

        assert len(results) > 0
        assert "params" in results[0]
        assert "sharpe" in results[0]
        assert backtester.call_count >= 10
        _assert_in_bounds(results[0]["params"])

    def test_island_model_cpu(self, sample_ohlcv, rsi_param_space):
        """Island model (fork-based multiprocessing) with a CPU backtester.

        Kept off the GPU on purpose: a CUDA context cannot be reused in forked
        children, so the island model is exercised with the CPU backtester.
        """
        optimizer = GeneticOptimizer(
            param_space=rsi_param_space,
            population_size=20,
            generations=5,
            n_islands=4,
            objectives=["sharpe"],
        )

        results = optimizer.optimize(
            strategy="rsi_crossover",
            data=sample_ohlcv,
            backtester=AnalyticBacktester(),
            verbose=False,
        )

        assert len(results) > 0
        _assert_in_bounds(results[0]["params"])
