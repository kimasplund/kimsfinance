"""
Python integration tests for GPU batch backtesting.

Tests the PyO3 bindings for BatchBacktestSweep.
"""

import numpy as np
import pytest

# Import will fail if GPU feature not enabled
try:
    from kimsfinance_core import batch_backtest, batch_backtest_info, BacktestResult

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    pytest.skip("GPU feature not enabled", allow_module_level=True)


class TestBatchBacktestInfo:
    """Test batch_backtest_info() function."""

    def test_info_returns_dict(self):
        """Test that batch_backtest_info() returns expected keys."""
        info = batch_backtest_info()

        assert "gpu_available" in info
        assert "expected_speedup" in info

        if info["gpu_available"]:
            assert "gpu_name" in info
            assert "cuda_version" in info
            assert "vram_gb" in info
            assert info["expected_speedup"] > 1.0
        else:
            assert "error" in info
            assert info["expected_speedup"] == 1.0


class TestBatchBacktestBasic:
    """Basic batch backtest functionality tests."""

    @pytest.fixture
    def synthetic_ohlcv(self):
        """Generate synthetic OHLCV data for testing."""
        np.random.seed(42)
        n_candles = 1000

        # Generate random walk prices
        returns = np.random.randn(n_candles) * 0.01
        close = 100.0 * np.exp(np.cumsum(returns))

        # Generate OHLC from close
        noise = np.random.randn(n_candles, 4) * 0.005
        ohlcv = np.zeros((n_candles, 5))
        ohlcv[:, 3] = close  # close
        ohlcv[:, 1] = close * (1 + np.abs(noise[:, 1]))  # high
        ohlcv[:, 2] = close * (1 - np.abs(noise[:, 2]))  # low
        ohlcv[:, 0] = close * (1 + noise[:, 0])  # open
        ohlcv[:, 4] = np.abs(np.random.randn(n_candles)) * 1000  # volume

        # Ensure OHLC invariants
        ohlcv[:, 1] = np.maximum(ohlcv[:, 1], ohlcv[:, 0])  # high >= open
        ohlcv[:, 1] = np.maximum(ohlcv[:, 1], ohlcv[:, 3])  # high >= close
        ohlcv[:, 2] = np.minimum(ohlcv[:, 2], ohlcv[:, 0])  # low <= open
        ohlcv[:, 2] = np.minimum(ohlcv[:, 2], ohlcv[:, 3])  # low <= close

        return ohlcv

    @pytest.fixture
    def timestamps(self, synthetic_ohlcv):
        """Generate timestamps for OHLCV data."""
        n_candles = len(synthetic_ohlcv)
        # Start from arbitrary timestamp, 1-minute bars
        start = 1700000000 * 1_000_000_000  # nanoseconds
        return np.arange(start, start + n_candles * 60_000_000_000, 60_000_000_000, dtype=np.int64)

    def test_batch_backtest_10_strategies(self, synthetic_ohlcv, timestamps):
        """Test batch backtest with 10 RSI strategies."""
        # 10 RSI strategies with different thresholds
        parameters = [[14.0, 20.0 + i * 2, 70.0 + i] for i in range(10)]

        # Run batch backtest
        results = batch_backtest(
            strategy="rsi_crossover",
            ohlcv=synthetic_ohlcv,
            parameters=parameters,
            timestamps=timestamps,
            initial_capital=10000.0,
            trading_fee=0.001,
            slippage=0.0001,
        )

        # Validate results
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

        for i, result in enumerate(results):
            # Check types
            assert isinstance(result, BacktestResult), f"Result {i} is not BacktestResult"

            # Check attributes exist
            assert hasattr(result, "sharpe_ratio")
            assert hasattr(result, "max_drawdown")
            assert hasattr(result, "win_rate")
            assert hasattr(result, "total_return")
            assert hasattr(result, "final_equity")
            assert hasattr(result, "num_trades")
            assert hasattr(result, "profit_factor")

            # Check value ranges
            assert 0.0 <= result.win_rate <= 1.0, f"Invalid win_rate: {result.win_rate}"
            assert (
                result.max_drawdown <= 0.0
            ), f"Max drawdown should be negative: {result.max_drawdown}"
            assert (
                result.final_equity >= 0.0
            ), f"Final equity should be positive: {result.final_equity}"
            assert result.num_trades >= 0, f"Num trades should be non-negative: {result.num_trades}"

    def test_batch_backtest_without_timestamps(self, synthetic_ohlcv):
        """Test batch backtest without providing timestamps (should auto-generate)."""
        parameters = [[14.0, 30.0, 70.0]]  # Single strategy

        results = batch_backtest(
            strategy="rsi_crossover",
            ohlcv=synthetic_ohlcv,
            parameters=parameters,
            initial_capital=10000.0,
        )

        assert len(results) == 1
        assert isinstance(results[0], BacktestResult)

    def test_batch_backtest_ma_crossover(self, synthetic_ohlcv, timestamps):
        """Test MA crossover strategy."""
        # 5 MA crossover strategies
        parameters = [[10.0 + i * 5, 50.0 + i * 10] for i in range(5)]  # [fast_period, slow_period]

        results = batch_backtest(
            strategy="ma_crossover",
            ohlcv=synthetic_ohlcv,
            parameters=parameters,
            timestamps=timestamps,
        )

        assert len(results) == 5
        for result in results:
            assert isinstance(result, BacktestResult)

    def test_batch_backtest_bollinger(self, synthetic_ohlcv, timestamps):
        """Test Bollinger Bands strategy."""
        # 3 Bollinger strategies
        parameters = [
            [20.0, 2.0, 1.5, 0.5],  # [period, std_dev, entry_std, exit_std]
            [15.0, 2.5, 2.0, 0.5],
            [25.0, 1.5, 1.0, 0.5],
        ]

        results = batch_backtest(
            strategy="bollinger",
            ohlcv=synthetic_ohlcv,
            parameters=parameters,
            timestamps=timestamps,
        )

        assert len(results) == 3
        for result in results:
            assert isinstance(result, BacktestResult)


class TestBatchBacktestStress:
    """Stress tests for batch backtesting."""

    @pytest.fixture
    def large_ohlcv(self):
        """Generate large OHLCV dataset (10K candles)."""
        np.random.seed(123)
        n_candles = 10000

        returns = np.random.randn(n_candles) * 0.01
        close = 100.0 * np.exp(np.cumsum(returns))

        noise = np.random.randn(n_candles, 4) * 0.005
        ohlcv = np.zeros((n_candles, 5))
        ohlcv[:, 3] = close
        ohlcv[:, 1] = close * (1 + np.abs(noise[:, 1]))
        ohlcv[:, 2] = close * (1 - np.abs(noise[:, 2]))
        ohlcv[:, 0] = close * (1 + noise[:, 0])
        ohlcv[:, 4] = np.abs(np.random.randn(n_candles)) * 1000

        # Fix OHLC invariants
        ohlcv[:, 1] = np.maximum.reduce([ohlcv[:, 1], ohlcv[:, 0], ohlcv[:, 3]])
        ohlcv[:, 2] = np.minimum.reduce([ohlcv[:, 2], ohlcv[:, 0], ohlcv[:, 3]])

        return ohlcv

    @pytest.mark.slow
    def test_100_strategies(self, large_ohlcv):
        """Test 100 strategies in parallel."""
        # 100 RSI strategies
        parameters = [[14.0, 20.0 + (i % 10) * 2, 70.0 + (i // 10)] for i in range(100)]

        results = batch_backtest(strategy="rsi_crossover", ohlcv=large_ohlcv, parameters=parameters)

        assert len(results) == 100

        # Check results are sorted by fitness (best first)
        fitness_scores = [r.fitness() for r in results]
        assert fitness_scores == sorted(
            fitness_scores, reverse=True
        ), "Results should be sorted by fitness (descending)"

    @pytest.mark.slow
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires GPU with >1GB VRAM")
    def test_1000_strategies(self, large_ohlcv):
        """Stress test with 1000 strategies (VRAM test)."""
        # 1000 strategies: 10 periods × 10 buy thresholds × 10 sell thresholds
        parameters = [
            [10.0 + p, 20.0 + b, 70.0 + s] for p in range(10) for b in range(10) for s in range(10)
        ]

        results = batch_backtest(strategy="rsi_crossover", ohlcv=large_ohlcv, parameters=parameters)

        assert len(results) == 1000

        # Verify all results are valid
        for result in results:
            assert result.final_equity >= 0.0
            assert 0.0 <= result.win_rate <= 1.0


class TestBatchBacktestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_strategy_name(self):
        """Test error for invalid strategy name."""
        ohlcv = np.random.randn(100, 5)
        parameters = [[14.0, 30.0, 70.0]]

        with pytest.raises(ValueError, match="Unknown strategy"):
            batch_backtest(strategy="invalid_strategy", ohlcv=ohlcv, parameters=parameters)

    def test_empty_parameters(self):
        """Test error for empty parameter list."""
        ohlcv = np.random.randn(100, 5)

        with pytest.raises(ValueError, match="parameters cannot be empty"):
            batch_backtest(strategy="rsi_crossover", ohlcv=ohlcv, parameters=[])

    def test_wrong_ohlcv_shape(self):
        """Test error for wrong OHLCV shape."""
        # Wrong: only 4 columns instead of 5
        ohlcv = np.random.randn(100, 4)
        parameters = [[14.0, 30.0, 70.0]]

        with pytest.raises(ValueError, match="ohlcv must have shape"):
            batch_backtest(strategy="rsi_crossover", ohlcv=ohlcv, parameters=parameters)

    def test_timestamp_length_mismatch(self):
        """Test error for timestamp length mismatch."""
        ohlcv = np.random.randn(100, 5)
        timestamps = np.arange(50, dtype=np.int64)  # Wrong length
        parameters = [[14.0, 30.0, 70.0]]

        with pytest.raises(ValueError, match="timestamps length.*must match"):
            batch_backtest(
                strategy="rsi_crossover", ohlcv=ohlcv, parameters=parameters, timestamps=timestamps
            )


class TestBacktestResultClass:
    """Test BacktestResult Python class methods."""

    @pytest.fixture
    def sample_result(self):
        """Create sample result by running minimal backtest."""
        ohlcv = np.random.randn(100, 5).cumsum(axis=0) + 100
        ohlcv[:, 0:4] = np.abs(ohlcv[:, 0:4])
        ohlcv[:, 4] = np.abs(ohlcv[:, 4]) * 1000

        results = batch_backtest(
            strategy="rsi_crossover", ohlcv=ohlcv, parameters=[[14.0, 30.0, 70.0]]
        )

        return results[0]

    def test_repr(self, sample_result):
        """Test __repr__ method."""
        repr_str = repr(sample_result)
        assert "BacktestResult" in repr_str
        assert "sharpe=" in repr_str
        assert "dd=" in repr_str

    def test_to_dict(self, sample_result):
        """Test to_dict() method."""
        d = sample_result.to_dict()

        assert isinstance(d, dict)
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d
        assert "win_rate" in d
        assert "params" in d
        assert isinstance(d["params"], dict)

    def test_fitness(self, sample_result):
        """Test fitness() method."""
        fitness = sample_result.fitness()

        assert isinstance(fitness, float)
        # Fitness should be Sharpe * (1 - abs(drawdown))
        expected = sample_result.sharpe_ratio * (1.0 - abs(sample_result.max_drawdown))
        assert abs(fitness - expected) < 1e-6


class TestPerformance:
    """Performance validation tests."""

    @pytest.fixture
    def perf_ohlcv(self):
        """Generate 10K candles for performance test."""
        np.random.seed(999)
        n = 10000
        returns = np.random.randn(n) * 0.01
        close = 100.0 * np.exp(np.cumsum(returns))

        ohlcv = np.zeros((n, 5))
        ohlcv[:, 3] = close
        ohlcv[:, 1] = close * 1.01  # high
        ohlcv[:, 2] = close * 0.99  # low
        ohlcv[:, 0] = close * (1 + np.random.randn(n) * 0.005)  # open
        ohlcv[:, 4] = np.abs(np.random.randn(n)) * 1000  # volume

        return ohlcv

    @pytest.mark.benchmark
    def test_100_strategies_10k_candles_performance(self, perf_ohlcv, benchmark):
        """Benchmark 100 strategies × 10K candles (target: <300ms)."""
        parameters = [[14.0, 20.0 + i, 70.0 + i * 0.5] for i in range(100)]

        def run_batch():
            return batch_backtest(strategy="rsi_crossover", ohlcv=perf_ohlcv, parameters=parameters)

        # Warmup
        run_batch()

        # Benchmark
        import time

        start = time.perf_counter()
        results = run_batch()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(results) == 100

        # Performance target: <300ms on RTX 3500 Ada
        # Allow 2x margin for slower GPUs or high load
        assert elapsed_ms < 600, f"Too slow: {elapsed_ms:.1f}ms (target: <300ms)"

        print(f"\n100 strategies × 10K candles: {elapsed_ms:.1f}ms")
        if elapsed_ms < 300:
            print("✓ Performance target met!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
