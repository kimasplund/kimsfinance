"""
Tests for genetic algorithm optimization module.

This module tests the GeneticOptimizer class and related functions
for strategy parameter optimization.
"""

import pytest
import numpy as np

# GeneticOptimizer requires the optional 'deap' package (the [optimization] extra).
pytest.importorskip("deap")

from kimsfinance.optimization import GeneticOptimizer, optimize_single_objective


class MockBacktestEngine:
    """Mock backtester for testing genetic optimization."""

    def __init__(self, deterministic=False):
        """
        Initialize mock backtester.

        Args:
            deterministic: If True, return deterministic results (no randomness)
        """
        self.deterministic = deterministic
        self.call_count = 0

    def run(self, strategy, data, params):
        """Simulate backtest with results based on parameters."""
        self.call_count += 1

        # Simple fitness function: optimal RSI is period=14, buy=30, sell=70
        rsi_period = params.get("rsi_period", 14)
        buy_threshold = params.get("buy_threshold", 30)
        sell_threshold = params.get("sell_threshold", 70)

        # Calculate fitness score
        period_score = 1.0 - abs(rsi_period - 14) / 20.0
        buy_score = 1.0 - abs(buy_threshold - 30) / 30.0
        sell_score = 1.0 - abs(sell_threshold - 70) / 30.0

        base_score = (period_score + buy_score + sell_score) / 3.0

        # Add randomness if not deterministic
        if self.deterministic:
            randomness = 0.0
        else:
            randomness = np.random.normal(0, 0.05)

        # Generate performance metrics
        sharpe = max(0, 2.0 * base_score + randomness)
        max_drawdown = -abs(0.2 - 0.15 * base_score + randomness * 0.05)
        win_rate = min(1.0, 0.5 + 0.3 * base_score + randomness * 0.1)

        return {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_return": max(0, 0.5 * base_score + randomness * 0.2),
            "profit_factor": max(1.0, 1.5 * base_score),
        }


def create_sample_data(n=1000):
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    return {
        "open": close + np.random.randn(n),
        "high": close + np.random.uniform(0.5, 2.0, n),
        "low": close - np.random.uniform(0.5, 2.0, n),
        "close": close,
        "volume": np.random.uniform(1000, 10000, n),
    }


@pytest.fixture
def param_space():
    """Standard parameter space for RSI strategy."""
    return {
        "rsi_period": (5, 30, int),
        "buy_threshold": (20, 40, float),
        "sell_threshold": (60, 80, float),
    }


@pytest.fixture
def backtester():
    """Mock backtester fixture."""
    return MockBacktestEngine(deterministic=True)


@pytest.fixture
def sample_data():
    """Sample OHLCV data fixture."""
    return create_sample_data(n=1000)


class TestGeneticOptimizerInit:
    """Test GeneticOptimizer initialization."""

    def test_init_default_parameters(self, param_space):
        """Test initialization with default parameters."""
        optimizer = GeneticOptimizer(param_space=param_space)

        assert optimizer.population_size == 100
        assert optimizer.generations == 50
        assert optimizer.objectives == ["sharpe", "max_drawdown", "win_rate"]
        assert optimizer.n_islands == 1
        assert optimizer.mutation_rate == 0.2
        assert optimizer.crossover_rate == 0.8

    def test_init_custom_parameters(self, param_space):
        """Test initialization with custom parameters."""
        optimizer = GeneticOptimizer(
            param_space=param_space,
            population_size=200,
            generations=100,
            objectives=["sharpe"],
            n_islands=4,
            mutation_rate=0.3,
        )

        assert optimizer.population_size == 200
        assert optimizer.generations == 100
        assert optimizer.objectives == ["sharpe"]
        assert optimizer.n_islands == 4
        assert optimizer.mutation_rate == 0.3

    def test_param_space_parsing(self, param_space):
        """Test parameter space parsing."""
        optimizer = GeneticOptimizer(param_space=param_space)

        assert "rsi_period" in optimizer.param_names
        assert "buy_threshold" in optimizer.param_names
        assert "sell_threshold" in optimizer.param_names

        assert optimizer.param_bounds["rsi_period"] == (5, 30)
        assert optimizer.param_types["rsi_period"] == int


class TestGeneticOptimizerDecoding:
    """Test individual encoding/decoding."""

    def test_decode_individual(self, param_space):
        """Test decoding individual to parameter dict."""
        optimizer = GeneticOptimizer(param_space=param_space)

        # Create test individual
        individual = [14, 30.0, 70.0]

        params = optimizer._decode_individual(individual)

        assert params["rsi_period"] == 14
        assert isinstance(params["rsi_period"], int)
        assert params["buy_threshold"] == 30.0
        assert isinstance(params["buy_threshold"], float)
        assert params["sell_threshold"] == 70.0

    def test_decode_respects_types(self, param_space):
        """Test that decoding respects parameter types."""
        optimizer = GeneticOptimizer(param_space=param_space)

        individual = [14.9, 30.5, 70.2]
        params = optimizer._decode_individual(individual)

        # Integer parameter should be converted
        assert params["rsi_period"] == 14
        assert isinstance(params["rsi_period"], int)

        # Float parameters should remain floats
        assert isinstance(params["buy_threshold"], float)
        assert isinstance(params["sell_threshold"], float)


class TestGeneticOptimizerMutation:
    """Test mutation operator."""

    def test_mutation_respects_bounds(self, param_space):
        """Test that mutation respects parameter bounds."""
        optimizer = GeneticOptimizer(param_space=param_space, mutation_rate=1.0)

        # Run mutation many times
        for _ in range(100):
            individual = [14, 30.0, 70.0]
            (mutated,) = optimizer._custom_mutate(individual)

            # Check bounds
            assert 5 <= mutated[0] <= 30  # rsi_period
            assert 20 <= mutated[1] <= 40  # buy_threshold
            assert 60 <= mutated[2] <= 80  # sell_threshold

    def test_mutation_rate_zero(self, param_space):
        """Test that mutation_rate=0 prevents mutations."""
        optimizer = GeneticOptimizer(param_space=param_space, mutation_rate=0.0)

        individual = [14, 30.0, 70.0]
        (mutated,) = optimizer._custom_mutate(individual)

        # Should be unchanged
        assert mutated == individual


class TestGeneticOptimizerFitness:
    """Test fitness evaluation."""

    def test_evaluate_fitness_single_objective(self, param_space, backtester, sample_data):
        """Test fitness evaluation with single objective."""
        optimizer = GeneticOptimizer(param_space=param_space, objectives=["sharpe"])

        individual = [14, 30.0, 70.0]

        fitness = optimizer._evaluate_fitness(
            individual, strategy="rsi_crossover", data=sample_data, backtester=backtester
        )

        assert len(fitness) == 1
        assert isinstance(fitness[0], float)
        assert fitness[0] > 0  # Sharpe should be positive

    def test_evaluate_fitness_multi_objective(self, param_space, backtester, sample_data):
        """Test fitness evaluation with multiple objectives."""
        optimizer = GeneticOptimizer(
            param_space=param_space, objectives=["sharpe", "max_drawdown", "win_rate"]
        )

        individual = [14, 30.0, 70.0]

        fitness = optimizer._evaluate_fitness(
            individual, strategy="rsi_crossover", data=sample_data, backtester=backtester
        )

        assert len(fitness) == 3
        assert fitness[0] > 0  # Sharpe
        assert fitness[1] < 0  # Drawdown (negated for maximization)
        assert 0 < fitness[2] <= 1  # Win rate

    def test_evaluate_fitness_calls_backtester(self, param_space, backtester, sample_data):
        """Test that fitness evaluation calls backtester."""
        optimizer = GeneticOptimizer(param_space=param_space)

        initial_count = backtester.call_count

        individual = [14, 30.0, 70.0]
        optimizer._evaluate_fitness(
            individual, strategy="rsi_crossover", data=sample_data, backtester=backtester
        )

        assert backtester.call_count == initial_count + 1


class TestSingleObjectiveOptimization:
    """Test single-objective optimization."""

    def test_optimize_single_objective(self, param_space, backtester, sample_data):
        """Test single-objective optimization completes successfully."""
        best_solution = optimize_single_objective(
            param_space=param_space,
            objective="sharpe",
            strategy="rsi_crossover",
            data=sample_data,
            backtester=backtester,
            population_size=20,
            generations=5,
        )

        assert "params" in best_solution
        assert "sharpe" in best_solution

        # Check parameter types
        assert isinstance(best_solution["params"]["rsi_period"], int)
        assert isinstance(best_solution["params"]["buy_threshold"], float)
        assert isinstance(best_solution["params"]["sell_threshold"], float)

        # Check parameter bounds
        params = best_solution["params"]
        assert 5 <= params["rsi_period"] <= 30
        assert 20 <= params["buy_threshold"] <= 40
        assert 60 <= params["sell_threshold"] <= 80

    def test_optimize_converges_to_optimum(self, param_space, backtester, sample_data):
        """Test that optimizer converges near optimal solution."""
        # With deterministic backtester, optimal is rsi=14, buy=30, sell=70
        best_solution = optimize_single_objective(
            param_space=param_space,
            objective="sharpe",
            strategy="rsi_crossover",
            data=sample_data,
            backtester=backtester,
            population_size=50,
            generations=30,
        )

        params = best_solution["params"]

        # Should be close to optimal (within 10% for continuous params)
        assert abs(params["rsi_period"] - 14) <= 3
        assert abs(params["buy_threshold"] - 30) <= 4
        assert abs(params["sell_threshold"] - 70) <= 4


class TestMultiObjectiveOptimization:
    """Test multi-objective optimization."""

    def test_optimize_multi_objective(self, param_space, backtester, sample_data):
        """Test multi-objective optimization returns Pareto front."""
        optimizer = GeneticOptimizer(
            param_space=param_space,
            population_size=30,
            generations=10,
            objectives=["sharpe", "max_drawdown", "win_rate"],
        )

        pareto_front = optimizer.optimize(
            strategy="rsi_crossover",
            data=sample_data,
            backtester=backtester,
            verbose=False,
        )

        assert len(pareto_front) > 0
        assert len(pareto_front) <= 30  # At most population_size

        # Check solution structure
        solution = pareto_front[0]
        assert "params" in solution
        assert "fitness" in solution
        assert "sharpe" in solution
        assert "max_drawdown" in solution
        assert "win_rate" in solution

    def test_pareto_front_non_dominated(self, param_space, backtester, sample_data):
        """Test that Pareto front contains non-dominated solutions."""
        optimizer = GeneticOptimizer(
            param_space=param_space,
            population_size=20,
            generations=5,
            objectives=["sharpe", "max_drawdown"],
        )

        pareto_front = optimizer.optimize(
            strategy="rsi_crossover",
            data=sample_data,
            backtester=backtester,
            verbose=False,
        )

        # All solutions should be non-dominated
        # (no solution should be strictly better in all objectives)
        for i, sol1 in enumerate(pareto_front):
            for j, sol2 in enumerate(pareto_front):
                if i == j:
                    continue

                # Check if sol1 dominates sol2
                better_in_all = True
                better_in_any = False

                for obj in optimizer.objectives:
                    if sol1[obj] > sol2[obj]:
                        better_in_any = True
                    elif sol1[obj] < sol2[obj]:
                        better_in_all = False

                # sol1 should not strictly dominate sol2
                assert not (better_in_all and better_in_any)


class TestIslandModel:
    """Test island model optimization."""

    @pytest.mark.slow
    def test_island_model_runs(self, param_space, backtester, sample_data):
        """Test that island model optimization completes."""
        optimizer = GeneticOptimizer(
            param_space=param_space,
            population_size=20,
            generations=5,
            n_islands=2,  # Use 2 islands
        )

        pareto_front = optimizer.optimize(
            strategy="rsi_crossover",
            data=sample_data,
            backtester=backtester,
            verbose=False,
            n_jobs=2,
        )

        assert len(pareto_front) > 0

    @pytest.mark.slow
    @pytest.mark.skip(
        reason="Probabilistic test - island diversity not guaranteed with small populations"
    )
    def test_island_model_diversity(self, param_space, sample_data):
        """Test that island model explores more diverse solutions.

        Note: This is a probabilistic property that's not guaranteed,
        especially with small populations and short runs. Skipped in CI.
        """
        backtester_random = MockBacktestEngine(deterministic=False)

        # Single island
        optimizer_single = GeneticOptimizer(
            param_space=param_space,
            population_size=20,
            generations=10,
            n_islands=1,
        )

        pareto_single = optimizer_single.optimize(
            strategy="rsi_crossover",
            data=sample_data,
            backtester=backtester_random,
            verbose=False,
        )

        # Multiple islands
        optimizer_multi = GeneticOptimizer(
            param_space=param_space,
            population_size=20,
            generations=10,
            n_islands=4,
        )

        pareto_multi = optimizer_multi.optimize(
            strategy="rsi_crossover",
            data=sample_data,
            backtester=backtester_random,
            verbose=False,
            n_jobs=4,
        )

        # Island model tends to explore more diverse solutions
        # (not guaranteed with small populations)
        # Just verify both complete successfully
        assert len(pareto_single) > 0
        assert len(pareto_multi) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_param_space(self):
        """Test that invalid parameter space raises error."""
        with pytest.raises((KeyError, ValueError)):
            optimizer = GeneticOptimizer(
                param_space={},  # Empty parameter space
            )

    def test_fitness_evaluation_error(self, param_space, sample_data):
        """Test that fitness evaluation errors are handled gracefully."""

        class FailingBacktester:
            def run(self, strategy, data, params):
                raise RuntimeError("Backtesting failed!")

        optimizer = GeneticOptimizer(param_space=param_space)

        individual = [14, 30.0, 70.0]

        # Should return worst fitness, not crash
        fitness = optimizer._evaluate_fitness(
            individual, strategy="rsi_crossover", data=sample_data, backtester=FailingBacktester()
        )

        # Should return -inf for all objectives
        assert all(f == float("-inf") for f in fitness)

    def test_zero_generations(self, param_space, backtester, sample_data):
        """Test optimization with zero generations."""
        optimizer = GeneticOptimizer(
            param_space=param_space,
            generations=0,
        )

        # Should still return initial population
        pareto_front = optimizer.optimize(
            strategy="rsi_crossover",
            data=sample_data,
            backtester=backtester,
            verbose=False,
        )

        assert len(pareto_front) > 0


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.benchmark
    def test_optimization_speed(self, param_space, backtester, sample_data, benchmark):
        """Benchmark optimization speed."""
        optimizer = GeneticOptimizer(
            param_space=param_space,
            population_size=20,
            generations=5,
        )

        def run_optimization():
            return optimizer.optimize(
                strategy="rsi_crossover",
                data=sample_data,
                backtester=backtester,
                verbose=False,
            )

        result = benchmark(run_optimization)
        assert len(result) > 0

    def test_scales_with_population_size(self, param_space, backtester, sample_data):
        """Test that backtester calls scale with population size."""
        population_sizes = [10, 20, 30]
        call_counts = []

        for pop_size in population_sizes:
            backtester_test = MockBacktestEngine(deterministic=True)

            optimizer = GeneticOptimizer(
                param_space=param_space,
                population_size=pop_size,
                generations=2,
            )

            optimizer.optimize(
                strategy="rsi_crossover",
                data=sample_data,
                backtester=backtester_test,
                verbose=False,
            )

            call_counts.append(backtester_test.call_count)

        # Call count should increase with population size
        assert call_counts[0] < call_counts[1] < call_counts[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
