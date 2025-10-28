"""
Optimization module for kimsfinance.

This module provides production-grade optimization algorithms for trading strategies,
including genetic algorithms, particle swarm optimization, and Bayesian optimization.

Available optimizers:
- GeneticOptimizer: Multi-objective genetic algorithm (NSGA-II) with island model
- optimize_single_objective: Convenience function for single-objective optimization

Example:
    ```python
    from kimsfinance.optimization import GeneticOptimizer

    param_space = {
        'rsi_period': (5, 30, int),
        'buy_threshold': (20, 40, float),
        'sell_threshold': (60, 80, float),
    }

    optimizer = GeneticOptimizer(
        param_space=param_space,
        population_size=100,
        generations=50,
        objectives=['sharpe', 'max_drawdown', 'win_rate'],
    )

    best_solutions = optimizer.optimize(
        strategy='rsi_crossover',
        data=ohlcv_data,
        backtester=BacktestEngine()
    )
    ```
"""

from .genetic import GeneticOptimizer, optimize_single_objective

__all__ = [
    "GeneticOptimizer",
    "optimize_single_objective",
]
