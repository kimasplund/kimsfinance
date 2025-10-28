"""High-level Python API for GPU batch backtesting.

This module provides a convenient Python interface for GPU-accelerated batch
backtesting, enabling 20-40x speedup over sequential CPU execution.

Example:
    >>> import pandas as pd
    >>> from kimsfinance.batch import batch_backtest, BacktestConfig
    >>>
    >>> # Load OHLCV data
    >>> data = pd.read_csv('BTC-USD.csv')
    >>>
    >>> # Define parameter sweep (100 strategies)
    >>> params = [
    ...     {'period': p, 'buy_threshold': b, 'sell_threshold': s}
    ...     for p in range(10, 20)
    ...     for b in [25, 30, 35]
    ...     for s in [65, 70, 75]
    ... ]
    >>>
    >>> # Run batch backtest on GPU
    >>> results = batch_backtest('rsi_crossover', data, params)
    >>>
    >>> # Find best strategy
    >>> best = max(results, key=lambda r: r['sharpe_ratio'])
    >>> print(f"Best Sharpe: {best['sharpe_ratio']:.2f}")
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np

try:
    from kimsfinance_core import (
        batch_backtest as _batch_backtest_rs,
        batch_backtest_info as _batch_backtest_info_rs,
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class BacktestConfig:
    """Configuration for batch backtesting.

    Attributes:
        initial_capital: Starting portfolio value (default: 10000.0)
        trading_fee: Fee per trade as fraction (default: 0.001 = 0.1%)
        slippage: Slippage per trade as fraction (default: 0.0001 = 0.01%)
    """
    initial_capital: float = 10000.0
    trading_fee: float = 0.001
    slippage: float = 0.0001


def get_gpu_info() -> Dict[str, Union[bool, str, float]]:
    """Get GPU availability and performance information.

    Returns:
        Dictionary with keys:
            - gpu_available: bool
            - gpu_name: str (if available)
            - cuda_version: str (if available)
            - vram_gb: int (if available)
            - expected_speedup: float (30.0 for GPU, 1.0 for CPU)
            - error: str (if GPU unavailable)

    Example:
        >>> info = get_gpu_info()
        >>> if info['gpu_available']:
        ...     print(f"GPU: {info['gpu_name']}")
        ...     print(f"Expected speedup: {info['expected_speedup']:.0f}x")
    """
    if not GPU_AVAILABLE:
        return {
            'gpu_available': False,
            'error': 'GPU feature not compiled. Install with: pip install kimsfinance[gpu]',
            'expected_speedup': 1.0,
        }

    return dict(_batch_backtest_info_rs())


def batch_backtest(
    strategy: str,
    data: pd.DataFrame,
    parameters: List[Dict[str, float]],
    config: Optional[BacktestConfig] = None,
    timestamps_col: str = 'timestamp',
) -> List[Dict]:
    """Run batch backtest on GPU for multiple parameter sets.

    Executes N strategies in parallel on GPU with a single data transfer.
    Delivers 20-40x speedup vs sequential CPU execution.

    Args:
        strategy: Strategy name. Options:
            - 'rsi_crossover': RSI crossover strategy
            - 'ma_crossover': Moving average crossover
            - 'bollinger': Bollinger Bands mean reversion

        data: DataFrame with OHLCV columns:
            - 'open', 'high', 'low', 'close', 'volume'
            - Optional: timestamps column (default: 'timestamp')

        parameters: List of parameter dicts. Format depends on strategy:
            - rsi_crossover: {'period': 14, 'buy_threshold': 30, 'sell_threshold': 70}
            - ma_crossover: {'fast_period': 10, 'slow_period': 50}
            - bollinger: {'period': 20, 'std_dev': 2.0, 'entry_std': 1.5, 'exit_std': 0.5}

        config: Backtest configuration (optional, uses defaults if None)

        timestamps_col: Column name for timestamps (default: 'timestamp').
            If column not found, auto-generated as [0, 1, 2, ...].

    Returns:
        List of result dicts, sorted by fitness (best first). Each dict contains:
            - sharpe_ratio: Sharpe ratio (annualized)
            - max_drawdown: Maximum drawdown (negative percentage)
            - win_rate: Win rate [0, 1]
            - total_return: Total return (percentage)
            - final_equity: Final portfolio value
            - num_trades: Number of trades executed
            - profit_factor: Gross profit / gross loss
            - params: Original parameter dict

    Raises:
        ValueError: Invalid strategy name, missing OHLCV columns, or empty parameters
        RuntimeError: GPU initialization failed or CUDA error
        ImportError: GPU feature not compiled

    Performance:
        - 1000 strategies × 10K candles: <250ms (RTX 3500 Ada)
        - Speedup: 20-40x vs sequential CPU
        - VRAM usage: <1GB for 1000 strategies

    Example:
        >>> import pandas as pd
        >>> from kimsfinance.batch import batch_backtest
        >>>
        >>> # Load data
        >>> data = pd.read_csv('BTC-USD.csv')
        >>>
        >>> # Define 90 parameter sets
        >>> params = [
        ...     {'period': p, 'buy_threshold': b, 'sell_threshold': s}
        ...     for p in range(10, 20)
        ...     for b in [25, 30, 35]
        ...     for s in [65, 70, 75]
        ... ]
        >>>
        >>> # Run batch backtest (all 90 at once on GPU!)
        >>> results = batch_backtest('rsi_crossover', data, params)
        >>>
        >>> # Find best Sharpe
        >>> best = max(results, key=lambda r: r['sharpe_ratio'])
        >>> print(f"Best Sharpe: {best['sharpe_ratio']:.2f}")
        >>> print(f"Parameters: {best['params']}")
    """
    if not GPU_AVAILABLE:
        raise ImportError(
            "GPU batch backtesting not available. "
            "Install with: pip install kimsfinance[gpu]"
        )

    # Validate inputs
    if not parameters:
        raise ValueError("parameters cannot be empty")

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"DataFrame must contain: {required_cols}"
        )

    # Use default config if not provided
    if config is None:
        config = BacktestConfig()

    # Convert DataFrame to NumPy array (N_candles, 5)
    ohlcv = data[required_cols].to_numpy()

    # Extract or generate timestamps
    if timestamps_col in data.columns:
        timestamps = data[timestamps_col].to_numpy(dtype=np.int64)
    else:
        timestamps = None  # Will be auto-generated in Rust

    # Convert parameter dicts to list of lists
    # (Order must match strategy parameter order!)
    param_lists = []
    for param_dict in parameters:
        if strategy == 'rsi_crossover':
            param_list = [
                param_dict.get('period', 14.0),
                param_dict.get('buy_threshold', 30.0),
                param_dict.get('sell_threshold', 70.0),
            ]
        elif strategy == 'ma_crossover':
            param_list = [
                param_dict.get('fast_period', 10.0),
                param_dict.get('slow_period', 50.0),
            ]
        elif strategy == 'bollinger':
            param_list = [
                param_dict.get('period', 20.0),
                param_dict.get('std_dev', 2.0),
                param_dict.get('entry_std', 1.5),
                param_dict.get('exit_std', 0.5),
            ]
        else:
            raise ValueError(
                f"Unknown strategy: '{strategy}'. "
                f"Valid options: 'rsi_crossover', 'ma_crossover', 'bollinger'"
            )

        param_lists.append(param_list)

    # Call Rust function
    rust_results = _batch_backtest_rs(
        strategy=strategy,
        ohlcv=ohlcv,
        parameters=param_lists,
        timestamps=timestamps,
        initial_capital=config.initial_capital,
        trading_fee=config.trading_fee,
        slippage=config.slippage,
    )

    # Convert to Python dicts
    results = []
    for i, r in enumerate(rust_results):
        result_dict = r.to_dict()
        result_dict['params'] = parameters[i]  # Original dict
        results.append(result_dict)

    return results


def find_best_parameters(
    strategy: str,
    data: pd.DataFrame,
    parameter_ranges: Dict[str, List[float]],
    config: Optional[BacktestConfig] = None,
    objective: str = 'sharpe_ratio',
) -> Dict:
    """Find best parameters via exhaustive grid search.

    Convenience function that generates all parameter combinations and runs
    batch backtest to find the optimal parameters.

    Args:
        strategy: Strategy name ('rsi_crossover', 'ma_crossover', 'bollinger')
        data: OHLCV DataFrame
        parameter_ranges: Dict of parameter name -> list of values to try
        config: Backtest configuration (optional)
        objective: Metric to optimize (default: 'sharpe_ratio')

    Returns:
        Dict with keys:
            - best_params: Best parameter dict
            - best_score: Best objective value
            - all_results: All results (sorted by objective)

    Example:
        >>> from kimsfinance.batch import find_best_parameters
        >>>
        >>> # Define parameter ranges
        >>> ranges = {
        ...     'period': [10, 14, 20],
        ...     'buy_threshold': [25, 30, 35],
        ...     'sell_threshold': [65, 70, 75],
        ... }
        >>>
        >>> # Find best parameters
        >>> result = find_best_parameters('rsi_crossover', data, ranges)
        >>>
        >>> print(f"Best parameters: {result['best_params']}")
        >>> print(f"Best Sharpe: {result['best_score']:.2f}")
    """
    # Generate all parameter combinations
    import itertools

    param_names = list(parameter_ranges.keys())
    param_values = [parameter_ranges[name] for name in param_names]

    all_combinations = itertools.product(*param_values)
    parameters = [
        dict(zip(param_names, combo))
        for combo in all_combinations
    ]

    # Run batch backtest
    results = batch_backtest(strategy, data, parameters, config)

    # Find best by objective
    best_result = max(results, key=lambda r: r[objective])

    return {
        'best_params': best_result['params'],
        'best_score': best_result[objective],
        'all_results': results,
    }


__all__ = [
    'batch_backtest',
    'get_gpu_info',
    'find_best_parameters',
    'BacktestConfig',
    'GPU_AVAILABLE',
]
