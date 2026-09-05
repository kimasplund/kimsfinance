"""Backtester adapters for driving ``GeneticOptimizer`` in tests.

``GeneticOptimizer.optimize()`` takes a ``backtester`` object exposing
``run(strategy=..., data=..., params=...) -> dict`` and has no GPU switch of its
own; whether fitness evaluation runs on the GPU is decided entirely by the
backtester it is given. These two adapters cover both cases:

* ``BatchBacktester``    - real GPU evaluation through ``kimsfinance.batch.batch_backtest``
                           (one parameter set per call). Needs a usable CUDA device
                           (gate with ``_gpu.requires_core_gpu``).
* ``AnalyticBacktester`` - deterministic closed-form CPU stand-in with a known
                           optimum (period=14, buy=30, sell=70). Needs nothing.

Both are plain module-level classes so they pickle for the island model
(``multiprocessing.Pool``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class BatchBacktester:
    """Adapter: one ``batch_backtest`` call per individual, returning its result dict."""

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config
        self.call_count = 0

    def run(self, strategy: str, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from kimsfinance.batch import batch_backtest

        self.call_count += 1
        return batch_backtest(strategy, data, [params], config=self.config)[0]


class AnalyticBacktester:
    """Deterministic CPU stand-in: fitness peaks at period=14, buy=30, sell=70."""

    def __init__(self) -> None:
        self.call_count = 0

    def run(self, strategy: str, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        self.call_count += 1
        period = float(params.get("period", params.get("rsi_period", 14)))
        buy = float(params.get("buy_threshold", 30.0))
        sell = float(params.get("sell_threshold", 70.0))

        score = (
            (1.0 - abs(period - 14) / 20.0)
            + (1.0 - abs(buy - 30) / 30.0)
            + (1.0 - abs(sell - 70) / 30.0)
        ) / 3.0

        return {
            "sharpe_ratio": max(0.0, 2.0 * score),
            "max_drawdown": -abs(0.2 - 0.15 * score),
            "win_rate": min(1.0, 0.5 + 0.3 * score),
            "total_return": max(0.0, 0.5 * score),
            "profit_factor": max(1.0, 1.5 * score),
        }


class FailingBacktester:
    """Every ``run`` raises - exercises the optimizer's worst-fitness fallback."""

    def run(self, strategy: str, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("simulated backtester failure")


__all__ = ["BatchBacktester", "AnalyticBacktester", "FailingBacktester"]
