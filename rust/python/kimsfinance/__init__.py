"""
kimsfinance - High-performance Python financial charting library with Rust acceleration

This package provides:
- Rust-accelerated backtesting engine
- Technical indicators (24+ indicators)
- Strategy library (momentum, trend, volatility)
- Visualization tools for backtest results
"""

__version__ = "0.2.0"

# Re-export core functionality
try:
    from kimsfinance_core import (
        calculate_sma,
        calculate_ema,
        calculate_rsi,
        calculate_atr,
        calculate_macd,
        calculate_bollinger_bands,
        run_backtest,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"kimsfinance_core not available: {e}. Run 'maturin develop' to build.", ImportWarning)

# Import strategy modules
from . import strategies
from . import visualization

__all__ = [
    "strategies",
    "visualization",
    "calculate_sma",
    "calculate_ema",
    "calculate_rsi",
    "calculate_atr",
    "calculate_macd",
    "calculate_bollinger_bands",
    "run_backtest",
]
