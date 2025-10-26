"""
Strategy library for kimsfinance backtesting

Pre-built strategies organized by category:
- momentum: RSI, ROC, Stochastic oscillator strategies
- trend: MACD, EMA crossover, trend following
- volatility: ATR, Bollinger Bands, volatility breakout
"""

from .momentum import (
    RSIStrategy,
    ROCStrategy,
    StochasticStrategy,
    WilliamsRStrategy,
    CCIStrategy,
)

from .trend import (
    MACDStrategy,
    EMACrossoverStrategy,
    TrendFollowingStrategy,
    DualMAStrategy,
)

from .volatility import (
    ATRBreakoutStrategy,
    BollingerBreakoutStrategy,
    KeltnerBreakoutStrategy,
    VolatilityContractionStrategy,
)

__all__ = [
    # Momentum
    "RSIStrategy",
    "ROCStrategy",
    "StochasticStrategy",
    "WilliamsRStrategy",
    "CCIStrategy",
    # Trend
    "MACDStrategy",
    "EMACrossoverStrategy",
    "TrendFollowingStrategy",
    "DualMAStrategy",
    # Volatility
    "ATRBreakoutStrategy",
    "BollingerBreakoutStrategy",
    "KeltnerBreakoutStrategy",
    "VolatilityContractionStrategy",
]
