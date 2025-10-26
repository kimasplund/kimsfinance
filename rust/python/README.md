# kimsfinance Python Library

High-performance Python library for algorithmic trading with Rust-accelerated backtesting engine.

## Features

- **Rust-Accelerated Backtesting**: 10-50x faster than pure Python implementations
- **24+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and more
- **Pre-Built Strategy Library**: Momentum, trend, and volatility strategies
- **GPU Support**: Optional CUDA acceleration for large datasets
- **Visualization Tools**: Equity curves, drawdowns, performance dashboards
- **Jupyter Integration**: Example notebooks for learning and experimentation

## Installation

### Build from Source

```bash
# From rust/ directory
maturin develop --release

# Or with GPU support (requires CUDA)
maturin develop --release --features gpu
```

### Install Optional Dependencies

```bash
# For visualization
pip install kimsfinance_core[visualization]

# For Jupyter notebooks
pip install kimsfinance_core[notebooks]

# For development
pip install kimsfinance_core[dev]
```

## Quick Start

### Simple RSI Backtest

```python
import numpy as np
import kimsfinance_core
from kimsfinance.strategies import RSIStrategy
from kimsfinance.visualization import print_performance_summary

# Load your OHLCV data
high = np.array([...])
low = np.array([...])
close = np.array([...])
open_prices = np.array([...])
volume = np.array([...])
timestamps = np.arange(len(close), dtype=np.int64) * 60

# Create strategy
strategy = RSIStrategy(period=14, buy_threshold=30, sell_threshold=70)

# Run backtest
result = kimsfinance_core.run_backtest(
    high=high,
    low=low,
    close=close,
    open_prices=open_prices,
    volume=volume,
    timestamps=timestamps,
    strategy=strategy,
    initial_capital=10000.0,
    trading_fee=0.001,  # 0.1% per trade
    slippage=0.0005,    # 0.05% slippage
    use_gpu=False       # Set True if GPU available
)

# Print results
print_performance_summary(result)
```

### Visualize Results

```python
from kimsfinance.visualization import plot_equity_curve, plot_drawdown, plot_performance_dashboard

# Plot equity curve
plot_equity_curve(result)

# Plot drawdown
plot_drawdown(result)

# Complete dashboard
plot_performance_dashboard(result)
```

## Strategy Library

### Momentum Strategies

```python
from kimsfinance.strategies import (
    RSIStrategy,          # RSI mean reversion
    ROCStrategy,          # Rate of change momentum
    StochasticStrategy,   # Stochastic oscillator
    WilliamsRStrategy,    # Williams %R
    CCIStrategy,          # Commodity Channel Index
)
```

### Trend Strategies

```python
from kimsfinance.strategies import (
    MACDStrategy,            # MACD crossover
    EMACrossoverStrategy,    # EMA golden/death cross
    DualMAStrategy,          # Dual moving average
    TrendFollowingStrategy,  # Multi-timeframe trend
)
```

### Volatility Strategies

```python
from kimsfinance.strategies import (
    ATRBreakoutStrategy,           # ATR-based breakouts
    BollingerBreakoutStrategy,     # Bollinger Bands breakout
    KeltnerBreakoutStrategy,       # Keltner Channels breakout
    VolatilityContractionStrategy, # Bollinger squeeze
)
```

## Creating Custom Strategies

```python
class MyCustomStrategy:
    def __init__(self, period=14):
        self.period = period

    def on_data(self, bar, indicators):
        """
        Trading logic called for each bar

        Args:
            bar: Dict with OHLCV data
            indicators: Dict with pre-calculated indicators

        Returns:
            str: Signal ('buy', 'sell', 'hold', 'short', 'cover')
        """
        rsi = indicators.get(f'rsi_{self.period}', 50.0)

        if rsi < 30:
            return 'buy'
        elif rsi > 70:
            return 'sell'
        return 'hold'

    def get_indicators(self):
        """
        Indicators required by this strategy

        Returns:
            list: List of indicator strings (e.g., 'rsi_14')
        """
        return [f'rsi_{self.period}']

    def position_size(self, equity, signal):
        """
        Position sizing logic (optional)

        Args:
            equity: Current account equity
            signal: Current signal

        Returns:
            float: Position size (1.0 = 100% of equity)
        """
        return 1.0  # Full allocation

# Use your strategy
strategy = MyCustomStrategy(period=14)
result = kimsfinance_core.run_backtest(...)
```

## Jupyter Notebooks

The `notebooks/` directory contains example notebooks:

1. **01_basic_backtesting.ipynb** - Introduction to backtesting with kimsfinance
2. **02_parameter_optimization.ipynb** - Grid search parameter optimization
3. **03_genetic_optimization.ipynb** - Genetic algorithm optimization
4. **04_multi_indicator_strategies.ipynb** - Combining multiple indicators

Run notebooks:

```bash
cd notebooks/
jupyter notebook
```

## Technical Indicators

All indicators are Rust-accelerated (5-10x faster than pandas):

### Moving Averages
- SMA, EMA, WMA, VWMA, DEMA, TEMA, HMA

### Momentum
- RSI, ROC, Williams %R, Stochastic, Aroon, CCI, MACD, TSI

### Volatility
- ATR, Bollinger Bands, Keltner Channels, Donchian Channels, Elder Ray

### Volume
- OBV, VWAP, CMF, Volume Profile

### Direct Indicator Usage

```python
import kimsfinance_core
import numpy as np

close = np.array([100.0, 102.0, 101.5, 103.0, 104.5])

# Single indicators
rsi = kimsfinance_core.calculate_rsi(close, period=14)
sma = kimsfinance_core.calculate_sma(close, period=20)
atr = kimsfinance_core.calculate_atr(high, low, close, period=14)

# Multi-output indicators
macd = kimsfinance_core.calculate_macd(close, fast_period=12, slow_period=26, signal_period=9)
print(f"MACD line: {macd['macd']}")
print(f"Signal line: {macd['signal']}")
print(f"Histogram: {macd['histogram']}")

bb = kimsfinance_core.calculate_bollinger_bands(close, period=20, std_dev=2.0)
print(f"Upper band: {bb['upper']}")
print(f"Middle band: {bb['middle']}")
print(f"Lower band: {bb['lower']}")
```

## Performance

Backtest performance benchmarks (1000 candles, RSI strategy):

| Implementation | Time | Speedup |
|---------------|------|---------|
| Pure Python | ~800ms | 1x |
| kimsfinance CPU | ~50ms | 16x |
| kimsfinance GPU | ~5ms | 160x |

Technical indicator performance (10K rows):

| Indicator | pandas | kimsfinance | Speedup |
|-----------|--------|-------------|---------|
| SMA | 2.1ms | 0.4ms | 5.2x |
| RSI | 3.8ms | 0.6ms | 6.3x |
| ATR | 4.2ms | 0.7ms | 6.0x |
| MACD | 5.1ms | 0.9ms | 5.7x |

## GPU Acceleration

Enable GPU acceleration for large datasets:

```python
# Check GPU availability
import kimsfinance_core
print(f"GPU available: {hasattr(kimsfinance_core, 'calculate_stochastic_gpu')}")

# Run backtest with GPU
result = kimsfinance_core.run_backtest(
    ...,
    use_gpu=True  # Enable GPU acceleration
)
```

Requirements:
- CUDA-capable GPU
- Build with `--features gpu`

## Project Structure

```
python/
├── kimsfinance/
│   ├── __init__.py
│   ├── visualization.py       # Plotting tools
│   └── strategies/
│       ├── __init__.py
│       ├── momentum.py        # Momentum strategies
│       ├── trend.py           # Trend strategies
│       └── volatility.py      # Volatility strategies
├── README.md                  # This file
└── requirements.txt

notebooks/
├── 01_basic_backtesting.ipynb
├── 02_parameter_optimization.ipynb
├── 03_genetic_optimization.ipynb
└── 04_multi_indicator_strategies.ipynb
```

## Development

### Build and Test

```bash
# Build library
maturin develop --release

# Run Python tests
pytest python_tests/

# Run Rust tests
cargo test

# Run benchmarks
cargo bench
```

### Contributing

Contributions welcome! Please follow the existing code style and add tests for new features.

## License

GNU Affero General Public License v3.0 or later (AGPLv3+)

## Links

- **Homepage**: https://github.com/kimasplund/kimsfinance
- **Documentation**: https://github.com/kimasplund/kimsfinance/tree/master/rust
- **Issues**: https://github.com/kimasplund/kimsfinance/issues

## Support

For questions and discussions, please use GitHub Issues.
