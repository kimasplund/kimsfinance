# Python API Tests for Backtesting Engine

This directory contains Python tests and examples for the `kimsfinance_core` backtesting API.

## Setup

1. **Build the Rust extension**:
   ```bash
   cd /home/kim-asplund/projects/kimsfinance/rust
   maturin develop --release
   ```

2. **Install test dependencies** (optional, for pytest):
   ```bash
   pip install pytest numpy
   ```

## Running Tests

### Run all tests with pytest:
```bash
pytest python_tests/test_backtest_api.py -v
```

### Run tests manually:
```bash
python python_tests/test_backtest_api.py
```

### Run the example:
```bash
python python_tests/example_backtest.py
```

## API Overview

### Strategy Interface

Create a Python strategy class with these methods:

```python
class MyStrategy:
    def on_data(self, bar, indicators):
        """
        Called for each bar with OHLCV data and pre-calculated indicators

        Args:
            bar: Dict with keys: timestamp, open, high, low, close, volume
            indicators: Dict with indicator values (e.g., {'rsi_14': 65.5})

        Returns:
            str: One of 'buy', 'sell', 'hold', 'short', 'cover'
        """
        rsi = indicators.get('rsi_14', 50.0)
        if rsi < 30:
            return 'buy'
        elif rsi > 70:
            return 'sell'
        return 'hold'

    def get_indicators(self):
        """
        List of required indicators

        Returns:
            list: Indicator names in format 'indicator_period'
                  Supported: rsi_N, atr_N, sma_N, ema_N, cci_N, roc_N, williamsr_N
        """
        return ['rsi_14']

    def position_size(self, equity, signal):
        """
        Optional: Position sizing logic

        Args:
            equity: Current equity value
            signal: Current signal string

        Returns:
            float: Position size (1.0 = 100% of capital)
        """
        return 1.0  # Full allocation
```

### Run Backtest

```python
import kimsfinance_core
import numpy as np

# Prepare OHLCV data (NumPy arrays)
timestamps = np.array([...], dtype=np.int64)
open_prices = np.array([...], dtype=np.float64)
high = np.array([...], dtype=np.float64)
low = np.array([...], dtype=np.float64)
close = np.array([...], dtype=np.float64)
volume = np.array([...], dtype=np.float64)

# Create strategy instance
strategy = MyStrategy()

# Run backtest
result = kimsfinance_core.run_backtest(
    high=high,
    low=low,
    close=close,
    open_prices=open_prices,
    volume=volume,
    timestamps=timestamps,
    strategy=strategy,
    initial_capital=10000.0,  # Optional, default: 10000.0
    trading_fee=0.001,         # Optional, default: 0.001 (0.1%)
    slippage=0.0005,          # Optional, default: 0.0005 (0.05%)
    use_gpu=False             # Optional, default: True
)
```

### Result Format

```python
{
    'final_equity': 10500.0,          # Final account value
    'total_return': 5.0,              # Total return percentage
    'sharpe_ratio': 2.5,              # Annualized Sharpe ratio
    'max_drawdown': 10.5,             # Maximum drawdown percentage
    'win_rate': 65.0,                 # Win rate percentage
    'num_trades': 42,                 # Number of trades executed
    'profit_factor': 1.8,             # Gross profit / gross loss
    'equity_curve': np.array([...]),  # Equity values over time
    'trades': [                       # List of trade dictionaries
        {
            'entry_time': 1234,       # Entry timestamp
            'exit_time': 5678,        # Exit timestamp
            'entry_price': 100.0,     # Entry price
            'exit_price': 105.0,      # Exit price
            'quantity': 1.0,          # Position size
            'direction': 'long',      # 'long' or 'short'
            'pnl': 100.0,             # Profit/Loss in dollars
            'pnl_percent': 5.0        # Profit/Loss percentage
        },
        ...
    ]
}
```

## Supported Indicators

The following indicators can be used in `get_indicators()`:

- `rsi_N` - Relative Strength Index (period N)
- `atr_N` - Average True Range (period N)
- `sma_N` - Simple Moving Average (period N)
- `ema_N` - Exponential Moving Average (period N)
- `cci_N` - Commodity Channel Index (period N)
- `roc_N` - Rate of Change (period N)
- `williamsr_N` - Williams %R (period N)

Example: `['rsi_14', 'atr_20', 'sma_50']`

## Test Coverage

- ✅ API availability and function signature
- ✅ Strategy integration and callbacks
- ✅ NumPy array conversions (f64, i64)
- ✅ Result dictionary structure
- ✅ Trade detail format
- ✅ Custom backtest parameters
- ✅ Error handling (empty data, mismatched arrays)
- ✅ Multiple strategy types (RSI, MA crossover, etc.)
- ✅ Uptrend, downtrend, and sideways markets

## Performance Notes

- **GPU Acceleration**: Set `use_gpu=True` to enable GPU-accelerated indicator calculation (requires CUDA)
- **CPU Fallback**: Automatically falls back to CPU if GPU is not available
- **Large Datasets**: GPU acceleration is most beneficial for datasets >10K rows
- **Batch Processing**: Indicators are calculated in batch for efficiency

## Files

- `test_backtest_api.py` - Comprehensive test suite (pytest)
- `example_backtest.py` - Usage examples with multiple strategies
- `README.md` - This file
