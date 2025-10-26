"""
Example: Using the kimsfinance_core backtest API

This example demonstrates how to:
1. Define a custom trading strategy in Python
2. Run a backtest with Rust acceleration
3. Analyze results

Run with: python python_tests/example_backtest.py
"""

import numpy as np

try:
    import kimsfinance_core
except ImportError:
    print("ERROR: kimsfinance_core not found")
    print("Run: maturin develop --release")
    exit(1)


class SimpleRSIStrategy:
    """
    Simple RSI mean reversion strategy

    Buy when RSI < 30 (oversold)
    Sell when RSI > 70 (overbought)
    """

    def __init__(self, rsi_period=14, buy_threshold=30.0, sell_threshold=70.0):
        self.rsi_period = rsi_period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def on_data(self, bar, indicators):
        """
        Trading logic called for each bar

        Args:
            bar: Dict with OHLCV data
            indicators: Dict with pre-calculated indicators

        Returns:
            str: Trading signal ('buy', 'sell', 'hold', 'short', 'cover')
        """
        # Get RSI value (defaults to 50 if not available)
        rsi = indicators.get(f'rsi_{self.rsi_period}', 50.0)

        # Mean reversion strategy
        if rsi < self.buy_threshold:
            return 'buy'
        elif rsi > self.sell_threshold:
            return 'sell'
        else:
            return 'hold'

    def get_indicators(self):
        """
        Indicators required by this strategy

        Returns:
            list: List of indicator strings in format 'name_period'
        """
        return [f'rsi_{self.rsi_period}']

    def position_size(self, equity, signal):
        """
        Position sizing logic (optional)

        Args:
            equity: Current account equity
            signal: Current signal ('buy', 'sell', etc.)

        Returns:
            float: Position size (1.0 = 100% of equity)
        """
        # Full allocation (100% of capital)
        return 1.0


class TrendFollowingStrategy:
    """
    Trend following strategy using dual moving averages

    Buy when fast MA crosses above slow MA
    Sell when fast MA crosses below slow MA
    """

    def __init__(self, fast_period=50, slow_period=200):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prev_fast = None
        self.prev_slow = None

    def on_data(self, bar, indicators):
        fast_ma = indicators.get(f'sma_{self.fast_period}', bar['close'])
        slow_ma = indicators.get(f'sma_{self.slow_period}', bar['close'])

        # Detect crossover
        if self.prev_fast is not None and self.prev_slow is not None:
            # Golden cross: fast MA crosses above slow MA (bullish)
            if self.prev_fast <= self.prev_slow and fast_ma > slow_ma:
                self.prev_fast = fast_ma
                self.prev_slow = slow_ma
                return 'buy'
            # Death cross: fast MA crosses below slow MA (bearish)
            elif self.prev_fast >= self.prev_slow and fast_ma < slow_ma:
                self.prev_fast = fast_ma
                self.prev_slow = slow_ma
                return 'sell'

        self.prev_fast = fast_ma
        self.prev_slow = slow_ma
        return 'hold'

    def get_indicators(self):
        return [f'sma_{self.fast_period}', f'sma_{self.slow_period}']


def generate_sample_data(n=1000, trend='up'):
    """Generate synthetic OHLCV data for testing"""
    timestamps = np.arange(n, dtype=np.int64) * 60  # 1-minute bars

    if trend == 'up':
        # Uptrend with volatility
        base = np.linspace(100.0, 200.0, n)
        noise = np.random.randn(n).cumsum() * 2
        close = base + noise
    elif trend == 'down':
        # Downtrend with volatility
        base = np.linspace(200.0, 100.0, n)
        noise = np.random.randn(n).cumsum() * 2
        close = base + noise
    else:  # sideways
        # Sideways with oscillation
        base = 150.0 + 20 * np.sin(np.arange(n) / 20)
        noise = np.random.randn(n) * 5
        close = base + noise

    # Generate realistic OHLC from close
    open_prices = close + np.random.randn(n) * 0.5
    high = np.maximum(open_prices, close) + np.abs(np.random.randn(n) * 2)
    low = np.minimum(open_prices, close) - np.abs(np.random.randn(n) * 2)
    volume = np.random.uniform(1000, 10000, n)

    return timestamps, open_prices, high, low, close, volume


def print_results(result, strategy_name):
    """Pretty print backtest results"""
    print(f"\n{'='*60}")
    print(f"  {strategy_name}")
    print(f"{'='*60}")
    print(f"Final Equity:      ${result['final_equity']:,.2f}")
    print(f"Total Return:      {result['total_return']:+.2f}%")
    print(f"Sharpe Ratio:      {result['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:      {result['max_drawdown']:.2f}%")
    print(f"Win Rate:          {result['win_rate']:.1f}%")
    print(f"Profit Factor:     {result['profit_factor']:.2f}")
    print(f"Number of Trades:  {result['num_trades']}")
    print(f"{'='*60}")

    # Print sample trades
    if result['num_trades'] > 0:
        print("\nSample Trades:")
        for i, trade in enumerate(result['trades'][:5]):  # First 5 trades
            print(f"\n  Trade #{i+1}")
            print(f"    Entry: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
            print(f"    Exit:  {trade['exit_time']} @ ${trade['exit_price']:.2f}")
            print(f"    Type:  {trade['direction']}")
            print(f"    P&L:   ${trade['pnl']:+.2f} ({trade['pnl_percent']:+.2f}%)")

        if result['num_trades'] > 5:
            print(f"\n  ... and {result['num_trades'] - 5} more trades")


def main():
    print("\n" + "="*60)
    print("  kimsfinance_core Backtest API Example")
    print("="*60)

    # Test 1: RSI strategy on uptrend
    print("\n[1] Testing RSI strategy on uptrending market...")
    timestamps, open_p, high, low, close, volume = generate_sample_data(1000, trend='up')

    strategy = SimpleRSIStrategy(rsi_period=14, buy_threshold=30, sell_threshold=70)

    result = kimsfinance_core.run_backtest(
        high=high,
        low=low,
        close=close,
        open_prices=open_p,
        volume=volume,
        timestamps=timestamps,
        strategy=strategy,
        initial_capital=10000.0,
        trading_fee=0.001,  # 0.1% per trade
        slippage=0.0005,    # 0.05% slippage
        use_gpu=False       # CPU mode for testing
    )

    print_results(result, "RSI Strategy (Uptrend)")

    # Test 2: RSI strategy on downtrend
    print("\n\n[2] Testing RSI strategy on downtrending market...")
    timestamps, open_p, high, low, close, volume = generate_sample_data(1000, trend='down')

    strategy = SimpleRSIStrategy(rsi_period=14, buy_threshold=30, sell_threshold=70)

    result = kimsfinance_core.run_backtest(
        high=high,
        low=low,
        close=close,
        open_prices=open_p,
        volume=volume,
        timestamps=timestamps,
        strategy=strategy,
        use_gpu=False
    )

    print_results(result, "RSI Strategy (Downtrend)")

    # Test 3: Trend following on sideways market
    print("\n\n[3] Testing trend following on sideways market...")
    timestamps, open_p, high, low, close, volume = generate_sample_data(1000, trend='sideways')

    strategy = TrendFollowingStrategy(fast_period=50, slow_period=200)

    result = kimsfinance_core.run_backtest(
        high=high,
        low=low,
        close=close,
        open_prices=open_p,
        volume=volume,
        timestamps=timestamps,
        strategy=strategy,
        use_gpu=False
    )

    print_results(result, "Trend Following (Sideways)")

    print("\n" + "="*60)
    print("  All backtests completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
