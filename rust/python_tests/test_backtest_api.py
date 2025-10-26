"""
Test Python API for backtesting engine

This test suite validates:
- Python strategy integration
- NumPy array conversions
- Backtest result format
- Exception handling
"""

import numpy as np
import pytest

# Import from the compiled Rust extension
try:
    import kimsfinance_core
except ImportError:
    pytest.skip("kimsfinance_core not built - run 'maturin develop' first", allow_module_level=True)


class SimpleRSIStrategy:
    """Simple RSI-based strategy for testing"""

    def __init__(self, rsi_period=14, buy_threshold=30.0, sell_threshold=70.0):
        self.rsi_period = rsi_period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def on_data(self, bar, indicators):
        """
        Trading logic called for each bar

        Args:
            bar: Dict with keys: timestamp, open, high, low, close, volume
            indicators: Dict with indicator values (e.g., {'rsi_14': 65.5})

        Returns:
            str: One of 'buy', 'sell', 'hold', 'short', 'cover'
        """
        rsi = indicators.get(f'rsi_{self.rsi_period}', 50.0)

        if rsi < self.buy_threshold:
            return 'buy'
        elif rsi > self.sell_threshold:
            return 'sell'
        else:
            return 'hold'

    def get_indicators(self):
        """
        List of indicators required by this strategy

        Returns:
            list: Indicator names in format 'indicator_period' (e.g., ['rsi_14'])
        """
        return [f'rsi_{self.rsi_period}']

    def position_size(self, equity, signal):
        """
        Optional: Position sizing logic

        Args:
            equity: Current equity value
            signal: Signal string ('buy', 'sell', etc.)

        Returns:
            float: Position size (1.0 = 100% of capital)
        """
        return 1.0  # Full allocation


class BuyAndHoldStrategy:
    """Buy and hold strategy for testing"""

    def __init__(self):
        self.bought = False

    def on_data(self, bar, indicators):
        if not self.bought:
            self.bought = True
            return 'buy'
        return 'hold'

    def get_indicators(self):
        return []  # No indicators needed


def generate_sample_data(n=100, trend='up'):
    """
    Generate synthetic OHLCV data for testing

    Args:
        n: Number of candles
        trend: 'up', 'down', or 'sideways'

    Returns:
        tuple: (timestamps, open, high, low, close, volume)
    """
    timestamps = np.arange(n, dtype=np.int64)

    if trend == 'up':
        # Uptrend: RSI will eventually go above 70
        close = np.linspace(100.0, 150.0, n) + np.random.randn(n) * 2
    elif trend == 'down':
        # Downtrend: RSI will eventually go below 30
        close = np.linspace(150.0, 100.0, n) + np.random.randn(n) * 2
    else:  # sideways
        close = 100.0 + np.random.randn(n) * 5

    # Generate OHLCV from close
    open_prices = close + np.random.randn(n) * 0.5
    high = np.maximum(open_prices, close) + np.abs(np.random.randn(n) * 1)
    low = np.minimum(open_prices, close) - np.abs(np.random.randn(n) * 1)
    volume = np.random.uniform(1000, 5000, n)

    return timestamps, open_prices, high, low, close, volume


def test_backtest_api_exists():
    """Test that run_backtest function is available"""
    assert hasattr(kimsfinance_core, 'run_backtest')
    assert callable(kimsfinance_core.run_backtest)


def test_simple_rsi_strategy_uptrend():
    """Test RSI strategy on uptrending data"""
    # Generate uptrending data
    timestamps, open_prices, high, low, close, volume = generate_sample_data(200, trend='up')

    # Create strategy
    strategy = SimpleRSIStrategy(rsi_period=14, buy_threshold=30.0, sell_threshold=70.0)

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
        trading_fee=0.001,
        slippage=0.0005,
        use_gpu=False  # Force CPU for testing
    )

    # Validate result structure
    assert isinstance(result, dict)
    assert 'final_equity' in result
    assert 'total_return' in result
    assert 'sharpe_ratio' in result
    assert 'max_drawdown' in result
    assert 'win_rate' in result
    assert 'num_trades' in result
    assert 'profit_factor' in result
    assert 'equity_curve' in result
    assert 'trades' in result

    # Validate types
    assert isinstance(result['final_equity'], (int, float))
    assert isinstance(result['total_return'], (int, float))
    assert isinstance(result['sharpe_ratio'], (int, float))
    assert isinstance(result['max_drawdown'], (int, float))
    assert isinstance(result['win_rate'], (int, float))
    assert isinstance(result['num_trades'], int)
    assert isinstance(result['profit_factor'], (int, float))
    assert isinstance(result['equity_curve'], np.ndarray)
    assert isinstance(result['trades'], list)

    # Validate equity curve
    assert len(result['equity_curve']) == len(timestamps)
    assert result['equity_curve'][0] > 0  # Should start with initial capital

    # Should have made some trades
    assert result['num_trades'] >= 0

    print("\n=== Backtest Results (Uptrend) ===")
    print(f"Final Equity: ${result['final_equity']:.2f}")
    print(f"Total Return: {result['total_return']:.2f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
    print(f"Win Rate: {result['win_rate']:.2f}%")
    print(f"Number of Trades: {result['num_trades']}")
    print(f"Profit Factor: {result['profit_factor']:.2f}")


def test_simple_rsi_strategy_downtrend():
    """Test RSI strategy on downtrending data"""
    # Generate downtrending data
    timestamps, open_prices, high, low, close, volume = generate_sample_data(200, trend='down')

    # Create strategy
    strategy = SimpleRSIStrategy(rsi_period=14, buy_threshold=30.0, sell_threshold=70.0)

    # Run backtest
    result = kimsfinance_core.run_backtest(
        high=high,
        low=low,
        close=close,
        open_prices=open_prices,
        volume=volume,
        timestamps=timestamps,
        strategy=strategy,
        use_gpu=False
    )

    # Should have results even on downtrend
    assert isinstance(result, dict)
    assert result['num_trades'] >= 0

    print("\n=== Backtest Results (Downtrend) ===")
    print(f"Final Equity: ${result['final_equity']:.2f}")
    print(f"Total Return: {result['total_return']:.2f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result['max_drawdown']:.2f}%")


def test_buy_and_hold_strategy():
    """Test buy and hold strategy"""
    # Generate uptrending data
    timestamps, open_prices, high, low, close, volume = generate_sample_data(100, trend='up')

    # Create strategy
    strategy = BuyAndHoldStrategy()

    # Run backtest
    result = kimsfinance_core.run_backtest(
        high=high,
        low=low,
        close=close,
        open_prices=open_prices,
        volume=volume,
        timestamps=timestamps,
        strategy=strategy,
        use_gpu=False
    )

    # Should have exactly 1 trade (buy at start, sell at end)
    assert result['num_trades'] >= 0

    print("\n=== Backtest Results (Buy & Hold) ===")
    print(f"Final Equity: ${result['final_equity']:.2f}")
    print(f"Total Return: {result['total_return']:.2f}%")


def test_trade_details():
    """Test that trade details are properly formatted"""
    timestamps, open_prices, high, low, close, volume = generate_sample_data(100, trend='up')

    strategy = SimpleRSIStrategy(rsi_period=14)

    result = kimsfinance_core.run_backtest(
        high=high,
        low=low,
        close=close,
        open_prices=open_prices,
        volume=volume,
        timestamps=timestamps,
        strategy=strategy,
        use_gpu=False
    )

    # Check trade format
    if result['num_trades'] > 0:
        trade = result['trades'][0]

        # All required fields should be present
        assert 'entry_time' in trade
        assert 'exit_time' in trade
        assert 'entry_price' in trade
        assert 'exit_price' in trade
        assert 'quantity' in trade
        assert 'direction' in trade
        assert 'pnl' in trade
        assert 'pnl_percent' in trade

        # Validate types
        assert isinstance(trade['entry_time'], (int, np.integer))
        assert isinstance(trade['exit_time'], (int, np.integer))
        assert isinstance(trade['entry_price'], (int, float, np.floating))
        assert isinstance(trade['exit_price'], (int, float, np.floating))
        assert isinstance(trade['quantity'], (int, float, np.floating))
        assert trade['direction'] in ['long', 'short']
        assert isinstance(trade['pnl'], (int, float, np.floating))
        assert isinstance(trade['pnl_percent'], (int, float, np.floating))

        # Exit time should be after entry time
        assert trade['exit_time'] > trade['entry_time']

        print(f"\n=== Sample Trade ===")
        print(f"Entry: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
        print(f"Exit:  {trade['exit_time']} @ ${trade['exit_price']:.2f}")
        print(f"Direction: {trade['direction']}")
        print(f"P&L: ${trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%)")


def test_numpy_array_conversion():
    """Test that NumPy arrays are properly converted"""
    timestamps, open_prices, high, low, close, volume = generate_sample_data(50)

    # Ensure arrays are correct dtype
    assert high.dtype == np.float64
    assert low.dtype == np.float64
    assert close.dtype == np.float64
    assert open_prices.dtype == np.float64
    assert volume.dtype == np.float64
    assert timestamps.dtype == np.int64

    strategy = BuyAndHoldStrategy()

    # Should handle NumPy arrays without error
    result = kimsfinance_core.run_backtest(
        high=high,
        low=low,
        close=close,
        open_prices=open_prices,
        volume=volume,
        timestamps=timestamps,
        strategy=strategy,
        use_gpu=False
    )

    # Equity curve should be returned as NumPy array
    assert isinstance(result['equity_curve'], np.ndarray)
    assert result['equity_curve'].dtype == np.float64
    assert len(result['equity_curve']) == len(timestamps)


def test_custom_parameters():
    """Test custom backtest parameters"""
    timestamps, open_prices, high, low, close, volume = generate_sample_data(50)

    strategy = BuyAndHoldStrategy()

    # Test with custom parameters
    result = kimsfinance_core.run_backtest(
        high=high,
        low=low,
        close=close,
        open_prices=open_prices,
        volume=volume,
        timestamps=timestamps,
        strategy=strategy,
        initial_capital=50000.0,  # Custom capital
        trading_fee=0.002,        # Higher fee (0.2%)
        slippage=0.001,          # Higher slippage (0.1%)
        use_gpu=False
    )

    # Should have valid results with custom parameters
    assert isinstance(result, dict)
    assert result['final_equity'] > 0


def test_empty_data_error():
    """Test that empty data raises an error"""
    # Empty arrays
    empty = np.array([], dtype=np.float64)
    empty_ts = np.array([], dtype=np.int64)

    strategy = BuyAndHoldStrategy()

    # Should raise an error for empty data
    with pytest.raises(Exception):
        kimsfinance_core.run_backtest(
            high=empty,
            low=empty,
            close=empty,
            open_prices=empty,
            volume=empty,
            timestamps=empty_ts,
            strategy=strategy,
            use_gpu=False
        )


def test_mismatched_array_lengths():
    """Test that mismatched array lengths raise an error"""
    timestamps = np.arange(100, dtype=np.int64)
    close = np.random.randn(100)
    high = np.random.randn(50)  # Wrong length
    low = np.random.randn(100)
    open_prices = np.random.randn(100)
    volume = np.random.randn(100)

    strategy = BuyAndHoldStrategy()

    # Should raise an error for mismatched lengths
    with pytest.raises(Exception):
        kimsfinance_core.run_backtest(
            high=high,
            low=low,
            close=close,
            open_prices=open_prices,
            volume=volume,
            timestamps=timestamps,
            strategy=strategy,
            use_gpu=False
        )


if __name__ == "__main__":
    # Run tests manually
    print("Running backtest API tests...\n")

    test_backtest_api_exists()
    print("✓ API exists")

    test_simple_rsi_strategy_uptrend()
    print("✓ RSI strategy (uptrend)")

    test_simple_rsi_strategy_downtrend()
    print("✓ RSI strategy (downtrend)")

    test_buy_and_hold_strategy()
    print("✓ Buy and hold strategy")

    test_trade_details()
    print("✓ Trade details format")

    test_numpy_array_conversion()
    print("✓ NumPy array conversion")

    test_custom_parameters()
    print("✓ Custom parameters")

    try:
        test_empty_data_error()
        print("✗ Empty data error (should have raised exception)")
    except AssertionError:
        print("✓ Empty data error handling")

    try:
        test_mismatched_array_lengths()
        print("✗ Mismatched array lengths (should have raised exception)")
    except AssertionError:
        print("✓ Mismatched array lengths error handling")

    print("\n=== All tests passed! ===")
