"""
Trend-following trading strategies

These strategies use trend indicators like MACD, moving average crossovers,
and trend strength to identify and follow market trends.
"""


class MACDStrategy:
    """
    MACD (Moving Average Convergence Divergence) crossover strategy

    Strategy logic:
    - Buy when MACD line crosses above signal line (bullish crossover)
    - Sell when MACD line crosses below signal line (bearish crossover)

    Parameters:
    - fast_period: Fast EMA period (default: 12)
    - slow_period: Slow EMA period (default: 26)
    - signal_period: Signal line period (default: 9)
    """

    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.prev_macd = None
        self.prev_signal = None

    def on_data(self, bar, indicators):
        """Trading logic with crossover detection"""
        # MACD returns dict with 'macd', 'signal', 'histogram'
        macd = indicators.get('macd_line', 0.0)
        signal = indicators.get('macd_signal', 0.0)

        if self.prev_macd is not None and self.prev_signal is not None:
            # Bullish crossover: MACD crosses above signal
            if self.prev_macd <= self.prev_signal and macd > signal:
                self.prev_macd, self.prev_signal = macd, signal
                return 'buy'
            # Bearish crossover: MACD crosses below signal
            elif self.prev_macd >= self.prev_signal and macd < signal:
                self.prev_macd, self.prev_signal = macd, signal
                return 'sell'

        self.prev_macd, self.prev_signal = macd, signal
        return 'hold'

    def get_indicators(self):
        # Note: Actual implementation would need multi-output indicator support
        return ['macd_line', 'macd_signal']


class EMACrossoverStrategy:
    """
    Exponential Moving Average crossover strategy

    Strategy logic:
    - Buy when fast EMA crosses above slow EMA (golden cross)
    - Sell when fast EMA crosses below slow EMA (death cross)

    Parameters:
    - fast_period: Fast EMA period (default: 12)
    - slow_period: Slow EMA period (default: 26)
    """

    def __init__(self, fast_period=12, slow_period=26):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prev_fast = None
        self.prev_slow = None

    def on_data(self, bar, indicators):
        """Trading logic with crossover detection"""
        fast_ema = indicators.get(f'ema_{self.fast_period}', bar['close'])
        slow_ema = indicators.get(f'ema_{self.slow_period}', bar['close'])

        if self.prev_fast is not None and self.prev_slow is not None:
            # Golden cross: fast EMA crosses above slow EMA
            if self.prev_fast <= self.prev_slow and fast_ema > slow_ema:
                self.prev_fast, self.prev_slow = fast_ema, slow_ema
                return 'buy'
            # Death cross: fast EMA crosses below slow EMA
            elif self.prev_fast >= self.prev_slow and fast_ema < slow_ema:
                self.prev_fast, self.prev_slow = fast_ema, slow_ema
                return 'sell'

        self.prev_fast, self.prev_slow = fast_ema, slow_ema
        return 'hold'

    def get_indicators(self):
        return [f'ema_{self.fast_period}', f'ema_{self.slow_period}']


class DualMAStrategy:
    """
    Dual Moving Average strategy with SMA

    Strategy logic:
    - Buy when fast SMA crosses above slow SMA
    - Sell when fast SMA crosses below slow SMA

    Parameters:
    - fast_period: Fast SMA period (default: 50)
    - slow_period: Slow SMA period (default: 200)
    """

    def __init__(self, fast_period=50, slow_period=200):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prev_fast = None
        self.prev_slow = None

    def on_data(self, bar, indicators):
        """Trading logic with crossover detection"""
        fast_sma = indicators.get(f'sma_{self.fast_period}', bar['close'])
        slow_sma = indicators.get(f'sma_{self.slow_period}', bar['close'])

        if self.prev_fast is not None and self.prev_slow is not None:
            # Bullish crossover
            if self.prev_fast <= self.prev_slow and fast_sma > slow_sma:
                self.prev_fast, self.prev_slow = fast_sma, slow_sma
                return 'buy'
            # Bearish crossover
            elif self.prev_fast >= self.prev_slow and fast_sma < slow_sma:
                self.prev_fast, self.prev_slow = fast_sma, slow_sma
                return 'sell'

        self.prev_fast, self.prev_slow = fast_sma, slow_sma
        return 'hold'

    def get_indicators(self):
        return [f'sma_{self.fast_period}', f'sma_{self.slow_period}']


class TrendFollowingStrategy:
    """
    Multi-timeframe trend following strategy

    Strategy logic:
    - Enter long when price > long-term MA AND short-term MA slopes up
    - Exit when short-term MA slopes down
    - Uses ATR for position sizing

    Parameters:
    - short_period: Short-term trend detection (default: 20)
    - long_period: Long-term trend filter (default: 50)
    - atr_period: ATR period for position sizing (default: 14)
    - atr_multiplier: Risk per trade in ATR units (default: 2.0)
    """

    def __init__(self, short_period=20, long_period=50, atr_period=14, atr_multiplier=2.0):
        self.short_period = short_period
        self.long_period = long_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.prev_short_ma = None

    def on_data(self, bar, indicators):
        """Trading logic with trend detection"""
        short_ma = indicators.get(f'ema_{self.short_period}', bar['close'])
        long_ma = indicators.get(f'ema_{self.long_period}', bar['close'])
        price = bar['close']

        # Check if price is above long-term trend
        in_uptrend = price > long_ma

        if self.prev_short_ma is not None:
            short_ma_slope = short_ma - self.prev_short_ma

            # Buy signal: uptrend + short MA slopes up
            if in_uptrend and short_ma_slope > 0 and self.prev_short_ma is not None:
                if self.prev_short_ma <= price and short_ma > price:  # Price crosses above short MA
                    self.prev_short_ma = short_ma
                    return 'buy'

            # Sell signal: short MA slopes down
            elif short_ma_slope < 0:
                if self.prev_short_ma >= price and short_ma < price:  # Price crosses below short MA
                    self.prev_short_ma = short_ma
                    return 'sell'

        self.prev_short_ma = short_ma
        return 'hold'

    def get_indicators(self):
        return [
            f'ema_{self.short_period}',
            f'ema_{self.long_period}',
            f'atr_{self.atr_period}'
        ]

    def position_size(self, equity, signal):
        """ATR-based position sizing"""
        # In production, would calculate: equity / (atr * atr_multiplier)
        # For now, return fixed allocation
        return 1.0
