"""
Volatility-based trading strategies

These strategies use volatility indicators like ATR, Bollinger Bands, and Keltner Channels
to identify breakouts, reversals, and volatility expansion/contraction.
"""


class ATRBreakoutStrategy:
    """
    ATR-based breakout strategy

    Strategy logic:
    - Buy when price breaks above (previous close + ATR * multiplier)
    - Sell when price breaks below (previous close - ATR * multiplier)

    Parameters:
    - period: ATR calculation period (default: 14)
    - multiplier: ATR multiplier for breakout threshold (default: 2.0)
    """

    def __init__(self, period=14, multiplier=2.0):
        self.period = period
        self.multiplier = multiplier
        self.prev_close = None

    def on_data(self, bar, indicators):
        """Trading logic for ATR breakout"""
        atr = indicators.get(f'atr_{self.period}', 0.0)
        current_close = bar['close']

        if self.prev_close is not None and atr > 0:
            upper_breakout = self.prev_close + (atr * self.multiplier)
            lower_breakout = self.prev_close - (atr * self.multiplier)

            # Bullish breakout
            if current_close > upper_breakout:
                self.prev_close = current_close
                return 'buy'
            # Bearish breakout
            elif current_close < lower_breakout:
                self.prev_close = current_close
                return 'sell'

        self.prev_close = current_close
        return 'hold'

    def get_indicators(self):
        return [f'atr_{self.period}']


class BollingerBreakoutStrategy:
    """
    Bollinger Bands breakout strategy

    Strategy logic:
    - Buy when price closes above upper band (breakout)
    - Sell when price closes below lower band (breakdown)
    - Exit when price returns to middle band

    Parameters:
    - period: Bollinger Bands period (default: 20)
    - std_dev: Standard deviation multiplier (default: 2.0)
    """

    def __init__(self, period=20, std_dev=2.0):
        self.period = period
        self.std_dev = std_dev

    def on_data(self, bar, indicators):
        """Trading logic for Bollinger breakout"""
        # Note: Would need multi-output indicator support
        bb_upper = indicators.get('bb_upper', float('inf'))
        bb_lower = indicators.get('bb_lower', float('-inf'))
        bb_middle = indicators.get('bb_middle', bar['close'])
        price = bar['close']

        # Breakout above upper band
        if price > bb_upper:
            return 'buy'
        # Breakdown below lower band
        elif price < bb_lower:
            return 'sell'
        # Mean reversion to middle band (exit signal)
        # In production, would track position state
        return 'hold'

    def get_indicators(self):
        return ['bb_upper', 'bb_lower', 'bb_middle']


class KeltnerBreakoutStrategy:
    """
    Keltner Channels breakout strategy

    Strategy logic:
    - Buy when price breaks above upper Keltner Channel
    - Sell when price breaks below lower Keltner Channel
    - More robust than Bollinger Bands (uses ATR instead of std dev)

    Parameters:
    - ema_period: EMA period for middle line (default: 20)
    - atr_period: ATR period for channel width (default: 10)
    - multiplier: ATR multiplier for channel width (default: 2.0)
    """

    def __init__(self, ema_period=20, atr_period=10, multiplier=2.0):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def on_data(self, bar, indicators):
        """Trading logic for Keltner breakout"""
        kc_upper = indicators.get('kc_upper', float('inf'))
        kc_lower = indicators.get('kc_lower', float('-inf'))
        price = bar['close']

        # Breakout above upper channel
        if price > kc_upper:
            return 'buy'
        # Breakdown below lower channel
        elif price < kc_lower:
            return 'sell'
        return 'hold'

    def get_indicators(self):
        return ['kc_upper', 'kc_lower', 'kc_middle']


class VolatilityContractionStrategy:
    """
    Volatility contraction/expansion strategy (Bollinger Squeeze)

    Strategy logic:
    - Detect volatility squeeze when BB bands are narrow (< threshold)
    - Enter on first breakout direction after squeeze
    - Exit when volatility expands significantly

    Parameters:
    - period: Bollinger Bands period (default: 20)
    - std_dev: Standard deviation multiplier (default: 2.0)
    - squeeze_threshold: Band width threshold for squeeze (default: 0.02 = 2%)
    """

    def __init__(self, period=20, std_dev=2.0, squeeze_threshold=0.02):
        self.period = period
        self.std_dev = std_dev
        self.squeeze_threshold = squeeze_threshold
        self.in_squeeze = False
        self.prev_close = None

    def on_data(self, bar, indicators):
        """Trading logic for volatility squeeze"""
        bb_upper = indicators.get('bb_upper', 0.0)
        bb_lower = indicators.get('bb_lower', 0.0)
        bb_middle = indicators.get('bb_middle', bar['close'])
        price = bar['close']

        # Calculate band width as percentage of middle band
        if bb_middle > 0:
            band_width = (bb_upper - bb_lower) / bb_middle
        else:
            return 'hold'

        # Detect squeeze (low volatility)
        if band_width < self.squeeze_threshold:
            self.in_squeeze = True
        elif self.in_squeeze and band_width > self.squeeze_threshold * 1.5:
            # Volatility expanding after squeeze
            if self.prev_close is not None:
                # Trade in direction of breakout
                if price > self.prev_close:
                    self.in_squeeze = False
                    self.prev_close = price
                    return 'buy'
                elif price < self.prev_close:
                    self.in_squeeze = False
                    self.prev_close = price
                    return 'sell'

        self.prev_close = price
        return 'hold'

    def get_indicators(self):
        return ['bb_upper', 'bb_lower', 'bb_middle']
