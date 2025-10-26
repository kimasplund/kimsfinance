"""
Momentum-based trading strategies

These strategies use momentum indicators like RSI, ROC, Stochastic, Williams %R, and CCI
to identify overbought/oversold conditions and momentum shifts.
"""


class RSIStrategy:
    """
    RSI (Relative Strength Index) mean reversion strategy

    Strategy logic:
    - Buy when RSI < buy_threshold (oversold)
    - Sell when RSI > sell_threshold (overbought)
    - Hold otherwise

    Parameters:
    - period: RSI calculation period (default: 14)
    - buy_threshold: RSI level to trigger buy (default: 30)
    - sell_threshold: RSI level to trigger sell (default: 70)
    - position_pct: Position size as percentage of equity (default: 1.0 = 100%)
    """

    def __init__(self, period=14, buy_threshold=30.0, sell_threshold=70.0, position_pct=1.0):
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.position_pct = position_pct

    def on_data(self, bar, indicators):
        """Trading logic called for each bar"""
        rsi = indicators.get(f'rsi_{self.period}', 50.0)

        if rsi < self.buy_threshold:
            return 'buy'
        elif rsi > self.sell_threshold:
            return 'sell'
        return 'hold'

    def get_indicators(self):
        """Indicators required by this strategy"""
        return [f'rsi_{self.period}']

    def position_size(self, equity, signal):
        """Position sizing logic"""
        return self.position_pct


class ROCStrategy:
    """
    ROC (Rate of Change) momentum strategy

    Strategy logic:
    - Buy when ROC crosses above buy_threshold (bullish momentum)
    - Sell when ROC crosses below sell_threshold (bearish momentum)

    Parameters:
    - period: ROC calculation period (default: 14)
    - buy_threshold: ROC threshold for buy signal (default: 5.0%)
    - sell_threshold: ROC threshold for sell signal (default: -5.0%)
    """

    def __init__(self, period=14, buy_threshold=5.0, sell_threshold=-5.0):
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.prev_roc = None

    def on_data(self, bar, indicators):
        """Trading logic with crossover detection"""
        roc = indicators.get(f'roc_{self.period}', 0.0)

        if self.prev_roc is not None:
            # Bullish crossover
            if self.prev_roc <= self.buy_threshold and roc > self.buy_threshold:
                self.prev_roc = roc
                return 'buy'
            # Bearish crossover
            elif self.prev_roc >= self.sell_threshold and roc < self.sell_threshold:
                self.prev_roc = roc
                return 'sell'

        self.prev_roc = roc
        return 'hold'

    def get_indicators(self):
        return [f'roc_{self.period}']


class StochasticStrategy:
    """
    Stochastic Oscillator strategy

    Strategy logic:
    - Buy when %K crosses above %D in oversold region (< 20)
    - Sell when %K crosses below %D in overbought region (> 80)

    Parameters:
    - k_period: %K line period (default: 14)
    - d_period: %D line period (default: 3)
    - oversold: Oversold threshold (default: 20)
    - overbought: Overbought threshold (default: 80)
    """

    def __init__(self, k_period=14, d_period=3, oversold=20, overbought=80):
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought
        self.prev_k = None
        self.prev_d = None

    def on_data(self, bar, indicators):
        """Trading logic with crossover detection"""
        # Note: Stochastic indicator would need multi-output support
        # For now, use simplified logic with %K only
        k = indicators.get(f'stochastic_k_{self.k_period}', 50.0)
        d = indicators.get(f'stochastic_d_{self.d_period}', 50.0)

        if self.prev_k is not None and self.prev_d is not None:
            # Bullish crossover in oversold region
            if k < self.oversold and self.prev_k <= self.prev_d and k > d:
                self.prev_k, self.prev_d = k, d
                return 'buy'
            # Bearish crossover in overbought region
            elif k > self.overbought and self.prev_k >= self.prev_d and k < d:
                self.prev_k, self.prev_d = k, d
                return 'sell'

        self.prev_k, self.prev_d = k, d
        return 'hold'

    def get_indicators(self):
        return [f'stochastic_k_{self.k_period}', f'stochastic_d_{self.d_period}']


class WilliamsRStrategy:
    """
    Williams %R oscillator strategy

    Strategy logic:
    - Buy when Williams %R crosses above oversold threshold (e.g., -80)
    - Sell when Williams %R crosses below overbought threshold (e.g., -20)

    Parameters:
    - period: Williams %R calculation period (default: 14)
    - oversold: Oversold threshold (default: -80)
    - overbought: Overbought threshold (default: -20)
    """

    def __init__(self, period=14, oversold=-80, overbought=-20):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.prev_wr = None

    def on_data(self, bar, indicators):
        """Trading logic with crossover detection"""
        wr = indicators.get(f'williamsr_{self.period}', -50.0)

        if self.prev_wr is not None:
            # Bullish crossover
            if self.prev_wr <= self.oversold and wr > self.oversold:
                self.prev_wr = wr
                return 'buy'
            # Bearish crossover
            elif self.prev_wr >= self.overbought and wr < self.overbought:
                self.prev_wr = wr
                return 'sell'

        self.prev_wr = wr
        return 'hold'

    def get_indicators(self):
        return [f'williamsr_{self.period}']


class CCIStrategy:
    """
    CCI (Commodity Channel Index) strategy

    Strategy logic:
    - Buy when CCI crosses above oversold threshold (typically -100)
    - Sell when CCI crosses below overbought threshold (typically +100)

    Parameters:
    - period: CCI calculation period (default: 20)
    - oversold: Oversold threshold (default: -100)
    - overbought: Overbought threshold (default: +100)
    """

    def __init__(self, period=20, oversold=-100, overbought=100):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.prev_cci = None

    def on_data(self, bar, indicators):
        """Trading logic with crossover detection"""
        cci = indicators.get(f'cci_{self.period}', 0.0)

        if self.prev_cci is not None:
            # Bullish crossover
            if self.prev_cci <= self.oversold and cci > self.oversold:
                self.prev_cci = cci
                return 'buy'
            # Bearish crossover
            elif self.prev_cci >= self.overbought and cci < self.overbought:
                self.prev_cci = cci
                return 'sell'

        self.prev_cci = cci
        return 'hold'

    def get_indicators(self):
        return [f'cci_{self.period}']
