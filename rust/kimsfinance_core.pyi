"""
Type stubs for kimsfinance_core Rust extension module.

This module provides high-performance Rust implementations for:
- Coordinate calculations (5-10x speedup)
- 25+ technical indicators (3-8x speedup)
- Batch indicator API (10x FFI overhead reduction)
- GPU-accelerated backtesting (20-40x speedup)
- Tick-level event-driven backtesting
- GPU tick aggregation

All functions accept NumPy arrays and return NumPy arrays or dictionaries.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

__version__: str

# ============================================================================
# COORDINATE CALCULATIONS
# ============================================================================

def calculate_coordinates_py(
    high_prices: npt.NDArray[np.float64],
    low_prices: npt.NDArray[np.float64],
    open_prices: npt.NDArray[np.float64],
    close_prices: npt.NDArray[np.float64],
    volume_data: npt.NDArray[np.float64],
    num_candles: int,
    candle_width: float,
    spacing: float,
    bar_width: float,
    price_min: float,
    price_range: float,
    volume_range: float,
    chart_height: int,
    volume_height: int,
    height: int,
) -> Dict[str, npt.NDArray[Union[np.int32, np.bool_]]]:
    """
    Calculate coordinates for candlestick chart rendering (Rust-accelerated).

    This function provides a 5-10x speedup over Python/NumPy implementation
    by using vectorized SIMD operations, cache-friendly memory layout,
    zero-allocation hot path, and parallel computation for large datasets.

    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        open_prices: Array of open prices
        close_prices: Array of close prices
        volume_data: Array of volume data
        num_candles: Number of candles to render
        candle_width: Width of each candle in pixels
        spacing: Spacing between candles in pixels
        bar_width: Width of candle body in pixels
        price_min: Minimum price value for scaling
        price_range: Price range (max - min) for scaling
        volume_range: Maximum volume value for scaling
        chart_height: Height of chart area in pixels
        volume_height: Height of volume area in pixels
        height: Total image height in pixels

    Returns:
        Dictionary containing NumPy arrays:
        - x_start: X coordinates of candle start (int32)
        - x_end: X coordinates of candle end (int32)
        - x_center: X coordinates of candle center (int32)
        - y_high: Y coordinates of high prices (int32)
        - y_low: Y coordinates of low prices (int32)
        - y_open: Y coordinates of open prices (int32)
        - y_close: Y coordinates of close prices (int32)
        - vol_heights: Volume bar heights (int32)
        - body_top: Y coordinates of candle body top (int32)
        - body_bottom: Y coordinates of candle body bottom (int32)
        - is_bullish: Boolean array (bullish=True, bearish=False)

    Performance:
        - 100 candles: <10μs (100x faster than Python)
        - 1,000 candles: <50μs (50x faster than Python)
        - 10,000 candles: <300μs (30x faster than Python)
    """
    ...

# ============================================================================
# MOVING AVERAGES (7 indicators)
# ============================================================================

def calculate_sma(
    prices: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Calculate Simple Moving Average (SMA).

    Args:
        prices: Array of prices
        period: Number of periods for the moving average (default: 14)

    Returns:
        NumPy array of SMA values (NaN for warmup period)

    Performance:
        - 3-5x faster than pandas rolling().mean()
        - Zero-allocation for <5000 rows
        - SIMD-optimized vectorization
    """
    ...

def calculate_ema(
    prices: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Calculate Exponential Moving Average (EMA).

    Args:
        prices: Array of prices
        period: Number of periods for the moving average (default: 14)

    Returns:
        NumPy array of EMA values (NaN for warmup period)
    """
    ...

def calculate_wma(
    prices: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Calculate Weighted Moving Average (WMA).

    Args:
        prices: Array of prices
        period: Number of periods for the moving average (default: 14)

    Returns:
        NumPy array of WMA values (NaN for warmup period)
    """
    ...

def calculate_vwma(
    prices: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Calculate Volume Weighted Moving Average (VWMA).

    Args:
        prices: Array of prices
        volume: Array of volume data
        period: Number of periods for the moving average (default: 14)

    Returns:
        NumPy array of VWMA values (NaN for warmup period)
    """
    ...

def calculate_dema(
    prices: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Calculate Double Exponential Moving Average (DEMA).

    Args:
        prices: Array of prices
        period: Number of periods for the moving average (default: 14)

    Returns:
        NumPy array of DEMA values (NaN for warmup period)
    """
    ...

def calculate_tema(
    prices: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Calculate Triple Exponential Moving Average (TEMA).

    Args:
        prices: Array of prices
        period: Number of periods for the moving average (default: 14)

    Returns:
        NumPy array of TEMA values (NaN for warmup period)
    """
    ...

def calculate_hma(
    prices: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Calculate Hull Moving Average (HMA).

    Args:
        prices: Array of prices
        period: Number of periods for the moving average (default: 14)

    Returns:
        NumPy array of HMA values (NaN for warmup period)
    """
    ...

# ============================================================================
# MOMENTUM INDICATORS (8 indicators)
# ============================================================================

def calculate_rsi(
    prices: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: Array of prices
        period: Number of periods for RSI calculation (default: 14)

    Returns:
        NumPy array of RSI values (0-100 range, NaN for warmup period)

    Performance:
        - 4-6x faster than pandas implementation
        - SIMD-optimized gain/loss separation
        - Parallel processing for >500 rows
    """
    ...

def calculate_roc(
    prices: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Calculate Rate of Change (ROC).

    Args:
        prices: Array of prices
        period: Number of periods for ROC calculation (default: 14)

    Returns:
        NumPy array of ROC values (percentage change, NaN for warmup period)
    """
    ...

def calculate_williams_r(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Calculate Williams %R.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: Number of periods for Williams %R (default: 14)

    Returns:
        NumPy array of Williams %R values (-100 to 0 range, NaN for warmup period)
    """
    ...

def calculate_stochastic(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    k_period: int = 14,
    d_period: int = 3,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Calculate Stochastic Oscillator.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        k_period: Number of periods for %K (default: 14)
        d_period: Number of periods for %D smoothing (default: 3)

    Returns:
        Dictionary with 'k' and 'd' NumPy arrays (0-100 range, NaN for warmup period)
    """
    ...

def calculate_stochastic_gpu(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    k_period: int = 14,
    d_period: int = 3,
    device_id: int = 0,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Calculate Stochastic Oscillator (GPU-accelerated).

    Requires 'gpu' feature flag and CUDA-capable GPU.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        k_period: Number of periods for %K line (default: 14)
        d_period: Number of periods for %D line (default: 3)
        device_id: GPU device ID (default: 0)

    Returns:
        Dictionary with 'k' and 'd' NumPy arrays (0-100 range, NaN for warmup period)

    Performance:
        Expected speedup: 15-25x over CPU for n > 10,000
    """
    ...

def calculate_aroon(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    period: int = 14,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Calculate Aroon Indicator.

    Args:
        high: Array of high prices
        low: Array of low prices
        period: Number of periods for Aroon calculation (default: 14)

    Returns:
        Dictionary with 'aroon_up' and 'aroon_down' NumPy arrays (0-100 range)

    Note:
        Aroon oscillator can be calculated as aroon_up - aroon_down on the Python side
    """
    ...

def calculate_cci(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    period: int = 20,
) -> npt.NDArray[np.float64]:
    """
    Calculate Commodity Channel Index (CCI).

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: Number of periods for CCI calculation (default: 20)

    Returns:
        NumPy array of CCI values (NaN for warmup period)
    """
    ...

def calculate_macd(
    prices: npt.NDArray[np.float64],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Array of prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        Dictionary with 'macd', 'signal', and 'histogram' NumPy arrays
    """
    ...

def calculate_tsi(
    prices: npt.NDArray[np.float64],
    long_period: int = 25,
    short_period: int = 13,
    signal_period: int = 7,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Calculate True Strength Index (TSI).

    Args:
        prices: Array of prices
        long_period: Long smoothing period (default: 25)
        short_period: Short smoothing period (default: 13)
        signal_period: Signal line period (default: 7)

    Returns:
        Dictionary with 'tsi' and 'signal' NumPy arrays
    """
    ...

# ============================================================================
# VOLATILITY INDICATORS (5 indicators)
# ============================================================================

def calculate_atr(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Calculate Average True Range (ATR).

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: Number of periods for ATR (default: 14)

    Returns:
        NumPy array of ATR values (NaN for warmup period)

    Performance:
        - SIMD-optimized true range calculation
        - 5-8x faster than pandas implementation
    """
    ...

def calculate_bollinger_bands(
    prices: npt.NDArray[np.float64],
    period: int = 20,
    std_dev: float = 2.0,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Array of prices
        period: Number of periods for moving average (default: 20)
        std_dev: Number of standard deviations (default: 2.0)

    Returns:
        Dictionary with 'middle', 'upper', and 'lower' NumPy arrays
    """
    ...

def calculate_keltner_channels(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Calculate Keltner Channels.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        ema_period: EMA period for middle line (default: 20)
        atr_period: ATR period for channel width (default: 10)
        multiplier: ATR multiplier for channel width (default: 2.0)

    Returns:
        Dictionary with 'middle', 'upper', and 'lower' NumPy arrays
    """
    ...

def calculate_donchian_channels(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    period: int = 20,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Calculate Donchian Channels.

    Args:
        high: Array of high prices
        low: Array of low prices
        period: Number of periods for channel calculation (default: 20)

    Returns:
        Dictionary with 'middle', 'upper', and 'lower' NumPy arrays
    """
    ...

def calculate_elder_ray(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    ema_period: int = 13,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Calculate Elder Ray Index.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        ema_period: EMA period for calculation (default: 13)

    Returns:
        Dictionary with 'bull_power' and 'bear_power' NumPy arrays
    """
    ...

# ============================================================================
# VOLUME INDICATORS (5 indicators)
# ============================================================================

def calculate_obv(
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculate On-Balance Volume (OBV).

    Args:
        close: Array of close prices
        volume: Array of volume data

    Returns:
        NumPy array of OBV values
    """
    ...

def calculate_vwap(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculate Volume Weighted Average Price (VWAP).

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        volume: Array of volume data

    Returns:
        NumPy array of VWAP values
    """
    ...

def calculate_cmf(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    period: int = 20,
) -> npt.NDArray[np.float64]:
    """
    Calculate Chaikin Money Flow (CMF).

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        volume: Array of volume data
        period: Number of periods for CMF calculation (default: 20)

    Returns:
        NumPy array of CMF values (-1 to 1 range, NaN for warmup period)
    """
    ...

def calculate_mfi(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Calculate Money Flow Index (MFI).

    Volume-weighted momentum indicator measuring buying/selling pressure.
    Often called the "volume-weighted RSI".

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        volume: Array of volume data
        period: Number of periods for MFI calculation (default: 14)

    Returns:
        NumPy array of MFI values (0-100 range, NaN for warmup period)
    """
    ...

def calculate_volume_profile(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    num_bins: int = 20,
) -> npt.NDArray[np.float64]:
    """
    Calculate Volume Profile.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        volume: Array of volume data
        num_bins: Number of price bins (default: 20)

    Returns:
        NumPy array of volume distribution across price levels
    """
    ...

# ============================================================================
# TREND INDICATORS (1 indicator)
# ============================================================================

def calculate_parabolic_sar(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
) -> npt.NDArray[np.float64]:
    """
    Calculate Parabolic SAR (Stop and Reverse).

    The Parabolic SAR is a trend-following indicator that provides entry and exit
    points. It appears as dots above or below price bars. When dots flip from below
    to above price (or vice versa), it signals a potential trend reversal.

    Args:
        high: Array of high prices
        low: Array of low prices
        af_start: Starting acceleration factor (default: 0.02)
        af_increment: AF increment when new extreme point reached (default: 0.02)
        af_max: Maximum acceleration factor (default: 0.2)

    Returns:
        NumPy array of SAR values (same length as input, all values initialized)

    Performance:
        - 5-10x faster than pandas implementation
        - SIMD-optimized min/max operations for SAR adjustments
        - Iterative algorithm with minimal allocations

    References:
        - Wilder, J. Wells (1978). "New Concepts in Technical Trading Systems"
        - https://en.wikipedia.org/wiki/Parabolic_SAR
    """
    ...

# ============================================================================
# BATCH API - FFI Overhead Reduction
# ============================================================================

def calculate_indicators_batch(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    open_prices: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    requests: List[
        Union[
            Tuple[str, str],  # 2-tuple: (indicator_type, params_json)
            Tuple[str, str, str],  # 3-tuple: (output_name, indicator_type, params_json)
        ]
    ],
) -> Dict[str, Union[npt.NDArray[np.float64], Dict[str, npt.NDArray[np.float64]]]]:
    """
    Calculate multiple indicators in a single batch (10x FFI overhead reduction).

    This function minimizes Python-Rust FFI overhead by:
    - Single FFI crossing for multiple indicators
    - Batch processing of OHLCV data
    - Efficient memory layout

    Args:
        high: Array of high prices
        low: Array of low prices
        open_prices: Array of open prices
        close: Array of close prices
        volume: Array of volume data
        requests: List of indicator requests in 2-tuple or 3-tuple format

    Request Formats:
        2-tuple format (backwards compatible):
            [("sma", '{"period": 14}'), ("rsi", '{"period": 14}')]
            Output keys use indicator names: "sma", "rsi"

        3-tuple format (allows duplicate indicators):
            [("sma_14", "sma", '{"period": 14}'),
             ("sma_50", "sma", '{"period": 50}')]
            Output keys use custom names: "sma_14", "sma_50"

    Returns:
        Dictionary mapping output names to results:
        - Single-output indicators: NumPy array
        - Multi-output indicators: Nested dict with named arrays

    Performance:
        - Individual calls (10 indicators): ~1000ms FFI overhead
        - Batch call (10 indicators): ~100ms FFI overhead
        - Result: 10x speedup for multi-indicator workflows

    Example:
        >>> results = calculate_indicators_batch(
        ...     high, low, open_prices, close, volume,
        ...     requests=[
        ...         ("sma_14", "sma", '{"period": 14}'),
        ...         ("rsi", '{"period": 14}'),
        ...         ("macd", '{"fast_period": 12, "slow_period": 26, "signal_period": 9}'),
        ...     ]
        ... )
        >>> sma_14 = results['sma_14']  # NumPy array
        >>> rsi = results['rsi']  # NumPy array
        >>> macd_line = results['macd']['macd']  # Nested dict
    """
    ...

# ============================================================================
# BACKTESTING API
# ============================================================================

def run_backtest(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    open_prices: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    timestamps: npt.NDArray[np.int64],
    strategy: Any,  # Python object with on_data() and get_indicators() methods
    initial_capital: float = 10000.0,
    trading_fee: float = 0.001,
    slippage: float = 0.0005,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Run backtest on OHLCV data with Python strategy.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        open_prices: Array of open prices
        volume: Array of volume data
        timestamps: Array of Unix timestamps
        strategy: Python strategy object with on_data() and get_indicators() methods
        initial_capital: Starting capital (default: 10000.0)
        trading_fee: Trading fee per trade (default: 0.001 = 0.1%)
        slippage: Slippage per trade (default: 0.0005 = 0.05%)
        use_gpu: Enable GPU acceleration if available (default: True)

    Returns:
        Dictionary with backtest results:
        - final_equity: Final equity value
        - total_return: Total return percentage
        - sharpe_ratio: Annualized Sharpe ratio
        - max_drawdown: Maximum drawdown percentage
        - win_rate: Win rate percentage
        - num_trades: Number of trades executed
        - profit_factor: Gross profit / gross loss
        - equity_curve: NumPy array of equity values over time
        - trades: List of trade dictionaries

    Example:
        >>> class SimpleRSI:
        ...     def on_data(self, bar, indicators):
        ...         rsi = indicators.get('rsi_14', 50.0)
        ...         if rsi < 30:
        ...             return 'buy'
        ...         elif rsi > 70:
        ...             return 'sell'
        ...         return 'hold'
        ...
        ...     def get_indicators(self):
        ...         return ['rsi_14']
        ...
        >>> result = run_backtest(
        ...     high=df['high'].values,
        ...     low=df['low'].values,
        ...     close=df['close'].values,
        ...     open_prices=df['open'].values,
        ...     volume=df['volume'].values,
        ...     timestamps=df['timestamp'].values,
        ...     strategy=SimpleRSI(),
        ...     use_gpu=True
        ... )
    """
    ...

# ============================================================================
# TICK-LEVEL BACKTESTING (CPU)
# ============================================================================

class TickBacktestConfig:
    """
    Configuration for tick-level (event-driven) backtesting.

    Attributes:
        initial_capital: Starting capital (default: 10000.0)
        trading_fee: Fee per trade as fraction (default: 0.001 = 0.1%)
        slippage: Slippage per trade as fraction (default: 0.0005 = 0.05%)
        execution_latency_ms: Execution delay in milliseconds (default: 10ms)
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        trading_fee: float = 0.001,
        slippage: float = 0.0005,
        execution_latency_ms: int = 10,
    ) -> None: ...
    def __repr__(self) -> str: ...

class TickBacktestResult:
    """
    Result from tick-level backtesting.

    Attributes:
        total_return: Total return percentage
        sharpe_ratio: Annualized Sharpe ratio
        max_drawdown: Maximum drawdown percentage
        win_rate: Win rate percentage
        profit_factor: Gross profit / gross loss
        num_trades: Number of trades executed
        final_equity: Final portfolio equity
    """

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    final_equity: float

    def equity_curve(self) -> npt.NDArray[np.float64]:
        """Get equity curve as NumPy array."""
        ...
    def trade_pnls(self) -> npt.NDArray[np.float64]:
        """Get trade P&Ls as NumPy array."""
        ...
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ...
    def __repr__(self) -> str: ...

class TickBacktestEngine:
    """
    Tick-level (event-driven) backtesting engine.

    Processes each trade as a separate event, simulating realistic
    execution with latency and slippage.
    """

    def __init__(self, config: TickBacktestConfig) -> None: ...
    def run(
        self,
        timestamps: npt.NDArray[np.int64],
        prices: npt.NDArray[np.float32],
        volumes: npt.NDArray[np.float32],
        is_buyer_maker: npt.NDArray[np.bool_],
        signals: npt.NDArray[np.int8],
        timeframe_ms: int,
    ) -> TickBacktestResult:
        """
        Run tick-level backtest with pre-computed signals.

        Args:
            timestamps: Trade timestamps in milliseconds (int64)
            prices: Trade prices (float32)
            volumes: Trade volumes (float32)
            is_buyer_maker: True if buyer is maker (bool)
            signals: Trading signals (0=Hold, 1=Buy, 2=Sell, int8)
            timeframe_ms: Timeframe in milliseconds (e.g., 300000 for 5min)

        Returns:
            TickBacktestResult with performance metrics

        Example:
            >>> config = TickBacktestConfig(initial_capital=10000.0)
            >>> engine = TickBacktestEngine(config)
            >>> result = engine.run(timestamps, prices, volumes, is_buyer_maker, signals, 300000)
        """
        ...
    def __repr__(self) -> str: ...

# ============================================================================
# GPU TICK AGGREGATION
# ============================================================================

class AggregatedCandles:
    """
    Result of GPU tick aggregation into OHLCV candles.

    Properties:
        timestamps: Candle timestamps (milliseconds since epoch)
        open: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        volume: Total volumes
        num_trades: Number of trades per candle
        num_candles: Total number of candles
    """

    @property
    def timestamps(self) -> npt.NDArray[np.int64]: ...
    @property
    def open(self) -> npt.NDArray[np.float32]: ...
    @property
    def high(self) -> npt.NDArray[np.float32]: ...
    @property
    def low(self) -> npt.NDArray[np.float32]: ...
    @property
    def close(self) -> npt.NDArray[np.float32]: ...
    @property
    def volume(self) -> npt.NDArray[np.float32]: ...
    @property
    def num_trades(self) -> npt.NDArray[np.int32]: ...
    @property
    def num_candles(self) -> int: ...
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy access."""
        ...
    def __repr__(self) -> str: ...

class GpuTickAggregator:
    """
    GPU-accelerated tick aggregation.

    Aggregates high-frequency trade data into OHLCV candles using
    JIT-compiled CUDA kernels for maximum performance.
    """

    def __init__(self) -> None:
        """
        Create a new GPU tick aggregator.

        Raises:
            RuntimeError: If GPU initialization fails
        """
        ...
    def aggregate(
        self,
        timestamps: npt.NDArray[np.int64],
        prices: npt.NDArray[np.float32],
        volumes: npt.NDArray[np.float32],
        sides: npt.NDArray[np.int8],
        timeframe_ms: int,
    ) -> AggregatedCandles:
        """
        Aggregate tick data into OHLCV candles.

        Args:
            timestamps: Tick timestamps (milliseconds since epoch)
            prices: Tick prices
            volumes: Tick volumes
            sides: Tick sides (1 for buy, -1 for sell)
            timeframe_ms: Candle timeframe in milliseconds (e.g., 300000 for 5 minutes)

        Returns:
            AggregatedCandles object with OHLCV data

        Raises:
            RuntimeError: If GPU aggregation fails

        Example:
            >>> aggregator = GpuTickAggregator()
            >>> candles = aggregator.aggregate(timestamps, prices, volumes, sides, 300000)
        """
        ...
    def __repr__(self) -> str: ...

def gpu_available() -> bool:
    """
    Check if GPU is available.

    Returns:
        True if GPU acceleration is available, False otherwise
    """
    ...

def gpu_info() -> Dict[str, Any]:
    """
    Get GPU device information.

    Returns:
        Dictionary with GPU information:
        - device_id: GPU device ID (int)
        - cuda_version: CUDA version (str)
        - compute_capability: Compute capability (str)
        - async_allocator: Whether async allocator is enabled (bool)

    Raises:
        RuntimeError: If GPU is not available
    """
    ...

# ============================================================================
# GPU BATCH BACKTESTING
# ============================================================================

class BacktestResult:
    """
    Result from GPU batch backtesting.

    Attributes:
        strategy_id: Strategy ID (index in parameter list)
        sharpe_ratio: Sharpe ratio (annualized, risk-free rate = 0)
        max_drawdown: Maximum drawdown (negative percentage, e.g., -0.15 = -15%)
        win_rate: Win rate [0, 1] (e.g., 0.65 = 65% of trades profitable)
        total_return: Total return (percentage, e.g., 0.25 = +25%)
        final_equity: Final portfolio equity (e.g., 12500.0)
        num_trades: Number of trades executed
        profit_factor: Profit factor (gross profit / gross loss)
    """

    strategy_id: int
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_return: float
    final_equity: float
    num_trades: int
    profit_factor: float

    def __repr__(self) -> str: ...
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Python dictionary."""
        ...
    def get_param(self, name: str) -> Optional[float]:
        """Get parameter by name."""
        ...
    def param_names(self) -> List[str]:
        """Get all parameter names."""
        ...
    def fitness(self) -> float:
        """Fitness score for genetic algorithm (Sharpe with drawdown penalty)."""
        ...

def batch_backtest(
    strategy: Literal["rsi_crossover", "ma_crossover", "bollinger"],
    ohlcv: npt.NDArray[np.float64],
    parameters: List[List[float]],
    timestamps: Optional[npt.NDArray[np.int64]] = None,
    initial_capital: float = 10000.0,
    trading_fee: float = 0.001,
    slippage: float = 0.0001,
    execution_mode: Literal["auto", "traditional", "fused", "async"] = "auto",
) -> List[BacktestResult]:
    """
    GPU batch backtest for genetic algorithm optimization.

    Executes N strategies in parallel on GPU with single data transfer.
    Delivers 20-40x speedup vs sequential CPU backtesting.

    Args:
        strategy: Strategy name ('rsi_crossover', 'ma_crossover', 'bollinger')
        ohlcv: NumPy array (N_candles, 5) with columns [open, high, low, close, volume]
        parameters: List of parameter lists (e.g., [[14, 30, 70], [14, 25, 75], ...])
        timestamps: Optional Unix timestamps (nanoseconds). If None, generated as [0, 1, 2, ...]
        initial_capital: Starting portfolio value (default: 10000.0)
        trading_fee: Fee per trade as fraction (default: 0.001 = 0.1%)
        slippage: Slippage per trade as fraction (default: 0.0001 = 0.01%)
        execution_mode: Execution mode (default: 'auto')
            - 'auto': Automatically selects best mode based on workload size
            - 'traditional': 4 separate GPU kernels (best for <150 strategies)
            - 'fused': Single persistent kernel (best for 150-999 strategies, 1.88-4.00x faster)
            - 'async': Triple-buffered pipeline (best for >=1000 strategies, 1.2-1.4x faster than fused)

    Returns:
        List of BacktestResult objects, sorted by fitness (best first)

    Raises:
        ValueError: Invalid strategy name or parameter shape
        RuntimeError: GPU initialization failed or CUDA error

    Performance:
        - 1000 strategies × 10K candles: <250ms (RTX 3500 Ada)
        - Speedup: 20-40x vs sequential CPU
        - VRAM usage: <1GB for 1000 strategies

    Example:
        >>> parameters = [[14.0, 20.0 + i, 70.0 + i] for i in range(100)]
        >>> results = batch_backtest(
        ...     strategy='rsi_crossover',
        ...     ohlcv=ohlcv,
        ...     parameters=parameters,
        ...     execution_mode='auto'  # Automatically selects best mode
        ... )
        >>> best = results[0]
        >>> print(f"Sharpe: {best.sharpe_ratio:.2f}")
    """
    ...

def batch_backtest_info() -> Dict[str, Any]:
    """
    Get batch backtest performance info (GPU vs CPU comparison).

    Returns:
        Dictionary with:
        - gpu_available: bool
        - gpu_name: str (e.g., 'NVIDIA RTX 3500 Ada')
        - cuda_version: str (e.g., '13.0')
        - vram_gb: int (e.g., 12)
        - expected_speedup: float (e.g., 30.0 for 30x)
        - error: str (if GPU not available)

    Example:
        >>> info = batch_backtest_info()
        >>> if info['gpu_available']:
        ...     print(f"GPU: {info['gpu_name']}")
        ...     print(f"Expected speedup: {info['expected_speedup']:.1f}x")
    """
    ...

# ============================================================================
# GPU PARAMETER OPTIMIZERS (requires 'gpu' feature)
# ============================================================================

class GridSearchOptimizer:
    """
    GPU-accelerated Grid Search optimizer for strategy parameter tuning.

    Exhaustively evaluates all parameter combinations using GPU batch backtesting.
    Provides guaranteed global optimum with 40x speedup vs sequential CPU.

    Performance:
        - 1000 combinations × 10K candles: <3 seconds (40x vs sequential)
        - Accuracy: Match CPU within 0.01% tolerance
        - GPU Utilization: >90% via batch execution

    Example:
        >>> optimizer = kimsfinance_core.GridSearchOptimizer(batch_size=1000)
        >>> param_ranges = {
        ...     'rsi_period': {'min': 10.0, 'max': 20.0, 'step': 2.0},
        ...     'buy_threshold': {'min': 20.0, 'max': 40.0, 'step': 5.0}
        ... }
        >>> result = optimizer.optimize(
        ...     timestamps=timestamps,
        ...     open=open_prices,
        ...     high=high,
        ...     low=low,
        ...     close=close,
        ...     volume=volume,
        ...     param_ranges=param_ranges,
        ...     strategy_type='RSI'
        ... )
        >>> print(f"Best Sharpe: {result.best_sharpe:.2f}")
    """

    def __init__(self, batch_size: int = 500) -> None:
        """
        Create new Grid Search optimizer.

        Args:
            batch_size: Number of parameter sets per GPU batch (default: 500)
                - 100: Safe for 4GB VRAM
                - 500: Optimal for 8-12GB VRAM (RTX 3500 Ada)
                - 1000: For 16GB+ VRAM or small datasets
        """
        ...

    def optimize(
        self,
        timestamps: npt.NDArray[np.int64],
        open: npt.NDArray[np.float64],
        high: npt.NDArray[np.float64],
        low: npt.NDArray[np.float64],
        close: npt.NDArray[np.float64],
        volume: npt.NDArray[np.float64],
        param_ranges: Dict[str, Dict[str, float]],
        strategy_type: Literal["RSI", "SMA_CROSS", "MACD", "BOLLINGER"],
        initial_capital: float = 10000.0,
        trading_fee: float = 0.001,
        slippage: float = 0.0005,
    ) -> "GridSearchResult":
        """
        Run grid search optimization.

        Exhaustively evaluates all parameter combinations from the grid.

        Args:
            timestamps: Unix timestamps (nanoseconds, int64)
            open, high, low, close, volume: OHLCV data (float64)
            param_ranges: Dictionary of parameter ranges:
                {
                    'param_name': {'min': 10.0, 'max': 20.0, 'step': 2.0},
                    # OR for discrete values:
                    'param_name': {'values': [1.0, 2.0, 5.0, 10.0]}
                }
            strategy_type: Strategy to optimize ('RSI', 'SMA_CROSS', 'MACD', 'BOLLINGER')
            initial_capital: Starting capital (default: 10000.0)
            trading_fee: Fee per trade as fraction (default: 0.001 = 0.1%)
            slippage: Slippage per trade as fraction (default: 0.0005 = 0.05%)

        Returns:
            GridSearchResult with best parameters, fitness, and convergence history

        Raises:
            ValueError: Invalid parameter ranges or strategy type
            RuntimeError: GPU initialization failed or CUDA error
        """
        ...

    def __repr__(self) -> str: ...

class GridSearchResult:
    """
    Grid Search optimization result.

    Attributes:
        best_parameters: Dictionary of best parameter values
        best_fitness: Fitness score (Sharpe with drawdown penalty)
        best_sharpe: Sharpe ratio of best parameters
        best_drawdown: Max drawdown of best parameters
        total_combinations: Total number of combinations evaluated
    """

    best_parameters: Dict[str, float]
    best_fitness: float
    best_sharpe: float
    best_drawdown: float
    total_combinations: int

    def convergence_history(self) -> npt.NDArray[np.float64]:
        """Get convergence history as NumPy array (best fitness per batch)."""
        ...

    def __repr__(self) -> str: ...

class EulerSearchOptimizer:
    """
    GPU-accelerated Euler Search optimizer for strategy parameter tuning.

    Implements QuantConnect's iterative grid refinement algorithm with GPU batch evaluation.
    Achieves 90% fewer evaluations than exhaustive grid search while converging to near-optimal.

    Algorithm:
        1. Test Grid: Evaluate N points across current search space
        2. Find Best: Identify parameter set with highest fitness
        3. Refine: Reduce step size and narrow boundaries around best
        4. Repeat: Until step size falls below minimum threshold

    Performance:
        - Evaluations: 90% fewer than exhaustive grid search
        - Convergence: Typical 5-10 iterations
        - GPU Batch: <250ms per iteration (1000 params)
        - Target: Sub-second optimization for 3-parameter strategies

    Example:
        >>> optimizer = kimsfinance_core.EulerSearchOptimizer(
        ...     segment_amount=4,
        ...     max_iterations=15,
        ...     batch_size=1000
        ... )
        >>> optimizer.add_parameter('rsi_period', 5.0, 30.0, 5.0, 1.0)
        >>> optimizer.add_parameter('buy_threshold', 20.0, 40.0, 5.0, 1.0)
        >>> result = optimizer.optimize(
        ...     timestamps=timestamps,
        ...     open=open_prices,
        ...     high=high,
        ...     low=low,
        ...     close=close,
        ...     volume=volume,
        ...     strategy_type='RSI'
        ... )
        >>> print(f"Converged in {result.iterations} iterations")
    """

    def __init__(
        self,
        segment_amount: int = 4,
        max_iterations: int = 20,
        batch_size: int = 1000,
    ) -> None:
        """
        Create new Euler Search optimizer.

        Args:
            segment_amount: Grid resolution per iteration (default: 4, QuantConnect default)
                - Higher values = finer grids, slower convergence
                - Lower values = coarser grids, faster convergence
            max_iterations: Maximum iterations before forced stop (default: 20)
            batch_size: GPU batch size (default: 1000)
        """
        ...

    def add_parameter(
        self,
        name: str,
        min_value: float,
        max_value: float,
        initial_step: float,
        min_step: float,
    ) -> None:
        """
        Add parameter to optimize.

        Args:
            name: Parameter name (e.g., 'rsi_period')
            min_value: Initial minimum value
            max_value: Initial maximum value
            initial_step: Initial step size
            min_step: Minimum step size (convergence threshold)

        Example:
            >>> optimizer.add_parameter('rsi_period', 5.0, 30.0, 5.0, 1.0)
            >>> optimizer.add_parameter('buy_threshold', 20.0, 40.0, 5.0, 1.0)
        """
        ...

    def optimize(
        self,
        timestamps: npt.NDArray[np.int64],
        open: npt.NDArray[np.float64],
        high: npt.NDArray[np.float64],
        low: npt.NDArray[np.float64],
        close: npt.NDArray[np.float64],
        volume: npt.NDArray[np.float64],
        strategy_type: Literal["RSI", "SMA_CROSS", "MACD", "BOLLINGER"],
        initial_capital: float = 10000.0,
        trading_fee: float = 0.001,
        slippage: float = 0.0005,
    ) -> "EulerSearchResult":
        """
        Run Euler Search optimization.

        Args:
            timestamps: Unix timestamps (nanoseconds, int64)
            open, high, low, close, volume: OHLCV data (float64)
            strategy_type: Strategy to optimize ('RSI', 'SMA_CROSS', 'MACD', 'BOLLINGER')
            initial_capital: Starting capital (default: 10000.0)
            trading_fee: Fee per trade as fraction (default: 0.001 = 0.1%)
            slippage: Slippage per trade as fraction (default: 0.0005 = 0.05%)

        Returns:
            EulerSearchResult with best parameters, convergence history, and refinement details

        Raises:
            ValueError: No parameters defined or invalid strategy type
            RuntimeError: GPU initialization failed or CUDA error
        """
        ...

    def __repr__(self) -> str: ...

class EulerSearchResult:
    """
    Euler Search optimization result.

    Attributes:
        best_parameters: Dictionary of best parameter values
        best_fitness: Fitness score (Sharpe with drawdown penalty)
        iterations: Number of iterations until convergence
        total_evaluations: Total parameter sets evaluated
        total_gpu_time_ms: Total GPU computation time (milliseconds)
        total_time_ms: Total wall-clock time (milliseconds)
    """

    best_parameters: Dict[str, float]
    best_fitness: float
    iterations: int
    total_evaluations: int
    total_gpu_time_ms: float
    total_time_ms: float

    def convergence_history(self) -> npt.NDArray[np.float64]:
        """Get convergence history as NumPy array (best fitness per iteration)."""
        ...

    def is_converged(self) -> bool:
        """
        Check if optimization converged to optimum.

        Returns True if final improvement was < 1% over 3 iterations.
        """
        ...

    def grid_search_speedup(self, grid_points_per_param: int = 10) -> float:
        """
        Calculate speedup vs exhaustive grid search.

        Args:
            grid_points_per_param: Number of grid points per parameter (default: 10)

        Returns:
            Estimated speedup factor (e.g., 5.0 = 5x faster than grid search)

        Example:
            >>> speedup = result.grid_search_speedup(grid_points_per_param=10)
            >>> print(f"Euler Search was {speedup:.1f}x faster")
        """
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# PARQUET DATA LOADER (requires 'data-downloaders' feature)
# ============================================================================

def load_parquet_file(parquet_path: str) -> List[Dict[str, Any]]:
    """
    Load tick data from a single Parquet file (zero-copy Arrow-based).

    Args:
        parquet_path: Path to Parquet file (e.g., "BTCUSDT-trades-2024-01-01.parquet")

    Returns:
        List of dictionaries with keys: id, price, qty, quote_qty, time, is_buyer_maker

    Performance:
        10-20M records/sec (zero-copy Arrow-based loader)

    Example:
        >>> trades = load_parquet_file(
        ...     "/data/trades_parquet/2024-01/BTCUSDT-trades-2024-01-01.parquet"
        ... )
        >>> print(f"Loaded {len(trades)} trades")
    """
    ...

def load_parquet_month(
    month_dir: str,
    max_trades: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load all tick data from a month directory.

    Loads and concatenates all Parquet files in a month directory.
    Files are sorted by name to ensure chronological order.

    Args:
        month_dir: Path to month directory (e.g., "/data/trades_parquet/2024-01")
        max_trades: Optional limit on number of trades to load (None = all)

    Returns:
        List of dictionaries with keys: id, price, qty, quote_qty, time, is_buyer_maker

    Example:
        >>> # Load full month
        >>> trades = load_parquet_month("/data/trades_parquet/2024-01")
        >>>
        >>> # Load first 1M trades only (for testing)
        >>> trades = load_parquet_month("/data/trades_parquet/2024-01", max_trades=1_000_000)
    """
    ...
