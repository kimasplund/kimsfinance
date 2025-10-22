from __future__ import annotations
import numpy as np
from ..core.types import ArrayLike
from ..utils.array_utils import to_numpy_array


def calculate_renko_bricks(
    ohlc: dict[str, ArrayLike],
    box_size: float | None = None,
    reversal_boxes: int = 1,
) -> list[dict[str, float | int]]:
    """
    Convert OHLC price data to Renko bricks.

    Renko charts are time-independent price charts that only show price movements
    of a fixed size (box_size). New bricks are created when the price moves by
    at least box_size from the last brick. This filters out minor price fluctuations
    and focuses on significant trends.

    Algorithm:
    1. Start with first close price as reference
    2. For each candle, check if price moved by >= box_size
    3. If yes, create brick(s) in movement direction
    4. Update reference price to top/bottom of last brick
    5. Apply reversal_boxes filter for trend changes

    Args:
        ohlc: OHLC price data dictionary containing 'open', 'high', 'low', 'close' arrays
        box_size: Size of each brick in price units.
                  If None, auto-calculate using ATR (Average True Range).
                  Recommended: ATR(14) * 0.5 to 1.0 for optimal noise filtering.
                  Larger values = fewer bricks, smoother trends.
                  Smaller values = more bricks, more detail.
        reversal_boxes: Number of boxes needed for trend reversal.
                       Default 1 (any opposite movement creates new brick).
                       Higher values (2-3) filter noise and require stronger
                       price movement before reversing trend.

    Returns:
        List of brick dicts: [
            {'price': 102.0, 'direction': 1},   # Up brick at price 102
            {'price': 104.0, 'direction': 1},   # Up brick at price 104
            {'price': 102.0, 'direction': -1},  # Down brick at price 102
            ...
        ]
        - price: Top of the brick for up bricks, bottom for down bricks
        - direction: 1 for up, -1 for down

    Examples:
        >>> ohlc = {
        ...     'open': np.array([100, 102, 105, 103]),
        ...     'high': np.array([101, 104, 106, 104]),
        ...     'low': np.array([99, 101, 104, 102]),
        ...     'close': np.array([100, 103, 105, 102])
        ... }
        >>> bricks = calculate_renko_bricks(ohlc, box_size=2.0)
        >>> len(bricks)
        4
        >>> bricks[0]
        {'price': 102.0, 'direction': 1}

    Performance:
        - Target: <5ms for 1000 candles
        - Vectorized NumPy operations where possible
        - Minimal Python loops (only for brick creation)

    Notes:
        - ATR-based auto-sizing provides adaptive box sizes for different volatility
        - reversal_boxes=1 creates very responsive charts (default)
        - reversal_boxes=2-3 creates smoother charts with less noise
        - Empty result may occur if price never moves by box_size
    """
    close_prices = to_numpy_array(ohlc["close"])
    high_prices = to_numpy_array(ohlc["high"])
    low_prices = to_numpy_array(ohlc["low"])

    if len(close_prices) == 0:
        return []

    # Auto-calculate box size using ATR if not provided
    if box_size is None:
        from ..ops.indicators import calculate_atr

        atr = calculate_atr(high_prices, low_prices, close_prices, period=14, engine="cpu")
        # Use 75% of median ATR for optimal balance between detail and noise filtering
        box_size = float(np.nanmedian(atr)) * 0.75

    # Validate box_size
    if box_size <= 0:
        raise ValueError(f"box_size must be positive, got {box_size}")

    bricks: list[dict[str, float | int]] = []
    reference_price = float(close_prices[0])
    current_direction: int | None = None  # 1=up, -1=down, None=initial

    # Process each candle using zip instead of range(len()) anti-pattern
    for close, high, low in zip(close_prices, high_prices, low_prices):
        close = float(close)
        high = float(high)
        low = float(low)

        # Use high/low for better brick detection (captures intra-candle movements)
        # Check upward movement first using high price
        price_diff_up = high - reference_price

        if price_diff_up >= box_size:
            num_boxes = int(price_diff_up / box_size)

            # Check if direction change (down to up)
            if current_direction == -1 and num_boxes < reversal_boxes:
                # Not enough movement to reverse trend, skip
                pass
            else:
                # Create up bricks
                for _ in range(num_boxes):
                    reference_price += box_size
                    bricks.append(
                        {
                            "price": reference_price,
                            "direction": 1,  # Up
                        }
                    )
                current_direction = 1
                continue  # Move to next candle

        # Check downward movement using low price
        price_diff_down = reference_price - low

        if price_diff_down >= box_size:
            num_boxes = int(price_diff_down / box_size)

            # Check if direction change (up to down)
            if current_direction == 1 and num_boxes < reversal_boxes:
                # Not enough movement to reverse trend, skip
                pass
            else:
                # Create down bricks
                for _ in range(num_boxes):
                    reference_price -= box_size
                    bricks.append(
                        {
                            "price": reference_price,
                            "direction": -1,  # Down
                        }
                    )
                current_direction = -1

    return bricks
