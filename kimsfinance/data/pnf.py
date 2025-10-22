from __future__ import annotations
import numpy as np
from ..core.types import ArrayLike
from ..utils.array_utils import to_numpy_array


def calculate_pnf_columns(
    ohlc: dict[str, ArrayLike],
    box_size: float | None = None,
    reversal_boxes: int = 3,
) -> list[dict]:
    """
    Convert OHLC price data to Point and Figure columns.

    Algorithm:
    1. Start with first close price, determine direction from second candle
    2. For each candle, check high for X boxes, low for O boxes
    3. If current column continues: add boxes
    4. If reversal detected (opposite direction >= reversal_boxes): start new column
    5. Ignore moves that don't meet box size threshold

    Args:
        ohlc: OHLC price data dictionary
        box_size: Price per box. If None, auto-calculate using ATR.
                  Typical: ATR(14) * 0.5 to 2.0
        reversal_boxes: Number of boxes needed for trend reversal.
                       Default 3 (traditional PNF standard).
                       Higher = fewer columns, smoother trend.

    Returns:
        List of column dicts: [
            {
                'type': 'X',  # or 'O'
                'boxes': [102, 104, 106, 108],  # Prices for each box
                'start_idx': 0,  # Index in original data where column started
            },
            {
                'type': 'O',
                'boxes': [106, 104, 102, 100],
                'start_idx': 10,
            },
            ...
        ]

    Notes:
        - Uses high/low prices, not just close
        - More accurate than close-based algorithms
        - Returns empty list if insufficient price movement
    """
    high_prices = to_numpy_array(ohlc["high"])
    low_prices = to_numpy_array(ohlc["low"])
    close_prices = to_numpy_array(ohlc["close"])

    # Auto-calculate box size using ATR
    if box_size is None:
        # Use ATR if we have enough data, otherwise use price range
        if len(close_prices) >= 14:
            from ..ops.indicators import calculate_atr

            atr = calculate_atr(ohlc["high"], ohlc["low"], ohlc["close"], period=14, engine="cpu")
            box_size = float(np.nanmedian(atr))  # Use median ATR
        else:
            # Fallback for small datasets: use 1% of price range
            price_range = float(np.max(high_prices) - np.min(low_prices))
            box_size = price_range * 0.01 if price_range > 0 else 1.0

    columns: list[dict] = []
    current_column: dict | None = None
    reference_price = close_prices[0]

    # Round reference price to nearest box
    reference_price = round(reference_price / box_size) * box_size

    for i in range(len(close_prices)):
        high = high_prices[i]
        low = low_prices[i]

        # How many boxes can we go up from reference?
        boxes_up = int((high - reference_price) / box_size)

        # How many boxes can we go down from reference?
        boxes_down = int((reference_price - low) / box_size)

        # Current column is rising (X column) or not yet started
        if current_column is None or current_column["type"] == "X":
            # Try to add X boxes
            if boxes_up > 0:
                if current_column is None:
                    current_column = {"type": "X", "boxes": [], "start_idx": i}

                # Add X boxes
                for j in range(boxes_up):
                    reference_price += box_size
                    current_column["boxes"].append(reference_price)

            # Check for reversal to O
            elif boxes_down >= reversal_boxes:
                # Save current X column
                if current_column and current_column["boxes"]:
                    columns.append(current_column)

                # Start new O column
                current_column = {"type": "O", "boxes": [], "start_idx": i}

                # Add O boxes (going down from previous high)
                # Go back to top of last X column
                if columns:
                    reference_price = (
                        columns[-1]["boxes"][-1] if columns[-1]["boxes"] else reference_price
                    )

                for j in range(boxes_down):
                    reference_price -= box_size
                    current_column["boxes"].append(reference_price)

        # Current column is falling (O column)
        elif current_column["type"] == "O":
            # Try to add O boxes
            if boxes_down > 0:
                for j in range(boxes_down):
                    reference_price -= box_size
                    current_column["boxes"].append(reference_price)

            # Check for reversal to X
            elif boxes_up >= reversal_boxes:
                # Save current O column
                if current_column["boxes"]:
                    columns.append(current_column)

                # Start new X column
                current_column = {"type": "X", "boxes": [], "start_idx": i}

                # Go back to bottom of last O column
                if columns:
                    reference_price = (
                        columns[-1]["boxes"][-1] if columns[-1]["boxes"] else reference_price
                    )

                for j in range(boxes_up):
                    reference_price += box_size
                    current_column["boxes"].append(reference_price)

    # Add final column
    if current_column and current_column["boxes"]:
        columns.append(current_column)

    return columns
