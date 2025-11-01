#!/usr/bin/env python3
"""
Demonstration of tick-level backtesting using Parquet trades data

This script shows how to:
1. Read partitioned Parquet trades data efficiently
2. Process trades sequentially (tick-by-tick)
3. Implement a simple strategy without OHLCV aggregation
4. Measure realistic execution with actual trade prices

Example:
    python demo_tick_backtest.py /tmp/test_trades_parquet 2021-01
"""

import polars as pl
from pathlib import Path
from datetime import datetime
import argparse


class SimpleTickStrategy:
    """
    Example tick-level strategy: Buy on dips, sell on rallies

    Strategy:
    - Track last N trades' prices
    - Buy if price drops below moving average
    - Sell if price rises above moving average
    """

    def __init__(self, window_size: int = 100, position_size: float = 0.1):
        self.window_size = window_size
        self.position_size = position_size
        self.price_history = []
        self.position = 0.0
        self.entry_price = None
        self.trades_executed = 0
        self.pnl = 0.0

    def on_trade(self, price: float, qty: float, side: str) -> dict:
        """
        Process each tick (trade) and return action

        Args:
            price: Trade price
            qty: Trade quantity
            side: "buy" or "sell"

        Returns:
            dict with action: "buy", "sell", or "hold"
        """
        # Update price history
        self.price_history.append(price)
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)

        # Need enough history to calculate MA
        if len(self.price_history) < self.window_size:
            return {"action": "hold"}

        # Calculate simple moving average
        ma = sum(self.price_history) / len(self.price_history)

        # Strategy logic
        action = {"action": "hold"}

        # Buy signal: price below MA and not in position
        if price < ma * 0.999 and self.position == 0:
            self.position = self.position_size
            self.entry_price = price
            self.trades_executed += 1
            action = {
                "action": "buy",
                "price": price,
                "qty": self.position_size,
                "ma": ma
            }

        # Sell signal: price above MA and in position
        elif price > ma * 1.001 and self.position > 0:
            self.pnl += self.position * (price - self.entry_price)
            self.position = 0
            self.entry_price = None
            self.trades_executed += 1
            action = {
                "action": "sell",
                "price": price,
                "qty": self.position_size,
                "ma": ma,
                "pnl": self.pnl
            }

        return action


def run_tick_backtest(
    parquet_dir: Path,
    month: str,
    strategy: SimpleTickStrategy,
    max_ticks: int = None
):
    """
    Run tick-level backtest on Parquet trades data

    Args:
        parquet_dir: Directory containing month-partitioned Parquet files
        month: Month to backtest (e.g., "2021-01")
        strategy: Strategy instance with on_trade() method
        max_ticks: Optional limit on number of ticks to process
    """
    month_dir = parquet_dir / month

    if not month_dir.exists():
        raise ValueError(f"Month directory not found: {month_dir}")

    # Find all Parquet files for this month
    parquet_files = sorted(month_dir.glob("*.parquet"))

    if not parquet_files:
        raise ValueError(f"No Parquet files found in {month_dir}")

    print(f"Backtesting month: {month}")
    print(f"Parquet files: {len(parquet_files)}")
    print()

    # Read all trades for the month (streaming)
    # Note: For very large months, you might want to process in chunks
    df = pl.read_parquet(parquet_files[0])

    if max_ticks:
        df = df.head(max_ticks)

    print(f"Total ticks to process: {len(df):,}")
    print()

    # Process each tick sequentially
    start_time = datetime.now()
    ticks_processed = 0
    last_report = 0

    for row in df.iter_rows(named=True):
        price = row["price"]
        qty = row["qty"]
        side = row["side"]

        # Execute strategy on this tick
        action = strategy.on_trade(price, qty, side)

        # Report significant actions
        if action["action"] in ["buy", "sell"]:
            print(f"[{row['timestamp']}] {action['action'].upper():4s} @ ${price:,.2f} | MA: ${action['ma']:,.2f} | PnL: ${strategy.pnl:,.2f}")

        ticks_processed += 1

        # Progress report every 1M ticks
        if ticks_processed - last_report >= 1_000_000:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = ticks_processed / elapsed
            print(f"  ... {ticks_processed:,} ticks processed ({rate:,.0f} ticks/sec)")
            last_report = ticks_processed

    # Final report
    elapsed = (datetime.now() - start_time).total_seconds()
    rate = ticks_processed / elapsed

    print()
    print("=" * 80)
    print("Backtest Complete")
    print("=" * 80)
    print(f"Month: {month}")
    print(f"Ticks processed: {ticks_processed:,}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Processing rate: {rate:,.0f} ticks/sec")
    print()
    print(f"Trades executed: {strategy.trades_executed}")
    print(f"Final position: {strategy.position} BTC")
    print(f"Total PnL: ${strategy.pnl:,.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Tick-level backtesting demo using Parquet trades data"
    )

    parser.add_argument(
        "parquet_dir",
        type=Path,
        help="Directory containing month-partitioned Parquet files"
    )

    parser.add_argument(
        "month",
        type=str,
        help="Month to backtest (e.g., '2021-01')"
    )

    parser.add_argument(
        "--max-ticks",
        type=int,
        default=1_000_000,
        help="Maximum number of ticks to process (default: 1M for demo)"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving average window size (default: 100)"
    )

    args = parser.parse_args()

    # Create strategy
    strategy = SimpleTickStrategy(window_size=args.window)

    # Run backtest
    run_tick_backtest(
        args.parquet_dir,
        args.month,
        strategy,
        max_ticks=args.max_ticks
    )


if __name__ == "__main__":
    main()
