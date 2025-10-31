#!/usr/bin/env python3
"""
Daily Historical Options Data Collector

Collects end-of-day options data for building a historical database.
Designed to run daily at 4:15 PM ET (after market close) via cron.

Storage: data/yfinance/options_historical/{symbol}/{date}.parquet

Usage:
    python scripts/download_options_daily_historical.py

Cron setup (4:15 PM ET Mon-Fri):
    15 16 * * 1-5 cd /home/kim-asplund/projects/kimsfinance/rust && source ../.venv/bin/activate && python scripts/download_options_daily_historical.py >> logs/options_daily.log 2>&1

Requirements:
    pip install yfinance pandas pyarrow
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import datetime, timezone
import time
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DailyOptionsCollector:
    """Collects EOD options data for historical database"""

    def __init__(self, base_dir: str = "data/yfinance/options_historical"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_backoff = 120  # Maximum backoff in seconds

        # Get today's date in market timezone (assume ET for US markets)
        self.snapshot_date = datetime.now().strftime("%Y-%m-%d")

    def download_symbol_options(
        self,
        symbol: str,
        max_expirations: Optional[int] = None
    ) -> Tuple[str, bool, str]:
        """
        Download all option expirations for a symbol and save to daily file

        Returns:
            Tuple of (symbol, success, message)
        """
        logger.info(f"Starting download for {symbol}")

        # Check if already collected today
        symbol_dir = self.base_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        output_file = symbol_dir / f"{self.snapshot_date}.parquet"

        if output_file.exists():
            logger.info(f"{symbol}: Already collected today, skipping")
            return (symbol, True, "Already collected today")

        retry_count = 0

        while True:
            try:
                # Get ticker
                ticker = yf.Ticker(symbol)

                # Get available expirations
                expirations = ticker.options

                if not expirations:
                    logger.warning(f"{symbol}: No options available")
                    return (symbol, False, "No options available")

                # Limit expirations if specified
                if max_expirations:
                    expirations = expirations[:max_expirations]

                logger.info(f"{symbol}: Found {len(expirations)} expirations")

                # Collect all options for all expirations
                all_options = []
                success_count = 0
                error_count = 0

                for i, expiration in enumerate(expirations, 1):
                    try:
                        logger.info(f"{symbol}: Downloading {i}/{len(expirations)} - {expiration}")

                        # Get options chain
                        opt = ticker.option_chain(expiration)

                        # Combine calls and puts
                        calls = opt.calls.copy()
                        calls['optionType'] = 'call'
                        puts = opt.puts.copy()
                        puts['optionType'] = 'put'

                        options_df = pd.concat([calls, puts], ignore_index=True)
                        options_df['expiration'] = expiration
                        options_df['symbol'] = symbol
                        options_df['snapshotDate'] = self.snapshot_date
                        options_df['downloadTime'] = datetime.now(timezone.utc).isoformat()

                        all_options.append(options_df)

                        logger.info(
                            f"{symbol}: ✓ {expiration} - "
                            f"{len(calls)} calls, {len(puts)} puts"
                        )
                        success_count += 1

                        # Small delay between expirations
                        time.sleep(0.5)

                    except Exception as e:
                        logger.error(f"{symbol}: ✗ {expiration} - {e}")
                        error_count += 1
                        continue

                if not all_options:
                    logger.error(f"{symbol}: No options data collected")
                    return (symbol, False, "No options data collected")

                # Combine all expirations into single DataFrame
                combined_df = pd.concat(all_options, ignore_index=True)

                # Save to daily file
                combined_df.to_parquet(output_file, index=False)

                logger.info(
                    f"{symbol}: ✓ Saved {len(combined_df)} total options to {output_file.name}"
                )
                logger.info(
                    f"{symbol}: Complete - {success_count} expirations, {error_count} errors"
                )

                return (symbol, True, f"{len(combined_df)} options from {success_count} expirations")

            except Exception as e:
                retry_count += 1

                # Exponential backoff: 2, 4, 8, 16, 32, 64, 120, 120, 120...
                delay = min(2 ** retry_count, self.max_backoff)

                if delay < self.max_backoff:
                    logger.warning(
                        f"{symbol}: Error (attempt {retry_count}). "
                        f"Exponential backoff: {delay}s. Error: {e}"
                    )
                else:
                    logger.warning(
                        f"{symbol}: Error (attempt {retry_count}). "
                        f"Steady retry: {delay}s. Error: {e}"
                    )

                time.sleep(delay)


def download_worker(args: Tuple[str, Optional[int]]) -> Tuple[str, bool, str]:
    """Worker function for parallel downloading"""
    symbol, max_expirations = args
    collector = DailyOptionsCollector()
    return collector.download_symbol_options(symbol, max_expirations)


def main():
    """Main entry point"""
    # Symbols to collect
    symbols = [
        "AAPL",
        "MSFT",
        "SPY",
        "QQQ",
        "TSLA",
        "NVDA",
        "AMZN",
        "GOOGL",
    ]

    # Collect up to 20 expirations per symbol (can be adjusted)
    max_expirations = 20

    # Number of parallel workers (use half of available CPUs to be nice)
    num_workers = max(1, cpu_count() // 2)

    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"=== Daily Historical Options Collector ===")
    logger.info(f"Snapshot date: {snapshot_date}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Max expirations per symbol: {max_expirations}")
    logger.info(f"Parallel workers: {num_workers}")
    logger.info(f"Output: data/yfinance/options_historical/")
    logger.info("")

    start_time = time.time()

    # Prepare arguments for parallel processing
    args_list = [(symbol, max_expirations) for symbol in symbols]

    # Download in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(download_worker, args_list)

    # Print summary
    elapsed = time.time() - start_time
    total_success = sum(1 for r in results if r[1])
    total_errors = len(results) - total_success

    logger.info("")
    logger.info("=== Collection Complete ===")
    logger.info(f"Time: {elapsed:.1f} seconds")
    logger.info(f"Successful collections: {total_success}/{len(symbols)}")
    logger.info(f"Failed collections: {total_errors}")
    logger.info("")
    logger.info("Results by symbol:")
    for symbol, success, message in results:
        status = "✓" if success else "✗"
        logger.info(f"  {status} {symbol}: {message}")

    logger.info("")
    logger.info(f"Data saved to: {Path('data/yfinance/options_historical/').absolute()}")
    logger.info("")
    logger.info("File structure:")
    logger.info(f"  data/yfinance/options_historical/")
    logger.info(f"    ├── AAPL/{snapshot_date}.parquet")
    logger.info(f"    ├── MSFT/{snapshot_date}.parquet")
    logger.info(f"    └── ...")


if __name__ == "__main__":
    main()
