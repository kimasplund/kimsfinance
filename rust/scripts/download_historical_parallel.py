#!/usr/bin/env python3
"""
Parallel Historical Stock Data Downloader using yfinance

Downloads 2 years of daily historical data for multiple symbols in parallel.
Automatically handles rate limiting with exponential backoff up to 120 seconds.

Usage:
    python scripts/download_historical_parallel.py

Requirements:
    pip install yfinance pandas pyarrow
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta
import time
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDownloader:
    """Downloads historical stock data with automatic retry and exponential backoff"""

    def __init__(self, base_dir: str = "data/yfinance/ohlcv"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_backoff = 120  # Maximum backoff in seconds

    def download_symbol_historical(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Tuple[str, bool, str]:
        """
        Download historical data for a symbol

        Returns:
            Tuple of (symbol, success, message)
        """
        logger.info(f"Starting download for {symbol}")
        retry_count = 0

        while True:
            try:
                # Download data using yfinance
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval="1d")

                if df.empty:
                    logger.warning(f"{symbol}: No data available")
                    return (symbol, False, "No data available")

                # Add metadata
                df['symbol'] = symbol
                df['downloadTime'] = datetime.now().isoformat()

                # Reset index to make Date a column
                df = df.reset_index()

                # Save to parquet (one file per symbol in ohlcv directory)
                output_file = self.base_dir / f"{symbol}.parquet"
                df.to_parquet(output_file, index=False)

                logger.info(
                    f"{symbol}: ✓ SUCCESS - {len(df)} days "
                    f"({df['Date'].min().date()} to {df['Date'].max().date()})"
                )
                return (symbol, True, f"{len(df)} days downloaded")

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


def download_worker(args: Tuple[str, str, str]) -> Tuple[str, bool, str]:
    """Worker function for parallel downloading"""
    symbol, start_date, end_date = args
    downloader = HistoricalDownloader()
    return downloader.download_symbol_historical(symbol, start_date, end_date)


def main():
    """Main entry point"""
    # Symbols to download (required for options strategy)
    symbols = [
        "AAPL",
        "SPY",
        "TSLA",
        "QQQ",
    ]

    # Full date range for options strategy backtesting
    start_str = "2016-01-01"
    end_str = "2025-10-30"

    # Number of parallel workers (use half of available CPUs to be nice)
    num_workers = max(1, cpu_count() // 2)

    logger.info(f"=== Parallel Historical Data Downloader ===")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Date range: {start_str} to {end_str} (9+ years)")
    logger.info(f"Parallel workers: {num_workers}")
    logger.info(f"Output: data/yfinance/ohlcv/")
    logger.info("")

    start_time = time.time()

    # Prepare arguments for parallel processing
    args_list = [(symbol, start_str, end_str) for symbol in symbols]

    # Download in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(download_worker, args_list)

    # Print summary
    elapsed = time.time() - start_time
    total_success = sum(1 for r in results if r[1])
    total_errors = len(results) - total_success

    logger.info("")
    logger.info("=== Download Complete ===")
    logger.info(f"Time: {elapsed:.1f} seconds")
    logger.info(f"Successful downloads: {total_success}/{len(symbols)}")
    logger.info(f"Failed downloads: {total_errors}")
    logger.info("")
    logger.info("Results by symbol:")
    for symbol, success, message in results:
        status = "✓" if success else "✗"
        logger.info(f"  {status} {symbol}: {message}")

    logger.info("")
    logger.info(f"Data saved to: {Path('data/yfinance/ohlcv/').absolute()}")


if __name__ == "__main__":
    main()
