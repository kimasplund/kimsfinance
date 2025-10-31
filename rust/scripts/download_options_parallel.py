#!/usr/bin/env python3
"""
Parallel Options Data Downloader using yfinance

Downloads options data for multiple symbols in parallel using multiprocessing.
Automatically handles rate limiting with exponential backoff up to 120 seconds.

Usage:
    python scripts/download_options_parallel.py

Requirements:
    pip install yfinance pandas pyarrow
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import datetime
import time
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptionsDownloader:
    """Downloads options data with automatic retry and exponential backoff"""

    def __init__(self, base_dir: str = "data/yfinance/options"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_backoff = 120  # Maximum backoff in seconds

    def download_symbol_options(
        self,
        symbol: str,
        max_expirations: Optional[int] = None
    ) -> Tuple[str, int, int]:
        """
        Download all option expirations for a symbol

        Returns:
            Tuple of (symbol, success_count, error_count)
        """
        logger.info(f"Starting download for {symbol}")
        retry_count = 0
        success_count = 0
        error_count = 0

        while True:
            try:
                # Get ticker
                ticker = yf.Ticker(symbol)

                # Get available expirations
                expirations = ticker.options

                if not expirations:
                    logger.warning(f"{symbol}: No options available")
                    return (symbol, 0, 1)

                # Limit expirations if specified
                if max_expirations:
                    expirations = expirations[:max_expirations]

                logger.info(f"{symbol}: Found {len(expirations)} expirations")

                # Create symbol directory
                symbol_dir = self.base_dir / symbol
                symbol_dir.mkdir(parents=True, exist_ok=True)

                # Download each expiration
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
                        options_df['downloadTime'] = datetime.now().isoformat()

                        # Save to parquet
                        output_file = symbol_dir / f"{expiration}_options.parquet"
                        options_df.to_parquet(output_file, index=False)

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

                logger.info(
                    f"{symbol}: Complete - {success_count} success, {error_count} errors"
                )
                return (symbol, success_count, error_count)

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


def download_worker(args: Tuple[str, Optional[int]]) -> Tuple[str, int, int]:
    """Worker function for parallel downloading"""
    symbol, max_expirations = args
    downloader = OptionsDownloader()
    return downloader.download_symbol_options(symbol, max_expirations)


def main():
    """Main entry point"""
    # Symbols to download
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

    # Limit to 20 expirations per symbol to avoid overwhelming
    max_expirations = 20

    # Number of parallel workers (use half of available CPUs to be nice)
    num_workers = max(1, cpu_count() // 2)

    logger.info(f"=== Parallel Options Downloader ===")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Max expirations per symbol: {max_expirations}")
    logger.info(f"Parallel workers: {num_workers}")
    logger.info(f"Output: data/yfinance/options/")
    logger.info("")

    start_time = time.time()

    # Prepare arguments for parallel processing
    args_list = [(symbol, max_expirations) for symbol in symbols]

    # Download in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(download_worker, args_list)

    # Print summary
    elapsed = time.time() - start_time
    total_success = sum(r[1] for r in results)
    total_errors = sum(r[2] for r in results)

    logger.info("")
    logger.info("=== Download Complete ===")
    logger.info(f"Time: {elapsed:.1f} seconds")
    logger.info(f"Total expirations downloaded: {total_success}")
    logger.info(f"Total errors: {total_errors}")
    logger.info("")
    logger.info("Results by symbol:")
    for symbol, success, errors in results:
        status = "✓" if errors == 0 else "⚠"
        logger.info(f"  {status} {symbol}: {success} success, {errors} errors")

    logger.info("")
    logger.info(f"Data saved to: {Path('data/yfinance/options/').absolute()}")


if __name__ == "__main__":
    main()
