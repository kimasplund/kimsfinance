#!/usr/bin/env python3
"""
Extensive 1-Minute Candle Library Builder
==========================================

Downloads 5 years of 1-minute candles for 100+ symbols in parallel
and converts to parquet format for fast loading.

Features:
- Downloads from Y Finance (free, no API key needed)
- Parallel downloading (20 workers by default)
- Automatic parquet conversion
- Converts existing CSV data
- Progress tracking
- Error handling and retry logic

Usage:
    python scripts/build_extensive_1min_library.py [--workers 20] [--years 5]
"""

import argparse
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Comprehensive symbol list (100+ symbols across sectors)
SYMBOLS = [
    # Indices
    "SPY", "QQQ", "IWM", "DIA", "VXX", "UVXY",

    # FAANG + Mega Cap Tech
    "AAPL", "GOOGL", "GOOG", "AMZN", "META", "MSFT", "NVDA", "TSLA",
    "NFLX", "AMD", "INTC", "CRM", "ORCL", "ADBE", "CSCO", "AVGO",

    # Semiconductors
    "TSM", "ASML", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC",

    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW",
    "V", "MA", "AXP", "PYPL", "SQ",

    # Healthcare & Biotech
    "UNH", "JNJ", "PFE", "ABBV", "TMO", "ABT", "LLY", "MRK",
    "AMGN", "GILD", "BIIB", "REGN", "VRTX", "MRNA", "BNTX",

    # Consumer
    "AMZN", "HD", "NKE", "SBUX", "MCD", "DIS", "COST", "TGT",
    "WMT", "PG", "KO", "PEP", "MDLZ", "CL",

    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",

    # Travel & Hospitality (already have some)
    "ABNB", "BKNG", "MAR", "HLT", "CCL", "RCL", "NCLH",
    "UAL", "DAL", "AAL", "LUV",
    "EXPE", "TRIP",

    # Gaming & Casino
    "LVS", "MGM", "WYNN", "CZR", "PENN", "HAS", "TTWO", "EA", "ATVI",

    # Industrials
    "BA", "CAT", "HON", "UNP", "RTX", "LMT", "GE", "MMM",
    "DE", "EMR",

    # Communications
    "GOOGL", "META", "DIS", "NFLX", "T", "VZ", "TMUS", "CMCSA",

    # EVs & Auto
    "TSLA", "F", "GM", "RIVN", "LCID", "NIO", "XPEV",

    # E-commerce & Retail
    "SHOP", "ETSY", "EBAY", "CHWY", "W", "FTCH",

    # Cloud & Software
    "NOW", "SNOW", "DDOG", "NET", "CRWD", "ZS", "OKTA", "PLTR",
    "WDAY", "TEAM", "ZM", "DOCU", "UBER", "LYFT",

    # REITs
    "AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "WELL", "AVB",
]

# Remove duplicates and sort
SYMBOLS = sorted(list(set(SYMBOLS)))

# Output directories
PARQUET_DIR = Path("data/1min_candles_parquet")
OLD_CSV_DIR = Path("/home/kim/projects/ib-data-downloader/data")


def download_symbol(symbol: str, years: int = 5) -> Optional[pd.DataFrame]:
    """
    Download 1-minute data for a symbol for the specified number of years.

    Args:
        symbol: Ticker symbol
        years: Number of years to download (default: 5)

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        logger.info(f"{symbol}: Downloading {years} years of 1min data...")

        # Download from yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval="1m",
            auto_adjust=False,  # Keep raw prices
            back_adjust=False,
            prepost=False,      # Regular hours only
        )

        if df.empty:
            logger.warning(f"{symbol}: No data available")
            return None

        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        df.index.name = 'timestamp'
        df.reset_index(inplace=True)

        # Add symbol column
        df['symbol'] = symbol

        logger.info(f"{symbol}: Downloaded {len(df):,} candles ({df['timestamp'].min()} to {df['timestamp'].max()})")
        return df

    except Exception as e:
        logger.error(f"{symbol}: Download failed - {e}")
        return None


def save_to_parquet(df: pd.DataFrame, symbol: str, output_dir: Path) -> bool:
    """
    Save DataFrame to parquet format.

    Args:
        df: DataFrame with candle data
        symbol: Ticker symbol
        output_dir: Output directory

    Returns:
        True if successful, False otherwise
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{symbol}_1min.parquet"

        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"{symbol}: Saved to {output_path} ({file_size_mb:.2f} MB)")
        return True

    except Exception as e:
        logger.error(f"{symbol}: Save to parquet failed - {e}")
        return False


def convert_existing_csvs(csv_dir: Path, parquet_dir: Path) -> Dict[str, int]:
    """
    Convert existing CSV files to parquet format.

    Args:
        csv_dir: Directory containing CSV files
        parquet_dir: Output directory for parquet files

    Returns:
        Dict mapping symbol to number of rows converted
    """
    logger.info(f"Converting existing CSVs from {csv_dir}...")

    if not csv_dir.exists():
        logger.warning(f"CSV directory {csv_dir} does not exist")
        return {}

    csv_files = list(csv_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files")

    results = {}

    for csv_file in csv_files:
        try:
            # Extract symbol from filename
            # Format: SYMBOL_1min_HISTORICAL_VOLATILITY_...csv
            symbol = csv_file.stem.split('_')[0]

            if symbol not in results:
                results[symbol] = []

            # Read CSV
            df = pd.read_csv(csv_file)
            results[symbol].append(df)

        except Exception as e:
            logger.error(f"Failed to read {csv_file}: {e}")

    # Concatenate all data for each symbol and save
    converted = {}
    for symbol, dfs in results.items():
        try:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df['symbol'] = symbol

            # Sort by timestamp if exists
            if 'timestamp' in combined_df.columns:
                combined_df = combined_df.sort_values('timestamp')
            elif 'date' in combined_df.columns:
                combined_df = combined_df.sort_values('date')

            # Remove duplicates
            combined_df = combined_df.drop_duplicates()

            if save_to_parquet(combined_df, symbol, parquet_dir):
                converted[symbol] = len(combined_df)
                logger.info(f"{symbol}: Converted {len(combined_df):,} rows from CSV")

        except Exception as e:
            logger.error(f"{symbol}: Failed to convert CSV data - {e}")

    return converted


def download_and_save(symbol: str, years: int, output_dir: Path) -> tuple:
    """
    Download data for symbol and save to parquet.

    Returns:
        (symbol, success, num_rows)
    """
    df = download_symbol(symbol, years)

    if df is None:
        return (symbol, False, 0)

    success = save_to_parquet(df, symbol, output_dir)
    return (symbol, success, len(df))


def main():
    parser = argparse.ArgumentParser(description="Build extensive 1-minute candle library")
    parser.add_argument('--workers', type=int, default=20, help="Number of parallel workers")
    parser.add_argument('--years', type=int, default=5, help="Years of historical data to download")
    parser.add_argument('--skip-existing', action='store_true', help="Skip symbols that already exist")
    parser.add_argument('--convert-only', action='store_true', help="Only convert existing CSVs, skip downloads")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Extensive 1-Minute Candle Library Builder")
    logger.info("=" * 80)
    logger.info(f"Symbols to download: {len(SYMBOLS)}")
    logger.info(f"Years of data: {args.years}")
    logger.info(f"Parallel workers: {args.workers}")
    logger.info(f"Output directory: {PARQUET_DIR}")
    logger.info("=" * 80)

    start_time = time.time()

    # Step 1: Convert existing CSVs
    if OLD_CSV_DIR.exists():
        logger.info("\n[Step 1/2] Converting existing CSV data...")
        converted = convert_existing_csvs(OLD_CSV_DIR, PARQUET_DIR)
        logger.info(f"Converted {len(converted)} symbols from CSV")
        for symbol, rows in converted.items():
            logger.info(f"  {symbol}: {rows:,} rows")
    else:
        logger.info(f"\n[Step 1/2] Skipping CSV conversion (directory {OLD_CSV_DIR} not found)")

    if args.convert_only:
        logger.info("\n--convert-only specified, skipping downloads")
        return

    # Step 2: Download new data in parallel
    logger.info(f"\n[Step 2/2] Downloading {len(SYMBOLS)} symbols in parallel...")

    # Filter out already existing if requested
    symbols_to_download = SYMBOLS
    if args.skip_existing:
        existing = [f.stem.replace('_1min', '') for f in PARQUET_DIR.glob("*_1min.parquet")]
        symbols_to_download = [s for s in SYMBOLS if s not in existing]
        logger.info(f"Skipping {len(SYMBOLS) - len(symbols_to_download)} existing symbols")

    successful = 0
    failed = 0
    total_rows = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(download_and_save, symbol, args.years, PARQUET_DIR): symbol
            for symbol in symbols_to_download
        }

        # Process results as they complete
        for future in as_completed(futures):
            symbol, success, num_rows = future.result()

            if success:
                successful += 1
                total_rows += num_rows
                logger.info(f"Progress: {successful + failed}/{len(symbols_to_download)} | Success: {successful} | Failed: {failed}")
            else:
                failed += 1

    elapsed = time.time() - start_time

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Successful: {successful}/{len(symbols_to_download)}")
    logger.info(f"Failed: {failed}/{len(symbols_to_download)}")
    logger.info(f"Total candles: {total_rows:,}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Output directory: {PARQUET_DIR}")
    logger.info("=" * 80)

    # List all parquet files
    parquet_files = sorted(PARQUET_DIR.glob("*_1min.parquet"))
    logger.info(f"\nTotal symbols in library: {len(parquet_files)}")

    total_size_mb = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)
    logger.info(f"Total library size: {total_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
