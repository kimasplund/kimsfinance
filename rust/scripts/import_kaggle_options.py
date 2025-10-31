#!/usr/bin/env python3
"""
Import Kaggle Historical Options Datasets

Downloads and imports Kyle Graupe's historical options datasets from Kaggle
into our historical database format.

Datasets:
- AAPL: Q1 2016 - Q1 2023 (7 years)
- SPY: Q1 2020 - Q4 2022 (3 years)
- TSLA: Q1 2019 - Q4 2022 (4 years)
- QQQ: Q1 2020 - Q4 2022 (3 years)

Requirements:
    pip install kaggle pandas pyarrow

Setup Kaggle API:
    1. Go to https://www.kaggle.com/settings
    2. Click "Create New Token" under API section
    3. Move downloaded kaggle.json to ~/.kaggle/
    4. chmod 600 ~/.kaggle/kaggle.json

Usage:
    python scripts/import_kaggle_options.py
"""

import os
import sys
import pandas as pd
from pathlib import Path
import subprocess
from typing import Dict, List
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KaggleOptionsImporter:
    """Import Kaggle options datasets into historical database"""

    # Kaggle datasets to download
    DATASETS = {
        'AAPL': {
            'kaggle_id': 'kylegraupe/aapl-options-data-2016-2020',
            'period': '2016-2023',
            'years': 7
        },
        'SPY': {
            'kaggle_id': 'kylegraupe/spy-daily-eod-options-quotes-2020-2022',
            'period': '2020-2022',
            'years': 3
        },
        'TSLA': {
            'kaggle_id': 'kylegraupe/tsla-daily-eod-options-quotes-2019-2022',
            'period': '2019-2022',
            'years': 4
        },
        'QQQ': {
            'kaggle_id': 'kylegraupe/qqq-daily-option-chains-q1-2020-to-q4-2022',
            'period': '2020-2022',
            'years': 3
        }
    }

    def __init__(
        self,
        kaggle_dir: str = "data/kaggle_raw",
        output_dir: str = "data/yfinance/options_historical"
    ):
        self.kaggle_dir = Path(kaggle_dir)
        self.output_dir = Path(output_dir)
        self.kaggle_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_kaggle_setup(self) -> bool:
        """Check if Kaggle API is set up"""
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'

        if not kaggle_json.exists():
            logger.error("Kaggle API not configured!")
            logger.error("Setup instructions:")
            logger.error("1. Go to https://www.kaggle.com/settings")
            logger.error("2. Click 'Create New Token' under API section")
            logger.error("3. Move downloaded kaggle.json to ~/.kaggle/")
            logger.error("4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False

        # Check kaggle CLI is installed
        try:
            subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
            logger.info("✓ Kaggle API configured")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Kaggle CLI not installed!")
            logger.error("Install: pip install kaggle")
            return False

    def download_dataset(self, symbol: str, kaggle_id: str) -> Path:
        """Download dataset from Kaggle"""
        logger.info(f"Downloading {symbol} dataset from Kaggle...")

        dataset_dir = self.kaggle_dir / symbol
        dataset_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download dataset
            cmd = [
                'kaggle', 'datasets', 'download',
                '-d', kaggle_id,
                '-p', str(dataset_dir),
                '--unzip'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"✓ Downloaded {symbol} dataset")
            return dataset_dir

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download {symbol}: {e.stderr}")
            raise

    def find_csv_files(self, dataset_dir: Path) -> List[Path]:
        """Find all CSV files in dataset directory"""
        csv_files = list(dataset_dir.glob('*.csv'))
        if not csv_files:
            csv_files = list(dataset_dir.glob('**/*.csv'))
        return csv_files

    def convert_to_historical_format(
        self,
        symbol: str,
        dataset_dir: Path
    ) -> int:
        """
        Convert Kaggle dataset to our historical database format

        Returns:
            Number of dates imported
        """
        logger.info(f"Converting {symbol} to historical format...")

        # Find CSV files
        csv_files = self.find_csv_files(dataset_dir)

        if not csv_files:
            logger.warning(f"No CSV files found for {symbol}")
            return 0

        logger.info(f"Found {len(csv_files)} CSV file(s)")

        imported_dates = 0

        for csv_file in csv_files:
            logger.info(f"Processing {csv_file.name}...")

            try:
                # Read CSV with proper handling for data quality issues
                # - UTF-8 BOM encoding
                # - Empty values as NaN
                # - Skip bad lines
                df = pd.read_csv(
                    csv_file,
                    encoding='utf-8-sig',  # Handle BOM
                    na_values=['', ' ', 'nan', 'NaN'],  # Treat empty/space as NaN
                    keep_default_na=True,
                    on_bad_lines='skip'  # Skip malformed rows
                )

                logger.info(f"  Loaded {len(df)} rows")
                logger.info(f"  Columns: {df.columns.tolist()}")

                # Normalize column names (Kaggle datasets may vary)
                df = self.normalize_columns(df)

                # Check for date column (after normalization, it should be 'snapshotDate')
                if 'snapshotDate' not in df.columns:
                    logger.error(f"  No snapshotDate column found in {csv_file.name}")
                    logger.error(f"  Available columns: {df.columns.tolist()}")
                    continue

                # Ensure date is in correct format
                df['snapshotDate'] = pd.to_datetime(df['snapshotDate']).dt.strftime('%Y-%m-%d')

                # Group by date and save each day
                for date, group in df.groupby('snapshotDate'):
                    self.save_daily_snapshot(symbol, date, group)
                    imported_dates += 1

                logger.info(f"  ✓ Imported {len(df['snapshotDate'].unique())} days from {csv_file.name}")

            except Exception as e:
                logger.error(f"  Error processing {csv_file.name}: {e}")
                continue

        return imported_dates

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to match our format"""
        # First, clean up column names (remove brackets, leading/trailing spaces)
        df.columns = df.columns.str.strip().str.replace('[', '').str.replace(']', '').str.lower()

        # Common column name mappings from Kaggle datasets
        rename_map = {
            # Date columns
            'quote_date': 'snapshotDate',
            'date': 'snapshotDate',
            'datadate': 'snapshotDate',

            # Basic columns
            'underlying_symbol': 'symbol',
            'root': 'symbol',
            'underlying': 'symbol',

            # Expiration
            'expire_date': 'expiration',
            'expiration': 'expiration',
            'expiration_date': 'expiration',

            # Strike
            'strike': 'strike',
            'strike_price': 'strike',

            # Options columns (we'll handle call/put split separately)
            'option_type': 'optionType',
            'type': 'optionType',
            'call_put': 'optionType',

            # Pricing columns
            'bid': 'bid',
            'ask': 'ask',
            'last': 'lastPrice',
            'last_price': 'lastPrice',

            'volume': 'volume',
            'open_interest': 'openInterest',
            'oi': 'openInterest',

            'implied_volatility': 'impliedVolatility',
            'iv': 'impliedVolatility',
            'imp_vol': 'impliedVolatility',

            # Greeks
            'delta': 'delta',
            'gamma': 'gamma',
            'theta': 'theta',
            'vega': 'vega',
            'rho': 'rho',
        }

        # Check if this is split call/put format (c_bid/p_bid instead of bid + optionType)
        has_call_put_split = 'c_bid' in df.columns and 'p_bid' in df.columns

        if has_call_put_split:
            # Split into calls and puts
            call_cols = {col: col.replace('c_', '') for col in df.columns if col.startswith('c_')}
            put_cols = {col: col.replace('p_', '') for col in df.columns if col.startswith('p_')}
            common_cols = [col for col in df.columns if not col.startswith('c_') and not col.startswith('p_')]

            # Create calls DataFrame
            calls = df[common_cols + list(call_cols.keys())].copy()
            calls = calls.rename(columns=call_cols)
            calls['optionType'] = 'call'

            # Create puts DataFrame
            puts = df[common_cols + list(put_cols.keys())].copy()
            puts = puts.rename(columns=put_cols)
            puts['optionType'] = 'put'

            # Combine
            df = pd.concat([calls, puts], ignore_index=True)

        # Rename columns
        df = df.rename(columns=rename_map)

        # Convert numeric columns, handling strings with spaces
        numeric_cols = [
            'strike', 'bid', 'ask', 'lastPrice', 'volume', 'openInterest',
            'impliedVolatility', 'delta', 'gamma', 'theta', 'vega', 'rho'
        ]

        for col in numeric_cols:
            if col in df.columns:
                # Strip whitespace and convert to numeric, coercing errors to NaN
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Normalize option_type values (call/put vs C/P)
        if 'optionType' in df.columns:
            df['optionType'] = df['optionType'].str.lower().map({
                'call': 'call',
                'put': 'put',
                'c': 'call',
                'p': 'put',
            })

        return df

    def save_daily_snapshot(
        self,
        symbol: str,
        date: str,
        df: pd.DataFrame
    ):
        """Save daily snapshot to historical database"""
        # Create symbol directory
        symbol_dir = self.output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Output file
        output_file = symbol_dir / f"{date}.parquet"

        # Check if already exists
        if output_file.exists():
            logger.debug(f"  Skipping {symbol} {date} (already exists)")
            return

        # Add metadata
        if 'snapshotDate' not in df.columns:
            df['snapshotDate'] = date

        if 'symbol' not in df.columns:
            df['symbol'] = symbol

        df['downloadTime'] = datetime.now().isoformat()

        # Save to parquet
        df.to_parquet(output_file, index=False)
        logger.debug(f"  Saved {symbol} {date}: {len(df)} options")

    def import_all(self) -> Dict[str, int]:
        """Import all Kaggle datasets"""
        results = {}

        for symbol, info in self.DATASETS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Importing {symbol} ({info['period']}, {info['years']} years)")
            logger.info(f"{'='*60}\n")

            try:
                # Download dataset
                dataset_dir = self.download_dataset(symbol, info['kaggle_id'])

                # Convert to historical format
                imported = self.convert_to_historical_format(symbol, dataset_dir)

                results[symbol] = imported
                logger.info(f"\n✓ {symbol}: Imported {imported} days")

            except Exception as e:
                logger.error(f"\n✗ {symbol}: Failed - {e}")
                results[symbol] = 0

        return results

    def print_summary(self, results: Dict[str, int]):
        """Print import summary"""
        logger.info(f"\n{'='*60}")
        logger.info("Import Complete")
        logger.info(f"{'='*60}\n")

        total_days = sum(results.values())

        logger.info("Results by symbol:")
        for symbol, days in results.items():
            info = self.DATASETS[symbol]
            status = "✓" if days > 0 else "✗"
            logger.info(f"  {status} {symbol}: {days} days ({info['period']})")

        logger.info(f"\nTotal days imported: {total_days}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")

        logger.info("\nData coverage:")
        logger.info("  AAPL: 2016-2023 (7 years)")
        logger.info("  SPY:  2020-2022 (3 years)")
        logger.info("  TSLA: 2019-2022 (4 years)")
        logger.info("  QQQ:  2020-2022 (3 years)")

        logger.info("\nYou can now query this data using:")
        logger.info("  from scripts.query_options_historical import OptionsHistoricalDB")
        logger.info("  db = OptionsHistoricalDB()")
        logger.info("  df = db.get_options('AAPL', '2020-01-02')")


def main():
    """Main entry point"""
    logger.info("=== Kaggle Options Data Importer ===\n")

    importer = KaggleOptionsImporter()

    # Check Kaggle setup
    if not importer.check_kaggle_setup():
        sys.exit(1)

    # Import all datasets
    results = importer.import_all()

    # Print summary
    importer.print_summary(results)


if __name__ == "__main__":
    main()
