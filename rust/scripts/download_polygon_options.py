#!/usr/bin/env python3
"""
Polygon.io Options Data Downloader

Downloads historical options data from Polygon.io REST API and flat files.
Requires Polygon.io API key.

API Documentation:
- REST API: https://polygon.io/docs/rest/options/overview
- Flat Files: https://polygon.io/docs/flat-files/options/overview

Usage:
    python scripts/download_polygon_options.py
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PolygonOptionsDownloader:
    """Download options data from Polygon.io"""

    def __init__(
        self,
        api_key: str,
        base_dir: str = "data/polygon/options_historical"
    ):
        self.api_key = api_key
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://api.polygon.io"

    def get_options_contracts(
        self,
        underlying_symbol: str,
        expiration_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get list of options contracts for an underlying symbol

        Args:
            underlying_symbol: Stock symbol (e.g., 'AAPL')
            expiration_date: Optional expiration date filter (YYYY-MM-DD)
            limit: Max results per request (default 1000)

        Returns:
            DataFrame with contract details
        """
        url = f"{self.base_url}/v3/reference/options/contracts"

        params = {
            'underlying_ticker': underlying_symbol,
            'apiKey': self.api_key,
            'limit': limit
        }

        if expiration_date:
            params['expiration_date'] = expiration_date

        logger.info(f"Fetching options contracts for {underlying_symbol}...")

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'results' not in data or not data['results']:
                logger.warning(f"No contracts found for {underlying_symbol}")
                return pd.DataFrame()

            contracts_df = pd.DataFrame(data['results'])
            logger.info(f"Found {len(contracts_df)} contracts")

            return contracts_df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching contracts: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def get_options_snapshot(
        self,
        option_ticker: str
    ) -> Dict:
        """
        Get current snapshot for an options contract

        Args:
            option_ticker: Option contract symbol (e.g., 'O:AAPL251121C00100000')

        Returns:
            Dictionary with snapshot data
        """
        url = f"{self.base_url}/v3/snapshot/options/{option_ticker}"

        params = {'apiKey': self.api_key}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'results' not in data:
                logger.warning(f"No snapshot data for {option_ticker}")
                return {}

            return data['results']

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching snapshot: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def get_options_chain_snapshot(
        self,
        underlying_symbol: str
    ) -> List[Dict]:
        """
        Get snapshot of entire options chain for underlying

        Args:
            underlying_symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            List of dictionaries with chain snapshot data
        """
        url = f"{self.base_url}/v3/snapshot/options/{underlying_symbol}"

        params = {'apiKey': self.api_key}

        logger.info(f"Fetching options chain snapshot for {underlying_symbol}...")

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'results' not in data or not data['results']:
                logger.warning(f"No chain snapshot for {underlying_symbol}")
                return []

            results = data['results']
            logger.info(f"Got snapshot with {len(results)} contracts")

            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching chain snapshot: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def get_aggregates(
        self,
        option_ticker: str,
        from_date: str,
        to_date: str,
        timespan: str = "day",
        limit: int = 5000
    ) -> pd.DataFrame:
        """
        Get aggregate bars (OHLCV) for an options contract

        Args:
            option_ticker: Option contract symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            timespan: Timespan (minute, hour, day, week, month)
            limit: Max results (default 5000)

        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.base_url}/v2/aggs/ticker/{option_ticker}/range/1/{timespan}/{from_date}/{to_date}"

        params = {
            'apiKey': self.api_key,
            'limit': limit
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'results' not in data or not data['results']:
                logger.warning(f"No aggregates data for {option_ticker}")
                return pd.DataFrame()

            df = pd.DataFrame(data['results'])

            # Convert Unix timestamp to datetime
            if 't' in df.columns:
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching aggregates: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def download_daily_chain(
        self,
        underlying_symbol: str,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download entire options chain for a specific date

        Args:
            underlying_symbol: Stock symbol
            date: Date (YYYY-MM-DD), defaults to today

        Returns:
            DataFrame with full options chain
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Downloading {underlying_symbol} options chain for {date}...")

        # Get chain snapshot (current data)
        chain_data = self.get_options_chain_snapshot(underlying_symbol)

        if not chain_data:
            logger.warning(f"No options data available for {underlying_symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(chain_data)

        # Add metadata
        df['snapshotDate'] = date
        df['symbol'] = underlying_symbol
        df['downloadTime'] = datetime.now().isoformat()

        # Save to file
        symbol_dir = self.base_dir / underlying_symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        output_file = symbol_dir / f"{date}.parquet"

        df.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(df)} contracts to {output_file}")

        return df

    def test_api_access(self) -> bool:
        """
        Test API key and check what endpoints are accessible

        Returns:
            True if API key is valid
        """
        logger.info("Testing Polygon.io API access...")

        tests = [
            ("Options Contracts", lambda: self.get_options_contracts('AAPL', limit=10)),
            ("Options Chain Snapshot", lambda: self.get_options_chain_snapshot('AAPL')),
        ]

        results = []

        for test_name, test_func in tests:
            try:
                logger.info(f"Testing: {test_name}...")
                result = test_func()

                if isinstance(result, pd.DataFrame):
                    success = not result.empty
                    logger.info(f"  ✓ {test_name}: {len(result)} results")
                elif isinstance(result, list):
                    success = len(result) > 0
                    logger.info(f"  ✓ {test_name}: {len(result)} results")
                else:
                    success = bool(result)
                    logger.info(f"  ✓ {test_name}: Success")

                results.append((test_name, success))
                time.sleep(1)  # Rate limiting

            except Exception as e:
                logger.error(f"  ✗ {test_name}: {e}")
                results.append((test_name, False))

        logger.info("\n=== API Access Test Results ===")
        for test_name, success in results:
            status = "✓" if success else "✗"
            logger.info(f"{status} {test_name}")

        successful = sum(1 for _, success in results if success)
        logger.info(f"\nTotal: {successful}/{len(tests)} endpoints accessible")

        return successful > 0


def main():
    """Main entry point"""
    import os

    # API key from environment variable
    API_KEY = os.environ.get("POLYGON_API_KEY")
    if not API_KEY:
        logger.error("POLYGON_API_KEY environment variable not set")
        logger.info("Set it with: export POLYGON_API_KEY=your_api_key")
        return

    downloader = PolygonOptionsDownloader(API_KEY)

    logger.info("=== Polygon.io Options Downloader ===\n")

    # Test API access
    if not downloader.test_api_access():
        logger.error("API access test failed. Check your API key and subscription.")
        return

    logger.info("\n=== Downloading Sample Data ===\n")

    # Download options chain for AAPL
    symbols = ["AAPL", "MSFT", "SPY"]

    for symbol in symbols:
        try:
            df = downloader.download_daily_chain(symbol)
            logger.info(f"✓ {symbol}: {len(df)} options downloaded\n")
            time.sleep(12)  # Free tier: 5 requests per minute = 12s delay

        except Exception as e:
            logger.error(f"✗ {symbol}: {e}\n")

    logger.info("\n=== Download Complete ===")
    logger.info(f"Data saved to: {downloader.base_dir.absolute()}")


if __name__ == "__main__":
    main()
