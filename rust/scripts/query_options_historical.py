#!/usr/bin/env python3
"""
Query Utilities for Historical Options Data

Efficient querying of historical options database.

Usage:
    from query_options_historical import OptionsHistoricalDB

    db = OptionsHistoricalDB()

    # Get all options for AAPL on a specific date
    df = db.get_options(symbol='AAPL', date='2025-10-30')

    # Get specific expiration
    df = db.get_options(symbol='AAPL', date='2025-10-30', expiration='2025-11-07')

    # Get ATM options
    df = db.get_atm_options(symbol='AAPL', date='2025-10-30', spot_price=220.0, window=5)

    # Get available dates for a symbol
    dates = db.get_available_dates('AAPL')
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta


class OptionsHistoricalDB:
    """Query interface for historical options database"""

    def __init__(self, base_dir: str = "data/yfinance/options_historical"):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise ValueError(f"Historical database not found: {self.base_dir}")

    def get_options(
        self,
        symbol: str,
        date: str,
        expiration: Optional[str] = None,
        option_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get options data for a specific symbol and date

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            date: Snapshot date (e.g., '2025-10-30')
            expiration: Optional expiration date filter
            option_type: Optional 'call' or 'put' filter

        Returns:
            DataFrame with options data
        """
        file_path = self.base_dir / symbol / f"{date}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"No data for {symbol} on {date}")

        df = pd.read_parquet(file_path)

        # Apply filters
        if expiration:
            df = df[df['expiration'] == expiration]

        if option_type:
            df = df[df['optionType'] == option_type.lower()]

        return df

    def get_options_range(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        expiration: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get options data for a date range

        Args:
            symbol: Stock symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            expiration: Optional expiration filter

        Returns:
            Combined DataFrame with all dates
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_data = []
        current = start

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                df = self.get_options(symbol, date_str, expiration)
                all_data.append(df)
            except FileNotFoundError:
                # Skip missing dates (weekends, holidays)
                pass

            current += timedelta(days=1)

        if not all_data:
            raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}")

        return pd.concat(all_data, ignore_index=True)

    def get_atm_options(
        self,
        symbol: str,
        date: str,
        spot_price: float,
        expiration: Optional[str] = None,
        window: float = 5.0
    ) -> pd.DataFrame:
        """
        Get at-the-money options (strikes within window of spot price)

        Args:
            symbol: Stock symbol
            date: Snapshot date
            spot_price: Current stock price
            expiration: Optional expiration filter
            window: Strike price window (±percentage, e.g., 5.0 = ±5%)

        Returns:
            DataFrame with ATM options
        """
        df = self.get_options(symbol, date, expiration)

        # Calculate strike window
        lower = spot_price * (1 - window / 100)
        upper = spot_price * (1 + window / 100)

        # Filter to ATM strikes
        atm = df[(df['strike'] >= lower) & (df['strike'] <= upper)]

        return atm.sort_values('strike')

    def get_available_dates(self, symbol: str) -> List[str]:
        """
        Get all available snapshot dates for a symbol

        Returns:
            Sorted list of date strings
        """
        symbol_dir = self.base_dir / symbol

        if not symbol_dir.exists():
            return []

        dates = []
        for file in symbol_dir.glob("*.parquet"):
            date_str = file.stem  # Filename without extension
            dates.append(date_str)

        return sorted(dates)

    def get_available_expirations(self, symbol: str, date: str) -> List[str]:
        """
        Get all available expirations for a symbol on a date

        Returns:
            Sorted list of expiration date strings
        """
        df = self.get_options(symbol, date)
        expirations = df['expiration'].unique().tolist()
        return sorted(expirations)

    def get_iv_surface(
        self,
        symbol: str,
        date: str,
        expiration: str
    ) -> pd.DataFrame:
        """
        Get implied volatility surface (IV by strike and option type)

        Returns:
            DataFrame with strikes and IV for calls/puts
        """
        df = self.get_options(symbol, date, expiration)

        # Pivot to get calls and puts side by side
        calls = df[df['optionType'] == 'call'][['strike', 'impliedVolatility']].rename(
            columns={'impliedVolatility': 'call_iv'}
        )
        puts = df[df['optionType'] == 'put'][['strike', 'impliedVolatility']].rename(
            columns={'impliedVolatility': 'put_iv'}
        )

        surface = pd.merge(calls, puts, on='strike', how='outer')
        return surface.sort_values('strike')

    def get_stats(self) -> dict:
        """
        Get database statistics

        Returns:
            Dict with symbol counts, date ranges, total size
        """
        stats = {}

        for symbol_dir in self.base_dir.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                dates = self.get_available_dates(symbol)

                if dates:
                    stats[symbol] = {
                        'count': len(dates),
                        'first_date': dates[0],
                        'last_date': dates[-1]
                    }

        return stats


def main():
    """Example usage"""
    db = OptionsHistoricalDB()

    print("=== Historical Options Database ===\n")

    # Get database stats
    stats = db.get_stats()
    print("Database Statistics:")
    for symbol, info in stats.items():
        print(f"  {symbol}: {info['count']} snapshots ({info['first_date']} to {info['last_date']})")

    print("\n" + "="*50 + "\n")

    # Example query
    if stats:
        symbol = list(stats.keys())[0]
        date = stats[symbol]['last_date']

        print(f"Example: Querying {symbol} options on {date}\n")

        df = db.get_options(symbol, date)
        print(f"Total options: {len(df)}")
        print(f"Expirations: {df['expiration'].nunique()}")
        print(f"Strikes: {df['strike'].nunique()}")

        print("\nFirst few rows:")
        print(df[['expiration', 'strike', 'optionType', 'bid', 'ask', 'impliedVolatility']].head(10))


if __name__ == "__main__":
    main()
