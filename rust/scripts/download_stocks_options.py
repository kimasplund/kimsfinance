#!/usr/bin/env python3
"""
Download stocks and options data for backtesting

Downloads data for 10 popular stocks with high options volume.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Top 10 stocks for options trading
SYMBOLS = [
    "AAPL",  # Apple - Tech bellwether
    "SPY",   # S&P 500 ETF - Most liquid options
    "TSLA",  # Tesla - High volatility
    "NVDA",  # NVIDIA - AI/chips
    "QQQ",   # Nasdaq ETF
    "AMD",   # AMD - Tech/chips
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "META",  # Meta
    "GOOGL", # Google
]

def download_stock_data(symbol, start_date, end_date, output_dir):
    """Download historical stock data"""
    print(f"\n📊 Downloading {symbol} stock data...")

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    if df.empty:
        print(f"  ⚠️  No data available for {symbol}")
        return None

    # Save to parquet
    output_path = output_dir / "stocks" / symbol / "daily"
    output_path.mkdir(parents=True, exist_ok=True)

    year = datetime.strptime(start_date, "%Y-%m-%d").year
    parquet_file = output_path / f"{year}.parquet"
    df.to_parquet(parquet_file, compression="snappy")

    print(f"  ✓ Saved: {parquet_file} ({len(df)} rows, {parquet_file.stat().st_size / 1024:.1f} KB)")

    return df

def download_options_chain(symbol, output_dir):
    """Download options chain for nearest expiration"""
    print(f"\n📈 Downloading {symbol} options chain...")

    try:
        ticker = yf.Ticker(symbol)

        # Get available expirations
        expirations = ticker.options
        if not expirations:
            print(f"  ⚠️  No options available for {symbol}")
            return None

        # Get nearest expiration
        expiration = expirations[0]
        print(f"  Expiration: {expiration}")

        # Get options chain
        opt_chain = ticker.option_chain(expiration)
        calls = opt_chain.calls
        puts = opt_chain.puts

        # Combine calls and puts
        calls['optionType'] = 'call'
        puts['optionType'] = 'put'
        options = pd.concat([calls, puts], ignore_index=True)

        # Save to parquet
        output_path = output_dir / "options" / symbol / "chain"
        output_path.mkdir(parents=True, exist_ok=True)

        parquet_file = output_path / f"{expiration}.parquet"
        options.to_parquet(parquet_file, compression="snappy")

        print(f"  ✓ Saved: {parquet_file}")
        print(f"    Calls: {len(calls)}, Puts: {len(puts)}, Total: {len(options)}")
        print(f"    Size: {parquet_file.stat().st_size / 1024:.1f} KB")

        return options

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    print("=" * 80)
    print("STOCK & OPTIONS DATA DOWNLOADER")
    print("=" * 80)
    print(f"System Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print()

    # Setup directories
    base_dir = Path("/home/kim-asplund/projects/kimsfinance/data")
    yahoo_dir = base_dir / "yahoo"

    # Date range for stock data (last year)
    end_date = "2025-10-30"
    start_date = "2024-01-01"

    print(f"Stock data period: {start_date} to {end_date}")

    # Download data for each symbol
    summary = {
        "download_date": datetime.now().isoformat(),
        "symbols": {},
    }

    for symbol in SYMBOLS:
        print(f"\n{'-' * 80}")
        print(f"Processing: {symbol}")
        print(f"{'-' * 80}")

        # Download stock data
        stock_df = download_stock_data(symbol, start_date, end_date, yahoo_dir)

        # Download options chain
        options_df = download_options_chain(symbol, yahoo_dir)

        # Summary
        summary["symbols"][symbol] = {
            "stock_rows": len(stock_df) if stock_df is not None else 0,
            "options_rows": len(options_df) if options_df is not None else 0,
            "stock_downloaded": stock_df is not None,
            "options_downloaded": options_df is not None,
        }

    # Save summary
    summary_file = yahoo_dir / "download_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print("DOWNLOAD COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nSummary saved to: {summary_file}")

    # Print summary table
    print("\nDownload Summary:")
    print(f"{'Symbol':<8} {'Stock Rows':<12} {'Options Rows':<14} {'Status':<10}")
    print("-" * 50)
    for symbol, data in summary["symbols"].items():
        stock_status = "✓" if data["stock_downloaded"] else "✗"
        options_status = "✓" if data["options_downloaded"] else "✗"
        status = f"{stock_status} {options_status}"
        print(f"{symbol:<8} {data['stock_rows']:<12} {data['options_rows']:<14} {status:<10}")

    print(f"\nData directory: {yahoo_dir}")
    print("\nDirectory structure:")
    print("data/yahoo/")
    print("├── stocks/")
    print("│   ├── AAPL/daily/2024.parquet")
    print("│   ├── SPY/daily/2024.parquet")
    print("│   └── ...")
    print("└── options/")
    print("    ├── AAPL/chain/2025-11-21.parquet")
    print("    ├── SPY/chain/2025-11-21.parquet")
    print("    └── ...")

if __name__ == "__main__":
    main()
