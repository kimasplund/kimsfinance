import os
import urllib.request
import zipfile
import pandas as pd
from datetime import datetime, UTC
from dateutil.relativedelta import relativedelta
import argparse

PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
]
SPOT_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/{pair}/5m/{pair}-5m-{year}-{month:02d}.zip"
FUTURES_UM_BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines/{pair}/5m/{pair}-5m-{year}-{month:02d}.zip"
BASE_DATA_DIR = "/home/kim/projects/kimsfinance/data"


def get_output_dir(market: str) -> str:
    if market == "spot":
        return os.path.join(BASE_DATA_DIR, "binance")
    return os.path.join(BASE_DATA_DIR, "binance_futures_um")


def download_data(market: str, months: int = 12, force: bool = False, pairs=None):
    # Download the latest completed rolling window, e.g. if now is 2026-06,
    # we fetch 2025-06..2026-05 for months=12.
    end_date = datetime.now(UTC).replace(day=1, tzinfo=None) - relativedelta(months=1)
    start_date = end_date - relativedelta(months=months - 1)
    base_url = SPOT_BASE_URL if market == "spot" else FUTURES_UM_BASE_URL
    data_dir = get_output_dir(market)
    os.makedirs(data_dir, exist_ok=True)

    print(f"Market: {market}")
    print(
        f"Window: {start_date.strftime('%Y-%m')} .. {end_date.strftime('%Y-%m')} ({months} months)"
    )
    print(f"Output: {data_dir}\n")

    selected_pairs = pairs if pairs else PAIRS

    for pair in selected_pairs:
        print(f"\nProcessing {pair}...")
        all_dfs = []
        parquet_path = os.path.join(data_dir, f"{pair}_5m_1y.parquet")

        if os.path.exists(parquet_path) and not force:
            print(f"Parquet already exists for {pair}, skipping download.")
            continue

        for i in range(months):
            target_date = start_date + relativedelta(months=i)
            year = target_date.year
            month = target_date.month

            url = base_url.format(pair=pair, year=year, month=month)
            zip_path = os.path.join(data_dir, f"temp_{pair}.zip")
            csv_name = f"{pair}-5m-{year}-{month:02d}.csv"

            try:
                # print(f"  Downloading {year}-{month:02d}...")
                urllib.request.urlretrieve(url, zip_path)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    with zip_ref.open(csv_name) as f:
                        df = pd.read_csv(
                            f,
                            header=None,
                            names=[
                                "open_time",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "close_time",
                                "quote_asset_volume",
                                "number_of_trades",
                                "taker_buy_base",
                                "taker_buy_quote",
                                "ignore",
                            ],
                        )
                        all_dfs.append(df)

                os.remove(zip_path)
            except Exception as e:
                print(f"  Failed for {year}-{month:02d}: {e}")

        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            # Keep open_time as int64 milliseconds for Rust parquet loader.
            final_df["open_time"] = pd.to_numeric(final_df["open_time"], errors="coerce")
            final_df.dropna(subset=["open_time"], inplace=True)
            # If timestamp is in microseconds, normalize to milliseconds.
            final_df.loc[final_df["open_time"] > 3000000000000, "open_time"] /= 1000
            final_df["open_time"] = final_df["open_time"].astype("int64")

            # Format columns like yfinance
            final_df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
            final_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
            final_df = final_df[["open_time", "Open", "High", "Low", "Close", "Volume"]]

            final_df.to_parquet(parquet_path)
            print(f"Saved {len(final_df)} rows for {pair} to {parquet_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance Vision 5m kline data to parquet")
    parser.add_argument(
        "--market", choices=["spot", "futures_um"], default="spot", help="Data source market"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Rolling months to download (latest completed months)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Redownload even if output parquet exists"
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="",
        help="Comma-separated symbols (e.g. BTCUSDT,ETHUSDT,SOLUSDT). Default: built-in symbol list",
    )
    args = parser.parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()] if args.pairs else None

    print("Starting Binance Vision download...")
    download_data(market=args.market, months=args.months, force=args.force, pairs=pairs)
    print("\nDownload complete.")
