import os
import urllib.request
import zipfile
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

PAIRS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT']
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/{pair}/5m/{pair}-5m-{year}-{month:02d}.zip"
DATA_DIR = "/home/kim/projects/kimsfinance/data/binance"

os.makedirs(DATA_DIR, exist_ok=True)

def download_data():
    # We are in May 2026, let's download from 2025-04 to 2026-03 (12 months)
    start_date = datetime(2025, 4, 1)
    
    for pair in PAIRS:
        print(f"\nProcessing {pair}...")
        all_dfs = []
        parquet_path = os.path.join(DATA_DIR, f"{pair}_5m_1y.parquet")
        
        if os.path.exists(parquet_path):
            print(f"Parquet already exists for {pair}, skipping download.")
            continue
            
        for i in range(12):
            target_date = start_date + relativedelta(months=i)
            year = target_date.year
            month = target_date.month
            
            url = BASE_URL.format(pair=pair, year=year, month=month)
            zip_path = os.path.join(DATA_DIR, f"temp_{pair}.zip")
            csv_name = f"{pair}-5m-{year}-{month:02d}.csv"
            
            try:
                # print(f"  Downloading {year}-{month:02d}...")
                urllib.request.urlretrieve(url, zip_path)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    with zip_ref.open(csv_name) as f:
                        df = pd.read_csv(f, header=None, names=[
                            'open_time', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_asset_volume', 'number_of_trades',
                            'taker_buy_base', 'taker_buy_quote', 'ignore'
                        ])
                        all_dfs.append(df)
                
                os.remove(zip_path)
            except Exception as e:
                print(f"  Failed for {year}-{month:02d}: {e}")
                
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            # Convert timestamp from ms to datetime
            final_df['open_time'] = pd.to_numeric(final_df['open_time'], errors='coerce')
            final_df.dropna(subset=['open_time'], inplace=True)
            # If timestamp is in microseconds (e.g. 2026 data), convert to milliseconds
            final_df.loc[final_df['open_time'] > 3000000000000, 'open_time'] /= 1000
            final_df['open_time'] = pd.to_datetime(final_df['open_time'], unit='ms')
            final_df.set_index('open_time', inplace=True)
            
            # Format columns like yfinance
            final_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            final_df = final_df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            final_df.to_parquet(parquet_path)
            print(f"Saved {len(final_df)} rows for {pair} to {parquet_path}")

if __name__ == "__main__":
    print("Starting Binance Vision download...")
    download_data()
    print("\nDownload complete.")
