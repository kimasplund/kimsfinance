#!/usr/bin/env python3
"""Quick script to inspect options data"""

import pandas as pd
import sys

if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = "data/yfinance/options_historical/AAPL/2020-01-02.parquet"

df = pd.read_parquet(file_path)

print(f"File: {file_path}")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Filter for puts
puts = df[df['optionType'] == 'put']
print(f"\nTotal puts: {len(puts)}")

# Check for volume and OI
print(f"\nVolume stats:")
print(puts['volume'].describe())
print(f"\nOpen Interest stats:")
print(puts['openInterest'].describe())

# Check for delta
if 'delta' in puts.columns:
    print(f"\nDelta stats:")
    print(puts['delta'].describe())

    # Count puts with delta in range
    delta_range = puts[(puts['delta'].abs() >= 0.15) & (puts['delta'].abs() <= 0.35)]
    print(f"\nPuts with delta 0.15-0.35: {len(delta_range)}")

    # Count with volume/OI requirements
    liquid = delta_range[(delta_range['volume'] >= 10) & (delta_range['openInterest'] >= 100)]
    print(f"  With volume>=10 and OI>=100: {len(liquid)}")
