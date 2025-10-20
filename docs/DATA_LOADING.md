# Data Loading Guide

kimsfinance is a **computation and visualization library**, not an I/O library. It accepts in-memory data structures (NumPy arrays, Polars DataFrames, Pandas DataFrames) and focuses on fast processing and rendering.

This design gives you **complete flexibility** to load data from any source:
- ✅ Parquet files (recommended for performance)
- ✅ CSV files
- ✅ SQL databases
- ✅ REST APIs
- ✅ WebSocket streams
- ✅ Pandas DataFrames
- ✅ Any other source

---

## Quick Start: Load and Plot

### From Parquet (Recommended)

```python
import polars as pl
import kimsfinance as kf

# Load data
df = pl.read_parquet('ohlcv_data.parquet')

# Plot directly - kimsfinance accepts Polars DataFrames
kf.plot(df, type='candle', savefig='chart.webp')
```

**Why Parquet?**
- 10-100x faster than CSV
- Smaller file sizes (compressed)
- Preserves data types
- Column-oriented (efficient for financial data)

---

## Loading from Different Sources

### 1. Parquet Files

**Single file:**
```python
import polars as pl
import kimsfinance as kf

# Load Parquet file
df = pl.read_parquet('btc_usd_1h.parquet')

# Expected columns: open, high, low, close, volume (optional)
# Timestamp column is ignored for charting (uses bar order)

# Plot candlestick chart
kf.plot(df, type='candle', savefig='btc_chart.webp', theme='tradingview')
```

**Multiple files (partitioned data):**
```python
import polars as pl
import kimsfinance as kf

# Load partitioned Parquet files
df = pl.read_parquet('data/year=2024/**/*.parquet')

# Filter and plot
recent = df.filter(pl.col('timestamp') > '2024-01-01')
kf.plot(recent, type='candle', savefig='recent.webp')
```

**Streaming large files:**
```python
import polars as pl
import kimsfinance as kf

# Use lazy loading for large files
df = pl.scan_parquet('massive_dataset.parquet') \
    .filter(pl.col('symbol') == 'BTCUSDT') \
    .tail(500) \
    .collect()

kf.plot(df, type='candle', savefig='last_500_bars.webp')
```

---

### 2. CSV Files

**Basic CSV loading:**
```python
import polars as pl
import kimsfinance as kf

# Load CSV with Polars (faster than pandas)
df = pl.read_csv('ohlcv_data.csv')

kf.plot(df, type='candle', savefig='chart.webp')
```

**CSV with custom columns:**
```python
import polars as pl
import kimsfinance as kf

# CSV has different column names
df = pl.read_csv('market_data.csv') \
    .rename({
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

kf.plot(df, type='ohlc', savefig='ohlc_chart.webp')
```

**CSV with timestamps:**
```python
import polars as pl
import kimsfinance as kf

# Parse timestamps and filter
df = pl.read_csv('historical_data.csv') \
    .with_columns([
        pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    ]) \
    .filter(pl.col('timestamp') > '2024-01-01')

kf.plot(df, type='candle', savefig='2024_data.webp')
```

---

### 3. SQL Databases

**PostgreSQL/MySQL:**
```python
import polars as pl
import kimsfinance as kf
from sqlalchemy import create_engine

# Connect to database
engine = create_engine('postgresql://user:pass@localhost/market_data')

# Query data
query = """
    SELECT open, high, low, close, volume
    FROM ohlcv_1h
    WHERE symbol = 'BTCUSDT'
    ORDER BY timestamp DESC
    LIMIT 500
"""

df = pl.read_database(query, connection=engine)

kf.plot(df, type='candle', savefig='db_chart.webp')
```

**SQLite:**
```python
import polars as pl
import kimsfinance as kf

# Read from SQLite database
df = pl.read_database(
    "SELECT * FROM candles WHERE symbol = 'AAPL' ORDER BY timestamp DESC LIMIT 200",
    connection="sqlite:///market_data.db"
)

kf.plot(df, type='candle', savefig='aapl_chart.webp')
```

---

### 4. REST APIs

**Fetch from Binance API:**
```python
import polars as pl
import kimsfinance as kf
import requests

# Fetch klines from Binance
response = requests.get(
    'https://api.binance.com/api/v3/klines',
    params={
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'limit': 500
    }
)

# Parse response
data = response.json()

# Convert to Polars DataFrame
df = pl.DataFrame({
    'timestamp': [int(x[0]) for x in data],
    'open': [float(x[1]) for x in data],
    'high': [float(x[2]) for x in data],
    'low': [float(x[3]) for x in data],
    'close': [float(x[4]) for x in data],
    'volume': [float(x[5]) for x in data],
})

kf.plot(df, type='candle', savefig='binance_btc.webp', theme='binance')
```

**Fetch from CoinGecko:**
```python
import polars as pl
import kimsfinance as kf
import requests
from datetime import datetime

# Fetch OHLC from CoinGecko
response = requests.get(
    'https://api.coingecko.com/api/v3/coins/bitcoin/ohlc',
    params={'vs_currency': 'usd', 'days': '30'}
)

data = response.json()

# CoinGecko returns [timestamp, open, high, low, close]
df = pl.DataFrame({
    'timestamp': [datetime.fromtimestamp(x[0]/1000) for x in data],
    'open': [x[1] for x in data],
    'high': [x[2] for x in data],
    'low': [x[3] for x in data],
    'close': [x[4] for x in data],
})

kf.plot(df, type='candle', savefig='coingecko_btc.webp', theme='coingecko')
```

---

### 5. Live WebSocket Streams

**Real-time candlestick updates:**
```python
import polars as pl
import kimsfinance as kf
import websocket
import json
from collections import deque

# Store last N candles
candles = deque(maxlen=100)

def on_message(ws, message):
    """Process WebSocket message."""
    data = json.loads(message)

    # Extract OHLCV from Binance kline message
    kline = data['k']
    candles.append({
        'open': float(kline['o']),
        'high': float(kline['h']),
        'low': float(kline['l']),
        'close': float(kline['c']),
        'volume': float(kline['v']),
    })

    # Update chart every 10 candles
    if len(candles) >= 10 and len(candles) % 10 == 0:
        df = pl.DataFrame(list(candles))
        kf.plot(df, type='candle', savefig='live_chart.webp')
        print(f"Chart updated with {len(candles)} candles")

# Connect to Binance WebSocket
ws = websocket.WebSocketApp(
    "wss://stream.binance.com:9443/ws/btcusdt@kline_1m",
    on_message=on_message
)

ws.run_forever()
```

---

### 6. Pandas DataFrames

**Convert from Pandas:**
```python
import pandas as pd
import polars as pl
import kimsfinance as kf

# Load with pandas (e.g., from Excel, HDF5, etc.)
df_pandas = pd.read_excel('market_data.xlsx')

# Convert to Polars (kimsfinance works with both)
df_polars = pl.from_pandas(df_pandas)

# Plot directly
kf.plot(df_polars, type='candle', savefig='chart.webp')

# Or pass pandas DataFrame directly (will be converted internally)
kf.plot(df_pandas, type='candle', savefig='chart.webp')
```

**From pandas with preprocessing:**
```python
import pandas as pd
import kimsfinance as kf

# Load and preprocess with pandas
df = pd.read_csv('data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').resample('1H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Plot resampled data
kf.plot(df.reset_index(), type='candle', savefig='resampled.webp')
```

---

### 7. NumPy Arrays (Lower-Level API)

**Direct NumPy arrays:**
```python
import numpy as np
import kimsfinance as kf
from kimsfinance.plotting import render_and_save

# Generate synthetic data
n = 100
closes = 100 + np.cumsum(np.random.randn(n) * 2)
highs = closes + np.abs(np.random.randn(n) * 1.5)
lows = closes - np.abs(np.random.randn(n) * 1.5)
opens = np.roll(closes, 1)
opens[0] = closes[0]
volumes = np.abs(np.random.randn(n) * 1_000_000)

# Use lower-level API
ohlc_data = {
    'open': opens,
    'high': highs,
    'low': lows,
    'close': closes,
}

render_and_save(
    ohlc=ohlc_data,
    volume=volumes,
    output_path='numpy_chart.webp',
    speed='fast'
)
```

---

## Expected Data Format

### Column Names

kimsfinance expects these column names (case-insensitive):

| Column | Required | Type | Description |
|--------|----------|------|-------------|
| `open` | ✅ Yes | float | Opening price |
| `high` | ✅ Yes | float | High price |
| `low` | ✅ Yes | float | Low price |
| `close` | ✅ Yes | float | Closing price |
| `volume` | ⚠️ Optional | float | Trading volume |
| `timestamp` | ⚠️ Ignored | datetime | Time (used for filtering only) |

**Notes:**
- Volume is optional but recommended for most chart types
- Timestamp is not used for plotting (uses sequential bar order)
- Other columns are ignored

### Data Validation

```python
import polars as pl
import kimsfinance as kf

# Load your data
df = pl.read_parquet('ohlcv.parquet')

# Validate required columns
required = ['open', 'high', 'low', 'close']
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Validate data types
df = df.with_columns([
    pl.col('open').cast(pl.Float64),
    pl.col('high').cast(pl.Float64),
    pl.col('low').cast(pl.Float64),
    pl.col('close').cast(pl.Float64),
])

# Validate OHLC relationships (high >= low, etc.)
invalid = df.filter(
    (pl.col('high') < pl.col('low')) |
    (pl.col('high') < pl.col('open')) |
    (pl.col('high') < pl.col('close')) |
    (pl.col('low') > pl.col('open')) |
    (pl.col('low') > pl.col('close'))
)

if len(invalid) > 0:
    print(f"Warning: {len(invalid)} bars have invalid OHLC relationships")

# Plot validated data
kf.plot(df, type='candle', savefig='validated_chart.webp')
```

---

## Performance Tips

### 1. Use Parquet Instead of CSV

**CSV (slow):**
```python
df = pl.read_csv('large_file.csv')  # 10-15 seconds for 10M rows
```

**Parquet (fast):**
```python
df = pl.read_parquet('large_file.parquet')  # 0.5-1 seconds for 10M rows
```

**Convert CSV to Parquet once:**
```python
import polars as pl

# One-time conversion
pl.read_csv('ohlcv_data.csv') \
    .write_parquet('ohlcv_data.parquet', compression='zstd')

# Future loads are 10-100x faster
df = pl.read_parquet('ohlcv_data.parquet')
```

### 2. Use Lazy Loading for Large Datasets

```python
import polars as pl
import kimsfinance as kf

# Lazy loading - processes only what you need
df = pl.scan_parquet('huge_dataset.parquet') \
    .filter(pl.col('symbol') == 'BTCUSDT') \
    .select(['open', 'high', 'low', 'close', 'volume']) \
    .tail(1000) \
    .collect()  # Only now does it load data

kf.plot(df, type='candle', savefig='chart.webp')
```

### 3. Filter Before Loading

**Bad - loads everything:**
```python
df = pl.read_parquet('all_symbols.parquet')
df = df.filter(pl.col('symbol') == 'AAPL')  # Too late
```

**Good - filters during read:**
```python
df = pl.scan_parquet('all_symbols.parquet') \
    .filter(pl.col('symbol') == 'AAPL') \
    .collect()  # Loads only AAPL data
```

### 4. Downsample Large Datasets

```python
import polars as pl
import kimsfinance as kf

# Load 1-minute data
df = pl.read_parquet('btc_1m.parquet')

# Resample to 1-hour candles (much faster to chart)
df_1h = df.group_by_dynamic('timestamp', every='1h').agg([
    pl.col('open').first(),
    pl.col('high').max(),
    pl.col('low').min(),
    pl.col('close').last(),
    pl.col('volume').sum(),
])

kf.plot(df_1h, type='candle', savefig='1h_chart.webp')
```

---

## Common Patterns

### Pattern 1: Load → Filter → Plot

```python
import polars as pl
import kimsfinance as kf

df = pl.read_parquet('market_data.parquet') \
    .filter(
        (pl.col('symbol') == 'BTCUSDT') &
        (pl.col('timestamp') > '2024-01-01')
    ) \
    .tail(500)

kf.plot(df, type='candle', savefig='recent_btc.webp', theme='tradingview')
```

### Pattern 2: Load → Calculate Indicators → Plot

```python
import polars as pl
import kimsfinance as kf

# Load data
df = pl.read_parquet('ohlcv.parquet')

# Calculate indicators
rsi = kf.calculate_rsi(df['close'].to_numpy(), period=14)
macd_line, signal, histogram = kf.calculate_macd(df['close'].to_numpy())

# Add to plot
addplot = [
    kf.make_addplot(rsi, panel=1, color='purple', ylabel='RSI'),
    kf.make_addplot(macd_line, panel=2, color='blue', ylabel='MACD'),
    kf.make_addplot(signal, panel=2, color='red'),
]

kf.plot(df, type='candle', addplot=addplot, savefig='with_indicators.webp')
```

### Pattern 3: Multi-Symbol Comparison

```python
import polars as pl
import kimsfinance as kf

# Load multiple symbols
btc = pl.read_parquet('btc_usd.parquet').tail(100)
eth = pl.read_parquet('eth_usd.parquet').tail(100)

# Plot side by side
kf.plot(btc, type='candle', savefig='btc_chart.webp', title='BTC/USD')
kf.plot(eth, type='candle', savefig='eth_chart.webp', title='ETH/USD')
```

### Pattern 4: Batch Processing

```python
import polars as pl
import kimsfinance as kf
from pathlib import Path

# Process all Parquet files in a directory
data_dir = Path('market_data/')
output_dir = Path('charts/')
output_dir.mkdir(exist_ok=True)

for parquet_file in data_dir.glob('*.parquet'):
    df = pl.read_parquet(parquet_file).tail(200)
    symbol = parquet_file.stem

    kf.plot(
        df,
        type='candle',
        savefig=output_dir / f'{symbol}_chart.webp',
        title=symbol
    )

    print(f"Generated chart for {symbol}")
```

---

## Troubleshooting

### Issue: "Column 'Open' not found"

**Problem:** Column names are case-sensitive and must be lowercase.

**Solution:**
```python
df = df.rename({
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})
```

### Issue: "ValueError: mismatched array lengths"

**Problem:** OHLC arrays must all have the same length.

**Solution:**
```python
# Check lengths
print(len(df['open']), len(df['high']), len(df['low']), len(df['close']))

# Remove NaN values
df = df.drop_nulls()
```

### Issue: Chart looks wrong (spikes, gaps)

**Problem:** Data may have invalid OHLC relationships or missing values.

**Solution:**
```python
# Validate OHLC relationships
df = df.filter(
    (pl.col('high') >= pl.col('low')) &
    (pl.col('high') >= pl.col('open')) &
    (pl.col('high') >= pl.col('close')) &
    (pl.col('low') <= pl.col('open')) &
    (pl.col('low') <= pl.col('close'))
)

# Remove outliers
df = df.filter(
    (pl.col('close') > 0) &  # Prices must be positive
    (pl.col('volume') >= 0)  # Volume can't be negative
)
```

---

## Summary

kimsfinance is **source-agnostic** - it accepts in-memory data structures:

| Source | Load With | Speed | Recommended |
|--------|-----------|-------|-------------|
| Parquet | `pl.read_parquet()` | ⚡ Very Fast | ✅ Yes |
| CSV | `pl.read_csv()` | 🐌 Slow | ⚠️ Convert to Parquet |
| Database | `pl.read_database()` | ⚡ Fast | ✅ Yes |
| REST API | `requests` + `pl.DataFrame()` | 🌐 Network | ✅ Yes |
| WebSocket | Custom + `pl.DataFrame()` | ⚡ Real-time | ✅ Yes |
| Pandas | `pl.from_pandas()` | ⚡ Fast | ✅ Yes |
| NumPy | Direct arrays | ⚡ Very Fast | ✅ Yes |

**Best practice**: Use Parquet for stored data (10-100x faster than CSV) and Polars for data manipulation (faster than pandas).

---

## Next Steps

- See [OUTPUT_FORMATS.md](OUTPUT_FORMATS.md) for image format options
- See [examples/](../examples/) for complete working examples
- See [API documentation](API.md) for full function reference

**Questions?** Open an issue at https://github.com/kimasplund/kimsfinance/issues
