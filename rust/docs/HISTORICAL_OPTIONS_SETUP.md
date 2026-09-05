# Historical Options Data Collection System

## Overview

This system automatically collects end-of-day options data to build a historical database for backtesting.

**Key Features:**
- Collects EOD (4:15 PM ET) options snapshots daily
- Automatic deduplication (skips if already collected)
- Parallel downloads (16 workers)
- Exponential backoff retry (handles rate limiting)
- Efficient storage (Parquet columnar format)

## Storage Structure

```
data/yfinance/options_historical/
  ├── AAPL/
  │   ├── 2025-10-30.parquet  # All expirations for this date
  │   ├── 2025-10-31.parquet
  │   └── ...
  ├── MSFT/
  │   ├── 2025-10-30.parquet
  │   └── ...
  └── ...
```

**Each file contains:**
- All expirations available on that date
- Both calls and puts
- Full options chain data (bid, ask, volume, OI, Greeks, IV)

**Storage estimates:**
- ~3 MB per day (8 symbols × 20 expirations)
- ~750 MB per year (250 trading days)
- ~1.5 GB for 2 years

## Quick Start

### 1. Install Dependencies

```bash
source ../.venv/bin/activate
pip install yfinance pandas pyarrow
```

### 2. Manual Collection (Test)

```bash
python scripts/download_options_daily_historical.py
```

This will:
- Collect today's EOD options data
- Save to `data/yfinance/options_historical/{symbol}/{date}.parquet`
- Skip if already collected today
- Take ~6 minutes (8 symbols × 20 expirations)

### 3. Query the Database

```python
from scripts.query_options_historical import OptionsHistoricalDB

db = OptionsHistoricalDB()

# Get all options for AAPL on a specific date
df = db.get_options(symbol='AAPL', date='2025-10-30')

# Get specific expiration
df = db.get_options(symbol='AAPL', date='2025-10-30', expiration='2025-11-07')

# Get ATM options (within ±5% of spot)
df = db.get_atm_options(symbol='AAPL', date='2025-10-30', spot_price=220.0, window=5)

# Get date range
df = db.get_options_range(symbol='AAPL', start_date='2025-10-01', end_date='2025-10-31')

# Get available dates
dates = db.get_available_dates('AAPL')

# Get database stats
stats = db.get_stats()
```

## Automated Daily Collection (Cron)

### Setup Cron Job

**When to run:** 4:15 PM ET (16:15), Monday-Friday

**Cron syntax:**
```bash
# Edit crontab
crontab -e

# Add this line (adjust paths):
15 16 * * 1-5 cd /home/kim/projects/kimsfinance/rust && source ../.venv/bin/activate && python scripts/download_options_daily_historical.py >> logs/options_daily.log 2>&1
```

**Explanation:**
- `15 16 * * 1-5` = 4:15 PM, Monday-Friday
- `cd ...` = Change to project directory
- `source ../.venv/bin/activate` = Activate virtualenv
- `>> logs/options_daily.log 2>&1` = Log output

**Create logs directory:**
```bash
mkdir -p logs
```

### Verify Cron Job

```bash
# List cron jobs
crontab -l

# Watch logs (after first run)
tail -f logs/options_daily.log
```

### Timezone Considerations

**Important:** Cron runs in server timezone. If your server is NOT in ET:

**Option 1: Convert time to your timezone**
```bash
# Example: Server is in UTC (ET = UTC-5 during standard time, UTC-4 during DST)
# 4:15 PM ET = 9:15 PM UTC (standard time) or 8:15 PM UTC (DST)
15 21 * * 1-5 ... # Standard time (Nov-Mar)
15 20 * * 1-5 ... # Daylight time (Mar-Nov)
```

**Option 2: Use TZ variable**
```bash
TZ=America/New_York 15 16 * * 1-5 cd ... && python ...
```

**Option 3: Let it run at any consistent time**
- EOD data doesn't change after 4:15 PM ET
- Running at 5 PM, 6 PM, or even midnight is fine
- Just keep it consistent

## Query Examples

### Example 1: Get Options Chain

```python
from scripts.query_options_historical import OptionsHistoricalDB

db = OptionsHistoricalDB()

# Get AAPL options on Oct 30, 2025
df = db.get_options('AAPL', '2025-10-30')

print(f"Total options: {len(df)}")
print(f"Expirations: {df['expiration'].nunique()}")
print(f"Strike range: ${df['strike'].min():.0f} - ${df['strike'].max():.0f}")

# Filter to calls only, Nov 7 expiration
calls = df[(df['optionType'] == 'call') & (df['expiration'] == '2025-11-07')]
print(f"\nNov 7 calls: {len(calls)}")
```

### Example 2: Backtest Covered Call Strategy

```python
db = OptionsHistoricalDB()

# Get historical stock prices
stock_prices = pd.read_parquet('data/yfinance/historical/AAPL/daily.parquet')

# For each trading day
for date, row in stock_prices.iterrows():
    spot_price = row['Close']

    # Get 30-day out calls
    expiration = (pd.Timestamp(date) + pd.Timedelta(days=30)).strftime('%Y-%m-%d')

    try:
        # Get ATM calls (within 5% of spot)
        options = db.get_atm_options('AAPL', date.strftime('%Y-%m-%d'),
                                      spot_price, expiration, window=5)

        # Select closest strike above spot (OTM call)
        otm_calls = options[(options['optionType'] == 'call') &
                           (options['strike'] > spot_price)]

        if not otm_calls.empty:
            best_call = otm_calls.iloc[0]
            premium = (best_call['bid'] + best_call['ask']) / 2

            print(f"{date}: Sell ${best_call['strike']:.0f} call for ${premium:.2f}")
    except FileNotFoundError:
        pass  # No options data for this date yet
```

### Example 3: IV Surface Analysis

```python
db = OptionsHistoricalDB()

# Get IV surface for AAPL on Oct 30, Nov 7 expiration
iv_surface = db.get_iv_surface('AAPL', '2025-10-30', '2025-11-07')

print("Implied Volatility by Strike:")
print(iv_surface[['strike', 'call_iv', 'put_iv']].head(10))

# Check for put-call skew
iv_surface['skew'] = iv_surface['put_iv'] - iv_surface['call_iv']
print(f"\nAverage skew: {iv_surface['skew'].mean():.4f}")
```

## Maintenance

### Check Database Size

```bash
du -sh data/yfinance/options_historical/
du -sh data/yfinance/options_historical/*
```

### Remove Old Data

```bash
# Remove data older than 1 year
find data/yfinance/options_historical/ -name "*.parquet" -mtime +365 -delete
```

### Verify Data Quality

```python
from scripts.query_options_historical import OptionsHistoricalDB

db = OptionsHistoricalDB()
stats = db.get_stats()

print("Database Statistics:")
for symbol, info in stats.items():
    print(f"{symbol}: {info['count']} days ({info['first_date']} to {info['last_date']})")

    # Check for gaps
    dates = db.get_available_dates(symbol)
    date_range = pd.date_range(dates[0], dates[-1], freq='B')  # Business days
    expected = len(date_range)
    actual = len(dates)
    missing = expected - actual

    if missing > 0:
        print(f"  ⚠ Missing {missing} days ({actual}/{expected})")
```

## Troubleshooting

### Problem: Cron job not running

**Check:**
```bash
# Check cron service
sudo systemctl status cron

# Check cron logs
grep CRON /var/log/syslog
tail -f /var/log/syslog | grep CRON
```

**Fix:**
```bash
# Start cron service
sudo systemctl start cron

# Check crontab syntax
crontab -l
```

### Problem: Rate limiting (HTTP 429)

**Symptom:** Logs show "Rate limited (429)" errors

**Fix:** The script automatically retries with exponential backoff. No action needed unless it fails after multiple retries.

**Adjust max expirations:**
```python
# In download_options_daily_historical.py, line 163
max_expirations = 10  # Reduce from 20 to 10
```

### Problem: Missing data for some dates

**Cause:** Markets closed (weekends, holidays), or cron job didn't run

**Fix:**
```bash
# Manually collect for specific date (if within last week)
# Edit script to set custom date, or just run it again
python scripts/download_options_daily_historical.py
```

### Problem: Disk space

**Check:**
```bash
df -h  # Check disk usage
du -sh data/yfinance/options_historical/
```

**Fix:**
```bash
# Remove old data or compress
# Parquet is already compressed, so gzip won't help much
# Best solution: Delete data older than needed
find data/yfinance/options_historical/ -name "*.parquet" -mtime +730 -delete  # 2 years
```

## Advanced Configuration

### Collect Multiple Times Per Day

Edit cron for intraday snapshots:
```bash
# Market open (9:45 AM ET)
45 9 * * 1-5 cd ... && python ...

# Mid-day (12:00 PM ET)
0 12 * * 1-5 cd ... && python ...

# Market close (4:15 PM ET)
15 16 * * 1-5 cd ... && python ...
```

**Note:** Modify script to append time to filename:
```python
# In download_options_daily_historical.py
self.snapshot_date = datetime.now().strftime("%Y-%m-%d_%H%M")  # e.g., 2025-10-30_1615
```

### Collect More Symbols

Edit `download_options_daily_historical.py`, line 153:
```python
symbols = [
    "AAPL", "MSFT", "SPY", "QQQ", "TSLA", "NVDA", "AMZN", "GOOGL",
    # Add more:
    "META", "NFLX", "AMD", "INTC", "DIS", "BA",
]
```

### Collect Fewer Expirations

To reduce API calls and storage:
```python
# In download_options_daily_historical.py, line 157
max_expirations = 10  # Collect only first 10 expirations
```

## Data Schema

Each Parquet file contains:

| Column | Type | Description |
|--------|------|-------------|
| contractSymbol | str | Unique option identifier |
| lastTradeDate | datetime | Last trade timestamp |
| strike | float | Strike price |
| lastPrice | float | Last trade price |
| bid | float | Bid price |
| ask | float | Ask price |
| change | float | Price change |
| percentChange | float | Percent change |
| volume | float | Trading volume |
| openInterest | int | Open interest |
| impliedVolatility | float | Implied volatility (IV) |
| inTheMoney | bool | ITM flag |
| contractSize | str | Contract size |
| currency | str | Currency |
| optionType | str | 'call' or 'put' |
| expiration | str | Expiration date |
| symbol | str | Underlying symbol |
| snapshotDate | str | Date of snapshot |
| downloadTime | str | Download timestamp (ISO 8601) |

## FAQ

**Q: Can I backfill historical data from before I started collecting?**
A: No, Yahoo Finance doesn't provide historical options data. You can only collect going forward. To get past data, you'd need to purchase it from a data vendor.

**Q: How long until I have enough data for backtesting?**
A: Depends on your strategy:
- Covered calls: 30-60 days useful
- Iron condors: 60-90 days useful
- Volatility trading: 6+ months ideal
- Long-term strategies: 1-2+ years

**Q: What if I miss a day?**
A: Not a problem. Backtests can handle gaps in data. Just continue collecting going forward.

**Q: Can I use this for intraday strategies?**
A: Not with EOD data. You'd need to collect multiple times per day (see Advanced Configuration) and modify the storage structure.

**Q: Should I collect on weekends?**
A: No need. Markets are closed, no new data. The cron job runs Mon-Fri only.

**Q: How do I migrate data to a different machine?**
A: Simply copy the entire `data/yfinance/options_historical/` directory. Parquet files are portable.

## Next Steps

1. **Set up cron job** (see "Automated Daily Collection" section)
2. **Wait 30-60 days** to build initial dataset
3. **Start backtesting** with query utilities
4. **Monitor logs** (`tail -f logs/options_daily.log`)
5. **Verify data quality** weekly (check for gaps)

## Related Documentation

- [Query Utilities](../scripts/query_options_historical.py) - Python interface for reading data
- [Collection Script](../scripts/download_options_daily_historical.py) - Daily collector
- [Original Options Downloader](../scripts/download_options_parallel.py) - One-time snapshot collector

## Support

For issues or questions:
1. Check logs: `tail -f logs/options_daily.log`
2. Verify cron: `crontab -l`
3. Test manually: `python scripts/download_options_daily_historical.py`
4. Check database stats: `python scripts/query_options_historical.py`
