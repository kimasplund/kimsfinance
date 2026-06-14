# Historical Options Data Sources

## Overview

This document summarizes all available data sources for options historical data, including free and paid options.

## Free Data Sources (✅ Working)

### 1. Kaggle Historical Datasets
**Status**: ✅ Implemented
**Script**: `scripts/import_kaggle_options.py`
**Documentation**: `docs/KAGGLE_IMPORT_GUIDE.md`

**Coverage**:
- **AAPL**: Q1 2016 - Q4 2025 (9+ years, 1,824 trading days)
- **SPY**: Q1 2020 - Q4 2025 (3+ years, 759 trading days)
- **TSLA**: Q1 2019 - Q4 2025 (4+ years, 1,011 trading days)
- **QQQ**: Q1 2021 - Q4 2025 (3+ years, 509 trading days)

**Total**: 4,103 trading days across 4 symbols

**Data Type**: End-of-day (EOD) options chains
**Format**: CSV → Parquet
**Cost**: Free
**Setup Time**: 5 minutes (Kaggle API token)
**Import Time**: 20-45 minutes

**Pros**:
- Completely free
- Years of historical data
- No rate limits after download
- All expirations included
- Greeks and IV included

**Cons**:
- One-time download (not updated)
- Data gaps (2023-2024 missing)
- Limited to 4 symbols
- No intraday data

**Usage**:
```bash
python scripts/import_kaggle_options.py
```

### 2. Yahoo Finance (yfinance)
**Status**: ✅ Implemented
**Scripts**:
- Daily collection: `scripts/download_options_daily_historical.py`
- Parallel download: `scripts/download_options_parallel.py`
- Query utilities: `scripts/query_options_historical.py`

**Documentation**: `docs/HISTORICAL_OPTIONS_SETUP.md`

**Coverage**:
- **Current data only** (no historical)
- **8 symbols**: AAPL, MSFT, SPY, QQQ, TSLA, NVDA, AMZN, GOOGL
- Collects at 4:15 PM ET daily (via cron)

**Data Type**: End-of-day (EOD) options chains
**Format**: Parquet
**Cost**: Free
**Rate Limiting**: ~5 requests/sec (handled with backoff)

**Pros**:
- Completely free
- No API key needed
- Builds database over time
- Up to 20 expirations per symbol
- Greeks and IV included

**Cons**:
- No historical data (Yahoo doesn't provide it)
- Must collect daily going forward
- Rate limiting (HTTP 429)
- Sometimes unreliable

**Usage**:
```bash
# One-time snapshot
python scripts/download_options_parallel.py

# Daily collection (add to cron)
python scripts/download_options_daily_historical.py
```

**Cron Setup** (4:15 PM ET, Mon-Fri):
```bash
15 16 * * 1-5 cd /home/kim/projects/kimsfinance/rust && source ../.venv/bin/activate && python scripts/download_options_daily_historical.py >> logs/options_daily.log 2>&1
```

## Paid/Limited Data Sources

### 3. Polygon.io
**Status**: ❌ Free tier insufficient
**Script**: `scripts/download_polygon_options.py`
**API Key**: Set via `POLYGON_API_KEY` environment variable

**Free Tier Access**:
- ✅ Options contracts list
- ❌ Options chain snapshots (403 Forbidden)
- ❌ Historical options data

**Paid Plans** (Required for options data):
- **Starter**: $99/month - Limited options data
- **Developer**: $249/month - Full options access
- **Advanced**: $499+/month - Flat files included

**Cost Analysis**: NOT RECOMMENDED for free tier users

**Pros** (Paid plans):
- Real-time data
- Tick-by-tick history
- Flat file downloads
- All exchanges (17 US options exchanges)

**Cons**:
- Expensive ($99-$499+/month)
- Free tier useless for options
- Requires paid subscription

**Verdict**: Skip unless you need real-time/intraday data and have budget.

### 4. Interactive Brokers (IBKR)
**Status**: ✅ Implemented (Rust)
**Module**: `src/data/ibkr/chunked.rs`
**Example**: `examples/test_ibkr_all_instruments.rs`

**Requirements**:
- IBKR account (free to open)
- TWS or IB Gateway running
- Market data subscription (varies)

**Coverage**:
- Historical options data via API
- All symbols (if subscribed)
- Intraday and EOD data

**Data Type**: OHLCV bars or trades
**Format**: Rust structs → Parquet
**Cost**: Account required + data subscriptions

**Pros**:
- Professional-grade data
- Intraday available
- All asset classes (stocks, options, futures, forex, crypto)
- Direct from broker

**Cons**:
- Requires IBKR account
- Market data subscriptions cost money
- TWS/Gateway must be running
- API learning curve

**Usage**:
```bash
cargo run --features data-ibkr --example test_ibkr_all_instruments
```

## Recommended Strategy

For most users building a historical database:

### Phase 1: Bootstrap (Day 1)
1. **Import Kaggle datasets** (20-45 minutes)
   - Gets you 3-7 years of data for 4 symbols
   - Run: `python scripts/import_kaggle_options.py`

2. **Verify import** (1 minute)
   - Run: `python scripts/query_options_historical.py`

### Phase 2: Ongoing Collection (Daily)
1. **Set up cron job** for Yahoo Finance daily collector
   - Collects at 4:15 PM ET, Mon-Fri
   - Fills the gap from 2023-present
   - Builds database going forward

2. **Monitor** weekly
   - Check for missing dates
   - Verify cron is running
   - Review logs

### Phase 3: Expand (Optional)
1. **Add more symbols** to Yahoo collector
   - Edit `scripts/download_options_daily_historical.py`
   - Add to `symbols` list (line 179)

2. **Fill 2023-2024 gap** (if critical)
   - Search Kaggle for more recent datasets
   - Consider paid data vendor
   - Or wait for daily collector to build up

## Data Quality Comparison

| Source | Quality | Coverage | Latency | Reliability | Cost |
|--------|---------|----------|---------|-------------|------|
| **Kaggle** | ⭐⭐⭐⭐ | 3-7 years (4 symbols) | One-time | ⭐⭐⭐⭐⭐ | Free |
| **Yahoo Finance** | ⭐⭐⭐ | Daily going forward | EOD + 15min | ⭐⭐⭐ | Free |
| **Polygon.io (Paid)** | ⭐⭐⭐⭐⭐ | Real-time + historical | Real-time | ⭐⭐⭐⭐⭐ | $99-$499/mo |
| **IBKR** | ⭐⭐⭐⭐⭐ | All historical | Real-time | ⭐⭐⭐⭐⭐ | Account + subs |

## Storage Structure

All data sources save to the same unified format:

```
data/yfinance/options_historical/
  ├── AAPL/
  │   ├── 2016-01-04.parquet  # Kaggle
  │   ├── 2016-01-05.parquet  # Kaggle
  │   ├── ...
  │   ├── 2023-03-31.parquet  # Kaggle (last day)
  │   ├── [GAP: 2023-04-01 to 2025-10-29]
  │   └── 2025-10-30.parquet  # Yahoo daily collector
  ├── SPY/
  │   └── ...
  └── ...
```

**Schema** (unified across all sources):
- `contractSymbol`: Option contract identifier
- `strike`: Strike price
- `expiration`: Expiration date
- `optionType`: 'call' or 'put'
- `bid`, `ask`, `lastPrice`: Pricing data
- `volume`, `openInterest`: Market data
- `impliedVolatility`: IV
- `delta`, `gamma`, `theta`, `vega`, `rho`: Greeks
- `snapshotDate`: Date of snapshot
- `symbol`: Underlying symbol
- `downloadTime`: When data was collected

## Query Interface

All data can be queried using the same Python interface:

```python
from scripts.query_options_historical import OptionsHistoricalDB

db = OptionsHistoricalDB()

# Get all available dates for AAPL
dates = db.get_available_dates('AAPL')

# Get options chain for specific date
df = db.get_options('AAPL', '2020-01-02')

# Get ATM options (±5% of spot)
atm = db.get_atm_options('AAPL', '2020-01-02', spot_price=75.0, window=5)

# Get date range
df = db.get_options_range('AAPL', '2020-01-01', '2020-12-31')

# Get IV surface
iv_surface = db.get_iv_surface('AAPL', '2020-01-02', '2020-02-21')

# Get database stats
stats = db.get_stats()
```

## Summary

**Best Free Setup**:
1. Kaggle datasets (one-time import) → 3-7 years historical
2. Yahoo Finance daily collector → Ongoing collection

**Total Cost**: $0
**Total Coverage**: 7+ years (and growing)
**Setup Time**: ~1 hour
**Maintenance**: Automatic (cron)

**For professionals needing real-time/intraday**:
- Polygon.io ($99-$499/month)
- IBKR (account + subscriptions)

---

**Last Updated**: 2025-10-30
**Status**: Kaggle import complete (4,103 days), Yahoo daily collector active
