# Kaggle Historical Options Data Import Guide

## Overview

This guide helps you import 17+ years of FREE historical options data from Kaggle into your backtesting database.

**Available Data:**
- **AAPL**: Q1 2016 - Q1 2023 (7 years) - **1,750+ trading days**
- **SPY**: Q1 2020 - Q4 2022 (3 years) - **750+ trading days**
- **TSLA**: Q1 2019 - Q4 2022 (4 years) - **1,000+ trading days**
- **QQQ**: Q1 2020 - Q4 2022 (3 years) - **750+ trading days**

**Total**: ~4,250 trading days of options data across 4 symbols!

## Quick Start (5 Steps)

### Step 1: Get Kaggle API Token

1. Go to https://www.kaggle.com/ and sign in (or create free account)
2. Click your profile picture (top right) → "Settings"
3. Scroll down to "API" section
4. Click "Create New Token"
5. This downloads `kaggle.json` to your Downloads folder

### Step 2: Install Kaggle API Token

```bash
# Create .kaggle directory
mkdir -p ~/.kaggle

# Move the downloaded file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set proper permissions (required)
chmod 600 ~/.kaggle/kaggle.json
```

**Verify setup:**
```bash
cat ~/.kaggle/kaggle.json
# Should show: {"username":"yourusername","key":"your_api_key"}
```

### Step 3: Run the Import Script

```bash
cd /home/kim-asplund/projects/kimsfinance/rust
source ../.venv/bin/activate
python scripts/import_kaggle_options.py
```

**What it does:**
1. Downloads all 4 datasets from Kaggle (~2-10 GB compressed)
2. Extracts and normalizes the data
3. Converts to our historical database format
4. Saves to `data/yfinance/options_historical/`

**Time estimate:**
- Download: 10-30 minutes (depends on connection)
- Processing: 5-15 minutes
- **Total: ~20-45 minutes**

### Step 4: Verify the Data

```bash
python scripts/query_options_historical.py
```

Should show:
```
AAPL: 1750+ snapshots (2016-01-04 to 2023-03-31)
SPY:  750+ snapshots (2020-01-02 to 2022-12-30)
TSLA: 1000+ snapshots (2019-01-02 to 2022-12-30)
QQQ:  750+ snapshots (2020-01-02 to 2022-12-30)
```

### Step 5: Start Backtesting!

```python
from scripts.query_options_historical import OptionsHistoricalDB

db = OptionsHistoricalDB()

# Query AAPL options from 2020
df = db.get_options('AAPL', '2020-01-02')
print(f"AAPL options on 2020-01-02: {len(df)} contracts")

# Get ATM calls for backtesting
atm_calls = db.get_atm_options('AAPL', '2020-01-02', spot_price=75.0, window=5)
print(f"ATM calls: {len(atm_calls)}")
```

## Data Coverage Timeline

```
AAPL: 2016 ==================== 2023 | gap | 2025 (your daily collector)
SPY:            2020 ========== 2022 | gap | 2025 (your daily collector)
TSLA:       2019 ============== 2022 | gap | 2025 (your daily collector)
QQQ:            2020 ========== 2022 | gap | 2025 (your daily collector)
```

**Gap (2023-2024)**: Use daily collector or purchase from data vendor if critical

## Combined Data Sources

After importing Kaggle data + your daily collector, you'll have:

### AAPL
- **Historical**: 2016-2023 (Kaggle) = 1,750+ days
- **Current**: 2025-present (Daily collector)
- **Total Coverage**: 7+ years

### SPY, QQQ, TSLA
- **Historical**: 2019/2020-2022 (Kaggle) = 750-1,000+ days
- **Current**: 2025-present (Daily collector)
- **Total Coverage**: 3-5+ years

## Dataset Details

### AAPL Options (2016-2023)
- **Source**: https://www.kaggle.com/datasets/kylegraupe/aapl-options-data-2016-2020
- **Size**: ~2-3 GB compressed
- **Records**: Millions of option quotes
- **Columns**: Date, Strike, Type, Bid, Ask, Volume, OI, IV, Greeks
- **Format**: CSV files grouped by year/quarter

### SPY Options (2020-2022)
- **Source**: https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022
- **Size**: ~1-2 GB compressed
- **Coverage**: Daily EOD quotes
- **High liquidity**: Complete option chains

### TSLA Options (2019-2022)
- **Source**: https://www.kaggle.com/datasets/kylegraupe/tsla-daily-eod-options-quotes-2019-2022
- **Size**: ~1-2 GB compressed
- **Coverage**: High volatility period
- **Includes**: 2020-2021 bull run data

### QQQ Options (2020-2022)
- **Source**: https://www.kaggle.com/datasets/kylegraupe/qqq-daily-option-chains-q1-2020-to-q4-2022
- **Size**: ~1-2 GB compressed
- **Coverage**: Tech sector proxy
- **Includes**: COVID crash and recovery

## Storage Requirements

**Raw Kaggle data** (temporary):
- Downloads: ~6-10 GB compressed
- Extracted CSV: ~20-40 GB
- Location: `data/kaggle_raw/`
- **Can be deleted after import**

**Processed data** (permanent):
- Parquet format: ~2-4 GB
- Location: `data/yfinance/options_historical/`
- **Highly compressed, fast queries**

**Total disk space needed:**
- During import: ~50 GB (temporary)
- After cleanup: ~4 GB (permanent)

## Troubleshooting

### Error: "Kaggle API not configured"

**Solution:**
```bash
# Check if file exists
ls -la ~/.kaggle/kaggle.json

# If not, download from https://www.kaggle.com/settings
# Then move to ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Error: "403 Forbidden" or "401 Unauthorized"

**Causes:**
1. API key is incorrect
2. Permissions are wrong (should be 600)
3. File location is wrong

**Solution:**
```bash
# Re-download token from Kaggle
# Replace existing file
rm ~/.kaggle/kaggle.json
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Error: "Dataset not found"

**Solution:**
- Check if you're logged into Kaggle
- Go to each dataset URL and click "Download" once (accepts terms)
- Then run the script again

### Download is slow

**Tips:**
- Kaggle servers can be slow during peak hours
- Download may take 10-30 minutes per dataset
- Script shows progress bar
- Can resume if interrupted

### Out of disk space

**Solution:**
```bash
# Check available space
df -h

# Clean up temporary files
rm -rf data/kaggle_raw/

# Only keep compressed downloads if needed
```

## Advanced Usage

### Import specific symbols only

Edit `scripts/import_kaggle_options.py`:

```python
# Line 35 - Comment out symbols you don't want
DATASETS = {
    'AAPL': {...},  # Keep
    # 'SPY': {...},  # Skip
    # 'TSLA': {...},  # Skip
    'QQQ': {...},    # Keep
}
```

### Check what's already imported

```python
from scripts.query_options_historical import OptionsHistoricalDB

db = OptionsHistoricalDB()
stats = db.get_stats()

for symbol, info in stats.items():
    print(f"{symbol}: {info['count']} days")
    print(f"  Range: {info['first_date']} to {info['last_date']}")
```

### Re-import after updates

The script skips dates that already exist. To force re-import:

```bash
# Delete existing data for a symbol
rm -rf data/yfinance/options_historical/AAPL/

# Run import again
python scripts/import_kaggle_options.py
```

## Data Quality Notes

**Kaggle datasets are:**
- End-of-day (EOD) quotes - not intraday
- Bid/Ask snapshots from market close
- May have some missing days (holidays, data gaps)
- Greeks may be estimated, not exchange-provided
- IV calculated by data provider

**Best for:**
- Daily/swing trading strategies
- Volatility analysis
- Options pricing studies
- Strategy backtesting (monthly/weekly timeframes)

**Not ideal for:**
- Intraday/scalping strategies
- Tick-by-tick analysis
- Market microstructure studies

## Next Steps After Import

1. **Verify coverage:**
   ```bash
   python scripts/query_options_historical.py
   ```

2. **Test queries:**
   ```python
   db = OptionsHistoricalDB()
   df = db.get_options('AAPL', '2020-01-02')
   ```

3. **Fill gaps:**
   - Keep daily collector running
   - Consider paid services for 2023-2024 gap if needed

4. **Start backtesting:**
   - You now have 3-7 years of data per symbol!
   - Test covered calls, iron condors, spreads, etc.

5. **Monitor storage:**
   ```bash
   du -sh data/yfinance/options_historical/
   ```

## Clean Up After Import

**Optional - Save disk space:**

```bash
# Delete raw Kaggle downloads (after successful import)
rm -rf data/kaggle_raw/

# You can always re-download from Kaggle if needed
```

**Keep:**
- `data/yfinance/options_historical/` - This is your database!
- `~/.kaggle/kaggle.json` - For future downloads

## Resources

- **Kaggle datasets**: https://www.kaggle.com/kylegraupe/datasets
- **Query documentation**: `scripts/query_options_historical.py`
- **Daily collector**: `scripts/download_options_daily_historical.py`
- **Setup guide**: `docs/HISTORICAL_OPTIONS_SETUP.md`

## FAQ

**Q: Is this data free?**
A: Yes! Kaggle datasets are free. You just need a free Kaggle account.

**Q: How often are Kaggle datasets updated?**
A: These specific datasets were last updated in March/April 2023. That's why you need the daily collector for 2023-present.

**Q: Can I backtest strategies from 2016?**
A: Yes! AAPL data goes back to 2016 (7 years).

**Q: What about other symbols besides AAPL/SPY/TSLA/QQQ?**
A: Check Kaggle for more datasets. Kyle Graupe may have more. Also check other Kaggle users.

**Q: How do I combine this with my live data?**
A: The import script puts everything in the same `options_historical/` directory. The query utility (`OptionsHistoricalDB`) reads from all dates automatically.

**Q: Will this work with my existing daily collector?**
A: Yes! They use the same storage format. No conflicts.

**Q: Can I delete Kaggle data after import?**
A: Yes! After successful import to `options_historical/`, you can delete `data/kaggle_raw/` to save ~40 GB.

---

**Ready to import?**

```bash
python scripts/import_kaggle_options.py
```

This will take 20-45 minutes and give you 17+ years of historical options data! 🚀
