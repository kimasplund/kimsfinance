# Paper Trading Deployment Guide

## Quick Answer: Architecture

Your bull put spread strategy **runs within the kimsfinance framework** but as a **standalone executable** that you can run daily.

```
paper_trading_scanner (executable)
    ↓ depends on
kimsfinance_core (library)
    ↓ provides
- Strategy framework
- Data loaders
- Black-Scholes calculator
- Transaction cost modeling
- Market regime detection
```

You don't need to understand the kimsfinance codebase to use the scanner - just run it!

---

## Daily Workflow (20 minutes)

### Morning Scan (Before Market Open)

```bash
cd /home/kim-asplund/projects/kimsfinance/rust

# Run the scanner
cargo run --release --features data-downloaders --example paper_trading_scanner

# Output: Trade opportunities ranked by quality
```

**What the scanner does**:
1. Loads latest market data (OHLCV + options)
2. Detects market regime (bull/bear, high/low vol)
3. Scans SPY, QQQ, AAPL, TSLA for bull put spread opportunities
4. Filters by proven strategy parameters (266% ROC)
5. Ranks opportunities by risk-reward
6. Shows exact IBKR order entry instructions

### Example Output

```
=== TRADE OPPORTUNITIES (Ranked Best to Worst) ===

┌─ Opportunity #1 - SPY ─────────────────────────
│
│ POSITION DETAILS:
│   Short PUT: $445.00 (delta: -0.25)
│   Long PUT:  $440.00
│   Width:     $5.00
│   DTE:       35 days
│   Expiration: 2024-12-20
│
│ FINANCIALS:
│   Credit Received:    $0.60 ($60 per contract)
│   Max Profit:         $60 per contract
│   Max Risk:           $440 per contract
│   Margin Required:    $500 per contract
│
│ RISK METRICS:
│   Risk/Capital:       4.40%
│   Margin/Capital:     5.00%
│   Credit/Width:       12.0%
│
│ EXIT TARGETS:
│   Profit Target (50%): Close at $0.30 debit
│   Stop Loss (200%):    Close at $1.20 debit
│   Max Hold:            42 days (from entry)
│
│ IBKR ORDER ENTRY:
│   1. Create vertical spread
│   2. Sell 1 2024-12-20 SPY PUT $445.00
│   3. Buy 1 2024-12-20 SPY PUT $440.00
│   4. Limit Order: $0.60 CREDIT (or better)
│   5. Time in Force: DAY ORDER
│
│ ALERTS TO SET:
│   - Profit target: Spread value drops to $0.30
│   - Stop loss: Spread value rises to $1.20
│   - Time exit: 42 days from entry
│
└─────────────────────────────────────────────────
```

---

## IBKR Paper Trading Setup

### One-Time Setup (15 minutes)

1. **Open IBKR Paper Trading Account** (already done ✓)
   - You mentioned "ibkr is open"
   - If not registered: https://www.interactivebrokers.com → Paper Trading

2. **Login to Trader Workstation (TWS)**
   - Download: https://www.interactivebrokers.com/en/trading/tws.php
   - Use paper trading credentials
   - Or use web platform: https://ndcdyn.interactivebrokers.com/sso/Login

3. **Fund Paper Account**
   - Set virtual capital to $10,000 (matches backtest)
   - Settings → Account → Paper Trading

4. **Create Trade Log Spreadsheet**
   - Copy template from PAPER_TRADING_GUIDE.md section "Trade Tracking Spreadsheet"
   - Google Sheets or Excel
   - Required fields: Trade ID, Entry Date, Strikes, Credit, Exit, P&L

---

## Entering Trades in IBKR

### Step-by-Step Order Entry

The scanner output provides exact instructions. Here's the general process:

**1. Open Option Trader Window**
- TWS → Trading → Option Trader
- Enter symbol (e.g., SPY)
- Select expiration date shown in scanner output

**2. Create Vertical Spread**
- Right-click on short PUT strike → Buy/Sell → Sell
- Right-click on long PUT strike → Buy/Sell → Buy
- TWS auto-creates spread order

**3. Configure Order**
- Order Type: **Limit Order** (not market!)
- Credit: Enter amount from scanner (e.g., $0.60)
- Quantity: 1 contract
- Time in Force: **DAY** (do not use GTC)

**4. Review and Submit**
- Check strikes match scanner exactly
- Verify credit matches (or better)
- Submit order

**5. Set Alerts**
- Right-click spread position → Create Alert
- Profit target: Spread value = 50% of entry credit
- Stop loss: Spread value = 200% of entry credit

**6. Log Trade**
- Record in trade log immediately
- Include: Entry date, strikes, DTE, credit, exit targets

---

## Monitoring Trades

### Mid-Day Check (10 minutes)

**Time**: 12:00-2:00 PM ET

```bash
# Optional: Create a position monitor script
# For now, manually check TWS:

1. Open Positions window in TWS
2. Check current spread values
3. Close if profit target or stop loss hit
4. Update trade log
```

**What to look for**:
- Spread value dropped to 50% of entry → **Close for profit**
- Spread value doubled (200% loss) → **Close to limit loss**
- Days in trade > 42 → **Close regardless of P&L**

### Evening Review (5 minutes)

**Time**: After market close (4:00 PM ET)

1. Update trade log with day's activity
2. Calculate current P&L
3. Check for upcoming expirations (close before expiration!)
4. Plan tomorrow's scan

---

## Example Trade Lifecycle

### Day 1: Entry
```
Scanner identifies: SPY $445/$440 bull put spread
- Credit: $0.60
- Max Risk: $440
- DTE: 35 days

Enter order in IBKR:
- Sell 1 SPY Dec 20 $445 PUT
- Buy 1 SPY Dec 20 $440 PUT
- Limit: $0.60 credit
- Filled at $0.61 (better than limit!)

Set alerts:
- Profit target: $0.305 (50%)
- Stop loss: $1.22 (200%)

Log trade:
- Trade ID: BPS_001
- Entry Date: 2024-11-15
- Entry Credit: $0.61
- Exit targets set
```

### Day 12: Profit Target Hit
```
Alert triggers: Spread value = $0.30

Close position:
- Buy 1 SPY Dec 20 $445 PUT
- Sell 1 SPY Dec 20 $440 PUT
- Limit: $0.30 debit
- Filled at $0.29 (better!)

P&L:
- Entry credit: $0.61
- Exit debit: $0.29
- Profit: $0.32 per spread = $32
- ROI: $32 / $500 margin = 6.4% in 12 days
- Annualized: ~195%

Update trade log:
- Exit Date: 2024-11-27
- Exit Debit: $0.29
- Net P&L: $32
- Days in Trade: 12
- Exit Reason: Profit Target
```

---

## Running the Scanner Daily

### Automation Options

**Option 1: Manual (Recommended for first month)**
```bash
# Every morning before market open
cd /home/kim-asplund/projects/kimsfinance/rust
cargo run --release --features data-downloaders --example paper_trading_scanner > ~/Desktop/today_opportunities.txt
```

**Option 2: Cron Job (After 1 month validation)**
```bash
# Add to crontab (crontab -e):
0 8 * * 1-5 cd /home/kim-asplund/projects/kimsfinance/rust && cargo run --release --features data-downloaders --example paper_trading_scanner > ~/Desktop/today_opportunities.txt
```

**Option 3: Live API (Future - requires IBKR API integration)**
- Automated order placement
- Real-time monitoring
- Automatic exits
- Requires additional development

---

## Data Requirements

The scanner needs historical market data to function:

**Already Available** (from previous work):
- ✅ 9,884 days of OHLCV data (AAPL, SPY, TSLA, QQQ)
- ✅ 4,103 days of options data from Kaggle

**Daily Updates** (optional but recommended):
```bash
# Update OHLCV data (latest spot prices)
cd scripts
python download_historical_parallel.py

# Update options data (current chains)
python download_options_daily_historical.py
```

**Note**: For paper trading, slightly stale data (1-2 days old) is acceptable. The scanner looks for structural opportunities (delta, DTE, regime) which don't change drastically day-to-day.

---

## Troubleshooting

### Scanner Returns "No Opportunities"

**Causes**:
1. **Bear market regime** → Strategy skips trading in bear markets
2. **No options meeting criteria** → Adjust DTE/delta ranges
3. **Risk limits too strict** → Currently set to 5% (very conservative)

**Solutions**:
```bash
# Check market regime
grep "Market Regime" ~/Desktop/today_opportunities.txt

# If bear market, wait for bullish conditions
# If no candidates, data might be stale - update:
cd scripts && python download_options_daily_historical.py
```

### IBKR Order Rejected

**Common Issues**:
1. **Insufficient margin** → Reduce position size or check paper account balance
2. **No bid/ask** → Option too illiquid, skip this trade
3. **Price moved** → Limit order too strict, adjust credit slightly

### Scanner Fails to Run

**Error**: "failed to load data"
```bash
# Check data directories exist
ls data/yfinance/options_historical/
ls ../data/yfinance/ohlcv/

# Re-download if missing
cd scripts
python download_historical_parallel.py
python download_options_parallel.py
```

---

## Performance Tracking

### Weekly Review (30 minutes every Friday)

```bash
# Calculate metrics from trade log:
1. Total P&L this week
2. Win rate (wins / total trades)
3. Average profit per winner
4. Average loss per loser
5. Sharpe ratio estimate

# Compare to backtest targets:
- Win Rate: 67% (target)
- Profit Factor: 2.45 (target)
- Average Days: 28.3 (target)
```

### Monthly Decision Point

**After 1 Month (Minimum 8 trades)**:

```
If paper trading shows:
✅ Win rate > 55%
✅ Sharpe ratio > 1.0
✅ Fill rate > 90%
✅ Max drawdown < 25%

→ Continue to Month 2-3

If metrics below targets:
⚠️  Analyze deviations from backtest
⚠️  Adjust parameters if needed
⚠️  Continue paper trading

If major issues:
❌ Stop and investigate
❌ Review PROFITABILITY_REPORT.md
❌ Check data quality
```

**After 3 Months (Minimum 20 trades)**:

```
If all targets met:
→ Proceed to live trading at 50% scale

If marginal performance:
→ Continue paper trading another 3 months

If poor performance:
→ Strategy validation failed, do not go live
```

---

## File Locations

**Scanner Executable**:
```
/home/kim-asplund/projects/kimsfinance/rust/examples/paper_trading_scanner.rs
```

**Documentation**:
```
/home/kim-asplund/projects/kimsfinance/rust/docs/
├── PAPER_TRADING_GUIDE.md          ← Entry/exit checklists
├── PAPER_TRADING_DEPLOYMENT.md     ← This file (deployment)
└── PROFITABILITY_REPORT.md         ← 266% ROC validation
```

**Data Directories**:
```
/home/kim-asplund/projects/kimsfinance/rust/data/yfinance/
├── options_historical/              ← Options chains
└── ohlcv/                          ← Spot prices (parent dir)
```

**Strategy Code** (for reference, not required to understand):
```
/home/kim-asplund/projects/kimsfinance/rust/src/strategy/
├── types.rs                        ← Data structures
├── strategies.rs                   ← Bull put spread logic
├── backtest.rs                     ← Backtesting engine
├── data_loader.rs                  ← Options data loader
├── spot_data.rs                    ← OHLCV loader
├── black_scholes.rs                ← IV calculator
├── transaction_costs.rs            ← Cost modeling
├── market_regime.rs                ← Regime detector
└── metrics.rs                      ← Performance metrics
```

---

## Next Steps

1. ✅ **Scanner created** - You can run it now!
2. ⏳ **Run first scan** - Execute the scanner to see today's opportunities
3. ⏳ **Enter first paper trade** - Follow IBKR instructions from scanner output
4. ⏳ **Set up trade log** - Use template from PAPER_TRADING_GUIDE.md
5. ⏳ **Monitor daily** - Morning scan + mid-day check + evening log

---

## Support

- **Scanner issues**: Check troubleshooting section above
- **Strategy questions**: See PROFITABILITY_REPORT.md for proven parameters
- **IBKR help**: See PAPER_TRADING_GUIDE.md for detailed order entry
- **Code questions**: See src/strategy/ directory (not required for trading)

---

**You're ready to start paper trading!**

Run the scanner now:
```bash
cd /home/kim-asplund/projects/kimsfinance/rust
cargo run --release --features data-downloaders --example paper_trading_scanner
```

The scanner will show you exact order entry instructions for IBKR. Good luck! 🎯
