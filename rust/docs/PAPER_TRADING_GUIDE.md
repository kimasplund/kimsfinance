# Paper Trading Implementation Guide

**Strategy**: Bull Put Spread
**Status**: Ready for paper trading validation
**Proven Performance**: 266% ROC over 4 years (with transaction costs)

---

## Quick Start Checklist

- [ ] Open paper trading account (IBKR or ToS)
- [ ] Fund with virtual $10,000
- [ ] Set up daily monitoring (20-30 min/day)
- [ ] Configure risk limits (5% per trade, 50% margin)
- [ ] Run backtest example to understand mechanics
- [ ] Execute first 5 paper trades
- [ ] Review after 1 month

---

## Paper Trading Platforms

### 1. Interactive Brokers (Recommended)

**Pros**:
- Professional-grade platform
- Real market data
- Free paper trading account
- API access for automation
- Matches live trading exactly

**Cons**:
- Complex interface (learning curve)
- Requires account verification

**Setup**:
1. Visit https://www.interactivebrokers.com
2. Click "Open Account" → "Paper Trading"
3. Complete registration (no funding required)
4. Download TWS (Trader Workstation) or use web platform
5. Fund paper account with $10,000 virtual capital

### 2. TD Ameritrade - Thinkorswim

**Pros**:
- User-friendly interface
- Excellent analysis tools
- Free paper trading (paperMoney)
- No account funding required

**Cons**:
- API access requires live account
- Manual execution only

**Setup**:
1. Visit https://www.tdameritrade.com/tools-and-platforms/thinkorswim.html
2. Download thinkorswim platform
3. Click "Login" → "paperMoney" tab
4. Create username/password
5. Paper account funded automatically

### 3. Tastytrade

**Pros**:
- Options-focused platform
- Built-in Greeks and analytics
- Free demo account

**Cons**:
- Smaller market share
- Less educational resources

---

## Daily Trading Workflow

### Morning Routine (15 minutes)

**Time**: Before market open (9:00 AM ET)

1. **Check Market Conditions**
   ```
   - SPY trend: Bullish/Bearish/Sideways?
   - VIX level: <20 (low vol) or >20 (high vol)?
   - Recent news: Any major events?
   ```

2. **Review Open Positions**
   ```
   - Any positions near stop loss?
   - Any positions hit profit target?
   - Days in trade for each position
   - Current P&L status
   ```

3. **Scan for New Opportunities**
   ```
   - Look for 30-45 DTE options expiring in 4-6 weeks
   - Find short puts with 0.15-0.35 delta (15-35% OTM)
   - Check strike width and credit received
   - Verify liquidity (volume >100, OI >1000)
   ```

### Intraday Monitoring (10 minutes)

**Time**: Mid-day (12:00-2:00 PM ET)

1. **Check Position Status**
   - Any stop losses triggered?
   - Any profit targets hit?
   - Update P&L tracking

2. **Adjust Alerts**
   - Set price alerts for stop losses
   - Set alerts for profit targets

### Evening Review (5 minutes)

**Time**: After market close (4:00 PM ET)

1. **Update Trading Log**
   - Record any trades executed
   - Note entry/exit prices
   - Calculate actual slippage vs expected

2. **Plan for Tomorrow**
   - Any positions needing attention?
   - New signals to investigate?

---

## Entry Checklist

Before entering any bull put spread position, verify:

### Position Criteria

- [ ] **DTE**: 30-45 days to expiration
- [ ] **Delta**: Short put delta between -0.15 and -0.35
- [ ] **Credit**: Minimum $0.20 credit per spread
- [ ] **Strike Width**: $5-10 wide (for $10K account)
- [ ] **Liquidity**: Volume >100, OI >1000 for short strike
- [ ] **Spread**: Bid-ask spread <10% of mid price

### Risk Management

- [ ] **Position Size**: <5% of capital at risk
- [ ] **Margin**: <50% total margin utilization
- [ ] **Open Positions**: <10 concurrent positions
- [ ] **Correlation**: Not overexposed to single symbol

### Market Conditions

- [ ] **Trend**: Bullish or neutral (avoid in bear markets)
- [ ] **Volatility**: Preferably low to moderate (VIX <25)
- [ ] **Earnings**: No earnings within position duration

---

## Entry Example: Step-by-Step

### Scenario
- **Date**: Monday, November 15, 2024
- **Symbol**: SPY (S&P 500 ETF)
- **Spot Price**: $450
- **Account Size**: $10,000
- **Target Risk**: 5% ($500 max risk)

### Step 1: Find Expiration
Look for options expiring in 30-45 days:
- Current date: Nov 15
- Target expiration: Dec 20 (35 DTE) ✅

### Step 2: Find Short Put Strike
Find put with delta -0.20 to -0.30:
- $435 PUT: Delta -0.25 ✅
- Bid/Ask: $2.40 / $2.50
- Volume: 5,000, OI: 15,000 ✅

### Step 3: Find Long Put (Protection)
Find put $5-10 below short strike:
- $430 PUT: Delta -0.15
- Bid/Ask: $1.80 / $1.90
- Volume: 3,000, OI: 10,000 ✅

### Step 4: Calculate Credit and Risk
```
Credit = (Short Put Bid) - (Long Put Ask)
       = $2.40 - $1.90
       = $0.50 per spread
       = $50 per contract (× 100 multiplier)

Width = Short Strike - Long Strike
      = $435 - $430
      = $5

Max Risk = (Width - Credit) × 100
         = ($5 - $0.50) × 100
         = $450 per spread

Margin Required = Width × 100 = $500
```

### Step 5: Verify Risk Management
```
Risk per trade = $450 / $10,000 = 4.5% ✅ (< 5% limit)
Margin = $500 / $10,000 = 5% ✅ (< 50% limit)
```

### Step 6: Place Order
**Order Type**: Vertical Spread
- Sell 1 SPY Dec 20 $435 PUT @ $2.40
- Buy 1 SPY Dec 20 $430 PUT @ $1.90
- **Net Credit**: $0.50 (minimum)
- **Order Type**: Limit Order (Credit $0.50 or better)
- **Time in Force**: Day Order

### Step 7: Set Alerts
```
Profit Target = 50% of max profit
              = $50 × 50% = $25
              → Close when credit drops to $0.25

Stop Loss = 200% of credit
          = $50 × 200% = $100 loss
          → Close when debit reaches $1.50
```

---

## Exit Checklist

Exit positions when ANY of the following occur:

### Profit Target (Priority 1)
- [ ] **50% Profit**: Credit dropped to 50% of entry
  - **Example**: Entered at $0.50 → Exit at $0.25
  - **Action**: Place buy-to-close order at $0.25 limit

### Stop Loss (Priority 2)
- [ ] **200% Loss**: Debit reached 200% of credit
  - **Example**: Entered at $0.50 credit → Exit at $1.00 debit
  - **Action**: Close immediately at market

### Time-Based Exit (Priority 3)
- [ ] **42 Days in Trade**: Max holding period reached
  - **Action**: Close at current market price
  - **Reason**: Theta decay diminishes, avoid expiration risk

### Expiration Exit (Priority 4)
- [ ] **At Expiration**: Options expire today
  - **Action**: Close before 3:00 PM ET to avoid assignment
  - **Reason**: Avoid physical settlement and fees

---

## Exit Example: Profit Target Hit

### Scenario
- **Entry Date**: Nov 15
- **Current Date**: Dec 1 (16 days in trade)
- **Entry Credit**: $0.50
- **Current Credit**: $0.25
- **Profit**: $25 per spread

### Action: Close Position
**Order**:
- Buy 1 SPY Dec 20 $435 PUT
- Sell 1 SPY Dec 20 $430 PUT
- **Net Debit**: $0.25 (max)
- **Order Type**: Limit Order

### Result
```
Profit = Entry Credit - Exit Debit
       = $0.50 - $0.25
       = $0.25 per spread
       = $25 per contract

ROI = Profit / Margin Required
    = $25 / $500
    = 5% in 16 days
    = ~114% annualized ✅
```

---

## Trade Tracking Spreadsheet

### Required Fields

| Field | Example | Purpose |
|-------|---------|---------|
| Trade ID | BPS_001 | Unique identifier |
| Symbol | SPY | Underlying |
| Entry Date | 2024-11-15 | Track duration |
| Exit Date | 2024-12-01 | Calculate holding period |
| Short Strike | $435 | Position detail |
| Long Strike | $430 | Position detail |
| DTE at Entry | 35 | Track parameter |
| Short Delta | -0.25 | Track parameter |
| Entry Credit | $0.50 | P&L calculation |
| Exit Debit | $0.25 | P&L calculation |
| Commissions | $4.80 | Transaction costs |
| Net P&L | $20.20 | After costs |
| Days in Trade | 16 | Performance metric |
| Exit Reason | Profit Target | Strategy validation |
| Slippage | $0.05 | Cost analysis |

### Sample Google Sheets Template

```
=GOOGLEFINANCE("SPY", "price")                    // Current price
=(C2-D2)*100-E2                                   // Net P&L
=IF(F2>0, "Win", "Loss")                          // Win/Loss
=COUNTIF(G:G, "Win")/COUNTA(G:G)                  // Win Rate
```

---

## Weekly Review Process

### Every Friday Evening (30 minutes)

1. **Calculate Weekly Metrics**
   ```
   - Total P&L this week
   - Win rate this week
   - Average profit/loss per trade
   - Capital utilization (avg margin%)
   ```

2. **Compare to Backtest**
   ```
   Target Metrics (from backtest):
   - Win Rate: 67%
   - Profit Factor: 2.45
   - Avg Days in Trade: 28.3
   - Sharpe Ratio: 1.40

   Current Paper Trading:
   - Win Rate: ___%
   - Profit Factor: ___
   - Avg Days: ___
   - Deviations from target?
   ```

3. **Identify Issues**
   ```
   - Slippage higher than expected?
   - Fill rates below 90%?
   - Stop losses triggered too often?
   - Profit targets not hitting?
   ```

4. **Adjustment Plan**
   ```
   - Parameter tweaks needed?
   - Position sizing adjustments?
   - Different symbols to try?
   ```

---

## Month 1 Milestones

### Week 1: Setup & First Trades
- [ ] Paper trading account opened
- [ ] Platform familiarized
- [ ] 2-3 practice trades executed
- [ ] Trade logging system established

### Week 2: Build Position
- [ ] 3-5 total positions opened
- [ ] Risk limits enforced (5% per trade)
- [ ] First exits executed (profit or stop)
- [ ] Slippage and costs tracked

### Week 3: Refine Process
- [ ] 5-7 total positions (ongoing)
- [ ] Entry timing optimized
- [ ] Exit discipline validated
- [ ] Comparison to backtest

### Week 4: Month 1 Review
- [ ] Minimum 8-10 trades completed
- [ ] Win rate calculated (target: >55%)
- [ ] Sharpe ratio estimated (target: >1.0)
- [ ] Decision: Continue or adjust?

---

## Common Mistakes to Avoid

### 1. Over-Trading
- **Mistake**: Taking every signal, >10 positions
- **Fix**: Max 10 concurrent positions, quality > quantity

### 2. Ignoring Stop Losses
- **Mistake**: "I'll wait for it to recover"
- **Fix**: Set alerts, honor stops religiously

### 3. Chasing Profit Targets
- **Mistake**: Moving target from 50% to 75%
- **Fix**: Stick to plan, take profits at 50%

### 4. Incorrect Position Sizing
- **Mistake**: Risking 20% per trade for "faster gains"
- **Fix**: Strict 5% limit, compound slowly

### 5. Trading in Bear Markets
- **Mistake**: Forcing trades when SPY is down-trending
- **Fix**: Skip trading during confirmed bear markets

### 6. Ignoring Earnings
- **Mistake**: Holding through earnings announcements
- **Fix**: Check earnings calendar, exit before earnings

### 7. Poor Liquidity
- **Mistake**: Trading options with <100 volume
- **Fix**: Minimum 100 volume, 1000 OI on short strike

### 8. No Trade Log
- **Mistake**: "I'll remember the details"
- **Fix**: Log EVERY trade immediately after execution

---

## Month 3 Decision Criteria

### Proceed to Live Trading IF:

✅ **Win Rate >55%** (paper trading)
✅ **Sharpe Ratio >1.0** (calculated)
✅ **Fill Rate >90%** (orders filled at desired price)
✅ **Max Drawdown <25%** (of paper capital)
✅ **Slippage <10%** (vs backtest expectation)
✅ **10+ Completed Trades** (sufficient sample)
✅ **Consistent Execution** (following plan)
✅ **Emotional Control** (no panic exits)

### Continue Paper Trading IF:

⚠️ Win rate 45-55% (marginal, needs refinement)
⚠️ Sharpe ratio 0.5-1.0 (acceptable but improvable)
⚠️ Fill rate 80-90% (some slippage issues)
⚠️ Drawdown 25-35% (higher than target)

### DO NOT Proceed to Live IF:

❌ Win rate <45% (strategy not working)
❌ Sharpe ratio <0.5 (poor risk-adjusted returns)
❌ Fill rate <80% (execution problems)
❌ Drawdown >35% (excessive risk)
❌ <10 trades (insufficient validation)
❌ Emotional trading (panic, revenge trading)

---

## Live Trading Transition Plan

### Phase 1: Soft Launch (Month 4)

**Capital**: 50% of intended amount
- If planning $10K live → Start with $5K

**Execution**:
- Execute 3-5 live trades at reduced size
- Monitor execution quality vs paper
- Track ACTUAL transaction costs
- Validate psychological comfort

**Success Criteria**:
- Win rate matches paper (±5%)
- No emotional decision-making
- Costs match expectations

### Phase 2: Scale Up (Month 5-6)

**Capital**: Gradually increase to 100%
- Month 5: 75% of target capital
- Month 6: 100% of target capital

**Execution**:
- Increase position sizes proportionally
- Maintain same risk percentages
- Continue rigorous tracking

**Success Criteria**:
- Consistent performance
- Risk limits respected
- Drawdown controlled

---

## Paper Trading vs Backtest Comparison

### Expected Deviations

| Metric | Backtest | Paper (Expected) | Reason |
|--------|----------|------------------|--------|
| Win Rate | 67% | 60-70% | Normal variance |
| Profit Factor | 2.45 | 2.0-3.0 | Small sample size |
| Avg Days | 28.3 | 25-35 | Timing differences |
| Sharpe | 1.40 | 1.0-1.8 | Market regime changes |
| Max DD | 60% | 20-30% | Shorter duration |
| Slippage | $0.05 | $0.05-0.15 | Live spreads wider |
| Fill Rate | 100% | 90-95% | Some orders unfilled |

### Red Flags

If paper trading shows:
- Win rate <55% → Review entry criteria
- Profit factor <1.5 → Check exit discipline
- Slippage >$0.25 → Improve order types (limit orders)
- Fill rate <85% → Trade more liquid options
- Max DD >40% → Reduce position sizes

---

## Resources

### Educational

- **TastyTrade**: Free options education (tastytrade.com)
- **CBOE Options Institute**: Professional courses (cboe.com)
- **OptionsAlpha**: Strategy guides (optionsalpha.com)

### Tools

- **OptionStrat**: Free P/L visualization (optionstrat.com)
- **Barchart**: Options chain analysis (barchart.com)
- **IBKR Risk Navigator**: Position risk analysis (TWS tool)

### Communities

- **r/thetagang**: Options selling strategies (reddit.com/r/thetagang)
- **r/options**: General options discussion (reddit.com/r/options)

---

## Support

### Getting Help

1. **Technical Issues**: Check `examples/backtest_bull_put_spread.rs` for reference implementation
2. **Strategy Questions**: See `docs/PROFITABILITY_REPORT.md` for detailed analysis
3. **Parameter Tuning**: Review `src/strategy/strategies.rs` for default parameters

### Contact

- **Documentation**: `/home/kim/projects/kimsfinance/rust/docs/`
- **Code Examples**: `/home/kim/projects/kimsfinance/rust/examples/`
- **Strategy Framework**: `/home/kim/projects/kimsfinance/rust/src/strategy/`

---

**Good luck with paper trading! Discipline and patience are key to success.**
