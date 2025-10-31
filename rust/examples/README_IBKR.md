# IBKR Paper Trading Integration

This directory contains examples for testing IBKR (Interactive Brokers) integration with paper trading account.

## Prerequisites

### 1. IBKR Account
- Active IBKR paper trading account
- TWS (Trader Workstation) or IB Gateway installed

### 2. TWS/Gateway Configuration

**Enable API Access:**
1. Open IB Gateway or TWS
2. Go to: **File → Global Configuration → API → Settings**
3. Enable checkboxes:
   - ✅ **Enable ActiveX and Socket Clients**
   - ✅ **Allow connections from localhost only** (security)
4. **Socket port**: Note the port number
   - Common for paper trading: **7497** or **4002**
   - Common for live trading: **7496** or **4001**
5. **Master API Client ID**: Leave blank or set to 0
6. Click **OK** and **restart** Gateway/TWS

### 3. Market Data Subscriptions

For equity options data (AAPL, TSLA, etc.):
1. Login to https://www.interactivebrokers.com
2. Go to: **Settings → User Settings → Market Data Subscriptions**
3. Subscribe to (for paper trading, these are usually free):
   - **US Securities Snapshot and Futures Value Bundle** (Level 1)
   - **US Equity and Options Add-On Streaming Bundle** (OPRA)

**Note**: Paper trading accounts typically have free real-time data. Live accounts may require paid subscriptions.

## Examples

### 1. Test IBKR Connection (`test_ibkr_paper_trading.rs`)

Tests basic connection to IBKR and fetches AAPL options chain.

**Run:**
```bash
cargo run --example test_ibkr_paper_trading --features data-ibkr --release
```

**What it does:**
1. Connects to TWS/Gateway at 127.0.0.1:7497 (default)
2. Fetches AAPL option chain (strikes, expirations)
3. Retrieves market data (bid/ask/last/IV/volume)
4. Filters for liquid options
5. Displays sample data and statistics

**Expected output:**
```
=== IBKR Paper Trading Test ===

Connecting to IBKR paper trading at 127.0.0.1:7497...
✓ Connected successfully!

Fetching AAPL option chain...
✓ Found 142 options

Sample options (first 10):
----------------------------------------------------------------------------------------------------
Expiration   Type     Strike     Bid          Ask          IV           Volume
----------------------------------------------------------------------------------------------------
2025-11-21   CALL     $150.00    $8.50        $8.80        28.5%        1250
...

=== Statistics ===
Calls: 71
Puts: 71
With IV: 142
With Greeks: 128
With Volume: 95
Average IV: 27.3%

✓ IBKR integration test PASSED!
```

**Configuration:**

If you need to use a different port, edit the default config in the code:

```rust
let config = IbkrConfig {
    host: "127.0.0.1".to_string(),
    port: 4002,  // Change to your port (7497 or 4002 typical)
    client_id: 1,
};
```

### 2. Heston Calibration with IBKR (`calibrate_heston_ibkr.rs`)

**Status**: ⚠️ Currently has compilation errors in Heston calibration module (unrelated to IBKR integration)

**Planned functionality:**
1. Connect to IBKR
2. Fetch real options data
3. Filter for liquid options with good data quality
4. Calibrate Heston model using GPU
5. Validate results and display calibrated parameters

**Once fixed, run:**
```bash
cargo run --example calibrate_heston_ibkr --features heston,data-ibkr --release
```

## Troubleshooting

### Connection Failed

**Error**: `Failed to connect to IBKR at 127.0.0.1:7497`

**Solutions**:
1. **Check if TWS/Gateway is running**
   - You should see the IB Gateway or TWS window open
   - Check system tray for IB icon

2. **Verify port number**
   - In IB Gateway/TWS: File → Global Configuration → API → Settings
   - Check "Socket port" matches your code (7497 or 4002 common)
   - Port 7497 is common for paper trading on TWS
   - Port 4002 is common for paper trading on IB Gateway

3. **Check API is enabled**
   - Ensure "Enable ActiveX and Socket Clients" is checked
   - Restart Gateway/TWS after changing settings

4. **Try alternative port**
   - Some systems use 4002 instead of 7497
   - Edit the config to try different port

### No Options Found

**Error**: `No option chains found for AAPL`

**Solutions**:
1. **Market is closed**
   - Options data may be limited after hours
   - Try during regular market hours (9:30 AM - 4:00 PM ET)

2. **Market data subscription not active**
   - Check Account Management for active subscriptions
   - Paper trading should have free real-time data

3. **Wrong symbol**
   - Try a different highly-liquid stock: SPY, QQQ, TSLA

### Data Quality Issues

**Issue**: Many options have no IV or volume

**This is normal**:
- Out-of-the-money options often have no volume
- The connector filters for liquid options (volume > 0, valid bid/ask)
- During after-hours, data may be stale

**Tips for better data**:
- Run during market hours
- Use highly-liquid underlyings (SPY, AAPL, TSLA)
- ATM (at-the-money) options typically have best data

## API Rate Limiting

IBKR enforces rate limits on market data requests:
- **~50 requests/second** maximum
- The connector implements reasonable delays
- If you see "Rate limit" errors, wait a few seconds and retry

## Port Reference

| Mode | TWS Port | IB Gateway Port |
|------|----------|----------------|
| Paper Trading | 7497 | 4002 |
| Live Trading | 7496 | 4001 |

**Note**: These are common defaults. Your installation may differ. Always check your TWS/Gateway configuration.

## Implementation Details

The IBKR connector (`src/data/ibkr/mod.rs`) implements:

- [x] Connection to TWS/Gateway
- [x] Fetch option chains (strikes, expirations)
- [x] Fetch market data (bid/ask/last)
- [x] Parse implied volatility
- [x] Parse Greeks (delta, gamma, vega, theta)
- [x] Filter liquid options
- [x] Error handling with descriptive messages
- [ ] Historical volatility (not available from IBKR API)
- [ ] Real-time streaming updates

## Next Steps

1. **Test Connection**: Run `test_ibkr_paper_trading`
2. **Verify Data Quality**: Check that options have IV and Greeks
3. **Fix Heston Calibration**: Resolve compilation errors in calibration module
4. **Run Full Calibration**: Test `calibrate_heston_ibkr` once fixed

## Support

- **IBKR API Docs**: https://interactivebrokers.github.io/tws-api/
- **ibapi Crate**: https://docs.rs/ibapi/2.0.0
- **Project Issues**: File an issue if you encounter problems
