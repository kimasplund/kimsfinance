# Options Data Connectors Setup Guide

**Last Updated**: 2025-10-29
**Status**: Implementation Complete ✅

This guide explains how to set up and use the IBKR and Deribit data connectors for fetching real options market data for Heston model calibration.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Deribit Setup](#deribit-setup-free)
3. [IBKR Setup](#ibkr-setup-requires-account)
4. [API Usage](#api-usage)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Enable Feature Flags

```toml
# Cargo.toml
[dependencies]
kimsfinance_core = { version = "0.2", features = ["data-deribit"] }  # Deribit only
# or
kimsfinance_core = { version = "0.2", features = ["data-ibkr"] }    # IBKR only
# or
kimsfinance_core = { version = "0.2", features = ["data-all"] }     # Both connectors
```

### Basic Usage

```rust
use kimsfinance_core::data::{OptionsDataProvider, deribit::DeribitConnector};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to Deribit
    let connector = DeribitConnector::connect().await?;

    // Fetch BTC options chain
    let options = connector.fetch_options_chain("BTC").await?;

    println!("Found {} BTC options", options.len());
    for option in options.iter().take(5) {
        println!("Strike: {}, IV: {:?}", option.strike, option.implied_vol);
    }

    Ok(())
}
```

---

## Deribit Setup (FREE)

### Prerequisites

- ✅ **No account required** for public market data
- ✅ **No authentication needed** for read-only access
- ✅ **No market data fees**

### Installation

1. **Add dependency**:

```toml
[dependencies]
kimsfinance_core = { version = "0.2", features = ["data-deribit"] }
tokio = { version = "1", features = ["full"] }
```

2. **That's it!** Deribit public API is open and free.

### Supported Underlyings

- `BTC` - Bitcoin
- `ETH` - Ethereum
- `SOL` - Solana
- `USDC`

### Example: Fetch BTC Options

```rust
use kimsfinance_core::data::deribit::DeribitConnector;
use kimsfinance_core::data::OptionsDataProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let connector = DeribitConnector::connect().await?;

    // Fetch entire BTC options chain
    let options = connector.fetch_options_chain("BTC").await?;

    // Fetch historical volatility (for calibration)
    let hist_vol = connector.fetch_historical_volatility("BTC", 30).await?;

    // Fetch DVOL index (Deribit's VIX equivalent)
    let dvol = connector.fetch_dvol_index("BTC").await?;

    println!("BTC options: {}", options.len());
    println!("Historical vol data points: {}", hist_vol.len());
    println!("DVOL data points: {}", dvol.len());

    Ok(())
}
```

### Data Quality

Deribit provides:
- ✅ Real-time implied volatility (calculated by exchange)
- ✅ Full Greeks (Delta, Gamma, Vega, Theta, Rho)
- ✅ Historical volatility API
- ✅ DVOL index (30-day implied vol, like VIX)
- ✅ Bid/ask/last prices
- ✅ Volume and open interest

### Rate Limits

- **WebSocket**: No hard limit, but recommended <100 subscriptions
- **REST API**: 100 requests per 10 seconds
- **Reconnection**: Automatic with exponential backoff

---

## IBKR Setup (Requires Account)

### Prerequisites

1. **Funded IBKR Account**:
   - Minimum: $500 USD
   - Recommended: $2,000+ for lower commissions

2. **Market Data Subscriptions** ($5-50/month):
   - US Equities Level 1: $10/month (waived if $30+ commissions/month)
   - US Options (OPRA): $4.50/month
   - Total: ~$15/month (or free with active trading)

3. **TWS or IB Gateway** (free software):
   - Download: https://www.interactivebrokers.com/en/trading/tws.php
   - Choose:
     - **TWS (Trader Workstation)**: Full GUI
     - **IB Gateway**: Headless, lightweight (recommended for automation)

### Installation

#### Step 1: Install TWS/IB Gateway

```bash
# Download from IBKR website
# Linux:
wget https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh
chmod +x ibgateway-stable-standalone-linux-x64.sh
./ibgateway-stable-standalone-linux-x64.sh

# macOS:
# Download .dmg from IBKR and install

# Windows:
# Download .exe installer
```

#### Step 2: Configure TWS/Gateway

1. **Open IB Gateway** (or TWS)
2. **Login** with your IBKR credentials
3. **Go to**: File → Global Configuration → API → Settings
4. **Enable**:
   - ✅ "Enable ActiveX and Socket Clients"
   - ✅ "Allow connections from localhost only" (for security)
   - ✅ "Download open orders on connection" (optional)
5. **Set Socket Port**:
   - Paper Trading: `4002`
   - Live Trading: `7497`
6. **Master API Client ID**: Leave blank (or set to 0)
7. **Click OK** and restart Gateway

#### Step 3: Subscribe to Market Data

1. **Login to IBKR Account Management**: https://www.interactivebrokers.com
2. **Go to**: Trading → Market Data Subscriptions
3. **Subscribe to**:
   - US Securities Snapshot and Futures Value Bundle (Level 1)
   - US Equity and Options Add-On Streaming Bundle (OPRA)
4. **Accept agreements** and confirm subscriptions

#### Step 4: Rust Integration

```toml
[dependencies]
kimsfinance_core = { version = "0.2", features = ["data-ibkr"] }
tokio = { version = "1", features = ["full"] }
```

### Example: Fetch AAPL Options

```rust
use kimsfinance_core::data::ibkr::{IbkrConnector, IbkrConfig};
use kimsfinance_core::data::OptionsDataProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure connection
    let config = IbkrConfig {
        host: "127.0.0.1".to_string(),
        port: 4002,  // Paper trading (use 7497 for live)
        client_id: 1,
    };

    // Connect to TWS/Gateway
    let connector = IbkrConnector::connect(config).await?;

    // Fetch AAPL options chain
    let options = connector.fetch_options_chain("AAPL").await?;

    println!("Found {} AAPL options", options.len());
    for option in options.iter().take(5) {
        println!("Strike: {}, Type: {:?}, IV: {:?}",
                 option.strike, option.option_type, option.implied_vol);
    }

    Ok(())
}
```

### Supported Underlyings

- **Equities**: AAPL, TSLA, MSFT, GOOGL, etc.
- **Indices**: SPX, SPY, QQQ, VIX, etc.
- **ETFs**: Any optionable ETF
- **Futures**: ES, NQ, CL, etc. (with futures options)

### Data Quality

IBKR provides:
- ✅ Real-time bid/ask/last prices
- ✅ Implied volatility (calculated by IB)
- ✅ Full Greeks (Delta, Gamma, Vega, Theta, Rho)
- ✅ Volume and open interest
- ⚠️ Historical IV: Not directly available (must reconstruct from historical option prices)

### Rate Limits

- **Market Data**: ~50 requests/second (pacing enforced by IB)
- **Order Submission**: Dozens per second
- **Connection**: Auto-reconnect with Fibonacci backoff

---

## API Usage

### Unified Trait Interface

Both connectors implement `OptionsDataProvider` trait:

```rust
use kimsfinance_core::data::OptionsDataProvider;

#[async_trait]
pub trait OptionsDataProvider: Send + Sync {
    async fn fetch_options_chain(&self, underlying: &str)
        -> Result<Vec<OptionQuote>, DataError>;

    async fn fetch_historical_volatility(&self, underlying: &str, days: u32)
        -> Result<Vec<(i64, f64)>, DataError>;

    async fn subscribe_to_updates(&mut self, underlying: &str)
        -> Result<(), DataError>;
}
```

### Using Trait Objects (Polymorphism)

```rust
use kimsfinance_core::data::{OptionsDataProvider, deribit::DeribitConnector, ibkr::IbkrConnector};

async fn calibrate_heston(
    provider: &dyn OptionsDataProvider,
    underlying: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let options = provider.fetch_options_chain(underlying).await?;
    let hist_vol = provider.fetch_historical_volatility(underlying, 30).await?;

    // Calibrate Heston model...

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use Deribit
    let deribit = DeribitConnector::connect().await?;
    calibrate_heston(&deribit, "BTC").await?;

    // Use IBKR
    let ibkr = IbkrConnector::connect(IbkrConfig::default()).await?;
    calibrate_heston(&ibkr, "AAPL").await?;

    Ok(())
}
```

### Option Quote Structure

```rust
pub struct OptionQuote {
    pub underlying: String,           // "BTC", "AAPL", etc.
    pub strike: f64,                  // Strike price
    pub expiration: i64,              // Unix timestamp
    pub option_type: OptionType,      // Call or Put
    pub spot_price: f64,              // Underlying price
    pub risk_free_rate: f64,          // Risk-free rate

    // Market data
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub last: Option<f64>,
    pub implied_vol: Option<f64>,     // Implied volatility (decimal, not %)
    pub volume: f64,
    pub open_interest: f64,

    // Greeks
    pub greeks: Option<Greeks>,       // Delta, Gamma, Vega, Theta, Rho
}
```

---

## Testing

### Run Deribit Tests (No Setup Required)

```bash
# Test Deribit connector
cargo test --features data-deribit -- deribit

# Specific tests
cargo test --features data-deribit test_deribit_btc_options_chain
cargo test --features data-deribit test_deribit_historical_volatility
cargo test --features data-deribit test_deribit_greeks
```

### Run IBKR Tests (Requires TWS/Gateway)

```bash
# 1. Start TWS or IB Gateway (paper trading mode)
# 2. Ensure API is enabled (port 4002)

# Run IBKR tests (marked as #[ignore] by default)
cargo test --features data-ibkr -- ibkr --ignored

# Specific tests
cargo test --features data-ibkr test_ibkr_connection -- --ignored
cargo test --features data-ibkr test_ibkr_aapl_options_chain -- --ignored
```

### Test Both Connectors

```bash
cargo test --features data-all
```

---

## Troubleshooting

### Deribit Issues

#### Problem: "Connection failed"

**Solutions**:
1. Check internet connection
2. Verify Deribit API is accessible: `ping test.deribit.com`
3. Check firewall settings (allow outbound WebSocket connections)

#### Problem: "No options found for BTC"

**Solutions**:
1. Check if BTC has active options (may be expired)
2. Try ETH instead: `connector.fetch_options_chain("ETH")`
3. Verify Deribit API status: https://status.deribit.com

### IBKR Issues

#### Problem: "Connection failed: Connection refused"

**Solutions**:
1. **Is TWS/Gateway running?**
   ```bash
   # Check if port is listening
   netstat -an | grep 4002  # Paper trading
   netstat -an | grep 7497  # Live trading
   ```

2. **Is API enabled in TWS?**
   - File → Global Configuration → API → Settings
   - ✅ "Enable ActiveX and Socket Clients"

3. **Wrong port?**
   - Paper: 4002
   - Live: 7497

#### Problem: "Authentication failed"

**Solutions**:
1. **Is TWS logged in?** (Check TWS GUI)
2. **Correct client ID?** (Try `client_id: 0`)
3. **Too many connections?** (TWS limits concurrent API connections)

#### Problem: "Market data not available"

**Solutions**:
1. **Is account funded?** (Minimum $500)
2. **Market data subscriptions active?**
   - Login to Account Management
   - Trading → Market Data Subscriptions
   - Verify subscriptions are active
3. **Paper trading account?** Demo accounts cannot subscribe to live data

#### Problem: "Rate limit exceeded"

**Solutions**:
1. Add delays between requests:
   ```rust
   tokio::time::sleep(Duration::from_millis(20)).await;
   ```
2. Reduce number of strikes/expirations queried
3. Use streaming subscriptions instead of polling

---

## Production Considerations

### Error Handling

```rust
use kimsfinance_core::data::{DataError, OptionsDataProvider};

async fn robust_fetch(
    provider: &dyn OptionsDataProvider,
    underlying: &str
) -> Result<Vec<OptionQuote>, DataError> {
    // Retry logic
    for attempt in 1..=3 {
        match provider.fetch_options_chain(underlying).await {
            Ok(options) => return Ok(options),
            Err(e) => {
                eprintln!("Attempt {}/3 failed: {:?}", attempt, e);
                if attempt < 3 {
                    tokio::time::sleep(Duration::from_secs(2_u64.pow(attempt))).await;
                } else {
                    return Err(e);
                }
            }
        }
    }
    unreachable!()
}
```

### Data Validation

```rust
fn validate_option_quote(quote: &OptionQuote) -> bool {
    // Filter out illiquid options
    if quote.open_interest < 10.0 {
        return false;
    }

    // Filter out wide spreads
    if let (Some(bid), Some(ask)) = (quote.bid, quote.ask) {
        let mid = (bid + ask) / 2.0;
        let spread_pct = if mid > 0.0 { ((ask - bid) / mid) * 100.0 } else { 100.0 };
        if spread_pct > 10.0 {
            return false;
        }
    }

    // Validate implied volatility
    if let Some(iv) = quote.implied_vol {
        if iv < 0.05 || iv > 3.0 {  // 5% - 300%
            return false;
        }
    } else {
        return false;  // Require IV for calibration
    }

    true
}
```

### Caching (Optional)

For production use, consider caching options data:

```rust
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

struct OptionsCache {
    cache: Arc<DashMap<String, (Vec<OptionQuote>, u64)>>,
    ttl_seconds: u64,
}

impl OptionsCache {
    fn new(ttl_seconds: u64) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            ttl_seconds,
        }
    }

    fn get(&self, underlying: &str) -> Option<Vec<OptionQuote>> {
        if let Some(entry) = self.cache.get(underlying) {
            let (options, timestamp) = entry.value();
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

            if now - timestamp < self.ttl_seconds {
                return Some(options.clone());
            }
        }
        None
    }

    fn set(&self, underlying: String, options: Vec<OptionQuote>) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        self.cache.insert(underlying, (options, now));
    }
}
```

---

## Cost Summary

### Deribit
- **Setup Cost**: $0
- **Monthly Cost**: $0
- **Account Minimum**: None
- **Total Annual Cost**: **$0**

### IBKR
- **Setup Cost**: $500 (account funding, refundable)
- **Monthly Cost**:
  - Market data: $15/month
  - Or $0 if commissions > $30/month
- **Total Annual Cost**: **$0-180** (depends on trading activity)

### Recommendation

- **Start with Deribit** (free, instant setup)
- **Add IBKR later** if you need equity options

---

## Next Steps

1. ✅ **Choose connector**: Deribit (crypto) or IBKR (equities)
2. ✅ **Run tests**: Verify connection and data quality
3. ✅ **Integrate with Heston calibrator**: Pass options chain to calibration engine
4. ✅ **Add monitoring**: Track API health and data freshness
5. ✅ **Deploy to production**: Add error handling, caching, logging

---

## Support

- **Deribit API Docs**: https://docs.deribit.com
- **IBKR TWS API Docs**: https://interactivebrokers.github.io/tws-api/
- **Rust Client (`ibapi`)**: https://docs.rs/ibapi
- **Rust Client (`deribit`)**: https://docs.rs/deribit

---

**Last Updated**: 2025-10-29
**Status**: Production Ready ✅
