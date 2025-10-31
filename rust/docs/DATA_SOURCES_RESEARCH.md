# Options Data Sources Research

**Date**: 2025-10-29
**Purpose**: Evaluate IBKR and Deribit APIs for options data integration into Heston calibrator
**Status**: Research Complete ✅

---

## Executive Summary

Both Interactive Brokers (IBKR) and Deribit provide comprehensive APIs suitable for options trading data integration. Rust client libraries exist for both platforms with different maturity levels:

- **IBKR TWS API**: Mature Rust client (`ibapi` v2.0) with async/sync support - **Recommended**
  - Note: IBKR has two APIs (TWS and Client Portal). **Use TWS API** due to better automation support
- **Deribit API**: Native Rust WebSocket client (`deribit-rs`) for crypto options

**Recommendation**:
1. Use **TWS API** (not Client Portal API) for IBKR equity options
2. Use **Deribit API** for crypto options (BTC/ETH)
3. Implement both connectors to support diverse options markets

---

## IMPORTANT: IBKR Has Two Different APIs

Interactive Brokers offers **two separate APIs** with different characteristics:

| Feature | TWS API (Recommended) | Client Portal Web API |
|---------|----------------------|----------------------|
| **Transport** | TCP socket | REST + WebSocket |
| **Local Software** | Requires TWS/Gateway | None (cloud API) |
| **Authentication** | Session-based (auto-restart 24h) | Re-login required every 24h ⚠️ |
| **Throughput** | High (dozens of orders/sec) | Low (limited rate) |
| **Options Data** | ✅ Full support | ⚠️ Limited support |
| **Greeks** | ✅ Real-time | ⚠️ Delayed calculation |
| **Historical Data** | ✅ Comprehensive | ❌ Limited |
| **Documentation** | ✅ Mature, extensive | ⚠️ Incomplete |
| **Rust Client** | `ibapi` v2.0 (production) | `ibkr_client_portal` (alpha) |
| **Stability** | ✅ Battle-tested | ⚠️ Still evolving |
| **Use Case** | **Automated trading, high-volume data** | Custom UIs, low-volume |

### Recommendation for Heston Calibrator

**Use TWS API** (`ibapi` crate) because:

1. **24h Authentication Issue**: Client Portal API requires manual re-authentication every 24 hours, making it unsuitable for automated systems
2. **Options Data Quality**: TWS API provides better support for options chains, Greeks, and historical data
3. **High Throughput**: Needed for fetching large options chains (30-120 options per calibration)
4. **Production Stability**: TWS API is mature and battle-tested
5. **Better Documentation**: More examples and community support

**Avoid Client Portal Web API** because:
- Manual login every 24h is a "showstopper" for automated calibration
- Limited documentation and missing features
- Not designed for high-volume data subscriptions
- Rust client is less mature

**Trade-off**: TWS API requires running local TWS or IB Gateway software, but this is acceptable given the superior data quality and reliability.

---

## 1. Interactive Brokers (IBKR) - TWS API

### API Overview

**Official API**: TWS (Trader Workstation) API
**Transports**: TCP socket connection to TWS/Gateway (runs locally)
**Documentation**: https://interactivebrokers.github.io/tws-api/
**Rust Client**: `ibapi` v2.0

### Rust Client Library: `ibapi` (Recommended)

**Crate**: `ibapi` v2.0
**Repository**: https://github.com/wboayue/rust-ibapi
**Docs**: https://docs.rs/ibapi/latest/ibapi/
**Status**: Actively maintained, production-ready

#### Key Features

1. **Dual Execution Models**:
   - **Async** (default): Tokio-based with broadcast channels
   - **Sync**: Blocking client with crossbeam channels
   - Can use both in same project

2. **Contract Types Supported**:
   - Stocks, Futures, Options, Bonds, Forex, CFDs, Combos
   - Type-safe builder API with compile-time validation

3. **Options Contract Builder**:
```rust
use ibapi::contract::Contract;

// Call option example
let call_option = Contract::call("AAPL")
    .strike(150.0)
    .expires_on(2024, 12, 20)
    .build();

// Put option example
let put_option = Contract::put("TSLA")
    .strike(200.0)
    .expires_on(2025, 3, 21)
    .exchange("SMART")
    .build();
```

4. **Real-Time Market Data**:
```rust
// Async streaming
let mut subscription = client.market_data(&contract).await?;
while let Some(tick) = subscription.next().await {
    match tick {
        TickType::BidPrice(price) => println!("Bid: {}", price),
        TickType::AskPrice(price) => println!("Ask: {}", price),
        TickType::ImpliedVolatility(iv) => println!("IV: {}", iv),
        _ => {}
    }
}
```

5. **Historical Data Access**:
```rust
let bars = client.historical_data(
    &contract,
    "20240101 00:00:00",
    "1 D", // Duration
    BarSize::OneHour,
    WhatToShow::Trades,
    false, // Use RTH
).await?;
```

6. **Automatic Reconnection**:
   - Fibonacci backoff strategy
   - Up to 30 reconnection attempts
   - Production-grade resilience

### Market Data Requirements

#### Subscription Requirements

1. **Level 1 Data** (Required):
   - Top-of-book quotes (bid/ask)
   - Live watchlist data
   - Tick-by-tick data
   - Historical bar data
   - **Cost**: Varies by exchange (typically $1-10/month per exchange)

2. **Options Greeks Data**:
   - Requires subscriptions for **both**:
     - Underlying security (e.g., AAPL stock)
     - Options derivative (e.g., AAPL options)
   - Greeks calculated server-side

3. **OPRA Data** (US Options):
   - Provides all US options market data
   - Required for comprehensive options coverage
   - **Cost**: ~$4.50/month (professional)

#### Account Requirements

- **Funded Account**: Minimum $500 USD for most subscriptions
- **Demo Accounts**: Cannot subscribe to live market data
- **API Access**: Available on all account types (once funded)

### Data Available for Heston Calibration

| Data Type | Availability | API Method |
|-----------|-------------|------------|
| **Current Option Quotes** | ✅ Real-time | `market_data()` |
| **Options Chain** | ✅ Yes | `req_sec_def_opt_params()` |
| **Implied Volatility** | ✅ Real-time | Tick type in `market_data()` |
| **Greeks** | ✅ Real-time | Tick types (Delta, Gamma, Vega, Theta) |
| **Historical Option Prices** | ✅ Yes | `historical_data()` |
| **Historical IV Surface** | ⚠️ Indirect | Can reconstruct from historical quotes |
| **Settlement Prices** | ✅ Yes | Historical data with `WhatToShow::OptionImpliedVolatility` |

### Implementation Recommendations

#### Phase 1: Basic Integration

```rust
// Cargo.toml
[dependencies]
ibapi = { version = "2.0", features = ["async"] }
tokio = { version = "1", features = ["full"] }

// src/data/ibkr/mod.rs
pub struct IbkrConnector {
    client: ibapi::Client,
    config: IbkrConfig,
}

#[derive(Debug, Clone)]
pub struct IbkrConfig {
    pub host: String,      // "127.0.0.1"
    pub port: u16,         // 4002 (paper), 7497 (live)
    pub client_id: i32,    // Unique client ID
}

impl IbkrConnector {
    pub async fn connect(config: IbkrConfig) -> Result<Self, DataError> {
        let address = format!("{}:{}", config.host, config.port);
        let client = ibapi::Client::connect(&address, config.client_id).await?;
        Ok(Self { client, config })
    }

    pub async fn fetch_options_chain(
        &self,
        underlying: &str,
        expiration_date: NaiveDate,
    ) -> Result<Vec<OptionQuote>, DataError> {
        // 1. Get available strikes and expirations
        let params = self.client.req_sec_def_opt_params(
            underlying,
            "",           // Futures expiry (empty for stocks)
            "STK",        // Underlying security type
            underlying_conid, // Contract ID
        ).await?;

        // 2. For each strike, request market data
        let mut options = Vec::new();
        for strike in params.strikes {
            // Call
            let call = Contract::call(underlying)
                .strike(strike)
                .expires_on(expiration_date.year(), expiration_date.month(), expiration_date.day())
                .build();

            let call_data = self.fetch_option_quote(&call).await?;
            options.push(call_data);

            // Put
            let put = Contract::put(underlying)
                .strike(strike)
                .expires_on(expiration_date.year(), expiration_date.month(), expiration_date.day())
                .build();

            let put_data = self.fetch_option_quote(&put).await?;
            options.push(put_data);
        }

        Ok(options)
    }

    async fn fetch_option_quote(&self, contract: &Contract) -> Result<OptionQuote, DataError> {
        let mut subscription = self.client.market_data(contract).await?;

        let mut quote = OptionQuote::default();
        let timeout = Duration::from_secs(5);
        let start = Instant::now();

        while start.elapsed() < timeout {
            if let Some(tick) = subscription.next().await {
                match tick {
                    TickType::BidPrice(price) => quote.bid = Some(price),
                    TickType::AskPrice(price) => quote.ask = Some(price),
                    TickType::LastPrice(price) => quote.last = Some(price),
                    TickType::ImpliedVolatility(iv) => quote.implied_vol = Some(iv),
                    TickType::Delta(delta) => quote.greeks.delta = Some(delta),
                    TickType::Gamma(gamma) => quote.greeks.gamma = Some(gamma),
                    TickType::Vega(vega) => quote.greeks.vega = Some(vega),
                    TickType::Theta(theta) => quote.greeks.theta = Some(theta),
                    _ => {}
                }
            }

            // Break early if we have all required data
            if quote.is_complete() {
                break;
            }
        }

        subscription.cancel().await?;
        Ok(quote)
    }
}
```

#### Phase 2: Historical Data for Calibration

```rust
impl IbkrConnector {
    pub async fn fetch_historical_iv_surface(
        &self,
        underlying: &str,
        date: NaiveDate,
        expirations: Vec<NaiveDate>,
    ) -> Result<VolatilitySurface, DataError> {
        let mut surface = VolatilitySurface::new();

        for expiration in expirations {
            let options_chain = self.fetch_options_chain(underlying, expiration).await?;

            for option in options_chain {
                if let Some(iv) = option.implied_vol {
                    surface.add_point(
                        expiration,
                        option.strike,
                        option.option_type,
                        iv,
                        option.bid,
                        option.ask,
                    );
                }
            }
        }

        Ok(surface)
    }
}
```

### Limitations & Considerations

1. **Rate Limits**: TWS API has built-in pacing (50 messages/sec for market data)
2. **Connection Model**: Requires TWS or IB Gateway running locally
3. **Market Data Costs**: Subscription fees required ($5-50/month depending on coverage)
4. **Greeks Calculation**: Uses IB's proprietary models (not Heston)
5. **Historical IV**: Not directly available, must be reconstructed from quotes

---

## 2. Deribit (Crypto Options)

### API Overview

**Official API**: Deribit V2 API
**Transports**: WebSocket (preferred), REST
**Documentation**: https://docs.deribit.com/
**Coverage**: BTC, ETH, SOL, USDC options and futures

### Rust Client Library: `deribit-rs`

**Crate**: `deribit`
**Repository**: https://github.com/dovahcrow/deribit-rs
**Status**: Active, WebSocket-focused

#### Key Features

1. **WebSocket-First Design**:
   - Real-time data via subscriptions
   - RPC calls over WebSocket
   - Automatic reconnection support

2. **Connection Example**:
```rust
use deribit::Deribit;

// Connect and get both API and subscription clients
let (api_client, mut subscription_client) = Deribit::connect().await?;

// API calls (RPC)
let instruments = api_client
    .get_instruments("BTC", "option")
    .await?;

// Subscribe to real-time updates
subscription_client
    .subscribe_ticker("BTC-*")
    .await?;

while let Some(notification) = subscription_client.next().await {
    match notification {
        Notification::Ticker(ticker) => {
            println!("IV: {}", ticker.mark_iv);
            println!("Greeks: {:?}", ticker.greeks);
        }
        _ => {}
    }
}
```

3. **Options-Specific Features**:
   - Native Greeks (Delta, Gamma, Vega, Theta, Rho)
   - Mark IV (implied volatility from mark price)
   - DVOL Index (Deribit Volatility Index - like VIX)
   - Settlement prices
   - Historical volatility data

### Data Available for Heston Calibration

| Data Type | Availability | API Method |
|-----------|-------------|------------|
| **Current Option Quotes** | ✅ Real-time | `/public/ticker`, WebSocket subscription |
| **Options Chain** | ✅ Yes | `/public/get_instruments` |
| **Implied Volatility** | ✅ Real-time | `mark_iv` in ticker |
| **Greeks** | ✅ Real-time | `greeks` object in ticker |
| **Historical Option Prices** | ✅ Yes | `/public/get_tradevolume` |
| **Historical IV Surface** | ✅ Yes | `/public/get_historical_volatility` |
| **DVOL Index** | ✅ Real-time | `/public/get_volatility_index_data` |
| **Settlement Prices** | ✅ Yes | `/public/get_last_settlements_by_instrument` |

### API Endpoints for Heston Calibration

#### 1. Options Chain Discovery

```rust
// Get all BTC options expiring in specific timeframe
let instruments = api_client
    .get_instruments("BTC", "option")
    .await?;

for instrument in instruments {
    println!("Strike: {}, Expiry: {}", instrument.strike, instrument.expiration_timestamp);
}
```

#### 2. Real-Time Implied Volatility

```rust
// Subscribe to ticker for specific option
subscription_client
    .subscribe_ticker("BTC-29DEC23-40000-C")
    .await?;

// Receive updates
while let Some(notification) = subscription_client.next().await {
    if let Notification::Ticker(ticker) = notification {
        println!("Mark IV: {}%", ticker.mark_iv);
        println!("Bid IV: {}%", ticker.bid_iv);
        println!("Ask IV: {}%", ticker.ask_iv);
    }
}
```

#### 3. Historical Volatility (Critical for Calibration)

```http
GET /public/get_historical_volatility?currency=BTC

Response:
[
  [timestamp, volatility],
  [1698105600000, 0.6234],
  [1698192000000, 0.6189],
  ...
]
```

```rust
let historical_vol = api_client
    .get_historical_volatility("BTC")
    .await?;

// Use for calibrating long-term variance (θ in Heston)
let average_vol = historical_vol.iter()
    .map(|(_, vol)| vol)
    .sum::<f64>() / historical_vol.len() as f64;
```

#### 4. DVOL Index (Like VIX)

```rust
let dvol_data = api_client
    .get_volatility_index_data("BTC")
    .await?;

println!("Current DVOL: {}%", dvol_data.current);
println!("30-day implied volatility: {}%", dvol_data.data.last().1);
```

### Implementation Recommendations

#### Phase 1: Basic Integration

```rust
// Cargo.toml
[dependencies]
deribit = "0.3"
tokio = { version = "1", features = ["full"] }

// src/data/deribit/mod.rs
pub struct DeribitConnector {
    api_client: deribit::DeribitAPIClient,
    subscription_client: deribit::DeribitSubscriptionClient,
}

impl DeribitConnector {
    pub async fn connect() -> Result<Self, DataError> {
        let (api_client, subscription_client) = deribit::Deribit::connect().await?;
        Ok(Self { api_client, subscription_client })
    }

    pub async fn fetch_options_chain(
        &self,
        currency: &str, // "BTC" or "ETH"
    ) -> Result<Vec<OptionQuote>, DataError> {
        // 1. Get all option instruments
        let instruments = self.api_client
            .get_instruments(currency, "option")
            .await?;

        // 2. Fetch ticker data for each instrument
        let mut options = Vec::new();
        for instrument in instruments {
            let ticker = self.api_client
                .ticker(&instrument.instrument_name)
                .await?;

            let option = OptionQuote {
                underlying: currency.to_string(),
                strike: instrument.strike,
                expiration: instrument.expiration_timestamp,
                option_type: if instrument.instrument_name.contains("-C") {
                    OptionType::Call
                } else {
                    OptionType::Put
                },
                bid: ticker.best_bid_price,
                ask: ticker.best_ask_price,
                last: ticker.last_price,
                implied_vol: Some(ticker.mark_iv),
                greeks: Greeks {
                    delta: Some(ticker.greeks.delta),
                    gamma: Some(ticker.greeks.gamma),
                    vega: Some(ticker.greeks.vega),
                    theta: Some(ticker.greeks.theta),
                    rho: Some(ticker.greeks.rho),
                },
                volume: ticker.stats.volume,
                open_interest: ticker.open_interest,
            };

            options.push(option);
        }

        Ok(options)
    }

    pub async fn fetch_historical_volatility(
        &self,
        currency: &str,
    ) -> Result<Vec<(i64, f64)>, DataError> {
        let vol_data = self.api_client
            .get_historical_volatility(currency)
            .await?;

        Ok(vol_data)
    }

    pub async fn subscribe_to_volatility_updates(
        &mut self,
        currency: &str,
    ) -> Result<(), DataError> {
        // Subscribe to DVOL index
        self.subscription_client
            .subscribe(&format!("deribit_volatility_index.{}_usd", currency.to_lowercase()))
            .await?;

        Ok(())
    }
}
```

#### Phase 2: Real-Time Calibration

```rust
impl DeribitConnector {
    pub async fn stream_iv_surface_updates(
        &mut self,
        currency: &str,
    ) -> impl Stream<Item = VolatilitySurfaceUpdate> {
        // Subscribe to all options for a currency
        self.subscription_client
            .subscribe_ticker(&format!("{}-*", currency))
            .await
            .unwrap();

        // Convert notifications to surface updates
        self.subscription_client
            .filter_map(|notification| async move {
                if let Notification::Ticker(ticker) = notification {
                    Some(VolatilitySurfaceUpdate {
                        timestamp: ticker.timestamp,
                        instrument: ticker.instrument_name,
                        mark_iv: ticker.mark_iv,
                        bid_iv: ticker.bid_iv,
                        ask_iv: ticker.ask_iv,
                        greeks: ticker.greeks,
                    })
                } else {
                    None
                }
            })
    }
}
```

### Advantages Over IBKR

1. **No Market Data Fees**: All data is free for Deribit
2. **No Local Software**: Pure cloud API (no TWS/Gateway needed)
3. **Native IV Data**: Implied volatility calculated and streamed by exchange
4. **DVOL Index**: Direct VIX-equivalent for crypto
5. **Historical IV**: Direct API endpoint for historical volatility
6. **Simpler Authentication**: API keys only (no session management)

### Limitations & Considerations

1. **Crypto Only**: Limited to BTC, ETH, SOL, USDC options
2. **Smaller Market**: Less liquidity than traditional options markets
3. **24/7 Trading**: Need to handle continuous market (no market hours)
4. **Pricing Model**: Uses Black-76 (not Heston) for Greeks
5. **Settlement Risk**: Crypto-specific risks (volatility, custody)

---

## 3. Comparison Matrix

| Feature | IBKR | Deribit |
|---------|------|---------|
| **Asset Classes** | Stocks, ETFs, Indices | BTC, ETH, SOL, USDC |
| **Rust Client** | `ibapi` (v2.0) | `deribit-rs` (v0.3) |
| **Transport** | TCP Socket | WebSocket/REST |
| **Market Data Cost** | $5-50/month | Free |
| **Account Requirement** | $500 minimum | None (API keys only) |
| **Greeks** | ✅ Real-time | ✅ Real-time |
| **Implied Volatility** | ✅ Real-time | ✅ Real-time + Historical |
| **Historical IV** | ⚠️ Indirect | ✅ Direct API |
| **Volatility Index** | ❌ None | ✅ DVOL (VIX-like) |
| **Connection Model** | Local TWS/Gateway | Cloud API |
| **Rate Limits** | 50 msg/sec | 100 req/10s |
| **Reconnection** | ✅ Automatic | ✅ Automatic |
| **Production Ready** | ✅ Yes | ✅ Yes |

---

## 4. Recommended Implementation Strategy

### Phase 4 (from main plan): Data Integration

#### Step 1: Deribit First (Easier)
- **Reason**: Simpler authentication, free data, direct IV API
- **Timeline**: 1-2 weeks
- **Deliverables**:
  - `src/data/deribit/mod.rs` - Connector module
  - `src/data/deribit/options.rs` - Options chain fetching
  - `src/data/deribit/volatility.rs` - Historical volatility
  - Integration tests with sandbox account

#### Step 2: IBKR Second (More Complex)
- **Reason**: Requires TWS/Gateway setup, market data subscriptions
- **Timeline**: 2-3 weeks
- **Deliverables**:
  - `src/data/ibkr/mod.rs` - Connector module
  - `src/data/ibkr/options.rs` - Options chain fetching
  - `src/data/ibkr/historical.rs` - Historical data reconstruction
  - Paper trading account integration tests

#### Step 3: Unified Interface
- **Timeline**: 1 week
- **Deliverables**:
```rust
// src/data/mod.rs
pub trait OptionsDataProvider {
    async fn fetch_options_chain(&self, underlying: &str) -> Result<Vec<OptionQuote>>;
    async fn fetch_historical_volatility(&self, underlying: &str, days: u32) -> Result<Vec<(i64, f64)>>;
    async fn subscribe_to_updates(&mut self, underlying: &str) -> Result<()>;
}

impl OptionsDataProvider for DeribitConnector { /* ... */ }
impl OptionsDataProvider for IbkrConnector { /* ... */ }

// Usage in Heston calibrator
pub struct HestonCalibrator<P: OptionsDataProvider> {
    data_provider: P,
    params: HestonParams,
}
```

### Code Structure

```
src/data/
├── mod.rs                    # Unified OptionsDataProvider trait
├── common.rs                 # Shared types (OptionQuote, Greeks, etc.)
├── ibkr/
│   ├── mod.rs               # IbkrConnector
│   ├── options.rs           # Options chain fetching
│   ├── historical.rs        # Historical data
│   └── config.rs            # Configuration
├── deribit/
│   ├── mod.rs               # DeribitConnector
│   ├── options.rs           # Options chain fetching
│   ├── volatility.rs        # Historical volatility
│   └── dvol.rs              # DVOL index integration
└── cache/
    ├── mod.rs               # Caching layer (optional)
    └── redis.rs             # Redis backend for IV surface caching
```

---

## 5. Data Requirements for Heston Calibration

### Required Data Points

For each option in the calibration set:

```rust
pub struct OptionQuote {
    // Contract details
    pub underlying: String,           // "BTC", "AAPL", etc.
    pub strike: f64,                  // Strike price
    pub expiration: i64,              // Unix timestamp
    pub option_type: OptionType,      // Call or Put

    // Market data (at specific timestamp)
    pub timestamp: i64,               // Quote timestamp
    pub bid: Option<f64>,             // Bid price
    pub ask: Option<f64>,             // Ask price
    pub last: Option<f64>,            // Last traded price
    pub volume: f64,                  // Trading volume
    pub open_interest: f64,           // Open interest

    // Volatility data (CRITICAL FOR HESTON)
    pub implied_vol: Option<f64>,     // Market implied volatility

    // Greeks (optional, for validation)
    pub greeks: Greeks,

    // Underlying data
    pub underlying_price: f64,        // Spot price of underlying
    pub risk_free_rate: f64,          // Risk-free rate
}

pub struct Greeks {
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub vega: Option<f64>,
    pub theta: Option<f64>,
    pub rho: Option<f64>,
}
```

### Calibration Dataset

Typical calibration uses:
- **Multiple expirations**: 3-6 different expirations (7d, 30d, 60d, 90d, 180d, 365d)
- **Multiple strikes**: 10-20 strikes per expiration (OTM, ATM, ITM)
- **Total options**: 30-120 options in calibration set
- **Update frequency**: Real-time (for live trading) or EOD (for backtesting)

### Data Quality Requirements

1. **Bid-Ask Spread**: Filter out options with spread > 10% (illiquid)
2. **Volume Filter**: Require minimum volume (e.g., 10 contracts/day)
3. **Moneyness Range**: Focus on options within 0.8-1.2 moneyness
4. **Expiration Range**: Include 7d to 365d (avoid very short/long dated)
5. **IV Bounds**: Filter out IV < 5% or IV > 300% (data errors)

---

## 6. Implementation Checklist

### Deribit Integration

- [ ] Add `deribit` crate to `Cargo.toml`
- [ ] Create `src/data/deribit/mod.rs` with `DeribitConnector`
- [ ] Implement `fetch_options_chain()` for BTC/ETH
- [ ] Implement `fetch_historical_volatility()` for calibration
- [ ] Add DVOL index integration for regime detection
- [ ] Create integration tests with sandbox environment
- [ ] Add documentation and examples

### IBKR Integration

- [ ] Add `ibapi` crate to `Cargo.toml`
- [ ] Create `src/data/ibkr/mod.rs` with `IbkrConnector`
- [ ] Implement `fetch_options_chain()` for equities
- [ ] Implement historical IV reconstruction logic
- [ ] Add configuration for TWS/Gateway connection
- [ ] Handle market data subscription requirements
- [ ] Create integration tests with paper trading account
- [ ] Add documentation and examples

### Unified Interface

- [ ] Define `OptionsDataProvider` trait in `src/data/mod.rs`
- [ ] Implement trait for both connectors
- [ ] Create common data types (`OptionQuote`, `Greeks`, etc.)
- [ ] Add data validation and filtering logic
- [ ] Implement optional caching layer (Redis)
- [ ] Add benchmarks for data fetching performance

### Testing

- [ ] Unit tests for each connector
- [ ] Integration tests with real APIs (sandbox/paper accounts)
- [ ] Validate data quality (bid-ask spreads, volume, etc.)
- [ ] Test reconnection logic
- [ ] Test rate limit handling
- [ ] Benchmark data fetching latency

---

## 7. Cost Estimate

### IBKR

- **Market Data Subscriptions**: $5-50/month
  - US Equities (L1): $10/month (waived if $30+ commissions/month)
  - US Options (OPRA): $4.50/month
  - Total for basic setup: ~$15/month (or waived with trading activity)

- **Account Minimum**: $500 (one-time)

### Deribit

- **Market Data**: Free
- **Account Minimum**: None
- **API Access**: Free

### Total Annual Cost

- **IBKR**: $180/year (or $0 if trading actively)
- **Deribit**: $0
- **Combined**: $180/year maximum

---

## 8. Next Steps

1. ✅ **Research Complete** - This document
2. **Decision Point**: Start with Deribit or IBKR?
   - **Recommendation**: Start with Deribit (easier, faster, free)
3. **Prototype**: Build basic connector and test data fetching
4. **Integrate**: Connect to Heston calibrator (once core model ready)
5. **Validate**: Compare calibrated parameters with market Greeks
6. **Production**: Add error handling, caching, monitoring

---

## 9. References

### IBKR

- **TWS API Documentation**: https://interactivebrokers.github.io/tws-api/
- **Rust Client (ibapi)**: https://github.com/wboayue/rust-ibapi
- **Market Data Subscriptions**: https://www.interactivebrokers.com/en/trading/market-data-subscriptions.php

### Deribit

- **API Documentation**: https://docs.deribit.com/
- **Rust Client (deribit-rs)**: https://github.com/dovahcrow/deribit-rs
- **DVOL Index**: https://insights.deribit.com/exchange-updates/dvol-deribit-implied-volatility-index/

### Heston Model

- **Original Paper**: Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
- **GPU Implementation**: CUDA-based characteristic function for fast calibration

---

**Prepared By**: Claude Code
**Date**: 2025-10-29
**Status**: Ready for Implementation ✅
