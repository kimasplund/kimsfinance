# Multi-Asset Support - Quick Reference

**Version**: kimsfinance_core v0.2.0
**Last Updated**: 2025-11-03

## Quick Start

```rust
use kimsfinance_core::assets::*;
```

---

## 1. EQUITY (Stocks)

### Create Asset
```rust
let aapl = EquityAsset::new("AAPL", Exchange::Nasdaq);
```

### Add Corporate Action
```rust
// 4-for-1 stock split
let split_date = NaiveDate::from_ymd_opt(2024, 6, 10).unwrap();
aapl.add_corporate_action(CorporateAction::new_split(split_date, 4, 1));

// $0.50 dividend
let dividend = CorporateAction::new_dividend(
    ex_date, record_date, payment_date, 0.50
);
aapl.add_corporate_action(dividend);
```

### Adjust Historical Data
```rust
let adjusted_price = aapl.adjust_price(600.0, split_date); // → $150
let adjusted_qty = aapl.adjust_quantity(100.0, split_date); // → 400 shares
```

---

## 2. FUTURES

### Create Contract
```rust
let expiration = DateTime::from_timestamp(1742864400, 0).unwrap();
let es = StandardFutures::es(FuturesMonthCode::March, 2025, expiration);
// Symbol: ESH25
```

### Calculate P&L
```rust
// Tick P&L
let ticks = (exit_price - entry_price) / es.tick_size();
let pnl = es.tick_pnl(ticks);

// Contract value
let value = es.calculate_value(5000.0, 1.0).unwrap(); // → $250,000
```

### Check Margin
```rust
let (initial, maintenance) = es.required_margin(2);
// → ($26,400, $24,000) for 2 contracts
```

---

## 3. OPTIONS

### Create Contract
```rust
let call = OptionsContract::new(
    "AAPL",
    OptionType::Call,
    150.0,  // strike
    expiration,
    Exchange::CBOE
);
```

### Price with Black-Scholes
```rust
let price = call.black_scholes_price(
    150.0,  // spot
    0.25,   // volatility (25%)
    0.05,   // risk-free rate (5%)
    0.5     // time to expiry (6 months)
);
```

### Calculate Greeks
```rust
let greeks = call.calculate_greeks(
    spot, volatility, risk_free_rate, time_to_expiry
);

println!("Delta: {}", greeks.delta);   // ~0.59 for ATM call
println!("Gamma: {}", greeks.gamma);   // ~0.015
println!("Theta: {}", greeks.theta);   // ~-0.04 per day
println!("Vega:  {}", greeks.vega);    // ~0.41 per 1%
println!("Rho:   {}", greeks.rho);     // ~0.38 per 1%
```

### Check Moneyness
```rust
if call.is_itm(spot_price) {
    // In-the-money
} else if call.is_atm(spot_price, 0.01) {
    // At-the-money (within 1%)
} else {
    // Out-of-the-money
}
```

---

## 4. FOREX

### Create Pair
```rust
let eurusd = StandardForexPairs::eurusd();
// Or custom:
let pair = ForexPair::new(Currency::EUR, Currency::USD);
```

### Calculate Pips
```rust
let pips = eurusd.calculate_pips(
    1.1000,  // entry
    1.1050,  // exit
    true     // long position
);
// → 50 pips profit
```

### Position Value
```rust
let value = eurusd.calculate_position_value(1.1000, 1.0);
// → $110,000 (1 standard lot)
```

---

## 5. CRYPTO

### Create Asset
```rust
let btcusd = StandardCryptoPairs::btcusd(Exchange::Binance);
```

### Satoshi Conversion
```rust
let sats = btcusd.to_satoshis(0.12345678);  // → 12,345,678
let btc = btcusd.from_satoshis(12345678);   // → 0.12345678
```

### Trading Fees
```rust
let btc = btcusd.with_fees(0.001, 0.0015);  // 0.1% maker, 0.15% taker
let fee = btc.calculate_fee(50_000.0, true); // → $50 (maker)
```

---

## 6. CFD

### Create Contract
```rust
let cfd = CFDContract::new(
    "AAPL",
    Exchange::OTC,
    10.0,   // 10x leverage
    0.01    // tick size
).with_financing_rate(0.05); // 5% annual
```

### Calculate Margin
```rust
let margin = cfd.calculate_margin(150.0, 100.0);
// → $1,500 (10% of $15,000 position)
```

### Overnight Financing
```rust
let charge = cfd.calculate_overnight_financing(15_000.0, 1.0);
// → $2.05 per day
```

### Max Position
```rust
let max_size = cfd.max_position_size(10_000.0, 150.0);
// → 666.67 shares (with 10x leverage)
```

---

## 7. INDEX

### Create Index
```rust
let sp500 = StandardIndices::sp500();
// Or custom:
let index = MarketIndex::new(
    "CUSTOM",
    "Custom Index",
    IndexMethodology::MarketCapWeighted
);
```

### Add Constituents
```rust
index.add_constituent("AAPL".to_string(), 0.07); // 7% weight
index.add_constituent("MSFT".to_string(), 0.06); // 6% weight
```

### Calculate Value
```rust
let mut prices = HashMap::new();
prices.insert("AAPL".to_string(), 150.0);
prices.insert("MSFT".to_string(), 300.0);

let value = index.calculate_value(&prices).unwrap();
// → (150 * 0.07) + (300 * 0.06) = 28.5
```

---

## Common Operations

### Validate Price
```rust
match asset.validate_price(150.05) {
    Ok(price) => println!("Valid: ${}", price),
    Err(e) => println!("Invalid: {}", e),
}
```

### Check Market Hours
```rust
let now = Utc::now();
if asset.is_market_open(now) {
    // Market is open
}
```

### Get Specifications
```rust
let spec = asset.specification();
println!("Tick size: {}", spec.tick_size);
println!("Multiplier: {}", spec.contract_multiplier);
println!("Currency: {}", spec.currency.code());
```

---

## Standard Contracts & Pairs

### Futures
```rust
StandardFutures::es(month, year, exp);     // E-mini S&P 500
StandardFutures::nq(month, year, exp);     // E-mini Nasdaq 100
StandardFutures::ym(month, year, exp);     // E-mini Dow
StandardFutures::cl(month, year, exp);     // Crude Oil
StandardFutures::gc(month, year, exp);     // Gold
```

### Forex
```rust
StandardForexPairs::eurusd();   // EUR/USD
StandardForexPairs::gbpusd();   // GBP/USD (cable)
StandardForexPairs::usdjpy();   // USD/JPY
StandardForexPairs::usdchf();   // USD/CHF (swissie)
StandardForexPairs::audusd();   // AUD/USD (aussie)
```

### Crypto
```rust
StandardCryptoPairs::btcusd(exchange);   // BTC/USD
StandardCryptoPairs::btcusdt(exchange);  // BTC/USDT
StandardCryptoPairs::ethusd(exchange);   // ETH/USD
StandardCryptoPairs::ethbtc(exchange);   // ETH/BTC
```

### Indices
```rust
StandardIndices::sp500();       // S&P 500 (SPX)
StandardIndices::djia();        // Dow Jones (DJI)
StandardIndices::nasdaq100();   // Nasdaq 100 (NDX)
StandardIndices::russell2000(); // Russell 2000 (RUT)
```

---

## Futures Month Codes

| Month     | Code | Example |
|-----------|------|---------|
| January   | F    | ESF25   |
| February  | G    | ESG25   |
| March     | H    | ESH25   |
| April     | J    | ESJ25   |
| May       | K    | ESK25   |
| June      | M    | ESM25   |
| July      | N    | ESN25   |
| August    | Q    | ESQ25   |
| September | U    | ESU25   |
| October   | V    | ESV25   |
| November  | X    | ESX25   |
| December  | Z    | ESZ25   |

---

## Exchange Codes

| Exchange | Asset Classes        |
|----------|----------------------|
| NYSE     | Equity               |
| Nasdaq   | Equity               |
| CME      | Futures              |
| CBOT     | Futures              |
| NYMEX    | Futures (Energy)     |
| COMEX    | Futures (Metals)     |
| CBOE     | Options              |
| Binance  | Crypto               |
| Coinbase | Crypto               |
| Forex    | Forex                |
| OTC      | CFD                  |

---

## Error Handling

```rust
use kimsfinance_core::assets::{AssetResult, AssetError};

fn trade(asset: &dyn Asset, price: f64) -> AssetResult<()> {
    let validated_price = asset.validate_price(price)?;

    if !asset.is_market_open(Utc::now()) {
        return Err(AssetError::MarketClosed("Market is closed".to_string()));
    }

    // Execute trade...
    Ok(())
}
```

---

## Testing

```bash
# Run all asset tests
cargo test --lib assets

# Run specific module tests
cargo test --lib assets::equity
cargo test --lib assets::futures
cargo test --lib assets::options
cargo test --lib assets::forex

# Run with output
cargo test --lib assets -- --nocapture

# Run example
cargo run --example multi_asset_demo
```

---

## Key Formulas

### Options Greeks (Black-Scholes)

```
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T

Call Price = S·N(d1) - K·e^(-rT)·N(d2)
Put Price = K·e^(-rT)·N(-d2) - S·N(-d1)

Delta (Call) = N(d1)
Delta (Put) = N(d1) - 1
Gamma = φ(d1) / (S·σ·√T)
Theta (per day) = -[S·φ(d1)·σ / (2√T) - r·K·e^(-rT)·N(d2)] / 365
Vega (per 1%) = S·φ(d1)·√T / 100
Rho (per 1%) = K·T·e^(-rT)·N(d2) / 100
```

Where:
- `N(x)` = Standard normal CDF
- `φ(x)` = Standard normal PDF
- `S` = Spot price
- `K` = Strike price
- `r` = Risk-free rate
- `σ` = Volatility
- `T` = Time to expiry (years)

---

## Performance Tips

1. **Reuse Assets**: Create once, use many times
2. **Cache Greeks**: Store calculated Greeks if spot/vol unchanged
3. **Batch Operations**: Use vectors for multiple calculations
4. **Avoid String Allocation**: Use `&str` where possible
5. **Compile-Time Optimization**: Let Rust monomorphize traits

---

## Common Pitfalls

1. **Tick Size Validation**: Always validate prices before trading
2. **Market Hours**: Check `is_market_open()` before orders
3. **Expiration**: Check `is_expired()` for derivatives
4. **Corporate Actions**: Apply adjustments before backtesting
5. **Fees**: Don't forget maker/taker fees in crypto
6. **Leverage**: CFD margin can be lower than expected

---

## Additional Resources

- **Full Documentation**: See `MULTI_ASSET_IMPLEMENTATION_REPORT.md`
- **Example Code**: `examples/multi_asset_demo.rs`
- **API Reference**: Run `cargo doc --open`
- **Source Code**: `src/assets/`

---

**Quick Reference Version**: 1.0
**Last Updated**: 2025-11-03
