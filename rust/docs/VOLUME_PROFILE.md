# Volume Profile Analysis

## Overview

Volume Profile is a charting technique that displays volume distribution by price level over a specific time period. Unlike traditional volume charts that show volume over time, volume profile shows **where** price traded, revealing the market's acceptance or rejection of specific price levels.

## Key Concepts

### Point of Control (POC)

The **Point of Control** is the price level with the highest traded volume during the analyzed period.

**Characteristics:**
- Represents fair value / equilibrium price
- Acts as a price magnet (prices tend to return to POC)
- Often becomes support or resistance
- Indicates where most market participants agreed on price

**Trading Applications:**
- Mean reversion: Buy below POC, sell above POC
- Breakout confirmation: Volume increase at POC confirms move
- Risk management: POC as a reference for stop-loss placement

### Value Area (VA)

The **Value Area** is the price range that contains 70% of the traded volume during the period.

**Components:**
- **Value Area High (VAH)**: Top of the value area (resistance)
- **Value Area Low (VAL)**: Bottom of the value area (support)

**Characteristics:**
- Represents "accepted" prices where most trading occurred
- 70% is the standard, but can be adjusted (60-80%)
- Prices outside VA are considered "rejected"

**Trading Applications:**
- Range trading: Buy at VAL, sell at VAH
- Breakout trading: Trade breaks above VAH or below VAL
- Support/Resistance: VA boundaries act as key levels

### High Volume Nodes (HVN)

Price levels with significantly higher volume than surrounding levels.

**Characteristics:**
- Strong support/resistance zones
- Prices "stick" at these levels
- Difficult to break through

**Trading Applications:**
- Expect price consolidation at HVNs
- Use as entry/exit points
- Require volume confirmation for breakouts

### Low Volume Nodes (LVN)

Price levels with significantly lower volume than surrounding levels.

**Characteristics:**
- Weak support/resistance
- Prices move quickly through these zones
- Market rejected these prices

**Trading Applications:**
- Expect fast price movement through LVNs
- Avoid placing stops in LVN zones
- Target LVNs for quick price transitions

## Implementation Details

### Algorithm

1. **Price Bucketing**
   ```rust
   // Round price to nearest tick_size
   let price_bucket = (price / tick_size).round() * tick_size;
   ```

2. **Volume Accumulation**
   ```rust
   // Accumulate volume at each price level
   price_levels.entry(price_bucket)
       .and_modify(|level| level.add_trade(trade))
       .or_insert(PriceLevel::new(price_bucket, trade));
   ```

3. **POC Calculation**
   ```rust
   // Find price level with maximum volume
   let poc = price_levels.iter()
       .max_by(|a, b| a.volume.partial_cmp(&b.volume).unwrap())
       .map(|level| level.price)
       .unwrap_or(0.0);
   ```

4. **Value Area Calculation**
   ```rust
   // Sort by volume descending
   price_levels.sort_by(|a, b| b.volume.partial_cmp(&a.volume).unwrap());

   // Accumulate until 70% of total volume
   let target_volume = total_volume * 0.70;
   let mut accumulated = 0.0;
   let mut value_area_prices = Vec::new();

   for level in &price_levels {
       if accumulated >= target_volume { break; }
       accumulated += level.volume;
       value_area_prices.push(level.price);
   }

   // Value area = [min, max] of these prices
   let value_area_low = value_area_prices.iter().min().unwrap();
   let value_area_high = value_area_prices.iter().max().unwrap();
   ```

### Performance Characteristics

- **Time Complexity**: O(n log n) where n = number of trades
  - O(n) for accumulation
  - O(k log k) for sorting price levels (k = unique price levels)
- **Space Complexity**: O(k) where k = number of unique price levels
- **Target Performance**: >100K trades/sec
- **Memory**: <1KB per profile

### Choosing Tick Size

The `tick_size` parameter determines price granularity:

| Tick Size | Use Case | Example |
|-----------|----------|---------|
| 0.01 | High precision, crypto pairs | $0.01 for BTCUSDT |
| 0.1 | Medium precision | $0.10 for stocks |
| 1.0 | Broader view, major levels | $1.00 for intraday analysis |
| 10.0 | Macro view | $10.00 for daily/weekly |

**Guidelines:**
- **Too small**: Noise, fragmented volume distribution
- **Too large**: Loss of detail, misses important levels
- **Rule of thumb**: 0.5-1% of average price

Example for BTCUSDT at $40,000:
- Fine: 0.01 (0.000025% of price)
- Medium: 1.0 (0.0025%)
- Coarse: 10.0 (0.025%)

## Trading Strategies

### 1. Range-Bound Strategy

**Concept**: Trade between Value Area boundaries

**Rules:**
- **Buy**: Price near VAL + confirmation
- **Sell**: Price near VAH + confirmation
- **Stop**: Outside value area

**Best For:**
- Consolidating markets
- Low volatility periods
- When POC is centered

```rust
if price <= value_area_low && distance < threshold {
    Signal::Buy
} else if price >= value_area_high && distance < threshold {
    Signal::Sell
} else {
    Signal::Hold
}
```

### 2. Mean Reversion Strategy

**Concept**: Prices return to POC (fair value)

**Rules:**
- **Buy**: Price significantly below POC
- **Sell**: Price significantly above POC
- **Target**: POC (fair value)

**Best For:**
- Established trends with pullbacks
- Markets with clear POC
- After extended moves

```rust
let distance_from_poc = (price - poc).abs() / poc;
if price < poc && distance_from_poc > 0.02 {
    Signal::Buy  // 2%+ below fair value
}
```

### 3. Breakout Strategy

**Concept**: Trade breaks outside value area with volume confirmation

**Rules:**
- **Buy**: Break above VAH with volume increase
- **Sell**: Break below VAL with volume increase
- **Confirm**: Volume > average at breakout level

**Best For:**
- Trending markets
- High volatility
- News-driven moves

```rust
if price > value_area_high && current_volume > avg_volume * 1.5 {
    Signal::Buy  // Breakout with volume confirmation
}
```

### 4. Volume Profile Confluence

**Concept**: Combine with other indicators

**Combine With:**
- **Moving Averages**: VA + MA crossover
- **RSI**: Oversold at VAL, overbought at VAH
- **Support/Resistance**: VA boundaries + horizontal levels

**Example:**
```rust
let at_support = price <= value_area_low;
let oversold = rsi < 30.0;
let above_ma = price > moving_average_50;

if at_support && oversold && above_ma {
    Signal::Buy  // Triple confluence
}
```

## Usage Examples

### Basic Profile

```rust
use kimsfinance_core::analysis::volume_profile::VolumeProfileBuilder;

let builder = VolumeProfileBuilder::new(1.0); // $1 tick size
let profile = builder.build(&trades);

println!("POC: ${:.2}", profile.point_of_control);
println!("VAH: ${:.2}", profile.value_area_high);
println!("VAL: ${:.2}", profile.value_area_low);
```

### Custom Value Area

```rust
// Use 80% value area instead of standard 70%
let builder = VolumeProfileBuilder::new(1.0)
    .value_area_pct(0.80);

let profile = builder.build(&trades);
```

### Multiple Timeframes

```rust
// Build hourly profiles for comparison
let profiles = builder.build_for_timeframe(
    &trades,
    Timeframe::hours(1)
);

for (i, profile) in profiles.iter().enumerate() {
    println!("Hour {}: POC = ${:.2}", i, profile.point_of_control);
}
```

### Trading Strategy

```rust
use kimsfinance_core::backtest::volume_profile_strategy::VolumeProfileStrategy;
use std::time::Duration;

let strategy = VolumeProfileStrategy::new(
    1.0,                           // $1 tick size
    Duration::from_secs(3600),     // 1 hour lookback
    0.02,                          // 2% distance threshold
);

// Use in tick engine
let engine = TickEngine::new(strategy);
let results = engine.run(&trades)?;
```

## Best Practices

### 1. Timeframe Selection

- **Intraday**: 5-15 minute profiles
- **Swing**: Daily profiles
- **Position**: Weekly/monthly profiles

Match profile period to your trading timeframe.

### 2. Volume Analysis

- **High Volume**: Strong support/resistance
- **Low Volume**: Weak levels, fast moves
- **Increasing Volume**: Trend confirmation
- **Decreasing Volume**: Potential reversal

### 3. Profile Shapes

**Balanced Profile (B-shape)**
- POC centered in range
- Normal distribution
- Range-bound market
- Trade mean reversion

**P-shape Profile**
- POC at top
- Distribution pullback after rally
- Potential reversal zone
- Watch for break below VAL

**b-shape Profile**
- POC at bottom
- Distribution bounce after decline
- Potential reversal zone
- Watch for break above VAH

### 4. Combining Sessions

- **Overnight**: Compare day vs night profiles
- **Multi-day**: Composite profiles show longer-term value
- **Weekly**: Identify macro support/resistance

### 5. Risk Management

- **Stop Loss**: Place outside value area
- **Position Size**: Reduce at LVNs (fast moves)
- **Take Profit**: VAH/VAL or POC depending on strategy

## Common Pitfalls

### 1. Wrong Tick Size

**Problem**: Too fine = noise, too coarse = missing levels

**Solution**: Start with 0.5-1% of price, adjust based on results

### 2. Insufficient Data

**Problem**: < 100 trades produces unreliable profile

**Solution**: Ensure minimum 500-1000 trades per profile

### 3. Ignoring Market Context

**Problem**: Trading VA in strong trend

**Solution**: Identify market regime first (trending vs ranging)

### 4. No Volume Confirmation

**Problem**: Trading breaks without volume

**Solution**: Require 1.5x+ average volume for breakouts

### 5. Fixed Timeframe

**Problem**: Using same period for all conditions

**Solution**: Adjust based on volatility and trading style

## Performance Tuning

### Memory Optimization

```rust
// Pre-allocate capacity
let mut builder = VolumeProfileBuilder::new(1.0);
let estimated_levels = (price_range / tick_size) as usize;
```

### Rebuild Frequency

```rust
// Balance between accuracy and performance
let strategy = VolumeProfileStrategy::new(1.0, lookback, threshold)
    .rebuild_interval(100);  // Rebuild every 100 trades
```

### Lookback Window

```rust
// Shorter = faster, longer = more context
let lookback = Duration::from_secs(1800);  // 30 minutes
```

## Further Reading

### Books
- **Market Profile**: "Mind Over Markets" by James Dalton
- **Volume Analysis**: "Volume Profile: The Insider's Guide" by Trader Dale

### Papers
- "Volume Profile Techniques for Financial Markets" (2015)
- "Price Discovery through Volume Analysis" (2018)

### Online Resources
- TradingView: Volume Profile indicator
- CME Group: Market Profile educational resources
- Jigsaw Trading: Order flow and volume analysis

## API Reference

See inline documentation:
- `VolumeProfileBuilder`: Profile configuration and building
- `VolumeProfile`: Result structure with POC and VA
- `PriceLevel`: Individual price level data
- `VolumeProfileStrategy`: Tick strategy implementation

Run `cargo doc --open` to view full API documentation.
