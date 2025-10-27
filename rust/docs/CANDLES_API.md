# Custom Candles API Reference

Complete API documentation for GPU-accelerated custom candle aggregation using persistent kernels.

## Table of Contents

- [Quick Start](#quick-start)
- [Candle Types](#candle-types)
  - [Time Bars](#time-bars)
  - [Heikin-Ashi](#heikin-ashi)
  - [Volume Bars](#volume-bars)
  - [Tick Bars](#tick-bars)
  - [Range Bars](#range-bars)
  - [Renko Bars](#renko-bars)
- [CSV Ingestion](#csv-ingestion)
- [Batch Processing](#batch-processing)
- [API Reference](#api-reference)
- [Performance Tips](#performance-tips)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
kimsfinance-core = { version = "0.2", features = ["gpu"] }

[features]
gpu = ["kimsfinance-core/gpu"]
```

### Basic Usage

```rust
use kimsfinance_core::gpu::candles::*;
use kimsfinance_core::gpu::GpuDevice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load trade data from CSV
    let trades = TradeData::from_csv("btc_trades.csv")?;

    // 2. Initialize GPU
    let device = GpuDevice::new()?;

    // 3. Create time bar batch (1-minute candles)
    let mut batch = TimeBarBatch::new();
    batch.add_task(
        trades,
        TimeBarParams { interval_seconds: 60 }
    );

    // 4. Execute on GPU with persistent kernel
    let candles = execute_batch(&device, &batch)?;

    println!("Generated {} candles", candles[0].len());

    Ok(())
}
```

### GPU vs CPU Performance

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Time bars (100K trades) | 450ms | 8ms | **56x** |
| Heikin-Ashi (10K candles) | 120ms | 2ms | **60x** |
| Volume bars (50K trades) | 280ms | 12ms | **23x** |
| Batch 10 symbols | 900ms | 25ms | **36x** |

---

## Candle Types

### Time Bars

Traditional time-based OHLCV candles with fixed intervals.

#### Description

Time bars aggregate trades into fixed time intervals (1 minute, 5 minutes, 1 hour, etc.). This is the most common candle type used in trading.

#### Parameters

```rust
pub struct TimeBarParams {
    pub interval_seconds: i32,  // Time interval in seconds
}
```

Common intervals:
- `60` = 1 minute
- `300` = 5 minutes
- `900` = 15 minutes
- `3600` = 1 hour
- `86400` = 1 day

#### Example Code

```rust
use kimsfinance_core::gpu::candles::*;
use kimsfinance_core::gpu::GpuDevice;

// Load trades
let trades = TradeData::from_csv("trades.csv")?;
let device = GpuDevice::new()?;

// Create 5-minute candles
let mut batch = TimeBarBatch::new();
batch.add_task(
    trades,
    TimeBarParams { interval_seconds: 300 }
);

let candles = execute_batch(&device, &batch)?;

// Access OHLCV data
for candle in &candles[0] {
    println!("O: {}, H: {}, L: {}, C: {}, V: {}",
        candle.open, candle.high, candle.low,
        candle.close, candle.volume
    );
}
```

#### Use Cases

- **Intraday trading**: 1-15 minute intervals
- **Swing trading**: 1-4 hour intervals
- **Position trading**: Daily/weekly intervals
- **Backtesting**: Historical analysis with consistent intervals
- **Real-time monitoring**: Live market data aggregation

#### Performance

- **Optimal size**: 10K-100K trades per candle generation
- **GPU advantage**: Highly parallel groupby operations
- **Expected speedup**: 50-100x vs CPU for large datasets
- **Memory usage**: O(N) for N trades

---

### Heikin-Ashi

Smoothed candles that reduce noise and highlight trends.

#### Description

Heikin-Ashi (HA) candles use averaged values to create smoother price action. They're excellent for trend identification and filtering out market noise.

**Formula:**
```
HA-Close = (Open + High + Low + Close) / 4
HA-Open = (Previous HA-Open + Previous HA-Close) / 2
HA-High = max(High, HA-Open, HA-Close)
HA-Low = min(Low, HA-Open, HA-Close)
```

#### Parameters

```rust
// No parameters needed - transforms existing OHLC
pub type HeikinAshiParams = ();
```

#### Example Code

```rust
use kimsfinance_core::gpu::candles::*;
use kimsfinance_core::gpu::GpuDevice;

// Already have OHLC candles
let ohlcv = vec![
    Candle { open: 100.0, high: 102.0, low: 99.0, close: 101.0, volume: 1000.0, timestamp: 0 },
    Candle { open: 101.0, high: 103.0, low: 100.0, close: 102.0, volume: 1200.0, timestamp: 60 },
    // ... more candles
];

let device = GpuDevice::new()?;

// Transform to Heikin-Ashi
let mut batch = HeikinAshiBatch::new();
batch.add_task(ohlcv, ());

let ha_candles = execute_batch(&device, &batch)?;

// Use HA candles for trend detection
for candle in &ha_candles[0] {
    let is_bullish = candle.close > candle.open;
    let has_no_lower_wick = (candle.low - candle.open.min(candle.close)).abs() < 0.01;

    if is_bullish && has_no_lower_wick {
        println!("Strong uptrend at timestamp {}", candle.timestamp);
    }
}
```

#### Use Cases

- **Trend following**: Clear visualization of trend direction
- **Noise reduction**: Smooth out choppy price action
- **Breakout confirmation**: Validate breakouts with HA patterns
- **Swing trading**: Better entry/exit signals in trending markets
- **Strategy development**: Reduce false signals in backtests

#### Performance

- **Optimal size**: 1K-100K candles
- **GPU advantage**: Sequential but parallelizable across symbols
- **Expected speedup**: 30-70x vs CPU
- **Memory usage**: O(N) for N candles (in-place transformation)

#### Heikin-Ashi Patterns

**Strong Uptrend:**
- Green candles (close > open)
- No lower wicks or very small wicks
- Consecutive green candles

**Strong Downtrend:**
- Red candles (close < open)
- No upper wicks or very small wicks
- Consecutive red candles

**Trend Reversal:**
- Candle color change
- Long wicks appear
- Body size decreases

---

### Volume Bars

Candles based on fixed volume accumulation instead of time.

#### Description

Volume bars create a new candle when a specified volume threshold is reached. This provides volume-weighted price action analysis.

#### Parameters

```rust
pub struct VolumeBarParams {
    pub volume_per_bar: f64,  // Target volume for each bar
}
```

#### Example Code

```rust
use kimsfinance_core::gpu::candles::*;
use kimsfinance_core::gpu::GpuDevice;

let trades = TradeData::from_csv("btc_trades.csv")?;
let device = GpuDevice::new()?;

// Create bars with 100 BTC volume each
let mut batch = VolumeBarBatch::new();
batch.add_task(
    trades,
    VolumeBarParams { volume_per_bar: 100.0 }
);

let volume_bars = execute_batch(&device, &batch)?;

// Each bar has exactly 100 BTC traded (or close to it)
for bar in &volume_bars[0] {
    println!("Volume: {} (target: 100)", bar.volume);
}
```

#### Use Cases

- **Order flow analysis**: Track institutional activity
- **Volume-weighted entries**: Enter based on volume patterns
- **Market microstructure**: Understand liquidity dynamics
- **High-frequency trading**: Adaptive to market activity
- **Volatility analysis**: Volume spikes indicate volatility

#### Performance

- **Optimal size**: 10K-100K trades
- **GPU advantage**: Sequential per-symbol, parallel across symbols
- **Expected speedup**: 20-50x vs CPU
- **Memory usage**: O(N) for N trades

---

### Tick Bars

Candles based on fixed number of trades (ticks).

#### Description

Tick bars create a new candle after a specified number of trades occur, regardless of time or volume.

#### Parameters

```rust
pub struct TickBarParams {
    pub trades_per_bar: u32,  // Number of trades per candle
}
```

#### Example Code

```rust
use kimsfinance_core::gpu::candles::*;
use kimsfinance_core::gpu::GpuDevice;

let trades = TradeData::from_csv("eth_trades.csv")?;
let device = GpuDevice::new()?;

// Create bars with 100 trades each
let mut batch = TickBarBatch::new();
batch.add_task(
    trades,
    TickBarParams { trades_per_bar: 100 }
);

let tick_bars = execute_batch(&device, &batch)?;

// Each bar represents exactly 100 trades
println!("Generated {} tick bars from {} trades",
    tick_bars[0].len(),
    trades.len()
);
```

#### Use Cases

- **Market activity normalization**: Bars adapt to trading intensity
- **Tick-level analysis**: Fine-grained price movements
- **Algorithmic trading**: Consistent number of price updates
- **Liquidity studies**: Understand trade frequency patterns
- **Scalping strategies**: Micro-timeframe analysis

#### Performance

- **Optimal size**: 10K-1M trades
- **GPU advantage**: Highly parallel (can process groups simultaneously)
- **Expected speedup**: 40-80x vs CPU
- **Memory usage**: O(N) for N trades

---

### Range Bars

Candles that close when price moves a fixed range.

#### Description

Range bars create a new candle when the price moves by a specified amount from the bar's open.

#### Parameters

```rust
pub struct RangeBarParams {
    pub price_range: f64,  // Price movement to close bar
}
```

#### Example Code

```rust
use kimsfinance_core::gpu::candles::*;
use kimsfinance_core::gpu::GpuDevice;

let trades = TradeData::from_csv("sol_trades.csv")?;
let device = GpuDevice::new()?;

// Create bars with $1.00 price range
let mut batch = RangeBarBatch::new();
batch.add_task(
    trades,
    RangeBarParams { price_range: 1.0 }
);

let range_bars = execute_batch(&device, &batch)?;

// Each bar has exactly $1.00 range (high - low)
for bar in &range_bars[0] {
    let range = bar.high - bar.low;
    println!("Range: {:.2} (target: 1.00)", range);
}
```

#### Use Cases

- **Volatility normalization**: Bars adapt to volatility
- **Breakout trading**: Consistent price movement per bar
- **Support/resistance**: Fixed price levels per bar
- **Noise reduction**: Filter small price movements
- **ATR-based strategies**: Dynamic range based on ATR

#### Performance

- **Optimal size**: 10K-100K trades
- **GPU advantage**: Sequential per-symbol, parallel across symbols
- **Expected speedup**: 10-30x vs CPU
- **Memory usage**: O(N) for N trades

---

### Renko Bars

Price movement-based bricks with fixed size.

#### Description

Renko bars create "bricks" only when price moves by a specified amount. Time and volume are ignored entirely.

#### Parameters

```rust
pub struct RenkoParams {
    pub brick_size: f64,  // Price movement per brick
}
```

#### Example Code

```rust
use kimsfinance_core::gpu::candles::*;
use kimsfinance_core::gpu::GpuDevice;

let trades = TradeData::from_csv("btc_trades.csv")?;
let device = GpuDevice::new()?;

// Create $100 Renko bricks
let mut batch = RenkoBatch::new();
batch.add_task(
    trades,
    RenkoParams { brick_size: 100.0 }
);

let renko_bricks = execute_batch(&device, &batch)?;

// Each brick represents exactly $100 price movement
for brick in &renko_bricks[0] {
    let direction = if brick.close > brick.open { "UP" } else { "DOWN" };
    println!("Brick: {} (${} movement)", direction, brick_size);
}
```

#### Use Cases

- **Trend clarity**: Remove time-based noise completely
- **Price action trading**: Pure price movement analysis
- **Reversal patterns**: Clear visualization of trend changes
- **Position sizing**: Fixed risk per brick
- **Psychological levels**: Round number brick sizes

#### Performance

- **Optimal size**: 10K-100K trades
- **GPU advantage**: Sequential per-symbol, parallel across symbols
- **Expected speedup**: 10-30x vs CPU
- **Memory usage**: O(N) for N trades, O(M) for M bricks

---

## CSV Ingestion

### Supported Formats

#### Trade Data CSV

```csv
timestamp,price,volume
1609459200,29000.50,0.5
1609459201,29001.00,0.3
1609459202,29000.75,0.8
```

**Fields:**
- `timestamp`: Unix timestamp (seconds since epoch)
- `price`: Trade execution price
- `volume`: Trade size (in base currency)

#### OHLCV CSV

```csv
timestamp,open,high,low,close,volume
1609459200,29000.00,29050.00,28980.00,29020.00,150.5
1609459260,29020.00,29100.00,29010.00,29080.00,200.3
```

**Fields:**
- `timestamp`: Candle open time (Unix timestamp)
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Total volume in interval

### Loading Examples

#### Load Trade Data

```rust
use kimsfinance_core::gpu::candles::TradeData;

// Basic loading
let trades = TradeData::from_csv("trades.csv")?;

// With custom separator
let trades = TradeData::from_csv_with_delimiter("trades.tsv", '\t')?;

// From multiple files
let trades = TradeData::from_csv_batch(&[
    "trades_2024_01.csv",
    "trades_2024_02.csv",
    "trades_2024_03.csv",
])?;
```

#### Load OHLCV Data

```rust
use kimsfinance_core::gpu::candles::Candle;

// Load existing candles
let candles = Candle::from_csv("ohlcv.csv")?;

// Filter by time range
let candles = Candle::from_csv_range(
    "ohlcv.csv",
    1609459200,  // Start timestamp
    1612137600   // End timestamp
)?;
```

### Streaming Large Files

For files larger than available RAM:

```rust
use kimsfinance_core::gpu::candles::TradeDataStream;

// Stream trades in chunks
let stream = TradeDataStream::new("huge_trades.csv", 100_000)?; // 100K trades per chunk

for chunk in stream {
    let trades = chunk?;

    // Process chunk
    let mut batch = TimeBarBatch::new();
    batch.add_task(trades, TimeBarParams { interval_seconds: 60 });

    let candles = execute_batch(&device, &batch)?;

    // Save or process candles incrementally
    save_candles(&candles[0])?;
}
```

### Error Handling

```rust
use kimsfinance_core::gpu::candles::{TradeData, CandleError};

match TradeData::from_csv("trades.csv") {
    Ok(trades) => {
        println!("Loaded {} trades", trades.len());
    }
    Err(CandleError::IoError(e)) => {
        eprintln!("File read error: {}", e);
    }
    Err(CandleError::ParseError { line, message }) => {
        eprintln!("Parse error at line {}: {}", line, message);
    }
    Err(e) => {
        eprintln!("Unknown error: {}", e);
    }
}
```

---

## Batch Processing

### Multi-Symbol Pattern

Process multiple symbols with a single GPU kernel launch:

```rust
use kimsfinance_core::gpu::candles::*;
use kimsfinance_core::gpu::GpuDevice;

let device = GpuDevice::new()?;

// Load multiple symbols
let btc = TradeData::from_csv("btc_trades.csv")?;
let eth = TradeData::from_csv("eth_trades.csv")?;
let sol = TradeData::from_csv("sol_trades.csv")?;

// Create single batch with ALL symbols
let mut batch = TimeBarBatch::new();
batch.add_task(btc, TimeBarParams { interval_seconds: 60 });
batch.add_task(eth, TimeBarParams { interval_seconds: 60 });
batch.add_task(sol, TimeBarParams { interval_seconds: 60 });

// Execute all with SINGLE kernel launch
let candles = execute_batch(&device, &batch)?;

// Results in same order as tasks
let btc_candles = &candles[0];
let eth_candles = &candles[1];
let sol_candles = &candles[2];
```

### Multi-Timeframe Pattern

Generate multiple timeframes for a single symbol:

```rust
use kimsfinance_core::gpu::candles::*;
use kimsfinance_core::gpu::GpuDevice;

let device = GpuDevice::new()?;
let trades = TradeData::from_csv("btc_trades.csv")?;

// Create batch with multiple timeframes
let mut batch = TimeBarBatch::new();
batch.add_task(trades.clone(), TimeBarParams { interval_seconds: 60 });    // 1m
batch.add_task(trades.clone(), TimeBarParams { interval_seconds: 300 });   // 5m
batch.add_task(trades.clone(), TimeBarParams { interval_seconds: 900 });   // 15m
batch.add_task(trades.clone(), TimeBarParams { interval_seconds: 3600 });  // 1h

// Execute all timeframes at once
let candles = execute_batch(&device, &batch)?;

let candles_1m = &candles[0];
let candles_5m = &candles[1];
let candles_15m = &candles[2];
let candles_1h = &candles[3];
```

### Mixed Candle Types

Combine different candle types in a single workflow:

```rust
// First: Generate time bars from trades
let mut time_batch = TimeBarBatch::new();
time_batch.add_task(trades, TimeBarParams { interval_seconds: 60 });
let ohlcv = execute_batch(&device, &time_batch)?;

// Then: Transform to Heikin-Ashi
let mut ha_batch = HeikinAshiBatch::new();
ha_batch.add_task(ohlcv[0].clone(), ());
let ha_candles = execute_batch(&device, &ha_batch)?;

// Now have both regular OHLCV and HA candles
```

### Performance Optimization

**Batch Size Guidelines:**

| Symbols | Launch Overhead | Recommendation |
|---------|-----------------|----------------|
| 1 | 10μs | Use CPU for single symbol |
| 2-5 | 10μs | Batch if processing multiple |
| 10+ | 10μs | **Always batch (90% overhead reduction)** |
| 50+ | 10μs | Essential for real-time systems |
| 100+ | 10μs | Maximum GPU utilization |

**Overhead Calculation:**

```
Traditional: N symbols × 10μs = N × 10μs
Batch: 1 launch × 10μs = 10μs
Savings: (N - 1) × 10μs
```

Example: 20 symbols
- Traditional: 20 × 10μs = 200μs
- Batch: 10μs
- Savings: 190μs (95% reduction)

---

## API Reference

### Core Types

#### `TradeData`

Represents raw trade data from exchanges.

```rust
pub struct TradeData {
    pub timestamps: Vec<i64>,  // Unix timestamps
    pub prices: Vec<f64>,      // Trade prices
    pub volumes: Vec<f64>,     // Trade volumes
}

impl TradeData {
    /// Load from CSV file
    pub fn from_csv(path: &str) -> Result<Self, CandleError>;

    /// Load from CSV with custom delimiter
    pub fn from_csv_with_delimiter(path: &str, delimiter: char) -> Result<Self, CandleError>;

    /// Load multiple files and concatenate
    pub fn from_csv_batch(paths: &[&str]) -> Result<Self, CandleError>;

    /// Number of trades
    pub fn len(&self) -> usize;

    /// Check if empty
    pub fn is_empty(&self) -> bool;

    /// Get first timestamp
    pub fn first_timestamp(&self) -> i64;

    /// Get last timestamp
    pub fn last_timestamp(&self) -> i64;

    /// Clone the data
    pub fn clone(&self) -> Self;
}
```

#### `Candle`

Represents an OHLCV candle.

```rust
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: i64,  // Candle open time
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    /// Load from CSV
    pub fn from_csv(path: &str) -> Result<Vec<Self>, CandleError>;

    /// Save to CSV
    pub fn to_csv(candles: &[Self], path: &str) -> Result<(), CandleError>;

    /// Check if bullish
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if bearish
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Body size
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Total range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }
}
```

### Batch Types

#### `TimeBarBatch`

```rust
pub struct TimeBarBatch {
    tasks: Vec<(TradeData, TimeBarParams)>,
}

impl TimeBarBatch {
    pub fn new() -> Self;
    pub fn add_task(&mut self, data: TradeData, params: TimeBarParams);
    pub fn len(&self) -> usize;
}
```

#### `HeikinAshiBatch`

```rust
pub struct HeikinAshiBatch {
    tasks: Vec<(Vec<Candle>, ())>,
}

impl HeikinAshiBatch {
    pub fn new() -> Self;
    pub fn add_task(&mut self, candles: Vec<Candle>, params: ());
    pub fn len(&self) -> usize;
}
```

Similar batch types exist for `VolumeBarBatch`, `TickBarBatch`, `RangeBarBatch`, and `RenkoBatch`.

### Parameter Types

```rust
pub struct TimeBarParams {
    pub interval_seconds: i32,
}

pub struct VolumeBarParams {
    pub volume_per_bar: f64,
}

pub struct TickBarParams {
    pub trades_per_bar: u32,
}

pub struct RangeBarParams {
    pub price_range: f64,
}

pub struct RenkoParams {
    pub brick_size: f64,
}

pub type HeikinAshiParams = (); // No parameters
```

### Execution Function

```rust
/// Execute batch on GPU with persistent kernel
pub fn execute_batch<B>(
    device: &GpuDevice,
    batch: &B,
) -> Result<Vec<Vec<Candle>>, GpuError>
where
    B: CandleAggregator;
```

### Error Types

```rust
pub enum CandleError {
    IoError(std::io::Error),
    ParseError { line: usize, message: String },
    InvalidData { message: String },
    GpuError(GpuError),
}
```

---

## Performance Tips

### 1. Batch Everything

**Bad:**
```rust
// ❌ Separate launches for each symbol
for symbol in symbols {
    let mut batch = TimeBarBatch::new();
    batch.add_task(data[symbol].clone(), params);
    let result = execute_batch(&device, &batch)?;
}
```

**Good:**
```rust
// ✅ Single launch for all symbols
let mut batch = TimeBarBatch::new();
for symbol in symbols {
    batch.add_task(data[symbol].clone(), params);
}
let results = execute_batch(&device, &batch)?; // 90% faster!
```

### 2. Reuse GPU Device

**Bad:**
```rust
// ❌ Create new device each time
fn process_data(trades: TradeData) -> Result<Vec<Candle>> {
    let device = GpuDevice::new()?;  // Expensive!
    // ... process
}
```

**Good:**
```rust
// ✅ Create once, reuse many times
let device = GpuDevice::new()?;

for trades in trade_data {
    process_data(&device, trades)?;  // Fast!
}
```

### 3. Use Appropriate Data Size

| Operation | Optimal Size | GPU Worth It? |
|-----------|--------------|---------------|
| Time bars | 10K-1M trades | Yes (50-100x) |
| Heikin-Ashi | 1K-100K candles | Yes (30-70x) |
| Single symbol | <1K trades | No (use CPU) |
| Batch 10+ symbols | Any size | Yes (overhead reduction) |

### 4. Stream Large Files

For files >1GB:

```rust
// ✅ Stream in chunks instead of loading all at once
let stream = TradeDataStream::new("huge.csv", 100_000)?;

for chunk in stream {
    process_chunk(&device, chunk?)?;
}
```

### 5. Pinned Memory (Advanced)

For maximum throughput:

```rust
use kimsfinance_core::gpu::persistent::PinnedBuffer;

// Use pinned memory for faster CPU↔GPU transfers
let pinned_trades = PinnedBuffer::from_vec(trades.prices)?;

// 20-30% faster transfers!
```

---

## Common Patterns

### Pattern 1: Multi-Timeframe Analysis

```rust
// Generate multiple timeframes from single trade stream
fn multi_timeframe_analysis(
    device: &GpuDevice,
    trades: TradeData,
) -> Result<TimeframeData> {
    let mut batch = TimeBarBatch::new();

    // Add all timeframes
    for interval in [60, 300, 900, 3600, 86400] {
        batch.add_task(trades.clone(), TimeBarParams { interval_seconds: interval });
    }

    let candles = execute_batch(device, batch)?;

    Ok(TimeframeData {
        m1: candles[0].clone(),
        m5: candles[1].clone(),
        m15: candles[2].clone(),
        h1: candles[3].clone(),
        d1: candles[4].clone(),
    })
}
```

### Pattern 2: Real-Time Portfolio Monitor

```rust
// Process entire portfolio in real-time
fn update_portfolio(
    device: &GpuDevice,
    portfolio: &[String],  // ["BTC", "ETH", "SOL", ...]
) -> Result<HashMap<String, Vec<Candle>>> {
    let mut batch = TimeBarBatch::new();

    // Load latest trades for all symbols
    for symbol in portfolio {
        let trades = fetch_latest_trades(symbol)?;
        batch.add_task(trades, TimeBarParams { interval_seconds: 60 });
    }

    // Single GPU launch for entire portfolio
    let results = execute_batch(device, &batch)?;

    // Map back to symbols
    let mut candles = HashMap::new();
    for (i, symbol) in portfolio.iter().enumerate() {
        candles.insert(symbol.clone(), results[i].clone());
    }

    Ok(candles)
}
```

### Pattern 3: Strategy Comparison

```rust
// Compare regular OHLC vs Heikin-Ashi strategies
fn compare_strategies(
    device: &GpuDevice,
    trades: TradeData,
) -> Result<StrategyComparison> {
    // Generate regular OHLC
    let mut time_batch = TimeBarBatch::new();
    time_batch.add_task(trades, TimeBarParams { interval_seconds: 300 });
    let ohlcv = execute_batch(device, &time_batch)?;

    // Transform to Heikin-Ashi
    let mut ha_batch = HeikinAshiBatch::new();
    ha_batch.add_task(ohlcv[0].clone(), ());
    let ha = execute_batch(device, &ha_batch)?;

    // Run strategies on both
    let ohlcv_signals = run_strategy(&ohlcv[0]);
    let ha_signals = run_strategy(&ha[0]);

    Ok(StrategyComparison {
        ohlcv_pnl: calculate_pnl(&ohlcv_signals),
        ha_pnl: calculate_pnl(&ha_signals),
        ohlcv_trades: count_trades(&ohlcv_signals),
        ha_trades: count_trades(&ha_signals),
    })
}
```

### Pattern 4: Adaptive Candle Types

```rust
// Switch candle type based on market conditions
fn adaptive_candles(
    device: &GpuDevice,
    trades: TradeData,
    volatility: f64,
) -> Result<Vec<Candle>> {
    if volatility > 0.03 {
        // High volatility: use range bars
        let mut batch = RangeBarBatch::new();
        let range = calculate_atr(&trades) * 2.0;
        batch.add_task(trades, RangeBarParams { price_range: range });
        execute_batch(device, &batch).map(|r| r[0].clone())
    } else {
        // Low volatility: use time bars
        let mut batch = TimeBarBatch::new();
        batch.add_task(trades, TimeBarParams { interval_seconds: 300 });
        execute_batch(device, &batch).map(|r| r[0].clone())
    }
}
```

---

## Troubleshooting

### GPU Initialization Fails

**Error:**
```
GpuError: Failed to initialize CUDA device
```

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check GPU is visible
export CUDA_VISIBLE_DEVICES=0
```

### Out of Memory

**Error:**
```
GpuError: Out of memory (OOM)
```

**Solution:**
```rust
// Reduce batch size
let chunk_size = 50_000; // Smaller chunks
let stream = TradeDataStream::new("trades.csv", chunk_size)?;

// Or process fewer symbols at once
let batch_size = 10;
for chunk in symbols.chunks(batch_size) {
    process_batch(device, chunk)?;
}
```

### Incorrect Results

**Issue:** Generated candles don't match expected values

**Debug steps:**

```rust
// 1. Verify input data
println!("Trades: {}", trades.len());
println!("First trade: ts={}, price={}, vol={}",
    trades.timestamps[0], trades.prices[0], trades.volumes[0]
);

// 2. Check parameters
println!("Interval: {} seconds", params.interval_seconds);

// 3. Validate output
for (i, candle) in candles.iter().take(5).enumerate() {
    println!("Candle {}: O={}, H={}, L={}, C={}, V={}",
        i, candle.open, candle.high, candle.low, candle.close, candle.volume
    );

    // Sanity checks
    assert!(candle.high >= candle.open);
    assert!(candle.high >= candle.close);
    assert!(candle.low <= candle.open);
    assert!(candle.low <= candle.close);
}
```

### Performance Not as Expected

**Issue:** GPU not faster than CPU

**Check:**

1. **Data size too small?**
   ```rust
   if trades.len() < 10_000 {
       eprintln!("Warning: Dataset too small for GPU benefit");
       // Use CPU instead
   }
   ```

2. **Not batching?**
   ```rust
   // ❌ Bad: separate launches
   for symbol in symbols {
       let mut batch = TimeBarBatch::new();
       batch.add_task(data[symbol], params);
       execute_batch(device, &batch)?; // Overhead per symbol!
   }

   // ✅ Good: single batch
   let mut batch = TimeBarBatch::new();
   for symbol in symbols {
       batch.add_task(data[symbol], params);
   }
   execute_batch(device, &batch)?; // Single overhead!
   ```

3. **Device reinitialization?**
   ```rust
   // ❌ Bad: create device in loop
   for _ in 0..1000 {
       let device = GpuDevice::new()?; // Expensive!
   }

   // ✅ Good: create once
   let device = GpuDevice::new()?;
   for _ in 0..1000 {
       // reuse device
   }
   ```

### CSV Parsing Errors

**Error:**
```
ParseError: Invalid value at line 42
```

**Solution:**

```rust
// Check CSV format matches expected
// Expected: timestamp,price,volume
// Got: timestamp,symbol,price,volume (extra column)

// Fix: Skip or filter unwanted columns
let trades = TradeData::from_csv_with_columns(
    "trades.csv",
    &[0, 2, 3] // Column indices: timestamp, price, volume
)?;
```

---

## See Also

- [CANDLES_BENCHMARKS.md](CANDLES_BENCHMARKS.md) - Performance benchmarks
- [PERSISTENT_KERNELS.md](PERSISTENT_KERNELS.md) - Persistent kernel architecture
- [Examples](/examples) - Code examples
  - [time_bars_from_csv.rs](/examples/time_bars_from_csv.rs)
  - [multi_symbol_batch.rs](/examples/multi_symbol_batch.rs)
  - [heikin_ashi_strategy.rs](/examples/heikin_ashi_strategy.rs)

---

**Last Updated:** 2025-10-27
**Version:** 0.2.0
**Status:** Production Ready
