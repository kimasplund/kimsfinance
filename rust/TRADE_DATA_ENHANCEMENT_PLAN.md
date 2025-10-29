# Trade Data Enhancement Plan

**Branch**: `feature/trade-data-support`
**Status**: Planning Phase
**Goal**: Enhanced tick-level backtesting with flexible timeframes

---

## Critical Architectural Questions (Answered)

### Q1: Tick-by-Tick vs Pre-Aggregated OHLCV?

**Answer: Support BOTH modes**

#### Mode 1: Tick-by-Tick Streaming (NEW)
- Strategy sees every trade as it happens
- Candles build incrementally: `on_tick(trade, incomplete_candle)`
- More realistic: mimics live trading
- Slower but allows intra-candle decisions
- Example: Enter on 3rd tick of candle if price moves 0.5%

```rust
trait TickStrategy {
    // Called for every trade
    fn on_tick(&mut self, trade: &Trade, current_candle: &IncompleteCandle) -> Signal;

    // Called when candle completes
    fn on_candle_complete(&mut self, candle: &Candle) -> Signal;
}
```

#### Mode 2: Pre-Aggregated Candles (EXISTING)
- Trade data aggregated to OHLCV first
- Strategy sees completed candles only
- Faster: processes millions of trades in seconds
- Good for trend-following, daily strategies
- Current implementation already does this

```rust
trait Strategy {
    // Called once per completed candle
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal;
}
```

**Decision**: Implement both, let user choose via config flag:
```rust
BacktestConfig {
    mode: BacktestMode::TickByTick,  // or BacktestMode::Candles
    ...
}
```

---

### Q2: Hardcoded Timeframes vs Flexible Duration?

**Answer: Replace enum with flexible Duration system**

#### Current (Hardcoded):
```rust
pub enum Timeframe {
    OneMinute,      // 60_000 ms
    FiveMinutes,    // 300_000 ms
    FifteenMinutes, // 900_000 ms
    // Only 6 options, no 3m, 2h, 45s, etc.
}
```

**Problems**:
- Can't do 3-minute candles
- Can't do 2-hour candles
- Can't do sub-minute (45s, 30s)
- Can't do custom periods

#### New (Flexible Duration):
```rust
pub struct Timeframe {
    duration: Duration,  // Any duration
}

impl Timeframe {
    // Convenience constructors
    pub fn seconds(s: u64) -> Self { Self { duration: Duration::from_secs(s) } }
    pub fn minutes(m: u64) -> Self { Self::seconds(m * 60) }
    pub fn hours(h: u64) -> Self { Self::minutes(h * 60) }
    pub fn days(d: u64) -> Self { Self::hours(d * 24) }

    // String parsing: "5m", "1h", "45s", "2D"
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let (num, unit) = parse_timeframe_string(s)?;
        match unit {
            's' | 'S' => Ok(Self::seconds(num)),
            'm' | 'M' => Ok(Self::minutes(num)),
            'h' | 'H' => Ok(Self::hours(num)),
            'd' | 'D' => Ok(Self::days(num)),
            _ => Err(ParseError::InvalidUnit(unit)),
        }
    }

    pub fn to_ms(&self) -> i64 {
        self.duration.as_millis() as i64
    }
}

// Usage examples:
let tf1 = Timeframe::minutes(5);         // 5m
let tf2 = Timeframe::parse("3m")?;       // 3m
let tf3 = Timeframe::parse("45s")?;      // 45s
let tf4 = Timeframe::parse("2h")?;       // 2h
let tf5 = Timeframe::seconds(137);       // 137s (any duration!)
```

**Benefits**:
- ✅ Any timeframe: 3m, 2h, 45s, 7m, 33s
- ✅ User-friendly parsing: `"5m"`, `"1h"`, `"2D"`
- ✅ Backward compatible (keep convenience methods)
- ✅ No enum limitations

**Migration Strategy**:
1. Keep old enum for compatibility
2. Add `From<TimeframeEnum>` trait
3. Deprecate enum gradually
4. All new code uses Duration-based system

---

## Enhancement Plan Overview

| # | Enhancement | Priority | Complexity | Impact | Dependencies |
|---|-------------|----------|------------|--------|--------------|
| 1 | Multi-file batch processing | HIGH | Low | High | None |
| 2 | Tick-level trading strategies | HIGH | Medium | Very High | Flexible timeframes |
| 3 | Market microstructure analysis | MEDIUM | Medium | High | Tick strategies |
| 4 | GPU-accelerated aggregation | MEDIUM | High | Medium | None |
| 5 | Data validation & quality | HIGH | Low | Medium | None |
| 6 | Real-time streaming | LOW | High | High | Tick strategies |

---

## Enhancement 1: Multi-File Batch Processing

**Status**: Not implemented
**Priority**: HIGH
**Effort**: 4-8 hours

### Current State
```rust
// Must specify exact file
let candles = process_binance_month(
    "/path/to/BTCUSDT-trades-2021-01-01.zip",
    Timeframe::FiveMinutes
)?;
```

### Target State
```rust
// Load entire year automatically
let candles = process_binance_directory(
    "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades/",
    "2021-01-01",  // Start date
    "2021-12-31",  // End date
    Timeframe::parse("5m")?
)?;

// Or load multiple months
let candles = process_binance_months(
    "/path/to/trades/",
    &["2021-01", "2021-02", "2021-03"],
    Timeframe::minutes(5)
)?;
```

### Implementation Plan

**1. Add date range utilities** (`src/binance/date_utils.rs`):
```rust
pub struct DateRange {
    start: NaiveDate,
    end: NaiveDate,
}

impl DateRange {
    pub fn parse(start: &str, end: &str) -> Result<Self, ParseError>;
    pub fn months(&self) -> Vec<String>;  // ["2021-01", "2021-02", ...]
    pub fn days(&self) -> Vec<String>;    // ["2021-01-01", "2021-01-02", ...]
}
```

**2. Add file discovery** (`src/binance/discovery.rs`):
```rust
pub struct BinanceDataFinder {
    base_path: PathBuf,
}

impl BinanceDataFinder {
    pub fn find_files(&self, pattern: &str) -> Vec<PathBuf>;
    pub fn find_by_date_range(&self, range: DateRange) -> Vec<PathBuf>;
    pub fn verify_checksums(&self) -> Result<(), ValidationError>;
}
```

**3. Add batch processor** (`src/binance/batch.rs`):
```rust
pub fn process_binance_directory(
    path: &str,
    start_date: &str,
    end_date: &str,
    timeframe: Timeframe,
) -> Result<Vec<Candle>, BinanceError> {
    let finder = BinanceDataFinder::new(path);
    let range = DateRange::parse(start_date, end_date)?;
    let files = finder.find_by_date_range(range);

    let mut all_candles = Vec::new();
    for file in files {
        let candles = process_binance_month(&file, timeframe)?;
        all_candles.extend(candles);
    }

    // Sort by timestamp (files might be out of order)
    all_candles.sort_by_key(|c| c.timestamp);

    Ok(all_candles)
}
```

**Files to Create**:
- `src/binance/date_utils.rs` (150 lines)
- `src/binance/discovery.rs` (200 lines)
- `src/binance/batch.rs` (250 lines)

**Tests**:
- Test date range parsing
- Test file discovery with real directory
- Test batch loading with 2021-01, 2021-02, 2021-03
- Benchmark: Load entire 2021 year (4.2M+ trades/month)

---

## Enhancement 2: Tick-Level Trading Strategies

**Status**: Not implemented
**Priority**: HIGH
**Effort**: 16-24 hours

### Current State
```rust
// Only sees completed candles
trait Strategy {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal;
}
```

**Limitations**:
- Can't react to intra-candle price movements
- Can't see order flow during candle formation
- Misses high-frequency opportunities
- Not realistic for short timeframes

### Target State
```rust
// NEW: Tick-aware strategy trait
trait TickStrategy {
    // Called for every trade (4.6M times per day!)
    fn on_tick(&mut self, trade: &Trade, candle: &IncompleteCandle) -> Signal;

    // Called when candle completes
    fn on_candle_complete(&mut self, candle: &Candle) -> Signal;

    // Optional: batch process N ticks at once
    fn on_tick_batch(&mut self, trades: &[Trade], candle: &IncompleteCandle) -> Signal {
        Signal::Hold  // Default: ignore batch, override if needed
    }
}

// Example: Momentum burst strategy
struct IntraCandleMomentum {
    entry_threshold: f64,  // 0.5% price move
    first_tick_price: Option<f64>,
}

impl TickStrategy for IntraCandleMomentum {
    fn on_tick(&mut self, trade: &Trade, candle: &IncompleteCandle) -> Signal {
        if self.first_tick_price.is_none() {
            self.first_tick_price = Some(trade.price);
            return Signal::Hold;
        }

        let first_price = self.first_tick_price.unwrap();
        let price_change_pct = (trade.price - first_price) / first_price * 100.0;

        // Enter on 0.5% move within candle
        if price_change_pct > self.entry_threshold {
            Signal::Buy
        } else if price_change_pct < -self.entry_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn on_candle_complete(&mut self, _candle: &Candle) -> Signal {
        self.first_tick_price = None;  // Reset for next candle
        Signal::Hold
    }
}
```

### IncompleteCandle Struct
```rust
/// Candle that's still forming (updated with each tick)
#[derive(Debug, Clone)]
pub struct IncompleteCandle {
    pub timestamp: i64,        // Candle start time
    pub open: f64,             // First trade price
    pub high: f64,             // Highest so far
    pub low: f64,              // Lowest so far
    pub close: f64,            // Latest trade price
    pub volume: f64,           // Volume so far
    pub quote_volume: f64,     // Quote volume so far
    pub num_trades: usize,     // Trades so far
    pub is_complete: bool,     // false until candle closes
}

impl IncompleteCandle {
    pub fn update(&mut self, trade: &Trade) {
        self.high = self.high.max(trade.price);
        self.low = self.low.min(trade.price);
        self.close = trade.price;
        self.volume += trade.quantity;
        self.quote_volume += trade.quote_quantity;
        self.num_trades += 1;
    }

    pub fn complete(self) -> Candle {
        Candle { /* convert */ }
    }
}
```

### Tick-by-Tick Backtest Engine
```rust
pub struct TickBacktestEngine {
    config: BacktestConfig,
    timeframe: Timeframe,
}

impl TickBacktestEngine {
    pub fn run_tick<S: TickStrategy>(
        &self,
        strategy: &mut S,
        trades: &[Trade],
    ) -> Result<BacktestResult, BacktestError> {
        let mut candle_builder: Option<IncompleteCandle> = None;
        let mut equity_curve = Vec::new();
        let mut position = Position::default();

        for trade in trades {
            let candle_timestamp = self.candle_timestamp(trade.timestamp_ms);

            // Start new candle or update existing
            let candle = candle_builder.get_or_insert_with(|| {
                IncompleteCandle::new(trade, candle_timestamp)
            });

            // Check if candle completed
            if candle.timestamp != candle_timestamp {
                // Previous candle completed
                strategy.on_candle_complete(&candle.clone().complete());
                candle_builder = Some(IncompleteCandle::new(trade, candle_timestamp));
            } else {
                // Update current candle
                candle.update(trade);
            }

            // Call strategy for this tick
            let signal = strategy.on_tick(trade, candle);

            // Execute signal, update position
            self.execute_signal(signal, trade, &mut position);
            equity_curve.push(position.equity);
        }

        // Calculate final metrics
        Ok(self.calculate_metrics(equity_curve, trades))
    }
}
```

### Performance Considerations

**Challenge**: 4.6M trades/day × 365 days = 1.68 BILLION ticks per year

**Solutions**:
1. **Tick Batching**: Process N ticks at once (configurable)
   ```rust
   config.tick_batch_size = 100;  // Call strategy every 100 ticks instead of every tick
   ```

2. **Sampling**: Sample 1 in N ticks (for testing)
   ```rust
   config.tick_sampling = 10;  // Process 1 in 10 ticks (10% sample)
   ```

3. **Time-based Updates**: Update every N milliseconds instead of every tick
   ```rust
   config.tick_interval_ms = 100;  // Update every 100ms instead of every tick
   ```

4. **Parallel Processing**: Split trades by date, process in parallel

**Files to Create**:
- `src/backtest/tick_strategy.rs` (300 lines)
- `src/backtest/tick_engine.rs` (500 lines)
- `src/binance/incomplete_candle.rs` (150 lines)
- `examples/tick_backtest_example.rs` (200 lines)

**Tests**:
- Test tick streaming with 100K trades
- Test candle formation correctness
- Test signal generation at tick level
- Benchmark: 1M ticks vs 10K candles (measure slowdown)

---

## Enhancement 3: Market Microstructure Analysis

**Status**: Not implemented
**Priority**: MEDIUM
**Effort**: 8-12 hours
**Requires**: Tick strategies (Enhancement 2)

### Concept

**Market Microstructure**: Study of how trades happen, order flow, and price formation

**Key Metrics**:
1. **Order Flow Imbalance**: Buy volume - Sell volume
2. **Trade Aggression**: Ratio of buyer-initiated vs seller-initiated
3. **Volume-Weighted Average Price (VWAP)**: Average execution price
4. **Delta**: Cumulative order flow
5. **Footprint Charts**: Volume at price levels

### Binance Trade Data Fields
```csv
trade_id,price,quantity,quote_quantity,timestamp_ms,is_buyer_maker
352562763,28948.19,0.052,1505.30,1609459200001,false
```

**`is_buyer_maker`**:
- `false` → Buyer is taker (aggressive buy, market order)
- `true` → Seller is taker (aggressive sell, market order)

### Implementation

**1. OrderFlow Struct** (`src/microstructure/order_flow.rs`):
```rust
#[derive(Debug, Clone)]
pub struct OrderFlow {
    pub buy_volume: f64,      // Aggressive buy volume
    pub sell_volume: f64,     // Aggressive sell volume
    pub delta: f64,           // buy - sell
    pub total_volume: f64,
    pub buy_ratio: f64,       // buy / total
}

impl OrderFlow {
    pub fn from_trades(trades: &[Trade]) -> Self {
        let buy_volume: f64 = trades
            .iter()
            .filter(|t| !t.is_buyer_maker)  // Taker buy = aggressive
            .map(|t| t.quantity)
            .sum();

        let sell_volume: f64 = trades
            .iter()
            .filter(|t| t.is_buyer_maker)   // Taker sell = aggressive
            .map(|t| t.quantity)
            .sum();

        let total_volume = buy_volume + sell_volume;

        Self {
            buy_volume,
            sell_volume,
            delta: buy_volume - sell_volume,
            total_volume,
            buy_ratio: if total_volume > 0.0 { buy_volume / total_volume } else { 0.5 },
        }
    }
}
```

**2. OrderFlow Strategy Example**:
```rust
struct OrderFlowStrategy {
    delta_threshold: f64,  // 100 BTC imbalance
    lookback_ticks: usize, // Last 1000 ticks
    tick_buffer: VecDeque<Trade>,
}

impl TickStrategy for OrderFlowStrategy {
    fn on_tick(&mut self, trade: &Trade, _candle: &IncompleteCandle) -> Signal {
        self.tick_buffer.push_back(trade.clone());

        if self.tick_buffer.len() < self.lookback_ticks {
            return Signal::Hold;
        }

        // Keep buffer size limited
        while self.tick_buffer.len() > self.lookback_ticks {
            self.tick_buffer.pop_front();
        }

        // Calculate order flow
        let flow = OrderFlow::from_trades(&self.tick_buffer.iter().collect::<Vec<_>>());

        // Trade on significant imbalance
        if flow.delta > self.delta_threshold {
            Signal::Buy  // More aggressive buying
        } else if flow.delta < -self.delta_threshold {
            Signal::Sell  // More aggressive selling
        } else {
            Signal::Hold
        }
    }
}
```

**3. Volume Profile** (`src/microstructure/volume_profile.rs`):
```rust
pub struct VolumeProfile {
    price_levels: HashMap<i64, f64>,  // price_bucket → volume
    tick_size: f64,                    // 0.01 for BTCUSDT
}

impl VolumeProfile {
    pub fn add_trade(&mut self, trade: &Trade) {
        let bucket = (trade.price / self.tick_size).round() as i64;
        *self.price_levels.entry(bucket).or_insert(0.0) += trade.quantity;
    }

    pub fn point_of_control(&self) -> f64 {
        // Price level with most volume
        self.price_levels
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(bucket, _)| *bucket as f64 * self.tick_size)
            .unwrap_or(0.0)
    }

    pub fn value_area(&self, percentage: f64) -> (f64, f64) {
        // Price range containing X% of volume
        // (e.g., 70% value area)
        // Implementation: sort by volume, accumulate until percentage reached
        unimplemented!()
    }
}
```

**Files to Create**:
- `src/microstructure/mod.rs` (50 lines)
- `src/microstructure/order_flow.rs` (200 lines)
- `src/microstructure/volume_profile.rs` (300 lines)
- `examples/order_flow_strategy.rs` (150 lines)

---

## Enhancement 4: GPU-Accelerated Aggregation

**Status**: Not implemented
**Priority**: MEDIUM
**Effort**: 20-30 hours

### Current Performance

**CPU Aggregation** (existing):
- 4.6M trades → 288 candles (5m)
- Time: 1.11s
- Speed: 4.1M trades/sec

**Target GPU Performance**:
- Same data: 0.1-0.2s (5-10x faster)
- Speed: 23-46M trades/sec

### Why GPU Acceleration?

**Current bottlenecks**:
1. HashMap lookups for each trade
2. Sequential processing
3. Memory allocation for builders

**GPU advantages**:
- Parallel sort of 4.6M trades
- Parallel reduction per bucket
- Zero-copy from CSV → GPU memory

### Implementation Approach

**Option 1: cuDF-Based** (Easiest):
```rust
use cudf::DataFrame;

pub fn aggregate_trades_gpu(
    trades: &[Trade],
    timeframe: Timeframe,
) -> Result<Vec<Candle>, GpuError> {
    // Convert to cuDF DataFrame
    let df = trades_to_dataframe(trades)?;

    // Group by candle timestamp
    let grouped = df.groupby(&["candle_timestamp"])?
        .agg(&[
            ("price", "first", "open"),
            ("price", "max", "high"),
            ("price", "min", "low"),
            ("price", "last", "close"),
            ("quantity", "sum", "volume"),
        ])?;

    // Convert back to Vec<Candle>
    dataframe_to_candles(grouped)
}
```

**Option 2: Custom CUDA Kernel** (Fastest):
```cuda
__global__ void aggregate_trades_kernel(
    const Trade* trades,
    int n_trades,
    Candle* candles,
    int64_t timeframe_ms
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_trades) return;

    Trade trade = trades[tid];
    int64_t candle_idx = trade.timestamp_ms / timeframe_ms;

    // Atomic operations to update candle
    atomicAdd(&candles[candle_idx].volume, trade.quantity);
    atomicMax(&candles[candle_idx].high, trade.price);
    atomicMin(&candles[candle_idx].low, trade.price);

    // ... etc
}
```

**Challenges**:
- Need to pre-allocate candle array (know max timestamp)
- Atomic operations on f64 (slower than int)
- Memory transfer overhead

**Files to Create**:
- `src/gpu/trade_aggregation.rs` (400 lines)
- `src/gpu/aggregation_kernels.cu` (300 lines CUDA)
- `benches/gpu_aggregation_benchmark.rs` (200 lines)

---

## Enhancement 5: Data Validation & Quality

**Status**: Not implemented
**Priority**: HIGH
**Effort**: 6-10 hours

### What to Validate

**1. Timestamp Gaps**:
```rust
pub struct GapDetector {
    max_gap_ms: i64,  // e.g., 10 minutes
}

impl GapDetector {
    pub fn find_gaps(&self, trades: &[Trade]) -> Vec<Gap> {
        trades.windows(2)
            .filter_map(|w| {
                let gap_ms = w[1].timestamp_ms - w[0].timestamp_ms;
                if gap_ms > self.max_gap_ms {
                    Some(Gap {
                        start: w[0].timestamp_ms,
                        end: w[1].timestamp_ms,
                        duration_ms: gap_ms,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}
```

**2. Price Outliers**:
```rust
pub struct OutlierDetector {
    std_dev_threshold: f64,  // e.g., 5.0
}

impl OutlierDetector {
    pub fn find_outliers(&self, trades: &[Trade]) -> Vec<Outlier> {
        let prices: Vec<f64> = trades.iter().map(|t| t.price).collect();
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();

        trades.iter()
            .enumerate()
            .filter_map(|(i, trade)| {
                let z_score = (trade.price - mean).abs() / std_dev;
                if z_score > self.std_dev_threshold {
                    Some(Outlier {
                        index: i,
                        trade: trade.clone(),
                        z_score,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}
```

**3. Checksum Verification**:
```rust
pub fn verify_checksums(data_dir: &Path) -> Result<ValidationReport, Error> {
    let mut report = ValidationReport::default();

    for entry in fs::read_dir(data_dir)? {
        let path = entry?.path();
        if path.extension() == Some(OsStr::new("zip")) {
            let checksum_file = path.with_extension("zip.CHECKSUM");
            if checksum_file.exists() {
                let expected = fs::read_to_string(&checksum_file)?;
                let actual = calculate_checksum(&path)?;

                if actual != expected.trim() {
                    report.failures.push(ChecksumFailure {
                        file: path,
                        expected,
                        actual,
                    });
                }
            }
        }
    }

    Ok(report)
}
```

**4. Data Quality Report**:
```rust
pub struct DataQualityReport {
    pub total_trades: usize,
    pub date_range: (i64, i64),
    pub gaps: Vec<Gap>,
    pub outliers: Vec<Outlier>,
    pub duplicate_trade_ids: Vec<u64>,
    pub negative_quantities: Vec<Trade>,
    pub zero_prices: Vec<Trade>,
}

impl DataQualityReport {
    pub fn generate(trades: &[Trade]) -> Self {
        // Run all validators
        let gap_detector = GapDetector::new(600_000); // 10 min
        let outlier_detector = OutlierDetector::new(5.0);

        Self {
            total_trades: trades.len(),
            date_range: (trades.first().unwrap().timestamp_ms,
                        trades.last().unwrap().timestamp_ms),
            gaps: gap_detector.find_gaps(trades),
            outliers: outlier_detector.find_outliers(trades),
            duplicate_trade_ids: find_duplicates(trades),
            negative_quantities: trades.iter()
                .filter(|t| t.quantity < 0.0)
                .cloned()
                .collect(),
            zero_prices: trades.iter()
                .filter(|t| t.price == 0.0)
                .cloned()
                .collect(),
        }
    }

    pub fn print_summary(&self) {
        println!("Data Quality Report");
        println!("==================");
        println!("Total trades: {}", self.total_trades);
        println!("Gaps found: {}", self.gaps.len());
        println!("Outliers: {}", self.outliers.len());
        println!("Duplicates: {}", self.duplicate_trade_ids.len());
        println!("Quality score: {:.1}%", self.quality_score());
    }
}
```

**Files to Create**:
- `src/validation/mod.rs` (100 lines)
- `src/validation/gap_detector.rs` (150 lines)
- `src/validation/outlier_detector.rs` (200 lines)
- `src/validation/checksum.rs` (100 lines)
- `examples/validate_data_quality.rs` (150 lines)

---

## Enhancement 6: Real-Time Streaming

**Status**: Not implemented
**Priority**: LOW (nice-to-have)
**Effort**: 30-40 hours

### Concept

Connect to live Binance websocket, backtest strategies in real-time.

**Use cases**:
- Paper trading with live data
- Strategy validation before deploying
- Live monitoring of backtest performance

### Implementation

**1. Binance Websocket Client**:
```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};

pub struct BinanceTradeStream {
    symbol: String,
    ws_url: String,
}

impl BinanceTradeStream {
    pub async fn connect(&self) -> Result<WebSocket, Error> {
        let url = format!("wss://fstream.binance.com/ws/{}@trade",
                         self.symbol.to_lowercase());
        let (ws_stream, _) = connect_async(url).await?;
        Ok(ws_stream)
    }

    pub async fn stream_trades(&self) -> impl Stream<Item = Trade> {
        let ws = self.connect().await.unwrap();
        ws.filter_map(|msg| {
            if let Ok(Message::Text(text)) = msg {
                serde_json::from_str::<BinanceTrade>(&text)
                    .ok()
                    .map(|bt| bt.into())
            } else {
                None
            }
        })
    }
}
```

**2. Live Backtest Engine**:
```rust
pub struct LiveBacktestEngine {
    strategy: Box<dyn TickStrategy>,
    trade_stream: BinanceTradeStream,
}

impl LiveBacktestEngine {
    pub async fn run(&mut self) -> Result<(), Error> {
        let mut stream = self.trade_stream.stream_trades().await;

        while let Some(trade) = stream.next().await {
            // Process trade
            let signal = self.strategy.on_tick(&trade, &incomplete_candle);

            // Log signal (don't actually trade in backtest mode)
            if signal != Signal::Hold {
                println!("SIGNAL: {:?} at price {}", signal, trade.price);
            }
        }

        Ok(())
    }
}
```

**Dependencies**:
- `tokio` - Async runtime
- `tokio-tungstenite` - Websocket client
- `serde_json` - JSON parsing

**Files to Create**:
- `src/streaming/mod.rs` (100 lines)
- `src/streaming/binance_ws.rs` (300 lines)
- `src/streaming/live_engine.rs` (400 lines)
- `examples/live_backtest.rs` (200 lines)

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
1. ✅ Flexible timeframe system (replace enum)
2. ✅ Multi-file batch processing
3. ✅ Data validation & quality

**Deliverable**: Load entire 2021 year with quality report

### Phase 2: Tick Infrastructure (Week 2)
4. ✅ Tick-level strategy trait
5. ✅ Tick backtest engine
6. ✅ IncompleteCandle implementation

**Deliverable**: Simple tick strategy working on historical data

### Phase 3: Advanced Features (Week 3)
7. ✅ Market microstructure (order flow)
8. ✅ GPU-accelerated aggregation
9. ✅ Performance benchmarks

**Deliverable**: Production-ready tick backtesting

### Phase 4: Real-Time (Optional)
10. ✅ Websocket integration
11. ✅ Live backtest engine

**Deliverable**: Paper trading mode

---

## Testing Strategy

### Unit Tests
- Timeframe parsing
- Date range calculations
- Order flow calculations
- Gap detection
- Outlier detection

### Integration Tests
- Load 2021-01 month
- Aggregate with multiple timeframes
- Tick-by-tick processing
- GPU vs CPU parity

### Performance Tests
- Load 1 year of data
- Tick processing 1M trades
- GPU aggregation benchmark
- Memory usage profiling

---

## Success Criteria

**Must Have**:
- ✅ Flexible timeframes (any duration)
- ✅ Tick-by-tick backtesting
- ✅ Multi-file batch loading
- ✅ Data quality validation

**Should Have**:
- ✅ Order flow analysis
- ✅ GPU aggregation (2x faster minimum)
- ✅ Comprehensive examples

**Nice to Have**:
- ✅ Real-time streaming
- ✅ Volume profile
- ✅ Footprint charts

---

## Files to Create Summary

```
src/
├── binance/
│   ├── timeframe.rs           (NEW - 300 lines)
│   ├── date_utils.rs          (NEW - 150 lines)
│   ├── discovery.rs           (NEW - 200 lines)
│   ├── batch.rs               (NEW - 250 lines)
│   └── incomplete_candle.rs   (NEW - 150 lines)
├── backtest/
│   ├── tick_strategy.rs       (NEW - 300 lines)
│   └── tick_engine.rs         (NEW - 500 lines)
├── microstructure/
│   ├── mod.rs                 (NEW - 50 lines)
│   ├── order_flow.rs          (NEW - 200 lines)
│   └── volume_profile.rs      (NEW - 300 lines)
├── validation/
│   ├── mod.rs                 (NEW - 100 lines)
│   ├── gap_detector.rs        (NEW - 150 lines)
│   ├── outlier_detector.rs    (NEW - 200 lines)
│   └── checksum.rs            (NEW - 100 lines)
├── gpu/
│   ├── trade_aggregation.rs   (NEW - 400 lines)
│   └── aggregation_kernels.cu (NEW - 300 lines CUDA)
└── streaming/
    ├── mod.rs                 (NEW - 100 lines)
    ├── binance_ws.rs          (NEW - 300 lines)
    └── live_engine.rs         (NEW - 400 lines)

examples/
├── tick_backtest_example.rs   (NEW - 200 lines)
├── order_flow_strategy.rs     (NEW - 150 lines)
├── validate_data_quality.rs   (NEW - 150 lines)
└── live_backtest.rs           (NEW - 200 lines)

benches/
└── gpu_aggregation_benchmark.rs (NEW - 200 lines)

Total: ~5,400 new lines of code
```

---

## Next Steps

**Question for User**:

Which enhancements would you like to implement first?

**Recommended order**:
1. Start with **Flexible Timeframes** (2-4 hours, high impact)
2. Then **Multi-file Batch Processing** (4-6 hours, immediate value)
3. Then **Data Validation** (4-6 hours, good to have)
4. Then **Tick-by-Tick Engine** (16-20 hours, core feature)
5. Then **Order Flow** (8-10 hours, builds on tick engine)
6. Later: **GPU Aggregation** (20-30 hours, optimization)
7. Optional: **Real-Time Streaming** (30-40 hours, nice-to-have)

**Or**: Start with specific use case and work backwards?

Let me know your preference!
