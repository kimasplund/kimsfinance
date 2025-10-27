# GPU-Accelerated Custom Candle Generation Design

## Overview

Extend the persistent kernel architecture to support:
1. **Trade-to-Candle aggregation** from raw CSV trade data
2. **Multiple candle types** (time, volume, tick, range, Heikin-Ashi, Renko)
3. **Batch processing** for multiple symbols simultaneously

## Architecture

### 1. Trade Data Ingestion (CSV → GPU)

**Input Format:**
```csv
timestamp,symbol,price,volume,side
1234567890,BTC,50000.0,0.5,buy
1234567891,BTC,50001.0,0.3,buy
```

**GPU Buffer Layout:**
```rust
struct TradeData {
    timestamps: Vec<i64>,    // Unix timestamps
    prices: Vec<f64>,        // Trade prices
    volumes: Vec<f64>,       // Trade volumes
    sides: Vec<i8>,          // 1=buy, -1=sell, 0=unknown
}
```

### 2. Candle Types Implementation

#### A. Time-Based Candles (Traditional OHLCV)

**Kernel:** `persistent_time_candles_kernel`

```cuda
// Aggregate trades into fixed time intervals (1m, 5m, 1h, etc.)
// Input: [timestamps(n), prices(n), volumes(n)]
// Output: [open(m), high(m), low(m), close(m), volume(m)]
```

**Algorithm:**
1. Sort trades by timestamp (if needed)
2. Group by time buckets (e.g., 60s intervals)
3. For each bucket:
   - Open = first trade price
   - High = max trade price
   - Low = min trade price
   - Close = last trade price
   - Volume = sum of trade volumes

**Performance:** Ideal for GPU - parallel bucket aggregation

---

#### B. Volume Bars

**Kernel:** `persistent_volume_bars_kernel`

```cuda
// Create bars with fixed volume per bar
// Input: [timestamps(n), prices(n), volumes(n)]
// Output: [open(m), high(m), low(m), close(m), volume(m), start_time(m), end_time(m)]
```

**Algorithm:**
1. Accumulate volume until reaching threshold (e.g., 100 BTC)
2. When threshold reached → close bar and start new one
3. Track OHLC within each volume bucket

**GPU Strategy:**
- Sequential per-symbol (volume accumulation has dependencies)
- Parallel across multiple symbols

---

#### C. Tick Bars (Fixed Number of Trades)

**Kernel:** `persistent_tick_bars_kernel`

```cuda
// Create bars from fixed number of trades
// Input: [timestamps(n), prices(n), volumes(n)]
// Output: [open(m), high(m), low(m), close(m), volume(m), num_trades(m)]
```

**Algorithm:**
1. Group every N trades (e.g., 100 trades per bar)
2. Calculate OHLC for each group

**GPU Strategy:** Highly parallel - can process multiple groups simultaneously

---

#### D. Range Bars

**Kernel:** `persistent_range_bars_kernel`

```cuda
// Create bars with fixed price range
// Input: [timestamps(n), prices(n), volumes(n)]
// Output: [open(m), high(m), low(m), close(m), volume(m)]
```

**Algorithm:**
1. Start bar at current price
2. Close bar when price moves fixed range (e.g., $100)
3. New bar starts at close of previous bar

**GPU Strategy:** Sequential within symbol, parallel across symbols

---

#### E. Heikin-Ashi Candles

**Kernel:** `persistent_heikin_ashi_kernel`

```cuda
// Smoothed candles using averages
// Input: [open(n), high(n), low(n), close(n)]
// Output: [ha_open(n), ha_high(n), ha_low(n), ha_close(n)]
```

**Algorithm:**
```
HA-Close = (Open + High + Low + Close) / 4
HA-Open = (Previous HA-Open + Previous HA-Close) / 2
HA-High = max(High, HA-Open, HA-Close)
HA-Low = min(Low, HA-Open, HA-Close)
```

**GPU Strategy:**
- Sequential (depends on previous bar)
- Can parallelize across multiple symbols
- Perfect fit for persistent kernel pattern!

---

#### F. Renko Bricks

**Kernel:** `persistent_renko_kernel`

```cuda
// Price movement based bricks (fixed size)
// Input: [timestamps(n), prices(n)]
// Output: [brick_price(m), direction(m), time(m)]
```

**Algorithm:**
1. Set brick size (e.g., $100)
2. New brick only forms when price moves full brick size
3. Direction: up (+1) or down (-1)

**GPU Strategy:** Sequential per symbol, parallel across symbols

---

## 3. Persistent Kernel Implementation

### Trait System

```rust
pub trait CandleAggregator: PersistentIndicator {
    type InputData;
    type OutputCandle;

    /// Number of input arrays (timestamp, price, volume, etc.)
    fn num_input_arrays() -> usize;

    /// Expected output fields per candle
    fn output_fields() -> usize;

    /// Kernel source code
    fn aggregation_kernel() -> &'static str;
}
```

### Example: Time Bars Implementation

```rust
pub struct TimeBarAggregator;

#[repr(C)]
pub struct TimeBarParams {
    pub interval_seconds: i32,  // 60 for 1m, 300 for 5m, etc.
}

impl CandleAggregator for TimeBarAggregator {
    type InputData = TradeData;
    type OutputCandle = OHLCVCandle;

    fn num_input_arrays() -> usize { 3 } // timestamp, price, volume
    fn output_fields() -> usize { 5 }    // O, H, L, C, V

    fn aggregation_kernel() -> &'static str {
        TIME_BAR_KERNEL
    }
}

const TIME_BAR_KERNEL: &str = r#"
extern "C" __global__ void persistent_time_bars_kernel(
    const double** __restrict__ input_batch,     // [timestamps, prices, volumes]
    double** __restrict__ output_batch,          // [O, H, L, C, V]
    const int* __restrict__ sizes,               // Number of trades per task
    const TimeBarParams* __restrict__ params,
    int num_tasks
) {
    cg::grid_group grid = cg::this_grid();
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* input = input_batch[task_id];
        int n = sizes[task_id];
        int interval = params[task_id].interval_seconds;

        // Split input buffer
        const double* timestamps = input;          // First n elements
        const double* prices = input + n;          // Next n elements
        const double* volumes = input + 2*n;       // Last n elements

        if (global_tid == task_id % grid_size) {
            // Aggregate trades into time buckets
            long long current_bucket = (long long)(timestamps[0]) / interval;
            double bucket_open = prices[0];
            double bucket_high = prices[0];
            double bucket_low = prices[0];
            double bucket_volume = 0.0;
            int candle_idx = 0;

            for (int i = 0; i < n; i++) {
                long long trade_bucket = (long long)(timestamps[i]) / interval;

                if (trade_bucket > current_bucket) {
                    // Close current candle
                    double* output = output_batch[task_id];
                    output[candle_idx * 5 + 0] = bucket_open;
                    output[candle_idx * 5 + 1] = bucket_high;
                    output[candle_idx * 5 + 2] = bucket_low;
                    output[candle_idx * 5 + 3] = prices[i-1];  // Last price is close
                    output[candle_idx * 5 + 4] = bucket_volume;
                    candle_idx++;

                    // Start new candle
                    current_bucket = trade_bucket;
                    bucket_open = prices[i];
                    bucket_high = prices[i];
                    bucket_low = prices[i];
                    bucket_volume = 0.0;
                }

                // Update current candle
                bucket_high = fmax(bucket_high, prices[i]);
                bucket_low = fmin(bucket_low, prices[i]);
                bucket_volume += volumes[i];
            }

            // Close final candle
            double* output = output_batch[task_id];
            output[candle_idx * 5 + 0] = bucket_open;
            output[candle_idx * 5 + 1] = bucket_high;
            output[candle_idx * 5 + 2] = bucket_low;
            output[candle_idx * 5 + 3] = prices[n-1];
            output[candle_idx * 5 + 4] = bucket_volume;
        }

        grid.sync();
    }
}
"#;
```

## 4. Usage Examples

### Example 1: Load Trades CSV and Create 1-Minute Candles

```rust
use kimsfinance_core::gpu::candles::*;

// Load trades from CSV
let trades = TradeData::from_csv("btc_trades.csv")?;

// Create time bar batch (1-minute candles)
let mut batch = TimeBarBatch::new();
batch.add_task(
    trades.concat_buffers(), // [timestamps, prices, volumes]
    TimeBarParams { interval_seconds: 60 }
);

// Execute on GPU with persistent kernel
let device = GpuDevice::new()?;
let candles = execute_batch(&device, &batch)?;

// Result: OHLCV candles aggregated by minute
```

### Example 2: Batch Process Multiple Symbols

```rust
// Load multiple CSV files
let btc_trades = TradeData::from_csv("btc_trades.csv")?;
let eth_trades = TradeData::from_csv("eth_trades.csv")?;
let sol_trades = TradeData::from_csv("sol_trades.csv")?;

// Create batch with all symbols (single GPU launch!)
let mut batch = TimeBarBatch::new();
batch.add_task(btc_trades.concat_buffers(), TimeBarParams { interval_seconds: 300 }); // 5m
batch.add_task(eth_trades.concat_buffers(), TimeBarParams { interval_seconds: 300 });
batch.add_task(sol_trades.concat_buffers(), TimeBarParams { interval_seconds: 300 });

// Process all 3 symbols in single persistent kernel launch
let candles = execute_batch(&device, &batch)?;
// candles[0] = BTC 5m candles
// candles[1] = ETH 5m candles
// candles[2] = SOL 5m candles
```

### Example 3: Heikin-Ashi from Existing OHLCV

```rust
// Already have OHLCV candles, convert to Heikin-Ashi
let mut batch = HeikinAshiBatch::new();
let ohlcv_concat = concat_ohlcv(&open, &high, &low, &close);
batch.add_task(ohlcv_concat, ());

let ha_candles = execute_batch(&device, &batch)?;
// Result: Smoothed Heikin-Ashi candles for trend following
```

## 5. Performance Benefits

**Persistent Kernel Advantages:**
1. **Single GPU Launch** - Process 10+ symbols → 10μs vs 100μs (90% reduction)
2. **Pinned Memory** - 20-30% faster CPU↔GPU transfers for large trade datasets
3. **Occupancy Optimization** - Dynamic block sizing per kernel type
4. **Batch Efficiency** - Process entire portfolio of symbols simultaneously

**Expected Speedups:**
- Time bars: **50-100x** vs CPU (highly parallel groupby operations)
- Volume/Tick bars: **20-50x** vs CPU (sequential per-symbol, parallel across symbols)
- Heikin-Ashi: **30-70x** vs CPU (simple transformations, very parallel)
- Renko/Range: **10-30x** vs CPU (more sequential logic)

## 6. Implementation Priority

**Phase 1: Foundation** (Essential)
- ✅ Time bars (most common use case)
- ✅ Heikin-Ashi (trend following)
- ✅ CSV ingestion pipeline

**Phase 2: Advanced** (High Value)
- ⏳ Volume bars (order flow analysis)
- ⏳ Tick bars (microstructure)
- ⏳ Range bars (volatility-adjusted)

**Phase 3: Specialized** (Niche)
- ⏳ Renko (price action traders)
- ⏳ Kagi charts
- ⏳ Point & Figure

## 7. Integration with Existing Code

```rust
// Candle aggregation fits naturally into persistent kernel architecture
pub mod candles {
    pub use time_bars::{TimeBarBatch, TimeBarAggregator, TimeBarParams};
    pub use volume_bars::{VolumeBarBatch, VolumeBarParams};
    pub use heikin_ashi::{HeikinAshiBatch, HeikinAshiAggregator};
    // ... other candle types
}

// Same execution pattern as indicators
pub use candles::execute_batch;  // Reuses existing infrastructure!
```

## 8. Next Steps

Would you like me to:
1. **Implement Time Bar aggregation** first (trades CSV → 1m/5m/1h candles)?
2. **Implement Heikin-Ashi** transformation (OHLC → smoothed candles)?
3. **Design the CSV ingestion pipeline** for trade data?
4. **Create a full example** with real trade data?

The persistent kernel architecture we just completed provides the perfect foundation for this!
