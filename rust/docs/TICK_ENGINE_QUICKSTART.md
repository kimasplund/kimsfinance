# Tick Engine Quick Start Guide

## Installation

The tick engine is part of `kimsfinance_core` crate (already installed).

```toml
[dependencies]
kimsfinance_core = { path = "../rust" }
```

## Basic Usage (3 Steps)

### 1. Create Engine

```rust
use kimsfinance_core::backtest::{TickEngine, BacktestConfig};

let config = BacktestConfig {
    initial_capital: 10_000.0,
    trading_fee: 0.001,  // 0.1%
    slippage: 0.0005,    // 0.05%
    ..Default::default()
};

let engine = TickEngine::new(config);
```

### 2. Create Strategy

```rust
use kimsfinance_core::backtest::IntraCandleMomentum;

// Use built-in strategy
let mut strategy = IntraCandleMomentum::new(0.5);  // 0.5% threshold
```

Or create custom strategy:

```rust
use kimsfinance_core::backtest::{TickStrategy, Signal};
use kimsfinance_core::binance::{Trade, IncompleteCandle, Candle};

struct MyStrategy {
    threshold: f64,
}

impl TickStrategy for MyStrategy {
    fn on_tick(&mut self, trade: &Trade, candle: &IncompleteCandle) -> Signal {
        let change = (trade.price - candle.open) / candle.open * 100.0;
        
        if change > self.threshold {
            Signal::Buy
        } else if change < -self.threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }
    
    fn name(&self) -> &str {
        "MyStrategy"
    }
}
```

### 3. Run Backtest

```rust
use kimsfinance_core::binance::Timeframe;

let timeframe = Timeframe::parse("5m")?;
let result = engine.run(&mut strategy, &trades, timeframe)?;

println!("Total Return: {:.2}%", result.total_return);
println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
println!("Max Drawdown: {:.2}%", result.max_drawdown);
println!("Win Rate: {:.2}%", result.win_rate);
println!("Profit Factor: {:.2}", result.profit_factor);
println!("Trades: {}", result.num_trades);
```

## Performance

- **64M trades/sec** throughput
- **72ms** for daily backtest (4.6M trades)
- **2.2 seconds** for monthly backtest (138M trades)

## Built-in Strategies

### 1. IntraCandleMomentum

Trades on price momentum within candle:

```rust
let mut strategy = IntraCandleMomentum::new(0.5);  // 0.5% threshold
```

### 2. VolumeSpikeStrategy

Trades on volume spikes:

```rust
use kimsfinance_core::backtest::VolumeSpikeStrategy;

let mut strategy = VolumeSpikeStrategy::new(3.0);  // 3x average volume
```

### 3. OrderFlowStrategy

Trades on aggressive buy/sell flow:

```rust
use kimsfinance_core::backtest::OrderFlowStrategy;

let mut strategy = OrderFlowStrategy::new(5.0);  // 5 BTC imbalance
```

## Loading Trade Data

### From Binance CSV

```rust
use std::fs::File;
use std::io::{BufRead, BufReader};
use kimsfinance_core::binance::Trade;

fn load_trades(path: &str) -> Result<Vec<Trade>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let mut trades = Vec::new();
    
    for line in reader.lines().skip(1) {  // Skip header
        let line = line?;
        let fields: Vec<&str> = line.split(',').collect();
        
        trades.push(Trade {
            trade_id: fields[0].parse()?,
            price: fields[1].parse()?,
            quantity: fields[2].parse()?,
            quote_quantity: fields[3].parse()?,
            timestamp_ms: fields[4].parse()?,
            is_buyer_maker: fields[5] == "true",
        });
    }
    
    Ok(trades)
}

// Usage
let trades = load_trades("BTCUSDT-trades-2024-01.csv")?;
```

## Results Structure

```rust
pub struct BacktestResult {
    pub equity_curve: Vec<f64>,      // Sampled equity values
    pub final_equity: f64,            // Final equity
    pub total_return: f64,            // Total return (%)
    pub sharpe_ratio: f64,            // Annualized Sharpe ratio
    pub max_drawdown: f64,            // Maximum drawdown (%)
    pub win_rate: f64,                // Win rate (%)
    pub profit_factor: f64,           // Gross profit / gross loss
    pub num_trades: usize,            // Total trades executed
    pub trades: Vec<Trade>,           // All trades
}
```

## Examples

### Run Built-in Example

```bash
cargo run --example tick_backtest_btc --release
```

### Run Benchmark

```bash
cargo run --example tick_benchmark --release
```

## Testing

```bash
# Run tick engine tests
cargo test --lib tick_engine

# Run all tick tests (engine + strategy)
cargo test --lib tick

# Run with output
cargo test --lib tick -- --nocapture
```

## Timeframes

Supported candle timeframes:

- `"1m"` - 1 minute
- `"5m"` - 5 minutes
- `"15m"` - 15 minutes
- `"1h"` - 1 hour
- `"4h"` - 4 hours
- `"1d"` - 1 day

```rust
let timeframe = Timeframe::parse("5m")?;
```

## Configuration Options

```rust
pub struct BacktestConfig {
    pub initial_capital: f64,  // Starting capital
    pub trading_fee: f64,      // Fee per trade (0.001 = 0.1%)
    pub slippage: f64,         // Slippage (0.0005 = 0.05%)
    pub use_gpu: bool,         // Enable GPU (N/A for tick engine)
    pub force_cpu: bool,       // Force CPU mode
}
```

## Common Patterns

### Parameter Sweep

```rust
let thresholds = vec![0.1, 0.25, 0.5, 0.75, 1.0];

for threshold in thresholds {
    let mut strategy = IntraCandleMomentum::new(threshold);
    let result = engine.run(&mut strategy, &trades, timeframe)?;
    
    println!("Threshold: {:.2}% → Return: {:.2}%", 
             threshold, result.total_return);
}
```

### Multi-Timeframe

```rust
let timeframes = vec!["1m", "5m", "15m", "1h"];

for tf_str in timeframes {
    let timeframe = Timeframe::parse(tf_str)?;
    let result = engine.run(&mut strategy, &trades, timeframe)?;
    
    println!("{} → Sharpe: {:.2}", tf_str, result.sharpe_ratio);
}
```

## Troubleshooting

### Slow Performance

```rust
// Check dataset size
println!("Processing {} trades", trades.len());

// Use release mode
cargo run --release
```

### No Trades Executed

```rust
// Lower threshold
let mut strategy = IntraCandleMomentum::new(0.1);  // Lower from 0.5

// Check candle formation
println!("Timeframe: {:?}", timeframe);
```

### Memory Issues

```rust
// Process in chunks
for chunk in trades.chunks(1_000_000) {
    let result = engine.run(&mut strategy, chunk, timeframe)?;
    // ... aggregate results
}
```

## Documentation

- Full documentation: `docs/TICK_BACKTEST_ENGINE.md`
- API docs: `cargo doc --open`
- Examples: `examples/tick_*.rs`

## Support

For issues or questions:
1. Check `docs/TICK_BACKTEST_ENGINE.md`
2. Run examples to verify installation
3. Check test suite: `cargo test --lib tick`
