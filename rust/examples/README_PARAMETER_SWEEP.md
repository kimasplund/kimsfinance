# Parameter Sweep Optimizer for Bull Put Spread Strategy

## Overview

The parameter sweep optimizer performs a comprehensive search across multiple parameter dimensions to find the optimal configuration for the bull put spread strategy. It tests hundreds of combinations in parallel using Rayon for maximum CPU utilization.

## Features

- **Parallel Processing**: Uses all CPU cores via Rayon for fast execution
- **Comprehensive Search**: Tests 4×4×4×3×3 = 576 parameter combinations
- **Fitness Scoring**: Composite metric balancing returns, risk, and consistency
- **Top 10 Tracking**: Identifies and ranks the best parameter sets
- **JSON Output**: Saves all results for further analysis

## Parameter Ranges

The optimizer tests the following parameter ranges:

### DTE (Days to Expiration)
- 21-35 days
- 30-45 days
- 35-50 days
- 45-60 days

### Delta Range
- 0.10-0.25
- 0.15-0.30
- 0.15-0.35
- 0.20-0.40

### Profit Targets
- 40%
- 50%
- 60%
- 75%

### Stop Losses
- 150%
- 200%
- 250%

### Max Hold Days
- 21 days
- 35 days
- 42 days

## Fitness Function

The optimizer uses a composite fitness score to rank parameter combinations:

```
Fitness = (ROC × 0.4) + (Sharpe × 20) + (Win Rate × 0.3) - (Max DD / 100)
```

Where:
- **ROC**: Return on Capital (40% weight) - emphasizes profitability
- **Sharpe**: Sharpe Ratio × 20 (scaled to ~20 points) - risk-adjusted returns
- **Win Rate**: Percentage of winning trades (30% weight) - consistency
- **Max DD**: Maximum Drawdown / 100 (penalty) - avoids risky strategies

## Usage

### Prerequisites

1. Download historical options data:
```bash
cargo run --release --features data-downloaders --example download_options_strategy
```

This creates the data directory structure:
```
data/yfinance/options_historical/
├── AAPL/
│   ├── 2020-01-02.parquet
│   ├── 2020-01-03.parquet
│   └── ...
```

### Run the Optimizer

```bash
cargo run --release --features data-downloaders --example parameter_sweep_optimizer
```

### Expected Output

```
=== Bull Put Spread Parameter Sweep Optimizer ===

Configuration:
  Symbol: AAPL
  Initial Capital: $10000.00
  Period: 2020-01-01 to 2023-12-31
  Output: results/parameter_sweep_results.json

Loading historical options data...
  AAPL has 950 days of historical data

Parameter Sweep Configuration:
  Total combinations: 576
  Parallel workers: 32 (Rayon)

Starting parameter sweep...

Progress: 10/576 (1.7%) - 2.5 tests/sec - ETA: 226s
Progress: 20/576 (3.5%) - 2.8 tests/sec - ETA: 199s
...

=== Parameter Sweep Complete ===
Total time: 205.32s
Tests per second: 2.81

=== Top 10 Parameter Combinations ===

#1 - Fitness: 45.23
  Parameters:
    DTE Range: 30 to 45
    Delta Range: 0.15 to 0.30
    Profit Target: 50%
    Stop Loss: 200%
    Max Hold Days: 35
  Results:
    Trades: 87
    Total P&L: $5432.10
    Win Rate: 78.2%
    Sharpe Ratio: 1.85
    Max Drawdown: $-423.50
    ROC: 54.32%
    Profit Factor: 2.31
...

=== Results Saved ===
Output file: results/parameter_sweep_results.json
  - Top 10 parameter sets
  - All 576 results
```

## Output Format

The optimizer saves results to `results/parameter_sweep_results.json`:

```json
{
  "metadata": {
    "symbol": "AAPL",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 10000.0,
    "total_combinations": 576,
    "execution_time_seconds": 205.32,
    "timestamp": "2025-10-30T14:30:00Z"
  },
  "top_10": [
    {
      "params": {
        "name": "DTE30-45_D0.15-0.30_PT50_SL200_MH35",
        "dte_min": 30,
        "dte_max": 45,
        "delta_min": 0.15,
        "delta_max": 0.30,
        "profit_target_pct": 50.0,
        "stop_loss_pct": 200.0,
        "max_hold_days": 35,
        "position_size_pct": 10.0,
        "min_credit": 0.30
      },
      "num_trades": 87,
      "total_pnl": 5432.10,
      "win_rate": 78.2,
      "sharpe_ratio": 1.85,
      "max_drawdown": -423.50,
      "return_on_capital": 54.32,
      "profit_factor": 2.31,
      "fitness": 45.23
    },
    ...
  ],
  "all_results": [...]
}
```

## Performance

### Execution Time

- **Single-threaded**: ~1,600 seconds (27 minutes) for 576 combinations
- **Multi-threaded (32 cores)**: ~200 seconds (3.3 minutes)
- **Speedup**: ~8x with 32 cores

### Memory Usage

- Peak RAM: ~2-4 GB (depends on data cache)
- Each thread creates its own data loader with cache

## Customization

### Change Parameter Ranges

Edit the `generate_parameter_combinations()` function:

```rust
let dte_ranges = vec![
    (21, 35),
    (30, 45),
    // Add more ranges...
];
```

### Modify Fitness Function

Edit the `calculate_fitness()` function:

```rust
fn calculate_fitness(result: &BacktestResult) -> f64 {
    // Custom fitness formula
    (result.return_on_capital * 0.5) + (result.sharpe_ratio * 25.0)
}
```

### Change Symbol or Date Range

Edit the `main()` function:

```rust
let symbol = "SPY";  // Different symbol
let start_date = NaiveDate::from_ymd_opt(2021, 1, 1).expect("Invalid start date");
let end_date = NaiveDate::from_ymd_opt(2024, 12, 31).expect("Invalid end date");
```

## Next Steps

After running the optimizer:

1. **Review top 10 results**: Identify common patterns in high-performing parameters
2. **Further validation**: Run walk-forward analysis on top parameters
3. **Out-of-sample testing**: Test top parameters on 2024+ data
4. **Robustness check**: Test across multiple symbols (SPY, QQQ, etc.)
5. **Live trading**: Use validated parameters for paper trading

## Troubleshooting

### Data Not Found

```
Error: No data available for AAPL
```

**Solution**: Download historical data first:
```bash
cargo run --release --features data-downloaders --example download_options_strategy
```

### Slow Performance

If progress is slower than expected:

1. Check CPU usage: `htop` or `top`
2. Verify Rayon is using all cores: Look for "Parallel workers: N"
3. Reduce combinations by limiting parameter ranges

### Out of Memory

If you run out of memory:

1. Reduce date range (test fewer years)
2. Clear cache periodically (modify data loader)
3. Process in batches instead of all at once

## Algorithm Details

### Thread Safety

Each thread creates its own `OptionsDataLoader` instance with independent cache. This avoids contention and allows true parallel execution.

### Progress Tracking

Progress is updated every 10 completions using a `Mutex`-protected counter. This minimizes lock contention while providing regular updates.

### Result Aggregation

Results are collected in a `Mutex`-protected vector. After all threads complete, results are sorted by fitness score in descending order.

## References

- Bull Put Spread Strategy: `examples/backtest_bull_put_spread.rs`
- Strategy Types: `src/strategy/types.rs`
- Backtest Engine: `src/strategy/backtest.rs`
- Data Loader: `src/strategy/data_loader.rs`

## License

Same as kimsfinance_core project.
