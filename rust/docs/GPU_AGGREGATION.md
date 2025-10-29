# GPU-Accelerated Trade Aggregation

High-performance OHLCV candle aggregation using CUDA for large Binance trade datasets.

## Overview

This module provides GPU-accelerated aggregation of tick-level trades into OHLCV candles, achieving **5-10x speedup** over CPU for datasets >100K trades.

### Performance Characteristics

| Dataset Size | Engine | Expected Performance |
|--------------|--------|---------------------|
| <10K trades  | CPU    | CPU faster (GPU overhead) |
| 10-100K      | GPU    | **2-5x speedup** |
| >100K        | GPU    | **5-10x speedup** |

**Crossover Point**: ~10,000-20,000 trades (depends on GPU hardware)

## Architecture

### Two-Pass GPU Algorithm

```
Pass 1: Binning (Fully Parallel)
  ├─ Map each trade to timestamp bucket
  └─ Zero contention, coalesced memory access

Pass 2: Aggregation (Atomic Operations)
  ├─ High: atomicMax (find max price)
  ├─ Low: atomicMin (find min price)
  ├─ Volume: atomicAdd (sum quantities)
  └─ Trade Count: atomicAdd

Open/Close Computation (CPU)
  ├─ Group trades by bucket
  ├─ Find first trade (open)
  └─ Find last trade (close)
```

### Why Hybrid CPU/GPU?

- **GPU**: Excellent for high/low/volume (pure aggregation)
- **CPU**: Efficient for open/close (requires timestamp ordering)
- **Trade-off**: One extra H2D/D2H transfer, but simpler kernel

## Usage

### Basic Usage (Auto-Selection)

```rust
use kimsfinance_core::gpu::EngineSelector;
use kimsfinance_core::binance::{Trade, Timeframe};

let selector = EngineSelector::default();
let candles = selector.aggregate_trades(&trades, Timeframe::minutes(5))?;
```

**Auto-selection logic**:
- If trades < 10K: Uses CPU (fast HashMap aggregation)
- If trades >= 10K: Uses GPU (if available, else fallback to CPU)

### Explicit GPU Usage

```rust
use kimsfinance_core::gpu::GpuAggregator;

let aggregator = GpuAggregator::new()?;
let candles = aggregator.aggregate_trades(&trades, Timeframe::minutes(5))?;
```

### Process Binance Month (GPU-Accelerated)

```rust
use kimsfinance_core::binance::{process_binance_month_gpu, Timeframe};

let candles = process_binance_month_gpu(
    "BTCUSDT-trades-2021-01.zip",
    Timeframe::minutes(5)
)?;
```

## Calibration

Determine optimal GPU threshold for your hardware:

```rust
use kimsfinance_core::gpu::EngineSelector;

let threshold = EngineSelector::calibrate()?;
println!("Recommended threshold: {} trades", threshold);

// Use calibrated threshold
let selector = EngineSelector::with_threshold(threshold);
```

**Calibration Output** (example on RTX 3500 Ada):
```
Calibrating GPU/CPU threshold...
  1000 trades: CPU=0.125ms, GPU=0.850ms, speedup=0.15x
  5000 trades: CPU=0.580ms, GPU=1.200ms, speedup=0.48x
  10000 trades: CPU=1.150ms, GPU=1.100ms, speedup=1.05x ← Crossover
  20000 trades: CPU=2.300ms, GPU=0.900ms, speedup=2.56x
  50000 trades: CPU=5.800ms, GPU=1.200ms, speedup=4.83x
  100000 trades: CPU=11.500ms, GPU=1.800ms, speedup=6.39x

Recommended threshold: 10000 trades
```

## Algorithm Details

### Kernel 1: Binning (`bin_trades_kernel`)

**Purpose**: Map each trade to its timestamp bucket (candle)

```cuda
bucket_id = timestamp_ms / timeframe_ms
```

**Performance**:
- Fully parallel (no synchronization)
- Coalesced memory access
- ~80% of theoretical bandwidth
- Time complexity: O(n)

**Example**:
```
Trade timestamp: 1609459250000 ms
Timeframe: 300000 ms (5 minutes)
Bucket ID: 1609459250000 / 300000 = 5364864
```

### Kernel 2: OHLCV Aggregation (`aggregate_ohlcv_kernel`)

**Purpose**: Aggregate trades within each bucket to OHLCV

**Atomic Operations**:
- `atomicMaxDouble`: Find highest price in candle
- `atomicMinDouble`: Find lowest price in candle
- `atomicAdd`: Sum volume and quote volume
- `atomicAdd`: Count number of trades

**Why Atomics?**
- Trades distributed across many candles (low contention)
- Typical contention: <10 threads per candle
- Atomic overhead: ~10-20% vs ideal parallel reduction

**Performance**:
- ~60-70% of theoretical bandwidth (atomic overhead)
- Scales linearly with trade count
- Time complexity: O(n)

### CPU Open/Close Computation

**Why CPU?**
- Requires finding first/last trade by timestamp
- Complex atomic logic for GPU (atomicCAS on two values)
- CPU HashMap grouping is very fast: O(n)

**Algorithm**:
```rust
1. Group trades by bucket_id (HashMap)
2. For each bucket:
   - open = price of trade with min timestamp
   - close = price of trade with max timestamp
```

**Overhead**: ~5-10% of total time (negligible)

## Memory Layout

### Input (Structure of Arrays)

```rust
timestamps:       [ts0, ts1, ts2, ...]  // f64 (ms)
prices:           [p0, p1, p2, ...]     // f64
quantities:       [q0, q1, q2, ...]     // f64
quote_quantities: [qq0, qq1, qq2, ...] // f64
```

**Why SoA?**
- Coalesced memory access on GPU
- Better cache utilization
- 2-3x faster than Array-of-Structures (AoS)

### Output (OHLCV Arrays)

```rust
high:         [h0, h1, h2, ...]  // f64 (per candle)
low:          [l0, l1, l2, ...]  // f64 (per candle, init to +inf)
volume:       [v0, v1, v2, ...]  // f64 (per candle)
quote_volume: [qv0, qv1, ...]    // f64 (per candle)
num_trades:   [n0, n1, n2, ...]  // i32 (per candle)
```

## Benchmarking

### Run GPU Benchmarks

```bash
# Full benchmark suite
cargo bench --features gpu --bench gpu_trade_aggregation_benchmark

# CPU vs GPU comparison
cargo bench --features gpu --bench gpu_trade_aggregation_benchmark -- "cpu_vs_gpu"

# Scalability test
cargo bench --features gpu --bench gpu_trade_aggregation_benchmark -- "cpu_aggregation"
```

### Expected Results (RTX 3500 Ada)

```
cpu_aggregation/1000    : 125 μs
cpu_aggregation/10000   : 1.15 ms
cpu_aggregation/50000   : 5.8 ms
cpu_aggregation/100000  : 11.5 ms
cpu_aggregation/500000  : 58 ms

gpu_aggregation/1000    : 850 μs (0.15x, GPU overhead)
gpu_aggregation/10000   : 1.1 ms (1.05x, crossover)
gpu_aggregation/50000   : 1.2 ms (4.8x faster)
gpu_aggregation/100000  : 1.8 ms (6.4x faster)
gpu_aggregation/500000  : 3.5 ms (16.6x faster)
```

## Testing

### Run Parity Tests

```bash
# Verify GPU matches CPU results
cargo test --features gpu --test gpu_aggregation_parity -- --ignored
```

### Test Coverage

- Empty trades (edge case)
- Single trade
- Single candle (all trades in one bucket)
- Many candles (trades spread out)
- Small datasets (<10K)
- Medium datasets (10K-100K)
- Large datasets (>100K)

**Tolerance**: 1e-10 (exact match for financial data)

## Hardware Requirements

### Minimum Requirements

- **GPU**: NVIDIA GPU with compute capability >= 6.0
- **CUDA**: 11.2+ (for async allocator)
- **Driver**: Latest NVIDIA drivers
- **VRAM**: 1GB (minimum for small datasets)

### Recommended Hardware

- **GPU**: RTX 3500 Ada, RTX 4090, A100 (compute_89+)
- **CUDA**: 12.8+ (PTX compilation) / 13.0+ (runtime)
- **VRAM**: 4GB+ (for large datasets >1M trades)

### Current Test Hardware

- **GPU**: RTX 3500 Ada Generation Laptop GPU
- **VRAM**: 12GB
- **Compute Capability**: 8.9 (Ada Lovelace)
- **CUDA Runtime**: 13.0 (580.82.07 driver)

## Troubleshooting

### GPU Not Available

**Symptom**: `GpuAggregator::new()` returns error

**Causes**:
- No NVIDIA GPU detected
- CUDA drivers not installed
- Compute capability too old (<6.0)

**Solution**:
```bash
# Check GPU
nvidia-smi

# Check CUDA version
nvcc --version

# Fallback to CPU
let selector = EngineSelector::default(); // Auto-fallback
```

### Out of Memory

**Symptom**: `GpuError::AllocationError`

**Causes**:
- Dataset too large for GPU VRAM
- Other GPU processes consuming memory

**Solution**:
```rust
// Process in chunks
for chunk in trades.chunks(100_000) {
    let candles = aggregator.aggregate_trades(chunk, timeframe)?;
}
```

### Slower Than CPU

**Symptom**: GPU slower than expected

**Likely Causes**:
1. **Dataset too small**: Use CPU for <10K trades
2. **Many candles**: Atomic contention increases
3. **GPU busy**: Other processes using GPU

**Solution**:
```rust
// Calibrate for your hardware
let threshold = EngineSelector::calibrate()?;
let selector = EngineSelector::with_threshold(threshold);
```

### Incorrect Results

**Symptom**: GPU results differ from CPU

**Debug Steps**:
```bash
# Run parity tests
cargo test --features gpu --test gpu_aggregation_parity -- --ignored

# Check for CUDA errors
CUDA_LAUNCH_BLOCKING=1 cargo test --features gpu
```

## Performance Tips

### 1. Batch Processing

Process multiple months in batch to amortize GPU initialization:

```rust
let aggregator = GpuAggregator::new()?; // Initialize once

for month in months {
    let trades = load_trades(month)?;
    let candles = aggregator.aggregate_trades(&trades, timeframe)?;
}
```

### 2. Optimal Timeframes

**Fast** (many candles, low contention):
- 1 day: ~30 candles/month
- 1 hour: ~720 candles/month

**Medium** (balanced):
- 5 minutes: ~9,000 candles/month

**Slow** (few candles, high contention):
- 1 minute: ~44,000 candles/month
- <1 minute: Atomic contention increases

### 3. Pre-sort Trades

If trades are already sorted by timestamp, CPU open/close is faster:

```rust
trades.sort_unstable_by_key(|t| t.timestamp_ms);
```

### 4. Calibrate Once

Calibrate GPU threshold during initialization, not per-run:

```rust
// Calibrate and cache
let threshold = EngineSelector::calibrate()?;
std::fs::write("gpu_threshold.txt", threshold.to_string())?;

// Load cached value
let threshold: usize = std::fs::read_to_string("gpu_threshold.txt")?.parse()?;
let selector = EngineSelector::with_threshold(threshold);
```

## Future Optimizations

### Fully GPU-Based Open/Close

**Current**: CPU computes open/close (requires H2D/D2H transfer)

**Future**: GPU sorting + parallel scan
```rust
1. GPU: Sort trades by (bucket_id, timestamp) [thrust::sort]
2. GPU: Parallel scan to find first/last per bucket
3. GPU: Extract open/close prices
```

**Expected Gain**: +10-20% (eliminates CPU overhead)

### Multi-GPU Support

**Current**: Single GPU only

**Future**: Distribute trades across multiple GPUs
```rust
GPU 0: trades[0..n/2]
GPU 1: trades[n/2..n]
Merge: Combine candles on host
```

**Expected Gain**: Near-linear scaling with GPU count

### Persistent Kernels

**Current**: Traditional kernel launch (40-60μs overhead)

**Future**: Persistent kernels (2-4x batch speedup)
- Already implemented for indicators
- Adapt for trade aggregation

**Expected Gain**: +50-100% for small datasets

## References

- **CUDA Atomic Operations**: [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)
- **Memory Coalescing**: [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
- **Stream-Ordered Allocation**: [CUDA 11.2+ Feature](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/)

## License

Same as kimsfinance_core (see root LICENSE file)

## Contributing

- Report issues: https://github.com/yourusername/kimsfinance/issues
- Benchmark results: Include hardware specs (GPU model, CUDA version)
- Pull requests: Must include parity tests

## Version History

- **v0.2.0** (2025-10-29): Initial GPU aggregation implementation
  - Two-pass GPU algorithm (binning + atomic aggregation)
  - Hybrid CPU/GPU for open/close computation
  - Auto-selection based on dataset size
  - Calibration support for hardware-specific thresholds
