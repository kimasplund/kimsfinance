# GPU Batch Indicator Benchmark

Comprehensive end-to-end benchmark for GPU batch indicator system using realistic OHLCV data.

## Overview

This benchmark validates the performance of the GPU batch indicator system by measuring:

- **Data generation time** (simulates CSV parse + OHLCV aggregation)
- **GPU memory transfer time** (implicit in batch operations)
- **Individual indicator computation time**
- **Total end-to-end time**
- **Throughput** (candles/sec, indicators/sec)
- **Speedup ratios** (batch vs individual calls)

## Scenarios

### 1. Single Indicator (Baseline)

Measures baseline GPU performance for single RSI calculation.

```bash
cargo bench --features gpu --bench binance_gpu_benchmark gpu_single_indicator
```

**Expected**: ~50ms for 44,640 candles (RTX 3500 Ada)

### 2. Batch Indicators (Memory Pooling Benefit)

Calculates all 9 indicators in batch vs sequential individual calls.

```bash
cargo bench --features gpu --bench binance_gpu_benchmark gpu_batch_indicators
```

**Expected**: 4-6x speedup over sequential execution

### 3. Multiple Timeframes

Benchmarks different dataset sizes representing 1m, 5m, 1h, 1d aggregations.

```bash
cargo bench --features gpu --bench binance_gpu_benchmark gpu_timeframes
```

**Datasets**:
- 1m: 44,640 candles (1 month)
- 5m: 8,928 candles (1 month)
- 1h: 744 candles (1 month)
- 1d: 31 candles (1 month)

### 4. Scalability

Tests performance across different time ranges.

```bash
cargo bench --features gpu --bench binance_gpu_benchmark gpu_scalability
```

**Datasets**:
- 1 day: 1,440 candles
- 1 week: 10,080 candles
- 1 month: 44,640 candles
- 3 months: 133,920 candles

## Running All Benchmarks

```bash
# Full benchmark suite (takes ~10-15 minutes)
cargo bench --features gpu --bench binance_gpu_benchmark

# Quick test with reduced samples
cargo bench --features gpu --bench binance_gpu_benchmark -- --sample-size 10

# Performance analysis test (detailed metrics)
cargo test --release --features gpu --bench binance_gpu_benchmark test_batch_performance_analysis -- --ignored --nocapture
```

## Sample Output

```
=== Performance Analysis ===
Dataset: 44640 candles (1 month)

Batch Indicators (9 indicators):
  Batch time:       245.3 ms
  Sequential time:  1,479.2 ms
  Speedup:          6.03x
  Throughput:       182,041 candles/sec
  Indicators/sec:   1,638,372
  Memory transfers: 88.9% reduction

Performance Validation:
  Expected speedup: 4-6x
  Actual speedup:   6.03x ✓
```

## Data Generation

The benchmark generates realistic Bitcoin price data with:

- **Base price**: ~$29,000 (Jan 2021)
- **Trend**: 10% monthly growth
- **Volatility**: 2-4% daily range
- **Patterns**: 4-hour oscillation cycles
- **Continuity**: No gaps or jumps

This simulates real Binance BTCUSDT futures data aggregated to 1-minute candles.

## Hardware Context

Benchmarks are calibrated for:

- **GPU**: NVIDIA RTX 3500 Ada (12GB VRAM)
- **CPU**: Intel i9-13980HX (24 cores)
- **RAM**: 64GB DDR5
- **CUDA**: 12.8.0+

Performance on different hardware will vary.

## Validation

Each benchmark run validates:

- ✅ Array lengths match input
- ✅ No NaN values outside warmup period (first 50 candles)
- ✅ Values within indicator bounds
- ✅ Minimum 2x speedup for batch operations

## Troubleshooting

### GPU Not Found

```
GPU not available: InitializationError(...). Skipping GPU benchmarks.
```

**Fix**: Ensure NVIDIA drivers and CUDA 12.8+ are installed.

### Compilation Errors

```bash
# Ensure GPU feature is enabled
cargo bench --features gpu --bench binance_gpu_benchmark

# Clean build if issues persist
cargo clean && cargo bench --features gpu --bench binance_gpu_benchmark
```

### Slow Performance

If speedup < 2x:

1. Check GPU utilization: `nvidia-smi dmon`
2. Verify no CPU throttling: `lscpu | grep MHz`
3. Close GPU-intensive applications
4. Use release mode: `cargo bench --release`

## Integration with CI/CD

```yaml
# GitHub Actions example
- name: GPU Benchmark
  run: |
    cargo bench --features gpu --bench binance_gpu_benchmark -- --sample-size 10
  if: runner.gpu == 'nvidia'
```

## Performance Regression Detection

To detect regressions, compare against baseline:

```bash
# Save baseline
cargo bench --features gpu --bench binance_gpu_benchmark -- --save-baseline main

# Compare after changes
cargo bench --features gpu --bench binance_gpu_benchmark -- --baseline main
```

## Related Benchmarks

- `momentum_indicators.rs` - CPU indicator benchmarks
- `moving_averages.rs` - CPU moving average benchmarks
- `volatility_indicators.rs` - CPU volatility benchmarks

## References

- GPU Batch System: `src/gpu/batch.rs`
- Memory Pool: `src/gpu/memory_pool.rs`
- Stream Manager: `src/gpu/streams.rs`
- Individual GPU Indicators: `src/gpu/*.rs`

---

**Last Updated**: 2025-10-25
**GPU Architecture**: CUDA 12.8.0 via cudarc 0.17.3
**Rust Edition**: 2024
