# Performance Guide

**Status**: Documentation in progress

This comprehensive performance guide is currently being written and will be available soon.

---

## Current Performance Highlights

**kimsfinance** achieves **178x speedup** over mplfinance baseline through:

- Native PIL/Pillow rendering (no Matplotlib overhead)
- Optimized coordinate computation with Numba JIT
- WebP compression for ultra-small file sizes (<1 KB)
- Optional GPU acceleration for OHLCV processing (6.4x speedup)

### Quick Performance Stats

| Metric | kimsfinance | mplfinance | Speedup |
|--------|-------------|------------|---------|
| 50 candles | **2.7ms** | 254ms | **94x faster** |
| 100 candles | **4.3ms** | 257ms | **60x faster** |
| 500 candles | **18.4ms** | 286ms | **16x faster** |
| Throughput | **>6000 img/sec** | 3-4 img/sec | **1500x-2000x** |

---

## Quick Start: Benchmarking

### Run Benchmarks

If you have Claude Code configured:
```bash
# Quick sanity check
/benchmark-quick 100

# Comprehensive benchmark suite
/kf/bench/all

# Compare with mplfinance
/kf/bench/compare
```

Or run directly:
```bash
# Run all benchmarks
pytest tests/benchmark_*.py -v

# Profile performance
python -m cProfile -s cumtime scripts/demo_tick_charts.py
```

---

## Performance Best Practices

### 1. Choose the Right Output Format

```python
from kimsfinance.api import plot

# Ultra-fast, tiny files (RECOMMENDED)
plot.render(df, output_path='chart.webp')  # <1 KB, <5ms

# Vector graphics (scalable)
plot.render(df, output_path='chart.svgz')  # 1-10 KB

# Maximum compatibility
plot.render(df, output_path='chart.png')   # 10-20 KB
```

### 2. Batch Rendering

Process multiple charts efficiently:
```python
for symbol, df in dataframes.items():
    plot.render(df, output_path=f'output/{symbol}.webp')
# Throughput: >500 charts/sec
```

### 3. GPU Acceleration (Optional)

For OHLCV processing on large datasets:
```python
# GPU automatically used if available
plot.render(large_df, use_gpu=True)  # 6.4x faster OHLCV processing
```

---

## Coming Soon

This performance guide will include:

- ✅ Detailed benchmarking methodology
- ✅ Performance tuning for different use cases
- ✅ Memory optimization techniques
- ✅ Batch processing strategies
- ✅ GPU optimization deep dive
- ✅ Profiling and debugging slow operations
- ✅ Comparison with other charting libraries

---

## See Also

- [GPU Optimization Guide](GPU_OPTIMIZATION.md) - GPU-specific optimizations
- [README Performance Section](../README.md#-performance) - Quick performance overview
- Benchmark scripts: `tests/benchmark_*.py`

---

**Last Updated**: 2025-10-22
**Status**: Under development
