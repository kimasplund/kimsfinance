# kimsfinance_core - GPU-Accelerated Financial Indicators

**Version**: 0.2.0
**Status**: Production Ready
**Language**: Rust (Edition 2024)
**GPU**: NVIDIA CUDA (via cudarc)

---

## Overview

High-performance GPU-accelerated financial technical indicators written in Rust with Python bindings (PyO3). Provides **1.5x to 80x speedup** over CPU implementations using NVIDIA CUDA.

**v0.2.0 Highlights**:
- **CPU-GPU Hybrid Architecture**: 1.5x - 6.8x faster than pure-GPU for sequential indicators
- **5 indicators optimized**: EMA, RSI, ATR, Elder Ray, Keltner Channels
- **Smart algorithm selection**: CPU for sequential operations, GPU for parallel operations

---

## Features

### Performance

**Sequential Indicators** (v0.2.0 hybrid architecture):
- **EMA**: 6.8x faster (CPU-optimized)
- **RSI**: 1.9x faster (GPU+CPU+GPU hybrid)
- **Elder Ray**: 2.0x faster (CPU+GPU hybrid)
- **ATR**: 1.5x faster (GPU+CPU hybrid)
- **Keltner**: 1.9x faster (cascades from EMA+ATR)

**Parallel Indicators** (GPU-accelerated):
- **SMA, WMA, VWMA**: 30-55x speedup
- **Bollinger Bands**: 20-30x speedup
- **Donchian Channels**: 50-80x speedup
- **ROC**: 30-50x speedup
- **Williams %R, Aroon**: 15-25x speedup
- **CCI, Stochastic**: 15-30x speedup

### Supported Indicators

**Trend Indicators**:
- EMA (Exponential Moving Average) - **CPU-optimized** ✨
- SMA (Simple Moving Average) - GPU
- WMA (Weighted Moving Average) - GPU
- Keltner Channels - **CPU+GPU hybrid** ✨

**Momentum Indicators**:
- RSI (Relative Strength Index) - **GPU+CPU+GPU hybrid** ✨
- ROC (Rate of Change) - GPU
- Williams %R - GPU
- Aroon - GPU
- Stochastic Oscillator - GPU
- CCI (Commodity Channel Index) - GPU
- MACD - GPU

**Volatility Indicators**:
- ATR (Average True Range) - **GPU+CPU hybrid** ✨
- Bollinger Bands - GPU
- Donchian Channels - GPU

**Volume Indicators**:
- OBV (On-Balance Volume) - GPU
- VWAP (Volume Weighted Average Price) - GPU
- CMF (Chaikin Money Flow) - GPU
- VWMA (Volume Weighted Moving Average) - GPU

**Price Action Indicators**:
- Elder Ray (Bull/Bear Power) - **CPU+GPU hybrid** ✨

✨ = **New in v0.2.0**: CPU-GPU Hybrid Architecture

---

## Installation

### Requirements

- Rust 1.90+ (Edition 2024)
- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- CUDA Toolkit 12.8+ (or compatible driver)

### Build from Source

```bash
git clone https://github.com/kimsfinance/kimsfinance_core.git
cd kimsfinance_core/rust

# Build with GPU support
cargo build --release --features gpu

# Run tests
cargo test --features gpu

# Run benchmarks
cargo bench --features gpu
```

### As a Rust Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
kimsfinance_core = { version = "0.2.0", features = ["gpu"] }
```

---

## Quick Start

### Example 1: EMA (CPU-optimized)

```rust
use kimsfinance_core::cpu::sequential::ema_cpu;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate sample data
    let close = Array1::from_vec((0..100_000).map(|i| 100.0 + i as f64 * 0.01).collect());

    // Calculate EMA (pure CPU - 6.8x faster than old GPU!)
    let ema = ema_cpu(&close, 20)?;

    println!("EMA calculated in ~25μs for 100K candles!");
    Ok(())
}
```

### Example 2: RSI (Hybrid GPU+CPU+GPU)

```rust
use kimsfinance_core::gpu::{GpuDevice, rsi_gpu};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU
    let device = GpuDevice::new()?;

    // Generate sample data
    let close = Array1::from_vec((0..100_000).map(|i| 100.0 + i as f64 * 0.01).collect());

    // Calculate RSI (hybrid: GPU parallel + CPU smoothing + GPU parallel)
    let rsi = rsi_gpu(&device, &close, 14, None)?;

    println!("RSI calculated in ~130μs for 100K candles (1.9x faster than old GPU!)");
    Ok(())
}
```

### Example 3: Multiple Indicators (Batch Processing)

```rust
use kimsfinance_core::gpu::{GpuDevice, BatchIndicatorRequest};
use kimsfinance_core::cpu::sequential::ema_cpu;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Sample OHLCV data
    let close = Array1::from_vec((0..100_000).map(|i| 100.0 + i as f64 * 0.01).collect());
    let high = close.mapv(|v| v * 1.02);
    let low = close.mapv(|v| v * 0.98);

    // Calculate EMA on CPU (fastest)
    let ema_20 = ema_cpu(&close, 20)?;

    // Calculate other indicators on GPU
    let rsi_14 = rsi_gpu(&device, &close, 14, None)?;
    let atr_14 = atr_gpu(&device, &high, &low, &close, 14, None)?;

    println!("All indicators calculated efficiently!");
    Ok(())
}
```

---

## Architecture

### CPU-GPU Hybrid Strategy (v0.2.0)

**Key Insight**: Not all algorithms benefit from GPU acceleration.

#### When to Use CPU

**Sequential algorithms** (data dependencies prevent parallelization):
- ✅ EMA, Wilder's smoothing (IIR filters)
- ✅ Cumulative operations with dependencies
- CPU is **4-7x faster** for single-threaded sequential loops

#### When to Use GPU

**Parallel algorithms** (independent operations):
- ✅ Element-wise operations (subtraction, division, multiplication)
- ✅ Rolling window operations (max, min, sum)
- ✅ Independent calculations per element
- GPU is **15-80x faster** for parallel operations

#### Hybrid Architecture

**Best of both worlds**:
```
Input Data
    ↓
CPU: Sequential operations (EMA, Wilder's smoothing)
    ↓
GPU: Parallel operations (subtraction, rolling windows)
    ↓
CPU/GPU: Final aggregation
    ↓
Output
```

**Example: RSI Pipeline**:
1. **GPU**: Parallel gains/losses calculation (~20μs)
2. **CPU**: Wilder's smoothing for gains (~15μs) - sequential, faster on CPU
3. **CPU**: Wilder's smoothing for losses (~15μs) - sequential, faster on CPU
4. **GPU**: Parallel RSI calculation (~15μs)
5. **Total**: ~130μs (vs ~250μs pure-GPU, 1.9x faster!)

Even with extra PCIe transfers (H2D + D2H = ~64μs), CPU smoothing is 3-4x faster than single-thread GPU, resulting in net performance win.

---

## Performance Benchmarks

### Hardware Configuration

- **CPU**: Intel i9-13980HX (24 cores, 32 threads, 5.6 GHz boost)
- **GPU**: NVIDIA RTX 3500 Ada Generation Laptop GPU (12GB VRAM, 5120 CUDA cores)
- **RAM**: 64GB DDR5
- **OS**: Linux 6.17.0-5-generic

### Benchmark Results (100K candles)

#### CPU-GPU Hybrid Indicators (v0.2.0)

| Indicator | Old (v0.1.0) | New (v0.2.0) | Speedup | Architecture |
|-----------|--------------|--------------|---------|--------------|
| **EMA** | 170μs | 25μs | **6.8x** | Pure CPU |
| **Elder Ray** | 200μs | 100μs | **2.0x** | CPU+GPU Hybrid |
| **RSI** | 250μs | 130μs | **1.9x** | GPU+CPU+GPU Hybrid |
| **ATR** | 238μs | 163μs | **1.5x** | GPU+CPU Hybrid |
| **Keltner** | 378μs | 198μs | **1.9x** | CPU+GPU Hybrid |

**Average improvement**: 2.8x faster than v0.1.0

#### Pure GPU Indicators (Unchanged)

| Indicator | Time (100K) | Speedup vs CPU | Architecture |
|-----------|-------------|----------------|--------------|
| **SMA** | 45μs | 30x | GPU Parallel |
| **WMA** | 38μs | 35x | GPU Parallel |
| **Bollinger** | 95μs | 25x | GPU Parallel |
| **Donchian** | 30μs | 60x | GPU Parallel |
| **ROC** | 28μs | 40x | GPU Parallel |
| **Williams %R** | 42μs | 20x | GPU Parallel |
| **Aroon** | 48μs | 18x | GPU Parallel |
| **CCI** | 52μs | 20x | GPU Parallel |

### Scaling Performance

**EMA (CPU-optimized)**:
```
1K candles:   2.5μs  (400K candles/sec)
10K candles:  6.2μs  (1.6M candles/sec)
100K candles: 25μs   (4M candles/sec)
1M candles:   250μs  (4M candles/sec)
```

**RSI (Hybrid)**:
```
1K candles:   16μs   (62K candles/sec)
10K candles:  33μs   (303K candles/sec)
100K candles: 130μs  (769K candles/sec)
1M candles:   1.3ms  (769K candles/sec)
```

**Donchian (Pure GPU)**:
```
1K candles:   12μs   (83K candles/sec)
10K candles:  18μs   (555K candles/sec)
100K candles: 30μs   (3.3M candles/sec)
1M candles:   180μs  (5.5M candles/sec)
```

---

## Migration from v0.1.0

### Breaking Changes

**EMA API Change**:

```rust
// ❌ Deprecated (v0.1.0)
use kimsfinance_core::gpu::{GpuDevice, ema_gpu};
let device = GpuDevice::new()?;
let ema = ema_gpu(&device, &close, 20, None)?;  // 6.8x SLOWER

// ✅ Recommended (v0.2.0)
use kimsfinance_core::cpu::sequential::ema_cpu;
let ema = ema_cpu(&close, 20)?;  // 6.8x FASTER

// ✅ Alternative (v0.2.0, backward compatible)
use kimsfinance_core::gpu::{GpuDevice, ema_hybrid};
let device = GpuDevice::new()?;
let ema = ema_hybrid(&device, &close, 20, None)?;  // Also 6.8x FASTER
```

### Other Indicators

**No code changes needed!** All other indicators automatically benefit from v0.2.0 optimizations:
- RSI: 1.9x faster (automatic)
- Elder Ray: 2.0x faster (automatic)
- ATR: 1.5x faster (automatic)
- Keltner: 1.9x faster (automatic)

See [`docs/MIGRATION_GUIDE_v0.2.0.md`](./docs/MIGRATION_GUIDE_v0.2.0.md) for detailed instructions.

---

## Documentation

- **[CPU-GPU Hybrid Strategy](./docs/CPU_GPU_HYBRID_STRATEGY.md)** - Technical deep-dive into hybrid architecture
- **[Migration Guide v0.2.0](./docs/MIGRATION_GUIDE_v0.2.0.md)** - Step-by-step migration from v0.1.0
- **[CHANGELOG](./CHANGELOG.md)** - Complete version history
- **[Benchmark Report](./HYBRID_BENCHMARK_REPORT.md)** - Detailed performance analysis
- **[Benchmark Usage](./benches/BENCHMARK_USAGE.md)** - How to run benchmarks

---

## Running Benchmarks

### Quick Benchmark

```bash
# Run all hybrid benchmarks
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark

# Run specific indicator
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark -- EMA

# Quick mode (faster iteration)
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark -- --quick
```

### Benchmark Output Example

```
EMA_Comparison/Old_GPU_SingleThread/100000
                        time:   [168.23 μs 170.45 μs 172.89 μs]
                        thrpt:  [578.42 Kelem/s 586.73 Kelem/s 594.54 Kelem/s]

EMA_Comparison/New_CPU/100000
                        time:   [24.12 μs 25.03 μs 25.98 μs]
                        thrpt:  [3.85 Melem/s 3.99 Melem/s 4.14 Melem/s]

Speedup: 6.8x ✅
```

### HTML Reports

```bash
# Generate HTML benchmark reports
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark

# View in browser
firefox target/criterion/report/index.html
```

---

## Testing

### Run All Tests

```bash
cargo test --features gpu
```

### Run Specific Tests

```bash
# Test EMA
cargo test --features gpu -- ema

# Test RSI
cargo test --features gpu -- rsi

# Test hybrid architecture
cargo test --features gpu -- hybrid
```

### Test Coverage

```bash
cargo tarpaulin --features gpu --out Html
```

---

## Development

### Project Structure

```
rust/
├── src/
│   ├── lib.rs              # PyO3 module definition
│   ├── cpu/
│   │   ├── mod.rs          # CPU module
│   │   └── sequential.rs   # CPU-optimized sequential algorithms (EMA, Wilder's, SMA)
│   └── gpu/
│       ├── mod.rs          # GPU module exports
│       ├── device.rs       # GPU device management
│       ├── ema.rs          # EMA (CPU-optimized) ✨
│       ├── rsi.rs          # RSI (GPU+CPU+GPU hybrid) ✨
│       ├── elder_ray.rs    # Elder Ray (CPU+GPU hybrid) ✨
│       ├── atr.rs          # ATR (GPU+CPU hybrid) ✨
│       ├── keltner.rs      # Keltner (CPU+GPU hybrid) ✨
│       ├── sma.rs          # SMA (GPU parallel)
│       ├── wma.rs          # WMA (GPU parallel)
│       └── ...             # Other indicators
├── benches/
│   ├── cpu_gpu_hybrid_benchmark.rs  # Hybrid architecture benchmarks
│   ├── README.md                     # Benchmark documentation
│   └── BENCHMARK_USAGE.md            # Usage guide
├── docs/
│   ├── CPU_GPU_HYBRID_STRATEGY.md   # Hybrid architecture design
│   ├── MIGRATION_GUIDE_v0.2.0.md    # Migration guide
│   └── ...
├── Cargo.toml
├── CHANGELOG.md
└── README.md (this file)
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - See [LICENSE](./LICENSE) for details

---

## Acknowledgments

- **cudarc**: Rust CUDA bindings
- **ndarray**: N-dimensional arrays in Rust
- **PyO3**: Rust Python bindings
- **Criterion**: Benchmarking framework

---

## Version History

- **v0.2.0** (2025-10-25) - CPU-GPU Hybrid Architecture ✨
  - 1.5x - 6.8x speedup for sequential indicators
  - Smart algorithm selection (CPU vs GPU)
  - Breaking change: `ema_gpu()` deprecated
- **v0.1.0** (2025-10-24) - Initial GPU Release
  - 15-80x speedup for parallel indicators
  - 20+ financial indicators
  - CUDA backend via cudarc

---

**Maintained By**: kimsfinance team
**Repository**: https://github.com/kimsfinance/kimsfinance_core
**Documentation**: https://docs.kimsfinance.io
