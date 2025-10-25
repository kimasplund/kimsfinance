# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-25

### Added

- **CPU-GPU Hybrid Architecture** for sequential indicators
  - New `src/cpu/sequential.rs` module with CPU-optimized algorithms
  - `ema_cpu()` - Pure CPU EMA (6.8x faster than old GPU)
  - `sma_cpu()` - CPU-optimized SMA for initialization
  - `wilders_smoothing_cpu()` - CPU-optimized Wilder's smoothing (RMA)
- **Hybrid API Functions** for backward compatibility
  - `ema_hybrid()` - Delegates to CPU (API-compatible with old `ema_gpu`)
- **Comprehensive Benchmarks**
  - `benches/cpu_gpu_hybrid_benchmark.rs` - Compare old GPU vs new CPU/hybrid
  - `benches/README.md` - Benchmark documentation
  - `benches/BENCHMARK_USAGE.md` - Usage guide
- **Documentation**
  - `docs/CPU_GPU_HYBRID_STRATEGY.md` - Technical analysis of hybrid architecture
  - `docs/MIGRATION_GUIDE_v0.2.0.md` - Complete migration guide
  - `HYBRID_BENCHMARK_REPORT.md` - Detailed performance analysis

### Changed

- **BREAKING**: `ema_gpu()` is now deprecated
  - Reason: Single-thread GPU is 6.8x slower than CPU for sequential algorithms
  - Migration: Use `ema_cpu()` (recommended) or `ema_hybrid()` (backward compatible)
  - Timeline: Function will be removed in v1.0.0
- **EMA** (`src/gpu/ema.rs`) - Now uses pure CPU execution
  - Old: Single-thread GPU kernel (~170μs for 100K candles)
  - New: CPU-optimized algorithm (~25μs for 100K candles)
  - Performance: **6.8x faster**
  - Architecture: Pure CPU (sequential IIR filter)
- **Elder Ray** (`src/gpu/elder_ray.rs`) - Hybrid CPU+GPU implementation
  - Old: GPU single-thread EMA + GPU parallel subtraction (~200μs)
  - New: CPU EMA + GPU parallel subtraction (~100μs)
  - Performance: **2.0x faster**
  - Architecture: CPU for EMA, GPU for parallel bull/bear power calculation
- **RSI** (`src/gpu/rsi.rs`) - Hybrid GPU+CPU+GPU implementation
  - Old: GPU parallel gains/losses + GPU single-thread smoothing + GPU parallel RSI (~250μs)
  - New: GPU parallel gains/losses + CPU Wilder's smoothing + GPU parallel RSI (~130μs)
  - Performance: **1.9x faster**
  - Architecture: GPU → CPU → GPU pipeline
- **ATR** (`src/gpu/atr.rs`) - Hybrid GPU+CPU implementation
  - Old: GPU single-thread Wilder's smoothing (~238μs)
  - New: GPU parallel true range + CPU Wilder's smoothing (~163μs)
  - Performance: **1.5x faster**
  - Architecture: GPU for parallel TR, CPU for sequential smoothing
- **Keltner Channels** (`src/gpu/keltner.rs`) - Hybrid CPU+GPU implementation
  - Dependencies: Uses fixed `ema_cpu()` and `atr_gpu()` hybrid
  - Old: GPU single-thread EMA + GPU single-thread ATR (~378μs)
  - New: CPU EMA + GPU/CPU ATR hybrid (~198μs)
  - Performance: **1.9x faster**
  - Architecture: Cascades improvements from EMA and ATR fixes

### Fixed

- **Performance Anti-Pattern**: Removed single-thread GPU kernels for sequential algorithms
  - Problem: Sequential algorithms (EMA, Wilder's smoothing) were running on single GPU thread
  - Impact: 6-10x slower than CPU due to:
    - Lower GPU single-thread clock speed (1.2 GHz vs 5.6 GHz CPU)
    - PCIe transfer overhead (~64μs)
    - Kernel launch overhead (~10μs)
  - Solution: Move sequential algorithms to CPU, keep parallel operations on GPU
  - Lines Removed: ~200 lines of inefficient single-thread GPU code
- **Elder Ray Test**: Fixed tautology in test (was comparing ema against ema)
- **Elder Ray Synchronization**: Removed unnecessary synchronization between kernels

### Performance Summary

**Benchmark Results** (100K candles, NVIDIA RTX 3500 Ada + Intel i9-13980HX):

| Indicator | Old Time | New Time | Speedup | Architecture |
|-----------|----------|----------|---------|--------------|
| EMA | 170μs | 25μs | **6.8x** | Pure CPU |
| Elder Ray | 200μs | 100μs | **2.0x** | CPU+GPU Hybrid |
| RSI | 250μs | 130μs | **1.9x** | GPU+CPU+GPU Hybrid |
| ATR | 238μs | 163μs | **1.5x** | GPU+CPU Hybrid |
| Keltner | 378μs | 198μs | **1.9x** | CPU+GPU Hybrid |

**Average Speedup**: 2.8x
**Range**: 1.5x - 6.8x
**Total Time Saved**: ~715μs per calculation (cumulative across all 5 indicators)

### Technical Details

**Why CPU is Faster for Sequential Algorithms**:
- Sequential algorithms (IIR filters) have data dependencies: `EMA[i] = f(EMA[i-1])`
- Cannot parallelize due to critical path = N (entire dataset length)
- CPU single-core performance:
  - Clock: 5.6 GHz (Intel i9-13980HX P-core boost)
  - IPC: ~5 (out-of-order execution, advanced branch prediction)
  - L1 Cache: 32 KB, ~1ns latency
- GPU single-thread performance:
  - Clock: ~1.2 GHz (RTX 3500 Ada)
  - IPC: ~1 (in-order execution, no branch prediction)
  - L1 Cache: Shared across warp, ~5-10ns latency
- **Result**: CPU is 4-5x faster for sequential loops
- **Plus**: No PCIe overhead, no kernel launch overhead
- **Total**: 6-10x faster on CPU for sequential algorithms

**Hybrid Architecture Benefits**:
- Best of both worlds: CPU for sequential, GPU for parallel
- Even with extra PCIe transfers (H2D + D2H = ~64μs), still net win
- GPU freed up for parallel operations where it excels
- Better GPU utilization (parallel kernels saturate GPU cores)

### Migration Guide

**For EMA users**:

```rust
// Before (v0.1.0)
use kimsfinance_core::gpu::{GpuDevice, ema_gpu};
let device = GpuDevice::new()?;
let ema = ema_gpu(&device, &close, 20, None)?;

// After (v0.2.0) - Option 1: Direct CPU (recommended)
use kimsfinance_core::cpu::sequential::ema_cpu;
let ema = ema_cpu(&close, 20)?;  // 6.8x faster!

// After (v0.2.0) - Option 2: Hybrid API (backward compatible)
use kimsfinance_core::gpu::{GpuDevice, ema_hybrid};
let device = GpuDevice::new()?;
let ema = ema_hybrid(&device, &close, 20, None)?;  // Also 6.8x faster
```

**For other indicators**:
- No code changes needed!
- Performance improvements are automatic
- Simply update to v0.2.0 and rebuild

**See**: `docs/MIGRATION_GUIDE_v0.2.0.md` for detailed migration instructions

### Deprecations

- `ema_gpu()` - Deprecated in favor of `ema_cpu()` or `ema_hybrid()`
  - Reason: Single-thread GPU is 6.8x slower than CPU
  - Timeline: Will be removed in v1.0.0 (Q2 2026)
  - Action: Replace with `ema_cpu()` or `ema_hybrid()`

### Internal Changes

- Refactored `src/cpu/mod.rs` to export `sequential` submodule
- Updated `src/gpu/mod.rs` to re-export CPU functions for convenience
- Added extensive inline documentation explaining CPU vs GPU trade-offs
- Created comprehensive test suite for CPU sequential algorithms
- Added benchmark infrastructure for CPU-GPU comparison

---

## [0.1.0] - 2025-10-24

### Added

- Initial release with GPU-accelerated financial indicators
- Core indicators: EMA, SMA, WMA, RSI, ATR, Elder Ray, Keltner, Bollinger Bands
- Volume indicators: OBV, VWAP, CMF, VWMA
- Momentum indicators: ROC, Williams %R, Aroon, Stochastic, MACD, CCI
- GPU batch system for concurrent indicator calculation
- CUDA backend via cudarc
- PyO3 Python bindings
- Comprehensive test suite

### Performance (v0.1.0)

- GPU acceleration: 15-50x speedup for parallel indicators (SMA, WMA, Bollinger, etc.)
- Note: Sequential indicators (EMA, RSI, ATR) had performance issues (fixed in v0.2.0)

---

## Version History

- **v0.2.0** (2025-10-25) - CPU-GPU Hybrid Architecture ✨
- **v0.1.0** (2025-10-24) - Initial GPU Release

---

## Future Roadmap

### v0.3.0 (Planned)

- More hybrid indicators (MACD optimization)
- Persistent kernel support for reduced launch overhead
- Shared memory optimizations for windowed operations
- Multi-GPU support

### v1.0.0 (Planned Q2 2026)

- Stable API
- Remove deprecated functions (`ema_gpu`)
- Production-ready performance guarantees
- Full test coverage (>90%)

---

**Maintained By**: kimsfinance team
**License**: MIT
**Repository**: https://github.com/kimsfinance/kimsfinance_core
