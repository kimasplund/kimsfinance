# Rust Coordinate Calculations - Performance Exploration

**Status**: Experimental (dev-rust branch)
**Conclusion**: Not recommended for production use

## Overview

This directory contains a Rust implementation of kimsfinance coordinate calculations using PyO3 bindings. The goal was to evaluate whether Rust could provide significant performance improvements over the existing NumPy implementation.

## Implementation Details

- **Language**: Rust 1.85+ Edition 2024
- **Python Bindings**: PyO3 0.27.1 with abi3-py313
- **Dependencies**: ndarray 0.16.1, rayon 1.11.0
- **Build System**: Maturin
- **Total Code**: 737 lines across 4 files

### Features

- SIMD vectorization via ndarray Zip
- Rayon parallel processing for datasets ≥5,000 candles
- Zero-copy NumPy array interface
- Pre-allocated output buffers (zero-allocation hot path)

## Performance Results (Verified)

Comprehensive benchmarks comparing Rust vs NumPy coordinate calculations:

| Dataset Size | NumPy Time | Rust Time | Speedup | Status |
|-------------|-----------|-----------|---------|--------|
| 100 candles | 0.073 ms | 0.018 ms | **4.12x** | ✅ Faster |
| 1,000 candles | 0.116 ms | 0.032 ms | **3.62x** | ✅ Faster |
| 10,000 candles | 0.364 ms | 0.541 ms | **0.67x** | ❌ SLOWER |
| 100,000 candles | 1.395 ms | 1.496 ms | **0.93x** | ❌ SLOWER |

**Benchmark Configuration**:
- Hardware: Intel i9-13980HX, 64GB DDR5
- Iterations: 100-1000 per test (adaptive)
- Methodology: Median of repeated runs with warmup

## Analysis

### Why Rust is Faster (Small Datasets)

For datasets <1,000 candles:
- Rust's compiled nature eliminates interpreter overhead
- Direct memory access without Python object allocation
- Efficient tight loops with predictable performance
- SIMD optimizations from LLVM backend

### Why NumPy is Faster (Large Datasets)

For datasets >10,000 candles:

1. **Optimized BLAS/LAPACK Backends**: NumPy uses highly optimized linear algebra libraries (OpenBLAS, Intel MKL) that have decades of optimization
2. **Vectorized Operations**: NumPy's vectorized operations stay in C/Fortran land without Python boundary crossings
3. **Python ↔ Rust Transfer Costs**: Converting PyReadonlyArray → ndarray and back has overhead
4. **Rayon Overhead**: Thread pool setup and task distribution costs dominate for the relatively simple coordinate calculations
5. **Cache Efficiency**: NumPy's memory layout and access patterns are extremely cache-friendly

## Conclusion

**Rust is NOT worth the complexity for this use case.**

### Reasons:

1. **Typical Workloads Are Large**: Most real-world charting involves 1,000-100,000 candles where Rust is slower
2. **Maintenance Burden**: Adds Rust toolchain requirement, complicates build process
3. **NumPy Already Optimal**: NumPy's BLAS/LAPACK backends are world-class
4. **Marginal Gains**: Even the 3-4x speedup for small datasets is negligible in absolute terms (0.05ms → 0.02ms)
5. **Ecosystem Integration**: NumPy integrates seamlessly with pandas, polars, GPU libraries

### When Rust MIGHT Be Worth It:

- Custom algorithms not available in NumPy
- Complex branching logic in hot loops
- Memory-constrained environments
- Sub-millisecond latency requirements for small batches

## Realistic Performance Claims (Python 3.13)

After validating all claims in \`validate_performance_claims.py\`:

| Optimization | Claimed | Actual | Status |
|--------------|---------|--------|--------|
| Polars GPU | 13x | **1.95x** | ⚠️ Lower than expected |
| Numba JIT | 1.2x | **1.2x** | ✅ Validated |
| Python 3.14t free-threading | 3.1x | **N/A** | ❌ Ecosystem not ready |
| Total vs mplfinance | 4,154x | **~67x** | ✅ Realistic |

### Ecosystem Limitations

**Python 3.14t (free-threading)** is NOT viable because:
- Numba: No Python 3.14 support (requires <3.14)
- cuDF: No Python 3.14 builds available
- CuPy: No Python 3.14 packages
- Polars: Re-enables GIL even on python3.14t (not GIL-safe yet)

**Recommendation**: Stay on Python 3.13 with GPU + JIT for best performance.

## Build Instructions

### Requirements

- Rust 1.85+ (\`rustup update\`)
- Python 3.13+
- Maturin (\`pip install maturin\`)

### Development Build

\`\`\`bash
cd rust/
maturin develop --release
\`\`\`

### Testing

\`\`\`bash
# Run benchmark
python benchmark_rust.py

# Expected output:
# Dataset: 100 candles - Speedup: 4.12x ✅
# Dataset: 1,000 candles - Speedup: 3.62x ✅
# Dataset: 10,000 candles - Speedup: 0.67x ❌ SLOWER
# Dataset: 100,000 candles - Speedup: 0.93x ❌ SLOWER
\`\`\`

## Files

- \`Cargo.toml\` - Rust project configuration
- \`pyproject.toml\` - Maturin build configuration
- \`src/lib.rs\` - PyO3 bindings and Python module definition
- \`src/coordinates.rs\` - Core coordinate calculation logic (337 lines)
- \`src/types.rs\` - Shared data structures
- \`README.md\` - This file

## Lessons Learned

1. **Benchmark First**: Always validate performance claims with realistic workloads
2. **Trust NumPy**: For array operations, NumPy's BLAS/LAPACK backends are hard to beat
3. **Measure Overhead**: Python ↔ FFI boundaries have costs
4. **Ecosystem Matters**: Rust adds toolchain complexity and build time
5. **Absolute vs Relative**: 3x speedup of 0.05ms is less valuable than 1.5x speedup of 100ms

## Future Considerations

This Rust implementation is preserved on the \`dev-rust\` branch for:
- Reference implementation demonstrating PyO3 patterns
- Future exploration if use cases change
- Potential use in batch processing of many small charts

For the main kimsfinance library, **stick with NumPy** - it's faster, simpler, and battle-tested.

---

**Last Updated**: 2025-10-25
**Branch**: dev-rust
**Benchmark Results**: Verified and reproducible
