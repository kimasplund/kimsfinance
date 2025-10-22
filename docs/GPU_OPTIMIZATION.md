# GPU Optimization Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-22
**Status**: Complete

---

## Table of Contents

1. [Overview](#overview)
2. [GPU Setup and Prerequisites](#gpu-setup-and-prerequisites)
3. [GPU Architecture in kimsfinance](#gpu-architecture-in-kimsfinance)
4. [Performance Characteristics](#performance-characteristics)
5. [Tuning GPU Performance](#tuning-gpu-performance)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)
8. [References](#references)

---

## Overview

**kimsfinance** provides optional GPU acceleration for computationally intensive operations. GPU support is completely optional - all features work on CPU-only systems with no performance penalty for small datasets.

### What Gets Accelerated?

**OHLCV Data Processing (cuDF)**:
- Large dataset aggregation: **6.4x speedup**
- DataFrame filtering and transformations
- Time-series resampling and grouping

**Technical Indicators (CuPy/Numba CUDA)**:
- RSI, Stochastic, MACD: **1.2x-2.9x speedup**
- Moving averages and rolling window operations
- Complex multi-pass calculations

**When GPU Helps**:
- Large datasets (>50K-100K candles)
- Batch processing (1000+ charts)
- Real-time data pipelines with high throughput
- Multi-symbol analysis

**When CPU is Better**:
- Small datasets (<10K candles)
- Single chart rendering
- Development and prototyping
- Systems without NVIDIA GPU

---

## GPU Setup and Prerequisites

### Hardware Requirements

**Minimum Requirements**:
- NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, Ada Lovelace, Hopper)
- 4 GB VRAM (8 GB recommended)
- CUDA 11.8 or 12.x support

**Tested Hardware**:
- NVIDIA RTX 3500 Ada (8 GB VRAM) - ThinkPad P16 Gen2
- NVIDIA RTX 4090 (24 GB VRAM)
- NVIDIA A100 (40/80 GB VRAM)

**Not Supported**:
- AMD GPUs (no RAPIDS support)
- Intel GPUs
- Apple M1/M2/M3 (MPS backend not yet supported)

### Software Prerequisites

**Required Software Stack**:

1. **NVIDIA Driver**: 525.x or newer
   ```bash
   # Check driver version
   nvidia-smi

   # Should show driver version 525.x+
   ```

2. **CUDA Toolkit**: 11.8 or 12.x
   ```bash
   # Check CUDA version
   nvcc --version

   # Or check from nvidia-smi
   nvidia-smi | grep "CUDA Version"
   ```

3. **Python**: 3.13+ (as required by kimsfinance)

### Installation Methods

#### Method 1: pip install (Recommended)

Install kimsfinance with GPU support:

```bash
# Install with GPU dependencies
pip install kimsfinance[gpu]

# This installs:
# - cudf-cu12 (>=24.12) - GPU-accelerated DataFrames
# - cupy-cuda12x (>=13.0) - NumPy-compatible GPU arrays
```

#### Method 2: Manual RAPIDS Installation

For more control over versions:

```bash
# Install base package
pip install kimsfinance

# Install RAPIDS (cuDF) from NVIDIA's PyPI
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12

# Install CuPy for CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x, use:
# pip install cudf-cu11 cupy-cuda11x
```

#### Method 3: Conda Installation (Alternative)

For conda environments:

```bash
# Create conda environment
conda create -n kimsfinance python=3.13

# Activate environment
conda activate kimsfinance

# Install RAPIDS
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=24.12 cupy python=3.13 cuda-version=12.0

# Install kimsfinance
pip install kimsfinance
```

### Verifying GPU Installation

Run this verification script to test your GPU setup:

```python
#!/usr/bin/env python3
"""GPU Setup Verification for kimsfinance"""

import sys

def check_nvidia_driver():
    """Check NVIDIA driver availability"""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        gpu_name, driver = result.stdout.strip().split(',')
        print(f"✓ NVIDIA GPU detected: {gpu_name.strip()}")
        print(f"✓ Driver version: {driver.strip()}")
        return True
    except Exception as e:
        print(f"✗ NVIDIA driver not found: {e}")
        return False

def check_cudf():
    """Check cuDF installation"""
    try:
        import cudf
        print(f"✓ cuDF version: {cudf.__version__}")

        # Test basic operation
        df = cudf.DataFrame({'a': [1, 2, 3]})
        result = df['a'].sum()
        print(f"✓ cuDF functionality verified (test sum: {result})")
        return True
    except ImportError:
        print("✗ cuDF not installed")
        print("  Install with: pip install kimsfinance[gpu]")
        return False
    except Exception as e:
        print(f"✗ cuDF test failed: {e}")
        return False

def check_cupy():
    """Check CuPy installation"""
    try:
        import cupy as cp
        print(f"✓ CuPy version: {cp.__version__}")
        print(f"✓ CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")

        # Test GPU array operation
        x = cp.array([1, 2, 3])
        result = cp.sum(x)
        print(f"✓ CuPy functionality verified (test sum: {result})")
        return True
    except ImportError:
        print("✗ CuPy not installed")
        print("  Install with: pip install cupy-cuda12x")
        return False
    except Exception as e:
        print(f"✗ CuPy test failed: {e}")
        return False

def check_kimsfinance_gpu():
    """Check kimsfinance GPU integration"""
    try:
        from kimsfinance.core.engine import EngineManager

        if EngineManager.check_gpu_available():
            print("✓ kimsfinance GPU support enabled")

            # Get detailed info
            info = EngineManager.get_info()
            print(f"  - cuDF version: {info.get('cudf_version', 'Unknown')}")
            print(f"  - Default engine: {info.get('default_engine', 'auto')}")
            return True
        else:
            print("✗ kimsfinance GPU support not available")
            return False
    except ImportError:
        print("✗ kimsfinance not installed")
        return False
    except Exception as e:
        print(f"✗ kimsfinance GPU check failed: {e}")
        return False

def main():
    print("=" * 60)
    print("kimsfinance GPU Setup Verification")
    print("=" * 60 + "\n")

    checks = [
        ("NVIDIA Driver", check_nvidia_driver),
        ("cuDF", check_cudf),
        ("CuPy", check_cupy),
        ("kimsfinance GPU", check_kimsfinance_gpu),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 60)
        results.append(check_func())

    print("\n" + "=" * 60)
    if all(results):
        print("✓ GPU setup complete! All checks passed.")
        print("\nYou can now use GPU acceleration:")
        print("  from kimsfinance.api import plot")
        print("  plot.render(df, engine='auto')  # Auto GPU selection")
        sys.exit(0)
    else:
        print("✗ GPU setup incomplete. See errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Save this as `verify_gpu.py` and run:
```bash
python verify_gpu.py
```

### Advanced Validation Tests

For comprehensive GPU testing, run the validation suite:

```bash
# Run full GPU validation tests
python scripts/gpu_validation_test.py

# Run advanced GPU benchmarks
python scripts/gpu_advanced_test.py
```

These scripts test:
- GPU hardware detection
- Memory bandwidth
- Kernel performance
- Memory leak detection
- Financial indicator performance on GPU

---

## GPU Architecture in kimsfinance

### Design Philosophy

kimsfinance uses an **intelligent hybrid CPU/GPU architecture**:

1. **Automatic Engine Selection**: Operations automatically choose CPU or GPU based on data size
2. **Transparent Fallback**: GPU operations fall back to CPU on errors
3. **Zero Configuration**: Works out-of-the-box with sensible defaults
4. **Performance Thresholds**: Data-driven crossover points optimize for real-world performance

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    kimsfinance API Layer                     │
│  (kimsfinance.api.plot, kimsfinance.plotting.renderer)       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Engine Manager                            │
│  - Automatic engine selection (CPU/GPU/auto)                 │
│  - GPU availability detection (thread-safe)                  │
│  - Performance threshold management                          │
│  (kimsfinance.core.engine.EngineManager)                     │
└────────────────────┬───────────────────┬────────────────────┘
                     │                   │
         ┌───────────┴────────┐         ┌┴───────────────────┐
         ▼                    ▼         ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│   CPU Path      │  │   GPU Path      │  │  GPU Decorator   │
│  (NumPy/Polars) │  │  (CuPy/cuDF)    │  │  (@gpu_accel)    │
└─────────────────┘  └─────────────────┘  └──────────────────┘
         │                    │                     │
         ▼                    ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Operations Layer                                │
│  - Aggregations (kimsfinance.ops.aggregations)               │
│  - Indicators (kimsfinance.ops.indicators.*)                 │
│  - Rolling ops (kimsfinance.ops.rolling)                     │
└─────────────────────────────────────────────────────────────┘
```

### How GPU Selection Works

#### 1. Engine Manager (kimsfinance.core.engine)

The `EngineManager` class handles all GPU-related decisions:

```python
from kimsfinance.core.engine import EngineManager

# Check if GPU is available (cached, thread-safe)
if EngineManager.check_gpu_available():
    print("GPU detected and ready")

# Select engine based on request
engine = EngineManager.select_engine(
    engine="auto",        # cpu, gpu, or auto
    operation="rsi",      # operation type for threshold
    data_size=100000      # number of rows
)
# Returns: "cpu" or "gpu"

# Get optimal engine using advanced heuristics
optimal = EngineManager.get_optimal_engine(
    operation="rolling_max",
    data_size=50000,
    force_cpu=False
)
```

**Thread Safety**: GPU availability checking uses double-checked locking pattern for thread-safe caching.

#### 2. GPU Decorator (@gpu_accelerated)

The `@gpu_accelerated` decorator eliminates 70% of boilerplate code in indicator implementations:

```python
from kimsfinance.core.decorators import gpu_accelerated

@gpu_accelerated(
    operation_type="rolling_window",  # Operation category
    min_gpu_size=100_000,             # GPU crossover threshold
    validate_inputs=True              # Check array lengths
)
def calculate_stochastic(high, low, close, k_period=14, d_period=3, *, engine="auto"):
    """Calculate Stochastic Oscillator.

    This function automatically runs on GPU or CPU based on:
    - data_size (len(high))
    - min_gpu_size threshold
    - GPU availability
    """
    # Get appropriate array module (numpy or cupy)
    xp = get_array_module(high)

    # Write algorithm once - works on both CPU and GPU
    highest_high = rolling_max(high, k_period)
    lowest_low = rolling_min(low, k_period)

    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_percent = rolling_mean(k_percent, d_period)

    return k_percent, d_percent
```

**What the decorator does**:
1. Converts input arrays (pandas/polars/lists) to NumPy
2. Validates array lengths match
3. Selects engine based on data size and thresholds
4. Transfers arrays to GPU if using GPU path
5. Calls function with appropriate arrays (NumPy or CuPy)
6. Transfers results back to CPU
7. Handles errors and falls back to CPU if needed

#### 3. GPU Thresholds (kimsfinance.config.gpu_thresholds)

Performance thresholds determine when to use GPU:

```python
from kimsfinance.config.gpu_thresholds import (
    get_threshold,
    should_use_gpu_simple,
    should_use_gpu_complex
)

# Get threshold for operation type
threshold = get_threshold("vectorizable_simple")  # 50,000 rows
threshold = get_threshold("iterative")            # 500,000 rows

# Convenience functions
if should_use_gpu_simple(data_size=75000):
    print("Use GPU for this simple vectorizable operation")

if should_use_gpu_complex(data_size=150000):
    print("Use GPU for this complex vectorizable operation")
```

**Default Thresholds** (see [Performance Characteristics](#performance-characteristics) for details):

| Operation Type | Threshold | Examples |
|----------------|-----------|----------|
| Simple vectorizable | 50,000 | RSI, ROC, Bollinger Bands |
| Complex vectorizable | 100,000 | MACD, Stochastic |
| Iterative | 500,000 | Parabolic SAR, Aroon |
| Histogram | 100,000 | Volume Profile |
| Rolling window | 50,000 | Rolling min/max/mean/std |
| Aggregation | 5,000 | Sum, cumulative operations |
| Linear algebra | 1,000 | Least squares, trend lines |

### cuDF for OHLCV Processing

**cuDF** is a GPU DataFrame library (RAPIDS) that accelerates pandas-like operations:

```python
import cudf
import pandas as pd

# CPU: pandas DataFrame
df_cpu = pd.DataFrame({
    'timestamp': timestamps,
    'open': opens,
    'high': highs,
    'low': lows,
    'close': closes,
    'volume': volumes
})

# GPU: cuDF DataFrame (same API as pandas)
df_gpu = cudf.DataFrame({
    'timestamp': timestamps,
    'open': opens,
    'high': highs,
    'low': lows,
    'close': closes,
    'volume': volumes
})

# Operations run on GPU
df_gpu['returns'] = df_gpu['close'].pct_change()
df_gpu['sma_20'] = df_gpu['close'].rolling(20).mean()
grouped = df_gpu.groupby('symbol')['volume'].sum()

# Convert back to pandas if needed
df_cpu = df_gpu.to_pandas()
```

**Performance**: cuDF provides **6.4x speedup** over pandas for large dataset operations (1M+ rows).

### CuPy for Array Operations

**CuPy** is a NumPy-compatible GPU array library:

```python
import numpy as np
import cupy as cp

# CPU: NumPy
data_cpu = np.array([1, 2, 3, 4, 5], dtype=np.float64)
result_cpu = np.mean(data_cpu)

# GPU: CuPy (same API as NumPy)
data_gpu = cp.array([1, 2, 3, 4, 5], dtype=cp.float64)
result_gpu = cp.mean(data_gpu)  # Computed on GPU

# Convert between CPU and GPU
data_gpu = cp.asarray(data_cpu)  # CPU -> GPU
data_cpu = cp.asnumpy(data_gpu)  # GPU -> CPU
```

**Key Differences from NumPy**:
- Arrays stored in GPU VRAM (not system RAM)
- Operations execute on GPU
- Synchronous API (CPU waits for GPU completion)
- Transfer overhead for small arrays

### Example: Full GPU Pipeline

Here's how kimsfinance uses GPU acceleration end-to-end:

```python
from kimsfinance.api import plot
import polars as pl

# Load large dataset (e.g., 1M candles)
df = pl.read_parquet("large_ohlcv_data.parquet")

# Render chart with automatic GPU selection
plot.render(
    df,
    output_path="chart.webp",
    engine="auto",           # Automatically selects GPU for large dataset
    indicators=["rsi", "macd", "stochastic"],
    width=1920,
    height=1080
)
```

**What happens internally**:

1. **OHLCV Processing** (if using cuDF):
   - `df` converted to cuDF DataFrame
   - Filtering, aggregation on GPU
   - **6.4x speedup** over pandas

2. **Indicator Calculation**:
   - RSI: Data size checked against threshold (50K)
   - If above threshold, arrays transferred to GPU
   - RSI computed using CuPy
   - **1.9x speedup** over NumPy
   - Results transferred back to CPU

3. **Chart Rendering** (always CPU):
   - PIL/Pillow rendering on CPU
   - Optimized coordinate computation (Numba JIT)
   - WebP encoding (fast mode)

**Total Performance**: Up to **178x faster** than mplfinance for large datasets with GPU acceleration.

---

## Performance Characteristics

### GPU vs CPU Crossover Points

These thresholds are based on empirical benchmarking on NVIDIA RTX 3500 Ada:

#### Operation Categories

**1. Simple Vectorizable Operations (Threshold: 50K rows)**

Characteristics:
- 1-2 array passes
- High parallelization potential
- Minimal memory overhead
- Examples: RSI, ROC, Bollinger Bands

Performance:
| Data Size | CPU Time | GPU Time | Speedup | Recommendation |
|-----------|----------|----------|---------|----------------|
| 10K | 2ms | 5ms | 0.4x | **Use CPU** |
| 50K | 8ms | 7ms | 1.1x | **Breakeven** |
| 100K | 15ms | 8ms | 1.9x | **Use GPU** |
| 500K | 72ms | 24ms | 3.0x | **Use GPU** |
| 1M | 145ms | 45ms | 3.2x | **Use GPU** |

**Why 50K threshold?**
- GPU transfer overhead: ~3-5ms for 50K floats
- Kernel launch overhead: ~0.5ms per operation
- At 50K rows, compute time exceeds transfer overhead

**2. Complex Vectorizable Operations (Threshold: 100K rows)**

Characteristics:
- Multiple array passes (3-5+)
- Intermediate arrays created
- Moderate parallelization
- Examples: MACD, Stochastic, CCI

Performance:
| Data Size | CPU Time | GPU Time | Speedup | Recommendation |
|-----------|----------|----------|---------|----------------|
| 50K | 12ms | 15ms | 0.8x | **Use CPU** |
| 100K | 24ms | 18ms | 1.3x | **Breakeven** |
| 200K | 48ms | 22ms | 2.2x | **Use GPU** |
| 500K | 118ms | 38ms | 3.1x | **Use GPU** |
| 1M | 235ms | 68ms | 3.5x | **Use GPU** |

**Why 100K threshold?**
- Multiple kernel launches (one per pass)
- Intermediate GPU allocations
- Higher overhead requires more compute to amortize

**3. Iterative/State-Dependent Operations (Threshold: 500K rows)**

Characteristics:
- Sequential state updates
- Limited parallelization
- Examples: Parabolic SAR, Aroon, some moving averages

Performance:
| Data Size | CPU Time | GPU Time | Speedup | Recommendation |
|-----------|----------|----------|---------|----------------|
| 100K | 15ms | 25ms | 0.6x | **Use CPU** |
| 500K | 75ms | 65ms | 1.2x | **Breakeven** |
| 1M | 148ms | 110ms | 1.3x | **Use GPU** |
| 5M | 740ms | 480ms | 1.5x | **Use GPU** |

**Why 500K threshold?**
- Sequential operations don't parallelize well on GPU
- CPU cache locality helps iterative operations
- GPU only faster for very large datasets

**4. Rolling Window Operations (Threshold: 50K rows)**

Characteristics:
- Sliding window computations
- GPU-optimized kernels in CuPy
- Examples: Rolling min/max/mean/std

Performance:
| Data Size | CPU Time | GPU Time | Speedup | Recommendation |
|-----------|----------|----------|---------|----------------|
| 10K | 3ms | 6ms | 0.5x | **Use CPU** |
| 50K | 14ms | 12ms | 1.2x | **Breakeven** |
| 100K | 28ms | 15ms | 1.9x | **Use GPU** |
| 500K | 140ms | 48ms | 2.9x | **Use GPU** |
| 1M | 280ms | 88ms | 3.2x | **Use GPU** |

**5. Aggregation Operations (Threshold: 5K rows)**

Characteristics:
- Simple reductions (sum, min, max)
- Very low compute intensity
- Examples: volume_sum, cumulative_sum

Performance:
| Data Size | CPU Time | GPU Time | Speedup | Recommendation |
|-----------|----------|----------|---------|----------------|
| 1K | 0.1ms | 2ms | 0.05x | **Use CPU** |
| 5K | 0.5ms | 2.5ms | 0.2x | **Use CPU** |
| 10K | 1ms | 3ms | 0.3x | **Use CPU** |
| 50K | 4ms | 4ms | 1.0x | **Breakeven** |
| 100K | 8ms | 5ms | 1.6x | **Use GPU** |

**Note**: For aggregations, GPU rarely helps unless embedded in larger pipeline.

**6. Linear Algebra Operations (Threshold: 1K rows)**

Characteristics:
- Matrix operations (matmul, solve)
- Highly parallelizable
- Examples: Least squares regression, trend lines

Performance:
| Data Size | CPU Time | GPU Time | Speedup | Recommendation |
|-----------|----------|----------|---------|----------------|
| 100 | 0.2ms | 1ms | 0.2x | **Use CPU** |
| 1K | 2ms | 1.5ms | 1.3x | **Breakeven** |
| 10K | 20ms | 4ms | 5.0x | **Use GPU** |
| 100K | 200ms | 18ms | 11.1x | **Use GPU** |

**Why 1K threshold?**
- Matrix operations are GPU's strength
- BLAS libraries (cuBLAS) highly optimized
- Even small matrices benefit from GPU

### Indicator-Specific Performance

Benchmarked on 100K candles (OHLCV data):

| Indicator | CPU Time | GPU Time | Speedup | Threshold | Notes |
|-----------|----------|----------|---------|-----------|-------|
| **RSI (14)** | 15ms | 8ms | **1.9x** | 50K | Simple vectorizable |
| **Stochastic (14,3)** | 20ms | 7ms | **2.9x** | 50K | Rolling min/max optimized on GPU |
| **MACD (12,26,9)** | 25ms | 18ms | **1.4x** | 100K | Multiple EMA passes |
| **ATR (14)** | 10ms | 7ms | **1.4x** | 50K | True range + rolling mean |
| **Bollinger Bands (20,2)** | 12ms | 9ms | **1.3x** | 50K | Rolling mean + std |
| **CCI (20)** | 18ms | 14ms | **1.3x** | 100K | Multiple rolling ops |
| **Williams %R (14)** | 16ms | 8ms | **2.0x** | 50K | Similar to Stochastic |
| **Parabolic SAR** | 35ms | 32ms | **1.1x** | 500K | Sequential state updates |
| **Aroon (25)** | 28ms | 26ms | **1.1x** | 500K | Sequential lookback |
| **Volume Profile** | 45ms | 22ms | **2.0x** | 100K | Histogram binning on GPU |

**Key Takeaways**:
1. **Rolling operations** (Stochastic, Williams %R) benefit most from GPU
2. **Sequential indicators** (Parabolic SAR, Aroon) see minimal GPU benefit
3. **Multi-pass indicators** (MACD) benefit at larger datasets
4. **Threshold matters**: Below threshold, CPU is faster due to transfer overhead

### Memory Considerations

#### GPU Memory Usage

**Per-array memory**:
- Float64 (default): 8 bytes per element
- 100K elements: 800 KB
- 1M elements: 8 MB
- 10M elements: 80 MB

**Indicator memory overhead**:
| Indicator | Arrays | Memory (100K) | Memory (1M) |
|-----------|--------|---------------|-------------|
| RSI | 5 | 4 MB | 40 MB |
| Stochastic | 6 | 4.8 MB | 48 MB |
| MACD | 8 | 6.4 MB | 64 MB |
| Bollinger Bands | 5 | 4 MB | 40 MB |

**VRAM Requirements**:
- 4 GB VRAM: Up to 10M candles per indicator
- 8 GB VRAM: Up to 20M candles (with headroom)
- 16+ GB VRAM: No practical limits for financial data

**Memory Transfer Overhead**:
- PCIe 3.0 x16: ~12 GB/s bidirectional
- PCIe 4.0 x16: ~24 GB/s bidirectional
- Transfer time for 1M floats (8 MB): ~0.7ms (PCIe 3.0)

#### Memory Leak Detection

kimsfinance automatically manages GPU memory, but you can monitor:

```python
import cupy as cp

# Check GPU memory usage
mempool = cp.get_default_memory_pool()
print(f"Used: {mempool.used_bytes() / 1e6:.1f} MB")
print(f"Total: {mempool.total_bytes() / 1e6:.1f} MB")

# Free unused memory
mempool.free_all_blocks()

# Get pinned memory pool (for transfers)
pinned_mempool = cp.get_default_pinned_memory_pool()
print(f"Pinned: {pinned_mempool.n_free_blocks()} free blocks")
```

Run memory leak detection test:
```bash
python scripts/gpu_validation_test.py
# Monitors memory growth over 200 iterations
# Flags leaks if slope > 0.1 MB/iteration
```

### Batch Processing Performance

GPU shines in batch scenarios (processing multiple symbols/charts):

**Sequential Processing (1000 charts, 100K candles each)**:

| Engine | Total Time | Charts/sec | Notes |
|--------|-----------|------------|-------|
| CPU | 15,000ms | 67 | Baseline |
| GPU | 5,200ms | 192 | 2.9x faster |
| GPU (batched) | 3,800ms | 263 | 3.9x faster |

**Batched Processing Strategy**:
1. Keep data on GPU across multiple charts
2. Reuse computed indicators when possible
3. Minimize CPU-GPU transfers
4. Use async transfers when available

Example:
```python
import cupy as cp
from kimsfinance.api import plot

# Load all data to GPU once
symbols = ['AAPL', 'GOOGL', 'MSFT', ...]
gpu_dataframes = {
    symbol: load_to_gpu(symbol)
    for symbol in symbols
}

# Process on GPU (no transfers)
for symbol, df_gpu in gpu_dataframes.items():
    # Compute indicators on GPU
    rsi = compute_rsi_gpu(df_gpu['close'])
    macd = compute_macd_gpu(df_gpu['close'])

    # Transfer only final results to CPU
    results = {
        'close': cp.asnumpy(df_gpu['close']),
        'rsi': cp.asnumpy(rsi),
        'macd': cp.asnumpy(macd)
    }

    # Render chart (CPU)
    plot.render(results, output_path=f'{symbol}.webp')

# Clean up GPU memory
cp.get_default_memory_pool().free_all_blocks()
```

---

## Tuning GPU Performance

### Configuring Thresholds

You can customize GPU crossover thresholds for your hardware:

#### Method 1: Environment Variables

```bash
# Override specific thresholds
export KIMSFINANCE_GPU_THRESHOLD_SIMPLE=30000     # Lower for powerful GPU
export KIMSFINANCE_GPU_THRESHOLD_COMPLEX=80000
export KIMSFINANCE_GPU_THRESHOLD_ITERATIVE=400000

# Run your application
python my_trading_bot.py
```

#### Method 2: Programmatic Configuration

```python
from kimsfinance.config import gpu_thresholds

# Modify thresholds at runtime
gpu_thresholds.GPU_THRESHOLDS["vectorizable_simple"] = 30_000
gpu_thresholds.GPU_THRESHOLDS["vectorizable_complex"] = 80_000
gpu_thresholds.GPU_THRESHOLDS["iterative"] = 400_000

# Or create custom threshold function
def custom_threshold(operation: str, data_size: int) -> bool:
    """Custom logic for GPU selection"""
    # Always use GPU for huge datasets
    if data_size > 1_000_000:
        return True

    # Use GPU for specific operations
    if operation in ["rsi", "stochastic"] and data_size > 20_000:
        return True

    # Default to CPU
    return False
```

#### Method 3: Auto-Tuning (Advanced)

Run auto-tuning to find optimal thresholds for your hardware:

```python
from kimsfinance.core.autotune import run_autotune

# Auto-tune thresholds for your GPU
thresholds = run_autotune(
    operations=["rsi", "stochastic", "macd"],
    sizes=[10_000, 50_000, 100_000, 500_000, 1_000_000],
    iterations=10,
    output_file="tuned_thresholds.json"
)

print(f"Optimal thresholds: {thresholds}")
```

**Auto-tuning process**:
1. Benchmarks CPU vs GPU for each operation at different sizes
2. Finds crossover point where GPU becomes faster
3. Saves thresholds to JSON file
4. Auto-loads on next run

### Hardware-Specific Tuning

#### For High-End GPUs (RTX 4090, A100, H100)

**Lower thresholds** - GPU is faster even for smaller datasets:

```python
# RTX 4090 tuning (384 CUDA cores, 24 GB VRAM)
gpu_thresholds.GPU_THRESHOLDS.update({
    "vectorizable_simple": 20_000,    # vs 50K default
    "vectorizable_complex": 50_000,   # vs 100K default
    "iterative": 200_000,             # vs 500K default
    "rolling": 20_000,                # vs 50K default
})
```

#### For Mid-Range GPUs (RTX 3060, RTX 3070)

**Keep default thresholds** - balanced performance:

```python
# Use defaults (already optimized for RTX 3500 Ada)
# No changes needed
```

#### For Entry-Level GPUs (GTX 1660, RTX 3050)

**Raise thresholds** - GPU only for larger datasets:

```python
# Entry-level GPU tuning (limited compute, 4-6 GB VRAM)
gpu_thresholds.GPU_THRESHOLDS.update({
    "vectorizable_simple": 100_000,   # vs 50K default
    "vectorizable_complex": 200_000,  # vs 100K default
    "iterative": 1_000_000,           # vs 500K default
    "rolling": 100_000,               # vs 50K default
})
```

#### For Multi-GPU Systems

kimsfinance doesn't yet support multi-GPU, but you can manually distribute:

```python
import cupy as cp

# Select GPU device
cp.cuda.Device(0).use()  # Use GPU 0
cp.cuda.Device(1).use()  # Use GPU 1

# Process different symbols on different GPUs
with cp.cuda.Device(0):
    process_symbols(['AAPL', 'GOOGL', ...])

with cp.cuda.Device(1):
    process_symbols(['MSFT', 'TSLA', ...])
```

### Batch Processing Optimization

For processing multiple charts/symbols efficiently:

#### Strategy 1: Keep Data on GPU

```python
import cupy as cp
import polars as pl
from kimsfinance.ops.indicators import rsi, macd

# Load all data to GPU once
symbols = load_symbols()  # ['AAPL', 'GOOGL', ...]

gpu_data = {}
for symbol in symbols:
    df = pl.read_parquet(f'data/{symbol}.parquet')
    gpu_data[symbol] = {
        'close': cp.array(df['close'].to_numpy()),
        'high': cp.array(df['high'].to_numpy()),
        'low': cp.array(df['low'].to_numpy()),
    }

# Compute indicators on GPU (no transfers)
gpu_indicators = {}
for symbol, data in gpu_data.items():
    gpu_indicators[symbol] = {
        'rsi': rsi.calculate_rsi(data['close'], period=14, engine='gpu'),
        'macd': macd.calculate_macd(data['close'], engine='gpu'),
    }

# Transfer results to CPU only once
for symbol in symbols:
    results = {
        k: cp.asnumpy(v)
        for k, v in gpu_indicators[symbol].items()
    }
    save_indicators(symbol, results)

# Cleanup
cp.get_default_memory_pool().free_all_blocks()
```

**Performance gain**: 2-3x faster than per-symbol transfer.

#### Strategy 2: Parallel Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor
import cupy as cp

def process_symbol_gpu(symbol: str, gpu_id: int):
    """Process one symbol on specific GPU"""
    with cp.cuda.Device(gpu_id):
        df = load_data(symbol)
        # ... compute indicators ...
        return results

# Process symbols in parallel across GPUs
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i, symbol in enumerate(symbols):
        gpu_id = i % 2  # Alternate between 2 GPUs
        future = executor.submit(process_symbol_gpu, symbol, gpu_id)
        futures.append(future)

    results = [f.result() for f in futures]
```

**Performance gain**: Near-linear scaling with number of GPUs.

### Memory Management

#### Explicit Memory Control

```python
import cupy as cp

# Pre-allocate GPU memory pool (reduces fragmentation)
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

# Set memory limit (prevent OOM)
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=6 * 1024**3)  # 6 GB limit

# Process with explicit cleanup
try:
    gpu_array = cp.array(large_dataset)
    result = compute_indicator(gpu_array)
finally:
    del gpu_array
    mempool.free_all_blocks()
```

#### Streaming for Huge Datasets

For datasets too large for GPU memory:

```python
def process_large_dataset_streaming(data, chunk_size=1_000_000):
    """Process data in chunks to fit in GPU memory"""
    results = []

    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]

        # Process chunk on GPU
        gpu_chunk = cp.array(chunk)
        gpu_result = compute_indicator(gpu_chunk)

        # Transfer result and free GPU memory
        results.append(cp.asnumpy(gpu_result))
        del gpu_chunk, gpu_result
        cp.get_default_memory_pool().free_all_blocks()

    # Combine results
    return np.concatenate(results)
```

---

## Advanced Topics

### Custom CUDA Kernels

For maximum performance, write custom CUDA kernels with CuPy:

#### Example: Custom RSI Kernel

```python
import cupy as cp

# Raw CUDA kernel for RSI calculation
rsi_kernel = cp.RawKernel(r'''
extern "C" __global__
void rsi_kernel(
    const double* prices,
    double* rsi_out,
    int n,
    int period
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= n - period) return;

    // Calculate gains and losses
    double gain_sum = 0.0;
    double loss_sum = 0.0;

    for (int i = 0; i < period; i++) {
        double delta = prices[idx + i + 1] - prices[idx + i];
        if (delta > 0) {
            gain_sum += delta;
        } else {
            loss_sum += -delta;
        }
    }

    // Calculate RS and RSI
    double avg_gain = gain_sum / period;
    double avg_loss = loss_sum / period;
    double rs = (avg_loss > 0) ? avg_gain / avg_loss : 0.0;
    rsi_out[idx] = 100.0 - (100.0 / (1.0 + rs));
}
''', 'rsi_kernel')

def calculate_rsi_custom_kernel(prices: cp.ndarray, period: int = 14) -> cp.ndarray:
    """Calculate RSI using custom CUDA kernel"""
    n = len(prices)
    rsi_out = cp.zeros(n, dtype=cp.float64)

    # Launch kernel
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    rsi_kernel(
        (blocks,), (threads_per_block,),
        (prices, rsi_out, n, period)
    )

    return rsi_out

# Usage
prices_gpu = cp.array(prices, dtype=cp.float64)
rsi_gpu = calculate_rsi_custom_kernel(prices_gpu, period=14)
rsi_cpu = cp.asnumpy(rsi_gpu)
```

**Performance**: Custom kernels can be 2-5x faster than CuPy for complex logic.

#### Using Numba CUDA

Alternative approach with Numba CUDA (more Pythonic):

```python
from numba import cuda
import numpy as np

@cuda.jit
def rsi_numba_kernel(prices, rsi_out, period):
    """RSI calculation with Numba CUDA"""
    idx = cuda.grid(1)
    n = prices.shape[0]

    if idx >= n - period:
        return

    gain_sum = 0.0
    loss_sum = 0.0

    for i in range(period):
        delta = prices[idx + i + 1] - prices[idx + i]
        if delta > 0:
            gain_sum += delta
        else:
            loss_sum += -delta

    avg_gain = gain_sum / period
    avg_loss = loss_sum / period

    if avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi_out[idx] = 100.0 - (100.0 / (1.0 + rs))
    else:
        rsi_out[idx] = 0.0

def calculate_rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using Numba CUDA"""
    # Transfer to GPU
    prices_gpu = cuda.to_device(prices)
    rsi_gpu = cuda.device_array(len(prices), dtype=np.float64)

    # Launch kernel
    threads_per_block = 256
    blocks_per_grid = (len(prices) + threads_per_block - 1) // threads_per_block
    rsi_numba_kernel[blocks_per_grid, threads_per_block](prices_gpu, rsi_gpu, period)

    # Transfer back
    return rsi_gpu.copy_to_host()
```

**Pros of Numba**:
- Pure Python syntax
- Easier to write and debug
- Good for prototyping

**Pros of CuPy RawKernel**:
- Direct CUDA C++
- Maximum performance
- Access to all CUDA features

### GPU Profiling with Nsight

Profile GPU kernels to find bottlenecks:

#### Using Nsight Systems

```bash
# Profile entire application
nsys profile --stats=true python my_script.py

# Focus on CUDA operations
nsys profile --trace=cuda,cudnn,cublas python my_script.py

# Generate timeline visualization
nsys profile -o timeline python my_script.py
# Open timeline.qdrep in Nsight Systems GUI
```

**What to look for**:
- Kernel execution time
- CPU-GPU transfer overhead
- GPU utilization (should be >70%)
- Memory bandwidth utilization

#### Using Nsight Compute

For detailed kernel analysis:

```bash
# Profile specific kernel
ncu --kernel-name=rsi_kernel python my_script.py

# Full metrics
ncu --set full python my_script.py

# Interactive profiling
ncu --mode=launch-and-attach python my_script.py
```

**Key metrics**:
- **SM Efficiency**: Should be >60%
- **Memory Throughput**: Compare to theoretical max
- **Occupancy**: Should be >50%
- **Warp Efficiency**: Should be >80%

#### CuPy Profiling

Built-in profiling for CuPy code:

```python
import cupy as cp
from cupyx.profiler import benchmark

def compute_rsi(prices, period=14):
    # ... RSI implementation ...
    return rsi

# Benchmark function
execution_time = benchmark(
    compute_rsi,
    (prices_gpu, 14),
    n_repeat=100
)

print(f"Average time: {execution_time.gpu_times.mean():.3f} ms")
print(f"Std dev: {execution_time.gpu_times.std():.3f} ms")
```

### Debugging GPU Code

#### Common Issues and Solutions

**1. Out of Memory (OOM)**

```python
# Problem: Large dataset doesn't fit in GPU memory
try:
    gpu_data = cp.array(huge_dataset)  # OOM!
except cp.cuda.memory.OutOfMemoryError:
    # Solution 1: Process in chunks
    process_in_chunks(huge_dataset, chunk_size=1_000_000)

    # Solution 2: Use CPU
    process_on_cpu(huge_dataset)
```

**2. Slow Transfers**

```python
# Problem: Too many small transfers
for i in range(1000):
    data_gpu = cp.array(small_array)  # 1000 transfers!
    result = compute(data_gpu)
    results.append(cp.asnumpy(result))

# Solution: Batch transfers
data_gpu = cp.array(np.vstack(small_arrays))  # 1 transfer
results_gpu = compute(data_gpu)
results = cp.asnumpy(results_gpu)  # 1 transfer
```

**3. Unexpected CPU Fallback**

```python
from kimsfinance.core.engine import EngineManager

# Check why GPU not used
if not EngineManager.check_gpu_available():
    print("GPU not available - check installation")

# Force GPU and see error
try:
    result = compute_indicator(data, engine='gpu')
except Exception as e:
    print(f"GPU error: {e}")
```

**4. Incorrect Results**

```python
# Problem: Floating-point differences between CPU and GPU
cpu_result = np.array([1.0, 2.0, 3.0])
gpu_result = cp.asnumpy(cp.array([1.0, 2.0, 3.0]))

# Use approximate comparison
np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-6)

# For indicators, differences of 1e-8 are acceptable
assert np.abs(cpu_rsi - gpu_rsi).max() < 1e-8
```

#### Debugging Tools

**1. CUDA Memory Checker**

```bash
# Detect memory errors
compute-sanitizer --tool memcheck python my_script.py

# Check for race conditions
compute-sanitizer --tool racecheck python my_script.py
```

**2. CuPy Debugging Mode**

```python
import os
os.environ['CUPY_DUMP_CUDA_SOURCE_ON_ERROR'] = '1'

# Now CuPy will dump kernel source on error
```

**3. Verbose Error Messages**

```python
import cupy as cp
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

# Enable memory pool debugging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Multi-GPU Support (Future)

kimsfinance doesn't currently support multi-GPU out-of-the-box, but you can implement it:

```python
import cupy as cp
from concurrent.futures import ThreadPoolExecutor

def process_on_gpu(data, gpu_id):
    """Process data on specific GPU"""
    with cp.cuda.Device(gpu_id):
        # All CuPy operations use this GPU
        gpu_data = cp.array(data)
        result = compute_indicator(gpu_data)
        return cp.asnumpy(result)

def multi_gpu_batch_process(datasets, n_gpus=2):
    """Distribute work across multiple GPUs"""
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = []
        for i, data in enumerate(datasets):
            gpu_id = i % n_gpus
            future = executor.submit(process_on_gpu, data, gpu_id)
            futures.append(future)

        results = [f.result() for f in futures]

    return results

# Usage
results = multi_gpu_batch_process(all_symbols_data, n_gpus=2)
```

**Expected performance**: Near-linear scaling (1.8-1.9x with 2 GPUs).

---

## Troubleshooting

### Installation Issues

#### Issue: cuDF installation fails

**Symptoms**:
```bash
ERROR: Could not find a version that satisfies the requirement cudf-cu12
```

**Solutions**:

1. Check Python version (must be 3.9-3.12, **3.13 not yet supported by RAPIDS**):
   ```bash
   python --version  # Must be 3.9-3.12
   ```

   If using Python 3.13:
   ```bash
   # Downgrade to Python 3.12 for GPU support
   conda create -n kimsfinance python=3.12
   conda activate kimsfinance
   pip install kimsfinance[gpu]
   ```

2. Use NVIDIA PyPI index:
   ```bash
   pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12
   ```

3. Use conda (more reliable):
   ```bash
   conda install -c rapidsai -c conda-forge -c nvidia cudf=24.12 python=3.12
   ```

#### Issue: CuPy installation fails

**Symptoms**:
```bash
ImportError: DLL load failed while importing cupy_backends
```

**Solutions**:

1. Install correct CUDA version:
   ```bash
   # For CUDA 12.x
   pip install cupy-cuda12x

   # For CUDA 11.x
   pip install cupy-cuda11x
   ```

2. Check CUDA installation:
   ```bash
   nvidia-smi | grep "CUDA Version"
   ```

3. Reinstall with conda:
   ```bash
   conda install -c conda-forge cupy
   ```

#### Issue: "GPU not detected" after installation

**Symptoms**:
```python
from kimsfinance.core.engine import EngineManager
EngineManager.check_gpu_available()  # Returns False
```

**Solutions**:

1. Verify NVIDIA driver:
   ```bash
   nvidia-smi  # Should show your GPU
   ```

2. Test CuPy directly:
   ```python
   import cupy as cp
   x = cp.array([1, 2, 3])
   print(x)  # Should work without error
   ```

3. Check for version conflicts:
   ```bash
   pip list | grep -E "cudf|cupy|cuda"
   ```

### Runtime Issues

#### Issue: OutOfMemoryError

**Symptoms**:
```python
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating X bytes
```

**Solutions**:

1. **Reduce dataset size** (process in chunks):
   ```python
   from kimsfinance.ops.indicators import rsi

   # Instead of processing all at once
   chunk_size = 500_000
   results = []
   for i in range(0, len(data), chunk_size):
       chunk = data[i:i+chunk_size]
       result = rsi.calculate_rsi(chunk, engine='gpu')
       results.append(result)

   full_result = np.concatenate(results)
   ```

2. **Free GPU memory explicitly**:
   ```python
   import cupy as cp

   # After processing
   cp.get_default_memory_pool().free_all_blocks()
   ```

3. **Use CPU for this operation**:
   ```python
   # Force CPU
   result = compute_indicator(data, engine='cpu')
   ```

4. **Increase threshold** so GPU only used for larger datasets:
   ```python
   from kimsfinance.config import gpu_thresholds
   gpu_thresholds.GPU_THRESHOLDS["vectorizable_simple"] = 200_000
   ```

#### Issue: Slow performance (GPU slower than CPU)

**Symptoms**:
- GPU operations take longer than CPU

**Diagnosis**:
```python
import time
import numpy as np
import cupy as cp

# Benchmark
data_cpu = np.random.randn(10000)
data_gpu = cp.random.randn(10000)

# CPU
start = time.time()
result_cpu = np.mean(data_cpu)
cpu_time = time.time() - start

# GPU
cp.cuda.Stream.null.synchronize()
start = time.time()
result_gpu = cp.mean(data_gpu)
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start

print(f"CPU: {cpu_time*1000:.2f}ms, GPU: {gpu_time*1000:.2f}ms")
```

**Solutions**:

1. **Dataset too small** - GPU overhead dominates:
   ```python
   # Increase threshold or use CPU explicitly
   result = compute_indicator(data, engine='cpu')
   ```

2. **Too many transfers**:
   ```python
   # Bad: Multiple transfers
   for symbol in symbols:
       data_gpu = cp.array(load_data(symbol))  # Transfer
       result = compute(data_gpu)
       results[symbol] = cp.asnumpy(result)  # Transfer

   # Good: Batch transfers
   all_data_gpu = {s: cp.array(load_data(s)) for s in symbols}
   all_results_gpu = {s: compute(d) for s, d in all_data_gpu.items()}
   results = {s: cp.asnumpy(r) for s, r in all_results_gpu.items()}
   ```

3. **GPU not fully utilized**:
   ```bash
   # Monitor GPU utilization while running
   watch -n 0.5 nvidia-smi

   # Should show 70-100% GPU utilization
   # If low (<30%), operation may not be GPU-friendly
   ```

#### Issue: Incorrect results from GPU

**Symptoms**:
- GPU results differ from CPU results beyond floating-point tolerance

**Diagnosis**:
```python
import numpy as np

cpu_result = compute_on_cpu(data)
gpu_result = compute_on_gpu(data)

diff = np.abs(cpu_result - gpu_result)
print(f"Max difference: {diff.max()}")
print(f"Mean difference: {diff.mean()}")

# Check for NaN or Inf
print(f"CPU NaN: {np.isnan(cpu_result).sum()}")
print(f"GPU NaN: {np.isnan(gpu_result).sum()}")
```

**Solutions**:

1. **Floating-point precision** (expected differences):
   ```python
   # Acceptable: differences < 1e-8
   np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-6, atol=1e-8)
   ```

2. **Algorithm order matters**:
   ```python
   # Bad: Non-associative operations may give different results
   # Sum in different order can accumulate different rounding errors

   # Use compensated summation (Kahan algorithm) if needed
   ```

3. **Check for NaN propagation**:
   ```python
   # GPU may handle NaN differently
   # Use explicit NaN handling
   data[np.isnan(data)] = 0.0
   ```

### Performance Issues

#### GPU utilization low (<30%)

**Diagnosis**:
```bash
nvidia-smi dmon -s u -c 10
# Monitor utilization for 10 seconds
```

**Solutions**:

1. **Increase batch size** (process more data at once)
2. **Reduce kernel launch overhead** (fewer, larger operations)
3. **Check if operation is CPU-bound** (use profiler)

#### High memory usage

**Diagnosis**:
```bash
nvidia-smi | grep "MiB"
# Check memory usage
```

**Solutions**:

1. **Use streaming/chunking** for large datasets
2. **Free memory explicitly** after operations
3. **Reduce precision** if acceptable (float32 instead of float64)

---

## References

### Documentation

- [RAPIDS cuDF Documentation](https://docs.rapids.ai/api/cudf/stable/)
- [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Numba CUDA Documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)

### Benchmarking Scripts

- `scripts/gpu_validation_test.py` - GPU setup validation
- `scripts/gpu_advanced_test.py` - Advanced GPU benchmarks
- `tests/test_engine_selection.py` - Engine selection tests
- `benchmarks/compare_gpu_cpu.py` - CPU vs GPU comparison

### Related Guides

- [Performance Guide](PERFORMANCE.md) - General performance optimization
- [API Reference](API.md) - Complete API documentation
- [README GPU Section](../README.md#-gpu-acceleration-optional) - Quick GPU overview

### Hardware References

**GPU Compute Capability**:
- 7.0: Volta (V100)
- 7.5: Turing (RTX 20 series, GTX 16 series)
- 8.0: Ampere (A100)
- 8.6: Ampere (RTX 30 series)
- 8.9: Ada Lovelace (RTX 40 series)
- 9.0: Hopper (H100)

**CUDA Versions**:
- CUDA 11.8: Widely supported, stable
- CUDA 12.0+: Latest features, better performance

**cuDF Compatibility**:
- cuDF 24.12: Python 3.9-3.12, CUDA 11.8+
- cuDF 24.10: Python 3.9-3.11, CUDA 11.4+

---

**Document Version**: 1.0.0
**Last Updated**: 2025-10-22
**Tested On**: NVIDIA RTX 3500 Ada, Ubuntu 22.04, CUDA 12.6
**Author**: kimsfinance Team

For issues or questions, see [GitHub Issues](https://github.com/kimasplund/kimsfinance/issues).
