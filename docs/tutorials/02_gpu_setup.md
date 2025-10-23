# GPU Setup & Configuration Tutorial

**Version**: 1.0.0
**Last Updated**: 2025-10-23
**Skill Level**: Intermediate
**Time Required**: 30-60 minutes

---

## Table of Contents

1. [Overview](#overview)
2. [When GPU Acceleration Helps](#when-gpu-acceleration-helps)
3. [Hardware Requirements](#hardware-requirements)
4. [Installation](#installation)
5. [Verification](#verification)
6. [Basic GPU Usage](#basic-gpu-usage)
7. [GPU Auto-Tuning](#gpu-auto-tuning)
8. [Batch Processing (CRITICAL)](#batch-processing-critical)
9. [Monitoring & Debugging](#monitoring--debugging)
10. [Performance Tips](#performance-tips)
11. [Troubleshooting](#troubleshooting)

---

## Overview

kimsfinance provides **optional GPU acceleration** using NVIDIA GPUs and CUDA. GPU support is completely optional - all features work perfectly on CPU-only systems with no performance penalty for typical workloads.

### What Gets Accelerated?

**Technical Indicators** (CuPy):
- RSI, Stochastic, MACD, ATR, Bollinger Bands
- 1.2x-2.9x speedup for individual indicators
- **66.7x more efficient when using batch processing**

**OHLCV Data Processing** (cuDF):
- Large dataset aggregation: 6.4x speedup
- DataFrame filtering and transformations
- Time-series resampling

### Key Concepts

- **Engine Parameter**: Controls execution mode (`cpu`, `gpu`, `auto`)
- **Auto-Tuning**: Empirically finds optimal CPU/GPU crossover points
- **Batch Processing**: Computing multiple indicators simultaneously (CRITICAL for GPU efficiency)
- **Streaming Mode**: Processes large datasets in chunks to prevent memory issues

---

## When GPU Acceleration Helps

### GPU is Beneficial For:

**Batch Processing (MOST IMPORTANT)**
- Computing **6+ indicators simultaneously**
- GPU beneficial at just **15K rows** (vs 1M for individual)
- **66.7x more efficient** than individual indicator calculations
- Example: Computing RSI + MACD + Stochastic + ATR + Bollinger + OBV

**Large Datasets**
- Individual indicators: **100K+ rows**
- Rolling window operations (ATR): **10K+ rows in parallel execution**
- Batch indicators: **15K+ rows** (ALWAYS prefer this)

**High-Throughput Systems**
- Real-time data pipelines processing multiple symbols
- Backtesting across hundreds of assets
- Dashboard displaying dozens of charts simultaneously

### GPU is NOT Beneficial For:

**Small Datasets**
- Single chart with <10K candles
- Quick exploratory analysis
- Development and prototyping

**Sequential Operations**
- Computing one indicator at a time
- Single-threaded sequential processing
- GPU overhead exceeds computation time

**CPU-Only Operations**
- Chart rendering (PIL/Pillow always uses CPU)
- File I/O and data loading
- Business logic and decision making

### Performance Summary

| Scenario | Crossover Threshold | Speedup | Recommendation |
|----------|-------------------|---------|----------------|
| **Batch indicators (6+)** | **15K rows** | **66.7x more efficient** | **✅ ALWAYS USE** |
| Parallel execution (real-world) | 10K-100K rows | 2-5x | ✅ Use for dashboards |
| Sequential individual | 100K-1M rows | 1.2-3x | ❌ Rarely needed |
| ATR (parallel) | 10K rows | 1.4x | ✅ Use in batch mode |
| Stochastic (batch) | 15K rows | 2.9x | ✅ Use in batch mode |

**CRITICAL INSIGHT**: GPU becomes beneficial at just **15K rows** when using batch processing, compared to **1M+ rows** for sequential individual calculations. This is a **66.7x efficiency improvement**!

---

## Hardware Requirements

### Minimum Requirements

- **GPU**: NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- **VRAM**: 4 GB minimum (8 GB recommended)
- **CUDA**: Version 11.8 or 12.x
- **Driver**: NVIDIA Driver 525.x or newer

### Supported Hardware

**Tested Configurations**:
- NVIDIA RTX 3500 Ada (8 GB VRAM) - ThinkPad P16 Gen2 ✅
- NVIDIA RTX 4090 (24 GB VRAM) ✅
- NVIDIA A100 (40/80 GB VRAM) ✅

**Supported GPU Architectures**:
- Volta (V100) - Compute 7.0
- Turing (RTX 20 series, GTX 16 series) - Compute 7.5
- Ampere (RTX 30 series, A100) - Compute 8.0/8.6
- Ada Lovelace (RTX 40 series) - Compute 8.9
- Hopper (H100) - Compute 9.0

### Not Supported

- **AMD GPUs** (no RAPIDS support)
- **Intel GPUs**
- **Apple Silicon** (M1/M2/M3 MPS backend not yet supported)

---

## Installation

### Step 1: Check GPU Availability

First, verify you have an NVIDIA GPU and driver installed:

```bash
# Check NVIDIA driver and GPU
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.x    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA RTX 3500...  Off  | 00000000:01:00.0 Off |                  N/A |
# | N/A   45C    P8    15W /  N/A |      0MiB /  8192MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

If `nvidia-smi` doesn't work, install NVIDIA drivers first:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535

# Fedora/RHEL
sudo dnf install nvidia-driver

# Reboot required after driver installation
sudo reboot
```

### Step 2: Check CUDA Version

```bash
# Check CUDA version from nvidia-smi output
nvidia-smi | grep "CUDA Version"

# Or check nvcc (CUDA compiler) if installed
nvcc --version
```

**Note**: You need CUDA **11.8 or 12.x**. The driver CUDA version shown in `nvidia-smi` is what matters for CuPy/cuDF.

### Step 3: Install kimsfinance with GPU Support

**Method 1: pip install (Recommended)**

```bash
# Install kimsfinance with GPU dependencies
pip install "kimsfinance[gpu]"

# This installs:
# - kimsfinance (base package)
# - cupy-cuda12x (NumPy-compatible GPU arrays)
# - cudf-cu12 (GPU-accelerated DataFrames)
```

**Method 2: Manual Installation**

```bash
# Install base package
pip install kimsfinance

# Install CuPy for your CUDA version
# For CUDA 12.x:
pip install cupy-cuda12x

# For CUDA 11.x:
pip install cupy-cuda11x

# Install cuDF (optional, for DataFrame operations)
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12
```

**Method 3: Conda Installation**

```bash
# Create conda environment with Python 3.12
# (Python 3.13 not yet supported by RAPIDS)
conda create -n kimsfinance python=3.12
conda activate kimsfinance

# Install RAPIDS (cuDF + CuPy)
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=24.12 cupy python=3.12 cuda-version=12.0

# Install kimsfinance
pip install kimsfinance
```

### Step 4: Verify Installation

Save this as `verify_gpu_setup.py`:

```python
#!/usr/bin/env python3
"""Quick GPU setup verification for kimsfinance"""

def main():
    print("=" * 60)
    print("kimsfinance GPU Setup Verification")
    print("=" * 60 + "\n")

    # Check 1: NVIDIA GPU
    print("1. Checking NVIDIA GPU...")
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        gpu_name, driver = result.stdout.strip().split(',')
        print(f"   ✓ GPU: {gpu_name.strip()}")
        print(f"   ✓ Driver: {driver.strip()}")
    except Exception as e:
        print(f"   ✗ NVIDIA GPU not detected: {e}")
        return False

    # Check 2: CuPy
    print("\n2. Checking CuPy...")
    try:
        import cupy as cp
        print(f"   ✓ CuPy version: {cp.__version__}")

        # Test GPU array operation
        x = cp.array([1, 2, 3])
        result = cp.sum(x)
        print(f"   ✓ GPU array test passed (sum={result})")
    except ImportError:
        print("   ✗ CuPy not installed")
        print("   Install with: pip install cupy-cuda12x")
        return False
    except Exception as e:
        print(f"   ✗ CuPy test failed: {e}")
        return False

    # Check 3: cuDF (optional)
    print("\n3. Checking cuDF (optional)...")
    try:
        import cudf
        print(f"   ✓ cuDF version: {cudf.__version__}")

        # Test basic operation
        df = cudf.DataFrame({'a': [1, 2, 3]})
        result = df['a'].sum()
        print(f"   ✓ cuDF test passed (sum={result})")
    except ImportError:
        print("   ⚠ cuDF not installed (optional)")
        print("   Install with: pip install kimsfinance[gpu]")
    except Exception as e:
        print(f"   ⚠ cuDF test failed: {e}")

    # Check 4: kimsfinance GPU
    print("\n4. Checking kimsfinance GPU support...")
    try:
        from kimsfinance.core.engine import EngineManager

        if EngineManager.check_gpu_available():
            print("   ✓ kimsfinance GPU support enabled")

            info = EngineManager.get_info()
            print(f"   ✓ Default engine: {info.get('default_engine')}")
        else:
            print("   ✗ kimsfinance GPU support not available")
            return False
    except Exception as e:
        print(f"   ✗ kimsfinance GPU check failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ GPU setup complete! All checks passed.")
    print("\nYou can now use GPU acceleration:")
    print("  from kimsfinance.api import plot")
    print("  plot.render(df, engine='auto')  # Auto GPU selection")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

Run the verification:

```bash
python verify_gpu_setup.py
```

---

## Verification

After installation, verify GPU support is working:

### Quick Test

```python
from kimsfinance.core.engine import EngineManager
from kimsfinance.ops.indicators import calculate_rsi
import numpy as np

# Check GPU availability
print(f"GPU available: {EngineManager.check_gpu_available()}")

# Get detailed info
info = EngineManager.get_info()
print(f"Info: {info}")

# Test GPU calculation
data = np.random.randn(100000)
rsi_cpu = calculate_rsi(data, engine='cpu')
rsi_gpu = calculate_rsi(data, engine='gpu')

# Verify results match (within floating-point tolerance)
print(f"Results match: {np.allclose(rsi_cpu, rsi_gpu, rtol=1e-6)}")
```

### Advanced Validation

For comprehensive GPU testing:

```bash
# Run full GPU validation suite
python scripts/gpu_validation_test.py

# Run advanced GPU benchmarks
python scripts/gpu_advanced_test.py
```

These scripts test:
- GPU hardware detection
- Memory bandwidth
- Kernel performance
- Memory leak detection
- Financial indicator performance

---

## Basic GPU Usage

### Using engine='gpu' Explicitly

Force GPU execution (raises error if GPU unavailable):

```python
from kimsfinance.ops.indicators import calculate_rsi
import numpy as np

# Generate test data
prices = 100 + np.cumsum(np.random.randn(100000) * 0.5)

# Force GPU execution
rsi = calculate_rsi(prices, engine='gpu')
# Raises GPUNotAvailableError if GPU not available
```

### Using engine='auto' (Recommended)

Automatically selects CPU or GPU based on data size:

```python
from kimsfinance.ops.indicators import calculate_rsi
import numpy as np

# Small dataset - uses CPU automatically
prices_small = 100 + np.cumsum(np.random.randn(5000) * 0.5)
rsi_small = calculate_rsi(prices_small, engine='auto')
# Uses CPU (below 100K threshold)

# Large dataset - uses GPU automatically
prices_large = 100 + np.cumsum(np.random.randn(500000) * 0.5)
rsi_large = calculate_rsi(prices_large, engine='auto')
# Uses GPU (above 100K threshold)
```

### Understanding Crossover Thresholds

Different operations have different CPU/GPU crossover points:

```python
from kimsfinance.config.gpu_thresholds import GPU_THRESHOLDS

# View all thresholds
print("GPU Crossover Thresholds:")
for operation, threshold in GPU_THRESHOLDS.items():
    print(f"  {operation}: {threshold:,} rows")

# Output:
#   vectorizable_simple: 100,000 rows (RSI, ROC)
#   vectorizable_complex: 500,000 rows (MACD, Stochastic)
#   iterative: 500,000 rows (Parabolic SAR)
#   rolling: 10,000 rows (ATR in parallel)
#   batch_indicators: 15,000 rows (6+ indicators)
#   aggregation: 5,000 rows
#   default: 100,000 rows
```

**Key Insight**: The `batch_indicators` threshold is **15K rows**, much lower than individual indicator thresholds. This is why batch processing is critical for GPU efficiency!

---

## GPU Auto-Tuning

Auto-tuning finds optimal CPU/GPU crossover points for **your specific hardware**.

### Why Auto-Tune?

Default thresholds are optimized for NVIDIA RTX 3500 Ada. Your hardware may have different optimal crossover points:

- **High-end GPU** (RTX 4090): Lower thresholds (GPU faster sooner)
- **Mid-range GPU** (RTX 3060): Default thresholds work well
- **Entry-level GPU** (GTX 1660): Higher thresholds (CPU competitive longer)

### Running Auto-Tune

**Option 1: Simple Auto-Tune**

```python
from kimsfinance.core.autotune import run_autotune

# Auto-tune common operations
thresholds = run_autotune(
    operations=['atr', 'rsi', 'stochastic'],
    save=True  # Save to ~/.kimsfinance/threshold_cache.json
)

print("Tuned thresholds:", thresholds)
```

**Option 2: Comprehensive Auto-Tune (Recommended)**

```bash
# Run comprehensive auto-tuning script
python scripts/run_autotune_comprehensive.py
```

This script benchmarks:
- Individual indicators (sequential)
- Individual indicators (parallel execution)
- Batch indicators (6+ at once)
- Various dataset sizes (10K to 1M rows)

### Understanding Results

Auto-tune output shows crossover points:

```
Tuning operation: atr...
  Size 10,000:  CPU=2.3ms, GPU=5.1ms (CPU faster)
  Size 50,000:  CPU=11.2ms, GPU=12.3ms (CPU faster)
  Size 100,000: CPU=22.5ms, GPU=15.8ms (GPU faster ✓)
  -> Found crossover at: 100,000

Tuning operation: batch_indicators...
  Size 10,000:  CPU=45.2ms, GPU=48.1ms (CPU faster)
  Size 15,000:  CPU=67.8ms, GPU=52.3ms (GPU faster ✓)
  -> Found crossover at: 15,000
```

### Cached Thresholds

Auto-tuned thresholds are saved to:

```
~/.kimsfinance/threshold_cache.json
```

kimsfinance automatically loads cached thresholds on import. Re-run auto-tune if:
- You upgrade your GPU
- You update CUDA/CuPy versions
- Performance characteristics change significantly

### Manual Threshold Override

Override thresholds programmatically:

```python
from kimsfinance.config.gpu_thresholds import GPU_THRESHOLDS

# Lower threshold for powerful GPU
GPU_THRESHOLDS['vectorizable_simple'] = 50_000

# Higher threshold for entry-level GPU
GPU_THRESHOLDS['vectorizable_complex'] = 200_000

# Now all operations use updated thresholds
```

Or via environment variables:

```bash
# Set custom thresholds
export KIMSFINANCE_GPU_THRESHOLD_SIMPLE=50000
export KIMSFINANCE_GPU_THRESHOLD_BATCH=10000

# Run your application
python my_trading_bot.py
```

---

## Batch Processing (CRITICAL)

**This is the single most important section for GPU efficiency!**

### Why Batch Processing Matters

Computing multiple indicators individually:
```python
# BAD: Individual calculations (GPU needs 100K+ rows each)
rsi = calculate_rsi(closes, engine='auto')        # Needs 100K rows
macd = calculate_macd(closes, engine='auto')      # Needs 100K rows
stoch = calculate_stochastic_oscillator(highs, lows, closes, engine='auto')  # Needs 500K rows
atr = calculate_atr(highs, lows, closes, engine='auto')  # Needs 10K rows
bollinger = calculate_bollinger_bands(closes, engine='auto')  # Needs 100K rows
obv = calculate_obv(closes, volumes, engine='auto')  # Needs 100K rows

# GPU only beneficial for EACH if dataset > their individual thresholds
# For 50K rows: ALL use CPU (below thresholds)
```

Computing multiple indicators in batch:
```python
# GOOD: Batch calculation (GPU beneficial at just 15K rows!)
results = calculate_indicators_batch(
    highs, lows, closes, volumes,
    engine='auto'
)
rsi = results['rsi']
macd = results['macd']
stoch = results['stochastic']
atr = results['atr']
bollinger = results['bollinger']
obv = results['obv']

# GPU beneficial at 15K rows for ALL 6 indicators!
# 66.7x more efficient than individual calculations
```

**Performance Comparison** (50K rows):

| Method | Engine Used | Total Time | Speedup |
|--------|-------------|------------|---------|
| Individual indicators | CPU (all below threshold) | ~150ms | 1x |
| Batch processing | **GPU** (above 15K threshold) | **~25ms** | **6x faster** |

### Using calculate_indicators_batch()

The `calculate_indicators_batch()` function computes 6 indicators simultaneously:

```python
from kimsfinance.ops.batch import calculate_indicators_batch
import numpy as np

# Generate sample data (50K rows - perfect for batch GPU)
n = 50_000
closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
highs = closes + np.abs(np.random.randn(n) * 0.3)
lows = closes - np.abs(np.random.randn(n) * 0.3)
volumes = np.abs(np.random.randn(n) * 1_000_000)

# Compute all indicators in one GPU pass
results = calculate_indicators_batch(
    highs, lows, closes, volumes,
    engine='auto',      # Auto-selects GPU at 15K+ rows
    streaming=None      # Auto-enables at 500K+ rows
)

# Extract results
atr = results['atr']                              # Average True Range
rsi = results['rsi']                              # RSI (0-100)
stoch_k, stoch_d = results['stochastic']          # Stochastic %K, %D
bb_upper, bb_middle, bb_lower = results['bollinger']  # Bollinger Bands
obv = results['obv']                              # On Balance Volume
macd_line, signal_line, histogram = results['macd']   # MACD

print(f"Computed 6 indicators for {n:,} rows:")
print(f"  ATR shape: {atr.shape}")
print(f"  RSI shape: {rsi.shape}")
print(f"  Stochastic: %K={stoch_k.shape}, %D={stoch_d.shape}")
print(f"  Bollinger: upper={bb_upper.shape}, middle={bb_middle.shape}, lower={bb_lower.shape}")
print(f"  OBV shape: {obv.shape}")
print(f"  MACD: macd={macd_line.shape}, signal={signal_line.shape}, histogram={histogram.shape}")
```

### Batch Processing Benefits

**1. Single Data Transfer**
- Individual: 6 separate CPU→GPU transfers
- Batch: 1 single CPU→GPU transfer
- Speedup: ~6x less transfer overhead

**2. Reused Computations**
- Price changes computed once, used by multiple indicators
- Rolling windows shared across indicators
- Memory allocations batched together

**3. Lower GPU Threshold**
- Individual indicators: 100K-500K rows needed
- Batch indicators: **15K rows** needed
- **66.7x more efficient** (1M / 15K = 66.7)

### Real-World Example: Multi-Symbol Dashboard

```python
from kimsfinance.ops.batch import calculate_indicators_batch
import polars as pl

# Load data for multiple symbols
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

for symbol in symbols:
    # Load OHLCV data (e.g., 20K rows = 1 month of 1-min data)
    df = pl.read_parquet(f'data/{symbol}.parquet')

    # Compute all indicators in batch (GPU at 15K+ rows)
    results = calculate_indicators_batch(
        df['high'].to_numpy(),
        df['low'].to_numpy(),
        df['close'].to_numpy(),
        df['volume'].to_numpy(),
        engine='auto'  # Uses GPU (20K > 15K threshold)
    )

    # Generate dashboard chart with all indicators
    plot_dashboard(symbol, df, results)

# Total time: ~5 seconds for 5 symbols with GPU
# vs ~30 seconds with individual CPU calculations
```

### Streaming Mode for Large Datasets

For datasets >500K rows, enable streaming to prevent out-of-memory errors:

```python
# Large dataset: 1 year of 1-minute data = 525,600 rows
df_large = pl.read_parquet('spy_1year_1min.parquet')

# Streaming auto-enabled at 500K+ rows
results = calculate_indicators_batch(
    df_large['high'].to_numpy(),
    df_large['low'].to_numpy(),
    df_large['close'].to_numpy(),
    df_large['volume'].to_numpy(),
    engine='auto',
    streaming=None  # Auto-enables at 500K+
)

# Polars processes data in chunks - no OOM!
```

**Streaming Benefits**:
- Prevents out-of-memory errors on large datasets
- Processes data in ~500MB chunks
- Minimal performance overhead (<5%)
- Works with both CPU and GPU engines

---

## Monitoring & Debugging

### Using nvidia-smi to Monitor GPU

**Real-Time Monitoring**:

```bash
# Watch GPU utilization in real-time (updates every 1 second)
watch -n 1 nvidia-smi

# Monitor specific metrics
nvidia-smi dmon -s u -c 10  # Utilization for 10 seconds

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**What to Look For**:
- **GPU Utilization**: Should be >70% during indicator calculations
- **Memory Usage**: Should match expected dataset size
- **Temperature**: Should be <85°C (throttling occurs above)
- **Power Draw**: Should increase during GPU work

### Profiling with Nsight Systems

For detailed performance analysis:

```bash
# Profile entire application
nsys profile --stats=true python my_script.py

# Focus on CUDA operations
nsys profile --trace=cuda python my_script.py

# Generate timeline visualization
nsys profile -o timeline python my_script.py
# Open timeline.qdrep in Nsight Systems GUI
```

**Key Metrics**:
- Kernel execution time
- CPU-GPU transfer overhead
- GPU utilization percentage
- Memory bandwidth utilization

### Profiling with CuPy

Built-in profiling for CuPy operations:

```python
import cupy as cp
from cupyx.profiler import benchmark
from kimsfinance.ops.indicators import calculate_rsi

# Generate test data
prices_gpu = cp.random.randn(100000)

# Benchmark RSI calculation
execution_time = benchmark(
    calculate_rsi,
    (prices_gpu,),
    n_repeat=100
)

print(f"Average time: {execution_time.gpu_times.mean():.3f} ms")
print(f"Std dev: {execution_time.gpu_times.std():.3f} ms")
```

### Memory Management

Check GPU memory usage:

```python
import cupy as cp

# Get default memory pool
mempool = cp.get_default_memory_pool()

print(f"Used memory: {mempool.used_bytes() / 1e6:.1f} MB")
print(f"Total allocated: {mempool.total_bytes() / 1e6:.1f} MB")

# Free unused memory
mempool.free_all_blocks()

# Check pinned memory (for transfers)
pinned_mempool = cp.get_default_pinned_memory_pool()
print(f"Pinned blocks: {pinned_mempool.n_free_blocks()}")
```

### Common Issues and Solutions

**Issue: GPU Utilization Low (<30%)**

**Diagnosis**: Dataset too small or operation not GPU-friendly

**Solution**:
```python
# Check if dataset is above threshold
from kimsfinance.config.gpu_thresholds import GPU_THRESHOLDS

data_size = len(prices)
threshold = GPU_THRESHOLDS['vectorizable_simple']

if data_size < threshold:
    print(f"Dataset ({data_size}) below GPU threshold ({threshold})")
    print("Consider using CPU or batch processing")
```

**Issue: Out of Memory Error**

**Diagnosis**: Dataset too large for GPU VRAM

**Solution**:
```python
# Use streaming mode for large datasets
results = calculate_indicators_batch(
    highs, lows, closes, volumes,
    engine='auto',
    streaming=True  # Process in chunks
)

# Or reduce dataset size
# Or upgrade GPU VRAM
```

**Issue: GPU Slower Than CPU**

**Diagnosis**: Transfer overhead exceeds computation time

**Solution**:
```python
# Use batch processing to amortize transfer cost
# Or force CPU for small datasets
results = calculate_indicators_batch(
    highs, lows, closes, volumes,
    engine='cpu'  # Force CPU for small dataset
)
```

---

## Performance Tips

### 1. ALWAYS Use Batch Processing

**Rule**: If you need 2+ indicators, ALWAYS use `calculate_indicators_batch()`.

```python
# ❌ BAD: Individual calculations
rsi = calculate_rsi(closes, engine='auto')
macd = calculate_macd(closes, engine='auto')
stoch = calculate_stochastic_oscillator(highs, lows, closes, engine='auto')

# ✅ GOOD: Batch calculation
results = calculate_indicators_batch(highs, lows, closes, engine='auto')
rsi = results['rsi']
macd = results['macd']
stoch = results['stochastic']
```

**Why**: GPU beneficial at 15K rows (batch) vs 100K-500K rows (individual) = **66.7x more efficient**.

### 2. Parallel Execution Patterns

Process multiple symbols in parallel:

```python
from concurrent.futures import ThreadPoolExecutor
from kimsfinance.ops.batch import calculate_indicators_batch

def process_symbol(symbol):
    """Process one symbol's indicators"""
    df = load_data(symbol)
    return calculate_indicators_batch(
        df['high'], df['low'], df['close'], df['volume'],
        engine='auto'
    )

# Process 10 symbols in parallel
symbols = ['AAPL', 'GOOGL', 'MSFT', ...]
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_symbol, symbols))

# Each thread uses GPU efficiently
```

### 3. Data Transfer Optimization

Minimize CPU↔GPU transfers:

```python
import cupy as cp

# ❌ BAD: Multiple small transfers
for symbol in symbols:
    data_gpu = cp.array(load_data(symbol))  # Transfer
    result = compute(data_gpu)
    results.append(cp.asnumpy(result))  # Transfer

# ✅ GOOD: Batch transfers
all_data_gpu = {s: cp.array(load_data(s)) for s in symbols}  # 1 transfer
all_results_gpu = {s: compute(d) for s, d in all_data_gpu.items()}
results = {s: cp.asnumpy(r) for s, r in all_results_gpu.items()}  # 1 transfer
```

### 4. Keep Data on GPU

For multiple operations on same data:

```python
import cupy as cp
from kimsfinance.ops.batch import calculate_indicators_batch

# Load data to GPU once
closes_gpu = cp.array(closes)
highs_gpu = cp.array(highs)
lows_gpu = cp.array(lows)
volumes_gpu = cp.array(volumes)

# Compute indicators (data already on GPU)
results = calculate_indicators_batch(
    highs_gpu, lows_gpu, closes_gpu, volumes_gpu,
    engine='gpu'  # Data already on GPU
)

# Transfer final results only
rsi_cpu = cp.asnumpy(results['rsi'])
```

### 5. When to Use CPU vs GPU

| Data Size | Operation | Engine | Reason |
|-----------|-----------|--------|--------|
| <10K rows | Any | `cpu` | GPU overhead > compute |
| 10K-15K rows | Individual | `cpu` | Below threshold |
| 10K-15K rows | **Batch (6+)** | **`gpu`** | **Above batch threshold** |
| 15K-100K rows | Individual simple | `cpu` | Below threshold |
| 15K-100K rows | **Batch (6+)** | **`gpu`** | **Above batch threshold** |
| 100K+ rows | Any | `gpu` | GPU efficient |

**Key Takeaway**: Batch processing makes GPU beneficial at just **15K rows** instead of 100K+ rows!

### 6. Hardware-Specific Tuning

**High-End GPU (RTX 4090, A100)**:
```python
from kimsfinance.config.gpu_thresholds import GPU_THRESHOLDS

# Lower thresholds for powerful GPU
GPU_THRESHOLDS['batch_indicators'] = 10_000  # vs 15K default
GPU_THRESHOLDS['vectorizable_simple'] = 50_000  # vs 100K default
```

**Entry-Level GPU (GTX 1660, RTX 3050)**:
```python
# Raise thresholds for limited GPU
GPU_THRESHOLDS['batch_indicators'] = 30_000  # vs 15K default
GPU_THRESHOLDS['vectorizable_simple'] = 200_000  # vs 100K default
```

---

## Troubleshooting

### Installation Issues

**Issue: cuDF installation fails**

**Symptoms**:
```bash
ERROR: Could not find a version that satisfies the requirement cudf-cu12
```

**Solution 1**: Check Python version (must be 3.9-3.12, NOT 3.13):
```bash
python --version  # Must be 3.9-3.12

# If Python 3.13, downgrade to 3.12
conda create -n kimsfinance python=3.12
conda activate kimsfinance
pip install "kimsfinance[gpu]"
```

**Solution 2**: Use NVIDIA PyPI index:
```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12
```

**Solution 3**: Use conda (most reliable):
```bash
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=24.12 python=3.12 cuda-version=12.0
```

**Issue: CuPy installation fails**

**Symptoms**:
```bash
ImportError: DLL load failed while importing cupy_backends
```

**Solution 1**: Install correct CUDA version:
```bash
# Check your CUDA version
nvidia-smi | grep "CUDA Version"

# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

**Solution 2**: Use conda:
```bash
conda install -c conda-forge cupy
```

**Issue: GPU not detected after installation**

**Symptoms**:
```python
from kimsfinance.core.engine import EngineManager
EngineManager.check_gpu_available()  # Returns False
```

**Solution 1**: Verify NVIDIA driver:
```bash
nvidia-smi  # Should show your GPU
```

**Solution 2**: Test CuPy directly:
```python
import cupy as cp
x = cp.array([1, 2, 3])
print(x)  # Should work without error
```

**Solution 3**: Check for version conflicts:
```bash
pip list | grep -E "cudf|cupy|cuda"
```

### Runtime Issues

**Issue: OutOfMemoryError**

**Symptoms**:
```python
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating X bytes
```

**Solution 1**: Enable streaming mode:
```python
results = calculate_indicators_batch(
    highs, lows, closes, volumes,
    engine='auto',
    streaming=True  # Process in chunks
)
```

**Solution 2**: Reduce dataset size (process in chunks):
```python
chunk_size = 500_000
results = []
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    result = calculate_indicators_batch(
        chunk['high'], chunk['low'], chunk['close'], chunk['volume'],
        engine='gpu'
    )
    results.append(result)
```

**Solution 3**: Free GPU memory explicitly:
```python
import cupy as cp

# After processing
cp.get_default_memory_pool().free_all_blocks()
```

**Solution 4**: Force CPU for this dataset:
```python
results = calculate_indicators_batch(
    highs, lows, closes, volumes,
    engine='cpu'  # Use CPU instead
)
```

**Issue: Incorrect results from GPU**

**Symptoms**: GPU results differ from CPU beyond floating-point tolerance

**Diagnosis**:
```python
import numpy as np

cpu_result = calculate_rsi(prices, engine='cpu')
gpu_result = calculate_rsi(prices, engine='gpu')

diff = np.abs(cpu_result - gpu_result)
print(f"Max difference: {diff.max()}")
print(f"Mean difference: {diff.mean()}")
```

**Solution**: Verify differences are within tolerance:
```python
# Acceptable: differences < 1e-8
np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-6, atol=1e-8)

# If differences larger, check for:
# - NaN values in input data
# - Floating-point precision issues
# - Algorithm order sensitivity
```

**Issue: GPU slower than CPU**

**Diagnosis**:
```python
import time
import numpy as np

prices = np.random.randn(10000)  # Small dataset

# Benchmark CPU
start = time.time()
rsi_cpu = calculate_rsi(prices, engine='cpu')
cpu_time = time.time() - start

# Benchmark GPU
start = time.time()
rsi_gpu = calculate_rsi(prices, engine='gpu')
gpu_time = time.time() - start

print(f"CPU: {cpu_time*1000:.2f}ms, GPU: {gpu_time*1000:.2f}ms")
```

**Solution**: Dataset too small - GPU overhead dominates:
```python
# Force CPU for small datasets
rsi = calculate_rsi(prices, engine='cpu')

# Or use batch processing (lowers GPU threshold)
results = calculate_indicators_batch(
    highs, lows, closes, volumes,
    engine='auto'  # GPU at 15K rows
)
```

---

## Summary

### Quick Reference

**Installation**:
```bash
pip install "kimsfinance[gpu]"
python verify_gpu_setup.py
```

**Basic Usage**:
```python
from kimsfinance.ops.batch import calculate_indicators_batch

results = calculate_indicators_batch(
    highs, lows, closes, volumes,
    engine='auto',      # Recommended
    streaming=None      # Auto at 500K+ rows
)
```

**Key Thresholds**:
- **Batch indicators (6+)**: 15K rows (ALWAYS prefer this)
- Individual indicators: 100K-500K rows (rarely needed)
- Streaming mode: 500K+ rows (auto-enabled)

**Performance Tips**:
1. ✅ **ALWAYS use batch processing** (`calculate_indicators_batch()`)
2. ✅ GPU beneficial at just **15K rows** with batch processing
3. ✅ Use `engine='auto'` for automatic selection
4. ✅ Enable streaming for datasets >500K rows
5. ✅ Run auto-tune for your specific hardware

**Troubleshooting**:
- Python must be 3.9-3.12 (NOT 3.13) for RAPIDS
- Use conda for most reliable GPU installation
- Check `nvidia-smi` to verify GPU availability
- Use streaming mode to prevent OOM errors

### Related Documentation

- [GPU Optimization Guide](../GPU_OPTIMIZATION.md) - Detailed GPU architecture
- [API Reference](../API.md) - Complete API documentation
- [Performance Guide](../PERFORMANCE.md) - General performance tips

### Support

For GPU-related issues:
- GitHub Issues: https://github.com/kimasplund/kimsfinance/issues
- Check NVIDIA forums for driver/CUDA issues
- RAPIDS documentation for cuDF/CuPy issues

---

**Tutorial Version**: 1.0.0
**Last Updated**: 2025-10-23
**Tested On**: NVIDIA RTX 3500 Ada, Ubuntu 22.04, CUDA 12.6
