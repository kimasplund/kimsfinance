# GPU Optimization Guide

**Status**: Documentation in progress

This comprehensive GPU optimization guide is currently being written and will be available soon.

---

## Overview

**kimsfinance** supports optional GPU acceleration for:

- **OHLCV data processing**: 6.4x speedup with cuDF
- **Technical indicators**: 1.2x-2.9x speedup with CuPy/Numba CUDA
- **Large dataset aggregation**: Parallel processing on GPU

**Important**: GPU acceleration is **completely optional**. All features work on CPU-only systems.

---

## Quick Start: GPU Setup

### 1. Check GPU Availability

```python
from kimsfinance.utils import gpu_available

if gpu_available():
    print("GPU acceleration enabled!")
else:
    print("Running on CPU (perfectly fine!)")
```

### 2. Install GPU Dependencies (Optional)

```bash
# NVIDIA GPU with CUDA 12.x
pip install cudf-cu12 cupy-cuda12x

# Verify installation
python -c "import cudf; print(cudf.__version__)"
```

### 3. Enable GPU Acceleration

```python
from kimsfinance.api import plot

# GPU automatically used if available
plot.render(df, use_gpu=True)

# Force CPU-only
plot.render(df, use_gpu=False)
```

---

## Performance Gains

### OHLCV Processing (cuDF)

| Operation | CPU (pandas) | GPU (cuDF) | Speedup |
|-----------|--------------|------------|---------|
| 1M candles aggregation | 1.28s | **0.2s** | **6.4x** |
| Large dataset filtering | 500ms | **80ms** | **6.3x** |

### Technical Indicators (CuPy/Numba)

| Indicator | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| ATR | 10ms | **7ms** | **1.4x** |
| RSI | 15ms | **8ms** | **1.9x** |
| Stochastic | 20ms | **7ms** | **2.9x** |

---

## When to Use GPU

**Use GPU for**:
- ✅ Large datasets (>100K candles)
- ✅ Batch processing (1000+ charts)
- ✅ Technical indicator computation
- ✅ High-frequency data aggregation

**Stick with CPU for**:
- ✅ Small datasets (<10K candles)
- ✅ Single chart rendering
- ✅ Development/testing
- ✅ No NVIDIA GPU available

---

## Testing GPU Setup

### Run GPU Validation Tests

```bash
# Comprehensive GPU test suite
/kf/test/gpu

# Or run directly
pytest tests/test_gpu_*.py -v
```

### Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Profile GPU kernels
/kf/profile/gpu-kernel
```

---

## Coming Soon

This GPU optimization guide will include:

- ✅ Detailed GPU architecture overview
- ✅ cuDF integration deep dive
- ✅ CuPy/Numba CUDA optimization techniques
- ✅ Memory management on GPU
- ✅ Multi-GPU support
- ✅ Profiling and debugging GPU code
- ✅ Performance benchmarking methodology
- ✅ Troubleshooting common GPU issues

---

## See Also

- [Performance Guide](PERFORMANCE.md) - General performance optimization
- [README GPU Section](../README.md#-gpu-acceleration-optional) - Quick GPU overview
- GPU test suite: `tests/test_gpu_*.py`

---

## Hardware Tested

- **GPU**: NVIDIA RTX 3500 Ada (8 GB)
- **CPU**: Intel i9-13980HX (24 cores)
- **System**: ThinkPad P16 Gen2
- **CUDA**: 12.6
- **cuDF**: 24.10+

---

**Last Updated**: 2025-10-22
**Status**: Under development
