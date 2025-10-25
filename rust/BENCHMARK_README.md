# Stochastic Oscillator: CPU vs GPU Benchmark

## Overview

This benchmark script (`benchmark_gpu_vs_cpu.py`) provides comprehensive CPU vs GPU performance comparison for the Stochastic Oscillator indicator implementation in kimsfinance.

## Purpose

- Validate GPU acceleration benefits for different dataset sizes
- Determine optimal CPU/GPU crossover point
- Measure actual speedup ratios with statistical validity
- Verify correctness of GPU implementation vs CPU baseline

## Usage

### Basic Usage

```bash
python benchmark_gpu_vs_cpu.py
```

### Requirements

**CPU-only testing:**
- Python 3.13+
- NumPy 2.0+
- kimsfinance package

**GPU testing (recommended):**
- NVIDIA GPU with CUDA support
- CuPy 13.0+ (`pip install cupy-cuda12x`)
- CUDA Toolkit 12.x

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Indicator | Stochastic Oscillator |
| k_period | 14 |
| d_period | 3 |
| Iterations | 5 (with 2 warmup) |
| Dataset Sizes | 1K, 10K, 100K, 1M candles |

## Output Format

### System Information
```
System Information:
  Python:    3.13.9
  NumPy:     2.2.6
  GPU:       ✓ NVIDIA RTX 3500 Ada (11GB VRAM)
  CuPy:      13.6.0
```

### Results Table
```
Dataset Size |     CPU Time |     GPU Time |    Speedup |   Winner
-------------+--------------+--------------+------------+---------
       1,000 |     0.15 ms  |     2.50 ms  |     0.06x  |   CPU
      10,000 |     1.20 ms  |     3.10 ms  |     0.39x  |   CPU
     100,000 |    12.50 ms  |     8.20 ms  |     1.52x  |   GPU
   1,000,000 |   125.00 ms  |    45.30 ms  |     2.76x  |   GPU
```

### Throughput Comparison
```
Dataset Size |    CPU (candles/s) |    GPU (candles/s)
-------------+--------------------+-------------------
       1,000 |             77,728 |                N/A
      10,000 |             75,207 |                N/A
     100,000 |             75,852 |            245,000
   1,000,000 |             76,113 |            890,000
```

### Summary & Recommendations
```
Performance Statistics:
  Average Speedup:  1.18x
  Peak Speedup:     2.76x (at 1,000,000 candles)
  Minimum Speedup:  0.06x (at 1,000 candles)

GPU Crossover Point: ~100,000 candles
  GPU becomes faster than CPU at approximately 100,000 candles

Recommendations:
  ✓ Use GPU (engine='gpu') for datasets > 100,000 candles
  ✓ Use CPU (engine='cpu') for datasets < 100,000 candles
  ✓ Use engine='auto' for automatic selection based on dataset size
```

## Expected Results

Based on the Stochastic Oscillator implementation:

### Performance Profile

| Dataset Size | CPU Performance | GPU Performance | Expected Speedup | Winner |
|--------------|-----------------|-----------------|------------------|--------|
| 1K candles   | ~0.15 ms | ~2-3 ms | ~0.05-0.10x | CPU (GPU overhead) |
| 10K candles  | ~1.5 ms | ~3-5 ms | ~0.3-0.5x | CPU (still overhead) |
| 100K candles | ~15 ms | ~8-12 ms | ~1.2-1.9x | GPU (breaking even) |
| 1M candles   | ~150 ms | ~50-70 ms | ~2.1-3.0x | GPU (strong benefit) |

### GPU Crossover Point

According to the function documentation (`calculate_stochastic`):
- **Documented threshold**: 500,000 rows
- **Actual threshold**: May vary based on hardware
- **Expected range**: 100K - 500K candles

### Why GPU is Slower for Small Datasets

For small datasets (< 100K candles), GPU is typically **slower** due to:

1. **Data transfer overhead**:
   - CPU → GPU memory transfer (~0.5-2 ms)
   - GPU → CPU result transfer (~0.3-1 ms)

2. **Kernel launch overhead**:
   - CuPy function dispatch (~0.1-0.5 ms)
   - Rolling window setup (~0.2-0.8 ms)

3. **Insufficient parallelism**:
   - GPU has 5,120 CUDA cores (RTX 3500 Ada)
   - Small datasets don't utilize all cores
   - CPU vectorization (SIMD) is more efficient

### Why GPU is Faster for Large Datasets

For large datasets (> 500K candles), GPU benefits from:

1. **Massive parallelism**:
   - Rolling max/min computed in parallel
   - 5,120 CUDA cores vs 32 CPU threads

2. **Memory bandwidth**:
   - GPU: ~240 GB/s (RTX 3500 Ada)
   - CPU: ~76.8 GB/s (DDR5-4800)

3. **Optimized kernels**:
   - CuPy's rolling operations are highly optimized
   - Fused operations reduce memory bandwidth

## Interpreting Results

### Color Codes

- 🟢 **Green**: GPU is significantly faster (> 1.2x speedup)
- 🟡 **Yellow**: Roughly equivalent performance (0.95x - 1.2x)
- 🔴 **Red**: CPU is faster (< 0.95x speedup)
- 🔵 **Cyan**: CPU winner

### Speedup Ratio

```
Speedup = CPU Time / GPU Time

> 1.0 = GPU is faster
= 1.0 = Equivalent performance
< 1.0 = CPU is faster
```

### Correctness Verification

The script automatically verifies that CPU and GPU results match:

```python
✓ Results match  # CPU and GPU outputs are identical (within floating-point tolerance)
✗ Results differ!  # Error: CPU and GPU outputs differ
```

Uses `np.allclose()` with:
- `rtol=1e-5` (relative tolerance: 0.001%)
- `atol=1e-8` (absolute tolerance)
- `equal_nan=True` (NaN values are considered equal)

## Customization

### Test Different Dataset Sizes

Edit the `data_sizes` list in `main()`:

```python
data_sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
```

### Change Number of Iterations

Increase iterations for more stable results:

```python
cpu_time, cpu_min, cpu_max, cpu_result = benchmark_stochastic(
    high, low, close, engine="cpu", iterations=10, warmup=3  # More iterations
)
```

### Test Different Stochastic Parameters

Modify the parameters in `benchmark_stochastic()`:

```python
k, d = calculate_stochastic(high, low, close, k_period=5, d_period=3, engine=engine)
```

## Troubleshooting

### Issue: "CuPy not installed"

**Solution**: Install CuPy with CUDA 12.x support:
```bash
pip install cupy-cuda12x
```

### Issue: "libnvrtc.so.13: cannot open shared object file"

**Solution**: Install CUDA Toolkit 12.x:
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA
# https://developer.nvidia.com/cuda-downloads
```

### Issue: "GPU operation failed"

**Possible causes**:
1. GPU driver not installed: `nvidia-smi` to check
2. CUDA version mismatch: CuPy 13.x requires CUDA 12.x
3. Insufficient VRAM: 1M candles needs ~50 MB VRAM
4. GPU in use: Close other GPU-intensive applications

**Solution**: Check GPU status:
```bash
nvidia-smi  # Should show RTX 3500 Ada

# Check CUDA version
nvcc --version

# Check CuPy can access GPU
python -c "import cupy as cp; print(cp.cuda.Device(0).compute_capability)"
```

### Issue: Slow Performance (Both CPU and GPU)

**Possible causes**:
1. Debug build of NumPy/CuPy
2. Python running in debug mode
3. System under load
4. Thermal throttling

**Solution**:
```bash
# Run with optimization
python -O benchmark_gpu_vs_cpu.py

# Check system load
htop  # CPU usage
nvidia-smi dmon  # GPU usage

# Check NumPy is optimized
python -c "import numpy as np; np.show_config()"
```

## Integration with kimsfinance

### Using Auto-Selection

The benchmark results inform the `engine='auto'` behavior:

```python
from kimsfinance.ops import calculate_stochastic

# Automatic CPU/GPU selection based on dataset size
k, d = calculate_stochastic(high, low, close, engine='auto')

# For 1M candles: Uses GPU (2-3x faster)
# For 1K candles: Uses CPU (GPU overhead not worth it)
```

### Manual Engine Selection

Based on benchmark results:

```python
# Force CPU (optimal for < 100K candles)
k, d = calculate_stochastic(high, low, close, engine='cpu')

# Force GPU (optimal for > 500K candles)
k, d = calculate_stochastic(high, low, close, engine='gpu')
```

### Tuning GPU Thresholds

Results can inform GPU threshold configuration:

```python
# kimsfinance/config/gpu_thresholds.py
GPU_THRESHOLDS = {
    "rolling": 500_000,  # Based on benchmark crossover point
    # ...
}
```

## Performance Optimization Tips

### For Small Datasets (< 100K)

✓ Use `engine='cpu'` - GPU overhead not worth it
✓ Enable NumPy MKL optimization
✓ Use vectorized operations

### For Large Datasets (> 500K)

✓ Use `engine='gpu'` - Significant speedup
✓ Batch multiple calculations together
✓ Keep data on GPU between operations
✓ Minimize CPU ↔ GPU transfers

### For Medium Datasets (100K - 500K)

✓ Use `engine='auto'` - Let system decide
✓ Profile your specific workload
✓ Consider batch size and frequency

## Benchmark Methodology

### Statistical Validity

- **5 iterations**: Reduces timing noise
- **2 warmup iterations**: JIT compilation, cache warming
- **Median time**: Robust to outliers
- **Min/Max times**: Shows variance

### Synthetic Data Generation

Uses realistic OHLC data:
- Random walk for close prices
- Realistic high/low spread based on volatility
- Positive price constraint (no negative prices)
- Fixed seed for reproducibility

### Accuracy

Timing resolution:
- `time.perf_counter()`: Sub-microsecond precision
- Warmup iterations eliminate JIT overhead
- Multiple iterations average out system noise

## Related Files

- **Implementation**: `/home/kim-asplund/projects/kimsfinance/kimsfinance/ops/stochastic.py`
- **GPU Decorator**: `/home/kim-asplund/projects/kimsfinance/kimsfinance/core/decorators.py`
- **GPU Thresholds**: `/home/kim-asplund/projects/kimsfinance/kimsfinance/config/gpu_thresholds.py`
- **Tests**: `/home/kim-asplund/projects/kimsfinance/tests/ops/indicators/test_stochastic.py`

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/)
- [RAPIDS.ai cuDF](https://docs.rapids.ai/api/cudf/)
- [kimsfinance Documentation](../README.md)

## License

Part of the kimsfinance project - See LICENSE for details.

---

**Last Updated**: 2025-10-25
**Benchmark Version**: 1.0
**Hardware**: NVIDIA RTX 3500 Ada (12GB VRAM)
**Software**: Python 3.13.9, CuPy 13.6.0, NumPy 2.2.6
