# GPU Kernels

High-performance CUDA kernels for GPU-accelerated operations, designed for NVRTC JIT compilation.

## Available Kernels

### 1. FP16 Tensor Core Matrix Multiply
**File**: `fp16_mma_ptx.cu`

FP16 matrix multiplication using tensor cores via raw PTX inline assembly.

**Features**:
- Raw PTX MMA instructions (no cuda_ptx namespace)
- NVRTC compatible (no SDK headers)
- 2x speedup vs FP32 on tensor cores
- Supports sm_70+ (Volta, Turing, Ampere, Ada)

**Usage**:
```python
import cupy as cp

kernel = cp.RawKernel(open("fp16_mma_ptx.cu").read(), "fp16_matmul_mma_ptx")
kernel((blocks_y, blocks_x), (32,), (A_fp16, B_fp16, C_fp16, m, n, k))
```

**Documentation**: See inline comments in file

---

### 2. FP16 Conversion Kernels
**File**: `fp16_conversions.cu`

FP32 ↔ FP16 conversion kernels for tensor core preparation.

**Features**:
- Hardware-accelerated (`__float2half_rn`, `__half2float`)
- Manual bitwise fallback (if intrinsics unavailable)
- Vectorized versions (4x throughput)
- Correct special value handling (Inf, NaN, Zero)
- Comprehensive test suite

**Kernels**:
1. `fp32_to_fp16` - FP32 → FP16 (scalar)
2. `fp16_to_fp32` - FP16 → FP32 (scalar)
3. `fp32_to_fp16_manual` - Manual conversion (fallback)
4. `fp16_to_fp32_manual` - Manual conversion (fallback)
5. `fp32_to_fp16_vectorized` - Vectorized (4x throughput)
6. `fp16_to_fp32_vectorized` - Vectorized (4x throughput)
7. `test_fp16_roundtrip` - Accuracy test
8. `test_fp16_special_values` - Special value test
9. `test_fp16_conversion_methods` - Method comparison
10. `benchmark_fp32_to_fp16` - Performance test
11. `benchmark_fp16_to_fp32` - Performance test

**Usage**:
```python
import cupy as cp

kernel = cp.RawKernel(open("fp16_conversions.cu").read(), "fp32_to_fp16_vectorized")

# Convert FP32 to FP16
threads = 256
blocks = ((n // 4) + threads - 1) // threads
kernel((blocks,), (threads,), (input_fp32, output_fp16, n))
```

**Documentation**: See `docs/FP16_CONVERSIONS.md`

**Tests**:
- `scripts/test_fp16_conversions_cupy.py` - CuPy-based tests (recommended)
- `scripts/test_fp16_conversions.py` - Raw CUDA Driver API tests

---

## Quick Start

### Testing FP16 Conversions

```bash
# Run CuPy test suite (recommended)
cd /home/kim/projects/kimsfinance/rust
python scripts/test_fp16_conversions_cupy.py

# Expected output:
# ✓ Compilation successful
# ✓ Round-trip accuracy within FP16 precision
# ✓ All special values handled correctly
# ✓ Vectorized version 2-4x faster
```

### Integration Example (FP32 → Tensor Core → FP32)

```python
import cupy as cp
import numpy as np

# Load kernels
conv_src = open("src/gpu/kernels/fp16_conversions.cu").read()
mma_src = open("src/gpu/kernels/fp16_mma_ptx.cu").read()

fp32_to_fp16 = cp.RawKernel(conv_src, "fp32_to_fp16_vectorized")
fp16_matmul = cp.RawKernel(mma_src, "fp16_matmul_mma_ptx")
fp16_to_fp32 = cp.RawKernel(conv_src, "fp16_to_fp32_vectorized")

# Prepare data
m, n, k = 1024, 1024, 1024
A_fp32 = cp.random.randn(m, k, dtype=cp.float32)
B_fp32 = cp.random.randn(k, n, dtype=cp.float32)

A_fp16 = cp.zeros((m, k), dtype=cp.uint16)
B_fp16 = cp.zeros((k, n), dtype=cp.uint16)
C_fp16 = cp.zeros((m, n), dtype=cp.uint16)
C_fp32 = cp.zeros((m, n), dtype=cp.float32)

# Convert FP32 → FP16
threads = 256
fp32_to_fp16(
    (((m*k)//4 + threads-1)//threads,), (threads,),
    (A_fp32, A_fp16, np.int32(m*k))
)
fp32_to_fp16(
    (((k*n)//4 + threads-1)//threads,), (threads,),
    (B_fp32, B_fp16, np.int32(k*n))
)

# Tensor core matmul (FP16)
MMA_M, MMA_N = 16, 8
fp16_matmul(
    ((m + MMA_M - 1) // MMA_M, (n + MMA_N - 1) // MMA_N), (32,),
    (A_fp16, B_fp16, C_fp16, np.int32(m), np.int32(n), np.int32(k))
)

# Convert FP16 → FP32
fp16_to_fp32(
    (((m*n)//4 + threads-1)//threads,), (threads,),
    (C_fp16, C_fp32, np.int32(m*n))
)

# Result: C_fp32 ≈ A_fp32 @ B_fp32 (with FP16 precision)
```

**Expected Performance**: 2x speedup vs FP32 matmul (from tensor cores)

---

## Design Principles

### 1. NVRTC Compatibility
All kernels avoid CUDA SDK headers (`cuda_fp16.h`, `mma.h`, `cuda::ptx`):
- Use raw PTX inline assembly instead of SDK templates
- Use `unsigned short` instead of `__half` type
- Use `__float2half_rn()` intrinsic (available in NVRTC)

### 2. Performance Optimization
- Coalesced memory access (consecutive threads → consecutive memory)
- Vectorized loads/stores (`float4`, `int4`)
- Optimal thread block size (256 threads = 8 warps)
- Minimal register pressure (high occupancy)

### 3. Correctness
- Comprehensive test coverage (accuracy, special values, edge cases)
- IEEE 754 compliance (round-to-nearest-even)
- Validation against CuPy native implementations

---

## GPU Requirements

### Minimum (FP16 Conversions)
- **Compute Capability**: 5.3+ (Maxwell, GTX 900 series)
- **CUDA Toolkit**: 11.0+
- **Intrinsics**: `__float2half_rn`, `__half2float`

### Recommended (Tensor Cores)
- **Compute Capability**: 7.0+ (Volta, V100, RTX 2000+)
- **CUDA Toolkit**: 11.0+
- **Features**: Tensor Cores, sm_70+ PTX

### Optimal (This Project)
- **GPU**: RTX 3500 Ada (Compute Capability 8.9)
- **VRAM**: 12GB
- **CUDA Toolkit**: 12.0+

---

## Performance Benchmarks

### FP16 Conversion (Expected on RTX 3500 Ada)

| Operation | Throughput | Memory BW | Notes |
|-----------|------------|-----------|-------|
| FP32→FP16 (scalar) | 1.5M elem/ms | ~300 GB/s | Baseline |
| FP32→FP16 (vector) | 5M elem/ms | ~600 GB/s | 3-4x faster |
| FP16→FP32 (scalar) | 1.8M elem/ms | ~350 GB/s | Baseline |
| FP16→FP32 (vector) | 6M elem/ms | ~700 GB/s | 3-4x faster |

### FP16 Tensor Core Matmul (Expected)

| Size | FP32 (ms) | FP16 TC (ms) | Speedup |
|------|-----------|--------------|---------|
| 1024×1024×1024 | 2.5 | 1.2 | 2.1x |
| 2048×2048×2048 | 20 | 9.5 | 2.1x |
| 4096×4096×4096 | 160 | 76 | 2.1x |

**Note**: Actual performance varies by GPU architecture and data layout.

---

## Troubleshooting

### Issue: "identifier __float2half_rn is undefined"
**Cause**: GPU compute capability < 5.3

**Solution**: Use manual conversion kernels:
```python
kernel = cp.RawKernel(source, "fp32_to_fp16_manual")  # Instead of fp32_to_fp16
```

### Issue: Incorrect results (large errors)
**Cause**: Expected precision loss for FP16

**Solution**: Verify error within FP16 precision (~1e-3 relative):
```python
max_rel_error = np.max(np.abs(result - expected) / (np.abs(expected) + 1e-10))
assert max_rel_error < 0.01, f"Error too large: {max_rel_error}"
```

### Issue: Slow performance
**Cause**: Using scalar instead of vectorized kernels

**Solution**: Use vectorized kernels for arrays >10K elements:
```python
# Use vectorized version
kernel = cp.RawKernel(source, "fp32_to_fp16_vectorized")
blocks = ((n // 4) + threads - 1) // threads  # Process 4 elements/thread
```

---

## References

1. **Tensor Core Programming**: https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d
2. **FP16 Format**: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
3. **CUDA Math API**: https://docs.nvidia.com/cuda/cuda-math-api/
4. **PTX ISA**: https://docs.nvidia.com/cuda/parallel-thread-execution/

---

## License

AGPL-3.0-or-later - see the project [LICENSE](../../../../LICENSE); commercial licensing is available, see [LICENSING.md](../../../../LICENSING.md) and [COMMERCIAL-LICENSE.md](../../../../COMMERCIAL-LICENSE.md)

---

**Last Updated**: 2025-11-01
**Status**: Production Ready ✅
