# FP8 E4M3 Conversion Kernels - Quick Reference

**File**: `/home/kim/projects/kimsfinance/rust/src/gpu/kernels/fp8_conversions.cu`

---

## Kernel API

### 1. `fp32_to_fp8_e4m3` - Main FP32 → FP8 Conversion

```cpp
extern "C" __global__ void fp32_to_fp8_e4m3(
    const float* input,      // FP32 input array
    unsigned char* output,   // FP8 E4M3 output (uint8)
    int n                    // Number of elements
);
```

**Launch Configuration**:
```cpp
threads_per_block = 256
blocks = (n + threads_per_block * 4 - 1) / (threads_per_block * 4)  // Each thread processes 4 elements
```

**Features**:
- Vectorized (4 elements per thread)
- Round-to-nearest, ties to even
- Saturates overflow (>448 → max FP8)
- Handles denormals, NaN, Inf

---

### 2. `fp8_e4m3_to_fp32` - Main FP8 → FP32 Conversion

```cpp
extern "C" __global__ void fp8_e4m3_to_fp32(
    const unsigned char* input,  // FP8 E4M3 input (uint8)
    float* output,               // FP32 output array
    int n                        // Number of elements
);
```

**Launch Configuration**: Same as above

**Features**:
- Vectorized (4 elements per thread)
- Exact reconstruction of FP8 values
- Denormal unpacking

---

### 3. `fp32_to_fp8_e4m3_saturate` - Explicit Clamping

```cpp
extern "C" __global__ void fp32_to_fp8_e4m3_saturate(
    const float* input,
    unsigned char* output,
    int n
);
```

**Use Case**: ML training where you want to **explicitly clamp** inputs to ±448 before conversion.

**Difference from main kernel**: Calls `fminf(fmaxf(val, -448), 448)` before conversion.

---

### 4. `fp32_to_fp8_e4m3_stochastic` - Stochastic Rounding

```cpp
extern "C" __global__ void fp32_to_fp8_e4m3_stochastic(
    const float* input,
    unsigned char* output,
    int n,
    unsigned int seed  // Random seed
);
```

**Use Case**: ML training to reduce quantization bias with randomized rounding.

**Rounding**: Uses LCG PRNG per thread instead of deterministic tie-breaking.

---

### 5. `test_fp8_conversions` - Validation Kernel

```cpp
extern "C" __global__ void test_fp8_conversions(
    float* test_values,      // Input: FP32 test values
    float* recovered,        // Output: FP32 after round-trip
    unsigned char* fp8_mid,  // Output: intermediate FP8 values
    int n
);
```

**Use Case**: Testing round-trip accuracy (FP32 → FP8 → FP32).

---

## FP8 E4M3 Format Cheat Sheet

| Component | Bits | Range | Notes |
|-----------|------|-------|-------|
| Sign | 1 (bit 7) | ±1 | 0=positive, 1=negative |
| Exponent | 4 (bits 6-3) | [0, 15] | Bias = 7 |
| Mantissa | 3 (bits 2-0) | [0, 7] | Fractional part |

**Value Ranges**:
- Normal: [0.015625, 448] = [2^-6, 448]
- Denormal: [0.001953125, 0.015625) = [2^-9, 2^-6)
- Max: ±448
- NaN: 0x7F (positive), 0xFF (negative)

**Special Encodings**:
```
Zero (+):      0x00
Zero (-):      0x80
Max positive:  0x7E (448.0)
Max negative:  0xFE (-448.0)
NaN positive:  0x7F
NaN negative:  0xFF
```

---

## Quick Test

```bash
cd /home/kim/projects/kimsfinance/rust
python scripts/test_fp8_conversions.py
```

**Expected**: ✅ ALL TESTS PASSED (3/3)

---

## Python Example (NVRTC)

```python
from cuda import cuda, nvrtc
import numpy as np

# 1. Compile kernel
with open('src/gpu/kernels/fp8_conversions.cu', 'r') as f:
    kernel_src = f.read()

err, program = nvrtc.nvrtcCreateProgram(kernel_src.encode(), b"fp8_conversions.cu", 0, [], [])
err = nvrtc.nvrtcCompileProgram(program, [b'--gpu-architecture=compute_89'])
err, ptx_size = nvrtc.nvrtcGetPTXSize(program)
ptx = b' ' * ptx_size
err = nvrtc.nvrtcGetPTX(program, ptx)
err, module = cuda.cuModuleLoadData(ptx)
err, kernel = cuda.cuModuleGetFunction(module, b'fp32_to_fp8_e4m3')

# 2. Prepare data
data = np.random.randn(1024).astype(np.float32)
output = np.zeros(1024, dtype=np.uint8)

d_in = cuda.cuMemAlloc(data.nbytes)[1]
d_out = cuda.cuMemAlloc(1024)[1]

cuda.cuMemcpyHtoD(d_in, data.ctypes.data, data.nbytes)

# 3. Launch
blocks = (1024 + 256*4 - 1) // (256*4)
cuda.cuLaunchKernel(kernel, blocks, 1, 1, 256, 1, 1, 0, 0, (d_in, d_out, 1024), 0)

# 4. Get results
cuda.cuMemcpyDtoH(output.ctypes.data, d_out, 1024)
```

---

## Performance Targets (RTX 3500 Ada)

| Operation | Array Size | Time (μs) | Bandwidth (GB/s) |
|-----------|------------|-----------|------------------|
| FP32→FP8  | 1K         | ~1        | ~5               |
| FP32→FP8  | 1M         | ~400      | ~350             |
| FP8→FP32  | 1M         | ~400      | ~350             |

**Bottleneck**: Memory bandwidth (432 GB/s theoretical)

---

## Common Pitfalls

### 1. **Wrong Grid Size**
```cpp
// ❌ WRONG: Doesn't account for 4x vectorization
blocks = (n + 255) / 256

// ✅ CORRECT: Each thread processes 4 elements
blocks = (n + 256*4 - 1) / (256*4)
```

### 2. **Forgetting Overflow Handling**
```python
# ❌ WRONG: Values >448 will become NaN
data = np.random.randn(1000) * 1000

# ✅ CORRECT: Use saturate kernel or clamp manually
data = np.clip(data, -448, 448)
```

### 3. **Precision Expectations**
```python
# ❌ WRONG: Expecting FP32 precision after round-trip
assert np.allclose(recovered, original, atol=1e-6)

# ✅ CORRECT: Account for FP8 quantization (~6% relative error)
assert np.allclose(recovered, original, rtol=0.1, atol=1e-3)
```

---

## Integration with Tensor Cores

```python
# 1. Convert inputs to FP8
A_fp32 = np.random.randn(128, 128).astype(np.float32)
B_fp32 = np.random.randn(128, 128).astype(np.float32)

A_fp8 = convert_fp32_to_fp8(A_fp32)  # Using kernel
B_fp8 = convert_fp32_to_fp8(B_fp32)

# 2. Run FP8 tensor core matmul
C_fp32 = fp8_matmul_mma_ptx(A_fp8, B_fp8)  # From fp8_mma_ptx.cu

# 3. Result is already FP32 (tensor cores accumulate in FP32)
print(f"Result shape: {C_fp32.shape}")
```

**Speedup**: 2-4x vs FP32 (memory bandwidth reduction + faster tensor cores)

---

## Debugging Tips

### Check FP8 Values
```python
fp8_val = output[0]
print(f"FP8: 0x{fp8_val:02X}")
print(f"Sign: {(fp8_val >> 7) & 1}")
print(f"Exp:  {(fp8_val >> 3) & 0xF}")
print(f"Mant: {fp8_val & 0x7}")
```

### Verify Round-Trip
```python
original = np.array([1.0, -1.0, 448.0], dtype=np.float32)
fp8 = convert_to_fp8(original)
recovered = convert_to_fp32(fp8)
print(f"Error: {np.abs(original - recovered)}")
```

### Profile Kernel
```python
import cupy as cp

# Time conversion
start = cp.cuda.Event()
end = cp.cuda.Event()

start.record()
convert_fp32_to_fp8(large_array)
end.record()
end.synchronize()

elapsed_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Conversion time: {elapsed_ms:.2f} ms")
```

---

## Summary Table

| Kernel | Purpose | Rounding | Use Case |
|--------|---------|----------|----------|
| `fp32_to_fp8_e4m3` | Main conversion | Deterministic | General-purpose |
| `fp8_e4m3_to_fp32` | Main conversion | Exact | General-purpose |
| `fp32_to_fp8_e4m3_saturate` | Explicit clamping | Deterministic | Training (prevent overflow) |
| `fp32_to_fp8_e4m3_stochastic` | Random rounding | Stochastic | Training (reduce bias) |
| `test_fp8_conversions` | Validation | Deterministic | Testing only |

---

**Documentation**: See `FP8_CONVERSIONS_IMPLEMENTATION.md` for full details.
