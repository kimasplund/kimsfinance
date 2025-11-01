# FP8 Kernel Build Script Guide

## Overview

The `build.rs` script compiles CUDA FP8 kernels during Rust build time when the `gpu` feature is enabled. This document explains the build process, configuration options, and troubleshooting.

**Status**: ✅ Implemented and tested
**Date**: 2025-11-01
**CUDA Version**: 13.0+
**Target GPU**: NVIDIA Ada Lovelace (sm_89, RTX 3500 Ada)

---

## Features

### 1. Automatic GPU Detection

The build script auto-detects your GPU's compute capability using `nvidia-smi`:

```bash
cargo build --features gpu
# Output: Auto-detected GPU architecture: sm_89 (compute cap 8.9)
```

**Fallback**: If detection fails, defaults to `sm_89` (RTX 3500 Ada).

### 2. Dual Kernel Compilation

Compiles two FP8 kernel variants:

1. **FP8 WMMA Kernel** (`src/gpu/kernels_fp8_wmma.cu`)
   - Uses NVIDIA's WMMA (Warp Matrix Multiply-Accumulate) API
   - Hardware tensor core acceleration
   - Simpler, more reliable compilation
   - Output: `fp8_wmma_kernels.cubin`

2. **FP8 CUTLASS Kernel** (`src/gpu/kernels/fp8_cutlass.cu`)
   - Uses NVIDIA CUTLASS library (experimental)
   - Direct FP8 E4M3 API access
   - Requires CUTLASS headers
   - Output: `fp8_kernels.cubin`

### 3. Graceful Degradation

- If `nvcc` not found: Skips CUDA compilation, warns user
- If CUDA toolkit not found: Skips compilation
- If CUTLASS not found: Skips CUTLASS kernel only
- If compilation fails: Non-critical warning, continues build

### 4. Environment Variable Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_HOME` | Auto-detect | CUDA toolkit path |
| `CUTLASS_PATH` | `/tmp/cutlass` | CUTLASS include path |
| `CUDA_ARCH` | Auto-detect or `sm_89` | Target GPU architecture |

---

## Build Process

### Standard Build (Auto-detect)

```bash
cargo build --features gpu
```

**What happens**:
1. Detects GPU architecture via `nvidia-smi` (e.g., sm_89)
2. Finds CUDA toolkit at `/usr/local/cuda-13.0`
3. Finds CUTLASS at `/tmp/cutlass`
4. Compiles FP8 WMMA kernel → `target/debug/build/.../out/fp8_wmma_kernels.cubin`
5. Compiles FP8 CUTLASS kernel → `target/debug/build/.../out/fp8_kernels.cubin`
6. Emits `cargo:rustc-env` variables for runtime access

### Custom Configuration

#### Custom CUDA Path

```bash
CUDA_HOME=/opt/cuda-13.0 cargo build --features gpu
```

#### Custom Architecture (Different GPU)

```bash
# RTX 4090 (sm_89)
CUDA_ARCH=sm_89 cargo build --features gpu

# RTX 3090 (sm_86)
CUDA_ARCH=sm_86 cargo build --features gpu

# A100 (sm_80)
CUDA_ARCH=sm_80 cargo build --features gpu
```

#### Custom CUTLASS Path

```bash
CUTLASS_PATH=/home/user/cutlass cargo build --features gpu
```

---

## Compilation Details

### FP8 WMMA Kernel

**Command**:
```bash
nvcc -cubin \
     -arch=sm_89 \
     -std=c++17 \
     -I/usr/local/cuda-13.0/include \
     -I/usr/local/cuda-13.0/targets/x86_64-linux/include \
     -O3 \
     -use_fast_math \
     --expt-relaxed-constexpr \
     --expt-extended-lambda \
     -D_FORCE_INLINES \
     -Xcompiler=-w \
     -o fp8_wmma_kernels.cubin \
     src/gpu/kernels_fp8_wmma.cu
```

**Flags Explained**:
- `-cubin`: Compile to CUBIN (device binary, not PTX)
- `-arch=sm_89`: Target Ada Lovelace compute capability
- `-std=c++17`: C++17 for modern features
- `-O3`: Maximum optimization
- `-use_fast_math`: Fast math operations (tensor cores)
- `--expt-relaxed-constexpr`: Relaxed constexpr for CUDA
- `--expt-extended-lambda`: Extended lambda support
- `-D_FORCE_INLINES`: Workaround for CUDA 13.0 math header conflicts
- `-Xcompiler=-w`: Suppress host compiler warnings

### FP8 CUTLASS Kernel

**Command** (same as WMMA, plus CUTLASS includes):
```bash
nvcc -cubin \
     -arch=sm_89 \
     -std=c++17 \
     -I/tmp/cutlass/include \
     -I/usr/local/cuda-13.0/include \
     -I/usr/local/cuda-13.0/targets/x86_64-linux/include \
     -O3 \
     -use_fast_math \
     --expt-relaxed-constexpr \
     --expt-extended-lambda \
     -D_FORCE_INLINES \
     -Xcompiler=-w \
     -o fp8_kernels.cubin \
     src/gpu/kernels/fp8_cutlass.cu
```

---

## Runtime Access

The build script emits environment variables for runtime access:

```rust
// In your Rust code
fn load_fp8_kernels() {
    // Path to WMMA kernel
    let wmma_cubin_path = env!("FP8_WMMA_CUBIN_PATH");

    // Path to CUTLASS kernel
    let cutlass_cubin_path = env!("FP8_CUTLASS_CUBIN_PATH");

    // Load kernels at runtime
    // (implementation depends on your GPU framework)
}
```

**Note**: These variables are only set if compilation succeeds.

---

## Rebuild Triggers

The build script will recompile kernels when:

- `src/gpu/kernels/fp8_cutlass.cu` changes
- `src/gpu/kernels_fp8_wmma.cu` changes
- `build.rs` changes
- `CUDA_HOME` environment variable changes
- `CUTLASS_PATH` environment variable changes
- `CUDA_ARCH` environment variable changes

**Cargo directives**:
```rust
println!("cargo:rerun-if-changed=src/gpu/kernels/fp8_cutlass.cu");
println!("cargo:rerun-if-changed=src/gpu/kernels_fp8_wmma.cu");
println!("cargo:rerun-if-changed=build.rs");
println!("cargo:rerun-if-env-changed=CUDA_HOME");
println!("cargo:rerun-if-env-changed=CUTLASS_PATH");
println!("cargo:rerun-if-env-changed=CUDA_ARCH");
```

---

## Troubleshooting

### 1. nvcc Not Found

**Symptom**:
```
cargo:warning=nvcc not found in PATH. Skipping FP8 CUTLASS kernel compilation.
```

**Solution**:
```bash
# Install CUDA Toolkit
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version

# Ensure nvcc is in PATH
export PATH=/usr/local/cuda-13.0/bin:$PATH
```

### 2. CUDA Toolkit Not Found

**Symptom**:
```
cargo:warning=CUDA toolkit not found. Set CUDA_HOME or install CUDA.
```

**Solution**:
```bash
# Option 1: Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-13.0
cargo build --features gpu

# Option 2: Create symlink
sudo ln -s /usr/local/cuda-13.0 /usr/local/cuda
```

### 3. CUTLASS Not Found

**Symptom**:
```
cargo:warning=CUTLASS not found. FP8 kernels require CUTLASS headers.
```

**Solution**:
```bash
# Option 1: Clone to /tmp/cutlass (default)
git clone https://github.com/NVIDIA/cutlass.git /tmp/cutlass

# Option 2: Set CUTLASS_PATH
export CUTLASS_PATH=/home/user/cutlass
cargo build --features gpu
```

### 4. Compilation Failures (Math Header Conflicts)

**Symptom**:
```
cargo:warning=Failed to compile FP8 WMMA kernel (non-critical)
cargo:warning=nvcc stderr: /usr/include/x86_64-linux-gnu/bits/mathcalls.h(206):
error: exception specification is incompatible with that of previous function "rsqrt"
```

**Root Cause**: CUDA 13.0 has conflicts with glibc 2.38+ math headers.

**Current Status**: Non-critical (build continues, kernels skip FP8 support).

**Future Fix Options**:
1. **Update kernel code** to avoid problematic headers
2. **Use CUDA 12.4** instead of 13.0 (known to work)
3. **Patch kernels** to use internal CUDA math only
4. **Wait for CUDA 13.1** (NVIDIA may fix this)

**Workaround** (temporary):
```bash
# Use CUDA 12.4 if available
CUDA_HOME=/usr/local/cuda-12.4 cargo build --features gpu
```

### 5. FP8 API Errors

**Symptom**:
```
error: no suitable conversion function from "const __nv_fp8_e4m3" to "__nv_fp8_storage_t"
```

**Root Cause**: CUDA 13.0 changed FP8 API (storage types required).

**Status**: Known issue in kernel code (not build.rs).

**Fix**: Update kernel to use `__nv_fp8_storage_t` type conversions.

---

## Performance Notes

### FP8 E4M3 Format

- **Range**: ±448
- **Precision**: ~2 decimal digits (0.01 resolution)
- **Hardware Acceleration**: 4x throughput vs FP32 on tensor cores
- **Use Case**: Genetic optimizer exploration phase (acceptable precision loss)

### When to Use FP8

✅ **Good for**:
- Large matrix multiplications (1000x1000+)
- Genetic algorithm parameter grids (10K+ evaluations)
- Approximate backtest simulations
- Exploration phase (wide search)

❌ **Not good for**:
- Final validation (use FP32)
- Small matrices (<100x100)
- High-precision requirements (>2 decimal digits)
- Critical financial calculations

---

## Architecture Support

| GPU | Compute Capability | FP8 Support | Recommended CUDA |
|-----|-------------------|-------------|------------------|
| RTX 3500 Ada | 8.9 | ✅ Yes | 13.0+ |
| RTX 4090 | 8.9 | ✅ Yes | 13.0+ |
| RTX 4080 | 8.9 | ✅ Yes | 13.0+ |
| RTX 3090 | 8.6 | ❌ No (FP16 only) | 12.4+ |
| A100 | 8.0 | ❌ No (TF32 only) | 12.4+ |
| V100 | 7.0 | ❌ No | 12.4+ |

**Note**: FP8 E4M3 tensor cores are exclusive to Ada Lovelace (sm_89) and Hopper (sm_90) GPUs.

---

## Example Build Output

### Successful Build

```
cargo:warning=GPU feature enabled, attempting CUDA kernel compilation
cargo:warning=Found nvcc at: /usr/bin/nvcc
cargo:warning=CUDA toolkit found at: /usr/local/cuda-13.0
cargo:warning=CUTLASS found at: /tmp/cutlass
cargo:warning=Auto-detected GPU architecture: sm_89 (compute cap 8.9)
cargo:warning=Compiling for CUDA architecture: sm_89
cargo:warning=Compiling FP8 WMMA kernel: src/gpu/kernels_fp8_wmma.cu
cargo:warning=Successfully compiled FP8 WMMA kernel to: target/debug/build/.../out/fp8_wmma_kernels.cubin
cargo:warning=Compiling FP8 CUTLASS kernel: src/gpu/kernels/fp8_cutlass.cu
cargo:warning=Successfully compiled FP8 CUTLASS kernel to: target/debug/build/.../out/fp8_kernels.cubin
```

### Build with Warnings (Current Status)

```
cargo:warning=GPU feature enabled, attempting CUDA kernel compilation
cargo:warning=Found nvcc at: /usr/bin/nvcc
cargo:warning=CUDA toolkit found at: /usr/local/cuda-13.0
cargo:warning=CUTLASS found at: /tmp/cutlass
cargo:warning=Auto-detected GPU architecture: sm_89 (compute cap 8.9)
cargo:warning=Compiling for CUDA architecture: sm_89
cargo:warning=Compiling FP8 WMMA kernel: src/gpu/kernels_fp8_wmma.cu
cargo:warning=Failed to compile FP8 WMMA kernel (non-critical)
cargo:warning=Exit code: Some(1)
cargo:warning=Compiling FP8 CUTLASS kernel: src/gpu/kernels/fp8_cutlass.cu
cargo:warning=Failed to compile FP8 CUTLASS kernel (non-critical)
cargo:warning=Note: FP8 WMMA kernel may still work. CUTLASS kernel is experimental.
```

**Interpretation**: Build continues successfully, but FP8 kernels are not available at runtime. System falls back to FP32 kernels.

---

## Future Improvements

### Phase 1: Fix Kernel Code (High Priority)

- [ ] Update FP8 conversion functions to use `__nv_fp8_storage_t`
- [ ] Replace glibc math headers with CUDA internal math
- [ ] Test with CUDA 12.4 for compatibility

### Phase 2: Enhanced Build Features (Medium Priority)

- [ ] Multi-architecture builds (`-gencode` for multiple GPUs)
- [ ] PTX fallback (if CUBIN fails)
- [ ] Precompiled kernel caching
- [ ] CI/CD integration tests

### Phase 3: Runtime Features (Low Priority)

- [ ] Dynamic kernel loading based on GPU detection
- [ ] Fallback to software FP8 simulation
- [ ] Performance benchmarking in build script

---

## References

- **CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads
- **CUTLASS Library**: https://github.com/NVIDIA/cutlass
- **FP8 Documentation**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fp8
- **Ada Lovelace Architecture**: https://www.nvidia.com/en-us/data-center/resources/ada-lovelace-architecture/
- **Build Scripts**: https://doc.rust-lang.org/cargo/reference/build-scripts.html

---

## License

Same as parent project (kimsfinance).

---

**Last Updated**: 2025-11-01
**Maintained By**: kimsfinance core team
**Status**: Production-ready (build.rs), kernels need fixes
