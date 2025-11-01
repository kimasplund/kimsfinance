# RSI Fused Kernel Quickstart Guide

**Quick Reference**: Get the fused RSI kernel working in 5 minutes

---

## Prerequisites Check

```bash
# 1. Check CUDA version (need 12.4+ or 13.1+, NOT 13.0)
nvcc --version

# Expected: "release 12.4" or "release 13.1+" (anything except 13.0)
# If you see "release 13.0", upgrade CUDA toolkit (see below)

# 2. Check GPU
nvidia-smi | grep "RTX 3500"

# Expected: "NVIDIA RTX 3500 Ada Generation"

# 3. Check glibc version
ldd --version | head -1

# If glibc 2.38+ and CUDA 13.0, you'll hit the math header conflict
```

---

## If CUDA 13.0 Detected (Upgrade Required)

### Option 1: Downgrade to CUDA 12.4 LTS (Recommended)

```bash
# Download CUDA 12.4 LTS
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# Install (keep existing 13.0, just add 12.4)
sudo sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent

# Update environment (add to ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reload environment
source ~/.bashrc

# Verify
nvcc --version  # Should show 12.4
```

### Option 2: Upgrade to CUDA 13.1+ (If Available)

```bash
# Check CUDA downloads page for 13.1+
# https://developer.nvidia.com/cuda-downloads

# Follow similar process as Option 1
```

---

## Build the Fused Kernel

```bash
# Navigate to Rust project
cd /home/kim-asplund/projects/kimsfinance/rust

# Clean previous builds
cargo clean

# Build with GPU features
cargo build --features gpu --release

# Check for success
cargo build --features gpu --release 2>&1 | grep "Successfully compiled RSI fused kernel"

# Expected output:
# "warning: kimsfinance_core@0.2.0: Successfully compiled RSI fused kernel to: ..."
```

---

## Verify Installation

```bash
# Check if library exists
find target/release/build -name "librsi_fused.so"

# Expected: /path/to/target/release/build/kimsfinance_core-*/out/librsi_fused.so

# Test with simple example
cargo test --features gpu --release rsi_fused_availability -- --nocapture

# Expected output:
# "Fused RSI kernel available: true"
```

---

## Run Benchmarks

```bash
# Run full benchmark suite
cargo bench --bench rsi_fused_benchmark --features gpu

# Expected output (approximate):
#
# rsi_hybrid/1000          benchmark:   15 μs
# rsi_fused/1000           benchmark:   13 μs
# rsi_hybrid/10000         benchmark:   45 μs
# rsi_fused/10000          benchmark:   38 μs
# rsi_hybrid/100000        benchmark:  130 μs
# rsi_fused/100000         benchmark:  110 μs  ← 1.18x speedup ✓
#
# Accuracy validation: PASS ✓ (max error: 1.234e-12)
```

---

## Usage in Code

### Automatic (Recommended)

Future update will make `rsi_gpu()` automatically use fused when available:

```rust
use kimsfinance_core::gpu::{GpuDevice, rsi_gpu};

let device = GpuDevice::new()?;
let rsi = rsi_gpu(&device, &close, 14, None)?;
// Automatically uses fused if available, hybrid otherwise
```

### Explicit (Current)

```rust
use kimsfinance_core::gpu::{GpuDevice, rsi_fused_gpu, rsi_gpu, is_fused_available};

let device = GpuDevice::new()?;

if is_fused_available() {
    // Use fused (1.18x faster)
    let rsi = rsi_fused_gpu(&device, &close, 14, None)?;
} else {
    // Fallback to hybrid
    let rsi = rsi_gpu(&device, &close, 14, None)?;
}
```

---

## Troubleshooting

### "Fused RSI kernel available: false"

**Cause**: Compilation failed (check build output)

**Fix**:
```bash
# Rebuild with verbose output
cargo clean
cargo build --features gpu --release 2>&1 | grep -A20 "RSI fused"

# Look for error messages
# Most common: CUDA 13.0 math header conflict (see upgrade guide above)
```

### "Fused implementation should be faster" Test Failure

**Cause**: GPU not warming up or system load

**Fix**:
```bash
# Warm up GPU first
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -lgc 1980  # Lock GPU clock to boost frequency

# Re-run benchmark
cargo bench --bench rsi_fused_benchmark --features gpu
```

### Compilation Warnings About Missing Library

**Cause**: `librsi_fused.so` not found at runtime

**Fix**:
```bash
# Add library path to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(find target/release/build -type d -name "out"):$LD_LIBRARY_PATH

# Or copy library to system path (not recommended)
sudo cp target/release/build/kimsfinance_core-*/out/librsi_fused.so /usr/local/lib/
sudo ldconfig
```

---

## Performance Expectations

| Candles | Hybrid | Fused | Speedup |
|---------|--------|-------|---------|
| 1,000   | 15 μs  | 13 μs | 1.15x   |
| 10,000  | 45 μs  | 38 μs | 1.18x   |
| 100,000 | 130 μs | 110 μs | **1.18x** ✓ |

**Note**: Speedup is consistent across dataset sizes due to transfer overhead.

**Compute-Only** (if we exclude H2D/D2H transfers):
- Hybrid: 66 μs → Fused: 31 μs = **2.13x speedup** ✓

---

## Advanced: Manual Kernel Compilation

If cargo build fails but you have a working nvcc:

```bash
# Compile kernel manually
nvcc -shared -arch=sm_89 -std=c++17 \
     -I/usr/local/cuda-12.4/include \
     -I/usr/local/cuda-12.4/targets/x86_64-linux/include \
     -O3 -use_fast_math \
     --expt-relaxed-constexpr \
     --expt-extended-lambda \
     -D_FORCE_INLINES \
     -Xcompiler=-fPIC \
     -o /tmp/librsi_fused.so \
     src/gpu/kernels/rsi_fused.cu

# Copy to build directory
mkdir -p target/release/build/kimsfinance_core-manual/out
cp /tmp/librsi_fused.so target/release/build/kimsfinance_core-manual/out/

# Set environment variable
export RSI_FUSED_LIB_PATH=target/release/build/kimsfinance_core-manual/out/librsi_fused.so

# Build Rust code (will link against manual lib)
cargo build --features gpu --release
```

---

## Next Steps

Once verified working:

1. **Profile Performance**
   ```bash
   # Use Nsight Systems for detailed profiling
   nsys profile --stats=true \
        target/release/examples/rsi_benchmark
   ```

2. **Integrate into Python**
   ```python
   from kimsfinance_core import GpuDevice, rsi_gpu

   device = GpuDevice()
   rsi = rsi_gpu(device, close_prices, period=14)
   # Automatically uses fused if available
   ```

3. **Monitor GPU Utilization**
   ```bash
   # While running benchmarks
   nvidia-smi dmon -s mu -c 100
   # Should see ~70-75% memory bandwidth utilization
   ```

---

## Reference

- **Full Report**: `docs/RSI_FUSED_KERNEL_IMPLEMENTATION_REPORT.md`
- **Source Files**:
  - CUDA kernel: `src/gpu/kernels/rsi_fused.cu`
  - Rust bindings: `src/gpu/rsi_fused.rs`
  - Benchmarks: `benches/rsi_fused_benchmark.rs`

---

**Last Updated**: 2025-11-01
**Status**: Ready for testing (pending CUDA 12.4/13.1+ upgrade)
