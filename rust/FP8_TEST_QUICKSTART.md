# FP8 Test Quickstart

## Quick Test Run

```bash
# Run all FP8 tests (with JIT fallback)
cargo test --test fp8_wmma_tests --features gpu -- --nocapture

# Run specific test
cargo test --test fp8_wmma_tests --features gpu test_fp8_conversion -- --nocapture
```

## Build Pre-Compiled Kernels (Optional)

```bash
# Compile .cubin manually (requires nvcc and CUTLASS)
nvcc -cubin -arch=sm_89 -std=c++17 \
     -I/tmp/cutlass/include \
     -I/usr/local/cuda-13.0/include \
     -O3 -use_fast_math \
     -o target/fp8_kernels.cubin \
     src/gpu/kernels/fp8_cutlass.cu

# Verify .cubin exists
ls -lh target/fp8_kernels.cubin

# Run tests (will use .cubin if available)
cargo test --test fp8_wmma_tests --features gpu -- --nocapture
```

## Expected Test Output

### Without .cubin (JIT Fallback)
```
test fp8_tests::test_fp8_kernel_loading ... ok
⚠️  Pre-compiled .cubin not found
   Skipping AOT kernel loading test
   Hint: Run 'nvcc -o target/fp8_kernels.cubin ...' to build kernels
```

### With .cubin (AOT)
```
test fp8_tests::test_fp8_kernel_loading ... ok
✓ Found .cubin at: target/fp8_kernels.cubin
✓ FP8 kernel loaded successfully
  Kernel: fp8_matmul_cutlass
```

## Test Coverage

| Test | Runtime | Purpose |
|------|---------|---------|
| `test_fp8_support_detection` | ~50ms | Hardware check |
| `test_quantize_fp8_cpu_accuracy` | <1ms | CPU quantization |
| `test_fp8_kernel_loading` | ~100ms | .cubin loading |
| `test_fp8_conversion` | ~200ms | Round-trip accuracy |
| `test_fp8_matmul_accuracy` | ~500ms | 3 matrix sizes |
| `test_fp8_matmul_edge_cases` | ~300ms | Boundary tests |
| `test_fp8_batch_performance` | ~1s | Throughput benchmark |
| **Total** | **~2.2s** | Full suite |

## Troubleshooting

### GPU Not Found
```
⚠️  GPU not available: ...
   Skipping FP8 support test
```
**Solution**: Check `nvidia-smi` output. Tests will skip gracefully.

### FP8 Not Supported
```
⚠️  FP8 not supported: FP8 requires compute capability >= 8.9, found 7.5
```
**Solution**: Need Ada Lovelace GPU (RTX 3500 Ada, RTX 4000+). Tests will skip gracefully.

### Kernel Compilation Failed
```
⚠️  Kernel compilation failed: PTX compilation failed: ...
```
**Solution**: Check CUDA toolkit installation. Tests fall back to software simulation.

## Performance Targets

| Metric | Target | Typical (RTX 3500 Ada) |
|--------|--------|------------------------|
| Round-trip error | < 2% | ~0.01-0.5% |
| Matmul accuracy | < 2% rel error | ~0.5-1.5% |
| Time per 32×32 matrix | < 1000 μs | ~200-500 μs |
| Throughput | >1000 mat/sec | ~2000-5000 mat/sec |

## Hardware Requirements

- **GPU**: NVIDIA Ada Lovelace (CC 8.9+)
- **CUDA**: 13.0+
- **VRAM**: 2GB+ (tests use <100MB)
- **Driver**: 525+

## Quick Commands

```bash
# Full test suite with verbose output
cargo test --test fp8_wmma_tests --features gpu -- --nocapture

# Just accuracy tests (skip performance)
cargo test --test fp8_wmma_tests --features gpu test_fp8_matmul_accuracy

# Just performance test
cargo test --test fp8_wmma_tests --features gpu test_fp8_batch_performance -- --nocapture

# Check compilation only (no GPU needed)
cargo test --test fp8_wmma_tests --features gpu --no-run
```

## Files

- **Test Suite**: `tests/fp8_wmma_tests.rs` (599 lines)
- **CUDA Kernel**: `src/gpu/kernels/fp8_cutlass.cu`
- **Implementation**: `src/gpu/fp8_wmma.rs`
- **Documentation**: `docs/FP8_AOT_TEST_UPDATE_SUMMARY.md`
