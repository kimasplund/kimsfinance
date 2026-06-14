# Tensor Core Benchmark Quickstart

**Status**: ⚠️  **PLACEHOLDER - AWAITING TENSOR CORE IMPLEMENTATION COMPLETION**

This benchmark suite will be ready to run once all 3 tensor core implementations are complete:
1. **Agent 1**: FP8 WMMA tensor cores
2. **Agent 2**: FP16 tensor cores
3. **Agent 3**: FP8 CUTLASS tensor cores

---

## Current Status

**Implemented**:
- ✅ Benchmark structure and test matrix
- ✅ Statistical rigor framework (n=10, p<0.05, confidence intervals)
- ✅ GFLOPS calculation methodology
- ✅ Accuracy validation framework
- ✅ Results template (`docs/TENSOR_CORE_BENCHMARK_RESULTS.md`)

**Not Yet Implemented** (blocked on tensor core implementations):
- ⏳ cuBLAS bindings for FP32/TF32 baseline (using native cuBLAS SGEMM)
- ⏳ FP16 tensor core integration (requires cuBLAS HGEMM bindings)
- ⏳ FP8 kernel compilation (requires CUDA kernel fixes)
- ⏳ Type conversions (GpuDevice currently only supports f64)

---

## Prerequisites

### 1. Hardware

- **GPU**: NVIDIA Ada Lovelace (RTX 3500 Ada, RTX 4000 series)
- **Compute Capability**: 8.9+
- **VRAM**: ≥8GB (12GB recommended for 4096² matrices)

### 2. Software

```bash
# Check GPU
nvidia-smi  # Should show RTX 3500 Ada or newer

# Check CUDA
nvcc --version  # Should show 12.4+

# Check driver
nvidia-smi | grep "Driver Version"  # Should show 580.82.07+

# Verify compute capability
nvidia-smi --query-gpu=compute_cap --format=csv  # Should show 8.9
```

### 3. Build Environment

```bash
# Install Rust (if not already)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install CUDA Toolkit 12.4+
# Download from https://developer.nvidia.com/cuda-downloads

# Verify environment
cargo --version  # 1.70+
nvcc --version   # 12.4+
```

---

## Running the Benchmark

### Quick Test (10 minutes)

```bash
# Run single precision comparison (once implemented)
cargo bench --features gpu --bench tensor_core_benchmark -- throughput/fp32
cargo bench --features gpu --bench tensor_core_benchmark -- throughput/fp16
```

### Full Benchmark Suite (60-90 minutes)

```bash
# Run all benchmarks with detailed output
cargo bench --features gpu --bench tensor_core_benchmark -- --verbose 2>&1 | tee tensor_core_results.txt

# This will run:
# 1. Matrix Multiplication Throughput (4 sizes × 4 precisions × 10 iterations)
# 2. Genetic Optimizer Workload (3 precisions × 10 iterations)
# 3. Conversion Overhead (2 conversion types × 4 sizes × 10 iterations)
# 4. Accuracy Analysis (2 precision tests)
```

### Individual Benchmark Groups

```bash
# Throughput benchmarks only
cargo bench --features gpu --bench tensor_core_benchmark -- throughput

# Genetic optimizer simulation only
cargo bench --features gpu --bench tensor_core_benchmark -- genetic_optimizer

# Conversion overhead only
cargo bench --features gpu --bench tensor_core_benchmark -- conversion

# Accuracy tests only
cargo test --features gpu --release test_fp8_accuracy -- --nocapture
cargo test --features gpu --release test_fp16_accuracy -- --nocapture
```

---

## Understanding the Output

### Criterion.rs Benchmark Output

```
throughput/fp32_baseline/512x512
                        time:   [0.85 ms 0.87 ms 0.90 ms]
Found 2 outliers among 10 measurements (20.00%)
  2 (20.00%) high mild

throughput/tf32_tensor/512x512
                        time:   [0.11 ms 0.12 ms 0.13 ms]
                        change: [-86.2% -86.0% -85.7%] (p = 0.00 < 0.05)
                        Performance has improved.
```

**Interpretation**:
- **time**: Mean execution time with 95% confidence interval
- **change**: Performance improvement vs previous run (if available)
- **p-value**: Statistical significance (p < 0.05 = significant)
- **Speedup**: 0.87ms / 0.12ms = 7.25x (close to expected 8x for TF32)

### GFLOPS Calculation

```
Matrix size: 512×512
Operations: 2 × 512 × 512 × 512 = 268,435,456 ops
Time: 0.12 ms = 0.00012 sec
GFLOPS: 268.4M ops / 0.00012 sec = 2,237 GFLOPS
```

### Statistical Validation

```
Statistical Analysis:
  FP32 baseline: mean=0.87ms, std=0.03ms, CV=3.4%
  TF32 tensor:   mean=0.12ms, std=0.01ms, CV=8.3%
  Speedup:       7.25x
  p-value:       0.001
  Cohen's d:     12.5 (large effect)
  Confidence:    95% CI: [6.8x, 7.7x]
  Verdict:       ✓ PASS (expected 8x ± 20%)
```

---

## Troubleshooting

### Issue: "GPU feature not enabled"

```
⚠️  GPU feature not enabled, skipping tensor core benchmarks
```

**Solution**:
```bash
cargo bench --features gpu --bench tensor_core_benchmark
#             ^^^^^^^^^^^^^^ Important!
```

### Issue: "FP8 tensor cores not supported"

```
⚠️  FP8 tensor cores not supported on this GPU, skipping
   Required: Compute capability >= 8.9 (Ada Lovelace or newer)
```

**Solution**:
- Check GPU model: `nvidia-smi`
- Check compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`
- If < 8.9, FP8 benchmarks will be skipped (FP16 will still run)

### Issue: "Failed to compile FP8 kernel"

```
warning: Failed to compile FP8 WMMA kernel (non-critical)
```

**Current Status**: This is **expected** until Agent 1 completes CUDA kernel fixes.

**Workaround**: Benchmarks will run with placeholder data once kernel compilation is fixed.

### Issue: Type mismatches (e.g., "expected &[f64], found &Vec<f32>")

**Current Status**: This is **expected** until GpuDevice API is extended to support f32/f16.

**Resolution**: Will be fixed as part of tensor core implementation.

### Issue: Out of memory

```
error: CUDA_ERROR_OUT_OF_MEMORY
```

**Solution**:
```bash
# Reduce matrix sizes (edit benches/tensor_core_benchmark.rs)
const THROUGHPUT_SIZES: &[usize] = &[512, 1024];  // Remove 2048, 4096

# Or reduce sample size
const SAMPLE_SIZE: usize = 5;  // Instead of 10
```

---

## Interpreting Results

### Pass/Fail Criteria

**Throughput Benchmarks**:
- ✓ PASS: Speedup within 80-120% of expected (accounts for real-world overhead)
- ✓ PASS: Statistical significance p < 0.05
- ✗ FAIL: Speedup < 80% of expected OR p ≥ 0.05

**Expected Speedups** (RTX 3500 Ada, sm_89):
| Comparison      | Expected Speedup | Pass Range     |
|-----------------|------------------|----------------|
| TF32 vs FP32    | 8.0x             | 6.4x - 9.6x    |
| FP16 vs TF32    | 2.0x             | 1.6x - 2.4x    |
| FP8 vs TF32     | 2.0x             | 1.6x - 2.4x    |

**Accuracy Benchmarks**:
- ✓ PASS (FP8): Maximum relative error < 5%
- ✓ PASS (FP16): Maximum relative error < 1%
- ⚠️ WARNING (FP8): Max error 5-10% (acceptable for genetic optimizer)
- ✗ FAIL: Max error > 10%

### Example Results Interpretation

**Scenario 1: Perfect Result**
```
TF32 vs FP32: 7.8x speedup, p=0.001, CI=[7.2x, 8.4x] ✓ PASS
FP16 vs TF32: 1.9x speedup, p=0.003, CI=[1.7x, 2.1x] ✓ PASS
FP8 accuracy: Max error 3.2% ✓ PASS
```

**Recommendation**: Use FP16 for genetic optimizer exploration, FP64 for refinement.

**Scenario 2: Conversion Overhead High**
```
FP8 conversion overhead: 45% of total time ⚠️ WARNING
Crossover point: 1024×1024 (larger than expected 512×512)
```

**Recommendation**: Only use FP8 for matrices ≥1024×1024. Use FP16 for smaller matrices.

**Scenario 3: Accuracy Too Low**
```
FP8 accuracy: Max error 12.5% ✗ FAIL (expected <10%)
Genetic optimizer quality: 85% retention (expected >90%)
```

**Recommendation**: Use FP16 instead of FP8 for genetic optimizer. FP8 only for rough approximations.

---

## Next Steps After Benchmark Completion

### 1. Analyze Results

```bash
# Fill in template with actual results
vim docs/TENSOR_CORE_BENCHMARK_RESULTS.md

# Replace all [TBD] placeholders with actual data
# Add statistical analysis
# Generate recommendations
```

### 2. Validate Against Hardware Specs

- Compare GFLOPS to RTX 3500 Ada specs (~240 GFLOPS FP16 peak)
- Check SM utilization with `nsys` profiling
- Verify memory bandwidth efficiency

### 3. Integration Recommendations

Based on results, integrate tensor cores into:
- Genetic algorithm optimizer (hybrid FP8/FP16/FP64 strategy)
- Batch backtest operations
- Technical indicator calculations (where precision allows)

### 4. CI/CD Integration

```bash
# Add performance regression tests
cargo test --features gpu --release test_fp8_accuracy
cargo test --features gpu --release test_fp16_accuracy

# Add to GitHub Actions (if applicable)
# Run on every PR to detect performance regressions
```

---

## Benchmark Maintenance

### Updating Expected Results

After hardware upgrades or CUDA version changes:

```bash
# Re-establish baseline
cargo bench --features gpu --bench tensor_core_benchmark -- --save-baseline v1.0

# Compare future runs
cargo bench --features gpu --bench tensor_core_benchmark -- --baseline v1.0
```

### Adding New Matrix Sizes

Edit `benches/tensor_core_benchmark.rs`:

```rust
const THROUGHPUT_SIZES: &[usize] = &[256, 512, 1024, 2048, 4096, 8192];
//                                   ^^^^ Add smaller sizes for edge case testing
//                                                                 ^^^^ Add larger if VRAM allows
```

### Adding New Precision Formats

Example: INT8 quantization:

```rust
fn bench_int8_tensor_cores(c: &mut Criterion) {
    // Similar structure to FP8 benchmark
    // Use cuBLAS GEMM with INT8 inputs
}
```

---

## References

### Documentation

- **NVIDIA Tensor Core Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-cores
- **cuBLAS Documentation**: https://docs.nvidia.com/cuda/cublas/index.html
- **CUTLASS Documentation**: https://github.com/NVIDIA/cutlass

### Kimsfinance Docs

- `/home/kim/projects/kimsfinance/rust/docs/TENSOR_CORE_BENCHMARK_RESULTS.md` - Results template
- `/home/kim/projects/kimsfinance/rust/benches/tensor_core_benchmark.rs` - Benchmark source
- `/home/kim/projects/kimsfinance/rust/benches/statistics.rs` - Statistical validation module

### Related Benchmarks

- `benches/fp8_tensor_cores.rs` - Original FP8-only benchmark
- `benches/genetic_optimizer_precision.rs` - Genetic optimizer precision benchmark
- `benches/genetic_optimizer_comparison.rs` - CPU vs GPU genetic optimizer

---

## FAQ

**Q: Why only 10 iterations? Isn't that too small for statistics?**

A: For GPU benchmarks with GFLOPS measurements, n=10 is sufficient because:
- GPU execution is highly deterministic (low variance)
- Typical CV < 5% for GFLOPS measurements
- Larger n doesn't significantly improve confidence intervals
- Trade-off: Statistical power vs benchmark runtime (10 iterations = ~60 min)

**Q: Why not use cuBLAS directly instead of custom kernels?**

A: We do both:
- **cuBLAS SGEMM/HGEMM**: Used for FP32/TF32/FP16 baselines (industry standard)
- **Custom CUTLASS kernels**: Used for FP8 (not yet in cuBLAS as of CUDA 12.4)
- **Validation**: Compare custom kernels to cuBLAS for accuracy

**Q: What if my GPU doesn't support FP8 (sm < 8.9)?**

A: Benchmarks gracefully skip FP8 tests and run only:
- FP32 baseline (CUDA cores)
- TF32 tensor cores (if sm ≥ 8.0, Ampere+)
- FP16 tensor cores (if sm ≥ 7.0, Volta+)

**Q: Can I run this on AMD GPUs?**

A: Not currently. This benchmark is NVIDIA-specific (CUDA, cuBLAS, tensor cores).
- AMD equivalent: ROCm, rocBLAS, Matrix Cores (CDNA architecture)
- Porting would require significant changes

**Q: How do I profile GPU kernel execution?**

A: Use NVIDIA Nsight tools:

```bash
# Timeline profiling
nsys profile --stats=true cargo bench --features gpu --bench tensor_core_benchmark

# Kernel analysis
ncu --set full cargo bench --features gpu --bench tensor_core_benchmark
```

---

**Quickstart Version**: 1.0.0
**Last Updated**: 2025-11-01
**Status**: ⚠️  AWAITING IMPLEMENTATION COMPLETION

**Next Steps**: Run this benchmark once all tensor core implementations are complete, then fill in `docs/TENSOR_CORE_BENCHMARK_RESULTS.md` with actual results.
