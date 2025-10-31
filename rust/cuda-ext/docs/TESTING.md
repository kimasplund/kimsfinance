# Testing Guide: Stream-Ordered Memory Allocation

Comprehensive guide for testing the stream-ordered memory allocation implementation.

## Table of Contents

1. [Test Overview](#test-overview)
2. [Running Tests](#running-tests)
3. [Test Categories](#test-categories)
4. [Benchmarking](#benchmarking)
5. [Memory Leak Detection](#memory-leak-detection)
6. [CI/CD Integration](#cicd-integration)

---

## Test Overview

The test suite covers:

- ✅ **Basic functionality** (15+ tests)
- ✅ **Edge cases** (zero-size, large allocations)
- ✅ **Error handling** (out-of-memory, invalid ops)
- ✅ **Concurrency** (multiple streams, thread safety)
- ✅ **Memory safety** (leak detection, proper cleanup)
- ✅ **Performance** (benchmarks with statistical analysis)

**Total Coverage:**
- Unit tests: 15+
- Integration tests: 5+
- Benchmarks: 3+
- Examples: 3+

---

## Running Tests

### Prerequisites

```bash
# CUDA driver >= 11.2
nvidia-smi

# Rust toolchain
rustc --version

# GPU device available
nvidia-smi -L
```

### Run All Tests

```bash
# Run all tests (requires GPU)
cargo test --test stream_malloc_comprehensive -- --ignored --test-threads=1

# Explanation:
# --test stream_malloc_comprehensive: Run specific test suite
# --ignored: Run tests marked with #[ignore] (GPU tests)
# --test-threads=1: Run serially (GPU resources)
```

### Run Specific Test

```bash
# Run single test
cargo test --test stream_malloc_comprehensive test_basic_alloc_free -- --ignored

# Run tests matching pattern
cargo test --test stream_malloc_comprehensive test_alloc -- --ignored
```

### Run with Output

```bash
# Show println! output
cargo test --test stream_malloc_comprehensive -- --ignored --nocapture

# Show detailed output
cargo test --test stream_malloc_comprehensive -- --ignored --nocapture --test-threads=1
```

---

## Test Categories

### 1. Basic Functionality Tests

**Purpose:** Verify core allocation/free operations work correctly.

```bash
# Run basic tests
cargo test --test stream_malloc_comprehensive \
    test_allocator_creation \
    test_basic_alloc_free \
    test_small_allocation \
    -- --ignored
```

**Tests:**
- `test_allocator_creation`: Verify allocator can be created
- `test_basic_alloc_free`: Basic alloc/free cycle
- `test_small_allocation`: 1-byte allocation (edge case)
- `test_multiple_allocations`: 100+ allocations

**Expected:** All pass in <1s

---

### 2. Edge Case Tests

**Purpose:** Test boundary conditions and unusual inputs.

```bash
# Run edge case tests
cargo test --test stream_malloc_comprehensive \
    test_zero_size_allocation \
    test_large_allocation \
    test_mixed_allocation_sizes \
    -- --ignored
```

**Tests:**
- `test_zero_size_allocation`: Zero-byte allocation
- `test_large_allocation`: 1GB allocation (may fail on small GPUs)
- `test_mixed_allocation_sizes`: Various sizes (1KB to 16MB)

**Expected:**
- Zero-size: May succeed or fail (implementation-defined)
- Large: May fail if insufficient GPU memory
- Mixed: All pass

---

### 3. Stress Tests

**Purpose:** Verify stability under heavy load.

```bash
# Run stress tests
cargo test --test stream_malloc_comprehensive \
    test_alloc_free_stress \
    test_concurrent_style_allocations \
    -- --ignored
```

**Tests:**
- `test_alloc_free_stress`: 1000 rapid alloc/free cycles
- `test_concurrent_style_allocations`: 4 "streams" × 50 allocations

**Expected:** All pass in <5s

---

### 4. Memory Safety Tests

**Purpose:** Ensure no memory leaks or corruption.

```bash
# Run memory tests
cargo test --test stream_malloc_comprehensive \
    test_memory_reuse \
    test_allocator_drop_cleanup \
    -- --ignored --nocapture
```

**Tests:**
- `test_memory_reuse`: Verify pool reuses memory
- `test_allocator_drop_cleanup`: Proper cleanup on drop

**Expected:**
- Memory reuse: >50% reuse rate
- Drop: No errors or warnings

---

### 5. Thread Safety Tests

**Purpose:** Verify thread safety (Send + Sync).

```bash
# Run thread safety tests
cargo test --test stream_malloc_comprehensive \
    test_send_sync_traits \
    test_shared_allocator \
    -- --ignored
```

**Tests:**
- `test_send_sync_traits`: Compile-time trait verification
- `test_shared_allocator`: Arc-based sharing across threads

**Expected:** All pass (tests thread safety at compile time)

---

## Benchmarking

### Run Benchmarks

```bash
# Run Criterion benchmarks
cargo bench --bench stream_malloc

# Run example benchmark
cargo run --release --example stream_malloc_benchmark

# Run basic example
cargo run --release --example stream_allocation_basics

# Run concurrent example
cargo run --release --example stream_allocation_concurrent
```

### Expected Benchmark Results

| Benchmark | Traditional | Stream-Ordered | Speedup |
|-----------|-------------|----------------|---------|
| 1KB allocation | ~12ms | ~7ms | 1.7x |
| 1MB allocation | ~10ms | ~6ms | 1.7x |
| 10MB allocation | ~15ms | ~10ms | 1.5x |
| 1000 allocations | 1.2-1.5s | 0.8-1.0s | 1.2-1.5x |

### Interpreting Results

**Speedup >= 1.5x:** ✅ Excellent!

**Speedup 1.2-1.5x:** ✅ Expected range

**Speedup < 1.2x:** ⚠️ Investigate:
- GPU memory caching (run multiple times)
- CUDA version (13.0+ is faster)
- System bottleneck (other GPU processes)
- Hardware differences (older GPUs may see less benefit)

---

## Memory Leak Detection

### Using CUDA Memcheck

```bash
# Run with cuda-memcheck (if available)
cuda-memcheck --leak-check full \
    cargo test --test stream_malloc_comprehensive \
    test_alloc_free_stress -- --ignored

# Expected output:
# ========= LEAK SUMMARY =========
# ========= 0 bytes leaked
# ========= No errors found
```

### Using Valgrind (for host-side leaks)

```bash
# Check for host memory leaks
valgrind --leak-check=full --show-leak-kinds=all \
    cargo test --test stream_malloc_comprehensive -- --ignored

# Expected: 0 definitely lost, 0 indirectly lost
```

### Manual Verification

```rust
// In test:
let initial_free = get_free_memory()?;

// ... allocate and free many times ...

let final_free = get_free_memory()?;

// Memory should be approximately the same
assert!((initial_free as i64 - final_free as i64).abs() < 1024 * 1024,
        "Memory leak detected: {} bytes", initial_free - final_free);
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: CUDA Tests

on: [push, pull_request]

jobs:
  test-cuda:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:13.0.0-devel-ubuntu22.04

    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        run: |
          curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
          echo "$HOME/.cargo/bin" >> $GITHUB_PATH

      - name: Check CUDA
        run: nvidia-smi

      - name: Run Tests
        run: |
          cd rust/cuda-ext
          cargo test --test stream_malloc_comprehensive -- --ignored --test-threads=1

      - name: Run Benchmarks
        run: |
          cd rust/cuda-ext
          cargo run --release --example stream_malloc_benchmark
```

### GitLab CI Example

```yaml
cuda-tests:
  image: nvidia/cuda:13.0.0-devel-ubuntu22.04
  tags:
    - gpu
  script:
    - curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    - source $HOME/.cargo/env
    - cd rust/cuda-ext
    - cargo test --test stream_malloc_comprehensive -- --ignored --test-threads=1
    - cargo run --release --example stream_malloc_benchmark
```

---

## Test Maintenance

### Adding New Tests

1. Add test to `tests/stream_malloc_comprehensive.rs`
2. Mark with `#[ignore]` if requires GPU
3. Document expected behavior
4. Run test: `cargo test test_name -- --ignored`

**Example:**
```rust
/// Test [feature description]
#[test]
#[ignore] // Requires GPU
fn test_new_feature() {
    let context = Arc::new(CudaContext::new(0).expect("GPU init failed"));
    let allocator = StreamOrderedAllocator::new(0).expect("Allocator creation failed");

    // Test logic...

    assert!(condition, "Failure message");
}
```

### Updating Benchmarks

1. Add benchmark to `benches/stream_malloc.rs`
2. Use Criterion for statistical analysis
3. Run: `cargo bench --bench stream_malloc`

**Example:**
```rust
fn bench_new_feature(c: &mut Criterion) {
    let context = Arc::new(CudaContext::new(0).unwrap());
    let allocator = StreamOrderedAllocator::new(0).unwrap();

    c.bench_function("new_feature", |b| {
        b.iter(|| {
            // Benchmark logic
        });
    });
}
```

---

## Troubleshooting Tests

### Test Failure: "Failed to initialize GPU"

**Cause:** No GPU available or CUDA driver issue

**Solution:**
```bash
# Check GPU
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify device permissions
ls -l /dev/nvidia*
```

### Test Failure: "Allocation failed"

**Cause:** Insufficient GPU memory

**Solution:**
```bash
# Check available memory
nvidia-smi --query-gpu=memory.free --format=csv

# Close other GPU processes
nvidia-smi | grep python  # or other processes

# Reduce test allocation sizes
```

### Test Hangs

**Cause:** GPU deadlock or timeout

**Solution:**
```bash
# Reset GPU
nvidia-smi --gpu-reset

# Run with timeout
timeout 60s cargo test test_name -- --ignored

# Check for infinite loops in test
```

### Benchmark Variance

**Cause:** GPU state, thermal throttling, other processes

**Solution:**
```bash
# Reset GPU state
nvidia-smi --gpu-reset

# Run multiple times
for i in {1..5}; do
    cargo bench --bench stream_malloc
done

# Check GPU clocks
nvidia-smi --query-gpu=clocks.current.graphics --format=csv

# Ensure no thermal throttling
nvidia-smi --query-gpu=temperature.gpu --format=csv
```

---

## Summary

**Running Tests:**
```bash
# Quick test (basic functionality)
cargo test --test stream_malloc_comprehensive -- --ignored --test-threads=1

# Full test suite
cargo test --test stream_malloc_comprehensive -- --ignored --nocapture --test-threads=1

# Benchmarks
cargo bench --bench stream_malloc
cargo run --release --example stream_allocation_basics
cargo run --release --example stream_allocation_concurrent
```

**Expected Results:**
- All tests pass
- Benchmarks show 1.2-1.5x speedup
- No memory leaks
- No crashes or hangs

**Common Issues:**
- No GPU: Skip tests (expected)
- Out of memory: Reduce allocation sizes
- Low speedup: Check CUDA version, GPU state
- Hangs: Reset GPU with `nvidia-smi --gpu-reset`

---

For more information, see:
- [API Reference](API_REFERENCE.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Examples](../examples/)
