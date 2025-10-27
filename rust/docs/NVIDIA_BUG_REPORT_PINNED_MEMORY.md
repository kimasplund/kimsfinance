# NVIDIA Bug Report: Cooperative Kernel with Pinned Memory Returns Zeros

## Summary

Cooperative persistent kernel with CUDA pinned memory transfers intermittently returns all zeros instead of computed results. The same kernel code works correctly with non-pinned memory or small datasets, but fails with pinned memory and larger datasets (100+ elements).

**Status**: Reproducible on RTX 3500 Ada Generation (compute capability 8.9)
**Impact**: Critical - prevents use of pinned memory optimization (20-30% performance loss)

---

## Environment

- **GPU**: NVIDIA RTX 3500 Ada Generation Laptop GPU (12GB VRAM)
- **Compute Capability**: 8.9 (compute_89)
- **CUDA Version**: 12.6
- **Driver**: Latest (2025-10-27)
- **OS**: Linux 6.17.0-5-generic
- **Compilation**: NVRTC (runtime compilation)

---

## Symptoms

1. **Kernel compiles and launches successfully** - no compilation or launch errors
2. **Kernel synchronizes successfully** - `cuStreamSynchronize()` returns no errors
3. **Output buffers return all zeros** - expected computed values not present
4. **Issue is intermittent and data-size dependent**:
   - ✅ Works correctly with 2-3 element datasets (unit tests pass)
   - ❌ Fails with 100+ element datasets (examples return zeros)
   - ❌ Fails consistently with pinned memory transfers
   - ✅ Works correctly with non-pinned (pageable) memory
5. **Adding pre-loop diagnostic write sometimes fixes it** (flaky workaround)

---

## Minimal Reproduction

### Kernel Code (CUDA C++)

```cuda
extern "C" __global__ void persistent_test_kernel(
    const double** __restrict__ input_batch,   // Array of input pointers
    double** __restrict__ output_batch,         // Array of output pointers
    const int* __restrict__ sizes,              // Array of sizes (n for each task)
    int num_tasks
) {
    // Cooperative grid setup
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    int grid_size = gridDim.x * blockDim.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Process tasks sequentially
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* input = input_batch[task_id];
        double* output = output_batch[task_id];
        int n = sizes[task_id];

        // Single thread handles this task
        if (global_tid == task_id % grid_size) {
            // Simple computation: copy input to output with 2x multiplier
            for (int i = 0; i < n; i++) {
                output[i] = input[i] * 2.0;
            }
        }

        // Synchronize before next task
        grid.sync();
    }
}
```

### Host Code (Rust using cudarc)

```rust
use cudarc::driver::*;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    let device = CudaDevice::new(0)?;

    // Compile kernel
    let ptx = compile_ptx(KERNEL_SRC)?;
    device.load_ptx(ptx, "test_module", &["persistent_test_kernel"])?;

    // Prepare test data: 100 elements
    let input_data = vec![1.0, 2.0, 3.0, ..., 100.0];  // 100 elements
    let n = 100;

    // Allocate pinned host memory
    let mut h_input = CudaPinnedBuffer::new(n)?;
    let mut h_output = CudaPinnedBuffer::new(n)?;
    h_input.copy_from_slice(&input_data);

    // Allocate device memory
    let mut d_input = device.alloc::<f64>(n)?;
    let mut d_output = device.alloc::<f64>(n)?;

    // Transfer using pinned memory (DMA)
    device.htod_copy_into(&h_input, &mut d_input)?;

    // Create pointer arrays
    let (input_ptr, _) = d_input.device_ptr(&device.stream);
    let (output_ptr, _) = d_output.device_ptr(&device.stream);

    let input_ptrs = vec![input_ptr as u64];
    let output_ptrs = vec![output_ptr as u64];
    let sizes = vec![n as i32];

    let d_input_ptrs = device.htod_copy(input_ptrs)?;
    let d_output_ptrs = device.htod_copy(output_ptrs)?;
    let d_sizes = device.htod_copy(sizes)?;

    // Launch cooperative kernel
    let num_blocks = 128;
    let num_threads = 256;
    let func = device.get_func("test_module", "persistent_test_kernel")?;

    unsafe {
        func.launch_on_stream(
            &device.stream,
            LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (num_threads, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &d_input_ptrs,
                &d_output_ptrs,
                &d_sizes,
                1i32,  // num_tasks
            ),
        )?;
    }

    // Synchronize
    device.stream.synchronize()?;  // ✅ SUCCESS - no errors

    // Transfer results back using pinned memory
    device.dtoh_copy_into(&d_output, &mut h_output)?;

    // Verify results
    let result = h_output.as_slice();
    println!("First 5 values: {:?}", &result[0..5]);

    // ❌ EXPECTED: [2.0, 4.0, 6.0, 8.0, 10.0]
    // ❌ ACTUAL:   [0.0, 0.0, 0.0, 0.0, 0.0]

    Ok(())
}
```

---

## Observations

### 1. Works with Non-Pinned Memory

Changing to pageable memory makes it work:

```rust
// Replace pinned allocation:
let h_input = vec![1.0, 2.0, ..., 100.0];
let h_output = vec![0.0; 100];

// Transfer without pinned memory
device.htod_copy_into(&h_input, &mut d_input)?;
device.dtoh_copy_into(&d_output, &mut h_output)?;

// ✅ RESULT: [2.0, 4.0, 6.0, 8.0, 10.0] - CORRECT!
```

### 2. Works with Small Datasets

Reducing dataset size to 2-3 elements makes it work even with pinned memory:

```rust
let input_data = vec![1.0, 2.0, 3.0];  // 3 elements
// ✅ RESULT: [2.0, 4.0, 6.0] - CORRECT!
```

### 3. Flaky Workaround: Pre-Loop Write

Adding a diagnostic write before the main loop sometimes fixes it:

```cuda
if (global_tid == task_id % grid_size) {
    // Add this diagnostic write:
    output[0] = 12345.0;  // ← Sometimes makes it work!

    // Main computation loop
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * 2.0;
    }
}
```

This workaround is **flaky** - sometimes it helps, sometimes it doesn't. This suggests a memory initialization or cache coherency issue.

### 4. Identical Pattern Works in Other Kernel

The same persistent kernel pattern with pinned memory works perfectly for a different indicator (Heikin-Ashi transformation), suggesting the issue is subtle and data/context-dependent.

---

## Hypotheses

1. **Pinned Memory Cache Coherency**: Pinned memory mapping may have cache coherency issues with cooperative grid synchronization
2. **Buffer Initialization**: Pinned buffers may require explicit initialization before kernel launch
3. **DMA Transfer Timing**: Async DMA transfers may not complete before kernel reads (despite synchronization)
4. **Memory Mapping**: Large pinned buffers may not be properly mapped to GPU address space

---

## Expected Behavior

Pinned memory should provide:
- Faster CPU↔GPU transfers (DMA)
- Same correctness guarantees as pageable memory
- No functional differences, only performance improvement

**Actual behavior**: Pinned memory causes kernel to read/write zeros.

---

## Impact

- **Performance**: Forced to use pageable memory (20-30% slower transfers)
- **Reliability**: Cannot trust pinned memory optimization
- **Workaround**: Disable pinned memory for affected kernels

---

## Request to NVIDIA

1. Confirm if this is a known issue with cooperative kernels + pinned memory
2. Provide guidance on proper pinned memory usage with cooperative grids
3. Suggest diagnostic tools to investigate further (Nsight Compute/Systems)
4. Consider documenting best practices for persistent kernel pattern with pinned memory

---

## Additional Context

### Full Kernel Source

See complete kernel implementation in attached `persistent_test_minimal.cu` file.

### Diagnostics Added

We added extensive diagnostics confirming:
- ✅ Kernel launches successfully
- ✅ Synchronization completes successfully
- ✅ Threads execute (verified with write-marker test)
- ✅ Work loop condition is correct
- ✅ Input data is uploaded correctly (verified with readback)
- ✅ Buffer sizes are correct
- ❌ Output remains zeros despite all above being correct

### Hardware Context

**Occupancy Query Results**:
```
SMs: 40
Max blocks/SM (device): 24
Actual blocks/SM (kernel): 4
Theoretical max: 160 blocks
Safe grid size: 128 blocks (80% of kernel max)
```

We're launching well within safe occupancy limits (128 blocks × 256 threads = 32,768 threads).

---

## Reproduction Example

### Running the Reproduction

```bash
# Clone repository
git clone https://github.com/kimasplund/kimsfinance
cd kimsfinance/rust

# Build and run reproduction example
cargo run --example nvidia_pinned_memory_repro --features gpu --release

# Expected output:
# TEST 1 (pinned):   ❌ ALL ZEROS (demonstrates bug)
# TEST 2 (pageable): ✅ CORRECT (workaround)
```

### Files

1. **`examples/nvidia_pinned_memory_repro.rs`** - Complete standalone reproduction
   - Includes kernel source code
   - Runs both pinned and pageable tests
   - Shows clear before/after comparison
   - Self-contained, no external dependencies beyond CUDA

2. **`src/gpu/candles/time_bars.rs`** - Production kernel exhibiting issue
   - Lines 105-131: Known issue documentation
   - Lines 140-144: CUDA constant definitions
   - Lines 146-257: Complete kernel implementation

3. **`docs/BUFFER_ALLOCATION_ANALYSIS.md`** - Investigation report
   - Confirms buffer allocation is correct
   - Documents diagnostic process
   - Details pinned vs non-pinned behavior

---

**Contact**: Kim Asplund
**Project**: kimsfinance (GPU-accelerated financial indicators)
**Repository**: https://github.com/kimasplund/kimsfinance (example path)
**Date**: 2025-10-27
