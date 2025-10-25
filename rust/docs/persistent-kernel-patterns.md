# Persistent Kernel Patterns for Ultra-Low Latency

**Target**: Reduce kernel launch overhead by 50-90% via persistent kernels with cooperative groups

---

## Pattern 1: Persistent Kernel Architecture

### Problem
Traditional CUDA programming launches one kernel per operation:
- Launch overhead: ~5-10μs per kernel
- N operations = N × 10μs wasted on launches
- CPU-GPU synchronization cost multiplied

### Solution
Launch kernel once, process multiple tasks in loop with grid-wide synchronization.

### Before (Traditional)
```rust
// Launch kernel N times
for task in tasks {
    let result = compute_indicator(&device, task.data, task.params)?;
    results.push(result);
}
// Total overhead: N × 10μs
```

### After (Persistent)
```rust
// Launch kernel once, process all tasks
let mut batch = TaskBatch::new();
for task in tasks {
    batch.add_task(task.data, task.params);
}
let results = manager.execute_batch(&batch)?;
// Total overhead: 1 × 10μs
```

### CUDA Kernel (Persistent ROC)
```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void persistent_roc_kernel(
    const double** __restrict__ input_batch,    // Array of input pointers
    double** __restrict__ output_batch,          // Array of output pointers
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of ROC periods
    int num_tasks                                // Number of tasks to process
) {
    // Get grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();
    
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;
    
    // Process each task sequentially (persistent kernel pattern)
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* input = input_batch[task_id];
        double* output = output_batch[task_id];
        int n = sizes[task_id];
        int period = periods[task_id];
        
        // Grid-stride loop for this task's data
        for (int idx = global_tid; idx < n; idx += grid_size) {
            if (idx < period) {
                output[idx] = CUDART_NAN;
            } else {
                // ROC = (price[i] / price[i-period] - 1) * 100
                output[idx] = (input[idx] / input[idx - period] - 1.0) * 100.0;
            }
        }
        
        // Synchronize entire grid before next task
        grid.sync();
    }
}
```

### Rust Launch Pattern
```rust
use cudarc::driver::LaunchConfig;

impl PersistentKernelManager {
    pub fn execute_batch(&self, batch: &TaskBatch) -> Result<Vec<Vec<f64>>, GpuError> {
        // 1. Compile persistent kernel
        let ptx = compile_ptx(PERSISTENT_ROC_KERNEL)?;
        let module = self._device.context().load_module(&ptx)?;
        let kernel = module.get_function("persistent_roc_kernel")?;
        
        // 2. Allocate GPU memory for all tasks
        let mut d_inputs = Vec::new();
        let mut d_outputs = Vec::new();
        for task_data in &batch.inputs {
            d_inputs.push(self._device.copy_to_device(task_data)?);
            d_outputs.push(self._device.alloc_buffer(task_data.len())?);
        }
        
        // 3. Create pointer arrays (double**)
        let d_input_ptrs = create_device_pointer_array(&d_inputs)?;
        let d_output_ptrs = create_device_pointer_array(&d_outputs)?;
        
        // 4. Launch persistent kernel with cooperative groups
        let config = self.get_launch_config();
        let mut builder = self._device.stream.launch_builder(&kernel);
        builder.arg(&d_input_ptrs);
        builder.arg(&d_output_ptrs);
        builder.arg(&d_sizes);
        builder.arg(&d_periods);
        builder.arg(&(batch.len() as i32));
        
        unsafe {
            // Cooperative launch - all blocks must be simultaneously resident
            builder.launch_cooperative(config)?;
        }
        
        // 5. Copy results back
        let mut results = Vec::new();
        for d_output in &d_outputs {
            results.push(self._device.copy_to_host(d_output)?);
        }
        
        Ok(results)
    }
}
```

### Trade-offs
- ✅ **50-90% launch overhead reduction** for N > 10
- ✅ **Better GPU utilization** (fewer idle cycles between kernels)
- ✅ **Simpler host-side code** (one launch instead of N)
- ❌ **More complex kernel code** (cooperative groups, task loop)
- ❌ **Grid size constraints** (all blocks must be simultaneously resident)
- ❌ **Less flexibility** (tasks must have homogeneous computation pattern)

### When to Use
- **Use persistent kernels when**:
  - Processing N > 10 similar tasks
  - Launch overhead > 10% of compute time
  - Tasks have homogeneous computational pattern
  - GPU has sufficient resources for cooperative launch

- **Use traditional kernels when**:
  - N < 5 tasks (overhead of persistent setup)
  - Tasks are heterogeneous
  - Need fine-grained error handling per task
  - Grid size limits are restrictive

---

## Pattern 2: Cooperative Groups Synchronization

### Problem
Traditional `__syncthreads()` only synchronizes within a thread block. For persistent kernels processing multiple tasks, we need grid-wide synchronization to ensure Task N completes before Task N+1 starts.

### Solution
Use `cooperative_groups::this_grid().sync()` for grid-wide barrier.

### Before (Block-level sync)
```cuda
extern "C" __global__ void traditional_kernel(const double* input, double* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute
    if (idx < n) {
        output[idx] = compute(input[idx]);
    }
    
    __syncthreads();  // Only synchronizes this block
}
```

### After (Grid-level sync)
```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void persistent_kernel(
    const double** inputs, 
    double** outputs, 
    int num_tasks
) {
    cg::grid_group grid = cg::this_grid();
    
    for (int task = 0; task < num_tasks; task++) {
        // Compute for this task
        // ...
        
        grid.sync();  // Synchronizes ALL blocks before next task
    }
}
```

### Requirements
1. **Cooperative Launch**: Must use `cudaLaunchCooperativeKernel` API
2. **Grid Size Limits**: All blocks must fit simultaneously on GPU
3. **Device Support**: Requires compute capability 6.0+ (Pascal or later)

### Rust Integration (cudarc)
```rust
use cudarc::driver::LaunchConfig;

// Query cooperative launch limits
let max_grid_size = device.get_cooperative_launch_max_grid_size()?;

// Ensure grid fits within limits
let config = LaunchConfig {
    grid_dim: (min(max_grid_size, optimal_grid), 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 0,
};

// Launch with cooperative flag
unsafe {
    builder.launch_cooperative(config)?;
}
```

### Trade-offs
- ✅ **Enables persistent kernel pattern**
- ✅ **Explicit synchronization semantics**
- ❌ **Grid size limited by GPU resources**
- ❌ **Requires newer GPU (compute 6.0+)**
- ❌ **Slightly higher synchronization cost than block-level**

---

## Pattern 3: Grid-Stride Loop for Task Processing

### Problem
Different tasks may have different data sizes. Need flexible loop structure that handles arbitrary sizes efficiently.

### Solution
Use grid-stride loop: each thread processes multiple elements with stride = grid_size.

### Implementation
```cuda
extern "C" __global__ void persistent_kernel(...) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;
    
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        int n = sizes[task_id];
        
        // Grid-stride loop: each thread handles multiple elements
        for (int idx = global_tid; idx < n; idx += grid_size) {
            output[idx] = compute(input[idx]);
        }
        
        grid.sync();  // Synchronize before next task
    }
}
```

### Benefits
1. **Handles arbitrary data sizes**: Works with any N (not just multiples of block size)
2. **Coalesced memory access**: Threads access contiguous memory
3. **Load balancing**: Work distributed evenly across threads
4. **Scalability**: Same code works on different GPU sizes

### Example (ROC calculation)
```cuda
// Each thread processes elements at: tid, tid+stride, tid+2*stride, ...
for (int idx = global_tid; idx < n; idx += grid_size) {
    if (idx < period) {
        output[idx] = NAN;
    } else {
        output[idx] = (input[idx] / input[idx - period] - 1.0) * 100.0;
    }
}
```

### Trade-offs
- ✅ **Flexible** (arbitrary data sizes)
- ✅ **Efficient memory access** (coalesced)
- ✅ **Scalable** (works on any GPU)
- ❌ **Slightly more complex** than simple 1:1 thread-to-element mapping

---

## Pattern 4: Batched Memory Management

### Problem
Persistent kernel needs to process multiple datasets. How to manage GPU memory for N datasets efficiently?

### Approach 1: Array of Pointers (double**)
```rust
// Allocate each dataset separately
let mut d_inputs = Vec::new();
for task_data in batch.inputs {
    d_inputs.push(device.copy_to_device(task_data)?);
}

// Create pointer array on GPU
let d_input_ptrs = create_device_pointer_array(&d_inputs)?;

// Kernel accesses: input_batch[task_id][idx]
```

**Trade-offs**:
- ✅ Simple implementation
- ✅ Handles varying dataset sizes easily
- ❌ Less cache-friendly (scattered allocations)
- ❌ Extra indirection (pointer dereference)

### Approach 2: Contiguous Buffer with Offsets
```rust
// Concatenate all datasets into single buffer
let total_size: usize = batch.inputs.iter().map(|v| v.len()).sum();
let mut all_data = Vec::with_capacity(total_size);
let mut offsets = vec![0];

for task_data in batch.inputs {
    all_data.extend_from_slice(task_data);
    offsets.push(all_data.len());
}

let d_all_data = device.copy_to_device(&all_data)?;
let d_offsets = device.copy_to_device(&offsets)?;

// Kernel accesses: all_data[offsets[task_id] + idx]
```

**Trade-offs**:
- ✅ **Better cache locality** (contiguous memory)
- ✅ **No pointer indirection**
- ✅ **Single memory transfer** (faster copy)
- ❌ More complex indexing
- ❌ Requires knowing all sizes upfront

### Recommendation
- **Use Approach 1** for initial implementation (simpler)
- **Optimize to Approach 2** if profiling shows memory access is bottleneck

---

## Pattern 5: Error Handling in Persistent Kernels

### Problem
If one task fails in a persistent kernel, traditional error handling (early return) would skip all subsequent tasks.

### Solution
Per-task error flags that allow kernel to continue processing even if some tasks fail.

### Implementation
```cuda
extern "C" __global__ void persistent_kernel(
    ...,
    int* error_flags  // Output: error code per task
) {
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        // Check for invalid input
        if (periods[task_id] <= 0) {
            error_flags[task_id] = ERROR_INVALID_PERIOD;
            continue;  // Skip this task, process others
        }
        
        if (sizes[task_id] < periods[task_id]) {
            error_flags[task_id] = ERROR_INSUFFICIENT_DATA;
            continue;
        }
        
        // Process task
        for (int idx = global_tid; idx < sizes[task_id]; idx += grid_size) {
            output[idx] = compute(input[idx]);
        }
        
        error_flags[task_id] = ERROR_NONE;  // Success
        grid.sync();
    }
}
```

### Rust Side
```rust
// Check error flags after kernel execution
let error_flags = device.copy_to_host(&d_error_flags)?;
for (task_id, &error_code) in error_flags.iter().enumerate() {
    if error_code != ERROR_NONE {
        return Err(GpuError::TaskFailed(task_id, error_code));
    }
}
```

### Trade-offs
- ✅ **Graceful degradation** (process valid tasks even if some fail)
- ✅ **Detailed error reporting** (per-task error codes)
- ❌ Extra memory for error flags
- ❌ Kernel continues even if all tasks fail (waste of GPU time)

### Alternative: Early Termination Flag
```cuda
__shared__ int should_terminate;

for (int task_id = 0; task_id < num_tasks; task_id++) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        should_terminate = 0;  // Reset flag
    }
    __syncthreads();
    
    // Check for fatal error
    if (fatal_condition) {
        atomicOr(&should_terminate, 1);
    }
    grid.sync();
    
    if (should_terminate) {
        break;  // All blocks exit early
    }
    
    // Process task...
}
```

---

## Performance Expectations

### Overhead Reduction by Batch Size

| Batch Size | Traditional Overhead | Persistent Overhead | Reduction |
|------------|---------------------|---------------------|-----------|
| 1          | ~10μs               | ~10μs               | 0%        |
| 5          | ~50μs               | ~10μs               | 80%       |
| 10         | ~100μs              | ~10μs               | 90%       |
| 50         | ~500μs              | ~10μs               | 98%       |
| 100        | ~1ms                | ~10μs               | 99%       |

### Throughput Improvement (End-to-End)

**Scenario**: Batch calculation of 9 indicators on 1000 candles

| Approach     | Launch Overhead | Compute Time | Total Time | Throughput |
|--------------|-----------------|--------------|------------|------------|
| Traditional  | ~150μs (15×10)  | ~500μs       | ~650μs     | 1,538 ops/s |
| Persistent   | ~30μs (3×10)    | ~500μs       | ~530μs     | 1,887 ops/s |
| **Speedup**  | **80% reduction** | Same       | **18% faster** | **1.23x** |

**Note**: Actual speedup depends on compute time. For smaller kernels (faster compute), launch overhead becomes more significant and speedup increases.

### Crossover Point Analysis

| Dataset Size | Compute Time | Launch % | Persistent Beneficial? |
|--------------|--------------|----------|------------------------|
| 100 candles  | ~10μs        | 50%      | ✅ Yes (2x speedup)    |
| 1K candles   | ~50μs        | 17%      | ✅ Yes (1.2x speedup)  |
| 10K candles  | ~500μs       | 2%       | ⚠️ Marginal (~5%)     |
| 100K candles | ~5ms         | 0.2%     | ❌ No benefit          |

**Recommendation**: Use persistent kernels for datasets < 10K candles where launch overhead matters.

---

## Implementation Checklist

### Phase 1: Infrastructure ✅ COMPLETE
- [x] Create `src/gpu/persistent.rs` module
- [x] Define `PersistentKernelManager` struct
- [x] Define `TaskBatch` struct
- [x] Add CUDA kernel template with cooperative groups
- [x] Export from `src/gpu/mod.rs`
- [x] Create benchmark harness

### Phase 2: Simple Prototype ⏳ PENDING
- [ ] Implement `execute_batch()` method
- [ ] Compile and load persistent ROC kernel
- [ ] Allocate GPU memory for batch inputs/outputs
- [ ] Create device pointer arrays
- [ ] Launch kernel with cooperative groups
- [ ] Copy results back to host
- [ ] Unit tests for correctness
- [ ] Benchmark: Traditional vs Persistent

### Phase 3: Multi-Indicator ⏳ PENDING
- [ ] Implement persistent kernels for:
  - [ ] Williams %R (simple parallel)
  - [ ] CCI (simple parallel)
  - [ ] SMA (rolling window)
  - [ ] Bollinger Bands (rolling window)
  - [ ] RSI (sequential EMA)
  - [ ] MACD (sequential EMA)
- [ ] Integrate with `src/gpu/batch.rs`
- [ ] End-to-end benchmarks
- [ ] GPU profiling (Nsight Systems)

### Phase 4: Optimizations ⏳ FUTURE
- [ ] Contiguous memory layout (Approach 2)
- [ ] Device property queries for grid size
- [ ] Per-task error handling
- [ ] Adaptive batch sizing (auto-select traditional vs persistent)

---

## References

- **CUDA Cooperative Groups**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups
- **Persistent Threads**: CUDA C Best Practices Guide, Section 9.3
- **Grid-Stride Loops**: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
- **Launch Overhead**: Typically 5-10μs on modern GPUs (measured via Nsight Systems)

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-25  
**Author**: Claude (rust-latency-optimizer agent)
