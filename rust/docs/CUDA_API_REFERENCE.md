# CUDA 13.0 API Reference for Financial Indicators

**Target Hardware:** RTX 3500 Ada Generation (Compute Capability 8.9)  
**CUDA Version:** 13.0  
**Document Date:** 2025-10-26  
**Audience:** kimsfinance Rust developers implementing GPU-accelerated financial indicators

---

## Table of Contents

1. [CUDA 13.0 Features Relevant to Financial Indicators](#cuda-130-features)
2. [Ada Lovelace Architecture Features](#ada-architecture-features)
3. [Memory Management APIs](#memory-management-apis)
4. [Kernel Launch and Execution](#kernel-launch-execution)
5. [Performance Optimization APIs](#performance-optimization-apis)
6. [Code Examples for Common Patterns](#code-examples)
7. [cudarc API Mapping](#cudarc-api-mapping)

---

## CUDA 13.0 Features Relevant to Financial Indicators {#cuda-130-features}

### 1. Stream-Ordered Memory Allocator

**Status:** CUDA 11.2+ (improved in 13.0)  
**cudarc 0.17.3 Support:** ❌ (requires FFI)  
**Relevance:** HIGH for batch processing and hybrid indicators

#### C/C++ API

```cpp
// Allocate memory asynchronously on stream
cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream);

// Free memory asynchronously on stream
cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream);

// Query memory pool properties
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device);
cudaError_t cudaMemPoolSetAttribute(
    cudaMemPool_t memPool,
    cudaMemPoolAttr attr,
    void* value
);
```

#### Rust FFI Wrapper (kimsfinance)

```rust
use std::ffi::c_void;

#[repr(C)]
pub struct cudaStream_t(*mut c_void);

#[repr(C)]
pub struct cudaMemPool_t(*mut c_void);

#[repr(u32)]
pub enum cudaMemPoolAttr {
    ReleaseThreshold = 0,
    ReuseFollowEventDependencies = 1,
    ReuseAllowOpportunistic = 2,
    ReuseAllowInternalDependencies = 3,
}

#[link(name = "cudart")]
extern "C" {
    pub fn cudaMallocAsync(
        dev_ptr: *mut *mut c_void,
        size: usize,
        stream: cudaStream_t,
    ) -> u32; // cudaError_t
    
    pub fn cudaFreeAsync(
        dev_ptr: *mut c_void,
        stream: cudaStream_t,
    ) -> u32;
    
    pub fn cudaDeviceGetDefaultMemPool(
        mem_pool: *mut cudaMemPool_t,
        device: i32,
    ) -> u32;
}

// Safe wrapper
impl GpuDevice {
    pub fn alloc_stream_ordered(&self, len: usize) -> Result<CudaSlice<f64>, GpuError> {
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let size = len * std::mem::size_of::<f64>();
            
            // Get raw stream handle (assumes cudarc exposes this)
            let stream_handle = self.stream.as_raw_cudaStream_t();
            
            let err = cudaMallocAsync(&mut ptr, size, stream_handle);
            if err != 0 {
                return Err(GpuError::AllocationError(
                    format!("cudaMallocAsync failed with error {}", err)
                ));
            }
            
            // Wrap in CudaSlice (requires custom implementation or unsafe casting)
            Ok(/* ... */)
        }
    }
}
```

#### Performance Characteristics

| Metric | Traditional (cudaMalloc) | Stream-Ordered (cudaMallocAsync) |
|--------|--------------------------|----------------------------------|
| Allocation Time (1 MB) | 50-100 μs | 5-10 μs (10-20x faster) |
| Host Blocking | Yes (global sync) | No (async on stream) |
| Concurrency | Blocks all streams | Stream-local |
| Best For | Long-lived buffers | Frequent allocation/deallocation |

#### Use Cases in kimsfinance

1. **Hybrid RSI Pipeline:**
   - Allocate gains/losses buffers: 2 × 100K × 8 bytes = 1.6 MB
   - Traditional: 100 μs allocation overhead
   - Stream-ordered: 10 μs allocation overhead
   - **Savings:** 90 μs per RSI call

2. **Batch Indicator Processing:**
   - Allocate 10 output buffers: 10 × 100K × 8 bytes = 8 MB
   - Traditional: 500 μs allocation overhead
   - Stream-ordered: 50 μs allocation overhead
   - **Savings:** 450 μs per batch

---

### 2. CUDA Graphs

**Status:** CUDA 10.0+ (improved memory management in 13.0)  
**cudarc 0.17.3 Support:** ⚠️ Placeholder API (not functional)  
**Relevance:** HIGH for batch processing (>5 indicators)

#### C/C++ API

```cpp
// Graph capture workflow
cudaGraph_t graph;
cudaGraphExec_t exec_graph;

// Step 1: Begin stream capture
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);

// Step 2: Launch kernels (recorded, not executed)
kernel1<<<grid, block, 0, stream>>>(...);
kernel2<<<grid, block, 0, stream>>>(...);
kernel3<<<grid, block, 0, stream>>>(...);

// Step 3: End capture and get graph
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph);

// Step 4: Instantiate graph for execution
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, u64 flags);

// Step 5: Launch graph (low overhead!)
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);

// Optional: Update graph parameters without re-capture
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo);

// Cleanup
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
```

#### Rust FFI Wrapper (kimsfinance)

```rust
#[repr(C)]
pub struct cudaGraph_t(*mut c_void);

#[repr(C)]
pub struct cudaGraphExec_t(*mut c_void);

#[repr(u32)]
pub enum cudaStreamCaptureMode {
    Global = 0,
    ThreadLocal = 1,
    Relaxed = 2,
}

#[link(name = "cuda")]
extern "C" {
    pub fn cudaStreamBeginCapture(
        stream: cudaStream_t,
        mode: cudaStreamCaptureMode,
    ) -> u32;
    
    pub fn cudaStreamEndCapture(
        stream: cudaStream_t,
        graph: *mut cudaGraph_t,
    ) -> u32;
    
    pub fn cudaGraphInstantiate(
        exec_graph: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        flags: u64,
    ) -> u32;
    
    pub fn cudaGraphLaunch(
        exec_graph: cudaGraphExec_t,
        stream: cudaStream_t,
    ) -> u32;
}

// Safe wrapper (see src/gpu/cuda_graphs.rs for full implementation)
impl IndicatorGraphBuilder {
    pub fn begin_capture(&mut self) -> Result<(), GpuError> {
        unsafe {
            let stream = self.device.stream.as_raw_cudaStream_t();
            let err = cudaStreamBeginCapture(stream, cudaStreamCaptureMode::ThreadLocal);
            if err != 0 {
                return Err(GpuError::ExecutionError(
                    format!("cudaStreamBeginCapture failed: {}", err)
                ));
            }
            self.state = GraphState::Capturing;
            Ok(())
        }
    }
    
    pub fn end_capture(mut self) -> Result<IndicatorGraph, GpuError> {
        unsafe {
            let mut graph: cudaGraph_t = cudaGraph_t(std::ptr::null_mut());
            let stream = self.device.stream.as_raw_cudaStream_t();
            
            let err = cudaStreamEndCapture(stream, &mut graph);
            if err != 0 {
                return Err(GpuError::ExecutionError(
                    format!("cudaStreamEndCapture failed: {}", err)
                ));
            }
            
            // Instantiate graph
            let mut exec_graph: cudaGraphExec_t = cudaGraphExec_t(std::ptr::null_mut());
            let err = cudaGraphInstantiate(&mut exec_graph, graph, 0);
            if err != 0 {
                return Err(GpuError::ExecutionError(
                    format!("cudaGraphInstantiate failed: {}", err)
                ));
            }
            
            Ok(IndicatorGraph {
                device: self.device,
                exec_graph,
                graph,
            })
        }
    }
}

impl IndicatorGraph {
    pub fn launch(&self) -> Result<(), GpuError> {
        unsafe {
            let stream = self.device.stream.as_raw_cudaStream_t();
            let err = cudaGraphLaunch(self.exec_graph, stream);
            if err != 0 {
                return Err(GpuError::ExecutionError(
                    format!("cudaGraphLaunch failed: {}", err)
                ));
            }
            Ok(())
        }
    }
}
```

#### Performance Characteristics

| Metric | Traditional Launches | CUDA Graphs |
|--------|---------------------|-------------|
| Kernel Launch Overhead | 5-10 μs per kernel | 2-3 μs total (all kernels) |
| CPU Overhead | High (per kernel) | Minimal (one call) |
| Best For | Single/few kernels | Batch of 5+ kernels |

#### Use Cases in kimsfinance

**Batch Indicator Processing (10 indicators, 1000 iterations):**

```rust
// Without CUDA Graphs (current)
for _ in 0..1000 {
    launch_roc_kernel();     // 5 μs overhead
    launch_rsi_kernel();     // 5 μs overhead
    launch_stochastic_kernel(); // 5 μs overhead
    // ... 7 more indicators
    synchronize(); // Total: 10 × 5 μs = 50 μs overhead
}
// Total overhead: 50 μs × 1000 = 50 ms

// With CUDA Graphs (optimized)
let mut builder = IndicatorGraphBuilder::new(&device)?;
builder.begin_capture()?;
launch_roc_kernel();
launch_rsi_kernel();
launch_stochastic_kernel();
// ... 7 more indicators
let graph = builder.end_capture()?;

for _ in 0..1000 {
    graph.launch()?; // 3 μs overhead (all 10 kernels!)
}
// Total overhead: 3 μs × 1000 = 3 ms
// Savings: 47 ms (94% reduction!)
```

---

### 3. L2 Cache Access Hints

**Status:** CUDA 11.0+  
**cudarc 0.17.3 Support:** ❌ (requires FFI)  
**Relevance:** MEDIUM (Ada's 32 MB L2 cache)

#### C/C++ API

```cpp
// Set L2 cache persisting access window for a stream
cudaStreamAttrValue stream_attr;
stream_attr.accessPolicyWindow.base_ptr = d_data;
stream_attr.accessPolicyWindow.num_bytes = data_size;
stream_attr.accessPolicyWindow.hitRatio = 0.8f;      // Expected hit rate
stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;  // Keep in L2
stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  // Don't cache misses

cudaError_t cudaStreamSetAttribute(
    cudaStream_t stream,
    cudaStreamAttrID attr,
    const cudaStreamAttrValue* value
);
```

#### Rust FFI Wrapper (kimsfinance)

```rust
#[repr(C)]
pub struct cudaAccessPolicyWindow {
    pub base_ptr: *mut c_void,
    pub num_bytes: usize,
    pub hit_ratio: f32,
    pub hit_prop: cudaAccessProperty,
    pub miss_prop: cudaAccessProperty,
}

#[repr(u32)]
pub enum cudaAccessProperty {
    Normal = 0,
    Streaming = 1,
    Persisting = 2,
}

#[repr(C)]
pub union cudaStreamAttrValue {
    pub access_policy_window: cudaAccessPolicyWindow,
}

#[link(name = "cudart")]
extern "C" {
    pub fn cudaStreamSetAttribute(
        stream: cudaStream_t,
        attr: u32, // cudaStreamAttributeAccessPolicyWindow = 3
        value: *const cudaStreamAttrValue,
    ) -> u32;
}

// Safe wrapper
impl GpuDevice {
    pub fn set_l2_cache_hint(
        &self,
        data_ptr: *const c_void,
        size_bytes: usize,
        expected_hit_ratio: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            let mut attr_value = cudaStreamAttrValue {
                access_policy_window: cudaAccessPolicyWindow {
                    base_ptr: data_ptr as *mut c_void,
                    num_bytes: size_bytes,
                    hit_ratio: expected_hit_ratio,
                    hit_prop: cudaAccessProperty::Persisting,
                    miss_prop: cudaAccessProperty::Streaming,
                },
            };
            
            let stream = self.stream.as_raw_cudaStream_t();
            let err = cudaStreamSetAttribute(
                stream,
                3, // cudaStreamAttributeAccessPolicyWindow
                &attr_value,
            );
            
            if err != 0 {
                return Err(GpuError::ExecutionError(
                    format!("cudaStreamSetAttribute failed: {}", err)
                ));
            }
            Ok(())
        }
    }
}
```

#### Use Cases in kimsfinance

**Batch Indicator Processing (OHLCV data reuse):**

```rust
// Load OHLCV data once (5 × 100K × 8 bytes = 4 MB)
let d_high = device.copy_to_device(&high)?;
let d_low = device.copy_to_device(&low)?;
let d_close = device.copy_to_device(&close)?;
let d_open = device.copy_to_device(&open)?;
let d_volume = device.copy_to_device(&volume)?;

// Hint to L2 cache: Keep OHLCV data in L2 (4 MB fits in 32 MB L2)
device.set_l2_cache_hint(d_close.as_ptr(), 100_000 * 8, 0.9)?; // 90% expected hit rate

// Calculate 10 indicators (all use close prices)
// With L2 hints: close prices stay in L2 across all kernels
// Without L2 hints: close prices may be evicted after each kernel
// Expected improvement: 10-20% for memory-bound kernels
for indicator in indicators {
    calculate_indicator(&device, &d_close, indicator)?;
}
```

---

## Ada Lovelace Architecture Features {#ada-architecture-features}

### 1. Compute Capability 8.9 Optimizations

#### Explicit PTX Targeting

**Current (Generic):**
```rust
// Compiles for generic compute_80+ (may miss Ada optimizations)
let ptx = compile_ptx(KERNEL_SOURCE)?;
```

**Optimized (Ada-specific):**
```rust
use cudarc::nvrtc::{CompileOptions, CompileTarget};

let opts = CompileOptions::default()
    .set_target(CompileTarget::Compute89)  // Explicit Ada target
    .set_opt_level(OptLevel::O3)
    .add_flag("--use_fast_math");  // Aggressive FP optimizations

let ptx = compile_ptx_with_opts(KERNEL_SOURCE, opts)?;
```

**Expected Impact:**
- FP32 throughput: **+15-30%** (2x FP32 ops/cycle on Ada vs Ampere)
- FP64 throughput: **0-5%** (not improved on Ada)
- Bandwidth-bound: **0-5%** (compute target doesn't affect memory)

---

### 2. FP8 Tensor Cores (4th Gen)

**Status:** Ada architecture feature  
**cudarc 0.17.3 Support:** ❌ (no high-level API)  
**Relevance:** LOW for current use case (financial indicators use FP64)

#### Note on FP8 for Financial Indicators

While Ada supports FP8 Tensor Cores, **financial indicators require FP64 precision** for accuracy. FP8 is suitable for:
- Deep learning inference (where precision loss is acceptable)
- Matrix multiplications (not common in technical indicators)

**Not recommended** for kimsfinance unless:
- Implementing ML-based indicators (e.g., neural network predictions)
- Can tolerate precision loss (<0.1% accuracy)

---

### 3. Shared Memory Configuration (100 KB per SM)

#### Dynamic Shared Memory Allocation

**Kernel Declaration:**
```cuda
extern "C" __global__ void sma_large_period_kernel(
    const double* close,
    double* sma,
    int n,
    int period
) {
    extern __shared__ double shared_data[]; // Dynamic allocation
    // ...
}
```

**Rust Launch Code:**
```rust
use cudarc::driver::LaunchConfig;

let block_size = 256;
let period = 200;

// Calculate shared memory needed
// (blockDim.x + period - 1) * sizeof(f64)
let shared_mem_bytes = (block_size + period - 1) * std::mem::size_of::<f64>();

// For large periods (>48 KB), request explicit carveout
if shared_mem_bytes > 49152 {
    // Set max dynamic shared memory for this kernel
    unsafe {
        cudarc::driver::sys::cuFuncSetAttribute(
            kernel.as_raw(),
            cudarc::driver::sys::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_mem_bytes as u32,
        );
    }
}

// Launch with dynamic shared memory
let config = LaunchConfig {
    grid_dim: (/* ... */),
    block_dim: (block_size, 1, 1),
    shared_mem_bytes: shared_mem_bytes as u32,
};

unsafe {
    kernel_stream.launch(&kernel, config, (d_close, d_sma, n, period))?;
}
```

**Best Practices:**
- **Small periods (<50):** Use 0 bytes shared memory (L2 cache sufficient)
- **Medium periods (50-100):** Use default 48 KB carveout
- **Large periods (100-500):** Request up to 100 KB carveout
- **Bank conflicts:** Pad arrays to avoid 32-way bank conflicts (use 33 instead of 32 for multiples of 32)

---

## Memory Management APIs {#memory-management-apis}

### 1. Pinned (Page-Locked) Memory

**Purpose:** 2-3x faster host-device transfers  
**Use Case:** Hybrid indicators (RSI, ATR, MACD with CPU smoothing)

#### C/C++ API

```cpp
// Allocate pinned host memory
void* host_ptr;
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);

// Flags:
// - cudaHostAllocDefault: Portable pinned memory
// - cudaHostAllocMapped: Zero-copy memory (GPU accesses host memory directly)
// - cudaHostAllocWriteCombined: Write-combined memory (faster writes, slower reads)

// Free pinned memory
cudaError_t cudaFreeHost(void* ptr);
```

#### Rust FFI Wrapper (kimsfinance)

```rust
pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> PinnedBuffer<T> {
    pub fn new(len: usize) -> Result<Self, GpuError> {
        unsafe {
            let size = len * std::mem::size_of::<T>();
            let mut ptr: *mut c_void = std::ptr::null_mut();
            
            let err = cudaHostAlloc(&mut ptr, size, 0); // cudaHostAllocDefault
            if err != 0 {
                return Err(GpuError::AllocationError(
                    format!("cudaHostAlloc failed: {}", err)
                ));
            }
            
            Ok(Self {
                ptr: ptr as *mut T,
                len,
                _marker: PhantomData,
            })
        }
    }
    
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            cudaFreeHost(self.ptr as *mut c_void);
        }
    }
}
```

#### Usage Example (RSI Hybrid)

```rust
// Allocate pinned buffers for CPU-GPU round-trip
let mut pinned_gains = PinnedBuffer::<f64>::new(n)?;
let mut pinned_losses = PinnedBuffer::<f64>::new(n)?;

// Copy from GPU to pinned host (2-3x faster than pageable)
device.memcpy_dtoh_pinned(&d_gains, pinned_gains.as_mut_slice())?;
device.memcpy_dtoh_pinned(&d_losses, pinned_losses.as_mut_slice())?;

// CPU smoothing uses pinned memory
let avg_gain = wilders_smoothing_cpu(pinned_gains.as_slice(), period);
let avg_loss = wilders_smoothing_cpu(pinned_losses.as_slice(), period);

// Copy back to GPU (2-3x faster)
let d_avg_gain = device.copy_to_device_pinned(&avg_gain)?;
let d_avg_loss = device.copy_to_device_pinned(&avg_loss)?;
```

**Performance:**
- Transfer 1 MB pageable memory: ~50 μs
- Transfer 1 MB pinned memory: ~18 μs
- **Speedup:** 2.8x

---

### 2. Unified Memory (Not Recommended for kimsfinance)

**Why Not?**
- Unified Memory has overhead for fine-grained access patterns
- Financial indicators use bulk transfers (copy once, compute many)
- Explicit memory management provides better control and performance

**When to Use:**
- Prototyping (easier than manual transfers)
- Irregular access patterns (GPU accesses random host memory locations)

---

## Kernel Launch and Execution {#kernel-launch-execution}

### 1. Launch Configuration Optimization

#### Grid/Block Sizing for Ada

**Ada SM Characteristics:**
- 48 warps/SM max
- 32 threads/warp
- 1024 threads/block max

**Recommended Configurations:**

| Kernel Type | Block Size | Grid Size | Rationale |
|-------------|-----------|-----------|-----------|
| Embarrassingly parallel (ROC, simple ops) | 256 | `(n + 255) / 256` | 8 warps/block, 6 blocks/SM, 100% occupancy |
| Shared memory (SMA large period) | 128 | `(n + 127) / 128` | More blocks fit when shared memory is high |
| Register-heavy (complex math) | 512 | `(n + 511) / 512` | Fewer blocks, but each block has more resources |

#### cudarc Launch API

```rust
use cudarc::driver::LaunchConfig;

// Automatic sizing (recommended for most cases)
let config = LaunchConfig::for_num_elems(n as u32);

// Manual sizing (when you need control)
let block_size = 256;
let grid_size = (n + block_size - 1) / block_size;
let config = LaunchConfig {
    grid_dim: (grid_size as u32, 1, 1),
    block_dim: (block_size as u32, 1, 1),
    shared_mem_bytes: 0,
};

// Launch with builder pattern
let mut builder = stream.launch_builder(&kernel);
builder.arg(&d_input);
builder.arg(&d_output);
builder.arg(&(n as i32));
unsafe { builder.launch(config)?; }
```

---

### 2. Stream Concurrency

**Use Case:** Launch fast/medium/slow indicators concurrently

```rust
use cudarc::driver::CudaStream;

// Create 3 streams for concurrent execution
let stream_fast = device.context().create_stream()?;
let stream_medium = device.context().create_stream()?;
let stream_slow = device.context().create_stream()?;

// Launch indicators on different streams (execute concurrently!)
launch_roc_kernel(&stream_fast, &d_close)?;       // Fast (5 μs)
launch_rsi_kernel(&stream_medium, &d_close)?;     // Medium (30 μs)
launch_stochastic_kernel(&stream_slow, &d_high, &d_low)?; // Slow (50 μs)

// Wait for all to complete
stream_fast.synchronize()?;
stream_medium.synchronize()?;
stream_slow.synchronize()?;

// Total time: max(5, 30, 50) = 50 μs (vs 5 + 30 + 50 = 85 μs sequential)
// Speedup: 1.7x
```

---

## Performance Optimization APIs {#performance-optimization-apis}

### 1. Occupancy Calculator

**Purpose:** Determine optimal block size for maximum SM utilization

#### C/C++ API

```cpp
int min_grid_size, block_size;
cudaError_t cudaOccupancyMaxPotentialBlockSize(
    int* minGridSize,
    int* blockSize,
    const void* func,
    size_t dynamicSMemSize,
    int blockSizeLimit
);
```

#### Rust Wrapper (Approximate - cudarc doesn't expose this)

```rust
// Not directly available in cudarc 0.17.3
// Workaround: Use Nsight Compute profiling to determine optimal block size

// Command-line profiling:
// ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./binary

// Expected output: Occupancy 90-100% = good, <70% = investigate
```

---

### 2. Profiling Hooks

**Recommended Tools:**
- **Nsight Compute:** Kernel-level profiling (latency, occupancy, memory bandwidth)
- **Nsight Systems:** Timeline view (kernel launch overhead, stream concurrency)

**Key Metrics for Financial Indicators:**

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Occupancy | >80% | GPU SMs well-utilized |
| Memory Bandwidth | >70% of peak (201 GB/s for RTX 3500) | Not memory-bound |
| L2 Cache Hit Rate | >60% | Good data locality |
| Kernel Duration | <100 μs for 100K candles | Compute efficiency |

---

## Code Examples for Common Patterns {#code-examples}

### Example 1: Parallel Element-Wise Operation (ROC)

**CUDA Kernel:**
```cuda
extern "C" __global__ void roc_kernel(
    const double* __restrict__ close,
    double* __restrict__ roc,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= period && idx < n) {
        double prev_close = close[idx - period];
        double curr_close = close[idx];
        roc[idx] = ((curr_close - prev_close) / prev_close) * 100.0;
    } else if (idx < period) {
        roc[idx] = NAN; // Not enough history
    }
}
```

**Rust Wrapper:**
```rust
pub fn roc_gpu(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
) -> Result<Array1<f64>, GpuError> {
    let n = close.len();
    
    // Compile kernel
    let ptx = compile_ptx_with_opts(ROC_KERNEL, CompileOptions::default()
        .set_target(CompileTarget::Compute89))?;
    let module = device.context().load_module(ptx)?;
    let kernel = module.load_function("roc_kernel")?;
    
    // Copy to device
    let d_close = device.copy_to_device(close.as_slice().unwrap())?;
    let mut d_roc = device.alloc_buffer(n)?;
    
    // Launch kernel
    let config = LaunchConfig::for_num_elems(n as u32);
    let mut builder = device.stream.launch_builder(&kernel);
    builder.arg(&d_close);
    builder.arg(&mut d_roc);
    builder.arg(&(n as i32));
    builder.arg(&(period as i32));
    unsafe { builder.launch(config)?; }
    
    // Copy result
    device.stream.synchronize()?;
    let result = device.copy_to_host(&d_roc)?;
    Ok(Array1::from_vec(result))
}
```

---

### Example 2: Rolling Window with Shared Memory (SMA)

**CUDA Kernel:**
```cuda
extern "C" __global__ void sma_shared_kernel(
    const double* __restrict__ close,
    double* __restrict__ sma,
    int n,
    int period
) {
    extern __shared__ double shared_close[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load data into shared memory (coalesced)
    if (idx < n) {
        shared_close[tid] = close[idx];
    }
    __syncthreads();
    
    // Compute SMA
    if (idx >= period - 1 && idx < n) {
        double sum = 0.0;
        
        // Sum from shared memory (if available)
        for (int j = 0; j < period; j++) {
            int offset = tid - j;
            if (offset >= 0) {
                sum += shared_close[offset];
            } else {
                sum += close[idx - j]; // Fallback to global memory
            }
        }
        
        sma[idx] = sum / (double)period;
    }
}
```

**Rust Wrapper:**
```rust
pub fn sma_gpu_shared(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
) -> Result<Array1<f64>, GpuError> {
    let n = close.len();
    let block_size = 256;
    
    // Calculate shared memory needed
    let shared_mem_bytes = block_size * std::mem::size_of::<f64>();
    
    // Compile and load kernel
    let ptx = compile_ptx(SMA_KERNEL)?;
    let module = device.context().load_module(ptx)?;
    let kernel = module.load_function("sma_shared_kernel")?;
    
    // Launch with dynamic shared memory
    let grid_size = (n + block_size - 1) / block_size;
    let config = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem_bytes as u32,
    };
    
    // ... (rest of launch code)
}
```

---

### Example 3: Hybrid CPU-GPU Pipeline (RSI)

**Stage 1: GPU Gains/Losses Calculation**
```cuda
extern "C" __global__ void calculate_gains_losses(
    const double* __restrict__ close,
    double* __restrict__ gains,
    double* __restrict__ losses,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx > 0 && idx < n) {
        double delta = close[idx] - close[idx - 1];
        gains[idx] = fmax(delta, 0.0);
        losses[idx] = fmax(-delta, 0.0);
    }
}
```

**Stage 2: CPU Wilder's Smoothing**
```rust
fn wilders_smoothing_cpu(data: &[f64], period: usize) -> Vec<f64> {
    let mut avg = vec![0.0; data.len()];
    
    // Initial average (SMA)
    avg[period - 1] = data[0..period].iter().sum::<f64>() / period as f64;
    
    // Wilder's smoothing (EMA with alpha = 1/period)
    let alpha = 1.0 / period as f64;
    for i in period..data.len() {
        avg[i] = alpha * data[i] + (1.0 - alpha) * avg[i - 1];
    }
    
    avg
}
```

**Stage 3: GPU RSI Calculation**
```cuda
extern "C" __global__ void calculate_rsi(
    const double* __restrict__ avg_gain,
    const double* __restrict__ avg_loss,
    double* __restrict__ rsi,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= period && idx < n) {
        double gain = avg_gain[idx];
        double loss = avg_loss[idx];
        
        if (loss < 1e-10) {
            rsi[idx] = 100.0;
        } else {
            double rs = gain / loss;
            rsi[idx] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
}
```

**Rust Orchestration:**
```rust
pub fn rsi_hybrid_gpu(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
) -> Result<Array1<f64>, GpuError> {
    let n = close.len();
    
    // Stage 1: GPU - Calculate gains/losses
    let d_close = device.copy_to_device(close.as_slice().unwrap())?;
    let mut d_gains = device.alloc_buffer(n)?;
    let mut d_losses = device.alloc_buffer(n)?;
    
    launch_gains_losses_kernel(&device, &d_close, &mut d_gains, &mut d_losses, n)?;
    
    // Stage 2: Transfer to pinned host memory (2-3x faster)
    let mut pinned_gains = PinnedBuffer::new(n)?;
    let mut pinned_losses = PinnedBuffer::new(n)?;
    device.memcpy_dtoh_pinned(&d_gains, pinned_gains.as_mut_slice())?;
    device.memcpy_dtoh_pinned(&d_losses, pinned_losses.as_mut_slice())?;
    
    // Stage 3: CPU - Wilder's smoothing (6x faster than GPU single-thread)
    let avg_gain = wilders_smoothing_cpu(pinned_gains.as_slice(), period);
    let avg_loss = wilders_smoothing_cpu(pinned_losses.as_slice(), period);
    
    // Stage 4: Transfer back to GPU
    let d_avg_gain = device.copy_to_device_pinned(&avg_gain)?;
    let d_avg_loss = device.copy_to_device_pinned(&avg_loss)?;
    
    // Stage 5: GPU - Calculate RSI
    let mut d_rsi = device.alloc_buffer(n)?;
    launch_rsi_kernel(&device, &d_avg_gain, &d_avg_loss, &mut d_rsi, n, period)?;
    
    // Stage 6: Copy result
    let result = device.copy_to_host(&d_rsi)?;
    Ok(Array1::from_vec(result))
}
```

**Performance Breakdown:**
- Stage 1 (GPU gains/losses): ~20 μs
- Stage 2 (D2H transfer, pinned): ~18 μs
- Stage 3 (CPU smoothing, 2x): ~30 μs
- Stage 4 (H2D transfer, pinned): ~18 μs
- Stage 5 (GPU RSI calc): ~15 μs
- **Total:** ~101 μs (vs ~250 μs pure GPU single-thread smoothing)

---

## cudarc API Mapping {#cudarc-api-mapping}

### Available in cudarc 0.17.3

| CUDA Feature | cudarc API | Notes |
|--------------|-----------|-------|
| Device Init | `CudaContext::new(device_id)` | ✅ Direct support |
| Memory Allocation | `CudaStream::alloc_zeros<T>(len)` | ✅ Direct support |
| Memory Copy H2D | `CudaStream::memcpy_htod(src, dst)` | ✅ Direct support |
| Memory Copy D2H | `CudaStream::memcpy_dtoh(src)` | ✅ Direct support |
| PTX Compilation | `compile_ptx(source)` | ✅ Via nvrtc |
| Module Load | `CudaContext::load_module(ptx)` | ✅ Direct support |
| Kernel Launch | `CudaStream::launch_builder(&kernel)` | ✅ Direct support |
| Synchronization | `CudaStream::synchronize()` | ✅ Direct support |
| Multiple Streams | `CudaContext::create_stream()` | ✅ Direct support |

### Requires FFI (Not in cudarc 0.17.3)

| CUDA Feature | FFI Required | Complexity | Priority |
|--------------|--------------|------------|----------|
| `cudaMallocAsync` | Yes | Medium (2-3 days) | HIGH |
| `cudaFreeAsync` | Yes | Medium (2-3 days) | HIGH |
| CUDA Graphs | Yes | High (3-5 days) | HIGH |
| L2 Cache Hints | Yes | Medium (2-3 days) | MEDIUM |
| Pinned Memory | Yes | Low (1-2 days) | HIGH |
| Occupancy Calculator | Yes | Low (1 day) | LOW (use Nsight instead) |

### Not Applicable/Not Recommended

| CUDA Feature | Reason |
|--------------|--------|
| Unified Memory | Explicit transfers more efficient for bulk operations |
| Zero-Copy Memory | Adds latency for large datasets |
| Texture Memory | Not beneficial for linear access patterns |
| Dynamic Parallelism | Not needed for indicator calculations |

---

## Profiling Commands Reference

### Nsight Compute (Kernel-level)

```bash
# Full profile (all metrics)
ncu --set full ./target/release/examples/binance_aggregation

# Specific metrics for financial indicators
ncu --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l2_tex_hit_rate,\
    smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_dmul_pred_on.sum \
    ./target/release/examples/binance_aggregation

# Roofline analysis
ncu --section SpeedOfLight ./target/release/examples/binance_aggregation

# Occupancy limiter
ncu --section Occupancy ./target/release/examples/binance_aggregation
```

### Nsight Systems (Timeline)

```bash
# System-wide timeline (kernel launches, memory transfers)
nsys profile --trace=cuda,nvtx ./target/release/examples/binance_aggregation

# Focus on GPU activity
nsys profile --trace=cuda --cuda-memory-usage=true \
    ./target/release/examples/binance_aggregation

# Export to SQLite for custom analysis
nsys profile --output=report.qdrep \
    ./target/release/examples/binance_aggregation
```

---

## References

1. **CUDA 13.0 Documentation:** https://docs.nvidia.com/cuda/
2. **Ada Tuning Guide:** https://docs.nvidia.com/cuda/ada-tuning-guide/
3. **Stream-Ordered Allocator:** https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
4. **CUDA Graphs:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
5. **cudarc Repository:** https://github.com/coreylowman/cudarc
6. **RTX 3500 Ada Specs:** https://www.techpowerup.com/gpu-specs/rtx-3500-mobile-ada-generation.c4098

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-26  
**Maintainer:** kimsfinance team  
**Review Status:** Draft
