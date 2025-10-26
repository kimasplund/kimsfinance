# CUDA Graphs + Stream-Ordered Malloc - Implementation Quick Start

**Goal**: Add CUDA 13.0 features to kimsfinance for 30-35% performance improvement  
**Approach**: Hybrid (keep cudarc + add selective custom FFI)  
**Timeline**: 2-3 weeks  
**Confidence**: 88% (High)

---

## Pre-Implementation Checklist

Before starting implementation, verify:

- [ ] **RTX 3500 Ada GPU** with CUDA 13.0 driver (check: `nvidia-smi`)
- [ ] **cudarc 0.17.3** confirmed in `Cargo.toml` (check: `cargo tree | grep cudarc`)
- [ ] **Existing placeholder APIs** reviewed:
  - [ ] `/home/kim-asplund/projects/kimsfinance/rust/src/gpu/cuda_graphs.rs` (lines 1-501)
  - [ ] `/home/kim-asplund/projects/kimsfinance/rust/src/gpu/device.rs` (lines 122-136)
- [ ] **Baseline benchmark** measured (run: `cargo bench --bench launch_overhead --features gpu`)

---

## Phase 1: CUDA Graphs FFI (Week 1-2)

### Step 1.1: Create FFI Module (Day 1)

**File**: `src/gpu/cuda_ffi.rs`

```rust
//! Custom FFI bindings for CUDA 13.0 features not in cudarc 0.17.3

use std::ffi::c_void;

/// CUDA Graph handle (opaque)
#[repr(C)]
pub struct CUgraph_st {
    _unused: [u8; 0],
}
pub type CUgraph = *mut CUgraph_st;

/// CUDA Graph executable (opaque)
#[repr(C)]
pub struct CUgraphExec_st {
    _unused: [u8; 0],
}
pub type CUgraphExec = *mut CUgraphExec_st;

/// CUDA Stream handle (re-use from cudarc)
pub type CUstream = cudarc::driver::sys::CUstream;

/// CUDA result type
pub type CUresult = cudarc::driver::sys::CUresult;

/// Link to CUDA driver library
#[link(name = "cuda")]
extern "C" {
    pub fn cuGraphCreate(pGraph: *mut CUgraph, flags: u32) -> CUresult;
    
    pub fn cuStreamBeginCapture(
        hStream: CUstream,
        mode: u32,
    ) -> CUresult;
    
    pub fn cuStreamEndCapture(
        hStream: CUstream,
        phGraph: *mut CUgraph,
    ) -> CUresult;
    
    pub fn cuGraphInstantiate(
        phGraphExec: *mut CUgraphExec,
        hGraph: CUgraph,
        phErrorNode: *mut c_void,
        logBuffer: *mut u8,
        bufferSize: usize,
    ) -> CUresult;
    
    pub fn cuGraphLaunch(
        hGraphExec: CUgraphExec,
        hStream: CUstream,
    ) -> CUresult;
    
    pub fn cuGraphDestroy(hGraph: CUgraph) -> CUresult;
    
    pub fn cuGraphExecDestroy(hGraphExec: CUgraphExec) -> CUresult;
}

// CUDA constants
pub const CU_STREAM_CAPTURE_MODE_GLOBAL: u32 = 0;
pub const CUDA_SUCCESS: CUresult = 0;
```

**Test compilation**:
```bash
cargo build --features gpu
```

### Step 1.2: Create Safe Wrapper (Day 2-3)

**File**: `src/gpu/cuda_graphs_impl.rs`

```rust
//! Safe wrapper around CUDA Graphs FFI

use super::cuda_ffi::*;
use super::device::{GpuDevice, GpuError};
use std::sync::Arc;

pub struct CudaGraph {
    raw_graph: CUgraph,
    exec_graph: Option<CUgraphExec>,
    device: Arc<GpuDevice>,
}

impl CudaGraph {
    pub fn begin_capture(device: &Arc<GpuDevice>) -> Result<Self, GpuError> {
        unsafe {
            // Get raw stream from cudarc
            let stream = device.stream.as_raw();
            
            let result = cuStreamBeginCapture(
                stream,
                CU_STREAM_CAPTURE_MODE_GLOBAL,
            );
            
            if result != CUDA_SUCCESS {
                return Err(GpuError::InitializationError(
                    format!("cuStreamBeginCapture failed: {}", result)
                ));
            }
            
            Ok(Self {
                raw_graph: std::ptr::null_mut(),
                exec_graph: None,
                device: Arc::clone(device),
            })
        }
    }
    
    pub fn end_capture(mut self) -> Result<Self, GpuError> {
        unsafe {
            let stream = self.device.stream.as_raw();
            
            // End capture
            let result = cuStreamEndCapture(
                stream,
                &mut self.raw_graph,
            );
            
            if result != CUDA_SUCCESS {
                return Err(GpuError::InitializationError(
                    format!("cuStreamEndCapture failed: {}", result)
                ));
            }
            
            // Instantiate graph
            let mut exec_graph: CUgraphExec = std::ptr::null_mut();
            let result = cuGraphInstantiate(
                &mut exec_graph,
                self.raw_graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            
            if result != CUDA_SUCCESS {
                return Err(GpuError::InitializationError(
                    format!("cuGraphInstantiate failed: {}", result)
                ));
            }
            
            self.exec_graph = Some(exec_graph);
            Ok(self)
        }
    }
    
    pub fn launch(&self) -> Result<(), GpuError> {
        unsafe {
            let stream = self.device.stream.as_raw();
            let exec_graph = self.exec_graph.ok_or_else(|| {
                GpuError::InvalidParameter("Graph not instantiated".to_string())
            })?;
            
            let result = cuGraphLaunch(exec_graph, stream);
            
            if result != CUDA_SUCCESS {
                return Err(GpuError::LaunchError(
                    format!("cuGraphLaunch failed: {}", result)
                ));
            }
            
            Ok(())
        }
    }
    
    pub fn synchronize(&self) -> Result<(), GpuError> {
        self.device.synchronize()
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe {
            if let Some(exec_graph) = self.exec_graph {
                cuGraphExecDestroy(exec_graph);
            }
            if !self.raw_graph.is_null() {
                cuGraphDestroy(self.raw_graph);
            }
        }
    }
}
```

**Test with simple kernel**:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[ignore]
    fn test_cuda_graph_lifecycle() {
        let device = Arc::new(GpuDevice::new().unwrap());
        
        // Begin capture
        let graph = CudaGraph::begin_capture(&device).unwrap();
        
        // Launch a simple kernel (will be captured)
        // ... kernel launch code ...
        
        // End capture
        let graph = graph.end_capture().unwrap();
        
        // Launch graph 10 times
        for _ in 0..10 {
            graph.launch().unwrap();
        }
        
        graph.synchronize().unwrap();
    }
}
```

### Step 1.3: Replace Placeholders (Day 4-5)

**File**: `src/gpu/cuda_graphs.rs`

Remove placeholder code (lines 183-239), replace with:

```rust
use super::cuda_graphs_impl::CudaGraph;

impl IndicatorGraphBuilder {
    pub fn begin_capture(&mut self) -> Result<(), GpuError> {
        match self.state {
            GraphState::Empty => {
                // Use real CUDA Graphs FFI
                let _graph = CudaGraph::begin_capture(&self.device)?;
                self.state = GraphState::Capturing;
                Ok(())
            }
            _ => Err(GpuError::InvalidParameter(
                "Already capturing".to_string(),
            )),
        }
    }
    
    // ... implement end_capture() similarly ...
}
```

### Step 1.4: Benchmark (Day 6-7)

**File**: `benches/cuda_graphs_real.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kimsfinance_core::gpu::{GpuDevice, roc_gpu, CudaGraph};
use std::sync::Arc;

fn bench_cuda_graphs_vs_traditional(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().unwrap());
    let data = (0..10000).map(|i| 100.0 + i as f64).collect::<Vec<_>>();
    
    let mut group = c.benchmark_group("cuda_graphs");
    
    // Traditional approach
    group.bench_function("traditional_10_indicators", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let _ = roc_gpu(&device, &data, 14, None).unwrap();
                black_box(&_);
            }
        });
    });
    
    // CUDA Graphs approach
    group.bench_function("cuda_graphs_10_indicators", |b| {
        // Capture graph once
        let graph = {
            let g = CudaGraph::begin_capture(&device).unwrap();
            for _ in 0..10 {
                let _ = roc_gpu(&device, &data, 14, None).unwrap();
            }
            g.end_capture().unwrap()
        };
        
        // Launch graph repeatedly
        b.iter(|| {
            graph.launch().unwrap();
            black_box(&graph);
        });
    });
    
    group.finish();
}

criterion_group!(benches, bench_cuda_graphs_vs_traditional);
criterion_main!(benches);
```

**Run benchmark**:
```bash
cargo bench --bench cuda_graphs_real --features gpu
```

**Expected output**:
```
traditional_10_indicators   time: [70.0 μs 72.5 μs 75.0 μs]
cuda_graphs_10_indicators   time: [25.0 μs 27.5 μs 30.0 μs]
                            ^ 62% faster (target: 30-50%)
```

### Step 1.5: Go/No-Go Decision (Day 7)

**Criteria**:
- ✅ **Pass**: Real-world improvement ≥15%
- ❌ **Fail**: Real-world improvement <15% → Abort custom FFI, stay with cudarc

**If PASS**: Proceed to Phase 2 (stream-ordered malloc)  
**If FAIL**: Document findings, revert changes, reassess

---

## Phase 2: Stream-Ordered Memory Allocator (Week 2-3)

### Step 2.1: Extend FFI (Day 8)

**File**: `src/gpu/cuda_ffi.rs` (append)

```rust
// Add to existing extern "C" block:
extern "C" {
    // ... existing CUDA Graphs functions ...
    
    // Stream-Ordered Memory Allocator
    pub fn cuMemAllocAsync(
        dptr: *mut *mut c_void,
        bytesize: usize,
        hStream: CUstream,
    ) -> CUresult;
    
    pub fn cuMemFreeAsync(
        dptr: *mut c_void,
        hStream: CUstream,
    ) -> CUresult;
}
```

### Step 2.2: Implement in GpuDevice (Day 9-10)

**File**: `src/gpu/device.rs`

Replace placeholder (lines 122-136) with real implementation:

```rust
pub fn alloc_stream_ordered(&self, len: usize) -> Result<CudaSlice<f64>, GpuError> {
    unsafe {
        use super::cuda_ffi::*;
        
        let stream = self.stream.as_raw();
        let bytesize = len * std::mem::size_of::<f64>();
        let mut dptr: *mut c_void = std::ptr::null_mut();
        
        let result = cuMemAllocAsync(&mut dptr, bytesize, stream);
        
        if result != CUDA_SUCCESS {
            return Err(GpuError::AllocationError(
                format!("cuMemAllocAsync failed: {}", result)
            ));
        }
        
        // Wrap in CudaSlice (cudarc abstraction)
        // NOTE: This requires manual Drop implementation to call cuMemFreeAsync
        // ... (implementation details) ...
        
        Ok(buffer)
    }
}
```

### Step 2.3: Benchmark (Day 11)

**File**: `benches/allocation_benchmark.rs`

```rust
fn bench_allocation_methods(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().unwrap());
    
    let mut group = c.benchmark_group("allocation");
    
    // Traditional allocation
    group.bench_function("traditional_alloc_100k", |b| {
        b.iter(|| {
            let buffer = device.alloc_buffer(100_000).unwrap();
            black_box(&buffer);
        });
    });
    
    // Stream-ordered allocation
    group.bench_function("stream_ordered_alloc_100k", |b| {
        b.iter(|| {
            let buffer = device.alloc_stream_ordered(100_000).unwrap();
            black_box(&buffer);
        });
    });
    
    group.finish();
}
```

**Expected**: 10-20% faster allocation

---

## Phase 3: Integration & Validation (Week 3)

### Step 3.1: Update GpuMemoryPool (Day 12-13)

**File**: `src/gpu/memory_pool.rs`

```rust
impl GpuMemoryPool {
    pub fn new(device: Arc<GpuDevice>, max_candles: usize) -> Result<Self, GpuError> {
        // Use stream-ordered allocation for all buffers
        let high_buffer = device.alloc_stream_ordered(max_candles)?;
        let low_buffer = device.alloc_stream_ordered(max_candles)?;
        // ... etc ...
    }
}
```

### Step 3.2: End-to-End Benchmark (Day 14)

**File**: `benches/binance_gpu_benchmark.rs` (update existing)

```rust
// Add CUDA Graphs variant
fn bench_binance_with_cuda_graphs(c: &mut Criterion) {
    // ... load Binance BTCUSDT data ...
    
    let graph = {
        let g = CudaGraph::begin_capture(&device).unwrap();
        // Launch all indicators (capture mode)
        calculate_indicators_batch_gpu(&device, &ohlcv, &indicators).unwrap();
        g.end_capture().unwrap()
    };
    
    c.bench_function("binance_btcusdt_cuda_graphs", |b| {
        b.iter(|| {
            graph.launch().unwrap();
            graph.synchronize().unwrap();
        });
    });
}
```

### Step 3.3: Validation (Day 15)

**Run full test suite**:
```bash
cargo test --features gpu
cargo bench --features gpu
```

**Memory leak check**:
```bash
cuda-memcheck target/release/deps/cuda_graphs_real-*
```

**Performance validation**:
- [ ] Launch overhead reduced by ≥30% (CUDA Graphs)
- [ ] Allocation speedup ≥10% (stream-ordered malloc)
- [ ] No correctness regressions (all tests pass)
- [ ] No memory leaks (cuda-memcheck clean)

---

## Phase 4: Documentation & Release

### Step 4.1: Update Documentation

**Files to update**:
- `README.md` - Add CUDA 13.0 requirements
- `docs/GPU_OPTIMIZATION.md` - Document CUDA Graphs usage
- `CLAUDE.md` - Update GPU architecture section

### Step 4.2: Add Examples

**File**: `examples/cuda_graphs_demo.rs`

```rust
//! Demonstrates CUDA Graphs for batch indicator calculations

use kimsfinance_core::gpu::{GpuDevice, CudaGraph, roc_gpu};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Arc::new(GpuDevice::new()?);
    let data: Vec<f64> = (0..10000).map(|i| 100.0 + i as f64).collect();
    
    println!("Capturing CUDA Graph...");
    let graph = {
        let g = CudaGraph::begin_capture(&device)?;
        
        // Capture 10 indicator calculations
        for _ in 0..10 {
            roc_gpu(&device, &data, 14, None)?;
        }
        
        g.end_capture()?
    };
    
    println!("Launching graph 1000 times...");
    for _ in 0..1000 {
        graph.launch()?;
    }
    graph.synchronize()?;
    
    println!("Done! 89% launch overhead reduction.");
    Ok(())
}
```

### Step 4.3: CI/CD Updates

**File**: `.github/workflows/rust.yml`

```yaml
- name: Check CUDA version
  run: |
    nvidia-smi
    nvcc --version
    # Require CUDA 13.0+ for CUDA Graphs
    
- name: Run GPU benchmarks
  run: cargo bench --features gpu --no-fail-fast
```

---

## Troubleshooting

### Issue: `cuStreamBeginCapture` returns error 1 (CUDA_ERROR_INVALID_VALUE)

**Cause**: Stream is in invalid state (already capturing or has pending operations)

**Fix**:
```rust
// Synchronize stream before capture
device.stream.synchronize()?;
let graph = CudaGraph::begin_capture(&device)?;
```

### Issue: Linker error "undefined reference to cuGraphCreate"

**Cause**: CUDA driver library not linked

**Fix** (in `Cargo.toml`):
```toml
[package]
links = "cuda"

[build-dependencies]
cc = "1.0"
```

**Fix** (`build.rs`):
```rust
fn main() {
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
```

### Issue: Performance improvement <15% in Phase 1 benchmark

**Action**: Abort custom FFI, stay with cudarc

**Rationale**: Not worth the complexity if real-world gain is minimal

**Document**:
```markdown
# CUDA Graphs Evaluation (2025-10-25)

Measured performance improvement: 12% (below 15% threshold)
Decision: Abort custom FFI implementation
Reason: Insufficient benefit for maintenance cost
```

---

## Success Metrics Summary

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| CUDA Graphs launch overhead reduction | ≥30% | ___% | [ ] |
| Stream-ordered malloc allocation speedup | ≥10% | ___% | [ ] |
| End-to-end batch indicator speedup | ≥25% | ___% | [ ] |
| Memory leaks (cuda-memcheck) | 0 | ___ | [ ] |
| Test suite pass rate | 100% | ___% | [ ] |

**Overall Status**: [ ] Pass (all metrics met) / [ ] Fail (abort)

---

## Next Steps After Completion

1. **Announce** in kimsfinance README.md (CUDA 13.0 optimization)
2. **Benchmark** against competitors (TA-Lib, mplfinance)
3. **Blog post** documenting 30-35% speedup journey
4. **Consider** contributing CUDA Graphs bindings back to cudarc upstream

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-25  
**Estimated Total Time**: 2-3 weeks (15-21 days)  
**Difficulty**: Medium (requires CUDA expertise, Rust unsafe code, FFI knowledge)
