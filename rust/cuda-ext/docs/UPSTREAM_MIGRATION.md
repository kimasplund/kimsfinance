# Migration to Upstream cudarc (When Released)

## Current State (Phase 1)

We currently have two approaches for stream-ordered allocation:

1. **Direct `malloc_async` usage** (cudarc built-in)
   - cudarc already uses `cuMemAllocAsync` when `has_async_alloc` is true
   - Used in `CudaStream::alloc()` automatically
   - Works but doesn't use memory pools

2. **FFI Overlay** (if we implemented one)
   - Custom `StreamOrderedAllocator` in `kimsfinance-cuda-ext`
   - Direct `sys::cuMemAllocFromPoolAsync` calls
   - Maximum performance with pools

## Future State (Phase 2)

When cudarc merges the stream-ordered malloc PR (expected: v0.18.0+):

### Timeline Estimate

- **PR Submitted**: 1-2 weeks from now
- **PR Review**: 2-6 weeks
- **Release**: 1-3 months after merge
- **Total**: 2-4 months

### Migration Steps

#### 1. Update cudarc Dependency

```toml
[dependencies]
cudarc = "0.18.0"  # Or whichever version includes memory pools
```

#### 2. Replace Custom Implementation

**Old (if using FFI overlay)**:
```rust
use kimsfinance_cuda_ext::stream_malloc::StreamOrderedAllocator;
let allocator = StreamOrderedAllocator::new(0)?;
let ptr = allocator.alloc_async(size, &stream)?;
```

**New (upstream cudarc)**:
```rust
use cudarc::driver::{CudaContext, MemoryPool, StreamMemPoolExt};

let ctx = CudaContext::new(0)?;
let stream = ctx.new_stream()?;
let pool = MemoryPool::new(&ctx)?;

unsafe {
    let slice = stream.alloc_from_pool::<f32>(len, &pool)?;
    // ... use slice ...
    stream.free_async(slice)?;
}
```

#### 3. Remove FFI Overlay Dependency (if applicable)

```toml
# Remove this line if it exists
[dependencies]
# kimsfinance-cuda-ext = { path = "./cuda-ext" }  # REMOVE
```

#### 4. Deprecate FFI Overlay (if we created one)

Mark `kimsfinance-cuda-ext` as deprecated:

```rust
#[deprecated(
    since = "0.2.0",
    note = "Use cudarc 0.18+ native stream-ordered allocation instead"
)]
pub mod stream_malloc;
```

## Benefits of Upstream Migration

### 1. Maintenance
- **Before**: Maintain FFI bindings ourselves
- **After**: Upstream maintains it for 1,200+ crates

### 2. Safety
- **Before**: Raw FFI calls, manual safety contracts
- **After**: Safe abstractions, event tracking integration

### 3. Compatibility
- **Before**: Our code only
- **After**: Works with entire cudarc ecosystem

### 4. Features
- **Before**: Basic alloc/free
- **After**: Pool configuration, trimming, statistics

## Code Migration Example

### Before (Current - if using cudarc built-in)

```rust
use cudarc::driver::CudaContext;

let ctx = CudaContext::new(0)?;
let stream = ctx.new_stream()?;

// Uses malloc_async internally, but no pool
unsafe {
    let data = stream.alloc::<f32>(1024)?;
}
```

**Performance**: ~3µs per allocation (CUDA 11.2+ driver)

### After (Upstream cudarc 0.18+)

```rust
use cudarc::driver::{CudaContext, MemoryPool, StreamMemPoolExt};

let ctx = CudaContext::new(0)?;
let stream = ctx.new_stream()?;
let pool = MemoryPool::new(&ctx)?;

// Uses malloc_async with memory pool
unsafe {
    let data = stream.alloc_from_pool::<f32>(1024, &pool)?;
    stream.free_async(data)?;
}
```

**Performance**: ~2-2.5µs per allocation (1.2-1.5x faster)

## Compatibility Matrix

| cudarc Version | Stream-Ordered Malloc | Memory Pools | kimsfinance Support |
|----------------|----------------------|--------------|---------------------|
| v0.17.3 (current) | ✅ (built-in) | ❌ | ✅ Works today |
| v0.18.0+ (future) | ✅ (built-in) | ✅ | ✅ Drop-in upgrade |

## Migration Checklist

When cudarc 0.18+ is released:

- [ ] Update `Cargo.toml` to cudarc 0.18+
- [ ] Replace custom allocator with `MemoryPool`
- [ ] Update imports to use `StreamMemPoolExt`
- [ ] Run benchmarks to validate performance
- [ ] Remove FFI overlay if it exists
- [ ] Update documentation
- [ ] Test on all GPUs (RTX 3500 Ada, etc.)
- [ ] Remove deprecation warnings

## Performance Validation

After migration, validate performance improvements:

```bash
# Run allocation benchmarks
cargo bench --bench allocation_benchmark

# Expected results:
# - malloc_async (current): ~3µs
# - malloc_async + pool (new): ~2-2.5µs
# - Speedup: 1.2-1.5x
```

## Rollback Plan

If upstream migration has issues:

1. **Revert dependency**: `cudarc = "0.17.3"`
2. **Keep existing code**: No changes needed
3. **Report issue**: Open issue on cudarc GitHub
4. **Temporary FFI overlay**: Re-enable if needed

## Current Status

- [x] cudarc fork implementation complete
- [x] Tests passing (5/5)
- [x] Documentation complete
- [ ] PR submitted to upstream
- [ ] PR reviewed
- [ ] PR merged
- [ ] Release published
- [ ] kimsfinance migrated

## Notes

- **No Action Required Now**: Current cudarc works fine
- **Monitor Upstream**: Watch https://github.com/coreylowman/cudarc for PR
- **Plan Ahead**: Migration will be trivial (1-2 hour effort)
- **Zero Risk**: Can keep current version indefinitely

---

**Summary**: When cudarc 0.18+ is released with memory pool support, we can upgrade in ~2 hours with 1.2-1.5x allocation speedup and reduced maintenance burden.
