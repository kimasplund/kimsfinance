# Quick Fix Guide for Integration Errors

## Fix 1: Add Debug constraint to PersistentIndicator

**File**: `src/gpu/persistent/mod.rs`

**Find** (around line 40):
```rust
pub trait PersistentIndicator {
    type Params: Clone;
```

**Replace with**:
```rust
pub trait PersistentIndicator {
    type Params: Clone + std::fmt::Debug;
```

---

## Fix 2: Make allocate_batch_buffers generic

**File**: `src/gpu/persistent/mod.rs`

**Find** (line 339):
```rust
fn allocate_batch_buffers(device: &GpuDevice, batch: &TaskBatch) -> Result<BatchBuffers, GpuError> {
```

**Replace with**:
```rust
fn allocate_batch_buffers<I: PersistentIndicator>(
    device: &GpuDevice,
    batch: &TaskBatch<I>
) -> Result<BatchBuffers, GpuError> {
```

**Update** (around line 346):
```rust
for &size in &batch.sizes {
```

**Replace with**:
```rust
for size in batch.tasks().iter().map(|t| t.data.len()) {
```

---

## Fix 3: Initialize pinned memory fields

**File**: `src/gpu/persistent/mod.rs`

**Find** (line 389):
```rust
Ok(BatchBuffers {
    d_inputs,
    d_outputs,
    d_param_buffer,
})
```

**Replace with**:
```rust
Ok(BatchBuffers {
    d_inputs,
    d_outputs,
    d_param_buffer,
    h_inputs: Vec::new(),     // Pinned memory disabled for now
    h_outputs: Vec::new(),
    using_pinned: false,
})
```

---

## Fix 4: Make launch_persistent_kernel generic

**File**: `src/gpu/persistent/mod.rs`

**Find** (line 402):
```rust
fn launch_persistent_kernel(
    kernel: &CudaFunction,
    batch: &TaskBatch,
```

**Replace with**:
```rust
fn launch_persistent_kernel<I: PersistentIndicator>(
    kernel: &CudaFunction,
    batch: &TaskBatch<I>,
```

---

## Fix 5: Make execute_batch generic

**File**: `src/gpu/persistent/mod.rs`

**Find** (line 525):
```rust
pub fn execute_batch(device: &GpuDevice, batch: &TaskBatch) -> Result<Vec<Vec<f64>>, GpuError> {
```

**Replace with**:
```rust
pub fn execute_batch<I: PersistentIndicator>(
    device: &GpuDevice,
    batch: &TaskBatch<I>
) -> Result<Vec<Vec<f64>>, GpuError> {
```

---

## Fix 6: Make PersistentKernelManager::execute_batch generic

**File**: `src/gpu/persistent/mod.rs`

**Find** (line 250):
```rust
pub fn execute_batch(&self, batch: &TaskBatch) -> Result<Vec<Vec<f64>>, GpuError> {
```

**Replace with**:
```rust
pub fn execute_batch<I: PersistentIndicator>(
    &self,
    batch: &TaskBatch<I>
) -> Result<Vec<Vec<f64>>, GpuError> {
```

---

## Fix 7: Add ValidAsZeroBits bound to GpuDevice::alloc_buffer

**File**: `src/gpu/device.rs`

**Find** (around line 250):
```rust
pub fn alloc_buffer<T>(&self, len: usize) -> Result<CudaSlice<T>, GpuError> {
```

**Replace with**:
```rust
pub fn alloc_buffer<T: cudarc::driver::ValidAsZeroBits>(
    &self,
    len: usize
) -> Result<CudaSlice<T>, GpuError> {
```

---

## Fix 8: Fix cudarc htod_copy_into API

**File**: `src/gpu/device.rs`

**Find** (line 287):
```rust
self.stream
    .htod_copy_into(pinned.as_slice(), dst)
```

**Replace with**:
```rust
self.stream
    .htod_sync_copy_into(pinned.as_slice(), dst)
```

---

## Fix 9: Verify dtoh_sync_copy_into API

**File**: `src/gpu/device.rs`

**Check** (line 318):
```rust
self.stream
    .dtoh_sync_copy_into(src, pinned.as_mut_slice())
```

**If error persists, try**:
```rust
self.stream
    .dtoh_sync_copy(src, pinned.as_mut_slice())
```

---

## Verification Commands

After each fix:
```bash
cargo check --features gpu
```

After all fixes:
```bash
cargo clippy --features gpu --allow=deprecated --allow=unused_variables
cargo build --release --features gpu
cargo test --features gpu
```

---

## Expected Results

✅ All compilation errors resolved
✅ Build succeeds
✅ Tests compile (may need to be written)
✅ Examples compile

⚠️ Clippy warnings may remain (can be addressed separately)
