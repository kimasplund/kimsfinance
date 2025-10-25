# cudarc 0.17.3 Migration Checklist

Quick checklist for fixing our GPU implementation to use correct cudarc 0.17.3 API.

## Files to Update

### 1. `/home/kim-asplund/projects/kimsfinance/rust/src/gpu/device.rs`

#### Changes Required:

- [ ] **Line 5**: Add `CudaContext` to imports
  ```rust
  use cudarc::driver::{CudaContext, CudaSlice, CudaStream, result::DriverError};
  ```

- [ ] **Line 11-13**: Update `GpuDevice` struct
  ```rust
  pub struct GpuDevice {
      context: Arc<CudaContext>,  // ADD THIS
      pub(crate) stream: Arc<CudaStream>,
  }
  ```

- [ ] **Line 34-40**: Store context in `with_device_id`
  ```rust
  let context = CudaContext::new(device_id)?;
  let stream = context.default_stream();

  Ok(Self {
      context: Arc::new(context),  // ADD THIS
      stream,
  })
  ```

- [ ] **Line 59-63**: Fix `copy_to_device` method
  ```rust
  pub fn copy_to_device(&self, data: &[f64]) -> Result<CudaSlice<f64>, GpuError> {
      let data_vec = data.to_vec();  // Convert to Vec
      self.stream
          .memcpy_htod(&data_vec)  // Changed from htod_sync_copy
          .map_err(|e| GpuError::MemoryCopyError(...))
  }
  ```

- [ ] **Line 70-74**: Fix `copy_to_host` method
  ```rust
  pub fn copy_to_host(&self, buffer: &CudaSlice<f64>) -> Result<Vec<f64>, GpuError> {
      self.stream
          .memcpy_dtov(buffer)  // Changed from dtoh_sync_copy
          .map_err(|e| GpuError::MemoryCopyError(...))
  }
  ```

- [ ] **Add accessor method for context**
  ```rust
  pub fn context(&self) -> &Arc<CudaContext> {
      &self.context
  }
  ```

---

### 2. `/home/kim-asplund/projects/kimsfinance/rust/src/gpu/stochastic.rs`

#### Changes Required:

- [ ] **Line 5**: Add missing imports
  ```rust
  use cudarc::driver::LaunchConfig;
  ```

- [ ] **Line 131-134**: Fix module loading (use context, not stream!)
  ```rust
  // OLD (wrong):
  device.stream.load_ptx(ptx, "stochastic_module", &["stochastic_oscillator_kernel"])?;

  // NEW (correct):
  let module = device.context()
      .load_module(ptx)
      .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;
  ```

- [ ] **Line 137-140**: Fix function retrieval (use module, not stream!)
  ```rust
  // OLD (wrong):
  let func = device.stream.get_func("stochastic_module", "stochastic_oscillator_kernel")?;

  // NEW (correct):
  let kernel = module
      .load_function("stochastic_oscillator_kernel")
      .map_err(|e| GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e)))?;
  ```

- [ ] **Line 155-172**: Fix kernel launch (use builder pattern!)
  ```rust
  // OLD (wrong):
  unsafe {
      func.clone()
          .launch_on_stream(
              &device.stream,
              (num_blocks as u32, 1, 1),
              (threads_per_block as u32, 1, 1),
              (
                  &d_high, &d_low, &d_close,
                  &d_k_line, &d_d_line,
                  n as i32, k_period as i32, d_period as i32,
              ),
          )?;
  }

  // NEW (correct):
  let mut builder = device.stream.launch_builder(&kernel);
  builder.arg(&d_high);
  builder.arg(&d_low);
  builder.arg(&d_close);
  builder.arg(&mut d_k_line);  // Note: mutable args need &mut
  builder.arg(&mut d_d_line);
  builder.arg(&(n as i32));
  builder.arg(&(k_period as i32));
  builder.arg(&(d_period as i32));

  let config = LaunchConfig::for_num_elems(n as u32);
  unsafe {
      builder.launch(config)
          .map_err(|e| GpuError::ExecutionError(format!("Kernel launch failed: {:?}", e)))?;
  }
  ```

---

## Testing After Migration

### 1. Compile Check
```bash
cd /home/kim-asplund/projects/kimsfinance/rust
cargo check --features gpu
```

### 2. Run Tests (requires GPU)
```bash
cargo test --features gpu -- --ignored
```

### 3. Specific GPU Tests
```bash
# Device initialization
cargo test --features gpu test_device_initialization -- --ignored --nocapture

# Memory operations
cargo test --features gpu test_memory_operations -- --ignored --nocapture

# Stochastic oscillator
cargo test --features gpu test_stochastic_gpu -- --ignored --nocapture
```

---

## Common Errors and Solutions

### Error: "no method named `htod_sync_copy`"
**Solution**: Replace with `memcpy_htod(&data.to_vec())`

### Error: "no method named `load_ptx` found for struct `CudaStream`"
**Solution**: Use `context.load_module(ptx)` instead

### Error: "no method named `get_func` found for struct `CudaStream`"
**Solution**: Use `module.load_function(name)` instead

### Error: "no method named `launch_on_stream`"
**Solution**: Use builder pattern: `stream.launch_builder(&kernel)`

### Error: "cannot borrow `*builder` as mutable, as it is also borrowed as immutable"
**Solution**: All args must be passed by reference: `builder.arg(&value)` or `builder.arg(&mut value)`

---

## Verification Checklist

Before committing:

- [ ] All compilation errors resolved
- [ ] `cargo check --features gpu` passes
- [ ] Test `test_device_initialization` passes
- [ ] Test `test_memory_operations` passes
- [ ] Test `test_stochastic_gpu` passes
- [ ] No clippy warnings: `cargo clippy --features gpu`
- [ ] Documentation updated if needed

---

## Quick Reference

See `/home/kim-asplund/projects/kimsfinance/rust/CUDARC_API_RESEARCH.md` for:
- Complete API documentation
- Working examples
- Performance considerations
- Method signatures
- Detailed explanations

---

**Last Updated**: 2025-10-25
**Target Version**: cudarc 0.17.3
