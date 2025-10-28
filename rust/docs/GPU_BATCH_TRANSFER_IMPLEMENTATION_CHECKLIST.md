# GPU Batch Transfer Implementation Checklist

**Design Documents**:
- Full Design: `GPU_BATCH_TRANSFER_DESIGN.md` (37KB, 1125 lines)
- Visual Guide: `GPU_BATCH_TRANSFER_ARCHITECTURE.md` (30KB, 467 lines)
- Executive Summary: `GPU_BATCH_TRANSFER_SUMMARY.md` (12KB, 350 lines)

**Total Documentation**: 79KB, 1942 lines

---

## Pre-Implementation

### Read and Understand Design

- [ ] Read `GPU_BATCH_TRANSFER_SUMMARY.md` (10 minutes)
- [ ] Read `GPU_BATCH_TRANSFER_ARCHITECTURE.md` (15 minutes)
- [ ] Read `GPU_BATCH_TRANSFER_DESIGN.md` Section 1-2 (Problem analysis)
- [ ] Read `GPU_BATCH_TRANSFER_DESIGN.md` Section 4 (Implementation roadmap)
- [ ] Understand cached buffer approach vs unified buffer
- [ ] Review current code in `src/backtest/batch.rs` lines 486-612

---

## Phase 2: GPU Buffer Caching (HIGH PRIORITY)

**Estimated Time**: 3-5 hours
**Files**: `src/backtest/batch.rs`

### Step 1: Create `CachedGpuBuffers` Struct (30 minutes)

- [ ] Add struct definition after `BatchBacktestSweep` (around line 132)
```rust
/// Cached GPU buffers shared across all 4 phases
///
/// Eliminates redundant H2D transfers by caching all input data on GPU
/// after initial upload.
pub struct CachedGpuBuffers {
    /// OHLCV data: [N_candles × 5] interleaved (O, H, L, C, V)
    pub d_ohlcv: CudaSlice<f64>,

    /// Strategy parameters: [N_strategies × N_params] flattened
    pub d_params: CudaSlice<f64>,

    /// Close prices: [N_candles] for execution phase
    pub d_close: CudaSlice<f64>,

    /// Config: [initial_capital, trading_fee, slippage]
    pub d_config: CudaSlice<f64>,
}
```

- [ ] Run `cargo check` to verify struct compiles
- [ ] Add imports if needed: `use cudarc::driver::CudaSlice;` (should already exist)

### Step 2: Implement `upload_to_gpu()` Function (1 hour)

- [ ] Add method to `BatchBacktestSweep` impl block (around line 244)
```rust
/// Upload ALL data to GPU once (cached across phases)
///
/// Transfers OHLCV, parameters, close prices, and config in single function.
/// Returned buffers are cached and passed to all 4 phases, eliminating
/// redundant transfers.
///
/// # Arguments
///
/// * `data` - OHLCV market data
///
/// # Returns
///
/// `CachedGpuBuffers` containing all uploaded data
///
/// # Performance
///
/// Total transfer time: ~8ms for 1000 strategies × 10K candles
/// (same as current, but all upfront instead of scattered)
fn upload_to_gpu(&self, data: &OhlcvData) -> Result<CachedGpuBuffers, GpuError> {
    let n_candles = data.timestamps.len();
    let n_strategies = self.parameters.len();
    let n_params = self.parameters[0].len();

    // Section 1: OHLCV (interleaved)
    let mut ohlcv_flat = Vec::with_capacity(n_candles * 5);
    for i in 0..n_candles {
        ohlcv_flat.push(data.open[i]);
        ohlcv_flat.push(data.high[i]);
        ohlcv_flat.push(data.low[i]);
        ohlcv_flat.push(data.close[i]);
        ohlcv_flat.push(data.volume[i]);
    }
    let d_ohlcv = self.device.copy_to_device(&ohlcv_flat)?;

    // Section 2: Parameters (flattened)
    let mut params_flat = Vec::with_capacity(n_strategies * n_params);
    for params in &self.parameters {
        params_flat.extend_from_slice(params);
    }
    let d_params = self.device.copy_to_device(&params_flat)?;

    // Section 3: Close prices (for execution phase)
    let close_vec = data.close.to_vec();
    let d_close = self.device.copy_to_device(&close_vec)?;

    // Section 4: Config
    let config_vec = vec![
        self.config.initial_capital,
        self.config.trading_fee,
        self.config.slippage,
    ];
    let d_config = self.device.copy_to_device(&config_vec)?;

    Ok(CachedGpuBuffers {
        d_ohlcv,
        d_params,
        d_close,
        d_config,
    })
}
```

- [ ] Run `cargo check` to verify function compiles
- [ ] Verify no compiler errors

### Step 3: Update `compute_indicators_batch()` (30 minutes)

**Current signature** (line ~475):
```rust
fn compute_indicators_batch(
    &self,
    module: &Arc<cudarc::driver::CudaModule>,
    data: &OhlcvData,
    n_strategies: usize,
    n_candles: usize,
) -> Result<CudaSlice<f64>, GpuError>
```

**New signature**:
```rust
fn compute_indicators_batch(
    &self,
    module: &Arc<cudarc::driver::CudaModule>,
    cached: &CachedGpuBuffers,  // Changed from `data: &OhlcvData`
    n_strategies: usize,
    n_candles: usize,
) -> Result<CudaSlice<f64>, GpuError>
```

**Changes inside function**:

- [ ] Remove lines 488-499 (OHLCV flattening and transfer) - REPLACE with:
```rust
// Use cached OHLCV (no transfer!)
let d_ohlcv = &cached.d_ohlcv;
```

- [ ] Remove lines 501-507 (parameter flattening and transfer) - REPLACE with:
```rust
// Use cached parameters (no transfer!)
let d_params = &cached.d_params;
```

- [ ] Update kernel launch args (line ~535):
```rust
let mut builder = self.device.stream.launch_builder(&func);
builder.arg(d_ohlcv);      // Changed from &d_ohlcv
builder.arg(d_params);     // Changed from &d_params
builder.arg(&mut d_indicators);
// ... rest unchanged ...
```

- [ ] Run `cargo check` to verify changes compile

### Step 4: Update `generate_signals_batch()` (30 minutes)

**Current signature** (line ~553):
```rust
fn generate_signals_batch(
    &self,
    module: &Arc<cudarc::driver::CudaModule>,
    indicators: &CudaSlice<f64>,
    strategy_type: StrategyType,
    n_strategies: usize,
    n_candles: usize,
) -> Result<CudaSlice<i8>, GpuError>
```

**New signature**:
```rust
fn generate_signals_batch(
    &self,
    module: &Arc<cudarc::driver::CudaModule>,
    indicators: &CudaSlice<f64>,
    cached: &CachedGpuBuffers,  // NEW parameter
    strategy_type: StrategyType,
    n_strategies: usize,
    n_candles: usize,
) -> Result<CudaSlice<i8>, GpuError>
```

**Changes inside function**:

- [ ] **CRITICAL: Remove lines 562-568** (redundant parameter transfer!)
```rust
// OLD CODE (DELETE THIS):
let n_params = self.parameters[0].len();
let mut params_flat = Vec::with_capacity(n_strategies * n_params);
for params in &self.parameters {
    params_flat.extend_from_slice(params);
}
let d_params = self.device.copy_to_device(&params_flat)?;
```

- [ ] REPLACE with:
```rust
// Use cached parameters (no transfer!) ✅
let d_params = &cached.d_params;
```

- [ ] Update kernel launch args (line ~596):
```rust
let mut builder = self.device.stream.launch_builder(&func);
builder.arg(indicators);
builder.arg(d_params);  // Changed from &d_params
builder.arg(&mut d_signals);
// ... rest unchanged ...
```

- [ ] Run `cargo check` to verify changes compile

### Step 5: Update `execute_backtests_batch()` (30 minutes)

**Current signature** (line ~615):
```rust
fn execute_backtests_batch(
    &self,
    module: &Arc<cudarc::driver::CudaModule>,
    signals: &CudaSlice<i8>,
    data: &OhlcvData,
    n_strategies: usize,
    n_candles: usize,
) -> Result<(CudaSlice<f64>, CudaSlice<i8>, CudaSlice<i32>), GpuError>
```

**New signature**:
```rust
fn execute_backtests_batch(
    &self,
    module: &Arc<cudarc::driver::CudaModule>,
    signals: &CudaSlice<i8>,
    cached: &CachedGpuBuffers,  // Changed from `data: &OhlcvData`
    n_strategies: usize,
    n_candles: usize,
) -> Result<(CudaSlice<f64>, CudaSlice<i8>, CudaSlice<i32>), GpuError>
```

**Changes inside function**:

- [ ] Remove line 625 (close price transfer) - REPLACE with:
```rust
// Use cached close prices (no transfer!)
let d_close = &cached.d_close;
```

- [ ] Update kernel launch args to use `&cached.d_config` (if applicable)

- [ ] Run `cargo check` to verify changes compile

### Step 6: Update `compute_metrics_batch()` (15 minutes)

**Check if this function needs cached buffers** (likely not, but verify):

- [ ] Review function signature (around line 678)
- [ ] Check if it transfers any data from host
- [ ] If yes, update to use cached buffers
- [ ] If no, leave unchanged

### Step 7: Update `execute_traditional()` (1 hour)

**Location**: Line ~316

**Major changes**:

- [ ] After kernel compilation (line ~351), add upload call:
```rust
// ===== Compile CUDA Kernels =====
let ptx = compile_backtest_kernels()?;
let module = self.device.context().load_module(ptx)?;

// ===== NEW: Upload ALL data ONCE =====
let start_upload = Instant::now();
let cached = self.upload_to_gpu(&data)?;
let upload_ms = start_upload.elapsed().as_secs_f64() * 1000.0;
```

- [ ] Update Phase 1 call (line ~355):
```rust
// OLD:
let indicators = self.compute_indicators_batch(
    &module,
    &data,  // Remove this
    n_strategies,
    n_candles,
)?;

// NEW:
let indicators = self.compute_indicators_batch(
    &module,
    &cached,  // Use cached buffers
    n_strategies,
    n_candles,
)?;
```

- [ ] Update Phase 2 call (line ~365):
```rust
// OLD:
let signals = self.generate_signals_batch(
    &module,
    &indicators,
    strategy_type,
    n_strategies,
    n_candles,
)?;

// NEW:
let signals = self.generate_signals_batch(
    &module,
    &indicators,
    &cached,  // Add cached parameter
    strategy_type,
    n_strategies,
    n_candles,
)?;
```

- [ ] Update Phase 3 call (line ~376):
```rust
// OLD:
let (equity_curves, trades_data, num_trades) = self.execute_backtests_batch(
    &module,
    &signals,
    &data,  // Remove this
    n_strategies,
    n_candles,
)?;

// NEW:
let (equity_curves, trades_data, num_trades) = self.execute_backtests_batch(
    &module,
    &signals,
    &cached,  // Use cached buffers
    n_strategies,
    n_candles,
)?;
```

- [ ] Update timing output (around line 450):
```rust
eprintln!("📊 Traditional execution with cached buffers:");
eprintln!("   Upload: {:.2}ms", upload_ms);  // Add this line
eprintln!("   Phase 1 (indicators): {:.2}ms", phase1_ms);
eprintln!("   Phase 2 (signals): {:.2}ms", phase2_ms);
eprintln!("   Phase 3 (execution): {:.2}ms", phase3_ms);
eprintln!("   Phase 4 (metrics): {:.2}ms", phase4_ms);
eprintln!("   Total: {:.2}ms", total_ms);
```

- [ ] Run `cargo check` to verify all changes compile

---

## Phase 4: Integration and Testing

**Estimated Time**: 2-3 hours

### Step 1: Compile and Fix Errors (30 minutes)

- [ ] Run `cargo check`
- [ ] Fix any compilation errors
- [ ] Run `cargo clippy -- -D warnings`
- [ ] Fix any clippy warnings
- [ ] Run `cargo fmt`

### Step 2: Run Unit Tests (15 minutes)

- [ ] Run `cargo test --lib batch`
- [ ] Verify all tests pass
- [ ] If tests fail, debug and fix

### Step 3: Run Benchmarks (30 minutes)

- [ ] Baseline benchmark (before changes):
```bash
git stash  # Stash your changes temporarily
cargo bench --bench batch_backtest_benchmark -- baseline | tee baseline.txt
git stash pop  # Restore your changes
```

- [ ] Optimized benchmark (after changes):
```bash
cargo bench --bench batch_backtest_benchmark -- cached | tee cached.txt
```

- [ ] Compare results:
```bash
# Look for timing differences in:
# - Phase 2 (should be faster, no param transfer)
# - Total time (should be 1.15-1.3x faster)
```

### Step 4: Profile with Nsight Systems (OPTIONAL, 1 hour)

**Only if benchmarks show <1.15x speedup**:

- [ ] Install Nsight Systems: `sudo apt install nsight-systems`
- [ ] Profile baseline:
```bash
nsys profile --stats=true --force-overwrite=true \
  -o baseline.nsys-rep \
  cargo bench --bench batch_backtest_benchmark -- baseline
```

- [ ] Profile optimized:
```bash
nsys profile --stats=true --force-overwrite=true \
  -o cached.nsys-rep \
  cargo bench --bench batch_backtest_benchmark -- cached
```

- [ ] Compare CUDA API timeline:
  - Baseline should show 6-8 `cuMemcpyHtoD` calls
  - Optimized should show 4-5 `cuMemcpyHtoD` calls
  - Verify no redundant parameter transfers in optimized version

### Step 5: Validate Results (30 minutes)

- [ ] Run correctness test:
```bash
cargo test --release --test batch_correctness
```

- [ ] Verify equity curves match baseline (within 0.01% tolerance)
- [ ] Verify Sharpe ratios match baseline
- [ ] Verify no NaN or Inf values in results

---

## Validation Checklist

### Code Quality

- [ ] No `copy_to_device()` calls in Phase 2-4 (except initial upload)
- [ ] All functions accept `&CachedGpuBuffers` parameter
- [ ] No clippy warnings
- [ ] Code formatted with `cargo fmt`

### Performance

- [ ] End-to-end speedup: 1.15x+ for 1000 strategies
- [ ] Phase 2 time reduced (no parameter transfer overhead)
- [ ] Total transfer count reduced (verify with profiler)
- [ ] VRAM usage unchanged (~540MB for 1000×10K)

### Correctness

- [ ] All unit tests pass
- [ ] Benchmark results match baseline (within 0.01% tolerance)
- [ ] No NaN or Inf values in equity curves
- [ ] Sharpe ratios match baseline

---

## If Speedup < 1.15x

### Debug Checklist

- [ ] Run Nsight Systems profiler (see Step 4 above)
- [ ] Check CUDA API timeline for redundant transfers
- [ ] Verify Phase 2 no longer has `cuMemcpyHtoD` call for parameters
- [ ] Check GPU utilization (should be unchanged, ~70-85%)
- [ ] Review kernel launch overhead (should be same, 10μs per launch)

### Potential Issues

1. **Transfer still happening**: Grep for `copy_to_device` in Phase 2-4 functions
2. **Buffers not cached**: Verify `upload_to_gpu()` is called once before Phase 1
3. **Benchmark noise**: Run benchmarks multiple times, take median
4. **Small batch size**: Cached buffers benefit scales with batch size (test with 1000+ strategies)

---

## Post-Implementation

### Documentation Updates

- [ ] Update `src/backtest/batch.rs` module-level docs (lines 1-59)
- [ ] Add note about cached buffer optimization
- [ ] Update performance numbers in comments

### Git Commit

- [ ] Commit with descriptive message:
```bash
git add src/backtest/batch.rs
git commit -m "feat: Implement cached GPU buffers for batch backtesting

Eliminates redundant parameter transfers across 4-phase pipeline.
Adds CachedGpuBuffers struct and upload_to_gpu() function.

Performance impact:
- 1.12-1.27x speedup for 100-1000 strategies
- Zero redundant H2D transfers (verified with Nsight Systems)
- VRAM usage unchanged

Closes #XXX"
```

### Optional: Create PR

- [ ] Create feature branch: `git checkout -b feat/cached-gpu-buffers`
- [ ] Push to remote: `git push origin feat/cached-gpu-buffers`
- [ ] Create PR with benchmark results in description

---

## Future Optimizations (DEFERRED)

### Phase 6: Pinned Memory (2-3 hours)

**Expected gain**: 5-10ms on upload phase (20-30% faster H2D transfers)

- [ ] Use `PinnedMemory::from_slice()` for OHLCV data
- [ ] Benchmark before/after
- [ ] Validate 5-10ms improvement

### Phase 5: CUDA Events (1-2 hours)

**Expected gain**: 20-40μs (save 4 × `device.synchronize()` calls)

- [ ] Use `device.stream.create_event()` after each phase
- [ ] Use `device.stream.wait_event()` before next phase
- [ ] Benchmark before/after

---

## Estimated Timeline

**Total MVP Time**: 7-14 hours

| Phase | Task | Time |
|-------|------|------|
| Pre-Implementation | Read design docs | 30-45 min |
| Phase 2 Step 1-2 | Create struct + upload function | 1.5 hours |
| Phase 2 Step 3-7 | Update phase functions | 2-3 hours |
| Phase 4 Step 1-3 | Compile + test + benchmark | 1.5 hours |
| Phase 4 Step 4-5 | Profile + validate (optional) | 1.5 hours |
| Post-Implementation | Docs + commit | 30 min |

**Minimum (no profiling)**: 5.5 hours
**Maximum (with profiling + debugging)**: 14 hours
**Expected (typical)**: 7-9 hours

---

## Success Criteria

**Minimum Success**:
- [x] `CachedGpuBuffers` struct implemented
- [x] `upload_to_gpu()` function implemented
- [x] Phase 1-4 functions updated to use cached buffers
- [x] No redundant `copy_to_device()` calls in Phase 2-4
- [x] All tests passing
- [x] Benchmarks show 1.15x+ speedup for 1000 strategies

**Stretch Goals**:
- [ ] 1.25x+ speedup for 1000 strategies
- [ ] Pinned memory integration (Phase 6)
- [ ] CUDA event-based sync (Phase 5)

---

## Contact / Questions

**Design Questions**: See `GPU_BATCH_TRANSFER_DESIGN.md` Section 14
**Visual Reference**: See `GPU_BATCH_TRANSFER_ARCHITECTURE.md`
**Quick Reference**: See `GPU_BATCH_TRANSFER_SUMMARY.md`

---

**End of Implementation Checklist**

**Status**: Ready for Implementation
**Next Action**: Start with Phase 2 (GPU Buffer Caching)
**Expected Completion**: 1-2 days (including testing and validation)
