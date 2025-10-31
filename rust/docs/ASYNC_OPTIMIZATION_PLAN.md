# Async Pinned Memory Optimization Plan

## Overview
Optimize all GPU kernels to use async pinned memory transfers instead of sync transfers.
Based on ATR benchmark: 163μs → 145μs (11% speedup)

## Files to Optimize (27 total)

### Group 1: Simple Indicators (2-3 transfers) - 6 files
**Agent 1 Task:**
- `src/gpu/ema.rs` (2 transfers)
- `src/gpu/roc.rs` (2 transfers)
- `src/gpu/wma.rs` (2 transfers)
- `src/gpu/batch.rs` (3 transfers)
- `src/gpu/obv.rs` (3 transfers)
- `src/gpu/vwma.rs` (3 transfers)

### Group 2: Medium Indicators (4-5 transfers) - 11 files
**Agent 2 Task:**
- `src/gpu/bollinger.rs` (4 transfers)
- `src/gpu/cci.rs` (4 transfers)
- `src/gpu/macd.rs` (4 transfers)
- `src/gpu/sma.rs` (4 transfers)
- `src/gpu/williams_r.rs` (4 transfers)
- `src/gpu/cmf.rs` (5 transfers)
- `src/gpu/donchian.rs` (5 transfers)
- `src/gpu/elder_ray.rs` (5 transfers)
- `src/gpu/keltner.rs` (5 transfers)
- `src/gpu/stochastic.rs` (5 transfers)
- `src/gpu/vwap.rs` (5 transfers)

### Group 3: Complex Indicators (6-10 transfers) - 7 files
**Agent 3 Task:**
- `src/gpu/rsi_sync.rs` (6 transfers)
- `src/gpu/supertrend.rs` (7 transfers)
- `src/gpu/heston_pricing.rs` (9 transfers)
- `src/gpu/pivot_points.rs` (10 transfers)
- `src/backtest/batch.rs` (8 transfers)
- `src/backtest/persistent.rs` (7 transfers)

### Group 4: Very Complex (11+ transfers) - 3 files
**Agent 4 Task:**
- `src/gpu/kernels_2d.rs` (16 transfers)
- `src/gpu/kernels_3d.rs` (12 transfers)
- `src/gpu/aggregation.rs` (13 transfers)

## Optimization Pattern

### Before (Sync):
```rust
let d_input = device.copy_to_device(input.as_slice().unwrap())?;
// ... kernel execution ...
let output_vec = device.copy_to_host(&d_output)?;
```

### After (Async with pinned memory):
```rust
// 1. Acquire pinned buffer
let mut pinned_input = device.pinned_pool.lock().acquire(n)?;
pinned_input.as_mut_slice()[..n].copy_from_slice(input.as_slice().unwrap());

// 2. Allocate device buffer
let mut d_input = device.alloc_buffer(n)?;

// 3. Async H2D transfer
kernel_stream.memcpy_htod(&pinned_input.as_slice()[..n], &mut d_input)?;

// 4. Release pinned buffer
device.pinned_pool.lock().release(pinned_input);

// ... kernel execution ...

// 5. Acquire pinned buffer for output
let mut pinned_output = device.pinned_pool.lock().acquire(n)?;

// 6. Async D2H transfer
kernel_stream.memcpy_dtoh(&d_output, &mut pinned_output.as_mut_slice()[..n])?;

// 7. Synchronize before CPU access
kernel_stream.synchronize()?;

// 8. Copy to output array
let output_vec = pinned_output.as_slice()[..n].to_vec();

// 9. Release pinned buffer
device.pinned_pool.lock().release(pinned_output);
```

## Reference Implementation
See `src/gpu/atr.rs` (lines 185-246) for complete working example.

## Expected Results
- 11% speedup per indicator (based on ATR: 163μs → 145μs)
- Better GPU saturation in multi-GPU clusters
- Professional memory management for institutional clients

## Testing
After optimization, run:
```bash
cargo test --lib --features gpu
cargo bench --features gpu
```
