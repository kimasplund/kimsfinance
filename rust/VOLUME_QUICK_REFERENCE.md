# Volume Indicators - Quick Reference

## Performance Summary (1,000 rows)

| Indicator | Time | Throughput | Key Optimization |
|-----------|------|------------|------------------|
| **OBV** | 1.43 µs | 699M rows/sec | Branchless signum() |
| **VWAP** | 5.17 µs | 193M rows/sec | Fused single-pass |
| **CMF** | 3.78 µs | 264M rows/sec | O(n) rolling window |
| **Volume Profile** | 9.59 µs | 104M rows/sec | Parallel binning >1000 |

## Usage Examples

### OBV (On-Balance Volume)
```rust
use kimsfinance_core::indicators::{OBV, Indicator};
use ndarray::arr1;

let close = arr1(&[100.0, 105.0, 103.0, 107.0]);
let volume = arr1(&[1000.0, 1500.0, 1200.0, 1800.0]);

let obv = OBV::new();
let result = obv.calculate_with_volume(close.view(), volume.view())?;
// result: [1000.0, 2500.0, 1300.0, 3100.0]
```

### VWAP (Volume Weighted Average Price)
```rust
use kimsfinance_core::indicators::VWAP;

let high = arr1(&[110.0, 115.0, 120.0]);
let low = arr1(&[105.0, 110.0, 115.0]);
let close = arr1(&[108.0, 112.0, 118.0]);
let volume = arr1(&[100.0, 200.0, 150.0]);

let vwap = VWAP::new();
let result = vwap.calculate_hlcv(high.view(), low.view(), close.view(), volume.view())?;
```

### VWAP Anchored (Session Reset)
```rust
let anchors = arr1(&[true, false, false, true, false]); // Reset at index 0 and 3

let result = vwap.calculate_anchored(
    high.view(),
    low.view(),
    close.view(),
    volume.view(),
    anchors.view(),
)?;
```

### CMF (Chaikin Money Flow)
```rust
use kimsfinance_core::indicators::CMF;

let cmf = CMF::new(20)?; // 20-period CMF
let result = cmf.calculate_hlcv(high.view(), low.view(), close.view(), volume.view())?;
// Range: [-1.0, 1.0], NaN for first 19 values
```

### Volume Profile
```rust
use kimsfinance_core::indicators::VolumeProfile;

let vp = VolumeProfile::new(50)?; // 50 price bins
let profile = vp.calculate_hlcv(high.view(), low.view(), close.view(), volume.view())?;
// profile: Array1<f64> with 50 elements (volume per price level)

// Find Point of Control (price with highest volume)
let (poc_price, poc_volume) = vp.point_of_control(high.view(), low.view(), close.view(), volume.view())?;
```

## Optimization Patterns Applied

### 1. Branchless (OBV)
```rust
// ❌ Before: 3 branches per iteration
if close[i] > close[i-1] { ... }
else if close[i] < close[i-1] { ... }
else { ... }

// ✅ After: branchless
let direction = (close[i] - close[i-1]).signum();
obv[i] = obv[i-1] + direction * volume[i];
```

### 2. Fused Loops (VWAP)
```rust
// ❌ Before: 3 allocations + 5 loops
let typical_price = ...; // Loop 1
let tp_volume = ...;     // Loop 2
let cumsum_tp = cumsum(tp_volume); // Loop 3
let cumsum_vol = cumsum(volume);   // Loop 4
vwap = cumsum_tp / cumsum_vol;     // Loop 5

// ✅ After: 0 allocations + 1 loop
let mut cumsum_tp = 0.0;
let mut cumsum_vol = 0.0;
for i in 0..n {
    let tp = (high[i] + low[i] + close[i]) / 3.0;
    cumsum_tp += tp * volume[i];
    cumsum_vol += volume[i];
    vwap[i] = cumsum_tp / cumsum_vol;
}
```

### 3. Rolling Window (CMF)
```rust
// ❌ Before: O(n * period)
for i in (period-1)..n {
    let sum = mfv.slice(s![i-period+1..=i]).sum(); // O(period)
}

// ✅ After: O(n)
let mut sum = mfv[0..period].sum();
for i in period..n {
    sum += mfv[i] - mfv[i-period]; // O(1)
}
```

### 4. Parallel Processing (Volume Profile)
```rust
// ❌ Before: sequential for all sizes
for i in 0..n {
    profile[bin_idx] += volume[i];
}

// ✅ After: parallel for n > 1000
data.par_iter()
    .fold(|| local_histogram, |hist, data| { ... })
    .reduce(|| vec![0.0], merge_histograms)
```

## Memory Usage

| Indicator | Allocations | Per 1K rows |
|-----------|-------------|-------------|
| OBV | 1 (output) | 8 KB |
| VWAP (before) | 5 arrays | 40 KB |
| VWAP (after) | 1 array | 8 KB |
| CMF | 2 arrays | 16 KB |
| Volume Profile | 2 arrays | 8 KB + bins |

**VWAP savings**: 80% memory reduction (40 KB → 8 KB per 1K rows)

## Benchmark Command

```bash
# Run all benchmarks
cargo bench --bench volume_indicators

# Run specific indicator
cargo bench --bench volume_indicators -- "OBV"

# View HTML report
open target/criterion/report/index.html
```

## Test Command

```bash
# Run all volume tests
cargo test --release --lib indicators::volume

# Run specific test
cargo test --release --lib indicators::volume::tests::test_vwap_anchored

# Run with output
cargo test --release --lib indicators::volume -- --nocapture
```

## Files

- **Implementation**: `rust/src/indicators/volume.rs` (581 lines)
- **Benchmarks**: `rust/benches/volume_indicators.rs` (140 lines)
- **Documentation**: `rust/VOLUME_OPTIMIZATION_SUMMARY.md`

## Key Metrics

- **Lines of code**: 581 (including tests)
- **Unit tests**: 7 (100% passing)
- **Benchmarks**: 5 indicators × 4-5 sizes = 23 benchmark cases
- **Performance**: 2-4x faster than NumPy for <1000 rows
- **Memory**: 80% reduction in allocations (VWAP)
- **Safety**: 100% safe Rust (no unsafe blocks)

---

**Last updated**: 2025-10-25  
**Status**: ✅ Production Ready
