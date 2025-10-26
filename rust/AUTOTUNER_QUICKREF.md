# Auto-Tuner Quick Reference

**Version**: 0.2.0 | **Status**: Production Ready ✅

---

## Quick Start

### 1. Basic Usage (Most Common)

```rust
use kimsfinance_core::autotuner::{AutoTuneProfile, ExecutionStrategy};
use kimsfinance_core::gpu::GpuDevice;

let device = GpuDevice::new()?;
let profile = AutoTuneProfile::get_or_init(&device);

// Auto-select strategy
match profile.select_rsi_strategy(data_size) {
    ExecutionStrategy::CPU => { /* use CPU */ }
    ExecutionStrategy::GPU => { /* use GPU */ }
    ExecutionStrategy::Hybrid => { /* fallback to CPU */ }
}
```

### 2. Manual Calibration

```bash
cargo build --release --features gpu --example calibrate
./target/release/examples/calibrate
```

### 3. Force CPU Mode

```bash
export KIMSFINANCE_FORCE_CPU=1
./my_app
```

### 4. Run Demo

```bash
cargo run --release --features gpu --example autotuner_demo
```

---

## Strategy Selection Cheatsheet

| Indicator          | Method                                  | Logic                                      |
|--------------------|-----------------------------------------|--------------------------------------------|
| EMA                | `select_ema_strategy(n)`                | Always CPU (sequential IIR filter)         |
| Wilder's smoothing | `select_wilders_strategy(n)`            | Always CPU (sequential IIR filter)         |
| RSI                | `select_rsi_strategy(n)`                | CPU if n<5K, else GPU (hybrid)             |
| ATR                | `select_atr_strategy(n)`                | CPU if n<5K, else GPU (hybrid)             |
| Stochastic         | `select_stochastic_strategy(n)`         | CPU if n<threshold (~5K-20K), else GPU     |
| ROC                | `select_roc_strategy(n)`                | CPU if n<threshold (~2K-10K), else GPU     |
| Williams %R        | `select_williams_r_strategy(n)`         | CPU if n<threshold (~5K-20K), else GPU     |
| Bollinger Bands    | `select_bollinger_strategy(n)`          | CPU if n<threshold (~3K-15K), else GPU     |
| MACD               | `select_macd_strategy(n)`               | CPU if n<threshold (~5K-25K), else GPU     |

**Note**: Thresholds are calibrated per machine. Values shown are typical ranges.

---

## Files & Locations

| File                                    | Purpose                          |
|-----------------------------------------|----------------------------------|
| `src/autotuner.rs`                      | Core implementation (1,078 lines)|
| `examples/autotuner_demo.rs`            | Interactive demo                 |
| `examples/calibrate.rs`                 | Manual calibration CLI           |
| `docs/AUTOTUNER_GUIDE.md`               | Full integration guide           |
| `~/.cache/kimsfinance/autotune.json`    | Cached calibration profile       |

---

## Cache Management

### View Cache

```bash
cat ~/.cache/kimsfinance/autotune.json
```

### Clear Cache (Force Re-Calibration)

```bash
rm ~/.cache/kimsfinance/autotune.json
cargo run --release --features gpu --example calibrate
```

### Cache Format

```json
{
  "hardware_id": "cpu:Intel(R) Core(TM) i9-13980HX_gpu:NVIDIA RTX 3500 Ada_ram:64gb",
  "cpu_clock_ghz": 5.6,
  "gpu_clock_ghz": 3.11,
  "vram_bandwidth_gbs": 288.0,
  "ram_bandwidth_gbs": 77.0,
  "thresholds": {
    "ema_crossover": 18446744073709551615,
    "wilders_crossover": 18446744073709551615,
    "stochastic_crossover": 5000,
    "roc_crossover": 2000,
    "williams_r_crossover": 5000,
    "bollinger_crossover": 8000,
    "macd_crossover": 15000,
    "parallel_operations": 1000
  },
  "calibration_timestamp": 1729900800
}
```

---

## Common Patterns

### Pattern 1: Conditional GPU Usage

```rust
use kimsfinance_core::autotuner::AutoTuneProfile;

fn calculate_indicator(device: &GpuDevice, data: &Array1<f64>) -> Result<Array1<f64>> {
    let profile = AutoTuneProfile::get_or_init(device);

    match profile.select_rsi_strategy(data.len()) {
        ExecutionStrategy::CPU => {
            // Fallback to CPU implementation
            rsi_cpu(data, 14)
        }
        ExecutionStrategy::GPU => {
            // Use GPU hybrid
            rsi_gpu(device, data, 14, None)
        }
        ExecutionStrategy::Hybrid => {
            // Custom strategy (future)
            rsi_cpu(data, 14)
        }
    }
}
```

### Pattern 2: Batch Processing with Auto-Selection

```rust
fn process_batch(device: &GpuDevice, batch: &[Array1<f64>]) -> Vec<Result<Array1<f64>>> {
    let profile = AutoTuneProfile::get_or_init(device);

    batch.iter().map(|data| {
        let strategy = profile.select_rsi_strategy(data.len());
        match strategy {
            ExecutionStrategy::CPU => rsi_cpu(data, 14),
            ExecutionStrategy::GPU => rsi_gpu(device, data, 14, None),
            ExecutionStrategy::Hybrid => rsi_cpu(data, 14),
        }
    }).collect()
}
```

### Pattern 3: Adaptive Threshold Override

```rust
// For testing: override auto-tuned thresholds
fn select_rsi_custom(profile: &AutoTuneProfile, data_size: usize, force_gpu: bool) -> ExecutionStrategy {
    if force_gpu {
        ExecutionStrategy::GPU
    } else {
        profile.select_rsi_strategy(data_size)
    }
}
```

---

## Troubleshooting

| Problem                              | Solution                                                   |
|--------------------------------------|------------------------------------------------------------|
| "GPU initialization failed"          | Check `nvidia-smi`, install CUDA driver                    |
| "Failed to create cache directory"   | Verify `$HOME` set, create `~/.cache/kimsfinance` manually |
| Calibration takes >10 seconds        | Normal for first run, reduce test sizes if needed          |
| Wrong strategy selected              | Delete cache, re-calibrate: `rm ~/.cache/kimsfinance/autotune.json` |
| Force CPU not working                | Verify `export KIMSFINANCE_FORCE_CPU=1` in current shell   |

---

## Performance Tips

### Tip 1: Pre-Calibrate in CI/CD

```bash
# In deployment script
cargo build --release --features gpu --example calibrate
./target/release/examples/calibrate
# Cache is now ready for production
```

### Tip 2: Monitor Cache Freshness

```bash
# Check calibration age
ls -lh ~/.cache/kimsfinance/autotune.json

# Re-calibrate monthly or after hardware upgrade
```

### Tip 3: Profile with Both Strategies

```rust
// Benchmark CPU vs GPU for your specific workload
let cpu_time = benchmark_cpu(&data);
let gpu_time = benchmark_gpu(&device, &data);

println!("CPU: {}μs, GPU: {}μs, Speedup: {:.2}x",
         cpu_time / 1000, gpu_time / 1000, cpu_time as f64 / gpu_time as f64);
```

---

## API Quick Reference

### AutoTuneProfile Methods

| Method                            | Returns             | Description                          |
|-----------------------------------|---------------------|--------------------------------------|
| `get_or_init(device)`             | `&'static Self`     | Get singleton profile (lazy init)    |
| `calibrate(device)`               | `Result<Self>`      | Run micro-benchmarks, cache results  |
| `load_from_cache()`               | `Option<Self>`      | Load from disk cache                 |
| `save_to_cache()`                 | `Result<()>`        | Save to disk cache                   |
| `select_ema_strategy(n)`          | `ExecutionStrategy` | Always CPU                           |
| `select_rsi_strategy(n)`          | `ExecutionStrategy` | CPU if n<5K, else GPU                |
| `select_stochastic_strategy(n)`   | `ExecutionStrategy` | Calibrated threshold                 |

### ExecutionStrategy Enum

```rust
pub enum ExecutionStrategy {
    CPU,     // Run on CPU
    GPU,     // Run on GPU (or GPU-heavy hybrid)
    Hybrid,  // Custom hybrid (future)
}
```

---

## Environment Variables

| Variable                  | Default | Effect                            |
|---------------------------|---------|-----------------------------------|
| `KIMSFINANCE_FORCE_CPU`   | 0       | Set to `1` to force CPU-only mode |
| `HOME`                    | N/A     | Cache directory base path          |

---

## Hardware Detection Reference

### CPU Clock Detection

```bash
# What auto-tuner reads
grep "cpu MHz" /proc/cpuinfo | head -1

# What you see (boost clock)
grep "model name" /proc/cpuinfo | head -1
```

### GPU Clock Detection

```bash
# What auto-tuner queries (boost clock)
nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits

# Verify GPU info
nvidia-smi --query-gpu=name,clocks.max.graphics,memory.total --format=csv
```

### RAM Size Detection

```bash
# What auto-tuner reads
grep MemTotal /proc/meminfo
```

---

## Example Output (Demo)

```
═══════════════════════════════════════════════════════
  kimsfinance Auto-Tuner Demo
═══════════════════════════════════════════════════════

📊 Hardware Configuration:
   CPU: 5.60 GHz
   GPU: 3.11 GHz (boost)
   VRAM: 288 GB/s
   RAM: 77 GB/s
   VRAM/RAM ratio: 3.7x

🎯 Calibrated Crossover Thresholds:
   EMA: N/A (always CPU)
   Wilder's (RSI/ATR): N/A (always CPU for sequential part)
   Stochastic: 5000 candles
   ROC: 2000 candles
   Williams %R: 5000 candles
   Bollinger: 8000 candles
   MACD: 15000 candles
   Parallel ops: 1000 elements

🔀 Adaptive Strategy Selection:

   Data Size  │  EMA  │  RSI  │ Stoch │  ROC  │ Will%R│  MACD
   ───────────┼───────┼───────┼───────┼───────┼───────┼───────
          100 │  CPU  │  CPU  │  CPU  │  CPU  │  CPU  │  CPU
         1000 │  CPU  │  CPU  │  CPU  │  CPU  │  CPU  │  CPU
         5000 │  CPU  │  GPU  │  GPU  │  GPU  │  GPU  │  CPU
        10000 │  CPU  │  GPU  │  GPU  │  GPU  │  GPU  │  CPU
        50000 │  CPU  │  GPU  │  GPU  │  GPU  │  GPU  │  GPU
       100000 │  CPU  │  GPU  │  GPU  │  GPU  │  GPU  │  GPU
```

---

## Build Commands Reference

```bash
# Check compilation
cargo check --features gpu

# Build examples
cargo build --release --features gpu --example autotuner_demo
cargo build --release --features gpu --example calibrate

# Run examples
./target/release/examples/autotuner_demo
./target/release/examples/calibrate

# Run tests
cargo test --features gpu autotuner

# Build with all features
cargo build --release --features "gpu,simd"
```

---

## Integration Checklist

When adding auto-tuner to existing indicator:

- [ ] Import auto-tuner: `use crate::autotuner::{AutoTuneProfile, ExecutionStrategy};`
- [ ] Replace hardcoded threshold with `profile.select_*_strategy(n)`
- [ ] Add CPU fallback for `ExecutionStrategy::Hybrid`
- [ ] Update function documentation
- [ ] Test with `KIMSFINANCE_FORCE_CPU=1`
- [ ] Benchmark before/after to verify improvement
- [ ] Update CHANGELOG.md

---

## Related Documentation

- **Full Guide**: `docs/AUTOTUNER_GUIDE.md` (623 lines, comprehensive)
- **Implementation Report**: `AUTOTUNER_IMPLEMENTATION_REPORT.md` (623 lines, technical details)
- **This Quick Ref**: `AUTOTUNER_QUICKREF.md` (you are here)

---

**Last Updated**: 2025-10-25
**Version**: 0.2.0
