# Auto-Tuner Architecture Diagram

**Version**: 0.2.0 | **Date**: 2025-10-25

---

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        kimsfinance Auto-Tuner                           │
│                    Adaptive CPU vs GPU Selection                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ├──── Hardware Detection
                                    ├──── Empirical Benchmarking
                                    ├──── Intelligent Caching
                                    └──── Strategy Selection
```

---

## Component Architecture

```text
┌───────────────────────────────────────────────────────────────────┐
│                      User Application                             │
│                                                                   │
│   fn calculate_rsi(data: &Array1<f64>) -> Result<Array1<f64>> {  │
│       let profile = AutoTuneProfile::get_or_init(&device);       │
│       match profile.select_rsi_strategy(data.len()) {            │
│           ExecutionStrategy::CPU => rsi_cpu(data, 14),           │
│           ExecutionStrategy::GPU => rsi_gpu(&device, data, 14),  │
│       }                                                           │
│   }                                                               │
└───────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                    AutoTuneProfile (Singleton)                    │
│                                                                   │
│   Lazy initialization with OnceLock:                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ PROFILE: OnceLock<AutoTuneProfile>                      │   │
│   │                                                          │   │
│   │ First call:  get_or_init() → calibrate() → cache        │   │
│   │ Subsequent:  get_or_init() → load cache → return        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│   Stored data:                                                   │
│   • hardware_id: String                                          │
│   • cpu_clock_ghz: f64                                           │
│   • gpu_clock_ghz: f64                                           │
│   • vram_bandwidth_gbs: f64                                      │
│   • ram_bandwidth_gbs: f64                                       │
│   • thresholds: IndicatorThresholds                              │
│   • calibration_timestamp: u64                                   │
└───────────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌─────────────────┐ ┌──────────────┐ ┌─────────────────┐
│ Hardware        │ │ Benchmarking │ │ Cache           │
│ Detection       │ │ Engine       │ │ Management      │
└─────────────────┘ └──────────────┘ └─────────────────┘
```

---

## Hardware Detection Pipeline

```text
┌──────────────────────────────────────────────────────────────────┐
│                    Hardware Detection                            │
└──────────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌─────────────────┐ ┌──────────────┐ ┌─────────────────┐
│ CPU Detection   │ │ GPU Detection│ │ RAM Detection   │
└─────────────────┘ └──────────────┘ └─────────────────┘
         │                 │                  │
         ▼                 ▼                  ▼
    /proc/cpuinfo     nvidia-smi        /proc/meminfo
         │                 │                  │
         ▼                 ▼                  ▼
   "cpu MHz: 5600"   "clocks.max: 3110"  "MemTotal: 64GB"
         │                 │                  │
         ▼                 ▼                  ▼
    5.6 GHz           3.11 GHz            77 GB/s
    (boost)           (boost)             (DDR5-4800)

┌──────────────────────────────────────────────────────────────────┐
│                    VRAM Bandwidth                                │
│                                                                  │
│   RTX 3500 Ada Specs:                                            │
│   • Memory: GDDR6                                                │
│   • Bus Width: 192-bit                                           │
│   • Memory Clock: 12 Gbps                                        │
│   • Bandwidth = (192/8) * 12000 * 2 = 288 GB/s                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Calibration Flow

```text
┌──────────────────────────────────────────────────────────────────┐
│                    CALIBRATION PROCESS                           │
│                     (First run: 2-5 sec)                         │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 1: Detect Hardware                                         │
│                                                                  │
│   detect_cpu_clock()     → 5.60 GHz                             │
│   detect_gpu_clock()     → 3.11 GHz                             │
│   detect_vram_bandwidth()→ 288 GB/s                             │
│   detect_ram_bandwidth() → 77 GB/s                              │
│   generate_hardware_id() → "cpu:Intel...gpu:RTX...ram:64gb"     │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 2: Benchmark Each Indicator                                │
│                                                                  │
│ For each indicator (Stochastic, ROC, Williams, Bollinger, MACD):│
│                                                                  │
│   For size in [100, 1K, 5K, 10K, 20K, 50K]:                     │
│       ┌─────────────────────────────────────────────┐           │
│       │ CPU Benchmark (10 iterations)              │           │
│       │   run 10x → [t1, t2, ..., t10]             │           │
│       │   median → cpu_time                        │           │
│       └─────────────────────────────────────────────┘           │
│                           │                                     │
│       ┌─────────────────────────────────────────────┐           │
│       │ GPU Benchmark (10 iterations)              │           │
│       │   run 10x → [t1, t2, ..., t10]             │           │
│       │   median → gpu_time                        │           │
│       └─────────────────────────────────────────────┘           │
│                           │                                     │
│                           ▼                                     │
│       ┌─────────────────────────────────────────────┐           │
│       │ Compare: gpu_time < cpu_time?              │           │
│       │   YES → Found crossover! Save size.        │           │
│       │   NO  → Continue to next size              │           │
│       └─────────────────────────────────────────────┘           │
│                                                                  │
│ Example result: Stochastic crossover = 5000 candles             │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 3: Cache Results                                           │
│                                                                  │
│   Serialize to JSON:                                            │
│   {                                                              │
│     "hardware_id": "cpu:Intel...gpu:RTX...ram:64gb",            │
│     "cpu_clock_ghz": 5.6,                                        │
│     "gpu_clock_ghz": 3.11,                                       │
│     "vram_bandwidth_gbs": 288.0,                                 │
│     "ram_bandwidth_gbs": 77.0,                                   │
│     "thresholds": {                                              │
│       "stochastic_crossover": 5000,                              │
│       "roc_crossover": 2000,                                     │
│       ...                                                        │
│     },                                                           │
│     "calibration_timestamp": 1729900800                          │
│   }                                                              │
│                                                                  │
│   Write to: ~/.cache/kimsfinance/autotune.json                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Strategy Selection Decision Tree

```text
┌──────────────────────────────────────────────────────────────────┐
│           User calls: profile.select_rsi_strategy(n)             │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Check indicator  │
                  │ category         │
                  └──────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Sequential   │ │ Hybrid       │ │ Parallel     │
    │ (EMA)        │ │ (RSI, ATR)   │ │ (Stochastic) │
    └──────────────┘ └──────────────┘ └──────────────┘
            │               │               │
            ▼               ▼               ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Always CPU   │ │ if n < 5K:   │ │ if n < thresh│
    │              │ │   CPU        │ │   CPU        │
    │ Reason:      │ │ else:        │ │ else:        │
    │ IIR filter   │ │   GPU hybrid │ │   GPU        │
    │ CPU 2-5x     │ │              │ │              │
    │ faster       │ │ Reason:      │ │ Reason:      │
    │              │ │ PCIe overhead│ │ Parallel ops │
    │              │ │ dominates    │ │ scale well   │
    └──────────────┘ └──────────────┘ └──────────────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
                ┌─────────────────────┐
                │ ExecutionStrategy   │
                │   - CPU             │
                │   - GPU             │
                │   - Hybrid          │
                └─────────────────────┘
```

---

## Cache Management Flow

```text
┌──────────────────────────────────────────────────────────────────┐
│                    First Application Run                         │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ AutoTuneProfile::        │
              │ get_or_init(&device)     │
              └──────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ Check cache file exists? │
              └──────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
          ┌─────────┐           ┌─────────────┐
          │ NO      │           │ YES         │
          └─────────┘           └─────────────┘
                │                       │
                ▼                       ▼
    ┌─────────────────────┐   ┌───────────────────┐
    │ calibrate()         │   │ load_from_cache() │
    │ (2-5 seconds)       │   │ (<1ms)            │
    └─────────────────────┘   └───────────────────┘
                │                       │
                │                       ▼
                │           ┌───────────────────────┐
                │           │ hardware_id matches?  │
                │           └───────────────────────┘
                │                       │
                │           ┌───────────┴────────────┐
                │           ▼                        ▼
                │     ┌─────────┐              ┌─────────┐
                │     │ YES     │              │ NO      │
                │     └─────────┘              └─────────┘
                │           │                        │
                │           ▼                        ▼
                │     ┌────────────┐       ┌─────────────────┐
                │     │ Use cached │       │ Hardware changed│
                │     │ profile    │       │ Re-calibrate    │
                │     └────────────┘       └─────────────────┘
                │           │                        │
                └───────────┴────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ Return &'static          │
              │ AutoTuneProfile          │
              └──────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                  Subsequent Application Runs                     │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ OnceLock already         │
              │ initialized?             │
              └──────────────────────────┘
                            │
                            ▼
                      ┌─────────┐
                      │ YES     │
                      └─────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ Return cached reference  │
              │ (<1ns, no disk I/O)      │
              └──────────────────────────┘
```

---

## Example: RSI Hybrid Pipeline

```text
┌──────────────────────────────────────────────────────────────────┐
│           RSI Calculation (100K candles, period=14)              │
│                                                                  │
│   profile.select_rsi_strategy(100_000) → ExecutionStrategy::GPU │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 1: GPU - Parallel Gains/Losses Calculation                 │
│                                                                  │
│   Input:  close[0..100K]                                         │
│   Kernel: calculate_gains_losses_kernel                          │
│   Threads: 100K - 1 (parallel)                                   │
│   Output: gains[0..100K], losses[0..100K]                        │
│   Time:   ~20μs                                                  │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 2: D2H Transfer (Device to Host)                           │
│                                                                  │
│   Transfer: gains[0..100K], losses[0..100K]                      │
│   Size:     2 × 100K × 8 bytes = 1.6 MB                          │
│   Time:     ~32μs                                                │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 3: CPU - Sequential Wilder's Smoothing (2x)                │
│                                                                  │
│   Input:  gains[0..100K]                                         │
│   Algo:   wilders_smoothing_cpu(gains, 14)                       │
│   Output: avg_gain[0..100K]                                      │
│   Time:   ~15μs                                                  │
│                                                                  │
│   Input:  losses[0..100K]                                        │
│   Algo:   wilders_smoothing_cpu(losses, 14)                      │
│   Output: avg_loss[0..100K]                                      │
│   Time:   ~15μs                                                  │
│                                                                  │
│   Reason CPU is faster:                                          │
│   • Sequential IIR filter (cannot parallelize)                   │
│   • CPU @ 5.6 GHz vs GPU thread @ 3.11 GHz                       │
│   • CPU is 3-4x faster for this step                             │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 4: H2D Transfer (Host to Device)                           │
│                                                                  │
│   Transfer: avg_gain[0..100K], avg_loss[0..100K]                 │
│   Size:     2 × 100K × 8 bytes = 1.6 MB                          │
│   Time:     ~32μs                                                │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 5: GPU - Parallel RSI Calculation                          │
│                                                                  │
│   Input:  avg_gain[0..100K], avg_loss[0..100K]                   │
│   Kernel: calculate_rsi_kernel                                   │
│   Threads: 100K (parallel)                                       │
│   Algo:   RSI[i] = 100 - (100 / (1 + avg_gain[i]/avg_loss[i]))  │
│   Output: rsi[0..100K]                                           │
│   Time:   ~15μs                                                  │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 6: D2H Transfer (Final Result)                             │
│                                                                  │
│   Transfer: rsi[0..100K]                                         │
│   Size:     100K × 8 bytes = 800 KB                              │
│   Time:     ~16μs                                                │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Total Time                                │
│                                                                  │
│   GPU gains/losses: ~20μs                                        │
│   D2H transfer:     ~32μs                                        │
│   CPU Wilder's (2x):~30μs                                        │
│   H2D transfer:     ~32μs                                        │
│   GPU RSI calc:     ~15μs                                        │
│   D2H final:        ~16μs                                        │
│   ────────────────────────                                       │
│   TOTAL:           ~145μs                                        │
│                                                                  │
│   vs CPU-only:     ~180μs                                        │
│   Speedup:         1.24x                                         │
│                                                                  │
│   vs Old GPU (single-thread Wilder's):                           │
│   Old GPU time:    ~250μs                                        │
│   Speedup:         1.72x                                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Environment Variable Override

```text
┌──────────────────────────────────────────────────────────────────┐
│                    KIMSFINANCE_FORCE_CPU=1                       │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ AutoTuneProfile::        │
              │ get_or_init(&device)     │
              └──────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ Check env var            │
              │ KIMSFINANCE_FORCE_CPU?   │
              └──────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
          ┌─────────┐           ┌─────────────┐
          │ == "1"  │           │ != "1"      │
          └─────────┘           └─────────────┘
                │                       │
                ▼                       ▼
    ┌─────────────────────┐   ┌───────────────────┐
    │ cpu_only_profile()  │   │ Normal calibration│
    │                     │   │ or load cache     │
    │ All thresholds:     │   └───────────────────┘
    │   usize::MAX        │
    │                     │
    │ Result: Always CPU  │
    └─────────────────────┘
```

---

## Data Structures

```text
┌──────────────────────────────────────────────────────────────────┐
│                      AutoTuneProfile                             │
├──────────────────────────────────────────────────────────────────┤
│ hardware_id: String                                              │
│   "cpu:Intel(R) Core(TM) i9-13980HX_gpu:NVIDIA RTX 3500 Ada..." │
│                                                                  │
│ cpu_clock_ghz: f64                                               │
│   5.6                                                            │
│                                                                  │
│ gpu_clock_ghz: f64                                               │
│   3.11                                                           │
│                                                                  │
│ vram_bandwidth_gbs: f64                                          │
│   288.0                                                          │
│                                                                  │
│ ram_bandwidth_gbs: f64                                           │
│   77.0                                                           │
│                                                                  │
│ thresholds: IndicatorThresholds ┐                               │
│   ┌─────────────────────────────┘                               │
│   │                                                              │
│   ├─ ema_crossover: usize::MAX (always CPU)                     │
│   ├─ wilders_crossover: usize::MAX (always CPU)                 │
│   ├─ stochastic_crossover: 5000                                 │
│   ├─ roc_crossover: 2000                                        │
│   ├─ williams_r_crossover: 5000                                 │
│   ├─ bollinger_crossover: 8000                                  │
│   ├─ macd_crossover: 15000                                      │
│   └─ parallel_operations: 1000                                  │
│                                                                  │
│ calibration_timestamp: u64                                       │
│   1729900800                                                     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    ExecutionStrategy (Enum)                      │
├──────────────────────────────────────────────────────────────────┤
│ CPU     → Run entirely on CPU                                    │
│ GPU     → Run entirely on GPU (or GPU-heavy hybrid like RSI)     │
│ Hybrid  → Custom hybrid strategy (future, currently falls back)  │
└──────────────────────────────────────────────────────────────────┘
```

---

## File System Layout

```text
~/.cache/kimsfinance/
└── autotune.json                   ← Cached calibration profile

/home/kim-asplund/projects/kimsfinance/rust/
├── src/
│   └── autotuner.rs                ← Core implementation (1,026 lines)
├── examples/
│   ├── autotuner_demo.rs           ← Interactive demo (188 lines)
│   └── calibrate.rs                ← Manual calibration CLI (128 lines)
├── docs/
│   ├── AUTOTUNER_GUIDE.md          ← Full integration guide (660 lines)
│   └── AUTOTUNER_ARCHITECTURE.md   ← This file
├── AUTOTUNER_IMPLEMENTATION_REPORT.md  ← Technical report (554 lines)
├── AUTOTUNER_QUICKREF.md           ← Quick reference (372 lines)
└── AUTOTUNER_SUMMARY.md            ← Executive summary (447 lines)
```

---

## Performance Comparison Matrix

```text
┌────────────────────────────────────────────────────────────────────┐
│         Hardware Configuration vs Strategy Performance            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ Scenario 1: RTX 3500 Ada + i9-13980HX (Your Hardware)             │
│ ─────────────────────────────────────────────────────────────────  │
│   CPU: 5.6 GHz boost                                               │
│   GPU: 3.11 GHz boost, 288 GB/s VRAM                               │
│   RAM: 77 GB/s (DDR5-4800)                                         │
│   VRAM/RAM ratio: 3.7x                                             │
│                                                                    │
│   Auto-Tuned Thresholds:                                           │
│   • EMA: Always CPU (sequential IIR)                               │
│   • RSI: GPU if n >= 5K (hybrid worth it)                          │
│   • Stochastic: GPU if n >= 5K (parallel speedup)                  │
│   • ROC: GPU if n >= 2K (simple parallel ops)                      │
│   • Bollinger: GPU if n >= 8K                                      │
│   • MACD: GPU if n >= 15K                                          │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ Scenario 2: RTX 4090 + Raspberry Pi 4                             │
│ ─────────────────────────────────────────────────────────────────  │
│   CPU: 1.5 GHz (ARM Cortex-A72)                                    │
│   GPU: 2.52 GHz boost, 1008 GB/s VRAM                              │
│   RAM: 8 GB/s (LPDDR4)                                             │
│   VRAM/RAM ratio: 126x (!!)                                        │
│                                                                    │
│   Auto-Tuned Thresholds (estimated):                               │
│   • EMA: CPU if n < 500 (sequential, but CPU very weak)            │
│   • RSI: GPU always (hybrid overhead worth it even at 100)         │
│   • Stochastic: GPU if n >= 100 (GPU dominates)                    │
│   • ROC: GPU if n >= 50                                            │
│   • Bollinger: GPU if n >= 100                                     │
│   • MACD: GPU if n >= 200                                          │
│                                                                    │
│   Key: Weak CPU + Strong GPU → GPU threshold much lower            │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ Scenario 3: Integrated GPU + i9-13980HX                           │
│ ─────────────────────────────────────────────────────────────────  │
│   CPU: 5.6 GHz boost (same as Scenario 1)                          │
│   GPU: 1.5 GHz, 68 GB/s VRAM (Intel Iris Xe)                       │
│   RAM: 77 GB/s (DDR5-4800)                                         │
│   VRAM/RAM ratio: 0.88x (VRAM slower!)                             │
│                                                                    │
│   Auto-Tuned Thresholds (estimated):                               │
│   • EMA: Always CPU (sequential)                                   │
│   • RSI: Always CPU (hybrid overhead not worth it)                 │
│   • Stochastic: GPU if n >= 50K (high threshold)                   │
│   • ROC: GPU if n >= 20K                                           │
│   • Bollinger: GPU if n >= 30K                                     │
│   • MACD: GPU if n >= 100K                                         │
│                                                                    │
│   Key: Weak GPU + Strong CPU → GPU threshold much higher           │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

**Version**: 0.2.0
**Last Updated**: 2025-10-25
**Status**: Production Ready ✅
