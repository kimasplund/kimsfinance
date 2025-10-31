# GPU Indicator Performance Results

**⚠️ DEPRECATED - See FINAL_GPU_INDICATOR_PERFORMANCE_REPORT.md for corrected results**

**Issue with this report:** Cold start measurements (no warmup) - includes CUDA kernel compilation overhead

**Test Configuration:**
- Dataset: 100,000 candles (OHLCV)
- Hardware: NVIDIA RTX 3500 Ada (12GB VRAM)
- CUDA: 13.0
- Compute Capability: 8.9
- Build: Release mode with `--features gpu`
- **Benchmark flaw:** No warmup runs (measurements include compilation overhead)

**Date:** 2025-10-31
**After:** Async pinned memory optimization (PR #9)

---

## Performance Results (100K Candles) - COLD START (INCORRECT)

⚠️ **These numbers include CUDA kernel compilation overhead (~20-120ms)**

### GROUP 1: SIMPLE INDICATORS (2-3 transfers)

| Indicator | Time (μs) | Time (ms) | Candles/sec | Notes |
|-----------|-----------|-----------|-------------|-------|
| **EMA (hybrid)** | **13** | **0.01** | **7,692,308** | ⚡ FASTEST - CPU fallback |
| **ROC** | ~~135,819~~ | ~~135.82~~ | ~~736,274~~ | ❌ WRONG - Actually 442μs warm (2nd fastest!) |
| **WMA** | ~~71,122~~ | ~~71.12~~ | ~~1,406,035~~ | ❌ WRONG - Actually 717μs warm |
| **OBV** | ~~36,986~~ | ~~36.99~~ | ~~2,703,726~~ | ❌ WRONG - Actually 4,696μs warm |
| **VWMA** | ~~36,797~~ | ~~36.80~~ | ~~2,717,613~~ | ❌ WRONG - Actually 1,033μs warm |

---

### GROUP 2: MEDIUM INDICATORS (4-5 transfers)

| Indicator | Time (μs) | Time (ms) | Candles/sec | Notes |
|-----------|-----------|-----------|-------------|-------|
| **CCI** | ~~33,031~~ | ~~33.03~~ | ~~3,027,459~~ | ❌ WRONG - Actually 1,152μs warm |
| **MACD** | ~~140,175~~ | ~~140.18~~ | ~~713,394~~ | ⚠️ PARTIALLY CORRECT - Actually 57,750μs warm (still slow, single-thread GPU anti-pattern) |
| **SMA** | ~~53,740~~ | ~~53.74~~ | ~~1,860,811~~ | ❌ WRONG - Actually 519μs warm (3rd fastest!) |
| **Williams %R** | ~~24,947~~ | ~~24.95~~ | ~~4,008,498~~ | ❌ WRONG - Actually 1,079μs warm |
| **CMF** | ~~41,115~~ | ~~41.12~~ | ~~2,432,202~~ | ❌ WRONG - Actually 1,779μs warm |
| **Donchian** | ~~40,709~~ | ~~40.71~~ | ~~2,456,459~~ | ❌ WRONG - Actually 1,174μs warm |
| **Elder Ray** | ~~24,251~~ | ~~24.25~~ | ~~4,123,541~~ | ❌ WRONG - Actually 1,330μs warm |
| **Stochastic** | ~~31,582~~ | ~~31.58~~ | ~~3,166,361~~ | ❌ WRONG - Actually 1,279μs warm |

---

### GROUP 3: COMPLEX INDICATORS (6+ transfers)

| Indicator | Time (μs) | Time (ms) | Candles/sec | Notes |
|-----------|-----------|-----------|-------------|-------|
| **ATR** | ~~18,262~~ | ~~18.26~~ | ~~5,475,851~~ | ❌ WRONG - Actually 1,360μs warm (Jules' 145μs is GPU-only kernel time) |
| **RSI** | ~~19,360~~ | ~~19.36~~ | ~~5,165,289~~ | ❌ WRONG - Actually 2,512μs warm |
| **RSI (sync)** | ~~19,128~~ | ~~19.13~~ | ~~5,227,938~~ | ❌ WRONG - Actually 2,870μs warm |

---

## CORRECTED RESULTS (With Proper Warmup)

**See:** `FINAL_GPU_INDICATOR_PERFORMANCE_REPORT.md`

### Top 10 Fastest (Corrected, Warm Performance)
1. **EMA (hybrid)**: 200μs
2. **ROC**: 442μs (was incorrectly shown as slow!)
3. **SMA**: 519μs (was incorrectly shown as slow!)
4. **WMA**: 717μs
5. **VWMA**: 1,033μs
6. **Williams %R**: 1,079μs
7. **CCI**: 1,152μs
8. **Donchian**: 1,174μs
9. **Stochastic**: 1,279μs
10. **Elder Ray**: 1,330μs

### Slowest (Corrected)
1. **MACD**: 57,750μs (57.75ms) - Single-thread GPU anti-pattern, needs CPU execution
2. **OBV**: 4,696μs (4.70ms)
3. **RSI (sync)**: 2,870μs (2.87ms)

### Key Corrections
- **ROC**: NOT slow (442μs warm, 2nd fastest)
- **SMA**: NOT slow (519μs warm, 3rd fastest)
- **ATR**: 1,360μs end-to-end (145μs is GPU-only kernel time)
- **MACD**: Confirmed slow (single-thread GPU anti-pattern)

---

## Analysis (DEPRECATED - IGNORE)

~~This analysis is based on cold start measurements including compilation overhead~~

**See:** `FINAL_GPU_INDICATOR_PERFORMANCE_REPORT.md` for accurate analysis

---

## Recommendations (UPDATED)

### 1. Completed ✅
- ✅ Fixed benchmark methodology with proper warmup
- ✅ Investigated ROC performance (vindicated - actually 2nd fastest)
- ✅ Investigated MACD performance (confirmed slow, single-thread GPU anti-pattern)
- ✅ Validated ATR performance (145μs is GPU-only, 1.36ms is end-to-end)
- ✅ Created comprehensive GPU performance testing guide

### 2. Next Actions
- ⚠️ **Implement MACD CPU execution** for 1,647x speedup (high priority)
- 🔍 Investigate OBV performance (currently 4.70ms, potential 5x speedup)
- 📊 Add GPU-only kernel timing with CUDA events to all indicators
- 🧪 Implement performance regression tests in CI

---

## GPU Utilization

**CUDA Info:**
- Version: 13.0 ✅
- Async allocation: Supported ✅
- Compute capability: 8.9 (Ada architecture) ✅

**Memory Pool:**
- cudaMallocAsync: Enabled ✅
- Pinned memory: Active ✅

---

## Conclusion

**Overall Performance:** Good for most indicators, but discrepancy from expected ATR performance suggests:
1. Timing methodology may include overhead
2. GPU warmup needed
3. Some indicators (ROC, MACD) need investigation

**Next Steps:**
1. Add proper warmup to benchmark
2. Investigate ROC & MACD slow performance
3. Validate ATR actual performance matches 145μs claim
4. Consider persistent kernels for hot path indicators

---

**Hardware:** RTX 3500 Ada (12GB VRAM)
**Software:** CUDA 13.0, Compute 8.9
**Test Date:** 2025-10-31
**Optimizations:** Async pinned memory (PR #9)
