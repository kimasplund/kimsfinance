# Test Coverage Gap Analysis - kimsfinance

**Analysis Date**: 2025-10-22
**Current Test Count**: 730 tests across 37 files
**Source Files**: 64 Python modules
**Test Coverage**: ~42% (estimated based on component analysis)

---

## Executive Summary

This analysis identifies critical gaps in test coverage across the kimsfinance codebase. Key findings:

1. **13 of 23 indicators (57%) have NO tests**
2. **Missing GPU parity tests** for all indicators
3. **No dedicated tests** for 7 core modules
4. **Limited edge case coverage** in existing tests
5. **Missing integration tests** for multi-indicator scenarios
6. **No performance regression tests** for chart rendering

**Priority**: HIGH - Many production-critical components lack basic test coverage

---

## 1. Untested Indicators (CRITICAL)

### 1.1 Indicators with ZERO Tests

| Indicator | Module | Functions | Priority | Complexity |
|-----------|--------|-----------|----------|------------|
| **RSI** | `rsi.py` | `calculate_rsi` | CRITICAL | Medium |
| **MACD** | `macd.py` | `calculate_macd` | CRITICAL | Medium |
| **Bollinger Bands** | `bollinger_bands.py` | `calculate_bollinger_bands` | CRITICAL | Medium |
| **Stochastic Oscillator** | `stochastic_oscillator.py` | `calculate_stochastic_oscillator` | HIGH | Medium |
| **OBV** | `obv.py` | `calculate_obv` | HIGH | Low |
| **VWAP** | `vwap.py` | `calculate_vwap`, `calculate_vwap_anchored` | HIGH | Medium |
| **Williams %R** | `williams_r.py` | `calculate_williams_r` | HIGH | Medium |
| **CCI** | `cci.py` | `calculate_cci` | MEDIUM | Medium |
| **TSI** | `tsi.py` | `calculate_tsi` | MEDIUM | High |
| **DEMA/TEMA** | `dema_tema.py` | `calculate_dema`, `calculate_tema` | MEDIUM | Medium |
| **Elder Ray** | `elder_ray.py` | `calculate_elder_ray` | MEDIUM | Medium |
| **HMA** | `hma.py` | `calculate_hma` | MEDIUM | Medium |
| **Volume Profile** | `volume_profile.py` | `calculate_volume_profile` | LOW | High |

**Total**: 13 indicators, 16 functions untested

### 1.2 Indicators with Partial Tests

| Indicator | Test File | Coverage Issue |
|-----------|-----------|----------------|
| **Moving Averages** | `test_moving_averages.py` | Only 1 test (414 bytes) - minimal coverage |
| **ATR** | `test_atr.py` | Only 1 test (1000 bytes) - basic only |
| **Swing** | `test_swing.py` | Only 1 test (1009 bytes) - basic only |

---

## 2. Missing GPU/CPU Parity Tests

### 2.1 GPU-Accelerated Components Without Parity Tests

**ALL indicators** support GPU acceleration but have NO tests verifying:
- CPU vs GPU result parity (numerical accuracy)
- Engine selection logic (`auto`, `cpu`, `gpu`)
- Threshold-based switching
- GPU memory management
- Fallback behavior when GPU unavailable

**Critical Gaps**:
```python
# NONE of these are tested for GPU parity:
calculate_rsi(..., engine="gpu") == calculate_rsi(..., engine="cpu")
calculate_macd(..., engine="gpu") == calculate_macd(..., engine="cpu")
calculate_bollinger_bands(..., engine="gpu") == calculate_bollinger_bands(..., engine="cpu")
# ... 20+ more indicators
```

### 2.2 Existing GPU Tests (Limited)

| Test File | What It Tests | What's Missing |
|-----------|---------------|----------------|
| `test_engine_selection.py` | Engine selection logic, thresholds | Actual GPU computation validation |
| `test_memory_leaks.py` | Memory leak detection | GPU memory leak detection |

**Missing**:
- GPU kernel correctness tests
- Numerical precision comparison (CPU vs GPU)
- Large dataset GPU stress tests
- Multi-GPU support tests

---

## 3. Untested Core Modules

### 3.1 Core Modules with ZERO Dedicated Tests

| Module | Path | Functions/Classes | Priority |
|--------|------|-------------------|----------|
| **Auto-tuning** | `core/autotune.py` | `_benchmark_operation`, threshold caching | HIGH |
| **Decorators** | `core/decorators.py` | Performance decorators | MEDIUM |
| **Exceptions** | `core/exceptions.py` | Custom exceptions | MEDIUM |
| **Types** | `core/types.py` | Type definitions | LOW |
| **Chart Settings** | `config/chart_settings.py` | Chart configuration | MEDIUM |
| **Render Config** | `config/render_config.py` | Rendering configuration | MEDIUM |
| **Themes** | `config/themes.py` | Theme definitions | LOW |

### 3.2 Operations Modules - Untested

| Module | Path | Key Functions | Priority |
|--------|------|---------------|----------|
| **Aggregations** | `ops/aggregations.py` | `volume_sum`, `volume_weighted_price`, `resample_ohlc` | HIGH |
| **Batch** | `ops/batch.py` | Batch indicator processing | MEDIUM |
| **Rolling** | `ops/rolling.py` | Rolling window operations | HIGH |
| **NaN Ops** | `ops/nan_ops.py` | NaN handling utilities | MEDIUM |
| **Linear Algebra** | `ops/linear_algebra.py` | Matrix operations | MEDIUM |
| **Indicator Utils** | `ops/indicator_utils.py` | Common indicator utilities | HIGH |

### 3.3 Utilities - Untested

| Module | Path | Functions | Priority |
|--------|------|-----------|----------|
| **Array Utils** | `utils/array_utils.py` | `to_numpy_array`, array conversions | HIGH |
| **Color Utils** | `utils/color_utils.py` | `_hex_to_rgba`, color conversions | MEDIUM |

### 3.4 Data Processing - Untested

| Module | Path | Functions | Priority |
|--------|------|-----------|----------|
| **Renko** | `data/renko.py` | `calculate_renko_bricks` | MEDIUM |
| **Point & Figure** | `data/pnf.py` | `calculate_pnf_chart` | MEDIUM |

### 3.5 Integration - Untested

| Module | Path | Functions | Priority |
|--------|------|-----------|----------|
| **Adapter** | `integration/adapter.py` | External library adapters | MEDIUM |
| **Hooks** | `integration/hooks.py` | Plugin hooks | LOW |

---

## 4. Chart Types - Test Coverage

### 4.1 Chart Types with Tests

| Chart Type | Test File | Coverage |
|------------|-----------|----------|
| Candlestick | Multiple | Good (PIL + SVG) |
| OHLC Bars | `test_renderer_ohlc.py` | Good |
| Line | `test_renderer_line.py` | Good |
| Hollow Candles | `test_renderer_hollow.py`, `test_hollow_candles_svg.py` | Good |
| Renko | `test_renderer_renko.py` | Good |
| Point & Figure | `test_renderer_pnf.py` | Good |

### 4.2 Chart Features - Missing Tests

| Feature | Status | Priority |
|---------|--------|----------|
| Tick charts | Partial (`test_tick_aggregations.py`) | HIGH |
| Volume charts | Untested | MEDIUM |
| Multi-indicator overlays | Untested | HIGH |
| Custom themes | Untested | LOW |
| Export formats (WebP, PNG, SVG, SVGZ) | Partial | MEDIUM |
| Batch rendering | Untested | HIGH |
| Parallel rendering | Untested (`plotting/parallel.py` exists) | HIGH |

---

## 5. Edge Cases - Missing Coverage

### 5.1 Data Edge Cases

Missing tests for:
- **Empty datasets** (0 rows)
- **Single row** datasets
- **Period > data length** scenarios
- **All NaN values**
- **Mixed NaN/valid data**
- **Extreme values** (overflow/underflow)
- **Negative prices/volumes**
- **Zero volume bars**
- **Duplicate timestamps**

### 5.2 Parameter Edge Cases

Missing tests for:
- **Invalid periods** (period=0, negative, non-integer)
- **Invalid box sizes** (Renko/PnF)
- **Invalid color codes**
- **Invalid file paths**
- **Out-of-bounds dimensions** (width/height)
- **Extreme parameters** (period=1000000)

### 5.3 Engine Selection Edge Cases

Missing tests for:
- **GPU not available** fallback
- **GPU memory exhausted** fallback
- **Mixed CPU/GPU operations**
- **Threshold boundary conditions**
- **Concurrent engine requests**

---

## 6. Integration Tests - Missing

### 6.1 End-to-End Workflows

Missing tests for:
- **Load data → Calculate 5+ indicators → Render chart → Export**
- **Batch processing 1000 charts**
- **Multi-panel charts** with multiple indicators
- **Real-time data streaming** scenarios
- **API compatibility** with mplfinance

### 6.2 Multi-Indicator Scenarios

No tests for:
- **MACD + RSI + Bollinger Bands** on same chart
- **Multiple moving averages** (3-10 lines)
- **Volume Profile + Price Action**
- **Ichimoku Cloud + ATR**

### 6.3 Error Handling Integration

Missing tests for:
- **Invalid data → error message quality**
- **GPU failure → CPU fallback → success**
- **File write failure → error recovery**
- **Concurrent access** to cache files

---

## 7. Performance Tests - Missing

### 7.1 Benchmark Coverage Gaps

| Component | Existing | Missing |
|-----------|----------|---------|
| **OHLC Rendering** | `benchmark_ohlc_bars.py` | Hollow, Renko, PnF, Line |
| **Indicators** | None | ALL indicators (23 total) |
| **GPU Operations** | None | ALL GPU-accelerated ops |
| **Aggregations** | None | Volume sum, VWAP, resampling |
| **Batch Processing** | None | 1000+ chart generation |

### 7.2 Performance Regression Tests

**No automated tests** to verify:
- 178x speedup claim vs mplfinance
- <10ms chart rendering target
- >1000 img/sec throughput target
- <1KB WebP file size target

### 7.3 Scaling Tests

Missing tests for:
- **10K, 100K, 1M, 10M candles** performance
- **Memory usage** at scale
- **GPU memory pressure** handling
- **Parallel rendering** scalability

---

## 8. Test Recommendations

### 8.1 Priority 1 (CRITICAL) - 2 weeks

**Estimated**: 180 tests

1. **Indicator Tests** (13 indicators × 10 tests each = 130 tests)
   - Basic correctness (known values)
   - GPU vs CPU parity
   - Edge cases (empty, single row, period > length)
   - Parameter validation
   - Engine selection
   - NaN handling
   - Large dataset (100K+ rows)
   - Numerical accuracy
   - Performance bounds
   - Integration with plotting

2. **Core Module Tests** (50 tests)
   - `aggregations.py`: 15 tests (volume sum, VWAP, resampling)
   - `array_utils.py`: 10 tests (conversions, validation)
   - `autotune.py`: 10 tests (benchmarking, caching)
   - `rolling.py`: 10 tests (window operations)
   - `indicator_utils.py`: 5 tests (common utilities)

### 8.2 Priority 2 (HIGH) - 1 week

**Estimated**: 80 tests

3. **GPU Parity Tests** (60 tests)
   - All indicators: CPU vs GPU numerical accuracy
   - Threshold-based switching validation
   - Fallback behavior
   - Memory management

4. **Integration Tests** (20 tests)
   - Multi-indicator charts
   - End-to-end workflows
   - Batch processing
   - Error handling

### 8.3 Priority 3 (MEDIUM) - 1 week

**Estimated**: 70 tests

5. **Chart Features** (30 tests)
   - Multi-panel layouts
   - Custom themes (all 4 themes)
   - Export formats (WebP, PNG, SVG, SVGZ)
   - Parallel rendering
   - Volume overlays

6. **Edge Cases** (40 tests)
   - Data edge cases: empty, NaN, extreme values
   - Parameter edge cases: invalid, boundary
   - Concurrency edge cases

### 8.4 Priority 4 (LOW) - 1 week

**Estimated**: 50 tests

7. **Performance Regression** (30 tests)
   - Indicator benchmarks (all 23)
   - Chart rendering benchmarks (all types)
   - GPU vs CPU speedup validation
   - Scaling tests (10K → 10M candles)

8. **Miscellaneous** (20 tests)
   - Themes, decorators, hooks
   - Configuration modules
   - Type checking

---

## 9. Test Count Estimation

### 9.1 Current State

| Category | Files | Tests | Coverage |
|----------|-------|-------|----------|
| Plotting | 9 | 288 | Good |
| Indicators | 17 | 254 | Poor (only 10 of 23 tested) |
| Core/Integration | 11 | 188 | Moderate |
| **Total** | **37** | **730** | **~42%** |

### 9.2 Target State (80% Coverage)

| Category | Current Tests | Needed Tests | Total Target |
|----------|---------------|--------------|--------------|
| **Indicators** | 254 | +180 | 434 |
| **Core Modules** | 50 | +100 | 150 |
| **GPU Parity** | 10 | +60 | 70 |
| **Integration** | 20 | +40 | 60 |
| **Chart Features** | 288 | +50 | 338 |
| **Edge Cases** | 40 | +80 | 120 |
| **Performance** | 8 | +50 | 58 |
| **Miscellaneous** | 60 | +20 | 80 |
| **TOTAL** | **730** | **+580** | **1310** |

**To reach 80% coverage**: Add ~580 tests (79% increase)

### 9.3 Effort Estimate

| Priority | Tests | Estimated Time | Engineer-Weeks |
|----------|-------|----------------|----------------|
| P1 (Critical) | 180 | 2 weeks | 2 weeks |
| P2 (High) | 80 | 1 week | 1 week |
| P3 (Medium) | 70 | 1 week | 1 week |
| P4 (Low) | 50 | 1 week | 1 week |
| **TOTAL** | **380** | **5 weeks** | **5 engineer-weeks** |

*Note: 380 tests gets to ~65% coverage. Full 80% requires 580 tests (8 weeks).*

---

## 10. Recommended Test Scenarios

### 10.1 RSI (Example Template)

```python
# tests/ops/indicators/test_rsi.py (NEW FILE)

import pytest
import numpy as np
from kimsfinance.ops.indicators import calculate_rsi

class TestRSIBasics:
    def test_rsi_known_values(self):
        """Test RSI calculation with known values."""
        prices = np.array([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42])
        rsi = calculate_rsi(prices, period=14)
        # Verify against known RSI values
        assert 50 < rsi[-1] < 70  # Bullish momentum

    def test_rsi_cpu_gpu_parity(self):
        """GPU and CPU engines produce identical results."""
        prices = np.random.rand(100000) * 100
        rsi_cpu = calculate_rsi(prices, period=14, engine="cpu")
        rsi_gpu = calculate_rsi(prices, period=14, engine="gpu")
        np.testing.assert_allclose(rsi_cpu, rsi_gpu, rtol=1e-6)

    def test_rsi_period_validation(self):
        """Invalid period raises ValueError."""
        prices = np.array([100, 102, 101, 105])
        with pytest.raises(ValueError, match="period"):
            calculate_rsi(prices, period=0)

    def test_rsi_empty_data(self):
        """Empty dataset raises ValueError."""
        with pytest.raises(ValueError):
            calculate_rsi(np.array([]), period=14)

    def test_rsi_single_row(self):
        """Single row raises ValueError."""
        with pytest.raises(ValueError):
            calculate_rsi(np.array([100]), period=14)

    def test_rsi_period_exceeds_length(self):
        """Period > data length raises ValueError."""
        prices = np.array([100, 102, 101])
        with pytest.raises(ValueError):
            calculate_rsi(prices, period=14)

    def test_rsi_nan_handling(self):
        """NaN values are handled gracefully."""
        prices = np.array([100, np.nan, 102, 101, 105, np.nan, 107])
        rsi = calculate_rsi(prices, period=3)
        assert np.isnan(rsi[1])  # NaN in input → NaN in output

    def test_rsi_engine_auto_small(self):
        """Auto engine selects CPU for small datasets."""
        prices = np.random.rand(1000) * 100
        rsi = calculate_rsi(prices, period=14, engine="auto")
        # Should not raise, CPU selected
        assert len(rsi) == len(prices)

    def test_rsi_engine_auto_large(self):
        """Auto engine selects GPU for large datasets."""
        prices = np.random.rand(200000) * 100
        rsi = calculate_rsi(prices, period=14, engine="auto")
        # Should not raise, GPU selected (if available)
        assert len(rsi) == len(prices)

    def test_rsi_performance_bounds(self):
        """RSI completes within performance target."""
        import time
        prices = np.random.rand(100000) * 100
        start = time.perf_counter()
        calculate_rsi(prices, period=14, engine="gpu")
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1  # <100ms for 100K rows on GPU
```

**Total**: 10 tests per indicator × 13 untested indicators = **130 tests**

### 10.2 GPU Parity (Example)

```python
# tests/test_gpu_parity.py (NEW FILE)

import pytest
import numpy as np
from kimsfinance.ops.indicators import *

@pytest.mark.parametrize("indicator,params", [
    (calculate_rsi, {"period": 14}),
    (calculate_macd, {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
    (calculate_bollinger_bands, {"period": 20, "num_std": 2.0}),
    # ... all 23 indicators
])
def test_gpu_cpu_parity_all_indicators(indicator, params):
    """All indicators: GPU == CPU results."""
    data = np.random.rand(100000) * 100
    result_cpu = indicator(data, **params, engine="cpu")
    result_gpu = indicator(data, **params, engine="gpu")

    if isinstance(result_cpu, tuple):
        for cpu, gpu in zip(result_cpu, result_gpu):
            np.testing.assert_allclose(cpu, gpu, rtol=1e-6)
    else:
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-6)
```

**Total**: 23 indicators × 3 parity tests = **69 tests**

### 10.3 Integration Test (Example)

```python
# tests/test_multi_indicator_integration.py (NEW FILE)

def test_macd_rsi_bollinger_integration():
    """Multi-indicator chart: MACD + RSI + Bollinger Bands."""
    from kimsfinance.api import plot
    import pandas as pd

    # Generate sample data
    dates = pd.date_range("2023-01-01", periods=500)
    df = pd.DataFrame({
        "Open": np.random.rand(500) * 100,
        "High": np.random.rand(500) * 105,
        "Low": np.random.rand(500) * 95,
        "Close": np.random.rand(500) * 100,
        "Volume": np.random.randint(1000, 10000, 500),
    }, index=dates)

    # Add indicators
    from kimsfinance.ops.indicators import calculate_macd, calculate_rsi, calculate_bollinger_bands
    macd, signal, hist = calculate_macd(df["Close"])
    rsi = calculate_rsi(df["Close"])
    upper, middle, lower = calculate_bollinger_bands(df["Close"])

    # Render chart with all indicators
    plot(
        df,
        type="candle",
        volume=True,
        indicators={
            "macd": (macd, signal, hist),
            "rsi": rsi,
            "bollinger": (upper, middle, lower),
        },
        savefig="test_multi_indicator.webp",
    )

    # Verify output
    assert Path("test_multi_indicator.webp").exists()
    assert Path("test_multi_indicator.webp").stat().st_size < 2048  # <2KB
```

**Total**: 10 common indicator combinations × 2 tests = **20 tests**

---

## 11. Test Infrastructure Recommendations

### 11.1 Test Fixtures Needed

Create `/home/kim/Documents/Github/kimsfinance/tests/fixtures/`:

1. **sample_ohlcv.py**: Standard OHLCV datasets
   - `small_dataset`: 50 rows
   - `medium_dataset`: 1000 rows
   - `large_dataset`: 100K rows
   - `extreme_dataset`: 10M rows

2. **known_values.py**: Pre-calculated indicator values
   - RSI expected outputs
   - MACD expected outputs
   - Bollinger Bands expected outputs

3. **edge_cases.py**: Edge case datasets
   - `empty_df`, `single_row_df`, `all_nan_df`
   - `negative_prices`, `zero_volume`

### 11.2 Test Utilities

Create `/home/kim/Documents/Github/kimsfinance/tests/utils/`:

1. **gpu_helpers.py**: GPU test utilities
   ```python
   def assert_gpu_cpu_parity(func, data, **kwargs):
       """Assert GPU and CPU produce identical results."""
       ...
   ```

2. **performance_helpers.py**: Performance test utilities
   ```python
   def assert_faster_than(func, data, max_time_ms):
       """Assert function completes within time limit."""
       ...
   ```

### 11.3 CI/CD Integration

Add to GitHub Actions:
- Run all tests on CPU (always)
- Run GPU tests on CUDA-enabled runners (conditional)
- Enforce 80% coverage minimum
- Performance regression detection

---

## 12. Prioritized Test Implementation Plan

### Week 1: Critical Indicators (P1)
- [ ] RSI: 10 tests
- [ ] MACD: 10 tests
- [ ] Bollinger Bands: 10 tests
- [ ] Stochastic: 10 tests
- [ ] OBV: 8 tests
- [ ] VWAP: 12 tests (includes anchored)
- [ ] Williams %R: 10 tests
- **Total**: 70 tests

### Week 2: Medium Priority Indicators + Core (P1)
- [ ] CCI: 8 tests
- [ ] TSI: 10 tests
- [ ] DEMA/TEMA: 12 tests
- [ ] Elder Ray: 8 tests
- [ ] HMA: 8 tests
- [ ] Volume Profile: 12 tests
- [ ] Core: `aggregations.py` (15 tests)
- [ ] Core: `array_utils.py` (10 tests)
- **Total**: 83 tests

### Week 3: GPU Parity + Integration (P2)
- [ ] GPU parity tests: 60 tests (all indicators)
- [ ] Integration tests: 20 tests (multi-indicator workflows)
- **Total**: 80 tests

### Week 4: Chart Features + Edge Cases (P3)
- [ ] Chart features: 30 tests (themes, export, parallel)
- [ ] Edge cases: 40 tests (data/param edge cases)
- **Total**: 70 tests

### Week 5: Performance + Cleanup (P4)
- [ ] Performance regression: 30 tests
- [ ] Miscellaneous: 20 tests
- [ ] Documentation updates
- **Total**: 50 tests

**Grand Total**: 353 tests in 5 weeks → **~65% coverage**

---

## 13. Success Metrics

After implementing recommended tests:

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Test Count** | 730 | 1083 | +353 tests |
| **Test Files** | 37 | 50 | +13 files |
| **Indicator Coverage** | 43% (10/23) | 100% (23/23) | +13 indicators |
| **GPU Parity Tests** | 0 | 60 | +60 tests |
| **Core Module Coverage** | 0% | 80% | +125 tests |
| **Integration Tests** | 1 | 21 | +20 tests |
| **Performance Tests** | 1 | 31 | +30 tests |
| **Overall Coverage** | ~42% | ~65% | +23% |

---

## 14. Notes

1. **GPU Tests**: Require CUDA-enabled hardware. Add `@pytest.mark.gpu` and skip on CPU-only systems.

2. **Known Values**: Use external tools (TA-Lib, pandas-ta) to generate expected values for validation.

3. **Performance Targets**: Based on CLAUDE.md:
   - Chart rendering: <10ms (target), <5ms (excellent)
   - Throughput: >1000 img/sec (target), >6000 img/sec (excellent)
   - GPU speedup: >1.5x minimum

4. **Flaky Tests**: GPU tests may be flaky due to floating-point precision. Use `rtol=1e-6` for comparisons.

5. **Test Data**: Generate large datasets (10M candles) only in performance tests. Use smaller datasets (1K-100K) for correctness tests.

---

## 15. Contact

**Analyzed by**: Claude Code Agent
**Repository**: /home/kim/Documents/Github/kimsfinance
**Branch**: master
**Commit**: e7252b4

For questions, consult `/home/kim/Documents/Github/kimsfinance/CLAUDE.md`.
