# Master Implementation Plan: 20 Missing Indicators

**Date**: 2025-10-20
**Status**: In Progress
**Target**: Implement all 20 high-priority missing indicators with GPU acceleration

---

## Executive Summary

Implementing **20 missing technical indicators** identified through comprehensive research:
- **5 Tier 1** (Critical - Highest Demand)
- **9 Tier 2** (High Value)
- **6 Tier 3** (Professional Tools)

All indicators will follow kimsfinance patterns:
- CPU and GPU implementations
- Engine routing (`auto`, `cpu`, `gpu`)
- Comprehensive test coverage
- Type hints throughout
- Performance benchmarks

---

## Parallel Implementation Strategy

### 20 Parallel Tasks (Grouped by Complexity)

**Group 1: Moving Averages (3 tasks)**
- Task 1: EMA + SMA (foundational, related)
- Task 2: WMA (weighted moving average)
- Task 3: DEMA + TEMA (double/triple exponential)

**Group 2: Trend Indicators (5 tasks)**
- Task 4: ADX + DI components
- Task 5: Parabolic SAR
- Task 6: Supertrend
- Task 7: Aroon
- Task 8: Ichimoku Cloud (5 components)

**Group 3: Channels (3 tasks)**
- Task 9: Keltner Channels
- Task 10: Donchian Channels
- Task 11: Fibonacci Retracement

**Group 4: Momentum (3 tasks)**
- Task 12: MFI (Money Flow Index)
- Task 13: ROC (Rate of Change)
- Task 14: TSI (True Strength Index)

**Group 5: Volume (2 tasks)**
- Task 15: Volume Profile / VPVR (GPU-intensive)
- Task 16: Chaikin Money Flow (CMF)

**Group 6: Other (4 tasks)**
- Task 17: Pivot Points
- Task 18: Elder Ray (Bull/Bear Power)
- Task 19-20: Reserved for integration/cleanup

---

## Shared Architecture

### File Structure
```
kimsfinance/ops/indicators.py  # Add all new indicators here
tests/test_indicators.py        # Add tests for all new indicators
tests/fixtures/indicators/      # Sample data for indicator tests
demo_output/indicators/         # Sample charts
```

### Function Template

```python
def calculate_<indicator_name>(
    data: ArrayLike,  # Or multiple arrays for OHLCV
    period: int = <default>,
    *,
    engine: Engine = "auto"
) -> ArrayResult:
    """
    Calculate <Indicator Name>.

    Automatically uses GPU for datasets > 500,000 rows when engine="auto".

    Args:
        data: Input data (price, OHLC, etc.)
        period: Calculation period
        engine: Computation engine ('auto', 'cpu', 'gpu')

    Returns:
        Array of indicator values

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs

    # Route to CPU or GPU
    if engine == "auto":
        use_gpu = _should_use_gpu(data)
    else:
        use_gpu = (engine == "gpu")

    if use_gpu:
        return _calculate_<indicator>_gpu(data, period)
    else:
        return _calculate_<indicator>_cpu(data, period)
```

### Test Template

```python
class Test<IndicatorName>:
    """Test <indicator name> calculation."""

    def test_basic_calculation(self):
        """Test basic indicator calculation."""
        # Test with known values

    def test_gpu_cpu_match(self):
        """Test GPU and CPU implementations match."""
        # Compare outputs

    def test_edge_cases(self):
        """Test edge cases (NaN, zeros, insufficient data)."""

    def test_performance(self):
        """Benchmark performance."""
```

---

## GPU Optimization Priority

**Highest GPU Value**:
1. **Volume Profile** - Histogram computation (very intensive)
2. **EMA/SMA** - Massive parallelization potential
3. **Ichimoku** - 5 components computed simultaneously
4. **Multi-period indicators** - Parallel across periods

**Medium GPU Value**:
- ADX, Supertrend, Keltner, MFI, TSI

**Lower GPU Value** (but still implement GPU for consistency):
- Parabolic SAR, Aroon, Pivot Points, Fibonacci

---

## Implementation Requirements

### Each Indicator Must Have:

1. **CPU Implementation** (mandatory)
   - Pure Python + NumPy
   - Handles all edge cases
   - Well-commented algorithm

2. **GPU Implementation** (for consistency)
   - CuPy-based (if available)
   - Fallback to CPU if GPU unavailable
   - Must match CPU output exactly

3. **Tests** (minimum 4 tests per indicator)
   - Basic calculation test
   - GPU/CPU parity test
   - Edge case tests
   - Known values verification

4. **Type Hints**
   - Full type annotations
   - Proper ArrayLike/ArrayResult usage

5. **Documentation**
   - Docstring with algorithm description
   - Args/Returns/Raises documented
   - Example usage

---

## Indicator Specifications

### Tier 1: Critical Indicators

#### 1. EMA (Exponential Moving Average)
- **Formula**: EMA = Price(t) * k + EMA(y) * (1 - k), k = 2/(N+1)
- **Inputs**: prices (array), period (int)
- **Output**: EMA values (array)
- **GPU**: Parallel scan operation
- **Complexity**: Low

#### 2. SMA (Simple Moving Average)
- **Formula**: SMA = sum(prices[-N:]) / N
- **Inputs**: prices (array), period (int)
- **Output**: SMA values (array)
- **GPU**: Rolling window operation
- **Complexity**: Very Low

#### 3. ADX (Average Directional Index)
- **Formula**: Complex (requires +DI, -DI, TR, smoothing)
- **Inputs**: high, low, close (arrays), period (int, default=14)
- **Output**: ADX values (array), +DI (array), -DI (array)
- **GPU**: Parallel directional movement calculations
- **Complexity**: High

#### 4. Volume Profile / VPVR
- **Formula**: Histogram of volume at each price level
- **Inputs**: prices (array), volume (array), num_bins (int, default=50)
- **Output**: price_levels (array), volume_profile (array)
- **GPU**: Parallel histogram computation
- **Complexity**: High (perfect for GPU!)

#### 5. Fibonacci Retracement
- **Formula**: Levels at 0%, 23.6%, 38.2%, 50%, 61.8%, 100%
- **Inputs**: high_price (float), low_price (float)
- **Output**: fibonacci_levels (dict)
- **GPU**: Not applicable (simple calculation)
- **Complexity**: Very Low

### Tier 2: High Value Indicators

#### 6. Parabolic SAR
- **Formula**: Iterative acceleration factor calculation
- **Inputs**: high, low, close (arrays), af_start=0.02, af_increment=0.02, af_max=0.2
- **Output**: SAR values (array)
- **GPU**: Iterative (challenging to parallelize)
- **Complexity**: Medium

#### 7. Supertrend
- **Formula**: (High + Low) / 2 ± (Multiplier * ATR)
- **Inputs**: high, low, close (arrays), period=10, multiplier=3
- **Output**: supertrend (array), direction (array)
- **GPU**: Parallel (uses ATR)
- **Complexity**: Medium

#### 8. MFI (Money Flow Index)
- **Formula**: 100 - (100 / (1 + money_flow_ratio))
- **Inputs**: high, low, close, volume (arrays), period=14
- **Output**: MFI values (array)
- **GPU**: Parallel
- **Complexity**: Medium

#### 9. Keltner Channels
- **Formula**: EMA ± (Multiplier * ATR)
- **Inputs**: high, low, close (arrays), period=20, multiplier=2
- **Output**: upper (array), middle (array), lower (array)
- **GPU**: Parallel (uses EMA, ATR)
- **Complexity**: Low (depends on EMA, ATR)

#### 10. Ichimoku Cloud
- **Formula**: 5 components (Tenkan, Kijun, Senkou A, Senkou B, Chikou)
- **Inputs**: high, low, close (arrays)
- **Output**: 5 arrays
- **GPU**: Highly parallel (5 independent calculations)
- **Complexity**: Medium

#### 11. Pivot Points
- **Formula**: PP = (H + L + C) / 3, R1/S1, R2/S2, R3/S3
- **Inputs**: high, low, close (floats or arrays)
- **Output**: pivot_point, resistances (3), supports (3)
- **GPU**: Trivial parallelization
- **Complexity**: Very Low

#### 12. Aroon
- **Formula**: Aroon Up/Down = ((period - periods since high/low) / period) * 100
- **Inputs**: high, low (arrays), period=25
- **Output**: aroon_up (array), aroon_down (array)
- **GPU**: Parallel
- **Complexity**: Low

#### 13. Chaikin Money Flow (CMF)
- **Formula**: Sum(money_flow_volume) / Sum(volume) over N periods
- **Inputs**: high, low, close, volume (arrays), period=20
- **Output**: CMF values (array)
- **GPU**: Parallel
- **Complexity**: Medium

#### 14. ROC (Rate of Change)
- **Formula**: ((Price - Price[n periods ago]) / Price[n periods ago]) * 100
- **Inputs**: prices (array), period=12
- **Output**: ROC values (array)
- **GPU**: Trivial parallelization
- **Complexity**: Very Low

### Tier 3: Professional Tools

#### 15. WMA (Weighted Moving Average)
- **Formula**: Sum(price[i] * weight[i]) / Sum(weights)
- **Inputs**: prices (array), period (int)
- **Output**: WMA values (array)
- **GPU**: Parallel weighted sum
- **Complexity**: Low

#### 16. DEMA (Double Exponential MA)
- **Formula**: 2 * EMA - EMA(EMA)
- **Inputs**: prices (array), period (int)
- **Output**: DEMA values (array)
- **GPU**: Parallel (uses EMA twice)
- **Complexity**: Low (depends on EMA)

#### 17. TEMA (Triple Exponential MA)
- **Formula**: 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
- **Inputs**: prices (array), period (int)
- **Output**: TEMA values (array)
- **GPU**: Parallel (uses EMA three times)
- **Complexity**: Low (depends on EMA)

#### 18. Donchian Channels
- **Formula**: Highest high and lowest low over N periods
- **Inputs**: high, low (arrays), period=20
- **Output**: upper (array), middle (array), lower (array)
- **GPU**: Parallel rolling max/min
- **Complexity**: Very Low

#### 19. TSI (True Strength Index)
- **Formula**: Double-smoothed momentum oscillator
- **Inputs**: prices (array), long_period=25, short_period=13
- **Output**: TSI values (array)
- **GPU**: Parallel (double smoothing)
- **Complexity**: Medium

#### 20. Elder Ray (Bull/Bear Power)
- **Formula**: Bull Power = High - EMA, Bear Power = Low - EMA
- **Inputs**: high, low, close (arrays), period=13
- **Output**: bull_power (array), bear_power (array)
- **GPU**: Parallel (uses EMA)
- **Complexity**: Low (depends on EMA)

---

## Success Criteria

### Each Indicator Implementation Must:

- ✅ CPU implementation (NumPy)
- ✅ GPU implementation (CuPy) with fallback
- ✅ Engine routing (`auto`, `cpu`, `gpu`)
- ✅ Minimum 4 tests per indicator (80 total tests)
- ✅ GPU/CPU parity verified
- ✅ Type hints throughout
- ✅ Full docstrings
- ✅ Performance benchmarks
- ✅ Edge case handling

### Integration Requirements:

- ✅ All indicators exported in `kimsfinance/ops/__init__.py`
- ✅ All tests passing (80+ new tests)
- ✅ Sample charts generated
- ✅ Documentation updated
- ✅ No performance regressions

---

## Performance Targets

| Indicator Type | CPU Target | GPU Target | Speedup Goal |
|---------------|------------|------------|--------------|
| Simple (SMA, ROC, Pivot) | <5ms (1M rows) | <1ms | >5x |
| Medium (EMA, ADX, MFI) | <10ms (1M rows) | <2ms | >5x |
| Complex (Ichimoku, Volume Profile) | <50ms (1M rows) | <5ms | >10x |

---

## Dependencies Between Indicators

**Must implement first** (other indicators depend on these):
1. **SMA** - Used by: DEMA, TEMA
2. **EMA** - Used by: DEMA, TEMA, Keltner, Elder Ray, Supertrend
3. **ATR** - Already exists, used by: Keltner, Supertrend

**Can implement in parallel** (no dependencies):
- Volume Profile, Fibonacci, Parabolic SAR, Pivot Points, Aroon, ROC, WMA, Donchian, MFI, CMF, TSI, ADX

**Implement after EMA**:
- DEMA, TEMA, Keltner Channels, Elder Ray, Supertrend

**Implement after DEMA/TEMA** (if time permits):
- None (these are leaf nodes)

---

## Timeline Estimate

**Parallel implementation with 20 agents**: 2-3 hours total
- Simultaneous development: ~1 hour
- Integration and testing: ~1 hour
- Documentation and samples: ~30 minutes

**Sequential implementation**: 10-15 hours

---

## Next Steps

1. ✅ Create master plan (this document)
2. ⏳ Create shared architecture documentation
3. ⏳ Launch 20 parallel implementation tasks
4. ⏳ Integrate all implementations
5. ⏳ Run comprehensive test suite
6. ⏳ Generate sample charts
7. ⏳ Update documentation

---

**Status**: Planning Complete, Ready for Parallel Execution
**Agents**: 20 parallel-task-executor-v2 agents
**Expected Completion**: 2-3 hours
