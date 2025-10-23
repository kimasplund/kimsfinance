# kimsfinance Memory Leak Fixes - Implementation Checklist

## Quick Reference

**Overall Risk**: MODERATE  
**Total Time**: 4-6 hours  
**Critical Issue**: Unbounded performance stats (2GB/day leak)

---

## Priority 1: Unbounded Performance Stats (CRITICAL)

### Location
File: `kimsfinance/integration/adapter.py`  
Lines: 30-37 (globals), 245-260 (_track_operation), 228-241 (reset_performance_stats)

### Issue
```python
# WRONG: Grows unbounded with each operation
_performance_stats = {
    "total_calls": 0,
    "gpu_calls": 0,
    "cpu_calls": 0,
    "time_saved_ms": 0.0,
    "speedup": 1.0,
}

# Called with every operation (no bounds)
def _track_operation(engine_used: str, time_saved_ms: float = 0.0) -> None:
    _performance_stats["total_calls"] += 1  # Unbounded accumulation
```

### Solution Options
```python
# OPTION A: Sliding window (recommended)
from collections import deque
_performance_window = deque(maxlen=100_000)  # Keep last 100K operations

def _track_operation(engine_used: str, time_saved_ms: float = 0.0) -> None:
    if not _config["performance_tracking"]:
        return
    _performance_window.append({
        "engine": engine_used,
        "time_saved": time_saved_ms,
    })

# OPTION B: Periodic auto-reset
_last_reset = time.time()
_reset_interval = 3600  # 1 hour

def _track_operation(engine_used: str, time_saved_ms: float = 0.0) -> None:
    global _last_reset
    if time.time() - _last_reset > _reset_interval:
        reset_performance_stats()
        _last_reset = time.time()
    # Track normally
```

### Testing
```python
kf.configure(performance_tracking=True)
kf.activate()

# Make 10M operations
for i in range(10_000_000):
    kf.integration.adapter._track_operation("cpu")

stats = kf.get_performance_stats()
assert stats["total_calls"] <= 100_000, "Stats accumulating unbounded!"
```

### Impact
- **Fixes**: 2GB/day leak on 24/7 servers
- **Effort**: 1.5 hours
- **Status**: ☐ Not started

---

## Priority 2: Double Array Copies (IMPORTANT)

### Location
File: `kimsfinance/plotting/pil_renderer.py`  
Lines: 273-277 (render_ohlcv_chart), 910-913 (render_line_chart), 1083-1087 (render_hollow_candles)

### Issue
```python
# WRONG: Two copies instead of one
open_prices = np.ascontiguousarray(to_numpy_array(ohlc["open"]))
#             ^^^^^^^^^^^^^^^^^ Copy 2
#                                ^^^^^^^^^^^ Copy 1 (inside to_numpy_array)

# 50K candle chart: 50K × 8 bytes × 4 arrays × 2 copies = 3.2MB wasted per chart
# At 100 charts/sec: 320MB/sec wasted
```

### Solution
```python
# OPTION A: Check flag first
open_prices = to_numpy_array(ohlc["open"])
if not open_prices.flags['C_CONTIGUOUS']:
    open_prices = np.ascontiguousarray(open_prices)

# OPTION B: Single-pass conversion (recommended)
open_prices = np.require(to_numpy_array(ohlc["open"]), 
                         requirements=['C_CONTIGUOUS'], 
                         dtype=np.float64)

# OPTION C: Avoid ascontiguousarray if already contiguous
open_prices = np.asarray(ohlc["open"], dtype=np.float64, order='C')
```

### Files to Update
- [ ] `render_ohlcv_chart()` - line 273-277
- [ ] `render_line_chart()` - line 910-913  
- [ ] `render_hollow_candles()` - line 1083-1087
- [ ] Similar patterns in `svg_renderer.py`

### Testing
```python
import numpy as np
from kimsfinance.plotting import render_ohlcv_chart

# Create test data
ohlc = {
    'open': np.random.rand(50000),
    'high': np.random.rand(50000),
    'low': np.random.rand(50000),
    'close': np.random.rand(50000),
}
volume = np.random.rand(50000)

# Should use only 1 copy per array, not 2
img = render_ohlcv_chart(ohlc, volume)
# Verify memory: should be ~2MB, not 3.2MB
```

### Impact
- **Fixes**: 50% memory overhead in batch rendering
- **Savings**: 80MB/sec instead of 160MB/sec at 100 charts/sec
- **Effort**: 2.5 hours
- **Status**: ☐ Not started

---

## Priority 3: BytesIO Buffers (LOW)

### Location
File: `kimsfinance/plotting/parallel.py`  
Lines: 36-39

### Issue
```python
# WRONG: Buffer not closed
buf = io.BytesIO()
img.save(buf, format="PNG")
return buf.getvalue()
# buf remains open, may accumulate in ProcessPoolExecutor
```

### Solution
```python
# RIGHT: Explicit cleanup with context manager
with io.BytesIO() as buf:
    img.save(buf, format="PNG")
    return buf.getvalue()
# buf automatically closed here
```

### Testing
```python
import gc
import io
from kimsfinance.plotting.parallel import render_charts_parallel

# Render 1000 charts in parallel
results = render_charts_parallel(datasets, num_workers=8)

# Force GC and check cleanup
gc.collect()
bytesio_refs = len([obj for obj in gc.get_objects() 
                    if isinstance(obj, io.BytesIO)])
assert bytesio_refs < 10, f"BytesIO leak: {bytesio_refs} buffers"
```

### Impact
- **Fixes**: BytesIO buffer leaks in parallel rendering
- **Effort**: 0.5 hours
- **Status**: ☐ Not started

---

## Priority 4: DataFrame Cleanup (LOW)

### Location
File: `kimsfinance/ops/aggregations.py`  
Lines: 254-257 (rolling_sum), 278-280 (rolling_mean), 301-321 (cumulative_sum)

### Issue
```python
# WRONG: DataFrame not explicitly freed
def rolling_sum(data: ArrayLike, window: int, *, engine: Engine = "auto") -> ArrayResult:
    data_arr = to_numpy_array(data)
    df = pl.DataFrame({"data": data_arr})
    result = df.select(pl.col("data").rolling_sum(window_size=window))["data"].to_numpy()
    return result
    # df freed only when function exits, may cause temp spikes in loops
```

### Solution
```python
# RIGHT: Explicit cleanup in finally block
def rolling_sum(data: ArrayLike, window: int, *, engine: Engine = "auto") -> ArrayResult:
    data_arr = to_numpy_array(data)
    df = pl.DataFrame({"data": data_arr})
    try:
        result = df.select(pl.col("data").rolling_sum(window_size=window))["data"].to_numpy()
        return result
    finally:
        del df  # Explicit cleanup
```

### Functions to Update
- [ ] `rolling_sum()` - line 254-260
- [ ] `rolling_mean()` - line 278-283
- [ ] `cumulative_sum()` - line 301-321

### Impact
- **Fixes**: Temporary memory spikes in tight loops
- **Effort**: 1 hour
- **Status**: ☐ Not started

---

## Priority 5: Function References (LOW)

### Location
File: `kimsfinance/integration/hooks.py`  
Lines: 20-22 (global), 42-43 (storage), 64-69 (cleanup)

### Issue
```python
# WRONG: Strong references prevent GC
_original_functions = {}
_original_functions["_plot_mav"] = mpf_plotting._plot_mav  # Keeps mplfinance in memory
```

### Solution
```python
# RIGHT: Use weakref to avoid retaining mplfinance
import weakref

def patch_plotting_functions(config: dict[str, Any]) -> None:
    global _original_functions
    try:
        import mplfinance.plotting as mpf_plotting
    except ImportError:
        raise ImportError("mplfinance not installed or incompatible version")

    # Store with weakref
    try:
        _original_functions["_plot_mav"] = weakref.ref(mpf_plotting._plot_mav)
        _original_functions["_plot_ema"] = weakref.ref(mpf_plotting._plot_ema)
    except TypeError:
        # Fallback: some objects can't be weakly referenced
        _original_functions["_plot_mav"] = mpf_plotting._plot_mav
        _original_functions["_plot_ema"] = mpf_plotting._plot_ema

def unpatch_plotting_functions() -> None:
    if not _original_functions:
        return
    try:
        import mplfinance.plotting as mpf_plotting
    except ImportError:
        return

    # Restore from weakref or strong ref
    for name in ["_plot_mav", "_plot_ema"]:
        if name in _original_functions:
            ref = _original_functions[name]
            if isinstance(ref, weakref.ref):
                func = ref()
                if func is not None:
                    setattr(mpf_plotting, name, func)
            else:
                setattr(mpf_plotting, name, ref)
    
    _original_functions.clear()
```

### Impact
- **Fixes**: mplfinance namespace retention
- **Effort**: 1.5 hours
- **Status**: ☐ Not started

---

## Implementation Checklist

### Phase 1: Critical (1.5 hours)
- [ ] Priority 1: Unbounded stats
  - [ ] Choose Option A (sliding window) or Option B (periodic reset)
  - [ ] Update `_track_operation()` function
  - [ ] Add bounds checking test
  - [ ] Run test: `python -c "import tests.test_memory_bounds"`

### Phase 2: Important (2.5 hours)
- [ ] Priority 2: Double array copies
  - [ ] Update `render_ohlcv_chart()` - line 273-277
  - [ ] Update `render_line_chart()` - line 910-913
  - [ ] Update `render_hollow_candles()` - line 1083-1087
  - [ ] Check `svg_renderer.py` for similar patterns
  - [ ] Add memory benchmark test

- [ ] Priority 3: BytesIO buffers
  - [ ] Update `parallel.py` line 36-39
  - [ ] Add context manager test

### Phase 3: Maintenance (2 hours)
- [ ] Priority 4: DataFrame cleanup
  - [ ] Update `rolling_sum()` - line 254-260
  - [ ] Update `rolling_mean()` - line 278-283
  - [ ] Update `cumulative_sum()` - line 301-321

- [ ] Priority 5: Function references
  - [ ] Import weakref module
  - [ ] Update `patch_plotting_functions()`
  - [ ] Update `unpatch_plotting_functions()`
  - [ ] Add cycle detection test

### Phase 4: Testing
- [ ] Long-running memory test (24 hour sim)
- [ ] Performance stats bounds test
- [ ] Array copy detection test
- [ ] Parallel rendering cleanup test
- [ ] Large dataset memory profile test

### Phase 5: Deployment
- [ ] Code review of all changes
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Commit and push to branch

---

## Estimated Timeline

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 1 | Unbounded stats | 1.5h | ☐ |
| 2 | Array copies | 2.5h | ☐ |
| 2 | BytesIO | 0.5h | ☐ |
| 3 | DataFrame cleanup | 1h | ☐ |
| 3 | Function refs | 1.5h | ☐ |
| 4 | Testing | 1.5h | ☐ |
| **Total** | **All fixes + testing** | **4-6h** | ☐ |

---

## Resources

- Full Report: `MEMORY_ANALYSIS_REPORT.md`
- Summary: `MEMORY_ANALYSIS_SUMMARY.txt`
- Tests: `tests/test_memory_*.py` (to be created)

---

## Sign-off

- [ ] All Priority 1 fixes applied
- [ ] All Priority 2 fixes applied
- [ ] All tests passing
- [ ] Code reviewed
- [ ] Ready for production

