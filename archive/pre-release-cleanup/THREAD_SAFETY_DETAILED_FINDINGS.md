# Thread Safety Issues - Detailed Findings & Code Locations

## Analysis Overview
- **Analyzed**: kimsfinance v0.1.0 codebase
- **Method**: Static analysis of global state, race conditions, and concurrency patterns
- **Scope**: Integration layer, engine management, file I/O, and multiprocessing
- **Total Issues Found**: 9 (4 CRITICAL, 3 HIGH, 2 MODERATE)

---

## Issue #1: CRITICAL - Unprotected _is_active Flag

### Location
- **File**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/integration/adapter.py`
- **Lines**: 21, 60, 83, 114, 126, 146

### Code
```python
# Line 21 - Module level
_is_active = False

# Line 60 - Function activate()
def activate(*, engine: str = "auto", strict: bool = False, verbose: bool = True) -> None:
    global _is_active
    
    if _is_active:  # LINE 62: Check without lock
        if verbose:
            print("⚠ kimsfinance already active")
        return
    
    # ... patch code ...
    
    _is_active = True  # LINE 83: Write without lock
```

### Race Condition
```
Thread A at line 62: reads _is_active = False
                     [CONTEXT SWITCH]
Thread B at line 62: reads _is_active = False
                     [CONTEXT SWITCH]
Thread A at line 83: writes _is_active = True, calls patch_plotting_functions()
                     [CONTEXT SWITCH]
Thread B at line 83: writes _is_active = True, calls patch_plotting_functions() AGAIN
                     
Result: mplfinance functions patched twice, inconsistent state
```

### Functions Affected
- `activate()` (line 60-83)
- `deactivate()` (line 114-126)
- `is_active()` (line 146)

### Severity
- **Probability**: HIGH
- **Impact**: HIGH (double-patching, state inconsistency)
- **Risk Level**: CRITICAL

---

## Issue #2: CRITICAL - Unprotected _config Dictionary

### Location
- **File**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/integration/adapter.py`
- **Lines**: 22-28, 68-70, 149-181

### Code
```python
# Lines 22-28 - Module level
_config = {
    "default_engine": "auto",
    "gpu_min_rows": 10_000,
    "strict_mode": False,
    "performance_tracking": False,
    "verbose": True,
}

# Lines 68-70 - activate() modifies _config
def activate(...):
    global _config
    _config["default_engine"] = engine  # LINE 68: Write without lock
    _config["strict_mode"] = strict     # LINE 69: Write without lock
    _config["verbose"] = verbose        # LINE 70: Write without lock

# Lines 149-181 - configure() modifies _config
def configure(**kwargs) -> None:
    global _config
    
    valid_keys = {...}
    
    for key, value in kwargs.items():
        if key not in valid_keys:
            raise ValueError(...)
        _config[key] = value  # LINE 181: Write without lock
```

### Race Condition (Multiple Updates)
```
Thread A: configure(default_engine="gpu", gpu_min_rows=50_000)
         Sets _config["default_engine"] = "gpu" at line 181
         [CONTEXT SWITCH before next iteration]

Thread B: Read _config["default_engine"] = "gpu" ✓ (correct)
         Read _config["gpu_min_rows"] = 10_000 ✗ (old value, should be 50_000)
         Inconsistent configuration!

Thread A: Sets _config["gpu_min_rows"] = 50_000 at line 181
         But Thread B already read old value
```

### Severity
- **Probability**: MEDIUM-HIGH
- **Impact**: MEDIUM (inconsistent config, wrong engine selection)
- **Risk Level**: CRITICAL

---

## Issue #3: CRITICAL - Unprotected _performance_stats Dictionary

### Location
- **File**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/integration/adapter.py`
- **Lines**: 31-37, 250-267

### Code
```python
# Lines 31-37 - Module level
_performance_stats = {
    "total_calls": 0,
    "gpu_calls": 0,
    "cpu_calls": 0,
    "time_saved_ms": 0.0,
    "speedup": 1.0,
}

# Lines 250-267 - _track_operation() modifies stats
def _track_operation(engine_used: str, time_saved_ms: float = 0.0) -> None:
    if not _config["performance_tracking"]:
        return

    global _performance_stats

    _performance_stats["total_calls"] += 1          # LINE 252: RMW not atomic
    
    if engine_used == "gpu":
        _performance_stats["gpu_calls"] += 1        # LINE 255: RMW not atomic
    else:
        _performance_stats["cpu_calls"] += 1        # LINE 257: RMW not atomic
    
    if time_saved_ms > 0:
        _performance_stats["time_saved_ms"] += time_saved_ms  # LINE 260: RMW not atomic
    
    # Compound operation without atomicity
    if _performance_stats["total_calls"] > 0:
        avg_speedup = 1.0 + (
            _performance_stats["time_saved_ms"] / (_performance_stats["total_calls"] * 10)
        )
        _performance_stats["speedup"] = avg_speedup  # LINE 267: RMW not atomic
```

### Race Condition (Lost Updates)
```
Thread A CPU instruction sequence:
  1. LOAD R1, [total_calls]     ; R1 = 100
  2. ADD R1, 1                  ; R1 = 101
  [CONTEXT SWITCH]

Thread B CPU instruction sequence:
  1. LOAD R1, [total_calls]     ; R1 = 100 (not yet updated by Thread A!)
  2. ADD R1, 1                  ; R1 = 101
  [CONTEXT SWITCH]

Thread A:
  3. STORE [total_calls], R1    ; [total_calls] = 101

Thread B:
  3. STORE [total_calls], R1    ; [total_calls] = 101 (overwrites Thread A's write!)

Expected: total_calls = 102
Actual: total_calls = 101 (lost one increment)
```

### Severity
- **Probability**: HIGH
- **Impact**: MEDIUM (corrupted statistics, misleading metrics)
- **Risk Level**: CRITICAL

---

## Issue #4: CRITICAL - Unprotected Hooks Global State

### Location
- **File**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/integration/hooks.py`
- **Lines**: 21-22, 32-33, 42-47, 64-69

### Code
```python
# Lines 21-22 - Module level
_original_functions = {}
_config = {}

# Lines 32-33 - patch_plotting_functions()
def patch_plotting_functions(config: dict[str, Any]) -> None:
    global _config
    _config = config  # LINE 33: Write without lock

    try:
        import mplfinance.plotting as mpf_plotting
        import mplfinance._utils as mpf_utils
    except ImportError:
        raise ImportError("mplfinance not installed or incompatible version")

    # Store original functions
    _original_functions["_plot_mav"] = mpf_plotting._plot_mav     # LINE 42
    _original_functions["_plot_ema"] = mpf_plotting._plot_ema     # LINE 43

    # Patch plotting functions
    mpf_plotting._plot_mav = _plot_mav_accelerated  # LINE 46: Non-atomic
    mpf_plotting._plot_ema = _plot_ema_accelerated  # LINE 47: Non-atomic

# Lines 64-69 - unpatch_plotting_functions()
def unpatch_plotting_functions() -> None:
    if not _original_functions:
        return

    try:
        import mplfinance.plotting as mpf_plotting
    except ImportError:
        return

    # Restore original functions
    if "_plot_mav" in _original_functions:
        mpf_plotting._plot_mav = _original_functions["_plot_mav"]  # LINE 65
    if "_plot_ema" in _original_functions:
        mpf_plotting._plot_ema = _original_functions["_plot_ema"]  # LINE 67

    _original_functions.clear()  # LINE 69: Clear without lock
```

### Race Condition (Loss of Originals)
```
Thread A: patch_plotting_functions(config_a)
         Stores _original_functions["_plot_mav"] = <original> at line 42
         Sets mpf_plotting._plot_mav = _plot_mav_accelerated at line 46
         Sets _config = config_a at line 33
         [CONTEXT SWITCH]

Thread B: unpatch_plotting_functions()
         Clears _original_functions at line 69
         [CONTEXT SWITCH]

Thread A: Continues with line 47...
         Now _original_functions is empty (cleared by Thread B)
         Loss of original function reference!

Thread C: patch_plotting_functions(config_c)
         Tries to access _original_functions["_plot_mav"] at line 42
         But it was cleared! Need to re-store...
```

### Severity
- **Probability**: MEDIUM-HIGH
- **Impact**: MEDIUM-HIGH (loss of original functions, undefined behavior)
- **Risk Level**: CRITICAL

---

## Issue #5: HIGH - EngineManager GPU Cache (TOCTOU)

### Location
- **File**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/core/engine.py`
- **Lines**: 54, 61-71, 74-76

### Code
```python
# Line 54 - Class variable
class EngineManager:
    _gpu_available: bool | None = None  # Shared mutable state

    @classmethod
    def check_gpu_available(cls) -> bool:
        """Check if GPU acceleration is available (lightweight check)."""
        if cls._gpu_available is not None:  # LINE 61: Check without lock
            return cls._gpu_available

        try:
            import cudf

            cls._gpu_available = True  # LINE 67: Write without lock
            return True
        except ImportError:
            cls._gpu_available = False  # LINE 70: Write without lock
            return False

    @classmethod
    def reset_gpu_cache(cls) -> None:
        """Reset the GPU availability cache."""
        cls._gpu_available = None  # LINE 76: Write without lock
```

### Race Condition (TOCTOU)
```
Time Event
--- -----
 0  Thread A: if cls._gpu_available is not None:  # False
 1  Thread B: if cls._gpu_available is not None:  # False
 2  Thread A: import cudf (starts)
 3  Thread B: import cudf (starts)
 4  Thread A: cls._gpu_available = True
 5  Thread B: cls._gpu_available = True (redundant, but okay)
 6  Thread C: reset_gpu_cache() -> cls._gpu_available = None
 7  Thread A: Later check -> cls._gpu_available is None, re-imports!
    (Import side effects, redundant work, state inconsistency)
```

### Severity
- **Probability**: MEDIUM-HIGH
- **Impact**: MEDIUM (redundant imports, state inconsistency)
- **Risk Level**: HIGH

---

## Issue #6: MODERATE - File I/O Race in autotune.py

### Location
- **File**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/core/autotune.py`
- **Lines**: 27, 110-113, 118-127

### Code
```python
# Line 27 - Module level
CACHE_FILE = Path.home() / ".kimsfinance" / "threshold_cache.json"

# Lines 110-113 - run_autotune()
def run_autotune(operations: list[str] | None = None, save: bool = True) -> dict[str, int]:
    # ... calculation code ...
    
    if save:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:  # LINE 111: Open for writing
            json.dump(tuned_thresholds, f, indent=4)  # LINE 112
        print(f"Saved tuned thresholds to: {CACHE_FILE}")  # LINE 113
    
    return tuned_thresholds

# Lines 118-127 - load_tuned_thresholds()
def load_tuned_thresholds() -> dict[str, int]:
    """Load tuned thresholds from the cache file."""
    if not CACHE_FILE.exists():
        return DEFAULT_THRESHOLDS

    with open(CACHE_FILE, "r") as f:  # LINE 123: Open for reading
        try:
            return json.load(f)  # LINE 125
        except json.JSONDecodeError:
            return DEFAULT_THRESHOLDS
```

### Race Condition (Partial Write)
```
Thread A: run_autotune(save=True)
         Opens CACHE_FILE for writing at line 111
         Writes partial JSON:
         {"atr": 100_000,
          "rsi": 100_000,
         (file still open, not flushed)
         [CONTEXT SWITCH - file handle still open]

Thread B: load_tuned_thresholds()
         Opens CACHE_FILE for reading at line 123
         File contains partial JSON!
         json.load(f) at line 125
         JSONDecodeError!
         Returns DEFAULT_THRESHOLDS instead

Thread A: (resumes) Finishes writing JSON and closes file
         But Thread B already read corrupted data
```

### Severity
- **Probability**: MEDIUM
- **Impact**: LOW-MEDIUM (corrupted cache, silent failure to defaults)
- **Risk Level**: MODERATE

---

## Issue #7: MODERATE - Monkey-Patching Race

### Location
- **File**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/integration/hooks.py`
- **Lines**: 42-47, 64-68

### Race Condition (Non-Atomic Patching)
```python
# Lines 42-47 - Storing and patching not atomic
_original_functions["_plot_mav"] = mpf_plotting._plot_mav     # LINE 42
_original_functions["_plot_ema"] = mpf_plotting._plot_ema     # LINE 43

# Between these lines and the patching, mplfinance could be called!
mpf_plotting._plot_mav = _plot_mav_accelerated  # LINE 46
mpf_plotting._plot_ema = _plot_ema_accelerated  # LINE 47

# Example:
Thread A at line 42: Stores _original_functions["_plot_mav"] = <original>
Thread B: Calls mplfinance.plot()
         mplfinance calls mpf_plotting._plot_mav()
         Is it original or patched? Unknown!
Thread A at line 46: Sets mpf_plotting._plot_mav = _plot_mav_accelerated
         Now it's patched
```

### Severity
- **Probability**: MEDIUM
- **Impact**: MEDIUM (inconsistent behavior, mixed execution)
- **Risk Level**: MODERATE

---

## Issue #8: LOW - Module Initialization Race

### Location
- **File**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/__init__.py`
- **Lines**: 404-421

### Code
```python
# Lines 404-405
_deps = _check_dependencies()  # Called at module import time

# Lines 407-412
if not _deps["polars"]:
    raise ImportError("Polars is required but not installed. " "Install with: pip install polars")

if not _deps["numpy"]:
    raise ImportError("NumPy is required but not installed. " "Install with: pip install numpy")

# Lines 414-421
if not (_deps["cupy"] and _deps["cudf"]):
    import warnings
    warnings.warn(
        "GPU acceleration not available. Install RAPIDS for GPU support:\n"
        "  pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12 cupy-cuda12x",
        UserWarning,
    )
```

### Risk
- **Probability**: LOW (Python's import lock prevents most issues)
- **Impact**: LOW (unlikely to cause problems in practice)
- **Mitigation**: Python's GIL and import lock handle module loading

### Severity
- **Risk Level**: LOW

---

## Safe Pattern: Multiprocessing in parallel.py

### Location
- **File**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/plotting/parallel.py`
- **Lines**: 148-150

### Code
```python
# Lines 148-150 - Safe multiprocessing pattern
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(_render_one_chart, args_list))

return results
```

### Why It's Safe
1. **Separate Processes**: Each worker is a separate OS process
2. **No Shared Memory**: Data is pickled/unpickled (deep copy)
3. **Isolation**: No race conditions across process boundaries
4. **Executor Management**: ProcessPoolExecutor handles cleanup

### Assessment
- **Thread Safety**: ✓ SAFE
- **Concurrency Pattern**: RECOMMENDED

---

## Summary: Quick Reference

| # | Issue | File | Severity | Risk |
|---|-------|------|----------|------|
| 1 | Unprotected `_is_active` | adapter.py:21,60,83,114,126 | CRITICAL | HIGH |
| 2 | Unprotected `_config` | adapter.py:22-28,68-70,181 | CRITICAL | HIGH |
| 3 | Unprotected `_performance_stats` | adapter.py:31-37,252-267 | CRITICAL | HIGH |
| 4 | Unprotected hooks globals | hooks.py:21-22,33,42-47,69 | CRITICAL | MEDIUM-HIGH |
| 5 | GPU cache TOCTOU | engine.py:54,61,67,70,76 | HIGH | MEDIUM-HIGH |
| 6 | File I/O race | autotune.py:27,111-112,123,125 | MODERATE | MEDIUM |
| 7 | Monkey-patching race | hooks.py:42-47,46-47 | MODERATE | MEDIUM |
| 8 | Module init race | __init__.py:405,407-421 | LOW | LOW |

---

## Recommendations

### Immediate (Critical)
- [ ] Add `threading.Lock()` to adapter.py
- [ ] Add `threading.Lock()` to hooks.py
- [ ] Protect all global state access

### This Sprint (High)
- [ ] Double-checked locking for EngineManager._gpu_available
- [ ] File locking for autotune.py
- [ ] Atomic patching in hooks.py

### Future (Moderate)
- [ ] Concurrent stress tests
- [ ] Thread safety documentation
- [ ] Code review for concurrency patterns

