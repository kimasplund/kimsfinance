# kimsfinance Thread Safety Analysis Report

**Analysis Date**: 2025-10-22  
**Thoroughness Level**: Medium  
**Scope**: Global state, race conditions, and concurrency issues

---

## Executive Summary

The kimsfinance codebase has **critical thread safety issues** in the integration layer and several **moderate concerns** in the engine management layer. The main problems are:

1. **Unprotected global mutable state** in `adapter.py` and `hooks.py`
2. **Non-atomic operations** on shared dictionaries without synchronization
3. **Class-level mutable cache** without locking in `EngineManager`
4. **File I/O race conditions** in `autotune.py`
5. **Monkey-patching without coordination** in hooks

**Risk Level**: HIGH (integration) to MEDIUM (core)

---

## Critical Issues

### 1. CRITICAL: Unprotected Global State in adapter.py

**Location**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/integration/adapter.py` (lines 21-37, 250-267)

**Issue**: Three module-level global variables are modified without any synchronization:

```python
# Line 21-37: Module-level globals
_is_active = False
_config = {
    "default_engine": "auto",
    "gpu_min_rows": 10_000,
    "strict_mode": False,
    "performance_tracking": False,
    "verbose": True,
}

_performance_stats = {
    "total_calls": 0,
    "gpu_calls": 0,
    "cpu_calls": 0,
    "time_saved_ms": 0.0,
    "speedup": 1.0,
}
```

**Functions Modifying These Globals**:
- `activate()` (line 60): `global _is_active` + modifies `_config`
- `deactivate()` (line 114): `global _is_active`
- `configure()` (line 168): `global _config` modifies dictionary
- `_track_operation()` (line 250): `global _performance_stats` modifies all dict values
- `reset_performance_stats()` (line 230): `global _performance_stats` reassigns entire dict

**Race Condition Scenario**:
```
Thread A: activate() reads _is_active (False), about to set it True
Thread B: activate() reads _is_active (False), also about to set it True
         Both threads execute patch_plotting_functions() - DOUBLE PATCHING

Thread C: _track_operation() increments _performance_stats["total_calls"]
Thread D: _track_operation() increments _performance_stats["total_calls"]
         Non-atomic dict update - lost updates possible
```

**Impact**: 
- Lost performance statistics (counter increments skipped)
- Double patching of mplfinance functions
- Inconsistent configuration state across threads
- Data corruption in performance tracking dictionaries

**Risk Assessment**: **CRITICAL** (HIGH probability, HIGH impact)

---

### 2. CRITICAL: Unprotected Global State in hooks.py

**Location**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/integration/hooks.py` (lines 21-22)

**Issue**: Global state without synchronization:

```python
# Line 21-22: Module-level globals
_original_functions = {}
_config = {}
```

**Functions Modifying These Globals**:
- `patch_plotting_functions()` (line 32-33): Sets `_config = config`
- `unpatch_plotting_functions()` (line 69): Clears `_original_functions`
- Multiple plotting functions read `_config` (lines 92, 114, 165)

**Race Condition Scenario**:
```
Thread A: patch_plotting_functions(config=cfg_a) sets _config = cfg_a
Thread B: patch_plotting_functions(config=cfg_b) sets _config = cfg_b
Thread C: _plot_mav_accelerated() reads _config, sees mixed/inconsistent state

Thread D: unpatch_plotting_functions() clears _original_functions
Thread E: _plot_mav_accelerated() tries to fallback using _original_functions
         KeyError because it was just cleared
```

**Impact**:
- Inconsistent configuration during concurrent plotting
- Loss of original function references
- Potential undefined behavior in plotting pipelines
- Silent corruption of mplfinance patches

**Risk Assessment**: **CRITICAL** (HIGH probability, MEDIUM-HIGH impact)

---

### 3. HIGH: Non-Atomic Performance Statistics Updates

**Location**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/integration/adapter.py` (lines 250-267)

**Issue**: `_track_operation()` performs non-atomic compound operations:

```python
def _track_operation(engine_used: str, time_saved_ms: float = 0.0) -> None:
    global _performance_stats
    
    _performance_stats["total_calls"] += 1  # RMW (Read-Modify-Write)
    
    if engine_used == "gpu":
        _performance_stats["gpu_calls"] += 1  # RMW
    else:
        _performance_stats["cpu_calls"] += 1  # RMW
    
    if time_saved_ms > 0:
        _performance_stats["time_saved_ms"] += time_saved_ms  # RMW
    
    # Compound operation without atomicity
    if _performance_stats["total_calls"] > 0:
        avg_speedup = 1.0 + (
            _performance_stats["time_saved_ms"] / (_performance_stats["total_calls"] * 10)
        )
        _performance_stats["speedup"] = avg_speedup  # RMW on derived value
```

**Thread Safety Problem**:
```
Thread A reads: total_calls = 10
Thread B reads: total_calls = 10
Thread A writes: total_calls = 11
Thread B writes: total_calls = 11  # Lost one increment!
Expected: 12, Got: 11

Same for gpu_calls, cpu_calls, time_saved_ms
```

**Impact**:
- Performance statistics become unreliable
- Speedup calculation based on corrupted data
- Misleading metrics for performance monitoring

**Risk Assessment**: **HIGH** (HIGH probability, MEDIUM impact)

---

### 4. HIGH: Race Condition in configure() Function

**Location**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/integration/adapter.py` (lines 149-184)

**Issue**: Dictionary modification without atomic guarantees:

```python
def configure(**kwargs) -> None:
    global _config
    
    valid_keys = {
        "default_engine",
        "gpu_min_rows",
        "strict_mode",
        "performance_tracking",
        "verbose",
    }
    
    for key, value in kwargs.items():
        if key not in valid_keys:
            raise ValueError(...)
        _config[key] = value  # Non-atomic dict update
```

**Race Condition Scenario**:
```
Thread A: configure(default_engine="gpu", gpu_min_rows=50_000)
          Sets: _config["default_engine"] = "gpu"
          [CONTEXT SWITCH]
Thread B: Reads _config["default_engine"] = "gpu" ✓
         But reads _config["gpu_min_rows"] = 10_000 (old value) ✗
         Inconsistent state!

Thread C: configure(default_engine="cpu")
         Sets: _config["default_engine"] = "cpu"
Thread A: Sets: _config["gpu_min_rows"] = 50_000
         Now _config is {"default_engine": "cpu", "gpu_min_rows": 50_000}
         Inconsistent configuration!
```

**Impact**:
- Inconsistent configuration values
- Engine selection based on partially-updated config
- Behavior changes mid-operation

**Risk Assessment**: **HIGH** (MEDIUM-HIGH probability, MEDIUM impact)

---

### 5. HIGH: Class-Level Mutable Cache Without Locking

**Location**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/core/engine.py` (lines 54, 61-71, 74-76)

**Issue**: Class variable `_gpu_available` used as cache without synchronization:

```python
class EngineManager:
    _gpu_available: bool | None = None  # Class variable (shared mutable state)
    
    @classmethod
    def check_gpu_available(cls) -> bool:
        if cls._gpu_available is not None:
            return cls._gpu_available  # TOCTOU race condition
        
        try:
            import cudf
            cls._gpu_available = True  # Write without lock
            return True
        except ImportError:
            cls._gpu_available = False  # Write without lock
            return False
    
    @classmethod
    def reset_gpu_cache(cls) -> None:
        cls._gpu_available = None  # Write without lock
```

**Race Condition Scenario**:
```
Thread A: check_gpu_available()
         Checks: if cls._gpu_available is not None: False
         About to import cudf...
         [CONTEXT SWITCH]

Thread B: check_gpu_available()
         Checks: if cls._gpu_available is not None: False
         Also starts importing cudf (TOCTOU - Time-Of-Check Time-Of-Use)
         [CONTEXT SWITCH]

Thread A: Completes import, sets cls._gpu_available = True
Thread B: Completes import, sets cls._gpu_available = True
         Both did redundant work, but okay...

Thread C: reset_gpu_cache() sets _gpu_available = None
Thread A: check_gpu_available() sees None, re-imports cudf
         Creates import side effects, state inconsistency
```

**Impact**:
- TOCTOU vulnerabilities
- Redundant GPU availability checks
- Cache invalidation during concurrent access
- Possible double initialization

**Risk Assessment**: **HIGH** (MEDIUM-HIGH probability, MEDIUM impact)

---

## Moderate Issues

### 6. MODERATE: File I/O Race Condition in autotune.py

**Location**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/core/autotune.py` (lines 27, 110-113, 118-127)

**Issue**: Cache file read/write without synchronization:

```python
CACHE_FILE = Path.home() / ".kimsfinance" / "threshold_cache.json"

def run_autotune(operations: list[str] | None = None, save: bool = True) -> dict[str, int]:
    # ...
    if save:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(tuned_thresholds, f, indent=4)  # Write without lock
        print(f"Saved tuned thresholds to: {CACHE_FILE}")
    
    return tuned_thresholds

def load_tuned_thresholds() -> dict[str, int]:
    if not CACHE_FILE.exists():
        return DEFAULT_THRESHOLDS
    
    with open(CACHE_FILE, "r") as f:  # Read without lock
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return DEFAULT_THRESHOLDS
```

**Race Condition Scenario**:
```
Thread A: run_autotune(save=True)
         Opens CACHE_FILE for writing
         [CONTEXT SWITCH before close]

Thread B: load_tuned_thresholds()
         Opens CACHE_FILE for reading
         Reads partially-written JSON
         json.JSONDecodeError or corrupted data

Thread C: load_tuned_thresholds()
         Opens CACHE_FILE while Thread A is writing
         File is in unknown state
```

**Impact**:
- Corrupted threshold cache
- JSONDecodeError exceptions
- Inconsistent tuning parameters
- Fallback to default thresholds silently

**Risk Assessment**: **MODERATE** (MEDIUM probability, LOW-MEDIUM impact)

---

### 7. MODERATE: Monkey-Patching Race Condition

**Location**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/integration/hooks.py` (lines 42-47, 64-68)

**Issue**: mplfinance module patching without coordination:

```python
def patch_plotting_functions(config: dict[str, Any]) -> None:
    # Store original functions
    _original_functions["_plot_mav"] = mpf_plotting._plot_mav
    _original_functions["_plot_ema"] = mpf_plotting._plot_ema
    
    # Patch plotting functions
    mpf_plotting._plot_mav = _plot_mav_accelerated
    mpf_plotting._plot_ema = _plot_ema_accelerated

def unpatch_plotting_functions() -> None:
    if not _original_functions:
        return
    
    # Restore original functions
    if "_plot_mav" in _original_functions:
        mpf_plotting._plot_mav = _original_functions["_plot_mav"]
    if "_plot_ema" in _original_functions:
        mpf_plotting._plot_ema = _original_functions["_plot_ema"]
    
    _original_functions.clear()
```

**Race Condition Scenario**:
```
Thread A: activate() calls patch_plotting_functions()
         Sets mpf_plotting._plot_mav = _plot_mav_accelerated
         [CONTEXT SWITCH]

Thread B: Uses mplfinance.plot()
         Calls mpf_plotting._plot_mav()
         Is it original or patched? Depends on timing!
         May see mixed behavior

Thread C: deactivate() calls unpatch_plotting_functions()
         Clears _original_functions
         [CONTEXT SWITCH]

Thread A: Tries to patch again, calls patch_plotting_functions()
         No original stored, patches with wrong fallback
```

**Impact**:
- Inconsistent behavior in mplfinance plots
- Wrong functions called
- Loss of original function references
- Potential fallback to wrong implementations

**Risk Assessment**: **MODERATE** (MEDIUM probability, MEDIUM impact)

---

## Low Severity Issues

### 8. LOW: Initialization Race in __init__.py

**Location**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/__init__.py` (line 405)

**Issue**: Module-level initialization without synchronization:

```python
def _check_dependencies():
    """Check and report on optional dependencies."""
    deps = {...}
    # Multiple try/except blocks checking imports
    return deps

# Check dependencies on import
_deps = _check_dependencies()

if not _deps["polars"]:
    raise ImportError(...)
```

**Impact**: 
- Import race during module load (rare in practice)
- Multiple threads could trigger dependency checks
- Minimal impact due to import lock in Python

**Risk Assessment**: **LOW** (LOW probability due to Python's import lock, LOW impact)

---

## Multiprocessing Analysis

### 9. Safe: Multiprocessing in parallel.py

**Location**: `/home/kim/Documents/Github/kimsfinance/kimsfinance/plotting/parallel.py`

**Analysis**: Multiprocessing approach is **SAFE for threads**:

```python
def render_charts_parallel(
    datasets: list[dict[str, Any]],
    output_paths: list[str] | None = None,
    num_workers: int | None = None,
    speed: str = "fast",
    **common_render_kwargs: Any,
) -> list[str | bytes]:
    # ...
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_render_one_chart, args_list))
    
    return results
```

**Why It's Safe**:
- Uses separate OS processes (not threads)
- Data is pickled/unpickled (deep copy)
- No shared mutable state between processes
- ProcessPoolExecutor handles isolation

**Notes**:
- Spawning processes is slower than threading
- Process isolation prevents all race conditions
- Good design choice for this use case

**Risk Assessment**: **SAFE** - No thread safety issues here

---

## Summary Table

| Issue | Severity | Location | Type | Probability | Impact |
|-------|----------|----------|------|-------------|--------|
| Unprotected `_is_active` | CRITICAL | adapter.py:21-37 | Global state race | HIGH | HIGH |
| Unprotected `_config` dict | CRITICAL | adapter.py:22-28 | Non-atomic updates | HIGH | HIGH |
| Unprotected `_performance_stats` | CRITICAL | adapter.py:31-37 | Non-atomic RMW | HIGH | MEDIUM |
| Unprotected hooks globals | CRITICAL | hooks.py:21-22 | Global state race | HIGH | MEDIUM |
| Non-atomic performance tracking | HIGH | adapter.py:250-267 | Race condition | HIGH | MEDIUM |
| configure() race condition | HIGH | adapter.py:168-181 | Dict corruption | MEDIUM-HIGH | MEDIUM |
| GPU cache without lock | HIGH | engine.py:54-76 | TOCTOU race | MEDIUM-HIGH | MEDIUM |
| Cache file I/O | MODERATE | autotune.py:110-127 | File corruption | MEDIUM | LOW-MEDIUM |
| Monkey-patching race | MODERATE | hooks.py:42-68 | State inconsistency | MEDIUM | MEDIUM |
| Module initialization | LOW | __init__.py:405 | Import race | LOW | LOW |

---

## Recommended Fixes

### Priority 1: Critical (Implement Immediately)

1. **Add threading.Lock to adapter.py**:
```python
import threading

_lock = threading.Lock()
_is_active = False
_config = {...}
_performance_stats = {...}

def activate(...):
    global _is_active
    with _lock:
        if _is_active:
            ...
        _config["default_engine"] = engine
        ...
        _is_active = True
```

2. **Add threading.Lock to hooks.py**:
```python
import threading

_lock = threading.Lock()
_original_functions = {}
_config = {}

def patch_plotting_functions(config):
    global _config
    with _lock:
        _config = config
        ...
```

3. **Make performance tracking thread-safe**:
```python
import threading

_perf_lock = threading.Lock()

def _track_operation(engine_used, time_saved_ms=0.0):
    if not _config["performance_tracking"]:
        return
    
    global _performance_stats
    with _perf_lock:
        _performance_stats["total_calls"] += 1
        if engine_used == "gpu":
            _performance_stats["gpu_calls"] += 1
        else:
            _performance_stats["cpu_calls"] += 1
        ...
```

### Priority 2: High (Implement Within Sprint)

1. **Add lock to EngineManager**:
```python
class EngineManager:
    _gpu_available: bool | None = None
    _gpu_lock = threading.Lock()
    
    @classmethod
    def check_gpu_available(cls) -> bool:
        if cls._gpu_available is not None:
            return cls._gpu_available
        
        with cls._gpu_lock:
            if cls._gpu_available is not None:  # Double-check
                return cls._gpu_available
            
            try:
                import cudf
                cls._gpu_available = True
                return True
            except ImportError:
                cls._gpu_available = False
                return False
```

2. **Add file locking to autotune.py**:
```python
import fcntl
import threading

_cache_lock = threading.Lock()

def load_tuned_thresholds() -> dict[str, int]:
    if not CACHE_FILE.exists():
        return DEFAULT_THRESHOLDS
    
    with _cache_lock:
        with open(CACHE_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return DEFAULT_THRESHOLDS

def run_autotune(...):
    ...
    if save:
        with _cache_lock:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(tuned_thresholds, f, indent=4)
```

### Priority 3: Moderate (Implement in Future)

1. **Coordinate monkey-patching with locks**
2. **Add integration tests for concurrent activation/deactivation**
3. **Document thread safety guarantees in API**

---

## Testing Recommendations

Add concurrent stress tests:

```python
import threading
import time

def test_concurrent_activation():
    """Test thread safety of activate/deactivate."""
    errors = []
    
    def activate_thread():
        try:
            for _ in range(100):
                activate()
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)
    
    threads = [threading.Thread(target=activate_thread) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert not errors, f"Thread safety violations: {errors}"

def test_concurrent_performance_tracking():
    """Test concurrent _track_operation updates."""
    initial = get_performance_stats()["total_calls"]
    
    def track_operations():
        for _ in range(1000):
            _track_operation("cpu")
    
    threads = [threading.Thread(target=track_operations) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    final = get_performance_stats()["total_calls"]
    expected = initial + 10000
    
    assert final == expected, f"Lost updates: expected {expected}, got {final}"
```

---

## Conclusion

**Overall Assessment**: The kimsfinance library has **critical thread safety issues** in its integration layer that **MUST be fixed** before using with threaded applications. The core engine and plotting modules are relatively safe, but the global state management in `adapter.py` and `hooks.py` creates significant race conditions.

**Recommendation**: Implement Priority 1 fixes (all critical) before next release. These are simple threading.Lock additions that don't change API or functionality.

