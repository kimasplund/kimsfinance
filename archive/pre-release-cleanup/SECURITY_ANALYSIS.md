# kimsfinance Security Analysis Report

## Executive Summary
Comprehensive security audit of kimsfinance codebase (v0.1.0) covering input validation, file operations, dangerous patterns, and cryptographic issues.

**Analysis Scope**: All Python files in `/kimsfinance` package (excluding .venv and tests)
**Severity Scale**: CRITICAL | HIGH | MEDIUM | LOW | INFORMATIONAL

---

## 1. INPUT VALIDATION ISSUES

### 1.1 Missing Validation on Color Parameters (MEDIUM Severity)
**Location**: `kimsfinance/api/plot.py` (lines 156-158)
**Location**: `kimsfinance/utils/color_utils.py` (lines 4-12)

**Issue**: The `_hex_to_rgba()` function does not validate hex color input format.

```python
def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """Convert hex color string to RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)  # No validation - can raise ValueError
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)
```

**Vulnerability**:
- Invalid hex strings (non-hex characters, wrong length) will raise unhandled `ValueError`
- User can cause application crashes by passing invalid color strings
- No bounds checking on alpha parameter (0-255)

**Risk**: DoS via invalid color values
**Recommendation**:
```python
def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """Convert hex color string to RGBA tuple."""
    if not isinstance(hex_color, str):
        raise ValueError(f"Color must be string, got {type(hex_color)}")
    
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6 or not all(c in '0123456789abcdefABCDEF' for c in hex_color):
        raise ValueError(f"Invalid hex color: #{hex_color}")
    
    if not (0 <= alpha <= 255):
        raise ValueError(f"Alpha must be 0-255, got {alpha}")
    
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)
```

---

### 1.2 Missing Validation on Numeric Parameters (MEDIUM Severity)
**Location**: `kimsfinance/api/plot.py` (lines 153-154)
**Location**: `kimsfinance/plotting/pil_renderer.py` (lines 389-422, 543-570)

**Issue**: Image dimensions and chart parameters lack validation:

```python
# In plot.py - no validation on width/height
width = kwargs.get("width", 1920)
height = kwargs.get("height", 1080)

# In pil_renderer.py - no validation on box_size, reversal_boxes, line_width
def render_renko_chart(
    ..., box_size: float | None = None, reversal_boxes: int = 1, ...
):
    # No validation that box_size > 0, reversal_boxes > 0
```

**Vulnerability**:
- Negative or zero width/height can cause PIL errors
- Negative box_size/reversal_boxes can cause rendering errors or infinite loops
- Extremely large dimensions can cause memory exhaustion (DoS)
- No upper bounds on line_width, wick_width_ratio

**Affected Parameters**:
- `width` (default 1920)
- `height` (default 1080)
- `box_size` (Renko/PNF)
- `reversal_boxes` (Renko/PNF)
- `line_width` (line charts)
- `wick_width_ratio` (candle charts)

**Risk**: DoS via resource exhaustion or application crashes
**Recommendation**: Add validation in `plot.py`:
```python
def _validate_numeric_params(width: int, height: int, **kwargs) -> None:
    if not (1 <= width <= 65536):
        raise ValueError(f"Width must be 1-65536, got {width}")
    if not (1 <= height <= 65536):
        raise ValueError(f"Height must be 1-65536, got {height}")
    
    if 'box_size' in kwargs and kwargs['box_size'] is not None:
        if kwargs['box_size'] <= 0:
            raise ValueError(f"box_size must be > 0")
    
    if 'reversal_boxes' in kwargs:
        if not (1 <= kwargs['reversal_boxes'] <= 100):
            raise ValueError(f"reversal_boxes must be 1-100")
    
    if 'line_width' in kwargs:
        if not (1 <= kwargs['line_width'] <= 50):
            raise ValueError(f"line_width must be 1-50")
```

---

### 1.3 Insufficient Validation on File Paths (MEDIUM-HIGH Severity)
**Location**: `kimsfinance/api/plot.py` (lines 163-165, 285-288)
**Location**: `kimsfinance/plotting/pil_renderer.py` (lines 30-197)

**Issue**: File path inputs lack validation for path traversal:

```python
# In plot.py - no path validation
is_svg_format = savefig and (
    savefig.lower().endswith(".svg") or savefig.lower().endswith(".svgz")
)

# Vulnerable to path traversal
savefig = "../../../etc/passwd.webp"  # No validation!

# In pil_renderer.py - direct save without path validation
img.save(output_path, "WEBP", **default_params)  # output_path not validated
```

**Vulnerability**:
- No path normalization (realpath check)
- No validation that path is within expected directory
- Susceptible to path traversal attacks: `../../sensitive/file.webp`
- Symlink attacks possible (no symlink detection)

**Risk**: Arbitrary file write (file overwrite, overwrite config files, etc.)

**Recommendation**:
```python
from pathlib import Path

def _validate_output_path(output_path: str, base_dir: Path | None = None) -> Path:
    """Validate output path prevents traversal attacks."""
    if not output_path:
        raise ValueError("output_path cannot be empty")
    
    path = Path(output_path).resolve()  # Resolve symlinks and relative paths
    
    # If base_dir specified, ensure path is within it
    if base_dir:
        base_dir = base_dir.resolve()
        try:
            path.relative_to(base_dir)
        except ValueError:
            raise ValueError(f"Path {path} is outside allowed directory {base_dir}")
    
    # Ensure parent directory exists or can be created
    path.parent.mkdir(parents=True, exist_ok=True)
    
    return path
```

**Usage in `plot.py`**:
```python
if savefig:
    safe_path = _validate_output_path(savefig)
    save_chart(img, str(safe_path), ...)
```

---

## 2. FILE OPERATIONS SECURITY

### 2.1 Unsafe JSON File Operations (MEDIUM Severity)
**Location**: `kimsfinance/core/autotune.py` (lines 27, 109-112, 123-127)

**Issue**: JSON cache file operations lack proper error handling:

```python
CACHE_FILE = Path.home() / ".kimsfinance" / "threshold_cache.json"

# In run_autotune():
with open(CACHE_FILE, "w") as f:
    json.dump(tuned_thresholds, f, indent=4)  # No error handling

# In load_tuned_thresholds():
with open(CACHE_FILE, "r") as f:
    try:
        return json.load(f)  # Untrusted JSON from disk!
    except json.JSONDecodeError:
        return DEFAULT_THRESHOLDS  # Silent fallback
```

**Vulnerability**:
1. **No file permission validation**: Home directory writable by other users
2. **Untrusted JSON deserialization**: If file is tampered with, malformed data could cause issues
3. **No atomic writes**: Concurrent writes could corrupt cache
4. **No file integrity check**: Can't detect if file was modified
5. **Poor error handling**: `IOError`, `PermissionError` not caught on write

**Risk**: Data corruption, privilege escalation (if home dir is shared), DoS

**Recommendation**:
```python
import json
import tempfile
from pathlib import Path

def _safe_write_json(path: Path, data: dict) -> None:
    """Safely write JSON with atomic operation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use temp file for atomic write
    with tempfile.NamedTemporaryFile(
        mode='w', dir=path.parent, delete=False, suffix='.tmp'
    ) as tmp:
        try:
            json.dump(data, tmp, indent=4)
            tmp.flush()
            tmp_path = Path(tmp.name)
        except Exception as e:
            tmp_path.unlink()
            raise IOError(f"Failed to write cache: {e}")
    
    # Atomic rename
    tmp_path.replace(path)
    path.chmod(0o600)  # Restrict to owner only

def _safe_read_json(path: Path) -> dict:
    """Safely read JSON with validation."""
    if not path.exists():
        return {}
    
    # Check file permissions (warn if world-readable)
    stat_info = path.stat()
    if stat_info.st_mode & 0o077:
        warnings.warn(f"Cache file {path} is world-readable")
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Validate data structure
        if not isinstance(data, dict):
            raise ValueError("Cache must be JSON object")
        
        return data
    except (json.JSONDecodeError, IOError, ValueError) as e:
        warnings.warn(f"Invalid cache file {path}: {e}")
        return {}
```

---

### 2.2 SVG File Operations with Compression (LOW-MEDIUM Severity)
**Location**: `kimsfinance/plotting/svg_renderer.py` (lines 20-39)

**Issue**: SVGZ compression uses gzip with insufficient error handling:

```python
def _save_svg_or_svgz(dwg: "svgwrite.Drawing", output_path: str) -> None:
    if output_path.endswith(".svgz"):
        svg_string = dwg.tostring()
        with open(output_path, "wb") as f:
            f.write(gzip.compress(svg_string.encode("utf-8"), compresslevel=9))
    else:
        dwg.saveas(output_path)
```

**Vulnerability**:
- No validation that `output_path` is string
- No path normalization (traversal possible)
- `gzip.compress()` with `compresslevel=9` could be resource-intensive
- No file permission handling

**Risk**: Path traversal, resource exhaustion
**Recommendation**: Combine with path validation from Section 2.1

---

## 3. MONKEY PATCHING & DYNAMIC MODIFICATION

### 3.1 Monkey Patching of mplfinance (MEDIUM Severity)
**Location**: `kimsfinance/integration/adapter.py` and `kimsfinance/integration/hooks.py`

**Issue**: The integration module monkey-patches mplfinance functions:

```python
# In hooks.py
mpf_plotting._plot_mav = _plot_mav_accelerated
mpf_plotting._plot_ema = _plot_ema_accelerated

# In adapter.py
_is_active = False  # Global state
_config = {...}      # Global mutable state
```

**Vulnerability**:
1. **Global state mutation**: Monkey patching affects entire process
2. **Thread safety issues**: Global `_is_active` and `_config` not thread-safe
3. **No restoration guarantee**: If error occurs, patches remain active
4. **State management**: activate/deactivate can be called multiple times, leading to inconsistent state

**Risk**: 
- Unexpected behavior in other code using mplfinance
- Race conditions in multi-threaded environments
- Difficult to debug side effects

**Recommendation**: Use context manager pattern:
```python
from contextlib import contextmanager

@contextmanager
def kimsfinance_acceleration(engine: str = "auto", strict: bool = False):
    """Context manager for safe GPU acceleration."""
    activate(engine=engine, strict=strict)
    try:
        yield
    finally:
        deactivate()

# Usage:
with kimsfinance_acceleration(engine="gpu"):
    result = mpf.plot(df, type='candle')
```

Add thread safety:
```python
import threading

_lock = threading.RLock()

def activate(*, engine: str = "auto", ...):
    global _is_active
    with _lock:
        if _is_active:
            return
        # ... rest of code
```

---

## 4. CONFIGURATION & VALIDATION

### 4.1 Configuration Injection (LOW-MEDIUM Severity)
**Location**: `kimsfinance/integration/adapter.py` (lines 149-182)

**Issue**: Configuration accepts arbitrary string values:

```python
def configure(**kwargs) -> None:
    valid_keys = {
        "default_engine",
        "gpu_min_rows",
        "strict_mode",
        "performance_tracking",
        "verbose",
    }
    
    for key, value in kwargs.items():
        if key not in valid_keys:
            raise ValueError(f"Invalid configuration key: {key!r}")
        _config[key] = value  # No type validation!
```

**Vulnerability**:
- No type checking on values (e.g., `default_engine` should be "cpu"/"gpu"/"auto")
- Boolean values not validated (accepts any truthy value)
- Numeric values (`gpu_min_rows`) not validated (could be negative, string, etc.)

**Recommendation**:
```python
def configure(**kwargs) -> None:
    """Configure kimsfinance with type validation."""
    valid_config = {
        "default_engine": {"type": str, "values": {"cpu", "gpu", "auto"}},
        "gpu_min_rows": {"type": int, "min": 100, "max": 10_000_000},
        "strict_mode": {"type": bool},
        "performance_tracking": {"type": bool},
        "verbose": {"type": bool},
    }
    
    for key, value in kwargs.items():
        if key not in valid_config:
            raise ValueError(f"Invalid config key: {key!r}")
        
        config_spec = valid_config[key]
        
        if not isinstance(value, config_spec["type"]):
            raise TypeError(f"{key} must be {config_spec['type']}, got {type(value)}")
        
        if "values" in config_spec:
            if value not in config_spec["values"]:
                raise ValueError(f"{key} must be one of {config_spec['values']}, got {value!r}")
        
        if "min" in config_spec and value < config_spec["min"]:
            raise ValueError(f"{key} must be >= {config_spec['min']}")
        
        if "max" in config_spec and value > config_spec["max"]:
            raise ValueError(f"{key} must be <= {config_spec['max']}")
        
        _config[key] = value
```

---

## 5. MISSING SECURITY FEATURES

### 5.1 No Input Sanitization on DataFrame Input (INFORMATIONAL)
**Location**: `kimsfinance/api/plot.py` (lines 415-449)

```python
def _prepare_data(data):
    """Convert DataFrame to OHLC dict - no validation of data integrity."""
    # ... code assumes data has correct structure ...
    ohlc_dict = {
        "open": df["Open"].to_numpy() if "Open" in df.columns else df["open"].to_numpy(),
        # ...
    }
```

**Note**: This is appropriate for a library (caller responsibility), but should document assumptions clearly.

---

### 5.2 No CORS/CSRF Protection (N/A)
The library is server-side Python only, not a web service. No CORS/CSRF issues.

---

### 5.3 No Cryptographic Operations (N/A)
The library performs no cryptographic operations. No crypto vulnerabilities.

---

### 5.4 No SQL Operations (N/A)
The library has no database operations. No SQL injection risks.

---

### 5.5 No Unsafe Deserialization (N/A)
The library does not use `pickle`, `eval()`, or `exec()`. No deserialization vulnerabilities found.

---

### 5.6 No Hard-coded Secrets (PASS)
No API keys, passwords, or tokens found in the codebase. Configuration is clean.

---

## 6. DEPENDENCY SECURITY

### 6.1 Pillow Library (PIL) (INFORMATIONAL)
**Location**: `pyproject.toml` (line 43)
- **Dependency**: `Pillow>=12.0`
- **Status**: Up-to-date
- **Note**: Pillow is well-maintained. Using latest version is good practice.

### 6.2 Polars Library (INFORMATIONAL)
**Location**: `pyproject.toml` (line 40)
- **Dependency**: `polars>=1.0`
- **Status**: Good (actively maintained)
- **Note**: No known security issues

### 6.3 NumPy Library (INFORMATIONAL)
**Location**: `pyproject.toml` (line 41)
- **Dependency**: `numpy>=2.0`
- **Status**: Good
- **Note**: No security concerns

### 6.4 Optional GPU Dependencies
- **cudf-cu12**: Not security critical, fails gracefully if unavailable
- **cupy-cuda12x**: Not security critical, fails gracefully if unavailable

---

## 7. SUMMARY OF FINDINGS

### By Severity:

| Severity | Issue | Location | Count |
|----------|-------|----------|-------|
| **MEDIUM-HIGH** | Path Traversal in File Operations | api/plot.py, pil_renderer.py | 1 |
| **MEDIUM** | Color Parameter Validation | utils/color_utils.py | 1 |
| **MEDIUM** | Numeric Parameter Validation | api/plot.py, plotting/ | 1 |
| **MEDIUM** | JSON Cache File Security | core/autotune.py | 1 |
| **MEDIUM** | SVG File Operations | plotting/svg_renderer.py | 1 |
| **MEDIUM** | Monkey Patching Safety | integration/ | 1 |
| **LOW-MEDIUM** | Configuration Type Validation | integration/adapter.py | 1 |
| **INFORMATIONAL** | Documentation | Various | - |

**Total Issues Found**: 8
- Critical: 0
- High: 1 (path traversal)
- Medium: 6
- Low: 1
- Informational: 1

---

## 8. RECOMMENDATIONS (Priority Order)

### Priority 1: CRITICAL (Path Traversal)
1. ✅ Implement path validation in `plot.py` and `pil_renderer.py`
2. ✅ Use `Path.resolve()` to eliminate relative paths and symlinks
3. ✅ Restrict file writes to allowed directories if applicable

### Priority 2: HIGH (Input Validation)
4. ✅ Add color format validation to `_hex_to_rgba()`
5. ✅ Add numeric parameter validation for width/height/box_size/etc.
6. ✅ Add type checking to `configure()` function

### Priority 3: MEDIUM (File Operations)
7. ✅ Implement atomic writes for JSON cache using temp files
8. ✅ Add file permission validation for home directory cache
9. ✅ Add error handling for file I/O operations

### Priority 4: MEDIUM (Thread Safety)
10. ✅ Add thread-safe locking to monkey patching
11. ✅ Implement context manager for activation/deactivation
12. ✅ Document thread safety limitations

### Priority 5: LOW (Documentation)
13. ✅ Document input assumptions and constraints
14. ✅ Add security advisory to README if applicable
15. ✅ Document file operation behavior and limitations

---

## 9. POSITIVE SECURITY FINDINGS

✅ **No eval/exec usage**: Code does not use dangerous Python functions
✅ **No pickle deserialization**: No unsafe deserialization patterns
✅ **No hard-coded secrets**: No API keys or credentials in code
✅ **No subprocess execution**: No command injection vulnerabilities
✅ **No SQL operations**: No SQL injection risks
✅ **Type hints enabled**: Uses Python type hints (mypy strict mode)
✅ **Good error handling**: Generally good exception handling patterns
✅ **Modern dependencies**: Uses current versions of libraries

---

## 10. TESTING RECOMMENDATIONS

Add security-focused tests:

```python
# tests/test_security.py
def test_hex_color_invalid_format():
    """Validate color input validation."""
    from kimsfinance.utils.color_utils import _hex_to_rgba
    
    with pytest.raises(ValueError):
        _hex_to_rgba("GGGGGG")  # Invalid hex
    
    with pytest.raises(ValueError):
        _hex_to_rgba("12345")  # Wrong length
    
    with pytest.raises(ValueError):
        _hex_to_rgba("#12345")  # Wrong length with #

def test_path_traversal_blocked():
    """Validate path traversal is prevented."""
    from kimsfinance.api import plot
    
    # Should reject paths with ../
    with pytest.raises(ValueError):
        plot(df, savefig="../../etc/passwd.webp")

def test_numeric_bounds():
    """Validate numeric parameters are bounded."""
    from kimsfinance.api import plot
    
    # Width too large
    with pytest.raises(ValueError):
        plot(df, width=999999)
    
    # Height zero
    with pytest.raises(ValueError):
        plot(df, height=0)
    
    # Negative box_size
    with pytest.raises(ValueError):
        plot(df, type="renko", box_size=-1)
```

---

## Conclusion

The kimsfinance codebase is generally secure with good practices in place. The main vulnerabilities are:

1. **Path traversal** (file write operations) - requires immediate fixes
2. **Input validation gaps** (colors, numeric parameters) - medium priority
3. **File operation safety** (JSON cache, error handling) - medium priority
4. **Thread safety** (monkey patching) - lower priority for library usage

All identified issues have recommended fixes above. With these corrections implemented, the codebase would achieve strong security posture suitable for production use.

