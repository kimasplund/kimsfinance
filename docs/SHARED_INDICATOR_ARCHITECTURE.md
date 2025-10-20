# Shared Indicator Architecture

**For**: All 20 parallel indicator implementation tasks
**Date**: 2025-10-20

---

## File Locations

### Implementation
- **File**: `kimsfinance/ops/indicators.py`
- **Add functions to existing file** (already has 10 indicators)
- **Preserve existing code** (do not modify ATR, RSI, MACD, etc.)

### Tests
- **File**: `tests/test_indicators.py`
- **Add test classes to existing file**
- **Follow existing test patterns**

### Exports
- **File**: `kimsfinance/ops/__init__.py`
- **Add new function names to `__all__` list**
- **Add new imports**

---

## Code Standards

### Function Signature Pattern

```python
def calculate_<indicator_name>(
    # Input arrays (use descriptive names)
    prices: ArrayLike,  # or highs, lows, closes, volumes
    # Parameters with defaults
    period: int = <standard_default>,
    # Keyword-only engine parameter
    *,
    engine: Engine = "auto"
) -> ArrayResult | tuple[ArrayResult, ...]:
    """
    Calculate <Full Indicator Name> (<ACRONYM>).

    Automatically uses GPU for datasets > 500,000 rows when engine="auto".

    <Brief description of what the indicator measures and how it's used>

    Args:
        prices: Input price data (or high, low, close, volume)
        period: Lookback period for calculation (default: X)
        engine: Computation engine ('auto', 'cpu', 'gpu')

    Returns:
        Array of <indicator> values (or tuple of arrays if multiple outputs)

    Raises:
        ValueError: If period < 1 or inputs have mismatched lengths

    Examples:
        >>> import polars as pl
        >>> df = pl.read_csv("ohlcv.csv")
        >>> <indicator> = calculate_<name>(df['Close'], period=14)
    """
```

### Implementation Pattern

```python
def calculate_<indicator>(
    data: ArrayLike,
    period: int = <default>,
    *,
    engine: Engine = "auto"
) -> ArrayResult:
    """Docstring here."""

    # 1. VALIDATE INPUTS
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    # 2. CONVERT TO NUMPY (standardize input)
    data_array = np.asarray(data, dtype=np.float64)

    if len(data_array) < period:
        raise ValueError(f"Insufficient data: need {period}, got {len(data_array)}")

    # 3. ENGINE ROUTING
    if engine == "auto":
        use_gpu = _should_use_gpu(data_array)
    elif engine == "gpu":
        use_gpu = True
    elif engine == "cpu":
        use_gpu = False
    else:
        raise ValueError(f"Invalid engine: {engine}")

    # 4. DISPATCH TO CPU OR GPU
    if use_gpu:
        return _calculate_<indicator>_gpu(data_array, period)
    else:
        return _calculate_<indicator>_cpu(data_array, period)


def _calculate_<indicator>_cpu(
    data: np.ndarray,
    period: int
) -> np.ndarray:
    """CPU implementation using NumPy."""

    # Initialize result array with NaN
    result = np.full(len(data), np.nan, dtype=np.float64)

    # Implement algorithm
    # ... calculation logic here ...

    return result


def _calculate_<indicator>_gpu(
    data: np.ndarray,
    period: int
) -> np.ndarray:
    """GPU implementation using CuPy."""

    try:
        import cupy as cp
    except ImportError:
        # Fallback to CPU if CuPy not available
        return _calculate_<indicator>_cpu(data, period)

    # Transfer to GPU
    data_gpu = cp.asarray(data, dtype=cp.float64)

    # Initialize result array with NaN
    result_gpu = cp.full(len(data_gpu), cp.nan, dtype=cp.float64)

    # Implement algorithm using CuPy
    # ... calculation logic here ...

    # Transfer back to CPU
    return cp.asnumpy(result_gpu)
```

---

## Test Pattern

### Test Class Structure

```python
class Test<IndicatorName>:
    """Test <Full Indicator Name> calculation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing."""
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 2)
        return prices

    def test_basic_calculation(self, sample_data):
        """Test basic <indicator> calculation."""
        result = calculate_<indicator>(sample_data, period=14, engine='cpu')

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))
        # First (period-1) values should be NaN
        assert np.all(np.isnan(result[:13]))
        # After warmup, should have valid values
        assert not np.isnan(result[14])

    def test_gpu_cpu_match(self, sample_data):
        """Test GPU and CPU implementations produce identical results."""
        cpu_result = calculate_<indicator>(sample_data, period=14, engine='cpu')
        gpu_result = calculate_<indicator>(sample_data, period=14, engine='gpu')

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10)

    def test_invalid_period(self, sample_data):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_<indicator>(sample_data, period=0)

    def test_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        short_data = np.array([100, 101, 102])
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_<indicator>(short_data, period=14)

    def test_known_values(self):
        """Test against known indicator values."""
        # Use small dataset with hand-calculated expected values
        data = np.array([...])  # Known test data
        result = calculate_<indicator>(data, period=X, engine='cpu')
        expected = np.array([...])  # Hand-calculated expected values

        np.testing.assert_allclose(result, expected, rtol=1e-6)
```

---

## Type Hints

### Import Required Types

```python
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike

# Type aliases (already defined in indicators.py)
ArrayResult = np.ndarray
Engine = Literal["auto", "cpu", "gpu"]
```

### Array Type Usage

- **Input parameters**: Use `ArrayLike` (accepts lists, arrays, Series, etc.)
- **Return values**: Use `ArrayResult` or `np.ndarray`
- **Internal functions**: Use `np.ndarray` (already converted)

---

## Common Algorithms

### Rolling Window Operations

```python
# CPU (NumPy)
for i in range(period - 1, len(data)):
    window = data[i - period + 1 : i + 1]
    result[i] = np.mean(window)  # or max, min, etc.

# GPU (CuPy) - use convolution or rolling
result_gpu = cp.convolve(data_gpu, cp.ones(period)/period, mode='same')
```

### Exponential Moving Average

```python
# CPU (NumPy)
alpha = 2.0 / (period + 1)
result[period - 1] = np.mean(data[:period])  # SMA as first value
for i in range(period, len(data)):
    result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

# GPU - iterative (challenging to parallelize, may need custom kernel)
```

### Directional Movement (for ADX)

```python
# +DM = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i]
# -DM = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1]

up_move = highs[1:] - highs[:-1]
down_move = lows[:-1] - lows[1:]

plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
```

---

## GPU Optimization Tips

### When to Use GPU

```python
def _should_use_gpu(data: np.ndarray, threshold: int = 500_000) -> bool:
    """Determine if GPU should be used based on data size."""
    try:
        import cupy as cp
        return len(data) >= threshold
    except ImportError:
        return False
```

### CuPy Equivalents

| NumPy | CuPy | Notes |
|-------|------|-------|
| `np.array` | `cp.array` | Direct equivalent |
| `np.mean` | `cp.mean` | Direct equivalent |
| `np.std` | `cp.std` | Direct equivalent |
| `np.cumsum` | `cp.cumsum` | Direct equivalent |
| `np.convolve` | `cp.convolve` | Direct equivalent |
| `np.maximum.accumulate` | `cp.maximum.accumulate` | Direct equivalent |
| `np.minimum.accumulate` | `cp.minimum.accumulate` | Direct equivalent |

### Custom CUDA Kernels (Advanced)

For complex iterative algorithms (Parabolic SAR, Ichimoku):

```python
# Use Numba CUDA for custom kernels
from numba import cuda

@cuda.jit
def custom_kernel(input_array, output_array, period):
    idx = cuda.grid(1)
    if idx < output_array.size:
        # Kernel logic
        pass
```

---

## Dependencies Between Indicators

### Reuse Existing Implementations

**Already Implemented** (can reuse):
- `calculate_atr` - Used by: Keltner Channels, Supertrend
- `calculate_rsi` - Reference implementation
- `calculate_macd` - Reference implementation

**Must Implement First**:
- `calculate_sma` - Used by: DEMA, TEMA
- `calculate_ema` - Used by: DEMA, TEMA, Keltner, Elder Ray, Supertrend

**Implementation Order**:
1. **SMA, EMA** (foundational)
2. **All others** (can be parallel, may depend on SMA/EMA)

---

## Error Handling

### Required Validations

```python
# 1. Check parameter ranges
if period < 1:
    raise ValueError(f"period must be >= 1, got {period}")

# 2. Check data length
if len(data) < period:
    raise ValueError(f"Insufficient data: need {period}, got {len(data)}")

# 3. Check for mismatched array lengths (multi-input functions)
if len(highs) != len(lows):
    raise ValueError(f"highs and lows must have same length")

# 4. Handle NaN/inf gracefully
if np.any(np.isnan(data)) or np.any(np.isinf(data)):
    warnings.warn("Input data contains NaN or inf values")
```

---

## Documentation Requirements

### Docstring Template

```python
"""
Calculate <Full Name> (<ACRONYM>).

Automatically uses GPU for datasets > 500,000 rows when engine="auto".

<1-2 sentence description of what the indicator measures>

<Brief explanation of how it's calculated>

<Common use cases and interpretation>

Args:
    <param>: <Description>
    period: Lookback period (default: X)
    engine: Computation engine ('auto', 'cpu', 'gpu')

Returns:
    Array of <indicator> values (length matches input)
    First (period-1) values are NaN due to warmup

Raises:
    ValueError: If period < 1 or insufficient data

Examples:
    >>> import polars as pl
    >>> df = pl.read_csv("ohlcv.csv")
    >>> <indicator> = calculate_<name>(df['Close'], period=14)

References:
    - <Link to Wikipedia or authoritative source>
    - <Link to original paper if applicable>
"""
```

---

## Integration Checklist

### After Implementation, Each Indicator Must:

- [ ] Be added to `kimsfinance/ops/indicators.py`
- [ ] Be exported in `kimsfinance/ops/__init__.py` (import and `__all__`)
- [ ] Have test class in `tests/test_indicators.py`
- [ ] Have minimum 4 tests (basic, gpu/cpu match, invalid input, known values)
- [ ] Pass all tests with `pytest tests/test_indicators.py -v`
- [ ] Have full type hints (no `Any` types)
- [ ] Have complete docstring
- [ ] Handle edge cases (NaN, inf, insufficient data)

---

## Example: Complete Implementation

See existing indicators in `kimsfinance/ops/indicators.py`:
- `calculate_atr` - Good example of OHLC input
- `calculate_rsi` - Good example of single input
- `calculate_macd` - Good example of multiple outputs

---

## Performance Benchmarking

### Add Benchmark Test (Optional)

```python
def test_performance_benchmark(self):
    """Benchmark CPU vs GPU performance."""
    n = 1_000_000
    data = np.random.randn(n).cumsum() + 100

    import time

    # CPU
    start = time.perf_counter()
    cpu_result = calculate_<indicator>(data, period=14, engine='cpu')
    cpu_time = time.perf_counter() - start

    # GPU
    start = time.perf_counter()
    gpu_result = calculate_<indicator>(data, period=14, engine='gpu')
    gpu_time = time.perf_counter() - start

    speedup = cpu_time / gpu_time
    print(f"\n<Indicator> Benchmark (1M rows):")
    print(f"  CPU: {cpu_time*1000:.2f}ms")
    print(f"  GPU: {gpu_time*1000:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")
```

---

## Common Pitfalls to Avoid

1. **Don't modify existing indicators** - Only add new ones
2. **Don't use `any` type** - Always use proper type hints
3. **Don't skip GPU implementation** - All indicators should have GPU path
4. **Don't forget warmup period** - First (period-1) values should be NaN
5. **Don't skip edge case tests** - Test invalid inputs, insufficient data
6. **Don't hardcode magic numbers** - Use named parameters with defaults
7. **Don't forget to export** - Add to `__all__` in `__init__.py`

---

## Questions or Issues?

Refer to existing indicators in `kimsfinance/ops/indicators.py` for patterns and examples.

**Good luck with parallel implementation!**
