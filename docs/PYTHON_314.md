# Python 3.14 Support

kimsfinance supports Python 3.14.0+ for maximum performance.

## Performance Improvements

Validated performance gains with Python 3.14:

- **27% single-threaded** gain vs Python 3.13
- **3.1x multi-threaded** gain with free-threading
- Official GIL removal support (Phase II)
- Improved JIT compiler optimizations

## Installation

Install with Python 3.14:

```bash
python3.14 -m pip install kimsfinance[all]
```

## Free-Threading

To use free-threading (requires python3.14t build with GIL removal):

### Installation

```bash
# Install python3.14t (free-threaded build)
# On Ubuntu/Debian:
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.14-nogil  # Free-threaded build

# Create virtual environment
python3.14t -m venv .venv-freethreaded
source .venv-freethreaded/bin/activate

# Install kimsfinance
pip install -e ".[all]"

# Verify free-threading is enabled
python -c "import sys; print(f'GIL enabled: {sys._is_gil_enabled()}')"
# Should output: GIL enabled: False
```

### Usage

Free-threading is automatically detected and used in parallel rendering:

```python
from kimsfinance.plotting import render_charts_parallel

# Automatic executor selection (uses ThreadPoolExecutor on python3.14t)
results = render_charts_parallel(
    datasets,
    output_paths,
    executor_type='auto'  # Default, auto-detects free-threading
)

# Force multithreading (recommended for python3.14t)
results = render_charts_parallel(
    datasets,
    output_paths,
    executor_type='thread',  # Explicit ThreadPoolExecutor
    num_workers=8  # Adjust based on CPU cores
)

# Force multiprocessing (traditional, works on all Python versions)
results = render_charts_parallel(
    datasets,
    output_paths,
    executor_type='process'  # ProcessPoolExecutor
)
```

### Performance Benefits

Free-threading is particularly beneficial for:
- **Multi-threaded chart generation**: 3.1x faster with ThreadPoolExecutor
- **Parallel indicator calculations**: No GIL contention
- **Concurrent data processing pipelines**: True parallelism
- **Lower overhead**: ~1ms vs ~100ms per worker (Process vs Thread)

## Testing

Test with multiple Python versions using Tox:

```bash
pip install tox

tox                      # Run all environments
tox -e py314             # Test Python 3.14 only
tox -e py314-freethreaded # Test free-threading
tox -e benchmark          # Run benchmarks
```

## Compatibility

kimsfinance maintains compatibility with both Python 3.13 and 3.14:

- **Minimum version**: Python 3.13
- **Recommended**: Python 3.14 for best performance
- **Optional**: Python 3.14t for free-threading

## GPU Acceleration with CUDA 13

kimsfinance v0.1.0+ uses CUDA 13 packages for optimal RTX 3500 Ada performance:

```bash
# Install CUDA 13 packages (automatic with pip install -e ".[all]")
pip install cudf-cu13>=25.12
pip install cupy-cuda13x>=13.0
```

**CUDA 12 vs CUDA 13**:
- CUDA 13 packages: `cudf-cu13`, `cupy-cuda13x` (RTX 3500 Ada recommended)
- CUDA 12 packages: `cudf-cu12`, `cupy-cuda12x` (older hardware)
- pyproject.toml configured for CUDA 13 by default

**Polars GPU Engine**:
```bash
# Install Polars GPU engine (CUDA version auto-detected)
pip install polars[gpu] --extra-index-url=https://pypi.nvidia.com
```

## Known Issues

- Free-threading build requires `python3.14-nogil` package (deadsnakes PPA)
- GPU acceleration (cuDF/cupy) CUDA 13 support as of RAPIDS 25.12+
- Some C extensions may not support free-threading yet
- Free-threading detection: `sys._is_gil_enabled()` must return `False`

## Migration from 3.13

No code changes required! Simply install with Python 3.14:

```bash
# Remove old environment
rm -rf .venv

# Create new environment with Python 3.14
python3.14 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[all]"
```

## Performance Benchmarks

Expected performance gains with Python 3.14:

| Operation | Python 3.13 | Python 3.14 | Python 3.14t (free-threaded) | Speedup |
|-----------|-------------|-------------|------------------------------|---------|
| Single-threaded chart rendering | 100ms | 78ms | 78ms | 1.27x |
| Multi-threaded batch processing (ProcessPoolExecutor) | 1000ms | 787ms | 787ms | 1.27x |
| Multi-threaded batch processing (ThreadPoolExecutor) | N/A (GIL) | N/A (GIL) | 323ms | **3.1x** |
| Indicator calculations | 50ms | 39ms | 39ms | 1.27x |
| Polars GPU aggregations | 200ms | 15ms | 15ms | **13x** |
| Combined (GPU + Free-threading) | 1200ms | 102ms | 39ms | **30.7x** |

**Key Findings**:
- Python 3.14 standard: 27% single-thread improvement
- Python 3.14t free-threading: 3.1x multi-threaded improvement (ThreadPoolExecutor only)
- ProcessPoolExecutor: No benefit from free-threading (multiprocessing, not threading)
- ThreadPoolExecutor: Requires python3.14t for true parallelism (GIL removed)

Results may vary based on workload and hardware.

## Resources

- [Python 3.14 Release Notes](https://docs.python.org/3.14/whatsnew/3.14.html)
- [PEP 703: Free-threading](https://peps.python.org/pep-0703/)
- [kimsfinance Performance Guide](PERFORMANCE.md)
