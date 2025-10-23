# Contributing to kimsfinance

Thank you for your interest in contributing to **kimsfinance**! We're thrilled to have you here. Whether you're fixing bugs, adding features, improving documentation, or simply asking questions, your contributions help make this library better for the entire quantitative finance community.

This guide will help you get started with contributing to the project.

---

## Table of Contents

1. [Ways to Contribute](#ways-to-contribute)
2. [Development Setup](#development-setup)
3. [Code Style Guidelines](#code-style-guidelines)
4. [Testing Requirements](#testing-requirements)
5. [Pull Request Process](#pull-request-process)
6. [Community Guidelines](#community-guidelines)
7. [Questions and Help](#questions-and-help)

---

## Ways to Contribute

### 1. Report Bugs

Found a bug? Help us fix it!

- **Search existing issues** first to avoid duplicates
- **Open a new issue** with a clear title and description
- **Include**:
  - Steps to reproduce the bug
  - Expected vs actual behavior
  - Your environment (Python version, OS, GPU info if relevant)
  - Minimal code example that demonstrates the issue
  - Error messages or stack traces

[Report a bug here](https://github.com/kimasplund/kimsfinance/issues/new?labels=bug)

### 2. Suggest Features

Have an idea for a new feature or improvement?

- **Check existing issues** to see if it's already been proposed
- **Open a feature request** describing:
  - The problem you're trying to solve
  - Your proposed solution
  - Why this would be useful to the community
  - Any implementation ideas or considerations

[Request a feature here](https://github.com/kimasplund/kimsfinance/issues/new?labels=enhancement)

### 3. Improve Documentation

Documentation is crucial! You can help by:

- Fixing typos or unclear explanations
- Adding code examples
- Writing tutorials or guides
- Improving API documentation
- Creating visual diagrams or charts

### 4. Contribute Code

Code contributions are welcome! Areas where we'd love help:

- **Bug fixes** - Fix reported issues
- **Performance optimizations** - Make things faster (with benchmarks!)
- **New chart types** - Line charts, area charts, Renko, Point & Figure
- **Technical indicators** - SMA, EMA, RSI, MACD, Bollinger Bands
- **GPU optimizations** - CuPy/cuDF improvements
- **Test coverage** - Expand our 329+ test suite
- **Platform support** - Windows, macOS, ARM architecture

### 5. Help Others

- Answer questions in GitHub Discussions
- Review pull requests
- Share your experience using kimsfinance
- Write blog posts or create tutorials

---

## Development Setup

### Prerequisites

- **Python 3.13+** (required)
- **Git** for version control
- **Virtual environment** (recommended)
- **NVIDIA GPU** (optional, for GPU development)

### Clone the Repository

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/kimsfinance.git
cd kimsfinance

# Add upstream remote
git remote add upstream https://github.com/kimasplund/kimsfinance.git
```

### Create a Virtual Environment

```bash
# Create virtual environment
python3.13 -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### Install Dependencies

#### Core Dependencies

```bash
# Install in development mode (editable)
pip install -e .
```

#### Development Tools

```bash
# Install testing and code quality tools
pip install -e ".[dev]"

# Or install individually
pip install pytest pytest-cov black mypy ruff
```

#### Optional: GPU Support

```bash
# For GPU-accelerated OHLCV processing
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12 cupy-cuda12x

# Verify GPU support
python -c "import cudf; import cupy; print('GPU ready!')"
```

#### Optional: JIT Compilation

```bash
# For 50-100% faster coordinate computation
pip install "kimsfinance[jit]"
# or manually:
pip install numba>=0.59
```

#### All Optional Dependencies

```bash
# Install everything for full development
pip install -e ".[dev,gpu,jit,test]"
```

### Verify Installation

```bash
# Run tests to ensure everything works
pytest tests/

# Should see: 329+ tests passed
```

---

## Code Style Guidelines

We maintain high code quality standards to ensure the codebase remains maintainable and professional.

### Type Hints (Required)

All functions must have type hints:

```python
# Good ✅
def calculate_moving_average(
    prices: np.ndarray,
    window: int
) -> np.ndarray:
    """Calculate simple moving average."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

# Bad ❌
def calculate_moving_average(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')
```

### Docstrings (NumPy Style)

All public functions, classes, and modules must have docstrings:

```python
def render_ohlcv_chart(
    ohlc: dict[str, ArrayLike],
    volume: ArrayLike,
    width: int = 300,
    height: int = 200,
    theme: str = 'classic'
) -> Image.Image:
    """
    Render a candlestick chart with volume bars.

    Parameters
    ----------
    ohlc : dict[str, ArrayLike]
        Dictionary containing 'open', 'high', 'low', 'close' arrays.
    volume : ArrayLike
        Volume data corresponding to each candle.
    width : int, default 300
        Chart width in pixels.
    height : int, default 200
        Chart height in pixels.
    theme : str, default 'classic'
        Color theme: 'classic', 'modern', 'tradingview', or 'light'.

    Returns
    -------
    Image.Image
        PIL Image object containing the rendered chart.

    Examples
    --------
    >>> ohlc = {
    ...     'open': [100, 102, 101],
    ...     'high': [103, 104, 102],
    ...     'low': [99, 101, 100],
    ...     'close': [102, 101, 103],
    ... }
    >>> volume = [1000, 1200, 900]
    >>> img = render_ohlcv_chart(ohlc, volume, theme='modern')
    """
    # Implementation...
```

### Code Formatting

We use **black** and **ruff** for consistent code formatting:

```bash
# Format all code
black kimsfinance/ tests/

# Check formatting without changes
black --check kimsfinance/

# Lint with ruff
ruff check kimsfinance/

# Auto-fix lint issues
ruff check --fix kimsfinance/
```

**Configuration** (already set in `pyproject.toml`):
- Line length: 100 characters
- Target: Python 3.13
- Style: black defaults

### Type Checking

We use **mypy** in strict mode:

```bash
# Run type checking
mypy kimsfinance/

# Check specific file
mypy kimsfinance/plotting/renderer.py
```

### Code Quality Checklist

Before submitting a PR, ensure:

- ✅ All functions have type hints
- ✅ All public APIs have NumPy-style docstrings
- ✅ Code formatted with `black`
- ✅ No lint errors from `ruff`
- ✅ No type errors from `mypy`
- ✅ All tests pass
- ✅ New features have tests

### Performance Considerations

kimsfinance is a **performance-focused library**. When contributing:

1. **Prefer NumPy vectorization** over Python loops
2. **Use C-contiguous arrays** for better cache performance
3. **Minimize allocations** in hot paths
4. **Consider optional Numba JIT** for critical functions
5. **Profile before optimizing** - measure, don't guess!

Example:

```python
# Good ✅ - Vectorized
body_heights = np.abs(close - open)

# Bad ❌ - Python loop
body_heights = np.array([abs(c - o) for c, o in zip(close, open)])
```

---

## Testing Requirements

All code contributions must include tests. We maintain **329+ tests** with comprehensive coverage.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=kimsfinance --cov-report=html tests/

# Run specific test file
pytest tests/test_renderer_ohlc.py

# Run specific test function
pytest tests/test_renderer_ohlc.py::test_render_basic_chart

# Run tests matching pattern
pytest -k "test_render"

# Run with verbose output
pytest -v tests/
```

### Writing Tests

Place tests in the appropriate location:

```
tests/
├── plotting/
│   ├── test_renderer_ohlc.py
│   ├── test_renderer_line.py
│   └── test_themes.py
├── ops/
│   ├── test_aggregations.py
│   └── test_indicators.py
└── test_api.py
```

Example test:

```python
import pytest
import numpy as np
from kimsfinance.plotting import render_ohlcv_chart


def test_render_basic_chart():
    """Test basic chart rendering with minimal data."""
    ohlc = {
        'open': np.array([100.0, 102.0, 101.0]),
        'high': np.array([103.0, 104.0, 102.0]),
        'low': np.array([99.0, 101.0, 100.0]),
        'close': np.array([102.0, 101.0, 103.0]),
    }
    volume = np.array([1000, 1200, 900])

    img = render_ohlcv_chart(ohlc, volume, width=300, height=200)

    assert img.size == (300, 200)
    assert img.mode == 'RGB'


def test_render_with_invalid_data():
    """Test that invalid data raises appropriate errors."""
    ohlc = {
        'open': np.array([100.0]),
        'high': np.array([103.0]),
        'low': np.array([99.0]),
        'close': np.array([102.0]),
    }
    volume = np.array([1000, 1200])  # Mismatched length!

    with pytest.raises(ValueError, match="Length mismatch"):
        render_ohlcv_chart(ohlc, volume)
```

### Test Guidelines

1. **Test new features** - Every new feature needs tests
2. **Test edge cases** - Empty data, single candle, large datasets
3. **Test error handling** - Invalid inputs should raise clear errors
4. **Use fixtures** - Share test data with pytest fixtures
5. **Keep tests fast** - Use small datasets when possible
6. **Test both CPU and GPU** paths (if applicable)

### Performance Testing

For performance optimizations, **always include benchmarks**:

```python
import time
import numpy as np
from kimsfinance.plotting import render_ohlcv_chart


def test_render_performance_1000_candles():
    """Benchmark: Rendering 1000 candles should be < 50ms."""
    # Generate test data
    n = 1000
    ohlc = {
        'open': np.random.random(n) * 100 + 100,
        'high': np.random.random(n) * 100 + 110,
        'low': np.random.random(n) * 100 + 90,
        'close': np.random.random(n) * 100 + 100,
    }
    volume = np.random.randint(1000, 10000, n)

    # Benchmark
    start = time.perf_counter()
    img = render_ohlcv_chart(ohlc, volume, width=300, height=200)
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

    assert elapsed < 50.0, f"Rendering took {elapsed:.2f}ms (target: <50ms)"
```

### Running Benchmarks

For comprehensive benchmarks:

```bash
# Quick performance check
pytest tests/benchmark_ohlc_bars.py

# Use the benchmark commands (see CLAUDE.md)
/kf/bench/compare
```

---

## Pull Request Process

### 1. Fork and Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout master
git merge upstream/master

# Create a feature branch
git checkout -b feature/my-awesome-feature
# or
git checkout -b fix/bug-description
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `perf/` - Performance optimizations
- `test/` - Test additions/improvements

### 2. Make Your Changes

Follow the [Code Style Guidelines](#code-style-guidelines) and add tests!

```bash
# Make changes
vim kimsfinance/plotting/renderer.py

# Format code
black kimsfinance/ tests/

# Run tests
pytest tests/

# Check types
mypy kimsfinance/
```

### 3. Commit Your Changes

We use **Conventional Commits** for clear commit messages:

```bash
# Format: <type>(<scope>): <description>

# Examples:
git commit -m "feat(plotting): add line chart renderer"
git commit -m "fix(renderer): correct wick width calculation"
git commit -m "docs(api): add examples to render_ohlcv_chart"
git commit -m "perf(drawing): vectorize coordinate computation (2.5x faster)"
git commit -m "test(renderer): add edge case tests for single candle"
```

**Commit types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `perf` - Performance improvements
- `test` - Test additions/changes
- `refactor` - Code refactoring (no behavior change)
- `style` - Code style changes (formatting, no logic change)
- `chore` - Build process, dependencies, tooling

**For performance improvements, include benchmark results:**

```bash
git commit -m "perf(renderer): vectorize grid drawing (1.8x faster)

Before: 12.5ms average for 500 candles
After: 6.9ms average for 500 candles
Speedup: 1.81x
Benchmark: pytest tests/benchmark_ohlc_bars.py"
```

### 4. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/my-awesome-feature
```

Then open a Pull Request on GitHub with:

**Title:** Clear, concise description
```
feat(plotting): Add line chart renderer with antialiasing support
```

**Description Template:**
```markdown
## Summary

Brief description of what this PR does.

## Changes

- Added line chart renderer
- Implemented antialiasing for smooth lines
- Added tests for line chart functionality

## Testing

- [ ] All existing tests pass
- [ ] New tests added for line chart
- [ ] Manual testing with various datasets
- [ ] Performance benchmarks (if applicable)

## Performance Impact

For line charts (500 candles):
- Rendering time: 8.2ms average
- Throughput: 122 charts/sec
- Compared to OHLC: 0.95x (slightly faster)

## Documentation

- [ ] Updated API documentation
- [ ] Added code examples
- [ ] Updated CHANGELOG.md (if applicable)

## Breaking Changes

None / List any breaking changes

## Checklist

- [x] Code follows style guidelines (black, ruff, mypy)
- [x] All tests pass
- [x] New functionality has tests
- [x] Documentation updated
- [x] Performance benchmarks included (if perf change)
- [x] Commit messages follow Conventional Commits
```

### 5. Code Review

- **Be responsive** to feedback
- **Make requested changes** in new commits (don't force-push)
- **Be patient** - maintainers will review when available
- **Ask questions** if feedback is unclear

### 6. Merge

Once approved, a maintainer will merge your PR. Thank you for contributing! 🎉

---

## Community Guidelines

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. We expect all participants to:

- **Be respectful** - Treat everyone with respect and kindness
- **Be constructive** - Provide helpful feedback and suggestions
- **Be inclusive** - Welcome newcomers and diverse perspectives
- **Be professional** - Keep discussions focused and on-topic

### Unacceptable Behavior

- Harassment, trolling, or personal attacks
- Offensive or discriminatory comments
- Publishing others' private information
- Spam or off-topic content

### Reporting Issues

If you experience or witness unacceptable behavior, please contact:
- **Email:** hello@asplund.kim
- **Confidential reporting:** [Provide confidential channel if available]

All reports will be handled with discretion and confidentiality.

### Full Code of Conduct

For detailed community guidelines, please see our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) (coming soon).

---

## Questions and Help

### Getting Help

**For bugs or issues:**
- [Open a GitHub issue](https://github.com/kimasplund/kimsfinance/issues/new)

**For questions or discussions:**
- [GitHub Discussions](https://github.com/kimasplund/kimsfinance/discussions) (coming soon)
- Check the [README.md](README.md) and documentation first

**For commercial licensing inquiries:**
- Email: licensing@asplund.kim
- See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)

**For general questions:**
- Email: hello@asplund.kim

### Resources

- **README.md** - Project overview and quick start
- **API Documentation** - In-depth API reference
- **Performance Guide** - Optimization tips and benchmarks
- **CLAUDE.md** - Development configuration and agent setup

---

## Development Tips

### Useful Commands

```bash
# Run quick tests
pytest tests/plotting/test_renderer_ohlc.py -v

# Watch for changes and re-run tests
pytest-watch tests/

# Generate coverage report
pytest --cov=kimsfinance --cov-report=html tests/
# Then open: htmlcov/index.html

# Profile performance
python -m cProfile -s cumulative scripts/benchmark_test.py

# Check for type errors
mypy kimsfinance/ --show-error-codes

# Format everything
black kimsfinance/ tests/ scripts/
ruff check --fix kimsfinance/
```

### Common Development Tasks

**Adding a new chart type:**
1. Create renderer in `kimsfinance/plotting/renderer_<type>.py`
2. Add tests in `tests/plotting/test_renderer_<type>.py`
3. Update `kimsfinance/plotting/__init__.py` to export
4. Update `kimsfinance/api/plot.py` to support new type
5. Add documentation and examples

**Adding a technical indicator:**
1. Implement in `kimsfinance/ops/indicators.py`
2. Add GPU version in `kimsfinance/ops/indicators_gpu.py` (optional)
3. Add tests in `tests/ops/test_indicators.py`
4. Add integration tests in `tests/test_all_operations.py`
5. Update documentation

**Performance optimization:**
1. Write benchmark first (`tests/benchmark_*.py`)
2. Profile to identify bottleneck (`cProfile`, `line_profiler`)
3. Implement optimization
4. Verify speedup with benchmark
5. Ensure all tests still pass
6. Document performance improvement in commit message

### GPU Development

If working on GPU features:

```bash
# Verify GPU available
nvidia-smi

# Test GPU functionality
pytest tests/test_gpu_operations.py

# Profile GPU kernels
/kf/profile/gpu-kernel

# Memory leak detection
/kf/test/memory
```

### Debugging

```bash
# Run with verbose output
pytest -vv tests/test_specific.py

# Drop into debugger on failure
pytest --pdb tests/

# Print all output (including print statements)
pytest -s tests/

# Run only last failed tests
pytest --lf tests/
```

---

## Recognition

Contributors are recognized in several ways:

- **CONTRIBUTORS.md** - Your name listed as a contributor
- **Release notes** - Mentioned in release announcements
- **Git history** - Your commits permanently in the project history

---

## License

By contributing to kimsfinance, you agree that your contributions will be licensed under the **AGPL-3.0** license. See [LICENSE](LICENSE) for details.

For commercial licensing inquiries, contact: licensing@asplund.kim

---

## Final Notes

Thank you for reading this guide! We're excited to see your contributions.

**Remember:**
- Start small - fix typos, improve docs, add tests
- Ask questions - we're here to help
- Be patient - reviews take time
- Have fun - you're making the library better!

**Let's build the fastest financial charting library together!** 🚀

---

**Questions?** Open an issue or email hello@asplund.kim

**Happy coding!**
