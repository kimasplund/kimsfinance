# PyPI Release Checklist for kimsfinance v0.1.0

**Package**: kimsfinance
**Version**: 0.1.0
**License**: Dual (AGPL-3.0-or-later + Commercial)
**Python**: 3.13+

---

## Pre-Release Verification

### Code Quality
- [ ] All tests passing (329+ tests)
  ```bash
  pytest tests/ -v
  ```
- [ ] Test coverage above 70%
  ```bash
  pytest --cov=kimsfinance tests/
  ```
- [ ] Type checking passes (mypy strict mode)
  ```bash
  mypy kimsfinance/
  ```
- [ ] Code formatting consistent
  ```bash
  black --check kimsfinance/
  ```
- [ ] No linting errors
  ```bash
  ruff check kimsfinance/
  ```

### Version Management
- [ ] Version bumped in `pyproject.toml` to `0.1.0`
- [ ] Version matches in `kimsfinance/__init__.py` (`__version__ = "0.1.0"`)
- [ ] CHANGELOG.md updated with release date (change `2025-01-XX` to actual date)
- [ ] All new features documented in CHANGELOG.md
- [ ] README.md reflects current capabilities

### Documentation
- [ ] README.md complete with installation instructions
- [ ] API documentation up to date
- [ ] Sample code tested and working
- [ ] Performance benchmarks validated
- [ ] LICENSE file present (AGPL-3.0)
- [ ] COMMERCIAL-LICENSE.md present
- [ ] CONTRIBUTING.md present

### Repository Cleanup
- [ ] No uncommitted changes
  ```bash
  git status
  ```
- [ ] All branches merged to master
- [ ] Git tag created for release
  ```bash
  git tag -a v0.1.0 -m "Release v0.1.0 - Beta"
  git push origin v0.1.0
  ```

---

## Build Package

### Clean Old Builds
```bash
# Remove old distribution files
rm -rf dist/ build/ *.egg-info

# Verify cleanup
ls -la dist/ build/ 2>/dev/null || echo "Cleanup successful"
```

### Build Distribution Files
```bash
# Install/upgrade build tools
pip install --upgrade build twine setuptools wheel

# Build source distribution and wheel
python -m build

# Expected output:
# dist/kimsfinance-0.1.0.tar.gz
# dist/kimsfinance-0.1.0-py3-none-any.whl
```

### Verify Package Contents
```bash
# Check source distribution contents
tar -tzf dist/kimsfinance-0.1.0.tar.gz | head -30

# Expected files:
# - kimsfinance-0.1.0/README.md
# - kimsfinance-0.1.0/LICENSE
# - kimsfinance-0.1.0/CHANGELOG.md
# - kimsfinance-0.1.0/pyproject.toml
# - kimsfinance-0.1.0/kimsfinance/__init__.py
# - kimsfinance-0.1.0/kimsfinance/py.typed
# - kimsfinance-0.1.0/PKG-INFO

# Check wheel contents
unzip -l dist/kimsfinance-0.1.0-py3-none-any.whl | head -30

# Expected files:
# - kimsfinance/__init__.py
# - kimsfinance/py.typed
# - kimsfinance-0.1.0.dist-info/METADATA
# - kimsfinance-0.1.0.dist-info/LICENSE
```

### Validate Package Metadata
```bash
# Check package metadata
twine check dist/*

# Expected output:
# Checking dist/kimsfinance-0.1.0-py3-none-any.whl: PASSED
# Checking dist/kimsfinance-0.1.0.tar.gz: PASSED
```

### Test Local Installation
```bash
# Create test virtual environment
python3.13 -m venv test_env
source test_env/bin/activate

# Install from wheel
pip install dist/kimsfinance-0.1.0-py3-none-any.whl

# Verify import works
python -c "import kimsfinance; print(f'Version: {kimsfinance.__version__}')"

# Test basic functionality
python -c "
from kimsfinance.plotting import render_ohlcv_chart
import numpy as np

ohlc = {
    'open': np.array([100, 102, 101]),
    'high': np.array([103, 104, 102]),
    'low': np.array([99, 101, 100]),
    'close': np.array([102, 101, 103])
}
volume = np.array([1000, 1200, 900])

img = render_ohlcv_chart(ohlc, volume, width=300, height=200)
print(f'Rendered chart: {img.size}')
"

# Deactivate and cleanup
deactivate
rm -rf test_env
```

---

## Test Upload (test.pypi.org)

### Setup Test PyPI Credentials
```bash
# Create/edit ~/.pypirc with test.pypi.org credentials
cat > ~/.pypirc <<EOF
[distutils]
index-servers =
    pypi
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmcC... # Your TestPyPI API token

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmcC... # Your PyPI API token (KEEP SECRET!)
EOF

chmod 600 ~/.pypirc
```

### Upload to Test PyPI
```bash
# Upload to test.pypi.org
python -m twine upload --repository testpypi dist/*

# Expected output:
# Uploading distributions to https://test.pypi.org/legacy/
# Uploading kimsfinance-0.1.0-py3-none-any.whl
# Uploading kimsfinance-0.1.0.tar.gz
# View at:
# https://test.pypi.org/project/kimsfinance/0.1.0/
```

### Verify Test PyPI Upload
- [ ] Visit https://test.pypi.org/project/kimsfinance/
- [ ] Check package description renders correctly
- [ ] Verify version is 0.1.0
- [ ] Check all links work (GitHub, homepage, etc.)
- [ ] Verify dependencies listed correctly
- [ ] Check license badge displays
- [ ] Verify Python version requirement (>=3.13)

### Test Installation from Test PyPI
```bash
# Create clean test environment
python3.13 -m venv testpypi_env
source testpypi_env/bin/activate

# Install from test.pypi.org
# Note: Dependencies will be installed from regular PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ kimsfinance

# Verify installation
python -c "import kimsfinance; print(f'Installed version: {kimsfinance.__version__}')"

# Expected output: Installed version: 0.1.0

# Test basic functionality
python -c "
from kimsfinance.plotting import render_ohlcv_chart, save_chart
import numpy as np

ohlc = {
    'open': np.array([100, 102, 101, 103, 102]),
    'high': np.array([103, 104, 102, 105, 103]),
    'low': np.array([99, 101, 100, 102, 101]),
    'close': np.array([102, 101, 103, 102, 103])
}
volume = np.array([1000, 1200, 900, 1100, 1050])

img = render_ohlcv_chart(ohlc, volume, width=300, height=200, theme='modern')
save_chart(img, '/tmp/test_chart.webp', speed='fast')
print('Test successful: chart saved to /tmp/test_chart.webp')
"

# Verify chart was created
ls -lh /tmp/test_chart.webp

# Test high-level API
python -c "
import kimsfinance as kf
import polars as pl
import numpy as np

# Create sample DataFrame
df = pl.DataFrame({
    'open': np.array([100, 102, 101, 103, 102]),
    'high': np.array([103, 104, 102, 105, 103]),
    'low': np.array([99, 101, 100, 102, 101]),
    'close': np.array([102, 101, 103, 102, 103]),
    'volume': np.array([1000, 1200, 900, 1100, 1050])
})

kf.plot(df, type='candle', volume=True, savefig='/tmp/test_kf_chart.webp')
print('High-level API test successful')
"

# Cleanup
deactivate
rm -rf testpypi_env
```

### Test PyPI Checklist
- [ ] Installation from test.pypi.org works
- [ ] Dependencies installed correctly (polars, numpy, pandas, Pillow)
- [ ] Import works without errors
- [ ] Basic chart rendering works
- [ ] Save to WebP works
- [ ] High-level API (`kf.plot()`) works
- [ ] No missing files or broken imports

---

## Production Upload (pypi.org)

**WARNING: This step is IRREVERSIBLE. You cannot delete or replace a version once uploaded to PyPI!**

### Final Pre-Flight Checks
- [ ] Test PyPI installation verified working
- [ ] All tests passed on clean environment
- [ ] Git tag pushed to GitHub
- [ ] GitHub release draft prepared
- [ ] No last-minute code changes

### Upload to Production PyPI
```bash
# Double-check package contents one more time
twine check dist/*

# Upload to production PyPI
python -m twine upload dist/*

# Expected output:
# Uploading distributions to https://upload.pypi.org/legacy/
# Uploading kimsfinance-0.1.0-py3-none-any.whl
# Uploading kimsfinance-0.1.0.tar.gz
# View at:
# https://pypi.org/project/kimsfinance/0.1.0/
```

### Verify Production PyPI Upload
- [ ] Visit https://pypi.org/project/kimsfinance/
- [ ] Check package description renders correctly
- [ ] Verify all badges display
- [ ] Test all external links
- [ ] Verify classifiers (Beta, Financial, AGPL-3.0)
- [ ] Check keywords for discoverability

### Test Installation from Production PyPI
```bash
# Create clean production test environment
python3.13 -m venv prod_test_env
source prod_test_env/bin/activate

# Install from production PyPI
pip install kimsfinance

# Verify installation
python -c "import kimsfinance; print(kimsfinance.__version__)"

# Expected output: 0.1.0

# Run comprehensive smoke tests
python -c "
import kimsfinance as kf
from kimsfinance.plotting import render_ohlcv_chart, save_chart, render_and_save
import numpy as np
import polars as pl

print('Testing basic chart rendering...')
ohlc = {
    'open': np.array([100, 102, 101, 103, 102]),
    'high': np.array([103, 104, 102, 105, 103]),
    'low': np.array([99, 101, 100, 102, 101]),
    'close': np.array([102, 101, 103, 102, 103])
}
volume = np.array([1000, 1200, 900, 1100, 1050])

img = render_ohlcv_chart(ohlc, volume, width=300, height=200)
print(f'✓ Rendered chart: {img.size}')

print('Testing WebP save...')
save_chart(img, '/tmp/prod_test.webp', speed='fast')
print('✓ WebP save successful')

print('Testing render_and_save...')
render_and_save(ohlc, volume, '/tmp/prod_direct.webp', width=300, height=200)
print('✓ Direct render and save successful')

print('Testing high-level API...')
df = pl.DataFrame({
    'open': ohlc['open'],
    'high': ohlc['high'],
    'low': ohlc['low'],
    'close': ohlc['close'],
    'volume': volume
})
kf.plot(df, type='candle', volume=True, savefig='/tmp/prod_kf.webp')
print('✓ High-level API successful')

print('\\n✅ ALL SMOKE TESTS PASSED')
"

# Cleanup
deactivate
rm -rf prod_test_env
```

### Production Checklist
- [ ] Installation from pypi.org works
- [ ] Smoke tests pass
- [ ] No import errors
- [ ] Chart rendering works
- [ ] WebP encoding works
- [ ] All APIs functional

---

## Post-Release

### GitHub Release
```bash
# Ensure tag is pushed
git push origin v0.1.0

# Create GitHub release via gh CLI
gh release create v0.1.0 \
    --title "kimsfinance v0.1.0 - Beta Release" \
    --notes "$(cat <<EOF
# kimsfinance v0.1.0 - Beta Release

High-performance financial charting library with optional GPU acceleration.

## Performance Highlights
- **28.8x average speedup** over mplfinance (validated: 7.3x - 70.1x range)
- **6,249 charts/sec** peak throughput (batch mode)
- **61x faster** image encoding (WebP fast mode)
- **79% smaller** file sizes (WebP vs PNG)

## Features
- **6 chart types**: Candlestick, OHLC, Line, Hollow, Renko, Point & Figure
- **32 technical indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, and more
- **GPU acceleration**: Optional cuDF/CuPy for 6.4x faster OHLCV processing
- **4 themes**: Classic, Modern, TradingView, Light

## Installation
\`\`\`bash
pip install kimsfinance
\`\`\`

## Quick Start
\`\`\`python
import kimsfinance as kf
import polars as pl

df = pl.read_parquet('ohlcv_data.parquet')
kf.plot(df, type='candle', volume=True, savefig='chart.webp')
\`\`\`

## Links
- PyPI: https://pypi.org/project/kimsfinance/
- Documentation: https://github.com/kimasplund/kimsfinance#readme
- Changelog: https://github.com/kimasplund/kimsfinance/blob/master/CHANGELOG.md

**Built with ⚡ for blazing-fast financial charting**
EOF
)"

# Or create release manually on GitHub
# Visit: https://github.com/kimasplund/kimsfinance/releases/new
```

### Update Documentation
- [ ] Update GitHub README.md with PyPI installation instructions
- [ ] Add PyPI badge to README.md
  ```markdown
  [![PyPI version](https://badge.fury.io/py/kimsfinance.svg)](https://pypi.org/project/kimsfinance/)
  [![Downloads](https://pepy.tech/badge/kimsfinance)](https://pepy.tech/project/kimsfinance)
  ```
- [ ] Update CHANGELOG.md with actual release date
- [ ] Create release notes blog post (if applicable)

### Community Announcements
- [ ] Post to Reddit (r/algotrading, r/Python, r/datascience)
- [ ] Post to Hacker News (Show HN: kimsfinance - 28.8x faster financial charting)
- [ ] Post to Twitter/X
- [ ] Post to LinkedIn
- [ ] Update personal website/portfolio
- [ ] Email to commercial license prospects

### Monitor & Support
- [ ] Watch GitHub issues for bug reports
- [ ] Monitor PyPI download statistics
- [ ] Respond to community questions
- [ ] Track performance feedback
- [ ] Plan v0.2.0 features based on feedback

---

## Rollback Procedure (If Needed)

**Note**: You CANNOT delete or overwrite a PyPI release. If v0.1.0 has critical bugs:

### Option 1: Yank the Release
```bash
# Mark release as yanked (users can still install with explicit version)
# This requires PyPI web interface - cannot be done via CLI
# Visit: https://pypi.org/manage/project/kimsfinance/release/0.1.0/
# Click "Options" → "Yank release"
```

### Option 2: Immediate Patch Release
```bash
# Fix critical bug
# Bump version to 0.1.1
# Upload fixed version
python -m build
python -m twine upload dist/*
```

---

## Dependencies Verification

### Core Dependencies (Required)
| Package | Minimum Version | PyPI Available | Python 3.13 Compatible |
|---------|----------------|----------------|------------------------|
| polars | 1.0 | ✅ Latest: 1.15+ | ✅ Yes |
| numpy | 2.0 | ✅ Latest: 2.2+ | ✅ Yes |
| pandas | 2.0 | ✅ Latest: 2.2+ | ✅ Yes |
| Pillow | 12.0 | ✅ Latest: 12.0+ | ✅ Yes |

### Optional Dependencies (GPU)
| Package | Minimum Version | PyPI Available | Python 3.13 Compatible |
|---------|----------------|----------------|------------------------|
| cudf-cu12 | 24.12 | ✅ pypi.nvidia.com | ⚠️ Limited (CUDA 12.x) |
| cupy-cuda12x | 13.0 | ✅ pypi.nvidia.com | ✅ Yes |

### Optional Dependencies (JIT)
| Package | Minimum Version | PyPI Available | Python 3.13 Compatible |
|---------|----------------|----------------|------------------------|
| numba | 0.59 | ✅ Latest: 0.60+ | ✅ Yes (0.59+) |

### Dev Dependencies
| Package | Minimum Version | PyPI Available | Python 3.13 Compatible |
|---------|----------------|----------------|------------------------|
| pytest | 7.0 | ✅ Latest: 8.3+ | ✅ Yes |
| pytest-cov | 4.0 | ✅ Latest: 6.0+ | ✅ Yes |
| black | 23.0 | ✅ Latest: 25.1+ | ✅ Yes |
| mypy | 1.0 | ✅ Latest: 1.14+ | ✅ Yes |

**All dependencies verified compatible with Python 3.13+**

---

## Common Issues & Solutions

### Issue: `twine upload` fails with authentication error
**Solution**:
- Ensure `~/.pypirc` has correct API token
- Use `--repository testpypi` or `--repository pypi` explicitly
- Generate new API token from PyPI account settings

### Issue: Package description doesn't render on PyPI
**Solution**:
- Check README.md uses valid Markdown (no unsupported syntax)
- Run `twine check dist/*` before uploading
- Test locally: `python -m readme_renderer README.md -o /tmp/output.html`

### Issue: Dependencies not installing from test.pypi.org
**Solution**:
- Use `--extra-index-url https://pypi.org/simple/` to pull deps from main PyPI
- Test PyPI doesn't host all dependencies (polars, numpy, etc.)

### Issue: Import fails after installation
**Solution**:
- Verify `kimsfinance/py.typed` included in wheel
- Check MANIFEST.in includes all necessary files
- Test in isolated environment (not development directory)

### Issue: Version conflict with existing installation
**Solution**:
```bash
pip uninstall kimsfinance
pip install --no-cache-dir kimsfinance
```

---

## Success Criteria

- ✅ Package visible on https://pypi.org/project/kimsfinance/
- ✅ Installation via `pip install kimsfinance` works
- ✅ Import works: `import kimsfinance`
- ✅ Basic functionality works (chart rendering)
- ✅ All dependencies install correctly
- ✅ No import errors or missing modules
- ✅ Documentation renders correctly on PyPI
- ✅ GitHub release created with tag v0.1.0

---

## Additional Resources

- **PyPI Help**: https://pypi.org/help/
- **Twine Documentation**: https://twine.readthedocs.io/
- **Python Packaging Guide**: https://packaging.python.org/
- **TestPyPI**: https://test.pypi.org/
- **PEP 621** (pyproject.toml): https://peps.python.org/pep-0621/

---

**Last Updated**: 2025-10-23
**Prepared by**: Claude Code (kimsfinance release automation)
**Version**: 1.0
