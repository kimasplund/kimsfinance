# PyPI Quick Reference - kimsfinance v0.1.0

**Quick command reference for PyPI upload process**

---

## Pre-Upload

```bash
# 1. Run tests
pytest tests/ -v

# 2. Clean old builds
rm -rf dist/ build/ *.egg-info

# 3. Build package
python -m build

# 4. Verify package
twine check dist/*
```

---

## Test Upload (test.pypi.org)

```bash
# Upload to test PyPI
python -m twine upload --repository testpypi dist/*

# Install from test PyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    kimsfinance

# Verify import
python -c "import kimsfinance; print(kimsfinance.__version__)"
```

---

## Production Upload (pypi.org)

```bash
# Upload to production PyPI
python -m twine upload dist/*

# Install from production PyPI
pip install kimsfinance

# Verify import
python -c "import kimsfinance; print(kimsfinance.__version__)"
```

---

## Smoke Test

```bash
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
save_chart(img, '/tmp/smoke_test.webp', speed='fast')
print('✅ Smoke test passed: chart saved to /tmp/smoke_test.webp')
"
```

---

## Git Tagging

```bash
# Create and push tag
git tag -a v0.1.0 -m "Release v0.1.0 - Beta"
git push origin v0.1.0

# Verify tag
git tag -l
```

---

## Troubleshooting

### Clean reinstall
```bash
pip uninstall kimsfinance
pip install --no-cache-dir kimsfinance
```

### Check package info
```bash
pip show kimsfinance
```

### List installed files
```bash
pip show -f kimsfinance
```

---

## Full Documentation

- **Complete Checklist**: `/home/kim/Documents/Github/kimsfinance/docs/PYPI_RELEASE_CHECKLIST.md`
- **Verification Report**: `/home/kim/Documents/Github/kimsfinance/docs/PYPI_VERIFICATION_REPORT.md`
