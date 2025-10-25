# Visual Regression Baseline Images

This directory contains baseline images for visual regression testing.

## Structure

```
baseline_images/
├── classic/       # Classic theme baselines
├── modern/        # Modern theme baselines
├── tradingview/   # TradingView theme baselines
└── light/         # Light theme baselines
```

## Generating Baselines

```bash
pytest tests/test_visual_regression.py --generate-baselines
```

## Running Visual Regression Tests

```bash
# Default tolerance (1%)
pytest tests/test_visual_regression.py

# Custom tolerance (0.5%)
pytest tests/test_visual_regression.py --tolerance=0.005
```

## Updating Baselines

When you intentionally change chart rendering:

1. Review current charts in `tests/visual_diffs/`
2. If changes are correct, regenerate baselines:
   ```bash
   pytest tests/test_visual_regression.py --generate-baselines
   ```
3. Commit updated baseline images

## Baseline Management

- **Commit baselines to git** - They're the source of truth
- **Review diffs carefully** - Visual regressions indicate bugs
- **Update selectively** - Only regenerate when intentional changes occur
