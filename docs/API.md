# API Reference

**Status**: Documentation in progress

This comprehensive API reference is currently being written and will be available soon.

---

## In the Meantime

### Quick Start

See the [README.md](../README.md) for a complete quick start guide with examples.

### Function Reference

For detailed API documentation, refer to:

1. **Function Docstrings** - All functions have comprehensive docstrings:
   ```python
   from kimsfinance.api import plot
   help(plot.render)  # View full API documentation
   ```

2. **Example Scripts** - Working examples in `scripts/`:
   - `scripts/demo_tick_charts.py` - Tick chart examples
   - `scripts/generate_tier1_indicators.py` - Indicator examples
   - `scripts/regenerate_all_samples.py` - Batch rendering

3. **Test Suite** - 329+ tests showing usage patterns:
   - `tests/test_api_native_routing.py` - API usage examples
   - `tests/test_renderer_*.py` - Renderer-specific examples
   - `tests/test_tick_aggregations.py` - Aggregation examples

---

## Core API Structure

### Main Entry Point

```python
from kimsfinance.api import plot

# Render chart from DataFrame
plot.render(
    df,
    chart_type='ohlc',  # ohlc, line, hollow, renko, pnf
    output_path='chart.webp',
    **options
)
```

### Key Modules

- **`kimsfinance.api.plot`** - Main plotting interface
- **`kimsfinance.ops.aggregations`** - Data aggregation functions
- **`kimsfinance.plotting.renderer`** - Low-level rendering engine

---

## Coming Soon

This API reference will include:

- ✅ Complete function signatures with type hints
- ✅ Parameter descriptions and defaults
- ✅ Return value specifications
- ✅ Usage examples for every function
- ✅ Error handling guidelines
- ✅ Performance optimization tips
- ✅ GPU acceleration configuration

---

**Last Updated**: 2025-10-22
**Status**: Under development
