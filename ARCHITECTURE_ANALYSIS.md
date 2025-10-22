# Architecture Analysis Report
**Project:** kimsfinance v0.1.0
**Analysis Date:** 2025-10-22
**Scope:** Post-PR #2 Major Refactoring

---

## Executive Summary

kimsfinance demonstrates **strong architectural foundations** with clear separation of concerns, consistent API design, and good performance patterns. The codebase is well-structured with 10 distinct modules, 65 files, and 329+ tests. However, several **medium-priority design issues** remain that could improve maintainability, extensibility, and testability.

**Overall Grade:** B+ (Good, with room for improvement)

**Key Strengths:**
- Clean module boundaries (core, ops, plotting, integration, config)
- Consistent use of Python 3.13+ type system
- Intelligent engine selection with fallback patterns
- Thread-safe state management

**Key Weaknesses:**
- Missing configuration management system (hardcoded values)
- Limited dependency injection (tight coupling to globals)
- No plugin architecture for custom indicators/themes
- Inconsistent error handling patterns
- API design has mplfinance compatibility baggage

---

## Severity Classification

### 🔴 Critical (0 issues)
No critical architectural issues found.

### 🟡 Medium (7 issues)
Issues that limit maintainability, extensibility, or testability but don't block functionality.

### 🟢 Low (5 issues)
Minor improvements that enhance code quality.

---

## Detailed Issues Analysis

### 🟡 MEDIUM-1: Missing Centralized Configuration System

**Location:** Throughout codebase (config/, core/engine.py, api/plot.py)

**Problem:**
- Configuration scattered across multiple locations:
  - `config/themes.py`: Hardcoded 4 themes
  - `config/chart_settings.py`: Hardcoded speed presets
  - `config/gpu_thresholds.py`: GPU operation thresholds
  - `integration/adapter.py`: Global `_config` dict with 5 hardcoded keys
  - `api/plot.py`: Magic numbers (width bounds, quality values)

**Evidence:**
```python
# api/plot.py - Hardcoded validation
if not (100 <= width <= 8192):
    raise ValueError(...)

# config/themes.py - Hardcoded themes
THEMES = {
    "classic": {"bg": "#000000", "up": "#00FF00", ...},
    "modern": {...},
}

# integration/adapter.py - Mutable global state
_config = {
    "default_engine": "auto",
    "gpu_min_rows": 10_000,  # Hardcoded threshold
    "strict_mode": False,
}
```

**Impact:**
- Cannot customize themes without modifying source code
- Cannot adjust performance thresholds per deployment
- Testing requires monkey-patching global state
- No environment-specific configuration (dev/prod)

**Recommendation:**

Implement a hierarchical configuration system:

```python
# kimsfinance/config/manager.py
from pathlib import Path
import json
from dataclasses import dataclass, field

@dataclass
class KimsFinanceConfig:
    """Centralized configuration with validation."""

    # Engine settings
    default_engine: Engine = "auto"
    gpu_min_rows: int = 10_000
    gpu_thresholds: dict[str, int] = field(default_factory=dict)

    # Rendering settings
    default_theme: str = "classic"
    image_width_bounds: tuple[int, int] = (100, 8192)
    image_height_bounds: tuple[int, int] = (100, 8192)

    # Performance settings
    enable_autotune: bool = True
    autotune_cache_ttl: int = 3600

    # Feature flags
    enable_svg_export: bool = True
    enable_parallel_rendering: bool = True

    @classmethod
    def from_file(cls, path: Path) -> "KimsFinanceConfig":
        """Load from JSON/TOML file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> "KimsFinanceConfig":
        """Load from environment variables (KIMSFINANCE_*)."""
        import os
        config = {}
        if val := os.getenv("KIMSFINANCE_DEFAULT_ENGINE"):
            config["default_engine"] = val
        # ... more env vars
        return cls(**config)

    def validate(self) -> None:
        """Validate configuration values."""
        if self.default_engine not in ("auto", "cpu", "gpu"):
            raise ValueError(f"Invalid engine: {self.default_engine}")
        if self.gpu_min_rows < 0:
            raise ValueError("gpu_min_rows must be non-negative")


class ConfigManager:
    """Global configuration manager with precedence hierarchy."""

    _instance: "ConfigManager | None" = None
    _config: KimsFinanceConfig

    def __init__(self):
        # Load with precedence: defaults < config file < env vars < API calls
        self._config = KimsFinanceConfig()

        # Try loading from ~/.kimsfinance/config.json
        user_config = Path.home() / ".kimsfinance" / "config.json"
        if user_config.exists():
            self._config = KimsFinanceConfig.from_file(user_config)

        # Override with environment variables
        env_config = KimsFinanceConfig.from_env()
        self._config = self._merge_configs(self._config, env_config)

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = ConfigManager()
        return cls._instance

    def get_config(self) -> KimsFinanceConfig:
        return self._config

    def update_config(self, **kwargs) -> None:
        """Update configuration at runtime."""
        for key, value in kwargs.items():
            setattr(self._config, key, value)
        self._config.validate()


# Usage:
config = ConfigManager.get_instance().get_config()
width_min, width_max = config.image_width_bounds
```

**Benefit:**
- ✅ User-configurable without source modifications
- ✅ Environment-specific configuration
- ✅ Testable (inject config objects)
- ✅ Type-safe with validation
- ✅ Clear precedence hierarchy

**Effort:** 2-3 days, Low risk

---

### 🟡 MEDIUM-2: Limited Dependency Injection (Tight Coupling)

**Location:** ops/, plotting/, integration/

**Problem:**
- Functions directly instantiate dependencies instead of receiving them
- Tight coupling to global singletons (EngineManager, ConfigManager)
- Hard to test with mocked dependencies
- Violates Dependency Inversion Principle

**Evidence:**
```python
# ops/indicators/rsi.py - Direct coupling to EngineManager
def calculate_rsi(prices, period=14, *, engine="auto"):
    exec_engine = EngineManager.select_engine(engine, ...)  # Global singleton
    result = df.lazy().select(...).collect(engine=exec_engine)
    return result["rsi"].to_numpy()

# plotting/pil_renderer.py - Direct theme access
from ..config.themes import THEMES  # Global import
def render_ohlcv_chart(..., theme="classic"):
    colors = THEMES[theme]  # Direct global access

# integration/hooks.py - Global state mutation
_config = {}  # Module-level global
def patch_plotting_functions(config):
    global _config
    _config = config.copy()  # Mutates global state
```

**Impact:**
- Cannot easily test with mock engines
- Cannot inject custom theme providers
- Hard to implement A/B testing of algorithms
- Limited to single configuration per process

**Recommendation:**

Implement lightweight dependency injection:

```python
# kimsfinance/core/context.py
from dataclasses import dataclass
from typing import Protocol

class ThemeProvider(Protocol):
    """Protocol for theme providers."""
    def get_theme(self, name: str) -> dict[str, str]: ...

class EngineSelector(Protocol):
    """Protocol for engine selection."""
    def select_engine(self, engine: Engine, operation: str, size: int) -> str: ...


@dataclass
class RenderContext:
    """Dependency injection container for rendering."""
    theme_provider: ThemeProvider
    engine_selector: EngineSelector
    config: KimsFinanceConfig

    @classmethod
    def default(cls) -> "RenderContext":
        """Create context with default dependencies."""
        return cls(
            theme_provider=DefaultThemeProvider(),
            engine_selector=EngineManager(),
            config=ConfigManager.get_instance().get_config(),
        )


# Usage in rendering functions:
def render_ohlcv_chart(
    ohlc_dict,
    volume_array,
    *,
    theme="classic",
    context: RenderContext | None = None,
    **kwargs,
):
    """Render with dependency injection."""
    ctx = context or RenderContext.default()

    # Use injected dependencies
    theme_colors = ctx.theme_provider.get_theme(theme)
    width_min, width_max = ctx.config.image_width_bounds

    # ... rendering logic


# Testing becomes trivial:
def test_render_with_custom_theme():
    class MockThemeProvider:
        def get_theme(self, name):
            return {"bg": "#CUSTOM", ...}

    ctx = RenderContext(
        theme_provider=MockThemeProvider(),
        engine_selector=MockEngineSelector(),
        config=test_config,
    )

    img = render_ohlcv_chart(..., context=ctx)
    # Assert custom theme applied
```

**Benefit:**
- ✅ Testable without mocks
- ✅ Extensible (custom providers)
- ✅ Supports A/B testing
- ✅ Multi-tenant configurations

**Effort:** 4-5 days, Medium risk (requires refactoring many functions)

---

### 🟡 MEDIUM-3: No Plugin Architecture for Extensibility

**Location:** ops/indicators/, config/themes.py, plotting/

**Problem:**
- No mechanism to register custom indicators without modifying source
- Themes hardcoded in source (cannot add themes externally)
- No chart type registry (cannot add custom chart types)
- Violates Open/Closed Principle

**Evidence:**
```python
# ops/indicators/__init__.py - Hardcoded indicator list
from .atr import calculate_atr
from .rsi import calculate_rsi
# ... 30+ imports

__all__ = ["calculate_atr", "calculate_rsi", ...]  # Hardcoded list

# config/themes.py - No extension mechanism
THEMES = {
    "classic": {...},
    "modern": {...},
    # Users cannot add "corporate" theme without editing source
}

# api/plot.py - Hardcoded chart type routing
if type == "candle":
    img = render_ohlcv_chart(...)
elif type == "ohlc":
    img = render_ohlc_bars(...)
# ... no extension point for custom types
```

**Impact:**
- Users cannot create custom indicators without forking
- No ecosystem for community-contributed indicators
- Cannot theme charts for corporate branding
- Limited to built-in chart types

**Recommendation:**

Implement plugin registry system:

```python
# kimsfinance/core/registry.py
from typing import Protocol, Callable
from collections.abc import Mapping

class IndicatorPlugin(Protocol):
    """Protocol for indicator plugins."""
    name: str
    description: str

    def calculate(self, *args, **kwargs) -> ArrayResult: ...
    def get_parameters(self) -> dict[str, type]: ...


class PluginRegistry:
    """Registry for plugins with validation."""

    def __init__(self):
        self._indicators: dict[str, IndicatorPlugin] = {}
        self._themes: dict[str, dict[str, str]] = {}
        self._chart_types: dict[str, Callable] = {}

    def register_indicator(self, plugin: IndicatorPlugin) -> None:
        """Register custom indicator."""
        if plugin.name in self._indicators:
            raise ValueError(f"Indicator '{plugin.name}' already registered")

        # Validate plugin implements protocol
        if not hasattr(plugin, "calculate"):
            raise TypeError("Indicator must implement calculate()")

        self._indicators[plugin.name] = plugin

    def register_theme(self, name: str, colors: dict[str, str]) -> None:
        """Register custom theme."""
        required_keys = {"bg", "up", "down", "grid"}
        if not required_keys.issubset(colors.keys()):
            raise ValueError(f"Theme must have keys: {required_keys}")

        self._themes[name] = colors

    def register_chart_type(self, name: str, renderer: Callable) -> None:
        """Register custom chart renderer."""
        self._chart_types[name] = renderer

    def get_indicator(self, name: str) -> IndicatorPlugin:
        if name not in self._indicators:
            raise KeyError(f"Unknown indicator: {name}")
        return self._indicators[name]

    def list_indicators(self) -> list[str]:
        return list(self._indicators.keys())


# Global registry instance
_registry = PluginRegistry()

def register_indicator(name: str):
    """Decorator for registering indicators."""
    def decorator(func):
        class WrappedPlugin:
            name = name
            calculate = staticmethod(func)
            def get_parameters(self):
                import inspect
                sig = inspect.signature(func)
                return {k: v.annotation for k, v in sig.parameters.items()}

        _registry.register_indicator(WrappedPlugin())
        return func
    return decorator


# Usage - Custom indicator in user code:
from kimsfinance import register_indicator

@register_indicator("custom_momentum")
def calculate_custom_momentum(prices: ArrayLike, period: int = 20) -> ArrayResult:
    """User's custom indicator."""
    return prices / np.roll(prices, period) - 1

# Usage - Custom theme:
from kimsfinance.core.registry import _registry

_registry.register_theme("corporate", {
    "bg": "#FFFFFF",
    "up": "#007ACC",
    "down": "#D13438",
    "grid": "#E0E0E0",
})

# Now available via API:
plot(df, theme="corporate")
```

**Benefit:**
- ✅ Open/Closed Principle compliance
- ✅ Community ecosystem potential
- ✅ Corporate customization
- ✅ A/B testing of indicators

**Effort:** 5-6 days, Medium risk

---

### 🟡 MEDIUM-4: Inconsistent Error Handling Strategy

**Location:** Throughout codebase (core/exceptions.py, ops/, api/)

**Problem:**
- Mix of error handling strategies:
  - Silent fallbacks (integration/hooks.py)
  - Exception propagation (core/engine.py)
  - ValueError vs custom exceptions
  - Inconsistent validation timing (some at entry, some deep in call stack)

**Evidence:**
```python
# integration/hooks.py - Silent fallback with warning
def _plot_mav_accelerated(...):
    try:
        # GPU calculation
        sma_results = calculate_sma(...)
    except Exception as e:
        warnings.warn(f"GPU failed, falling back: {e}")
        _original_functions["_plot_mav"](...)  # Silent fallback

# core/engine.py - Explicit exception
def select_engine(self, engine, ...):
    if engine == "gpu" and not self.check_gpu_available():
        raise GPUNotAvailableError()  # Explicit exception

# api/plot.py - ValueError for validation
def _validate_numeric_params(width, height, **kwargs):
    if not (100 <= width <= 8192):
        raise ValueError(f"width must be...")  # ValueError, not custom

# ops/indicators/rsi.py - ValueError deep in call stack
def calculate_rsi(prices, period=14, *, engine="auto"):
    # No validation here
    ...
    if len(prices_arr) < period + 1:  # Validation deep inside
        raise ValueError(...)
```

**Impact:**
- Users don't know if errors are recoverable
- Inconsistent error messages
- Hard to implement error monitoring
- Testing requires catching different exception types

**Recommendation:**

Standardize error handling strategy:

```python
# kimsfinance/core/exceptions.py - Enhanced hierarchy
class KimsFinanceError(Exception):
    """Base exception with context."""

    def __init__(self, message: str, *, recoverable: bool = False, context: dict | None = None):
        super().__init__(message)
        self.recoverable = recoverable
        self.context = context or {}


class ValidationError(KimsFinanceError):
    """Input validation failed."""

    def __init__(self, param_name: str, value: object, expected: str):
        super().__init__(
            f"Invalid {param_name}: {value!r}. Expected: {expected}",
            recoverable=False,
            context={"param": param_name, "value": value},
        )


class EngineUnavailableError(KimsFinanceError):
    """Requested engine not available (may fallback)."""

    def __init__(self, engine: str, reason: str, *, can_fallback: bool = True):
        super().__init__(
            f"{engine.upper()} engine unavailable: {reason}",
            recoverable=can_fallback,
            context={"engine": engine, "reason": reason},
        )


# Standardized error handler decorator
def with_error_context(operation: str):
    """Add error context and convert exceptions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KimsFinanceError:
                raise  # Already our exception
            except ValueError as e:
                # Convert stdlib exceptions to our hierarchy
                raise ValidationError(
                    param_name="unknown",
                    value=None,
                    expected=str(e),
                ) from e
            except Exception as e:
                # Wrap unexpected exceptions
                raise KimsFinanceError(
                    f"Unexpected error in {operation}: {e}",
                    recoverable=False,
                    context={"operation": operation, "error_type": type(e).__name__},
                ) from e
        return wrapper
    return decorator


# Usage:
@with_error_context("calculate_rsi")
def calculate_rsi(prices, period=14, *, engine="auto"):
    # Validate at entry point
    if period < 1:
        raise ValidationError("period", period, ">= 1")

    if len(prices) < period + 1:
        raise ValidationError("prices", f"length={len(prices)}", f">= {period + 1}")

    # ... calculation
```

**Benefit:**
- ✅ Consistent error handling
- ✅ Error monitoring integration
- ✅ Clear recovery guidance
- ✅ Better error messages

**Effort:** 3-4 days, Low risk

---

### 🟡 MEDIUM-5: API Design - mplfinance Compatibility Baggage

**Location:** api/plot.py, integration/

**Problem:**
- Primary API (`plot()`) designed for mplfinance compatibility
- Inconsistent parameter naming (mav vs windows, style vs theme)
- Fallback to mplfinance adds dependency and complexity
- Native capabilities hidden behind compatibility layer

**Evidence:**
```python
# api/plot.py - Compatibility layer
def plot(
    data,
    *,
    type="candle",         # mplfinance naming
    style="binance",       # Alias for theme
    mav=None,              # mplfinance parameter (not implemented)
    ema=None,              # mplfinance parameter (not implemented)
    ...
):
    """
    Native PIL-based financial plotting achieving 178x speedup vs mplfinance.

    This function uses kimsfinance's native PIL renderer instead of mplfinance...
    """
    # Check for unsupported features
    if mav is not None or ema is not None:
        warnings.warn("Using mplfinance fallback for mav/ema features...")
        return _plot_mplfinance(...)  # Falls back to slower library!

    # Map style aliases
    style = _map_style(style)  # "binance" -> "tradingview"
    theme = kwargs.get("theme", style)  # Confusion between style/theme
```

**Impact:**
- API confusion (style vs theme, mav vs windows)
- Performance degradation when using unsupported params
- Maintenance burden (supporting two APIs)
- Users unclear which API to use

**Recommendation:**

Provide separate, clean native API alongside compatibility layer:

```python
# kimsfinance/api/native.py - Clean native API
def render_chart(
    data: DataFrameInput,
    *,
    chart_type: ChartType = "candlestick",
    theme: str = "classic",
    indicators: list[Indicator] | None = None,
    output: str | None = None,
    **options,
) -> Image.Image | None:
    """
    Native kimsfinance rendering API (178x faster than mplfinance).

    This is the recommended API for new projects. For mplfinance
    compatibility, use `kimsfinance.compat.plot()`.

    Args:
        data: OHLCV DataFrame (pandas or polars)
        chart_type: Chart type ("candlestick", "ohlc", "line", "hollow", "renko", "pnf")
        theme: Visual theme ("classic", "modern", "light", "tradingview")
        indicators: List of indicator overlays (not fallback to mplfinance)
        output: Output file path (None = return Image)
        **options: Rendering options (width, height, volume, etc.)

    Returns:
        PIL Image if output=None, otherwise None (saves to file)

    Example:
        >>> from kimsfinance.api import render_chart
        >>> from kimsfinance.indicators import SMA, RSI
        >>>
        >>> img = render_chart(
        ...     df,
        ...     chart_type="candlestick",
        ...     theme="tradingview",
        ...     indicators=[SMA(20), SMA(50), RSI(14)],
        ...     width=1920,
        ...     height=1080,
        ... )
    """
    # Pure native implementation - no mplfinance fallback
    config = RenderConfig.from_kwargs(**options)
    config.validate()

    # Render indicators natively
    indicator_overlays = []
    if indicators:
        for ind in indicators:
            result = ind.calculate(data)
            indicator_overlays.append(result)

    # Pure native rendering
    img = _render_native(data, chart_type, theme, indicator_overlays, config)

    if output:
        img.save(output)
        return None
    return img


# Keep compatibility layer separate
# kimsfinance/api/compat.py
def plot(data, *, type="candle", style="binance", mav=None, **kwargs):
    """
    mplfinance-compatible API (for migration).

    **Deprecated**: Use `render_chart()` for better performance
    and native indicator support.
    """
    warnings.warn(
        "plot() is compatibility layer. Use render_chart() for native API.",
        DeprecationWarning,
    )
    # ... existing implementation
```

**Benefit:**
- ✅ Clear API boundaries
- ✅ No fallback complexity
- ✅ Better performance (no compatibility checks)
- ✅ Gradual migration path

**Effort:** 3-4 days, Medium risk

---

### 🟡 MEDIUM-6: Tight Coupling in Indicator Implementations

**Location:** ops/indicators/, ops/

**Problem:**
- Indicators directly import utility functions
- No abstraction layer for data operations
- Hard to swap implementations (e.g., Polars vs Pandas vs GPU)
- Code duplication in validation and conversion logic

**Evidence:**
```python
# ops/indicators/rsi.py - Direct Polars coupling
import polars as pl

def calculate_rsi(prices, period=14, *, engine="auto"):
    df = pl.DataFrame({"price": prices_arr})  # Direct Polars usage
    delta = pl.col("price").diff()            # Polars expressions
    df = df.with_columns(...)                 # Polars API
    result = df.lazy().select(...).collect(engine=exec_engine)
    return result["rsi"].to_numpy()

# ops/indicators/bollinger_bands.py - Same pattern repeated
def calculate_bollinger_bands(...):
    df = pl.DataFrame({"price": prices_arr})  # Duplicate conversion
    # ... same pattern


# ops/indicators/macd.py - Same pattern again
def calculate_macd(...):
    df = pl.DataFrame({"close": prices})      # Duplicate conversion
    # ... same pattern
```

**Impact:**
- Cannot swap data backend (locked to Polars)
- Code duplication (DataFrame conversion 50+ times)
- Hard to optimize across indicators
- Testing requires real Polars DataFrames

**Recommendation:**

Introduce data abstraction layer:

```python
# kimsfinance/core/dataframe.py
from typing import Protocol
from abc import abstractmethod

class DataFrameBackend(Protocol):
    """Protocol for DataFrame operations."""

    @abstractmethod
    def create_dataframe(self, data: dict[str, ArrayLike]) -> object: ...

    @abstractmethod
    def add_column(self, df: object, name: str, expr: object) -> object: ...

    @abstractmethod
    def rolling_mean(self, series: object, window: int) -> object: ...

    @abstractmethod
    def ewm_mean(self, series: object, span: int) -> object: ...

    @abstractmethod
    def to_array(self, series: object) -> ArrayResult: ...


class PolarsBackend:
    """Polars implementation."""

    def create_dataframe(self, data):
        import polars as pl
        return pl.DataFrame(data)

    def add_column(self, df, name, expr):
        return df.with_columns(**{name: expr})

    def rolling_mean(self, series, window):
        return series.rolling_mean(window_size=window)

    # ... more operations


class DataFrameOps:
    """High-level operations using backend."""

    def __init__(self, backend: DataFrameBackend | None = None):
        self.backend = backend or PolarsBackend()

    def calculate_rsi_dataframe(
        self,
        prices: ArrayLike,
        period: int,
    ) -> ArrayResult:
        """RSI using backend abstraction."""
        df = self.backend.create_dataframe({"price": prices})

        # Use backend operations
        delta = self.backend.diff(df["price"])
        gain = self.backend.where(delta > 0, delta, 0)
        loss = self.backend.where(delta < 0, -delta, 0)

        avg_gain = self.backend.ewm_mean(gain, span=2 * period - 1)
        avg_loss = self.backend.ewm_mean(loss, span=2 * period - 1)

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return self.backend.to_array(rsi)


# Refactored indicator:
def calculate_rsi(prices, period=14, *, engine="auto", backend=None):
    ops = DataFrameOps(backend=backend)
    return ops.calculate_rsi_dataframe(prices, period)
```

**Benefit:**
- ✅ Swappable backends (Polars, Pandas, cuDF)
- ✅ Reduced code duplication
- ✅ Easier testing (mock backend)
- ✅ Performance optimization opportunities

**Effort:** 6-8 days, High risk (requires refactoring all indicators)

---

### 🟡 MEDIUM-7: Missing Observability and Monitoring Hooks

**Location:** Throughout codebase

**Problem:**
- No structured logging
- Limited performance telemetry (only in integration layer)
- No hooks for monitoring (APM, metrics)
- Hard to debug performance issues in production

**Evidence:**
```python
# integration/adapter.py - Performance tracking exists but limited
class BoundedPerformanceStats:
    """Track performance but only in integration layer."""
    def record(self, engine_used, time_saved_ms):
        # Basic tracking but no extensibility
        self._aggregated_stats["total_calls"] += 1

# No observability in:
# - ops/indicators/*.py (no timing, no engine selection logging)
# - plotting/*.py (no render time tracking)
# - core/engine.py (no engine selection rationale logging)

# Only print statements and warnings:
if verbose:
    print("✓ kimsfinance activated!")  # No structured logging
```

**Impact:**
- Cannot integrate with observability platforms (Datadog, Prometheus)
- No insight into production performance
- Hard to debug "why GPU not used?"
- Cannot track SLA compliance

**Recommendation:**

Add observability hooks:

```python
# kimsfinance/core/observability.py
from typing import Protocol, Any
from dataclasses import dataclass
from contextlib import contextmanager
import time

class ObservabilityHook(Protocol):
    """Protocol for observability backends."""

    def record_metric(self, name: str, value: float, tags: dict[str, str]) -> None: ...
    def log_event(self, level: str, message: str, context: dict[str, Any]) -> None: ...
    def start_span(self, name: str) -> "Span": ...


@dataclass
class Span:
    """Tracing span."""
    name: str
    start_time: float
    tags: dict[str, str]

    def finish(self):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        # Report to observability backend


class ObservabilityManager:
    """Centralized observability management."""

    def __init__(self):
        self._hooks: list[ObservabilityHook] = []

    def register_hook(self, hook: ObservabilityHook):
        self._hooks.append(hook)

    @contextmanager
    def trace_operation(self, operation: str, **tags):
        """Trace operation with timing."""
        start = time.perf_counter()
        span = Span(operation, start, tags)

        try:
            yield span
        finally:
            duration_ms = (time.perf_counter() - start) * 1000

            # Notify all hooks
            for hook in self._hooks:
                hook.record_metric(
                    f"kimsfinance.{operation}.duration_ms",
                    duration_ms,
                    tags,
                )


# Global instance
_observability = ObservabilityManager()


# Usage in indicator:
def calculate_rsi(prices, period=14, *, engine="auto"):
    with _observability.trace_operation(
        "calculate_rsi",
        engine=engine,
        data_size=len(prices),
    ) as span:
        # Calculation
        exec_engine = EngineManager.select_engine(...)
        span.tags["engine_used"] = exec_engine

        result = ...
        return result


# Integration with Datadog (user code):
from datadog import statsd
from kimsfinance.core.observability import _observability

class DatadogHook:
    def record_metric(self, name, value, tags):
        tag_list = [f"{k}:{v}" for k, v in tags.items()]
        statsd.histogram(name, value, tags=tag_list)

_observability.register_hook(DatadogHook())
```

**Benefit:**
- ✅ Production observability
- ✅ Performance debugging
- ✅ SLA monitoring
- ✅ Pluggable backends

**Effort:** 4-5 days, Low risk

---

## Low Priority Issues

### 🟢 LOW-1: RenderConfig Not Fully Adopted

**Problem:** `RenderConfig` class exists but only used in one place. Most functions still use `**kwargs`.

**Location:** config/render_config.py (defined but underutilized)

**Recommendation:** Gradually migrate functions to accept `RenderConfig` instead of `**kwargs`.

**Effort:** 2 days

---

### 🟢 LOW-2: Inconsistent Module Naming

**Problem:** Mix of naming conventions:
- `ops/indicators/moving_averages.py` (snake_case)
- `ops/indicators/__init__.py` imports as `calculate_sma` (function style)
- `core/EngineManager` (class style)

**Recommendation:** Standardize on Python conventions:
- Modules: snake_case
- Classes: PascalCase
- Functions: snake_case
- Constants: UPPER_SNAKE_CASE

**Effort:** 1 day

---

### 🟢 LOW-3: Missing Type Aliases for Complex Types

**Problem:** Complex types repeated throughout codebase:
```python
dict[str, str | bool]  # Repeated 10+ times
tuple[ArrayResult, ArrayResult, ArrayResult]  # Repeated for MACD
```

**Recommendation:** Add type aliases in `core/types.py`:
```python
type EngineInfo = dict[str, str | bool]
type MACDResult = tuple[ArrayResult, ArrayResult, ArrayResult]
```

**Effort:** 1 day

---

### 🟢 LOW-4: Limited Documentation for Extension Points

**Problem:** No docs on how to:
- Add custom themes
- Create custom indicators
- Implement custom backends

**Recommendation:** Add docs/EXTENDING.md with examples.

**Effort:** 2 days

---

### 🟢 LOW-5: No CI/CD Pipeline Configuration

**Problem:** No .github/workflows/ or CI configuration found.

**Recommendation:** Add GitHub Actions for:
- Linting (ruff, mypy)
- Testing (pytest with coverage)
- Benchmarking (performance regression detection)

**Effort:** 2 days

---

## Design Pattern Recommendations

### Pattern 1: Strategy Pattern for Engine Selection

**Current:** Direct if/else in `EngineManager.select_engine()`

**Recommended:**
```python
class EngineSelectionStrategy(Protocol):
    def should_use_gpu(self, operation: str, data_size: int) -> bool: ...

class ThresholdStrategy:
    def should_use_gpu(self, operation, data_size):
        threshold = GPU_CROSSOVER_THRESHOLDS.get(operation, 100_000)
        return data_size >= threshold

class MLBasedStrategy:
    """Machine learning based selection."""
    def should_use_gpu(self, operation, data_size):
        # Predict based on historical performance
        return self.model.predict(operation, data_size)
```

---

### Pattern 2: Factory Pattern for Renderers

**Current:** Large if/elif chain in `plot()`

**Recommended:**
```python
class RendererFactory:
    _renderers: dict[str, Callable] = {
        "candle": render_candlestick,
        "ohlc": render_ohlc_bars,
        "line": render_line_chart,
        # Extensible!
    }

    @classmethod
    def register_renderer(cls, chart_type: str, renderer: Callable):
        cls._renderers[chart_type] = renderer

    @classmethod
    def get_renderer(cls, chart_type: str) -> Callable:
        if chart_type not in cls._renderers:
            raise ValueError(f"Unknown chart type: {chart_type}")
        return cls._renderers[chart_type]
```

---

### Pattern 3: Builder Pattern for Complex Configurations

**Current:** Functions with 15+ parameters

**Recommended:**
```python
class ChartBuilder:
    def __init__(self, data: DataFrameInput):
        self.data = data
        self.config = RenderConfig()
        self.indicators = []

    def with_theme(self, theme: str) -> "ChartBuilder":
        self.config.theme = theme
        return self

    def with_size(self, width: int, height: int) -> "ChartBuilder":
        self.config.width = width
        self.config.height = height
        return self

    def add_indicator(self, indicator: Indicator) -> "ChartBuilder":
        self.indicators.append(indicator)
        return self

    def build(self) -> Image.Image:
        return render_chart(self.data, self.config, self.indicators)

# Usage:
img = (
    ChartBuilder(df)
    .with_theme("tradingview")
    .with_size(1920, 1080)
    .add_indicator(SMA(20))
    .add_indicator(RSI(14))
    .build()
)
```

---

## Refactoring Roadmap

### Phase 1: Foundation (2 weeks)
**Priority:** High
**Risk:** Low

1. Implement centralized configuration system (MEDIUM-1)
2. Standardize error handling (MEDIUM-4)
3. Add observability hooks (MEDIUM-7)

**Deliverables:**
- `config/manager.py` with `KimsFinanceConfig`
- Enhanced exception hierarchy with context
- Basic observability infrastructure

---

### Phase 2: Extensibility (3 weeks)
**Priority:** Medium
**Risk:** Medium

1. Implement plugin registry (MEDIUM-3)
2. Introduce dependency injection (MEDIUM-2)
3. Create clean native API (MEDIUM-5)

**Deliverables:**
- Plugin system for indicators/themes
- RenderContext for DI
- `api/native.py` with clean API

---

### Phase 3: Abstraction (4 weeks)
**Priority:** Medium
**Risk:** High

1. Data abstraction layer (MEDIUM-6)
2. Adopt RenderConfig throughout (LOW-1)
3. Apply design patterns (Strategy, Factory, Builder)

**Deliverables:**
- `core/dataframe.py` backend abstraction
- RenderConfig used in all renderers
- Refactored plotting layer

---

### Phase 4: Polish (1 week)
**Priority:** Low
**Risk:** Low

1. Standardize naming conventions (LOW-2)
2. Add type aliases (LOW-3)
3. Write extension documentation (LOW-4)
4. Setup CI/CD (LOW-5)

**Deliverables:**
- Consistent naming throughout
- docs/EXTENDING.md
- GitHub Actions workflows

---

## Testing Recommendations

### Current State
- ✅ 329+ tests with good coverage
- ✅ Unit tests for indicators
- ✅ Integration tests for rendering
- ✅ Performance benchmarks

### Gaps
- ❌ No architecture tests (check dependencies)
- ❌ Limited contract tests (protocols)
- ❌ No mutation testing
- ❌ No property-based testing

### Recommended Additions

```python
# tests/architecture/test_dependencies.py
def test_core_has_no_external_dependencies():
    """Ensure core/ doesn't depend on ops/ or plotting/."""
    import importlib
    import kimsfinance.core as core

    # Core should only depend on stdlib and typing libs
    for module in core.__all__:
        mod = importlib.import_module(f"kimsfinance.core.{module}")
        # Assert no imports from kimsfinance.ops or kimsfinance.plotting


# tests/contracts/test_protocols.py
def test_theme_provider_protocol():
    """Ensure all theme providers match protocol."""
    from kimsfinance.core.context import ThemeProvider
    from kimsfinance.config.themes import DefaultThemeProvider

    # Protocol check
    assert isinstance(DefaultThemeProvider(), ThemeProvider)


# tests/properties/test_indicator_properties.py
from hypothesis import given, strategies as st

@given(
    prices=st.lists(st.floats(min_value=1, max_value=1000), min_size=50, max_size=100),
    period=st.integers(min_value=5, max_value=20),
)
def test_rsi_properties(prices, period):
    """RSI should always be in 0-100 range."""
    rsi = calculate_rsi(prices, period)
    assert np.all((rsi >= 0) & (rsi <= 100) | np.isnan(rsi))
```

---

## Metrics & Monitoring

### Architectural Health Metrics

Track these metrics over time:

1. **Coupling Metrics:**
   - `coupling_ratio = external_imports / total_imports`
   - Target: < 0.3 (currently ~0.4)

2. **Abstraction Metrics:**
   - `abstraction_score = interfaces / concrete_classes`
   - Target: > 0.2 (currently ~0.05)

3. **Extensibility Metrics:**
   - `extension_points = len(registries) + len(protocols)`
   - Target: > 10 (currently 0)

4. **Configuration Metrics:**
   - `config_centralization = centralized_config / total_config`
   - Target: > 0.9 (currently ~0.3)

---

## Conclusion

kimsfinance has a **solid architectural foundation** with clear benefits:

✅ **Strengths:**
- Clean module boundaries
- Modern Python 3.13+ type system
- Thread-safe state management
- Intelligent engine selection
- Strong performance focus

⚠️ **Areas for Improvement:**
- Configuration management (scattered, hardcoded)
- Dependency injection (tight coupling to globals)
- Extensibility (no plugin system)
- Error handling (inconsistent patterns)
- API design (compatibility baggage)

**Recommended Priority:**
1. **Phase 1** (Foundation) - 2 weeks, high impact
2. **Phase 2** (Extensibility) - 3 weeks, medium impact
3. **Phase 3** (Abstraction) - 4 weeks, long-term benefit
4. **Phase 4** (Polish) - 1 week, quality of life

**Total Effort:** 10 weeks for full implementation

**Risk Mitigation:**
- Implement changes incrementally
- Maintain backward compatibility
- Add deprecation warnings for breaking changes
- Comprehensive testing at each phase

---

**Report Author:** Claude (Sonnet 4.5)
**Analysis Methodology:** Static code analysis, pattern recognition, architecture review
**Files Analyzed:** 65 Python files, 10 modules, 329+ tests
