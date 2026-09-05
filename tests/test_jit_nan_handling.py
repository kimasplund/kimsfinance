"""
Regression tests: NaN/inf handling in Numba-JIT kernels
=======================================================

``@njit(fastmath=True)`` enables *every* LLVM fast-math flag, including
``nnan`` and ``ninf``.  Those two let the optimiser assume NaN/inf never occur,
so ``np.isnan(x)`` folds to ``False`` and NaN-aware comparisons are rewritten.
With Numba installed this silently turned ``replace_nan``,
``fill_nan_forward_jit``/``fill_nan_backward_jit`` into no-ops and made
``_wilder_smoothing`` return all-NaN for any input containing a NaN.  CI never
installs Numba, so the pure-Python fallback masked the bug there.

The fix is the shared ``FASTMATH_SAFE`` flag set (everything except
``nnan``/``ninf``).  These tests exercise the *public* entry points on inputs
containing NaN and inf and assert the NaN-aware behaviour.  When Numba is not
installed the same functions run as plain Python; the tests then still pass
but the ``requires_numba`` cases are skipped because they exist to pin the
compiled behaviour.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from kimsfinance.utils.array_utils import (
    FASTMATH_SAFE,
    clip_array_jit,
    fill_nan_backward_jit,
    fill_nan_forward_jit,
    normalize_array_jit,
)
from kimsfinance.ops.nan_ops import replace_nan
from kimsfinance.ops import indicator_utils, rolling
from kimsfinance.ops.indicators.aroon import _calculate_aroon_cpu, calculate_aroon
from kimsfinance.ops.indicators.elder_ray import calculate_elder_ray
from kimsfinance.ops.indicators.roc import calculate_roc

try:
    import numba  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

requires_numba = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")

NAN = np.nan
INF = np.inf


def assert_equal_nan(actual: np.ndarray, expected, msg: str = "") -> None:
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float64), equal_nan=True)


# ============================================================================
# The shared flag set
# ============================================================================


class TestFastmathSafeFlags:
    def test_is_plain_set_without_nan_inf_flags(self):
        """Numba only accepts a plain ``set``; it must not contain nnan/ninf/fast."""
        assert type(FASTMATH_SAFE) is set
        assert FASTMATH_SAFE.isdisjoint({"nnan", "ninf", "fast"})
        assert FASTMATH_SAFE == {"nsz", "arcp", "contract", "afn", "reassoc"}

    @requires_numba
    def test_numba_accepts_flag_set(self):
        from numba import njit

        @njit(fastmath=FASTMATH_SAFE)
        def count_nan(arr):
            n = 0
            for i in range(len(arr)):
                if np.isnan(arr[i]):
                    n += 1
            return n

        assert count_nan(np.array([1.0, NAN, NAN, 4.0])) == 2

    def test_full_fastmath_only_on_whitelisted_kernels(self):
        """
        Guard: every ``fastmath=True`` site left in the package must be one of
        the kernels audited as pure arithmetic over finite data.  Anything that
        inspects or must preserve NaN/inf has to use ``FASTMATH_SAFE``.
        """
        allowed = {
            "array_diff_jit",  # utils/array_utils.py
            "_rolling_mean_jit",  # ops/rolling.py (guarded by a has_nan check)
            "_calculate_wma_jit",  # ops/indicators/moving_averages.py
            "_calculate_vwma_jit",  # ops/indicators/moving_averages.py
            "_calculate_ohlc_bar_coordinates",  # plotting/pil_renderer.py
            "_calculate_line_chart_coordinates",
            "_calculate_grid_coordinates",
            "_calculate_coordinates_jit",
        }
        package_root = Path(__file__).resolve().parent.parent / "kimsfinance"
        found: dict[str, str] = {}
        for path in package_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                for dec in node.decorator_list:
                    if not isinstance(dec, ast.Call):
                        continue
                    for kw in dec.keywords:
                        if (
                            kw.arg == "fastmath"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            found[node.name] = str(path.relative_to(package_root))
        unexpected = {name: where for name, where in found.items() if name not in allowed}
        assert not unexpected, (
            "New fastmath=True kernels must be audited for NaN/inf handling; "
            f"use FASTMATH_SAFE unless they are pure finite arithmetic: {unexpected}"
        )


# ============================================================================
# nan_ops.replace_nan
# ============================================================================


class TestReplaceNan:
    @pytest.mark.parametrize("value", [0.0, -1.5, 99.0])
    def test_replaces_every_nan(self, value):
        arr = np.array([1.0, NAN, 3.0, NAN, 5.0])
        result = replace_nan(arr, value, engine="cpu")
        assert_equal_nan(result, [1.0, value, 3.0, value, 5.0])
        assert not np.isnan(result).any()

    def test_input_not_mutated(self):
        arr = np.array([1.0, NAN, 3.0])
        replace_nan(arr, 0.0, engine="cpu")
        assert np.isnan(arr[1])

    def test_inf_is_not_nan(self):
        arr = np.array([INF, NAN, -INF, 2.0])
        assert_equal_nan(replace_nan(arr, 0.0, engine="cpu"), [INF, 0.0, -INF, 2.0])

    def test_all_nan_and_no_nan(self):
        assert_equal_nan(replace_nan(np.full(5, NAN), 7.0, engine="cpu"), [7.0] * 5)
        clean = np.array([1.0, 2.0, 3.0])
        assert_equal_nan(replace_nan(clean, 7.0, engine="cpu"), clean)

    def test_large_array(self):
        rng = np.random.default_rng(0)
        arr = rng.standard_normal(50_000)
        mask = rng.random(arr.size) < 0.1
        arr[mask] = NAN
        result = replace_nan(arr, 0.0, engine="cpu")
        assert not np.isnan(result).any()
        assert np.array_equal(result[mask], np.zeros(mask.sum()))
        assert np.array_equal(result[~mask], arr[~mask])


# ============================================================================
# utils.array_utils fill / clip / normalize
# ============================================================================


class TestFillNan:
    def test_forward_fill(self):
        arr = np.array([NAN, 1.0, NAN, NAN, 4.0, NAN])
        assert_equal_nan(fill_nan_forward_jit(arr), [NAN, 1.0, 1.0, 1.0, 4.0, 4.0])

    def test_backward_fill(self):
        arr = np.array([NAN, 1.0, NAN, NAN, 4.0, NAN])
        assert_equal_nan(fill_nan_backward_jit(arr), [1.0, 1.0, 4.0, 4.0, 4.0, NAN])

    def test_inf_is_a_valid_fill_value(self):
        arr = np.array([INF, NAN, -INF, NAN])
        assert_equal_nan(fill_nan_forward_jit(arr), [INF, INF, -INF, -INF])
        assert_equal_nan(fill_nan_backward_jit(arr), [INF, -INF, -INF, NAN])

    def test_all_nan_stays_nan(self):
        arr = np.full(4, NAN)
        assert np.isnan(fill_nan_forward_jit(arr)).all()
        assert np.isnan(fill_nan_backward_jit(arr)).all()

    def test_no_nan_is_identity_and_not_mutated(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert_equal_nan(fill_nan_forward_jit(arr), arr)
        assert_equal_nan(fill_nan_backward_jit(arr), arr)
        src = np.array([NAN, 1.0, NAN])
        fill_nan_forward_jit(src)
        fill_nan_backward_jit(src)
        assert_equal_nan(src, [NAN, 1.0, NAN])


class TestClipAndNormalize:
    def test_clip_preserves_nan_and_clips_inf(self):
        arr = np.array([NAN, -1.0, 5.0, 2.0, INF, -INF])
        assert_equal_nan(clip_array_jit(arr, 0.0, 3.0), [NAN, 0.0, 3.0, 2.0, 3.0, 0.0])

    def test_normalize_propagates_nan(self):
        arr = np.array([NAN, 1.0, 2.0])
        assert_equal_nan(normalize_array_jit(arr, 0.0, 2.0), [NAN, 0.5, 1.0])


# ============================================================================
# indicator_utils._wilder_smoothing (ATR / ADX building block)
# ============================================================================


def _wilder_reference(arr: np.ndarray, period: int) -> np.ndarray:
    """Pure NumPy version of the documented semantics: nanmean seed, carry across gaps."""
    n = len(arr)
    out = np.full(n, NAN)
    if n < period:
        return out
    seed_window = arr[:period]
    if np.isnan(seed_window).all():
        return out
    out[period - 1] = np.nanmean(seed_window)
    alpha = 1.0 / period
    for i in range(period, n):
        out[i] = out[i - 1] if np.isnan(arr[i]) else alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


WILDER_CASES = {
    "clean": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    "leading_nan": np.array([NAN, NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    "mid_gap": np.array([1.0, 2.0, 3.0, 4.0, NAN, NAN, 7.0, 8.0, 9.0]),
    "all_nan": np.full(8, NAN),
    "flat": np.full(8, 5.0),
    "too_short": np.array([1.0, NAN]),
}


class TestWilderSmoothing:
    @pytest.mark.parametrize("name", sorted(WILDER_CASES))
    def test_matches_reference(self, name):
        arr = WILDER_CASES[name]
        with np.errstate(all="ignore"):
            result = indicator_utils._wilder_smoothing(arr, 3)
        assert_equal_nan(result, _wilder_reference(arr, 3))

    @requires_numba
    @pytest.mark.parametrize("name", sorted(WILDER_CASES))
    def test_jit_path_matches_generic_path(self, name, monkeypatch):
        """The compiled kernel must agree with the generic NumPy/CuPy loop."""
        arr = WILDER_CASES[name]
        jit_result = indicator_utils._wilder_smoothing(arr, 3)
        monkeypatch.setattr(indicator_utils, "NUMBA_AVAILABLE", False)
        with pytest.warns(RuntimeWarning) if name == "all_nan" else np.errstate(all="ignore"):
            generic_result = indicator_utils._wilder_smoothing(arr, 3)
        assert_equal_nan(jit_result, generic_result)

    @requires_numba
    def test_jit_kernel_directly(self):
        """Regression: with nnan the kernel returned all-NaN for any NaN input."""
        arr = WILDER_CASES["leading_nan"]
        result = indicator_utils._wilder_smoothing_jit(arr, 3, 1.0 / 3)
        assert np.isnan(result[:2]).all()
        assert np.isfinite(result[2:]).all()
        assert result[2] == pytest.approx(1.0)


# ============================================================================
# rolling helpers
# ============================================================================


class TestRollingNanPropagation:
    x = np.array([NAN, NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_rolling_max_min_propagate_nan_through_window(self):
        assert_equal_nan(rolling.rolling_max(self.x, 3), [NAN, NAN, NAN, NAN, 3.0, 4.0, 5.0, 6.0])
        assert_equal_nan(rolling.rolling_min(self.x, 3), [NAN, NAN, NAN, NAN, 1.0, 2.0, 3.0, 4.0])

    def test_rolling_max_min_handle_inf(self):
        arr = np.array([1.0, INF, 2.0, -INF, 3.0])
        assert_equal_nan(rolling.rolling_max(arr, 2), [NAN, INF, INF, 2.0, 3.0])
        assert_equal_nan(rolling.rolling_min(arr, 2), [NAN, 1.0, 2.0, -INF, -INF])

    def test_rolling_std_propagates_nan(self):
        assert_equal_nan(rolling.rolling_std(self.x, 3), [NAN, NAN, NAN, NAN, 1.0, 1.0, 1.0, 1.0])

    @pytest.mark.parametrize("ddof", [0, 1])
    def test_rolling_std_matches_numpy(self, ddof):
        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 8.0, 7.5])
        expected = [NAN, NAN] + [np.std(arr[i : i + 3], ddof=ddof) for i in range(len(arr) - 2)]
        assert_equal_nan(rolling.rolling_std(arr, 3, ddof=ddof), expected)

    def test_ewm_mean_propagates_nan_from_seed_window(self):
        # Documented behaviour of both code paths: a NaN inside the seed window
        # poisons the recurrence.  Pinned so JIT and generic paths stay aligned.
        assert np.isnan(rolling.ewm_mean(self.x, 3)).all()
        clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert_equal_nan(rolling.ewm_mean(clean, 3), [NAN, NAN, 2.0, 2.6666666667, 3.4444444444])

    @requires_numba
    @pytest.mark.parametrize("func", ["rolling_max", "rolling_min", "rolling_std", "ewm_mean"])
    def test_jit_matches_generic_path(self, func, monkeypatch):
        fn = getattr(rolling, func)
        jit_result = fn(self.x, 3)
        monkeypatch.setattr(rolling, "NUMBA_AVAILABLE", False)
        assert_equal_nan(jit_result, fn(self.x, 3))


# ============================================================================
# Indicators whose JIT kernels compare or subtract possibly-NaN values
# ============================================================================


class TestIndicatorKernels:
    def test_aroon_with_nan_matches_vectorized_cpu_path(self):
        highs = np.array([1.0, 2.0, NAN, 4.0, 3.0, 2.0, 5.0])
        lows = highs - 1.0
        up, down = calculate_aroon(highs, lows, period=3, engine="cpu")
        ref_up, ref_down = _calculate_aroon_cpu(highs, lows, 3)
        assert_equal_nan(up, ref_up)
        assert_equal_nan(down, ref_down)

    def test_roc_with_nan(self):
        prices = np.array([1.0, NAN, 2.0, 4.0, 5.0])
        result = calculate_roc(prices, period=1, engine="cpu")
        assert_equal_nan(result, [NAN, NAN, NAN, 100.0, 25.0])

    def test_elder_ray_keeps_ema_warmup_nan(self):
        highs = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        lows = highs - 1.0
        closes = highs - 0.5
        bull, bear = calculate_elder_ray(highs, lows, closes, period=3, engine="cpu")
        assert np.isnan(bull[:2]).all() and np.isnan(bear[:2]).all()
        assert np.isfinite(bull[2:]).all() and np.isfinite(bear[2:]).all()
        np.testing.assert_allclose((bull - bear)[2:], np.ones(4))
