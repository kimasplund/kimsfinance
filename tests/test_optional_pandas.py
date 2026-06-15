"""Regression test: pandas is an OPTIONAL dependency.

`import kimsfinance` and its core (Polars/NumPy) paths must work when pandas is
not installed. We assert this in a clean subprocess that blocks the `pandas`
import, so the check is hermetic and doesn't perturb the rest of the test run
(where pandas IS installed via the test extra).
"""

import subprocess
import sys
import textwrap


def test_import_kimsfinance_without_pandas():
    code = textwrap.dedent("""
        import sys, builtins
        _real = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "pandas" or name.startswith("pandas."):
                raise ImportError("pandas blocked for optional-dependency test")
            return _real(name, *args, **kwargs)

        builtins.__import__ = _blocked
        for _m in [m for m in sys.modules if m == "pandas" or m.startswith("pandas.")]:
            del sys.modules[_m]

        try:
            import pandas  # noqa: F401
            raise SystemExit("pandas was importable; test setup failed")
        except ImportError:
            pass

        import numpy as np
        import kimsfinance  # must import with no pandas

        # Core conversion + type guards must work (and treat pandas as absent).
        from kimsfinance.utils.array_utils import to_numpy_array
        from kimsfinance.core.types import (
            is_pandas_dataframe,
            is_pandas_series,
            is_array_like,
        )

        assert to_numpy_array(np.array([1.0, 2.0, 3.0])).tolist() == [1.0, 2.0, 3.0]
        assert is_pandas_dataframe(object()) is False
        assert is_pandas_series(object()) is False
        assert is_array_like([1, 2, 3]) is True
        print("OK_NO_PANDAS")
        """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert "OK_NO_PANDAS" in result.stdout, (
        "import kimsfinance failed without pandas:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
