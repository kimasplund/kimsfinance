"""Regression test: ``import kimsfinance`` must succeed without reportlab.

Bug (pre-existing, fixed): ``kimsfinance/reporting/pdf_report.py`` used
``letter`` (a reportlab page-size constant) as a class-level default in
``ReportConfig``. ``letter`` is only bound inside a ``try/except ImportError``
guarding the reportlab import, so when reportlab was absent the class body
raised ``NameError`` at import time. ``NameError`` is not ``ImportError``, so
the ``except ImportError`` guard in ``kimsfinance/__init__.py`` did not catch
it and ``import kimsfinance`` failed outright.

CI installs only the ``[dev]`` extra (no reportlab), so this broke collection
of the entire test suite on every platform. This test reproduces that exact
condition in a subprocess (reportlab/matplotlib blocked from importing) and
asserts the package imports and ``ReportConfig`` is usable.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_import_kimsfinance_without_reportlab() -> None:
    code = textwrap.dedent("""
        import sys
        import importlib.abc

        BLOCK = {"reportlab", "matplotlib"}

        class _Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCK:
                    raise ImportError(f"blocked optional dependency: {name}")
                return None

        sys.meta_path.insert(0, _Blocker())

        import kimsfinance
        from kimsfinance.reporting import pdf_report

        assert pdf_report.REPORTLAB_AVAILABLE is False
        cfg = pdf_report.ReportConfig()
        assert cfg.page_size is not None  # falls back to the letter tuple
        print("OK", kimsfinance.__version__)
        """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "import kimsfinance failed when reportlab was absent:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "OK" in result.stdout
