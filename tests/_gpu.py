"""Device-based GPU availability probes shared by the test suite.

GPU-dependent tests must gate on one of the probes below (or the ``requires_*``
skip markers built from them), never on ``import cupy`` succeeding. On a machine
where the CUDA libraries are installed but no device is usable (driver needs a
reset, ``CUDA_VISIBLE_DEVICES=""``, a CPU-only container with the GPU wheels)
``import cupy`` works while every kernel launch raises, so an import-based gate
turns "skipped" into "failed". In CI, which installs no CUDA wheels at all, every
probe is False and the marked tests are skipped exactly as before.

Probes (each computed once at import time):

* ``GPU_AVAILABLE``        - cupy imports and reports at least one CUDA device.
                             Use for the cupy/cuDF code path
                             (``EngineManager.select_engine``).
* ``POLARS_GPU_AVAILABLE`` - a trivial ``LazyFrame.collect(engine="gpu")``
                             succeeds. Use for indicators that go through
                             ``EngineManager.select_polars_engine``.
* ``CORE_GPU_AVAILABLE``   - ``kimsfinance_core.gpu_available()`` is True, i.e.
                             the Rust crate can create a CUDA context. Use for
                             the Rust GPU batch-backtest bindings.

Usage::

    from _gpu import requires_polars_gpu

    @requires_polars_gpu
    def test_rsi_gpu_matches_cpu(): ...

``tests/`` is put on ``sys.path`` by pytest when it loads ``tests/conftest.py``
(default ``prepend`` import mode), so ``from _gpu import ...`` works from every
test module regardless of nesting.
"""

from __future__ import annotations

import pytest


def _probe_cupy_device() -> bool:
    try:
        import cupy

        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # ImportError, CUDARuntimeError, driver errors, ...
        return False


def _probe_polars_gpu() -> bool:
    try:
        import polars as pl

        pl.LazyFrame({"probe": [1, 2, 3]}).collect(engine="gpu")
        return True
    except Exception:
        return False


def _probe_core_gpu() -> bool:
    try:
        import kimsfinance_core

        return bool(kimsfinance_core.gpu_available())
    except Exception:
        return False


GPU_AVAILABLE: bool = _probe_cupy_device()
POLARS_GPU_AVAILABLE: bool = _probe_polars_gpu()
CORE_GPU_AVAILABLE: bool = _probe_core_gpu()

requires_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE, reason="no usable CUDA device (cupy reports 0 devices)"
)
requires_polars_gpu = pytest.mark.skipif(
    not POLARS_GPU_AVAILABLE, reason="Polars GPU engine not available (no usable CUDA device)"
)
requires_core_gpu = pytest.mark.skipif(
    not CORE_GPU_AVAILABLE, reason="kimsfinance_core reports no usable CUDA device"
)

__all__ = [
    "GPU_AVAILABLE",
    "POLARS_GPU_AVAILABLE",
    "CORE_GPU_AVAILABLE",
    "requires_gpu",
    "requires_polars_gpu",
    "requires_core_gpu",
]
