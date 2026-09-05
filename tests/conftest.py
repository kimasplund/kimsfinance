"""Shared pytest configuration for the kimsfinance test suite."""

import pytest

from _gpu import CORE_GPU_AVAILABLE, GPU_AVAILABLE, POLARS_GPU_AVAILABLE


def pytest_addoption(parser):
    """Add command-line options for baseline management."""
    parser.addoption(
        "--generate-baselines",
        action="store_true",
        default=False,
        help="Generate new baseline images",
    )
    parser.addoption(
        "--tolerance",
        type=float,
        default=0.01,
        help="Acceptable difference percentage (default: 1%%)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: needs a usable CUDA device; auto-skipped when tests/_gpu.py finds none "
        "(prefer the requires_gpu/requires_polars_gpu/requires_core_gpu markers from tests/_gpu.py)",
    )


def pytest_report_header(config):
    return (
        f"gpu probes: cupy device={GPU_AVAILABLE}, "
        f"polars gpu engine={POLARS_GPU_AVAILABLE}, "
        f"kimsfinance_core={CORE_GPU_AVAILABLE}"
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.gpu`` tests when no CUDA device is usable."""
    if GPU_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="no usable CUDA device (tests/_gpu.py probe)")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip)
