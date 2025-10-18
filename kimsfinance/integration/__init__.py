"""Integration modules for mplfinance compatibility."""

from .adapter import activate, deactivate, is_active, configure, get_config
from .hooks import patch_plotting_functions, unpatch_plotting_functions

__all__ = [
    "activate",
    "deactivate",
    "is_active",
    "configure",
    "get_config",
    "patch_plotting_functions",
    "unpatch_plotting_functions",
]
