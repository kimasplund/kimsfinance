"""Core modules for kimsfinance."""

from .types import (
    ArrayResult,
    SeriesResult,
    DataFrameResult,
    DataFrameInput,
    ArrayLike,
    WindowSize,
    ShiftPeriods,
    Engine,
    BoundsResult,
    LinearFitResult,
    MACDResult,
    EngineConfig,
)

from .exceptions import (
    KimsFinanceError,
    GPUNotAvailableError,
    DataValidationError,
    EngineError,
    OperationNotSupportedError,
    ConfigurationError,
)

from .engine import EngineManager, with_engine_fallback

from .decorators import (
    gpu_accelerated,
    get_array_module,
    to_gpu,
    to_cpu,
)

from .utils import to_numpy_array

__all__ = [
    # Types
    "ArrayResult",
    "SeriesResult",
    "DataFrameResult",
    "DataFrameInput",
    "ArrayLike",
    "WindowSize",
    "ShiftPeriods",
    "Engine",
    "BoundsResult",
    "LinearFitResult",
    "MACDResult",
    "EngineConfig",
    # Exceptions
    "KimsFinanceError",
    "GPUNotAvailableError",
    "DataValidationError",
    "EngineError",
    "OperationNotSupportedError",
    "ConfigurationError",
    # Engine
    "EngineManager",
    "with_engine_fallback",
    # Decorators
    "gpu_accelerated",
    "get_array_module",
    "to_gpu",
    "to_cpu",
    # Utils
    "to_numpy_array",
]
