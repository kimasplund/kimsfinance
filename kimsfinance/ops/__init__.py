"""Operations modules for kimsfinance."""

from .moving_averages import (
    calculate_sma,
    calculate_ema,
    calculate_multiple_mas,
    from_pandas_series,
)

from .nan_ops import (
    nanmin_gpu,
    nanmax_gpu,
    nan_bounds,
    isnan_gpu,
    nan_indices,
    replace_nan,
    should_use_gpu_for_nan_ops,
)

from .batch import (
    calculate_indicators_batch,
)

# Phase 0: Abstraction Layer Components
from .rolling import (
    rolling_max,
    rolling_min,
    rolling_mean,
    rolling_std,
    rolling_sum,
    rolling_sum_optimized,
    ewm_mean,
)

from .indicator_utils import (
    true_range,
    gain_loss_separation,
    typical_price,
    money_flow,
    positive_negative_money_flow,
    directional_movement,
    percentage_change,
    median_price,
    weighted_close,
)

# Phase 1: Technical Indicators
from .stochastic import (
    calculate_stochastic,
    calculate_stochastic_rsi,
)

from .indicators import (
    calculate_atr,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_stochastic_oscillator,
    calculate_obv,
    calculate_vwap,
    calculate_vwap_anchored,
    calculate_williams_r,
    calculate_cci,
)

from .ichimoku import calculate_ichimoku
from .adx import calculate_adx
from .mfi import calculate_mfi
from .supertrend import calculate_supertrend
from .atr import calculate_atr as calculate_atr_standalone
from .picks import calculate_picks_momentum_ratio
from .swing import find_swing_points


__all__ = [
    # Moving averages
    "calculate_sma",
    "calculate_ema",
    "calculate_multiple_mas",
    "from_pandas_series",
    # NaN operations
    "nanmin_gpu",
    "nanmax_gpu",
    "nan_bounds",
    "isnan_gpu",
    "nan_indices",
    "replace_nan",
    "should_use_gpu_for_nan_ops",
    # Batch indicators
    "calculate_indicators_batch",
    # Rolling operations (Phase 0)
    "rolling_max",
    "rolling_min",
    "rolling_mean",
    "rolling_std",
    "rolling_sum",
    "rolling_sum_optimized",
    "ewm_mean",
    # Indicator utilities (Phase 0)
    "true_range",
    "gain_loss_separation",
    "typical_price",
    "money_flow",
    "positive_negative_money_flow",
    "directional_movement",
    "percentage_change",
    "median_price",
    "weighted_close",
    # Phase 1 indicators
    "calculate_stochastic",
    "calculate_stochastic_rsi",
    "calculate_ichimoku",
    "calculate_adx",
    "calculate_mfi",
    "calculate_supertrend",
    "calculate_picks_momentum_ratio",
    "find_swing_points",
    # Consolidated indicators
    "calculate_atr",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_stochastic_oscillator",
    "calculate_obv",
    "calculate_vwap",
    "calculate_vwap_anchored",
    "calculate_williams_r",
    "calculate_cci",
]
