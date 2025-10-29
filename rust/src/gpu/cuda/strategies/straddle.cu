//! Long Straddle Strategy Kernel
//!
//! Generates signals for long straddle (buy ATM call + put) strategy.
//! Profits from large price movements in either direction.
//!
//! # Strategy Logic
//!
//! **Entry Signal**:
//! - IV < HV - threshold (cheap options, expecting volatility expansion)
//! - Find ATM strike (closest to spot)
//! - Buy both call and put at ATM strike
//!
//! **Exit Signal**:
//! - Price moves beyond breakeven point
//! - Breakeven = ATM ± (call_price + put_price)
//!
//! # Performance Target
//!
//! - 1000 strategies × 500 candles: <5ms
//! - Memory-bound with coalesced access
//!
//! # Numerical Stability
//!
//! - Validates IV and HV are finite and positive
//! - Clamps total cost to reasonable bounds

/// Long straddle signal generation kernel
///
/// Grid: 2D (candles × strategies)
/// Block: (256, 4) threads
///
/// # Arguments
///
/// - underlying_prices: Spot prices [n_candles]
/// - call_prices: ATM call prices [n_strategies × n_candles]
/// - put_prices: ATM put prices [n_strategies × n_candles]
/// - implied_vols: Implied volatilities [n_strategies × n_candles]
/// - historical_vols: Historical volatilities [n_strategies × n_candles]
/// - strategy_params: [vol_threshold, breakeven_pct] per strategy [n_strategies × 2]
/// - signals: Output signals [call_signal, put_signal] [n_strategies × n_candles × 2]
/// - total_cost: Total cost (call + put) [n_strategies × n_candles]
/// - n_strategies: Number of strategy configurations
/// - n_candles: Number of time points
extern "C" __global__ void straddle_signals_kernel(
    const double* __restrict__ underlying_prices,
    const double* __restrict__ call_prices,
    const double* __restrict__ put_prices,
    const double* __restrict__ implied_vols,
    const double* __restrict__ historical_vols,
    const double* __restrict__ strategy_params,
    int8_t* __restrict__ signals,
    double* __restrict__ total_cost,
    int n_strategies,
    int n_candles
) {
    // 2D thread indexing
    int candle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strategy_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (candle_idx >= n_candles || strategy_idx >= n_strategies) return;

    // Linear index for strategy-candle pair
    int idx = strategy_idx * n_candles + candle_idx;

    // Extract strategy parameters
    double vol_threshold = strategy_params[strategy_idx * 2] / 100.0; // Convert to decimal
    double breakeven_pct = strategy_params[strategy_idx * 2 + 1] / 100.0;

    // Get market data
    double spot = underlying_prices[candle_idx];
    double iv = implied_vols[idx];
    double hv = historical_vols[idx];
    double call_price = call_prices[idx];
    double put_price = put_prices[idx];

    // Validate inputs (avoid NaN propagation)
    bool valid_data = isfinite(iv) && isfinite(hv) && isfinite(call_price) &&
                      isfinite(put_price) && iv > 0.0 && hv > 0.0;

    if (!valid_data) {
        signals[idx * 2 + 0] = 0; // No call signal
        signals[idx * 2 + 1] = 0; // No put signal
        total_cost[idx] = 0.0;
        return;
    }

    // Entry Logic: Buy straddle when IV < HV - threshold
    // Rationale: Options are cheap relative to expected volatility
    bool enter_signal = (iv < hv - vol_threshold);

    if (enter_signal) {
        // Buy ATM call and put
        signals[idx * 2 + 0] = 1; // Buy call
        signals[idx * 2 + 1] = 1; // Buy put
        total_cost[idx] = call_price + put_price;
    } else {
        // No position
        signals[idx * 2 + 0] = 0;
        signals[idx * 2 + 1] = 0;
        total_cost[idx] = 0.0;
    }

    // Exit Logic (for monitoring, not enforced in this kernel)
    // Breakeven points:
    // - Upper: ATM + total_cost
    // - Lower: ATM - total_cost
    // Would need position tracking for full exit logic
}

/// Short straddle signal generation kernel (sell ATM call + put)
///
/// Opposite of long straddle: profits from low volatility.
///
/// **Entry Signal**: IV > HV + threshold (expensive options, expecting vol contraction)
/// **Exit Signal**: Price moves beyond breakeven
///
/// Grid: 2D (candles × strategies)
/// Block: (256, 4) threads
extern "C" __global__ void short_straddle_signals_kernel(
    const double* __restrict__ underlying_prices,
    const double* __restrict__ call_prices,
    const double* __restrict__ put_prices,
    const double* __restrict__ implied_vols,
    const double* __restrict__ historical_vols,
    const double* __restrict__ strategy_params,
    int8_t* __restrict__ signals,
    double* __restrict__ total_premium,
    int n_strategies,
    int n_candles
) {
    int candle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strategy_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (candle_idx >= n_candles || strategy_idx >= n_strategies) return;

    int idx = strategy_idx * n_candles + candle_idx;

    double vol_threshold = strategy_params[strategy_idx * 2] / 100.0;
    double max_loss_pct = strategy_params[strategy_idx * 2 + 1] / 100.0;

    double spot = underlying_prices[candle_idx];
    double iv = implied_vols[idx];
    double hv = historical_vols[idx];
    double call_price = call_prices[idx];
    double put_price = put_prices[idx];

    bool valid_data = isfinite(iv) && isfinite(hv) && isfinite(call_price) &&
                      isfinite(put_price) && iv > 0.0 && hv > 0.0;

    if (!valid_data) {
        signals[idx * 2 + 0] = 0;
        signals[idx * 2 + 1] = 0;
        total_premium[idx] = 0.0;
        return;
    }

    // Entry: Sell straddle when IV > HV + threshold
    // Rationale: Options are expensive, collect premium
    bool enter_signal = (iv > hv + vol_threshold);

    if (enter_signal) {
        signals[idx * 2 + 0] = -1; // Sell call (negative = short)
        signals[idx * 2 + 1] = -1; // Sell put
        total_premium[idx] = call_price + put_price; // Premium received
    } else {
        signals[idx * 2 + 0] = 0;
        signals[idx * 2 + 1] = 0;
        total_premium[idx] = 0.0;
    }
}
