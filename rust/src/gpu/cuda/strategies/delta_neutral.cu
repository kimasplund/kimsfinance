//! Delta-Neutral Volatility Trading Strategy Kernel
//!
//! Generates signals for delta-neutral volatility trading via dynamic hedging.
//! Maintains portfolio delta near zero while capturing gamma/vega profit.
//!
//! # Strategy Logic
//!
//! **Entry Signal**:
//! - IV < HV - vol_threshold (cheap volatility)
//! - Buy options to gain vega exposure
//! - Immediately delta hedge with underlying
//!
//! **Rebalancing**:
//! - Monitor portfolio delta continuously
//! - When |delta| > rebalance_threshold, adjust underlying position
//! - Hedge delta = -option_delta × option_quantity
//!
//! **Exit Signal**:
//! - IV converges to HV (spread < vol_threshold)
//! - Close option position and unwind hedge
//!
//! # Performance Target
//!
//! - 1000 strategies × 500 candles: <10ms
//! - Memory-bound with coalesced access
//!
//! # Numerical Stability
//!
//! - Validates all inputs are finite
//! - Clamps delta to reasonable bounds [-1, 1]
//! - Prevents division by zero in hedge ratio calculation

/// Delta-neutral signal generation kernel
///
/// Grid: 2D (candles × strategies)
/// Block: (256, 4) threads
///
/// # Arguments
///
/// - underlying_prices: Spot prices [n_candles]
/// - option_prices: Option prices [n_strategies × n_candles]
/// - option_deltas: Option deltas from Greeks [n_strategies × n_candles]
/// - implied_vols: Implied volatilities [n_strategies × n_candles]
/// - historical_vols: Historical volatilities [n_strategies × n_candles]
/// - strategy_params: [delta_threshold, rebalance_threshold, vol_threshold] per strategy [n_strategies × 3]
/// - option_signals: Output option signals (1=buy, -1=sell, 0=hold) [n_strategies × n_candles]
/// - hedge_signals: Output hedge signals (quantity of underlying) [n_strategies × n_candles]
/// - portfolio_delta: Output portfolio delta after hedging [n_strategies × n_candles]
/// - n_strategies: Number of strategy configurations
/// - n_candles: Number of time points
extern "C" __global__ void delta_neutral_signals_kernel(
    const double* __restrict__ underlying_prices,
    const double* __restrict__ option_prices,
    const double* __restrict__ option_deltas,
    const double* __restrict__ implied_vols,
    const double* __restrict__ historical_vols,
    const double* __restrict__ strategy_params,
    char* __restrict__ option_signals,
    double* __restrict__ hedge_signals,
    double* __restrict__ portfolio_delta,
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
    int param_idx = strategy_idx * 3;
    double delta_threshold = strategy_params[param_idx] / 100.0; // Convert to decimal
    double rebalance_threshold = strategy_params[param_idx + 1] / 100.0;
    double vol_threshold = strategy_params[param_idx + 2] / 100.0;

    // Get market data
    double spot = underlying_prices[candle_idx];
    double option_price = option_prices[idx];
    double option_delta = option_deltas[idx];
    double iv = implied_vols[idx];
    double hv = historical_vols[idx];

    // Validate inputs (avoid NaN propagation)
    bool valid_data = isfinite(spot) && isfinite(option_price) && isfinite(option_delta) &&
                      isfinite(iv) && isfinite(hv) &&
                      spot > 0.0 && option_price >= 0.0 && iv > 0.0 && hv > 0.0;

    if (!valid_data) {
        option_signals[idx] = 0;
        hedge_signals[idx] = 0.0;
        portfolio_delta[idx] = 0.0;
        return;
    }

    // Clamp option delta to reasonable bounds [-1, 1]
    option_delta = fmax(-1.0, fmin(1.0, option_delta));

    // Entry Logic: Buy options when IV < HV - vol_threshold
    // This means volatility is cheap relative to historical levels
    bool enter_signal = (iv < hv - vol_threshold);

    // Exit Logic: Close position when IV-HV spread narrows
    bool exit_signal = (fabs(iv - hv) < vol_threshold * 0.5); // Exit at half threshold

    // Determine option signal
    char option_signal = 0;
    if (enter_signal && !exit_signal) {
        option_signal = 1; // Buy option (long vega exposure)
    } else if (exit_signal) {
        option_signal = -1; // Close position (sell if long)
    }

    // Calculate hedge signal (delta hedge the option position)
    // For long option: hedge = -delta × quantity
    // For call: delta > 0, so sell underlying to hedge
    // For put: delta < 0, so buy underlying to hedge
    double hedge_signal = 0.0;
    double portfolio_delta_value = 0.0;

    if (option_signal == 1) {
        // Entering long option position
        // Need to short underlying to hedge positive delta (for calls)
        // or long underlying to hedge negative delta (for puts)
        hedge_signal = -option_delta; // Hedge ratio
        portfolio_delta_value = option_delta + hedge_signal; // Should be near zero
    } else if (option_signal == -1) {
        // Exiting position - unwind hedge
        hedge_signal = option_delta; // Reverse the hedge
        portfolio_delta_value = 0.0;
    }

    // Rebalancing logic: if portfolio delta drifts beyond threshold
    // This would require tracking previous hedge positions (stateful)
    // For this kernel, we assume stateless per-candle evaluation
    // Real implementation would track positions across time

    // Check if rebalancing is needed (portfolio delta too large)
    if (fabs(portfolio_delta_value) > rebalance_threshold) {
        // Adjust hedge to bring portfolio delta back to zero
        // This is already handled in the hedge_signal calculation above
        // In a stateful system, this would adjust existing hedge
    }

    // Output signals
    option_signals[idx] = option_signal;
    hedge_signals[idx] = hedge_signal;
    portfolio_delta[idx] = portfolio_delta_value;
}

/// Delta-neutral rebalancing kernel (stateful version)
///
/// This kernel handles rebalancing of existing positions based on delta drift.
/// Requires tracking current positions across time.
///
/// Grid: 2D (candles × strategies)
/// Block: (256, 4) threads
///
/// # Arguments
///
/// - current_option_positions: Current option quantities [n_strategies × n_candles]
/// - current_hedge_positions: Current hedge positions [n_strategies × n_candles]
/// - option_deltas: Current option deltas [n_strategies × n_candles]
/// - strategy_params: [delta_threshold, rebalance_threshold, vol_threshold] per strategy [n_strategies × 3]
/// - rebalance_signals: Output rebalance signals (hedge adjustment) [n_strategies × n_candles]
/// - new_portfolio_delta: Output portfolio delta after rebalancing [n_strategies × n_candles]
/// - n_strategies: Number of strategy configurations
/// - n_candles: Number of time points
extern "C" __global__ void delta_neutral_rebalance_kernel(
    const double* __restrict__ current_option_positions,
    const double* __restrict__ current_hedge_positions,
    const double* __restrict__ option_deltas,
    const double* __restrict__ strategy_params,
    double* __restrict__ rebalance_signals,
    double* __restrict__ new_portfolio_delta,
    int n_strategies,
    int n_candles
) {
    int candle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strategy_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (candle_idx >= n_candles || strategy_idx >= n_strategies) return;

    int idx = strategy_idx * n_candles + candle_idx;

    // Extract rebalance threshold
    int param_idx = strategy_idx * 3;
    double rebalance_threshold = strategy_params[param_idx + 1] / 100.0;

    // Get current positions
    double option_qty = current_option_positions[idx];
    double hedge_qty = current_hedge_positions[idx];
    double option_delta = option_deltas[idx];

    // Validate inputs
    bool valid = isfinite(option_qty) && isfinite(hedge_qty) && isfinite(option_delta);
    if (!valid) {
        rebalance_signals[idx] = 0.0;
        new_portfolio_delta[idx] = 0.0;
        return;
    }

    // Calculate current portfolio delta
    // Portfolio delta = option_qty × option_delta + hedge_qty × 1.0
    // (hedge is in underlying, so delta = 1.0)
    double current_portfolio_delta = option_qty * option_delta + hedge_qty;

    // Check if rebalancing is needed
    if (fabs(current_portfolio_delta) > rebalance_threshold) {
        // Calculate hedge adjustment needed to bring delta to zero
        double hedge_adjustment = -current_portfolio_delta;
        rebalance_signals[idx] = hedge_adjustment;
        new_portfolio_delta[idx] = 0.0; // Target delta after rebalancing
    } else {
        // No rebalancing needed
        rebalance_signals[idx] = 0.0;
        new_portfolio_delta[idx] = current_portfolio_delta;
    }
}
