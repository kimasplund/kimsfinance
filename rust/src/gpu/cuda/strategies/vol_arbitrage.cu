//! Volatility Arbitrage Strategy Kernel
//!
//! Identifies and exploits mispricing between implied volatility (IV) and
//! historical volatility (HV) with immediate delta hedging.
//!
//! # Strategy Logic
//!
//! **Buy Signal** (Long Volatility):
//! - IV < HV - vol_threshold (options are cheap)
//! - Expected edge: (HV - IV) > min_edge
//! - Buy options, immediately delta hedge
//! - Profit from volatility mean reversion
//!
//! **Sell Signal** (Short Volatility):
//! - IV > HV + vol_threshold (options are expensive)
//! - Expected edge: (IV - HV) > min_edge
//! - Sell options, immediately delta hedge
//! - Profit from IV contraction
//!
//! **Exit Signal**:
//! - IV-HV spread narrows below min_edge
//! - Close option position and unwind hedge
//!
//! # Performance Target
//!
//! - 1000 strategies × 500 candles: <10ms
//! - Memory-bound with coalesced access
//!
//! # Numerical Stability
//!
//! - Validates all inputs are finite and positive
//! - Calculates expected profit based on vol spread
//! - Handles both call and put options

/// Volatility arbitrage signal generation kernel
///
/// Grid: 2D (candles × strategies)
/// Block: (256, 4) threads
///
/// # Arguments
///
/// - underlying_prices: Spot prices [n_candles]
/// - option_prices: Option prices [n_strategies × n_candles]
/// - option_deltas: Option deltas from Greeks [n_strategies × n_candles]
/// - option_vegas: Option vegas from Greeks [n_strategies × n_candles]
/// - implied_vols: Implied volatilities [n_strategies × n_candles]
/// - historical_vols: Historical volatilities [n_strategies × n_candles]
/// - strategy_params: [vol_threshold, hedge_delta, min_edge] per strategy [n_strategies × 3]
/// - option_signals: Output option signals (1=buy, -1=sell, 0=hold) [n_strategies × n_candles]
/// - hedge_signals: Output hedge signals (quantity of underlying) [n_strategies × n_candles]
/// - expected_profit: Expected profit from vol spread [n_strategies × n_candles]
/// - vol_edge: Actual vol edge (HV - IV) [n_strategies × n_candles]
/// - n_strategies: Number of strategy configurations
/// - n_candles: Number of time points
extern "C" __global__ void vol_arbitrage_signals_kernel(
    const double* __restrict__ underlying_prices,
    const double* __restrict__ option_prices,
    const double* __restrict__ option_deltas,
    const double* __restrict__ option_vegas,
    const double* __restrict__ implied_vols,
    const double* __restrict__ historical_vols,
    const double* __restrict__ strategy_params,
    int8_t* __restrict__ option_signals,
    double* __restrict__ hedge_signals,
    double* __restrict__ expected_profit,
    double* __restrict__ vol_edge,
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
    double vol_threshold = strategy_params[param_idx] / 100.0; // Convert to decimal
    double hedge_delta_enabled = strategy_params[param_idx + 1]; // 0 or 1 (boolean)
    double min_edge = strategy_params[param_idx + 2] / 100.0;

    // Get market data
    double spot = underlying_prices[candle_idx];
    double option_price = option_prices[idx];
    double option_delta = option_deltas[idx];
    double option_vega = option_vegas[idx];
    double iv = implied_vols[idx];
    double hv = historical_vols[idx];

    // Validate inputs (avoid NaN propagation)
    bool valid_data = isfinite(spot) && isfinite(option_price) &&
                      isfinite(option_delta) && isfinite(option_vega) &&
                      isfinite(iv) && isfinite(hv) &&
                      spot > 0.0 && option_price >= 0.0 &&
                      iv > 0.0 && hv > 0.0 && option_vega >= 0.0;

    if (!valid_data) {
        option_signals[idx] = 0;
        hedge_signals[idx] = 0.0;
        expected_profit[idx] = 0.0;
        vol_edge[idx] = 0.0;
        return;
    }

    // Calculate volatility edge (HV - IV)
    // Positive edge: IV is cheap (buy volatility)
    // Negative edge: IV is expensive (sell volatility)
    double vol_spread = hv - iv;

    // Store vol edge for monitoring
    vol_edge[idx] = vol_spread;

    // Determine if there's a tradeable mispricing
    bool buy_vol_signal = (vol_spread > vol_threshold) && (vol_spread > min_edge);
    bool sell_vol_signal = (vol_spread < -vol_threshold) && (fabs(vol_spread) > min_edge);
    bool exit_signal = (fabs(vol_spread) < min_edge * 0.5); // Exit at half min_edge

    // Determine option signal
    int8_t option_signal = 0;
    if (buy_vol_signal && !exit_signal) {
        option_signal = 1; // Buy options (long volatility)
    } else if (sell_vol_signal && !exit_signal) {
        option_signal = -1; // Sell options (short volatility)
    } else if (exit_signal) {
        option_signal = 0; // Close position
    }

    // Calculate expected profit from volatility arbitrage
    // Profit ≈ Vega × (realized_vol_change)
    // If we expect IV to converge to HV, then:
    // Expected profit = Vega × vol_spread (for long vol)
    double expected_pnl = 0.0;
    if (option_signal == 1) {
        // Long volatility: profit if IV rises toward HV
        expected_pnl = option_vega * vol_spread * 100.0; // Vega is per 1% vol change
    } else if (option_signal == -1) {
        // Short volatility: profit if IV falls toward HV
        expected_pnl = option_vega * (-vol_spread) * 100.0;
    }

    // Calculate hedge signal (delta hedge if enabled)
    double hedge_signal = 0.0;
    if (hedge_delta_enabled > 0.5 && option_signal != 0) {
        // Delta hedge: offset option delta with underlying position
        // For long option with delta > 0: short underlying
        // For long option with delta < 0: long underlying
        hedge_signal = -option_signal * option_delta;
    }

    // Output signals
    option_signals[idx] = option_signal;
    hedge_signals[idx] = hedge_signal;
    expected_profit[idx] = expected_pnl;
}

/// Volatility arbitrage profit calculation kernel
///
/// Calculates realized P&L from volatility arbitrage positions.
/// This is used for backtesting and performance analysis.
///
/// Grid: 2D (candles × strategies)
/// Block: (256, 4) threads
///
/// # Arguments
///
/// - entry_prices: Option entry prices [n_strategies × n_candles]
/// - current_prices: Current option prices [n_strategies × n_candles]
/// - entry_iv: Implied vol at entry [n_strategies × n_candles]
/// - current_iv: Current implied vol [n_strategies × n_candles]
/// - option_positions: Option position sizes (signed) [n_strategies × n_candles]
/// - option_vegas: Option vegas [n_strategies × n_candles]
/// - realized_pnl: Output realized P&L [n_strategies × n_candles]
/// - vol_pnl_component: P&L attributable to vol change [n_strategies × n_candles]
/// - n_strategies: Number of strategy configurations
/// - n_candles: Number of time points
extern "C" __global__ void vol_arbitrage_pnl_kernel(
    const double* __restrict__ entry_prices,
    const double* __restrict__ current_prices,
    const double* __restrict__ entry_iv,
    const double* __restrict__ current_iv,
    const double* __restrict__ option_positions,
    const double* __restrict__ option_vegas,
    double* __restrict__ realized_pnl,
    double* __restrict__ vol_pnl_component,
    int n_strategies,
    int n_candles
) {
    int candle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strategy_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (candle_idx >= n_candles || strategy_idx >= n_strategies) return;

    int idx = strategy_idx * n_candles + candle_idx;

    // Get position data
    double entry_price = entry_prices[idx];
    double current_price = current_prices[idx];
    double iv_entry = entry_iv[idx];
    double iv_current = current_iv[idx];
    double position = option_positions[idx];
    double vega = option_vegas[idx];

    // Validate inputs
    bool valid = isfinite(entry_price) && isfinite(current_price) &&
                 isfinite(iv_entry) && isfinite(iv_current) &&
                 isfinite(position) && isfinite(vega);

    if (!valid || position == 0.0) {
        realized_pnl[idx] = 0.0;
        vol_pnl_component[idx] = 0.0;
        return;
    }

    // Calculate total P&L
    // P&L = position × (current_price - entry_price)
    // Positive position: profit when price increases
    // Negative position: profit when price decreases
    double total_pnl = position * (current_price - entry_price);

    // Calculate P&L component from volatility change
    // Vol P&L ≈ position × Vega × (current_iv - entry_iv) × 100
    // (Vega is per 1% vol change)
    double iv_change = (iv_current - iv_entry) * 100.0; // Convert to percentage points
    double vol_pnl = position * vega * iv_change;

    // Output
    realized_pnl[idx] = total_pnl;
    vol_pnl_component[idx] = vol_pnl;
}

/// Volatility arbitrage edge monitoring kernel
///
/// Monitors the volatility edge (IV vs HV spread) across multiple options
/// and identifies the best opportunities.
///
/// Grid: 2D (candles × strategies)
/// Block: (256, 4) threads
///
/// # Arguments
///
/// - implied_vols: Implied volatilities [n_strategies × n_candles]
/// - historical_vols: Historical volatilities [n_strategies × n_candles]
/// - option_prices: Option prices [n_strategies × n_candles]
/// - option_vegas: Option vegas [n_strategies × n_candles]
/// - vol_edge: Output vol edge (HV - IV) [n_strategies × n_candles]
/// - edge_quality: Quality score (|edge| × vega) [n_strategies × n_candles]
/// - n_strategies: Number of strategy configurations
/// - n_candles: Number of time points
extern "C" __global__ void vol_edge_monitor_kernel(
    const double* __restrict__ implied_vols,
    const double* __restrict__ historical_vols,
    const double* __restrict__ option_prices,
    const double* __restrict__ option_vegas,
    double* __restrict__ vol_edge,
    double* __restrict__ edge_quality,
    int n_strategies,
    int n_candles
) {
    int candle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strategy_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (candle_idx >= n_candles || strategy_idx >= n_strategies) return;

    int idx = strategy_idx * n_candles + candle_idx;

    double iv = implied_vols[idx];
    double hv = historical_vols[idx];
    double price = option_prices[idx];
    double vega = option_vegas[idx];

    // Validate inputs
    bool valid = isfinite(iv) && isfinite(hv) && isfinite(price) &&
                 isfinite(vega) && iv > 0.0 && hv > 0.0 && vega >= 0.0;

    if (!valid) {
        vol_edge[idx] = 0.0;
        edge_quality[idx] = 0.0;
        return;
    }

    // Calculate volatility edge
    double edge = hv - iv;

    // Calculate edge quality (edge × vega)
    // Higher quality = larger edge × more sensitive option
    double quality = fabs(edge) * vega * 100.0;

    vol_edge[idx] = edge;
    edge_quality[idx] = quality;
}
