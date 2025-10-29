//! Covered Call Strategy Kernel
//!
//! Generates signals for covered call strategy: own stock + sell OTM call.
//! Income-generating strategy that caps upside but provides premium income.
//!
//! # Strategy Logic
//!
//! **Entry Signal**:
//! - Long 100 shares of underlying
//! - Sell OTM call at strike = spot * (1 + strike_offset_pct/100)
//! - Only enter if call_premium >= spot * min_premium_pct/100
//!
//! **Exit Signal**:
//! - Expiration: Collect premium if spot < strike (max profit)
//! - Early assignment: If spot > strike (called away)
//!
//! # Performance Target
//!
//! - 1000 strategies × 500 candles: <10ms
//! - Memory-bound with coalesced access
//!
//! # Numerical Stability
//!
//! - Validates all prices are finite and positive
//! - Prevents negative premiums
//! - Checks strike > spot (OTM call)

/// Covered call signal generation kernel
///
/// Grid: 2D (candles × strategies)
/// Block: (256, 4) threads
///
/// # Arguments
///
/// - underlying_prices: Spot prices [n_candles]
/// - call_prices: OTM call prices [n_strategies × n_candles]
/// - strikes: Call strike prices [n_strategies × n_candles]
/// - strategy_params: [strike_offset_pct, min_premium_pct] per strategy [n_strategies × 2]
/// - stock_signals: Output stock signals (1=buy, 0=hold) [n_strategies × n_candles]
/// - call_signals: Output call signals (-1=sell, 0=hold) [n_strategies × n_candles]
/// - premium_collected: Premium received from selling call [n_strategies × n_candles]
/// - n_strategies: Number of strategy configurations
/// - n_candles: Number of time points
extern "C" __global__ void covered_call_signals_kernel(
    const double* __restrict__ underlying_prices,
    const double* __restrict__ call_prices,
    const double* __restrict__ strikes,
    const double* __restrict__ strategy_params,
    int8_t* __restrict__ stock_signals,
    int8_t* __restrict__ call_signals,
    double* __restrict__ premium_collected,
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
    double strike_offset_pct = strategy_params[strategy_idx * 2];
    double min_premium_pct = strategy_params[strategy_idx * 2 + 1];

    // Get market data
    double spot = underlying_prices[candle_idx];
    double call_price = call_prices[idx];
    double strike = strikes[idx];

    // Validate inputs (avoid NaN propagation)
    bool valid_data = isfinite(spot) && isfinite(call_price) && isfinite(strike) &&
                      spot > 0.0 && call_price >= 0.0 && strike > 0.0;

    if (!valid_data) {
        stock_signals[idx] = 0;
        call_signals[idx] = 0;
        premium_collected[idx] = 0.0;
        return;
    }

    // Calculate expected strike based on offset
    double expected_strike = spot * (1.0 + strike_offset_pct / 100.0);
    
    // Check if strike is OTM (above spot)
    bool is_otm = strike > spot;
    
    // Check if premium meets minimum requirement
    double min_premium = spot * (min_premium_pct / 100.0);
    bool premium_sufficient = call_price >= min_premium;

    // Entry Logic: Buy stock and sell OTM call if premium is sufficient
    if (is_otm && premium_sufficient) {
        stock_signals[idx] = 1;   // Buy 100 shares
        call_signals[idx] = -1;   // Sell 1 OTM call (short)
        premium_collected[idx] = call_price;
    } else {
        // Don't enter position if conditions not met
        stock_signals[idx] = 0;
        call_signals[idx] = 0;
        premium_collected[idx] = 0.0;
    }

    // P&L Calculation (for monitoring):
    // - Max profit: premium + (strike - spot) if spot >= strike at expiry
    // - Max loss: (spot - 0) - premium (if stock goes to zero)
    // - Breakeven: spot - premium
}

/// Covered call P&L calculation kernel
///
/// Calculates profit/loss at expiration for covered call positions.
///
/// Grid: 2D (candles × strategies)
/// Block: (256, 4) threads
///
/// # Arguments
///
/// - entry_prices: Spot price at entry [n_strategies × n_candles]
/// - exit_prices: Spot price at exit/expiry [n_strategies × n_candles]
/// - strikes: Call strike prices [n_strategies × n_candles]
/// - premiums: Premium collected [n_strategies × n_candles]
/// - stock_signals: Stock position signals [n_strategies × n_candles]
/// - pnl: Output P&L per share [n_strategies × n_candles]
/// - n_strategies: Number of strategy configurations
/// - n_candles: Number of time points
extern "C" __global__ void covered_call_pnl_kernel(
    const double* __restrict__ entry_prices,
    const double* __restrict__ exit_prices,
    const double* __restrict__ strikes,
    const double* __restrict__ premiums,
    const int8_t* __restrict__ stock_signals,
    double* __restrict__ pnl,
    int n_strategies,
    int n_candles
) {
    int candle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strategy_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (candle_idx >= n_candles || strategy_idx >= n_strategies) return;

    int idx = strategy_idx * n_candles + candle_idx;

    // Only calculate P&L if position was entered
    if (stock_signals[idx] != 1) {
        pnl[idx] = 0.0;
        return;
    }

    double entry_price = entry_prices[idx];
    double exit_price = exit_prices[idx];
    double strike = strikes[idx];
    double premium = premiums[idx];

    // Validate inputs
    if (!isfinite(entry_price) || !isfinite(exit_price) || 
        !isfinite(strike) || !isfinite(premium)) {
        pnl[idx] = 0.0;
        return;
    }

    // Calculate P&L components:
    // 1. Stock P&L: exit_price - entry_price
    // 2. Call P&L: 
    //    - If exit_price <= strike: Keep full premium (call expires worthless)
    //    - If exit_price > strike: Premium - (exit_price - strike) (called away)
    double stock_pnl = exit_price - entry_price;
    double call_pnl;

    if (exit_price <= strike) {
        // Call expires worthless, keep premium
        call_pnl = premium;
    } else {
        // Stock called away at strike
        // Effective stock exit at strike, not exit_price
        stock_pnl = strike - entry_price;
        call_pnl = premium; // Already collected premium upfront
    }

    // Total P&L per share
    pnl[idx] = stock_pnl + call_pnl;
}
