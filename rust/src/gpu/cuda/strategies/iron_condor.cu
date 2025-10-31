//! Iron Condor Strategy Kernel
//!
//! Generates signals for iron condor: sell OTM put + call, buy further OTM put + call.
//! Income-generating strategy that profits from low volatility (range-bound markets).
//!
//! # Strategy Logic
//!
//! **4-Leg Structure**:
//! 1. Buy OTM put at strike_long_put = spot * (1 - (short_put_offset + long_offset)/100)
//! 2. Sell OTM put at strike_short_put = spot * (1 - short_put_offset/100)
//! 3. Sell OTM call at strike_short_call = spot * (1 + short_call_offset/100)
//! 4. Buy OTM call at strike_long_call = spot * (1 + (short_call_offset + long_offset)/100)
//!
//! **Entry Signal**:
//! - Net credit = (short_put_premium + short_call_premium) - (long_put_premium + long_call_premium)
//! - Only enter if net_credit >= min_credit
//!
//! **P&L Profile**:
//! - Max profit: net_credit (if spot stays between short strikes at expiry)
//! - Max loss: (width of put or call spread) - net_credit
//! - Breakeven: short_strike ± net_credit
//!
//! # Performance Target
//!
//! - 1000 strategies × 500 candles: <10ms
//! - 4 legs per strategy = 4x memory bandwidth
//!
//! # Numerical Stability
//!
//! - Validates all prices are finite and positive
//! - Checks strike ordering: long_put < short_put < spot < short_call < long_call
//! - Prevents negative net credit (invalid spread)

/// Iron condor signal generation kernel
///
/// Grid: 2D (candles × strategies)
/// Block: (256, 4) threads
///
/// # Arguments
///
/// - underlying_prices: Spot prices [n_candles]
/// - put_prices: Put option prices [n_strategies × n_candles × 2] (long, short)
/// - call_prices: Call option prices [n_strategies × n_candles × 2] (short, long)
/// - put_strikes: Put strike prices [n_strategies × n_candles × 2] (long, short)
/// - call_strikes: Call strike prices [n_strategies × n_candles × 2] (short, long)
/// - strategy_params: [short_put_offset, short_call_offset, long_offset, min_credit] [n_strategies × 4]
/// - put_signals: Output put signals [long_put, short_put] [n_strategies × n_candles × 2]
/// - call_signals: Output call signals [short_call, long_call] [n_strategies × n_candles × 2]
/// - net_credit: Net credit received [n_strategies × n_candles]
/// - max_loss: Maximum loss potential [n_strategies × n_candles]
/// - n_strategies: Number of strategy configurations
/// - n_candles: Number of time points
extern "C" __global__ void iron_condor_signals_kernel(
    const double* __restrict__ underlying_prices,
    const double* __restrict__ put_prices,
    const double* __restrict__ call_prices,
    const double* __restrict__ put_strikes,
    const double* __restrict__ call_strikes,
    const double* __restrict__ strategy_params,
    int8_t* __restrict__ put_signals,
    int8_t* __restrict__ call_signals,
    double* __restrict__ net_credit,
    double* __restrict__ max_loss,
    int n_strategies,
    int n_candles
) {
    // 2D thread indexing
    int candle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strategy_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (candle_idx >= n_candles || strategy_idx >= n_strategies) return;

    // Linear index for strategy-candle pair
    int idx = strategy_idx * n_candles + candle_idx;
    int idx_2legs = idx * 2; // For 2-element arrays (put/call pairs)

    // Extract strategy parameters
    double short_put_offset = strategy_params[strategy_idx * 4];
    double short_call_offset = strategy_params[strategy_idx * 4 + 1];
    double long_offset = strategy_params[strategy_idx * 4 + 2];
    double min_credit_threshold = strategy_params[strategy_idx * 4 + 3];

    // Get market data
    double spot = underlying_prices[candle_idx];

    // Get option prices and strikes
    // Put side: [0] = long (buy), [1] = short (sell)
    double long_put_price = put_prices[idx_2legs + 0];
    double short_put_price = put_prices[idx_2legs + 1];
    double long_put_strike = put_strikes[idx_2legs + 0];
    double short_put_strike = put_strikes[idx_2legs + 1];

    // Call side: [0] = short (sell), [1] = long (buy)
    double short_call_price = call_prices[idx_2legs + 0];
    double long_call_price = call_prices[idx_2legs + 1];
    double short_call_strike = call_strikes[idx_2legs + 0];
    double long_call_strike = call_strikes[idx_2legs + 1];

    // Validate inputs (avoid NaN propagation)
    bool valid_data = 
        isfinite(spot) && spot > 0.0 &&
        isfinite(long_put_price) && long_put_price >= 0.0 &&
        isfinite(short_put_price) && short_put_price >= 0.0 &&
        isfinite(short_call_price) && short_call_price >= 0.0 &&
        isfinite(long_call_price) && long_call_price >= 0.0 &&
        isfinite(long_put_strike) && long_put_strike > 0.0 &&
        isfinite(short_put_strike) && short_put_strike > 0.0 &&
        isfinite(short_call_strike) && short_call_strike > 0.0 &&
        isfinite(long_call_strike) && long_call_strike > 0.0;

    if (!valid_data) {
        // Invalid data, no position
        put_signals[idx_2legs + 0] = 0;
        put_signals[idx_2legs + 1] = 0;
        call_signals[idx_2legs + 0] = 0;
        call_signals[idx_2legs + 1] = 0;
        net_credit[idx] = 0.0;
        max_loss[idx] = 0.0;
        return;
    }

    // Validate strike ordering: long_put < short_put < spot < short_call < long_call
    bool valid_strikes = 
        (long_put_strike < short_put_strike) &&
        (short_put_strike < spot) &&
        (spot < short_call_strike) &&
        (short_call_strike < long_call_strike);

    if (!valid_strikes) {
        // Invalid strike configuration
        put_signals[idx_2legs + 0] = 0;
        put_signals[idx_2legs + 1] = 0;
        call_signals[idx_2legs + 0] = 0;
        call_signals[idx_2legs + 1] = 0;
        net_credit[idx] = 0.0;
        max_loss[idx] = 0.0;
        return;
    }

    // Calculate net credit (premium received - premium paid)
    double credit_received = short_put_price + short_call_price; // Sell (receive)
    double debit_paid = long_put_price + long_call_price;         // Buy (pay)
    double calculated_net_credit = credit_received - debit_paid;

    // Check if net credit meets minimum threshold
    bool credit_sufficient = calculated_net_credit >= min_credit_threshold;

    if (!credit_sufficient || calculated_net_credit <= 0.0) {
        // Insufficient credit or negative credit (invalid spread)
        put_signals[idx_2legs + 0] = 0;
        put_signals[idx_2legs + 1] = 0;
        call_signals[idx_2legs + 0] = 0;
        call_signals[idx_2legs + 1] = 0;
        net_credit[idx] = 0.0;
        max_loss[idx] = 0.0;
        return;
    }

    // Entry Logic: Enter iron condor
    // Put spread: Buy long put, Sell short put
    put_signals[idx_2legs + 0] = 1;   // Buy long put (lower strike)
    put_signals[idx_2legs + 1] = -1;  // Sell short put (higher strike)

    // Call spread: Sell short call, Buy long call
    call_signals[idx_2legs + 0] = -1; // Sell short call (lower strike)
    call_signals[idx_2legs + 1] = 1;  // Buy long call (higher strike)

    net_credit[idx] = calculated_net_credit;

    // Calculate maximum loss
    // Max loss occurs if price moves beyond either long strike
    // Max loss = width of spread - net credit
    double put_spread_width = short_put_strike - long_put_strike;
    double call_spread_width = long_call_strike - short_call_strike;
    
    // Max loss is the larger of the two spreads minus credit
    double max_put_loss = put_spread_width - calculated_net_credit;
    double max_call_loss = call_spread_width - calculated_net_credit;
    max_loss[idx] = fmax(max_put_loss, max_call_loss);
}

/// Iron condor P&L calculation kernel
///
/// Calculates profit/loss at expiration for iron condor positions.
///
/// Grid: 2D (candles × strategies)
/// Block: (256, 4) threads
///
/// # Arguments
///
/// - exit_prices: Spot price at exit/expiry [n_strategies × n_candles]
/// - put_strikes: Put strike prices [long, short] [n_strategies × n_candles × 2]
/// - call_strikes: Call strike prices [short, long] [n_strategies × n_candles × 2]
/// - net_credits: Net credit received [n_strategies × n_candles]
/// - put_signals: Put position signals [n_strategies × n_candles × 2]
/// - pnl: Output P&L [n_strategies × n_candles]
/// - n_strategies: Number of strategy configurations
/// - n_candles: Number of time points
extern "C" __global__ void iron_condor_pnl_kernel(
    const double* __restrict__ exit_prices,
    const double* __restrict__ put_strikes,
    const double* __restrict__ call_strikes,
    const double* __restrict__ net_credits,
    const int8_t* __restrict__ put_signals,
    double* __restrict__ pnl,
    int n_strategies,
    int n_candles
) {
    int candle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strategy_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (candle_idx >= n_candles || strategy_idx >= n_strategies) return;

    int idx = strategy_idx * n_candles + candle_idx;
    int idx_2legs = idx * 2;

    // Only calculate P&L if position was entered
    if (put_signals[idx_2legs + 1] != -1) { // Check if short put was sold
        pnl[idx] = 0.0;
        return;
    }

    double exit_price = exit_prices[idx];
    double long_put_strike = put_strikes[idx_2legs + 0];
    double short_put_strike = put_strikes[idx_2legs + 1];
    double short_call_strike = call_strikes[idx_2legs + 0];
    double long_call_strike = call_strikes[idx_2legs + 1];
    double credit = net_credits[idx];

    // Validate inputs
    if (!isfinite(exit_price) || !isfinite(credit)) {
        pnl[idx] = 0.0;
        return;
    }

    // Calculate P&L based on exit price position
    double pnl_value;

    if (exit_price <= long_put_strike) {
        // Max loss on put side
        // Put spread loss = (short_put_strike - long_put_strike)
        pnl_value = credit - (short_put_strike - long_put_strike);
    } else if (exit_price < short_put_strike) {
        // Partial loss on put side
        // Loss = (short_put_strike - exit_price)
        pnl_value = credit - (short_put_strike - exit_price);
    } else if (exit_price <= short_call_strike) {
        // Max profit: inside the profit zone
        pnl_value = credit;
    } else if (exit_price < long_call_strike) {
        // Partial loss on call side
        // Loss = (exit_price - short_call_strike)
        pnl_value = credit - (exit_price - short_call_strike);
    } else {
        // Max loss on call side
        // Call spread loss = (long_call_strike - short_call_strike)
        pnl_value = credit - (long_call_strike - short_call_strike);
    }

    pnl[idx] = pnl_value;
}
