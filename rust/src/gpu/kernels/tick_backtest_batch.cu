//! GPU Tick-Level Batch Backtest Kernel with Pending Orders Queue
//!
//! # Architecture
//!
//! - **Sequential per-strategy**: Maintains position state correctness
//! - **Parallel across strategies**: 10-20 strategies in parallel (one block per strategy)
//! - **Pending orders queue**: 10ms execution delay simulation
//! - **Exact CPU matching**: Matches tick_engine.rs calculations exactly
//!
//! # Performance Target
//!
//! - **Throughput**: 1-1.5B ticks/sec (10-20 strategies in parallel)
//! - **Latency**: 10ms execution delay (configurable)
//! - **Accuracy**: <0.01% deviation from CPU backtest
//!
//! # Memory Layout
//!
//! - Pending orders: 2KB per strategy (100 orders × 20 bytes)
//! - Position state: Stored in registers (zero contention)
//! - Trades array: 1000 trades max per strategy (MAX_TRADES)
//!
//! # Queue Overflow Handling
//!
//! If pending queue exceeds 100 orders:
//! - Log warning (printf)
//! - Execute immediately (graceful degradation)
//! - Continue processing (no crash)

// NVRTC Kernel - Do NOT include system headers
// Built-in CUDA types and functions provided by NVRTC

// Type definitions
typedef signed char int8_t;
typedef long long int64_t;

// Constants
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)
#define MAX_TRADES 1000
#define MAX_PENDING_ORDERS 100
#define DEFAULT_EXECUTION_DELAY_MS 10

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Trade signal enumeration (matches Rust Signal enum)
enum Signal : int8_t {
    HOLD = 0,
    BUY = 1,
    SELL = 2,
    SHORT = 3,
    COVER = 4
};

/// Completed trade record
struct Trade {
    double entry_price;
    double exit_price;
    int64_t entry_time;
    int64_t exit_time;
    double pnl;
    int8_t direction; // 1=Long, -1=Short
};

/// Pending order in execution queue
struct PendingOrder {
    Signal signal;
    int64_t execution_time;  // timestamp_ms + delay
    double price;            // Price at signal time
    bool active;             // Is this slot active?
};

/// Position state (stored in registers for performance)
struct Position {
    double cash;              // Available cash
    double position_size;     // >0=long, <0=short, 0=flat
    double position_value;    // Net position value after fees
    double entry_price;       // Entry price
    int64_t entry_timestamp;  // Entry timestamp
};

// ============================================================================
// WELFORD'S ALGORITHM FOR NUMERICAL STABILITY
// ============================================================================

/// Online mean and variance calculation (numerically stable)
struct WelfordAccumulator {
    double mean;
    double M2;  // Sum of squared differences from mean
    int n;      // Sample count
};

__device__ inline void welford_init(WelfordAccumulator* acc) {
    acc->mean = 0.0;
    acc->M2 = 0.0;
    acc->n = 0;
}

__device__ inline void welford_update(WelfordAccumulator* acc, double value) {
    acc->n++;
    double delta = value - acc->mean;
    acc->mean += delta / acc->n;
    double delta2 = value - acc->mean;
    acc->M2 += delta * delta2;
}

__device__ inline double welford_variance(const WelfordAccumulator* acc) {
    if (acc->n < 2) return 0.0;
    return acc->M2 / (acc->n - 1);
}

__device__ inline double welford_std_dev(const WelfordAccumulator* acc) {
    return sqrt(welford_variance(acc));
}

// ============================================================================
// CIRCULAR QUEUE OPERATIONS
// ============================================================================

__device__ void queue_init(
    PendingOrder* queue,
    int* head,
    int* tail,
    int* size
) {
    *head = 0;
    *tail = 0;
    *size = 0;

    // Initialize all slots as inactive
    for (int i = 0; i < MAX_PENDING_ORDERS; i++) {
        queue[i].active = false;
    }
}

__device__ bool queue_add(
    PendingOrder* queue,
    int* head,
    int* tail,
    int* size,
    Signal signal,
    int64_t execution_time,
    double price
) {
    if (*size >= MAX_PENDING_ORDERS) {
        return false;  // Queue full
    }

    queue[*tail].signal = signal;
    queue[*tail].execution_time = execution_time;
    queue[*tail].price = price;
    queue[*tail].active = true;

    *tail = (*tail + 1) % MAX_PENDING_ORDERS;
    (*size)++;

    return true;
}

__device__ PendingOrder queue_peek(
    const PendingOrder* queue,
    int head
) {
    return queue[head];
}

__device__ void queue_remove(
    PendingOrder* queue,
    int* head,
    int* size
) {
    queue[*head].active = false;
    *head = (*head + 1) % MAX_PENDING_ORDERS;
    (*size)--;
}

// ============================================================================
// POSITION MANAGEMENT (MATCHES CPU tick_engine.rs EXACTLY!)
// ============================================================================

/// Open position (matches tick_engine.rs:331-356)
__device__ void open_position(
    Position* pos,
    double price,
    int64_t timestamp,
    double direction,  // 1.0 = long, -1.0 = short
    double trading_fee,
    double slippage,
    Trade* trades,
    int* trade_count,
    int max_trades
) {
    // Close existing position first (if any)
    // This will be handled by caller to match CPU logic

    // Calculate position size (use all available cash)
    double gross_position_value = pos->cash / price;
    double fee = gross_position_value * price * trading_fee;
    double slippage_cost = gross_position_value * price * slippage;
    double total_cost = fee + slippage_cost;

    pos->position_size = gross_position_value * direction;
    pos->entry_price = price;
    pos->entry_timestamp = timestamp;
    pos->position_value = pos->cash - total_cost;  // NET value after costs
    pos->cash = 0.0;  // All cash converted to position
}

/// Close position (matches tick_engine.rs:367-418)
__device__ void close_position(
    Position* pos,
    double exit_price,
    int64_t exit_timestamp,
    double trading_fee,
    double slippage,
    Trade* trades,
    int* trade_count,
    int max_trades
) {
    if (pos->position_size == 0.0) {
        return;  // No position to close
    }

    double exit_value = fabs(pos->position_size) * exit_price;
    double fee = exit_value * trading_fee;
    double slippage_cost = exit_value * slippage;

    // Calculate P&L (matches CPU logic exactly)
    double pnl;
    if (pos->position_size > 0.0) {
        // Long position
        pnl = exit_value - pos->position_value;
    } else {
        // Short position
        pnl = pos->position_value - exit_value;
    }

    pos->cash += pos->position_value + pnl - fee - slippage_cost;

    // Record trade (if space available)
    if (*trade_count < max_trades) {
        trades[*trade_count].entry_price = pos->entry_price;
        trades[*trade_count].exit_price = exit_price;
        trades[*trade_count].entry_time = pos->entry_timestamp;
        trades[*trade_count].exit_time = exit_timestamp;
        trades[*trade_count].pnl = pnl;
        trades[*trade_count].direction = (pos->position_size > 0.0) ? 1 : -1;
        (*trade_count)++;
    }

    // Reset position
    pos->position_size = 0.0;
    pos->position_value = 0.0;
    pos->entry_price = 0.0;
    pos->entry_timestamp = 0;
}

/// Update equity with mark-to-market (matches tick_engine.rs:426-438)
__device__ double calculate_equity(
    const Position* pos,
    double current_price
) {
    if (pos->position_size == 0.0) {
        return pos->cash;
    } else if (pos->position_size > 0.0) {
        // Long position: current market value
        return pos->position_size * current_price;
    } else {
        // Short position: cash + unrealized P&L
        return pos->cash + pos->position_size * (pos->entry_price - current_price);
    }
}

// ============================================================================
// TRADE EXECUTION WITH PENDING ORDERS
// ============================================================================

/// Execute signal with slippage and fees
__device__ void execute_signal(
    Signal signal,
    Position* pos,
    double current_price,
    int64_t current_time,
    double trading_fee,
    double slippage,
    Trade* trades,
    int* trade_count,
    int max_trades
) {
    // Apply slippage (matches batch_backtest.cu:344-345)
    double buy_price = current_price * (1.0 + slippage + trading_fee);
    double sell_price = current_price * (1.0 - slippage - trading_fee);

    switch (signal) {
        case BUY:
            if (pos->position_size <= 0.0) {
                // Close short if exists
                if (pos->position_size < 0.0) {
                    close_position(pos, buy_price, current_time, trading_fee, slippage,
                                   trades, trade_count, max_trades);
                }
                // Open long
                open_position(pos, buy_price, current_time, 1.0, trading_fee, slippage,
                              trades, trade_count, max_trades);
            }
            break;

        case SELL:
            if (pos->position_size >= 0.0) {
                // Close long if exists
                if (pos->position_size > 0.0) {
                    close_position(pos, sell_price, current_time, trading_fee, slippage,
                                   trades, trade_count, max_trades);
                }
            }
            break;

        case SHORT:
            if (pos->position_size >= 0.0) {
                // Close long if exists
                if (pos->position_size > 0.0) {
                    close_position(pos, sell_price, current_time, trading_fee, slippage,
                                   trades, trade_count, max_trades);
                }
                // Open short
                open_position(pos, sell_price, current_time, -1.0, trading_fee, slippage,
                              trades, trade_count, max_trades);
            }
            break;

        case COVER:
            if (pos->position_size <= 0.0) {
                // Close short if exists
                if (pos->position_size < 0.0) {
                    close_position(pos, buy_price, current_time, trading_fee, slippage,
                                   trades, trade_count, max_trades);
                }
            }
            break;

        case HOLD:
            // Do nothing
            break;
    }
}

/// Process expired orders from pending queue
__device__ void process_pending_orders(
    PendingOrder* queue,
    int* head,
    int* tail,
    int* size,
    int64_t current_time,
    Position* pos,
    double trading_fee,
    double slippage,
    Trade* trades,
    int* trade_count,
    int max_trades
) {
    // Process all expired orders (FIFO order)
    while (*size > 0) {
        PendingOrder order = queue_peek(queue, *head);

        if (!order.active || current_time < order.execution_time) {
            break;  // No more expired orders
        }

        // Execute order at recorded price
        execute_signal(order.signal, pos, order.price, order.execution_time,
                       trading_fee, slippage, trades, trade_count, max_trades);

        // Remove from queue
        queue_remove(queue, head, size);
    }
}

// ============================================================================
// MAIN KERNEL
// ============================================================================

extern "C" __global__ void tick_backtest_batch_kernel(
    // Inputs
    const int8_t* __restrict__ signals,        // [N_strategies × N_ticks]
    const double* __restrict__ prices,         // [N_ticks]
    const int64_t* __restrict__ timestamps,    // [N_ticks] (milliseconds)

    // Outputs
    double* __restrict__ equity_curves,        // [N_strategies × N_ticks]
    Trade* __restrict__ trades,                // [N_strategies × MAX_TRADES]
    int* __restrict__ num_trades,              // [N_strategies]

    // Metrics outputs
    double* __restrict__ final_equity,         // [N_strategies]
    double* __restrict__ total_return,         // [N_strategies]
    double* __restrict__ sharpe_ratios,        // [N_strategies]
    double* __restrict__ max_drawdowns,        // [N_strategies]
    double* __restrict__ win_rates,            // [N_strategies]

    // Configuration
    int N_strategies,
    int N_ticks,
    double initial_capital,
    double trading_fee,
    double slippage,
    int execution_delay_ms                     // Default: 10ms
) {
    // One block per strategy (parallel across strategies)
    int strategy_idx = blockIdx.x;

    if (strategy_idx >= N_strategies) return;

    // Only thread 0 in block executes (sequential per-strategy)
    if (threadIdx.x != 0) return;

    // ========================================================================
    // SHARED MEMORY: PENDING ORDERS QUEUE
    // ========================================================================

    __shared__ PendingOrder pending_queue[MAX_PENDING_ORDERS];
    __shared__ int queue_head;
    __shared__ int queue_tail;
    __shared__ int queue_size;

    // Initialize queue
    queue_init(pending_queue, &queue_head, &queue_tail, &queue_size);

    // ========================================================================
    // REGISTER STATE: POSITION TRACKING
    // ========================================================================

    Position pos;
    pos.cash = initial_capital;
    pos.position_size = 0.0;
    pos.position_value = 0.0;
    pos.entry_price = 0.0;
    pos.entry_timestamp = 0;

    int trade_count = 0;

    // Base offsets
    int signal_base = strategy_idx * N_ticks;
    int equity_base = strategy_idx * N_ticks;
    int trade_base = strategy_idx * MAX_TRADES;

    // Welford accumulator for Sharpe ratio
    WelfordAccumulator returns_acc;
    welford_init(&returns_acc);

    // Max drawdown tracking
    double running_peak = initial_capital;
    double max_dd = 0.0;

    double prev_equity = initial_capital;

    // ========================================================================
    // MAIN PROCESSING LOOP (SEQUENTIAL PER-STRATEGY)
    // ========================================================================

    for (int tick = 0; tick < N_ticks; tick++) {
        int64_t current_time = timestamps[tick];
        double current_price = prices[tick];
        Signal signal = (Signal)signals[signal_base + tick];

        // Process expired pending orders FIRST
        process_pending_orders(
            pending_queue,
            &queue_head,
            &queue_tail,
            &queue_size,
            current_time,
            &pos,
            trading_fee,
            slippage,
            &trades[trade_base],
            &trade_count,
            MAX_TRADES
        );

        // Add new signal to pending queue (if not HOLD)
        if (signal != HOLD) {
            int64_t execution_time = current_time + execution_delay_ms;

            bool added = queue_add(
                pending_queue,
                &queue_head,
                &queue_tail,
                &queue_size,
                signal,
                execution_time,
                current_price
            );

            if (!added) {
                // Queue overflow: Execute immediately (graceful degradation)
                printf("WARNING: Pending queue overflow for strategy %d at tick %d - executing immediately\n",
                       strategy_idx, tick);
                execute_signal(signal, &pos, current_price, current_time,
                               trading_fee, slippage, &trades[trade_base],
                               &trade_count, MAX_TRADES);
            }
        }

        // Calculate mark-to-market equity
        double current_equity = calculate_equity(&pos, current_price);
        equity_curves[equity_base + tick] = current_equity;

        // Update Welford accumulator (for Sharpe ratio)
        if (tick > 0 && prev_equity > 1e-10) {
            double ret = (current_equity - prev_equity) / prev_equity;
            welford_update(&returns_acc, ret);
        }

        // Update max drawdown
        running_peak = fmax(running_peak, current_equity);
        if (running_peak > 1e-10) {
            double dd = (running_peak - current_equity) / running_peak;
            max_dd = fmax(max_dd, dd);
        }

        prev_equity = current_equity;
    }

    // ========================================================================
    // FINAL METRICS CALCULATION
    // ========================================================================

    // Store final trade count
    num_trades[strategy_idx] = trade_count;

    // Final equity and total return
    double final_eq = equity_curves[equity_base + N_ticks - 1];
    final_equity[strategy_idx] = final_eq;
    total_return[strategy_idx] = ((final_eq - initial_capital) / initial_capital) * 100.0;

    // Sharpe ratio (annualized, assumes tick data is ~daily equivalent)
    if (returns_acc.n > 1) {
        double std_dev = welford_std_dev(&returns_acc);
        if (std_dev > 1e-10) {
            sharpe_ratios[strategy_idx] = (returns_acc.mean / std_dev) * sqrt(252.0);
        } else {
            sharpe_ratios[strategy_idx] = 0.0;
        }
    } else {
        sharpe_ratios[strategy_idx] = 0.0;
    }

    // Max drawdown (already calculated incrementally)
    max_drawdowns[strategy_idx] = max_dd;

    // Win rate
    if (trade_count > 0) {
        int wins = 0;
        for (int t = 0; t < trade_count; t++) {
            if (trades[trade_base + t].pnl > 0.0) {
                wins++;
            }
        }
        win_rates[strategy_idx] = (double)wins / (double)trade_count;
    } else {
        win_rates[strategy_idx] = 0.0;
    }
}
