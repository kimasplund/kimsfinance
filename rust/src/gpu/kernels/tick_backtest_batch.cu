/**
 * GPU Tick-Level Batch Backtest Kernel with Pending Orders Queue
 *
 * NVRTC-JIT-compiled at runtime (gpu/compile.rs, targets compute_89):
 * no header inclusion of any kind, extern "C" __global__ entry points only,
 * NVRTC built-in types/intrinsics exclusively.
 *
 * ## Normative reference
 *
 * rust/src/backtest/tick_engine.rs defines the ground-truth cost model and
 * equity semantics. This kernel mirrors it exactly:
 *
 * - Orders execute at the CURRENT tick's market price and timestamp once
 *   current_time >= signal_time + execution_delay_ms (tick_engine.rs:218-232
 *   executes pending orders with trade.price / trade.timestamp_ms — NOT the
 *   stale signal-time price the previous kernel used).
 * - Fees and slippage are charged exactly once, inside open_position /
 *   close_position (tick_engine.rs:331-418). The previous kernel ALSO
 *   pre-adjusted the execution price by (1 +/- slippage +/- fee),
 *   double-charging every trade.
 * - Equity is marked to market as cash + position_value + unrealized P&L
 *   (tick_engine.rs:426-441). The previous kernel returned size * price for
 *   longs (overstating by entry costs) and dropped position_value for shorts.
 * - Any position still open after the last tick is force-closed at the last
 *   price/timestamp (recording the trade), while final_equity reports the
 *   last in-loop mark-to-market value — tick_engine.rs:259-271 passes the
 *   pre-close `position.equity` to calculate_metrics. Quirky, but it is the
 *   normative CPU behavior and the GPU-vs-CPU parity tests assert it.
 *
 * ## Signal contract (pipeline boundary)
 *
 * The orderflow producer (kernels/orderflow_signals_batch.cu and
 * cpu/orderflow.rs) emits i8 signals BUY=1, SELL=-1, HOLD=0. This kernel
 * remaps raw signals at execution time (see remap_signal):
 *
 *   raw  1  -> BUY (close short if any, open long)
 *   raw -1  -> SELL (close long) when allow_short == 0,
 *              SHORT (close long, open short) when allow_short != 0
 *   raw  0  -> HOLD
 *   raw 2-4 -> legacy enum encoding, unchanged (SELL=2, SHORT=3, COVER=4)
 *   other   -> HOLD (defensive)
 *
 * The previous kernel interpreted raw bytes directly as the 0..4 enum,
 * silently dropping every -1 sell signal from the orderflow pipeline.
 *
 * ## Occupancy
 *
 * One THREAD per strategy (grid = ceil(N_strategies / THREADS_PER_BLOCK),
 * block = THREADS_PER_BLOCK). Every thread walks the tick loop in lockstep,
 * so prices[tick] / timestamps[tick] loads are warp-uniform broadcasts.
 * The previous kernel launched one single-thread block per strategy with all
 * other lanes returning immediately — 10-20 active threads on the whole GPU.
 * The pending-order queue lives in a per-thread local array (1.6KB/thread);
 * the old shared-memory queue was both pointless for one thread and broken
 * for more than one.
 *
 * ## Precision
 *
 * Position/cash accumulators are deliberately FP64 despite Ada's 1:64
 * FP64:FP32 throughput: equity compounds over up to 1e8 ticks and the parity
 * contract with the f64 CPU engine is 1e-9. The tick loop is dominated by
 * global memory latency (sequential per-strategy scan), not arithmetic, so
 * the FP64 penalty is hidden. f32 price migration is deferred until runtime
 * benchmarking is possible.
 *
 * ## Memory
 *
 * - equity_stride parameter: 0 = no equity curve stored, k = every k-th tick
 *   (ceil(N_ticks / k) points per strategy). Sharpe/drawdown/returns are
 *   computed incrementally in registers, so a 106M-tick x 10-strategy run no
 *   longer needs the 8B/tick/strategy store (and the matching multi-GB D2H).
 * - Queue overflow increments overflow_counts[strategy] via atomicAdd and
 *   executes the order immediately (graceful degradation). The previous
 *   kernel issued a device-side print from the hot loop.
 *
 * ## Preconditions
 *
 * timestamps must be non-decreasing (tick data). FIFO order + a constant
 * delay then guarantees the queue head always holds the earliest execution
 * time, so the head-only expiry check below is exhaustive.
 */

// NVRTC kernel - no system headers; built-in types only
typedef signed char int8_t;
typedef long long int64_t;

// Layout contract mirrored in rust/src/gpu/tick_backtest_batch.rs
// (host-side tests assert these #define lines verbatim)
#define MAX_TRADES 1000
#define MAX_PENDING_ORDERS 100
#define THREADS_PER_BLOCK 128

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Trade signal enumeration (kernel-internal encoding; see remap_signal for
/// the accepted wire encodings)
enum Signal : int8_t {
    HOLD = 0,
    BUY = 1,
    SELL = 2,
    SHORT = 3,
    COVER = 4
};

/// Completed trade record.
///
/// Layout contract with Rust `GpuTrade` (#[repr(C)]): 3 x double + 2 x
/// int64_t + int8_t, natural alignment 8 -> sizeof == 48 (asserted host-side).
struct Trade {
    double entry_price;
    double exit_price;
    int64_t entry_time;
    int64_t exit_time;
    double pnl;
    int8_t direction; // 1=Long, -1=Short
};

/// Pending order: 16-byte packed record in per-thread local memory.
///
/// `signal` stores the RAW pipeline byte (remapped at execution time so the
/// allow_short policy applies uniformly to queued and overflow-executed
/// orders). `signal_price` records the f32 market price at signal time for
/// diagnostics / future limit-order semantics; execution itself uses the
/// CURRENT tick's price for CPU parity (see file header).
struct PendingOrder {
    int64_t execution_time; // signal timestamp_ms + execution_delay_ms
    float signal_price;     // price at signal time (reference only)
    int8_t signal;          // raw pipeline signal byte
    // 3 padding bytes -> sizeof == 16
};

/// Position state (registers; mirrors tick_engine.rs `Position` fields)
struct Position {
    double cash;             // Available cash
    double position_size;    // >0=long, <0=short, 0=flat
    double position_value;   // Entry value NET of entry costs
    double entry_price;      // Entry price
    int64_t entry_timestamp; // Entry timestamp
};

// ============================================================================
// WELFORD'S ALGORITHM (numerically stable running mean/variance for Sharpe)
// ============================================================================

struct WelfordAccumulator {
    double mean;
    double M2; // Sum of squared differences from mean
    int n;     // Sample count
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
// SIGNAL REMAP (pipeline contract; see file header)
// ============================================================================

__device__ inline Signal remap_signal(int8_t raw, int allow_short) {
    if (raw == -1) {
        // Orderflow SELL: close long; additionally open short when the
        // allow_short policy permits (matches tick_engine.rs Signal::Sell,
        // which closes any long and opens a short via open_position(-1.0)).
        return allow_short ? SHORT : SELL;
    }
    if (raw >= 0 && raw <= 4) {
        return (Signal)raw; // legacy 0..4 encoding, unchanged
    }
    return HOLD; // defensive: unknown bytes do nothing
}

// ============================================================================
// POSITION MANAGEMENT (mirrors tick_engine.rs open/close/update exactly)
// ============================================================================

/// Open position with all available cash (tick_engine.rs:331-357).
/// Caller closes any existing position first (CPU does the same).
__device__ void open_position(
    Position* pos,
    double price,
    int64_t timestamp,
    double direction, // 1.0 = long, -1.0 = short
    double trading_fee,
    double slippage
) {
    double gross_position_value = pos->cash / price;
    double fee = gross_position_value * price * trading_fee;
    double slippage_cost = gross_position_value * price * slippage;
    double total_cost = fee + slippage_cost;

    pos->position_size = gross_position_value * direction;
    pos->entry_price = price;
    pos->entry_timestamp = timestamp;
    pos->position_value = pos->cash - total_cost; // NET value after costs
    pos->cash = 0.0;                              // All cash converted to position
}

/// Close position at exit_price (tick_engine.rs:367-418). The ONLY place
/// exit fees/slippage are charged; pnl itself is gross of exit costs,
/// exactly like the CPU trade records.
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
        return; // No position to close
    }

    double exit_value = fabs(pos->position_size) * exit_price;
    double fee = exit_value * trading_fee;
    double slippage_cost = exit_value * slippage;

    double pnl;
    if (pos->position_size > 0.0) {
        pnl = exit_value - pos->position_value; // Long
    } else {
        pnl = pos->position_value - exit_value; // Short
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

/// Mark-to-market equity (tick_engine.rs:426-441):
/// flat  -> cash
/// long  -> cash + position_value + (price - entry) * size
/// short -> cash + position_value + (entry - price) * |size|
__device__ double calculate_equity(const Position* pos, double current_price) {
    if (pos->position_size == 0.0) {
        return pos->cash;
    }
    double unrealized;
    if (pos->position_size > 0.0) {
        unrealized = (current_price - pos->entry_price) * pos->position_size;
    } else {
        unrealized = (pos->entry_price - current_price) * fabs(pos->position_size);
    }
    return pos->cash + pos->position_value + unrealized;
}

// ============================================================================
// TRADE EXECUTION
// ============================================================================

/// Execute one raw pipeline signal at the RAW market price. No price
/// pre-adjustment here: fees and slippage are charged once inside
/// open_position / close_position, mirroring tick_engine.rs (the previous
/// kernel multiplied price by (1 +/- slippage +/- fee) on top, double
/// charging every trade).
__device__ void execute_signal(
    int8_t raw_signal,
    int allow_short,
    Position* pos,
    double current_price,
    int64_t current_time,
    double trading_fee,
    double slippage,
    Trade* trades,
    int* trade_count,
    int max_trades
) {
    Signal signal = remap_signal(raw_signal, allow_short);

    switch (signal) {
        case BUY:
            if (pos->position_size <= 0.0) {
                if (pos->position_size < 0.0) {
                    close_position(pos, current_price, current_time, trading_fee, slippage,
                                   trades, trade_count, max_trades);
                }
                open_position(pos, current_price, current_time, 1.0, trading_fee, slippage);
            }
            break;

        case SELL:
            // Close long only (long-only exit; backwards-compatible with the
            // legacy 0..4 contract). tick_engine.rs Signal::Sell additionally
            // opens a short — route -1 with allow_short=1 (-> SHORT) for that.
            if (pos->position_size > 0.0) {
                close_position(pos, current_price, current_time, trading_fee, slippage,
                               trades, trade_count, max_trades);
            }
            break;

        case SHORT:
            if (pos->position_size >= 0.0) {
                if (pos->position_size > 0.0) {
                    close_position(pos, current_price, current_time, trading_fee, slippage,
                                   trades, trade_count, max_trades);
                }
                open_position(pos, current_price, current_time, -1.0, trading_fee, slippage);
            }
            break;

        case COVER:
            if (pos->position_size < 0.0) {
                close_position(pos, current_price, current_time, trading_fee, slippage,
                               trades, trade_count, max_trades);
            }
            break;

        case HOLD:
        default:
            break;
    }
}

// ============================================================================
// MAIN KERNEL
// ============================================================================

extern "C" __global__ void tick_backtest_batch_kernel(
    // Inputs
    const int8_t* __restrict__ signals,     // [N_strategies x N_ticks] raw pipeline bytes
    const double* __restrict__ prices,      // [N_ticks]
    const int64_t* __restrict__ timestamps, // [N_ticks] (ms, non-decreasing)

    // Outputs
    double* __restrict__ equity_curves,        // [N_strategies x ceil(N_ticks/equity_stride)];
                                               // unused when equity_stride == 0
    Trade* __restrict__ trades,                // [N_strategies x MAX_TRADES]
    int* __restrict__ num_trades,              // [N_strategies]
    unsigned int* __restrict__ overflow_counts, // [N_strategies], host pre-zeroed

    // Metrics outputs
    double* __restrict__ final_equity,  // [N_strategies]
    double* __restrict__ total_return,  // [N_strategies] (percent)
    double* __restrict__ sharpe_ratios, // [N_strategies]
    double* __restrict__ max_drawdowns, // [N_strategies] (fraction)
    double* __restrict__ win_rates,     // [N_strategies] (fraction)

    // Configuration
    int N_strategies,
    int N_ticks,
    double initial_capital,
    double trading_fee,
    double slippage,
    int execution_delay_ms,
    int allow_short,  // -1 signals: 0 -> SELL (close long), nonzero -> SHORT
    int equity_stride // 0 = no equity curve, k = store every k-th tick
) {
    // One THREAD per strategy; the whole warp walks the same tick index, so
    // prices/timestamps loads are warp-uniform broadcasts.
    int strategy_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (strategy_idx >= N_strategies) return;

    // ------------------------------------------------------------------
    // Per-thread pending-orders ring buffer (local memory, 1.6KB/thread).
    // ------------------------------------------------------------------
    PendingOrder pending_queue[MAX_PENDING_ORDERS];
    int queue_head = 0;
    int queue_tail = 0;
    int queue_size = 0;

    // ------------------------------------------------------------------
    // Register state
    // ------------------------------------------------------------------
    Position pos;
    pos.cash = initial_capital;
    pos.position_size = 0.0;
    pos.position_value = 0.0;
    pos.entry_price = 0.0;
    pos.entry_timestamp = 0;

    int trade_count = 0;

    long long signal_base = (long long)strategy_idx * N_ticks;
    Trade* strategy_trades = &trades[(long long)strategy_idx * MAX_TRADES];

    // Stored equity points per strategy (matches host equity_points())
    int n_eq = (equity_stride > 0) ? (N_ticks + equity_stride - 1) / equity_stride : 0;
    long long equity_base = (long long)strategy_idx * n_eq;

    // Incremental metrics (registers; independent of equity_stride)
    WelfordAccumulator returns_acc;
    welford_init(&returns_acc);

    double running_peak = initial_capital;
    double max_dd = 0.0;
    double prev_equity = initial_capital;

    // ------------------------------------------------------------------
    // Main loop: sequential per-strategy (position state is path-dependent)
    // ------------------------------------------------------------------
    for (int tick = 0; tick < N_ticks; tick++) {
        int64_t current_time = timestamps[tick];
        double current_price = prices[tick];
        int8_t raw_signal = signals[signal_base + tick];

        // 1) Execute every due pending order, FIFO, at the CURRENT tick's
        //    price/time (CPU parity; see file header). Monotone timestamps +
        //    constant delay keep execution times FIFO-ordered, so stopping at
        //    the first unexpired head order is exhaustive.
        while (queue_size > 0 && current_time >= pending_queue[queue_head].execution_time) {
            execute_signal(pending_queue[queue_head].signal, allow_short, &pos,
                           current_price, current_time, trading_fee, slippage,
                           strategy_trades, &trade_count, MAX_TRADES);
            queue_head = (queue_head + 1) % MAX_PENDING_ORDERS;
            queue_size--;
        }

        // 2) Queue this tick's signal (raw 0 is HOLD in both encodings)
        if (raw_signal != 0) {
            if (queue_size >= MAX_PENDING_ORDERS) {
                // Overflow: count it host-visibly and execute immediately
                // (graceful degradation; no hot-loop device print).
                atomicAdd(&overflow_counts[strategy_idx], 1u);
                execute_signal(raw_signal, allow_short, &pos, current_price, current_time,
                               trading_fee, slippage, strategy_trades, &trade_count, MAX_TRADES);
            } else {
                pending_queue[queue_tail].execution_time = current_time + execution_delay_ms;
                pending_queue[queue_tail].signal_price = (float)current_price;
                pending_queue[queue_tail].signal = raw_signal;
                queue_tail = (queue_tail + 1) % MAX_PENDING_ORDERS;
                queue_size++;
            }
        }

        // 3) Mark-to-market equity
        double current_equity = calculate_equity(&pos, current_price);
        if (equity_stride > 0 && (tick % equity_stride) == 0) {
            equity_curves[equity_base + tick / equity_stride] = current_equity;
        }

        // Per-tick return for Sharpe (Welford, registers)
        if (tick > 0 && prev_equity > 1e-10) {
            double ret = (current_equity - prev_equity) / prev_equity;
            welford_update(&returns_acc, ret);
        }

        // Max drawdown (incremental)
        running_peak = fmax(running_peak, current_equity);
        if (running_peak > 1e-10) {
            double dd = (running_peak - current_equity) / running_peak;
            max_dd = fmax(max_dd, dd);
        }

        prev_equity = current_equity;
    }

    // ------------------------------------------------------------------
    // Final metrics
    // ------------------------------------------------------------------

    // CPU parity quirk (tick_engine.rs:259-271): final_equity is the LAST
    // mark-to-market value; the forced close below only updates cash and the
    // trade list. Orders still pending after the last tick are dropped,
    // exactly like the CPU's leftover pending_orders vec.
    double final_eq = prev_equity;
    if (pos.position_size != 0.0) {
        close_position(&pos, prices[N_ticks - 1], timestamps[N_ticks - 1],
                       trading_fee, slippage, strategy_trades, &trade_count, MAX_TRADES);
    }

    num_trades[strategy_idx] = trade_count;
    final_equity[strategy_idx] = final_eq;
    total_return[strategy_idx] = ((final_eq - initial_capital) / initial_capital) * 100.0;

    // Sharpe ratio (annualized; per-tick returns, sample variance). The CPU
    // engine computes Sharpe over a 1-in-100 sampled curve with population
    // variance — Sharpe is intentionally NOT part of the 1e-9 parity contract.
    if (returns_acc.n > 1) {
        double std_dev = welford_std_dev(&returns_acc);
        sharpe_ratios[strategy_idx] =
            (std_dev > 1e-10) ? (returns_acc.mean / std_dev) * sqrt(252.0) : 0.0;
    } else {
        sharpe_ratios[strategy_idx] = 0.0;
    }

    max_drawdowns[strategy_idx] = max_dd;

    // Win rate (fraction of recorded trades with positive pnl)
    if (trade_count > 0) {
        int wins = 0;
        for (int t = 0; t < trade_count; t++) {
            if (strategy_trades[t].pnl > 0.0) {
                wins++;
            }
        }
        win_rates[strategy_idx] = (double)wins / (double)trade_count;
    } else {
        win_rates[strategy_idx] = 0.0;
    }
}
