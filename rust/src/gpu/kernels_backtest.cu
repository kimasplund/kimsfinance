//! CUDA Kernels for GPU Batch Backtesting
//!
//! Production-ready 4-phase batch backtesting system for genetic optimization:
//! 1. Batch indicator calculation (extend kernels_3d pattern)
//! 2. Strategy signal generation (NEW - core innovation)
//! 3. Backtest execution (sequential per strategy, parallel across strategies)
//! 4. Metrics calculation (warp-level primitive reductions)
//!
//! # Source Assembly Contract (NVRTC)
//!
//! This file depends on the warp/block reduction primitives declared in
//! gpu/kernels/warp_primitives.cuh. NVRTC compiles from an in-memory string
//! with an EMPTY include path (see gpu/compile.rs), so a `#include` directive
//! cannot be resolved at runtime. The header is therefore PREPENDED to this
//! file at Rust compile time — see BACKTEST_KERNELS_SRC in
//! src/backtest/batch.rs. Do NOT add #include directives to this file.
//!
//! # Performance Targets
//!
//! - 1000 strategies × 10K candles: <250ms (40x vs sequential)
//! - VRAM usage: <1GB for typical workload
//! - Accuracy: Match CPU within 0.01% tolerance
//!
//! # Architecture
//!
//! RTX 3500 Ada: 14,336 CUDA cores, 12GB VRAM, 32MB L2 cache
//! - Execution: strategy-packed launch — 128 threads/block, each thread runs
//!   ONE strategy's candle loop sequentially (wall time = single strategy
//!   time, not 1000×). The old 1-thread-per-block launch left 127/128 of
//!   every SM partition idle.
//! - close_prices is shared by every strategy and stays L2-resident; the
//!   former single-thread shared-memory staging loop was a pessimization and
//!   has been removed.
//! - Max drawdown is tracked sequentially inside the execution loop (the old
//!   per-thread strided running_max in the metrics kernel systematically
//!   underestimated drawdowns).
//!
//! # Precision
//!
//! Kernels intentionally use double for equity/PnL accumulation: results are
//! compared 1:1 against the CPU reference (backtest/metrics.rs) at 0.01%
//! tolerance, and the sequential per-strategy loops are memory-latency bound,
//! not FP64-throughput bound, so Ada's 1:64 FP64 rate is not the bottleneck
//! here.

// NVRTC Kernel - Do NOT include system headers
// NVRTC provides built-in CUDA types and functions
// Including <cuda_runtime.h> or <stdint.h> causes JIT compilation errors

// Type definitions for NVRTC (built-in types)
typedef signed char int8_t;
typedef long long int64_t;

// Constants
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)
#define MAX_TRADES 1000

// Trade structure
//
// LAYOUT CONTRACT: mirrored byte-for-byte by `GpuTrade` in
// src/backtest/batch.rs. 3×double + 2×int64_t + int8_t, explicitly padded to
// 48 bytes (8-byte alignment). Device buffers MUST be sized as
// n_strategies * MAX_TRADES * 48 bytes.
struct Trade {
    double entry_price;   // offset  0
    double exit_price;    // offset  8
    int64_t entry_time;   // offset 16
    int64_t exit_time;    // offset 24
    double pnl;           // offset 32
    int8_t direction;     // offset 40: 1=Long, -1=Short
    int8_t _pad[7];       // offset 41..48: explicit padding
};

// Signal enumeration
enum Signal : int8_t {
    HOLD = 0,
    BUY = 1,
    SELL = 2,
    SHORT = 3,
    COVER = 4
};

// ============================================================================
// KERNEL 1: BATCH INDICATOR CALCULATION (EXTEND EXISTING)
// ============================================================================

// RSI Calculation Helper (device function)
__device__ double calculate_rsi_point(
    const double* __restrict__ close,
    int candle_idx,
    int period,
    int n_candles
) {
    if (candle_idx < period) {
        return CUDART_NAN;
    }

    // Calculate average gains and losses
    double avg_gain = 0.0;
    double avg_loss = 0.0;

    // Initial average (first 'period' values)
    for (int i = candle_idx - period + 1; i <= candle_idx; i++) {
        if (i > 0) {
            double delta = close[i] - close[i - 1];
            if (delta > 0.0) {
                avg_gain += delta;
            } else {
                avg_loss += -delta;
            }
        }
    }

    avg_gain /= (double)period;
    avg_loss /= (double)period;

    // Calculate RSI
    if (avg_loss < 1e-10) {
        return 100.0;
    }

    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

// ATR Calculation Helper (device function)
__device__ double calculate_atr_point(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int candle_idx,
    int period,
    int n_candles
) {
    if (candle_idx < period) {
        return CUDART_NAN;
    }

    // Calculate true ranges
    double sum_tr = 0.0;

    for (int i = candle_idx - period + 1; i <= candle_idx; i++) {
        double h_l = high[i] - low[i];
        double h_pc = (i > 0) ? fabs(high[i] - close[i - 1]) : 0.0;
        double l_pc = (i > 0) ? fabs(low[i] - close[i - 1]) : 0.0;

        double tr = fmax(h_l, fmax(h_pc, l_pc));
        sum_tr += tr;
    }

    return sum_tr / (double)period;
}

// SMA Calculation Helper (device function)
__device__ double calculate_sma_point(
    const double* __restrict__ close,
    int candle_idx,
    int period,
    int n_candles
) {
    if (candle_idx < period - 1) {
        return CUDART_NAN;
    }

    double sum = 0.0;
    for (int i = 0; i < period; i++) {
        sum += close[candle_idx - i];
    }

    return sum / (double)period;
}

// Batch Indicator Kernel (3D Grid: Strategy × Indicator × Candle)
extern "C" __global__ void batch_indicators_kernel(
    const double* __restrict__ ohlcv,           // [N_candles × 5] O, H, L, C, V
    const double* __restrict__ params,          // [N_strategies × N_params]
    double* __restrict__ indicators,            // [N_strategies × N_indicators × N_candles]
    int N_strategies,
    int N_indicators,
    int N_candles,
    int N_params
) {
    int strategy_idx = blockIdx.x;
    int indicator_idx = blockIdx.y;
    int candle_chunk = blockIdx.z;
    int thread_id = threadIdx.x;
    int candle_idx = candle_chunk * blockDim.x + thread_id;

    if (strategy_idx >= N_strategies || indicator_idx >= N_indicators || candle_idx >= N_candles) {
        return;
    }

    // Extract OHLCV pointers
    const double* close = &ohlcv[3 * N_candles];  // Close is at offset 3
    const double* high = &ohlcv[1 * N_candles];   // High is at offset 1
    const double* low = &ohlcv[2 * N_candles];    // Low is at offset 2

    // Get strategy parameters
    int param_offset = strategy_idx * N_params;
    double period = params[param_offset + indicator_idx];  // Each indicator has its own period

    // Calculate indicator
    double value = CUDART_NAN;

    if (indicator_idx == 0) {
        // RSI
        value = calculate_rsi_point(close, candle_idx, (int)period, N_candles);
    } else if (indicator_idx == 1) {
        // ATR
        value = calculate_atr_point(high, low, close, candle_idx, (int)period, N_candles);
    } else if (indicator_idx == 2) {
        // SMA
        value = calculate_sma_point(close, candle_idx, (int)period, N_candles);
    }

    // Write to output [strategy][indicator][candle]
    int out_idx = strategy_idx * (N_indicators * N_candles) +
                  indicator_idx * N_candles +
                  candle_idx;
    indicators[out_idx] = value;
}

// ============================================================================
// KERNEL 2: STRATEGY SIGNAL GENERATION (NEW - CORE INNOVATION)
// ============================================================================

extern "C" __global__ void strategy_signals_kernel(
    const double* __restrict__ indicators,       // [N_strategies × N_indicators × N_candles]
    const double* __restrict__ params,           // [N_strategies × N_indicators × 3]
    int8_t* __restrict__ signals,                // [N_strategies × N_candles]
    int N_strategies,
    int N_indicators,
    int N_candles,
    int strategy_type                            // 0=RSI, 1=MA, 2=Bollinger
) {
    int strategy_idx = blockIdx.x;
    int candle_chunk = blockIdx.y;
    int thread_id = threadIdx.x;
    int candle_idx = candle_chunk * blockDim.x + thread_id;

    if (strategy_idx >= N_strategies || candle_idx >= N_candles) {
        return;
    }

    // PARAM LAYOUT CONTRACT: the host packs parameters with a stride of
    // exactly N_indicators * 3 doubles per strategy: 3 slots per indicator.
    // Wrappers assert this before launch — see pad_params_to_kernel_layout
    // in src/backtest/batch.rs. A mismatched stride reads out of bounds for
    // every strategy after the first.
    int param_base = strategy_idx * N_indicators * 3;
    double buy_threshold = params[param_base + 1];
    double sell_threshold = params[param_base + 2];

    // Get indicator base offset
    int indicator_base = strategy_idx * N_indicators * N_candles;

    // Read indicator value (RSI for now)
    double rsi = indicators[indicator_base + candle_idx];

    // Apply strategy logic
    int8_t signal = HOLD;

    if (!isnan(rsi)) {
        if (strategy_type == 0) {  // RSI Crossover
            if (rsi < buy_threshold) {
                signal = BUY;
            } else if (rsi > sell_threshold) {
                signal = SELL;
            }
        }
        // Add more strategy types here: MA crossover, Bollinger, etc.
    }

    // Write signal
    int signal_idx = strategy_idx * N_candles + candle_idx;
    signals[signal_idx] = signal;
}

// ============================================================================
// KERNEL 3: BACKTEST EXECUTION (SEQUENTIAL PER STRATEGY, PACKED THREADS)
// ============================================================================
//
// Launch config (both variants):
//   block_dim = (128, 1, 1)
//   grid_dim  = (ceil(N_strategies / 128), 1, 1)
//   shared_mem_bytes = 0
//
// Each thread executes exactly one strategy's candle loop sequentially —
// identical per-strategy semantics to the old 1-thread-per-block launch,
// but with ~128x better SM occupancy. The early bounds-check return is safe
// because the kernels contain no __syncthreads().
//
// Max drawdown is tracked in registers during the sequential loop, mirroring
// the CPU reference calculate_max_drawdown (src/backtest/metrics.rs): a
// single running peak over the whole equity curve, in order. (Written as a
// FRACTION; the CPU helper returns a percentage and callers scale.)

extern "C" __global__ void backtest_execution_kernel(
    const int8_t* __restrict__ signals,          // [N_strategies × N_candles]
    const double* __restrict__ close_prices,     // [N_candles]
    double* __restrict__ equity_curves,          // [N_strategies × N_candles]
    Trade* __restrict__ trades,                  // [N_strategies × MAX_TRADES]
    int* __restrict__ num_trades,                // [N_strategies]
    double* __restrict__ max_drawdowns,          // [N_strategies] fraction, not percent
    double initial_capital,
    double trading_fee,
    double slippage,
    int N_strategies,
    int N_candles
) {
    int strategy_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (strategy_idx >= N_strategies) {
        return;
    }

    // Per-strategy state - stored in registers, very fast
    double equity = initial_capital;
    double position = 0.0;  // 0=flat, >0=long, <0=short
    double entry_price = 0.0;
    int64_t entry_time = 0;
    int trade_count = 0;

    // Sequential running-peak drawdown state
    double running_max = -CUDART_INF;
    double max_dd = 0.0;

    // Base offsets
    int signal_base = strategy_idx * N_candles;
    int equity_base = strategy_idx * N_candles;
    int trade_base = strategy_idx * MAX_TRADES;

    // Sequential loop through candles - this is OK, parallel across strategies
    for (int candle = 0; candle < N_candles; candle++) {
        int8_t signal = signals[signal_base + candle];
        double close = close_prices[candle];

        // Apply fees and slippage
        double buy_price = close * (1.0 + slippage + trading_fee);
        double sell_price = close * (1.0 - slippage - trading_fee);

        // Execute signal
        if (signal == BUY && position <= 0.0) {
            // Close short if exists
            if (position < 0.0) {
                double exit_price = buy_price;
                double pnl = position * (entry_price - exit_price);

                if (trade_count < MAX_TRADES) {
                    trades[trade_base + trade_count].entry_price = entry_price;
                    trades[trade_base + trade_count].exit_price = exit_price;
                    trades[trade_base + trade_count].entry_time = entry_time;
                    trades[trade_base + trade_count].exit_time = candle;
                    trades[trade_base + trade_count].pnl = pnl;
                    trades[trade_base + trade_count].direction = -1;  // Short
                    trade_count++;
                }

                equity += pnl;
                position = 0.0;
            }

            // Open long
            position = equity / buy_price;  // Full position
            entry_price = buy_price;
            entry_time = candle;
            equity = 0.0;  // All capital in position
        }
        else if (signal == SELL && position >= 0.0) {
            // Close long if exists
            if (position > 0.0) {
                double exit_price = sell_price;
                double pnl = position * (exit_price - entry_price);

                if (trade_count < MAX_TRADES) {
                    trades[trade_base + trade_count].entry_price = entry_price;
                    trades[trade_base + trade_count].exit_price = exit_price;
                    trades[trade_base + trade_count].entry_time = entry_time;
                    trades[trade_base + trade_count].exit_time = candle;
                    trades[trade_base + trade_count].pnl = pnl;
                    trades[trade_base + trade_count].direction = 1;  // Long
                    trade_count++;
                }

                equity += position * exit_price;
                position = 0.0;
            }
        }

        // Mark-to-market equity
        double mtm_equity;
        if (position > 0.0) {
            mtm_equity = position * close;
        } else if (position < 0.0) {
            mtm_equity = equity + position * (entry_price - close);
        } else {
            mtm_equity = equity;
        }

        equity_curves[equity_base + candle] = mtm_equity;

        // Sequential drawdown tracking (CPU-reference semantics)
        running_max = fmax(running_max, mtm_equity);
        if (running_max > 1e-10) {
            double dd = (running_max - mtm_equity) / running_max;
            max_dd = fmax(max_dd, dd);
        }
    }

    // Store final per-strategy outputs
    num_trades[strategy_idx] = trade_count;
    max_drawdowns[strategy_idx] = max_dd;
}

// ============================================================================
// KERNEL 3 OPTIMIZED: REGISTER-RESIDENT STATE, HOISTED MULTIPLIERS
// ============================================================================
//
// The previous version staged close_prices through __shared__ memory with a
// single-thread copy loop per 128-candle chunk — a pessimization at 1 thread
// per block, and a deadlock hazard with packed threads since divergent
// threads would skip its __syncthreads(). close_prices is identical for all
// strategies and stays hot in Ada's 32MB L2, so it is read directly.

extern "C" __global__ void backtest_execution_kernel_optimized(
    const int8_t* __restrict__ signals,          // [N_strategies × N_candles]
    const double* __restrict__ close_prices,     // [N_candles]
    double* __restrict__ equity_curves,          // [N_strategies × N_candles]
    Trade* __restrict__ trades,                  // [N_strategies × MAX_TRADES]
    int* __restrict__ num_trades,                // [N_strategies]
    double* __restrict__ max_drawdowns,          // [N_strategies] fraction, not percent
    double initial_capital,
    double trading_fee,
    double slippage,
    int N_strategies,
    int N_candles
) {
    // Strategy-packed indexing: one thread per strategy.
    int strategy_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (strategy_idx >= N_strategies) {
        return;
    }

    // Per-strategy state - registers
    double equity = initial_capital;
    double position = 0.0;       // 0=flat, >0=long, <0=short
    double entry_price = 0.0;
    int64_t entry_time = 0;
    int trade_count = 0;

    // Sequential running-peak drawdown state
    double running_max = -CUDART_INF;
    double max_dd = 0.0;

    // Precompute fee/slippage multipliers - hoisted out of loop
    const double buy_mult = 1.0 + slippage + trading_fee;
    const double sell_mult = 1.0 - slippage - trading_fee;

    // Base offsets
    const int signal_base = strategy_idx * N_candles;
    const int equity_base = strategy_idx * N_candles;
    const int trade_base = strategy_idx * MAX_TRADES;

    for (int candle = 0; candle < N_candles; candle++) {
        int8_t signal = signals[signal_base + candle];
        double close = close_prices[candle];  // L2-hot, shared by all strategies

        // Compute trade prices using precomputed multipliers
        double buy_price = close * buy_mult;
        double sell_price = close * sell_mult;

        // Execute signal
        if (signal == BUY && position <= 0.0) {
            // Close short if exists
            if (position < 0.0) {
                double pnl = position * (entry_price - buy_price);

                if (trade_count < MAX_TRADES) {
                    Trade* t = &trades[trade_base + trade_count];
                    t->entry_price = entry_price;
                    t->exit_price = buy_price;
                    t->entry_time = entry_time;
                    t->exit_time = candle;
                    t->pnl = pnl;
                    t->direction = -1;
                    trade_count++;
                }

                equity += pnl;
                position = 0.0;
            }

            // Open long
            position = equity / buy_price;
            entry_price = buy_price;
            entry_time = candle;
            equity = 0.0;
        }
        else if (signal == SELL && position >= 0.0) {
            // Close long if exists
            if (position > 0.0) {
                double pnl = position * (sell_price - entry_price);

                if (trade_count < MAX_TRADES) {
                    Trade* t = &trades[trade_base + trade_count];
                    t->entry_price = entry_price;
                    t->exit_price = sell_price;
                    t->entry_time = entry_time;
                    t->exit_time = candle;
                    t->pnl = pnl;
                    t->direction = 1;
                    trade_count++;
                }

                equity += position * sell_price;
                position = 0.0;
            }
        }

        // Mark-to-market equity
        double mtm_equity = equity;
        if (position > 0.0) {
            mtm_equity = position * close;
        } else if (position < 0.0) {
            mtm_equity = equity + position * (entry_price - close);
        }

        equity_curves[equity_base + candle] = mtm_equity;

        // Sequential drawdown tracking (CPU-reference semantics)
        running_max = fmax(running_max, mtm_equity);
        if (running_max > 1e-10) {
            double dd = (running_max - mtm_equity) / running_max;
            max_dd = fmax(max_dd, dd);
        }
    }

    // Store final per-strategy outputs
    num_trades[strategy_idx] = trade_count;
    max_drawdowns[strategy_idx] = max_dd;
}

// ============================================================================
// KERNEL 4: METRICS CALCULATION (PARALLEL REDUCTION)
// ============================================================================
//
// Launch config: grid = (N_strategies, 1, 1), block = (256, 1, 1),
// shared_mem_bytes = 0. All block reductions use the static __shared__
// buffers inside warp_primitives' block_reduce_* helpers; no dynamic shared
// memory is needed.
//
// Max drawdown is NOT computed here anymore: the per-thread strided
// running_max approach systematically underestimated it (each thread only
// saw the peak of its own stride-256 subsequence). The execution kernels now
// write max_drawdowns directly from their sequential loops.

extern "C" __global__ void metrics_calculation_kernel(
    const double* __restrict__ equity_curves,    // [N_strategies × N_candles]
    const Trade* __restrict__ trades,            // [N_strategies × MAX_TRADES]
    const int* __restrict__ num_trades,          // [N_strategies]
    double* __restrict__ sharpe_ratios,          // [N_strategies]
    double* __restrict__ win_rates,              // [N_strategies]
    int N_strategies,
    int N_candles
) {
    int strategy_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Uniform per-block early exit - safe: taken by the whole block, so the
    // block_reduce_* barriers below are still reached by all live threads.
    if (strategy_idx >= N_strategies) {
        return;
    }

    int equity_base = strategy_idx * N_candles;

    // ========== SHARPE RATIO CALCULATION (WARP OPTIMIZED) ==========

    // Phase 1: Each thread calculates partial sums over a strided subset of
    // returns, counting only valid samples (finite values, prev > 0).
    double local_sum = 0.0;
    double local_sq_sum = 0.0;
    int local_count = 0;

    for (int i = tid + 1; i < N_candles; i += block_size) {
        double curr = equity_curves[equity_base + i];
        double prev = equity_curves[equity_base + i - 1];

        if (!isnan(curr) && !isnan(prev) && prev > 1e-10) {
            double ret = (curr - prev) / prev;
            local_sum += ret;
            local_sq_sum += ret * ret;
            local_count++;
        }
    }

    // Phase 2: Block-level reductions via warp shuffle primitives.
    double total_sum, total_sq_sum;
    block_reduce_sum_pair<double>(local_sum, local_sq_sum, total_sum, total_sq_sum);
    double total_count = block_reduce_sum<double>((double)local_count);

    // Thread 0 calculates final Sharpe ratio. The divisor is the VALID
    // sample count - the old code counted valid samples and then divided by
    // N_candles - 1 anyway, deflating mean/variance whenever the curve
    // contained NaNs or non-positive equity values. Matches CPU
    // calculate_sharpe_ratio_scalar: mean over returns.len(), annualized by
    // sqrt(252): mean * 252 / (std * sqrt(252)) == (mean / std) * sqrt(252).
    if (tid == 0) {
        if (total_count > 0.5) {
            double mean = total_sum / total_count;
            double variance = (total_sq_sum / total_count) - (mean * mean);

            if (variance > 1e-10) {
                double std_dev = sqrt(variance);
                // Annualized Sharpe ratio - daily data, 252 trading days
                sharpe_ratios[strategy_idx] = (mean / std_dev) * sqrt(252.0);
            } else {
                sharpe_ratios[strategy_idx] = 0.0;
            }
        } else {
            sharpe_ratios[strategy_idx] = 0.0;
        }
    }

    // ========== WIN RATE CALCULATION (PARALLEL) ==========

    // Strided over the recorded trades instead of the old thread-0-only loop.
    int total_trades = num_trades[strategy_idx];
    int trade_base = strategy_idx * MAX_TRADES;
    int local_wins = 0;

    for (int t = tid; t < total_trades; t += block_size) {
        if (trades[trade_base + t].pnl > 0.0) {
            local_wins++;
        }
    }

    int total_wins = block_reduce_sum<int>(local_wins);

    if (tid == 0) {
        win_rates[strategy_idx] =
            (total_trades > 0) ? ((double)total_wins / (double)total_trades) : 0.0;
    }
}
