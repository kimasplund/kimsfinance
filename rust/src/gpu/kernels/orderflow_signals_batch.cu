/**
 * Fused GPU Kernel: Orderflow Feature Extraction + Signal Generation
 *
 * Eliminates 48-60MB intermediate memory transfer by computing features and signals
 * in a single kernel launch. Features stay in registers/shared memory.
 *
 * ## Performance Targets
 *
 * - **Orderflow**: 500M-1B features/sec (6 features × 100M ticks/sec)
 * - **Signals**: 3-4B signals/sec (10-20 strategies × 100M ticks/sec)
 * - **Memory**: 6 bytes per tick per strategy (INT8 quantized features)
 * - **Fusion Savings**: Avoids 48-60MB intermediate write/read (10 strategies × 106M ticks × 6 features × 8 bytes)
 *
 * ## Architecture
 *
 * **Warp-per-strategy** parallelization:
 * - 32 threads per strategy (one warp)
 * - 10-20 warps per block (10-20 strategies)
 * - Each thread processes subset of ticks
 *
 * **Circular buffer** for sliding window:
 * - Fixed-size shared memory buffer (window_size elements)
 * - Head/tail pointers track position
 * - O(1) insert/evict (beats O(n) recalculation)
 *
 * **Per-feature dynamic range** quantization:
 * - Each feature gets its own [min, max] range
 * - Quantize: value → int((value - min) / (max - min) * 255)
 * - 8x compression: 6 features × 8 bytes = 48 bytes → 6 bytes
 *
 * ## Memory Layout
 *
 * **Input** (from Agent 1 - tick aggregation):
 * - `timestamps`: Unix timestamps in milliseconds (i64)
 * - `open/high/low/close`: OHLCV data (f32)
 * - `volume`: Trade volume (f32)
 * - `buy_volume`: Buy-side volume (f32)
 * - `sell_volume`: Sell-side volume (f32)
 *
 * **Output** (to Agent 3 - backtester):
 * - `signals`: [num_strategies][num_ticks] (i8: 1=buy, -1=sell, 0=hold)
 * - `features`: [num_strategies][num_ticks * 6] (i8 quantized)
 *
 * ## Feature Definitions
 *
 * 1. **Order imbalance**: buy_volume / total_volume (range: 0.0-1.0)
 * 2. **Volume delta**: cumulative (buy_volume - sell_volume)
 * 3. **Trade intensity**: trades_per_second (estimated from tick count)
 * 4. **Price velocity**: (price[t] - price[t-window]) / time_delta
 * 5. **Volume-weighted spread**: (bid_volume - ask_volume) / total_volume
 * 6. **Trade size distribution**: median(trade_size) over window
 *
 * ## Hardcoded Strategies (Phase 1)
 *
 * 1. **Simple momentum**: imbalance > 0.6 && volume_delta > 1000
 * 2. **Mean reversion**: imbalance < 0.4 && volume_delta < -1000
 * 3. **Breakout**: trade_intensity > 100 && price_velocity > 0.001
 * 4. **Scalping**: imbalance > 0.55 && abs(volume_delta) < 500
 * 5. **Trend following**: volume_delta > 5000 && price_velocity > 0.002
 */

// ============================================================================
// Constants and Configuration
// ============================================================================

#define WINDOW_SIZE 20          // Sliding window size for features
#define NUM_FEATURES 6          // Number of orderflow features
#define WARP_SIZE 32           // CUDA warp size
#define MAX_STRATEGIES 20      // Maximum strategies per block

// Signal types
#define SIGNAL_HOLD 0
#define SIGNAL_BUY 1
#define SIGNAL_SELL -1

// Strategy IDs (hardcoded Phase 1)
#define STRATEGY_MOMENTUM 0
#define STRATEGY_MEAN_REVERSION 1
#define STRATEGY_BREAKOUT 2
#define STRATEGY_SCALPING 3
#define STRATEGY_TREND_FOLLOWING 4

// ============================================================================
// Circular Buffer (Shared Memory)
// ============================================================================

/**
 * Circular buffer for sliding window
 *
 * Maintains last N values for feature calculation without O(n) recalculation.
 *
 * Layout in shared memory:
 * - buffer[WINDOW_SIZE]: Data values
 * - head: Next write position (0 to WINDOW_SIZE-1)
 */
struct CircularBuffer {
    float buffer[WINDOW_SIZE];
    int head;
    int count; // Number of elements currently in buffer
};

/**
 * Initialize circular buffer
 */
__device__ inline void circ_init(CircularBuffer* cb) {
    cb->head = 0;
    cb->count = 0;
}

/**
 * Add value to circular buffer (overwrites oldest if full)
 */
__device__ inline void circ_push(CircularBuffer* cb, float value) {
    cb->buffer[cb->head] = value;
    cb->head = (cb->head + 1) % WINDOW_SIZE;
    if (cb->count < WINDOW_SIZE) {
        cb->count++;
    }
}

/**
 * Get value at index (0 = oldest, count-1 = newest)
 */
__device__ inline float circ_get(const CircularBuffer* cb, int index) {
    if (index >= cb->count) return 0.0f;
    int pos = (cb->head - cb->count + index + WINDOW_SIZE) % WINDOW_SIZE;
    return cb->buffer[pos];
}

/**
 * Calculate sum of all values in buffer
 */
__device__ inline float circ_sum(const CircularBuffer* cb) {
    float sum = 0.0f;
    for (int i = 0; i < cb->count; i++) {
        sum += circ_get(cb, i);
    }
    return sum;
}

/**
 * Calculate median of buffer (simplified: returns middle element after sort)
 */
__device__ inline float circ_median(const CircularBuffer* cb) {
    if (cb->count == 0) return 0.0f;

    // Simple bubble sort (acceptable for WINDOW_SIZE=20)
    float sorted[WINDOW_SIZE];
    for (int i = 0; i < cb->count; i++) {
        sorted[i] = circ_get(cb, i);
    }

    for (int i = 0; i < cb->count - 1; i++) {
        for (int j = 0; j < cb->count - i - 1; j++) {
            if (sorted[j] > sorted[j + 1]) {
                float temp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = temp;
            }
        }
    }

    return sorted[cb->count / 2];
}

// ============================================================================
// Orderflow Feature Calculation
// ============================================================================

/**
 * Feature 1: Order imbalance
 *
 * buy_volume / total_volume (range: 0.0-1.0)
 */
__device__ inline float feature_order_imbalance(float buy_volume, float total_volume) {
    if (total_volume < 1e-6f) return 0.5f; // Neutral if no volume
    return buy_volume / total_volume;
}

/**
 * Feature 2: Volume delta (cumulative)
 *
 * Cumulative sum of (buy_volume - sell_volume)
 */
__device__ inline float feature_volume_delta(float buy_volume, float sell_volume, float prev_delta) {
    return prev_delta + (buy_volume - sell_volume);
}

/**
 * Feature 3: Trade intensity
 *
 * Trades per second (estimated from tick count over window)
 */
__device__ inline float feature_trade_intensity(const CircularBuffer* price_buffer, float time_delta_ms) {
    if (time_delta_ms < 1.0f) return 0.0f;
    float trades_per_ms = (float)price_buffer->count / time_delta_ms;
    return trades_per_ms * 1000.0f; // Convert to trades per second
}

/**
 * Feature 4: Price velocity
 *
 * (price[t] - price[t-window]) / time_delta
 */
__device__ inline float feature_price_velocity(const CircularBuffer* price_buffer, float time_delta_ms) {
    if (price_buffer->count < 2 || time_delta_ms < 1.0f) return 0.0f;

    float price_now = circ_get(price_buffer, price_buffer->count - 1);
    float price_old = circ_get(price_buffer, 0);

    return (price_now - price_old) / (time_delta_ms / 1000.0f); // Price change per second
}

/**
 * Feature 5: Volume-weighted spread
 *
 * (bid_volume - ask_volume) / total_volume
 * Approximation: Use (buy_volume - sell_volume) / total_volume
 */
__device__ inline float feature_volume_weighted_spread(float buy_volume, float sell_volume, float total_volume) {
    if (total_volume < 1e-6f) return 0.0f;
    return (buy_volume - sell_volume) / total_volume;
}

/**
 * Feature 6: Trade size distribution (median)
 *
 * Median trade size over sliding window
 */
__device__ inline float feature_trade_size_median(const CircularBuffer* volume_buffer) {
    return circ_median(volume_buffer);
}

// ============================================================================
// Quantization (Per-Feature Dynamic Range)
// ============================================================================

/**
 * Quantize float to INT8 using per-feature dynamic range
 *
 * Formula: int((value - min) / (max - min) * 255)
 *
 * @param value Raw feature value
 * @param min Minimum value for this feature (learned from calibration)
 * @param max Maximum value for this feature
 * @return Quantized value in range [0, 255] as signed int8 (cast to i8 later)
 */
__device__ inline char quantize_feature(float value, float min_val, float max_val) {
    if (max_val <= min_val) return 0; // Avoid division by zero

    // Clamp value to [min, max]
    float clamped = fminf(fmaxf(value, min_val), max_val);

    // Normalize to [0, 1]
    float normalized = (clamped - min_val) / (max_val - min_val);

    // Scale to [0, 255] and round
    int quantized = (int)(normalized * 255.0f + 0.5f);

    // Clamp to [0, 255] (safety)
    quantized = min(max(quantized, 0), 255);

    return (char)quantized;
}

// ============================================================================
// Strategy Signal Generation (Hardcoded Phase 1)
// ============================================================================

/**
 * Strategy 1: Simple Momentum
 *
 * Buy: imbalance > 0.6 && volume_delta > 1000
 * Sell: imbalance < 0.4 && volume_delta < -1000
 */
__device__ inline char strategy_momentum(float imbalance, float volume_delta) {
    if (imbalance > 0.6f && volume_delta > 1000.0f) {
        return SIGNAL_BUY;
    } else if (imbalance < 0.4f && volume_delta < -1000.0f) {
        return SIGNAL_SELL;
    }
    return SIGNAL_HOLD;
}

/**
 * Strategy 2: Mean Reversion
 *
 * Buy: imbalance < 0.4 && volume_delta < -1000 (oversold)
 * Sell: imbalance > 0.6 && volume_delta > 1000 (overbought)
 */
__device__ inline char strategy_mean_reversion(float imbalance, float volume_delta) {
    if (imbalance < 0.4f && volume_delta < -1000.0f) {
        return SIGNAL_BUY;
    } else if (imbalance > 0.6f && volume_delta > 1000.0f) {
        return SIGNAL_SELL;
    }
    return SIGNAL_HOLD;
}

/**
 * Strategy 3: Breakout
 *
 * Buy: trade_intensity > 100 && price_velocity > 0.001
 * Sell: trade_intensity > 100 && price_velocity < -0.001
 */
__device__ inline char strategy_breakout(float trade_intensity, float price_velocity) {
    if (trade_intensity > 100.0f && price_velocity > 0.001f) {
        return SIGNAL_BUY;
    } else if (trade_intensity > 100.0f && price_velocity < -0.001f) {
        return SIGNAL_SELL;
    }
    return SIGNAL_HOLD;
}

/**
 * Strategy 4: Scalping
 *
 * Buy: imbalance > 0.55 && abs(volume_delta) < 500
 * Sell: imbalance < 0.45 && abs(volume_delta) < 500
 */
__device__ inline char strategy_scalping(float imbalance, float volume_delta) {
    float abs_delta = fabsf(volume_delta);
    if (imbalance > 0.55f && abs_delta < 500.0f) {
        return SIGNAL_BUY;
    } else if (imbalance < 0.45f && abs_delta < 500.0f) {
        return SIGNAL_SELL;
    }
    return SIGNAL_HOLD;
}

/**
 * Strategy 5: Trend Following
 *
 * Buy: volume_delta > 5000 && price_velocity > 0.002
 * Sell: volume_delta < -5000 && price_velocity < -0.002
 */
__device__ inline char strategy_trend_following(float volume_delta, float price_velocity) {
    if (volume_delta > 5000.0f && price_velocity > 0.002f) {
        return SIGNAL_BUY;
    } else if (volume_delta < -5000.0f && price_velocity < -0.002f) {
        return SIGNAL_SELL;
    }
    return SIGNAL_HOLD;
}

/**
 * Generate signal for given strategy
 */
__device__ inline char generate_signal(
    int strategy_id,
    float f1_imbalance,
    float f2_volume_delta,
    float f3_trade_intensity,
    float f4_price_velocity,
    float f5_vw_spread,
    float f6_trade_size_median
) {
    switch (strategy_id) {
        case STRATEGY_MOMENTUM:
            return strategy_momentum(f1_imbalance, f2_volume_delta);
        case STRATEGY_MEAN_REVERSION:
            return strategy_mean_reversion(f1_imbalance, f2_volume_delta);
        case STRATEGY_BREAKOUT:
            return strategy_breakout(f3_trade_intensity, f4_price_velocity);
        case STRATEGY_SCALPING:
            return strategy_scalping(f1_imbalance, f2_volume_delta);
        case STRATEGY_TREND_FOLLOWING:
            return strategy_trend_following(f2_volume_delta, f4_price_velocity);
        default:
            return SIGNAL_HOLD;
    }
}

// ============================================================================
// Main Fused Kernel
// ============================================================================

/**
 * Fused orderflow feature extraction + signal generation kernel
 *
 * Processes multiple strategies in parallel using warp-per-strategy pattern.
 * Features are computed and immediately consumed for signal generation,
 * eliminating intermediate memory transfer.
 *
 * ## Grid Configuration
 *
 * - Grid: (num_strategies / strategies_per_block, 1, 1)
 * - Block: (strategies_per_block * WARP_SIZE, 1, 1)
 *   - Example: 10 strategies/block × 32 threads/warp = 320 threads/block
 *
 * ## Thread Assignment
 *
 * - Thread 0-31: Strategy 0
 * - Thread 32-63: Strategy 1
 * - ...
 * - Thread (N-1)*32 to N*32-1: Strategy N-1
 *
 * ## Shared Memory Usage
 *
 * Per-strategy circular buffers (WINDOW_SIZE × 4 bytes × 3 buffers):
 * - price_buffer: Last WINDOW_SIZE prices
 * - volume_buffer: Last WINDOW_SIZE volumes
 * - time_buffer: Last WINDOW_SIZE timestamps (for velocity calculation)
 *
 * Total: 20 × 4 × 3 = 240 bytes per strategy
 * For 10 strategies: 2.4 KB shared memory
 *
 * @param timestamps      Input: Unix timestamps in milliseconds [num_ticks]
 * @param close_prices    Input: Close prices [num_ticks]
 * @param volumes         Input: Total volumes [num_ticks]
 * @param buy_volumes     Input: Buy-side volumes [num_ticks]
 * @param sell_volumes    Input: Sell-side volumes [num_ticks]
 * @param strategy_ids    Input: Strategy ID for each strategy [num_strategies]
 * @param feature_mins    Input: Min values for quantization [num_strategies * NUM_FEATURES]
 * @param feature_maxs    Input: Max values for quantization [num_strategies * NUM_FEATURES]
 * @param out_signals     Output: Trading signals [num_strategies * num_ticks] (i8)
 * @param out_features    Output: Quantized features [num_strategies * num_ticks * NUM_FEATURES] (i8)
 * @param num_strategies  Number of strategies to process
 * @param num_ticks       Number of ticks to process
 */
extern "C" __global__ void orderflow_signals_fused_kernel(
    const long long* __restrict__ timestamps,
    const float* __restrict__ close_prices,
    const float* __restrict__ volumes,
    const float* __restrict__ buy_volumes,
    const float* __restrict__ sell_volumes,
    const int* __restrict__ strategy_ids,
    const float* __restrict__ feature_mins,
    const float* __restrict__ feature_maxs,
    char* __restrict__ out_signals,
    char* __restrict__ out_features,
    int num_strategies,
    int num_ticks
) {
    // Thread identification
    int thread_id = threadIdx.x;
    int strategy_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + (thread_id / WARP_SIZE);
    int lane_id = thread_id % WARP_SIZE; // Position within warp (0-31)

    // Bounds check
    if (strategy_idx >= num_strategies) return;

    // Get strategy configuration
    int strategy_id = strategy_ids[strategy_idx];

    // Shared memory: circular buffers per strategy
    extern __shared__ char shared_mem[];
    CircularBuffer* price_buffer = (CircularBuffer*)&shared_mem[strategy_idx * sizeof(CircularBuffer) * 3];
    CircularBuffer* volume_buffer = (CircularBuffer*)&shared_mem[strategy_idx * sizeof(CircularBuffer) * 3 + sizeof(CircularBuffer)];
    CircularBuffer* time_buffer = (CircularBuffer*)&shared_mem[strategy_idx * sizeof(CircularBuffer) * 3 + sizeof(CircularBuffer) * 2];

    // Initialize circular buffers (one thread per strategy)
    if (lane_id == 0) {
        circ_init(price_buffer);
        circ_init(volume_buffer);
        circ_init(time_buffer);
    }
    __syncwarp();

    // State variables (per-strategy, stored in registers)
    float cumulative_volume_delta = 0.0f;

    // Process ticks (stride loop for thread cooperation)
    for (int tick = lane_id; tick < num_ticks; tick += WARP_SIZE) {
        // Load tick data (coalesced reads within warp)
        long long timestamp = timestamps[tick];
        float price = close_prices[tick];
        float volume = volumes[tick];
        float buy_vol = buy_volumes[tick];
        float sell_vol = sell_volumes[tick];

        // Update circular buffers (thread 0 of warp does sequential updates)
        if (lane_id == 0) {
            circ_push(price_buffer, price);
            circ_push(volume_buffer, volume);
            circ_push(time_buffer, (float)timestamp);
        }
        __syncwarp();

        // ===== FEATURE EXTRACTION =====

        // Feature 1: Order imbalance
        float total_volume = buy_vol + sell_vol;
        float f1_imbalance = feature_order_imbalance(buy_vol, total_volume);

        // Feature 2: Volume delta (cumulative)
        cumulative_volume_delta = feature_volume_delta(buy_vol, sell_vol, cumulative_volume_delta);
        float f2_volume_delta = cumulative_volume_delta;

        // Feature 3: Trade intensity (requires window)
        float time_delta = 0.0f;
        if (time_buffer->count >= 2) {
            float time_now = circ_get(time_buffer, time_buffer->count - 1);
            float time_old = circ_get(time_buffer, 0);
            time_delta = time_now - time_old;
        }
        float f3_trade_intensity = feature_trade_intensity(price_buffer, time_delta);

        // Feature 4: Price velocity
        float f4_price_velocity = feature_price_velocity(price_buffer, time_delta);

        // Feature 5: Volume-weighted spread
        float f5_vw_spread = feature_volume_weighted_spread(buy_vol, sell_vol, total_volume);

        // Feature 6: Trade size median
        float f6_trade_size_median = feature_trade_size_median(volume_buffer);

        // ===== QUANTIZATION =====

        // Get quantization ranges for this strategy
        int feature_base = strategy_idx * NUM_FEATURES;
        char q1 = quantize_feature(f1_imbalance, feature_mins[feature_base + 0], feature_maxs[feature_base + 0]);
        char q2 = quantize_feature(f2_volume_delta, feature_mins[feature_base + 1], feature_maxs[feature_base + 1]);
        char q3 = quantize_feature(f3_trade_intensity, feature_mins[feature_base + 2], feature_maxs[feature_base + 2]);
        char q4 = quantize_feature(f4_price_velocity, feature_mins[feature_base + 3], feature_maxs[feature_base + 3]);
        char q5 = quantize_feature(f5_vw_spread, feature_mins[feature_base + 4], feature_maxs[feature_base + 4]);
        char q6 = quantize_feature(f6_trade_size_median, feature_mins[feature_base + 5], feature_maxs[feature_base + 5]);

        // ===== SIGNAL GENERATION (FUSED!) =====

        char signal = generate_signal(
            strategy_id,
            f1_imbalance,
            f2_volume_delta,
            f3_trade_intensity,
            f4_price_velocity,
            f5_vw_spread,
            f6_trade_size_median
        );

        // ===== WRITE OUTPUTS =====

        // Write signal (coalesced within warp)
        int signal_idx = strategy_idx * num_ticks + tick;
        out_signals[signal_idx] = signal;

        // Write quantized features (coalesced within warp)
        int feature_idx = (strategy_idx * num_ticks + tick) * NUM_FEATURES;
        out_features[feature_idx + 0] = q1;
        out_features[feature_idx + 1] = q2;
        out_features[feature_idx + 2] = q3;
        out_features[feature_idx + 3] = q4;
        out_features[feature_idx + 4] = q5;
        out_features[feature_idx + 5] = q6;
    }
}

// ============================================================================
// Atomic Operations for Floats (Forward Declarations)
// ============================================================================

/**
 * Helper: Atomic min for float (using atomicCAS)
 *
 * CUDA provides atomicMin for integers but not floats, so we implement it using CAS.
 */
__device__ inline void atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        int new_val = __float_as_int(fminf(__int_as_float(assumed), val));
        old = atomicCAS(address_as_int, assumed, new_val);
    } while (assumed != old);
}

/**
 * Helper: Atomic max for float (using atomicCAS)
 *
 * CUDA provides atomicMax for integers but not floats, so we implement it using CAS.
 */
__device__ inline void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        int new_val = __float_as_int(fmaxf(__int_as_float(assumed), val));
        old = atomicCAS(address_as_int, assumed, new_val);
    } while (assumed != old);
}

// ============================================================================
// Calibration Kernel
// ============================================================================

/**
 * Calibration kernel: Compute min/max ranges for feature quantization
 *
 * Runs a first pass over data to determine dynamic ranges for each feature.
 * Required for per-feature quantization.
 *
 * Uses parallel reduction within block to find min/max efficiently.
 *
 * @param timestamps      Input: Unix timestamps [num_ticks]
 * @param close_prices    Input: Close prices [num_ticks]
 * @param volumes         Input: Total volumes [num_ticks]
 * @param buy_volumes     Input: Buy-side volumes [num_ticks]
 * @param sell_volumes    Input: Sell-side volumes [num_ticks]
 * @param out_mins        Output: Min values [NUM_FEATURES]
 * @param out_maxs        Output: Max values [NUM_FEATURES]
 * @param num_ticks       Number of ticks
 */
extern "C" __global__ void calibrate_feature_ranges_kernel(
    const long long* __restrict__ timestamps,
    const float* __restrict__ close_prices,
    const float* __restrict__ volumes,
    const float* __restrict__ buy_volumes,
    const float* __restrict__ sell_volumes,
    float* __restrict__ out_mins,
    float* __restrict__ out_maxs,
    int num_ticks
) {
    // Shared memory for reduction
    __shared__ float s_mins[NUM_FEATURES][256];
    __shared__ float s_maxs[NUM_FEATURES][256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize reduction arrays
    for (int f = 0; f < NUM_FEATURES; f++) {
        s_mins[f][tid] = 1e30f;  // Large positive
        s_maxs[f][tid] = -1e30f; // Large negative
    }

    // Process ticks (grid-stride loop)
    CircularBuffer price_buf, volume_buf, time_buf;
    circ_init(&price_buf);
    circ_init(&volume_buf);
    circ_init(&time_buf);

    float cumulative_delta = 0.0f;

    for (int tick = idx; tick < num_ticks; tick += blockDim.x * gridDim.x) {
        // Load data
        long long timestamp = timestamps[tick];
        float price = close_prices[tick];
        float volume = volumes[tick];
        float buy_vol = buy_volumes[tick];
        float sell_vol = sell_volumes[tick];

        // Update buffers
        circ_push(&price_buf, price);
        circ_push(&volume_buf, volume);
        circ_push(&time_buf, (float)timestamp);

        // Compute features
        float total_vol = buy_vol + sell_vol;
        float f1 = feature_order_imbalance(buy_vol, total_vol);
        cumulative_delta += (buy_vol - sell_vol);
        float f2 = cumulative_delta;

        float time_delta = 0.0f;
        if (time_buf.count >= 2) {
            time_delta = circ_get(&time_buf, time_buf.count - 1) - circ_get(&time_buf, 0);
        }
        float f3 = feature_trade_intensity(&price_buf, time_delta);
        float f4 = feature_price_velocity(&price_buf, time_delta);
        float f5 = feature_volume_weighted_spread(buy_vol, sell_vol, total_vol);
        float f6 = feature_trade_size_median(&volume_buf);

        // Update min/max
        s_mins[0][tid] = fminf(s_mins[0][tid], f1);
        s_maxs[0][tid] = fmaxf(s_maxs[0][tid], f1);
        s_mins[1][tid] = fminf(s_mins[1][tid], f2);
        s_maxs[1][tid] = fmaxf(s_maxs[1][tid], f2);
        s_mins[2][tid] = fminf(s_mins[2][tid], f3);
        s_maxs[2][tid] = fmaxf(s_maxs[2][tid], f3);
        s_mins[3][tid] = fminf(s_mins[3][tid], f4);
        s_maxs[3][tid] = fmaxf(s_maxs[3][tid], f4);
        s_mins[4][tid] = fminf(s_mins[4][tid], f5);
        s_maxs[4][tid] = fmaxf(s_maxs[4][tid], f5);
        s_mins[5][tid] = fminf(s_mins[5][tid], f6);
        s_maxs[5][tid] = fmaxf(s_maxs[5][tid], f6);
    }

    __syncthreads();

    // Reduction: parallel min/max across block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int f = 0; f < NUM_FEATURES; f++) {
                s_mins[f][tid] = fminf(s_mins[f][tid], s_mins[f][tid + stride]);
                s_maxs[f][tid] = fmaxf(s_maxs[f][tid], s_maxs[f][tid + stride]);
            }
        }
        __syncthreads();
    }

    // Thread 0 writes block results (atomic min/max across blocks)
    if (tid == 0) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            atomicMinFloat(&out_mins[f], s_mins[f][0]);
            atomicMaxFloat(&out_maxs[f], s_maxs[f][0]);
        }
    }
}
