/**
 * GPU Tick-Level Aggregation with Hash-Based Bucketing
 *
 * Implements high-throughput trade→OHLCV aggregation using shared memory hash tables
 * for improved performance over global memory atomics.
 *
 * ## Algorithm: Two-Pass Hash-Based Aggregation
 *
 * **Pass 1: Parallel Binning**
 * - Each thread computes timestamp bucket for one trade
 * - Fully parallel, no contention
 * - Coalesced memory access (SoA layout)
 *
 * **Pass 2: Hash-Based Aggregation**
 * - Block-level shared memory hash table (99KB on CUDA 13.0)
 * - Atomic updates within shared memory (10-20x faster than global atomics)
 * - Flush to global memory at end of block
 * - Multiple strategies processed in parallel
 *
 * ## Performance Target
 *
 * - **Throughput**: 1-2B trades/sec (vs 100M/sec CPU)
 * - **GPU Utilization**: >80% during kernel execution
 * - **Memory Efficiency**: Minimize H2D/D2H transfers (pinned memory)
 * - **Numerical Accuracy**: Match CPU aggregation exactly
 *
 * ## Memory Layout: Structure-of-Arrays (SoA)
 *
 * **Input**:
 *   - timestamps[N]  - i64 (milliseconds since epoch)
 *   - prices[N]      - f32 (price per trade)
 *   - volumes[N]     - f32 (volume per trade)
 *   - sides[N]       - i8 (1=buy, -1=sell)
 *
 * **Output**:
 *   - open[C]        - f32 (first trade price in candle)
 *   - high[C]        - f32 (max price)
 *   - low[C]         - f32 (min price)
 *   - close[C]       - f32 (last trade price)
 *   - volume[C]      - f32 (sum of volumes)
 *   - num_trades[C]  - i32 (trade count)
 *
 * C = number of unique candles (determined after Pass 1)
 *
 * ## Quantization (Post-Aggregation)
 *
 * After aggregation, convert f32→i8 using per-feature dynamic range:
 * - open: quantize([min_open, max_open] → [0, 255])
 * - high: quantize([min_high, max_high] → [0, 255])
 * - low: quantize([min_low, max_low] → [0, 255])
 * - close: quantize([min_close, max_close] → [0, 255])
 * - volume: log-scale quantize (financial data is skewed)
 *
 * Quantization is done in separate kernel (not in this file).
 *
 * ## Shared Memory Layout (per block)
 *
 * 99KB shared memory budget (CUDA 13.0):
 * - Hash table entries: (bucket_id, ohlcv_data)
 * - ~1024 buckets per block × 32 bytes/entry = 32KB
 * - Remaining 67KB for intermediate buffers
 *
 * ## CUDA 13.0 Optimizations
 *
 * - L2 cache persistence hints for input arrays
 * - Async memory allocator for temporary buffers
 * - Pinned memory for faster H2D/D2H transfers
 * - Warp-level primitives for hash table probing
 */

// ============================================================================
// Type Definitions
// ============================================================================

// NVRTC built-in types (no includes needed)
typedef signed char int8_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;

// Constants
#define CUDART_NAN __int_as_float(0x7fc00000)
#define CUDART_INF __int_as_float(0x7f800000)
#define LLONG_MAX 9223372036854775807LL
#define LLONG_MIN (-9223372036854775807LL - 1LL)

// Hash table configuration
#define HASH_TABLE_SIZE 1024  // Power of 2 for fast modulo
#define HASH_EMPTY_BUCKET (-1)  // Sentinel for empty hash slot

// ============================================================================
// Hash Table Entry Structure
// ============================================================================

/**
 * Hash table entry stored in shared memory
 *
 * Total size: 36 bytes (aligned to 8 bytes → 40 bytes)
 * 1024 entries × 40 bytes = 40KB shared memory usage
 */
struct HashEntry {
    int32_t bucket_id;      // Timestamp bucket ID (-1 = empty)
    int64_t first_ts;       // Timestamp of first trade (for open)
    int64_t last_ts;        // Timestamp of last trade (for close)
    double first_price;     // Open price
    double last_price;      // Close price
    double high;            // Highest price
    double low;             // Lowest price
    double volume;          // Sum of volumes
    int32_t num_trades;     // Trade count
};

// ============================================================================
// Atomic Helpers for Float/Double
// ============================================================================

/**
 * Atomic maximum for double-precision floats
 */
__device__ inline double atomicMaxDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);

        // Only update if val > old_val
        if (val <= old_val) {
            break;
        }

        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

/**
 * Atomic minimum for double-precision floats
 */
__device__ inline double atomicMinDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);

        // Only update if val < old_val
        if (val >= old_val) {
            break;
        }

        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

/**
 * Atomic compare-and-swap for int64_t (timestamp tracking)
 *
 * Updates timestamp and associated price if new timestamp is smaller/larger.
 */
__device__ inline void atomicMinTimestampAndPrice(
    int64_t* ts_address,
    double* price_address,
    int64_t new_ts,
    double new_price
) {
    unsigned long long* ts_as_ull = (unsigned long long*)ts_address;
    unsigned long long old_ts = *ts_as_ull;
    unsigned long long assumed_ts;

    do {
        assumed_ts = old_ts;
        int64_t old_ts_val = (int64_t)assumed_ts;

        // Only update if new timestamp is earlier
        if (new_ts >= old_ts_val) {
            break;
        }

        // Try to update timestamp
        old_ts = atomicCAS(ts_as_ull, assumed_ts, (unsigned long long)new_ts);

        // If successful, update price (not atomic, but timestamp guarantees uniqueness)
        if (old_ts == assumed_ts) {
            *price_address = new_price;
        }
    } while (assumed_ts != old_ts);
}

__device__ inline void atomicMaxTimestampAndPrice(
    int64_t* ts_address,
    double* price_address,
    int64_t new_ts,
    double new_price
) {
    unsigned long long* ts_as_ull = (unsigned long long*)ts_address;
    unsigned long long old_ts = *ts_as_ull;
    unsigned long long assumed_ts;

    do {
        assumed_ts = old_ts;
        int64_t old_ts_val = (int64_t)assumed_ts;

        // Only update if new timestamp is later
        if (new_ts <= old_ts_val) {
            break;
        }

        // Try to update timestamp
        old_ts = atomicCAS(ts_as_ull, assumed_ts, (unsigned long long)new_ts);

        // If successful, update price
        if (old_ts == assumed_ts) {
            *price_address = new_price;
        }
    } while (assumed_ts != old_ts);
}

// ============================================================================
// Hash Function
// ============================================================================

/**
 * Fast hash function for bucket IDs
 *
 * Uses multiplicative hashing with prime number for good distribution.
 */
__device__ inline int32_t hash_bucket_id(int32_t bucket_id) {
    // Multiplicative hash with prime (avoids clustering)
    return (bucket_id * 2654435761u) & (HASH_TABLE_SIZE - 1);
}

// ============================================================================
// Kernel 1: Parallel Binning (Unchanged from aggregation.cu)
// ============================================================================

/**
 * Bin trades into timestamp buckets (fully parallel, no contention)
 *
 * Each thread maps one trade to its candle bucket based on timeframe.
 *
 * @param timestamps      Trade timestamps in milliseconds (i64 array)
 * @param bucket_ids      Output: Bucket ID for each trade (i32 array)
 * @param n_trades        Number of trades
 * @param timeframe_ms    Candle timeframe in milliseconds (e.g., 300000 for 5m)
 */
extern "C" __global__ void bin_trades_kernel(
    const int64_t* timestamps,
    int32_t* bucket_ids,
    int32_t n_trades,
    int64_t timeframe_ms
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_trades) {
        // Convert timestamp to bucket ID
        int64_t ts = timestamps[idx];
        bucket_ids[idx] = (int32_t)(ts / timeframe_ms);
    }
}

// ============================================================================
// Kernel 2: Hash-Based Aggregation (NEW IMPLEMENTATION)
// ============================================================================

/**
 * Aggregate trades into OHLCV candles using shared memory hash table
 *
 * **Algorithm**:
 * 1. Each block maintains a hash table in shared memory (40KB)
 * 2. Threads process trades and update hash table entries
 * 3. At end of block, flush hash table to global memory
 *
 * **Performance**:
 * - Shared memory atomics: 10-20x faster than global memory
 * - Reduced memory bandwidth (only final results written to global)
 * - Better cache locality (hash table fits in shared memory)
 *
 * @param timestamps       Trade timestamps (i64, for open/close tracking)
 * @param prices           Trade prices (f32 array)
 * @param volumes          Trade volumes (f32 array)
 * @param bucket_ids       Bucket ID for each trade (i32, from Pass 1)
 * @param n_trades         Number of trades
 * @param out_timestamps   Output: Candle timestamps (i64 array, size n_candles)
 * @param out_open         Output: Open prices (f32 array, size n_candles)
 * @param out_high         Output: High prices (f32 array, size n_candles)
 * @param out_low          Output: Low prices (f32 array, size n_candles, init to +inf)
 * @param out_close        Output: Close prices (f32 array, size n_candles)
 * @param out_volume       Output: Volumes (f32 array, size n_candles)
 * @param out_num_trades   Output: Trade counts (i32 array, size n_candles)
 * @param bucket_to_idx    Mapping: bucket_id → candle_index (i32 array, size max_bucket_id)
 * @param timeframe_ms     Candle timeframe (for timestamp reconstruction)
 *
 * ## Launch Configuration
 *
 * - Block size: 256 threads (good occupancy)
 * - Grid size: ceil(n_trades / 256)
 * - Shared memory: 40KB + 1KB (warp reduction buffers)
 */
/*
 * COMMENTED OUT: Hash kernel with __shared__ memory causes PTX loading failure with JIT compilation
 * on sm_89 architecture. Use aggregate_ohlcv_direct_kernel instead.
 *
extern "C" __global__ void aggregate_ohlcv_hash_kernel(
    const int64_t* timestamps,
    const double* prices,
    const double* volumes,
    const int32_t* bucket_ids,
    int32_t n_trades,
    int64_t* out_timestamps,
    double* out_open,
    double* out_high,
    double* out_low,
    double* out_close,
    double* out_volume,
    int32_t* out_num_trades,
    const int32_t* bucket_to_idx,
    int64_t timeframe_ms
) {
    // Shared memory hash table (40KB)
    __shared__ HashEntry hash_table[HASH_TABLE_SIZE];

    int32_t tid = threadIdx.x;
    int32_t idx = blockIdx.x * blockDim.x + tid;

    // Initialize hash table (each thread initializes multiple entries)
    for (int32_t i = tid; i < HASH_TABLE_SIZE; i += blockDim.x) {
        hash_table[i].bucket_id = HASH_EMPTY_BUCKET;
        hash_table[i].first_ts = LLONG_MAX;
        hash_table[i].last_ts = LLONG_MIN;
        hash_table[i].first_price = 0.0f;
        hash_table[i].last_price = 0.0f;
        hash_table[i].high = -CUDART_INF;
        hash_table[i].low = CUDART_INF;
        hash_table[i].volume = 0.0f;
        hash_table[i].num_trades = 0;
    }

    __syncthreads();  // Wait for initialization

    // Process trades assigned to this block
    if (idx < n_trades) {
        int32_t bucket_id = bucket_ids[idx];
        int64_t ts = timestamps[idx];
        double price = prices[idx];
        double volume = volumes[idx];

        // Hash bucket_id to find slot in hash table
        int32_t hash_idx = hash_bucket_id(bucket_id);

        // Linear probing for collision resolution
        int32_t probe_count = 0;
        while (probe_count < HASH_TABLE_SIZE) {
            int32_t current_bucket = atomicCAS(
                &hash_table[hash_idx].bucket_id,
                HASH_EMPTY_BUCKET,
                bucket_id
            );

            // Found empty slot or matching bucket
            if (current_bucket == HASH_EMPTY_BUCKET || current_bucket == bucket_id) {
                // Update OHLCV in shared memory
                atomicMinTimestampAndPrice(
                    &hash_table[hash_idx].first_ts,
                    &hash_table[hash_idx].first_price,
                    ts,
                    price
                );

                atomicMaxTimestampAndPrice(
                    &hash_table[hash_idx].last_ts,
                    &hash_table[hash_idx].last_price,
                    ts,
                    price
                );

                atomicMaxDouble(&hash_table[hash_idx].high, price);
                atomicMinDouble(&hash_table[hash_idx].low, price);
                atomicAdd(&hash_table[hash_idx].volume, volume);
                atomicAdd(&hash_table[hash_idx].num_trades, 1);

                break;  // Successfully updated
            }

            // Collision - try next slot (linear probing)
            hash_idx = (hash_idx + 1) & (HASH_TABLE_SIZE - 1);
            probe_count++;
        }

        // If probe_count == HASH_TABLE_SIZE, hash table is full!
        // This shouldn't happen if HASH_TABLE_SIZE is large enough.
        // TODO: Add overflow handling (spill to global memory)
    }

    __syncthreads();  // Wait for all threads to finish aggregation

    // Flush hash table to global memory (only one warp does this)
    if (tid < 32) {  // Warp 0
        for (int32_t i = tid; i < HASH_TABLE_SIZE; i += 32) {
            if (hash_table[i].bucket_id != HASH_EMPTY_BUCKET) {
                int32_t bucket_id = hash_table[i].bucket_id;
                int32_t candle_idx = bucket_to_idx[bucket_id];

                // Write to global memory (no contention across blocks)
                if (candle_idx >= 0) {
                    out_timestamps[candle_idx] = bucket_id * timeframe_ms;
                    out_open[candle_idx] = hash_table[i].first_price;
                    out_high[candle_idx] = hash_table[i].high;
                    out_low[candle_idx] = hash_table[i].low;
                    out_close[candle_idx] = hash_table[i].last_price;
                    out_volume[candle_idx] = hash_table[i].volume;
                    out_num_trades[candle_idx] = hash_table[i].num_trades;
                }
            }
        }
    }
}
*/

// ============================================================================
// Kernel 3: Simplified Aggregation (Direct Global Memory, for comparison)
// ============================================================================

/**
 * Fallback kernel using global memory atomics (no shared memory optimization)
 *
 * Useful for:
 * - Benchmarking hash-based vs direct approach
 * - Small datasets where shared memory overhead isn't worth it
 * - Validation (simpler implementation)
 */
extern "C" __global__ void aggregate_ohlcv_direct_kernel(
    const int64_t* timestamps,
    const double* prices,
    const double* volumes,
    const int32_t* bucket_ids,
    int32_t n_trades,
    int64_t* out_timestamps,
    double* out_open,
    double* out_high,
    double* out_low,
    double* out_close,
    double* out_volume,
    int32_t* out_num_trades,
    const int32_t* bucket_to_idx,
    int64_t timeframe_ms
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_trades) {
        int32_t bucket_id = bucket_ids[idx];
        int32_t candle_idx = bucket_to_idx[bucket_id];

        if (candle_idx >= 0) {
            int64_t ts = timestamps[idx];
            double price = prices[idx];
            double volume = volumes[idx];

            // Update OHLCV using global memory atomics
            // Note: This is slower than shared memory but simpler
            atomicMaxDouble(&out_high[candle_idx], price);
            atomicMinDouble(&out_low[candle_idx], price);
            atomicAdd(&out_volume[candle_idx], volume);
            atomicAdd(&out_num_trades[candle_idx], 1);

            // Open/close require timestamp tracking (use helper functions)
            // For now, compute on CPU (same as aggregation.cu)
            // TODO: Add GPU-based first/last tracking with atomics
        }
    }
}

// ============================================================================
// Kernel 4: Post-Aggregation Quantization (INT8 Compression)
// ============================================================================

/**
 * Quantize float32 OHLCV arrays to int8 using dynamic range
 *
 * Each feature (O, H, L, C, V) is quantized independently based on its range.
 * This preserves relative differences while achieving 4x compression.
 *
 * @param in_values     Input float32 array
 * @param out_values    Output int8 array
 * @param n             Array length
 * @param min_val       Minimum value in array (computed beforehand)
 * @param max_val       Maximum value in array (computed beforehand)
 *
 * Quantization formula:
 *   out = (in - min) / (max - min) * 255
 *   out = clamp(out, 0, 255)
 */
extern "C" __global__ void quantize_to_int8_kernel(
    const float* in_values,
    int8_t* out_values,
    int32_t n,
    float min_val,
    float max_val
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float val = in_values[idx];
        float range = max_val - min_val;

        // Handle edge case: constant values
        if (range < 1e-6f) {
            out_values[idx] = 127;  // Middle of range
        } else {
            // Normalize to [0, 1] then scale to [0, 255]
            float normalized = (val - min_val) / range;
            int32_t quantized = (int32_t)(normalized * 255.0f);

            // Clamp to [0, 255]
            quantized = max(0, min(255, quantized));

            // Convert to signed int8 (subtract 128 for zero-centered)
            out_values[idx] = (int8_t)(quantized - 128);
        }
    }
}

/**
 * Dequantize int8 back to float32 (for validation)
 */
extern "C" __global__ void dequantize_from_int8_kernel(
    const int8_t* in_values,
    float* out_values,
    int32_t n,
    float min_val,
    float max_val
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int8_t quantized = in_values[idx];
        float range = max_val - min_val;

        // Convert from signed int8 to [0, 255]
        int32_t unsigned_val = (int32_t)quantized + 128;

        // Denormalize: [0, 255] → [min, max]
        float normalized = (float)unsigned_val / 255.0f;
        out_values[idx] = min_val + normalized * range;
    }
}
