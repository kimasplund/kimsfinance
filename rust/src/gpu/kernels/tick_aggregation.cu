/**
 * GPU Tick-Level Aggregation (Sorted Boundary Detection)
 *
 * Aggregates timestamp-sorted trade streams into OHLCV candles in a single
 * pass with zero atomics for open/close and single hardware atomics for
 * high/low/volume/count.
 *
 * ## Algorithm
 *
 * Exchange tick data is timestamp-sorted. The host verifies this with
 * `check_sorted_kernel` (unsorted input falls back to the CPU aggregator).
 * For a sorted stream, candle membership changes are detectable locally:
 *
 * - bucket(i)   = timestamps[i] / timeframe_ms          (i64 math throughout)
 * - candle c    = bucket(i) - first_bucket              (dense i32 index)
 * - bucket(i) != bucket(i-1) (or i == 0)   => trade i opens candle c
 * - bucket(i) != bucket(i+1) (or i == n-1) => trade i closes candle c
 *
 * Open/close/timestamp writes are therefore race-free plain stores: exactly
 * one trade per candle satisfies each boundary predicate. High/low use one
 * hardware atomicMax/atomicMin each on a monotonic ordered-uint image of the
 * f32 price (see encoding below). Volume uses native atomicAdd(float).
 *
 * ## Precision
 *
 * All price/volume math is f32. Ada (sm_89) executes FP64 at 1/64 the FP32
 * rate, and the input feed is f32 to begin with, so widening adds cost
 * without adding information. Bucket/timestamp math stays in i64 (exact).
 *
 * ## Ordered-uint encoding of f32 (layout contract with Rust host)
 *
 * encode(x): b = bits(x); b has sign bit clear (x >= 0.0 or +NaN payloads)
 *            -> b | 0x80000000, otherwise -> ~b
 * decode(e): e has top bit set -> bits = e & 0x7FFFFFFF, otherwise bits = ~e
 *
 * Under *unsigned* integer comparison the encoded values are strictly
 * monotonic in float order (negatives handled, -0.0 < +0.0). Identity values
 * for the reductions:
 *   encoded(-inf) = 0x007FFFFF  (high buffer init, neutral for atomicMax)
 *   encoded(+inf) = 0xFF800000  (low  buffer init, neutral for atomicMin)
 * These constants and the decode transform are mirrored in
 * rust/src/gpu/tick_aggregation.rs -- keep them in sync.
 *
 * ## NVRTC compatibility
 *
 * No include directives, extern "C" __global__ entry points only, no shared
 * memory (shared-memory kernels in this file previously failed PTX module
 * loading under NVRTC JIT on sm_89).
 */

// ============================================================================
// Type Definitions (NVRTC built-in widths, no includes needed)
// ============================================================================

typedef signed char int8_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define LLONG_MAX 9223372036854775807LL
#define LLONG_MIN (-9223372036854775807LL - 1LL)

// Ordered-uint encodings of +/-infinity (see header layout contract)
#define ENCODED_NEG_INF 0x007FFFFFu
#define ENCODED_POS_INF 0xFF800000u

// ============================================================================
// Ordered-uint encoding helpers
// ============================================================================

/**
 * Map f32 to a uint32 whose unsigned ordering matches float ordering.
 *
 * Non-negative floats (sign bit clear): set the top bit -> [0x80000000, ...]
 * Negative floats (sign bit set): bitwise NOT reverses their ordering and
 * clears the top bit -> [0x0, 0x7FFFFFFF].
 */
__device__ __forceinline__ uint32_t float_to_ordered_uint(float x) {
    uint32_t b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// ============================================================================
// Warp shuffle helper (i64 via two 32-bit shuffles)
// ============================================================================

/**
 * Warp shuffle-down of an i64 value built from two 32-bit shuffles.
 *
 * NVRTC provides the 32-bit __shfl_down_sync overloads as compiler builtins;
 * the 64-bit overloads are SDK-header inline functions (sm_30_intrinsics.hpp)
 * that NVRTC is not guaranteed to pre-include, so the halves are shuffled
 * explicitly and reassembled in unsigned math (well-defined for negatives).
 */
__device__ __forceinline__ int64_t shfl_down_i64(uint32_t mask, int64_t v, int offset) {
    int32_t lo = (int32_t)(uint32_t)((uint64_t)v & 0xFFFFFFFFu);
    int32_t hi = (int32_t)(uint32_t)((uint64_t)v >> 32);
    lo = __shfl_down_sync(mask, lo, offset);
    hi = __shfl_down_sync(mask, hi, offset);
    return (int64_t)(((uint64_t)(uint32_t)hi << 32) | (uint64_t)(uint32_t)lo);
}

// ============================================================================
// Kernel 1: Parallel Binning (standalone utility, rebased indices)
// ============================================================================

/**
 * Bin trades into rebased timestamp-bucket indices (fully parallel).
 *
 * Bucket math is i64; only the index rebased to `first_bucket` is narrowed
 * to i32, which cannot overflow when the candle range fits i32 (the host
 * enforces this). Writing the raw bucket `ts / timeframe_ms` as i32 would
 * overflow for epoch-millisecond timestamps with sub-second timeframes.
 *
 * Not launched by the main aggregation path (which computes buckets inline);
 * retained as a standalone utility kernel.
 *
 * @param timestamps      Trade timestamps in milliseconds (i64 array)
 * @param bucket_indices  Output: rebased bucket index per trade (i32 array)
 * @param n_trades        Number of trades
 * @param timeframe_ms    Candle timeframe in milliseconds (e.g., 300000 for 5m)
 * @param first_bucket    Smallest bucket id in the stream (from
 *                        compute_bucket_range_kernel)
 */
extern "C" __global__ void bin_trades_kernel(
    const int64_t* timestamps,
    int32_t* bucket_indices,
    int32_t n_trades,
    int64_t timeframe_ms,
    int64_t first_bucket
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_trades) {
        int64_t bucket = timestamps[idx] / timeframe_ms;
        bucket_indices[idx] = (int32_t)(bucket - first_bucket);
    }
}

// ============================================================================
// Kernel 2: Sortedness Check
// ============================================================================

/**
 * Set *out_unsorted_flag to 1 if any timestamps[i] < timestamps[i-1].
 *
 * The flag must be zero-initialized by the host. All racing writers store
 * the same 32-bit value, so a plain store is sufficient (no atomic needed).
 *
 * @param timestamps        Trade timestamps (i64 array)
 * @param out_unsorted_flag Output flag (single i32, host-initialized to 0)
 * @param n_trades          Number of trades
 */
extern "C" __global__ void check_sorted_kernel(
    const int64_t* timestamps,
    int32_t* out_unsorted_flag,
    int32_t n_trades
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 1 && idx < n_trades) {
        if (timestamps[idx] < timestamps[idx - 1]) {
            *out_unsorted_flag = 1;
        }
    }
}

// ============================================================================
// Kernel 3: Bucket Range Reduction
// ============================================================================

/**
 * Grid-stride min/max reduction over bucket = ts / timeframe_ms.
 *
 * Produces the first and last bucket ids so the host can size the dense
 * candle range (n_candles = last - first + 1) without copying per-trade
 * bucket ids back to the host.
 *
 * Reduction strategy: per-thread local min/max over a grid-stride loop,
 * then a warp shuffle reduction (no shared memory, see header note), then
 * one atomicMin/atomicMax per warp on the global results.
 *
 * @param timestamps       Trade timestamps (i64 array)
 * @param n_trades         Number of trades
 * @param timeframe_ms     Candle timeframe in milliseconds
 * @param out_first_bucket Output (single i64, host-initialized to LLONG_MAX)
 * @param out_last_bucket  Output (single i64, host-initialized to LLONG_MIN)
 *
 * Launch: 256-thread blocks; grid capped by the host (grid-stride loop
 * covers any n_trades).
 */
extern "C" __global__ void compute_bucket_range_kernel(
    const int64_t* timestamps,
    int32_t n_trades,
    int64_t timeframe_ms,
    int64_t* out_first_bucket,
    int64_t* out_last_bucket
) {
    int64_t local_min = LLONG_MAX;
    int64_t local_max = LLONG_MIN;

    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_trades;
         i += stride) {
        int64_t bucket = timestamps[i] / timeframe_ms;
        if (bucket < local_min) local_min = bucket;
        if (bucket > local_max) local_max = bucket;
    }

    // Warp shuffle reduction. Block size is a multiple of 32, so every warp
    // is full and the full mask is valid; threads with no elements hold the
    // neutral LLONG_MAX/LLONG_MIN values.
    for (int offset = 16; offset > 0; offset >>= 1) {
        int64_t other_min = shfl_down_i64(0xFFFFFFFFu, local_min, offset);
        int64_t other_max = shfl_down_i64(0xFFFFFFFFu, local_max, offset);
        if (other_min < local_min) local_min = other_min;
        if (other_max > local_max) local_max = other_max;
    }

    if ((threadIdx.x & 31) == 0 && local_min != LLONG_MAX) {
        atomicMin(out_first_bucket, local_min);
        atomicMax(out_last_bucket, local_max);
    }
}

// ============================================================================
// Kernel 4: High/Low Buffer Initialization
// ============================================================================

/**
 * Initialize encoded high/low buffers to the reduction identities.
 *
 * Required because zero-fill is NOT neutral for the ordered-uint min/max
 * reductions (0x00000000 decodes to NaN-payload territory below -inf, which
 * would poison atomicMin on the low buffer).
 *
 * @param out_high_enc Encoded high buffer (set to encoded(-inf))
 * @param out_low_enc  Encoded low buffer (set to encoded(+inf))
 * @param n_candles    Number of dense candle slots
 */
extern "C" __global__ void init_ohlcv_extrema_kernel(
    uint32_t* out_high_enc,
    uint32_t* out_low_enc,
    int32_t n_candles
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_candles) {
        out_high_enc[idx] = ENCODED_NEG_INF;
        out_low_enc[idx] = ENCODED_POS_INF;
    }
}

// ============================================================================
// Kernel 5: Sorted OHLCV Aggregation (f32, boundary detection)
// ============================================================================

/**
 * Aggregate a timestamp-sorted trade stream into dense OHLCV candle slots.
 *
 * Preconditions (enforced by the host wrapper):
 * - timestamps are non-decreasing (check_sorted_kernel passed)
 * - first_bucket = min(ts / timeframe_ms) over the stream
 * - candle range (last_bucket - first_bucket + 1) fits the output buffers
 * - out_high_enc/out_low_enc initialized by init_ohlcv_extrema_kernel
 * - out_volume/out_num_trades zero-initialized
 *
 * Open/close/timestamps: race-free plain stores at bucket boundaries
 * (exactly one trade per candle satisfies each predicate on a sorted
 * stream). Ties in timestamp are fine: boundaries compare bucket ids, so
 * open/close follow stream order exactly like the CPU reference
 * (binance::aggregate_trades_to_candles).
 *
 * Empty candle slots (gaps in the bucket range) keep num_trades == 0 and
 * are filtered by the host during candle construction.
 *
 * @param timestamps     Trade timestamps (i64, sorted non-decreasing)
 * @param prices         Trade prices (f32)
 * @param volumes        Trade volumes (f32)
 * @param n_trades       Number of trades
 * @param timeframe_ms   Candle timeframe in milliseconds
 * @param first_bucket   Smallest bucket id (dense index origin)
 * @param out_timestamps Output: candle open times, bucket * timeframe_ms (i64)
 * @param out_open       Output: open prices (f32)
 * @param out_high_enc   Output: encoded high prices (uint32, ordered image)
 * @param out_low_enc    Output: encoded low prices (uint32, ordered image)
 * @param out_close      Output: close prices (f32)
 * @param out_volume     Output: volume sums (f32)
 * @param out_num_trades Output: trade counts (i32)
 */
extern "C" __global__ void aggregate_ohlcv_sorted_kernel(
    const int64_t* timestamps,
    const float* prices,
    const float* volumes,
    int32_t n_trades,
    int64_t timeframe_ms,
    int64_t first_bucket,
    int64_t* out_timestamps,
    float* out_open,
    uint32_t* out_high_enc,
    uint32_t* out_low_enc,
    float* out_close,
    float* out_volume,
    int32_t* out_num_trades
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_trades) {
        int64_t ts = timestamps[idx];
        int64_t bucket = ts / timeframe_ms;
        int32_t c = (int32_t)(bucket - first_bucket);
        float price = prices[idx];

        // First trade of the candle: plain stores, exactly one writer.
        if (idx == 0 || (timestamps[idx - 1] / timeframe_ms) != bucket) {
            out_open[c] = price;
            out_timestamps[c] = bucket * timeframe_ms;
        }

        // Last trade of the candle: plain store, exactly one writer.
        if (idx == n_trades - 1 || (timestamps[idx + 1] / timeframe_ms) != bucket) {
            out_close[c] = price;
        }

        uint32_t enc = float_to_ordered_uint(price);
        atomicMax(&out_high_enc[c], enc);
        atomicMin(&out_low_enc[c], enc);
        atomicAdd(&out_volume[c], volumes[idx]);
        atomicAdd(&out_num_trades[c], 1);
    }
}

// ============================================================================
// Kernel 6: Post-Aggregation Quantization (INT8 Compression)
// ============================================================================

/**
 * Quantize f32 arrays to raw 0-255 codes stored through the i8 ABI.
 *
 * Convention (shared with rust/src/gpu/quantization.rs and
 * kernels/quantize_int8.cu): the full byte range 0-255 is used and carried
 * through int8 via `(int8_t)(unsigned char)code`. A plain `(int8_t)` cast of
 * the float would compile to a saturating cvt and collapse codes 128-255 to
 * 127 (losing the top half of every feature range). Rounding (not
 * truncation) preserves a half-code of accuracy.
 *
 * Degenerate range (max ~= min): emit code 0 so dequantization returns
 * min_val, round-tripping the constant exactly.
 *
 * @param in_values  Input f32 array
 * @param out_values Output i8 array (raw 0-255 codes)
 * @param n          Array length
 * @param min_val    Minimum value in array (computed beforehand)
 * @param max_val    Maximum value in array (computed beforehand)
 */
extern "C" __global__ void quantize_to_int8_kernel(
    const float* in_values,
    int8_t* out_values,
    int32_t n,
    float min_val,
    float max_val
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float range = max_val - min_val;

        if (range < 1e-6f) {
            out_values[idx] = (int8_t)(unsigned char)0;
        } else {
            float normalized = (in_values[idx] - min_val) / range;
            int32_t quantized = __float2int_rn(normalized * 255.0f);

            if (quantized < 0) quantized = 0;
            if (quantized > 255) quantized = 255;

            out_values[idx] = (int8_t)(unsigned char)quantized;
        }
    }
}

/**
 * Dequantize raw 0-255 codes (stored as i8) back to f32.
 *
 * Reads the code through `(unsigned char)` to recover the raw 0-255 value
 * (a direct i8 read would interpret codes 128-255 as negative). For a
 * degenerate range the stored code is 0 and the result is min_val.
 */
extern "C" __global__ void dequantize_from_int8_kernel(
    const int8_t* in_values,
    float* out_values,
    int32_t n,
    float min_val,
    float max_val
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int32_t code = (int32_t)(unsigned char)in_values[idx];
        float range = max_val - min_val;
        out_values[idx] = fmaf((float)code * (1.0f / 255.0f), range, min_val);
    }
}
