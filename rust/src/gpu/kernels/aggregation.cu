/**
 * GPU Trade Aggregation Kernels
 *
 * High-performance OHLCV candle aggregation for large trade datasets.
 *
 * ## Pipeline (mirrors rust/src/gpu/aggregation.rs — keep in sync)
 *
 * 1. init_ohlcv_state_kernel : write ordered-encoded -inf/+inf into high/low
 * 2. bin_trades_kernel       : i64 timestamp -> dense candle index (rebased)
 * 3. aggregate_ohlcv_kernel  : single pass; high/low/volume/count via atomics,
 *                              open/close via segment-boundary detection
 *
 * ## Preconditions (enforced on the host in aggregation.rs)
 *
 * - Trades are sorted by timestamp (non-decreasing). Dense candle indices are
 *   therefore non-decreasing and each candle's trades occupy one contiguous
 *   index range, which makes segment-boundary open/close detection race-free
 *   (exactly one writer per open/close slot).
 * - n_candles = last_bucket - first_bucket + 1 fits in i32.
 *
 * ## Precision rationale (Ada FP64:FP32 = 1:64)
 *
 * Prices/volumes stay f64 to match the CPU reference (`CandleBuilder` in
 * rust/src/binance/trades.rs) bit-for-bit. These kernels are memory/atomic
 * bound, not FP-ALU bound: the only f64 arithmetic is atomicAdd(double)
 * (native since sm_60, limited by L2 atomic throughput, not the FP64 ALU).
 * High/low comparisons run as *integer* u64 atomics on an order-preserving
 * encoding of f64, so the slow FP64 ALU is not on the hot path at all.
 *
 * ## Ordered-int encoding of double (layout contract with aggregation.rs)
 *
 * encode(v) = sign-bit set ? ~bits(v) : bits(v) | 0x8000000000000000
 *
 * is strictly monotonic over all non-NaN doubles (financial data is NaN-free),
 * so native integer atomics replace the old CAS retry loops:
 *
 *   max(price) == decode(atomicMax(encoded_high, encode(price)))
 *   min(price) == decode(atomicMin(encoded_low,  encode(price)))
 *
 * Initialization values (written by init_ohlcv_state_kernel — NOT zero;
 * a zero-initialized high buffer floors at 0.0 and corrupts results for
 * negative-price instruments):
 *
 *   encode(-inf) = 0x000FFFFFFFFFFFFF  (high identity)
 *   encode(+inf) = 0xFFF0000000000000  (low  identity)
 *
 * Host-side decode lives in aggregation.rs (decode_ordered_u64) and MUST
 * stay in sync with this transform.
 *
 * ## Launch configuration (must match aggregation.rs)
 *
 * - 256 threads per block for all kernels
 * - init_ohlcv_state_kernel : grid = ceil(n_candles / 256)
 * - bin_trades_kernel       : grid = ceil(n_trades  / 256)
 * - aggregate_ohlcv_kernel  : grid = ceil(n_trades  / 256)
 *
 * NVRTC-compatible: no #include directives, extern "C" entry points only.
 */

// ============================================================================
// Ordered-int encoding helpers
// ============================================================================

// Order-preserving u64 image of an f64 (see header contract).
__device__ __forceinline__ unsigned long long ordered_encode_double(double v) {
    unsigned long long bits = (unsigned long long)__double_as_longlong(v);
    return (bits & 0x8000000000000000ULL) ? ~bits : (bits | 0x8000000000000000ULL);
}

// encode(-infinity): identity element for atomicMax on the encoded domain
#define ORDERED_ENCODED_NEG_INF 0x000FFFFFFFFFFFFFULL
// encode(+infinity): identity element for atomicMin on the encoded domain
#define ORDERED_ENCODED_POS_INF 0xFFF0000000000000ULL

// ============================================================================
// Kernel 0: High/Low State Initialization
// ============================================================================

/**
 * Initialize encoded high/low buffers to their atomic identity values.
 *
 * @param out_high_bits  Encoded high prices (u64 array, size n_candles)
 * @param out_low_bits   Encoded low prices  (u64 array, size n_candles)
 * @param n_candles      Number of dense candle slots
 */
extern "C" __global__ void init_ohlcv_state_kernel(
    unsigned long long* __restrict__ out_high_bits,
    unsigned long long* __restrict__ out_low_bits,
    int n_candles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_candles) {
        out_high_bits[idx] = ORDERED_ENCODED_NEG_INF;
        out_low_bits[idx] = ORDERED_ENCODED_POS_INF;
    }
}

// ============================================================================
// Kernel 1: Trade Binning (dense candle indices)
// ============================================================================

/**
 * Bin trades into dense candle indices (fully parallel, no contention).
 *
 * @param timestamps    Trade timestamps in epoch milliseconds (i64 array)
 * @param bucket_ids    Output: dense candle index per trade (i32 array)
 * @param n_trades      Number of trades
 * @param timeframe_ms  Candle timeframe in milliseconds (e.g., 300000 for 5m)
 * @param first_bucket  trades[0].timestamp_ms / timeframe_ms (host-computed)
 *
 * Truncating i64 division matches the CPU reference bucket math
 * (`trade.timestamp_ms / timeframe_ms` in rust/src/binance/trades.rs).
 * Rebasing by first_bucket yields dense candle indices starting at 0 that
 * fit in i32 even for sub-second timeframes on epoch-ms timestamps — the
 * raw quotient ts / timeframe_ms exceeds i32::MAX for timeframe_ms < ~800
 * on current timestamps and used to overflow the old (int) cast.
 */
extern "C" __global__ void bin_trades_kernel(
    const long long* __restrict__ timestamps,
    int* __restrict__ bucket_ids,
    int n_trades,
    long long timeframe_ms,
    long long first_bucket
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_trades) {
        bucket_ids[idx] = (int)(timestamps[idx] / timeframe_ms - first_bucket);
    }
}

// ============================================================================
// Kernel 2: OHLCV Aggregation (atomics + segment-boundary open/close)
// ============================================================================

/**
 * Aggregate trades into OHLCV candles in a single pass.
 *
 * Each thread processes one trade:
 * - High/Low: native u64 atomicMax/atomicMin on the order-preserving
 *   encoding (no CAS retry loop, correct for negative prices)
 * - Volume / Quote Volume: atomicAdd(double) (native on sm_60+)
 * - Trade count: atomicAdd(int)
 * - Open/Close: segment-boundary detection. Trades are time-ordered, so
 *   bucket indices are non-decreasing and each candle owns a contiguous
 *   trade range. Thread i compares bucket(i) with bucket(i-1): on a
 *   discontinuity it is the unique first trade of bucket(i) (writes open)
 *   and trade i-1 is the unique last trade of bucket(i-1) (writes close).
 *   Edges: thread 0 opens the first candle, thread n-1 closes the last.
 *   No atomics needed — exactly one writer per slot.
 *
 * Candle slots with no trades (time gaps) receive no writes and are
 * filtered on the host via out_num_trades == 0.
 *
 * @param prices           Trade prices (f64 array, size n_trades)
 * @param quantities       Trade base-asset quantities (f64 array)
 * @param quote_quantities Trade quote-asset quantities (f64 array)
 * @param bucket_ids       Dense candle index per trade (from bin_trades_kernel)
 * @param n_trades         Number of trades
 * @param out_high_bits    Encoded highs (u64, size n_candles, init: encode(-inf))
 * @param out_low_bits     Encoded lows  (u64, size n_candles, init: encode(+inf))
 * @param out_open         Open prices  (f64, size n_candles)
 * @param out_close        Close prices (f64, size n_candles)
 * @param out_volume       Base volume sums  (f64, size n_candles, zero-init)
 * @param out_quote_volume Quote volume sums (f64, size n_candles, zero-init)
 * @param out_num_trades   Trade counts (i32, size n_candles, zero-init)
 */
extern "C" __global__ void aggregate_ohlcv_kernel(
    const double* __restrict__ prices,
    const double* __restrict__ quantities,
    const double* __restrict__ quote_quantities,
    const int* __restrict__ bucket_ids,
    int n_trades,
    unsigned long long* __restrict__ out_high_bits,
    unsigned long long* __restrict__ out_low_bits,
    double* __restrict__ out_open,
    double* __restrict__ out_close,
    double* __restrict__ out_volume,
    double* __restrict__ out_quote_volume,
    int* __restrict__ out_num_trades
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_trades) {
        return;
    }

    int c = bucket_ids[idx];
    double price = prices[idx];
    unsigned long long encoded = ordered_encode_double(price);

    // High/Low as integer atomics on the order-preserving encoding.
    atomicMax(&out_high_bits[c], encoded);
    atomicMin(&out_low_bits[c], encoded);

    // Sums and counts.
    atomicAdd(&out_volume[c], quantities[idx]);
    atomicAdd(&out_quote_volume[c], quote_quantities[idx]);
    atomicAdd(&out_num_trades[c], 1);

    // Open/Close via segment boundaries (single writer per slot).
    if (idx == 0) {
        out_open[c] = price;
    } else {
        int c_prev = bucket_ids[idx - 1];
        if (c_prev != c) {
            out_open[c] = price;
            out_close[c_prev] = prices[idx - 1];
        }
    }
    if (idx == n_trades - 1) {
        out_close[c] = price;
    }
}
