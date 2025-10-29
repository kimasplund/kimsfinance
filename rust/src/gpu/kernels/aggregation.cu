/**
 * GPU Trade Aggregation Kernels
 *
 * High-performance OHLCV candle aggregation for large trade datasets.
 *
 * ## Performance Characteristics
 *
 * - **Binning**: O(n) fully parallel, no shared memory, coalesced access
 * - **Aggregation**: O(n) with atomic contention (low for typical data)
 * - **Expected speedup**: 5-10x vs CPU for >100K trades
 *
 * ## Memory Layout
 *
 * Inputs:  Structure-of-Arrays (SoA) for coalesced access
 * Outputs: Separate arrays per OHLCV field
 *
 * ## Atomic Operations
 *
 * - Open/Close: Use first/last trade (tracked via timestamp atomics)
 * - High/Low: atomicMax/atomicMin on double (requires atomicCAS)
 * - Volume: atomicAdd on double (native support on compute_60+)
 * - Count: atomicAdd on int (native support)
 */

// ============================================================================
// Atomic Helpers for Double-Precision
// ============================================================================

/**
 * Atomic maximum for double-precision floats
 *
 * Uses compare-and-swap (CAS) loop since there's no native atomicMax for doubles.
 * This is safe for OHLCV because:
 * - Contention is low (trades distributed across candles)
 * - NaN/Inf not present in financial data
 * - Eventual consistency is sufficient (all threads converge to true max)
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
 *
 * Uses compare-and-swap (CAS) loop similar to atomicMaxDouble.
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

// ============================================================================
// Kernel 1: Trade Binning
// ============================================================================

/**
 * Bin trades into timestamp buckets (fully parallel, no contention)
 *
 * Each thread maps one trade to its candle bucket based on timeframe.
 *
 * @param timestamps      Trade timestamps in milliseconds (f64 array)
 * @param bucket_ids      Output: Bucket ID for each trade (i32 array)
 * @param n_trades        Number of trades
 * @param timeframe_ms    Candle timeframe in milliseconds (e.g., 300000 for 5m)
 *
 * ## Performance
 *
 * - Memory-bound kernel (1 read, 1 write per thread)
 * - Coalesced access pattern (sequential memory access)
 * - No shared memory needed
 * - Expected bandwidth: ~80% of theoretical peak
 *
 * ## Grid Configuration
 *
 * - Block size: 256 threads (8 warps, good occupancy)
 * - Grid size: ceil(n_trades / 256)
 */
extern "C" __global__ void bin_trades_kernel(
    const double* timestamps,
    int* bucket_ids,
    int n_trades,
    long long timeframe_ms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_trades) {
        // Convert timestamp to bucket ID
        // Example: timestamp=1609459250000ms, timeframe=300000ms → bucket=5364864
        long long ts = (long long)timestamps[idx];
        bucket_ids[idx] = (int)(ts / timeframe_ms);
    }
}

// ============================================================================
// Kernel 2: OHLCV Aggregation with Atomics
// ============================================================================

/**
 * Aggregate trades into OHLCV candles using atomic operations
 *
 * Each thread processes one trade and atomically updates the corresponding candle.
 * This approach works well when trades are distributed across many candles
 * (typical for 1m-1h timeframes). For very short timeframes (<1s), contention
 * may increase.
 *
 * @param prices           Trade prices (f64 array)
 * @param quantities       Trade quantities (base asset, f64 array)
 * @param quote_quantities Trade quote quantities (USDT, f64 array)
 * @param bucket_mapping   Bucket index for each trade (i32 array)
 * @param n_trades         Number of trades
 * @param out_high         Output: High prices (f64 array, size n_candles)
 * @param out_low          Output: Low prices (f64 array, size n_candles, init to +inf)
 * @param out_volume       Output: Base volume (f64 array, size n_candles)
 * @param out_quote_volume Output: Quote volume (f64 array, size n_candles)
 * @param out_num_trades   Output: Trade counts (i32 array, size n_candles)
 *
 * ## Atomic Strategy
 *
 * - **High**: atomicMaxDouble (find max price)
 * - **Low**: atomicMinDouble (find min price) - must be initialized to +inf
 * - **Volume**: atomicAdd (sum quantities)
 * - **Quote Volume**: atomicAdd (sum quote quantities)
 * - **Num Trades**: atomicAdd (count trades)
 *
 * ## Note on Open/Close
 *
 * Open and close are computed on CPU because they require timestamp ordering:
 * - **Open**: First trade in bucket (min timestamp)
 * - **Close**: Last trade in bucket (max timestamp)
 *
 * **Why CPU?**
 * - Tracking min/max timestamp + associated price requires complex atomic logic
 * - CPU can easily group by bucket and find first/last trade
 * - Cost is minimal: O(n) scan on CPU vs GPU kernel overhead
 *
 * **Future optimization**: Use GPU sorting (thrust::sort) + parallel scan
 * for fully GPU-based open/close computation.
 *
 * ## Performance
 *
 * - Atomic contention: Low for typical data (trades spread across candles)
 * - Memory bandwidth: ~60-70% of theoretical (atomic overhead)
 * - Scalability: Linear with n_trades (until atomic contention saturates)
 * - Expected speedup: 5-10x vs CPU for >100K trades
 */
extern "C" __global__ void aggregate_ohlcv_kernel(
    const double* prices,
    const double* quantities,
    const double* quote_quantities,
    const int* bucket_mapping,
    int n_trades,
    double* out_high,
    double* out_low,
    double* out_volume,
    double* out_quote_volume,
    int* out_num_trades
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_trades) {
        // Get bucket index for this trade
        int bucket_idx = bucket_mapping[idx];

        // Load trade data
        double price = prices[idx];
        double quantity = quantities[idx];
        double quote_qty = quote_quantities[idx];

        // Update OHLCV using atomic operations
        // Note: Open and close are computed on CPU (requires timestamp ordering)

        // High: atomic max
        atomicMaxDouble(&out_high[bucket_idx], price);

        // Low: atomic min (must be initialized to +inf before kernel launch)
        atomicMinDouble(&out_low[bucket_idx], price);

        // Volume: atomic sum
        atomicAdd(&out_volume[bucket_idx], quantity);

        // Quote volume: atomic sum
        atomicAdd(&out_quote_volume[bucket_idx], quote_qty);

        // Trade count: atomic increment
        atomicAdd(&out_num_trades[bucket_idx], 1);
    }
}

/**
 * Simplified aggregation kernel (for testing/validation)
 *
 * This version only computes volume and trade count (no atomicMax/Min complexity).
 * Useful for benchmarking atomic contention vs pure summation.
 */
extern "C" __global__ void aggregate_volume_only_kernel(
    const double* quantities,
    const int* bucket_mapping,
    int n_trades,
    double* out_volume,
    int* out_num_trades
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_trades) {
        int bucket_idx = bucket_mapping[idx];
        double quantity = quantities[idx];

        atomicAdd(&out_volume[bucket_idx], quantity);
        atomicAdd(&out_num_trades[bucket_idx], 1);
    }
}
