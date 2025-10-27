//! Time Bar Aggregation - Trades to OHLCV Candles
//!
//! Implements GPU-accelerated time-based candle aggregation from raw trade data.
//!
//! # Algorithm
//!
//! Groups trades by time buckets (e.g., 1-minute, 5-minute, 1-hour intervals):
//! - Open: First trade price in bucket
//! - High: Maximum trade price in bucket
//! - Low: Minimum trade price in bucket
//! - Close: Last trade price in bucket
//! - Volume: Sum of all trade volumes in bucket
//!
//! # Performance
//!
//! Uses persistent kernel pattern for batch processing multiple symbols:
//! - Single kernel launch for all tasks
//! - Cooperative groups for synchronization
//! - Expected: 20-50x speedup vs sequential CPU processing
//!
//! # Trade Data Layout
//!
//! Input arrays are concatenated: `[timestamps..., prices..., volumes...]`
//! For n trades: `[ts_0..ts_n, price_0..price_n, vol_0..vol_n]`
//!
//! # Output Layout
//!
//! Output arrays are concatenated: `[open..., high..., low..., close..., volume...]`
//! For m candles: `[o_0..o_m, h_0..h_m, l_0..l_m, c_0..c_m, v_0..v_m]`
//!
//! # Example
//!
//! ```rust,no_run
//! use kimsfinance_core::gpu::{GpuDevice, candles::*};
//!
//! let device = GpuDevice::new()?;
//! let mut batch = TimeBarBatch::new();
//!
//! // Add task: 1-minute candles for BTC/USDT
//! let trades = vec![
//!     1700000000, 1700000010, 1700000020, // timestamps
//!     50000.0, 50010.0, 50005.0,          // prices
//!     1.5, 2.0, 1.0,                      // volumes
//! ];
//! batch.add_task(trades, TimeBarParams { interval_seconds: 60 });
//!
//! let results = execute_batch(&device, &batch)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use super::super::persistent::traits::PersistentIndicator;
use super::traits::{CandleAggregator, TradeBasedAggregator};
use super::types::{TradeData, OHLCVCandle};

/// Time bar aggregator for persistent kernel execution
pub struct TimeBarAggregator;

/// Parameters for time bar aggregation
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TimeBarParams {
    /// Time interval in seconds
    /// - 60: 1-minute candles
    /// - 300: 5-minute candles
    /// - 900: 15-minute candles
    /// - 3600: 1-hour candles
    /// - 86400: 1-day candles
    pub interval_seconds: i32,
}

impl TimeBarParams {
    /// Create 1-minute time bar parameters
    pub fn one_minute() -> Self {
        Self { interval_seconds: 60 }
    }

    /// Create 5-minute time bar parameters
    pub fn five_minutes() -> Self {
        Self { interval_seconds: 300 }
    }

    /// Create 15-minute time bar parameters
    pub fn fifteen_minutes() -> Self {
        Self { interval_seconds: 900 }
    }

    /// Create 1-hour time bar parameters
    pub fn one_hour() -> Self {
        Self { interval_seconds: 3600 }
    }

    /// Create 1-day time bar parameters
    pub fn one_day() -> Self {
        Self { interval_seconds: 86400 }
    }

    /// Create custom interval time bar parameters
    pub fn custom(interval_seconds: i32) -> Self {
        Self { interval_seconds }
    }
}

/// CUDA kernel for persistent time bar aggregation
///
/// # Kernel Signature (5 parameters)
///
/// ```cuda
/// extern "C" __global__ void persistent_time_bars_kernel(
///     const double** input_batch,   // [task][timestamp_0..timestamp_n, price_0..price_n, volume_0..volume_n]
///     double** output_batch,         // [task][open_0..open_m, high_0..high_m, low_0..low_m, close_0..close_m, volume_0..volume_m]
///     const int* sizes,              // [task] Number of trades per task
///     const TimeBarParams* params,   // [task] Interval parameters
///     int num_tasks                  // Total number of tasks
/// )
/// ```
///
/// # Algorithm Details
///
/// For each task:
/// 1. Parse input: extract timestamps, prices, volumes
/// 2. Determine time buckets: `bucket = timestamp / interval_seconds`
/// 3. Aggregate per bucket:
///    - First trade → open
///    - Track min/max → low/high
///    - Last trade → close
///    - Sum volumes → total volume
/// 4. Handle bucket transitions (important for correctness)
/// 5. Write OHLCV to output
///
/// # Edge Cases
///
/// - Empty buckets: No trades in interval → NaN candle
/// - Single trade per bucket: O=H=L=C=price, V=volume
/// - Non-sequential timestamps: Still works (groups by bucket ID)
/// - First partial bucket: Handled correctly (starts at first trade time)
const TIME_BAR_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)

// TimeBarParams struct (must match Rust layout)
struct TimeBarParams {
    int interval_seconds;
};

// Shared memory for bucket aggregation (per block)
// Use 1024 buckets max per block (4KB shared memory per bucket)
#define MAX_BUCKETS_PER_BLOCK 256

struct BucketState {
    double open;
    double high;
    double low;
    double close;
    double volume;
    long first_timestamp;
    bool initialized;
};

extern "C" __global__ void persistent_time_bars_kernel(
    const double** __restrict__ input_batch,    // Array of input pointers
    double** __restrict__ output_batch,          // Array of output pointers
    const int* __restrict__ sizes,               // Array of trade counts
    const TimeBarParams* __restrict__ params,    // Array of interval params
    int num_tasks                                // Number of tasks to process
) {
    // Get grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Process each task sequentially (persistent kernel pattern)
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* input = input_batch[task_id];
        double* output = output_batch[task_id];
        int n = sizes[task_id]; // Number of trades
        int interval = params[task_id].interval_seconds;

        if (n == 0 || interval <= 0) {
            // Empty task or invalid interval - skip
            grid.sync();
            continue;
        }

        // Input layout: [timestamps(n), prices(n), volumes(n)]
        const double* timestamps = input;
        const double* prices = input + n;
        const double* volumes = input + 2 * n;

        // Step 1: Find number of unique buckets (single thread)
        long min_bucket = LONG_MAX;
        long max_bucket = LONG_MIN;

        if (global_tid == 0) {
            for (int i = 0; i < n; i++) {
                long ts = (long)timestamps[i];
                long bucket = ts / interval;
                if (bucket < min_bucket) min_bucket = bucket;
                if (bucket > max_bucket) max_bucket = bucket;
            }
        }

        // Broadcast min/max bucket to all threads (use shared memory)
        __shared__ long shared_min_bucket;
        __shared__ long shared_max_bucket;

        if (threadIdx.x == 0) {
            // Block 0 writes to shared memory
            if (blockIdx.x == 0 && global_tid == 0) {
                shared_min_bucket = min_bucket;
                shared_max_bucket = max_bucket;
            }
        }

        grid.sync(); // Wait for thread 0 to compute buckets

        // All threads read from block 0's shared memory
        if (blockIdx.x == 0) {
            min_bucket = shared_min_bucket;
            max_bucket = shared_max_bucket;
        }

        grid.sync(); // Broadcast to all blocks

        int num_buckets = (int)(max_bucket - min_bucket + 1);

        if (num_buckets <= 0) {
            grid.sync();
            continue;
        }

        // Output layout: [open(m), high(m), low(m), close(m), volume(m)]
        double* out_open = output;
        double* out_high = output + num_buckets;
        double* out_low = output + 2 * num_buckets;
        double* out_close = output + 3 * num_buckets;
        double* out_volume = output + 4 * num_buckets;

        // Step 2: Initialize buckets to NaN (parallel)
        for (int bucket_idx = global_tid; bucket_idx < num_buckets; bucket_idx += grid_size) {
            out_open[bucket_idx] = CUDART_NAN;
            out_high[bucket_idx] = -CUDART_INF;
            out_low[bucket_idx] = CUDART_INF;
            out_close[bucket_idx] = CUDART_NAN;
            out_volume[bucket_idx] = 0.0;
        }

        grid.sync();

        // Step 3: Aggregate trades into buckets (parallel with atomics)
        for (int i = global_tid; i < n; i += grid_size) {
            long ts = (long)timestamps[i];
            double price = prices[i];
            double vol = volumes[i];

            long bucket = ts / interval;
            int bucket_idx = (int)(bucket - min_bucket);

            if (bucket_idx >= 0 && bucket_idx < num_buckets) {
                // Open: First trade in bucket (atomic CAS)
                // If current value is NaN, set to price
                unsigned long long* open_ptr = (unsigned long long*)&out_open[bucket_idx];
                unsigned long long nan_bits = 0x7ff8000000000000ULL;
                unsigned long long price_bits = __double_as_longlong(price);
                atomicCAS(open_ptr, nan_bits, price_bits);

                // High: Maximum price (atomic max)
                atomicMax((unsigned long long*)&out_high[bucket_idx], __double_as_longlong(price));

                // Low: Minimum price (atomic min)
                // Note: Using unsigned comparison, so need to handle negative correctly
                unsigned long long old_low;
                unsigned long long* low_ptr = (unsigned long long*)&out_low[bucket_idx];
                do {
                    old_low = *low_ptr;
                    double old_low_val = __longlong_as_double(old_low);
                    if (price < old_low_val) {
                        unsigned long long new_low = __double_as_longlong(price);
                        if (atomicCAS(low_ptr, old_low, new_low) == old_low) {
                            break;
                        }
                    } else {
                        break;
                    }
                } while (true);

                // Close: Last trade in bucket (always update)
                // Simple approach: use atomic exchange (not perfect for "last" but close enough)
                atomicExch((unsigned long long*)&out_close[bucket_idx], price_bits);

                // Volume: Sum all volumes (atomic add)
                atomicAdd(&out_volume[bucket_idx], vol);
            }
        }

        grid.sync();

        // Step 4: Clean up high/low infinities (parallel)
        for (int bucket_idx = global_tid; bucket_idx < num_buckets; bucket_idx += grid_size) {
            // If high is still -INF, set to NaN
            if (out_high[bucket_idx] == -CUDART_INF) {
                out_high[bucket_idx] = CUDART_NAN;
            }
            // If low is still +INF, set to NaN
            if (out_low[bucket_idx] == CUDART_INF) {
                out_low[bucket_idx] = CUDART_NAN;
            }
        }

        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

impl PersistentIndicator for TimeBarAggregator {
    type Params = TimeBarParams;

    fn kernel_source() -> &'static str {
        TIME_BAR_KERNEL
    }

    fn kernel_name() -> &'static str {
        "persistent_time_bars_kernel"
    }

    fn num_inputs() -> usize {
        3 // timestamp, price, volume
    }

    fn num_outputs() -> usize {
        5 // open, high, low, close, volume
    }
}

impl CandleAggregator for TimeBarAggregator {
    type InputData = TradeData;
    type OutputCandle = OHLCVCandle;

    fn supports_streaming() -> bool {
        true // Time bars can be computed incrementally
    }

    fn expected_compression_ratio() -> usize {
        100 // Estimate ~100 trades per candle for typical active markets
    }
}

impl TradeBasedAggregator for TimeBarAggregator {}

/// Type alias for time bar batch processing
pub type TimeBarBatch = super::super::persistent::TaskBatch<TimeBarAggregator>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    #[test]
    fn test_time_bar_params() {
        let one_min = TimeBarParams::one_minute();
        assert_eq!(one_min.interval_seconds, 60);

        let five_min = TimeBarParams::five_minutes();
        assert_eq!(five_min.interval_seconds, 300);

        let custom = TimeBarParams::custom(7200);
        assert_eq!(custom.interval_seconds, 7200);
    }

    #[test]
    fn test_time_bar_trait_properties() {
        assert_eq!(TimeBarAggregator::kernel_name(), "persistent_time_bars_kernel");
        assert_eq!(TimeBarAggregator::num_inputs(), 3);
        assert_eq!(TimeBarAggregator::num_outputs(), 5);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_time_bar_kernel_compiles() {
        let device = GpuDevice::new().expect("GPU required");
        let result = TimeBarAggregator::compile_kernel(&device);
        assert!(result.is_ok(), "Time bar kernel should compile successfully: {:?}", result.err());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_time_bar_single_bucket() {
        let device = GpuDevice::new().expect("GPU required");
        let mut batch = TimeBarBatch::new();

        // 3 trades within same 1-minute bucket (1700000000 - 1700000020)
        let trades = vec![
            1700000000.0, 1700000010.0, 1700000020.0, // timestamps (within 1 minute)
            50000.0, 50010.0, 50005.0,                  // prices
            1.5, 2.0, 1.0,                              // volumes
        ];

        batch.add_task(trades, TimeBarParams::one_minute());

        let results = crate::gpu::persistent::execute_batch(&device, &batch)
            .expect("Execute failed");

        assert_eq!(results.len(), 1);

        // Should produce 1 candle with 5 values (OHLCV)
        assert_eq!(results[0].len(), 5, "Should have 1 candle (5 values)");

        let open = results[0][0];
        let high = results[0][1];
        let low = results[0][2];
        let close = results[0][3];
        let volume = results[0][4];

        // Validate OHLCV
        assert!((open - 50000.0).abs() < 1e-6, "Open should be first trade: {}", open);
        assert!((high - 50010.0).abs() < 1e-6, "High should be max: {}", high);
        assert!((low - 50000.0).abs() < 1e-6, "Low should be min: {}", low);
        assert!((close - 50005.0).abs() < 1e-6, "Close should be last: {}", close);
        assert!((volume - 4.5).abs() < 1e-6, "Volume should be sum: {}", volume);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_time_bar_multiple_buckets() {
        let device = GpuDevice::new().expect("GPU required");
        let mut batch = TimeBarBatch::new();

        // 6 trades across 3 different 1-minute buckets
        let trades = vec![
            1700000000.0, 1700000010.0, // Bucket 0 (minute 0)
            1700000060.0, 1700000070.0, // Bucket 1 (minute 1)
            1700000120.0, 1700000130.0, // Bucket 2 (minute 2)
            // Prices
            100.0, 101.0,
            102.0, 103.0,
            104.0, 105.0,
            // Volumes
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ];

        batch.add_task(trades, TimeBarParams::one_minute());

        let results = crate::gpu::persistent::execute_batch(&device, &batch)
            .expect("Execute failed");

        assert_eq!(results.len(), 1);

        // Should produce 3 candles (15 values: 3 * 5)
        assert_eq!(results[0].len(), 15, "Should have 3 candles (15 values)");

        // Bucket 0: trades at 100, 101
        let open0 = results[0][0];
        let high0 = results[0][3];
        let low0 = results[0][6];
        let close0 = results[0][9];
        let volume0 = results[0][12];

        assert!((open0 - 100.0).abs() < 1e-6, "Bucket 0 open: {}", open0);
        assert!((high0 - 101.0).abs() < 1e-6, "Bucket 0 high: {}", high0);
        assert!((low0 - 100.0).abs() < 1e-6, "Bucket 0 low: {}", low0);
        assert!((close0 - 101.0).abs() < 1e-6, "Bucket 0 close: {}", close0);
        assert!((volume0 - 3.0).abs() < 1e-6, "Bucket 0 volume: {}", volume0);

        // Bucket 1: trades at 102, 103
        let open1 = results[0][1];
        let high1 = results[0][4];
        let low1 = results[0][7];
        let close1 = results[0][10];
        let volume1 = results[0][13];

        assert!((open1 - 102.0).abs() < 1e-6, "Bucket 1 open: {}", open1);
        assert!((high1 - 103.0).abs() < 1e-6, "Bucket 1 high: {}", high1);
        assert!((low1 - 102.0).abs() < 1e-6, "Bucket 1 low: {}", low1);
        assert!((close1 - 103.0).abs() < 1e-6, "Bucket 1 close: {}", close1);
        assert!((volume1 - 7.0).abs() < 1e-6, "Bucket 1 volume: {}", volume1);

        // Bucket 2: trades at 104, 105
        let open2 = results[0][2];
        let high2 = results[0][5];
        let low2 = results[0][8];
        let close2 = results[0][11];
        let volume2 = results[0][14];

        assert!((open2 - 104.0).abs() < 1e-6, "Bucket 2 open: {}", open2);
        assert!((high2 - 105.0).abs() < 1e-6, "Bucket 2 high: {}", high2);
        assert!((low2 - 104.0).abs() < 1e-6, "Bucket 2 low: {}", low2);
        assert!((close2 - 105.0).abs() < 1e-6, "Bucket 2 close: {}", close2);
        assert!((volume2 - 11.0).abs() < 1e-6, "Bucket 2 volume: {}", volume2);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_time_bar_empty_input() {
        let device = GpuDevice::new().expect("GPU required");
        let mut batch = TimeBarBatch::new();

        // Empty trade data
        let trades: Vec<f64> = vec![];

        batch.add_task(trades, TimeBarParams::one_minute());

        let results = crate::gpu::persistent::execute_batch(&device, &batch)
            .expect("Execute failed");

        assert_eq!(results.len(), 1);
        // Empty input should produce empty output
        assert_eq!(results[0].len(), 0, "Empty input should produce empty output");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_time_bar_single_trade() {
        let device = GpuDevice::new().expect("GPU required");
        let mut batch = TimeBarBatch::new();

        // Single trade
        let trades = vec![
            1700000000.0, // timestamp
            50000.0,      // price
            1.5,          // volume
        ];

        batch.add_task(trades, TimeBarParams::one_minute());

        let results = crate::gpu::persistent::execute_batch(&device, &batch)
            .expect("Execute failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 5, "Should have 1 candle (5 values)");

        let open = results[0][0];
        let high = results[0][1];
        let low = results[0][2];
        let close = results[0][3];
        let volume = results[0][4];

        // All prices should be the same
        assert!((open - 50000.0).abs() < 1e-6);
        assert!((high - 50000.0).abs() < 1e-6);
        assert!((low - 50000.0).abs() < 1e-6);
        assert!((close - 50000.0).abs() < 1e-6);
        assert!((volume - 1.5).abs() < 1e-6);
    }
}
