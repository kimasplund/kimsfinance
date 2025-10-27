//! Trait system for custom candle aggregation
//!
//! Extends the persistent kernel pattern to support candle generation from trade data.

use super::super::persistent::PersistentIndicator;

/// Trait for candle aggregators that convert trade data into OHLCV candles
///
/// Extends `PersistentIndicator` to provide candle-specific functionality while
/// maintaining compatibility with the persistent kernel batch execution infrastructure.
///
/// # Type Parameters
///
/// - `InputData`: Input data type (typically `TradeData` or concatenated buffers)
/// - `OutputCandle`: Output candle type (typically `OHLCVCandle` or variants)
///
/// # Design Philosophy
///
/// This trait follows the "interface-first" design from the persistent kernel pattern:
/// - Reuses existing batch execution infrastructure (`execute_batch`)
/// - Generic over input/output types (extensible to new candle types)
/// - Static dispatch for zero-cost abstraction
/// - Consistent API across all candle aggregators
///
/// # Example Implementation
///
/// ```rust,ignore
/// pub struct TimeBarAggregator;
///
/// #[repr(C)]
/// pub struct TimeBarParams {
///     pub interval_seconds: i32,
/// }
///
/// impl PersistentIndicator for TimeBarAggregator {
///     type Params = TimeBarParams;
///
///     fn kernel_source() -> &'static str {
///         TIME_BAR_KERNEL
///     }
///
///     fn kernel_name() -> &'static str {
///         "persistent_time_bars_kernel"
///     }
///
///     fn num_inputs() -> usize { 3 } // timestamp, price, volume
///     fn num_outputs() -> usize { 5 } // O, H, L, C, V
/// }
///
/// impl CandleAggregator for TimeBarAggregator {
///     type InputData = TradeData;
///     type OutputCandle = OHLCVCandle;
///
///     fn supports_streaming() -> bool {
///         true  // Time bars can be computed incrementally
///     }
/// }
/// ```
///
/// # Usage with Persistent Kernels
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::candles::*;
/// use kimsfinance_core::gpu::persistent::{execute_batch, TaskBatch};
///
/// // Create batch of time bar aggregation tasks
/// let mut batch = TaskBatch::<TimeBarAggregator>::new();
/// batch.add_task(btc_trades.concat_buffers(), TimeBarParams { interval_seconds: 60 });
/// batch.add_task(eth_trades.concat_buffers(), TimeBarParams { interval_seconds: 60 });
///
/// // Execute all tasks with single kernel launch (90% overhead reduction!)
/// let device = GpuDevice::new()?;
/// let candles = execute_batch(&device, &batch)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait CandleAggregator: PersistentIndicator {
    /// Input data type (e.g., `TradeData`)
    type InputData;

    /// Output candle type (e.g., `OHLCVCandle`)
    type OutputCandle;

    /// Whether this aggregator supports streaming/incremental updates
    ///
    /// Streaming aggregators can compute new candles as trades arrive without
    /// reprocessing the entire history.
    ///
    /// # Returns
    ///
    /// - `true`: Supports incremental updates (e.g., time bars, volume bars)
    /// - `false`: Requires full recomputation (e.g., Renko, Heikin-Ashi with lookback)
    ///
    /// # Default
    ///
    /// Returns `false` for safety (conservative default)
    fn supports_streaming() -> bool {
        false
    }

    /// Number of input time series required
    ///
    /// # Returns
    ///
    /// - Time bars: 3 (timestamp, price, volume)
    /// - Heikin-Ashi: 4 (open, high, low, close)
    /// - Renko: 2 (timestamp, price)
    ///
    /// # Note
    ///
    /// This is separate from `PersistentIndicator::num_inputs()` which returns
    /// the number of buffers per task. This method describes the logical time series.
    fn num_input_series() -> usize {
        Self::num_inputs()
    }

    /// Number of output time series produced
    ///
    /// # Returns
    ///
    /// - Standard OHLCV: 5 (open, high, low, close, volume)
    /// - Heikin-Ashi: 4 (ha_open, ha_high, ha_low, ha_close)
    /// - Renko: 3 (brick_price, direction, timestamp)
    ///
    /// # Note
    ///
    /// This is separate from `PersistentIndicator::num_outputs()` which describes
    /// the GPU buffer layout.
    fn num_output_series() -> usize {
        Self::num_outputs()
    }

    /// Expected compression ratio (trades per candle)
    ///
    /// Used for buffer pre-allocation. If unknown, return 1 (no compression).
    ///
    /// # Returns
    ///
    /// Estimated trades per output candle:
    /// - Time bars (1m on active markets): ~100-1000 trades/candle
    /// - Volume bars (100 BTC): ~50-500 trades/candle
    /// - Tick bars (100 trades): exactly 100 trades/candle
    ///
    /// # Default
    ///
    /// Returns 1 (no compression assumed, safe but may over-allocate)
    fn expected_compression_ratio() -> usize {
        1
    }
}

/// Marker trait for candle aggregators that process raw trade data
///
/// Distinguishes trade-to-candle aggregators from candle-to-candle transformations.
///
/// # Examples
///
/// - `TimeBarAggregator` ✅ (trades → time bars)
/// - `VolumeBarAggregator` ✅ (trades → volume bars)
/// - `HeikinAshiAggregator` ❌ (candles → modified candles)
pub trait TradeBasedAggregator: CandleAggregator {}

/// Marker trait for candle aggregators that transform existing candles
///
/// Distinguishes candle-to-candle transformations from trade-to-candle aggregations.
///
/// # Examples
///
/// - `HeikinAshiAggregator` ✅ (OHLCV → Heikin-Ashi)
/// - `TimeBarAggregator` ❌ (trades → OHLCV)
pub trait CandleBasedAggregator: CandleAggregator {}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementations for testing trait bounds

    struct MockTradeAggregator;
    struct MockCandleTransform;

    #[derive(Debug, Clone, Copy)]
    struct MockParams {
        period: i32,
    }

    impl PersistentIndicator for MockTradeAggregator {
        type Params = MockParams;

        fn kernel_source() -> &'static str {
            "mock_kernel_source"
        }

        fn kernel_name() -> &'static str {
            "mock_kernel"
        }

        fn num_outputs() -> usize {
            5
        }
    }

    impl CandleAggregator for MockTradeAggregator {
        type InputData = ();
        type OutputCandle = ();

        fn supports_streaming() -> bool {
            true
        }

        fn expected_compression_ratio() -> usize {
            100
        }
    }

    impl TradeBasedAggregator for MockTradeAggregator {}

    impl PersistentIndicator for MockCandleTransform {
        type Params = ();

        fn kernel_source() -> &'static str {
            "mock_transform_source"
        }

        fn kernel_name() -> &'static str {
            "mock_transform"
        }

        fn num_outputs() -> usize {
            4
        }
    }

    impl CandleAggregator for MockCandleTransform {
        type InputData = ();
        type OutputCandle = ();

        fn supports_streaming() -> bool {
            false
        }
    }

    impl CandleBasedAggregator for MockCandleTransform {}

    #[test]
    fn test_candle_aggregator_defaults() {
        assert_eq!(MockCandleTransform::expected_compression_ratio(), 1);
        assert_eq!(MockCandleTransform::num_input_series(), 1);
        assert_eq!(MockCandleTransform::num_output_series(), 4);
    }

    #[test]
    fn test_candle_aggregator_overrides() {
        assert!(MockTradeAggregator::supports_streaming());
        assert_eq!(MockTradeAggregator::expected_compression_ratio(), 100);
    }

    #[test]
    fn test_trait_bounds() {
        // Verify that trait implementations compile correctly
        fn requires_trade_based<T: TradeBasedAggregator>() {}
        fn requires_candle_based<T: CandleBasedAggregator>() {}

        requires_trade_based::<MockTradeAggregator>();
        requires_candle_based::<MockCandleTransform>();
    }
}
