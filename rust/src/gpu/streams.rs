//! CUDA Stream Management for Concurrent Kernel Execution
//!
//! Manages 3 CUDA streams to execute GPU indicators concurrently based on their
//! computational complexity. This provides 15-30% throughput improvement by
//! overlapping kernel execution.
//!
//! # Stream Classification
//!
//! - **Stream 0 (Fast)**: < 5μs/candle - ROC, Williams %R, CCI, SMA, WMA, VWMA,
//!   Donchian, OBV, VWAP, Pivot Points
//! - **Stream 1 (Medium)**: 5-15μs/candle - RSI, ATR, Aroon, Bollinger Bands,
//!   EMA, DEMA, TEMA, HMA, Keltner, Elder Ray, CMF, MFI
//! - **Stream 2 (Slow)**: > 15μs/candle - Stochastic, MACD, TSI, Parabolic SAR,
//!   Volume Profile
//!
//! # Architecture
//!
//! ```text
//! StreamManager
//!   ├── stream_fast (Stream 0)    - Embarrassingly parallel ops
//!   ├── stream_medium (Stream 1)  - Rolling window ops
//!   └── stream_slow (Stream 2)    - Complex multi-pass ops
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::{GpuDevice, StreamManager, IndicatorSpeed};
//!
//! let device = GpuDevice::new()?;
//! let stream_mgr = StreamManager::new(Arc::new(device))?;
//!
//! // Launch ROC on fast stream
//! stream_mgr.launch_kernel(IndicatorSpeed::Fast, |stream| {
//!     roc_kernel.launch(stream, &config, (...))
//! })?;
//!
//! // Launch RSI on medium stream (concurrent with ROC)
//! stream_mgr.launch_kernel(IndicatorSpeed::Medium, |stream| {
//!     rsi_kernel.launch(stream, &config, (...))
//! })?;
//!
//! // Wait for all streams
//! stream_mgr.synchronize_all()?;
//! ```

use super::device::GpuDevice;
use cudarc::driver::CudaStream;
use std::sync::{Arc, OnceLock};

/// Process-wide StreamManager, constructed once on first use.
///
/// CUDA stream creation is cheap (~10μs) but the previous batch pipeline
/// re-created a full `GpuDevice` (including its pinned-memory pool) per chunk
/// just to obtain streams. Constructing the manager once amortizes that cost
/// across the process lifetime.
///
/// Safety/correctness note: cudarc 0.17 `CudaContext::new()` retains the CUDA
/// *primary* context (`cuDevicePrimaryCtxRetain`), so every `GpuDevice` handle
/// for device 0 shares the same underlying context. Streams created here are
/// therefore valid for kernels/modules loaded through any other `GpuDevice`
/// handle on device 0.
static GLOBAL_STREAM_MANAGER: OnceLock<StreamManager> = OnceLock::new();

/// Indicator execution speed classification
///
/// Based on empirical GPU kernel timing:
/// - Fast: < 5μs/candle (embarrassingly parallel)
/// - Medium: 5-15μs/candle (rolling windows, partial dependencies)
/// - Slow: > 15μs/candle (complex multi-pass, sequential bottlenecks)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndicatorSpeed {
    /// Fast indicators (< 5μs/candle): ROC, Williams %R, CCI
    Fast,

    /// Medium indicators (5-15μs/candle): RSI, ATR, Aroon, Bollinger Bands
    Medium,

    /// Slow indicators (> 15μs/candle): Stochastic, MACD
    Slow,
}

/// GPU operation errors
///
/// Re-exported from device module for convenience
pub use super::device::GpuError;

/// CUDA stream manager for concurrent indicator execution
///
/// Manages 3 non-blocking CUDA streams to overlap kernel execution based on
/// indicator computational complexity. This enables 15-30% throughput gains
/// when computing multiple indicators simultaneously.
///
/// # Memory Safety
///
/// All streams share the same CUDA context. Kernels on different streams can
/// execute concurrently but must not write to overlapping memory regions.
///
/// # Performance Characteristics
///
/// - Stream creation: ~10μs overhead (one-time)
/// - Stream switching: ~1μs overhead per kernel launch
/// - Synchronization: ~5-20μs depending on pending work
pub struct StreamManager {
    device: Arc<GpuDevice>,
    stream_fast: Arc<CudaStream>,
    stream_medium: Arc<CudaStream>,
    stream_slow: Arc<CudaStream>,
}

impl StreamManager {
    /// Create new stream manager with 3 CUDA streams
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device handle with initialized CUDA context
    ///
    /// # Errors
    ///
    /// Returns error if CUDA stream creation fails (rare - indicates driver issue)
    ///
    /// # Performance
    ///
    /// Stream creation overhead: ~10μs per stream (30μs total)
    pub fn new(device: Arc<GpuDevice>) -> Result<Self, GpuError> {
        let context = device.context();

        // Create 3 non-blocking CUDA streams
        // These enable concurrent kernel execution on the same GPU
        let stream_fast = context.new_stream().map_err(|e| {
            GpuError::InitializationError(format!("Failed to create fast stream: {:?}", e))
        })?;

        let stream_medium = context.new_stream().map_err(|e| {
            GpuError::InitializationError(format!("Failed to create medium stream: {:?}", e))
        })?;

        let stream_slow = context.new_stream().map_err(|e| {
            GpuError::InitializationError(format!("Failed to create slow stream: {:?}", e))
        })?;

        Ok(Self {
            device,
            stream_fast,
            stream_medium,
            stream_slow,
        })
    }

    /// Get the process-wide StreamManager, creating it on first use
    ///
    /// The manager (and its 3 CUDA streams) is constructed exactly once and
    /// shared for the lifetime of the process. Because cudarc retains the CUDA
    /// primary context, the streams are valid for any `GpuDevice` handle on
    /// device 0 (see `GLOBAL_STREAM_MANAGER` docs).
    ///
    /// # Concurrency
    ///
    /// The streams are shared process-wide: `synchronize_all()` on the global
    /// manager waits for work submitted by *all* callers. This is safe
    /// (over-synchronization only), but callers requiring private streams
    /// should construct their own manager via [`StreamManager::new`].
    ///
    /// # Errors
    ///
    /// Returns error if no CUDA device is available or stream creation fails.
    /// Failed initialization is not cached - subsequent calls retry.
    pub fn global() -> Result<&'static StreamManager, GpuError> {
        if let Some(manager) = GLOBAL_STREAM_MANAGER.get() {
            return Ok(manager);
        }

        // Construct outside get_or_init so initialization errors propagate.
        // Benign race: if two threads initialize concurrently, one manager
        // is dropped (its streams are destroyed unused).
        let device = Arc::new(GpuDevice::new()?);
        let manager = StreamManager::new(device)?;
        Ok(GLOBAL_STREAM_MANAGER.get_or_init(|| manager))
    }

    /// Classify indicator by execution speed
    ///
    /// Based on empirical GPU kernel benchmarks with 10K candles plus the
    /// computational structure of each indicator:
    ///
    /// **Fast (< 5μs/candle)** - embarrassingly parallel or trivial sequential:
    /// - ROC: price[i] / price[i-period] - 1
    /// - Williams %R, Donchian: simple rolling max/min windows
    /// - CCI: two-pass but fully parallel (mean, then deviation)
    /// - SMA, WMA, VWMA: independent rolling-window averages
    /// - OBV, VWAP: single cheap prefix-sum pass
    /// - PivotPoints: per-candle arithmetic
    ///
    /// **Medium (5-15μs/candle)** - one smoothing/recursive stage:
    /// - RSI, ATR: Wilder's smoothing (sequential IIR bottleneck)
    /// - EMA, DEMA, TEMA, HMA, ElderRay: EMA-family recursions
    /// - Aroon: argmax/argmin search (O(n*period))
    /// - Bollinger: rolling std dev (two-pass)
    /// - Keltner: EMA + ATR combination
    /// - CMF, MFI: rolling money-flow sums
    ///
    /// **Slow (> 15μs/candle)** - multi-stage or strongly sequential:
    /// - Stochastic: %K rolling window + %D smoothing
    /// - MACD: three sequential EMAs (fast, slow, signal)
    /// - TSI: cascaded double-EMA on momentum and |momentum|
    /// - ParabolicSAR: sequential state machine (no parallelism)
    /// - VolumeProfile: histogram over the full series
    ///
    /// # Arguments
    ///
    /// * `indicator` - Indicator type from batch module
    ///
    /// # Returns
    ///
    /// IndicatorSpeed classification for stream assignment
    pub fn classify_indicator(indicator: &crate::batch::IndicatorRequest) -> IndicatorSpeed {
        use crate::batch::IndicatorRequest;

        match indicator {
            // Fast indicators (< 5μs/candle)
            IndicatorRequest::ROC { .. } => IndicatorSpeed::Fast,
            IndicatorRequest::WilliamsR { .. } => IndicatorSpeed::Fast,
            IndicatorRequest::CCI { .. } => IndicatorSpeed::Fast,
            IndicatorRequest::SMA { .. } => IndicatorSpeed::Fast,
            IndicatorRequest::WMA { .. } => IndicatorSpeed::Fast,
            IndicatorRequest::VWMA { .. } => IndicatorSpeed::Fast,
            IndicatorRequest::DonchianChannels { .. } => IndicatorSpeed::Fast,
            IndicatorRequest::OBV => IndicatorSpeed::Fast,
            IndicatorRequest::VWAP => IndicatorSpeed::Fast,
            IndicatorRequest::PivotPoints => IndicatorSpeed::Fast,

            // Medium indicators (5-15μs/candle)
            IndicatorRequest::RSI { .. } => IndicatorSpeed::Medium,
            IndicatorRequest::ATR { .. } => IndicatorSpeed::Medium,
            IndicatorRequest::Aroon { .. } => IndicatorSpeed::Medium,
            IndicatorRequest::BollingerBands { .. } => IndicatorSpeed::Medium,
            IndicatorRequest::EMA { .. } => IndicatorSpeed::Medium,
            IndicatorRequest::DEMA { .. } => IndicatorSpeed::Medium,
            IndicatorRequest::TEMA { .. } => IndicatorSpeed::Medium,
            IndicatorRequest::HMA { .. } => IndicatorSpeed::Medium,
            IndicatorRequest::KeltnerChannels { .. } => IndicatorSpeed::Medium,
            IndicatorRequest::ElderRay { .. } => IndicatorSpeed::Medium,
            IndicatorRequest::CMF { .. } => IndicatorSpeed::Medium,
            IndicatorRequest::MFI { .. } => IndicatorSpeed::Medium,

            // Slow indicators (> 15μs/candle)
            IndicatorRequest::Stochastic { .. } => IndicatorSpeed::Slow,
            IndicatorRequest::MACD { .. } => IndicatorSpeed::Slow,
            IndicatorRequest::TSI { .. } => IndicatorSpeed::Slow,
            IndicatorRequest::ParabolicSAR { .. } => IndicatorSpeed::Slow,
            IndicatorRequest::VolumeProfile { .. } => IndicatorSpeed::Slow,

            // Default to Medium for indicators added after this classification.
            // Conservative: better to slightly underutilize the fast stream.
            #[allow(unreachable_patterns)]
            _ => IndicatorSpeed::Medium,
        }
    }

    /// Get CUDA stream for specified speed classification
    ///
    /// # Arguments
    ///
    /// * `speed` - Indicator speed classification
    ///
    /// # Returns
    ///
    /// Reference to appropriate CUDA stream
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let stream = stream_mgr.get_stream(IndicatorSpeed::Fast);
    /// // Launch kernel on this stream
    /// ```
    pub fn get_stream(&self, speed: IndicatorSpeed) -> &Arc<CudaStream> {
        match speed {
            IndicatorSpeed::Fast => &self.stream_fast,
            IndicatorSpeed::Medium => &self.stream_medium,
            IndicatorSpeed::Slow => &self.stream_slow,
        }
    }

    /// Launch kernel on appropriate stream based on indicator speed
    ///
    /// # Arguments
    ///
    /// * `speed` - Indicator speed classification
    /// * `f` - Closure that launches kernel using provided stream
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// stream_mgr.launch_kernel(IndicatorSpeed::Fast, |stream| {
    ///     // Launch kernel on provided stream
    ///     Ok(())
    /// })?;
    /// ```
    pub fn launch_kernel<F>(&self, speed: IndicatorSpeed, f: F) -> Result<(), GpuError>
    where
        F: FnOnce(&Arc<CudaStream>) -> Result<(), GpuError>,
    {
        let stream = self.get_stream(speed);
        f(stream)
    }

    /// Synchronize all streams (wait for all kernels to complete)
    ///
    /// Blocks until all pending kernels on all 3 streams have finished execution.
    /// This is required before:
    /// - Copying results back to host memory
    /// - Freeing GPU memory buffers
    /// - Shutting down the stream manager
    ///
    /// # Performance
    ///
    /// Synchronization overhead: 5-20μs depending on pending work
    /// - No pending kernels: ~5μs (just checks status)
    /// - Active kernels: blocks until completion
    ///
    /// # Errors
    ///
    /// Returns error if any stream synchronization fails (indicates kernel error)
    pub fn synchronize_all(&self) -> Result<(), GpuError> {
        self.stream_fast.synchronize().map_err(|e| {
            GpuError::SynchronizationError(format!("Fast stream sync failed: {:?}", e))
        })?;

        self.stream_medium.synchronize().map_err(|e| {
            GpuError::SynchronizationError(format!("Medium stream sync failed: {:?}", e))
        })?;

        self.stream_slow.synchronize().map_err(|e| {
            GpuError::SynchronizationError(format!("Slow stream sync failed: {:?}", e))
        })?;

        Ok(())
    }

    /// Get reference to underlying GPU device
    pub fn device(&self) -> &Arc<GpuDevice> {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::IndicatorRequest;

    #[test]
    #[ignore] // Requires GPU
    fn test_stream_manager_creation() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let stream_mgr =
            StreamManager::new(Arc::new(device)).expect("Failed to create StreamManager");

        // Verify streams are different (by checking Arc pointer identity)
        assert!(!Arc::ptr_eq(
            stream_mgr.get_stream(IndicatorSpeed::Fast),
            stream_mgr.get_stream(IndicatorSpeed::Medium)
        ));
        assert!(!Arc::ptr_eq(
            stream_mgr.get_stream(IndicatorSpeed::Fast),
            stream_mgr.get_stream(IndicatorSpeed::Slow)
        ));
        assert!(!Arc::ptr_eq(
            stream_mgr.get_stream(IndicatorSpeed::Medium),
            stream_mgr.get_stream(IndicatorSpeed::Slow)
        ));
    }

    #[test]
    fn test_indicator_classification() {
        // Fast indicators
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::ROC { period: 14 }),
            IndicatorSpeed::Fast
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::WilliamsR { period: 14 }),
            IndicatorSpeed::Fast
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::CCI { period: 20 }),
            IndicatorSpeed::Fast
        );

        // Medium indicators
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::RSI { period: 14 }),
            IndicatorSpeed::Medium
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::ATR { period: 14 }),
            IndicatorSpeed::Medium
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::Aroon { period: 25 }),
            IndicatorSpeed::Medium
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::BollingerBands {
                period: 20,
                std_dev: 2.0
            }),
            IndicatorSpeed::Medium
        );

        // Slow indicators
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::Stochastic {
                k_period: 14,
                d_period: 3
            }),
            IndicatorSpeed::Slow
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::MACD {
                fast_period: 12,
                slow_period: 26,
                signal_period: 9
            }),
            IndicatorSpeed::Slow
        );
    }

    #[test]
    fn test_indicator_classification_extended() {
        // Fast: rolling-window averages and cheap prefix-sum indicators
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::SMA { period: 20 }),
            IndicatorSpeed::Fast
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::WMA { period: 20 }),
            IndicatorSpeed::Fast
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::VWMA { period: 20 }),
            IndicatorSpeed::Fast
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::DonchianChannels { period: 20 }),
            IndicatorSpeed::Fast
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::OBV),
            IndicatorSpeed::Fast
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::VWAP),
            IndicatorSpeed::Fast
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::PivotPoints),
            IndicatorSpeed::Fast
        );

        // Medium: single smoothing/recursive stage
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::EMA { period: 20 }),
            IndicatorSpeed::Medium
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::DEMA { period: 20 }),
            IndicatorSpeed::Medium
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::TEMA { period: 20 }),
            IndicatorSpeed::Medium
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::HMA { period: 20 }),
            IndicatorSpeed::Medium
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::KeltnerChannels {
                ema_period: 20,
                atr_period: 10,
                atr_multiplier: 2.0
            }),
            IndicatorSpeed::Medium
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::ElderRay { ema_period: 13 }),
            IndicatorSpeed::Medium
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::CMF { period: 20 }),
            IndicatorSpeed::Medium
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::MFI { period: 14 }),
            IndicatorSpeed::Medium
        );

        // Slow: multi-stage or strongly sequential
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::TSI {
                long_period: 25,
                short_period: 13,
                signal_period: 7
            }),
            IndicatorSpeed::Slow
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::ParabolicSAR {
                af_start: 0.02,
                af_increment: 0.02,
                af_max: 0.2
            }),
            IndicatorSpeed::Slow
        );
        assert_eq!(
            StreamManager::classify_indicator(&IndicatorRequest::VolumeProfile { num_bins: 24 }),
            IndicatorSpeed::Slow
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_stream_selection() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let stream_mgr =
            StreamManager::new(Arc::new(device)).expect("Failed to create StreamManager");

        // Verify correct stream returned for each speed
        let fast_stream = stream_mgr.get_stream(IndicatorSpeed::Fast);
        let medium_stream = stream_mgr.get_stream(IndicatorSpeed::Medium);
        let slow_stream = stream_mgr.get_stream(IndicatorSpeed::Slow);

        // Should be consistent
        assert!(Arc::ptr_eq(
            fast_stream,
            stream_mgr.get_stream(IndicatorSpeed::Fast)
        ));
        assert!(Arc::ptr_eq(
            medium_stream,
            stream_mgr.get_stream(IndicatorSpeed::Medium)
        ));
        assert!(Arc::ptr_eq(
            slow_stream,
            stream_mgr.get_stream(IndicatorSpeed::Slow)
        ));
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_synchronize_all() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let stream_mgr =
            StreamManager::new(Arc::new(device)).expect("Failed to create StreamManager");

        // Launch no-op on each stream (just testing synchronization)
        stream_mgr
            .launch_kernel(IndicatorSpeed::Fast, |_stream| Ok(()))
            .expect("Fast kernel launch failed");

        stream_mgr
            .launch_kernel(IndicatorSpeed::Medium, |_stream| Ok(()))
            .expect("Medium kernel launch failed");

        stream_mgr
            .launch_kernel(IndicatorSpeed::Slow, |_stream| Ok(()))
            .expect("Slow kernel launch failed");

        // Should complete without error
        stream_mgr
            .synchronize_all()
            .expect("Synchronization failed");
    }
}
