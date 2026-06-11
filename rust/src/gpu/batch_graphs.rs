//! Batch Indicator Calculation with CUDA Graphs - DISABLED
//!
//! # Status: graph replay is disabled
//!
//! The original implementation had two fatal problems:
//!
//! 1. **Guaranteed panic**: graphs were cached under a *sorted* indicator key
//!    but fetched with the *unsorted* key followed by `.unwrap()`, so any
//!    indicator order that differed from the sorted order panicked on the
//!    first call.
//! 2. **Net-negative "fast path"**: captured graphs never wired up result
//!    buffers, so after replaying the graphs the executor recomputed every
//!    indicator through the traditional path anyway. Replay added pure
//!    overhead (graph launch + sync) on top of full recomputation while
//!    claiming a 16.7x launch-overhead reduction.
//!
//! Until graph capture stores result buffers, [`BatchGraphExecutor::calculate_batch`]
//! returns an honest error instead of silently doing extra work. Use
//! [`crate::gpu::batch::calculate_indicators_batch_gpu`] for batch indicator
//! calculation.

use super::batch::{BatchIndicatorParams, BatchIndicatorType, IndicatorResult};
use super::cuda_graphs::IndicatorGraph;
use super::device::{GpuDevice, GpuError};
use super::streams::StreamManager;
use ndarray::Array1;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Error message returned while the CUDA Graph replay path is disabled.
pub(crate) const GRAPH_REPLAY_DISABLED_MSG: &str = "CUDA Graph batch replay is disabled: \
captured graphs do not wire result buffers, so replay recomputed every indicator \
traditionally on top of the graph launch (net-negative). Use \
gpu::batch::calculate_indicators_batch_gpu instead.";

/// Batch executor with CUDA Graphs optimization - currently disabled
///
/// See the module-level documentation for why the replay path is gated off.
///
/// # Thread Safety
///
/// This executor is NOT thread-safe. Create one per thread or use external
/// synchronization. CUDA Graphs are not internally synchronized per NVIDIA docs.
pub struct BatchGraphExecutor {
    _device: Arc<GpuDevice>,
    _stream_mgr: Arc<StreamManager>,
    /// Cached graphs for each unique indicator set.
    ///
    /// Invariant: keys are sorted by `format!("{:?}", ind)` before insertion
    /// AND lookup (the original code inserted sorted keys but looked up
    /// unsorted keys, guaranteeing a panic). The cache is currently always
    /// empty because graph capture is disabled.
    graph_cache: Mutex<HashMap<Vec<BatchIndicatorType>, Arc<IndicatorGraph>>>,
}

impl BatchGraphExecutor {
    /// Create new batch executor
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device handle
    ///
    /// # Errors
    ///
    /// Returns error if stream manager creation fails
    pub fn new(device: Arc<GpuDevice>) -> Result<Self, GpuError> {
        let stream_mgr = Arc::new(StreamManager::new(device.clone())?);

        Ok(Self {
            _device: device,
            _stream_mgr: stream_mgr,
            graph_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Calculate batch indicators - currently returns an error
    ///
    /// The CUDA Graph capture/replay path is disabled because it produced no
    /// speedup (it recomputed every indicator traditionally after replay) and
    /// panicked on unsorted indicator lists. Until result buffers are wired
    /// into the captured graphs this returns an explanatory error after input
    /// validation.
    ///
    /// Use [`crate::gpu::batch::calculate_indicators_batch_gpu`] instead.
    ///
    /// # Errors
    ///
    /// - `InvalidParameter` if input arrays mismatch or no indicators given
    /// - `ComputationErrorStatic` (always, after validation) while the graph
    ///   path is disabled
    pub fn calculate_batch(
        &self,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        indicators: &[BatchIndicatorType],
        _params: &HashMap<BatchIndicatorType, BatchIndicatorParams>,
    ) -> Result<HashMap<BatchIndicatorType, IndicatorResult>, GpuError> {
        let n = close.len();

        // Validate inputs
        if high.len() != n || low.len() != n {
            return Err(GpuError::InvalidParameter(
                "High, low, and close arrays must have same length".to_string(),
            ));
        }

        if indicators.is_empty() {
            return Err(GpuError::InvalidParameter(
                "No indicators specified".to_string(),
            ));
        }

        Err(GpuError::ComputationErrorStatic(GRAPH_REPLAY_DISABLED_MSG))
    }

    /// Clear graph cache (useful for memory management)
    pub fn clear_cache(&self) {
        let mut cache = self.graph_cache.lock().unwrap();
        cache.clear();
    }

    /// Get number of cached graphs
    pub fn cache_size(&self) -> usize {
        let cache = self.graph_cache.lock().unwrap();
        cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let high: Array1<f64> = Array1::linspace(100.0, 110.0, n);
        let low: Array1<f64> = Array1::linspace(95.0, 105.0, n);
        let close: Array1<f64> = Array1::linspace(97.0, 107.0, n);
        (high, low, close)
    }

    #[test]
    fn test_disabled_message_mentions_alternative() {
        // Host-side sanity check: the error message must point users at the
        // working batch API.
        assert!(GRAPH_REPLAY_DISABLED_MSG.contains("calculate_indicators_batch_gpu"));
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_batch_graph_executor_returns_unsupported() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let executor = BatchGraphExecutor::new(device).expect("Executor creation failed");

        let (high, low, close) = generate_test_data(1000);
        let indicators = vec![BatchIndicatorType::RSI];
        let params = HashMap::new();

        // The graph replay path is disabled: an honest error, not a panic
        // and not silently-recomputed results.
        let result = executor.calculate_batch(&high, &low, &close, &indicators, &params);
        match result {
            Err(GpuError::ComputationErrorStatic(msg)) => {
                assert!(msg.contains("disabled"));
            }
            other => panic!("Expected ComputationErrorStatic, got {:?}", other.map(|_| ())),
        }

        // Nothing is ever cached while the path is disabled.
        assert_eq!(executor.cache_size(), 0);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_batch_graph_executor_input_validation() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let executor = BatchGraphExecutor::new(device).expect("Executor creation failed");

        let (high, low, close) = generate_test_data(1000);
        let params = HashMap::new();

        // Empty indicator list is rejected before the disabled-path error.
        let result = executor.calculate_batch(&high, &low, &close, &[], &params);
        assert!(matches!(result, Err(GpuError::InvalidParameter(_))));

        // Mismatched lengths are rejected.
        let short_low = Array1::linspace(95.0, 105.0, 10);
        let indicators = vec![BatchIndicatorType::RSI];
        let result = executor.calculate_batch(&high, &short_low, &close, &indicators, &params);
        assert!(matches!(result, Err(GpuError::InvalidParameter(_))));
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_batch_graph_executor_cache_clear() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let executor = BatchGraphExecutor::new(device).expect("Executor creation failed");

        assert_eq!(executor.cache_size(), 0);
        executor.clear_cache();
        assert_eq!(executor.cache_size(), 0);
    }
}
