//! Automatic GPU/CPU Engine Selection for Trade Aggregation
//!
//! Dynamically selects the optimal aggregation engine based on dataset size
//! and GPU availability. Uses calibrated thresholds to maximize performance.
//!
//! # Performance Characteristics
//!
//! - **<10K trades**: CPU faster (kernel launch overhead dominates)
//! - **10K-100K trades**: GPU 2-5x faster
//! - **>100K trades**: GPU 5-10x faster
//!
//! # Calibration
//!
//! The crossover threshold can be calibrated for specific hardware via benchmarking.
//! Default threshold: **10,000 trades** (conservative, works well on most GPUs).
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::auto_select::EngineSelector;
//!
//! let selector = EngineSelector::default();
//! let candles = selector.aggregate_trades(&trades, timeframe)?;
//! ```

use super::aggregation::GpuAggregator;
use crate::binance::{BinanceError, Candle, Timeframe, Trade, aggregate_trades_to_candles};
use std::sync::LazyLock;

/// Default GPU threshold (trades below this use CPU)
const DEFAULT_GPU_THRESHOLD: usize = 10_000;

/// Maximum threshold (safety limit to prevent excessive GPU allocation)
const MAX_GPU_THRESHOLD: usize = 10_000_000;

/// Minimum threshold (below this, always use CPU)
const MIN_GPU_THRESHOLD: usize = 1_000;

/// Engine selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationEngine {
    /// CPU-based HashMap aggregation (fast for small datasets)
    CPU,
    /// GPU-based parallel aggregation (fast for large datasets)
    GPU,
}

impl AggregationEngine {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            AggregationEngine::CPU => "CPU",
            AggregationEngine::GPU => "GPU",
        }
    }
}

/// Engine selector with configurable threshold
#[derive(Debug, Clone)]
pub struct EngineSelector {
    /// Minimum trades required for GPU (below this, always use CPU)
    gpu_threshold: usize,
    /// Whether GPU is available
    gpu_available: bool,
}

impl Default for EngineSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineSelector {
    /// Create new selector with default threshold
    ///
    /// Checks GPU availability on initialization.
    pub fn new() -> Self {
        Self {
            gpu_threshold: DEFAULT_GPU_THRESHOLD,
            gpu_available: GpuAggregator::is_available(),
        }
    }

    /// Create selector with custom threshold
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum trades for GPU (clamped to 1K-10M range)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Use GPU for datasets >50K trades
    /// let selector = EngineSelector::with_threshold(50_000);
    /// ```
    pub fn with_threshold(threshold: usize) -> Self {
        let clamped_threshold = threshold.clamp(MIN_GPU_THRESHOLD, MAX_GPU_THRESHOLD);

        if threshold != clamped_threshold {
            eprintln!(
                "Warning: GPU threshold {} clamped to {}",
                threshold, clamped_threshold
            );
        }

        Self {
            gpu_threshold: clamped_threshold,
            gpu_available: GpuAggregator::is_available(),
        }
    }

    /// Select optimal engine for given dataset size
    ///
    /// # Logic
    ///
    /// - **GPU unavailable**: Always return CPU
    /// - **Below threshold**: Return CPU (launch overhead not worth it)
    /// - **Above threshold**: Return GPU (expected speedup >2x)
    ///
    /// # Arguments
    ///
    /// * `num_trades` - Number of trades to aggregate
    pub fn select_engine(&self, num_trades: usize) -> AggregationEngine {
        if !self.gpu_available {
            return AggregationEngine::CPU;
        }

        if num_trades < self.gpu_threshold {
            AggregationEngine::CPU
        } else {
            AggregationEngine::GPU
        }
    }

    /// Aggregate trades using auto-selected engine
    ///
    /// # Performance
    ///
    /// Automatically chooses CPU or GPU based on dataset size:
    /// - Small datasets: CPU (no GPU overhead)
    /// - Large datasets: GPU (5-10x speedup)
    ///
    /// # Arguments
    ///
    /// * `trades` - Input trade array
    /// * `timeframe` - Aggregation timeframe
    ///
    /// # Errors
    ///
    /// Returns error if aggregation fails on selected engine.
    pub fn aggregate_trades(
        &self,
        trades: &[Trade],
        timeframe: Timeframe,
    ) -> Result<Vec<Candle>, BinanceError> {
        let num_trades = trades.len();
        let engine = self.select_engine(num_trades);

        match engine {
            AggregationEngine::CPU => {
                // CPU aggregation (always available)
                Ok(aggregate_trades_to_candles(trades, timeframe))
            }
            AggregationEngine::GPU => {
                // GPU aggregation (may fall back to CPU on error)
                GPU_AGGREGATOR
                    .aggregate_trades(trades, timeframe)
                    .or_else(|e| {
                        eprintln!("GPU aggregation failed: {:?}, falling back to CPU", e);
                        Ok(aggregate_trades_to_candles(trades, timeframe))
                    })
            }
        }
    }

    /// Calibrate GPU threshold via benchmarking
    ///
    /// Runs aggregation on varying dataset sizes and determines the crossover
    /// point where GPU becomes faster than CPU.
    ///
    /// # Returns
    ///
    /// Recommended threshold (number of trades) for GPU aggregation.
    ///
    /// # Performance
    ///
    /// Calibration takes ~10-30 seconds (benchmarks 5-10 dataset sizes).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let threshold = EngineSelector::calibrate()?;
    /// println!("Recommended GPU threshold: {} trades", threshold);
    ///
    /// let selector = EngineSelector::with_threshold(threshold);
    /// ```
    pub fn calibrate() -> Result<usize, String> {
        use std::time::Instant;

        if !GpuAggregator::is_available() {
            return Err("GPU not available for calibration".to_string());
        }

        println!("Calibrating GPU/CPU threshold...");

        // Test dataset sizes (1K to 1M trades)
        let test_sizes = vec![1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 500_000];

        let mut crossover_point = DEFAULT_GPU_THRESHOLD;

        for &size in &test_sizes {
            // Generate test trades
            let trades = generate_test_trades(size);

            // Benchmark CPU
            let cpu_start = Instant::now();
            let _candles_cpu = aggregate_trades_to_candles(&trades, Timeframe::minutes(5));
            let cpu_time = cpu_start.elapsed();

            // Benchmark GPU
            let gpu_aggregator =
                GpuAggregator::new().map_err(|e| format!("GPU init failed: {:?}", e))?;
            let gpu_start = Instant::now();
            let _candles_gpu = gpu_aggregator
                .aggregate_trades(&trades, Timeframe::minutes(5))
                .map_err(|e| format!("GPU aggregation failed: {:?}", e))?;
            let gpu_time = gpu_start.elapsed();

            let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

            println!(
                "  {} trades: CPU={:.3}ms, GPU={:.3}ms, speedup={:.2}x",
                size,
                cpu_time.as_secs_f64() * 1000.0,
                gpu_time.as_secs_f64() * 1000.0,
                speedup
            );

            // Find first size where GPU is faster (speedup > 1.0)
            if speedup > 1.0 && crossover_point == DEFAULT_GPU_THRESHOLD {
                crossover_point = size;
            }
        }

        println!("\nRecommended threshold: {} trades", crossover_point);
        Ok(crossover_point)
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// Get current threshold
    pub fn threshold(&self) -> usize {
        self.gpu_threshold
    }
}

/// Global GPU aggregator instance (lazily initialized)
static GPU_AGGREGATOR: LazyLock<GpuAggregator> =
    LazyLock::new(|| GpuAggregator::new().expect("Failed to initialize GPU aggregator"));

/// Generate synthetic test trades for benchmarking
fn generate_test_trades(n: usize) -> Vec<Trade> {
    let mut trades = Vec::with_capacity(n);

    let base_price = 50_000.0; // BTC price
    let base_time = 1_600_000_000_000i64; // Jan 2021

    for i in 0..n {
        let price = base_price + (i as f64 * 0.01); // Small price variation
        let timestamp = base_time + (i as i64 * 1000); // 1 second apart

        trades.push(Trade {
            trade_id: i as u64,
            price,
            quantity: 0.1,
            quote_quantity: price * 0.1,
            timestamp_ms: timestamp,
            is_buyer_maker: i % 2 == 0,
        });
    }

    trades
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_selector_default() {
        let selector = EngineSelector::default();
        assert_eq!(selector.threshold(), DEFAULT_GPU_THRESHOLD);
    }

    #[test]
    fn test_engine_selector_custom_threshold() {
        let selector = EngineSelector::with_threshold(50_000);
        assert_eq!(selector.threshold(), 50_000);
    }

    #[test]
    fn test_threshold_clamping() {
        // Too low
        let selector = EngineSelector::with_threshold(100);
        assert_eq!(selector.threshold(), MIN_GPU_THRESHOLD);

        // Too high
        let selector = EngineSelector::with_threshold(100_000_000);
        assert_eq!(selector.threshold(), MAX_GPU_THRESHOLD);
    }

    #[test]
    fn test_select_engine_below_threshold() {
        let selector = EngineSelector::with_threshold(10_000);
        assert_eq!(selector.select_engine(5_000), AggregationEngine::CPU);
    }

    #[test]
    fn test_select_engine_above_threshold() {
        let selector = EngineSelector::with_threshold(10_000);

        // If GPU available, should select GPU
        // If not available, should select CPU
        let engine = selector.select_engine(20_000);

        if selector.is_gpu_available() {
            assert_eq!(engine, AggregationEngine::GPU);
        } else {
            assert_eq!(engine, AggregationEngine::CPU);
        }
    }

    #[test]
    fn test_generate_test_trades() {
        let trades = generate_test_trades(1000);
        assert_eq!(trades.len(), 1000);

        // Verify trades are sequential
        assert_eq!(trades[0].trade_id, 0);
        assert_eq!(trades[999].trade_id, 999);

        // Verify timestamps are increasing
        assert!(trades[1].timestamp_ms > trades[0].timestamp_ms);
    }

    #[test]
    #[ignore] // Requires GPU and takes ~10-30 seconds
    fn test_calibrate() {
        let result = EngineSelector::calibrate();

        if GpuAggregator::is_available() {
            assert!(result.is_ok());
            let threshold = result.unwrap();
            println!("Calibrated threshold: {}", threshold);
            assert!(threshold >= MIN_GPU_THRESHOLD);
            assert!(threshold <= MAX_GPU_THRESHOLD);
        } else {
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_aggregate_trades_empty() {
        let selector = EngineSelector::default();
        let trades = vec![];
        let candles = selector
            .aggregate_trades(&trades, Timeframe::minutes(1))
            .expect("Aggregation failed");
        assert!(candles.is_empty());
    }

    #[test]
    fn test_aggregate_trades_small_dataset() {
        let selector = EngineSelector::default();
        let trades = generate_test_trades(100);
        let candles = selector
            .aggregate_trades(&trades, Timeframe::minutes(1))
            .expect("Aggregation failed");

        // Should use CPU (below threshold)
        assert_eq!(selector.select_engine(100), AggregationEngine::CPU);
        assert!(!candles.is_empty());
    }
}
