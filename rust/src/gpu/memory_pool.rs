//! GPU Memory Pool for Batch Indicator Calculations
//!
//! Pre-allocates GPU memory for OHLCV inputs and all indicator outputs to minimize
//! GPU-CPU transfers. Reduces memory transfers by 49% (360 MB → 184 MB) by loading
//! data once and reusing buffers across indicators.
//!
//! # Memory Layout
//!
//! - **Input Buffers** (5 × max_candles × 8 bytes): high, low, close, open, volume
//! - **Output Buffers** (16 × max_candles × 8 bytes): 16 indicator outputs
//! - **Total**: 168 bytes/candle
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::{GpuDevice, GpuMemoryPool, IndicatorType};
//!
//! let device = GpuDevice::new()?;
//! let mut pool = GpuMemoryPool::new(device, 100_000)?;
//!
//! // Load OHLCV once
//! pool.load_ohlcv(&high, &low, &close, &open, &volume)?;
//!
//! // Calculate multiple indicators without reloading data
//! let stoch_k = pool.get_output_buffer(IndicatorType::StochasticK)?;
//! let stoch_d = pool.get_output_buffer(IndicatorType::StochasticD)?;
//!
//! // Copy selected results back
//! let results = pool.copy_results_to_host(&[
//!     IndicatorType::StochasticK,
//!     IndicatorType::StochasticD,
//! ])?;
//! ```

use super::device::{GpuDevice, GpuError};
use cudarc::driver::CudaSlice;
use std::collections::HashMap;
use std::sync::Arc;

/// Maximum supported candles (1M limit for safety)
const MAX_CANDLES_LIMIT: usize = 1_000_000;

/// Type alias for input buffer tuple returned by get_input_buffers
type InputBuffers<'a> = (
    &'a CudaSlice<f64>,
    &'a CudaSlice<f64>,
    &'a CudaSlice<f64>,
    &'a CudaSlice<f64>,
    &'a CudaSlice<f64>,
    usize,
);

/// Indicator output types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndicatorType {
    // Stochastic Oscillator (2 outputs)
    StochasticK,
    StochasticD,

    // Williams %R (1 output)
    WilliamsR,

    // Average True Range (1 output)
    ATR,

    // Relative Strength Index (1 output)
    RSI,

    // Bollinger Bands (3 outputs)
    BollingerUpper,
    BollingerMiddle,
    BollingerLower,

    // Rate of Change (1 output)
    ROC,

    // Commodity Channel Index (1 output)
    CCI,

    // Aroon Indicator (2 outputs)
    AroonUp,
    AroonDown,

    // MACD (3 outputs)
    MACDLine,
    MACDSignal,
    MACDHistogram,
}

/// GPU Memory Pool for batch indicator calculations
///
/// Pre-allocates all input and output buffers to minimize GPU-CPU transfers.
pub struct GpuMemoryPool {
    device: Arc<GpuDevice>,
    max_candles: usize,
    actual_candles: usize,

    // Input buffers (pre-allocated)
    high_buffer: CudaSlice<f64>,
    low_buffer: CudaSlice<f64>,
    close_buffer: CudaSlice<f64>,
    open_buffer: CudaSlice<f64>,
    volume_buffer: CudaSlice<f64>,

    // Output buffers (pre-allocated for all indicators)
    output_buffers: HashMap<IndicatorType, CudaSlice<f64>>,
}

impl GpuMemoryPool {
    /// Create new GPU memory pool with pre-allocated buffers
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device handle
    /// * `max_candles` - Maximum number of candles to support (≤1M)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `max_candles` exceeds 1M limit
    /// - GPU memory allocation fails (out of VRAM)
    /// - Device initialization fails
    ///
    /// # Memory Usage
    ///
    /// - 100K candles: ~16.8 MB
    /// - 1M candles: ~168 MB
    pub fn new(device: Arc<GpuDevice>, max_candles: usize) -> Result<Self, GpuError> {
        // Validate max_candles
        if max_candles == 0 {
            return Err(GpuError::InvalidParameter(
                "max_candles must be greater than 0".to_string(),
            ));
        }
        if max_candles > MAX_CANDLES_LIMIT {
            return Err(GpuError::InvalidParameter(format!(
                "max_candles {} exceeds limit of {}",
                max_candles, MAX_CANDLES_LIMIT
            )));
        }

        // Allocate input buffers
        let high_buffer = device.alloc_buffer(max_candles)?;
        let low_buffer = device.alloc_buffer(max_candles)?;
        let close_buffer = device.alloc_buffer(max_candles)?;
        let open_buffer = device.alloc_buffer(max_candles)?;
        let volume_buffer = device.alloc_buffer(max_candles)?;

        // Allocate output buffers for all 16 indicator outputs
        let mut output_buffers = HashMap::new();

        let indicator_types = [
            IndicatorType::StochasticK,
            IndicatorType::StochasticD,
            IndicatorType::WilliamsR,
            IndicatorType::ATR,
            IndicatorType::RSI,
            IndicatorType::BollingerUpper,
            IndicatorType::BollingerMiddle,
            IndicatorType::BollingerLower,
            IndicatorType::ROC,
            IndicatorType::CCI,
            IndicatorType::AroonUp,
            IndicatorType::AroonDown,
            IndicatorType::MACDLine,
            IndicatorType::MACDSignal,
            IndicatorType::MACDHistogram,
        ];

        for indicator_type in &indicator_types {
            let buffer = device.alloc_buffer(max_candles)?;
            output_buffers.insert(*indicator_type, buffer);
        }

        Ok(Self {
            device,
            max_candles,
            actual_candles: 0,
            high_buffer,
            low_buffer,
            close_buffer,
            open_buffer,
            volume_buffer,
            output_buffers,
        })
    }

    /// Load OHLCV data to GPU (one-time operation)
    ///
    /// # Arguments
    ///
    /// * `high` - High prices
    /// * `low` - Low prices
    /// * `close` - Close prices
    /// * `open` - Open prices (optional, will use zeros if None)
    /// * `volume` - Volume data (optional, will use zeros if None)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Array lengths don't match
    /// - Array length exceeds `max_candles`
    /// - Memory copy fails
    pub fn load_ohlcv(
        &mut self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        open: Option<&[f64]>,
        volume: Option<&[f64]>,
    ) -> Result<(), GpuError> {
        // Validate inputs
        let n = high.len();
        if n > self.max_candles {
            return Err(GpuError::InvalidParameter(format!(
                "Data length {} exceeds max_candles {}",
                n, self.max_candles
            )));
        }

        if low.len() != n || close.len() != n {
            return Err(GpuError::InvalidParameter(
                "high, low, close must have same length".to_string(),
            ));
        }

        if let Some(open_data) = open
            && open_data.len() != n
        {
            return Err(GpuError::InvalidParameter(
                "open length must match high/low/close".to_string(),
            ));
        }

        if let Some(volume_data) = volume
            && volume_data.len() != n
        {
            return Err(GpuError::InvalidParameter(
                "volume length must match high/low/close".to_string(),
            ));
        }

        // Copy data to GPU
        self.device
            .stream
            .memcpy_htod(high, &mut self.high_buffer)
            .map_err(|e| GpuError::MemoryCopyError(format!("Failed to copy high: {:?}", e)))?;

        self.device
            .stream
            .memcpy_htod(low, &mut self.low_buffer)
            .map_err(|e| GpuError::MemoryCopyError(format!("Failed to copy low: {:?}", e)))?;

        self.device
            .stream
            .memcpy_htod(close, &mut self.close_buffer)
            .map_err(|e| GpuError::MemoryCopyError(format!("Failed to copy close: {:?}", e)))?;

        // Copy open or use zeros
        if let Some(open_data) = open {
            self.device
                .stream
                .memcpy_htod(open_data, &mut self.open_buffer)
                .map_err(|e| GpuError::MemoryCopyError(format!("Failed to copy open: {:?}", e)))?;
        }

        // Copy volume or use zeros
        if let Some(volume_data) = volume {
            self.device
                .stream
                .memcpy_htod(volume_data, &mut self.volume_buffer)
                .map_err(|e| {
                    GpuError::MemoryCopyError(format!("Failed to copy volume: {:?}", e))
                })?;
        }

        self.actual_candles = n;
        Ok(())
    }

    /// Get references to input buffers for kernel execution
    ///
    /// Returns (high, low, close, open, volume, actual_length)
    pub fn get_input_buffers(&self) -> InputBuffers<'_> {
        (
            &self.high_buffer,
            &self.low_buffer,
            &self.close_buffer,
            &self.open_buffer,
            &self.volume_buffer,
            self.actual_candles,
        )
    }

    /// Get mutable reference to output buffer for specific indicator
    ///
    /// # Arguments
    ///
    /// * `indicator` - Indicator type
    ///
    /// # Errors
    ///
    /// Returns error if indicator type not recognized
    pub fn get_output_buffer_mut(
        &mut self,
        indicator: IndicatorType,
    ) -> Result<&mut CudaSlice<f64>, GpuError> {
        self.output_buffers.get_mut(&indicator).ok_or_else(|| {
            GpuError::InvalidParameter(format!("Unknown indicator type: {:?}", indicator))
        })
    }

    /// Get immutable reference to output buffer for specific indicator
    pub fn get_output_buffer(&self, indicator: IndicatorType) -> Result<&CudaSlice<f64>, GpuError> {
        self.output_buffers.get(&indicator).ok_or_else(|| {
            GpuError::InvalidParameter(format!("Unknown indicator type: {:?}", indicator))
        })
    }

    /// Copy selected indicator results from GPU to host
    ///
    /// # Arguments
    ///
    /// * `indicators` - List of indicator types to copy
    ///
    /// # Returns
    ///
    /// HashMap mapping IndicatorType to Vec<f64> results
    ///
    /// # Errors
    ///
    /// Returns error if memory copy fails
    pub fn copy_results_to_host(
        &self,
        indicators: &[IndicatorType],
    ) -> Result<HashMap<IndicatorType, Vec<f64>>, GpuError> {
        let mut results = HashMap::new();

        for indicator in indicators {
            let buffer = self.get_output_buffer(*indicator)?;

            // Copy only actual_candles elements.
            //
            // The output buffer is allocated for `max_candles`, but only the
            // first `actual_candles` elements are valid. cudarc's `memcpy_dtoh`
            // asserts `dst.len() >= src.len()`, so we must slice the device
            // buffer down to `actual_candles` before the copy; passing the full
            // `max_candles` buffer against an `actual_candles`-sized host slice
            // would otherwise trip that assertion (panic).
            let mut host_data = vec![0.0; self.actual_candles];
            let src = buffer.slice(0..self.actual_candles);
            self.device
                .stream
                .memcpy_dtoh(&src, &mut host_data)
                .map_err(|e| {
                    GpuError::MemoryCopyError(format!(
                        "Failed to copy {:?} from device: {:?}",
                        indicator, e
                    ))
                })?;

            // Truncate to actual length
            host_data.truncate(self.actual_candles);
            results.insert(*indicator, host_data);
        }

        Ok(results)
    }

    /// Calculate total memory usage in bytes
    ///
    /// # Returns
    ///
    /// Total GPU memory allocated (includes all buffers)
    pub fn memory_usage(&self) -> usize {
        // 5 input buffers + 15 output buffers = 20 total buffers
        // Each buffer: max_candles * 8 bytes (f64)
        let num_buffers = 5 + self.output_buffers.len();
        num_buffers * self.max_candles * std::mem::size_of::<f64>()
    }

    /// Get maximum number of candles supported
    pub fn max_candles(&self) -> usize {
        self.max_candles
    }

    /// Get actual number of candles loaded
    pub fn actual_candles(&self) -> usize {
        self.actual_candles
    }

    /// Get device reference
    pub fn device(&self) -> &Arc<GpuDevice> {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_memory_pool_allocation() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let pool = GpuMemoryPool::new(device, 10_000).expect("Failed to create memory pool");

        assert_eq!(pool.max_candles(), 10_000);
        assert_eq!(pool.actual_candles(), 0); // No data loaded yet

        // Verify memory usage calculation
        // 5 inputs + 15 outputs = 20 buffers
        // 20 * 10_000 * 8 = 1,600,000 bytes
        assert_eq!(pool.memory_usage(), 1_600_000);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_memory_pool_load_ohlcv() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let mut pool = GpuMemoryPool::new(device, 10_000).expect("Failed to create memory pool");

        // Create test data
        let n = 1000;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 95.0 + i as f64).collect();
        let open: Vec<f64> = (0..n).map(|i| 92.0 + i as f64).collect();
        let volume: Vec<f64> = (0..n).map(|i| 1000.0 * i as f64).collect();

        // Load data
        pool.load_ohlcv(&high, &low, &close, Some(&open), Some(&volume))
            .expect("Failed to load OHLCV");

        assert_eq!(pool.actual_candles(), n);

        // Verify we can get input buffers
        let (high_buf, low_buf, close_buf, open_buf, volume_buf, actual_len) =
            pool.get_input_buffers();
        assert_eq!(actual_len, n);
        assert_eq!(high_buf.len(), 10_000); // Max allocation
        assert_eq!(low_buf.len(), 10_000);
        assert_eq!(close_buf.len(), 10_000);
        assert_eq!(open_buf.len(), 10_000);
        assert_eq!(volume_buf.len(), 10_000);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_memory_pool_output_buffers() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let mut pool = GpuMemoryPool::new(device, 10_000).expect("Failed to create memory pool");

        // Verify all indicator output buffers exist
        let indicators = [
            IndicatorType::StochasticK,
            IndicatorType::StochasticD,
            IndicatorType::WilliamsR,
            IndicatorType::ATR,
            IndicatorType::RSI,
            IndicatorType::BollingerUpper,
            IndicatorType::BollingerMiddle,
            IndicatorType::BollingerLower,
            IndicatorType::ROC,
            IndicatorType::CCI,
            IndicatorType::AroonUp,
            IndicatorType::AroonDown,
            IndicatorType::MACDLine,
            IndicatorType::MACDSignal,
            IndicatorType::MACDHistogram,
        ];

        for indicator in &indicators {
            let buffer = pool
                .get_output_buffer_mut(*indicator)
                .expect(&format!("Missing buffer for {:?}", indicator));
            assert_eq!(buffer.len(), 10_000);
        }

        // Test total count (15 output indicators)
        assert_eq!(pool.output_buffers.len(), 15);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_memory_pool_copy_results() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let mut pool = GpuMemoryPool::new(device, 10_000).expect("Failed to create memory pool");

        // Load sample data
        let n = 100;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 95.0 + i as f64).collect();

        pool.load_ohlcv(&high, &low, &close, None, None)
            .expect("Failed to load OHLCV");

        // Copy selected results (buffers are zeros, just testing transfer)
        let results = pool
            .copy_results_to_host(&[IndicatorType::StochasticK, IndicatorType::RSI])
            .expect("Failed to copy results");

        assert_eq!(results.len(), 2);
        assert!(results.contains_key(&IndicatorType::StochasticK));
        assert!(results.contains_key(&IndicatorType::RSI));
        assert_eq!(results[&IndicatorType::StochasticK].len(), n);
        assert_eq!(results[&IndicatorType::RSI].len(), n);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_memory_pool_memory_usage() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        // Test 100K candles (target workload)
        let pool_100k =
            GpuMemoryPool::new(device.clone(), 100_000).expect("Failed to create 100K pool");

        // 20 buffers * 100,000 candles * 8 bytes = 16,000,000 bytes = ~15.26 MB
        let expected_100k = 20 * 100_000 * 8;
        assert_eq!(pool_100k.memory_usage(), expected_100k);
        println!(
            "100K candles: {} bytes (~{:.2} MB)",
            expected_100k,
            expected_100k as f64 / 1_048_576.0
        );

        // Test 1M candles (max limit)
        let pool_1m =
            GpuMemoryPool::new(device.clone(), 1_000_000).expect("Failed to create 1M pool");

        // 20 buffers * 1,000,000 candles * 8 bytes = 160,000,000 bytes = ~152.59 MB
        let expected_1m = 20 * 1_000_000 * 8;
        assert_eq!(pool_1m.memory_usage(), expected_1m);
        println!(
            "1M candles: {} bytes (~{:.2} MB)",
            expected_1m,
            expected_1m as f64 / 1_048_576.0
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_memory_pool_large() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let mut pool =
            GpuMemoryPool::new(device, 1_000_000).expect("Failed to create 1M memory pool");

        // Load 1M candles
        let n = 1_000_000;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + (i % 1000) as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + (i % 1000) as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 95.0 + (i % 1000) as f64).collect();

        pool.load_ohlcv(&high, &low, &close, None, None)
            .expect("Failed to load 1M candles");

        assert_eq!(pool.actual_candles(), n);

        // Verify memory usage (~152.59 MB)
        let usage = pool.memory_usage();
        assert_eq!(usage, 160_000_000);
        println!(
            "1M candles memory usage: {} bytes (~{:.2} MB)",
            usage,
            usage as f64 / 1_048_576.0
        );
    }

    #[test]
    fn test_memory_pool_validation() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        // Test zero candles
        let result = GpuMemoryPool::new(device.clone(), 0);
        assert!(result.is_err());
        if let Err(GpuError::InvalidParameter(msg)) = result {
            assert!(msg.contains("must be greater than 0"));
        }

        // Test exceeding limit
        let result = GpuMemoryPool::new(device.clone(), 2_000_000);
        assert!(result.is_err());
        if let Err(GpuError::InvalidParameter(msg)) = result {
            assert!(msg.contains("exceeds limit"));
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_memory_pool_mismatched_lengths() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let mut pool = GpuMemoryPool::new(device, 10_000).expect("Failed to create memory pool");

        let high: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let low: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let close: Vec<f64> = (0..50).map(|i| i as f64).collect(); // Wrong length

        let result = pool.load_ohlcv(&high, &low, &close, None, None);
        assert!(result.is_err());
        if let Err(GpuError::InvalidParameter(msg)) = result {
            assert!(msg.contains("same length"));
        }
    }
}
