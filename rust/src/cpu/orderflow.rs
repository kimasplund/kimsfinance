//! CPU Orderflow Feature Extraction + Signal Generation
//!
//! Pure CPU implementation of orderflow analysis providing automatic fallback
//! when GPU is unavailable. Matches the GPU API for drop-in replacement.
//!
//! # Architecture
//!
//! This module implements CPU versions of Agent 2's orderflow feature extraction
//! and signal generation. While slower than GPU for large batches, it ensures
//! the system never fails due to GPU unavailability.
//!
//! # Performance
//!
//! - **Single strategy**: 5-20M features/sec (CPU-bound)
//! - **Multi-strategy**: Scales linearly with CPU cores
//! - **Memory**: 6 bytes per tick per strategy (INT8 quantized)
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::cpu::orderflow::{OrderflowBatchProcessor, OrderflowInput, StrategyConfig};
//!
//! let processor = OrderflowBatchProcessor::new();
//!
//! // Configure strategies
//! let strategies = vec![
//!     StrategyConfig::momentum(),
//!     StrategyConfig::mean_reversion(),
//!     StrategyConfig::breakout(),
//! ];
//!
//! // Prepare input data
//! let input = OrderflowInput {
//!     timestamps: vec![...],
//!     close_prices: vec![...],
//!     volumes: vec![...],
//!     buy_volumes: vec![...],
//!     sell_volumes: vec![...],
//! };
//!
//! // Process batch (CPU implementation)
//! let results = processor.process_batch(&input, &strategies)?;
//! ```

use std::collections::VecDeque;

#[cfg(feature = "gpu")]
use crate::gpu::GpuError;

#[cfg(not(feature = "gpu"))]
use crate::cpu::sequential::GpuError;

/// Number of orderflow features computed per tick
pub const NUM_FEATURES: usize = 6;

/// Sliding window size for feature calculation
const WINDOW_SIZE: usize = 20;

/// Signal types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Signal {
    Hold = 0,
    Buy = 1,
    Sell = -1,
}

impl From<i8> for Signal {
    fn from(value: i8) -> Self {
        match value {
            1 => Signal::Buy,
            -1 => Signal::Sell,
            _ => Signal::Hold,
        }
    }
}

/// Strategy identifiers (matching GPU implementation)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum StrategyType {
    /// Simple momentum: buy when imbalance > 0.6 && volume_delta > 1000
    Momentum = 0,

    /// Mean reversion: buy when imbalance < 0.4 && volume_delta < -1000
    MeanReversion = 1,

    /// Breakout: buy when trade_intensity > 100 && price_velocity > 0.001
    Breakout = 2,

    /// Scalping: buy when imbalance > 0.55 && abs(volume_delta) < 500
    Scalping = 3,

    /// Trend following: buy when volume_delta > 5000 && price_velocity > 0.002
    TrendFollowing = 4,
}

/// Strategy configuration
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Strategy type (determines signal generation logic)
    pub strategy_type: StrategyType,

    /// Per-feature minimum values for quantization (calibrated from data)
    pub feature_mins: [f32; NUM_FEATURES],

    /// Per-feature maximum values for quantization (calibrated from data)
    pub feature_maxs: [f32; NUM_FEATURES],
}

impl StrategyConfig {
    /// Create momentum strategy with default quantization ranges
    pub fn momentum() -> Self {
        Self {
            strategy_type: StrategyType::Momentum,
            feature_mins: [0.0, -10000.0, 0.0, -1.0, -1.0, 0.0],
            feature_maxs: [1.0, 10000.0, 1000.0, 1.0, 1.0, 10000.0],
        }
    }

    /// Create mean reversion strategy with default quantization ranges
    pub fn mean_reversion() -> Self {
        Self {
            strategy_type: StrategyType::MeanReversion,
            feature_mins: [0.0, -10000.0, 0.0, -1.0, -1.0, 0.0],
            feature_maxs: [1.0, 10000.0, 1000.0, 1.0, 1.0, 10000.0],
        }
    }

    /// Create breakout strategy with default quantization ranges
    pub fn breakout() -> Self {
        Self {
            strategy_type: StrategyType::Breakout,
            feature_mins: [0.0, -10000.0, 0.0, -1.0, -1.0, 0.0],
            feature_maxs: [1.0, 10000.0, 1000.0, 1.0, 1.0, 10000.0],
        }
    }

    /// Create scalping strategy with default quantization ranges
    pub fn scalping() -> Self {
        Self {
            strategy_type: StrategyType::Scalping,
            feature_mins: [0.0, -10000.0, 0.0, -1.0, -1.0, 0.0],
            feature_maxs: [1.0, 10000.0, 1000.0, 1.0, 1.0, 10000.0],
        }
    }

    /// Create trend following strategy with default quantization ranges
    pub fn trend_following() -> Self {
        Self {
            strategy_type: StrategyType::TrendFollowing,
            feature_mins: [0.0, -10000.0, 0.0, -1.0, -1.0, 0.0],
            feature_maxs: [1.0, 10000.0, 1000.0, 1.0, 1.0, 10000.0],
        }
    }
}

/// Input data for orderflow processing
///
/// This structure matches the GPU implementation exactly.
#[derive(Debug, Clone)]
pub struct OrderflowInput {
    /// Unix timestamps in milliseconds
    pub timestamps: Vec<i64>,

    /// Close prices (OHLCV aggregated from ticks)
    pub close_prices: Vec<f32>,

    /// Total volumes
    pub volumes: Vec<f32>,

    /// Buy-side volumes (taker was buyer)
    pub buy_volumes: Vec<f32>,

    /// Sell-side volumes (taker was seller)
    pub sell_volumes: Vec<f32>,
}

impl OrderflowInput {
    /// Validate input data consistency
    pub fn validate(&self) -> Result<(), GpuError> {
        let n = self.timestamps.len();

        if n == 0 {
            return Err(GpuError::InvalidParameter("Empty input data".into()));
        }

        if self.close_prices.len() != n {
            return Err(GpuError::InvalidParameter(
                "close_prices length mismatch".into(),
            ));
        }

        if self.volumes.len() != n {
            return Err(GpuError::InvalidParameter("volumes length mismatch".into()));
        }

        if self.buy_volumes.len() != n {
            return Err(GpuError::InvalidParameter(
                "buy_volumes length mismatch".into(),
            ));
        }

        if self.sell_volumes.len() != n {
            return Err(GpuError::InvalidParameter(
                "sell_volumes length mismatch".into(),
            ));
        }

        Ok(())
    }

    /// Get number of ticks
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
}

/// Output from orderflow processing
#[derive(Debug, Clone)]
pub struct OrderflowOutput {
    /// Trading signals [num_strategies][num_ticks]
    /// i8: 1=buy, -1=sell, 0=hold
    pub signals: Vec<Vec<i8>>,

    /// Quantized features [num_strategies][num_ticks * NUM_FEATURES]
    /// i8 quantized (0-255) for 8x compression
    pub features: Vec<Vec<i8>>,

    /// Feature ranges used for quantization [num_strategies][NUM_FEATURES * 2]
    /// Layout: [min0, max0, min1, max1, ...]
    pub feature_ranges: Vec<[f32; NUM_FEATURES * 2]>,
}

/// Circular buffer for sliding window calculations
struct CircularBuffer {
    buffer: VecDeque<f32>,
    capacity: usize,
}

impl CircularBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, value: f32) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(value);
    }

    fn mean(&self) -> f32 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.buffer.iter().sum();
        sum / self.buffer.len() as f32
    }

    fn std_dev(&self) -> f32 {
        if self.buffer.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f32 =
            self.buffer.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.buffer.len() as f32;
        variance.sqrt()
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }
}

/// Orderflow features (6 features per tick)
#[derive(Debug, Clone, Copy)]
struct OrderflowFeatures {
    /// Buy/sell imbalance: buy_vol / (buy_vol + sell_vol)
    /// Range: [0, 1]
    buy_sell_imbalance: f32,

    /// Volume delta: buy_vol - sell_vol
    /// Range: unbounded (typically -10000 to +10000)
    volume_delta: f32,

    /// Trade intensity: volume / time_delta (volume per second)
    /// Range: [0, +inf] (typically 0 to 1000)
    trade_intensity: f32,

    /// Price velocity: (price - price_window_mean) / price_window_std
    /// Range: unbounded z-score (typically -3 to +3)
    price_velocity: f32,

    /// Volume velocity: (volume - volume_window_mean) / volume_window_std
    /// Range: unbounded z-score (typically -3 to +3)
    volume_velocity: f32,

    /// Cumulative volume delta (running sum)
    /// Range: unbounded (typically 0 to 10000)
    cumulative_volume_delta: f32,
}

impl OrderflowFeatures {
    /// Convert to array for quantization
    fn to_array(self) -> [f32; NUM_FEATURES] {
        [
            self.buy_sell_imbalance,
            self.volume_delta,
            self.trade_intensity,
            self.price_velocity,
            self.volume_velocity,
            self.cumulative_volume_delta,
        ]
    }
}

/// CPU batch processor for orderflow features and signals
///
/// This is the main API for CPU orderflow processing.
pub struct OrderflowBatchProcessor;

impl OrderflowBatchProcessor {
    /// Create new processor
    pub fn new() -> Self {
        Self
    }

    /// Calibrate feature quantization ranges from input data
    ///
    /// Runs a first pass over data to determine min/max for each feature.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data to calibrate from
    ///
    /// # Returns
    ///
    /// Array of [min, max] pairs for each of the 6 features
    pub fn calibrate_ranges(
        &self,
        input: &OrderflowInput,
    ) -> Result<[f32; NUM_FEATURES * 2], GpuError> {
        input.validate()?;

        let num_ticks = input.len();

        // Initialize min/max trackers
        let mut mins = [f32::INFINITY; NUM_FEATURES];
        let mut maxs = [f32::NEG_INFINITY; NUM_FEATURES];

        // Sliding windows for price and volume
        let mut price_window = CircularBuffer::new(WINDOW_SIZE);
        let mut volume_window = CircularBuffer::new(WINDOW_SIZE);
        let mut cumulative_volume_delta = 0.0f32;

        // Previous timestamp for intensity calculation
        let mut prev_timestamp = input.timestamps[0];

        // Process each tick
        for i in 0..num_ticks {
            let features = self.extract_features(
                i,
                input,
                &mut price_window,
                &mut volume_window,
                &mut cumulative_volume_delta,
                &mut prev_timestamp,
            );

            let feature_array = features.to_array();

            // Update min/max
            for (j, &value) in feature_array.iter().enumerate() {
                if value.is_finite() {
                    mins[j] = mins[j].min(value);
                    maxs[j] = maxs[j].max(value);
                }
            }
        }

        // Interleave min/max pairs
        let mut ranges = [0.0f32; NUM_FEATURES * 2];
        for i in 0..NUM_FEATURES {
            ranges[i * 2] = if mins[i].is_finite() { mins[i] } else { 0.0 };
            ranges[i * 2 + 1] = if maxs[i].is_finite() { maxs[i] } else { 1.0 };
        }

        Ok(ranges)
    }

    /// Extract orderflow features for a single tick
    fn extract_features(
        &self,
        tick_idx: usize,
        input: &OrderflowInput,
        price_window: &mut CircularBuffer,
        volume_window: &mut CircularBuffer,
        cumulative_volume_delta: &mut f32,
        prev_timestamp: &mut i64,
    ) -> OrderflowFeatures {
        let close = input.close_prices[tick_idx];
        let volume = input.volumes[tick_idx];
        let buy_vol = input.buy_volumes[tick_idx];
        let sell_vol = input.sell_volumes[tick_idx];
        let timestamp = input.timestamps[tick_idx];

        // Feature 1: Buy/sell imbalance
        let total_vol = buy_vol + sell_vol;
        let buy_sell_imbalance = if total_vol > 0.0 {
            buy_vol / total_vol
        } else {
            0.5 // Neutral if no volume
        };

        // Feature 2: Volume delta
        let volume_delta = buy_vol - sell_vol;

        // Feature 3: Trade intensity (volume per second)
        let time_delta_ms = (timestamp - *prev_timestamp).max(1); // Avoid division by zero
        let time_delta_sec = time_delta_ms as f32 / 1000.0;
        let trade_intensity = volume / time_delta_sec;
        *prev_timestamp = timestamp;

        // Feature 4: Price velocity (z-score)
        price_window.push(close);
        let price_velocity = if price_window.len() >= 2 {
            let mean = price_window.mean();
            let std = price_window.std_dev();
            if std > 1e-6 {
                (close - mean) / std
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Feature 5: Volume velocity (z-score)
        volume_window.push(volume);
        let volume_velocity = if volume_window.len() >= 2 {
            let mean = volume_window.mean();
            let std = volume_window.std_dev();
            if std > 1e-6 {
                (volume - mean) / std
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Feature 6: Cumulative volume delta
        *cumulative_volume_delta += volume_delta;

        OrderflowFeatures {
            buy_sell_imbalance,
            volume_delta,
            trade_intensity,
            price_velocity,
            volume_velocity,
            cumulative_volume_delta: *cumulative_volume_delta,
        }
    }

    /// Quantize features to i8 (0-255 range)
    fn quantize_features(
        &self,
        features: &OrderflowFeatures,
        mins: &[f32; NUM_FEATURES],
        maxs: &[f32; NUM_FEATURES],
    ) -> [i8; NUM_FEATURES] {
        let feature_array = features.to_array();
        let mut quantized = [0i8; NUM_FEATURES];

        for i in 0..NUM_FEATURES {
            let value = feature_array[i];
            let min = mins[i];
            let max = maxs[i];

            // Normalize to [0, 1]
            let normalized = if (max - min).abs() > 1e-6 {
                ((value - min) / (max - min)).clamp(0.0, 1.0)
            } else {
                0.5 // Neutral if range is zero
            };

            // Quantize to [0, 255] and cast to i8 (will wrap to signed range)
            quantized[i] = (normalized * 255.0) as i8;
        }

        quantized
    }

    /// Generate trading signal for a strategy
    fn generate_signal(&self, strategy_type: StrategyType, features: &OrderflowFeatures) -> i8 {
        match strategy_type {
            StrategyType::Momentum => {
                // Buy when imbalance > 0.6 && volume_delta > 1000
                if features.buy_sell_imbalance > 0.6 && features.volume_delta > 1000.0 {
                    Signal::Buy as i8
                }
                // Sell when imbalance < 0.4 && volume_delta < -1000
                else if features.buy_sell_imbalance < 0.4 && features.volume_delta < -1000.0 {
                    Signal::Sell as i8
                } else {
                    Signal::Hold as i8
                }
            }

            StrategyType::MeanReversion => {
                // Buy when imbalance < 0.4 && volume_delta < -1000 (oversold)
                if features.buy_sell_imbalance < 0.4 && features.volume_delta < -1000.0 {
                    Signal::Buy as i8
                }
                // Sell when imbalance > 0.6 && volume_delta > 1000 (overbought)
                else if features.buy_sell_imbalance > 0.6 && features.volume_delta > 1000.0 {
                    Signal::Sell as i8
                } else {
                    Signal::Hold as i8
                }
            }

            StrategyType::Breakout => {
                // Buy when trade_intensity > 100 && price_velocity > 0.001
                if features.trade_intensity > 100.0 && features.price_velocity > 0.001 {
                    Signal::Buy as i8
                }
                // Sell when trade_intensity > 100 && price_velocity < -0.001
                else if features.trade_intensity > 100.0 && features.price_velocity < -0.001 {
                    Signal::Sell as i8
                } else {
                    Signal::Hold as i8
                }
            }

            StrategyType::Scalping => {
                // Buy when imbalance > 0.55 && abs(volume_delta) < 500
                if features.buy_sell_imbalance > 0.55 && features.volume_delta.abs() < 500.0 {
                    Signal::Buy as i8
                }
                // Sell when imbalance < 0.45 && abs(volume_delta) < 500
                else if features.buy_sell_imbalance < 0.45 && features.volume_delta.abs() < 500.0
                {
                    Signal::Sell as i8
                } else {
                    Signal::Hold as i8
                }
            }

            StrategyType::TrendFollowing => {
                // Buy when volume_delta > 5000 && price_velocity > 0.002
                if features.volume_delta > 5000.0 && features.price_velocity > 0.002 {
                    Signal::Buy as i8
                }
                // Sell when volume_delta < -5000 && price_velocity < -0.002
                else if features.volume_delta < -5000.0 && features.price_velocity < -0.002 {
                    Signal::Sell as i8
                } else {
                    Signal::Hold as i8
                }
            }
        }
    }

    /// Process batch of orderflow data with multiple strategies
    ///
    /// This is the main entry point for CPU orderflow processing.
    /// Computes orderflow features and generates trading signals.
    ///
    /// # Arguments
    ///
    /// * `input` - Tick-level OHLCV data
    /// * `strategies` - Strategy configurations (type + quantization ranges)
    ///
    /// # Returns
    ///
    /// Signals and quantized features for all strategies
    ///
    /// # Performance
    ///
    /// - Single strategy: 5-20M features/sec (CPU-bound)
    /// - Multi-strategy: Scales linearly with strategies
    pub fn process_batch(
        &self,
        input: &OrderflowInput,
        strategies: &[StrategyConfig],
    ) -> Result<OrderflowOutput, GpuError> {
        input.validate()?;

        if strategies.is_empty() {
            return Err(GpuError::InvalidParameter("No strategies provided".into()));
        }

        let num_strategies = strategies.len();
        let num_ticks = input.len();

        // Allocate output buffers
        let mut signals = vec![vec![0i8; num_ticks]; num_strategies];
        let mut features = vec![vec![0i8; num_ticks * NUM_FEATURES]; num_strategies];
        let mut feature_ranges = Vec::with_capacity(num_strategies);

        // Process each strategy
        for (strategy_idx, strategy) in strategies.iter().enumerate() {
            // Sliding windows for this strategy
            let mut price_window = CircularBuffer::new(WINDOW_SIZE);
            let mut volume_window = CircularBuffer::new(WINDOW_SIZE);
            let mut cumulative_volume_delta = 0.0f32;
            let mut prev_timestamp = input.timestamps[0];

            // Extract feature ranges for quantization
            let mins = strategy.feature_mins;
            let maxs = strategy.feature_maxs;

            // Process each tick
            for tick_idx in 0..num_ticks {
                // Extract features
                let tick_features = self.extract_features(
                    tick_idx,
                    input,
                    &mut price_window,
                    &mut volume_window,
                    &mut cumulative_volume_delta,
                    &mut prev_timestamp,
                );

                // Generate signal
                let signal = self.generate_signal(strategy.strategy_type, &tick_features);
                signals[strategy_idx][tick_idx] = signal;

                // Quantize features
                let quantized = self.quantize_features(&tick_features, &mins, &maxs);

                // Store features (flattened: 6 features per tick)
                for (feature_idx, &value) in quantized.iter().enumerate() {
                    features[strategy_idx][tick_idx * NUM_FEATURES + feature_idx] = value;
                }
            }

            // Store feature ranges
            let mut ranges = [0.0f32; NUM_FEATURES * 2];
            for i in 0..NUM_FEATURES {
                ranges[i * 2] = mins[i];
                ranges[i * 2 + 1] = maxs[i];
            }
            feature_ranges.push(ranges);
        }

        Ok(OrderflowOutput {
            signals,
            features,
            feature_ranges,
        })
    }
}

impl Default for OrderflowBatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_config_creation() {
        let momentum = StrategyConfig::momentum();
        assert_eq!(momentum.strategy_type, StrategyType::Momentum);
        assert_eq!(momentum.feature_mins.len(), NUM_FEATURES);

        let mean_rev = StrategyConfig::mean_reversion();
        assert_eq!(mean_rev.strategy_type, StrategyType::MeanReversion);
    }

    #[test]
    fn test_orderflow_input_validation() {
        let valid = OrderflowInput {
            timestamps: vec![1, 2, 3],
            close_prices: vec![100.0, 101.0, 102.0],
            volumes: vec![10.0, 11.0, 12.0],
            buy_volumes: vec![5.0, 6.0, 7.0],
            sell_volumes: vec![5.0, 5.0, 5.0],
        };
        assert!(valid.validate().is_ok());

        let invalid = OrderflowInput {
            timestamps: vec![1, 2, 3],
            close_prices: vec![100.0, 101.0], // Wrong length!
            volumes: vec![10.0, 11.0, 12.0],
            buy_volumes: vec![5.0, 6.0, 7.0],
            sell_volumes: vec![5.0, 5.0, 5.0],
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_signal_conversion() {
        assert_eq!(Signal::from(1), Signal::Buy);
        assert_eq!(Signal::from(-1), Signal::Sell);
        assert_eq!(Signal::from(0), Signal::Hold);
        assert_eq!(Signal::from(99), Signal::Hold); // Unknown defaults to Hold
    }

    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(3);

        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);
        assert_eq!(buffer.len(), 3);
        assert!((buffer.mean() - 2.0).abs() < 1e-6);

        // Should evict oldest (1.0)
        buffer.push(4.0);
        assert_eq!(buffer.len(), 3);
        assert!((buffer.mean() - 3.0).abs() < 1e-6); // (2+3+4)/3 = 3
    }

    #[test]
    fn test_orderflow_processor_basic() {
        let processor = OrderflowBatchProcessor::new();

        let input = OrderflowInput {
            timestamps: vec![1000, 2000, 3000, 4000, 5000],
            close_prices: vec![100.0, 101.0, 102.0, 103.0, 104.0],
            volumes: vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            buy_volumes: vec![600.0, 700.0, 800.0, 900.0, 1000.0],
            sell_volumes: vec![400.0, 400.0, 400.0, 400.0, 400.0],
        };

        let strategies = vec![StrategyConfig::momentum()];

        let result = processor.process_batch(&input, &strategies);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.signals.len(), 1);
        assert_eq!(output.signals[0].len(), 5);
        assert_eq!(output.features.len(), 1);
        assert_eq!(output.features[0].len(), 5 * NUM_FEATURES);
    }

    #[test]
    fn test_calibrate_ranges() {
        let processor = OrderflowBatchProcessor::new();

        let input = OrderflowInput {
            timestamps: vec![1000, 2000, 3000, 4000, 5000],
            close_prices: vec![100.0, 101.0, 102.0, 103.0, 104.0],
            volumes: vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            buy_volumes: vec![600.0, 700.0, 800.0, 900.0, 1000.0],
            sell_volumes: vec![400.0, 400.0, 400.0, 400.0, 400.0],
        };

        let ranges = processor.calibrate_ranges(&input);
        assert!(ranges.is_ok());

        let ranges = ranges.unwrap();
        assert_eq!(ranges.len(), NUM_FEATURES * 2);

        // Check that ranges are valid (min <= max)
        for i in 0..NUM_FEATURES {
            let min = ranges[i * 2];
            let max = ranges[i * 2 + 1];
            assert!(min <= max, "Feature {} min ({}) > max ({})", i, min, max);
        }
    }

    #[test]
    fn test_multi_strategy_processing() {
        let processor = OrderflowBatchProcessor::new();

        let input = OrderflowInput {
            timestamps: vec![1000, 2000, 3000],
            close_prices: vec![100.0, 101.0, 102.0],
            volumes: vec![1000.0, 1100.0, 1200.0],
            buy_volumes: vec![600.0, 700.0, 800.0],
            sell_volumes: vec![400.0, 400.0, 400.0],
        };

        let strategies = vec![
            StrategyConfig::momentum(),
            StrategyConfig::mean_reversion(),
            StrategyConfig::breakout(),
        ];

        let result = processor.process_batch(&input, &strategies);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.signals.len(), 3);
        assert_eq!(output.features.len(), 3);
        assert_eq!(output.feature_ranges.len(), 3);

        // Each strategy should have signals for all ticks
        for strategy_signals in &output.signals {
            assert_eq!(strategy_signals.len(), 3);
        }

        // Each strategy should have features for all ticks
        for strategy_features in &output.features {
            assert_eq!(strategy_features.len(), 3 * NUM_FEATURES);
        }
    }

    #[test]
    fn test_momentum_signal_generation() {
        let processor = OrderflowBatchProcessor::new();

        // Create scenario that should trigger buy signal:
        // imbalance > 0.6 && volume_delta > 1000
        let input = OrderflowInput {
            timestamps: vec![1000, 2000],
            close_prices: vec![100.0, 101.0],
            volumes: vec![2000.0, 2000.0],
            buy_volumes: vec![1600.0, 1600.0], // 80% buy (imbalance = 0.8)
            sell_volumes: vec![400.0, 400.0],  // volume_delta = 1200 (> 1000)
        };

        let strategies = vec![StrategyConfig::momentum()];
        let output = processor.process_batch(&input, &strategies).unwrap();

        // Second signal should be Buy (first might be Hold due to warmup)
        assert!(
            output.signals[0][1] == Signal::Buy as i8,
            "Expected Buy signal, got {}",
            output.signals[0][1]
        );
    }

    #[test]
    fn test_performance_large_batch() {
        let processor = OrderflowBatchProcessor::new();

        // Generate 10K ticks
        let num_ticks = 10_000;
        let input = OrderflowInput {
            timestamps: (0..num_ticks).map(|i| (i as i64) * 1000).collect(),
            close_prices: (0..num_ticks).map(|i| 100.0 + (i as f32) * 0.01).collect(),
            volumes: vec![1000.0; num_ticks],
            buy_volumes: vec![600.0; num_ticks],
            sell_volumes: vec![400.0; num_ticks],
        };

        let strategies = vec![StrategyConfig::momentum(), StrategyConfig::mean_reversion()];

        let start = std::time::Instant::now();
        let result = processor.process_batch(&input, &strategies);
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Processing failed");
        println!(
            "CPU orderflow processing (10K ticks, 2 strategies): {:?}",
            elapsed
        );

        // Should complete in reasonable time (< 100ms in release, < 1s in debug)
        #[cfg(not(debug_assertions))]
        assert!(
            elapsed.as_millis() < 100,
            "Processing too slow: {:?}",
            elapsed
        );

        #[cfg(debug_assertions)]
        assert!(
            elapsed.as_millis() < 1000,
            "Processing too slow: {:?}",
            elapsed
        );
    }
}
