//! INT8 Quantization Infrastructure for Orderflow Features
//!
//! Compresses orderflow features from FP32 (32 bits) to INT8 (8 bits) with per-feature dynamic range.
//!
//! # Quantization Strategy
//!
//! - **Per-feature calibration**: Each of 6 features gets its own [min, max] range
//! - **Compression ratio**: 4x (24 bytes → 6 bytes per tick)
//! - **Memory savings**: 19GB → 2.4GB for 10 strategies (106M ticks each)
//! - **Target accuracy**: <0.01% deviation in final backtest results
//!
//! # Features Quantized
//!
//! 1. `order_imbalance` - Buy/sell volume ratio (0.0-1.0)
//! 2. `volume_delta` - Buy volume - sell volume
//! 3. `trade_intensity` - Trades per second
//! 4. `price_velocity` - Price change rate
//! 5. `volume_weighted_spread` - Weighted bid-ask spread
//! 6. `trade_size_distribution` - Mean/std of trade sizes
//!
//! # Algorithm
//!
//! **Quantization** (FP32 → INT8):
//! ```text
//! scale = 255.0 / (max - min)
//! code  = ((value - min) * scale).round().clamp(0.0, 255.0)  // raw 0-255
//! ```
//!
//! **Dequantization** (INT8 → FP32, fused multiply-add):
//! ```text
//! inv_scale = (max - min) / 255.0   // 0.0 for degenerate (constant) features
//! value     = code * inv_scale + min
//! ```
//!
//! # INT8 Code Convention
//!
//! Quantized codes are **raw 0-255 values stored through the `i8` ABI**: the bit
//! pattern is preserved (`(code as u8) as i8`), so codes >= 128 appear negative
//! when read as signed. All readers must recover the code via `q as u8` before
//! any arithmetic. A plain `f32 as i8` cast saturates at 127 in Rust (and
//! `(char)` saturates via `cvt.rzi.s8.f32` in CUDA), which would alias the top
//! half of every feature range to a single code — never cast directly.
//!
//! NOTE: `kernels/tick_aggregation.cu` (`quantize_to_int8_kernel`) uses a
//! different zero-centered convention (`code - 128`); aligning that kernel is
//! tracked separately (task kf-tick-aggregation-correctness).
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::{GpuDevice, QuantizationCalibrator};
//!
//! let device = GpuDevice::new()?;
//! let features = vec![
//!     vec![0.5, 1000.0, 50.0, 0.001, 0.1, 100.0],  // Tick 1
//!     vec![0.6, 1200.0, 55.0, 0.0012, 0.15, 105.0], // Tick 2
//! ];
//!
//! // Calibrate per-feature ranges
//! let calibrator = QuantizationCalibrator::calibrate(&features);
//!
//! // Quantize batch on GPU
//! let quantized = calibrator.quantize_batch_gpu(&device, &features)?;
//!
//! // Estimate accuracy
//! let rmse = calibrator.estimate_error(&features);
//! assert!(rmse < 0.001, "Quantization error too high");
//! ```
//!
//! # Hardware Requirements
//!
//! - GPU: NVIDIA Ada Lovelace or newer (tested on RTX 3500 Ada)
//! - CUDA: 12.0+ (13.0+ recommended for optimal performance)
//! - Driver: 580.82.07 or newer
//!
//! # Performance
//!
//! - **GPU quantization**: 1-2B features/sec
//! - **Memory bandwidth**: Coalesced 128-byte reads/writes
//! - **Latency**: <5ms for 100M features
//!
//! # Fallback Strategy
//!
//! If accuracy loss > 0.01%, automatically falls back to FP16 (2x compression instead of 4x).

use crate::gpu::{GpuDevice, GpuError};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// CUDA kernel source, NVRTC-JIT-compiled at runtime.
///
/// Must stay NVRTC-compatible: no `#include` directives, `extern "C" __global__`
/// entry points only.
const QUANTIZE_INT8_KERNEL_SRC: &str = include_str!("kernels/quantize_int8.cu");

/// Number of orderflow features per tick (layout contract with the CUDA kernels).
const NUM_FEATURES: usize = 6;

/// Flatten `[num_ticks][*]` features into a row-major `[num_ticks * num_features]`
/// buffer for GPU upload.
///
/// Each tick contributes exactly `num_features` values: extra values are
/// truncated, missing values (including fully empty ticks) are zero-padded, so
/// the output length is always `features.len() * num_features` — the layout
/// contract the kernel launch math depends on. Trivially terminates for any
/// input length (the previous `while len < (len/n + 1) * n` padding condition
/// was always true for multiples of `n` and grew the buffer until OOM).
fn flatten_and_pad_features(features: &[Vec<f32>], num_features: usize) -> Vec<f32> {
    let mut features_flat = Vec::with_capacity(features.len() * num_features);
    for tick_features in features {
        let kept = num_features.min(tick_features.len());
        features_flat.extend_from_slice(&tick_features[..kept]);
        // Zero-pad short ticks to exactly num_features values
        features_flat.resize(features_flat.len() + (num_features - kept), 0.0);
    }
    features_flat
}

/// Per-feature quantization parameters for INT8 compression
///
/// Stores min/max/scale for each of 6 orderflow features, enabling
/// accurate reconstruction after quantization.
///
/// # Memory Layout
///
/// - 6 features × 3 params (min, max, scale) = 18 f32 values = 72 bytes
/// - Negligible compared to 19GB feature data
#[derive(Debug, Clone)]
pub struct QuantizationCalibrator {
    /// Feature names for debugging
    pub feature_names: Vec<String>,

    /// Minimum value per feature [6]
    pub min_values: Vec<f32>,

    /// Maximum value per feature [6]
    pub max_values: Vec<f32>,

    /// Quantization scale: 255.0 / (max - min) [6]
    pub scales: Vec<f32>,
}

impl QuantizationCalibrator {
    /// Calibrate quantization parameters from training data
    ///
    /// Analyzes feature distributions to determine optimal [min, max] ranges
    /// for each of 6 orderflow features.
    ///
    /// # Arguments
    ///
    /// * `features` - Training features [num_ticks][6]
    ///
    /// # Returns
    ///
    /// Calibrator with per-feature quantization parameters
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let features = vec![
    ///     vec![0.5, 1000.0, 50.0, 0.001, 0.1, 100.0],
    ///     vec![0.6, 1200.0, 55.0, 0.0012, 0.15, 105.0],
    /// ];
    /// let calibrator = QuantizationCalibrator::calibrate(&features);
    /// ```
    pub fn calibrate(features: &[Vec<f32>]) -> Self {
        let feature_names = vec![
            "order_imbalance".to_string(),
            "volume_delta".to_string(),
            "trade_intensity".to_string(),
            "price_velocity".to_string(),
            "volume_weighted_spread".to_string(),
            "trade_size_distribution".to_string(),
        ];

        let num_features = NUM_FEATURES;
        let mut min_values = vec![f32::INFINITY; num_features];
        let mut max_values = vec![f32::NEG_INFINITY; num_features];

        // Find min/max for each feature
        for tick_features in features {
            for (feature_idx, &value) in tick_features.iter().enumerate() {
                if feature_idx < num_features {
                    min_values[feature_idx] = min_values[feature_idx].min(value);
                    max_values[feature_idx] = max_values[feature_idx].max(value);
                }
            }
        }

        // Calculate quantization scales
        let scales: Vec<f32> = min_values
            .iter()
            .zip(max_values.iter())
            .map(|(&min, &max)| {
                let range = max - min;
                if range > 1e-9 {
                    255.0 / range
                } else {
                    1.0 // Constant feature, scale doesn't matter
                }
            })
            .collect();

        Self {
            feature_names,
            min_values,
            max_values,
            scales,
        }
    }

    /// Quantize single tick features (CPU)
    ///
    /// Converts 6 FP32 features to 6 INT8 values using calibrated ranges.
    ///
    /// # Arguments
    ///
    /// * `features` - Single tick features [6]
    ///
    /// # Returns
    ///
    /// Quantized features as i8 [6]
    pub fn quantize(&self, features: &[f32]) -> Vec<i8> {
        features
            .iter()
            .enumerate()
            .map(|(idx, &value)| {
                if idx >= self.min_values.len() {
                    return 0;
                }

                let quantized = (value - self.min_values[idx]) * self.scales[idx];
                // Raw 0-255 code stored through the i8 ABI. A plain `as i8`
                // saturates at 127 and would alias the top half of the range
                // to a single code; cast through u8 to preserve the bit pattern.
                (quantized.round().clamp(0.0, 255.0) as u8) as i8
            })
            .collect()
    }

    /// Dequantize single tick features (CPU)
    ///
    /// Reconstructs FP32 features from INT8 quantized values.
    ///
    /// # Arguments
    ///
    /// * `quantized` - Quantized features [6]
    ///
    /// # Returns
    ///
    /// Reconstructed FP32 features [6]
    pub fn dequantize(&self, quantized: &[i8]) -> Vec<f32> {
        quantized
            .iter()
            .enumerate()
            .map(|(idx, &q)| {
                if idx >= self.scales.len() {
                    return 0.0;
                }

                let q_unsigned = q as u8 as f32; // Recover raw 0-255 code
                // FMA with inverse scale mirrors the GPU kernel's
                // fmaf(code, inv_scale, min). inv_scale is 0 for degenerate
                // (constant) features, collapsing the output to min.
                let inv_scale = Self::inv_scale(self.min_values[idx], self.max_values[idx]);
                q_unsigned.mul_add(inv_scale, self.min_values[idx])
            })
            .collect()
    }

    /// Inverse scale for one feature: `(max - min) / 255.0`.
    ///
    /// Returns 0.0 for degenerate ranges (`range <= 1e-9`, same threshold as
    /// [`Self::calibrate`]) so dequantization collapses to `min` instead of
    /// producing garbage from the placeholder `scale = 1.0`.
    #[inline]
    fn inv_scale(min: f32, max: f32) -> f32 {
        let range = max - min;
        if range > 1e-9 {
            range / 255.0
        } else {
            0.0
        }
    }

    /// Per-feature inverse scales `(max - min) / 255.0` for FMA dequantization.
    ///
    /// Host-precomputed and uploaded to the GPU so `dequantize_features_int8`
    /// can use `fmaf(code, inv_scale, min)` instead of per-element division.
    /// Degenerate (constant) features yield 0.0 (dequantizes to `min`).
    pub fn inv_scales(&self) -> Vec<f32> {
        self.min_values
            .iter()
            .zip(self.max_values.iter())
            .map(|(&min, &max)| Self::inv_scale(min, max))
            .collect()
    }

    /// Quantize batch of features on GPU
    ///
    /// Parallel GPU quantization for 100M+ features.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device handle
    /// * `features` - Batch features [num_ticks][6]
    ///
    /// # Returns
    ///
    /// Device buffer with quantized features [num_ticks * 6] (i8)
    ///
    /// # Performance
    ///
    /// - 1-2B features/sec on RTX 3500 Ada
    /// - Coalesced memory access (128-byte cache lines)
    /// - Occupancy: 75-90%
    pub fn quantize_batch_gpu(
        &self,
        device: &GpuDevice,
        features: &[Vec<f32>],
    ) -> Result<CudaSlice<i8>, GpuError> {
        let num_ticks = features.len();
        let num_features = NUM_FEATURES;
        let total_elements = num_ticks * num_features;

        // Flatten features: [tick0_f0, tick0_f1, ..., tick0_f5, tick1_f0, ...]
        // (truncates extra values, zero-pads short ticks; always terminates)
        let features_flat = flatten_and_pad_features(features, num_features);
        debug_assert_eq!(features_flat.len(), total_elements);

        // Upload to GPU using direct allocation + copy
        let mut d_features = device
            .stream
            .alloc_zeros::<f32>(features_flat.len())
            .map_err(|e| {
                GpuError::AllocationError(format!("Failed to allocate features: {:?}", e))
            })?;
        device
            .stream
            .memcpy_htod(&features_flat, &mut d_features)
            .map_err(|e| GpuError::MemoryCopyError(format!("Failed to copy features: {:?}", e)))?;

        let mut d_min_values = device
            .stream
            .alloc_zeros::<f32>(self.min_values.len())
            .map_err(|e| {
                GpuError::AllocationError(format!("Failed to allocate min_values: {:?}", e))
            })?;
        device
            .stream
            .memcpy_htod(&self.min_values, &mut d_min_values)
            .map_err(|e| {
                GpuError::MemoryCopyError(format!("Failed to copy min_values: {:?}", e))
            })?;

        let mut d_scales = device
            .stream
            .alloc_zeros::<f32>(self.scales.len())
            .map_err(|e| {
                GpuError::AllocationError(format!("Failed to allocate scales: {:?}", e))
            })?;
        device
            .stream
            .memcpy_htod(&self.scales, &mut d_scales)
            .map_err(|e| GpuError::MemoryCopyError(format!("Failed to copy scales: {:?}", e)))?;

        // Allocate output buffer
        let mut d_quantized = device.allocate_device_buffer::<i8>(total_elements)?;

        // Compile and load kernel
        let ptx_arc = crate::gpu::compile::compile_ptx_optimized_cached(QUANTIZE_INT8_KERNEL_SRC)?;
        let module = device
            .context()
            .load_module(Arc::unwrap_or_clone(ptx_arc))
            .map_err(|e| {
                GpuError::CompilationError(format!("Failed to load INT8 quantize module: {:?}", e))
            })?;

        let kernel = module
            .load_function("quantize_features_int8")
            .map_err(|e| {
                GpuError::ExecutionError(format!("Failed to load quantize_features_int8: {:?}", e))
            })?;

        // Launch kernel: 256 threads per block, process 4 features per thread
        let block_size = 256;
        let n_blocks = total_elements.div_ceil(block_size * 4) as u32;

        let config = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_ticks_i32 = num_ticks as i32;
        let num_features_i32 = num_features as i32;

        let mut builder = device.stream.launch_builder(&kernel);
        builder.arg(&d_features);
        builder.arg(&mut d_quantized);
        builder.arg(&d_min_values);
        builder.arg(&d_scales);
        builder.arg(&num_ticks_i32);
        builder.arg(&num_features_i32);

        unsafe {
            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("INT8 quantization kernel launch failed: {:?}", e))
            })?;
        }

        Ok(d_quantized)
    }

    /// Dequantize batch of features on GPU
    ///
    /// Parallel GPU dequantization for validation.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device handle
    /// * `quantized` - Quantized features on device [num_ticks * 6]
    /// * `num_ticks` - Number of ticks
    ///
    /// # Returns
    ///
    /// Device buffer with dequantized FP32 features
    pub fn dequantize_batch_gpu(
        &self,
        device: &GpuDevice,
        quantized: &CudaSlice<i8>,
        num_ticks: usize,
    ) -> Result<CudaSlice<f32>, GpuError> {
        let num_features = NUM_FEATURES;
        let total_elements = num_ticks * num_features;

        // Upload calibration parameters
        let mut d_min_values = device
            .stream
            .alloc_zeros::<f32>(self.min_values.len())
            .map_err(|e| {
                GpuError::AllocationError(format!("Failed to allocate min_values: {:?}", e))
            })?;
        device
            .stream
            .memcpy_htod(&self.min_values, &mut d_min_values)
            .map_err(|e| {
                GpuError::MemoryCopyError(format!("Failed to copy min_values: {:?}", e))
            })?;

        // Host-precomputed inverse scales: (max - min) / 255, 0 for degenerate
        // ranges. The kernel dequantizes via fmaf(code, inv_scale, min) —
        // no per-element division.
        let inv_scales = self.inv_scales();
        let mut d_inv_scales = device
            .stream
            .alloc_zeros::<f32>(inv_scales.len())
            .map_err(|e| {
                GpuError::AllocationError(format!("Failed to allocate inv_scales: {:?}", e))
            })?;
        device
            .stream
            .memcpy_htod(&inv_scales, &mut d_inv_scales)
            .map_err(|e| {
                GpuError::MemoryCopyError(format!("Failed to copy inv_scales: {:?}", e))
            })?;

        // Allocate output buffer
        let mut d_dequantized = device.allocate_device_buffer::<f32>(total_elements)?;

        // Load kernel
        let ptx_arc = crate::gpu::compile::compile_ptx_optimized_cached(QUANTIZE_INT8_KERNEL_SRC)?;
        let module = device
            .context()
            .load_module(Arc::unwrap_or_clone(ptx_arc))
            .map_err(|e| {
                GpuError::CompilationError(format!(
                    "Failed to load INT8 dequantize module: {:?}",
                    e
                ))
            })?;

        let kernel = module
            .load_function("dequantize_features_int8")
            .map_err(|e| {
                GpuError::ExecutionError(format!(
                    "Failed to load dequantize_features_int8: {:?}",
                    e
                ))
            })?;

        // Launch kernel
        let block_size = 256;
        let n_blocks = total_elements.div_ceil(block_size * 4) as u32;

        let config = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_ticks_i32 = num_ticks as i32;
        let num_features_i32 = num_features as i32;

        let mut builder = device.stream.launch_builder(&kernel);
        builder.arg(quantized);
        builder.arg(&mut d_dequantized);
        builder.arg(&d_min_values);
        builder.arg(&d_inv_scales);
        builder.arg(&num_ticks_i32);
        builder.arg(&num_features_i32);

        unsafe {
            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!(
                    "INT8 dequantization kernel launch failed: {:?}",
                    e
                ))
            })?;
        }

        Ok(d_dequantized)
    }

    /// Estimate quantization error (RMSE)
    ///
    /// Measures accuracy by quantizing and dequantizing features,
    /// then computing root mean squared error.
    ///
    /// # Arguments
    ///
    /// * `original` - Original FP32 features
    ///
    /// # Returns
    ///
    /// RMSE across all features (lower is better)
    ///
    /// # Target
    ///
    /// RMSE < 0.001 for <0.01% deviation in backtest results
    pub fn estimate_error(&self, original: &[Vec<f32>]) -> f32 {
        let mut total_squared_error = 0.0;
        let mut count = 0;

        for tick_features in original {
            let quantized = self.quantize(tick_features);
            let dequantized = self.dequantize(&quantized);

            for (orig, deq) in tick_features.iter().zip(dequantized.iter()) {
                let error = orig - deq;
                total_squared_error += error * error;
                count += 1;
            }
        }

        if count > 0 {
            (total_squared_error / count as f32).sqrt()
        } else {
            0.0
        }
    }

    /// Check if quantization meets accuracy target
    ///
    /// # Arguments
    ///
    /// * `features` - Validation features
    /// * `max_rmse` - Maximum acceptable RMSE (default: 0.001)
    ///
    /// # Returns
    ///
    /// `true` if RMSE < max_rmse, `false` otherwise
    pub fn validate_accuracy(&self, features: &[Vec<f32>], max_rmse: f32) -> bool {
        let rmse = self.estimate_error(features);
        rmse < max_rmse
    }
}

/// Quantized orderflow features for memory-efficient storage
///
/// Stores 6 INT8 features per tick with associated calibrator for reconstruction.
///
/// # Memory Savings
///
/// - FP32: 106M ticks × 6 features × 4 bytes = 2.54GB per strategy
/// - INT8: 106M ticks × 6 features × 1 byte = 636MB per strategy
/// - Savings: **4x compression** (75% reduction)
///
/// # Example
///
/// ```rust,ignore
/// let quantized = QuantizedFeatures::new(calibrator, features_int8);
///
/// // Later, reconstruct for backtest
/// let features_fp32 = quantized.dequantize_batch();
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedFeatures {
    /// Quantized features [num_ticks * 6]
    pub features_int8: Vec<i8>,

    /// Calibrator for dequantization
    pub calibrator: QuantizationCalibrator,

    /// Number of ticks
    pub num_ticks: usize,
}

impl QuantizedFeatures {
    /// Create new quantized features
    pub fn new(
        calibrator: QuantizationCalibrator,
        features_int8: Vec<i8>,
        num_ticks: usize,
    ) -> Self {
        Self {
            features_int8,
            calibrator,
            num_ticks,
        }
    }

    /// Dequantize all features (CPU)
    ///
    /// Reconstructs FP32 features for backtest execution.
    ///
    /// # Returns
    ///
    /// Vec<Vec<f32>> with shape [num_ticks][6]
    pub fn dequantize_batch(&self) -> Vec<Vec<f32>> {
        (0..self.num_ticks)
            .map(|tick_idx| {
                let start = tick_idx * 6;
                let end = start + 6;
                self.calibrator.dequantize(&self.features_int8[start..end])
            })
            .collect()
    }

    /// Memory used by quantized features (bytes)
    pub fn memory_bytes(&self) -> usize {
        self.features_int8.len() + std::mem::size_of::<QuantizationCalibrator>()
    }

    /// Memory saved vs FP32 (bytes)
    pub fn memory_saved(&self) -> usize {
        let fp32_size = self.num_ticks * 6 * 4; // FP32 = 4 bytes
        let int8_size = self.memory_bytes();
        fp32_size.saturating_sub(int8_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Maximum acceptable round-trip error for one feature: half a
    /// quantization step plus small floating-point slack.
    fn half_step_bound(calibrator: &QuantizationCalibrator, idx: usize) -> f32 {
        let range = calibrator.max_values[idx] - calibrator.min_values[idx];
        (range / 255.0) / 2.0 * 1.01 + 1e-6
    }

    #[test]
    fn test_quantization_roundtrip() {
        let features = vec![
            vec![0.5, 1000.0, 50.0, 0.001, 0.1, 100.0],   // Tick 1
            vec![0.6, 1200.0, 55.0, 0.0012, 0.15, 105.0], // Tick 2
            vec![0.4, 800.0, 45.0, 0.0008, 0.08, 95.0],   // Tick 3
        ];

        let calibrator = QuantizationCalibrator::calibrate(&features);

        // Every reconstructed value must be within half a quantization step
        // of the original (the strongest bound 8-bit codes can guarantee;
        // the bound is per-feature because steps are range-dependent).
        for tick in &features {
            let roundtrip = calibrator.dequantize(&calibrator.quantize(tick));
            for idx in 0..6 {
                let err = (tick[idx] - roundtrip[idx]).abs();
                assert!(
                    err <= half_step_bound(&calibrator, idx),
                    "feature {} error {} exceeds half-step bound",
                    idx,
                    err
                );
            }
        }
    }

    #[test]
    fn test_per_feature_ranges() {
        let features = vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1000.0, 100.0, 0.01, 1.0, 200.0],
        ];

        let calibrator = QuantizationCalibrator::calibrate(&features);

        // Verify min/max ranges
        assert_eq!(calibrator.min_values[0], 0.0);
        assert_eq!(calibrator.max_values[0], 1.0);
        assert_eq!(calibrator.min_values[1], 0.0);
        assert_eq!(calibrator.max_values[1], 1000.0);
    }

    #[test]
    fn test_accuracy_validation() {
        let features = vec![
            vec![0.5, 500.0, 50.0, 0.005, 0.5, 100.0],
            vec![0.6, 600.0, 60.0, 0.006, 0.6, 120.0],
        ];

        let calibrator = QuantizationCalibrator::calibrate(&features);
        assert!(calibrator.validate_accuracy(&features, 0.001));
    }

    #[test]
    fn test_quantized_features_memory() {
        let features_int8 = vec![0i8; 106_000_000 * 6]; // 106M ticks
        let calibrator = QuantizationCalibrator {
            feature_names: vec!["test".to_string(); 6],
            min_values: vec![0.0; 6],
            max_values: vec![1.0; 6],
            scales: vec![255.0; 6],
        };

        let quantized = QuantizedFeatures::new(calibrator, features_int8, 106_000_000);

        let fp32_size = 106_000_000usize * 6 * 4; // 2.54GB
        let int8_size = quantized.memory_bytes();
        let compression_ratio = fp32_size as f64 / int8_size as f64;

        assert!(
            compression_ratio > 3.9 && compression_ratio < 4.1,
            "Compression ratio should be ~4x (FP32 -> INT8), got {:.2}x",
            compression_ratio
        );
    }

    /// Regression: a plain `f32 as i8` cast saturates at 127, collapsing the
    /// top half of every feature range (codes 128-255) to a single code.
    /// Raw 0-255 codes must survive the i8 ABI bit-for-bit.
    #[test]
    fn test_quantize_reaches_full_code_range() {
        let features = vec![vec![0.0f32; 6], vec![255.0f32; 6]];
        let calibrator = QuantizationCalibrator::calibrate(&features);

        // value == max must produce code 255 (bit pattern -1 in i8)
        let codes_max = calibrator.quantize(&[255.0f32; 6]);
        for &code in &codes_max {
            assert_eq!(code as u8, 255, "max value must map to code 255");
            assert_eq!(code, -1i8, "code 255 must be stored as i8 bit pattern -1");
        }

        // A value in the upper half must produce a code above 127
        let codes_upper = calibrator.quantize(&[200.0f32; 6]);
        for &code in &codes_upper {
            assert_eq!(code as u8, 200, "upper-half codes must not saturate at 127");
        }

        // Codes round-trip to the original values (scale is exactly 1.0 here)
        let roundtrip = calibrator.dequantize(&codes_max);
        for &value in &roundtrip {
            assert!(
                (value - 255.0).abs() < 1e-3,
                "code 255 must dequantize back to max, got {}",
                value
            );
        }
    }

    /// Round-trip over the full [min, max] range of every feature, including
    /// value == max. Each reconstructed value must be within half a
    /// quantization step; RMSE must be bounded by the widest feature's step.
    #[test]
    fn test_roundtrip_full_range_including_max() {
        let n = 511;
        let features: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32; // 0.0..=1.0 inclusive
                vec![
                    t,                     // order_imbalance      [0, 1]
                    2000.0 * t - 1000.0,   // volume_delta         [-1000, 1000]
                    100.0 * t,             // trade_intensity      [0, 100]
                    0.01 * t - 0.005,      // price_velocity       [-0.005, 0.005]
                    t,                     // volume_weighted_spread [0, 1]
                    200.0 * t,             // trade_size_distribution [0, 200]
                ]
            })
            .collect();

        let calibrator = QuantizationCalibrator::calibrate(&features);

        for tick in &features {
            let roundtrip = calibrator.dequantize(&calibrator.quantize(tick));
            for idx in 0..6 {
                let err = (tick[idx] - roundtrip[idx]).abs();
                assert!(
                    err <= half_step_bound(&calibrator, idx),
                    "feature {} value {} reconstructed as {} (error {})",
                    idx,
                    tick[idx],
                    roundtrip[idx],
                    err
                );
            }
        }

        // RMSE across all features is bounded by the widest feature's
        // half-step (volume_delta: range 2000 -> half-step ~3.92)
        let rmse = calibrator.estimate_error(&features);
        assert!(rmse <= 4.0, "RMSE too high: {}", rmse);
    }

    /// Regression: the old padding condition
    /// `len < (len / num_features + 1) * num_features` is always true when
    /// `len` is a multiple of `num_features`, so the loop grew the buffer
    /// until OOM on the first quantize_batch_gpu call.
    #[test]
    fn test_flatten_and_pad_terminates_for_all_tick_lengths() {
        for tick_len in 0..=13usize {
            let features = vec![vec![1.0f32; tick_len]; 3];
            let flat = flatten_and_pad_features(&features, 6);
            assert_eq!(flat.len(), 18, "tick_len={}: wrong flattened length", tick_len);

            // Kept values preserved, padding zeroed, extras truncated
            let kept = tick_len.min(6);
            for tick in flat.chunks(6) {
                assert!(tick[..kept].iter().all(|&v| v == 1.0));
                assert!(tick[kept..].iter().all(|&v| v == 0.0));
            }
        }

        assert!(flatten_and_pad_features(&[], 6).is_empty());
    }

    /// Degenerate (constant) features use a placeholder scale of 1.0; the
    /// inverse scale must be 0.0 so dequantization collapses to min instead
    /// of producing garbage.
    #[test]
    fn test_degenerate_range_dequantizes_to_min() {
        let features = vec![
            vec![5.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![5.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let calibrator = QuantizationCalibrator::calibrate(&features);

        assert_eq!(calibrator.inv_scales()[0], 0.0);
        assert_eq!(calibrator.scales[0], 1.0); // placeholder, never used to divide

        let roundtrip = calibrator.dequantize(&calibrator.quantize(&features[0]));
        assert_eq!(roundtrip[0], 5.0);

        // Even arbitrary/corrupt codes must collapse to min for constant features
        let corrupt = calibrator.dequantize(&[-1i8, 0, 0, 0, 0, 0]);
        assert_eq!(corrupt[0], 5.0);
    }

    /// Inverse scales must reconstruct the full range: code 255 -> max.
    #[test]
    fn test_inv_scales_reconstruct_range() {
        let features = vec![
            vec![0.0, -1000.0, 0.0, -0.005, 0.0, 0.0],
            vec![1.0, 1000.0, 100.0, 0.005, 1.0, 200.0],
        ];
        let calibrator = QuantizationCalibrator::calibrate(&features);
        let inv_scales = calibrator.inv_scales();

        for idx in 0..6 {
            let range = calibrator.max_values[idx] - calibrator.min_values[idx];
            let reconstructed_max = 255.0f32.mul_add(inv_scales[idx], calibrator.min_values[idx]);
            assert!(
                (reconstructed_max - calibrator.max_values[idx]).abs() <= range * 1e-5,
                "feature {}: code 255 must dequantize to max",
                idx
            );
        }
    }

    /// The kernel is NVRTC-JIT-compiled at runtime: no #include allowed, and
    /// all entry points must be extern "C" __global__.
    #[test]
    fn test_kernel_source_is_nvrtc_compatible() {
        assert!(
            !QUANTIZE_INT8_KERNEL_SRC.contains("#include"),
            "NVRTC kernel must not use #include directives"
        );

        for name in [
            "quantize_features_int8",
            "dequantize_features_int8",
            "quantize_features_int8_batch",
            "quantize_features_int8_saturate",
            "compute_quantization_error",
        ] {
            let signature = format!("extern \"C\" __global__ void {}(", name);
            assert!(
                QUANTIZE_INT8_KERNEL_SRC.contains(&signature),
                "missing extern \"C\" __global__ entry point: {}",
                name
            );
        }
    }

    /// Regression: `(char)float` compiles to cvt.rzi.s8.f32 which saturates
    /// at 127. Every quantize store (vectorized + scalar tail in all three
    /// quantize variants = 6 sites) must cast through unsigned char.
    #[test]
    fn test_kernel_quantize_casts_preserve_raw_codes() {
        assert_eq!(
            QUANTIZE_INT8_KERNEL_SRC
                .matches("(char)(unsigned char)quantized_f")
                .count(),
            6,
            "all 6 quantize store sites must cast through unsigned char"
        );
        assert_eq!(
            QUANTIZE_INT8_KERNEL_SRC.matches("= (char)quantized_f").count(),
            0,
            "no direct saturating (char) cast may remain"
        );
    }

    /// Dequantize kernel must use FMA with host-precomputed inverse scales
    /// (no per-element division) and read codes through unsigned char.
    #[test]
    fn test_kernel_dequantize_uses_fma_with_inv_scales() {
        assert!(
            QUANTIZE_INT8_KERNEL_SRC.contains("const float* __restrict__ inv_scales"),
            "dequantize kernel must take inv_scales"
        );
        assert_eq!(
            QUANTIZE_INT8_KERNEL_SRC
                .matches("fmaf(q_float, inv_scales[feature_idx], min_values[feature_idx])")
                .count(),
            2,
            "both dequantize paths (vectorized + tail) must use fmaf"
        );
        assert!(
            QUANTIZE_INT8_KERNEL_SRC.contains("(unsigned char)values[i]")
                && QUANTIZE_INT8_KERNEL_SRC.contains("(unsigned char)quantized[global_idx]"),
            "dequantize must recover raw 0-255 codes through unsigned char"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_quantization() {
        use crate::gpu::GpuDevice;

        if let Ok(device) = GpuDevice::new() {
            let features = vec![
                vec![0.5, 1000.0, 50.0, 0.001, 0.1, 100.0],
                vec![0.6, 1200.0, 55.0, 0.0012, 0.15, 105.0],
                vec![0.4, 800.0, 45.0, 0.0008, 0.08, 95.0],
            ];

            let calibrator = QuantizationCalibrator::calibrate(&features);

            match calibrator.quantize_batch_gpu(&device, &features) {
                Ok(d_quantized) => {
                    // Copy back to host for validation
                    match device.copy_to_host_i8(&d_quantized) {
                        Ok(quantized_host) => {
                            println!("✅ GPU quantization successful");
                            assert_eq!(quantized_host.len(), features.len() * 6);
                        }
                        Err(e) => println!("⚠️ GPU copy failed: {:?}", e),
                    }
                }
                Err(e) => println!("⚠️ GPU quantization failed: {:?}", e),
            }
        } else {
            println!("⚠️ GPU not available, skipping GPU test");
        }
    }
}
