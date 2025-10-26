//! GPU-Accelerated Keltner Channels - CPU-GPU Hybrid
//!
//! Provides improved performance using CPU-GPU hybrid architecture.
//! Keltner Channels are volatility-based envelopes around an EMA, similar to Bollinger Bands
//! but using ATR (Average True Range) instead of standard deviation.
//!
//! # Hybrid Architecture (v0.2.0)
//!
//! - **CPU**: EMA calculation (~25μs) - Sequential, faster on CPU
//! - **GPU**: ATR calculation (hybrid, ~163μs) - Uses `atr_gpu` hybrid
//! - **GPU**: Parallel band calculation (~10μs)
//! - **Total**: ~198μs (vs ~368μs for old pure-GPU)
//!
//! # Why Hybrid?
//!
//! Both EMA and ATR have sequential components (IIR filters) that cannot be parallelized.
//! Single-thread GPU kernels are 6-8x slower than CPU for these operations.
//!
//! - **Old (v0.1.0 - Anti-pattern)**:
//!   - GPU: Single-thread EMA (~130μs) ← Bottleneck!
//!   - GPU: Single-thread ATR (~238μs) ← Bottleneck!
//!   - GPU: Parallel bands (~10μs)
//!   - **Total**: ~378μs
//!
//! - **New (v0.2.0 - Hybrid)**:
//!   - CPU: EMA (~25μs) ← 5x faster!
//!   - GPU: ATR hybrid (~163μs) ← 1.5x faster!
//!   - GPU: Parallel bands (~10μs)
//!   - **Total**: ~198μs (1.9x faster!)
//!
//! # Algorithm
//!
//! 1. **CPU**: Middle = EMA(close, ema_period) - typically 20
//! 2. **GPU Hybrid**: ATR = Average True Range(high, low, close, atr_period) - typically 10
//! 3. **GPU Parallel**: Upper = Middle + (ATR * multiplier), Lower = Middle - (ATR * multiplier)
//!
//! # Advantages over Bollinger Bands
//!
//! - ATR adapts to true volatility (includes gaps)
//! - Less prone to whipsaws in choppy markets
//! - More responsive to expanding/contracting ranges

use super::atr::atr_gpu;
use super::device::{GpuDevice, GpuError};
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use crate::gpu::compile::compile_ptx_optimized;
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for Keltner Channels calculation (Hybrid v0.2.0)
///
/// Only contains parallel band calculation kernel.
/// EMA is calculated on CPU (6x faster than single-thread GPU).
/// ATR is calculated using hybrid `atr_gpu` (GPU parallel TR + CPU Wilder's).
const KELTNER_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

// Calculate Keltner Channel bands (PARALLEL - Good for GPU)
// Upper = EMA + (ATR * multiplier)
// Middle = EMA
// Lower = EMA - (ATR * multiplier)
extern "C" __global__ void keltner_bands_kernel(
    const double* __restrict__ ema,
    const double* __restrict__ atr,
    double* __restrict__ upper,
    double* __restrict__ middle,
    double* __restrict__ lower,
    double multiplier,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Check if both EMA and ATR are valid
    if (!isnan(ema[idx]) && !isnan(atr[idx])) {
        middle[idx] = ema[idx];
        double offset = atr[idx] * multiplier;
        upper[idx] = ema[idx] + offset;
        lower[idx] = ema[idx] - offset;
    } else {
        // Not enough data yet
        middle[idx] = CUDART_NAN;
        upper[idx] = CUDART_NAN;
        lower[idx] = CUDART_NAN;
    }
}
"#;

/// GPU-accelerated Keltner Channels - CPU-GPU Hybrid
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `ema_period` - EMA period for middle line (typically 20)
/// * `atr_period` - ATR period for volatility (typically 10)
/// * `atr_multiplier` - Multiplier for ATR bands (typically 2.0)
/// * `stream` - Optional CUDA stream for concurrent execution (None uses device default)
///
/// # Returns
///
/// Tuple of (upper_band, middle_band, lower_band) as Array1<f64>
/// Early values will be NaN until enough data is available.
///
/// # Performance
///
/// Expected performance: **~198μs** for 100K candles (1.9x faster than old pure-GPU)
///
/// Breakdown:
/// - CPU EMA: ~25μs
/// - GPU ATR (hybrid): ~163μs
/// - GPU parallel bands: ~10μs
/// - **Total**: ~198μs
///
/// Old pure-GPU: ~378μs (two single-thread bottlenecks)
///
/// # Stream Concurrency
///
/// When a stream is provided, kernel launches execute on that stream, enabling
/// concurrent execution with other operations on different streams. This is used
/// in the batch pipeline for 4-6x speedup across Fast/Medium/Slow indicator groups.
///
/// Classification: **MEDIUM** indicator (hybrid CPU-GPU approach)
///
/// # Why Hybrid?
///
/// Both EMA and ATR contain sequential IIR filters that cannot be parallelized.
/// Single-thread GPU kernels are 6-8x slower than CPU for these operations.
/// Hybrid approach leverages CPU for sequential parts and GPU for parallel parts.
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, keltner_channels_gpu};
/// use ndarray::Array1;
///
/// let device = GpuDevice::new()?;
/// let high = Array1::from_vec(vec![100.0, 102.0, /* ... */]);
/// let low = Array1::from_vec(vec![98.0, 99.0, /* ... */]);
/// let close = Array1::from_vec(vec![99.0, 101.0, /* ... */]);
///
/// let (upper, middle, lower) = keltner_channels_gpu(
///     &device, &high, &low, &close, 20, 10, 2.0, None
/// )?;
/// ```
///
/// # Errors
///
/// Returns error if:
/// - Arrays have different lengths
/// - Periods < 1
/// - Not enough data (n < max(ema_period, atr_period))
/// - GPU operations fail
pub fn keltner_channels_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    ema_period: usize,
    atr_period: usize,
    atr_multiplier: f64,
    stream: Option<&Arc<CudaStream>>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), GpuError> {
    let n = high.len();

    // Validate inputs
    if low.len() != n || close.len() != n {
        return Err(GpuError::InvalidParameter(
            "High, low, and close arrays must have same length".to_string(),
        ));
    }

    if ema_period < 1 {
        return Err(GpuError::InvalidParameter(
            "EMA period must be >= 1".to_string(),
        ));
    }

    if atr_period < 1 {
        return Err(GpuError::InvalidParameter(
            "ATR period must be >= 1".to_string(),
        ));
    }

    let min_required = ema_period.max(atr_period);
    if n < min_required {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need {} points, got {}",
            min_required, n
        )));
    }

    // Select stream: use provided stream or fallback to device.stream
    let exec_stream = stream.unwrap_or(&device.stream);

    // === Step 1: CPU - Calculate EMA (middle line, sequential, 6x faster than GPU) ===
    use crate::cpu::sequential::ema_cpu;
    let ema = ema_cpu(close, ema_period)?;

    // === Step 2: GPU Hybrid - Calculate ATR (uses hybrid implementation) ===
    // Pass the exec_stream to ATR for proper stream coordination
    let atr = atr_gpu(device, high, low, close, atr_period, Some(exec_stream))?;

    // === Step 3: GPU Parallel - Calculate bands ===
    let ptx = compile_ptx_optimized(KELTNER_KERNEL)
        .map_err(|e| GpuError::CompilationError(format!("Failed to compile kernel: {:?}", e)))?;

    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel function (only parallel bands kernel - EMA moved to CPU)
    let bands_kernel = module.load_function("keltner_bands_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load bands kernel function: {:?}", e))
    })?;

    // Copy EMA and ATR to GPU for band calculation
    let d_ema = device.copy_to_device(ema.as_slice().unwrap())?;
    let d_atr = device.copy_to_device(atr.as_slice().unwrap())?;

    let mut d_upper = device.alloc_buffer(n)?;
    let mut d_middle = device.alloc_buffer(n)?;
    let mut d_lower = device.alloc_buffer(n)?;

    let n_i32 = n as i32;

    let mut bands_builder = exec_stream.launch_builder(&bands_kernel);
    bands_builder.arg(&d_ema);
    bands_builder.arg(&d_atr);
    bands_builder.arg(&mut d_upper);
    bands_builder.arg(&mut d_middle);
    bands_builder.arg(&mut d_lower);
    bands_builder.arg(&atr_multiplier);
    bands_builder.arg(&n_i32);

    let bands_config = LaunchConfig::for_num_elems(n as u32);

    unsafe {
        bands_builder.launch(bands_config).map_err(|e| {
            GpuError::ExecutionError(format!("Bands kernel launch failed: {:?}", e))
        })?;
    }

    // Synchronize and copy results back
    exec_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Bands kernel synchronization failed: {:?}", e))
    })?;

    let upper_vec = device.copy_to_host(&d_upper)?;
    let middle_vec = device.copy_to_host(&d_middle)?;
    let lower_vec = device.copy_to_host(&d_lower)?;

    Ok((
        Array1::from_vec(upper_vec),
        Array1::from_vec(middle_vec),
        Array1::from_vec(lower_vec),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_keltner_channels_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Sample OHLC data with clear trend
        let high = arr1(&[
            48.70, 48.72, 48.90, 48.87, 48.82, 49.05, 49.20, 49.35, 49.92, 50.19, 50.12, 49.66,
            49.88, 50.19, 50.36, 50.57, 50.65, 50.43, 50.75, 51.38, 51.19, 52.51, 53.87, 53.75,
            53.71, 53.09,
        ]);
        let low = arr1(&[
            47.79, 48.14, 48.39, 48.37, 48.24, 48.64, 48.94, 48.86, 49.50, 49.87, 49.20, 48.90,
            49.43, 49.73, 49.26, 49.32, 50.08, 49.21, 49.77, 50.57, 50.09, 51.18, 52.69, 52.23,
            52.40, 52.12,
        ]);
        let close = arr1(&[
            48.16, 48.61, 48.75, 48.63, 48.74, 49.03, 49.07, 49.32, 49.91, 50.13, 49.53, 49.50,
            49.75, 50.03, 50.31, 50.52, 50.41, 49.34, 49.93, 51.37, 50.23, 52.46, 53.83, 53.48,
            53.00, 52.91,
        ]);

        let ema_period = 20;
        let atr_period = 10;
        let multiplier = 2.0;

        let (upper, middle, lower) = keltner_channels_gpu(
            &device, &high, &low, &close, ema_period, atr_period, multiplier, None,
        )
        .expect("Keltner Channels GPU calculation failed");

        // Verify lengths
        assert_eq!(upper.len(), close.len());
        assert_eq!(middle.len(), close.len());
        assert_eq!(lower.len(), close.len());

        // Verify early values are NaN (not enough data)
        let warmup = ema_period.max(atr_period);
        for i in 0..warmup - 1 {
            assert!(
                middle[i].is_nan(),
                "Middle[{}] should be NaN during warmup",
                i
            );
        }

        // Verify middle line values start appearing after warmup
        assert!(
            !middle[warmup - 1].is_nan(),
            "Middle should have value after warmup"
        );

        // Verify band relationships: Lower < Middle < Upper
        for i in warmup - 1..close.len() {
            if !middle[i].is_nan() && !upper[i].is_nan() && !lower[i].is_nan() {
                assert!(
                    lower[i] < middle[i],
                    "Lower[{}] = {} should be < Middle[{}] = {}",
                    i,
                    lower[i],
                    i,
                    middle[i]
                );
                assert!(
                    middle[i] < upper[i],
                    "Middle[{}] = {} should be < Upper[{}] = {}",
                    i,
                    middle[i],
                    i,
                    upper[i]
                );
            }
        }

        // Verify channel symmetry: Middle = (Upper + Lower) / 2
        for i in warmup - 1..close.len() {
            if !middle[i].is_nan() && !upper[i].is_nan() && !lower[i].is_nan() {
                let expected_middle = (upper[i] + lower[i]) / 2.0;
                assert!(
                    (middle[i] - expected_middle).abs() < 1e-10,
                    "Middle[{}] should equal (Upper + Lower) / 2",
                    i
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_keltner_channels_gpu_validation() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = arr1(&[10.0, 11.0, 12.0]);
        let low = arr1(&[8.0, 9.0, 10.0]);
        let close = arr1(&[9.0, 10.0, 11.0]);

        // Mismatched lengths
        let high_wrong = arr1(&[10.0, 11.0]);
        let result = keltner_channels_gpu(&device, &high_wrong, &low, &close, 20, 10, 2.0, None);
        assert!(result.is_err(), "Should fail with mismatched lengths");

        // Invalid EMA period
        let result = keltner_channels_gpu(&device, &high, &low, &close, 0, 10, 2.0, None);
        assert!(result.is_err(), "Should fail with zero EMA period");

        // Invalid ATR period
        let result = keltner_channels_gpu(&device, &high, &low, &close, 20, 0, 2.0, None);
        assert!(result.is_err(), "Should fail with zero ATR period");

        // Not enough data
        let result = keltner_channels_gpu(&device, &high, &low, &close, 20, 10, 2.0, None);
        assert!(result.is_err(), "Should fail with insufficient data");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_keltner_channels_gpu_custom_params() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with shorter periods
        let n = 50;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.2).collect());
        let low = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64) * 0.2).collect());
        let close = Array1::from_vec((0..n).map(|i| 99.0 + (i as f64) * 0.2).collect());

        let ema_period = 10;
        let atr_period = 5;
        let multiplier = 1.5;

        let (upper, middle, lower) = keltner_channels_gpu(
            &device, &high, &low, &close, ema_period, atr_period, multiplier, None,
        )
        .expect("Keltner Channels GPU calculation failed");

        // Verify data starts at max(ema_period, atr_period) - 1
        let warmup = ema_period.max(atr_period);
        assert!(
            !middle[warmup - 1].is_nan(),
            "Should have valid data after warmup"
        );

        // Verify all valid values maintain band relationships
        for i in warmup - 1..n {
            if !middle[i].is_nan() {
                assert!(lower[i] < middle[i] && middle[i] < upper[i]);
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_keltner_channels_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Test with large dataset to verify performance
        let n = 100_000;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());
        let low = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64) * 0.01).collect());
        let close = Array1::from_vec((0..n).map(|i| 99.0 + (i as f64) * 0.01).collect());

        let start = std::time::Instant::now();
        let (upper, middle, lower) =
            keltner_channels_gpu(&device, &high, &low, &close, 20, 10, 2.0, None)
                .expect("Keltner Channels GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU Keltner Channels (n={}): {:.2}ms",
            n,
            elapsed.as_secs_f64() * 1000.0
        );

        assert_eq!(upper.len(), n);
        assert_eq!(middle.len(), n);
        assert_eq!(lower.len(), n);

        // Verify most values are valid
        let valid_count = middle.iter().filter(|&&x| !x.is_nan()).count();
        assert!(
            valid_count > n - 25,
            "Most values should be valid in large dataset"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_keltner_channels_gpu_multiplier_effect() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 30;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.5).collect());
        let low = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64) * 0.5).collect());
        let close = Array1::from_vec((0..n).map(|i| 99.0 + (i as f64) * 0.5).collect());

        // Calculate with multiplier = 1.0
        let (upper1, middle1, lower1) =
            keltner_channels_gpu(&device, &high, &low, &close, 10, 5, 1.0, None)
                .expect("Keltner Channels GPU calculation failed");

        // Calculate with multiplier = 2.0
        let (upper2, middle2, lower2) =
            keltner_channels_gpu(&device, &high, &low, &close, 10, 5, 2.0, None)
                .expect("Keltner Channels GPU calculation failed");

        // Middle line should be identical (EMA doesn't depend on multiplier)
        for i in 0..n {
            if !middle1[i].is_nan() && !middle2[i].is_nan() {
                assert!(
                    (middle1[i] - middle2[i]).abs() < 1e-10,
                    "Middle lines should be identical"
                );
            }
        }

        // Channel width with multiplier=2.0 should be exactly 2x wider than multiplier=1.0
        for i in 10..n {
            if !upper1[i].is_nan() && !upper2[i].is_nan() {
                let width1 = upper1[i] - middle1[i];
                let width2 = upper2[i] - middle2[i];
                assert!(
                    (width2 - 2.0 * width1).abs() < 1e-10,
                    "Width2 should be 2x Width1"
                );
            }
        }
    }
}
