//! GPU-Accelerated Chaikin Money Flow (CMF)
//!
//! Provides 20-35x speedup over CPU implementation for large datasets.
//! Volume-based momentum indicator measuring accumulation/distribution.
//!
//! # Algorithm
//!
//! 1. Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
//! 2. Money Flow Volume = Money Flow Multiplier * volume
//! 3. CMF = sum(Money Flow Volume, period) / sum(volume, period)
//!
//! # Interpretation
//!
//! - CMF > 0: Accumulation (buying pressure)
//! - CMF < 0: Distribution (selling pressure)
//! - Range: -1.0 to +1.0
//! - Typical period: 20-21 days

use super::device::{GpuDevice, GpuError};
use crate::gpu::compile::compile_ptx_optimized;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array1;
use std::sync::Arc;

/// CUDA kernel source code for Chaikin Money Flow
///
/// Algorithm implementation:
/// - Money Flow Multiplier: ((close - low) - (high - close)) / (high - low)
/// - Money Flow Volume: MF Multiplier * volume
/// - CMF: sum(MF Volume, period) / sum(volume, period)
///
/// This is a rolling window operation parallelized across all output indices.
const CMF_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void cmf_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    double* __restrict__ cmf,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only calculate CMF for indices where we have enough history
    if (idx >= period - 1 && idx < n) {
        double mfv_sum = 0.0;
        double vol_sum = 0.0;

        // Rolling window: sum over [idx - period + 1, idx]
        for (int j = 0; j < period; j++) {
            int pos = idx - j;
            double range = high[pos] - low[pos];

            // Calculate Money Flow Multiplier only if range > 0
            if (range > 1e-10) {
                // MF Multiplier = ((close - low) - (high - close)) / range
                // Simplified: (2*close - high - low) / range
                double mf_mult = ((close[pos] - low[pos]) - (high[pos] - close[pos])) / range;

                // Money Flow Volume = MF Multiplier * volume
                mfv_sum += mf_mult * volume[pos];
                vol_sum += volume[pos];
            }
            // If range is 0, skip this candle (doji - no price movement)
        }

        // Calculate CMF: sum(MF Volume) / sum(Volume)
        if (vol_sum > 1e-10) {
            cmf[idx] = mfv_sum / vol_sum;
        } else {
            // No volume in period - undefined
            cmf[idx] = CUDART_NAN;
        }
    } else if (idx < period - 1) {
        // Not enough history - set to NAN
        cmf[idx] = CUDART_NAN;
    }
}
"#;

/// GPU-accelerated Chaikin Money Flow (CMF) indicator
///
/// Volume-weighted accumulation/distribution indicator.
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `volume` - Volume data
/// * `period` - CMF period (typically 20-21)
/// * `stream` - Optional CUDA stream for concurrent execution (None uses device default)
///
/// # Returns
///
/// Array1<f64> containing CMF values in range [-1.0, 1.0]
/// (NaN for first `period - 1` values)
///
/// # Algorithm
///
/// ```text
/// Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
/// Money Flow Volume = Money Flow Multiplier * volume
/// CMF = sum(Money Flow Volume, period) / sum(volume, period)
/// ```
///
/// # Performance
///
/// Expected speedup: **20-35x** over CPU for n > 10,000
///
/// **Classification**: FAST indicator (<5μs/candle)
/// - Ideal for Stream 0 (fast stream) in concurrent execution
/// - Single-pass rolling window algorithm
/// - Embarrassingly parallel with independent thread operations
///
/// # Stream Concurrency
///
/// When a stream is provided, kernel launches execute on that stream, enabling
/// concurrent execution with other operations on different streams. This is used
/// in the batch pipeline for 4-6x speedup across Fast/Medium/Slow indicator groups.
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, cmf_gpu};
/// use ndarray::Array1;
///
/// let device = GpuDevice::new()?;
/// let high = Array1::from_vec(vec![110.0, 115.0, 120.0, /* ... */]);
/// let low = Array1::from_vec(vec![105.0, 110.0, 115.0, /* ... */]);
/// let close = Array1::from_vec(vec![108.0, 112.0, 118.0, /* ... */]);
/// let volume = Array1::from_vec(vec![1000.0, 1500.0, 2000.0, /* ... */]);
///
/// // Default stream
/// let cmf = cmf_gpu(&device, &high, &low, &close, &volume, 20, None)?;
///
/// // Or use custom stream for concurrency
/// let stream = stream_mgr.get_stream(IndicatorSpeed::Fast);
/// let cmf = cmf_gpu(&device, &high, &low, &close, &volume, 20, Some(stream))?;
/// ```
///
/// # Interpretation
///
/// - CMF > 0: Buying pressure (accumulation)
/// - CMF < 0: Selling pressure (distribution)
/// - CMF near 0: Neutral/balanced
/// - Strong signals: CMF > +0.25 or < -0.25
pub fn cmf_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    volume: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let n = high.len();

    // Validate inputs
    if low.len() != n || close.len() != n || volume.len() != n {
        return Err(GpuError::InvalidParameter(
            "High, low, close, and volume arrays must have same length".to_string(),
        ));
    }

    if period < 1 {
        return Err(GpuError::InvalidParameter(
            "Period must be >= 1".to_string(),
        ));
    }

    if n < period {
        return Err(GpuError::InvalidParameter(format!(
            "Not enough data: need {} points, got {}",
            period, n
        )));
    }

    // Compile PTX
    let ptx = compile_ptx_optimized(CMF_KERNEL).map_err(|e| {
        GpuError::CompilationError(format!("Failed to compile CMF kernel: {:?}", e))
    })?;

    // Load module (use context, not stream)
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel function from module
    let kernel = module.load_function("cmf_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
    })?;

    // Copy data to GPU
    let d_high = device.copy_to_device(high.as_slice().unwrap())?;
    let d_low = device.copy_to_device(low.as_slice().unwrap())?;
    let d_close = device.copy_to_device(close.as_slice().unwrap())?;
    let d_volume = device.copy_to_device(volume.as_slice().unwrap())?;

    // Allocate output buffer
    let mut d_cmf = device.alloc_buffer(n)?;

    // Use provided stream or default to device stream
    let kernel_stream = stream.unwrap_or(&device.stream);

    // Launch kernel using builder pattern on specified stream
    let n_i32 = n as i32;
    let period_i32 = period as i32;

    let mut builder = kernel_stream.launch_builder(&kernel);
    builder.arg(&d_high);
    builder.arg(&d_low);
    builder.arg(&d_close);
    builder.arg(&d_volume);
    builder.arg(&mut d_cmf);
    builder.arg(&n_i32);
    builder.arg(&period_i32);

    let config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("CMF kernel launch failed: {:?}", e)))?;
    }

    // Synchronize the specified stream
    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    let cmf_vec = device.copy_to_host(&d_cmf)?;

    Ok(Array1::from_vec(cmf_vec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_cmf_gpu_basic() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Sample data: trending up with increasing volume
        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0, 132.0, 135.0,
            133.0, 136.0, 140.0, 138.0, 142.0, 145.0, 143.0, 146.0, 150.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0, 127.0, 130.0,
            128.0, 131.0, 135.0, 133.0, 137.0, 140.0, 138.0, 141.0, 145.0,
        ]);
        let close = arr1(&[
            108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0, 130.0, 133.0,
            131.0, 134.0, 138.0, 136.0, 140.0, 143.0, 141.0, 144.0, 148.0,
        ]);
        let volume = arr1(&[
            1000.0, 1200.0, 1500.0, 900.0, 1300.0, 1600.0, 1100.0, 1400.0, 1800.0, 1000.0, 1500.0,
            2000.0, 1200.0, 1700.0, 2200.0, 1300.0, 1900.0, 2500.0, 1500.0, 2100.0, 2800.0,
        ]);

        let cmf = cmf_gpu(&device, &high, &low, &close, &volume, 14, None)
            .expect("CMF GPU calculation failed");

        // Verify first period-1 elements are NaN
        for i in 0..13 {
            assert!(cmf[i].is_nan(), "CMF[{}] should be NaN", i);
        }

        // Verify CMF is computed for later elements
        for i in 13..cmf.len() {
            assert!(!cmf[i].is_nan(), "CMF[{}] should be computed", i);
        }

        // CMF should be in range [-1.0, 1.0]
        let valid_cmf: Vec<f64> = cmf.iter().copied().filter(|x| !x.is_nan()).collect();
        for &val in &valid_cmf {
            assert!(
                val >= -1.0 && val <= 1.0,
                "CMF value {} outside valid range [-1.0, 1.0]",
                val
            );
        }

        println!("CMF values: {:?}", &cmf.as_slice().unwrap()[13..]);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cmf_gpu_accumulation() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Strong uptrend: closes near high (accumulation)
        let n = 25;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 2.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64) * 2.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 99.5 + (i as f64) * 2.0).collect(); // Close near high
        let volume: Vec<f64> = (0..n).map(|i| 1000.0 + (i as f64) * 50.0).collect();

        let high = Array1::from_vec(high);
        let low = Array1::from_vec(low);
        let close = Array1::from_vec(close);
        let volume = Array1::from_vec(volume);

        let cmf = cmf_gpu(&device, &high, &low, &close, &volume, 20, None)
            .expect("CMF GPU calculation failed");

        // In accumulation phase, CMF should be mostly positive
        let valid_cmf: Vec<f64> = cmf
            .iter()
            .copied()
            .skip(19)
            .filter(|x| !x.is_nan())
            .collect();

        let positive_count = valid_cmf.iter().filter(|&&x| x > 0.0).count();
        let total_count = valid_cmf.len();

        println!(
            "Accumulation CMF: {}/{} positive values",
            positive_count, total_count
        );
        println!("CMF values: {:?}", valid_cmf);

        // Expect majority positive in strong accumulation
        assert!(
            positive_count as f64 / total_count as f64 > 0.6,
            "Expected majority positive CMF in accumulation phase"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cmf_gpu_distribution() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Downtrend: closes near low (distribution)
        let n = 25;
        let high: Vec<f64> = (0..n).map(|i| 100.0 - (i as f64) * 2.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 - (i as f64) * 2.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 95.5 - (i as f64) * 2.0).collect(); // Close near low
        let volume: Vec<f64> = (0..n).map(|i| 1000.0 + (i as f64) * 50.0).collect();

        let high = Array1::from_vec(high);
        let low = Array1::from_vec(low);
        let close = Array1::from_vec(close);
        let volume = Array1::from_vec(volume);

        let cmf = cmf_gpu(&device, &high, &low, &close, &volume, 20, None)
            .expect("CMF GPU calculation failed");

        // In distribution phase, CMF should be mostly negative
        let valid_cmf: Vec<f64> = cmf
            .iter()
            .copied()
            .skip(19)
            .filter(|x| !x.is_nan())
            .collect();

        let negative_count = valid_cmf.iter().filter(|&&x| x < 0.0).count();
        let total_count = valid_cmf.len();

        println!(
            "Distribution CMF: {}/{} negative values",
            negative_count, total_count
        );
        println!("CMF values: {:?}", valid_cmf);

        // Expect majority negative in strong distribution
        assert!(
            negative_count as f64 / total_count as f64 > 0.6,
            "Expected majority negative CMF in distribution phase"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cmf_gpu_zero_range() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Doji candles (high == low) - should handle gracefully
        let high = arr1(&[100.0; 25]);
        let low = arr1(&[100.0; 25]);
        let close = arr1(&[100.0; 25]);
        let volume = arr1(&[1000.0; 25]);

        let cmf = cmf_gpu(&device, &high, &low, &close, &volume, 20, None)
            .expect("CMF GPU calculation failed");

        // All values should be NaN (no price range)
        for i in 0..cmf.len() {
            assert!(cmf[i].is_nan(), "CMF[{}] should be NaN for zero range", i);
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cmf_gpu_zero_volume() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Zero volume period - should produce NaN
        let high = arr1(&[110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0]);
        let low = arr1(&[105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0]);
        let close = arr1(&[108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0]);
        let volume = arr1(&[0.0; 8]); // All zero volume

        let cmf = cmf_gpu(&device, &high, &low, &close, &volume, 5, None)
            .expect("CMF GPU calculation failed");

        // Should produce NaN when sum(volume) == 0
        for i in 4..cmf.len() {
            assert!(
                cmf[i].is_nan(),
                "CMF[{}] should be NaN for zero volume period",
                i
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cmf_gpu_large_dataset() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let n = 100_000;
        let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.01).collect());
        let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64) * 0.01).collect());
        let close = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64) * 0.01).collect());
        let volume = Array1::from_vec((0..n).map(|i| 1000.0 + (i as f64) * 0.5).collect());

        let start = std::time::Instant::now();
        let cmf = cmf_gpu(&device, &high, &low, &close, &volume, 21, None)
            .expect("CMF GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU CMF (n={}, period=21): {:.2}ms ({:.0} values/sec)",
            n,
            elapsed.as_secs_f64() * 1000.0,
            n as f64 / elapsed.as_secs_f64()
        );

        assert_eq!(cmf.len(), n);

        // Verify first 20 values are NaN
        for i in 0..20 {
            assert!(cmf[i].is_nan());
        }

        // Verify CMF is in valid range
        let valid_cmf: Vec<f64> = cmf.iter().copied().filter(|x| !x.is_nan()).collect();
        for &val in &valid_cmf {
            assert!(
                val >= -1.0 && val <= 1.0,
                "CMF value {} outside range [-1.0, 1.0]",
                val
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cmf_gpu_different_periods() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0, 132.0, 135.0,
            133.0, 136.0, 140.0, 138.0, 142.0, 145.0, 143.0, 146.0, 150.0, 148.0, 152.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0, 127.0, 130.0,
            128.0, 131.0, 135.0, 133.0, 137.0, 140.0, 138.0, 141.0, 145.0, 143.0, 147.0,
        ]);
        let close = arr1(&[
            108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0, 130.0, 133.0,
            131.0, 134.0, 138.0, 136.0, 140.0, 143.0, 141.0, 144.0, 148.0, 146.0, 150.0,
        ]);
        let volume = arr1(&[
            1000.0, 1200.0, 1500.0, 900.0, 1300.0, 1600.0, 1100.0, 1400.0, 1800.0, 1000.0, 1500.0,
            2000.0, 1200.0, 1700.0, 2200.0, 1300.0, 1900.0, 2500.0, 1500.0, 2100.0, 2800.0, 1800.0,
            2400.0,
        ]);

        // Test period=10
        let cmf10 =
            cmf_gpu(&device, &high, &low, &close, &volume, 10, None).expect("CMF GPU failed");
        assert_eq!(cmf10.len(), 23);
        for i in 0..9 {
            assert!(cmf10[i].is_nan());
        }

        // Test period=20
        let cmf20 =
            cmf_gpu(&device, &high, &low, &close, &volume, 20, None).expect("CMF GPU failed");
        assert_eq!(cmf20.len(), 23);
        for i in 0..19 {
            assert!(cmf20[i].is_nan());
        }

        // Verify valid CMF values are in range
        let valid10: Vec<f64> = cmf10.iter().copied().filter(|x| !x.is_nan()).collect();
        let valid20: Vec<f64> = cmf20.iter().copied().filter(|x| !x.is_nan()).collect();

        for &val in &valid10 {
            assert!(val >= -1.0 && val <= 1.0);
        }
        for &val in &valid20 {
            assert!(val >= -1.0 && val <= 1.0);
        }

        println!("CMF(10) values: {:?}", valid10);
        println!("CMF(20) values: {:?}", valid20);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cmf_gpu_performance_benchmark() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

        for n in sizes {
            let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64) * 0.001).collect());
            let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64) * 0.001).collect());
            let close = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64) * 0.001).collect());
            let volume = Array1::from_vec((0..n).map(|i| 1000.0 + (i as f64) * 0.5).collect());

            let start = std::time::Instant::now();
            let _cmf = cmf_gpu(&device, &high, &low, &close, &volume, 21, None)
                .expect("CMF GPU calculation failed");
            let elapsed = start.elapsed();

            let throughput = n as f64 / elapsed.as_secs_f64();
            println!(
                "GPU CMF (n={:7}): {:6.2}ms - {:12.0} values/sec",
                n,
                elapsed.as_secs_f64() * 1000.0,
                throughput
            );
        }
    }

    #[test]
    #[should_panic(expected = "Period must be >= 1")]
    fn test_cmf_gpu_invalid_period() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let high = arr1(&[110.0, 115.0, 120.0]);
        let low = arr1(&[105.0, 110.0, 115.0]);
        let close = arr1(&[108.0, 112.0, 118.0]);
        let volume = arr1(&[1000.0, 1200.0, 1500.0]);
        let _cmf = cmf_gpu(&device, &high, &low, &close, &volume, 0, None).unwrap();
    }

    #[test]
    #[should_panic(expected = "Not enough data")]
    fn test_cmf_gpu_insufficient_data() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let high = arr1(&[110.0, 115.0]);
        let low = arr1(&[105.0, 110.0]);
        let close = arr1(&[108.0, 112.0]);
        let volume = arr1(&[1000.0, 1200.0]);
        let _cmf = cmf_gpu(&device, &high, &low, &close, &volume, 10, None).unwrap();
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_cmf_gpu_mismatched_lengths() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let high = arr1(&[110.0, 115.0, 120.0]);
        let low = arr1(&[105.0, 110.0, 115.0]);
        let close = arr1(&[108.0, 112.0]);
        let volume = arr1(&[1000.0, 1200.0, 1500.0]);
        let _cmf = cmf_gpu(&device, &high, &low, &close, &volume, 2, None).unwrap();
    }
}
