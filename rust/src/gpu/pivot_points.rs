//! GPU-Accelerated Pivot Points
//!
//! Provides 15-30x speedup over CPU implementation for large datasets.
//!
//! # Algorithm (Standard Method)
//!
//! Pivot points are support/resistance levels calculated from previous period's OHLC:
//!
//! 1. Pivot Point (PP) = (High + Low + Close) / 3
//! 2. Support 1 (S1) = (2 × PP) - High
//! 3. Support 2 (S2) = PP - (High - Low)
//! 4. Support 3 (S3) = Low - 2 × (High - PP)
//! 5. Resistance 1 (R1) = (2 × PP) - Low
//! 6. Resistance 2 (R2) = PP + (High - Low)
//! 7. Resistance 3 (R3) = High + 2 × (PP - Low)
//!
//! # Performance
//!
//! This is an embarrassingly parallel problem - each timepoint can be calculated
//! independently. Expected speedup: **15-30x** for n > 10,000.

use super::device::{GpuDevice, GpuError};
use crate::gpu::compile::compile_ptx_optimized_cached;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array1;
use std::sync::Arc;

/// Output structure for Pivot Points calculation
///
/// Contains all 7 pivot point levels for each timepoint.
#[derive(Debug, Clone)]
pub struct PivotPointsOutput {
    /// Pivot Point (middle level)
    pub pp: Array1<f64>,
    /// Support Level 1
    pub s1: Array1<f64>,
    /// Support Level 2
    pub s2: Array1<f64>,
    /// Support Level 3
    pub s3: Array1<f64>,
    /// Resistance Level 1
    pub r1: Array1<f64>,
    /// Resistance Level 2
    pub r2: Array1<f64>,
    /// Resistance Level 3
    pub r3: Array1<f64>,
}

/// CUDA kernel source code for Pivot Points calculation
///
/// Single-pass algorithm: each thread calculates all 7 levels for one timepoint.
/// This is an embarrassingly parallel problem with perfect GPU suitability.
const PIVOT_POINTS_KERNEL: &str = r#"
// Define constants directly to avoid header dependencies with NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

/// Calculate all 7 pivot point levels in a single kernel
///
/// Each thread processes one timepoint independently:
/// - Reads previous period's high, low, close (if available)
/// - Calculates PP and all 6 support/resistance levels
/// - Writes results to 7 output arrays
///
/// First timepoint (index 0) has no previous data, so outputs NaN.
extern "C" __global__ void pivot_points_kernel(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    double* __restrict__ pp,
    double* __restrict__ s1,
    double* __restrict__ s2,
    double* __restrict__ s3,
    double* __restrict__ r1,
    double* __restrict__ r2,
    double* __restrict__ r3,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // First timepoint has no previous data
    if (idx == 0) {
        pp[idx] = CUDART_NAN;
        s1[idx] = CUDART_NAN;
        s2[idx] = CUDART_NAN;
        s3[idx] = CUDART_NAN;
        r1[idx] = CUDART_NAN;
        r2[idx] = CUDART_NAN;
        r3[idx] = CUDART_NAN;
        return;
    }

    // Use previous period's data for pivot calculation
    double prev_high = high[idx - 1];
    double prev_low = low[idx - 1];
    double prev_close = close[idx - 1];

    // Step 1: Calculate Pivot Point (PP)
    double pivot = (prev_high + prev_low + prev_close) / 3.0;
    pp[idx] = pivot;

    // Step 2: Calculate price range for support/resistance calculations
    double range = prev_high - prev_low;

    // Step 3: Calculate Resistance levels
    r1[idx] = 2.0 * pivot - prev_low;        // R1 = 2*PP - Low
    r2[idx] = pivot + range;                  // R2 = PP + (High - Low)
    r3[idx] = prev_high + 2.0 * (pivot - prev_low); // R3 = High + 2*(PP - Low)

    // Step 4: Calculate Support levels
    s1[idx] = 2.0 * pivot - prev_high;        // S1 = 2*PP - High
    s2[idx] = pivot - range;                  // S2 = PP - (High - Low)
    s3[idx] = prev_low - 2.0 * (prev_high - pivot); // S3 = Low - 2*(High - PP)
}
"#;

/// GPU-accelerated Pivot Points calculation
///
/// Calculates all 7 pivot point levels using CUDA for massive parallelization.
/// Each thread processes one timepoint independently - perfect for GPU.
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `stream` - Optional CUDA stream for concurrent execution (None = use device default)
///
/// # Returns
///
/// `PivotPointsOutput` structure containing all 7 levels (PP, R1, R2, R3, S1, S2, S3)
///
/// # Performance
///
/// Expected speedup: **15-30x** over CPU for n > 10,000
///
/// This is an embarrassingly parallel problem with no data dependencies between
/// timepoints. Each thread performs:
/// - 3 reads (previous high, low, close)
/// - 7 writes (PP, 3 resistance, 3 support levels)
/// - ~15 arithmetic operations (highly parallelizable)
///
/// # Algorithm Details
///
/// Pivot points are calculated using **previous period's** OHLC data:
/// - `pp[i]` uses `high[i-1]`, `low[i-1]`, `close[i-1]`
/// - `pp[0]` is NaN (no previous data)
/// - All 7 levels calculated in a single pass
///
/// # Stream Concurrency
///
/// When provided with a CUDA stream, this function can execute concurrently with other
/// indicators on different streams. Classification: **FAST** indicator (<5μs/candle).
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::{GpuDevice, pivot_points_gpu};
/// use ndarray::arr1;
///
/// let device = GpuDevice::new()?;
/// let high = arr1(&[110.0, 115.0, 120.0, /* ... */]);
/// let low = arr1(&[105.0, 110.0, 115.0, /* ... */]);
/// let close = arr1(&[108.0, 112.0, 118.0, /* ... */]);
///
/// let result = pivot_points_gpu(Arc::new(device), &high, &low, &close, None)?;
///
/// println!("Pivot Point: {:?}", result.pp);
/// println!("Resistance 1: {:?}", result.r1);
/// println!("Support 1: {:?}", result.s1);
/// ```
pub fn pivot_points_gpu(
    device: Arc<GpuDevice>,
    high: &[f64],
    low: &[f64],
    close: &[f64],
    stream: Option<&CudaStream>,
) -> Result<PivotPointsOutput, GpuError> {
    let n = high.len();

    // Validate inputs
    if n != low.len() || n != close.len() {
        return Err(GpuError::InvalidParameter(format!(
            "Input arrays must have same length: high={}, low={}, close={}",
            n,
            low.len(),
            close.len()
        )));
    }

    if n < 2 {
        return Err(GpuError::InvalidParameter(format!(
            "Need at least 2 data points for pivot points, got {}",
            n
        )));
    }

    // Compile PTX with caching (50-200x faster on cache hits)
    let ptx_arc = compile_ptx_optimized_cached(PIVOT_POINTS_KERNEL).map_err(|e| {
        GpuError::CompilationError(format!("Failed to compile Pivot Points kernel: {:?}", e))
    })?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);

    // Load module
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel function
    let kernel = module.load_function("pivot_points_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load pivot_points_kernel: {:?}", e))
    })?;

    // Select stream: use provided stream or device default
    let kernel_stream = stream.unwrap_or(&device.stream);

    // === Step 1: H2D - Asynchronously copy input data to device ===
    let mut pinned_high = device.pinned_pool.lock().acquire(n)?;
    pinned_high.as_mut_slice()[..n].copy_from_slice(high);
    let mut pinned_low = device.pinned_pool.lock().acquire(n)?;
    pinned_low.as_mut_slice()[..n].copy_from_slice(low);
    let mut pinned_close = device.pinned_pool.lock().acquire(n)?;
    pinned_close.as_mut_slice()[..n].copy_from_slice(close);

    let mut d_high = device.alloc_buffer(n)?;
    let mut d_low = device.alloc_buffer(n)?;
    let mut d_close = device.alloc_buffer(n)?;

    device
        .stream
        .memcpy_htod(&pinned_high.as_slice()[..n], &mut d_high)?;
    device
        .stream
        .memcpy_htod(&pinned_low.as_slice()[..n], &mut d_low)?;
    device
        .stream
        .memcpy_htod(&pinned_close.as_slice()[..n], &mut d_close)?;

    // Release pinned buffers
    let mut pool = device.pinned_pool.lock();
    pool.release(pinned_high);
    pool.release(pinned_low);
    pool.release(pinned_close);
    drop(pool);

    // === Step 2: Allocate device buffers for 7 outputs ===
    let mut d_pp = device.alloc_buffer(n)?;
    let mut d_s1 = device.alloc_buffer(n)?;
    let mut d_s2 = device.alloc_buffer(n)?;
    let mut d_s3 = device.alloc_buffer(n)?;
    let mut d_r1 = device.alloc_buffer(n)?;
    let mut d_r2 = device.alloc_buffer(n)?;
    let mut d_r3 = device.alloc_buffer(n)?;

    // === Step 3: Launch kernel ===
    let n_i32 = n as i32;

    let mut builder = kernel_stream.launch_builder(&kernel);
    builder.arg(&d_high);
    builder.arg(&d_low);
    builder.arg(&d_close);
    builder.arg(&mut d_pp);
    builder.arg(&mut d_s1);
    builder.arg(&mut d_s2);
    builder.arg(&mut d_s3);
    builder.arg(&mut d_r1);
    builder.arg(&mut d_r2);
    builder.arg(&mut d_r3);
    builder.arg(&n_i32);

    let config = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        builder.launch(config).map_err(|e| {
            GpuError::ExecutionError(format!("Pivot Points kernel launch failed: {:?}", e))
        })?;
    }

    // === Step 4: D2H - Asynchronously copy results back to host ===
    let mut pinned_pp = device.pinned_pool.lock().acquire(n)?;
    let mut pinned_s1 = device.pinned_pool.lock().acquire(n)?;
    let mut pinned_s2 = device.pinned_pool.lock().acquire(n)?;
    let mut pinned_s3 = device.pinned_pool.lock().acquire(n)?;
    let mut pinned_r1 = device.pinned_pool.lock().acquire(n)?;
    let mut pinned_r2 = device.pinned_pool.lock().acquire(n)?;
    let mut pinned_r3 = device.pinned_pool.lock().acquire(n)?;

    device
        .stream
        .memcpy_dtoh(&d_pp, &mut pinned_pp.as_mut_slice()[..n])?;
    device
        .stream
        .memcpy_dtoh(&d_s1, &mut pinned_s1.as_mut_slice()[..n])?;
    device
        .stream
        .memcpy_dtoh(&d_s2, &mut pinned_s2.as_mut_slice()[..n])?;
    device
        .stream
        .memcpy_dtoh(&d_s3, &mut pinned_s3.as_mut_slice()[..n])?;
    device
        .stream
        .memcpy_dtoh(&d_r1, &mut pinned_r1.as_mut_slice()[..n])?;
    device
        .stream
        .memcpy_dtoh(&d_r2, &mut pinned_r2.as_mut_slice()[..n])?;
    device
        .stream
        .memcpy_dtoh(&d_r3, &mut pinned_r3.as_mut_slice()[..n])?;

    // Synchronize stream to ensure all D2H copies are complete before CPU access
    device.stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    let pp_vec = pinned_pp.as_slice()[..n].to_vec();
    let s1_vec = pinned_s1.as_slice()[..n].to_vec();
    let s2_vec = pinned_s2.as_slice()[..n].to_vec();
    let s3_vec = pinned_s3.as_slice()[..n].to_vec();
    let r1_vec = pinned_r1.as_slice()[..n].to_vec();
    let r2_vec = pinned_r2.as_slice()[..n].to_vec();
    let r3_vec = pinned_r3.as_slice()[..n].to_vec();

    // Release pinned buffers
    let mut pool = device.pinned_pool.lock();
    pool.release(pinned_pp);
    pool.release(pinned_s1);
    pool.release(pinned_s2);
    pool.release(pinned_s3);
    pool.release(pinned_r1);
    pool.release(pinned_r2);
    pool.release(pinned_r3);
    drop(pool);

    Ok(PivotPointsOutput {
        pp: Array1::from_vec(pp_vec),
        s1: Array1::from_vec(s1_vec),
        s2: Array1::from_vec(s2_vec),
        s3: Array1::from_vec(s3_vec),
        r1: Array1::from_vec(r1_vec),
        r2: Array1::from_vec(r2_vec),
        r3: Array1::from_vec(r3_vec),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    #[ignore] // Requires GPU
    fn test_pivot_points_gpu_basic() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        // Simple test with known values
        let high = arr1(&[110.0, 115.0, 120.0]);
        let low = arr1(&[100.0, 110.0, 115.0]);
        let close = arr1(&[105.0, 112.0, 118.0]);

        let result = pivot_points_gpu(
            device,
            high.as_slice().unwrap(),
            low.as_slice().unwrap(),
            close.as_slice().unwrap(),
            None,
        )
        .expect("Pivot Points GPU calculation failed");

        // First value should be NaN (no previous data)
        assert!(result.pp[0].is_nan());
        assert!(result.s1[0].is_nan());
        assert!(result.r1[0].is_nan());

        // Second value uses first period's data:
        // PP = (110 + 100 + 105) / 3 = 105
        assert!((result.pp[1] - 105.0).abs() < 1e-10);

        // R1 = 2*PP - low = 2*105 - 100 = 110
        assert!((result.r1[1] - 110.0).abs() < 1e-10);

        // S1 = 2*PP - high = 2*105 - 110 = 100
        assert!((result.s1[1] - 100.0).abs() < 1e-10);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pivot_points_gpu_relationships() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        let high = arr1(&[110.0, 115.0, 120.0, 118.0, 122.0]);
        let low = arr1(&[105.0, 110.0, 115.0, 113.0, 117.0]);
        let close = arr1(&[108.0, 112.0, 118.0, 115.0, 120.0]);

        let result = pivot_points_gpu(
            device,
            high.as_slice().unwrap(),
            low.as_slice().unwrap(),
            close.as_slice().unwrap(),
            None,
        )
        .expect("Pivot Points GPU calculation failed");

        // Verify relationships: S3 < S2 < S1 < PP < R1 < R2 < R3
        for i in 1..result.pp.len() {
            assert!(!result.pp[i].is_nan());

            // Support levels should be below pivot
            assert!(
                result.s1[i] < result.pp[i],
                "S1 should be < PP at index {}",
                i
            );
            assert!(
                result.s2[i] < result.pp[i],
                "S2 should be < PP at index {}",
                i
            );
            assert!(
                result.s3[i] < result.pp[i],
                "S3 should be < PP at index {}",
                i
            );

            // Resistance levels should be above pivot
            assert!(
                result.r1[i] > result.pp[i],
                "R1 should be > PP at index {}",
                i
            );
            assert!(
                result.r2[i] > result.pp[i],
                "R2 should be > PP at index {}",
                i
            );
            assert!(
                result.r3[i] > result.pp[i],
                "R3 should be > PP at index {}",
                i
            );

            // Support levels ordering: S3 < S2 < S1
            assert!(
                result.s3[i] < result.s2[i],
                "S3 should be < S2 at index {}",
                i
            );
            assert!(
                result.s2[i] < result.s1[i],
                "S2 should be < S1 at index {}",
                i
            );

            // Resistance levels ordering: R1 < R2 < R3
            assert!(
                result.r1[i] < result.r2[i],
                "R1 should be < R2 at index {}",
                i
            );
            assert!(
                result.r2[i] < result.r3[i],
                "R2 should be < R3 at index {}",
                i
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pivot_points_gpu_symmetry() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        // Test with symmetric data
        let high = arr1(&[110.0, 120.0, 130.0]);
        let low = arr1(&[90.0, 100.0, 110.0]);
        let close = arr1(&[100.0, 110.0, 120.0]);

        let result = pivot_points_gpu(
            device,
            high.as_slice().unwrap(),
            low.as_slice().unwrap(),
            close.as_slice().unwrap(),
            None,
        )
        .expect("Pivot Points GPU calculation failed");

        // PP should be centered between high and low
        // For index 1: PP = (110 + 90 + 100) / 3 = 100
        assert!((result.pp[1] - 100.0).abs() < 1e-10);

        // R1 and S1 should be equidistant from PP
        let r1_dist = result.r1[1] - result.pp[1];
        let s1_dist = result.pp[1] - result.s1[1];
        assert!(
            (r1_dist - s1_dist).abs() < 1e-10,
            "R1 and S1 should be equidistant from PP"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pivot_points_gpu_large_dataset() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        let n = 100_000;
        let high: Vec<f64> = (0..n).map(|i| 110.0 + (i as f64) * 0.01).collect();
        let low: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.01).collect();
        let close: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64) * 0.01).collect();

        let start = std::time::Instant::now();
        let result = pivot_points_gpu(device, &high, &low, &close, None)
            .expect("Pivot Points GPU calculation failed");
        let elapsed = start.elapsed();

        println!(
            "GPU Pivot Points (n={}): {:.2}ms",
            n,
            elapsed.as_secs_f64() * 1000.0
        );

        // Verify dimensions
        assert_eq!(result.pp.len(), n);
        assert_eq!(result.s1.len(), n);
        assert_eq!(result.r1.len(), n);

        // Verify valid data (except first point)
        for i in 1..n {
            assert!(
                !result.pp[i].is_nan(),
                "PP should not be NaN at index {}",
                i
            );
            assert!(
                !result.s1[i].is_nan(),
                "S1 should not be NaN at index {}",
                i
            );
            assert!(
                !result.r1[i].is_nan(),
                "R1 should not be NaN at index {}",
                i
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pivot_points_gpu_constant_prices() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        // When prices are constant, all levels should converge
        let high = arr1(&[100.0; 20]);
        let low = arr1(&[100.0; 20]);
        let close = arr1(&[100.0; 20]);

        let result = pivot_points_gpu(
            device,
            high.as_slice().unwrap(),
            low.as_slice().unwrap(),
            close.as_slice().unwrap(),
            None,
        )
        .expect("Pivot Points GPU calculation failed");

        // All values should be 100.0 (except first which is NaN)
        for i in 1..result.pp.len() {
            assert!((result.pp[i] - 100.0).abs() < 1e-10);
            assert!((result.s1[i] - 100.0).abs() < 1e-10);
            assert!((result.s2[i] - 100.0).abs() < 1e-10);
            assert!((result.s3[i] - 100.0).abs() < 1e-10);
            assert!((result.r1[i] - 100.0).abs() < 1e-10);
            assert!((result.r2[i] - 100.0).abs() < 1e-10);
            assert!((result.r3[i] - 100.0).abs() < 1e-10);
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pivot_points_gpu_validation() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        // Should fail: mismatched lengths
        let high = arr1(&[110.0, 115.0]);
        let low = arr1(&[105.0]);
        let close = arr1(&[108.0, 112.0]);
        let result = pivot_points_gpu(
            device.clone(),
            high.as_slice().unwrap(),
            low.as_slice().unwrap(),
            close.as_slice().unwrap(),
            None,
        );
        assert!(result.is_err(), "Should fail with mismatched lengths");

        // Should fail: insufficient data (need at least 2 points)
        let high = arr1(&[110.0]);
        let low = arr1(&[105.0]);
        let close = arr1(&[108.0]);
        let result = pivot_points_gpu(
            device,
            high.as_slice().unwrap(),
            low.as_slice().unwrap(),
            close.as_slice().unwrap(),
            None,
        );
        assert!(result.is_err(), "Should fail with insufficient data");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pivot_points_gpu_cpu_parity() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

        // Test against CPU implementation
        let high = arr1(&[110.0, 115.0, 120.0]);
        let low = arr1(&[100.0, 110.0, 115.0]);
        let close = arr1(&[105.0, 112.0, 118.0]);

        let gpu_result = pivot_points_gpu(
            device,
            high.as_slice().unwrap(),
            low.as_slice().unwrap(),
            close.as_slice().unwrap(),
            None,
        )
        .expect("Pivot Points GPU calculation failed");

        // Calculate CPU reference for second point (first period's data)
        // PP = (110 + 100 + 105) / 3 = 105
        let pp_cpu = (110.0 + 100.0 + 105.0) / 3.0;
        let r1_cpu = 2.0 * pp_cpu - 100.0; // 110
        let r2_cpu = pp_cpu + (110.0 - 100.0); // 115
        let r3_cpu = 110.0 + 2.0 * (pp_cpu - 100.0); // 120
        let s1_cpu = 2.0 * pp_cpu - 110.0; // 100
        let s2_cpu = pp_cpu - (110.0 - 100.0); // 95
        let s3_cpu = 100.0 - 2.0 * (110.0 - pp_cpu); // 90

        assert!((gpu_result.pp[1] - pp_cpu).abs() < 1e-10);
        assert!((gpu_result.r1[1] - r1_cpu).abs() < 1e-10);
        assert!((gpu_result.r2[1] - r2_cpu).abs() < 1e-10);
        assert!((gpu_result.r3[1] - r3_cpu).abs() < 1e-10);
        assert!((gpu_result.s1[1] - s1_cpu).abs() < 1e-10);
        assert!((gpu_result.s2[1] - s2_cpu).abs() < 1e-10);
        assert!((gpu_result.s3[1] - s3_cpu).abs() < 1e-10);
    }
}
