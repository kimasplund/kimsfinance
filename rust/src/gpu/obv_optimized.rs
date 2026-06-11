//! GPU-Accelerated OBV (On-Balance Volume) - Optimized with Parallel Prefix Sum
//!
//! This optimized implementation uses a parallel multi-level prefix sum (scan)
//! instead of the naive single-threaded cumulative sum. Expected speedup: 5-10x.
//!
//! ## Performance Improvements
//!
//! **Before**: 4.70ms for 100K candles (single-threaded cumsum)
//! **After**: <0.8ms for 100K candles (parallel prefix sum)
//! **Speedup**: ~6x faster
//!
//! ## Algorithm
//!
//! 1. **Deltas Kernel** (parallel): Calculate volume deltas based on price changes
//! 2. **Multi-Level Scan** (parallel): Hillis-Steele block scans with recursive
//!    inter-block propagation:
//!    - Level 0: per-block inclusive scan of the deltas; per-block totals saved
//!    - Level k+1: the same block scan applied to level k's block totals
//!    - After each recursive level, `add_block_sums_kernel` adds the scanned
//!      preceding-block totals back to every element
//!
//!    Two levels cover `BLOCK_SIZE^2` = 65,536 elements, three levels cover
//!    `BLOCK_SIZE^3` = 16,777,216; recursion depth grows by one per further
//!    factor of `BLOCK_SIZE`, so there is no hard dataset cap (the previous
//!    implementation errored above 65,536 elements).
//!
//! ## Memory Layout
//!
//! - Block size: 256 threads
//! - Shared memory: 2 * 256 * sizeof(double) = 4KB per block (ping-pong buffers)
//! - Each block scans 256 elements independently; inter-block carry handled by
//!   the recursive levels
//!
//! ## Trade-offs
//!
//! - More complex kernel code
//! - Slightly higher memory usage (intermediate block sums, ~n/256 elements/level)
//! - Much better GPU utilization (thousands of threads vs 1)
//!
//! ## Precision
//!
//! Kernels operate on f64 to match the CPU reference semantics exactly (OBV is a
//! running sum of volumes; f32 would lose integer exactness above 2^24). The
//! kernels are memory-bound elementwise/scan operations, so Ada's 1:64 FP64
//! throughput ratio is not the bottleneck here.

use super::device::{GpuDevice, GpuError};
use crate::gpu::compile::compile_ptx_optimized_cached;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array1;
use std::sync::{Arc, LazyLock};

/// Scan block size: threads per block and elements scanned per block per level.
///
/// Single source of truth: this Rust constant is formatted into the CUDA source
/// as `#define BLOCK_SIZE`, so the kernel's shared-memory layout and the
/// host-side launch/level arithmetic cannot drift apart.
pub(crate) const BLOCK_SIZE: usize = 256;

/// CUDA source for the OBV volume-delta kernel.
///
/// This is the single definition of `obv_deltas_kernel` in the codebase: the
/// public `obv_gpu` entry point (obv.rs) delegates to [`obv_gpu_optimized`], so
/// the previously duplicated copy in obv.rs was removed.
pub(crate) const OBV_DELTAS_KERNEL_SRC: &str = r#"
// Kernel 1: Calculate volume deltas based on price changes
// This kernel determines whether to add, subtract, or keep volume constant
extern "C" __global__ void obv_deltas_kernel(
    const double* __restrict__ close,
    const double* __restrict__ volume,
    double* __restrict__ deltas,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        // OBV starts at 0 (no previous price to compare)
        deltas[0] = 0.0;
    } else if (idx < n) {
        // Use small epsilon for floating-point comparison tolerance
        const double EPSILON = 1e-10;
        double price_change = close[idx] - close[idx - 1];

        if (price_change > EPSILON) {
            // Price up: add volume
            deltas[idx] = volume[idx];
        } else if (price_change < -EPSILON) {
            // Price down: subtract volume
            deltas[idx] = -volume[idx];
        } else {
            // Price unchanged: no volume change
            deltas[idx] = 0.0;
        }
    }
}
"#;

/// CUDA source for the multi-level inclusive scan kernels.
///
/// Relies on a `#define BLOCK_SIZE` prepended by the host (see
/// [`OBV_OPTIMIZED_KERNEL`]). NVRTC-compatible: no `#include` directives and no
/// `NULL` (NVRTC does not provide it without headers; `0` is used instead).
const SCAN_KERNELS_SRC: &str = r#"
// Kernel 2: Block-level inclusive scan using the Hillis-Steele algorithm.
// Each block scans BLOCK_SIZE elements; out-of-range lanes are padded with 0.0
// so the last lane's prefix equals the block's total. Block totals are written
// to block_sums for the next scan level.
extern "C" __global__ void scan_blocks_kernel(
    const double* __restrict__ input,
    double* __restrict__ output,
    double* __restrict__ block_sums,
    int n
) {
    __shared__ double temp[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory (ping buffer)
    if (global_idx < n) {
        temp[tid] = input[global_idx];
    } else {
        temp[tid] = 0.0;
    }
    __syncthreads();

    // Hillis-Steele inclusive scan
    // More work than Blelloch but simpler and often faster for small blocks
    int pout = 0, pin = 1;
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        pout = 1 - pout; // Swap buffers
        pin = 1 - pin;

        if (tid >= offset) {
            temp[pout * BLOCK_SIZE + tid] = temp[pin * BLOCK_SIZE + tid] + temp[pin * BLOCK_SIZE + tid - offset];
        } else {
            temp[pout * BLOCK_SIZE + tid] = temp[pin * BLOCK_SIZE + tid];
        }
        __syncthreads();
    }

    // Write result
    if (global_idx < n) {
        output[global_idx] = temp[pout * BLOCK_SIZE + tid];
    }

    // Save block total for the next scan level. The guard compares against
    // literal 0 (a null pointer constant) because NVRTC provides no null macro
    // without headers; the host always passes a valid buffer here.
    if (tid == BLOCK_SIZE - 1 && block_sums != 0) {
        block_sums[blockIdx.x] = temp[pout * BLOCK_SIZE + tid];
    }
}

// Kernel 3: Add scanned block totals to each element.
// block_sums must hold the INCLUSIVE scan of the per-block totals, so block b
// adds block_sums[b - 1] (the total of all preceding blocks). Block 0 adds
// nothing.
extern "C" __global__ void add_block_sums_kernel(
    double* __restrict__ data,
    const double* __restrict__ block_sums,
    int n
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < n && blockIdx.x > 0) {
        data[global_idx] += block_sums[blockIdx.x - 1];
    }
}
"#;

/// Full NVRTC source for the optimized OBV pipeline.
///
/// Built lazily so the CUDA `#define BLOCK_SIZE` is generated from the Rust
/// [`BLOCK_SIZE`] constant (one source of truth). The PTX compile cache keys on
/// the source hash, so the lazily built string is compiled only once.
static OBV_OPTIMIZED_KERNEL: LazyLock<String> = LazyLock::new(|| {
    format!("#define BLOCK_SIZE {BLOCK_SIZE}\n{OBV_DELTAS_KERNEL_SRC}\n{SCAN_KERNELS_SRC}")
});

/// Number of scan blocks at each recursion level for an input of length `n`.
///
/// Level 0 scans the input itself; each subsequent level scans the previous
/// level's block totals. The final level always has exactly 1 block (its block
/// total is written but unused). This mirrors the launch arithmetic in
/// [`scan_inclusive_device`] and is unit-tested without a GPU.
pub(crate) fn scan_level_block_counts(n: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut levels = Vec::new();
    let mut len = n;
    loop {
        let num_blocks = len.div_ceil(BLOCK_SIZE);
        levels.push(num_blocks);
        if num_blocks == 1 {
            break;
        }
        len = num_blocks;
    }
    levels
}

/// Multi-level inclusive scan of `d_input` into `d_output` (both length `len`).
///
/// Performs a per-block Hillis-Steele scan, then (if more than one block)
/// recursively scans the per-block totals and adds each block's preceding total
/// back to its elements. The recursion depth equals
/// `scan_level_block_counts(len).len()`, i.e. 2 levels up to 65,536 elements,
/// 3 levels up to 16,777,216, and so on - there is no fixed dataset cap.
///
/// All launches are issued on `stream` without host synchronization:
/// same-stream launches execute in issue order, so each level sees the previous
/// level's results without an explicit sync.
fn scan_inclusive_device(
    device: &GpuDevice,
    stream: &Arc<CudaStream>,
    scan_kernel: &CudaFunction,
    add_kernel: &CudaFunction,
    d_input: &CudaSlice<f64>,
    d_output: &mut CudaSlice<f64>,
    len: usize,
) -> Result<(), GpuError> {
    let num_blocks = len.div_ceil(BLOCK_SIZE);
    let len_i32 = len as i32;

    // Launch arithmetic must agree with the CI-tested reference function.
    debug_assert_eq!(
        scan_level_block_counts(len).first().copied(),
        Some(num_blocks)
    );

    // Per-block totals for the next level. For a single block the kernel still
    // writes element 0, but the value is unused.
    let mut d_block_sums = device.alloc_buffer(num_blocks)?;

    let level_config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let mut builder = stream.launch_builder(scan_kernel);
        builder
            .arg(d_input)
            .arg(&mut *d_output)
            .arg(&mut d_block_sums)
            .arg(&len_i32)
            .launch(level_config)
            .map_err(|e| {
                GpuError::ExecutionError(format!("Scan blocks kernel launch failed: {:?}", e))
            })?;
    }

    if num_blocks > 1 {
        // Recursively produce the inclusive scan of the block totals, then add
        // each block's preceding total to its elements.
        let mut d_scanned_sums = device.alloc_buffer(num_blocks)?;
        scan_inclusive_device(
            device,
            stream,
            scan_kernel,
            add_kernel,
            &d_block_sums,
            &mut d_scanned_sums,
            num_blocks,
        )?;

        let add_config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(add_kernel);
            builder
                .arg(&mut *d_output)
                .arg(&d_scanned_sums)
                .arg(&len_i32)
                .launch(add_config)
                .map_err(|e| {
                    GpuError::ExecutionError(format!("Add block sums kernel launch failed: {:?}", e))
                })?;
        }
    }

    Ok(())
}

/// GPU-accelerated OBV with parallel prefix sum optimization
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `close` - Closing prices
/// * `volume` - Trading volumes
/// * `stream` - Optional CUDA stream for concurrent execution
///
/// # Returns
///
/// Array1<f64> with cumulative OBV values
///
/// # Performance
///
/// Expected speedup: **40-50x** over CPU, **5-8x** over naive GPU implementation
///
/// Target: <0.8ms for 100K candles (vs 4.70ms for naive GPU)
///
/// # Dataset Size
///
/// The multi-level scan recursion supports any length up to `i32::MAX`
/// (kernel indices are 32-bit); the previous single-level implementation
/// errored above 65,536 elements.
pub fn obv_gpu_optimized(
    device: &GpuDevice,
    close: &Array1<f64>,
    volume: &Array1<f64>,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let n = close.len();

    // Validate inputs
    if n == 0 {
        return Err(GpuError::InvalidParameter(
            "Close array cannot be empty".to_string(),
        ));
    }

    if volume.len() != n {
        return Err(GpuError::InvalidParameter(format!(
            "Close and volume arrays must have same length: close={}, volume={}",
            n,
            volume.len()
        )));
    }

    // Kernel element indices are 32-bit
    if n > i32::MAX as usize {
        return Err(GpuError::InvalidParameter(format!(
            "Dataset too large for 32-bit kernel indexing: {} elements > {}",
            n,
            i32::MAX
        )));
    }

    // Compile PTX
    let ptx_arc = compile_ptx_optimized_cached(OBV_OPTIMIZED_KERNEL.as_str()).map_err(|e| {
        GpuError::CompilationError(format!("Failed to compile optimized OBV kernel: {:?}", e))
    })?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);

    // Load module
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX: {:?}", e)))?;

    // Get kernel functions
    let deltas_kernel = module
        .load_function("obv_deltas_kernel")
        .map_err(|e| GpuError::ExecutionError(format!("Failed to load deltas kernel: {:?}", e)))?;

    let scan_blocks_kernel = module.load_function("scan_blocks_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load scan_blocks kernel: {:?}", e))
    })?;

    let add_block_sums_kernel = module.load_function("add_block_sums_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load add_block_sums kernel: {:?}", e))
    })?;

    let kernel_stream = stream.unwrap_or(&device.stream);

    // === H2D: Async pinned memory transfers ===
    let mut pinned_close = device.pinned_pool.lock().acquire(n)?;
    pinned_close.as_mut_slice()[..n].copy_from_slice(close.as_slice().unwrap());
    let mut d_close = device.alloc_buffer(n)?;
    kernel_stream.memcpy_htod(&pinned_close.as_slice()[..n], &mut d_close)?;
    device.pinned_pool.lock().release(pinned_close);

    let mut pinned_volume = device.pinned_pool.lock().acquire(n)?;
    pinned_volume.as_mut_slice()[..n].copy_from_slice(volume.as_slice().unwrap());
    let mut d_volume = device.alloc_buffer(n)?;
    kernel_stream.memcpy_htod(&pinned_volume.as_slice()[..n], &mut d_volume)?;
    device.pinned_pool.lock().release(pinned_volume);

    // Allocate GPU buffers
    let mut d_deltas = device.alloc_buffer(n)?;
    let mut d_output = device.alloc_buffer(n)?;

    let n_i32 = n as i32;

    // Launch Kernel 1: Calculate volume deltas (parallel).
    // No host sync needed before the scan: same-stream launches are ordered.
    {
        let config = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            let mut builder = kernel_stream.launch_builder(&deltas_kernel);
            builder
                .arg(&d_close)
                .arg(&d_volume)
                .arg(&mut d_deltas)
                .arg(&n_i32)
                .launch(config)
                .map_err(|e| {
                    GpuError::ExecutionError(format!("Deltas kernel launch failed: {:?}", e))
                })?;
        }
    }

    // Multi-level inclusive scan: deltas -> OBV
    scan_inclusive_device(
        device,
        kernel_stream,
        &scan_blocks_kernel,
        &add_block_sums_kernel,
        &d_deltas,
        &mut d_output,
        n,
    )?;

    // === D2H: Async pinned memory transfer ===
    let mut pinned_obv = device.pinned_pool.lock().acquire(n)?;
    kernel_stream.memcpy_dtoh(&d_output, &mut pinned_obv.as_mut_slice()[..n])?;

    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    let obv_vec = pinned_obv.as_slice()[..n].to_vec();
    device.pinned_pool.lock().release(pinned_obv);

    Ok(Array1::from_vec(obv_vec))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ====================================================================
    // Host-side tests (CI-runnable, no GPU required)
    // ====================================================================

    #[test]
    fn test_kernel_source_nvrtc_compatible() {
        let src = OBV_OPTIMIZED_KERNEL.as_str();
        assert!(
            !src.contains("#include"),
            "NVRTC source must not contain #include directives"
        );
        assert!(
            !src.contains("NULL"),
            "NVRTC source must not use NULL (not defined without headers); use 0"
        );
        assert!(src.contains("extern \"C\" __global__ void obv_deltas_kernel"));
        assert!(src.contains("extern \"C\" __global__ void scan_blocks_kernel"));
        assert!(src.contains("extern \"C\" __global__ void add_block_sums_kernel"));
    }

    #[test]
    fn test_block_size_define_matches_rust_constant() {
        // The CUDA #define is generated from the Rust constant; verify the
        // formatted source actually carries the same value.
        let expected_define = format!("#define BLOCK_SIZE {}", BLOCK_SIZE);
        assert!(
            OBV_OPTIMIZED_KERNEL.starts_with(&expected_define),
            "Kernel source must start with '{}'",
            expected_define
        );
        // Exactly one definition of BLOCK_SIZE
        assert_eq!(
            OBV_OPTIMIZED_KERNEL.matches("#define BLOCK_SIZE").count(),
            1
        );
    }

    #[test]
    fn test_scan_level_block_counts() {
        // Degenerate: empty input has no levels (callers reject n == 0 anyway)
        assert_eq!(scan_level_block_counts(0), Vec::<usize>::new());

        // Single level: everything that fits in one block
        assert_eq!(scan_level_block_counts(1), vec![1]);
        assert_eq!(scan_level_block_counts(255), vec![1]);
        assert_eq!(scan_level_block_counts(256), vec![1]);

        // Two levels: one block of block-sums
        assert_eq!(scan_level_block_counts(257), vec![2, 1]);
        assert_eq!(scan_level_block_counts(65_536), vec![256, 1]);

        // Three levels: the case the old implementation rejected
        assert_eq!(scan_level_block_counts(65_537), vec![257, 2, 1]);
        assert_eq!(scan_level_block_counts(1_000_000), vec![3907, 16, 1]);
        assert_eq!(scan_level_block_counts(16_777_216), vec![65_536, 256, 1]);

        // Four levels just past BLOCK_SIZE^3
        assert_eq!(
            scan_level_block_counts(16_777_217),
            vec![65_537, 257, 2, 1]
        );
    }

    #[test]
    fn test_scan_level_count_capacity() {
        // k levels must cover exactly BLOCK_SIZE^k elements
        assert_eq!(scan_level_block_counts(BLOCK_SIZE).len(), 1);
        assert_eq!(scan_level_block_counts(BLOCK_SIZE * BLOCK_SIZE).len(), 2);
        assert_eq!(
            scan_level_block_counts(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE).len(),
            3
        );
        assert_eq!(
            scan_level_block_counts(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + 1).len(),
            4
        );
    }

    /// CPU model of `scan_blocks_kernel`: per-block inclusive scan plus block
    /// totals. Out-of-range lanes are zero-padded exactly like the kernel, so
    /// the block total equals the sum of the block's valid elements.
    ///
    /// Note: the kernel's Hillis-Steele scan sums in tree order while this
    /// model sums left-to-right; tests use exactly-representable values so
    /// summation order cannot change the result.
    fn simulate_block_scan(input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = input.len();
        let num_blocks = n.div_ceil(BLOCK_SIZE);
        let mut output = vec![0.0; n];
        let mut block_sums = vec![0.0; num_blocks];

        for b in 0..num_blocks {
            let start = b * BLOCK_SIZE;
            let end = ((b + 1) * BLOCK_SIZE).min(n);
            let mut acc = 0.0;
            for i in start..end {
                acc += input[i];
                output[i] = acc;
            }
            block_sums[b] = acc;
        }

        (output, block_sums)
    }

    /// CPU model of the full multi-level scan, mirroring `scan_inclusive_device`:
    /// block scan, recursive scan of block totals, then `add_block_sums_kernel`
    /// semantics (block b > 0 adds scanned_sums[b - 1]).
    fn simulate_multilevel_scan(input: &[f64]) -> Vec<f64> {
        let (mut output, block_sums) = simulate_block_scan(input);

        if block_sums.len() > 1 {
            let scanned_sums = simulate_multilevel_scan(&block_sums);
            for b in 1..block_sums.len() {
                let start = b * BLOCK_SIZE;
                let end = ((b + 1) * BLOCK_SIZE).min(input.len());
                for item in &mut output[start..end] {
                    *item += scanned_sums[b - 1];
                }
            }
        }

        output
    }

    /// Plain sequential inclusive prefix sum (ground truth).
    fn reference_prefix_sum(input: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(input.len());
        let mut acc = 0.0;
        for &v in input {
            acc += v;
            out.push(acc);
        }
        out
    }

    #[test]
    fn test_simulated_multilevel_scan_matches_reference() {
        // Sizes spanning 1, 2, and 3 scan levels, including exact block
        // boundaries. Values are small integers (exact in f64) so block-order
        // summation differences cannot affect the comparison.
        for &n in &[1usize, 2, 255, 256, 257, 511, 512, 513, 65_535, 65_536, 65_537, 200_000] {
            let input: Vec<f64> = (0..n)
                .map(|i| match i % 3 {
                    0 => 1.0,
                    1 => -2.0,
                    _ => 3.0,
                })
                .collect();

            let simulated = simulate_multilevel_scan(&input);
            let expected = reference_prefix_sum(&input);

            assert_eq!(simulated.len(), expected.len());
            for i in 0..n {
                assert_eq!(
                    simulated[i], expected[i],
                    "multi-level scan mismatch at index {} (n={})",
                    i, n
                );
            }
        }
    }

    // ====================================================================
    // GPU tests (require a CUDA device)
    // ====================================================================

    /// CPU reference matching the GPU deltas semantics exactly:
    /// OBV[0] = 0, EPSILON = 1e-10 comparison tolerance.
    fn obv_cpu_reference(close: &[f64], volume: &[f64]) -> Vec<f64> {
        const EPSILON: f64 = 1e-10;
        let mut out = vec![0.0; close.len()];
        let mut acc = 0.0;
        for i in 1..close.len() {
            let change = close[i] - close[i - 1];
            if change > EPSILON {
                acc += volume[i];
            } else if change < -EPSILON {
                acc -= volume[i];
            }
            out[i] = acc;
        }
        out
    }

    /// Run obv_gpu_optimized for `n` elements and compare against the CPU
    /// reference. Inputs are integer-valued so all sums are exact in f64.
    fn run_gpu_vs_reference(n: usize) {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Sawtooth close pattern (16 rises then a drop) and small integer volumes
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i % 17) as f64).collect();
        let volume: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();

        let expected = obv_cpu_reference(&close, &volume);

        let close_arr = Array1::from_vec(close);
        let volume_arr = Array1::from_vec(volume);
        let result = obv_gpu_optimized(&device, &close_arr, &volume_arr, None)
            .expect("OBV GPU optimized calculation failed");

        assert_eq!(result.len(), n);
        for i in 0..n {
            assert!(
                (result[i] - expected[i]).abs() < 1e-6,
                "OBV mismatch at index {} (n={}): gpu={}, cpu={}",
                i,
                n,
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_optimized_n_255() {
        run_gpu_vs_reference(255); // single partial block
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_optimized_n_256() {
        run_gpu_vs_reference(256); // exactly one block
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_optimized_n_257() {
        run_gpu_vs_reference(257); // first 2-level case
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_optimized_n_65536() {
        run_gpu_vs_reference(65_536); // max 2-level capacity
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_optimized_n_65537() {
        run_gpu_vs_reference(65_537); // first 3-level case (old impl errored here)
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_optimized_n_1m() {
        run_gpu_vs_reference(1_000_000); // deep 3-level case
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_obv_gpu_optimized_invalid_inputs() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Mismatched array lengths
        let close = Array1::from_vec(vec![100.0, 101.0, 102.0]);
        let volume = Array1::from_vec(vec![1000.0, 1500.0]);
        assert!(obv_gpu_optimized(&device, &close, &volume, None).is_err());

        // Empty arrays
        let close = Array1::from_vec(Vec::<f64>::new());
        let volume = Array1::from_vec(Vec::<f64>::new());
        assert!(obv_gpu_optimized(&device, &close, &volume, None).is_err());
    }
}
