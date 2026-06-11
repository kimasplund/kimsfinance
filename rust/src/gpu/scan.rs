//! Reusable GPU scan primitives: prefix sum/max + affine linear-recurrence scan
//!
//! This module is the shared building block that five audit areas need:
//!
//! - **RSI / ADX / MFI / MACD**: Wilder/EMA smoothing currently round-trips to
//!   the CPU (2-6 PCIe transfers + host syncs per indicator). The affine scan
//!   here parallelizes those first-order recurrences entirely on-device.
//! - **OBV / VWAP**: cumulative sums currently run on a single GPU thread;
//!   [`inclusive_scan_f32`] / [`inclusive_scan_pair_sum_f32`] replace them
//!   with O(n/p + log n) parallel scans.
//! - **Drawdown**: needs a prefix max ([`ScanOp::Max`]).
//!
//! # Why an affine scan parallelizes EMA/Wilder
//!
//! The recurrence `y[i] = a*y[i-1] + b[i]` is **not** scannable with the raw
//! recurrence operator (it is not associative - the mathematical flaw in
//! `rsi_fused.cu`'s CUB approach). It **is** scannable over affine-transform
//! pairs `(m, c)` (representing `y -> m*y + c`) under function composition:
//!
//! ```text
//! (m1, c1) then (m2, c2)  =>  (m1*m2, m2*c1 + c2)
//! ```
//!
//! which is associative (see [`affine_compose`] and the property tests in
//! this module). The inclusive scan at index `i` yields `(M, C)` with
//! `y[i] = M*y[-1] + C`; the SMA seed is injected as the pair `(0, SMA)` so
//! `M` collapses to 0 there, making the seed **exact** (deliberately avoiding
//! `rsi_fused.cu`'s `alpha*SMA` seed bug).
//!
//! # Architecture (deterministic 3-kernel scan)
//!
//! 1. `scan_partials_<op>`: 256 threads x 4 items/thread per 1024-element
//!    tile; intra-warp scan via `__shfl_up_sync`, cross-warp via 8 shared
//!    warp aggregates; per-tile aggregate written to a partials array.
//! 2. `scan_aggregates_<op>`: a single block scans up to 1024 partials in
//!    place; the host wrapper recurses for larger inputs (1M+ tiles).
//! 3. `scan_fixup_<op>`: combines the exclusive tile prefix into every
//!    element of tiles `1..`.
//!
//! Chosen over single-pass decoupled-lookback because correctness is fully
//! reviewable without a GPU: no spin-waits or inter-block protocols.
//!
//! All entry points operate on **device-resident buffers** and never
//! synchronize the host, so they compose into existing pipelines and remain
//! CUDA-graph capturable.
//!
//! # Precision
//!
//! Kernels default to f32: on Ada (sm_89) FP64 throughput is 1/64 of FP32.
//! For the affine scan, composed `m`-products are products of `(1-alpha)`
//! factors in `[0,1)` that decay geometrically, so old contributions are
//! down-weighted and f32 rounding error self-heals rather than accumulating.
//! [`AffinePrecision::F64Acc`] keeps float2 pairs in global memory but does
//! all accumulation in double as a precision fallback, and f64 sum/max/affine
//! paths exist for full-precision consumers.
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::{GpuDevice, scan};
//!
//! let device = GpuDevice::new()?;
//! let d_gains = device.copy_to_device_f32(&gains)?;
//! let mut d_smooth = device.allocate_device_buffer::<f32>(gains.len())?;
//!
//! // Wilder smoothing entirely on-device (no host round-trip):
//! scan::wilder_smooth_f32(&device, None, &d_gains, 14, &mut d_smooth)?;
//! ```

use super::device::{GpuDevice, GpuError};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use std::sync::Arc;

/// CUDA source for all scan kernels (NVRTC-compiled at runtime via the cached
/// `GpuDevice::get_or_load_function` path: PTX is compiled once per process,
/// the module loaded once per device).
pub(crate) const SCAN_KERNELS_SRC: &str = include_str!("kernels/scan.cu");

// Layout contract mirrored in kernels/scan.cu (keep in sync; asserted by
// test_scan_kernel_layout_contract_matches_rust_consts).
/// Threads per scan block.
pub const SCAN_BLOCK_THREADS: usize = 256;
/// Items processed per thread in the tile scan.
pub const SCAN_ITEMS_PER_THREAD: usize = 4;
/// Logical elements per tile (= SCAN_BLOCK_THREADS * SCAN_ITEMS_PER_THREAD).
pub const SCAN_TILE: usize = SCAN_BLOCK_THREADS * SCAN_ITEMS_PER_THREAD;

/// Exact identity element of the affine composition: `(1, 0)` (`y -> y`).
pub const AFFINE_IDENTITY: (f64, f64) = (1.0, 0.0);

/// Scan operator selector for the scalar entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOp {
    /// Inclusive prefix sum.
    Sum,
    /// Inclusive prefix max (identity: -inf).
    Max,
}

/// Accumulation precision for the affine (linear-recurrence) scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffinePrecision {
    /// Pure f32 (default; self-healing error, see module docs).
    F32,
    /// float2 pairs in global memory, double accumulation in
    /// registers/shared/partials. Fallback for precision-critical callers.
    /// FP64 ALU cost is bounded because scans are bandwidth-bound.
    F64Acc,
}

/// Compose two affine transforms `(m, c)` representing `y -> m*y + c`.
///
/// `a` is applied **first** (earlier in the sequence), `b` second:
/// `b(a(y)) = b.m*(a.m*y + a.c) + b.c = (a.m*b.m)*y + (b.m*a.c + b.c)`.
///
/// This is the exact host-side mirror of the `combine` operator in
/// `kernels/scan.cu` and is associative - the property the parallel scan
/// relies on (verified by unit tests below).
#[inline]
pub fn affine_compose(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0, b.0 * a.1 + b.1)
}

// ============================================================================
// Module / kernel loading
// ============================================================================

fn load_scan_module(device: &GpuDevice) -> Result<Arc<CudaModule>, GpuError> {
    let ptx_arc = compile_ptx_optimized_cached(SCAN_KERNELS_SRC).map_err(|e| {
        GpuError::CompilationError(format!("Failed to compile scan kernels: {:?}", e))
    })?;
    // PTX compilation is cached (SHA-256 keyed); module load itself is cheap.
    let ptx = Arc::unwrap_or_clone(ptx_arc);
    device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load scan PTX: {:?}", e)))
}

fn get_function(module: &Arc<CudaModule>, name: &str) -> Result<CudaFunction, GpuError> {
    module
        .load_function(name)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load kernel {}: {:?}", name, e)))
}

/// The three kernels of one scan operator instantiation.
struct ScanKernelSet {
    partials: CudaFunction,
    aggregates: CudaFunction,
    fixup: CudaFunction,
}

impl ScanKernelSet {
    fn load(module: &Arc<CudaModule>, base: &str) -> Result<Self, GpuError> {
        Ok(Self {
            partials: get_function(module, &format!("scan_partials_{}", base))?,
            aggregates: get_function(module, &format!("scan_aggregates_{}", base))?,
            fixup: get_function(module, &format!("scan_fixup_{}", base))?,
        })
    }
}

// ============================================================================
// Launch helpers
// ============================================================================

/// Number of 1024-element tiles covering `n` logical elements.
#[inline]
fn num_tiles(n: usize) -> usize {
    n.div_ceil(SCAN_TILE)
}

fn tile_cfg(tiles: usize) -> Result<LaunchConfig, GpuError> {
    let grid = u32::try_from(tiles).map_err(|_| {
        GpuError::InvalidParameter(format!("scan input too large: {} tiles exceed grid limit", tiles))
    })?;
    Ok(LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (SCAN_BLOCK_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn elementwise_cfg(count: usize) -> Result<LaunchConfig, GpuError> {
    let blocks = count.div_ceil(SCAN_BLOCK_THREADS).max(1);
    let grid = u32::try_from(blocks).map_err(|_| {
        GpuError::InvalidParameter(format!("launch too large: {} blocks exceed grid limit", blocks))
    })?;
    Ok(LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (SCAN_BLOCK_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[inline]
fn launch_failed(kernel: &str, e: impl std::fmt::Debug) -> GpuError {
    GpuError::ExecutionError(format!("{} kernel launch failed: {:?}", kernel, e))
}

// ============================================================================
// Generic 3-kernel scan driver
// ============================================================================

/// Recursively scan a partials buffer **in place** (stream-ordered, no host
/// sync). `n_items` counts logical elements; `part_units` is the number of
/// `P` words per logical element (1 for scalars, 2 for float2/double2 pairs).
///
/// In-place tile scanning is safe: the partials kernel loads each tile fully
/// before writing it, and tiles are disjoint across blocks (see scan.cu).
fn scan_inplace_levels<P>(
    stream: &Arc<CudaStream>,
    set: &ScanKernelSet,
    buf: &CudaSlice<P>,
    n_items: usize,
    part_units: usize,
) -> Result<(), GpuError>
where
    P: DeviceRepr + ValidAsZeroBits,
{
    let n_i64 = n_items as i64;

    if n_items <= SCAN_TILE {
        // Single block scans up to SCAN_TILE aggregates in place.
        let cfg = tile_cfg(1)?;
        let mut b = stream.launch_builder(&set.aggregates);
        b.arg(buf).arg(&n_i64);
        unsafe { b.launch(cfg) }.map_err(|e| launch_failed("scan_aggregates", e))?;
        return Ok(());
    }

    let tiles = num_tiles(n_items);
    let partials: CudaSlice<P> = stream
        .alloc_zeros(tiles * part_units)
        .map_err(|e| GpuError::AllocationError(format!("scan partials alloc failed: {:?}", e)))?;

    let cfg = tile_cfg(tiles)?;
    {
        // In-place tile scan (in == out is safe, see above).
        let mut b = stream.launch_builder(&set.partials);
        b.arg(buf).arg(buf).arg(&partials).arg(&n_i64);
        unsafe { b.launch(cfg) }.map_err(|e| launch_failed("scan_partials", e))?;
    }

    scan_inplace_levels(stream, set, &partials, tiles, part_units)?;

    {
        let mut b = stream.launch_builder(&set.fixup);
        b.arg(buf).arg(&partials).arg(&n_i64);
        unsafe { b.launch(cfg) }.map_err(|e| launch_failed("scan_fixup", e))?;
    }
    Ok(())
}

/// Top-level scan driver. `E` is the global-memory word type of `d_in` /
/// `d_out` (f32 or f64; pairs are 2 words per logical element), `P` the word
/// type of the partials buffer (differs from `E` only for the F64Acc affine
/// variant). `d_in == d_out` aliasing is allowed (in-place).
#[allow(clippy::too_many_arguments)]
fn scan_device_buffers<E, P>(
    stream: &Arc<CudaStream>,
    partials0: &CudaFunction,
    fixup0: &CudaFunction,
    rec_set: &ScanKernelSet,
    d_in: &CudaSlice<E>,
    d_out: &CudaSlice<E>,
    n_items: usize,
    part_units: usize,
) -> Result<(), GpuError>
where
    E: DeviceRepr,
    P: DeviceRepr + ValidAsZeroBits,
{
    if n_items == 0 {
        return Ok(());
    }

    let tiles = num_tiles(n_items);
    let n_i64 = n_items as i64;

    let partials: CudaSlice<P> = stream
        .alloc_zeros(tiles * part_units)
        .map_err(|e| GpuError::AllocationError(format!("scan partials alloc failed: {:?}", e)))?;

    let cfg = tile_cfg(tiles)?;
    {
        let mut b = stream.launch_builder(partials0);
        b.arg(d_in).arg(d_out).arg(&partials).arg(&n_i64);
        unsafe { b.launch(cfg) }.map_err(|e| launch_failed("scan_partials", e))?;
    }

    if tiles > 1 {
        scan_inplace_levels(stream, rec_set, &partials, tiles, part_units)?;
        let mut b = stream.launch_builder(fixup0);
        b.arg(d_out).arg(&partials).arg(&n_i64);
        unsafe { b.launch(cfg) }.map_err(|e| launch_failed("scan_fixup", e))?;
    }
    Ok(())
}

fn check_equal_lens(in_len: usize, out_len: usize) -> Result<(), GpuError> {
    if in_len != out_len {
        return Err(GpuError::InvalidParameter(format!(
            "scan input/output length mismatch: {} vs {}",
            in_len, out_len
        )));
    }
    Ok(())
}

fn check_pair_len(len: usize) -> Result<usize, GpuError> {
    if !len.is_multiple_of(2) {
        return Err(GpuError::InvalidParameter(format!(
            "pair scan buffer length must be even (2 words per pair), got {}",
            len
        )));
    }
    Ok(len / 2)
}

// ============================================================================
// Public scan entry points (device-resident, no transfers, no host sync)
// ============================================================================

/// Inclusive scan (prefix sum or prefix max) over a device-resident f32
/// buffer. Composable: runs entirely on `stream` (or the device default
/// stream) without host synchronization.
///
/// `d_in` and `d_out` must have equal lengths. The caller is responsible for
/// synchronizing the stream before reading `d_out` from the host.
pub fn inclusive_scan_f32(
    device: &GpuDevice,
    stream: Option<&Arc<CudaStream>>,
    d_in: &CudaSlice<f32>,
    d_out: &mut CudaSlice<f32>,
    op: ScanOp,
) -> Result<(), GpuError> {
    check_equal_lens(d_in.len(), d_out.len())?;
    let n = d_in.len();
    if n == 0 {
        return Ok(());
    }
    let module = load_scan_module(device)?;
    let base = match op {
        ScanOp::Sum => "sum_f32",
        ScanOp::Max => "max_f32",
    };
    let set = ScanKernelSet::load(&module, base)?;
    let stream = stream.unwrap_or(&device.stream);
    scan_device_buffers::<f32, f32>(stream, &set.partials, &set.fixup, &set, d_in, d_out, n, 1)
}

/// Inclusive scan (prefix sum or prefix max) over a device-resident f64
/// buffer.
///
/// Precision note: FP64 ALU throughput on Ada is 1/64 of FP32, but scans are
/// bandwidth-bound (~2 flops per 8 bytes), so the f64 path is primarily
/// limited by the doubled memory traffic. Prefer f32 where the consumer
/// tolerates it.
pub fn inclusive_scan_f64(
    device: &GpuDevice,
    stream: Option<&Arc<CudaStream>>,
    d_in: &CudaSlice<f64>,
    d_out: &mut CudaSlice<f64>,
    op: ScanOp,
) -> Result<(), GpuError> {
    check_equal_lens(d_in.len(), d_out.len())?;
    let n = d_in.len();
    if n == 0 {
        return Ok(());
    }
    let module = load_scan_module(device)?;
    let base = match op {
        ScanOp::Sum => "sum_f64",
        ScanOp::Max => "max_f64",
    };
    let set = ScanKernelSet::load(&module, base)?;
    let stream = stream.unwrap_or(&device.stream);
    scan_device_buffers::<f64, f64>(stream, &set.partials, &set.fixup, &set, d_in, d_out, n, 1)
}

/// Inclusive element-wise prefix sum over float2 pairs, e.g. running
/// `(typical_price*volume, volume)` numerator/denominator pairs for VWAP, or
/// `(money_flow_volume, volume)` for MFI/CMF consumers.
///
/// Buffers hold `2*n_pairs` f32 words in interleaved `(x, y)` layout
/// (= `float2[n_pairs]` on the device; cudarc allocations are 256-byte
/// aligned, satisfying float2's 8-byte alignment). Lengths must be equal and
/// even.
pub fn inclusive_scan_pair_sum_f32(
    device: &GpuDevice,
    stream: Option<&Arc<CudaStream>>,
    d_in: &CudaSlice<f32>,
    d_out: &mut CudaSlice<f32>,
) -> Result<(), GpuError> {
    check_equal_lens(d_in.len(), d_out.len())?;
    let n_pairs = check_pair_len(d_in.len())?;
    if n_pairs == 0 {
        return Ok(());
    }
    let module = load_scan_module(device)?;
    let set = ScanKernelSet::load(&module, "pair_sum_f32")?;
    let stream = stream.unwrap_or(&device.stream);
    scan_device_buffers::<f32, f32>(
        stream,
        &set.partials,
        &set.fixup,
        &set,
        d_in,
        d_out,
        n_pairs,
        2,
    )
}

/// Inclusive affine-composition scan over `(m, c)` pairs (see module docs and
/// [`affine_compose`]). Parallelizes first-order recurrences
/// `y[i] = m[i]*y[i-1] + c[i]`; the scanned `c` component at index `i` equals
/// `y[i]` when there is no prior state (`y[-1] = 0`).
///
/// Buffers hold `2*n_pairs` f32 words in interleaved `(m, c)` layout.
/// `d_in == d_out` in-place operation is supported via
/// [`wilder_smooth_f32`]/[`ema_f32`], which use this scan internally.
pub fn inclusive_scan_affine_f32(
    device: &GpuDevice,
    stream: Option<&Arc<CudaStream>>,
    d_in: &CudaSlice<f32>,
    d_out: &mut CudaSlice<f32>,
    precision: AffinePrecision,
) -> Result<(), GpuError> {
    check_equal_lens(d_in.len(), d_out.len())?;
    let n_pairs = check_pair_len(d_in.len())?;
    if n_pairs == 0 {
        return Ok(());
    }
    let module = load_scan_module(device)?;
    let stream = stream.unwrap_or(&device.stream);
    match precision {
        AffinePrecision::F32 => {
            let set = ScanKernelSet::load(&module, "affine_f32")?;
            scan_device_buffers::<f32, f32>(
                stream,
                &set.partials,
                &set.fixup,
                &set,
                d_in,
                d_out,
                n_pairs,
                2,
            )
        }
        AffinePrecision::F64Acc => {
            // Level-0 kernels read/write float2 but accumulate in double;
            // partials are double2, so the recursion uses the affine_f64 set.
            let partials0 = get_function(&module, "scan_partials_affine_f32_f64acc")?;
            let fixup0 = get_function(&module, "scan_fixup_affine_f32_f64acc")?;
            let rec_set = ScanKernelSet::load(&module, "affine_f64")?;
            scan_device_buffers::<f32, f64>(
                stream, &partials0, &fixup0, &rec_set, d_in, d_out, n_pairs, 2,
            )
        }
    }
}

/// Inclusive affine-composition scan over double2 `(m, c)` pairs (full f64
/// path). Buffers hold `2*n_pairs` f64 words in interleaved `(m, c)` layout.
pub fn inclusive_scan_affine_f64(
    device: &GpuDevice,
    stream: Option<&Arc<CudaStream>>,
    d_in: &CudaSlice<f64>,
    d_out: &mut CudaSlice<f64>,
) -> Result<(), GpuError> {
    check_equal_lens(d_in.len(), d_out.len())?;
    let n_pairs = check_pair_len(d_in.len())?;
    if n_pairs == 0 {
        return Ok(());
    }
    let module = load_scan_module(device)?;
    let set = ScanKernelSet::load(&module, "affine_f64")?;
    let stream = stream.unwrap_or(&device.stream);
    scan_device_buffers::<f64, f64>(
        stream,
        &set.partials,
        &set.fixup,
        &set,
        d_in,
        d_out,
        n_pairs,
        2,
    )
}

// ============================================================================
// EMA / Wilder smoothing on top of the affine scan
// ============================================================================

/// Derive the warmup period from an EMA alpha: `alpha = 2/(period+1)` =>
/// `period = 2/alpha - 1` (exact round-trip for alphas derived from integer
/// periods; rounded to the nearest integer otherwise).
fn ema_period_from_alpha(alpha: f32) -> usize {
    let p = (2.0 / alpha as f64) - 1.0;
    if !p.is_finite() {
        return 1;
    }
    p.round().max(1.0) as usize
}

/// Shared driver for Wilder/EMA smoothing. Pipeline (all device-side, no
/// host sync, CUDA-graph capturable):
///
/// 1. `scan_store_i32` writes the "no valid window" sentinel `n`
///    (Wilder only; EMA keeps the zero-initialized start index).
/// 2. `scan_first_valid_window_f32` atomicMin's the first index where
///    `period` consecutive finite values exist (Wilder only).
/// 3. `scan_recurrence_build_pairs_f32` maps inputs to `(m, c)` pairs with
///    the exact `(0, SMA)` seed.
/// 4. Affine scan (3-kernel design above), in place on the pairs buffer.
/// 5. `scan_recurrence_finalize_f32` extracts `y[i]` and applies the NaN
///    warmup prefix.
fn run_affine_recurrence_f32(
    device: &GpuDevice,
    stream: Option<&Arc<CudaStream>>,
    d_x: &CudaSlice<f32>,
    d_out: &CudaSlice<f32>,
    period: usize,
    alpha: f32,
    search_window: bool,
    check_finite: bool,
) -> Result<(), GpuError> {
    let n = d_x.len();
    let stream = stream.unwrap_or(&device.stream);
    let module = load_scan_module(device)?;

    let build_fn = get_function(&module, "scan_recurrence_build_pairs_f32")?;
    let finalize_fn = get_function(&module, "scan_recurrence_finalize_f32")?;
    let affine_set = ScanKernelSet::load(&module, "affine_f32")?;

    let n_i32 = n as i32;
    let period_i32 = period as i32;
    let check_i32 = check_finite as i32;

    // First-valid-window start index, kept device-resident so no host sync
    // is required. alloc_zeros leaves it at 0 for the EMA path.
    let d_start: CudaSlice<i32> = stream
        .alloc_zeros(1)
        .map_err(|e| GpuError::AllocationError(format!("scan start-index alloc failed: {:?}", e)))?;

    if search_window {
        let store_fn = get_function(&module, "scan_store_i32")?;
        let window_fn = get_function(&module, "scan_first_valid_window_f32")?;

        // Sentinel n => "no valid window" until atomicMin lowers it.
        let cfg_one = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut b = stream.launch_builder(&store_fn);
            b.arg(&d_start).arg(&n_i32);
            unsafe { b.launch(cfg_one) }.map_err(|e| launch_failed("scan_store_i32", e))?;
        }

        let candidates = n - period + 1;
        let cfg = elementwise_cfg(candidates)?;
        let mut b = stream.launch_builder(&window_fn);
        b.arg(d_x).arg(&n_i32).arg(&period_i32).arg(&d_start);
        unsafe { b.launch(cfg) }.map_err(|e| launch_failed("scan_first_valid_window_f32", e))?;
    }

    // (m, c) pairs: n float2 stored as 2n f32 words.
    let d_pairs: CudaSlice<f32> = stream
        .alloc_zeros(2 * n)
        .map_err(|e| GpuError::AllocationError(format!("scan pairs alloc failed: {:?}", e)))?;

    {
        let cfg = elementwise_cfg(n)?;
        let mut b = stream.launch_builder(&build_fn);
        b.arg(d_x)
            .arg(&d_pairs)
            .arg(&n_i32)
            .arg(&period_i32)
            .arg(&alpha)
            .arg(&d_start)
            .arg(&check_i32);
        unsafe { b.launch(cfg) }
            .map_err(|e| launch_failed("scan_recurrence_build_pairs_f32", e))?;
    }

    // Inclusive affine scan, in place on the pairs buffer.
    scan_device_buffers::<f32, f32>(
        stream,
        &affine_set.partials,
        &affine_set.fixup,
        &affine_set,
        &d_pairs,
        &d_pairs,
        n,
        2,
    )?;

    {
        let cfg = elementwise_cfg(n)?;
        let mut b = stream.launch_builder(&finalize_fn);
        b.arg(&d_pairs)
            .arg(d_out)
            .arg(&n_i32)
            .arg(&period_i32)
            .arg(&d_start);
        unsafe { b.launch(cfg) }.map_err(|e| launch_failed("scan_recurrence_finalize_f32", e))?;
    }
    Ok(())
}

/// Wilder's smoothing (RMA, `alpha = 1/period`) on a device-resident f32
/// buffer, replicating the **exact** semantics of
/// `rust/src/cpu/sequential.rs::wilders_smoothing_cpu`:
///
/// - First-valid-window search: handles NaN-prefixed inputs (e.g. ADX's DX
///   series); output is NaN before `start + period - 1`.
/// - SMA seed at the end of the first valid window, injected exactly via an
///   `(0, SMA)` affine pair (double-accumulated; not fed through the
///   recurrence operator like `rsi_fused.cu`'s buggy `alpha*SMA` seed).
/// - Non-finite inputs after the seed produce NaN that propagates to all
///   later outputs, exactly like the CPU recurrence.
/// - No valid window at all => entire output is NaN.
///
/// Runs entirely on `stream` with no host synchronization or transfers.
pub fn wilder_smooth_f32(
    device: &GpuDevice,
    stream: Option<&Arc<CudaStream>>,
    d_x: &CudaSlice<f32>,
    period: usize,
    d_out: &mut CudaSlice<f32>,
) -> Result<(), GpuError> {
    let n = d_x.len();
    if period == 0 {
        return Err(GpuError::InvalidParameterStatic(
            "Wilder's smoothing period must be >= 1",
        ));
    }
    if n < period {
        return Err(GpuError::InvalidParameter(format!(
            "Insufficient data: need at least {} elements, got {}",
            period, n
        )));
    }
    check_equal_lens(n, d_out.len())?;
    if n > i32::MAX as usize {
        return Err(GpuError::InvalidParameterStatic(
            "wilder_smooth_f32 supports at most i32::MAX elements",
        ));
    }

    let alpha = 1.0f32 / period as f32;
    run_affine_recurrence_f32(device, stream, d_x, d_out, period, alpha, true, true)
}

/// Exponential moving average on a device-resident f32 buffer, replicating
/// the semantics of `rust/src/cpu/sequential.rs::ema_cpu`: NaN warmup for the
/// first `period - 1` outputs, SMA seed at `period - 1`, then
/// `y[i] = alpha*x[i] + (1-alpha)*y[i-1]`.
///
/// The warmup period is derived from `alpha` as `period = 2/alpha - 1`
/// (exact for `alpha = 2/(period+1)`; rounded to the nearest integer for
/// other alphas).
///
/// Runs entirely on `stream` with no host synchronization or transfers.
pub fn ema_f32(
    device: &GpuDevice,
    stream: Option<&Arc<CudaStream>>,
    d_x: &CudaSlice<f32>,
    alpha: f32,
    d_out: &mut CudaSlice<f32>,
) -> Result<(), GpuError> {
    let n = d_x.len();
    if !(alpha > 0.0 && alpha <= 1.0) {
        return Err(GpuError::InvalidParameter(format!(
            "EMA alpha must be in (0, 1], got {}",
            alpha
        )));
    }
    let period = ema_period_from_alpha(alpha);
    if n < period {
        return Err(GpuError::InvalidParameter(format!(
            "Insufficient data: need at least {} elements (alpha {} => period {}), got {}",
            period, alpha, period, n
        )));
    }
    check_equal_lens(n, d_out.len())?;
    if n > i32::MAX as usize {
        return Err(GpuError::InvalidParameterStatic(
            "ema_f32 supports at most i32::MAX elements",
        ));
    }

    run_affine_recurrence_f32(device, stream, d_x, d_out, period, alpha, false, false)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::sequential::{ema_cpu, wilders_smoothing_cpu};
    use ndarray::Array1;

    // ------------------------------------------------------------------
    // Helpers (pure CPU)
    // ------------------------------------------------------------------

    fn xorshift32(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    /// Deterministic pseudo-random values in [0, 1).
    fn gen_unit_f64(n: usize, seed: u32) -> Vec<f64> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| xorshift32(&mut s) as f64 / (u32::MAX as f64 + 1.0))
            .collect()
    }

    fn assert_close(got: f64, want: f64, rel: f64, abs: f64, ctx: &str) {
        if want.is_nan() {
            assert!(got.is_nan(), "{}: expected NaN, got {}", ctx, got);
            return;
        }
        if want.is_infinite() {
            assert_eq!(got, want, "{}: expected {}, got {}", ctx, want, got);
            return;
        }
        let diff = (got - want).abs();
        let tol = abs + rel * want.abs();
        assert!(
            diff <= tol,
            "{}: got {}, want {}, |diff| {} > tol {}",
            ctx,
            got,
            want,
            diff,
            tol
        );
    }

    fn assert_vec_close(got: &[f64], want: &[f64], rel: f64, abs: f64, ctx: &str) {
        assert_eq!(got.len(), want.len(), "{}: length mismatch", ctx);
        for i in 0..got.len() {
            assert_close(got[i], want[i], rel, abs, &format!("{}[{}]", ctx, i));
        }
    }

    // ------------------------------------------------------------------
    // Host-side mirror of the GPU recurrence pipeline (f64). The mirror
    // follows kernels/scan.cu line-for-line so that the algorithm (window
    // search, pair construction, seed injection, finalize) is validated in
    // normal CI without a GPU. The GPU parity tests below then validate
    // only the CUDA translation.
    // ------------------------------------------------------------------

    fn host_first_valid_window(x: &[f64], period: usize) -> usize {
        let n = x.len();
        if n < period {
            return n;
        }
        for s in 0..=(n - period) {
            if x[s..s + period].iter().all(|v| v.is_finite()) {
                return s;
            }
        }
        n // sentinel: no valid window
    }

    fn host_build_pairs(
        x: &[f64],
        period: usize,
        alpha: f64,
        start: usize,
        check_finite: bool,
    ) -> Vec<(f64, f64)> {
        let n = x.len();
        let no_window = start + period > n; // mirrors `start > n - period`
        (0..n)
            .map(|i| {
                if no_window {
                    return AFFINE_IDENTITY;
                }
                let sma_idx = start + period - 1;
                if i < sma_idx {
                    AFFINE_IDENTITY
                } else if i == sma_idx {
                    let sum: f64 = x[start..start + period].iter().sum();
                    (0.0, sum / period as f64)
                } else {
                    let v = x[i];
                    let c = if check_finite && !v.is_finite() {
                        f64::NAN
                    } else {
                        alpha * v
                    };
                    (1.0 - alpha, c)
                }
            })
            .collect()
    }

    fn host_affine_scan(pairs: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let mut acc = AFFINE_IDENTITY;
        pairs
            .iter()
            .map(|&p| {
                acc = affine_compose(acc, p);
                acc
            })
            .collect()
    }

    fn host_finalize(scanned: &[(f64, f64)], period: usize, start: usize) -> Vec<f64> {
        let n = scanned.len();
        let no_window = start + period > n;
        (0..n)
            .map(|i| {
                if no_window {
                    return f64::NAN;
                }
                let sma_idx = start + period - 1;
                if i < sma_idx { f64::NAN } else { scanned[i].1 }
            })
            .collect()
    }

    fn host_wilder(x: &[f64], period: usize) -> Vec<f64> {
        let start = host_first_valid_window(x, period);
        let pairs = host_build_pairs(x, period, 1.0 / period as f64, start, true);
        host_finalize(&host_affine_scan(&pairs), period, start)
    }

    fn host_ema(x: &[f64], period: usize) -> Vec<f64> {
        let alpha = 2.0 / (period as f64 + 1.0);
        let pairs = host_build_pairs(x, period, alpha, 0, false);
        host_finalize(&host_affine_scan(&pairs), period, 0)
    }

    // ------------------------------------------------------------------
    // Kernel source contract tests (no GPU required)
    // ------------------------------------------------------------------

    #[test]
    fn test_scan_kernel_source_is_nvrtc_safe() {
        assert!(
            !SCAN_KERNELS_SRC.contains("#include"),
            "scan.cu must not use include directives (NVRTC constraint)"
        );
        // CUB was the invalid approach in rsi_fused.cu; the scan must not
        // depend on it.
        assert!(
            !SCAN_KERNELS_SRC.contains("cub::"),
            "scan.cu must not depend on CUB"
        );
    }

    #[test]
    fn test_scan_kernel_source_contains_all_entry_points() {
        let entry_points = [
            "scan_partials_sum_f32",
            "scan_aggregates_sum_f32",
            "scan_fixup_sum_f32",
            "scan_partials_sum_f64",
            "scan_aggregates_sum_f64",
            "scan_fixup_sum_f64",
            "scan_partials_max_f32",
            "scan_aggregates_max_f32",
            "scan_fixup_max_f32",
            "scan_partials_max_f64",
            "scan_aggregates_max_f64",
            "scan_fixup_max_f64",
            "scan_partials_pair_sum_f32",
            "scan_aggregates_pair_sum_f32",
            "scan_fixup_pair_sum_f32",
            "scan_partials_affine_f32",
            "scan_aggregates_affine_f32",
            "scan_fixup_affine_f32",
            "scan_partials_affine_f64",
            "scan_aggregates_affine_f64",
            "scan_fixup_affine_f64",
            "scan_partials_affine_f32_f64acc",
            "scan_fixup_affine_f32_f64acc",
            "scan_store_i32",
            "scan_first_valid_window_f32",
            "scan_recurrence_build_pairs_f32",
            "scan_recurrence_finalize_f32",
        ];
        for name in entry_points {
            let decl = format!("__global__ void {}(", name);
            assert!(
                SCAN_KERNELS_SRC.contains(&decl),
                "missing extern \"C\" entry point: {}",
                name
            );
        }
    }

    #[test]
    fn test_scan_kernel_layout_contract_matches_rust_consts() {
        // Layout contract mirrored between CUDA and Rust; a drift here would
        // silently corrupt every scan.
        assert!(SCAN_KERNELS_SRC.contains("#define SCAN_BLOCK_THREADS 256"));
        assert!(SCAN_KERNELS_SRC.contains("#define SCAN_ITEMS_PER_THREAD 4"));
        assert!(SCAN_KERNELS_SRC.contains("#define SCAN_TILE 1024"));
        assert_eq!(SCAN_BLOCK_THREADS, 256);
        assert_eq!(SCAN_ITEMS_PER_THREAD, 4);
        assert_eq!(SCAN_TILE, 1024);
        assert_eq!(SCAN_TILE, SCAN_BLOCK_THREADS * SCAN_ITEMS_PER_THREAD);
        // 8 warps of 32 threads per block.
        assert!(SCAN_KERNELS_SRC.contains("#define SCAN_WARPS 8"));
        assert_eq!(SCAN_BLOCK_THREADS / 32, 8);
    }

    // ------------------------------------------------------------------
    // Affine composition property tests (the math the scan relies on)
    // ------------------------------------------------------------------

    #[test]
    fn test_affine_identity_is_exact() {
        let samples = [
            (0.9, 1.5),
            (0.0, 42.0),
            (1.0, 0.0),
            (0.5, -3.25),
            (1.0 - 1.0 / 14.0, 0.07),
        ];
        for &p in &samples {
            // Identity must be exact (bit-for-bit) on both sides; the GPU
            // tile scan pads tail tiles with it.
            assert_eq!(affine_compose(AFFINE_IDENTITY, p), p);
            assert_eq!(affine_compose(p, AFFINE_IDENTITY), p);
        }
    }

    #[test]
    fn test_affine_compose_is_associative() {
        let mut seed = 0xC0FFEEu32;
        for trial in 0..1000 {
            // m in [0.5, 1), c in [-1, 1): numerically benign ranges that
            // match (1-alpha) factors of real smoothing periods.
            let mut next = |lo: f64, hi: f64| {
                lo + (hi - lo) * (xorshift32(&mut seed) as f64 / (u32::MAX as f64 + 1.0))
            };
            let a = (next(0.5, 1.0), next(-1.0, 1.0));
            let b = (next(0.5, 1.0), next(-1.0, 1.0));
            let c = (next(0.5, 1.0), next(-1.0, 1.0));

            let left = affine_compose(affine_compose(a, b), c);
            let right = affine_compose(a, affine_compose(b, c));
            assert_close(
                left.0,
                right.0,
                1e-12,
                1e-300,
                &format!("trial {} m-component", trial),
            );
            assert_close(
                left.1,
                right.1,
                1e-12,
                1e-12,
                &format!("trial {} c-component", trial),
            );
        }
    }

    #[test]
    fn test_affine_compose_seed_pair_severs_history() {
        // The (0, SMA) seed must make the result independent of everything
        // composed before it - the property that makes the SMA seed exact.
        let history = (0.123, 9.87);
        let seed = (0.0, 5.0);
        let composed = affine_compose(history, seed);
        assert_eq!(composed, (0.0, 5.0));
        // ...and NaN-free identity pairs before the seed keep it clean
        // (IEEE: 0 * NaN == NaN would otherwise poison it).
        let nan_history = (f64::NAN, f64::NAN);
        let poisoned = affine_compose(nan_history, seed);
        assert!(poisoned.0.is_nan() && poisoned.1.is_nan());
    }

    #[test]
    fn test_affine_tree_scan_matches_sequential() {
        // The parallel scan evaluates the composition in a different
        // association order than the sequential recurrence; associativity
        // guarantees the same result up to rounding.
        fn tree_scan(pairs: &mut [(f64, f64)]) {
            let n = pairs.len();
            if n <= 1 {
                return;
            }
            let mid = n / 2;
            let (l, r) = pairs.split_at_mut(mid);
            tree_scan(l);
            tree_scan(r);
            let lt = l[mid - 1];
            for v in r.iter_mut() {
                *v = affine_compose(lt, *v);
            }
        }

        for &n in &[1usize, 2, 3, 255, 256, 257, 1023, 1024, 1025, 4096] {
            let ms = gen_unit_f64(n, 7 + n as u32);
            let cs = gen_unit_f64(n, 11 + n as u32);
            let pairs: Vec<(f64, f64)> = (0..n)
                .map(|i| (0.5 + 0.5 * ms[i], 2.0 * cs[i] - 1.0))
                .collect();

            let sequential = host_affine_scan(&pairs);
            let mut tree = pairs.clone();
            tree_scan(&mut tree);

            for i in 0..n {
                assert_close(
                    tree[i].1,
                    sequential[i].1,
                    1e-12,
                    1e-12,
                    &format!("n={} index {}", n, i),
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Host pipeline vs CPU reference (normative semantics, no GPU needed)
    // ------------------------------------------------------------------

    #[test]
    fn test_host_wilder_pipeline_matches_cpu_reference() {
        for &(n, period) in &[(5usize, 3usize), (64, 14), (255, 14), (256, 7), (1023, 20)] {
            let x = gen_unit_f64(n, 1000 + n as u32);
            let expected = wilders_smoothing_cpu(&Array1::from_vec(x.clone()), period)
                .expect("CPU reference failed");
            let got = host_wilder(&x, period);
            assert_vec_close(
                &got,
                expected.as_slice().unwrap(),
                1e-12,
                1e-12,
                &format!("wilder n={} period={}", n, period),
            );
        }
    }

    #[test]
    fn test_host_wilder_pipeline_nan_prefix() {
        // ADX's DX series: NaN warmup prefix, then finite values.
        let n = 200;
        let period = 14;
        let mut x = gen_unit_f64(n, 42);
        for v in x.iter_mut().take(27) {
            *v = f64::NAN;
        }
        let expected =
            wilders_smoothing_cpu(&Array1::from_vec(x.clone()), period).expect("CPU failed");
        let got = host_wilder(&x, period);
        assert_vec_close(&got, expected.as_slice().unwrap(), 1e-12, 1e-12, "nan-prefix");
        // Sanity: warmup really is start + period - 1 = 27 + 13 = 40.
        assert!(got[39].is_nan());
        assert!(got[40].is_finite());
    }

    #[test]
    fn test_host_wilder_pipeline_interior_nan() {
        // A NaN after the seed poisons everything from that point on, exactly
        // like the CPU recurrence.
        let n = 100;
        let period = 5;
        let mut x = gen_unit_f64(n, 77);
        x[50] = f64::NAN;
        let expected =
            wilders_smoothing_cpu(&Array1::from_vec(x.clone()), period).expect("CPU failed");
        let got = host_wilder(&x, period);
        assert_vec_close(&got, expected.as_slice().unwrap(), 1e-12, 1e-12, "interior-nan");
        for i in 50..n {
            assert!(got[i].is_nan(), "index {} should be NaN", i);
        }
    }

    #[test]
    fn test_host_wilder_pipeline_no_valid_window() {
        let n = 32;
        let period = 8;
        // Every 4th value NaN => no window of 8 consecutive finite values.
        let mut x = gen_unit_f64(n, 5);
        for i in (0..n).step_by(4) {
            x[i] = f64::NAN;
        }
        let expected =
            wilders_smoothing_cpu(&Array1::from_vec(x.clone()), period).expect("CPU failed");
        let got = host_wilder(&x, period);
        assert_vec_close(&got, expected.as_slice().unwrap(), 1e-12, 1e-12, "no-window");
        assert!(got.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_host_wilder_pipeline_edge_periods() {
        // period == 1 (alpha == 1 => y[i] = x[i]) and period == n.
        let x = gen_unit_f64(64, 9);
        for &period in &[1usize, 64] {
            let expected =
                wilders_smoothing_cpu(&Array1::from_vec(x.clone()), period).expect("CPU failed");
            let got = host_wilder(&x, period);
            assert_vec_close(
                &got,
                expected.as_slice().unwrap(),
                1e-12,
                1e-12,
                &format!("period={}", period),
            );
        }
    }

    #[test]
    fn test_host_ema_pipeline_matches_cpu_reference() {
        for &(n, period) in &[(5usize, 3usize), (64, 12), (257, 26), (1025, 9)] {
            let x = gen_unit_f64(n, 2000 + n as u32);
            let expected =
                ema_cpu(&Array1::from_vec(x.clone()), period).expect("CPU reference failed");
            let got = host_ema(&x, period);
            assert_vec_close(
                &got,
                expected.as_slice().unwrap(),
                1e-12,
                1e-12,
                &format!("ema n={} period={}", n, period),
            );
        }
    }

    #[test]
    fn test_ema_period_from_alpha_roundtrip() {
        for period in 1usize..=500 {
            let alpha = (2.0 / (period as f64 + 1.0)) as f32;
            assert_eq!(
                ema_period_from_alpha(alpha),
                period,
                "alpha {} should round-trip to period {}",
                alpha,
                period
            );
        }
        // alpha = 1.0 => period 1 (y[i] = x[i]).
        assert_eq!(ema_period_from_alpha(1.0), 1);
    }

    #[test]
    fn test_num_tiles_boundaries() {
        assert_eq!(num_tiles(1), 1);
        assert_eq!(num_tiles(1023), 1);
        assert_eq!(num_tiles(1024), 1);
        assert_eq!(num_tiles(1025), 2);
        assert_eq!(num_tiles(2048), 2);
        assert_eq!(num_tiles(2049), 3);
        assert_eq!(num_tiles(1_000_000), 977);
        assert_eq!(num_tiles(1024 * 1024 + 1), 1025);
    }

    // ------------------------------------------------------------------
    // GPU parity tests (require a CUDA device; validated in the GPU phase)
    // ------------------------------------------------------------------

    /// Boundary sizes exercising single-tile, multi-tile and recursive
    /// (>1024 tiles) paths.
    const PARITY_SIZES: &[usize] = &[
        1, 2, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537, 1_000_000,
    ];

    fn gpu_or_skip() -> Option<GpuDevice> {
        match GpuDevice::new() {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("GPU not available, skipping test: {:?}", e);
                None
            }
        }
    }

    fn gen_unit_f32(n: usize, seed: u32) -> Vec<f32> {
        gen_unit_f64(n, seed).into_iter().map(|v| v as f32).collect()
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_scan_sum_f32_parity() {
        let Some(device) = gpu_or_skip() else { return };
        for &n in PARITY_SIZES {
            let x = gen_unit_f32(n, 100 + n as u32);
            let d_in = device.copy_to_device_f32(&x).unwrap();
            let mut d_out = device.allocate_device_buffer::<f32>(n).unwrap();
            inclusive_scan_f32(&device, None, &d_in, &mut d_out, ScanOp::Sum).unwrap();
            device.synchronize().unwrap();
            let got = device.copy_to_host_f32(&d_out).unwrap();

            let mut acc = 0.0f64;
            for i in 0..n {
                acc += x[i] as f64;
                assert_close(
                    got[i] as f64,
                    acc,
                    1e-5,
                    1e-6,
                    &format!("sum_f32 n={} index {}", n, i),
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_scan_sum_f64_parity() {
        let Some(device) = gpu_or_skip() else { return };
        for &n in PARITY_SIZES {
            let x = gen_unit_f64(n, 200 + n as u32);
            let d_in = device.copy_to_device(&x).unwrap();
            let mut d_out = device.alloc_buffer(n).unwrap();
            inclusive_scan_f64(&device, None, &d_in, &mut d_out, ScanOp::Sum).unwrap();
            device.synchronize().unwrap();
            let got = device.copy_to_host(&d_out).unwrap();

            let mut acc = 0.0f64;
            for i in 0..n {
                acc += x[i];
                assert_close(got[i], acc, 1e-12, 1e-14, &format!("sum_f64 n={} idx {}", n, i));
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_scan_max_f32_parity() {
        let Some(device) = gpu_or_skip() else { return };
        for &n in PARITY_SIZES {
            // Values both above and below zero so -inf identity is exercised.
            let x: Vec<f32> = gen_unit_f32(n, 300 + n as u32)
                .into_iter()
                .map(|v| v * 200.0 - 100.0)
                .collect();
            let d_in = device.copy_to_device_f32(&x).unwrap();
            let mut d_out = device.allocate_device_buffer::<f32>(n).unwrap();
            inclusive_scan_f32(&device, None, &d_in, &mut d_out, ScanOp::Max).unwrap();
            device.synchronize().unwrap();
            let got = device.copy_to_host_f32(&d_out).unwrap();

            // Prefix max over f32 is exact (comparisons only).
            let mut acc = f32::NEG_INFINITY;
            for i in 0..n {
                acc = acc.max(x[i]);
                assert_eq!(got[i], acc, "max_f32 n={} index {}", n, i);
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_scan_pair_sum_f32_parity() {
        let Some(device) = gpu_or_skip() else { return };
        for &n in &[1usize, 255, 256, 257, 1024, 1025, 65537, 500_000] {
            // Interleaved (tpv, vol)-style pairs.
            let buf = gen_unit_f32(2 * n, 400 + n as u32);
            let d_in = device.copy_to_device_f32(&buf).unwrap();
            let mut d_out = device.allocate_device_buffer::<f32>(2 * n).unwrap();
            inclusive_scan_pair_sum_f32(&device, None, &d_in, &mut d_out).unwrap();
            device.synchronize().unwrap();
            let got = device.copy_to_host_f32(&d_out).unwrap();

            let (mut ax, mut ay) = (0.0f64, 0.0f64);
            for i in 0..n {
                ax += buf[2 * i] as f64;
                ay += buf[2 * i + 1] as f64;
                assert_close(got[2 * i] as f64, ax, 1e-5, 1e-6, &format!("pair x n={} i={}", n, i));
                assert_close(
                    got[2 * i + 1] as f64,
                    ay,
                    1e-5,
                    1e-6,
                    &format!("pair y n={} i={}", n, i),
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_scan_affine_f32_parity() {
        let Some(device) = gpu_or_skip() else { return };
        for &precision in &[AffinePrecision::F32, AffinePrecision::F64Acc] {
            for &n in &[1usize, 257, 1024, 1025, 65537, 1_000_000] {
                // EMA-like pairs: m = 1 - alpha (alpha = 2/15), c = alpha*x.
                let alpha = 2.0f32 / 15.0;
                let xs = gen_unit_f32(n, 500 + n as u32);
                let mut buf = Vec::with_capacity(2 * n);
                for &x in &xs {
                    buf.push(1.0 - alpha);
                    buf.push(alpha * x);
                }
                let d_in = device.copy_to_device_f32(&buf).unwrap();
                let mut d_out = device.allocate_device_buffer::<f32>(2 * n).unwrap();
                inclusive_scan_affine_f32(&device, None, &d_in, &mut d_out, precision).unwrap();
                device.synchronize().unwrap();
                let got = device.copy_to_host_f32(&d_out).unwrap();

                // f64 sequential reference on the same f32-rounded pairs.
                let pairs: Vec<(f64, f64)> = (0..n)
                    .map(|i| (buf[2 * i] as f64, buf[2 * i + 1] as f64))
                    .collect();
                let want = host_affine_scan(&pairs);
                for i in 0..n {
                    assert_close(
                        got[2 * i + 1] as f64,
                        want[i].1,
                        1e-5,
                        1e-6,
                        &format!("affine {:?} n={} c[{}]", precision, n, i),
                    );
                }
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_scan_affine_f64_parity() {
        let Some(device) = gpu_or_skip() else { return };
        for &n in &[1usize, 1025, 65537] {
            let ms = gen_unit_f64(n, 600 + n as u32);
            let cs = gen_unit_f64(n, 601 + n as u32);
            let mut buf = Vec::with_capacity(2 * n);
            for i in 0..n {
                buf.push(0.5 + 0.5 * ms[i]);
                buf.push(2.0 * cs[i] - 1.0);
            }
            let d_in = device.copy_to_device(&buf).unwrap();
            let mut d_out = device.alloc_buffer(2 * n).unwrap();
            inclusive_scan_affine_f64(&device, None, &d_in, &mut d_out).unwrap();
            device.synchronize().unwrap();
            let got = device.copy_to_host(&d_out).unwrap();

            let pairs: Vec<(f64, f64)> = (0..n).map(|i| (buf[2 * i], buf[2 * i + 1])).collect();
            let want = host_affine_scan(&pairs);
            for i in 0..n {
                assert_close(
                    got[2 * i + 1],
                    want[i].1,
                    1e-10,
                    1e-12,
                    &format!("affine_f64 n={} c[{}]", n, i),
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_wilder_smooth_parity_boundary_sizes() {
        let Some(device) = gpu_or_skip() else { return };
        let period = 14;
        for &n in &[255usize, 256, 257, 1023, 1025, 65535, 65537, 1_000_000] {
            let x_f32 = gen_unit_f32(n, 700 + n as u32);
            // CPU reference computed on the same f32-rounded inputs so that
            // the comparison isolates algorithmic error from input
            // quantization.
            let x_f64: Vec<f64> = x_f32.iter().map(|&v| v as f64).collect();
            let expected =
                wilders_smoothing_cpu(&Array1::from_vec(x_f64), period).expect("CPU failed");

            let d_x = device.copy_to_device_f32(&x_f32).unwrap();
            let mut d_out = device.allocate_device_buffer::<f32>(n).unwrap();
            wilder_smooth_f32(&device, None, &d_x, period, &mut d_out).unwrap();
            device.synchronize().unwrap();
            let got = device.copy_to_host_f32(&d_out).unwrap();

            for i in 0..n {
                assert_close(
                    got[i] as f64,
                    expected[i],
                    1e-5,
                    1e-6,
                    &format!("wilder n={} index {}", n, i),
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_wilder_smooth_nan_prefix_parity() {
        let Some(device) = gpu_or_skip() else { return };
        let n = 5000;
        let period = 14;
        let mut x_f32 = gen_unit_f32(n, 800);
        for v in x_f32.iter_mut().take(27) {
            *v = f32::NAN;
        }
        let x_f64: Vec<f64> = x_f32.iter().map(|&v| v as f64).collect();
        let expected =
            wilders_smoothing_cpu(&Array1::from_vec(x_f64), period).expect("CPU failed");

        let d_x = device.copy_to_device_f32(&x_f32).unwrap();
        let mut d_out = device.allocate_device_buffer::<f32>(n).unwrap();
        wilder_smooth_f32(&device, None, &d_x, period, &mut d_out).unwrap();
        device.synchronize().unwrap();
        let got = device.copy_to_host_f32(&d_out).unwrap();

        for i in 0..n {
            assert_close(
                got[i] as f64,
                expected[i],
                1e-5,
                1e-6,
                &format!("wilder nan-prefix index {}", i),
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_ema_parity() {
        let Some(device) = gpu_or_skip() else { return };
        for &(n, period) in &[(255usize, 12usize), (1025, 26), (65537, 9), (1_000_000, 20)] {
            let alpha = (2.0 / (period as f64 + 1.0)) as f32;
            let x_f32 = gen_unit_f32(n, 900 + n as u32);
            let x_f64: Vec<f64> = x_f32.iter().map(|&v| v as f64).collect();
            let expected = ema_cpu(&Array1::from_vec(x_f64), period).expect("CPU failed");

            let d_x = device.copy_to_device_f32(&x_f32).unwrap();
            let mut d_out = device.allocate_device_buffer::<f32>(n).unwrap();
            ema_f32(&device, None, &d_x, alpha, &mut d_out).unwrap();
            device.synchronize().unwrap();
            let got = device.copy_to_host_f32(&d_out).unwrap();

            for i in 0..n {
                assert_close(
                    got[i] as f64,
                    expected[i],
                    1e-5,
                    1e-6,
                    &format!("ema n={} period={} index {}", n, period, i),
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_scan_validation_errors() {
        let Some(device) = gpu_or_skip() else { return };
        let x = gen_unit_f32(16, 1);
        let d_x = device.copy_to_device_f32(&x).unwrap();
        let mut d_out = device.allocate_device_buffer::<f32>(16).unwrap();
        let mut d_short = device.allocate_device_buffer::<f32>(8).unwrap();

        // period 0 / insufficient data / length mismatch.
        assert!(wilder_smooth_f32(&device, None, &d_x, 0, &mut d_out).is_err());
        assert!(wilder_smooth_f32(&device, None, &d_x, 17, &mut d_out).is_err());
        assert!(wilder_smooth_f32(&device, None, &d_x, 4, &mut d_short).is_err());

        // EMA alpha validation.
        assert!(ema_f32(&device, None, &d_x, 0.0, &mut d_out).is_err());
        assert!(ema_f32(&device, None, &d_x, 1.5, &mut d_out).is_err());
        assert!(ema_f32(&device, None, &d_x, f32::NAN, &mut d_out).is_err());

        // Pair buffers must have even length.
        let d_odd_in = device.copy_to_device_f32(&x[..15]).unwrap();
        let mut d_odd_out = device.allocate_device_buffer::<f32>(15).unwrap();
        assert!(inclusive_scan_pair_sum_f32(&device, None, &d_odd_in, &mut d_odd_out).is_err());
        assert!(
            inclusive_scan_affine_f32(
                &device,
                None,
                &d_odd_in,
                &mut d_odd_out,
                AffinePrecision::F32
            )
            .is_err()
        );
    }
}
