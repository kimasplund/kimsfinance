//! GPU-Accelerated Advanced Moving Averages (DEMA, TEMA, HMA, KAMA)
//!
//! # Parallelism Model (honest assessment)
//!
//! DEMA, TEMA and KAMA are EMA-recurrence-shaped: each output depends on the
//! previous output, so they are **inherently serial along the time axis** and
//! cannot be element-parallelized (see `gpu/ema.rs` for why a single-thread
//! GPU recurrence is an anti-pattern). The GPU win for this family comes from
//! the **batch dimension**: parameter sweeps and multi-series workloads run
//! one thread per `(series, parameter)` pair, with each thread executing the
//! sequential recurrence over its own series. With thousands of pairs the GPU
//! saturates and amortizes transfer/launch overhead.
//!
//! The single-series convenience wrappers (`dema_gpu`, `tema_gpu`, `kama_gpu`)
//! launch the batch kernel with one thread. They only win for very long
//! series; for a single short series prefer the CPU implementations in
//! `crate::indicators::moving_averages` / `moving_averages_advanced`.
//!
//! HMA is different: it decomposes into three windowed WMAs, which are
//! element-parallel. `hma_gpu` uses a fused kernel: each block stages the
//! intermediate series `2*WMA(period/2) - WMA(period)` in shared memory
//! (with a `sqrt(period)-1` halo) and then applies the final
//! `WMA(sqrt(period))` from shared memory.
//!
//! # Memory Layout Contract (mirrored in the CUDA source)
//!
//! ```text
//! prices: [num_series, series_len]               row-major, flattened
//! out:    [num_series * num_params, series_len]  row-major, flattened
//! output row index = series * num_params + param
//! ```
//!
//! # Precision (why f32 device math)
//!
//! All device arithmetic is f32. Ada Lovelace (sm_89, RTX 3500 Ada) executes
//! FP64 at 1/64 of the FP32 rate, so f64 device code would be catastrophically
//! slow. EMA-style recurrences are convex combinations (contractive), so f32
//! rounding does not amplify over long series; window sums (HMA, KAMA
//! volatility) are short enough that f32 keeps ~6 significant digits, which is
//! ample for price-scale data. The host API stays `f64` and converts at the
//! boundary, matching the established `gpu/*.rs` patterns.
//!
//! # Warmup / NaN Semantics
//!
//! * `DEMA(p)`: NaN for indices `< 2*(p-1)`. EMA1 seeded with `SMA(prices[0..p])`
//!   at index `p-1`; EMA2 seeded with the SMA of the first `p` *valid* EMA1
//!   values at index `2*(p-1)`. Requires `series_len >= 2*p` (matches CPU
//!   `DEMA::min_periods`).
//! * `TEMA(p)`: NaN for indices `< 3*(p-1)`, EMA3 seeded analogously at
//!   `3*(p-1)`. Requires `series_len >= 3*p`.
//! * `KAMA(p)`: NaN for indices `< p`, `out[p] = prices[p]`, recurrence from
//!   `p+1`. Matches `indicators::moving_averages_advanced::KAMA` exactly.
//! * `HMA(p)`: NaN for indices `< (p-1) + (floor(sqrt(p))-1)`. Matches
//!   `indicators::moving_averages::HMA` (whose intermediate-NaN windows
//!   propagate NaN until exactly that index).
//!
//! NOTE on DEMA/TEMA vs the CPU reference: the CPU implementations compose
//! `utils::ema` directly on the NaN-prefixed intermediate arrays, which poisons
//! the second/third SMA seed and propagates NaN through the entire output
//! (a latent CPU bug; its own tests only assert length). This module
//! implements the *intended* textbook cascade — each EMA stage is applied to
//! the valid suffix of the previous stage — which is what the CPU docs and
//! `min_periods` (2p / 3p) describe. GPU parity tests below validate against
//! a host-side reference with these semantics (anchored to `utils::ema`).

use super::device::{GpuDevice, GpuError};
use crate::gpu::compile::compile_ptx_optimized_cached;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Block size for the batch recurrence kernels (one long-running thread per
/// (series, param) pair; smaller blocks help the scheduler balance them).
const RECURRENCE_BLOCK_SIZE: u32 = 128;

/// Block size for the element-parallel HMA kernel.
const HMA_BLOCK_SIZE: u32 = 256;

/// CUDA kernel source for the advanced moving-average family.
///
/// NVRTC-compiled: no `#include` directives, `extern "C" __global__` entry
/// points only, and only NVRTC built-in intrinsics (`__int_as_float`,
/// `fabsf`). All device math is f32 (Ada FP64:FP32 = 1:64 — see module docs).
const MA_ADVANCED_KERNEL: &str = r#"
// f32 quiet NaN without header dependencies (NVRTC-safe)
#define CUDART_NAN_F __int_as_float(0x7fc00000)

// ---------------------------------------------------------------------------
// Layout contract (mirrored by the Rust wrappers in gpu/ma_advanced.rs):
//   prices: [num_series, series_len]               row-major, flattened
//   out:    [num_series * num_params, series_len]  row-major, flattened
//   output row index = series * num_params + param
// Batch kernels run ONE thread per (series, param) pair: the recurrences are
// serial in time, so parallelism comes from the batch dimension only.
// ---------------------------------------------------------------------------

// DEMA = 2*EMA1 - EMA2 with EMA2 = EMA(EMA1).
// EMA1 seeded with SMA(prices[0..p]) at index p-1; EMA2 seeded with the SMA of
// the first p valid EMA1 values at index 2*(p-1). Warmup NaN below 2*(p-1).
// Host guarantees: p >= 1 and series_len >= 2*p.
extern "C" __global__ void dema_batch_kernel(
    const float* __restrict__ prices,
    float* __restrict__ out,
    const int* __restrict__ periods,
    int num_series,
    int num_params,
    int series_len
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_series * num_params;
    if (t >= total) return;

    int series = t / num_params;
    int param = t - series * num_params;
    const float* x = prices + (long long)series * (long long)series_len;
    float* y = out + (long long)t * (long long)series_len;

    int p = periods[param];
    float alpha = 2.0f / ((float)p + 1.0f);
    float beta = 1.0f - alpha;

    // Seed EMA1 = SMA of the first p prices (value at index p-1)
    float sum1 = 0.0f;
    for (int i = 0; i < p; ++i) {
        sum1 += x[i];
    }
    float ema1 = sum1 / (float)p;

    // Seed EMA2 = SMA of EMA1 over indices [p-1, 2p-2]
    // (loop is empty when p == 1, leaving ema2 = ema1)
    float sum2 = ema1;
    for (int i = p; i <= 2 * (p - 1); ++i) {
        ema1 = alpha * x[i] + beta * ema1;
        sum2 += ema1;
    }
    float ema2 = sum2 / (float)p;

    int first_valid = 2 * (p - 1);
    for (int i = 0; i < first_valid; ++i) {
        y[i] = CUDART_NAN_F;
    }
    y[first_valid] = 2.0f * ema1 - ema2;

    for (int i = first_valid + 1; i < series_len; ++i) {
        ema1 = alpha * x[i] + beta * ema1;
        ema2 = alpha * ema1 + beta * ema2;
        y[i] = 2.0f * ema1 - ema2;
    }
}

// TEMA = 3*EMA1 - 3*EMA2 + EMA3 with EMA2 = EMA(EMA1), EMA3 = EMA(EMA2).
// Seeding cascades like DEMA; EMA3 seeded at index 3*(p-1).
// Host guarantees: p >= 1 and series_len >= 3*p.
extern "C" __global__ void tema_batch_kernel(
    const float* __restrict__ prices,
    float* __restrict__ out,
    const int* __restrict__ periods,
    int num_series,
    int num_params,
    int series_len
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_series * num_params;
    if (t >= total) return;

    int series = t / num_params;
    int param = t - series * num_params;
    const float* x = prices + (long long)series * (long long)series_len;
    float* y = out + (long long)t * (long long)series_len;

    int p = periods[param];
    float alpha = 2.0f / ((float)p + 1.0f);
    float beta = 1.0f - alpha;

    // Seed EMA1 = SMA of the first p prices (value at index p-1)
    float sum1 = 0.0f;
    for (int i = 0; i < p; ++i) {
        sum1 += x[i];
    }
    float ema1 = sum1 / (float)p;

    // Seed EMA2 = SMA of EMA1 over indices [p-1, 2p-2]
    float sum2 = ema1;
    for (int i = p; i <= 2 * (p - 1); ++i) {
        ema1 = alpha * x[i] + beta * ema1;
        sum2 += ema1;
    }
    float ema2 = sum2 / (float)p;

    // Seed EMA3 = SMA of EMA2 over indices [2p-2, 3p-3]
    float sum3 = ema2;
    for (int i = 2 * (p - 1) + 1; i <= 3 * (p - 1); ++i) {
        ema1 = alpha * x[i] + beta * ema1;
        ema2 = alpha * ema1 + beta * ema2;
        sum3 += ema2;
    }
    float ema3 = sum3 / (float)p;

    int first_valid = 3 * (p - 1);
    for (int i = 0; i < first_valid; ++i) {
        y[i] = CUDART_NAN_F;
    }
    y[first_valid] = 3.0f * ema1 - 3.0f * ema2 + ema3;

    for (int i = first_valid + 1; i < series_len; ++i) {
        ema1 = alpha * x[i] + beta * ema1;
        ema2 = alpha * ema1 + beta * ema2;
        ema3 = alpha * ema2 + beta * ema3;
        y[i] = 3.0f * ema1 - 3.0f * ema2 + ema3;
    }
}

// KAMA (Kaufman Adaptive Moving Average), matching the CPU reference
// (indicators::moving_averages_advanced::KAMA) exactly:
//   warmup:   out[0..p) = NaN, out[p] = prices[p]
//   ER       = |x[i] - x[i-p]| / sum_{j=i-p+1..i} |x[j] - x[j-1]|  (0 if denom == 0)
//   SC       = (ER * (fast_sc - slow_sc) + slow_sc)^2
//   out[i]   = out[i-1] + SC * (x[i] - out[i-1])
// The volatility window sum is recomputed per step (O(p)) like the CPU code:
// a rolling add/subtract would drift in f32 over long series.
// Host guarantees: p >= 1, fast/slow >= 1, fast < slow, series_len >= p + 1.
extern "C" __global__ void kama_batch_kernel(
    const float* __restrict__ prices,
    float* __restrict__ out,
    const int* __restrict__ er_periods,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int num_series,
    int num_params,
    int series_len
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_series * num_params;
    if (t >= total) return;

    int series = t / num_params;
    int param = t - series * num_params;
    const float* x = prices + (long long)series * (long long)series_len;
    float* y = out + (long long)t * (long long)series_len;

    int p = er_periods[param];
    float fast_sc = 2.0f / ((float)fast_periods[param] + 1.0f);
    float slow_sc = 2.0f / ((float)slow_periods[param] + 1.0f);

    for (int i = 0; i < p; ++i) {
        y[i] = CUDART_NAN_F;
    }

    float kama = x[p];
    y[p] = kama;

    for (int i = p + 1; i < series_len; ++i) {
        float direction = fabsf(x[i] - x[i - p]);

        float volatility = 0.0f;
        for (int j = i - p + 1; j <= i; ++j) {
            volatility += fabsf(x[j] - x[j - 1]);
        }

        float er = (volatility > 0.0f) ? (direction / volatility) : 0.0f;
        float sc = er * (fast_sc - slow_sc) + slow_sc;
        sc = sc * sc;

        kama = kama + sc * (x[i] - kama);
        y[i] = kama;
    }
}

// HMA = WMA(2*WMA(prices, period/2) - WMA(prices, period), floor(sqrt(period)))
// Element-parallel fused kernel. Each block stages the intermediate series
// diff[g] = 2*WMA_half(g) - WMA_full(g) in shared memory for global indices
// [block_first - (sqrt_period-1), block_first + blockDim.x) (left halo), then
// applies the final WMA(sqrt_period) from shared memory.
//
// Validity is decided BY INDEX (not by NaN arithmetic, which fast-math may
// break): diff is valid for g >= period-1; the final value is valid for
// idx >= (period-1) + (sqrt_period-1), matching the CPU NaN propagation.
//
// Dynamic shared memory: (blockDim.x + sqrt_period - 1) floats.
// Host guarantees: period >= 2, half_period = period/2 >= 1,
// sqrt_period = floor(sqrt(period)) >= 1, n >= period.
extern "C" __global__ void hma_kernel(
    const float* __restrict__ prices,
    float* __restrict__ out,
    int n,
    int period,
    int half_period,
    int sqrt_period
) {
    extern __shared__ float diff_tile[];

    int halo = sqrt_period - 1;
    int tile_size = blockDim.x + halo;
    int block_first = blockIdx.x * blockDim.x;

    // Stage 1: cooperatively compute the intermediate diff series for this
    // block's tile (including the left halo).
    for (int s = threadIdx.x; s < tile_size; s += blockDim.x) {
        int g = block_first - halo + s;
        float val = CUDART_NAN_F;
        if (g >= period - 1 && g < n) {
            // WMA(half_period) at g: newest weight = half_period, oldest = 1
            float wsum_h = 0.0f;
            for (int j = 0; j < half_period; ++j) {
                wsum_h += prices[g - j] * (float)(half_period - j);
            }
            float denom_h = (float)half_period * ((float)half_period + 1.0f) * 0.5f;

            // WMA(period) at g
            float wsum_f = 0.0f;
            for (int j = 0; j < period; ++j) {
                wsum_f += prices[g - j] * (float)(period - j);
            }
            float denom_f = (float)period * ((float)period + 1.0f) * 0.5f;

            val = 2.0f * (wsum_h / denom_h) - (wsum_f / denom_f);
        }
        diff_tile[s] = val;
    }
    __syncthreads();

    int idx = block_first + threadIdx.x;
    if (idx >= n) return;

    int first_valid = (period - 1) + (sqrt_period - 1);
    if (idx < first_valid) {
        out[idx] = CUDART_NAN_F;
        return;
    }

    // Stage 2: final WMA(sqrt_period) over diff from shared memory.
    // diff at global index (idx - j) lives at tile slot (threadIdx.x + halo - j);
    // all slots in the window are valid because idx >= first_valid.
    float wsum = 0.0f;
    for (int j = 0; j < sqrt_period; ++j) {
        wsum += diff_tile[threadIdx.x + halo - j] * (float)(sqrt_period - j);
    }
    float denom = (float)sqrt_period * ((float)sqrt_period + 1.0f) * 0.5f;
    out[idx] = wsum / denom;
}
"#;

/// KAMA parameter triple for one batch slot.
///
/// Mirrors `indicators::moving_averages_advanced::KAMA::new(period, fast, slow)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KamaParams {
    /// Efficiency-ratio lookback window (CPU `period`)
    pub er_period: usize,
    /// Fast smoothing-constant period (must be < `slow_period`)
    pub fast_period: usize,
    /// Slow smoothing-constant period
    pub slow_period: usize,
}

impl Default for KamaParams {
    /// Kaufman's conventional defaults: ER period 10, fast 2, slow 30.
    fn default() -> Self {
        Self {
            er_period: 10,
            fast_period: 2,
            slow_period: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// Validation helpers (pure host logic — unit-tested without a GPU)
// ---------------------------------------------------------------------------

/// Validate batch dimensions and return the total output element count.
///
/// Guards the `int` arithmetic inside the kernels (thread count and
/// `series_len` must fit in i32) and checks for `usize` overflow of the
/// flattened output buffer.
fn validate_batch_dims(
    num_series: usize,
    series_len: usize,
    num_params: usize,
) -> Result<usize, GpuError> {
    if num_series == 0 {
        return Err(GpuError::InvalidParameter(
            "prices must contain at least one series (got 0 rows)".to_string(),
        ));
    }
    if series_len == 0 {
        return Err(GpuError::InvalidParameter(
            "series length must be >= 1 (got empty series)".to_string(),
        ));
    }
    if num_params == 0 {
        return Err(GpuError::InvalidParameter(
            "parameter list must not be empty".to_string(),
        ));
    }
    if series_len > i32::MAX as usize {
        return Err(GpuError::InvalidParameter(format!(
            "series_len {} exceeds i32::MAX (kernel index limit)",
            series_len
        )));
    }
    let total_threads = num_series.checked_mul(num_params).ok_or_else(|| {
        GpuError::InvalidParameter("num_series * num_params overflows usize".to_string())
    })?;
    if total_threads > i32::MAX as usize {
        return Err(GpuError::InvalidParameter(format!(
            "num_series * num_params = {} exceeds i32::MAX (kernel index limit)",
            total_threads
        )));
    }
    total_threads.checked_mul(series_len).ok_or_else(|| {
        GpuError::InvalidParameter(
            "output buffer size (num_series * num_params * series_len) overflows usize".to_string(),
        )
    })
}

/// Validate periods for the EMA-cascade indicators.
///
/// `smoothing_passes` is 2 for DEMA and 3 for TEMA, matching the CPU
/// `min_periods` contracts (`2*period` / `3*period`).
fn validate_cascade_periods(
    periods: &[usize],
    series_len: usize,
    smoothing_passes: usize,
    indicator: &str,
) -> Result<(), GpuError> {
    if periods.is_empty() {
        return Err(GpuError::InvalidParameter(format!(
            "{}: periods list must not be empty",
            indicator
        )));
    }
    for &p in periods {
        if p < 1 {
            return Err(GpuError::InvalidParameter(format!(
                "{}: period must be >= 1",
                indicator
            )));
        }
        let min_len = p.checked_mul(smoothing_passes).ok_or_else(|| {
            GpuError::InvalidParameter(format!("{}: period {} overflows usize", indicator, p))
        })?;
        if series_len < min_len {
            return Err(GpuError::InvalidParameter(format!(
                "{}: not enough data: need >= {} points for period {}, got {}",
                indicator, min_len, p, series_len
            )));
        }
    }
    Ok(())
}

/// Validate KAMA parameter triples (mirrors CPU `KAMA::new` + `min_periods`).
fn validate_kama_params(params: &[KamaParams], series_len: usize) -> Result<(), GpuError> {
    if params.is_empty() {
        return Err(GpuError::InvalidParameter(
            "KAMA: parameter list must not be empty".to_string(),
        ));
    }
    for kp in params {
        if kp.er_period == 0 || kp.fast_period == 0 || kp.slow_period == 0 {
            return Err(GpuError::InvalidParameter(format!(
                "KAMA: all periods must be >= 1 (got {}/{}/{})",
                kp.er_period, kp.fast_period, kp.slow_period
            )));
        }
        if kp.fast_period >= kp.slow_period {
            return Err(GpuError::InvalidParameter(format!(
                "KAMA: fast_period ({}) must be < slow_period ({})",
                kp.fast_period, kp.slow_period
            )));
        }
        // CPU KAMA validates len >= period + 1
        if series_len < kp.er_period + 1 {
            return Err(GpuError::InvalidParameter(format!(
                "KAMA: not enough data: need >= {} points for er_period {}, got {}",
                kp.er_period + 1,
                kp.er_period,
                series_len
            )));
        }
    }
    Ok(())
}

/// Validate HMA parameters (mirrors CPU `HMA::new` + `min_periods`).
fn validate_hma_params(n: usize, period: usize) -> Result<(), GpuError> {
    if period < 2 {
        return Err(GpuError::InvalidParameter(format!(
            "HMA: period must be >= 2, got {}",
            period
        )));
    }
    if n < period {
        return Err(GpuError::InvalidParameter(format!(
            "HMA: not enough data: need >= {} points, got {}",
            period, n
        )));
    }
    if n > i32::MAX as usize {
        return Err(GpuError::InvalidParameter(format!(
            "HMA: input length {} exceeds i32::MAX (kernel index limit)",
            n
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Host <-> device plumbing
// ---------------------------------------------------------------------------

/// Compile (cached) and load one of the MA-advanced kernel functions.
///
/// The returned `CudaFunction` holds an `Arc` to its module, so it stays
/// valid after this helper returns.
fn load_kernel_function(device: &GpuDevice, name: &str) -> Result<CudaFunction, GpuError> {
    let ptx_arc = compile_ptx_optimized_cached(MA_ADVANCED_KERNEL).map_err(|e| {
        GpuError::CompilationError(format!("Failed to compile MA-advanced kernels: {:?}", e))
    })?;
    let ptx = Arc::unwrap_or_clone(ptx_arc);

    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX module: {:?}", e)))?;

    module.load_function(name).map_err(|e| {
        GpuError::ExecutionError(format!(
            "Failed to load kernel function '{}': {:?}",
            name, e
        ))
    })
}

/// Flatten an f64 price matrix to a row-major f32 vec (the device layout).
///
/// `ndarray` iteration is in logical (row-major) order regardless of the
/// underlying memory layout, so this also handles non-contiguous views.
fn flatten_prices_f32(prices: &Array2<f64>) -> Vec<f32> {
    prices.iter().map(|&v| v as f32).collect()
}

/// Convert the flattened f32 device output back to a row-major f64 matrix.
fn output_to_array2(out_f32: Vec<f32>, rows: usize, cols: usize) -> Result<Array2<f64>, GpuError> {
    let out_f64: Vec<f64> = out_f32.into_iter().map(|v| v as f64).collect();
    Array2::from_shape_vec((rows, cols), out_f64)
        .map_err(|e| GpuError::ComputationError(format!("Failed to reshape GPU output: {:?}", e)))
}

/// Shared launch path for the DEMA/TEMA batch kernels (identical signatures).
fn ema_cascade_batch_gpu(
    device: &GpuDevice,
    kernel_name: &str,
    smoothing_passes: usize,
    indicator: &str,
    prices: &Array2<f64>,
    periods: &[usize],
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array2<f64>, GpuError> {
    let (num_series, series_len) = prices.dim();
    let num_params = periods.len();

    let output_len = validate_batch_dims(num_series, series_len, num_params)?;
    validate_cascade_periods(periods, series_len, smoothing_passes, indicator)?;

    let kernel = load_kernel_function(device, kernel_name)?;
    let kernel_stream = stream.unwrap_or(&device.stream);

    // H2D: prices (f32) and periods (i32) on the selected stream.
    // The f64 pinned pool is type-specific, so f32 transfers use pageable
    // memory here; batch payloads are dominated by kernel time anyway.
    let prices_f32 = flatten_prices_f32(prices);
    let mut d_prices = device.allocate_device_buffer::<f32>(prices_f32.len())?;
    kernel_stream.memcpy_htod(&prices_f32, &mut d_prices)?;

    let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
    let mut d_periods = device.allocate_device_buffer::<i32>(periods_i32.len())?;
    kernel_stream.memcpy_htod(&periods_i32, &mut d_periods)?;

    let mut d_out = device.allocate_device_buffer::<f32>(output_len)?;

    let num_series_i32 = num_series as i32;
    let num_params_i32 = num_params as i32;
    let series_len_i32 = series_len as i32;

    let mut builder = kernel_stream.launch_builder(&kernel);
    builder.arg(&d_prices);
    builder.arg(&mut d_out);
    builder.arg(&d_periods);
    builder.arg(&num_series_i32);
    builder.arg(&num_params_i32);
    builder.arg(&series_len_i32);

    let total_threads = (num_series * num_params) as u32;
    let config = LaunchConfig {
        grid_dim: (total_threads.div_ceil(RECURRENCE_BLOCK_SIZE), 1, 1),
        block_dim: (RECURRENCE_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        builder.launch(config).map_err(|e| {
            GpuError::ExecutionError(format!("{} kernel launch failed: {:?}", indicator, e))
        })?;
    }

    // D2H + sync on the same stream before host access.
    let mut out_f32 = vec![0.0f32; output_len];
    kernel_stream.memcpy_dtoh(&d_out, &mut out_f32)?;
    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    output_to_array2(out_f32, num_series * num_params, series_len)
}

/// Wrap a single series as a 1-row batch matrix.
fn single_series_matrix(close: &Array1<f64>) -> Result<Array2<f64>, GpuError> {
    Array2::from_shape_vec((1, close.len()), close.to_vec()).map_err(|e| {
        GpuError::InvalidParameter(format!("Failed to view series as 1-row batch: {:?}", e))
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// GPU-accelerated batch DEMA (Double Exponential Moving Average).
///
/// `DEMA = 2*EMA(p) - EMA(EMA(p))`, computed for every `(series, period)`
/// pair with one GPU thread per pair (the recurrence is serial in time; see
/// module docs for the parallelism model).
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `prices` - `[num_series, series_len]` price matrix (row-major)
/// * `periods` - DEMA periods to evaluate against every series
/// * `stream` - Optional CUDA stream (None uses the device default)
///
/// # Returns
///
/// `[num_series * periods.len(), series_len]` matrix; row `s * periods.len() + k`
/// holds `DEMA(periods[k])` of series `s`. Indices `< 2*(period-1)` are NaN.
///
/// # Errors
///
/// Returns `GpuError::InvalidParameter` if the price matrix is empty, the
/// period list is empty, any period is 0, or any `series_len < 2*period`
/// (matching CPU `DEMA::min_periods`).
pub fn dema_batch_gpu(
    device: &GpuDevice,
    prices: &Array2<f64>,
    periods: &[usize],
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array2<f64>, GpuError> {
    ema_cascade_batch_gpu(
        device,
        "dema_batch_kernel",
        2,
        "DEMA",
        prices,
        periods,
        stream,
    )
}

/// GPU DEMA for a single series (convenience wrapper over [`dema_batch_gpu`]).
///
/// Runs the batch kernel with `num_series = 1` and a single period — i.e. a
/// single GPU thread. This only wins for very long series; for one short
/// series prefer the CPU `indicators::moving_averages::DEMA`.
pub fn dema_gpu(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let prices = single_series_matrix(close)?;
    let out = dema_batch_gpu(device, &prices, &[period], stream)?;
    Ok(out.row(0).to_owned())
}

/// GPU-accelerated batch TEMA (Triple Exponential Moving Average).
///
/// `TEMA = 3*EMA1 - 3*EMA2 + EMA3` with `EMA2 = EMA(EMA1)`, `EMA3 = EMA(EMA2)`,
/// computed for every `(series, period)` pair with one GPU thread per pair.
///
/// # Returns
///
/// `[num_series * periods.len(), series_len]` matrix; row `s * periods.len() + k`
/// holds `TEMA(periods[k])` of series `s`. Indices `< 3*(period-1)` are NaN.
///
/// # Errors
///
/// `GpuError::InvalidParameter` on empty inputs, zero periods, or any
/// `series_len < 3*period` (matching CPU `TEMA::min_periods`).
pub fn tema_batch_gpu(
    device: &GpuDevice,
    prices: &Array2<f64>,
    periods: &[usize],
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array2<f64>, GpuError> {
    ema_cascade_batch_gpu(
        device,
        "tema_batch_kernel",
        3,
        "TEMA",
        prices,
        periods,
        stream,
    )
}

/// GPU TEMA for a single series (convenience wrapper over [`tema_batch_gpu`]).
///
/// Single GPU thread — only wins for very long series; prefer the CPU
/// `indicators::moving_averages::TEMA` for one short series.
pub fn tema_gpu(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let prices = single_series_matrix(close)?;
    let out = tema_batch_gpu(device, &prices, &[period], stream)?;
    Ok(out.row(0).to_owned())
}

/// GPU-accelerated batch KAMA (Kaufman Adaptive Moving Average).
///
/// Matches `indicators::moving_averages_advanced::KAMA` numerics exactly:
/// efficiency ratio over `er_period`, smoothing constant
/// `(ER*(fast_sc-slow_sc)+slow_sc)^2`, adaptive recurrence seeded with
/// `prices[er_period]`. One GPU thread per `(series, params)` pair.
///
/// # Arguments
///
/// * `prices` - `[num_series, series_len]` price matrix (row-major)
/// * `params` - KAMA parameter triples to evaluate against every series
///
/// # Returns
///
/// `[num_series * params.len(), series_len]` matrix; row `s * params.len() + k`
/// holds `KAMA(params[k])` of series `s`. Indices `< er_period` are NaN and
/// index `er_period` equals the input price (CPU warmup convention).
///
/// # Errors
///
/// `GpuError::InvalidParameter` on empty inputs, zero periods,
/// `fast_period >= slow_period`, or `series_len < er_period + 1`.
pub fn kama_batch_gpu(
    device: &GpuDevice,
    prices: &Array2<f64>,
    params: &[KamaParams],
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array2<f64>, GpuError> {
    let (num_series, series_len) = prices.dim();
    let num_params = params.len();

    let output_len = validate_batch_dims(num_series, series_len, num_params)?;
    validate_kama_params(params, series_len)?;

    let kernel = load_kernel_function(device, "kama_batch_kernel")?;
    let kernel_stream = stream.unwrap_or(&device.stream);

    let prices_f32 = flatten_prices_f32(prices);
    let mut d_prices = device.allocate_device_buffer::<f32>(prices_f32.len())?;
    kernel_stream.memcpy_htod(&prices_f32, &mut d_prices)?;

    let er_i32: Vec<i32> = params.iter().map(|kp| kp.er_period as i32).collect();
    let fast_i32: Vec<i32> = params.iter().map(|kp| kp.fast_period as i32).collect();
    let slow_i32: Vec<i32> = params.iter().map(|kp| kp.slow_period as i32).collect();

    let mut d_er = device.allocate_device_buffer::<i32>(num_params)?;
    kernel_stream.memcpy_htod(&er_i32, &mut d_er)?;
    let mut d_fast = device.allocate_device_buffer::<i32>(num_params)?;
    kernel_stream.memcpy_htod(&fast_i32, &mut d_fast)?;
    let mut d_slow = device.allocate_device_buffer::<i32>(num_params)?;
    kernel_stream.memcpy_htod(&slow_i32, &mut d_slow)?;

    let mut d_out = device.allocate_device_buffer::<f32>(output_len)?;

    let num_series_i32 = num_series as i32;
    let num_params_i32 = num_params as i32;
    let series_len_i32 = series_len as i32;

    let mut builder = kernel_stream.launch_builder(&kernel);
    builder.arg(&d_prices);
    builder.arg(&mut d_out);
    builder.arg(&d_er);
    builder.arg(&d_fast);
    builder.arg(&d_slow);
    builder.arg(&num_series_i32);
    builder.arg(&num_params_i32);
    builder.arg(&series_len_i32);

    let total_threads = (num_series * num_params) as u32;
    let config = LaunchConfig {
        grid_dim: (total_threads.div_ceil(RECURRENCE_BLOCK_SIZE), 1, 1),
        block_dim: (RECURRENCE_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("KAMA kernel launch failed: {:?}", e)))?;
    }

    let mut out_f32 = vec![0.0f32; output_len];
    kernel_stream.memcpy_dtoh(&d_out, &mut out_f32)?;
    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    output_to_array2(out_f32, num_series * num_params, series_len)
}

/// GPU KAMA for a single series (convenience wrapper over [`kama_batch_gpu`]).
///
/// Single GPU thread — only wins for very long series; prefer the CPU
/// `indicators::moving_averages_advanced::KAMA` for one short series.
pub fn kama_gpu(
    device: &GpuDevice,
    close: &Array1<f64>,
    er_period: usize,
    fast_period: usize,
    slow_period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let prices = single_series_matrix(close)?;
    let params = [KamaParams {
        er_period,
        fast_period,
        slow_period,
    }];
    let out = kama_batch_gpu(device, &prices, &params, stream)?;
    Ok(out.row(0).to_owned())
}

/// GPU-accelerated HMA (Hull Moving Average) — element-parallel fused kernel.
///
/// `HMA = WMA(2*WMA(period/2) - WMA(period), floor(sqrt(period)))`, with the
/// intermediate series staged in shared memory per block (left halo of
/// `sqrt(period)-1` elements) so the final WMA reads from shared memory.
///
/// Matches `indicators::moving_averages::HMA`: integer `period/2` and
/// `floor(sqrt(period))` sub-periods, NaN for indices
/// `< (period-1) + (floor(sqrt(period))-1)`.
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `close` - Close prices
/// * `period` - HMA period (must be >= 2)
/// * `stream` - Optional CUDA stream (None uses the device default)
///
/// # Errors
///
/// `GpuError::InvalidParameter` if `period < 2` or `close.len() < period`.
pub fn hma_gpu(
    device: &GpuDevice,
    close: &Array1<f64>,
    period: usize,
    stream: Option<&Arc<CudaStream>>,
) -> Result<Array1<f64>, GpuError> {
    let n = close.len();
    validate_hma_params(n, period)?;

    // Mirror the CPU sub-period derivation exactly (integer floor semantics).
    let half_period = period / 2;
    let sqrt_period = (period as f64).sqrt() as usize;

    let kernel = load_kernel_function(device, "hma_kernel")?;
    let kernel_stream = stream.unwrap_or(&device.stream);

    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let mut d_close = device.allocate_device_buffer::<f32>(n)?;
    kernel_stream.memcpy_htod(&close_f32, &mut d_close)?;

    let mut d_out = device.allocate_device_buffer::<f32>(n)?;

    let n_i32 = n as i32;
    let period_i32 = period as i32;
    let half_i32 = half_period as i32;
    let sqrt_i32 = sqrt_period as i32;

    let mut builder = kernel_stream.launch_builder(&kernel);
    builder.arg(&d_close);
    builder.arg(&mut d_out);
    builder.arg(&n_i32);
    builder.arg(&period_i32);
    builder.arg(&half_i32);
    builder.arg(&sqrt_i32);

    // Tile = block + left halo of (sqrt_period - 1) intermediate values.
    let tile_elems = HMA_BLOCK_SIZE as usize + sqrt_period - 1;
    let shared_mem_bytes = (tile_elems * std::mem::size_of::<f32>()) as u32;

    let config = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(HMA_BLOCK_SIZE), 1, 1),
        block_dim: (HMA_BLOCK_SIZE, 1, 1),
        shared_mem_bytes,
    };

    unsafe {
        builder
            .launch(config)
            .map_err(|e| GpuError::ExecutionError(format!("HMA kernel launch failed: {:?}", e)))?;
    }

    let mut out_f32 = vec![0.0f32; n];
    kernel_stream.memcpy_dtoh(&d_out, &mut out_f32)?;
    kernel_stream.synchronize().map_err(|e| {
        GpuError::SynchronizationError(format!("Stream synchronization failed: {:?}", e))
    })?;

    Ok(Array1::from_vec(
        out_f32.into_iter().map(|v| v as f64).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    // -----------------------------------------------------------------------
    // Host-side tests (no GPU required)
    // -----------------------------------------------------------------------

    #[test]
    fn test_kernel_source_has_no_includes() {
        assert!(
            !MA_ADVANCED_KERNEL.contains("#include"),
            "NVRTC kernel source must not contain #include directives"
        );
    }

    #[test]
    fn test_kernel_source_has_expected_entry_points() {
        for entry in [
            "extern \"C\" __global__ void dema_batch_kernel",
            "extern \"C\" __global__ void tema_batch_kernel",
            "extern \"C\" __global__ void kama_batch_kernel",
            "extern \"C\" __global__ void hma_kernel",
        ] {
            assert!(
                MA_ADVANCED_KERNEL.contains(entry),
                "kernel source missing entry point: {}",
                entry
            );
        }
    }

    #[test]
    fn test_kernel_source_uses_f32_only() {
        // Ada (sm_89) FP64:FP32 throughput is 1:64 — device code must be f32.
        assert!(
            !MA_ADVANCED_KERNEL.contains("double"),
            "kernel source must not use f64 ('double') device math on Ada"
        );
    }

    #[test]
    fn test_validate_batch_dims() {
        assert!(validate_batch_dims(0, 100, 1).is_err(), "0 series");
        assert!(validate_batch_dims(1, 0, 1).is_err(), "empty series");
        assert!(validate_batch_dims(1, 100, 0).is_err(), "0 params");

        let output_len = validate_batch_dims(2, 100, 3).unwrap();
        assert_eq!(output_len, 600);
    }

    #[test]
    fn test_validate_cascade_periods_errors() {
        // Empty period list
        assert!(validate_cascade_periods(&[], 100, 2, "DEMA").is_err());
        // Zero period
        assert!(validate_cascade_periods(&[0], 100, 2, "DEMA").is_err());
        // DEMA needs series_len >= 2*period
        assert!(validate_cascade_periods(&[60], 100, 2, "DEMA").is_err());
        assert!(validate_cascade_periods(&[50], 100, 2, "DEMA").is_ok());
        // TEMA needs series_len >= 3*period
        assert!(validate_cascade_periods(&[40], 100, 3, "TEMA").is_err());
        assert!(validate_cascade_periods(&[33], 100, 3, "TEMA").is_ok());
        // One bad period poisons the whole batch
        assert!(validate_cascade_periods(&[10, 60], 100, 2, "DEMA").is_err());
    }

    #[test]
    fn test_validate_kama_params_errors() {
        let ok = KamaParams::default(); // 10 / 2 / 30
        assert!(validate_kama_params(&[], 100).is_err(), "empty params");
        assert!(validate_kama_params(&[ok], 100).is_ok());

        let zero_er = KamaParams { er_period: 0, ..ok };
        assert!(validate_kama_params(&[zero_er], 100).is_err());

        let fast_ge_slow = KamaParams {
            fast_period: 30,
            slow_period: 30,
            ..ok
        };
        assert!(validate_kama_params(&[fast_ge_slow], 100).is_err());

        // Needs series_len >= er_period + 1
        let long_er = KamaParams {
            er_period: 100,
            ..ok
        };
        assert!(validate_kama_params(&[long_er], 100).is_err());
        assert!(validate_kama_params(&[long_er], 101).is_ok());
    }

    #[test]
    fn test_validate_hma_params_errors() {
        assert!(validate_hma_params(100, 0).is_err(), "period 0");
        assert!(
            validate_hma_params(100, 1).is_err(),
            "period 1 (CPU requires >= 2)"
        );
        assert!(validate_hma_params(3, 4).is_err(), "window > series length");
        assert!(validate_hma_params(4, 4).is_ok());
    }

    // --- Host reference implementations for DEMA/TEMA parity tests ---------
    //
    // The CPU DEMA/TEMA compose utils::ema directly on NaN-prefixed arrays,
    // which poisons the second SMA seed and yields all-NaN output (latent CPU
    // bug — see module docs). The intended semantics apply each EMA stage to
    // the valid suffix of the previous stage; these references implement that,
    // anchored to utils::ema for the base EMA semantics.

    /// Mirror of `indicators::utils::ema` on a plain slice.
    fn ema_seq(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        if n < period {
            return result;
        }
        let alpha = 2.0 / (period as f64 + 1.0);
        let first_sum: f64 = data[0..period].iter().sum();
        result[period - 1] = first_sum / period as f64;
        for i in period..n {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }
        result
    }

    /// EMA applied to the valid (non-NaN) suffix, re-padded with NaN.
    fn ema_on_valid(data: &[f64], period: usize) -> Vec<f64> {
        let start = data.iter().position(|v| !v.is_nan()).unwrap_or(data.len());
        let mut out = vec![f64::NAN; data.len()];
        if data.len() - start >= period {
            let inner = ema_seq(&data[start..], period);
            for (k, v) in inner.into_iter().enumerate() {
                out[start + k] = v;
            }
        }
        out
    }

    fn dema_reference(prices: &[f64], period: usize) -> Vec<f64> {
        let ema1 = ema_seq(prices, period);
        let ema2 = ema_on_valid(&ema1, period);
        ema1.iter()
            .zip(ema2.iter())
            .map(|(&e1, &e2)| {
                if !e1.is_nan() && !e2.is_nan() {
                    2.0 * e1 - e2
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    fn tema_reference(prices: &[f64], period: usize) -> Vec<f64> {
        let ema1 = ema_seq(prices, period);
        let ema2 = ema_on_valid(&ema1, period);
        let ema3 = ema_on_valid(&ema2, period);
        (0..prices.len())
            .map(|i| {
                let (e1, e2, e3) = (ema1[i], ema2[i], ema3[i]);
                if !e1.is_nan() && !e2.is_nan() && !e3.is_nan() {
                    3.0 * e1 - 3.0 * e2 + e3
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Deterministic price-like test series (~100 +/- 13, no NaN).
    fn gen_series(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let x = i as f64;
                100.0 + 10.0 * (x * 0.05).sin() + 3.0 * (x * 0.013).cos()
            })
            .collect()
    }

    /// NaN-aware closeness assertion: NaN masks must match exactly; finite
    /// values must agree within `abs_tol + rel_tol * |expected|`.
    fn assert_series_close(actual: &[f64], expected: &[f64], rel_tol: f64, abs_tol: f64) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for i in 0..actual.len() {
            let (a, e) = (actual[i], expected[i]);
            if e.is_nan() {
                assert!(a.is_nan(), "index {}: expected NaN, got {}", i, a);
            } else {
                assert!(!a.is_nan(), "index {}: expected {}, got NaN", i, e);
                let tol = abs_tol + rel_tol * e.abs();
                assert!(
                    (a - e).abs() <= tol,
                    "index {}: gpu={} expected={} (tol={})",
                    i,
                    a,
                    e,
                    tol
                );
            }
        }
    }

    #[test]
    fn test_ema_seq_matches_indicators_utils_ema() {
        // Anchor the test reference to the normative CPU EMA semantics.
        let data = gen_series(200);
        let arr = Array1::from_vec(data.clone());
        let expected = crate::indicators::utils::ema(arr.view(), 14);
        let actual = ema_seq(&data, 14);
        assert_series_close(&actual, expected.as_slice().unwrap(), 0.0, 1e-12);
    }

    #[test]
    fn test_dema_reference_warmup_and_identity() {
        // period = 1: alpha = 1, every EMA stage is the identity => DEMA = price
        let data = gen_series(16);
        let dema1 = dema_reference(&data, 1);
        assert_series_close(&dema1, &data, 0.0, 1e-12);

        // period = 3: NaN strictly below index 2*(p-1) = 4, finite from there
        let dema3 = dema_reference(&data, 3);
        for (i, v) in dema3.iter().enumerate() {
            if i < 4 {
                assert!(v.is_nan(), "index {} should be warmup NaN", i);
            } else {
                assert!(v.is_finite(), "index {} should be finite", i);
            }
        }
    }

    #[test]
    fn test_tema_reference_warmup_and_identity() {
        let data = gen_series(20);
        let tema1 = tema_reference(&data, 1);
        assert_series_close(&tema1, &data, 0.0, 1e-12);

        // period = 3: NaN strictly below index 3*(p-1) = 6
        let tema3 = tema_reference(&data, 3);
        for (i, v) in tema3.iter().enumerate() {
            if i < 6 {
                assert!(v.is_nan(), "index {} should be warmup NaN", i);
            } else {
                assert!(v.is_finite(), "index {} should be finite", i);
            }
        }
    }

    // -----------------------------------------------------------------------
    // GPU-gated tests (parity vs reference / CPU implementations)
    // -----------------------------------------------------------------------
    //
    // f32 device math vs f64 reference: tolerances are rel 1e-3 / abs 1e-2 on
    // ~100-scale prices, far above accumulated f32 rounding for these stable
    // (contractive) recurrences but tight enough to catch semantic errors.

    const GPU_REL_TOL: f64 = 1e-3;
    const GPU_ABS_TOL: f64 = 1e-2;

    #[test]
    #[ignore] // Requires GPU
    fn test_dema_gpu_matches_reference() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let data = gen_series(2_000);
        let close = Array1::from_vec(data.clone());

        for &period in &[1, 5, 10, 20] {
            let gpu = dema_gpu(&device, &close, period, None).expect("DEMA GPU failed");
            let expected = dema_reference(&data, period);
            assert_series_close(gpu.as_slice().unwrap(), &expected, GPU_REL_TOL, GPU_ABS_TOL);
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_tema_gpu_matches_reference() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let data = gen_series(2_000);
        let close = Array1::from_vec(data.clone());

        for &period in &[1, 5, 10, 20] {
            let gpu = tema_gpu(&device, &close, period, None).expect("TEMA GPU failed");
            let expected = tema_reference(&data, period);
            assert_series_close(gpu.as_slice().unwrap(), &expected, GPU_REL_TOL, GPU_ABS_TOL);
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_kama_gpu_matches_cpu() {
        use crate::indicators::Indicator;
        use crate::indicators::moving_averages_advanced::KAMA;

        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let data = gen_series(2_000);
        let close = Array1::from_vec(data.clone());

        for &(er, fast, slow) in &[(10usize, 2usize, 30usize), (5, 2, 20), (21, 3, 40)] {
            let gpu = kama_gpu(&device, &close, er, fast, slow, None).expect("KAMA GPU failed");
            let cpu = KAMA::new(er, fast, slow)
                .unwrap()
                .calculate(close.view())
                .unwrap();
            assert_series_close(
                gpu.as_slice().unwrap(),
                cpu.as_slice().unwrap(),
                GPU_REL_TOL,
                GPU_ABS_TOL,
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_kama_gpu_warmup_convention() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let data = gen_series(100);
        let close = Array1::from_vec(data.clone());
        let er = 10;

        let gpu = kama_gpu(&device, &close, er, 2, 30, None).expect("KAMA GPU failed");
        for i in 0..er {
            assert!(gpu[i].is_nan(), "KAMA[{}] should be warmup NaN", i);
        }
        // CPU convention: first valid value is the raw price at index er_period
        assert!(
            (gpu[er] - data[er]).abs() <= GPU_ABS_TOL + GPU_REL_TOL * data[er].abs(),
            "KAMA[{}] should equal price {}, got {}",
            er,
            data[er],
            gpu[er]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_hma_gpu_matches_cpu() {
        use crate::indicators::Indicator;
        use crate::indicators::moving_averages::HMA;

        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let data = gen_series(1_000);
        let close = Array1::from_vec(data.clone());

        // Periods chosen to exercise odd/even period/2 and sqrt floor cases
        for &period in &[2usize, 4, 9, 16, 50] {
            let gpu = hma_gpu(&device, &close, period, None).expect("HMA GPU failed");
            let cpu = HMA::new(period).unwrap().calculate(close.view()).unwrap();
            assert_series_close(
                gpu.as_slice().unwrap(),
                cpu.as_slice().unwrap(),
                GPU_REL_TOL,
                GPU_ABS_TOL,
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_hma_gpu_block_boundaries() {
        use crate::indicators::Indicator;
        use crate::indicators::moving_averages::HMA;

        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Length straddles multiple 256-thread blocks to exercise the shared
        // memory halo at block boundaries.
        let data = gen_series(256 * 3 + 17);
        let close = Array1::from_vec(data.clone());
        let period = 21; // half = 10, sqrt = 4 -> halo of 3

        let gpu = hma_gpu(&device, &close, period, None).expect("HMA GPU failed");
        let cpu = HMA::new(period).unwrap().calculate(close.view()).unwrap();
        assert_series_close(
            gpu.as_slice().unwrap(),
            cpu.as_slice().unwrap(),
            GPU_REL_TOL,
            GPU_ABS_TOL,
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_dema_batch_gpu_matches_single_series_calls() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let series_len = 500;
        let s0 = gen_series(series_len);
        let s1: Vec<f64> = gen_series(series_len)
            .iter()
            .map(|v| v * 1.5 + 7.0)
            .collect();
        let periods = [5usize, 10, 20];

        let mut flat = s0.clone();
        flat.extend_from_slice(&s1);
        let prices = Array2::from_shape_vec((2, series_len), flat).unwrap();

        let batch = dema_batch_gpu(&device, &prices, &periods, None).expect("batch DEMA failed");
        assert_eq!(batch.dim(), (2 * periods.len(), series_len));

        for (s, series) in [&s0, &s1].iter().enumerate() {
            let close = Array1::from_vec((*series).clone());
            for (k, &period) in periods.iter().enumerate() {
                let single = dema_gpu(&device, &close, period, None).expect("single DEMA failed");
                let row = batch.row(s * periods.len() + k);
                // Same kernel, same inputs: results should agree to f32 noise
                assert_series_close(
                    row.as_slice().unwrap(),
                    single.as_slice().unwrap(),
                    0.0,
                    1e-6,
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_kama_batch_gpu_multi_param_layout() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let series_len = 400;
        let data = gen_series(series_len);
        let prices = Array2::from_shape_vec((1, series_len), data.clone()).unwrap();
        let close = Array1::from_vec(data);

        let params = [
            KamaParams::default(),
            KamaParams {
                er_period: 5,
                fast_period: 2,
                slow_period: 20,
            },
        ];

        let batch = kama_batch_gpu(&device, &prices, &params, None).expect("batch KAMA failed");
        assert_eq!(batch.dim(), (params.len(), series_len));

        for (k, kp) in params.iter().enumerate() {
            let single = kama_gpu(
                &device,
                &close,
                kp.er_period,
                kp.fast_period,
                kp.slow_period,
                None,
            )
            .expect("single KAMA failed");
            assert_series_close(
                batch.row(k).as_slice().unwrap(),
                single.as_slice().unwrap(),
                0.0,
                1e-6,
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_dema_gpu_invalid_inputs() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Zero period
        let close = arr1(&[100.0, 101.0, 102.0, 103.0]);
        assert!(dema_gpu(&device, &close, 0, None).is_err());

        // series_len < 2*period
        assert!(dema_gpu(&device, &close, 3, None).is_err());

        // Empty series
        let empty: Array1<f64> = arr1(&[]);
        assert!(dema_gpu(&device, &empty, 3, None).is_err());
    }
}
