/**
 * Fused GPU Kernels: Orderflow Feature Extraction + Signal Generation
 *
 * NVRTC-JIT-compiled at runtime (gpu/compile.rs, targets compute_89):
 * no header inclusion of any kind, extern "C" __global__ entry points only,
 * NVRTC built-in types/intrinsics exclusively.
 *
 * ## Normative reference
 *
 * rust/src/cpu/orderflow.rs is the numerical ground truth. Every feature
 * formula below mirrors `OrderflowBatchProcessor::extract_features` line by
 * line; GPU-vs-CPU parity is asserted by #[ignore]-gated tests in
 * rust/src/gpu/orderflow_batch.rs.
 *
 * ## Features (per tick, FP32)
 *
 * 0. buy_sell_imbalance       buy_vol / (buy_vol + sell_vol); 0.5 when total == 0
 * 1. volume_delta             buy_vol - sell_vol (per-tick, NOT cumulative)
 * 2. trade_intensity          volume / (max(ts[i] - ts[i-1], 1) ms as seconds)
 * 3. price_velocity           z-score of close over trailing WINDOW_SIZE window
 * 4. volume_velocity          z-score of volume over trailing WINDOW_SIZE window
 * 5. cumulative_volume_delta  inclusive prefix sum of volume_delta
 *
 * All device math is FP32: Ada (sm_89) runs FP64 at 1/64 the FP32 rate, and
 * the CPU reference itself accumulates in f32, so FP32 is simultaneously the
 * fast choice AND the parity-exact one. The only 64-bit arithmetic is the
 * i64 timestamp difference, which MUST happen before any float conversion
 * (millisecond epochs ~1.7e12 are far outside f32's 24-bit integer range).
 *
 * ## Execution plan (3 deterministic passes)
 *
 * Feature 5 is a global inclusive prefix sum, which a single kernel cannot
 * produce without grid-wide synchronization. A reviewable 3-pass scan is
 * used (no decoupled lookback; same house style as kernels/scan.cu):
 *
 *   1. orderflow_block_scan_kernel       block-local inclusive scan of
 *                                        (buy - sell), per-block totals out
 *   2. orderflow_scan_block_sums_kernel  single block converts per-block
 *                                        totals into exclusive block prefixes
 *   3. orderflow_features_signals_kernel computes all 6 features once per
 *      (or calibrate_feature_ranges_     tick (cumulative = block-local scan
 *       kernel)                          + block prefix) and fuses the
 *                                        per-strategy epilogue / reduction
 *
 * Passes 1 and 3 MUST use the same tick->block mapping: TICKS_PER_BLOCK
 * contiguous ticks per block, blockDim.x == TICKS_PER_BLOCK,
 * gridDim.x == ceil(num_ticks / TICKS_PER_BLOCK). The Rust wrapper
 * (orderflow_batch.rs) owns the launch configs and enforces this.
 *
 * ## Stage A: features once (chunk + halo)
 *
 * Each block cooperatively stages the close prices and volumes of its
 * TICKS_PER_BLOCK ticks plus a HALO = WINDOW_SIZE - 1 tick lookback in
 * shared memory; each thread then computes its OWN tick's trailing-window
 * statistics from the shared halo. (The previous kernel had warp lane 0
 * push only its own every-32nd strided ticks into a shared circular buffer,
 * with no loop-end __syncwarp and shared memory indexed by the GLOBAL
 * strategy index — strided-garbage windows, a data race, and an
 * out-of-bounds write for >10 strategies. All three are fixed by the
 * block-local chunk+halo layout.)
 *
 * Buy/sell volumes and timestamps are read straight from global memory:
 * each thread needs only its own tick's values (plus ts[i-1]), both reads
 * are coalesced, and there is no cross-thread reuse to justify staging.
 *
 * ## Stage B: per-strategy epilogue (fused)
 *
 * Signals: a handful of threshold compares per strategy on the in-register
 * features -> out_signals[num_strategies][num_ticks] (i8 -1/0/+1).
 *
 * Quantized features: the host deduplicates strategies that share identical
 * (mins, maxs) ranges into "range groups" (all 5 default StrategyConfigs
 * share one group); the kernel quantizes once per GROUP, not per strategy
 * -> out_features_q[num_groups][num_ticks * 6]. The host broadcasts group
 * rows back to the stable per-strategy output shape.
 *
 * ## INT8 convention (must match kernels/quantize_int8.cu / quantization.rs)
 *
 * code = clamp(roundf((value - min) * scale), 0, 255) with
 * scale = 255/(max - min) host-precomputed (0.0 for degenerate ranges
 * <= 1e-9, so the code collapses to 0 and dequantizes to min). Codes are
 * RAW 0-255 values stored through the char ABI via (char)(unsigned char):
 * a direct (char) cast compiles to cvt.rzi.s8.f32 and SATURATES at 127.
 *
 * ## Calibration
 *
 * calibrate_feature_ranges_kernel reuses the exact Stage A feature math
 * (same chunk + halo, same block mapping, same scan inputs) and reduces
 * per-feature min/max with warp shuffles plus one CAS-float atomic per
 * block per feature. out_mins/out_maxs MUST be initialized to +inf/-inf
 * via init_calibration_ranges_kernel first: zero-initialized buffers would
 * silently clamp mins <= 0 and maxs >= 0, which is wrong for 4 of the 6
 * features (volume_delta, both z-scores, and cumulative delta can be
 * entirely positive or entirely negative).
 */

// ============================================================================
// Constants — layout contract mirrored in rust/src/gpu/orderflow_batch.rs
// (host-side tests assert these #define lines verbatim)
// ============================================================================

#define WINDOW_SIZE 20
#define HALO (WINDOW_SIZE - 1)
#define NUM_FEATURES 6
#define TICKS_PER_BLOCK 256
#define NUM_WARPS (TICKS_PER_BLOCK / 32)
#define FULL_MASK 0xFFFFFFFFu

// Signal values (match cpu/orderflow.rs Signal)
#define SIGNAL_HOLD 0
#define SIGNAL_BUY 1
#define SIGNAL_SELL (-1)

// Strategy IDs (match cpu/orderflow.rs StrategyType discriminants)
#define STRATEGY_MOMENTUM 0
#define STRATEGY_MEAN_REVERSION 1
#define STRATEGY_BREAKOUT 2
#define STRATEGY_SCALPING 3
#define STRATEGY_TREND_FOLLOWING 4

// ============================================================================
// Small device helpers
// ============================================================================

/**
 * Finite test without header-only functions: true iff not NaN and
 * |v| <= FLT_MAX. Kept manual (no isfinite) for NVRTC portability — same
 * convention as kernels/scan.cu.
 */
__device__ inline bool is_finite_f32(float v) {
    return (v == v) && (fabsf(v) <= 3.402823466e38f);
}

/**
 * Atomic min for float via integer CAS (CUDA has no native float atomicMin).
 * Correct for all non-NaN inputs; callers filter NaN via is_finite_f32.
 */
__device__ inline void atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        int new_val = __float_as_int(fminf(__int_as_float(assumed), val));
        old = atomicCAS(address_as_int, assumed, new_val);
    } while (assumed != old);
}

/**
 * Atomic max for float via integer CAS (CUDA has no native float atomicMax).
 */
__device__ inline void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        int new_val = __float_as_int(fmaxf(__int_as_float(assumed), val));
        old = atomicCAS(address_as_int, assumed, new_val);
    } while (assumed != old);
}

// ============================================================================
// Block-wide inclusive scan (Kogge-Stone within warps + cross-warp carry)
// ============================================================================

/**
 * Inclusive prefix sum of one float per thread across the whole block.
 *
 * Requires blockDim.x == TICKS_PER_BLOCK. `warp_sums` is caller-provided
 * shared storage of NUM_WARPS floats. Contains two __syncthreads(): EVERY
 * thread of the block must reach this call (no early returns before it).
 *
 * Deterministic: the combination order is fixed by lane/warp indices, so
 * repeated runs are bit-identical. The host-side simulation in
 * orderflow_batch.rs tests mirrors exactly this order for review.
 */
__device__ inline float block_inclusive_scan(float value, float* warp_sums) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    // Warp-level Kogge-Stone inclusive scan
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(FULL_MASK, value, offset);
        if (lane >= offset) value += n;
    }

    // Last lane of each warp publishes its warp total
    if (lane == 31) warp_sums[warp] = value;
    __syncthreads();

    // First warp scans the warp totals (NUM_WARPS <= 32)
    if (warp == 0) {
        float ws = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 1; offset < NUM_WARPS; offset <<= 1) {
            float n = __shfl_up_sync(FULL_MASK, ws, offset);
            if (lane >= offset) ws += n;
        }
        if (lane < NUM_WARPS) warp_sums[lane] = ws;
    }
    __syncthreads();

    // Add the inclusive total of all preceding warps
    if (warp > 0) value += warp_sums[warp - 1];
    return value;
}

// ============================================================================
// Stage A: per-tick features (mirrors cpu/orderflow.rs extract_features)
// ============================================================================

/**
 * Compute all 6 features for `tick` (global index, caller guarantees
 * tick < num_ticks).
 *
 * `s_price` / `s_volume` are the block's shared staging arrays laid out as
 * [HALO lookback | TICKS_PER_BLOCK chunk]; `shared_idx` = HALO + threadIdx.x
 * points at this thread's own tick. Out-of-range lookback slots are staged
 * as zero but never read: window_len = min(tick + 1, WINDOW_SIZE) keeps the
 * window inside the valid global range, exactly like the CPU's growing
 * VecDeque during warmup.
 *
 * Summation order inside the window is oldest -> newest, matching the CPU's
 * VecDeque iteration, so f32 rounding tracks the reference up to FMA
 * contraction (covered by the 1e-4 parity tolerance).
 */
__device__ inline void compute_tick_features(
    int tick,
    int shared_idx,
    const float* s_price,
    const float* s_volume,
    const long long* __restrict__ timestamps,
    const float* __restrict__ buy_volumes,
    const float* __restrict__ sell_volumes,
    const float* __restrict__ cum_delta_partial,
    float block_prefix,
    float* feats // [NUM_FEATURES]
) {
    float price = s_price[shared_idx];
    float volume = s_volume[shared_idx];
    float buy_vol = buy_volumes[tick];
    float sell_vol = sell_volumes[tick];

    // Feature 0: buy/sell imbalance (CPU compares total_vol > 0.0 — an exact
    // zero test, not an epsilon test)
    float total_vol = buy_vol + sell_vol;
    feats[0] = (total_vol > 0.0f) ? (buy_vol / total_vol) : 0.5f;

    // Feature 1: per-tick volume delta (the CPU strategies consume THIS, not
    // the cumulative sum)
    float volume_delta = buy_vol - sell_vol;
    feats[1] = volume_delta;

    // Feature 2: trade intensity = volume per second. CPU initializes
    // prev_timestamp to timestamps[0], so tick 0 sees dt = max(0, 1) = 1 ms.
    // The difference is taken in i64 BEFORE the f32 conversion (ms epochs are
    // not representable in f32).
    long long ts = timestamps[tick];
    long long prev_ts = (tick > 0) ? timestamps[tick - 1] : ts;
    long long dt_ms = ts - prev_ts;
    if (dt_ms < 1) dt_ms = 1;
    float dt_sec = (float)dt_ms / 1000.0f;
    feats[2] = volume / dt_sec;

    // Features 3 & 4: z-scores over the trailing window. Population variance
    // (divide by len, not len - 1) and the std > 1e-6 guard mirror the CPU
    // window std_dev / extract_features.
    int window_len = min(tick + 1, WINDOW_SIZE);
    feats[3] = 0.0f;
    feats[4] = 0.0f;
    if (window_len >= 2) {
        int first = shared_idx - (window_len - 1);

        float psum = 0.0f;
        float vsum = 0.0f;
        for (int k = first; k <= shared_idx; k++) {
            psum += s_price[k];
            vsum += s_volume[k];
        }
        float pmean = psum / (float)window_len;
        float vmean = vsum / (float)window_len;

        float pvar = 0.0f;
        float vvar = 0.0f;
        for (int k = first; k <= shared_idx; k++) {
            float pd = s_price[k] - pmean;
            float vd = s_volume[k] - vmean;
            pvar += pd * pd;
            vvar += vd * vd;
        }
        float pstd = sqrtf(pvar / (float)window_len);
        float vstd = sqrtf(vvar / (float)window_len);

        if (pstd > 1e-6f) feats[3] = (price - pmean) / pstd;
        if (vstd > 1e-6f) feats[4] = (volume - vmean) / vstd;
    }

    // Feature 5: cumulative volume delta = block-local inclusive scan
    // (pass 1) + exclusive block prefix (pass 2). The previous kernel kept 32
    // disjoint per-lane partial sums here — never a true global running sum.
    feats[5] = cum_delta_partial[tick] + block_prefix;
}

// ============================================================================
// Stage B helpers: signals + quantization
// ============================================================================

/**
 * Threshold-compare signal generation; mirrors cpu/orderflow.rs
 * generate_signal exactly (same comparisons, same f32 literals).
 * Strategies consume features 0-3 only.
 */
__device__ inline char generate_signal(
    int strategy_id,
    float imbalance,
    float volume_delta,
    float trade_intensity,
    float price_velocity
) {
    switch (strategy_id) {
        case STRATEGY_MOMENTUM:
            if (imbalance > 0.6f && volume_delta > 1000.0f) return SIGNAL_BUY;
            if (imbalance < 0.4f && volume_delta < -1000.0f) return SIGNAL_SELL;
            return SIGNAL_HOLD;
        case STRATEGY_MEAN_REVERSION:
            if (imbalance < 0.4f && volume_delta < -1000.0f) return SIGNAL_BUY;
            if (imbalance > 0.6f && volume_delta > 1000.0f) return SIGNAL_SELL;
            return SIGNAL_HOLD;
        case STRATEGY_BREAKOUT:
            if (trade_intensity > 100.0f && price_velocity > 0.001f) return SIGNAL_BUY;
            if (trade_intensity > 100.0f && price_velocity < -0.001f) return SIGNAL_SELL;
            return SIGNAL_HOLD;
        case STRATEGY_SCALPING: {
            float abs_delta = fabsf(volume_delta);
            if (imbalance > 0.55f && abs_delta < 500.0f) return SIGNAL_BUY;
            if (imbalance < 0.45f && abs_delta < 500.0f) return SIGNAL_SELL;
            return SIGNAL_HOLD;
        }
        case STRATEGY_TREND_FOLLOWING:
            if (volume_delta > 5000.0f && price_velocity > 0.002f) return SIGNAL_BUY;
            if (volume_delta < -5000.0f && price_velocity < -0.002f) return SIGNAL_SELL;
            return SIGNAL_HOLD;
        default:
            return SIGNAL_HOLD;
    }
}

/**
 * Quantize one feature to a raw 0-255 code (char ABI).
 *
 * `scale` = 255/(max - min), host-precomputed; 0.0 for degenerate ranges so
 * the code collapses to 0 (dequantizes to min — same convention as
 * quantization.rs). roundf rounds half away from zero, matching Rust
 * f32::round on the CPU path. The cast goes through unsigned char to
 * preserve the raw bit pattern: a direct (char) cast saturates at 127.
 */
__device__ inline char quantize_feature_code(float value, float min_val, float scale) {
    float q = (value - min_val) * scale;
    q = fmaxf(0.0f, fminf(255.0f, roundf(q)));
    return (char)(unsigned char)q;
}

// ============================================================================
// Pass 1: block-local inclusive scan of per-tick volume delta
// ============================================================================

/**
 * cum_delta_partial[tick] = inclusive prefix sum of (buy - sell) WITHIN the
 * tick's block; block_sums[blockIdx.x] = block total.
 *
 * Launch: grid = ceil(num_ticks / TICKS_PER_BLOCK), block = TICKS_PER_BLOCK.
 * MUST use the same mapping as the pass-3 kernels that consume the outputs.
 */
extern "C" __global__ void orderflow_block_scan_kernel(
    const float* __restrict__ buy_volumes,
    const float* __restrict__ sell_volumes,
    float* __restrict__ cum_delta_partial, // [num_ticks]
    float* __restrict__ block_sums,        // [gridDim.x]
    int num_ticks
) {
    __shared__ float warp_sums[NUM_WARPS];

    int tick = blockIdx.x * TICKS_PER_BLOCK + threadIdx.x;

    // Threads past the end contribute 0 so the block total stays correct.
    // No early return: block_inclusive_scan synchronizes the whole block.
    float delta = (tick < num_ticks) ? (buy_volumes[tick] - sell_volumes[tick]) : 0.0f;

    float inclusive = block_inclusive_scan(delta, warp_sums);

    if (tick < num_ticks) {
        cum_delta_partial[tick] = inclusive;
    }
    if (threadIdx.x == TICKS_PER_BLOCK - 1) {
        // Last lane's inclusive value == chunk total (padding lanes added 0)
        block_sums[blockIdx.x] = inclusive;
    }
}

// ============================================================================
// Pass 2: exclusive scan of per-block totals (single block, deterministic)
// ============================================================================

/**
 * block_prefixes[b] = sum of block_sums[0..b) (exclusive prefix).
 *
 * Launch: grid = (1,1,1), block = TICKS_PER_BLOCK. One block walks the
 * totals in TICKS_PER_BLOCK-sized chunks with a running shared-memory carry:
 * even 100M ticks yield only ~390K totals (~1.5K chunk iterations),
 * negligible next to passes 1/3, and the fixed order keeps the result
 * bit-deterministic (no decoupled lookback).
 */
extern "C" __global__ void orderflow_scan_block_sums_kernel(
    const float* __restrict__ block_sums,  // [num_blocks]
    float* __restrict__ block_prefixes,    // [num_blocks]
    int num_blocks
) {
    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float s_inclusive[TICKS_PER_BLOCK];
    __shared__ float s_carry;

    if (threadIdx.x == 0) s_carry = 0.0f;
    __syncthreads();

    for (int base = 0; base < num_blocks; base += TICKS_PER_BLOCK) {
        int i = base + threadIdx.x;
        float v = (i < num_blocks) ? block_sums[i] : 0.0f;

        float inclusive = block_inclusive_scan(v, warp_sums);
        s_inclusive[threadIdx.x] = inclusive;
        __syncthreads();

        if (i < num_blocks) {
            // Exclusive prefix = inclusive total of the PREVIOUS lane. Exact:
            // subtracting v from this lane's inclusive value would
            // reintroduce f32 rounding into the composition.
            float prev_inclusive = (threadIdx.x > 0) ? s_inclusive[threadIdx.x - 1] : 0.0f;
            block_prefixes[i] = s_carry + prev_inclusive;
        }
        __syncthreads(); // all s_carry reads complete before the update below

        if (threadIdx.x == TICKS_PER_BLOCK - 1) {
            s_carry += inclusive; // chunk total
        }
        __syncthreads(); // carry visible before the next chunk reads it
    }
}

// ============================================================================
// Pass 3a: fused feature extraction + per-strategy signal generation
// ============================================================================

/**
 * Stage A: each block stages close/volume for its TICKS_PER_BLOCK ticks plus
 * a HALO lookback in shared memory; each thread computes its own tick's 6
 * features ONCE. Stage B: per-strategy threshold signals plus per-range-group
 * INT8 quantization, all from the in-register features (no per-strategy
 * feature recompute — ~num_strategies x less feature work than the previous
 * warp-per-strategy kernel).
 *
 * Launch: grid = ceil(num_ticks / TICKS_PER_BLOCK) (identical to pass 1),
 * block = TICKS_PER_BLOCK, dynamic shared mem = 0 (static arrays only).
 *
 * Outputs:
 * - out_features_f32 [num_ticks * 6]              raw FP32 features, written once
 * - out_features_q   [num_groups][num_ticks * 6]  raw 0-255 codes (char ABI)
 * - out_signals      [num_strategies][num_ticks]  -1 / 0 / +1
 */
extern "C" __global__ void orderflow_features_signals_kernel(
    const long long* __restrict__ timestamps,
    const float* __restrict__ close_prices,
    const float* __restrict__ volumes,
    const float* __restrict__ buy_volumes,
    const float* __restrict__ sell_volumes,
    const float* __restrict__ cum_delta_partial, // [num_ticks]   pass-1 output
    const float* __restrict__ block_prefixes,    // [gridDim.x]   pass-2 output
    const int* __restrict__ strategy_ids,        // [num_strategies]
    const float* __restrict__ group_mins,        // [num_groups * NUM_FEATURES]
    const float* __restrict__ group_scales,      // [num_groups * NUM_FEATURES]
    float* __restrict__ out_features_f32,        // [num_ticks * NUM_FEATURES]
    char* __restrict__ out_features_q,           // [num_groups * num_ticks * NUM_FEATURES]
    char* __restrict__ out_signals,              // [num_strategies * num_ticks]
    int num_strategies,
    int num_groups,
    int num_ticks
) {
    __shared__ float s_price[HALO + TICKS_PER_BLOCK];
    __shared__ float s_volume[HALO + TICKS_PER_BLOCK];

    int chunk_start = blockIdx.x * TICKS_PER_BLOCK;

    // Cooperative staging of chunk + lookback halo. Block-LOCAL indices only:
    // the previous kernel indexed shared memory by the global strategy index
    // and wrote out of bounds for >10 strategies.
    for (int s = threadIdx.x; s < TICKS_PER_BLOCK + HALO; s += TICKS_PER_BLOCK) {
        int g = chunk_start - HALO + s;
        bool in_range = (g >= 0) && (g < num_ticks);
        s_price[s] = in_range ? close_prices[g] : 0.0f;
        s_volume[s] = in_range ? volumes[g] : 0.0f;
    }
    __syncthreads();

    int tick = chunk_start + threadIdx.x;
    if (tick >= num_ticks) return; // safe: no barriers after this point

    float feats[NUM_FEATURES];
    compute_tick_features(
        tick, HALO + threadIdx.x, s_price, s_volume,
        timestamps, buy_volumes, sell_volumes,
        cum_delta_partial, block_prefixes[blockIdx.x], feats);

    // FP32 features, written exactly once per tick
    int f32_base = tick * NUM_FEATURES;
    #pragma unroll
    for (int f = 0; f < NUM_FEATURES; f++) {
        out_features_f32[f32_base + f] = feats[f];
    }

    // INT8 features once per unique quantization-range group (the host
    // deduplicates; all 5 default StrategyConfigs collapse to one group)
    for (int g = 0; g < num_groups; g++) {
        int q_base = (g * num_ticks + tick) * NUM_FEATURES;
        #pragma unroll
        for (int f = 0; f < NUM_FEATURES; f++) {
            out_features_q[q_base + f] = quantize_feature_code(
                feats[f],
                group_mins[g * NUM_FEATURES + f],
                group_scales[g * NUM_FEATURES + f]);
        }
    }

    // Signals: a handful of compares per strategy
    for (int s = 0; s < num_strategies; s++) {
        out_signals[s * num_ticks + tick] =
            generate_signal(strategy_ids[s], feats[0], feats[1], feats[2], feats[3]);
    }
}

// ============================================================================
// Calibration support
// ============================================================================

/**
 * Initialize calibration range buffers to +inf / -inf identity elements.
 *
 * Launch: grid = (1,1,1), block = (32,1,1) (num_features <= 32).
 */
extern "C" __global__ void init_calibration_ranges_kernel(
    float* __restrict__ out_mins,
    float* __restrict__ out_maxs,
    int num_features
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_features) {
        out_mins[i] = __uint_as_float(0x7f800000u); // +inf
        out_maxs[i] = __uint_as_float(0xff800000u); // -inf
    }
}

/**
 * Pass 3b: per-feature [min, max] calibration over REAL trailing windows.
 *
 * Identical Stage A feature math and block mapping as
 * orderflow_features_signals_kernel (consumes the same pass-1/2 scan
 * outputs), then block-reduces per-feature min/max with warp shuffles and
 * publishes one CAS-float atomic per block per feature.
 *
 * The previous version gave every thread a PRIVATE circular buffer with a
 * grid sized so each thread saw exactly one tick: every window had
 * count == 1 and time_delta == 0, so the reported ranges collapsed to [0, 0].
 *
 * Non-finite feature values are skipped, mirroring the CPU calibrate_ranges
 * `value.is_finite()` filter; the wrapper maps a still-infinite min/max
 * (feature never finite) to the CPU fallbacks 0.0 / 1.0.
 *
 * Launch: grid = ceil(num_ticks / TICKS_PER_BLOCK), block = TICKS_PER_BLOCK.
 * out_mins/out_maxs MUST be pre-initialized by init_calibration_ranges_kernel.
 */
extern "C" __global__ void calibrate_feature_ranges_kernel(
    const long long* __restrict__ timestamps,
    const float* __restrict__ close_prices,
    const float* __restrict__ volumes,
    const float* __restrict__ buy_volumes,
    const float* __restrict__ sell_volumes,
    const float* __restrict__ cum_delta_partial, // [num_ticks] pass-1 output
    const float* __restrict__ block_prefixes,    // [gridDim.x] pass-2 output
    float* __restrict__ out_mins,                // [NUM_FEATURES], pre-init +inf
    float* __restrict__ out_maxs,                // [NUM_FEATURES], pre-init -inf
    int num_ticks
) {
    __shared__ float s_price[HALO + TICKS_PER_BLOCK];
    __shared__ float s_volume[HALO + TICKS_PER_BLOCK];
    __shared__ float s_warp_mins[NUM_WARPS];
    __shared__ float s_warp_maxs[NUM_WARPS];

    int chunk_start = blockIdx.x * TICKS_PER_BLOCK;

    for (int s = threadIdx.x; s < TICKS_PER_BLOCK + HALO; s += TICKS_PER_BLOCK) {
        int g = chunk_start - HALO + s;
        bool in_range = (g >= 0) && (g < num_ticks);
        s_price[s] = in_range ? close_prices[g] : 0.0f;
        s_volume[s] = in_range ? volumes[g] : 0.0f;
    }
    __syncthreads();

    int tick = chunk_start + threadIdx.x;
    bool valid = (tick < num_ticks);
    // No early return: every thread participates in the reductions below.

    float feats[NUM_FEATURES] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (valid) {
        compute_tick_features(
            tick, HALO + threadIdx.x, s_price, s_volume,
            timestamps, buy_volumes, sell_volumes,
            cum_delta_partial, block_prefixes[blockIdx.x], feats);
    }

    const float POS_INF = __uint_as_float(0x7f800000u);
    const float NEG_INF = __uint_as_float(0xff800000u);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    for (int f = 0; f < NUM_FEATURES; f++) {
        // Invalid lanes and non-finite values contribute identity elements
        bool use = valid && is_finite_f32(feats[f]);
        float vmin = use ? feats[f] : POS_INF;
        float vmax = use ? feats[f] : NEG_INF;

        // Warp-level min/max reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            vmin = fminf(vmin, __shfl_down_sync(FULL_MASK, vmin, offset));
            vmax = fmaxf(vmax, __shfl_down_sync(FULL_MASK, vmax, offset));
        }
        if (lane == 0) {
            s_warp_mins[warp] = vmin;
            s_warp_maxs[warp] = vmax;
        }
        __syncthreads();

        // First warp reduces the per-warp partials, then one atomic per block
        if (warp == 0) {
            vmin = (lane < NUM_WARPS) ? s_warp_mins[lane] : POS_INF;
            vmax = (lane < NUM_WARPS) ? s_warp_maxs[lane] : NEG_INF;
            #pragma unroll
            for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
                vmin = fminf(vmin, __shfl_down_sync(FULL_MASK, vmin, offset));
                vmax = fmaxf(vmax, __shfl_down_sync(FULL_MASK, vmax, offset));
            }
            if (lane == 0) {
                if (vmin < POS_INF) atomicMinFloat(&out_mins[f], vmin);
                if (vmax > NEG_INF) atomicMaxFloat(&out_maxs[f], vmax);
            }
        }
        __syncthreads(); // s_warp_* are reused by the next feature iteration
    }
}
