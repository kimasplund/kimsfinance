// scan.cu - Deterministic 3-kernel inclusive-scan primitives (NVRTC-safe)
//
// Reusable GPU prefix-scan building blocks:
//
//   * prefix sum  (f32 / f64)
//   * prefix max  (f32 / f64)
//   * pair-wise prefix sum (float2: e.g. (typical_price*vol, vol) running
//     numerator/denominator pairs for VWAP / MFI consumers)
//   * affine linear-recurrence scan (float2 = (m, c) pairs) which parallelizes
//     first-order recurrences y[i] = m[i] * y[i-1] + c[i] (EMA / Wilder / RMA)
//
// Why a 3-kernel design (instead of single-pass decoupled lookback):
//   1. scan_partials_<op>   : each 256-thread block scans one 1024-element
//                             tile (4 items/thread) and emits its tile
//                             aggregate into a partials array.
//   2. scan_aggregates_<op> : a single block scans up to 1024 tile aggregates
//                             in place. The host wrapper recurses on the
//                             partials array when there are more than 1024
//                             tiles, so 1M+ tiles (1B+ elements) are
//                             supported.
//   3. scan_fixup_<op>      : combines the exclusive tile prefix into every
//                             element of tiles 1..N-1.
// The dataflow is fully deterministic and reviewable without a GPU: no
// spin-waits, no inter-block communication other than stream ordering
// between the three launches.
//
// Affine recurrence math (the key primitive for EMA / Wilder smoothing):
//   A first-order recurrence y[i] = m[i]*y[i-1] + c[i] is NOT a scan over the
//   raw values with the recurrence operator - that operator is not
//   associative (the flaw in rsi_fused.cu's CUB approach). It IS a scan over
//   affine-transform pairs under function composition:
//       (m1, c1) followed by (m2, c2)  =>  (m1*m2, m2*c1 + c2)
//   Composition of affine maps is associative, so any reduction tree yields
//   the same result (up to floating-point rounding). The inclusive scan at
//   index i yields (M, C) with y[i] = M * y[-1] + C; with no prior state
//   (y[-1] := 0, enforced exactly by an m=0 seed pair) y[i] = C.
//
//   Seeding: wilders_smoothing_cpu (rust/src/cpu/sequential.rs) seeds the
//   recurrence with the SMA of the first valid window. We inject the pair
//   (0, SMA) at the seed index: m=0 makes the seed value exact and severs all
//   dependence on earlier pairs. (rsi_fused.cu fed the SMA through the
//   recurrence operator, effectively seeding with alpha*SMA - wrong. This
//   implementation avoids that by construction.)
//
// Precision (Ada sm_89: FP64 throughput is 1/64 of FP32):
//   Kernels default to f32. For the affine scan the composed m-products are
//   products of (1-alpha) factors in [0,1) which decay geometrically, so
//   contributions from the distant past are heavily down-weighted and f32
//   rounding error self-heals instead of accumulating. For precision-critical
//   callers, scan_*_affine_f32_f64acc keeps float2 pairs in global memory but
//   performs all accumulation in double (scans are bandwidth-bound, so the
//   bounded FP64 ALU cost is acceptable), and scan_*_affine_f64 /
//   scan_*_sum_f64 provide full f64 paths. The single-thread SMA seed in
//   scan_recurrence_build_pairs_f32 also accumulates in double: it is one
//   thread doing `period` adds, so the FP64 cost is negligible while the seed
//   (whose weight persists as (1-alpha)^k) stays exact.
//
// NVRTC constraints honored: no include directives, only extern "C"
// __global__ entry points, only NVRTC built-in types/intrinsics
// (float2/double2 vector types, __shfl_up_sync, __int_as_float,
// __longlong_as_double, fmaxf/fmax/fabsf, atomicMin).
//
// Layout contract mirrored in rust/src/gpu/scan.rs (keep in sync):
//   SCAN_BLOCK_THREADS = 256, SCAN_ITEMS_PER_THREAD = 4, SCAN_TILE = 1024.

#define SCAN_BLOCK_THREADS 256
#define SCAN_ITEMS_PER_THREAD 4
#define SCAN_TILE 1024
#define SCAN_WARPS 8
#define SCAN_FULL_MASK 0xffffffffu

// IEEE-754 special values built without headers (NVRTC-safe).
#define SCAN_NAN_F32 (__int_as_float(0x7fc00000))
#define SCAN_NEG_INF_F32 (__int_as_float(0xff800000))
#define SCAN_NEG_INF_F64 (__longlong_as_double(0xfff0000000000000ULL))

//==============================================================================
// Warp shuffle helpers
//
// __shfl_up_sync natively supports 32/64-bit scalars; vector pairs are
// shuffled component-wise. Every overload must be executed by all lanes of
// the warp (callers keep the shuffle itself outside divergent branches).
//==============================================================================

__device__ __forceinline__ float shfl_up_val(float v, int delta) {
    return __shfl_up_sync(SCAN_FULL_MASK, v, delta);
}

__device__ __forceinline__ double shfl_up_val(double v, int delta) {
    return __shfl_up_sync(SCAN_FULL_MASK, v, delta);
}

__device__ __forceinline__ float2 shfl_up_val(float2 v, int delta) {
    float2 r;
    r.x = __shfl_up_sync(SCAN_FULL_MASK, v.x, delta);
    r.y = __shfl_up_sync(SCAN_FULL_MASK, v.y, delta);
    return r;
}

__device__ __forceinline__ double2 shfl_up_val(double2 v, int delta) {
    double2 r;
    r.x = __shfl_up_sync(SCAN_FULL_MASK, v.x, delta);
    r.y = __shfl_up_sync(SCAN_FULL_MASK, v.y, delta);
    return r;
}

//==============================================================================
// Scan operators
//
// Each operator defines:
//   T        - element type in global memory
//   Acc      - accumulator type (registers, shared memory, partials array)
//   identity - exact identity element: combine(identity, x) == x and
//              combine(x, identity) == x bit-for-bit for finite x
//   combine  - ASSOCIATIVE binary operator; `a` is the EARLIER prefix,
//              `b` the LATER one (order matters for the affine operator)
//   load / store - T <-> Acc conversion
//==============================================================================

struct SumF32 {
    typedef float T;
    typedef float Acc;
    static __device__ __forceinline__ Acc identity() { return 0.0f; }
    static __device__ __forceinline__ Acc combine(Acc a, Acc b) { return a + b; }
    static __device__ __forceinline__ Acc load(T v) { return v; }
    static __device__ __forceinline__ T store(Acc v) { return v; }
};

struct SumF64 {
    typedef double T;
    typedef double Acc;
    static __device__ __forceinline__ Acc identity() { return 0.0; }
    static __device__ __forceinline__ Acc combine(Acc a, Acc b) { return a + b; }
    static __device__ __forceinline__ Acc load(T v) { return v; }
    static __device__ __forceinline__ T store(Acc v) { return v; }
};

struct MaxF32 {
    typedef float T;
    typedef float Acc;
    static __device__ __forceinline__ Acc identity() { return SCAN_NEG_INF_F32; }
    static __device__ __forceinline__ Acc combine(Acc a, Acc b) { return fmaxf(a, b); }
    static __device__ __forceinline__ Acc load(T v) { return v; }
    static __device__ __forceinline__ T store(Acc v) { return v; }
};

struct MaxF64 {
    typedef double T;
    typedef double Acc;
    static __device__ __forceinline__ Acc identity() { return SCAN_NEG_INF_F64; }
    static __device__ __forceinline__ Acc combine(Acc a, Acc b) { return fmax(a, b); }
    static __device__ __forceinline__ Acc load(T v) { return v; }
    static __device__ __forceinline__ T store(Acc v) { return v; }
};

struct PairSumF32 {
    typedef float2 T;
    typedef float2 Acc;
    static __device__ __forceinline__ Acc identity() {
        Acc v; v.x = 0.0f; v.y = 0.0f; return v;
    }
    static __device__ __forceinline__ Acc combine(Acc a, Acc b) {
        Acc r; r.x = a.x + b.x; r.y = a.y + b.y; return r;
    }
    static __device__ __forceinline__ Acc load(T v) { return v; }
    static __device__ __forceinline__ T store(Acc v) { return v; }
};

// Affine composition: a = (Ma, Ca) is the earlier composition, b = (mb, cb)
// the later element. Result f(y) = mb*(Ma*y + Ca) + cb
//                                = (Ma*mb)*y + (mb*Ca + cb).
struct AffineF32 {
    typedef float2 T;
    typedef float2 Acc;
    static __device__ __forceinline__ Acc identity() {
        Acc v; v.x = 1.0f; v.y = 0.0f; return v;
    }
    static __device__ __forceinline__ Acc combine(Acc a, Acc b) {
        Acc r;
        r.x = a.x * b.x;
        r.y = b.x * a.y + b.y;
        return r;
    }
    static __device__ __forceinline__ Acc load(T v) { return v; }
    static __device__ __forceinline__ T store(Acc v) { return v; }
};

struct AffineF64 {
    typedef double2 T;
    typedef double2 Acc;
    static __device__ __forceinline__ Acc identity() {
        Acc v; v.x = 1.0; v.y = 0.0; return v;
    }
    static __device__ __forceinline__ Acc combine(Acc a, Acc b) {
        Acc r;
        r.x = a.x * b.x;
        r.y = b.x * a.y + b.y;
        return r;
    }
    static __device__ __forceinline__ Acc load(T v) { return v; }
    static __device__ __forceinline__ T store(Acc v) { return v; }
};

// f64-accumulation fallback: float2 pairs in global memory, double2 in
// registers/shared/partials. Used when callers need tighter error bounds
// than pure f32 (see precision note at the top of this file).
struct AffineF32F64Acc {
    typedef float2 T;
    typedef double2 Acc;
    static __device__ __forceinline__ Acc identity() {
        Acc v; v.x = 1.0; v.y = 0.0; return v;
    }
    static __device__ __forceinline__ Acc combine(Acc a, Acc b) {
        Acc r;
        r.x = a.x * b.x;
        r.y = b.x * a.y + b.y;
        return r;
    }
    static __device__ __forceinline__ Acc load(T v) {
        Acc r; r.x = (double)v.x; r.y = (double)v.y; return r;
    }
    static __device__ __forceinline__ T store(Acc v) {
        T r; r.x = (float)v.x; r.y = (float)v.y; return r;
    }
};

//==============================================================================
// Generic scan implementation (Kogge-Stone warp scan + warp-aggregate pass)
//==============================================================================

template <typename Op>
__device__ __forceinline__ typename Op::Acc warp_inclusive_scan(
    typename Op::Acc v,
    unsigned int lane
) {
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        // All lanes execute the shuffle (full mask); only lanes with a
        // predecessor at distance d apply the combine.
        typename Op::Acc up = shfl_up_val(v, d);
        if (lane >= (unsigned int)d) {
            v = Op::combine(up, v);
        }
    }
    return v;
}

// Phase 1 of the 3-kernel scan: inclusive scan of one 1024-element tile.
//
// When `partials` is non-null, partials[blockIdx.x] receives the tile
// aggregate (the combination of all valid items in the tile - out-of-range
// items load the exact identity, so partial tail tiles are handled).
//
// IN-PLACE SAFETY (in == out): every thread loads its 4 items in step (1),
// before the first __syncthreads(); writes happen only in step (5), after
// it. Tiles never overlap across blocks, so in-place operation is safe both
// within and across blocks. For this reason `in`/`out` are deliberately NOT
// __restrict__-qualified.
template <typename Op>
__device__ __forceinline__ void scan_tile_impl(
    const typename Op::T* in,
    typename Op::T* out,
    typename Op::Acc* partials,
    long long n
) {
    typedef typename Op::Acc Acc;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;

    const long long base =
        (long long)blockIdx.x * SCAN_TILE + (long long)tid * SCAN_ITEMS_PER_THREAD;

    // (1) Per-thread sequential inclusive scan over 4 consecutive items.
    Acc items[SCAN_ITEMS_PER_THREAD];
    #pragma unroll
    for (int k = 0; k < SCAN_ITEMS_PER_THREAD; ++k) {
        long long idx = base + k;
        Acc v = (idx < n) ? Op::load(in[idx]) : Op::identity();
        items[k] = (k == 0) ? v : Op::combine(items[k - 1], v);
    }

    // (2) Warp-level inclusive scan of per-thread totals.
    Acc warp_incl = warp_inclusive_scan<Op>(items[SCAN_ITEMS_PER_THREAD - 1], lane);

    // (3) Cross-warp pass: last lane of each warp publishes its warp total,
    //     then warp 0 scans the SCAN_WARPS totals.
    __shared__ Acc warp_aggs[SCAN_WARPS];
    if (lane == 31u) {
        warp_aggs[warp] = warp_incl;
    }
    __syncthreads();

    if (warp == 0u) {
        Acc v = (lane < SCAN_WARPS) ? warp_aggs[lane] : Op::identity();
        v = warp_inclusive_scan<Op>(v, lane);
        if (lane < SCAN_WARPS) {
            warp_aggs[lane] = v;
        }
    }
    __syncthreads();

    // (4) Exclusive prefix for this thread:
    //     (exclusive warp prefix) combined with (exclusive lane prefix).
    Acc up1 = shfl_up_val(warp_incl, 1);
    Acc lane_excl = (lane == 0u) ? Op::identity() : up1;
    Acc warp_prefix = (warp == 0u) ? Op::identity() : warp_aggs[warp - 1];
    Acc thread_excl = Op::combine(warp_prefix, lane_excl);

    // (5) Write inclusive results.
    #pragma unroll
    for (int k = 0; k < SCAN_ITEMS_PER_THREAD; ++k) {
        long long idx = base + k;
        if (idx < n) {
            out[idx] = Op::store(Op::combine(thread_excl, items[k]));
        }
    }

    // (6) Tile aggregate = inclusive scan over all warp totals -> last slot.
    if (partials != nullptr && tid == 0u) {
        partials[blockIdx.x] = warp_aggs[SCAN_WARPS - 1];
    }
}

// Phase 3 of the 3-kernel scan: combine the exclusive tile prefix
// (partials[blockIdx.x - 1], where `partials` now holds the INCLUSIVE scan
// of tile aggregates) into every element of this tile. Tile 0 has no prefix.
//
// Operand order matters for the affine operator: the prefix is EARLIER, so
// combine(prefix, element).
template <typename Op>
__device__ __forceinline__ void scan_fixup_impl(
    typename Op::T* out,
    const typename Op::Acc* partials,
    long long n
) {
    if (blockIdx.x == 0u) return;

    typename Op::Acc prefix = partials[blockIdx.x - 1];
    const long long base =
        (long long)blockIdx.x * SCAN_TILE + (long long)threadIdx.x * SCAN_ITEMS_PER_THREAD;

    #pragma unroll
    for (int k = 0; k < SCAN_ITEMS_PER_THREAD; ++k) {
        long long idx = base + k;
        if (idx < n) {
            out[idx] = Op::store(Op::combine(prefix, Op::load(out[idx])));
        }
    }
}

//==============================================================================
// extern "C" entry points
//
// Written out explicitly (no token-pasting macros) so every launchable name
// is greppable in this file and host-side tests can assert their presence.
//
// scan_partials_<op>(in, out, partials, n)  : grid = ceil(n / 1024) blocks
// scan_aggregates_<op>(data, n)             : grid = 1 block, n <= 1024,
//                                             in-place
// scan_fixup_<op>(out, partials, n)         : grid = ceil(n / 1024) blocks
//
// All blocks are SCAN_BLOCK_THREADS (256) threads; `n` counts LOGICAL
// elements (pairs count as one element for the float2/double2 operators).
//==============================================================================

// ---- sum f32 ----------------------------------------------------------------

extern "C" __global__ void scan_partials_sum_f32(
    const float* in, float* out, float* partials, long long n
) {
    scan_tile_impl<SumF32>(in, out, partials, n);
}

extern "C" __global__ void scan_aggregates_sum_f32(float* data, long long n) {
    scan_tile_impl<SumF32>(data, data, (float*)nullptr, n);
}

extern "C" __global__ void scan_fixup_sum_f32(
    float* out, const float* partials, long long n
) {
    scan_fixup_impl<SumF32>(out, partials, n);
}

// ---- sum f64 ----------------------------------------------------------------

extern "C" __global__ void scan_partials_sum_f64(
    const double* in, double* out, double* partials, long long n
) {
    scan_tile_impl<SumF64>(in, out, partials, n);
}

extern "C" __global__ void scan_aggregates_sum_f64(double* data, long long n) {
    scan_tile_impl<SumF64>(data, data, (double*)nullptr, n);
}

extern "C" __global__ void scan_fixup_sum_f64(
    double* out, const double* partials, long long n
) {
    scan_fixup_impl<SumF64>(out, partials, n);
}

// ---- max f32 ----------------------------------------------------------------

extern "C" __global__ void scan_partials_max_f32(
    const float* in, float* out, float* partials, long long n
) {
    scan_tile_impl<MaxF32>(in, out, partials, n);
}

extern "C" __global__ void scan_aggregates_max_f32(float* data, long long n) {
    scan_tile_impl<MaxF32>(data, data, (float*)nullptr, n);
}

extern "C" __global__ void scan_fixup_max_f32(
    float* out, const float* partials, long long n
) {
    scan_fixup_impl<MaxF32>(out, partials, n);
}

// ---- max f64 ----------------------------------------------------------------

extern "C" __global__ void scan_partials_max_f64(
    const double* in, double* out, double* partials, long long n
) {
    scan_tile_impl<MaxF64>(in, out, partials, n);
}

extern "C" __global__ void scan_aggregates_max_f64(double* data, long long n) {
    scan_tile_impl<MaxF64>(data, data, (double*)nullptr, n);
}

extern "C" __global__ void scan_fixup_max_f64(
    double* out, const double* partials, long long n
) {
    scan_fixup_impl<MaxF64>(out, partials, n);
}

// ---- pair sum f32 (float2) --------------------------------------------------

extern "C" __global__ void scan_partials_pair_sum_f32(
    const float2* in, float2* out, float2* partials, long long n
) {
    scan_tile_impl<PairSumF32>(in, out, partials, n);
}

extern "C" __global__ void scan_aggregates_pair_sum_f32(float2* data, long long n) {
    scan_tile_impl<PairSumF32>(data, data, (float2*)nullptr, n);
}

extern "C" __global__ void scan_fixup_pair_sum_f32(
    float2* out, const float2* partials, long long n
) {
    scan_fixup_impl<PairSumF32>(out, partials, n);
}

// ---- affine f32 (float2 = (m, c) pairs) -------------------------------------

extern "C" __global__ void scan_partials_affine_f32(
    const float2* in, float2* out, float2* partials, long long n
) {
    scan_tile_impl<AffineF32>(in, out, partials, n);
}

extern "C" __global__ void scan_aggregates_affine_f32(float2* data, long long n) {
    scan_tile_impl<AffineF32>(data, data, (float2*)nullptr, n);
}

extern "C" __global__ void scan_fixup_affine_f32(
    float2* out, const float2* partials, long long n
) {
    scan_fixup_impl<AffineF32>(out, partials, n);
}

// ---- affine f64 (double2 pairs; also the recursion type for f64acc) ---------

extern "C" __global__ void scan_partials_affine_f64(
    const double2* in, double2* out, double2* partials, long long n
) {
    scan_tile_impl<AffineF64>(in, out, partials, n);
}

extern "C" __global__ void scan_aggregates_affine_f64(double2* data, long long n) {
    scan_tile_impl<AffineF64>(data, data, (double2*)nullptr, n);
}

extern "C" __global__ void scan_fixup_affine_f64(
    double2* out, const double2* partials, long long n
) {
    scan_fixup_impl<AffineF64>(out, partials, n);
}

// ---- affine f32 with f64 accumulation (fallback precision variant) ----------
//
// Level-0 kernels only: partials are double2, so the host wrapper scans the
// partials array with the affine_f64 kernel set above. No aggregates variant
// is needed for this operator.

extern "C" __global__ void scan_partials_affine_f32_f64acc(
    const float2* in, float2* out, double2* partials, long long n
) {
    scan_tile_impl<AffineF32F64Acc>(in, out, partials, n);
}

extern "C" __global__ void scan_fixup_affine_f32_f64acc(
    float2* out, const double2* partials, long long n
) {
    scan_fixup_impl<AffineF32F64Acc>(out, partials, n);
}

//==============================================================================
// First-order recurrence helpers (EMA / Wilder smoothing)
//
// Pipeline driven by rust/src/gpu/scan.rs::wilder_smooth_f32 / ema_f32:
//   scan_store_i32              (Wilder only: write "no window" sentinel n)
//   scan_first_valid_window_f32 (Wilder only: atomicMin over valid starts)
//   scan_recurrence_build_pairs_f32
//   scan_partials/aggregates/fixup_affine_f32 (the scan above)
//   scan_recurrence_finalize_f32
//
// The start index never leaves the device, so the whole pipeline runs
// without host synchronization and is CUDA-graph capturable.
//==============================================================================

// Matches Rust f64::is_finite semantics in f32: rejects NaN (v != v) and
// +/-inf (|v| > FLT_MAX). Kept manual (no isfinite) for NVRTC portability;
// the compile options do not enable finite-math-only, so the NaN comparison
// is not folded away.
__device__ __forceinline__ int scan_is_finite_f32(float v) {
    return (v == v) && (fabsf(v) <= 3.402823466e+38f);
}

// Single-thread scalar store, used to initialize the first-valid-window
// sentinel on-device (avoids a host->device memcpy that would break CUDA
// graph capture). Launch with 1 block x 1 thread.
extern "C" __global__ void scan_store_i32(int* dst, int value) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        *dst = value;
    }
}

// Finds min{s : x[s .. s+period) all finite} via atomicMin, replicating the
// first-valid-window search of wilders_smoothing_cpu (handles NaN-prefixed
// inputs such as ADX's DX series). `first_start` must be pre-initialized to
// the sentinel n (any value > n - period means "no valid window").
//
// Cost: O(n * period) reads worst case, but each thread early-exits at its
// first non-finite value and neighboring threads read overlapping windows
// (L1/L2 resident), so the practical cost is ~O(n) memory traffic.
extern "C" __global__ void scan_first_valid_window_f32(
    const float* x, int n, int period, int* first_start
) {
    long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (s > (long long)(n - period)) return;

    const int si = (int)s;
    for (int k = 0; k < period; ++k) {
        if (!scan_is_finite_f32(x[si + k])) return;
    }
    atomicMin(first_start, si);
}

// Builds affine pairs (m, c) such that y[i] = m*y[i-1] + c reproduces the
// EXACT warmup semantics of rust/src/cpu/sequential.rs::wilders_smoothing_cpu
// (check_finite=1) and ema_cpu (check_finite=0, *first_start == 0):
//
//   start = *first_start, sma_idx = start + period - 1
//   no valid window (start > n-period): all pairs identity; finalize emits NaN
//   i <  sma_idx : identity (1, 0)  - kept exactly NaN-free so the m=0 seed
//                  below is not poisoned (IEEE: 0 * NaN == NaN); outputs here
//                  are overwritten to NaN by the finalize kernel
//   i == sma_idx : (0, SMA(x[start .. start+period)))  - m=0 makes the seed
//                  exact and independent of earlier pairs
//   i >  sma_idx : (1-alpha, alpha*x[i]); with check_finite, a non-finite
//                  x[i] injects c=NaN, which propagates through every later
//                  prefix exactly like the CPU recurrence does
extern "C" __global__ void scan_recurrence_build_pairs_f32(
    const float* x,
    float2* pairs,
    int n,
    int period,
    float alpha,
    const int* first_start,
    int check_finite
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n) return;

    const int start = *first_start;

    float2 p;
    p.x = 1.0f;
    p.y = 0.0f;

    if (start > n - period) {
        // No valid window: identity pairs; finalize writes NaN everywhere.
        pairs[i] = p;
        return;
    }

    const long long sma_idx = (long long)start + period - 1;
    if (i < sma_idx) {
        // identity (set above)
    } else if (i == sma_idx) {
        // Single thread computes the SMA seed; double accumulation keeps the
        // seed exact (one thread x `period` adds - FP64 cost negligible).
        double sum = 0.0;
        for (int k = 0; k < period; ++k) {
            sum += (double)x[start + k];
        }
        p.x = 0.0f;
        p.y = (float)(sum / (double)period);
    } else {
        const float v = x[i];
        p.x = 1.0f - alpha;
        p.y = (check_finite != 0 && !scan_is_finite_f32(v)) ? SCAN_NAN_F32
                                                            : alpha * v;
    }
    pairs[i] = p;
}

// Extracts y[i] from the scanned pairs (y[i] = C component) and applies the
// NaN warmup prefix, matching wilders_smoothing_cpu / ema_cpu output layout.
extern "C" __global__ void scan_recurrence_finalize_f32(
    const float2* pairs,
    float* out,
    int n,
    int period,
    const int* first_start
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n) return;

    const int start = *first_start;
    if (start > n - period) {
        out[i] = SCAN_NAN_F32;
        return;
    }

    const long long sma_idx = (long long)start + period - 1;
    out[i] = (i < sma_idx) ? SCAN_NAN_F32 : pairs[i].y;
}
