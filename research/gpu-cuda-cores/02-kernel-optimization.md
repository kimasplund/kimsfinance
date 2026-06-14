# CUDA Kernel Optimization (Ada-Aware)

**Research date:** 2026-06-14
**Target hardware:** NVIDIA RTX 3500 Ada Generation Laptop GPU — Ada Lovelace, compute capability **sm_89**, 12 GB GDDR6, CUDA Toolkit 13.1.
**Scope:** Single authoritative reference for optimizing the `rust/src/gpu` CUDA core, with emphasis on rolling-window / scan-style financial-indicator kernels.

---

## 1. Ada (sm_89) Hardware Limits You Must Design Around

These are the hard per-SM constraints from the [NVIDIA Ada GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/). Every occupancy/register decision is bounded by them:

| Resource | sm_89 limit |
|---|---|
| Max concurrent warps / SM | **48** (1536 threads) |
| Register file / SM | **65,536** 32-bit registers |
| Max registers / thread | **255** |
| Shared memory / SM | **100 KB** (carveout: 0/8/16/32/64/100 KB) |
| Max shared mem / block | **99 KB** (opt-in above 48 KB via `cudaFuncSetAttribute`) |
| Max thread blocks / SM | **24** |

Two derived rules dominate kernel design:
- **Register budget for full occupancy:** 65,536 / 1536 ≈ **42 registers/thread**. Above ~42 regs you cannot reach 48 warps; above ~64 you drop to 50% occupancy.
- **48 KB static shared-memory wall:** anything beyond 48 KB requires a runtime opt-in (`cudaFuncAttributeMaxDynamicSharedMemorySize`). Static `__shared__` arrays are still capped at 48 KB for binary compatibility.

Ada also ships a very large device-wide L2 (e.g. **48 MB on AD104**, per [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-corrects-mistake-with-one-of-its-new-rtx-40-super-gpus)). For financial workloads where a rolling window of prices fits in tens of MB, L2 residency — not DRAM bandwidth — is often the real ceiling.

---

## 2. Occupancy vs. ILP — Stop Maximizing Occupancy Blindly

The single most over-applied "rule" is "maximize occupancy." Volkov's [*Better Performance at Lower Occupancy*](https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf) (GTC 2010) showed kernels hitting near-peak at **~25% occupancy** by exploiting **instruction-level parallelism (ILP)** instead of thread-level parallelism. Occupancy only hides latency via *more warps*; ILP hides it via *independent instructions inside one thread*.

Practical consequences for indicator kernels:
- **Spend registers to gain ILP.** Have each thread process N independent window positions (register blocking / "thread coarsening"). Keep N partial sums in registers so the warp scheduler has independent FMAs to issue while loads are in flight. This *lowers* occupancy but raises throughput.
- **Target the sweet spot, not the ceiling.** On Ada, ~50% occupancy (24 warps) is typically enough to saturate memory-bound kernels. Use the freed register/shared budget for blocking factor.
- **Decide empirically.** Use the Nsight Compute *Occupancy* + *Warp State* sections (§9) to see whether you are *latency-bound* (more warps/ILP helps) or *throughput-bound* (already saturated; occupancy is irrelevant).

This is a genuine trade-off, not a free win: vectorized/blocked loads "increase register pressure and reduce overall parallelism" ([CUDA Pro Tip: Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)).

---

## 3. Global Memory Coalescing

A warp's 32 threads should access a **contiguous, aligned** region so the hardware merges them into the minimum number of 32-byte sectors / 128-byte cache lines. The canonical pattern is `data[blockIdx.x * blockDim.x + threadIdx.x]` (thread *i* → element *i*).

- **Stride-1 across the warp** is the goal. Strided or random per-thread access multiplies the number of memory transactions and tanks effective bandwidth.
- **Rolling-window trap:** a naive kernel where thread *i* loads `price[i-W .. i]` makes *adjacent threads re-read overlapping data with offset strides*. The fix is to coalesce the *load* (each thread loads one contiguous element into shared memory) and do the *windowing* from shared memory (§4).
- Verify with Nsight Compute's *Memory Workload Analysis*: watch **sectors-per-request** (ideal: 4 sectors per 32-thread `LD.32` request) and global load efficiency ([Unlock GPU Performance: Global Memory Access](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/)).

---

## 4. Shared-Memory Tiling & Bank-Conflict Avoidance

Shared memory is organized into **32 banks**; successive 32-bit words map to successive banks, and all 32 banks can be serviced in one cycle ([Lei Mao — CUDA Shared Memory Bank](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/)). A **bank conflict** occurs when ≥2 threads in a warp hit *different addresses in the same bank*; the access serializes (a 32-way conflict is 32× slower). Same-address broadcast is conflict-free.

Avoidance techniques:
- **Padding:** declare `__shared__ float tile[BDIM][BDIM + 1];`. The +1 column shifts the bank mapping so column accesses (the classic transpose case) stop colliding ([cuda-programming.blogspot](http://cuda-programming.blogspot.com/2013/02/bank-conflicts-in-shared-memory-in-cuda.html)). Costs one column of shared memory.
- **Swizzling:** XOR-based index permutation gives conflict-free access *without* wasting padding bytes — preferred when shared memory is the occupancy limiter ([Lei Mao — Swizzling](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/)).
- **Halo loading for windows:** load `blockDim.x + W` contiguous elements (block tile + window halo) into shared memory once, coalesced; every thread then computes its window from on-chip data. This converts O(W) redundant DRAM reads per thread into a single shared-memory pass — the dominant win for rolling indicators (SMA, ATR, Bollinger).

---

## 5. Register Pressure & Spilling

Each thread's registers come from the 64K/SM file. Two failure modes:
1. **Occupancy collapse:** crossing 42 → 64 → 128 regs/thread steps occupancy down (48 → 32 → 16 warps).
2. **Register spilling:** exceeding the per-thread budget spills to **local memory** (which lives in L1/L2/DRAM). Spills add load/store traffic on the hot path.

Controls and diagnostics:
- Compile with `-Xptxas -v` to print *registers, spill stores, spill loads* per kernel.
- Cap registers with `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` or `-maxrregcount`. `__launch_bounds__` is preferred — it lets ptxas trade registers for the occupancy you actually want.
- Some spilling is acceptable if it buys ILP/blocking that nets out positive — measure, don't assume. Check Nsight Compute's *local memory* traffic to confirm spills aren't dominating.

---

## 6. Vectorized Loads (float4 / int4)

A 128-bit `LD.128` (via `float4`/`int4`) moves 4× the data per instruction, cutting instruction count, reducing index arithmetic, and saturating the bus better than four `LD.32`s ([CUDA Pro Tip](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)).

```cuda
// 4 floats per thread, one 128-bit transaction
float4 v = reinterpret_cast<const float4*>(prices)[idx];
```

Requirements and caveats:
- **Alignment is mandatory:** the pointer must be 16-byte aligned for `float4`. Device allocations are aligned to the type size, but *offset pointers must preserve alignment* — a `+1` float offset breaks `float4` loads.
- **N must be divisible by 4** (or handle the tail scalar).
- **Register cost:** each `float4` consumes 4 registers; on register-limited or low-parallelism kernels, scalar loads may win. This is the §2 trade-off in concrete form.

For OHLCV indicator kernels, packing the 4-or-5 series (O/H/L/C/V) so that one vector load fetches a full bar is a natural fit.

---

## 7. Async Global→Shared Copy (`cp.async` / `cuda::memcpy_async`)

sm_80+ (including Ada sm_89) provides the `cp.async` PTX instruction: it copies **global → shared directly, bypassing the register file and L1**, and is *asynchronous* so compute can overlap the copy ([Controlling Data Movement on Ampere](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/), [Async Data Copies — Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)).

- Requirements: source = global, dest = shared, alignment ≥ 4 bytes (16-byte alignment enables the more efficient `cp.async.cg` cache-global variant).
- Use the modern API: `cuda::memcpy_async` bound to a `cuda::pipeline` / `cuda::barrier` ([cuda::memcpy_async docs](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html)). The group overload cooperatively issues the copy across the block.
- **Software pipelining ("double buffering"):** issue the async copy for tile *k+1* while computing on tile *k*; `pipeline.commit()` / `pipeline.consumer_wait()` gate the stages. This hides DRAM latency behind compute and is the highest-leverage transform for streaming/scan kernels that march a window across a long time series.
- Note: TMA / `cp.async.bulk` is **Hopper-only**; on Ada you use classic `cp.async`.

---

## 8. Warp-Level Primitives — Reductions Without Shared Memory

Warp shuffles (`__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`) exchange registers directly between lanes via the `SHFL` instruction, **bypassing the memory hierarchy entirely** — no shared memory, no `__syncthreads()` ([Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)).

```cuda
// Warp-level sum reduction (full mask)
for (int off = 16; off > 0; off >>= 1)
    val += __shfl_down_sync(0xffffffff, val, off);   // lane 0 holds the sum
```

- Always pass an explicit **mask** (`0xffffffff` for the full warp). The `_sync` forms are mandatory since CUDA 9 — independent thread scheduling means non-sync shuffles are unsafe.
- `__ballot_sync(mask, pred)` returns a 32-bit mask of which lanes' predicate is true — ideal for counting signals (e.g. "how many bars in this warp crossed the threshold") and for `__popc()`-based stream compaction of trade signals.
- **Scan/prefix-sum** (cumulative sum, CVD, running max) maps to `__shfl_up_sync` for the intra-warp inclusive scan, then a shared-memory pass to stitch warps — the textbook pattern for cumulative-volume-delta and rolling-cumulative indicators.
- Warp shuffles are generally faster than shared memory for reductions and free up the occupancy that shared memory would have consumed ([Warp Shuffle vs Shared Memory](https://medium.com/a-gpu-crash-course-for-embedded-engineers/warp-shuffle-vs-shared-memory-which-is-faster-f8ed254a7c29)).

---

## 9. L2 Cache Persistence (`cudaAccessPolicyWindow`)

sm_80+ can pin a hot data window in L2. For financial kernels that repeatedly re-read the *same* recent-price/indicator-parameter buffer across many launches, this is a large win ([L2 Cache Control — Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html), [Lei Mao — L2 Persistent Cache](https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/)).

```cuda
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, setAsideBytes);
cudaStreamAttrValue attr = {};
attr.accessPolicyWindow.base_ptr  = hotData;
attr.accessPolicyWindow.num_bytes = windowBytes;
attr.accessPolicyWindow.hitRatio  = 0.6f;            // fraction tagged "persisting"
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

Rules:
- Keep `hitRatio * num_bytes ≤ cudaLimitPersistingL2CacheSize` or you thrash the set-aside region.
- The set-aside pool is **shared across all concurrent kernels** — over-reserving starves neighbors.
- Reset with `cudaCtxResetPersistingL2Cache()` when the hot buffer changes; otherwise stale lines stay pinned.
- Tag streaming output (write-once results) as `cudaAccessPropertyStreaming` so it doesn't evict your persisting window.

---

## 10. Launch-Configuration Tuning

- Block size: a multiple of **32** (warp size); 128–256 threads is the usual starting band. Use `cudaOccupancyMaxPotentialBlockSize` to get a launch config that maximizes occupancy *for the kernel's actual register/shared use*, then sweep ±1–2 block sizes empirically.
- Choose `minBlocksPerSM` in `__launch_bounds__` to express the occupancy you proved best in §2, letting ptxas cap registers accordingly.
- Grid size: at least `2–4 × SM_count` blocks to keep all SMs fed; for grid-stride loops, size to a few waves rather than one element per thread.

---

## 11. Profiling — Nsight Compute Metrics That Matter

Drive every change with [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html). Read sections in this order:

1. **GPU Speed of Light (SOL):** top-line Compute% vs Memory%. Tells you immediately whether you are compute- or memory-bound — optimize the dominant one.
2. **Memory Workload Analysis:** *sectors per request* (coalescing — §3), L1/L2 hit rates, DRAM throughput. High sectors/request ⇒ uncoalesced access.
3. **Occupancy:** achieved vs theoretical; the limiter (registers / shared mem / block size). Gap between the two ⇒ load imbalance or tail effect.
4. **Warp State (Scheduler) Statistics:** *stall reasons*. `Long Scoreboard` = memory-latency-bound (add ILP/`cp.async`); `MIO Throttle` / `LG Throttle` = saturated memory pipes (already throughput-bound, occupancy won't help).
5. **Launch / Register stats:** registers/thread, shared bytes, spill loads/stores (§5).

Use the built-in **rules-based guided analysis** — it flags uncoalesced access, bank conflicts, and low occupancy automatically with remediation hints.

---

## 12. Prioritized Checklist — Rolling-Window / Scan Financial Kernels

Apply top-down; re-profile after each step.

1. **Coalesce the base load.** One contiguous, aligned element per thread; never let adjacent threads issue overlapping strided window reads (§3).
2. **Tile the window into shared memory with a halo.** Load `blockDim + W` contiguous elements once; compute all windows from on-chip data. Eliminates O(W) redundant DRAM reads (§4).
3. **Vectorize the load** to `float4`/`int4` where alignment and N%4 allow — fewer instructions, full-width transactions (§6).
4. **Double-buffer with `cp.async` / `memcpy_async`** for long series: prefetch tile *k+1* while computing tile *k* to hide DRAM latency (§7).
5. **Use warp shuffles for the reduction/scan core.** `__shfl_down_sync` for window sums/extrema; `__shfl_up_sync` for cumulative scans (SMA, CVD, running max). Avoids shared memory and `__syncthreads()` (§8).
6. **Coarsen threads / block in registers** to raise ILP — process several window positions per thread; accept lower occupancy if Warp-State shows latency-bound stalls (§2).
7. **Pad or swizzle shared memory** to kill bank conflicts on column/transpose access (§4).
8. **Cap registers with `__launch_bounds__`**, check `-Xptxas -v` for spills, retune block size with the occupancy API (§5, §10).
9. **Pin the recurring read-only buffer** (parameter set / recent-price window reused across launches) in L2 via `cudaAccessPolicyWindow` (§9).
10. **Profile end-to-end in Nsight Compute** — confirm SOL shifted toward the resource you targeted and that stalls/sectors-per-request improved (§11).

---

### Source Credibility

| Source | Type | Credibility |
|---|---|---|
| NVIDIA Ada Tuning Guide, CUDA Programming Guide, Nsight Compute Guide | Primary / vendor docs | High |
| NVIDIA Developer Blogs (Vectorized Access, Warp Primitives, Ampere Data Movement, Global Memory Access) | Vendor engineering | High |
| Volkov, *Better Performance at Lower Occupancy* (GTC 2010) | Peer-known primary | High |
| Lei Mao's Log Book (bank conflicts, swizzling, L2 persistence, vectorized access) | Expert engineering blog, code-verified | Medium-High |
| Tom's Hardware (AD104 48 MB L2 correction) | Tech journalism | Medium (corroborated) |

**Overall confidence: ~90%.** Hardware limits and APIs are vendor-documented for sm_80/sm_89; the one area to validate empirically on the actual RTX 3500 Ada is the occupancy-vs-ILP sweet spot and L2 set-aside sizing, since both depend on the specific kernel's register/shared footprint.
