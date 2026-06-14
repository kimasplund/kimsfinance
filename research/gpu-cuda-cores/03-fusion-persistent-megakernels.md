# Combined Cores: Kernel Fusion, Persistent Kernels, Megakernels & CUDA Graphs

**Research date:** 2026-06-14
**Target HW:** NVIDIA RTX 3500 Ada Generation Laptop (Ada Lovelace, sm_89, 12 GB), driver 595.x, CUDA Toolkit 13.1
**Scope:** Design space for a "combined core" that processes a batch of technical indicators with minimal overhead and maximal data reuse, plus a decision framework for when a unified megakernel wins.

---

## 1. The cost model you are optimizing against

Every separate kernel launch on Ada pays a fixed tax. Measured host-side launch overhead is roughly **2–5 µs per kernel** with the standard `<<<>>>` / stream submission path, and it becomes a visible fraction of end-to-end latency once a kernel's own runtime drops to tens of microseconds ([CUDA Graphs — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)). For a batch of *N* indicators, the naive "one kernel each" path costs:

```
T_naive ≈ N × (launch_overhead + sync) + Σ compute_i + Σ (global writes + re-reads of shared inputs)
```

Three distinct overheads are in play, and the four techniques in this document each attack a different subset:

| Overhead | Attacked by |
|---|---|
| Per-kernel **CPU launch cost** (2–5 µs) | CUDA Graphs, megakernels, persistent kernels |
| **Global-memory round-trips** of shared inputs (close/high/low re-read N times) | Fusion, megakernels |
| **Tail/load-balancing** idle time between launches | Persistent kernels, megakernels |

A key truth for indicator workloads: on a 100K-candle window, OHLC inputs are ~0.8 MB each. Twelve indicators each re-reading `close` from global memory is ~10 MB of redundant traffic that fusion can collapse to a single load held in registers/shared memory. The Ada L2 (the RTX 3500 Ada has a large L2 relative to its class, per [Ada Tuning Guide §4.2.1, Increased L2 capacity](https://docs.nvidia.com/cuda/ada-tuning-guide/)) absorbs much of this, so the real fusion win on Ada is often **launch-overhead amortization and occupancy of the SMs, not raw DRAM bandwidth**.

---

## 2. Kernel fusion — when it helps, when it hurts

**Fusion** merges the bodies of several kernels into one launch so that intermediate results stay in registers/shared memory instead of round-tripping through global memory.

### Helps when
- **Inputs are shared.** Indicators over the same `close[]` (RSI, ROC, EMA, momentum) load it once. This is the dominant win for technical-indicator batches.
- **There is a producer→consumer chain.** Gains/losses → Wilder smoothing → RSI can be fused into one pass; the project's `rsi_fused.cu` does exactly this (gains/losses → CUB `DeviceScan` for the IIR smoothing → RSI), targeting **~130 µs hybrid → ~61 µs fused (2.13×)** by eliminating D2H/H2D transfers and the CPU leg.
- **Kernels are short.** When each kernel runs <50 µs, launch overhead is 7–10%+ of runtime; fusing K of them removes (K−1) launches ([pegainfer: From Launch Overhead to CUDA Graph](https://susun-blog.com/blog/pegainfer-3-cuda-graph/)).

### Hurts when
- **Register pressure rises.** Fused kernels accumulate live state. Ada caps **255 registers/thread** and has a **64 K-register file per SM** ([Ada Tuning Guide §4.1.1, Occupancy](https://docs.nvidia.com/cuda/ada-tuning-guide/)). At 64 regs/thread you fit 1024 threads/SM; at 128 regs/thread only 512; beyond ~168 you spill. *"Fusing multiple routines increases on-chip memory demands, which may limit occupancy or restrict block size"* and over-fusion causes **register spilling or decreased blocks/SM** ([Kernel Fusion in GPU Computing — EmergentMind](https://www.emergentmind.com/topics/kernel-fusion); [Reducing register pressure](https://app.studyraid.com/en/read/11728/371500/reducing-register-pressure)).
- **Shared memory becomes the limiter.** Ada offers **100 KB shared memory/SM, max 99 KB/block** ([Ada Tuning Guide §4.2.2](https://docs.nvidia.com/cuda/ada-tuning-guide/)). A fused kernel staging many indicators' tiles can hit this wall before registers, dropping to 1 block/SM.
- **Lost parallelism / divergence.** Indicators with different access patterns (Wilder IIR vs a windowed max for Donchian) fused vertically serialize phases that could otherwise overlap on separate SMs. Empirically there are **diminishing returns past ~3 fused kernels** due to bandwidth saturation, register pressure, and occupancy drops ([Analyzing the Impact of Kernel Fusion — MDPI Electronics 2025](https://www.mdpi.com/2079-9292/15/5/1034)).
- **Mitigation exists:** CUDA 13's **shared-memory register spilling** stores spills in on-chip shared memory instead of local/L2, cutting spill latency for register-heavy fused kernels ([NVIDIA: Improve CUDA Kernel Performance with Shared Memory Register Spilling](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/)). Useful, but it trades shared-memory capacity for register relief — i.e. it moves the occupancy bottleneck rather than removing it.

**Horizontal vs vertical fusion.** *Horizontal* fusion runs independent indicators side-by-side in one launch (different warps/blocks do different indicators) to reclaim thread-level parallelism even at the cost of block-level parallelism ([Automatic Horizontal Fusion for GPU Kernels](https://arxiv.org/pdf/2007.01277)). *Vertical* fusion chains dependent stages to keep intermediates on-chip (the FlashAttention pattern — [A Case Study in CUDA Kernel Fusion: FlashAttention-2 on Hopper with CUTLASS](https://arxiv.org/html/2312.11918v1)). For an indicator batch, **horizontal fusion of independent indicators + vertical fusion within each multi-stage indicator** is the natural decomposition.

---

## 3. Persistent kernels (grid-resident)

A **persistent kernel** launches exactly enough blocks to fill the GPU once, then each block loops pulling work until the queue drains — circumscribing the per-item logic in a `while(work)` loop instead of relying on the scheduler to launch fresh waves ([Gupta et al., *A Study of Persistent Threads Style GPU Programming*](https://escholarship.org/content/qt3j76d3td/qt3j76d3td.pdf)).

### Three sub-patterns

1. **Grid-stride loop (static).** Each thread strides by `gridDim.x * blockDim.x` over the data. Trivial, decoupled from data size, and already used across this project's `quantize_int8.cu`, `tick_aggregation.cu`, and `tick_backtest_batch.cu`. This is the safe default and composes cleanly with cooperative launches that constrain grid size.

2. **Cooperative-groups grid sync.** `cudaLaunchCooperativeKernel` + `cooperative_groups::this_grid().sync()` gives a **device-wide barrier inside one launch** (CC ≥ 6.0), enabling multi-phase algorithms (e.g. global reduction then broadcast) without returning to the host ([Cooperative Groups — CUDA Programming Guide §4.4](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html); [Grid Synchronization with Cooperative Groups — Acceleware](https://training.acceleware.com/blog/grid-synchronization-cooperative-groups)). **Hard constraint:** every block must be co-resident, so the grid is capped at `numSM × maxActiveBlocksPerSM` from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. Oversubscribing deadlocks. This caps the launch to a modest, occupancy-bounded grid — fine for indicators, restrictive for huge irregular workloads.

3. **Dynamic work-stealing / producer-consumer queue.** Blocks pull tasks from a global queue via `atomicAdd` on a counter, or use per-worker deques (owner pops LIFO from the tail, thieves steal FIFO from the head). On irregular workloads, centralized and distributed queues showed **>100× more idle time than task-stealing/donation** schemes ([Gupta et al.](https://escholarship.org/content/qt3j76d3td/qt3j76d3td.pdf); [GTaP: GPU-Resident Fork-Join Task-Parallel Runtime](https://arxiv.org/pdf/2604.05982)). The persistent style serves CPU-GPU sync, load balancing/irregular parallelism, producer-consumer locality, and global sync — and can deliver up to an order-of-magnitude speedup, **but also loses performance in many cases** when work is regular and the bookkeeping overhead isn't amortized.

**When persistent helps for indicators:** ragged batches (many symbols × different window lengths) where a static grid leaves SMs idle in the tail. A single persistent launch with an atomic work counter keeps all SMs busy and pays the 2–5 µs launch cost exactly **once for the whole batch**.

**When it hurts:** uniform, embarrassingly parallel indicator passes. Here a plain grid-stride launch already saturates the GPU; adding atomics and a global queue only injects contention.

---

## 4. Megakernels — fusion taken to its limit

A **megakernel** (a.k.a. uberkernel) is one persistent kernel that contains the *entire* pipeline; blocks loop and select among multiple code paths/tasks, eliminating both inter-kernel launches and device-wide sync at kernel boundaries ([Gupta et al. — uberkernel definition](https://escholarship.org/content/qt3j76d3td/qt3j76d3td.pdf)). The 2025 LLM-serving wave proved the model end-to-end: **Mirage Persistent Kernel (MPK, June 2025)** compiles an entire LLM forward pass into a single megakernel, using an **SM-level task graph** with **decentralized in-kernel scheduling across SMs** to enable cross-operator software pipelining, reporting **1.2×–6.7× lower inference latency** by removing launch overhead and overlapping compute/data-load/communication ([Compiling LLMs into a MegaKernel — Zhihao Jia](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17); [MPK arXiv 2512.22219](https://arxiv.org/pdf/2512.22219)). Kog's "monokernel" variant uses compile-time, programmer-managed work partitioning to drop even the in-kernel scheduler overhead ([Kog: single-kernel LLM inference on MI300X](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/)).

**Why a megakernel can beat a graph:** a CUDA Graph still executes node-by-node with real (if tiny) inter-node scheduling and cannot overlap the *tail* of node A's last blocks with the *head* of node B. A megakernel can pipeline at SM granularity — block 3 starts indicator B as soon as it finishes its slice of indicator A. **The cost:** it is one giant kernel with the union of all register and shared-memory requirements, so it inherits §2's occupancy ceiling at its worst, plus substantial engineering and debugging complexity.

---

## 5. CUDA Graphs — capturing many small kernels without rewriting them

CUDA Graphs let you **define a DAG of kernels/memcpys once and replay it cheaply**. Separating definition from execution reduces CPU launch cost versus streams and lets the driver optimize across the whole workflow — *"presenting the whole workflow to CUDA enables optimizations which might not be possible with the piecewise work submission mechanism of streams"* ([CUDA Graphs — Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)). Graphs are typically built via **stream capture** (run the pipeline once in capture mode; the runtime records kernels, memcpys and their dependencies) and replayed many times ([Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)). The benefit grows with kernel count: launch cost is paid once at instantiation, then **graph replay overhead is ~2–3 µs for the whole graph** vs N × (2–5 µs) for individual launches.

**Graphs vs fusion — they are complementary, not alternatives.** Fusion removes global-memory round-trips and reduces *kernel count*; graphs remove *launch overhead* for whatever kernels remain. For an indicator batch you often want both: fuse what shares data, then wrap the remaining distinct kernels in a graph.

### The project's own cautionary tale (load-bearing)
This repo's `batch_graphs.rs` documents two real failure modes that gate graphs off today:
1. **Key-ordering panic** — graphs cached under a *sorted* indicator key but fetched with the *unsorted* key, panicking on any non-sorted order.
2. **Net-negative fast path** — captured graphs never wired up result buffers, so after replay the executor recomputed every indicator the traditional way; replay added pure overhead while *claiming* a 16.7× reduction.

The takeaway is methodological: **graphs only pay off if (a) the captured nodes actually own their input/output buffers, and (b) the workflow is stable across replays.** Variable-size inputs force re-capture (cheap-ish but not free), and conditional execution breaks the static-DAG assumption ([CUDA Graphs — Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)). Always validate against the naive path; a graph that recomputes is slower than no graph.

---

## 6. Maintainability trade-off (do not ignore)

| Technique | Perf ceiling | Engineering cost | Debuggability |
|---|---|---|---|
| Separate kernels | Baseline | Lowest | Easiest (per-kernel profiling) |
| CUDA Graphs | +launch savings | Low (no kernel rewrite) | Moderate (graph capture pitfalls) |
| Pairwise fusion | +data reuse | Moderate | Moderate |
| Persistent kernel | +load balance | High (queues/atomics) | Hard (deadlock risk) |
| Megakernel | Highest | Very high | Hardest (one opaque kernel) |

Each step up the ladder trades a modular, individually-profilable kernel for a monolith whose occupancy is set by its hungriest component. For a research/quant library that adds indicators frequently, the modularity cost is real: a megakernel must be re-tuned every time an indicator changes its register footprint.

---

## 7. Decision framework — when a unified megakernel beats many specialized kernels

Apply in order; stop at the first decisive branch.

1. **Is the batch latency-critical and replayed many times** (live tick loop, optimization sweep) **with a fixed kernel set?** If no → ship specialized kernels; you are done. Per-launch overhead doesn't matter for one-shot work.

2. **Do the indicators share inputs or form producer→consumer chains?**
   - No shared data, all independent, uniform sizes → **specialized kernels wrapped in a CUDA Graph.** You get the 2–3 µs replay cost without paying fusion's occupancy tax. (Fix the buffer-ownership bug from §5 first.)
   - Shared inputs / multi-stage chains → continue.

3. **Estimate the fused register + shared-memory footprint.** Sum the live state. If the fused kernel exceeds **~168 regs/thread** (drops below ~33% occupancy on Ada's 64 K-register file) or **>99 KB shared/block**, full fusion will collapse occupancy ([Ada Tuning Guide §4.1.1 / §4.2.2](https://docs.nvidia.com/cuda/ada-tuning-guide/)). In that case → **fuse in small clusters of ≤3 indicators** (the empirical sweet spot — [MDPI 2025](https://www.mdpi.com/2079-9292/15/5/1034)) and graph the clusters together. Do **not** build one megakernel.

4. **Is the workload ragged** (many symbols × heterogeneous window lengths) so a static grid leaves SMs idle in the tail? → add a **persistent grid-stride or work-stealing layer** so one launch keeps all SMs busy. This is orthogonal to fusion and usually the bigger win for ragged batches than fusion itself.

5. **Only build a true megakernel when ALL hold:**
   - Latency floor matters more than throughput (you are chasing the last few µs).
   - The pipeline has cross-stage overlap a graph cannot exploit (tail of stage A overlapping head of stage B), as MPK demonstrated for LLMs ([MPK arXiv](https://arxiv.org/pdf/2512.22219)).
   - The fused footprint still fits Ada's occupancy budget (passes step 3), or you accept 1–2 blocks/SM because the work is latency- not throughput-bound.
   - The kernel set is stable enough to justify the re-tuning cost on every change.

If any of those fail, the **CUDA Graph + small-cluster fusion (+ persistent layer for ragged work)** combination captures ~80% of the megakernel's benefit at a fraction of the complexity, and keeps each indicator independently profilable.

---

## Confidence

**Medium-High (82%).** Architectural limits (registers, shared memory, occupancy, launch overhead) are pulled directly from the [Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/) and [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) and corroborated by the project's own `rsi_fused.cu` and `batch_graphs.rs`. Megakernel speedup ranges (1.2–6.7×) are from LLM-serving systems, not technical-indicator batches — directionally valid but not a like-for-like benchmark, hence the held-back confidence. The exact fusion-cluster crossover (≤3) is empirical and workload-dependent; measure on the RTX 3500 Ada before committing.
