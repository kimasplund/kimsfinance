# Ada Lovelace (sm_89) Microarchitecture — RTX 3500 Ada Reference

**Research date:** 2026-06-14 · **Scope:** SM microarchitecture, Tensor Cores, memory hierarchy, and occupancy limits for NVIDIA Ada Lovelace compute capability 8.9, specialized to the RTX 3500 Ada Generation Laptop GPU (AD104, sm_89, 12 GB). Target context: CUDA Toolkit 13.1 (nvcc V13.1), driver 595.71.05.

Local probe confirms the target device:
```
name = NVIDIA RTX 3500 Ada Generation Laptop GPU
compute_cap = 8.9
memory.total = 12282 MiB
driver_version = 595.71.05
```

---

## 1. Concrete numbers — RTX 3500 Ada

The RTX 3500 Ada is a **cut-down AD104** die (TSMC 4N / "5nm", ~35.8 B transistors, 294.5 mm²). Full AD104 ships 60 SMs / 7680 CUDA cores / 48 MB L2; the RTX 3500 enables **40 of 60 SMs** ([VideoCardz](https://videocardz.net/nvidia-rtx-3500-ada-laptop-gpu); [Notebookcheck](https://www.notebookcheck.net/NVIDIA-RTX-3500-Ada-Generation-Laptop-GPU-Benchmarks-and-Specs.744890.0.html)).

| Property | RTX 3500 Ada | Source |
|---|---|---|
| Architecture / die | Ada Lovelace, AD104 (cut) | [VideoCardz](https://videocardz.net/nvidia-rtx-3500-ada-laptop-gpu) |
| Compute capability | **8.9** (sm_89) | local `nvidia-smi` |
| SMs | **40** | [Notebookcheck](https://www.notebookcheck.net/NVIDIA-RTX-3500-Ada-Generation-Laptop-GPU-Benchmarks-and-Specs.744890.0.html) |
| CUDA cores (FP32 lanes) | **5120** (= 40 × 128) | [VideoCardz](https://videocardz.net/nvidia-rtx-3500-ada-laptop-gpu) |
| 4th-gen Tensor Cores | **160** (= 40 × 4) | [Notebookcheck](https://www.notebookcheck.net/NVIDIA-RTX-3500-Ada-Generation-Laptop-GPU-Benchmarks-and-Specs.744890.0.html) |
| RT cores (3rd gen) | 40 | same |
| Boost clock | ~2250 MHz (config-dependent) | [VideoCardz](https://videocardz.net/nvidia-rtx-3500-ada-laptop-gpu) |
| Peak FP32 | ~15.8–23 TFLOPS (TGP 60–140 W dependent) | [VideoCardz](https://videocardz.net/nvidia-rtx-3500-ada-laptop-gpu) |
| Memory | 12 GB GDDR6, 192-bit, **~432 GB/s** | [LaptopMedia](https://laptopmedia.com/video-card/nvidia-rtx-3500-ada-generation/) |
| L2 cache | **~32 MB** (40/60 of full-die 48 MB; see §4) | derived from [Ryan Smith / AD104 full = 48 MB](https://x.com/ryansmithat/status/1573190320019615747) |
| TGP | configurable **60–140 W** | [Notebookcheck](https://www.notebookcheck.net/NVIDIA-RTX-3500-Ada-Generation-Laptop-GPU-Benchmarks-and-Specs.744890.0.html) |

> **Clock/TFLOPS caveat (time-sensitive, per-laptop):** TGP is OEM-configurable 60–140 W, so sustained boost clock and the 15.8–23 TFLOPS FP32 range vary by chassis and thermals. Peak FP32 = `2 × CUDA_cores × clock` = `2 × 5120 × 2.25 GHz ≈ 23 TFLOPS` at full boost; the 15.8 TFLOPS figure reflects a ~100 W base-clock operating point. Benchmark your specific unit.

---

## 2. SM layout: CUDA cores, datapaths, schedulers

Each Ada SM is partitioned into **4 processing blocks (sub-cores / SMSPs)**, identical in structure to Ampere GA10x ([NVIDIA Ada GPU Architecture whitepaper v2.02](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)). Per SM:

- **128 FP32 "CUDA cores"** = 4 sub-cores × 32 FP32 lanes.
- **64 INT32 lanes** = 4 × 16. The other 16 lanes per sub-core are **shared FP32/INT32** (one FP32 *or* one INT32 op/clock). So per sub-core: 16 lanes are FP32-only + 16 lanes are FP32-or-INT32, giving **32 FP32 lanes** but only **16 INT32 lanes**.
- Net: **128 FP32 ops/clk** when all lanes do FP32, but a mixed FP32+INT32 workload contends — the dual-datapath only delivers full 128-wide FP32 *when there is no concurrent INT32 traffic on the shared lanes*.

This is the architectural meaning of NVIDIA's statement that **CC 8.9 has 2× the FP32 ops/cycle/SM of CC 8.0** (Volta-style separate INT path was the bottleneck; Ada/Ampere put FP32 on both paths) — [Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html). NVIDIA explicitly recommends recompiling for `sm_89` rather than running an `sm_80` binary, to realize this FP32 throughput.

**Schedulers / dispatch:** Each sub-core has **1 warp scheduler + 1 dispatch unit** (32 threads/clk) and its own **L0 instruction cache** and 64 KB register sub-file. That is **4 warp schedulers per SM**, each managing up to **12 resident warps** (48 warps / 4). One double-precision (FP64) unit per sub-core gives a token 1:64 FP64 rate — **FP64 is effectively unusable for throughput** on this part; keep financial math in FP32/TF32/FP16 ([whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf); [Wccftech SM breakdown](https://wccftech.com/nvidia-details-ada-lovelace-gpu-dlss-3-geforce-rtx-40-founders-edition-cooler/)).

---

## 3. 4th-gen Tensor Cores (FP8 Transformer Engine)

Ada's **4th-generation Tensor Cores** add the **FP8 Transformer Engine** with two formats ([NVIDIA Ada architecture](https://www.nvidia.com/en-us/technologies/ada-architecture/); [whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)):

- **E4M3** (4 exp / 3 mantissa) — higher precision, narrower range; forward-pass weights/activations.
- **E5M2** (5 exp / 2 mantissa) — wider range, lower precision; gradients/backward pass.

Throughput hierarchy (per Tensor Core, with **2:4 structured sparsity** roughly doubling dense rates): FP16 ⟶ TF32 (½ FP16) ⟶ FP8 (2× FP16). NVIDIA cites up to **5× tensor throughput** vs Ampere and ~1.4 PFLOPS FP8 (sparse) at the AD102 flagship level. For the RTX 3500 (160 Tensor Cores) the absolute number is ~40/144 of AD102's, but the **relative format ladder is identical**. Sparsity gives up to **2× throughput** only on the dense matrix when it satisfies the 2:4 structured pattern.

**Implication for quant work:** the financial-kernel sweet spot is **TF32 GEMM** (drop-in for FP32 matmul with ~10-bit mantissa, no code change beyond enabling TF32) and **FP16/FP8 for ML inference** (the orderflow INT8-quantized feature pipeline in this repo maps naturally to Tensor-Core INT8 paths). Plain elementwise indicator math (EMA, ATR, rolling stats) gets **zero benefit** from Tensor Cores — they only accelerate dense matmul/conv-shaped GEMMs. Use CUTLASS / cuBLASLt to reach them ([CUTLASS docs](https://github.com/NVIDIA/cutlass)).

---

## 4. Memory hierarchy

| Level | Per-SM | Per-GPU (RTX 3500) | Notes |
|---|---|---|---|
| Register file | **64K × 32-bit = 256 KB** | 10 MB (40 SM) | 65,536 regs/SM; max **255 regs/thread** |
| L1 / shared (unified) | **128 KB** | 5 MB | configurable carveout; **100 KB max shared/SM** |
| L2 cache | — | **~32 MB** | shared, set-aside persistence (§4.3) |
| Global (VRAM) | — | 12 GB GDDR6 | 192-bit, **~432 GB/s** |

Sources: [Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), [Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html), [CUDA C++ Programming Guide — Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).

### 4.1 Registers
65,536 32-bit registers per SM. To run the **full 1536 threads/SM** (48 warps) you must stay **≤ 42 regs/thread** (`65536 / 1536 ≈ 42.7`). Exceeding this drops occupancy in 64-register-granularity steps; at the 255-reg ceiling only `65536/255 ≈ 256` threads (8 warps, 16.7% occupancy) fit. **Register spilling to local memory** (L1→L2→DRAM backed) is the single most common quant-kernel performance cliff — watch `-Xptxas -v` register counts.

### 4.2 Shared memory / L1 carveout
The 128 KB unified L1/shared block is configurable, but **shared memory caps at 100 KB/SM**, and a single block can statically request only 48 KB — beyond that you must opt in dynamically (`cudaFuncAttributeMaxDynamicSharedMemorySize`) up to **99 KB usable** (CUDA reserves 1 KB/block) ([Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html); [CUDA C++ PG Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)). Requesting near-max shared memory limits you to **1 block/SM**, killing occupancy — balance carefully for tiled sliding-window kernels (e.g., the 20-tick orderflow window).

### 4.3 L2 cache (~32 MB) and persistence
Ada's headline is a **massive L2** — full AD104 = 48 MB (16× the GA102 6 MB). The 40-SM RTX 3500 maps to **~32 MB** (the 46-SM RTX 4070 desktop ships 36 MB; the 40-SM AD104 config = 32 MB; **confidence: medium** — NVIDIA does not publish an L2 line item for the pro/laptop SKU). This is large enough to **resident-cache an entire mid-size tick window or a multi-asset OHLCV working set**, dramatically cutting DRAM traffic.

You can pin hot data with the **L2 set-aside (persistence) API**: query `cudaDeviceProp::persistingL2CacheMaxSize` and `accessPolicyMaxWindowSize`, then `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)` and attach a `cudaAccessPolicyWindow{base_ptr, num_bytes, hitRatio, hitProp=cudaAccessPropertyPersisting}` to a stream/graph. The common recipe sets aside ~75% of L2: `min(int(prop.l2CacheSize*0.75), prop.persistingL2CacheMaxSize)` ([CUDA PG — L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html); [Lei Mao, L2 Persistent Cache](https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/)). Ideal for **reused lookup tables, strategy-config arrays, and rolling-stat state** that every kernel launch re-reads.

---

## 5. Occupancy limits (CC 8.9)

From [CUDA C++ Programming Guide — Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) and the [Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html):

| Limit (per SM) | CC 8.9 value |
|---|---|
| Max resident threads | **1536** |
| Max resident warps | **48** |
| Max resident blocks | **24** |
| Max threads / block | 1024 |
| 32-bit registers / SM | **65,536** |
| Max registers / thread | **255** |
| Max shared memory / SM | **100 KB** (99 KB addressable/block) |
| Warp size | 32 |

Note the **1536-thread (not 2048) ceiling**: a 512-thread block config yields only 3 blocks (1536 threads) → 75% of the 2048 some occupancy calculators assume from Ampere data-center parts. Prefer block sizes that divide 1536 evenly — **128, 192, 256, or 384 threads** — to avoid stranded warps. The 24-block cap means tiny blocks (≤64 threads) hit the block limit before the thread limit (24×64 = 1536, exactly full).

---

## 6. Implications: compute-bound vs memory-bound financial kernels

**Arithmetic-intensity / roofline frame.** Balance point ≈ `peak_FP32 / bandwidth` ≈ `15.8e12 / 432e9 ≈ 37 FLOP/byte` (≈53 at full-boost 23 TFLOPS). Below ~37 FLOP/byte the kernel is **memory-bound**; above it, compute-bound.

- **Memory-bound (the common case for quant indicators):** EMA/SMA, ATR, RSI, rolling z-scores, CVD — all do O(1) arithmetic per loaded value (well under 37 FLOP/byte). These are **bandwidth-limited at ~432 GB/s**. Levers: (1) **coalesced, vectorized loads** (`float4`); (2) **fuse multi-indicator passes** into one kernel to amortize the global read (this repo's fused orderflow kernel is exactly right); (3) **L2 persistence** to keep the sliding window resident; (4) maximize occupancy to hide DRAM latency (~300–500 cyc) — aim for ≥50% occupancy via ≤42 regs/thread. GPU/CPU crossover here is dominated by PCIe transfer cost, matching this repo's `>10K rows` GPU threshold.

- **Compute-bound:** dense covariance/correlation matrices, Monte-Carlo option pricing, large GEMM-shaped feature transforms, batched linear algebra. Route through **Tensor Cores (TF32/FP16/FP8) via cuBLASLt/CUTLASS** for the 2–5× uplift; mind FP32/INT32 datapath contention (§2) by keeping integer index math off the critical FP32 path.

- **Latency-bound small kernels:** sub-10K-row launches are dominated by **kernel-launch overhead (~5–10 µs)** and the 24-block/1536-thread ceiling, not throughput. **Batch many series into one launch** or use **CUDA Graphs** to amortize launch cost — this is the decisive optimization for the orderflow multi-strategy path.

**Net:** On the RTX 3500 Ada, most kimsfinance indicator kernels are **memory-bound** — the 32 MB L2 + persistence + kernel fusion matter far more than raw FLOPS. Tensor Cores only pay off for the genuinely GEMM-shaped or ML-inference parts of the pipeline (orderflow INT8 features, covariance, batched pricing).

---

## Sources
- [NVIDIA Ada GPU Architecture Whitepaper v2.02](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [NVIDIA Ada GPU Architecture Tuning Guide (CUDA 13.x)](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
- [CUDA C++ Programming Guide — Compute Capabilities appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [CUDA C++ Programming Guide — L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)
- [NVIDIA Ada Architecture — Tensor Cores / FP8](https://www.nvidia.com/en-us/technologies/ada-architecture/)
- [CUTLASS (NVIDIA) — Tensor Core GEMM templates](https://github.com/NVIDIA/cutlass)
- [VideoCardz — RTX 3500 Ada Laptop GPU specs](https://videocardz.net/nvidia-rtx-3500-ada-laptop-gpu)
- [Notebookcheck — RTX 3500 Ada benchmarks & specs](https://www.notebookcheck.net/NVIDIA-RTX-3500-Ada-Generation-Laptop-GPU-Benchmarks-and-Specs.744890.0.html)
- [LaptopMedia — RTX 3500 Ada specs (432 GB/s, 192-bit)](https://laptopmedia.com/video-card/nvidia-rtx-3500-ada-generation/)
- [Ryan Smith (AnandTech) — full AD104 = 60 SM / 7680 cores / 48 MB L2](https://x.com/ryansmithat/status/1573190320019615747)
- [Wccftech — Ada SM block diagram breakdown](https://wccftech.com/nvidia-details-ada-lovelace-gpu-dlss-3-geforce-rtx-40-founders-edition-cooler/)
- [Lei Mao — CUDA L2 Persistent Cache walkthrough](https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/)

**Confidence:** High (90%) on SM/occupancy/memory-hierarchy spec (corroborated by NVIDIA primary docs + local probe). Medium on the RTX 3500's exact L2 size (~32 MB, derived not datasheet-stated) and on per-laptop sustained TFLOPS/clock (TGP-dependent).
