# Tensor-Core / Low-Precision GEMM Cores — Codebase Analysis

**Scope:** `rust/src/gpu/{fp8_gemm_cutlass,fp8_wmma,quantization}.rs` + `rust/src/gpu/kernels/{fp8_conversions,fp8_cutlass,fp8_gemm_cutlass,fp8_jit_fallback,fp16_conversions,fp16_mma_ptx,fp16_wmma,fp32_mma_ptx}.cu`
**Target HW:** RTX 3500 Ada (sm_89), 12 GB, CUDA 13.1 / NVRTC JIT path
**Mode:** Read-only analysis. No source modified.

---

## 0. TL;DR — what is actually wired up

This subsystem is **~80% experimental/dead code**. The only numerically-functional tensor-core path is `FP8TensorCore::matmul_fp16` (an FP16 MMA via raw PTX), and even it has a correctness/precision concern (F16 accumulation, not the FP32 the docs claim) and is **never called by any finance compute path** — no Python binding, no production caller, only tests/benches/one broken example. The genuinely production-wired core in this file set is the **INT8 quantizer** (`quantization.rs` → `quantize_int8.cu`), which is the real orderflow memory-compression workhorse, not a GEMM.

Wiring summary (verified via `include_str!`, `build.rs`, and caller grep):

| File | Wired? | How | Evidence |
|---|---|---|---|
| `fp8_wmma.rs` (`FP8TensorCore`) | Partially | `include_str!` of `fp8_mma_ptx.cu`, `fp16_mma_ptx.cu`, `fp16_conversions.cu` | `fp8_wmma.rs:275,308,309,372` |
| `quantization.rs` | **Yes (prod)** | `include_str!` of `quantize_int8.cu` | `quantization.rs:94` |
| `fp8_gemm_cutlass.rs` (`FP8GemmCutlass`) | **No** — `new()` always errors | constructor returns `ComputationErrorStatic` | `fp8_gemm_cutlass.rs:69-72` |
| `kernels/fp8_gemm_cutlass.cu` | **No** — only `include_str!`'d by a *test that asserts it cannot compile* | NVRTC can't resolve `#include "cutlass/..."` | `fp8_gemm_cutlass.rs:462-467`; kernel `:32-36` |
| `kernels/fp8_conversions.cu` | **No** — orphaned | 0 `include_str!`/`build.rs` refs | grep: 0 code refs |
| `kernels/fp8_cutlass.cu` | **No** — orphaned (AOT path removed) | only mentioned in `build.rs:8` comment | grep: comment-only |
| `kernels/fp8_jit_fallback.cu` | **No** — orphaned | 0 refs anywhere | grep: 0 refs |
| `kernels/fp16_wmma.cu` | **No** — superseded by `fp16_mma_ptx.cu` | code comment says it requires `mma.h` (NVRTC-incompatible) | `fp8_wmma.rs:307` |
| `kernels/fp32_mma_ptx.cu` (real TF32) | **No** — orphaned | 0 refs; `matmul_tf32` disabled instead | grep: 0 refs; `fp8_wmma.rs:624-635` |
| `kernels/fp16_mma_ptx.cu` | **Yes** | the one live MMA kernel | `fp8_wmma.rs:308,372` |

Of 16 `.cu` files in `kernels/`, only 10 are wired; **5 of the 8 in-scope kernels are orphaned**, and a 6th (`fp8_gemm_cutlass.cu`) exists only to be asserted un-compilable.

---

## 1. Core Inventory

### Rust orchestration

**`fp8_gemm_cutlass.rs` — `FP8GemmCutlass` (487 LOC).**
A CUTLASS 3.5 FP8 GEMM manager. `new()` (`:57-72`) checks sm_89 then **unconditionally returns an error** (`CUTLASS_REQUIRES_HOST_BUILD_MSG`, `:26-29`): NVRTC cannot compile the CUTLASS-header-dependent `.cu`, and no CUBIN-load path exists. The rest of the struct (`fp32_to_fp8`, `fp8_to_fp32`, `gemm`, `matmul`, `gemm_batched`, `test` — `:86-439`) is **unreachable** API kept "for stability." Note `gemm` launches with `grid_dim=(1,1,1), block_dim=(1,1,1)` (`:247-251`) — even if it ran, it calls CUTLASS device-side from a single thread, which is not how CUTLASS GEMM is meant to be driven.

**`fp8_wmma.rs` — `FP8TensorCore` (883 LOC).** The main tensor-core entry point. Detects precision support (`supports_fp8_hardware`, `:74-76`; fp16 ≥ sm_70; tf32 ≥ sm_80). JIT-compiles three modules at first use (`load_fp8_kernels` `:273`, `load_fp16_kernels` `:305`, `load_fp32_kernels` `:370`). Public matmuls:
- `matmul_fp16` (`:439-485`) — **the only functional path**. FP32→FP16 convert, MMA, FP16→FP32 convert.
- `matmul_fp8` (`:417-428`) — **disabled**, always errors (`FP8_MATMUL_UNSUPPORTED_MSG` `:58-61`): prior impl fed f32-stored "quantized" values to a byte-oriented FP8 MMA → garbage.
- `matmul_tf32` (`:624-635`) — **disabled**, always errors: prior impl aliased the FP16 kernel and fed it f32 → garbage. (A correct TF32 kernel literally exists at `fp32_mma_ptx.cu` but is not wired.)
- `quantize_fp8_batch` (`:649-700`) — software FP8 *simulation* (round to 0.01, clamp ±448), stored as f32. Compiles `QUANTIZE_KERNEL` (`:83-111`) inline. Not real E4M3 bytes.

**`quantization.rs` — `QuantizationCalibrator` / `QuantizedFeatures` (978 LOC).** **The production core.** Per-feature INT8 (0-255) affine quantization of 6 orderflow features for 4× memory compression (19 GB → 2.4 GB for 10 strategies). CPU paths (`quantize`/`dequantize` `:223-269`) and GPU batch paths (`quantize_batch_gpu` `:317`, `dequantize_batch_gpu` `:428`) backed by `quantize_int8.cu`. Strong invariant handling: raw 0-255 codes carried through the i8 ABI to avoid the `f32 as i8` saturate-at-127 bug (`:233-237`, tests `:744-771`), FMA dequant with host-precomputed inverse scales (`:286-297`), degenerate-range collapse to `min`. This file is the most mature and best-tested in the set.

### CUDA kernels

**`fp16_mma_ptx.cu` (168 LOC) — LIVE.** Raw-PTX FP16 tensor-core MMA (`m16n8k16`, `HMMA16816` `:39-46`), NVRTC-compatible (no headers). Single warp per 16×8 output tile, `ldmatrix.x4/x2` from shared. **Concern:** the MMA descriptor is `f16.f16.f16.f16` (`:41`) — **F16 accumulation**, contradicting `fp8_wmma.rs:432` ("FP32 accumulation"). Accumulator `RC[2]` is `unsigned int` holding 4 packed halves.

**`fp16_conversions.cu` (445 LOC) — LIVE.** FP32↔FP16 bitwise convert (`unsigned short` for NVRTC). Scalar + vectorized (float4) variants, special-value tests, throughput benches. **Denormals flushed to zero** (`:83`), banker's rounding (`:104`).

**`fp32_mma_ptx.cu` (203 LOC) — ORPHANED but correct.** Real TF32 MMA (`m16n8k8`, `f32.tf32.tf32.f32` `:48-56`) with **FP32 accumulation** (4 float regs). This is exactly the TF32 kernel `matmul_tf32` claims doesn't exist. Not `include_str!`'d.

**`fp8_conversions.cu` (441 LOC) — ORPHANED.** Pure-bitwise FP32↔FP8 E4M3 (real packed bytes), saturate + **stochastic-rounding** variants. NVRTC-compatible. This is the *correct* E4M3 byte-packing the disabled `matmul_fp8` says it needs — but it is not connected to anything.

**`fp8_gemm_cutlass.cu` (646 LOC) — DEAD.** CUTLASS 3.5 `GemmUniversalWithAbsMax` for sm_89, three tile sizes (64³ / 128²×64 / 128×256×64), FP32 accumulate, auto-select + batched. Requires host nvcc+CUTLASS build (`:18-25`); cannot be JIT-compiled. Each `extern "C"` kernel runs CUTLASS from `threadIdx==0` (`:353`, `:388`...), which is structurally wrong for runtime-launched CUTLASS.

**`fp8_cutlass.cu` (96 LOC) — DEAD.** Native `cuda_fp8.h` matmul (AOT-only, `#include <cuda_fp8.h>` `:9`). Naive scalar O(MNK) dot-product per thread (`:33-41`) — **not tensor cores** despite the name. AOT compile path was removed (`build.rs:7-8`).

**`fp8_jit_fallback.cu` (112 LOC) — DEAD.** Software FP8 sim matmul, scalar O(MNK), quantize-on-the-fly to 0.01 (`:14-32`). NVRTC-compatible but never referenced.

**`fp16_wmma.cu` (113 LOC) — DEAD.** `mma.h`/`nvcuda::wmma` FP16 GEMM with **FP32 accumulate** (the "correct" version), but requires headers → NVRTC-incompatible, so superseded by `fp16_mma_ptx.cu` (`fp8_wmma.rs:307`).

---

## 2. Current Optimization State (evidence-based)

**Precision handling.**
- INT8 quantizer (`quantization.rs`): correct, FMA dequant, no division (`:935`), full 0-255 range preserved. **This is the high-quality part.**
- FP16 live path: **F16 accumulation** (`fp16_mma_ptx.cu:41`) — risks accuracy loss for long-K finance reductions and mislabels itself as FP32-accumulating (`fp8_wmma.rs:432`).
- FP8/TF32: disabled with honest errors. The *correct* building blocks (E4M3 bytes in `fp8_conversions.cu`, TF32 MMA in `fp32_mma_ptx.cu`) exist but are unwired.

**Memory pattern.**
- Conversions are well-vectorized: float4 loads, packed-uint32 stores (`fp8_conversions.cu:223-241`, `fp16_conversions.cu:190-216`), `__restrict__` throughout.
- INT8 kernel coalesced, 4-features/thread (`quantization.rs:385-387`).
- **Heavy redundant H2D + alloc churn:** `matmul_fp16` does 3 kernel launches + intermediate FP16 buffers + **4 `synchronize`/alloc points** (grep count 4 in `fp8_wmma.rs`); each quantize call re-uploads `min_values`/`scales` every invocation (`quantization.rs:343-365`, `438-466`) instead of caching on device.

**Fusion.**
- **None across this set.** `matmul_fp16` is 3 separate kernels (convert-in, MMA, convert-out) with no epilogue fusion — the FP32→FP16 cast should be fused into the MMA's `ldmatrix` load, and FP16→FP32 into the store. CUTLASS path *would* have had epilogue fusion but it's dead.
- INT8 quantize and dequantize are separate kernels and separate from the orderflow feature kernels (`orderflow_signals_batch.cu` etc.), so features are written FP32 → read back → quantized in a second pass rather than quantized in the producing kernel's epilogue.

**Persistent kernels / CUDA graphs.**
- **None.** grep for `graph|capture|persistent` across all three Rust modules → 0 hits. Every op is a discrete `launch` + `synchronize`. For the genetic-optimizer use case (many tiny 32×32 GEMMs, `fp8_gemm_cutlass.cu:623-628`) this is the dominant cost — launch latency ≫ compute.

**Launch pattern (inefficiencies).**
- `FP8GemmCutlass::gemm` launches `(1,1,1)/(1,1,1)` (`fp8_gemm_cutlass.rs:248-251`) — single thread.
- `matmul_fp16` uses **1 warp/block, no K-blocking across blocks** (`fp8_wmma.rs:509`); a single warp walks the entire K dimension (`fp16_mma_ptx.cu:93-140`) with `#pragma unroll` over a runtime trip count and `__syncthreads()` inside a 1-warp block (pointless barrier). Occupancy will be poor for large K.
- Dead CUTLASS kernels run CUTLASS from one thread (`:353` etc.) — non-functional design even if compiled.

**Obvious inefficiencies (ranked by concreteness).**
1. `matmul_fp16` re-JITs nothing but re-allocs + 4 syncs per call; no buffer reuse (`fp8_wmma.rs:474-485,537-605`).
2. Calibration params re-uploaded to device every quantize/dequantize call (`quantization.rs:343,438`).
3. FP16 MMA single-warp / F16-accumulate (`fp16_mma_ptx.cu:41,93`).
4. Five orphaned kernels compiled into the binary as string blobs but never used (dead weight, ~1.5 KLOC of `.cu`).

---

## 3. Ranked Optimization Opportunities

| # | Opportunity | Impact | Effort | Evidence / Rationale |
|---|---|---|---|---|
| **1** | **Replace bespoke MMA with cuBLAS/cuBLASLt for FP16+TF32 GEMM** and delete the hand-rolled single-warp kernels. The benches *already* call cuBLAS for TF32/FP16 (`benches/tensor_core_benchmark.rs:213,263`), proving the dep is acceptable. cuBLASLt gives FP32-accumulate, multi-warp tiling, autotuned, and FP8 on Ada. | **High** | Med | `fp16_mma_ptx.cu:41,93` (F16 accum, 1 warp); `fp8_wmma.rs:509`. Immediately fixes precision + occupancy + gives a real FP8 path. |
| **2** | **Wire the existing correct TF32 kernel** (`fp32_mma_ptx.cu`, FP32 accumulate, `m16n8k8`) into `matmul_tf32` instead of returning an error. Lowest-risk real tensor-core win for finance (TF32 keeps FP32 range, ~8× CUDA-core throughput, tolerable mantissa loss for indicators). | **High** | Low | Kernel exists and is correct (`fp32_mma_ptx.cu:48-56,99`); `matmul_tf32` is disabled (`fp8_wmma.rs:624-635`). Just unwired. |
| **3** | **Fuse INT8 quantization into the orderflow feature-producing kernels' epilogue** (write INT8 directly), eliminating the FP32 write + reread + second pass. Also cache calibration params on device once. | **High** | Med | `quantization.rs:343-365` re-uploads per call; separate kernel from `orderflow_signals_batch.cu`. This is the *production* path → real $ win. |
| **4** | **Fuse FP32↔FP16 casts into the MMA load/store** (or use cuBLASLt fp16 with fp32 I/O) to kill 2 of 3 launches + intermediate buffers + 2 syncs in `matmul_fp16`. | **Med** | Med | `fp8_wmma.rs:474-485` (3 kernels), 4 sync/alloc points. |
| **5** | **Wire `fp8_conversions.cu` (real E4M3 bytes) + a u8 GEMM to re-enable `matmul_fp8`** end-to-end (u8 buffers throughout), with runtime validation vs FP32 ref. | **Med** | High | The disabled error explicitly names this as the missing piece (`fp8_wmma.rs:58-61`); the kernel already produces correct packed bytes incl. stochastic rounding (`fp8_conversions.cu:371-441`). |
| **6** | **CUDA Graphs + buffer pool for the genetic-optimizer batch GEMMs** (many tiny 32×32). Capture the convert→GEMM→convert (or quantize→score) sequence once, replay per generation; launch latency currently dominates. | **Med** | Med | No graph usage anywhere (grep=0); tiny-matrix use case documented `fp8_gemm_cutlass.cu:623-628`. |
| **7** | **Delete dead cores** (`fp8_cutlass.cu`, `fp8_jit_fallback.cu`, `fp16_wmma.cu`, and `fp8_gemm_cutlass.rs`/`.cu` unless a host CUTLASS build is committed). Removes ~1.5 KLOC of misleading "production-ready" code and an `example` that calls the now-erroring `matmul_fp8` (`examples/fp8_genetic_optimizer.rs:118`). | **Low** (clarity/maintenance) | Low | grep: 0 live refs; example is broken against current API. |
| **8** | **Fix the docstring/precision claim** on `matmul_fp16` (says FP32 accumulate, kernel is F16) — or switch the MMA to `f16.f16.f16.f32` accumulation. | **Low** | Low | `fp8_wmma.rs:432` vs `fp16_mma_ptx.cu:41`. |

---

## 4. Fusion / Combine / Megakernel Potential

**Where finance compute could exploit this (and what to fuse):**

- **Orderflow feature → INT8 quantize (highest value).** The feature kernels (`orderflow_signals_batch.cu`, `tick_aggregation.cu`) currently emit FP32; `quantize_int8.cu` is a separate pass. **Fuse the affine quantize into the feature kernel's store epilogue.** This is the one path with real production traffic (4× memory, 19 GB→2.4 GB) and is a pure win — no extra global round-trip. (Opportunity #3.)

- **Convert-GEMM-convert collapse.** `matmul_fp16`'s three kernels are a textbook fusion target: fold the FP32→FP16 narrowing into the `ldmatrix` load and FP16→FP32 widening into the store, i.e. a single GEMM kernel that takes FP32 I/O and runs FP16/TF32 tensor cores internally (exactly what cuBLASLt's `CUDA_R_32F` compute with `CUDA_R_16F` data does). Eliminates 2 launches, 2 intermediate buffers, 2 syncs. (Opportunities #1/#4.)

- **Genetic-optimizer batch megakernel.** The intended workload is *many small independent GEMMs* (100 param sets × 32×32, `fp8_gemm_cutlass.cu:623-628`). These should NOT be N separate launches. Two viable shapes: (a) **batched/strided GEMM** (cuBLASLt grouped GEMM), or (b) **persistent megakernel + CUDA Graph** that processes the whole population per generation. The current `gemm_batched` (`fp8_gemm_cutlass.rs:316`) is the right *idea* but dead and one-block-per-matrix-single-thread. (Opportunities #1/#6.)

- **Shared conversion library.** `fp8_conversions.cu`, `fp16_conversions.cu`, and the inline `QUANTIZE_KERNEL` all implement narrowing converters independently. They can share one device-function header and, more importantly, be inlined as epilogues rather than standalone kernels — no standalone convert kernel should survive once GEMM/feature kernels fuse their own casts.

**What should NOT be combined:** the precision modes themselves (FP16 vs TF32 vs FP8) want distinct tensor-core instruction shapes (`m16n8k16` vs `m16n8k8` vs FP8) and accumulator widths — these stay as selectable kernels, not one über-kernel. The fusion wins are *vertical* (convert/quantize into the producer/consumer), not *horizontal* (merging precisions).

---

## 5. Risks / Caveats for Downstream Work

- **`FP8GemmCutlass` and `fp8_gemm_cutlass.cu` are aspirational.** Enabling them requires committing a host-side nvcc+CUTLASS build step and a CUBIN-load path in `build.rs` (currently only `rsi_fused.cu` is AOT-compiled, `build.rs:95`). Do not treat the "Expected Performance" table (`fp8_gemm_cutlass.cu:611-645`) or CLAUDE.md FP8 throughput numbers as measured — the module never ran.
- **No FP8/TF32 numbers in this set are validated.** Both Rust docstrings explicitly state no hardware figures were measured (`fp8_wmma.rs:19-20`, `fp8_gemm_cutlass.rs:16`).
- The cleanest, lowest-risk near-term wins are **#2 (wire existing TF32 kernel)** and **#3 (fuse INT8 quantize)** — both use code that already exists and is correct.
