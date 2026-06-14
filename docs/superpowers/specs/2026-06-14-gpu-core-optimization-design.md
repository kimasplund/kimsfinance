# GPU Core Optimization — Design Spec

**Date:** 2026-06-14
**Branch:** `feature/gpu-core-optimization`
**Status:** Approved (design); pending Gate 0 results + spec review
**Author:** Kim Asplund (with Claude)
**Hardware:** NVIDIA RTX 3500 Ada (sm_89, 12 GB), CUDA Toolkit 13.1, driver 595.x

---

## 1. Goal & scope

Upgrade/enhance the GPU compute cores in `rust/src/gpu` (~52K LOC) for the
RTX 3500 Ada, **measured** (not theoretical), and make an **evidence-based**
decision on whether a "fully combined core" (unified/persistent megakernel) is
worth building.

**Scope:** all of `rust/src/gpu` was *analyzed* (see `research/gpu-cuda-cores/`);
we *optimize* the high-impact Pareto subset first. "Instruments" = the
`rust/src/assets` asset classes drive heterogeneous inputs to the cores.

**Non-goals:** rewriting working reference cores; chasing every kernel; building a
megakernel on faith; touching the CPU path.

## 2. How the design was reached

- **Research** (`research/gpu-cuda-cores/`, 11 docs): authoritative web research
  (Ada/sm_89, kernel optimization, fusion/persistent/megakernels, tensor cores,
  instruments) + code-grounded analysis of every GPU subsystem + a synthesis.
- **IR-v2** selected **Adversarial Reasoning** (affinity 4.70; Solution-Exists=5,
  Robustness=5) — i.e. *attack the research's conclusion before committing*.
- The strategy **survived** the adversarial pass with six refinements (§5).

## 3. Key findings (from research)

- The GPU core is **bimodal**: a few f32+SoA+fused reference cores
  (`tick_aggregation`, `orderflow_batch`, `quantization`, `scan`) vs a **legacy
  FP64 majority** (15/17 indicators, all Heston pricing, 2D/3D batch+sweep,
  candles) fighting Ada's **1/64** FP64:FP32 rate.
- Work is **memory-bound** (roofline ~37 FLOP/byte) → wins are traffic and
  round-trip cuts, not raw FLOPs.
- Expensive infra is **built but disconnected**: device-resident `scan.rs`
  (1478 LOC, imported by nobody), CUDA Graphs disabled, async pool a no-op,
  multi-stream gated off, a correct TF32 kernel reachable only behind a
  `matmul_tf32` that returns an error.

## 4. Architecture decision

**Keep per-kernel cores. Pursue targeted fusion of data-sharing families + CUDA
Graphs. Do NOT build a unified megakernel** unless a benchmark on the real hot
path justifies it.

Rationale (held up under adversarial attack): a megakernel inherits the union of
every component's register/shared budget, sees diminishing returns past ~3 fused
kernels on Ada, and — decisively — must be **re-tuned every time an indicator is
added**, which this library does routinely. Targeted fusion + graphs captures
~80% of the benefit at a fraction of the complexity while keeping each indicator
independently profilable.

## 5. Adversarial refinements (baked into the plan)

1. **FP64→FP32 is per-indicator with a numerical gate**, not a blanket flip.
   Variance/cumulative indicators (Bollinger, Keltner, CCI, cumulative sums)
   risk catastrophic cancellation in f32 → use stable algorithms (Welford for
   variance, Kahan/pairwise for long sums) and add **f32-vs-f64 tolerance tests**
   per indicator.
2. **Wire disconnected infra only after verifying correctness.** Investigate
   *why* each piece is disabled (git history/comments); it may be unfinished WIP,
   not merely unwired. Treat as "verify → fix → benchmark", not "flip switch".
3. **Match optimization to the dominant workload.** FP64→FP32 helps large
   memory-bound batches; launch-overhead-bound small/latency workloads need
   graphs/fusion. Characterize the real workloads first.
4. **The combined-core decision gate uses the REAL hot path** (genetic-sweep /
   batch backtest, where the same OHLCV is evaluated millions of times), not a
   synthetic batch.
5. **Every fusion passes an occupancy gate** (Nsight Compute, >33%, before/after).
   Stateful/branchy kernels (ADX, Supertrend) can blow the register budget.
6. **Gate 0 feasibility first:** confirm the `gpu` feature builds and existing
   GPU tests/benchmarks run on this machine before any optimization.

## 6. Phased plan

**Gate 0 — feasibility (no code changes).**
Build `--features gpu`; run existing GPU tests + a baseline benchmark on the
RTX 3500 Ada; record the baseline numbers and establish the measurement harness.
Exit criterion: green build + reproducible baseline.

**Phase 1 — quick wins (no architecture commitment).**
- FP64→FP32 per indicator, each behind an f32-vs-f64 tolerance test; stable
  algorithms for variance/cumulative cases.
- Investigate + wire the disconnected infra that proves correct (`scan.rs`,
  module cache, `alloc_uninit`, TF32 `matmul_tf32`).
- Benchmark each change individually. Target: documented ≥1.5–2× on
  memory-bound paths; no correctness regressions (existing 1376-test suite +
  new tolerance tests + visual regression as the safety net).

**Phase 2 — targeted fusion + graphs.**
- Fuse only data-sharing families behind occupancy gates: True-Range
  (ATR/ADX/Supertrend), Typical-Price (Σx, Σx²), EMA-consumers, Heston
  CF→weight→FFT→extract.
- Remove mid-pipeline host syncs; enable CUDA Graphs + multi-stream.

**Decision gate — the "maybe combined core?".**
Benchmark (A) cluster-fused-in-a-graph vs (B) unfused-graph vs (C) hand-built
megakernel on the **real sweep/backtest hot path**, profiled in Nsight Compute.
Build (C) only if it beats (A) by a margin justifying the re-tuning tax **and**
holds >33% occupancy. Strong prior: (A) wins.

## 7. Success criteria & measurement

- **Correctness:** all existing tests pass; new f32-vs-f64 tolerance tests pass;
  visual-regression unchanged.
- **Performance:** per-change before/after benchmarks on the RTX 3500 Ada;
  report wall-clock + Nsight metrics (occupancy, achieved bandwidth, DRAM
  throughput). No regression on any path.
- **Honesty:** every speedup claim is benchmarked on this hardware; no
  fabricated numbers (project rule).

## 8. Risks

| Risk | Mitigation |
|---|---|
| `gpu` feature doesn't build cleanly here | Gate 0 verifies first; fix build before anything else |
| f32 accuracy regressions (variance/cumulative) | per-indicator tolerance tests + stable algorithms |
| Disconnected infra is buggy WIP | investigate-before-wire; benchmark before adopting |
| Fusion hurts occupancy | per-fusion Nsight gate (>33%) |
| Scope sprawl over 52K LOC | optimize Pareto subset first; phases gate the rest |

## 9. Out of scope (this round)

- macOS/Windows GPU support; multi-GPU; datacenter-class assumptions.
- The unified megakernel unless the decision gate justifies it.
