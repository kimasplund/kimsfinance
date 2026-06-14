# GPU/CUDA Cores — Synthesis & Optimization Roadmap

**Synthesis date:** 2026-06-14
**Target HW:** NVIDIA RTX 3500 Ada Generation Laptop GPU — Ada Lovelace, AD104, **sm_89**, 40 SMs / 5120 FP32 cores / 160 4th-gen Tensor Cores, 12 GB GDDR6 @ ~432 GB/s, ~32 MB L2. CUDA Toolkit 13.1, driver 595.x.
**Scope:** Unifies the architecture research (`01`–`05`) with the read-only codebase audits (`codebase/indicators.md`, `options-pricing.md`, `batch-infra.md`, `tensor-gemm.md`, `candles-tick-misc.md`) into a single ranked plan for the `rust/src/gpu` core (~52K LOC).

---

## 1. Executive Summary — state of the cores + the biggest levers

The GPU core is **bimodal**. A small set of *reference-grade* cores — `tick_aggregation.rs`, `orderflow_batch.rs`, `quantization.rs`, and `scan.rs` — already do exactly what the Ada hardware wants: **f32 + SoA layout, fused multi-stage kernels, native ordered-integer atomics, device-resident scans, batched readbacks, and process-wide PTX caching** (`candles-tick-misc.md` §2 "Tier A"; `tensor-gemm.md` §0). The rest of the codebase — 15 of 17 technical indicators, the entire Heston/Greeks pricing path, the 2D/3D batch+sweep kernels, and the candle generators — is a **legacy FP64 tier** that fights the hardware on the single most important axis.

Three structural facts dominate every recommendation:

1. **FP64 is a 1/64 throughput tax on this GPU.** Ada sm_89 runs FP64 at 1/64 of FP32 (`01-ada-architecture.md` §2; `04` §1). Yet 15/17 indicators, all Heston kernels, and the 2D/3D/candle kernels are `double` (`indicators.md` §2.1: 15/17 f64; `options-pricing.md` §2.1: FP64 transcendentals in the CF kernel; `candles-tick-misc.md` §2: `kernels_2d` 54 `double` refs, `kernels_3d` 41). The project *already proved it knows better* — `ma_advanced.rs:34-41` and `scan.rs:48-55` document why f32 is both faster and numerically safe for these contractive/window patterns — but only 2 cores adopted it. This is the largest single lever, and it is mostly mechanical.

2. **The expensive infrastructure is built but disconnected.** The 1478-LOC device-resident `scan.rs` (Wilder/EMA, graph-capturable) is imported by **nobody** (`indicators.md` §2.3: `grep scan::` → only itself). CUDA Graphs are **disabled** (`batch_graphs.rs` always errors; `cuda_graphs.rs` 732 LOC has zero production callers — `batch-infra.md` §2.5). The async memory pool is a **no-op** (`async_alloc.rs` falls back to `alloc_zeros` — `batch-infra.md` §2.8). Multi-stream dispatch is **gated off** (`batch.rs:65` — §2.6). A correct TF32 tensor-core kernel exists but `matmul_tf32` returns an error instead of calling it (`tensor-gemm.md` opp #2). The wins here are *wiring*, not new kernel math.

3. **Most indicator/aggregation work is memory-bound, not compute-bound.** Roofline balance is ~37 FLOP/byte; EMA/ATR/RSI/rolling stats sit far below it (`01` §6). So the highest-value transforms are the ones that cut **global traffic and PCIe round-trips**: f32 (halves transfer + storage), fusing shared inputs (HLC re-uploaded N times per batch — `batch-infra.md` §2.2, `indicators.md` §2.4), replacing O(n·period) window scans with the existing prefix-sum, and removing mid-pipeline host syncs that forbid graph capture (`indicators.md` §2.4). Tensor Cores matter only for the genuinely GEMM-shaped minority (covariance, batched regression, MC correlation — `04` §6; `options-pricing.md` §4), and even there the Ada laptop part is half-rate FP32-accumulate, so budget ~115–130 TFLOPS, not datacenter numbers (`04-tensor-cores-low-precision.md` §1).

**The single biggest levers, in order:** (1) **f32 conversion** of the legacy tier; (2) **wire `scan.rs`** into RSI/ADX/MFI/ATR/Supertrend and the 2D/3D RSI paths, deleting the CPU Wilder round-trips; (3) **fix + enable CUDA Graphs** on the stable repetitive workloads (genetic backtest, parameter sweeps, live per-bar loop); (4) **share device-resident inputs** so a batch uploads HLC once. These four account for the overwhelming majority of recoverable performance and are largely independent of any "combined core" decision.

---

## 2. Ranked optimization-opportunity matrix

Ranked by **impact ÷ effort** (highest leverage first). "Impact" is expected throughput / latency / VRAM win on the RTX 3500 Ada. File refs point at the load-bearing evidence in the underlying docs.

| # | Opportunity | Cores affected | Expected impact | Effort | Risk | Evidence |
|---|-------------|----------------|-----------------|--------|------|----------|
| 1 | **Wire the existing device-resident `scan.rs` Wilder/EMA into RSI, ADX, MFI, ATR, Supertrend** — delete the D2H→CPU-smooth→H2D round-trips and mid-pipeline syncs. | RSI, ADX, MFI, ATR, Supertrend (+2D/3D RSI) | High (removes 2 syncs + 4–9 PCIe transfers/call; unblocks graph capture) | Low–Med | Low (primitive is tested, CPU-parity-verified, graph-capturable) | `indicators.md` §2.3, opp #1; `scan.rs:616/656`, `adx.rs:381/477`, `rsi.rs:264/335`, `mfi.rs:402/472`; `candles-tick-misc.md` opp #6 |
| 2 | **Convert legacy FP64 kernels → f32** (indicators, 2D/3D batch+sweep, candles, Heston CF). Keep i64 timestamps; keep f64 *in-register* only for Bollinger var / equity accumulation. | 15/17 indicators, `kernels_2d/3d`, candle kernels, Heston CF/Greeks | High (≤64× ALU headroom on compute-heavy scans; ~2× from halved global+PCIe traffic; halves VRAM) | Med | Low–Med (variance cancellation in Bollinger/CCI — mitigate w/ Welford or f64 in-register accumulator) | `01` §2; `04` §1; `indicators.md` §2.1; `options-pricing.md` H2; `candles-tick-misc.md` opp #1; precedent `ma_advanced.rs:34-41`, `scan.rs:48-55`, `tick_aggregation.rs:289-294` |
| 3 | **Make `batch_backtest_genetic` use the module cache + `alloc_uninit`, then capture it as a CUDA Graph.** This is the de-facto combined core today. | `gpu/mod.rs` backtest host, `device.rs` | High (wins compound every invocation; ~50–70% launch-overhead cut on replay) | Low–Med | Low (module cache + alloc_uninit already exist); Med for graph (needs stable result-buffer binding) | `batch-infra.md` §2.2, opp #2; `mod.rs:585-621` reloads PTX every call, `:565-578` redundant zeroing; `device.rs:165,253` |
| 4 | **Fix + enable CUDA Graphs for the stable repetitive paths** (param sweeps, genetic backtest, Greeks/strategy sequence, live per-bar loop). Bind result buffers; stabilize the sorted-key lookup. | `batch_graphs.rs`, `cuda_graphs.rs`, Heston Greeks/strategies | High (launch cost paid once → ~2–3 µs replay vs N×2–5 µs) | High | Med (the two documented failure modes: unbound result buffers → recompute; key-ordering panic) | `03` §5 (project cautionary tale); `batch-infra.md` §2.5, opp #1; `options-pricing.md` M2 |
| 5 | **Replace O(n·period) sliding windows with the existing prefix/pair-sum scans.** Shared Typical-Price + (Σx, Σx²) pass for CCI + Bollinger + CMF + MFI; same for 2D/3D SMA/Bollinger. | SMA, Bollinger, CCI, CMF, MFI, `kernels_2d/3d` | High (O(n·period) → O(n) global traffic; Bollinger walks window twice, CCI recomputes TP period×) | Med | Low (prefix-sum machinery exists in `scan.rs`) | `indicators.md` §2.2, opp #3, §4.3; `bollinger.rs:45,59`, `cci.rs:61`; `candles-tick-misc.md` opp #5; `02` §12 |
| 6 | **Share device-resident HLC/OHLCV inputs across a batch** — upload once per chunk, pass `CudaSlice` views to indicator kernels. | `batch.rs`, all `*_gpu` indicator fns | Med (a 5-indicator OHLC batch pays ~5× H2D today) | Med–High | Med (signature changes across all `*_gpu` fns) | `batch-infra.md` §2.2, opp #4; `batch.rs:683-684`; `indicators.md` §2.4, opp #5 |
| 7 | **Fuse the Heston pricing pipeline on-GPU** (revive `carr_madan_weight.cu` + `extract_prices.cu`, add cuFFT) and replace 8× full re-pricing in Greeks with one bumped megakernel; drop the per-launch `synchronize`. | Heston CF, Greeks, strategies | High (eliminates ~13 MB/batch D2H + 8 CPU integration loops + ~16 syncs → ~1) | Med–High | Med (cuFFT is a new dep; precision validation vs Lewis-vs-BS test) | `options-pricing.md` H1/H3/M1, §4; `carr_madan_weight.cu:23-24` (intent existed); `greeks_gpu.rs:148-165`, `heston_pricing.rs:651-654` |
| 8 | **Make `sweep.rs` actually batch** — dispatch to the existing `kernels_3d.rs` (period×asset×candle) megakernel + on-device Sharpe reduction instead of the sequential per-parameter loop; metrics on-device. | `sweep.rs`, `kernels_3d.rs` | High (delivers the 10–50× the docs already claim) | Med | Low (kernels + reduction already exist — pure wiring) | `candles-tick-misc.md` opp #3; `sweep.rs:409-413` loops, `:538-680` host metrics, `:700-720` stubbed; `kernels_3d.rs:174-230` |
| 9 | **Wire the existing correct TF32 kernel** (`fp32_mma_ptx.cu`, FP32 accumulate, `m16n8k8`) into `matmul_tf32`; replace the bespoke single-warp FP16 MMA with cuBLAS/cuBLASLt. | `fp8_wmma.rs`, tensor GEMM | High (for GEMM-shaped work: covariance, batched regression) | Low (TF32 wire) / Med (cuBLASLt) | Low (kernel exists & is correct; benches already call cuBLAS) | `tensor-gemm.md` opp #1/#2; `fp32_mma_ptx.cu:48-56`, `fp8_wmma.rs:624-635`; `04` §3 |
| 10 | **Retire the 4 single-thread candle kernels; re-parallelize block-per-symbol.** Route time bars to the f32 dense-atomic aggregator (fixes the all-zeros flaky bug); make volume/range/renko/HA grid-stride across symbols, drop cooperative `grid.sync`. | `candles/{time,volume,range,renko,heikin_ashi}` | High (fixes <1% utilization *and* a shipped flaky bug) | Med–High | Med (re-parallelization touches 5 kernels) | `candles-tick-misc.md` opp #2; `time_bars.rs:113-141,227`, `volume_bars.rs:123`, `renko.rs:107` |
| 11 | **Fuse INT8 quantization into the orderflow feature kernel's epilogue** (write INT8 directly); cache calibration params on device once. | `quantization.rs`, `orderflow_signals_batch.cu` | Med (kills FP32 write + reread + 2nd pass; production traffic path) | Med | Low (quantizer is the most mature, best-tested core) | `tensor-gemm.md` opp #3, §4; `quantization.rs:343-365` |
| 12 | **Enable multi-stream dispatch in `batch.rs`** via event-gated pinned release (the primitives exist). | `batch.rs`, `streams.rs`, `pinned_memory.rs` | Med (15–30% from Fast/Med/Slow H2D/compute/D2H overlap) | Med | Med (cross-stream pinned reuse correctness) | `batch-infra.md` §2.6, opp #3; `batch.rs:65`, `pinned_memory.rs:640` |
| 13 | **Add rolling-extremum (monotonic-deque / sparse-table) shared kernel** for Donchian, Aroon, Fibonacci swings, Ichimoku Tenkan/Kijun. | Donchian, Aroon, Fibonacci, Ichimoku | Med (4 indicators share it; O(n·period) → O(n)) | Med–High (max isn't invertible) | Med | `indicators.md` §4.4; `donchian.rs:43`, `aroon.rs:49` |
| 14 | **Fuse aggregation → orderflow → backtest device residency** — pass device pointers instead of D2H/H2D between the three already-f32-compatible tick cores. | `tick_aggregation`, `orderflow_batch`, `tick_backtest_batch` | Med (eliminates `N_strat × N_ticks` signal round-trip) | Med–High (backtest is f64 → precision decision) | Med | `candles-tick-misc.md` §4.2–4.3; `orderflow_batch.rs:688` D2H → `tick_backtest_batch.rs:366` H2D |
| 15 | **Fix `PersistentKernelManager` double-device alloc + unsafe `transmute_copy` param packing.** | `persistent/mod.rs` | Med (correctness for any unified core; ~128 MB startup) | Low–Med | Low | `batch-infra.md` opp #6/#7; `persistent/mod.rs:347-348,647-661` |
| 16 | **Cleanup: delete dead/orphaned code** — `sma_gpu_shared` (0–3%), disabled `rsi_fused.rs`, 5 orphaned tensor `.cu` files, `*.backup/*.orig` artifacts. | indicators, tensor GEMM, Heston | Low (maintenance/clarity) | Low | Low | `indicators.md` opp #7; `tensor-gemm.md` opp #7; `options-pricing.md` L1 |

---

## 3. "Combined core" feasibility assessment

**Question:** Is a unified/persistent megakernel (or fused batch core) worth it here, or is targeted fusion better?

**Verdict: A single grand megakernel is NOT warranted. Targeted fusion + CUDA Graphs is the right architecture for this codebase.** A megakernel only wins under four conditions that this library largely fails (`03-fusion-persistent-megakernels.md` §7):

1. *Latency floor matters more than throughput, replayed many times with a fixed kernel set* — true for the live per-bar loop and genetic sweeps, but **false** for the one-shot indicator/pricing calls that dominate the API.
2. *Cross-stage overlap a graph cannot exploit (tail of A overlapping head of B)* — the MPK/LLM result that justified megakernels (1.2–6.7×) came from transformer pipelines, **not** technical-indicator batches; the evidence is directionally valid but not like-for-like (`03` confidence note).
3. *Fused footprint still fits Ada's occupancy budget* — a megakernel inherits the **union** of every component's register + shared-memory demand. Ada caps 255 regs/thread, 64K regs/SM, 100 KB shared/SM; empirical fusion returns diminish past **~3 fused kernels** (`03` §2, §7; MDPI 2025). A library that mixes Wilder IIR recurrences, windowed maxima, and FP64 transcendentals would collapse to 1 block/SM.
4. *Kernel set stable enough to justify re-tuning on every change* — **false**: this is a research library that adds indicators frequently; a megakernel must be re-tuned every time an indicator's register footprint changes (`03` §6 maintainability table).

**Why targeted fusion wins instead.** The CUDA-Graph + small-cluster-fusion combination captures ~80% of a megakernel's benefit at a fraction of the complexity, and keeps each indicator independently profilable (`03` §7). The codebase already validates both halves of this: `orderflow_batch.rs` is a *correct, fused, f32 3-pass scan* that dedups shared work across 5 strategies (`candles-tick-misc.md` §2 Tier A), and `batch_backtest_genetic` is a *fused-data 4-phase pipeline* (`batch-infra.md` §2.3). Neither is a single opaque kernel; both fuse only what shares data.

**Where to fuse (the natural decomposition is horizontal-across-independent + vertical-within-multistage — `03` §2):**

- **True-Range family** (ATR, ADX, Supertrend) compute a *byte-for-byte identical* True Range from HLC (`indicators.md` §4.1). **Compute TR once, Wilder-smooth once via `scan.rs`, fan out to all three.** Keltner already proves the shared-ATR/EMA pattern works.
- **EMA-consumer family** (Elder Ray, Keltner already take EMA as input; Bollinger mid-band, MACD, DEMA/TEMA seeds) share one device-resident `ema_f32` output (`indicators.md` §4.2).
- **Typical-Price / windowed-sum family** (CCI, MFI, Bollinger) share one prefix-sum producing `(Σx, Σx²)` — the single highest-leverage indicator fusion (`indicators.md` §4.3).
- **Heston pricing chain** `CF → weight → FFT → extract` is the canonical *vertical* fusion: 3 of 4 stages are element-wise/reduction over the same `[n_options × n_fft]` grid; only the FFT stays a cuFFT call (`options-pricing.md` §4, H1).
- **Tick→candle→features→signals→backtest** is the natural *device-residency* fusion (not a megakernel): keep the SoA f32 buffer resident, pass device pointers stage-to-stage (`candles-tick-misc.md` §4.1–4.3).

**Recommended architecture sketch (targeted, not monolithic):**

```
Per chunk (L2-sized OHLCV slab, uploaded ONCE, SoA f32):
  ┌─────────────────────────────────────────────────────────────┐
  │ Shared-intermediate stage (compute once, fan out):           │
  │   TR  ← max(hl,|hc|,|lc|)        [ATR,ADX,Supertrend]         │
  │   TP  ← (h+l+c)/3                 [CCI,MFI]                    │
  │   prefix(Σx,Σx²) over close/TP   [SMA,Boll,CCI,CMF,MFI]      │
  │   ema_f32 / wilder_f32 (scan.rs) [EMA-consumers, RSI,ADX...] │
  ├─────────────────────────────────────────────────────────────┤
  │ Per-indicator epilogue kernels (small clusters ≤3):          │
  │   read shared intermediates from device, write outputs       │
  ├─────────────────────────────────────────────────────────────┤
  │ Wrap the whole fixed sequence in ONE CUDA Graph → replay     │
  │ per chunk / per generation / per bar (result buffers BOUND)  │
  └─────────────────────────────────────────────────────────────┘
  + persistent grid-stride layer ONLY for ragged batches
    (many symbols × heterogeneous window lengths) to fill the tail
```

This is `ma_advanced.rs`'s `[num_series, series_len]` batch layout (`indicators.md` §4.5) generalized, fed by one shared upload, with the `scan.rs` primitives finally wired in, and the fixed DAG captured as a graph. The **persistent cooperative-grid kernel is reserved for ragged batches only** — for uniform indicator passes a plain grid-stride launch already saturates the GPU, and adding atomics/`grid.sync` only injects contention (`03` §3; the candle kernels are the cautionary example of misapplied persistence — `candles-tick-misc.md` §2).

---

## 4. Recommended phased plan

### Phase 0 — Foundation fixes (low risk, compounding, no architecture commitment)
Independent of the combined-core decision; do these first regardless.

1. **f32 conversion** of the legacy indicator + 2D/3D + candle + Heston-CF tier (opp #2). Keep f64 in-register only for variance/equity accumulation. Validate each against the existing CPU-parity tests.
2. **Wire `scan.rs`** into RSI/ADX/MFI/ATR/Supertrend and 2D/3D RSI, deleting CPU Wilder round-trips (opp #1).
3. **Module-cache + `alloc_uninit`** in `batch_backtest_genetic` (opp #3, low-effort half).
4. **Wire the existing TF32 kernel** into `matmul_tf32` (opp #9, low-effort half).
5. **Cleanup** dead code (opp #16) to stop drift.

**Quick-win benchmark gate:** re-run `/kf:bench:scaling` and `/kf:bench:compare` across 1K/10K/100K candles. Expect ≥1.5–2× on the converted memory-bound indicators (halved traffic + ALU) and removal of all mid-pipeline syncs. This phase needs no megakernel.

### Phase 1 — Targeted fusion + traffic reduction
6. **Prefix-sum window replacement** for SMA/Bollinger/CCI/CMF/MFI (opp #5) + shared Typical-Price/`(Σx,Σx²)` fusion.
7. **Share device-resident HLC inputs** across the batch (opp #6).
8. **Fuse INT8 quantize into the orderflow epilogue** (opp #11); **make `sweep.rs` dispatch to `kernels_3d`** (opp #8); **re-parallelize candle kernels** (opp #10).
9. **Fuse the Heston pipeline** + bumped Greeks megakernel (opp #7).

### Phase 2 — Launch-overhead amortization
10. **Fix + enable CUDA Graphs** on the now-stable, sync-free pipelines (opp #4) and **multi-stream dispatch** (opp #12). These only pay off *after* Phase 0–1 remove the mid-pipeline host syncs that forbid capture.

### Combined-core decision gate (after Phase 0–1)

**Run this benchmark before committing any persistent-megakernel work:**

> On the RTX 3500 Ada, take the most fusion-favorable real workload (a 5-indicator batch over the True-Range + Typical-Price families, sharing inputs, replayed N≥1000 times as in a genetic sweep). Measure end-to-end latency for three implementations:
> **(A)** small-cluster fused kernels (≤3) wrapped in a CUDA Graph (the Phase-1/2 architecture);
> **(B)** the same kernels unfused but graph-wrapped;
> **(C)** a hand-built persistent megakernel with a task-descriptor table.
>
> Profile each in Nsight Compute for achieved occupancy, register/thread, and SOL (`02` §11). **Build the megakernel (C) only if it beats (A) by a margin that justifies the re-tuning-on-every-indicator-change maintenance cost AND its fused footprint stays >33% occupancy (≤~168 regs/thread, ≤99 KB shared/block — `03` §7 step 3).** If (A) is within ~10–15% of (C), ship (A): the graph + cluster-fusion captures ~80% of the benefit at a fraction of the complexity and keeps every indicator independently profilable.

The strong prior from all five architecture docs and the codebase audits is that **(A) wins** for this library's add-indicators-frequently, mixed-pattern, mostly-memory-bound workload — and that the persistent layer should be deployed *narrowly*, only for ragged multi-symbol batches where a static grid leaves SMs idle in the tail.

---

## Confidence

**High (88%)** on the f32/scan-wiring/graph-fix priorities and the "targeted fusion beats megakernel" verdict — corroborated by primary NVIDIA docs (`01`–`04`) and the project's own `scan.rs`, `batch_graphs.rs`, `ma_advanced.rs`, and `orderflow_batch.rs` evidence. **Medium** on the exact megakernel-vs-cluster crossover and on per-laptop sustained TFLOPS — both must be measured on the actual RTX 3500 Ada at the decision gate, since they depend on the specific fused kernel's register/shared footprint and the OEM TGP cap (`01` §1 caveat; `03` confidence note).
