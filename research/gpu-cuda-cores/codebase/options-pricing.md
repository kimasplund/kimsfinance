# GPU/CUDA Cores Analysis: Options & Derivatives Pricing

**Scope:** Heston pricing, Greeks, options strategies GPU cores
**Target HW:** NVIDIA RTX 3500 Ada (sm_89, 12 GB), CUDA 13.1, driver 595.x
**Mode:** Read-only analysis (no source modified)
**Repo:** `/home/kim/projects/kimsfinance`

---

## 1. Core Inventory

### 1.1 Host orchestration (Rust)

| File | LOC | Role |
|------|-----|------|
| `rust/src/gpu/heston_pricing.rs` | 1209 | Main Heston GPU pricer. Owns the only GPU kernel actually launched in this path (`heston_characteristic_function`). All FFT / Lewis integration / price extraction runs **on the CPU**. Manages pinned + device buffers. |
| `rust/src/quantitative/heston/greeks_gpu.rs` | 624 | GPU Greeks calculator. Computes 5 Greeks via finite differences. **Calls `price_options` 8× per batch** (base + spot± + vol± + time + rate±), then launches 5 tiny element-wise difference kernels. |
| `rust/src/quantitative/heston/strategies_gpu.rs` | 880 | Launchers for straddle / covered-call / iron-condor signal+P&L kernels. 2D grid `(candles × strategies)`, block `(256,4)`. |
| `rust/src/quantitative/heston/strategies_vol_arbitrage.rs` | 694 | Launchers for vol-arbitrage and delta-neutral kernels. |
| `rust/src/gpu/compile.rs` | (host) | NVRTC compile options. `arch=sm_89`, `use_fast_math`, `ftz`, `prec_sqrt=false`, `prec_div=false` — all FP32-throughput tuning, but the pricing kernels are FP64 (see §2.1). |

### 1.2 Device kernels (CUDA `.cu`)

| File | LOC | Entry point(s) | Status | What it does |
|------|-----|----------------|--------|--------------|
| `cuda/heston/characteristic_function.cu` | 251 | `heston_characteristic_function` | **LIVE** | Per `(option, frequency)` thread: full Gatheral-stable Heston CF in FP64 complex. Writes `[n_options × n_fft]` real+imag arrays. |
| `cuda/heston/carr_madan_weight.cu` | 165 | `carr_madan_weight` | **DEAD** | Carr-Madan weighting (discount × CF / denom × Simpson). Only referenced by an NVRTC-compatibility unit test (`heston_pricing.rs:1062-1085`); never compiled into a `CudaFunction` or launched. |
| `cuda/heston/extract_prices.cu` | 142 | `extract_prices` | **DEAD** | Extracts price from FFT bin (incl. an **O(n_fft) linear search** per option for the closest log-strike bin). Never launched. |
| `cuda/greeks/delta.cu` | 54 | `calculate_delta_kernel` | LIVE | `(up-down)/(2ΔS)`, clamp `[-1,1]`. |
| `cuda/greeks/gamma.cu` | 57 | `calculate_gamma_kernel` | LIVE | `(up-2·mid+down)/ΔS²`, clamp `≥0`. |
| `cuda/greeks/vega.cu` | 52 | `calculate_vega_kernel` | LIVE | `(up-down)/(2·0.01)`. |
| `cuda/greeks/theta.cu` | 48 | `calculate_theta_kernel` | LIVE | `-(tomorrow-now)/1`. |
| `cuda/greeks/rho.cu` | 47 | `calculate_rho_kernel` | LIVE | `(up-down)/(2·0.01)`. |
| `cuda/strategies/straddle.cu` | 170 | `straddle_signals_kernel`, `short_straddle_signals_kernel` | LIVE | Long/short straddle entry signals. |
| `cuda/strategies/covered_call.cu` | 184 | `covered_call_signals_kernel`, `covered_call_pnl_kernel` | LIVE | Covered-call signals + P&L. |
| `cuda/strategies/iron_condor.cu` | 261 | `iron_condor_signals_kernel`, `iron_condor_pnl_kernel` | LIVE | 4-leg iron condor signals + P&L. |
| `cuda/strategies/vol_arbitrage.cu` | 293 | `vol_arbitrage_signals_kernel`, `vol_arbitrage_pnl_kernel`, `vol_edge_monitor_kernel` | LIVE | IV/HV vol-arb signals, P&L, edge monitor. |
| `cuda/strategies/delta_neutral.cu` | 221 | `delta_neutral_signals_kernel`, `delta_neutral_rebalance_kernel` | LIVE | Delta-neutral hedging signals. |

**Out of scope / not options-pricing:** `rust/src/gpu/fibonacci.rs` (708 LOC) is a rolling-max/min retracement *indicator*, not derivatives pricing — no pricing, Greeks, or CF math. Excluded from optimization ranking below except as a possible co-tenant for a megakernel batch (§4).

**Stale artifacts present:** `heston_pricing.rs.backup`, `heston_pricing.rs.orig` (867 LOC each), `characteristic_function.cu.backup` (198 LOC). Not compiled; flagged for cleanup.

---

## 2. Current Optimization State (evidence-based)

### 2.1 Precision: FP64 everywhere — fights the hardware

Every pricing/Greeks/strategy kernel uses `double`. The CF kernel does FP64 **complex** `exp/log/sqrt/atan2/sin/cos` per thread:
- `characteristic_function.cu:71-89` — `Complex::sqrt/exp/log` built on `::sqrt`, `atan2`, `::exp`, `cos`, `sin` (all FP64 transcendentals).
- `characteristic_function.cu:172-246` — the full CF: ~6 complex mults, 3 complex divs, 2 complex exps, 1 complex log, 1 complex sqrt, per `(option, frequency)` thread.

On **Ada (sm_89) the FP64:FP32 throughput ratio is 1:64**. The compile options in `compile.rs:137-157` (`use_fast_math`, `ftz`, `prec_sqrt=false`, `prec_div=false`, and the comment at `:138-139` boasting "128 FP32 ops/cycle … 2x vs Ampere") are tuned for FP32 — yet **none of that helps an FP64 kernel**. Fast-math intrinsics (`__expf`, `__logf`, `__sincosf`) and the 128 FP32 cores are simply not reached. This is the single largest mismatch between the code and the target hardware.

### 2.2 Memory pattern

- CF output is `[n_options × n_fft]` FP64 = **for 100 options × 4096 = 6.4 MB per array × 2 (real+imag) = ~13 MB**, fully written by the kernel, fully D2H-copied, then consumed on CPU (`heston_pricing.rs:544-563`). The kernel comment at `carr_madan_weight.cu:23-24` even acknowledges "Eliminates need to download 6.5 MB of CF data to CPU" — but that elimination was never wired up.
- Indexing `idx = option_idx * n_fft + phi_idx` (`characteristic_function.cu:152`) with `phi_idx` on the x-thread is **coalesced** along the FFT axis — good.
- Greeks/strategy kernels are simple coalesced 1D/2D element-wise reads — memory-bound and fine as written (`delta.cu:36-53`, etc.).

### 2.3 Fusion: essentially none on the hot path

- The CF kernel writes raw CF; **weighting, FFT, and extraction all happen on CPU** via `rustfft` (`heston_pricing.rs:885-1049`, `price_with_lewis_method:695-838`). The two GPU kernels that *would* fuse weighting/extraction (`carr_madan_weight.cu`, `extract_prices.cu`) are dead code (§1.2).
- Greeks: 5 separate trivial kernels (`delta/gamma/vega/theta/rho`) each launched separately with its own H2D of prices+spots and D2H of results (`greeks_gpu.rs:307-513`). These are perfectly fusable into one kernel.

### 2.4 Persistent kernels / CUDA graphs: not used here

CUDA-graph infrastructure exists (`rust/src/gpu/cuda_graphs.rs`, `rust/src/gpu/batch_graphs.rs`) but is **not referenced** anywhere in `heston_pricing.rs`, `greeks_gpu.rs`, or the strategy launchers (grep: zero hits). The Greeks path issues ~8 pricer invocations + 5 kernel launches + ~20 H2D/D2H copies + many `stream.synchronize()` calls **per batch**, with no graph capture/replay.

### 2.5 Launch pattern & synchronization — the dominant cost

Concrete evidence in `greeks_gpu.rs:148-165`:
```
let (prices_base, prices_spot_up, prices_spot_down) = calculate_spot_bumped_prices(...);  // 3× price_options
let (prices_vol_up, prices_vol_down)              = calculate_vol_bumped_prices(...);     // 2× price_options
let prices_tomorrow                               = calculate_time_bumped_prices(...);    // 1× price_options
let (prices_rate_up, prices_rate_down)            = calculate_rate_bumped_prices(...);    // 2× price_options
```
That is **8 full Heston pricings** for one Greeks batch. Each `price_options` →
`price_with_pageable_memory` (`heston_pricing.rs:468-572`) does: 4 pinned-pool acquires, 4 fresh `alloc_buffer`, 4 H2D, kernel launch, **a `stream.synchronize()` inside `launch_kernel` (`:652`)**, then 2 D2H of the 6.4 MB CF arrays + **another `stream.synchronize()` (`:558`)**, plus CPU Lewis/FFT integration. So one Greeks batch ≈ **16 blocking syncs + 16 large D2H transfers + 8× CPU integration loops**, fully serialized.

`launch_kernel` synchronizes immediately after every launch (`:651-654`), so even the 8 pricings cannot overlap. The "<5 ms / 1000 options" target in the header docs is not credibly reachable with this structure.

### 2.6 Numerical / correctness notes (not perf, but relevant)

- `extract_prices.cu:103-111` does an **O(n_fft) linear scan** to find the nearest log-strike bin — pointless; the bin is `round((k+b)·N/λ)`, O(1). (Dead code today, but if revived this is a 4096× waste per option.)
- The pinned download fast-path (`price_with_pinned_memory:384-465`) is **disabled** due to a `dtoh_pinned` bug (`heston_pricing.rs:324-326`, `:378-383`), so the slower pageable path runs unconditionally.
- `strikes` is passed to the CF kernel but unused (`characteristic_function.cu:154-157`) — harmless, just ABI padding.
- Greeks `price_options` re-derives `time_to_expiry` from `chrono::Utc::now()` repeatedly; the time bump subtracts one day from `expiration` and re-prices fully rather than bumping a scalar.

---

## 3. Ranked Optimization Opportunities

Ranking = (Impact) × (Effort). Impact estimated against the current serialized FP64 baseline.

### HIGH impact

**H1 — Fuse the full pricing pipeline on-GPU; stop downloading CF arrays.** *(Effort: Med-High)*
Revive/replace the dead `carr_madan_weight.cu` + `extract_prices.cu` and add cuFFT (CUDA 13.1 ships cuFFT; it is **not** currently a dependency — `Cargo.toml:85-86` only has CPU `rustfft`). Pipeline becomes: `CF kernel → carr_madan_weight kernel → cuFFT (batched) → extract_prices kernel → D2H of n_options scalars`. Eliminates the ~13 MB/batch D2H (`heston_pricing.rs:544-563`) and the per-option CPU integration loops (`:710-835`). Evidence the design intent already existed: `carr_madan_weight.cu:23-24`. Alternatively keep Lewis but do the integration in a reduction kernel rather than CPU. This is the structural fix that unlocks everything else.

**H2 — FP32 (or mixed-precision) CF kernel for Ada.** *(Effort: Med)*
FP64 on sm_89 runs at 1/64 FP32 rate; the CF kernel (`characteristic_function.cu:172-246`) is transcendental-heavy and is the arithmetic bottleneck. A single-precision (or FP32-compute / FP64-accumulate) variant with `__expf/__logf/__sincosf` fast intrinsics can plausibly give an order-of-magnitude kernel speedup, and finally makes the existing `use_fast_math`/sm_89 compile flags (`compile.rs:137-157`) actually matter. Validate against the existing Lewis-vs-BS test (`heston_pricing.rs:1100-1156`, 5% tolerance) — option premia tolerate FP32 well; keep an FP64 path for calibration if needed.

**H3 — Replace 8× full re-pricing in Greeks with one bumped megakernel + remove per-launch syncs.** *(Effort: Med)*
`greeks_gpu.rs:148-165` re-prices the whole CF 8 times. Instead, evaluate the CF once over a bumped parameter grid (spot±, vol±, time, rate±) in a single kernel launch (extra grid dimension = bump-id), or at minimum drop the unconditional `stream.synchronize()` in `launch_kernel` (`heston_pricing.rs:651-654`) and pipeline the 8 pricings on one stream so they overlap. Combined with H1 this collapses ~16 syncs + 16 transfers to ~1.

### MEDIUM impact

**M1 — Fuse the 5 Greeks difference kernels into one.** *(Effort: Low)*
`delta/gamma/vega/theta/rho` are trivial element-wise ops over the same `n_options` index, launched separately with redundant H2D of prices/spots and 5 D2H (`greeks_gpu.rs:307-513`). One kernel taking all bumped price arrays → all 5 Greeks in a single launch removes 4 launches + most of the redundant copies.

**M2 — Capture the strategies/Greeks launch sequence in a CUDA graph.** *(Effort: Low-Med)*
Infra already exists (`gpu/cuda_graphs.rs`, `gpu/batch_graphs.rs`) but is unused on this path. The strategy launchers (`strategies_gpu.rs`) and the multi-kernel Greeks sequence are fixed-shape, repeated per backtest step — ideal for graph capture/replay to amortize launch overhead. Pair with persistent device buffers (today every strategy call does fresh `copy_to_device`/`allocate_device_buffer`/`copy_to_host` — 38 such calls in `strategies_gpu.rs`).

**M3 — O(1) strike-bin lookup if `extract_prices.cu` is revived.** *(Effort: Low)*
Replace the O(n_fft) scan (`extract_prices.cu:103-111`) with the closed-form `best_idx = round((k+b)·N/λ)`. Only matters once H1 makes this kernel live, but it is a 4096× per-option win when it does.

**M4 — Fix and re-enable the pinned D2H fast path.** *(Effort: Med)*
The disabled pinned path (`heston_pricing.rs:324-326, 378-465`) promises 20-30% faster transfers. Largely moot if H1 removes the big CF transfer entirely, so do H1 first; otherwise this is the cheapest transfer win.

### LOW impact

**L1 — Delete stale artifacts** (`*.rs.backup`, `*.rs.orig`, `*.cu.backup`) to prevent drift/confusion. *(Effort: trivial)*

**L2 — Drop the unused `strikes` arg from the CF kernel ABI** (`characteristic_function.cu:134, 154-157`; launch site `heston_pricing.rs:636`) — minor register/clarity gain. *(Effort: trivial)*

**L3 — Monte-Carlo pricing path (new capability, not an optimization of existing code).** *(Effort: High)*
The current method is Fourier (CF + FFT/Lewis), which is the right choice for European vanillas and does **not** benefit from tensor cores. A separate QMC/MC Heston engine (Andersen QE) would unlock path-dependent payoffs and could use cuRAND + warp-level reductions; FP32 MC maps well to Ada's FP32 cores. This is roadmap, not a fix.

---

## 4. Fusion / Combine Potential (megakernel & batching)

**Strong megakernel candidate — the Heston pricing pipeline (H1).** The four-stage chain `CF → weight → FFT → extract` is the canonical fusion target. Three of the four stages are pure element-wise/reduction ops over the same `[n_options × n_fft]` grid and can share one kernel (CF + weight fused; extract fused with the FFT-output reduction). Only the FFT itself must stay a cuFFT batched call. This keeps the entire `[n_options × n_fft]` working set in device memory, eliminating the ~13 MB/batch round-trip.

**Strong batch candidate — Greeks bump grid (H3 + M1).** All 8 bump scenarios differ only by a scalar perturbation to `(S, v0, T, r)`. They share strikes, expirations, and Heston params. A single CF launch over an added "bump" grid dimension, followed by **one** fused finite-difference kernel producing all 5 Greeks, replaces 8 pricings + 5 kernels + ~20 transfers with ~2 launches.

**Strategies share a common ABI shape and can be co-scheduled.** Every strategy kernel (`straddle`, `covered_call`, `iron_condor`, `vol_arbitrage`, `delta_neutral`) uses the identical 2D `(candle_idx, strategy_idx)` indexing, `(256,4)` block, and reads from `underlying_prices / option_prices / vols / strategy_params`. They are independent per `(candle, strategy)` cell, so:
- They can run **concurrently on separate streams** (or as nodes in one CUDA graph, M2) since they consume the same inputs and write disjoint outputs.
- A "signal megakernel" could compute several strategies in one launch (branch by `strategy_type` param) to amortize the per-launch H2D of the shared market-data arrays — useful when backtesting many strategy families over the same candle series.

**Greeks → strategies coupling.** `vol_arbitrage` and `delta_neutral` consume `option_deltas`/`option_vegas` (`vol_arbitrage.cu:58-59`, `delta_neutral.cu:54`). Today those come from the separate Greeks path. A fully fused backtest graph would: price (H1) → Greeks (H3/M1) → strategy signals → P&L, all as one captured CUDA graph (M2) with no intermediate D2H, only final signal/P&L scalars copied back.

**Tensor cores:** Not applicable to the current Fourier pricing or finite-difference Greeks (no dense GEMM-shaped work). They would only become relevant under an MC/QMC engine (L3) for batched payoff/regression steps, or if a neural surrogate pricer were introduced — both are new capabilities, not fusions of existing kernels.

---

## 5. Summary Table

| ID | Opportunity | Impact | Effort | Key evidence |
|----|-------------|--------|--------|--------------|
| H1 | Fuse on-GPU pipeline (revive weight+extract, add cuFFT) | High | Med-High | `carr_madan_weight.cu` & `extract_prices.cu` dead; CPU FFT at `heston_pricing.rs:885-1049`; 13 MB D2H at `:544-563` |
| H2 | FP32 / mixed-precision CF kernel for Ada (1:64 FP64 penalty) | High | Med | FP64 throughout `characteristic_function.cu:172-246`; FP32 compile flags wasted `compile.rs:137-157` |
| H3 | One bumped megakernel for Greeks; drop per-launch sync | High | Med | 8× re-pricing `greeks_gpu.rs:148-165`; sync `heston_pricing.rs:651-654` |
| M1 | Fuse 5 Greeks diff kernels into 1 | Med | Low | 5 separate launches `greeks_gpu.rs:307-513` |
| M2 | CUDA-graph capture for strategies/Greeks sequence | Med | Low-Med | Unused infra `gpu/cuda_graphs.rs`, `gpu/batch_graphs.rs`; 38 alloc/copy calls in `strategies_gpu.rs` |
| M3 | O(1) strike-bin lookup | Med | Low | O(n_fft) scan `extract_prices.cu:103-111` |
| M4 | Fix/re-enable pinned D2H path | Med | Med | Disabled `heston_pricing.rs:324-326, 378-465` |
| L1 | Delete stale `*.backup/*.orig` | Low | Trivial | 3 stale files, ~1900 LOC |
| L2 | Drop unused `strikes` arg | Low | Trivial | `characteristic_function.cu:154-157` |
| L3 | New MC/QMC engine (tensor-core relevant) | Low(now) | High | No MC path exists; Fourier-only |
