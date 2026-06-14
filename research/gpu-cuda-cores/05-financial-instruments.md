# Financial Instruments & Their GPU Compute Mapping

Research date: 2026-06-14
Scope: Map each major asset class to its contract specifications and the numerical compute it demands, then derive what a unified, instrument-aware GPU compute core (Rust + CUDA, targeting RTX 3500 Ada / sm_89) must provide. Grounded in standard quant references and the NVIDIA CUDA programming/tuning guides.

---

## 1. Why instrument specs matter to a GPU kernel

A GPU kernel does not care that a number represents the S&P 500; it cares about *layout*, *control flow*, and *arithmetic*. Instrument heterogeneity attacks exactly those three axes:

1. **Multipliers / tick sizes** turn a "price" into a P&L scalar (`pnl = (exit - entry) * multiplier * contracts`). A batch mixing ES ($50/pt) and CL ($1000/pt) needs per-instrument scalars — a parameter-broadcast problem.
2. **Settlement / funding / expiry** introduce *time-dependent cash flows* (funding accruals, roll, dividend adjustments) that are absent for spot equities — a branching / per-class code-path problem.
3. **Pricing math** ranges from O(1) closed forms (European Greeks) to O(N²) lattices (American early exercise) — a workload-imbalance problem.

On the SIMT hardware, divergent per-instrument code paths inside a warp are serialized: "if threads within a warp take different paths on conditional branches, execution of those paths becomes serialized... in the worst case, only 1 of the 32 threads makes progress per cycle" ([Warp Divergence overview, ScienceDirect](https://www.sciencedirect.com/topics/computer-science/thread-divergence)). So instrument-awareness is not a data-modeling nicety — it is the dominant determinant of kernel efficiency.

---

## 2. Asset classes: contract specs and compute demanded

### 2.1 Equities (cash/spot)
- **Specs**: tick typically $0.01 (sub-penny for some venues), multiplier = 1 share, T+1 settlement (US moved to T+1 in 2024), no expiry, discrete dividends, corporate actions (splits) require historical price adjustment.
- **Compute**: technical indicators (SMA/EMA, RSI, ATR, VWAP), rolling/windowed reductions, correlation/cointegration matrices for stat-arb, factor regressions. All are embarrassingly parallel, memory-bandwidth-bound, branch-light — ideal GPU fodder. (This repo already shows indicators are 1.2–2.9x on GPU and moving averages favor vectorized CPU.)

### 2.2 Futures
- **Specs (CME)**: ES = 0.25 pt tick, $50/pt multiplier → $12.50/tick; NQ = 0.25 pt, $20/pt → $5/tick; CL = $0.01 tick, 1000 bbl → $10/tick. Quarterly expiry codes H/M/U/Z on the third Friday; financial futures cash-settle, commodities physically deliver ([QuantVPS futures tick cheatsheet](https://www.quantvps.com/blog/futures-tick-cheatsheet); [Optimus contract specs](https://learn.optimusfutures.com/contract-specifications-and-values)).
- **Compute**: same indicator suite as equities **plus** continuous-contract construction (roll/stitch at expiry), basis/carry calc, and per-contract P&L using the multiplier. The multiplier and tick are *per-symbol constants* that must travel with each row.

### 2.3 Options
- **Specs**: strike grid, expiry, call/put, exercise style (European vs American), multiplier (US equity options = 100 shares), settlement (physical vs cash). Style determines the algorithm.
- **Compute — European**: every Greek has a closed form. Price uses N(d₁),N(d₂); Δ = e^(−rτ)N(d₁), Γ = e^(−rτ)n(d₁)/(Sσ√τ), Vega = Se^(−rτ)√τ·n(d₁), Θ and ρ likewise — "every single Greek has an analytic solution" ([Macroption Black-Scholes formulas](https://www.macroption.com/black-scholes-formula/); [Greeks (finance), Wikipedia](https://en.wikipedia.org/wiki/Greeks_(finance))). This is O(1) per option, perfectly parallel: one thread per (strike,expiry) cell. The only non-trivial kernel piece is a vectorized erf/normal-CDF.
- **Compute — American**: "an exact analytical expression... does not exist"; early exercise is a free-boundary optimization solved by binomial trees, finite-difference PDE, or Longstaff–Schwartz Monte Carlo ([DayTrading.com American vs European](https://www.daytrading.com/american-options-vs-european-options-mathematical-modeling); [UT Austin binomial American notes](https://web.ma.utexas.edu/users/mcudina/binomial_american.pdf)). Cost is O(N²) for an N-step lattice with sequential backward induction — a dependency chain that maps poorly to one-thread-per-step but well to one-thread-per-option (each thread owns a full small tree).
- **Vol surface**: implied vol per strike/expiry, then a surface fit. The standard arbitrage-free parameterization is Gatheral–Jacquier SVI, which guarantees absence of calendar-spread and butterfly arbitrage with a closed-form representation ([Gatheral & Jacquier, Arbitrage-free SVI, arXiv:1204.0646](https://arxiv.org/abs/1204.0646)). IV inversion is a per-cell Newton/Brent root-find (variable iteration count → **divergence risk**); SVI calibration is a small nonlinear least-squares per expiry slice.

### 2.4 Forex (spot/FX)
- **Specs**: pip = 0.0001 (0.01 for JPY pairs), standard lot = 100,000 base units → $10/pip on EURUSD, mini = 10,000, micro = 1,000; T+2 settlement; positions held past the value date accrue a swap/rollover reflecting the interest-rate differential, charged on Wednesdays at triple ([Afterprime spot forex glossary](https://afterprime.com/glossary/spot-forex); [TIOmarkets lot sizes](https://tiomarkets.com/article/forex-lot-sizes-explained-how-to-calculate-position-size)).
- **Compute**: indicators + carry/swap accrual + cross-rate triangulation (EURJPY from EURUSD·USDJPY). Pip-value depends on the quote currency, so P&L needs a per-pair conversion factor — another per-instrument scalar, sometimes itself a function of another live rate.

### 2.5 Crypto (perpetuals & spot)
- **Spot**: like equities, 24/7, no settlement lag.
- **Perpetuals**: no expiry; price anchored to spot via **funding**. Binance: `FundingRate = PremiumIndex + clamp(InterestRate − PremiumIndex, −0.05%, 0.05%)`, interest ~0.01% per 8h, settled every 8h (00:00/08:00/16:00 UTC); positive rate → longs pay shorts. Mark price uses a median of index-based fair price, basis MA and contract price to resist liquidation manipulation ([Coinbase funding rates](https://www.coinbase.com/learn/perpetual-futures/understanding-funding-rates-in-perpetual-futures); [Binance index/mark price docs](https://developers.binance.com/docs/derivatives/coin-margined-futures/market-data/rest-api/Index-Price-and-Mark-Price)).
- **Compute**: indicators + funding accrual (clamp, premium-index reductions over the order book) + mark-price median + index = volume-weighted average across exchanges. The clamp and median are cheap but introduce small per-element branches.

### 2.6 CFDs
- **Specs**: cash-settled derivative on an underlying (share/index/commodity/FX), usually **open-ended, no fixed maturity**; longs pay an overnight financing rate (benchmark + spread), shorts may pay or receive; dividend/corporate-action adjustments are passed through ([Wikipedia, Contract for difference](https://en.wikipedia.org/wiki/Contract_for_difference); [Standard Bank CFD product page](https://webtrading.standardbank.com/webtrader/products/contracts-for-difference.html)).
- **Compute**: underlying's indicator/pricing math + daily overnight-financing accrual + dividend adjustment events. Computationally a "spot underlying + financing overlay."

### 2.7 Indices
- **Specs**: not directly tradable; accessed via index futures, index options, index CFDs, or ETFs. Index CFDs are price-adjusted when a constituent goes ex-dividend, with the weighted dividend credited to longs/debited to shorts ([LiteFinance dividend adjustment](https://www.litefinance.org/trading/trading-instruments/cfd-xetra/dividend-adjustment/)).
- **Compute**: index level reconstruction (weighted sum of constituents — a large reduction), then whichever wrapper instrument's math applies. The constituent weighting is a sparse matrix-vector product, very GPU-friendly.

---

## 3. The heterogeneous-batch problem on SIMT hardware

A real portfolio batch mixes all of the above. Three concrete GPU pathologies arise:

**(a) Parameter broadcast.** Multiplier, tick, pip-value, lot size, funding interval, and expiry differ per row. Storing them inline as an Array-of-Structures (`struct Instrument{price; mult; tick; ...}`) wrecks coalescing: with AoS, "consecutive CUDA threads access memory locations that are not consecutive... memory accesses cannot be coalesced," whereas SoA lets "each thread do coalesced access for continuous memory address" ([NVIDIA forums, SoA vs AoS](https://forums.developer.nvidia.com/t/structures-of-arrays-vs-arrays-of-structures/13581)). **Verdict: store every per-instrument scalar in its own contiguous array (SoA).** The repo's existing INT8-quantized, `[strategy, tick, feature]` orderflow layout is already SoA-shaped — extend that discipline to instrument metadata.

**(b) Control-flow divergence.** A naive kernel `if(class==OPTION_AM) tree(); else if(class==PERP) funding(); else indicator();` serializes per branch inside each warp. Cost of divergence is real and measurable ([Benchmarking thread divergence in CUDA, arXiv:1504.01650](https://arxiv.org/pdf/1504.01650)). **Mitigations:**
   - *Stream compaction / sort-by-class*: group rows so each warp (32 lanes) is homogeneous; then the `if` is uniform across the warp and divergence vanishes. This is the single highest-leverage technique.
   - *Kernel-per-class dispatch*: launch separate kernels (European-Greeks kernel, American-tree kernel, funding kernel) over compacted sub-batches; lets you tune block size and registers per workload.
   - *Predication* only for short, balanced branches (the clamp in funding, JPY-pip selection) where the compiler converts the branch to masked execution cheaply.

**(c) Workload imbalance.** O(1) European cells and O(N²) American trees in one launch cause tail effects: a few long-running threads keep an SM busy while the rest idle. **Mitigation:** route American/lattice/MC work to its own kernel with persistent-thread or work-queue scheduling, keep the closed-form sweep separate. On sm_89 each SM has 128 FP32 cores, a 256 KB register file and 128 KB combined L1/shared ([NVIDIA Ada GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)); a small binomial tree (≤256 steps) fits in registers/shared per thread, so "one option per thread" keeps occupancy high without spilling.

---

## 4. What a unified, instrument-aware compute core needs

Synthesizing the above into requirements for the Rust+CUDA core:

1. **Instrument descriptor table (SoA, device-resident).** Columns: `class_id, multiplier, tick_size, pip_value_ccy, lot_size, settlement_lag, funding_interval, funding_clamp, expiry_ts, exercise_style, settlement_type`. Keep numeric scalars in parallel `f32`/`i32` arrays; never inline as AoS structs in the hot path.

2. **Class-tagged dispatch with pre-sort.** Maintain a stable `class_id` per row and a sort/segment step (`thrust::sort_by_key` or a counting-sort) that yields contiguous, warp-aligned class segments. Launch one tuned kernel per segment. This turns divergence into uniform-branch execution.

3. **A small library of numeric primitives, each its own kernel:**
   - vectorized indicators (already present),
   - vectorized normal CDF/PDF + erf for European Greeks (closed-form, one thread/cell),
   - IV root-finder with a **fixed iteration budget + fallback** (cap Newton steps to bound divergence; flag non-convergence rather than looping unboundedly),
   - SVI/eSSVI slice calibration (small NLS per expiry),
   - binomial/PDE/LSMC engine for American/path-dependent, one instrument per thread,
   - funding/roll/swap/dividend accrual reducer (time-cash-flow overlay).

4. **A cash-flow overlay layer** that, after pricing, applies per-class time effects uniformly: futures roll at expiry, perp funding every 8h, FX swap on value date, CFD/index overnight financing and ex-div adjustments. Express these as data (interval, rate, clamp) consumed by *one* accrual kernel rather than as bespoke branches — data-driven, not code-driven heterogeneity.

5. **Per-instrument P&L scalarization.** All P&L flows through `pnl = Δprice * multiplier * size * fx_conv`, with `multiplier`/`fx_conv` pulled from the descriptor SoA. Quote-currency conversion (FX, crypto cross) must be resolvable from the same device tables to avoid host round-trips.

6. **Validation against canonical references.** European prices/Greeks vs Black-Scholes closed form; American vs a CPU binomial reference; SVI fits checked for butterfly/calendar arbitrage per Gatheral–Jacquier; funding vs the exchange formula. Bit-for-bit CPU fallback (the repo already keeps CPU/GPU dual paths) is the regression oracle.

---

## 5. Confidence & limitations

- **High confidence** on contract specs, the closed-form/numerical split for options, and the SoA + sort-by-class + kernel-per-class strategy — all cross-referenced against primary NVIDIA docs and standard quant sources.
- **Medium confidence** on exact funding/mark-price formulas: these are *venue-specific* (Binance shown here; Bybit/OKX/dYdX differ in clamp bounds, interval, and index basket). Any production core must parameterize them, not hardcode.
- **Limitations**: exchange spec values drift (CME relists micros, equity options add weeklies/dailies, US equity settlement moved to T+1); treat all numeric specs as *configuration*, refreshed from venue reference data, never compiled in.

---

### Sources
- [QuantVPS — Futures Tick Values Cheatsheet](https://www.quantvps.com/blog/futures-tick-cheatsheet)
- [Optimus Futures — Contract Specifications and Values](https://learn.optimusfutures.com/contract-specifications-and-values)
- [Macroption — Black-Scholes Formulas and Greeks](https://www.macroption.com/black-scholes-formula/)
- [Wikipedia — Greeks (finance)](https://en.wikipedia.org/wiki/Greeks_(finance))
- [DayTrading.com — American vs European Options Modeling](https://www.daytrading.com/american-options-vs-european-options-mathematical-modeling)
- [UT Austin — American Options (Binomial)](https://web.ma.utexas.edu/users/mcudina/binomial_american.pdf)
- [Gatheral & Jacquier — Arbitrage-free SVI Volatility Surfaces (arXiv:1204.0646)](https://arxiv.org/abs/1204.0646)
- [Afterprime — Spot Forex Glossary](https://afterprime.com/glossary/spot-forex)
- [TIOmarkets — Forex Lot Sizes Explained](https://tiomarkets.com/article/forex-lot-sizes-explained-how-to-calculate-position-size)
- [Coinbase — Understanding Funding Rates in Perpetual Futures](https://www.coinbase.com/learn/perpetual-futures/understanding-funding-rates-in-perpetual-futures)
- [Binance — Index Price and Mark Price](https://developers.binance.com/docs/derivatives/coin-margined-futures/market-data/rest-api/Index-Price-and-Mark-Price)
- [Wikipedia — Contract for Difference](https://en.wikipedia.org/wiki/Contract_for_difference)
- [LiteFinance — CFD Dividend Adjustment](https://www.litefinance.org/trading/trading-instruments/cfd-xetra/dividend-adjustment/)
- [NVIDIA — Ada GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [ScienceDirect — Warp/Thread Divergence overview](https://www.sciencedirect.com/topics/computer-science/thread-divergence)
- [Benchmarking the cost of thread divergence in CUDA (arXiv:1504.01650)](https://arxiv.org/pdf/1504.01650)
- [NVIDIA Developer Forums — Structures of Arrays vs Arrays of Structures](https://forums.developer.nvidia.com/t/structures-of-arrays-vs-arrays-of-structures/13581)
