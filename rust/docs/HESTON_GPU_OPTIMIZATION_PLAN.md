# Heston GPU Characteristic Function Optimization Plan

**Status**: Phase 1 In Progress  
**Timeline**: 1 week  
**Current Performance**: 40-45μs per option (100 options in 4.5ms)  
**Target Performance**: <30μs per option (100 options in <3ms)  

---

## Current Bottleneck Analysis

### Benchmark Results (Baseline)
```
Single option:   4.96ms  (includes kernel compilation)
10 options:      0.092ms (9.2μs per option)
100 options:     4.49ms  (44.9μs per option) ❌ Target: <3ms
500 options:     20.29ms (40.6μs per option) ❌ Target: <10ms
```

### Root Causes

1. **Memory Transfer Overhead** (Est. 30-40% of time)
   - Using pageable memory (20-30% slower than pinned)
   - 5 H2D transfers + 2 D2H transfers per pricing call
   - No buffer reuse between calls

2. **FFT Pricing Placeholder** (Not Real Pricing)
   - Returns mid prices, not characteristic function prices
   - No Carr-Madan formula implementation

3. **Kernel Not Optimized** (Est. 10-20% overhead)
   - No shared memory for Heston parameters
   - Complex arithmetic not using fast math
   - Suboptimal block size

4. **Launch Overhead** (For Calibration Only)
   - Each pricing call launches new kernel
   - For 1000s of calibration calls, launch overhead dominates

---

## 4-Phase Optimization Plan

### Phase 1: Pinned Memory (20-30% speedup) ⚙️ IN PROGRESS
**Target**: 100 options in <3.5ms (from 4.49ms)

**Implementation**:
- Pre-allocate pinned host buffers for `max_batch_size` options
- Pre-allocate device buffers (eliminate per-call allocation)
- Use `htod_pinned()` and `dtoh_pinned()` for transfers
- Graceful fallback to pageable memory if pinned allocation fails

**Code Changes**:
1. `HestonGpuPricer` struct: Add pinned buffer fields
2. `new()` constructor: Allocate pinned + device buffers
3. `price_options()`: Use pre-allocated buffers with pinned transfers
4. Add helper methods for buffer management

**Expected Result**: 100 options in 3.2-3.5ms

---

### Phase 2: Optimized CUDA Kernel (10-20% speedup)
**Target**: 100 options in <3ms

**Implementation**:
- Add shared memory for Heston parameters (40 bytes)
- Use fast math (`__fmul_rn`, `__fadd_rn`, `__fma_rn`)
- Optimize complex arithmetic (inline operations)
- Tune block size (currently 256, may benefit from 512)

**Kernel Changes** (`characteristic_function.cu`):
```cuda
extern "C" __global__ void heston_characteristic_function_optimized(
    // ... parameters ...
) {
    // Shared memory for Heston params (40 bytes, accessed by all threads)
    __shared__ double s_kappa;
    __shared__ double s_theta;
    __shared__ double s_sigma;
    __shared__ double s_rho;
    __shared__ double s_v0;
    
    if (threadIdx.x == 0) {
        s_kappa = kappa;
        s_theta = theta;
        s_sigma = sigma;
        s_rho = rho;
        s_v0 = v0;
    }
    __syncthreads();
    
    // Use shared params instead of re-reading from global memory
    // Use __fmul_rn() for faster multiplication
    // ...
}
```

**Expected Result**: 100 options in 2.8-3.0ms

---

### Phase 3: Carr-Madan FFT Pricing (Correctness)
**Target**: Correct option prices (not mid prices)

**Options**:
1. **rustfft** (CPU): Simple, portable, ~0.5-1ms overhead
2. **cuFFT** (GPU): Faster, complex integration, ~0.2-0.4ms

**Recommendation**: Start with rustfft for correctness, migrate to cuFFT if needed

**Implementation** (rustfft):
```rust
use rustfft::{FftPlanner, num_complex::Complex};

fn fft_to_option_prices(
    &self,
    char_func_real: &[f64],
    char_func_imag: &[f64],
    options: &[OptionQuote],
) -> Result<Vec<f64>, GpuError> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(self.fft_size);
    
    let mut prices = Vec::with_capacity(options.len());
    
    for (i, option) in options.iter().enumerate() {
        // Extract characteristic function for this option
        let start = i * self.fft_size;
        let end = start + self.fft_size;
        
        let mut char_func: Vec<Complex<f64>> = char_func_real[start..end]
            .iter()
            .zip(&char_func_imag[start..end])
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        
        // Apply FFT
        fft.process(&mut char_func);
        
        // Carr-Madan formula: C(K) = exp(-αk) / π * Re(∫ e^(-iφk) ψ(φ) dφ)
        let alpha = 1.5; // Damping factor
        let k = (option.strike / option.spot_price).ln();
        
        let mut integral_sum = 0.0;
        for (j, cf) in char_func.iter().enumerate() {
            let phi = (j as f64) * 2.0 * PI / (self.fft_size as f64);
            let damping = (-alpha * k).exp();
            integral_sum += damping * cf.re;
        }
        
        let price = (integral_sum / PI) * option.spot_price;
        prices.push(price.max(0.0)); // Price cannot be negative
    }
    
    Ok(prices)
}
```

**Expected Result**: Real option prices

---

### Phase 4: Persistent Kernel (For Calibration)
**Target**: 1000x pricing calls with <1ms amortized overhead

**When to Use**:
- Calibration workloads (100s-1000s of pricing calls)
- Parameter sweeps
- **NOT** for single pricing calls (overhead not worth it)

**Implementation**:
- Integrate with existing `persistent::PersistentKernelManager`
- Single kernel launch, process all parameter sets in loop
- Use cooperative groups for synchronization

**Expected Result**: 1000 pricing calls in <100ms (from ~4000ms)

---

## Implementation Priority

1. ✅ **Phase 1** (pinned memory) - Highest ROI, lowest risk
2. ✅ **Phase 2** (kernel optimization) - Medium ROI, low risk
3. ⚠️ **Phase 3** (FFT pricing) - Correctness critical, medium risk
4. ⏭️ **Phase 4** (persistent kernel) - Only for calibration, defer if not needed

---

## Risk Assessment

**Phase 1 Risks**:
- Pinned allocation may fail (graceful fallback implemented ✅)
- Memory limit (~50% of system RAM for pinned buffers)

**Phase 2 Risks**:
- Shared memory bank conflicts (unlikely with 40 bytes)
- Fast math accuracy (negligible for option pricing)

**Phase 3 Risks**:
- FFT size tuning (too small = inaccurate, too large = slow)
- Damping factor α sensitivity (need to validate against Black-Scholes)

**Phase 4 Risks**:
- Cooperative launch support (RTX 3500 Ada supports it ✅)
- Max grid size limits (need to query at runtime)

---

## Success Criteria

- [x] Profile current implementation (Done: 40-45μs/option baseline)
- [ ] Phase 1: 100 options in <3.5ms with pinned memory
- [ ] Phase 2: 100 options in <3ms with optimized kernel
- [ ] Phase 3: Correct option prices (validate against Black-Scholes)
- [ ] Phase 4 (Optional): 1000 calibration calls in <100ms

---

## Next Steps

1. Complete Phase 1 implementation
2. Update `test_heston_pricer.rs` to use new API (`HestonGpuPricer::new(device, 4096, 1000)`)
3. Benchmark and validate 20-30% speedup
4. Proceed to Phase 2 if Phase 1 successful

---

**Last Updated**: 2025-10-29  
**Owner**: Rust Latency Optimizer Agent
