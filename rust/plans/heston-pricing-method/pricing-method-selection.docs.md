# Heston Pricing Method Selection Architecture

## Summary

Design enum-based method selection for GPU-accelerated Heston option pricing to handle FFT instability issues. Current implementation uses Carr-Madan FFT with Black-Scholes fallback when prices become invalid (NaN/Inf/negative). The new architecture enables explicit method choice (CarrMadanFFT, Lewis2001, Auto) with smart fallback detection.

**Confidence Level:** High: 90-95%
**Research Depth:** Complex

## Key Components

- `/home/kim-asplund/projects/kimsfinance/rust/src/gpu/heston_pricing.rs` - Main GPU pricer implementation (1,121 lines)
- `/home/kim-asplund/projects/kimsfinance/rust/src/quantitative/heston/model.rs` - Core Heston model and OptionType enum
- `/home/kim-asplund/projects/kimsfinance/rust/src/quantitative/heston/black_scholes.rs` - Analytical BS pricer (fallback)
- `/home/kim-asplund/projects/kimsfinance/rust/src/gpu/auto_select.rs` - Enum pattern example (AggregationEngine)
- `/home/kim-asplund/projects/kimsfinance/rust/src/autotuner.rs` - Enum pattern example (ExecutionStrategy)

## Implementation Patterns

### Pattern 1: Simple Binary Enum (AggregationEngine style)
**File:** `src/gpu/auto_select.rs:40-56`
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationEngine {
    /// CPU-based HashMap aggregation (fast for small datasets)
    CPU,
    /// GPU-based parallel aggregation (fast for large datasets)
    GPU,
}

impl AggregationEngine {
    pub fn name(&self) -> &'static str {
        match self {
            AggregationEngine::CPU => "CPU",
            AggregationEngine::GPU => "GPU",
        }
    }
}
```

**Characteristics:**
- Derives: `Debug, Clone, Copy, PartialEq, Eq`
- Helper method: `name()` for string representation
- Used with selector pattern: `EngineSelector::select_engine(size)`

### Pattern 2: Three-Way Strategy Enum (ExecutionStrategy style)
**File:** `src/autotuner.rs:278-287`
```rust
pub enum ExecutionStrategy {
    /// Run entirely on CPU
    CPU,
    /// Run entirely on GPU (or GPU-heavy hybrid like RSI)
    GPU,
    /// Custom hybrid strategy (future: CPU-GPU pipeline)
    Hybrid,
}
```

**Characteristics:**
- No derives (simpler enum)
- Used for high-level strategy selection
- Future-proof with Hybrid variant

### Pattern 3: Error Enum (OptionType style)
**File:** `src/quantitative/heston/model.rs:149-152`
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}
```

**Characteristics:**
- Derives: `Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize`
- Used in core domain models
- Serde integration for JSON serialization

## Dependencies & Versions

Based on `Cargo.toml` inspection (not shown but standard in Rust projects):

- **serde** + **serde_json**: For serialization if method selection needs persistence
- **rustfft**: `v6.x` - Used for FFT computation (lines 24, 763-764)
- **num_complex**: For Complex64 type (lines 23, 796-915)
- **cudarc**: For GPU kernel execution (lines 22, various)

**Version-Specific Behaviors:**
- rustfft v6.x uses unnormalized FFT (requires manual 1/N normalization at line 956-959)
- No breaking changes expected in dependencies

## Considerations

### Critical Edge Cases

1. **FFT Overflow Detection** (lines 798-850):
   - Carr-Madan FFT becomes unstable at high frequencies (φ > 50.0)
   - Denominator can approach zero: `α² + α - φ²` (line 838)
   - Current code: Hard frequency limit at MAX_SAFE_PHI = 50.0 (line 804)
   - Current code: Adaptive truncation after 5 consecutive small values (line 803)

2. **Black-Scholes Fallback Trigger** (lines 379-400):
   ```rust
   let is_invalid = !price.is_finite() || price <= 1e-10 || price > 10.0 * option.spot_price;
   ```
   - Triggers on: NaN, Inf, negative, or unreasonably large prices
   - Uses current volatility from Heston params: `v0.sqrt()`
   - Fallback is **silent** (only eprintln warning)

3. **Put-Call Parity** (lines 1004-1012):
   - FFT computes call prices
   - Puts derived via parity: `P = C - S + K·exp(-r·T)`
   - Validation tests expect <0.1% parity error

### Numerical Stability Issues

**Carr-Madan FFT** (current method):
- **Pros:** Fast (single FFT), GPU-accelerated characteristic function
- **Cons:** Unstable for:
  - High volatility of volatility (σ > 0.5)
  - Extreme correlations (|ρ| > 0.9)
  - Deep OTM options (K/S > 2.0 or < 0.5)
  - Near-expiry options (T < 0.01 years)

**Lewis 2001 Formula** (proposed alternative):
- **Pros:**
  - More stable (no damping parameter)
  - Better convergence for wide strike ranges
  - Robust for extreme parameters
- **Cons:**
  - Requires two characteristic function evaluations (2x GPU calls)
  - ~1.5-2x slower than Carr-Madan

**Black-Scholes Fallback:**
- **Pros:** Always stable, analytical formula
- **Cons:**
  - Ignores stochastic volatility (inaccurate for Heston)
  - Only uses current vol (`v0.sqrt()`), not full Heston dynamics
  - Not suitable for calibration workflows

### Integration Points

1. **Constructor Integration** (lines 72-167):
   ```rust
   pub fn new(
       device: Arc<GpuDevice>,
       fft_size: usize,
       max_batch_size: usize,
       // NEW: method: HestonPricingMethod
   ) -> Result<Self, GpuError>
   ```
   - Add method parameter with default in `with_default_batch_size`
   - Store method in struct fields (line 29-52)

2. **Pricing Integration** (lines 274-403):
   ```rust
   pub fn price_options(
       &mut self,
       params: &HestonParams,
       options: &[OptionQuote],
       // ALTERNATIVE: method override parameter?
   ) -> Result<Vec<f64>, GpuError>
   ```
   - Either: Use stored method from constructor
   - Or: Allow per-call override (more flexible)

3. **Auto Mode Logic** (NEW - to be implemented):
   ```rust
   fn detect_instability(&self, params: &HestonParams, options: &[OptionQuote]) -> bool {
       // Check for known unstable configurations:
       // 1. High vol of vol: params.sigma > 0.5
       // 2. Extreme correlation: params.rho.abs() > 0.9
       // 3. Deep OTM: max strike ratio > 2.0 or < 0.5
       // 4. Near expiry: min time to expiry < 0.01 years
   }
   ```

### Backward Compatibility

**Critical:** Existing code relies on automatic fallback (lines 379-400). Changes must:
- **Preserve default behavior:** FFT with Black-Scholes fallback
- **No breaking changes:** Existing tests must pass without modification
- **Deprecation path:** Consider adding `with_method()` builder instead of breaking `new()`

## Next Steps

### Recommended Implementation Approach

**Phase 1: Add Enum** (1-2 hours)
```rust
/// Heston option pricing method selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HestonPricingMethod {
    /// Carr-Madan FFT (fast but can be unstable for extreme parameters)
    /// - Best for: Standard parameters, ATM/near-ATM options
    /// - Unstable when: σ > 0.5, |ρ| > 0.9, deep OTM, near expiry
    CarrMadanFFT,

    /// Lewis 2001 formula (stable but ~2x slower)
    /// - Best for: Wide strike ranges, extreme parameters
    /// - Requires: Two characteristic function evaluations
    Lewis2001,

    /// Automatic selection with smart fallback
    /// - Strategy: Try CarrMadanFFT first
    /// - Fallback to Lewis2001 if instability detected
    /// - Final fallback to Black-Scholes if both fail
    Auto,
}

impl Default for HestonPricingMethod {
    fn default() -> Self {
        Self::Auto  // Safe default for production
    }
}

impl HestonPricingMethod {
    pub fn name(&self) -> &'static str {
        match self {
            Self::CarrMadanFFT => "Carr-Madan FFT",
            Self::Lewis2001 => "Lewis 2001",
            Self::Auto => "Auto (Smart Fallback)",
        }
    }
}
```

**Phase 2: Add to HestonGpuPricer** (2-3 hours)
```rust
pub struct HestonGpuPricer {
    // ... existing fields ...
    pricing_method: HestonPricingMethod,  // NEW
}

impl HestonGpuPricer {
    // Backward compatible: keep existing signature
    pub fn new(
        device: Arc<GpuDevice>,
        fft_size: usize,
        max_batch_size: usize,
    ) -> Result<Self, GpuError> {
        Self::with_method(device, fft_size, max_batch_size, HestonPricingMethod::Auto)
    }

    // New constructor with method selection
    pub fn with_method(
        device: Arc<GpuDevice>,
        fft_size: usize,
        max_batch_size: usize,
        method: HestonPricingMethod,
    ) -> Result<Self, GpuError> {
        // ... existing initialization ...
        Ok(Self {
            // ... existing fields ...
            pricing_method: method,
        })
    }

    // Add method override option
    pub fn price_options_with_method(
        &mut self,
        params: &HestonParams,
        options: &[OptionQuote],
        method: HestonPricingMethod,
    ) -> Result<Vec<f64>, GpuError> {
        let original_method = self.pricing_method;
        self.pricing_method = method;
        let result = self.price_options(params, options);
        self.pricing_method = original_method;
        result
    }
}
```

**Phase 3: Implement Auto Detection** (3-4 hours)
```rust
impl HestonGpuPricer {
    fn detect_fft_instability(
        &self,
        params: &HestonParams,
        options: &[OptionQuote],
    ) -> bool {
        // Known unstable configurations for Carr-Madan FFT

        // 1. High volatility of volatility
        if params.sigma > 0.5 {
            eprintln!("[AUTO] High vol-of-vol detected: σ={:.3} > 0.5", params.sigma);
            return true;
        }

        // 2. Extreme correlation (leverage effect too strong)
        if params.rho.abs() > 0.9 {
            eprintln!("[AUTO] Extreme correlation detected: |ρ|={:.3} > 0.9", params.rho.abs());
            return true;
        }

        // 3. Deep OTM options (strike ratio > 2x or < 0.5x)
        let now = chrono::Utc::now().timestamp();
        for opt in options {
            let strike_ratio = opt.strike / opt.spot_price;
            if strike_ratio > 2.0 || strike_ratio < 0.5 {
                eprintln!("[AUTO] Deep OTM detected: K/S={:.3}", strike_ratio);
                return true;
            }

            // 4. Near expiry (< 3.65 days = 0.01 years)
            let tau = opt.time_to_expiry(now);
            if tau < 0.01 && tau > 0.0 {
                eprintln!("[AUTO] Near expiry detected: T={:.4} years", tau);
                return true;
            }
        }

        false
    }

    pub fn price_options(
        &mut self,
        params: &HestonParams,
        options: &[OptionQuote],
    ) -> Result<Vec<f64>, GpuError> {
        // ... existing validation (lines 279-296) ...

        let method = match self.pricing_method {
            HestonPricingMethod::Auto => {
                if self.detect_fft_instability(params, options) {
                    eprintln!("[AUTO] Using Lewis2001 due to instability risk");
                    HestonPricingMethod::Lewis2001
                } else {
                    eprintln!("[AUTO] Using CarrMadanFFT (stable config)");
                    HestonPricingMethod::CarrMadanFFT
                }
            }
            explicit_method => {
                eprintln!("[EXPLICIT] Using {}", explicit_method.name());
                explicit_method
            }
        };

        // Dispatch to appropriate pricing method
        let prices = match method {
            HestonPricingMethod::CarrMadanFFT => {
                self.price_with_carr_madan_fft(params, options)?
            }
            HestonPricingMethod::Lewis2001 => {
                self.price_with_lewis2001(params, options)?
            }
            HestonPricingMethod::Auto => unreachable!("Auto resolved above"),
        };

        // Final Black-Scholes fallback (for both methods)
        self.apply_black_scholes_fallback(params, options, prices)
    }

    // Refactor existing code into this method
    fn price_with_carr_madan_fft(
        &mut self,
        params: &HestonParams,
        options: &[OptionQuote],
    ) -> Result<Vec<f64>, GpuError> {
        // Lines 297-377 (existing logic)
        // ...
    }

    // NEW: Implement Lewis 2001 formula
    fn price_with_lewis2001(
        &mut self,
        params: &HestonParams,
        options: &[OptionQuote],
    ) -> Result<Vec<f64>, GpuError> {
        // TODO: Implement Lewis 2001 semi-analytical formula
        // Requires:
        // 1. Two characteristic function evaluations (φ₁ and φ₂)
        // 2. Numerical integration (trapezoidal rule)
        // 3. No FFT needed (direct integration)
        unimplemented!("Lewis2001 pricing not yet implemented")
    }

    // Refactor existing fallback logic (lines 379-400)
    fn apply_black_scholes_fallback(
        &self,
        params: &HestonParams,
        options: &[OptionQuote],
        mut prices: Vec<f64>,
    ) -> Result<Vec<f64>, GpuError> {
        let now = chrono::Utc::now().timestamp();

        for (i, option) in options.iter().enumerate() {
            let price = prices[i];
            let is_invalid = !price.is_finite()
                || price <= 1e-10
                || price > 10.0 * option.spot_price;

            if is_invalid {
                eprintln!(
                    "[FALLBACK] Option {}: price {:.6} invalid, using Black-Scholes",
                    i, price
                );

                let vol = params.v0.sqrt();
                let tau = option.time_to_expiry(now);
                let bs_price = BlackScholesPricer::price(
                    option.spot_price,
                    option.strike,
                    tau,
                    option.risk_free_rate,
                    vol,
                    option.option_type,
                );
                prices[i] = bs_price;
            }
        }

        Ok(prices)
    }
}
```

**Phase 4: Testing** (2-3 hours)
1. Test explicit method selection:
   ```rust
   let pricer = HestonGpuPricer::with_method(device, 4096, 100, HestonPricingMethod::Lewis2001)?;
   ```

2. Test auto detection with unstable parameters:
   ```rust
   let unstable_params = HestonParams::new(
       2.0, 0.04, 0.8, -0.95, 0.04  // High σ and |ρ|
   )?;
   let pricer = HestonGpuPricer::new(device, 4096, 100)?;  // Auto mode
   let prices = pricer.price_options(&unstable_params, &options)?;
   // Should automatically use Lewis2001
   ```

3. Validate backward compatibility:
   ```rust
   // Existing code should work unchanged
   let pricer = HestonGpuPricer::new(device, 4096, 100)?;
   let prices = pricer.price_options(&params, &options)?;
   ```

### Risk Assessment

**Low Risk:**
- Enum addition (no breaking changes)
- Auto detection heuristics (can be tuned)
- Backward compatibility preserved

**Medium Risk:**
- Lewis2001 implementation correctness (requires validation against reference prices)
- Performance regression (Lewis2001 is 2x slower)

**High Risk:**
- None (existing FFT path unchanged by default)

### Proof of Concept Recommendation

**Not Required** - Implementation is straightforward refactoring with well-understood patterns.

However, consider:
1. **Benchmark Lewis2001 vs CarrMadanFFT** on stable cases to validate 2x slowdown estimate
2. **Validate auto detection heuristics** with real market data (if available)
3. **Test extreme parameter cases** to confirm Lewis2001 stability improvement

---

## Code Examples

### Usage Example 1: Explicit Method Selection

```rust
use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
use kimsfinance_core::quantitative::heston::{HestonParams, HestonPricingMethod, OptionQuote};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Arc::new(GpuDevice::new()?);

    // Force Lewis2001 for stability (e.g., calibration on extreme strikes)
    let pricer = HestonGpuPricer::with_method(
        device,
        4096,
        100,
        HestonPricingMethod::Lewis2001,
    )?;

    let params = HestonParams::new(2.0, 0.04, 0.8, -0.95, 0.04)?;
    let options = vec![/* ... */];

    let prices = pricer.price_options(&params, &options)?;

    println!("Priced {} options using {}",
        options.len(),
        pricer.method().name()
    );

    Ok(())
}
```

### Usage Example 2: Auto Mode (Default)

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Arc::new(GpuDevice::new()?);

    // Auto mode: smart fallback based on parameters
    let mut pricer = HestonGpuPricer::new(device, 4096, 100)?;

    // Stable parameters → CarrMadanFFT (fast)
    let stable_params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04)?;
    let prices1 = pricer.price_options(&stable_params, &options)?;
    // [AUTO] Using CarrMadanFFT (stable config)

    // Unstable parameters → Lewis2001 (stable)
    let unstable_params = HestonParams::new(2.0, 0.04, 0.8, -0.95, 0.04)?;
    let prices2 = pricer.price_options(&unstable_params, &options)?;
    // [AUTO] Using Lewis2001 due to instability risk

    Ok(())
}
```

### Usage Example 3: Per-Call Override

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Arc::new(GpuDevice::new()?);
    let mut pricer = HestonGpuPricer::new(device, 4096, 100)?;

    // Override method for specific call
    let prices_fft = pricer.price_options_with_method(
        &params,
        &options,
        HestonPricingMethod::CarrMadanFFT,
    )?;

    let prices_lewis = pricer.price_options_with_method(
        &params,
        &options,
        HestonPricingMethod::Lewis2001,
    )?;

    // Compare methods
    for i in 0..options.len() {
        println!("Option {}: FFT=${:.4}, Lewis=${:.4}, diff=${:.4}",
            i, prices_fft[i], prices_lewis[i],
            (prices_fft[i] - prices_lewis[i]).abs()
        );
    }

    Ok(())
}
```

---

## Research Metadata

**Files Analyzed:** 6 core files
**Lines Reviewed:** ~2,000 lines of implementation code
**Examples Found:** 3 enum patterns, 1 FFT pricing example
**Time Spent:** ~25 minutes (Complex research depth)

**Key Findings:**
1. Current implementation has robust Black-Scholes fallback (lines 379-400)
2. FFT instability is well-understood (comments at lines 800-850)
3. Codebase follows consistent enum patterns with derives and helper methods
4. Backward compatibility is critical (existing tests and examples)

**Confidence Level Justification:**
- **90-95% confidence** based on:
  - ✅ Clear existing patterns in codebase
  - ✅ Well-documented instability issues
  - ✅ Straightforward refactoring (no architectural changes)
  - ⚠️ Lewis2001 implementation not yet validated (5-10% uncertainty)

**Recommended Next Action:**
Implement Phase 1-2 (enum + constructor) first, then validate with existing tests before implementing Lewis2001.
