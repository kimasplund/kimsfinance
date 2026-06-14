# Heston Debug Quick Reference

## Run Test (Copy-Paste)
```bash
cd /home/kim/projects/kimsfinance/rust
cargo run --example test_heston_debug --features heston --release 2>&1 | tee heston_debug_output.txt
```

## What to Look For

### Critical Values That MUST Have Non-Zero Imaginary Parts

| Variable | Expected | Why |
|----------|----------|-----|
| **z** | `(u_real, -2.5)` | Input is complex by construction |
| **i_z** | `(2.5, u_real)` | i × (u, -2.5) = (2.5, u) |
| **z_squared** | `(?, ?)` | Complex square has imaginary part |
| **d_squared** | `(?, ?)` | Contains complex terms |
| **d** | `(?, ?)` | Square root of complex number |
| **g** | `(?, ?)` | Ratio of complex numbers |
| **D** | `(?, ?)` | Complex coefficient |
| **C** | `(?, ?)` | Complex coefficient |
| **phi** | `(?, ?)` | Final characteristic function |

### First Three Steps Are Critical

1. **z = (u, -2.5)** ← Input, should be correct
2. **i_z = i × z** ← First complex operation
3. **z_squared = z × z** ← Tests complex multiplication

**If i_z or z_squared have zero imag → Bug in complex multiplication operator**

## Debugging Decision Tree

```
Is z.imag = -2.5?
  NO → Check z construction (line 159)
  YES ↓

Is i_z.imag ≠ 0?
  NO → Bug in Complex operator* (lines 44-48)
  YES ↓

Is z_squared.imag ≠ 0?
  NO → Bug in Complex operator* (lines 44-48)
  YES ↓

Is d_squared.imag ≠ 0?
  NO → Check arithmetic (lines 205-213)
  YES ↓

Is d.imag ≠ 0?
  NO → Bug in Complex::sqrt() (lines 71-76)
  YES ↓

Is g.imag ≠ 0?
  NO → Bug in Complex operator/ (lines 62-68)
  YES ↓

Is D.imag ≠ 0?
  NO → Check D calculation (lines 244-250)
  YES ↓

Is C.imag ≠ 0?
  NO → Bug in Complex::log() (lines 85-89) or arithmetic
  YES ↓

Is phi.imag ≠ 0?
  NO → Bug in Complex::exp() (lines 79-82)
  YES → SUCCESS! (but why is output still zero?)
```

## Expected First Few Values

For typical input (S=100, K=100, T=1, r=0.05, α=1.5, u=0):

```
z = (0.000000, -2.500000)  ← u=0, imag=-2.5
i_z = (2.500000, 0.000000)  ← i × (0, -2.5) = (2.5, 0)
```

**WAIT! i_z should be (2.5, 0) not (?, non-zero)?**

Let me verify: i × (0, -2.5) = (0, 1) × (0, -2.5)
Using (a+bi)(c+di) = (ac-bd) + (ad+bc)i:
- a=0, b=1, c=0, d=-2.5
- real = 0×0 - 1×(-2.5) = 2.5 ✓
- imag = 0×(-2.5) + 1×0 = 0 ✓

**OK, so i_z CAN have zero imaginary for u=0!**

Next check: z² = (0, -2.5)² = (0, -2.5) × (0, -2.5)
- a=0, b=-2.5, c=0, d=-2.5
- real = 0×0 - (-2.5)×(-2.5) = -6.25 ✓
- imag = 0×(-2.5) + (-2.5)×0 = 0 ✓

**z_squared also has zero imaginary for u=0!**

## Key Insight

For u=0 (first FFT point), many intermediate values will have zero imaginary parts.
**We need to check a non-zero u value!**

### Better Test: Check u≠0 FFT Points

Look at debug output for idx>0 (second FFT point where u≠0).
The kernel only prints idx==0, so we need to modify the printf condition!

## Action Items

1. **Modify CUDA kernel** to print idx==1 or idx==16 (where u≠0)
2. **Verify** that for u≠0, we get non-zero imaginary parts throughout
3. **Check** if the problem is that ALL FFT points have u=0 (phi_values array issue)

## Quick Formula Check

Carr-Madan frequency points:
```
u[j] = j × η
where η = λ / N, λ = ln(strike_high / strike_low)

For single strike: η = 2π / (N × Δk) where Δk is log-strike spacing
```

**Check**: Are phi_values being set correctly?
**Check**: Is eta being computed correctly?
**Check**: Are we only getting u=0 for all points?
