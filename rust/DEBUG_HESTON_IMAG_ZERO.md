# Heston Characteristic Function Debug Investigation

## Problem Statement

The Heston characteristic function kernel is producing **all zeros for imaginary parts** while real parts are non-zero and correct.

**Current Output:**
- Real part: 90489, 38754, -57665, etc. (non-zero, correct)
- Imaginary part: ALL ZEROS (bug!)

**Expected Behavior:**
- Both real and imaginary parts should be non-zero for complex characteristic function
- The Carr-Madan FFT formula requires z = u - (α+1)i, which should produce complex output

## Debug Instrumentation Added

### File Modified
`src/gpu/cuda/heston/characteristic_function.cu`

### Debug Printfs Added (Thread 0 Only)

The kernel now prints these intermediate values at each step of the computation:

1. **Initial z value** (line 161-166)
   - `z.real`, `z.imag`
   - Parameters: kappa, theta, sigma, rho, v0, alpha
   - Option params: S, K, T, r

2. **i_z = i × z** (line 184-186)
   - Result of multiplying imaginary unit by z

3. **rho_sigma_i_z = ρσ × i_z** (line 191-194)
   - Correlation and vol-of-vol term

4. **z_squared** (line 199-201)
   - z × z

5. **d_squared** (line 215-217)
   - Intermediate for square root: (ρσiz - b)² + σ²(2iz - z²)

6. **d = sqrt(d_squared)** (line 221-223)
   - Square root of d_squared

7. **g** (line 231-233)
   - Ratio term: (b - ρσiz - d) / (b - ρσiz + d)

8. **exp_neg_d_T** (line 238-241)
   - exp(-d × T)

9. **D** (line 250-252)
   - D(T,z) coefficient

10. **C** (line 267-269)
    - C(T,z) coefficient

11. **D_v0** and **iz_ln_S** (line 275-278)
    - Components of final exponent

12. **exponent = C + D×v0 + iz×ln(S)** (line 282-285)
    - Total exponent before exp()

13. **phi = exp(exponent)** (line 289-291)
    - Final characteristic function value

## How to Run Debug Test

### Option 1: Minimal Test (Fastest)
```bash
cd /home/kim-asplund/projects/kimsfinance/rust
cargo run --example test_heston_debug --features heston --release 2>&1 | grep CUDA_DEBUG
```

### Option 2: Full Test Script
```bash
cd /home/kim-asplund/projects/kimsfinance/rust
bash scripts/test_heston_debug.sh
```

### Option 3: Original FFT Test
```bash
cd /home/kim-asplund/projects/kimsfinance/rust
cargo run --example test_fft_pricing --features heston --release 2>&1 | grep CUDA_DEBUG
```

## Expected Debug Output Format

```
CUDA_DEBUG: Initial z = (0.000000, -2.500000)
CUDA_DEBUG: Parameters: kappa=5.000000, theta=0.040000, sigma=0.300000, rho=-0.500000, v0=0.040000, alpha=1.500000
CUDA_DEBUG: Option params: S=100.000000, K=100.000000, T=1.000000, r=0.050000
CUDA_DEBUG: i_z = i * z = (2.500000, 0.000000)
CUDA_DEBUG: rho_sigma_i_z = rho*sigma*i_z = (-0.375000, 0.000000)
CUDA_DEBUG: z_squared = (-6.250000, 0.000000)
CUDA_DEBUG: d_squared = (X.XXXXXX, Y.YYYYYY)
CUDA_DEBUG: d = sqrt(d_squared) = (X.XXXXXX, Y.YYYYYY)
CUDA_DEBUG: g = (X.XXXXXX, Y.YYYYYY)
CUDA_DEBUG: exp_neg_d_T = exp(-d*T) = (X.XXXXXX, Y.YYYYYY)
CUDA_DEBUG: D = (X.XXXXXX, Y.YYYYYY)
CUDA_DEBUG: C = (X.XXXXXX, Y.YYYYYY)
CUDA_DEBUG: D_v0 = (X.XXXXXX, Y.YYYYYY)
CUDA_DEBUG: iz_ln_S = (X.XXXXXX, Y.YYYYYY)
CUDA_DEBUG: exponent = C + D*v0 + iz*ln(S) = (X.XXXXXX, Y.YYYYYY)
CUDA_DEBUG: phi = exp(exponent) = (X.XXXXXX, Y.YYYYYY)
```

## Analysis Strategy

### Step 1: Find First Zero Imaginary Part
Run the debug test and examine output line by line:
- **If `i_z` has zero imag**: Bug in complex multiplication operator (line 44-48)
- **If `z_squared` has zero imag**: Bug in complex multiplication operator (line 44-48)
- **If `d_squared` has zero imag**: Bug in complex arithmetic (lines 205-213)
- **If `d` has zero imag**: Bug in complex sqrt (lines 71-76)
- **If `g` has zero imag**: Bug in complex division (lines 62-68)
- **If `D` has zero imag**: Bug in complex division or multiplication (lines 244-250)
- **If `C` has zero imag**: Bug in complex log or arithmetic (lines 254-267)
- **If `phi` has zero imag**: Bug in complex exp (lines 79-82)

### Step 2: Root Cause Identification
Once we find which calculation first loses the imaginary part, we can:
1. Examine the exact formula at that line
2. Check if the complex operator implementation is correct
3. Verify input values are as expected
4. Check for any numerical edge cases

### Step 3: Fix Implementation
Based on root cause, fix the buggy operator or calculation.

## Complex Arithmetic Reference

### Complex Multiplication (should preserve imaginary parts)
```cuda
(a + bi)(c + di) = (ac - bd) + (ad + bc)i
```

### Complex Division (should preserve imaginary parts)
```cuda
(a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
```

### Complex Square Root (should preserve imaginary parts)
```cuda
√(a + bi) = √r × (cos(θ/2) + i×sin(θ/2))
where r = √(a² + b²), θ = atan2(b, a)
```

### Complex Exponential (should preserve imaginary parts)
```cuda
e^(a + bi) = e^a × (cos(b) + i×sin(b))
```

### Complex Logarithm (should preserve imaginary parts)
```cuda
ln(a + bi) = ln(r) + i×θ
where r = √(a² + b²), θ = atan2(b, a)
```

## Success Criteria

✅ Debug output shows exactly where imaginary part first becomes zero
✅ Root cause identified with exact line number and operation
✅ Mathematical formula verified against reference implementation
✅ Bug fix proposed with specific code changes

## Next Steps After Debug

1. **Capture full debug output** from test run
2. **Identify first zero imaginary part** in the sequence
3. **Examine operator implementation** at that step
4. **Verify formula correctness** against Heston literature
5. **Implement fix** in the buggy operator
6. **Re-test** to confirm imaginary parts are now non-zero
7. **Validate prices** match Black-Scholes in limit case

---

**Generated:** 2025-10-29
**Status:** Debug instrumentation complete, ready to run test
**Files Modified:**
- `src/gpu/cuda/heston/characteristic_function.cu` (added printf debugging)
- `examples/test_heston_debug.rs` (new minimal test)
- `scripts/test_heston_debug.sh` (new test script)
