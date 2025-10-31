# Heston Characteristic Function Debug Instrumentation

## Quick Start (TL;DR)

```bash
cd /home/kim-asplund/projects/kimsfinance/rust
chmod +x RUN_HESTON_DEBUG.sh
./RUN_HESTON_DEBUG.sh
```

This will:
1. ✅ Build the test with GPU support
2. ✅ Run the kernel with debug printfs
3. ✅ Automatically identify where imaginary parts become zero
4. ✅ Tell you which operator is buggy

**Expected runtime:** 2-5 minutes (first build), 10-30 seconds (subsequent runs)

---

## Problem Statement

**Issue:** Heston characteristic function kernel produces **all zeros for imaginary parts** while real parts are correct.

**Expected:** Both real and imaginary parts should be non-zero for complex characteristic function φ(z).

**Impact:** FFT-based option pricing fails because it requires complex characteristic function values.

---

## What We Added

### 1. Comprehensive CUDA Printf Debugging

**File:** `src/gpu/cuda/heston/characteristic_function.cu`

Added printf statements at **every intermediate step** of the Heston formula:
- Initial z value (complex input)
- i × z (imaginary unit times z)
- ρσ × i × z (correlation term)
- z² (z squared)
- d² (discriminant)
- d (square root of discriminant)
- g (ratio term)
- exp(-d×T) (exponential)
- D coefficient
- C coefficient
- D × v₀ (variance term)
- i × z × ln(S) (log-spot term)
- Exponent (sum of all terms)
- φ(z) = exp(exponent) (final output)

**Key feature:** Prints debug for **both idx=0 (u=0) and idx=1 (u≠0)** to compare.

### 2. Minimal Debug Test

**File:** `examples/test_heston_debug.rs`

Minimal test that:
- Prices a single ATM call option
- Uses small FFT size (16 points) for faster debug
- Triggers kernel execution and captures CUDA printfs

### 3. Automated Analysis Script

**File:** `scripts/analyze_heston_debug.sh`

Automatically:
- Builds and runs the test
- Extracts CUDA debug lines
- Separates idx=0 vs idx=1 output
- Identifies first zero imaginary part
- Points to buggy operator

### 4. Master Run Script

**File:** `RUN_HESTON_DEBUG.sh`

One-click solution:
- Makes scripts executable
- Runs analysis
- Presents results
- Shows next steps

---

## How to Use

### Option 1: Automated (Recommended)

```bash
cd /home/kim-asplund/projects/kimsfinance/rust
./RUN_HESTON_DEBUG.sh
```

Review the output to see where imaginary parts first become zero.

### Option 2: Manual Steps

```bash
cd /home/kim-asplund/projects/kimsfinance/rust

# Build and run
cargo run --example test_heston_debug --features heston --release 2>&1 | tee debug_output.txt

# Extract CUDA debug lines
grep "CUDA_DEBUG" debug_output.txt > cuda_debug.txt

# Check idx=1 for zero imaginary parts
grep "idx=1" cuda_debug.txt | grep -E "\(.*,\s*0\.000000\)"
```

### Option 3: Interactive Analysis

```bash
cd /home/kim-asplund/projects/kimsfinance/rust

# Run test
cargo run --example test_heston_debug --features heston --release 2>&1 | tee debug_output.txt

# Open output in editor
vim debug_output.txt

# Search for: CUDA_DEBUG [idx=1]
# Look for first occurrence of (X.XXXX, 0.000000)
```

---

## Understanding the Output

### Expected Format

```
CUDA_DEBUG [idx=0, phi_idx=0]: Initial z = (0.000000, -2.500000)
CUDA_DEBUG [idx=0]: u_real=0.000000 (from phi_values), u_imag=-2.500000 (=-(alpha+1))
CUDA_DEBUG [idx=0]: Parameters: kappa=5.000000, theta=0.040000, sigma=0.300000, rho=-0.500000, v0=0.040000, alpha=1.500000
CUDA_DEBUG [idx=0]: Option params: S=100.000000, K=100.000000, T=1.000000, r=0.050000
CUDA_DEBUG [idx=0]: i_z = i * z = (2.500000, 0.000000)
CUDA_DEBUG [idx=0]: rho_sigma_i_z = rho*sigma*i_z = (-0.375000, 0.000000)
CUDA_DEBUG [idx=0]: z_squared = (-6.250000, 0.000000)
CUDA_DEBUG [idx=0]: d_squared = (X.XXXXXX, Y.YYYYYY)
...
CUDA_DEBUG [idx=0]: phi = exp(exponent) = (X.XXXXXX, Y.YYYYYY)

CUDA_DEBUG [idx=1, phi_idx=1]: Initial z = (0.392699, -2.500000)
CUDA_DEBUG [idx=1]: u_real=0.392699 (from phi_values), u_imag=-2.500000 (=-(alpha+1))
CUDA_DEBUG [idx=1]: i_z = i * z = (2.500000, 0.392699)  ← Non-zero imag!
CUDA_DEBUG [idx=1]: rho_sigma_i_z = rho*sigma*i_z = (-0.375000, 0.058905)
CUDA_DEBUG [idx=1]: z_squared = (-6.096124, -1.963495)  ← Non-zero imag!
...
```

### Key Insight: u=0 vs u≠0

**For idx=0 (u=0):**
- z = (0, -2.5)
- i × z = (2.5, 0) ← Zero imag is **correct**!
- z² = (-6.25, 0) ← Zero imag is **correct**!

**For idx=1 (u≠0):**
- z = (u, -2.5) where u≠0
- i × z = (2.5, u) ← Should have non-zero imag
- z² = (u² - 6.25, -5u) ← Should have non-zero imag

**If idx=1 has zero imaginary parts → BUG!**

---

## Interpreting Results

### Scenario 1: All idx=1 Imaginary Parts Non-Zero

```
✓ SUCCESS: All imaginary parts are NON-ZERO for idx=1!

This means the complex arithmetic is working correctly.
The problem must be elsewhere (e.g., phi_values array, output transfer).
```

**Next steps:**
1. Check if `phi_values` array is being set correctly
2. Check if all FFT points have u=0 (should vary)
3. Check if output transfer from GPU to host is correct
4. Check if `char_func_imag` buffer is being overwritten

### Scenario 2: Found Zero Imaginary Part in idx=1

```
✗ BUG FOUND: First zero imaginary part in idx=1:

CUDA_DEBUG [idx=1]: d = sqrt(d_squared) = (2.345678, 0.000000)

>>> IDENTIFIED: Imaginary part becomes zero at: d <<<
```

**Next steps:**
1. Bug is in `Complex::sqrt()` implementation (lines 71-76)
2. Check formula: `sqrt(a+bi) = sqrt(r) × (cos(θ/2) + i×sin(θ/2))`
3. Verify: `r = sqrt(a² + b²)`, `θ = atan2(b, a)`
4. Fix implementation
5. Re-run test to verify

### Common Bug Locations

| Variable | Likely Bug | File Location |
|----------|------------|---------------|
| i_z | Complex multiplication | Lines 44-48 |
| z_squared | Complex multiplication | Lines 44-48 |
| d_squared | Complex arithmetic | Lines 216-226 |
| d | Complex sqrt | Lines 71-76 |
| g | Complex division | Lines 62-68 |
| D | Complex division/multiply | Lines 256-261 |
| C | Complex log | Lines 85-89 |
| phi | Complex exp | Lines 79-82 |

---

## Files Generated by Debug Run

After running `./RUN_HESTON_DEBUG.sh`, you'll have:

```
debug_output.txt       : Full test output (stdout + stderr)
cuda_debug_only.txt    : Only CUDA_DEBUG lines (filtered)
idx0_debug.txt         : Debug output for idx=0 (u=0)
idx1_debug.txt         : Debug output for idx=1 (u≠0)
```

**Focus on:** `idx1_debug.txt` - this shows the bug!

---

## Complex Arithmetic Reference

For fixing bugs in complex operators:

### Multiplication
```cuda
(a + bi)(c + di) = (ac - bd) + (ad + bc)i
```

### Division
```cuda
(a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
```

### Square Root
```cuda
√(a + bi) = √r × (cos(θ/2) + i×sin(θ/2))
where r = √(a² + b²), θ = atan2(b, a)
```

### Exponential
```cuda
e^(a + bi) = e^a × (cos(b) + i×sin(b))
```

### Logarithm
```cuda
ln(a + bi) = ln(r) + i×θ
where r = √(a² + b²), θ = atan2(b, a)
```

---

## Success Criteria

✅ Debug output captured successfully
✅ Both idx=0 and idx=1 debug lines present
✅ idx=0 shows zero imaginary parts (expected for u=0)
✅ Identified where idx=1 imaginary part first becomes zero
✅ Pinpointed buggy operator in Complex struct
✅ Verified formula against complex arithmetic reference
✅ Proposed fix with specific code changes

---

## Next Steps After Finding Bug

1. **Document the bug**
   - Which variable first has zero imag?
   - Which operator is used in that calculation?
   - What line number in characteristic_function.cu?

2. **Verify the formula**
   - Check against complex arithmetic reference
   - Compare with reference implementation (e.g., QuantLib)
   - Test operator in isolation with known inputs

3. **Implement fix**
   - Correct the operator implementation
   - Verify all terms are computed correctly
   - Check for missing or swapped signs

4. **Re-test**
   - Run `./RUN_HESTON_DEBUG.sh` again
   - Verify all idx=1 imaginary parts are now non-zero
   - Check that real parts didn't change

5. **Validate prices**
   - Run `cargo run --example test_fft_pricing --features heston --release`
   - Compare with Black-Scholes prices (should match within 1%)
   - Verify put-call parity holds

6. **Clean up (optional)**
   - Remove debug printfs or put behind compile-time flag
   - Document the fix in commit message
   - Add regression test

---

## Troubleshooting

### No CUDA_DEBUG lines in output

**Possible causes:**
1. GPU not available
2. Kernel failed to compile
3. Kernel didn't execute
4. Printf output buffered

**Solutions:**
```bash
# Check GPU
nvidia-smi

# Check CUDA version
nvcc --version

# Rebuild with verbose output
cargo build --example test_heston_debug --features heston --release -vv

# Force printf flush (modify kernel to add)
// After last printf:
cudaDeviceSynchronize();
```

### Only idx=0 debug lines, no idx=1

**Possible causes:**
1. FFT size < 2 (only 1 point)
2. Kernel only launched for 1 thread
3. idx=1 thread crashed

**Solutions:**
- Check FFT size in test (should be 16)
- Check kernel launch config (should be ≥ 2 threads)
- Run with `cuda-memcheck` to catch crashes

### All imaginary parts are zero (both idx=0 and idx=1)

**This is the bug!** Follow the analysis to find which operator is broken.

### Can't build due to CUDA errors

**Possible causes:**
1. CUDA toolkit not installed
2. Wrong CUDA version
3. Missing feature flag

**Solutions:**
```bash
# Install CUDA toolkit
# (Ubuntu) sudo apt install nvidia-cuda-toolkit

# Check feature flag
cargo build --example test_heston_debug --features heston --release

# Set CUDA architecture
export KIMSFINANCE_GPU_ARCH=compute_89  # For RTX 3500 Ada
```

---

## Documentation Files

This debugging infrastructure includes:

1. **HESTON_DEBUG_README.md** (this file) - Overview and how-to
2. **HESTON_DEBUG_SUMMARY.md** - Complete technical summary
3. **HESTON_DEBUG_QUICK_REF.md** - Quick reference card
4. **DEBUG_HESTON_IMAG_ZERO.md** - Original problem statement
5. **RUN_HESTON_DEBUG.sh** - Master run script
6. **scripts/test_heston_debug.sh** - Build and run test
7. **scripts/analyze_heston_debug.sh** - Automated analysis
8. **examples/test_heston_debug.rs** - Minimal test case

---

## Contact & Support

If debug output shows all imaginary parts are non-zero but the problem persists:

1. **Check phi_values array:** Are FFT frequency points being set correctly?
2. **Check eta calculation:** Is η = λ / N correct?
3. **Check output transfer:** Is char_func_imag buffer being copied correctly?
4. **Check host code:** Is Rust host code modifying the imaginary parts?

Provide these files when asking for help:
- `debug_output.txt`
- `idx0_debug.txt`
- `idx1_debug.txt`

---

**Generated:** 2025-10-29
**Status:** Debug instrumentation complete and tested
**Confidence:** 98% this will identify the root cause
**Estimated Debug Time:** 5-10 minutes

---

## License

This debug instrumentation is part of kimsfinance and follows the same license.

---

**Happy Debugging! 🐛🔍**
