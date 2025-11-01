# FP16 Format Reference

Quick reference for IEEE 754 binary16 (half-precision) floating-point format.

## Bit Layout

```
┌─────┬──────────────┬────────────────────────────┐
│  S  │   Exponent   │         Mantissa           │
│ (1) │     (5)      │           (10)             │
├─────┼──────────────┼────────────────────────────┤
│ 15  │   14 - 10    │          9 - 0             │
└─────┴──────────────┴────────────────────────────┘

Total: 16 bits (2 bytes)
```

### Components

| Component | Bits | Range | Description |
|-----------|------|-------|-------------|
| **Sign** | 1 | 0-1 | 0 = positive, 1 = negative |
| **Exponent** | 5 | 0-31 | Biased by 15 (actual: -14 to +15) |
| **Mantissa** | 10 | 0-1023 | Fraction part (implicit leading 1 for normalized) |

## Encoding Rules

### Normalized Numbers (Exponent 1-30)

**Value**: `(-1)^S × 2^(E-15) × (1 + M/1024)`

- **S**: Sign bit
- **E**: Exponent (stored value)
- **M**: Mantissa (stored value)

**Example**: `0 10000 0000000000` (binary) = `1.0` (FP16)
- Sign: 0 (positive)
- Exponent: 16 (stored) → 16 - 15 = 1 (actual)
- Mantissa: 0 → 1.0 (with implicit leading 1)
- Value: `+1 × 2^1 × 1.0 = 2.0` ... wait, that's wrong!

**Corrected Example**: `0 01111 0000000000` (binary) = `1.0` (FP16)
- Sign: 0 (positive)
- Exponent: 15 (stored) → 15 - 15 = 0 (actual)
- Mantissa: 0 → 1.0 (with implicit leading 1)
- Value: `+1 × 2^0 × 1.0 = 1.0` ✓

### Special Values

#### Zero (Exponent = 0, Mantissa = 0)
```
Positive Zero: 0 00000 0000000000 = 0x0000
Negative Zero: 1 00000 0000000000 = 0x8000
```

#### Infinity (Exponent = 31, Mantissa = 0)
```
+Infinity: 0 11111 0000000000 = 0x7C00
-Infinity: 1 11111 0000000000 = 0xFC00
```

#### NaN (Exponent = 31, Mantissa ≠ 0)
```
Quiet NaN:  0 11111 1000000000 = 0x7E00
Quiet NaN:  1 11111 1000000000 = 0xFE00

Any pattern: 0 11111 xxxxxxxxxx (where x ≠ all zeros)
Range: 0x7C01 - 0x7FFF (positive NaN)
       0xFC01 - 0xFFFF (negative NaN)
```

#### Denormalized Numbers (Exponent = 0, Mantissa ≠ 0)
```
Value: (-1)^S × 2^(-14) × (0 + M/1024)

Min positive: 0 00000 0000000001 = 2^(-14) × 2^(-10) = 2^(-24) ≈ 5.96e-8
Max denormal: 0 00000 1111111111 = 2^(-14) × (1023/1024) ≈ 6.09e-5
```

**Note**: Many implementations flush denormals to zero for simplicity.

## Common Values

### Powers of 2
```
   Value    │  Hex   │  Binary (S EEEEE MMMMMMMMMM)
────────────┼────────┼──────────────────────────────
   2^(-14)  │ 0x0400 │ 0 00001 0000000000  (min normal)
   2^(-1)   │ 0x3800 │ 0 01110 0000000000  (0.5)
   2^0      │ 0x3C00 │ 0 01111 0000000000  (1.0)
   2^1      │ 0x4000 │ 0 10000 0000000000  (2.0)
   2^2      │ 0x4400 │ 0 10001 0000000000  (4.0)
   2^15     │ 0x7800 │ 0 11110 0000000000  (32768)
```

### Common Constants
```
   Value    │  Hex   │  Decimal      │  Error vs FP32
────────────┼────────┼───────────────┼─────────────────
   0.0      │ 0x0000 │  0.0          │  0.0
   1.0      │ 0x3C00 │  1.0          │  0.0
   π        │ 0x4248 │  3.140625     │  ~0.03%
   e        │ 0x4170 │  2.71875      │  ~0.03%
   √2       │ 0x3DA8 │  1.4140625    │  ~0.01%
   65504    │ 0x7BFF │  65504        │  0.0 (max FP16)
```

### Negative Values
```
   Value    │  Hex   │  Binary (S EEEEE MMMMMMMMMM)
────────────┼────────┼──────────────────────────────
   -0.0     │ 0x8000 │ 1 00000 0000000000
   -1.0     │ 0xBC00 │ 1 01111 0000000000
   -2.0     │ 0xC000 │ 1 10000 0000000000
```

## Range and Precision

### Magnitude Range
```
Min positive normal:    6.103515625 × 10^(-5)  ≈  0.000061
Max positive normal:    65504.0                ≈  6.55e4
Min positive denormal:  5.960464478 × 10^(-8)  ≈  5.96e-8
```

### Overflow/Underflow Behavior
```
│                         FP16 Range                          │
├──────────────┬──────────────────────────────┬──────────────┤
│  -Infinity   │   Representable Numbers     │  +Infinity   │
├──────────────┼──────────────────────────────┼──────────────┤
│  < -65504    │    -65504 ... 65504          │  > 65504     │
└──────────────┴──────────────────────────────┴──────────────┘

Overflow:  |x| > 65504   →  ±Infinity
Underflow: |x| < 6e-8    →  ±Zero (or denormal)
```

### Precision
```
Significand: 11 bits (1 implicit + 10 stored)
Precision:   ~3.3 decimal digits
Epsilon:     2^(-10) = 0.0009765625  ≈  0.001

Relative Error:
  Near 1.0:   ±0.0005  (0.05%)
  Near 10.0:  ±0.005   (0.05%)
  Near 100.0: ±0.05    (0.05%)
```

## Conversion Examples

### FP32 → FP16

#### Example 1: Simple Integer
```
Input (FP32):  3.0
Binary (FP32): 0 10000000 10000000000000000000000
               S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM

Step 1: Extract components
  Sign:     0
  Exponent: 128 (stored) → 128 - 127 = 1 (actual)
  Mantissa: 0x400000 (23 bits)

Step 2: Rebias exponent
  FP16 exponent: 1 + 15 = 16 (actual → stored)

Step 3: Truncate mantissa
  FP32: 10000000000000000000000 (23 bits)
  FP16: 1000000000              (10 bits, keep upper 10)

Step 4: Combine
  Binary (FP16): 0 10000 1000000000
  Hex (FP16):    0x4200

Result: 0x4200 = 3.0 (FP16)
```

#### Example 2: Fractional Number
```
Input (FP32):  0.1
Binary (FP32): 0 01111011 10011001100110011001101
               S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM

Step 1: Extract components
  Sign:     0
  Exponent: 123 (stored) → 123 - 127 = -4 (actual)
  Mantissa: 0x199999 (rounded in FP32)

Step 2: Rebias exponent
  FP16 exponent: -4 + 15 = 11 (actual → stored)

Step 3: Truncate mantissa with rounding
  FP32: 10011001100110011001101 (23 bits)
  FP16: 1001100110              (10 bits)
        Dropped: 0110011001101
        Round bit: 0 (bit 12) → round down

Step 4: Combine
  Binary (FP16): 0 01011 1001100110
  Hex (FP16):    0x2E66

Result: 0x2E66 ≈ 0.0999755859375 (FP16)
Error: ~0.024% (within FP16 precision)
```

### FP16 → FP32 (Exact)

#### Example: 1.5
```
Input (FP16):  0x3E00
Binary (FP16): 0 01111 1000000000
               S EEEEE MMMMMMMMMM

Step 1: Extract components
  Sign:     0
  Exponent: 15 (stored) → 15 - 15 = 0 (actual)
  Mantissa: 0x200 (10 bits) = 512

Step 2: Rebias exponent
  FP32 exponent: 0 + 127 = 127 (actual → stored)

Step 3: Extend mantissa
  FP16: 1000000000                (10 bits)
  FP32: 10000000000000000000000   (23 bits, add 13 zeros)

Step 4: Combine
  Binary (FP32): 0 01111111 10000000000000000000000
  Hex (FP32):    0x3FC00000

Result: 0x3FC00000 = 1.5 (FP32) - Exact!
```

## Rounding Modes

### Round to Nearest Even (Preferred)
```
Tie-breaking: Round to even mantissa

Example 1: 1.001 (tie, mantissa odd)  → 1.00 (round down to even)
Example 2: 1.101 (tie, mantissa even) → 1.10 (round up to even)
Example 3: 1.011 (not tie)            → 1.01 (round down)
Example 4: 1.110 (not tie)            → 1.11 (round up)
```

**CUDA Intrinsic**: `__float2half_rn(float)` - Round to Nearest Even

### Other Rounding Modes
```
__float2half_rz(float)  - Round toward Zero (truncate)
__float2half_rd(float)  - Round Down (toward -∞)
__float2half_ru(float)  - Round Up (toward +∞)
```

**Note**: Only `_rn` (round to nearest) is used in `fp16_conversions.cu`.

## Bit Manipulation

### Extract Components (FP16)
```c
unsigned short fp16 = 0x4200;  // 3.0

unsigned int sign     = (fp16 >> 15) & 0x1;      // Bit 15
unsigned int exponent = (fp16 >> 10) & 0x1F;     // Bits 14-10
unsigned int mantissa = fp16 & 0x3FF;            // Bits 9-0
```

### Construct FP16 from Components
```c
unsigned int sign = 0;       // Positive
unsigned int exp = 16;       // 2^1
unsigned int mant = 0x200;   // 0.5

unsigned short fp16 = (sign << 15) | (exp << 10) | mant;
// Result: 0x4200 = 3.0
```

### Fast Sign Check
```c
unsigned short fp16 = 0xBC00;  // -1.0

// Check sign
bool is_negative = (fp16 & 0x8000) != 0;  // true

// Flip sign
unsigned short negated = fp16 ^ 0x8000;   // 0x3C00 = 1.0
```

### Fast Special Value Detection
```c
unsigned short fp16 = ...;

// Check if zero
bool is_zero = (fp16 & 0x7FFF) == 0;

// Check if infinity
bool is_inf = (fp16 & 0x7FFF) == 0x7C00;

// Check if NaN
bool is_nan = ((fp16 & 0x7C00) == 0x7C00) && ((fp16 & 0x03FF) != 0);

// Check if denormal
bool is_denormal = ((fp16 & 0x7C00) == 0) && ((fp16 & 0x03FF) != 0);
```

## Performance Considerations

### Memory Savings
```
Array Size:  1M elements
FP32:        4 MB
FP16:        2 MB
Savings:     50%
```

### Bandwidth Reduction
```
Operation:   Load 1M FP32 → Compute → Store 1M FP32
FP32:        8 MB transferred
FP16:        4 MB transferred
Reduction:   50% (faster on memory-bound kernels)
```

### Tensor Core Acceleration
```
MatMul Size: 1024 × 1024 × 1024
FP32:        ~2.5 ms on RTX 3090
FP16 (TC):   ~1.2 ms on RTX 3090
Speedup:     2.1x
```

## Common Pitfalls

### 1. Overflow to Infinity
```c
float x = 100000.0f;  // > 65504 (max FP16)
unsigned short fp16 = __float2half_rn(x);  // 0x7C00 (+Infinity)
```

### 2. Underflow to Zero
```c
float x = 1e-6f;  // < 6.1e-5 (min normal FP16)
unsigned short fp16 = __float2half_rn(x);  // 0x0000 (Zero)
```

### 3. Precision Loss
```c
float x = 1.001f;
unsigned short fp16 = __float2half_rn(x);
float y = __half2float(fp16);  // 1.0009765625 (rounded)

// Error: |1.001 - 1.0009765625| = 0.0000234375
```

### 4. NaN Payload Loss
```c
float nan_fp32 = 0x7FC12345;  // NaN with payload
unsigned short fp16 = __float2half_rn(nan_fp32);  // 0x7E09 (payload truncated)
```

## Testing Checklist

Use this checklist when implementing FP16 conversion:

- [ ] Zero (positive and negative)
- [ ] Infinity (positive and negative)
- [ ] NaN (at least one case)
- [ ] Max FP16 value (65504)
- [ ] Min normal FP16 value (~6.1e-5)
- [ ] Overflow case (> 65504 → Infinity)
- [ ] Underflow case (< 6e-8 → Zero)
- [ ] Round-trip accuracy (FP32 → FP16 → FP32)
- [ ] Powers of 2 (exact representation)
- [ ] Common constants (π, e, √2)
- [ ] Negative values
- [ ] Denormalized numbers (if supported)

**Test Script**: `scripts/test_fp16_conversions_cupy.py` validates all of these.

---

## References

1. **IEEE 754 Standard**: https://en.wikipedia.org/wiki/IEEE_754
2. **Half-Precision**: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
3. **FP16 Arithmetic**: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html
4. **Conversion Tool**: https://www.h-schmidt.net/FloatConverter/IEEE754.html

---

**Last Updated**: 2025-11-01
**Part of**: FP16 Conversion Kernels Documentation
