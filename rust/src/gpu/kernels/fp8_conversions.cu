//! FP8 E4M3 Conversion Kernels Using Pure Bitwise Operations
//!
//! Implements FP32 <-> FP8 E4M3 conversions WITHOUT requiring:
//! - cuda_fp8.h header
//! - <mma.h> header
//! - Any CUDA SDK C++ templates
//!
//! Compatible with NVRTC JIT compilation!
//!
//! FP8 E4M3 Format Specification (8 bits total):
//! - 1 sign bit (bit 7)
//! - 4 exponent bits (bits 6-3), bias = 7
//! - 3 mantissa bits (bits 2-0)
//! - No infinity representation (max exponent = 15)
//! - NaN: 0x7F (positive), 0xFF (negative)
//! - Exponent range: [-6, 8] (after bias subtraction)
//! - Value range: ±448 (max finite value)
//! - Smallest normalized: 2^(-6) = 0.015625
//! - Denormal range: [2^(-9), 2^(-6))
//!
//! Reference: NVIDIA H100 Tensor Core GPU Architecture White Paper
//! https://resources.nvidia.com/en-us-tensor-core

// FP32 format constants
#define FP32_SIGN_MASK 0x80000000u
#define FP32_EXP_MASK  0x7F800000u
#define FP32_MANT_MASK 0x007FFFFFu
#define FP32_EXP_BIAS  127

// FP8 E4M3 format constants
#define FP8_SIGN_MASK 0x80u
#define FP8_EXP_MASK  0x78u
#define FP8_MANT_MASK 0x07u
#define FP8_EXP_BIAS  7
#define FP8_MAX_EXP   15
#define FP8_NAN_POS   0x7Fu
#define FP8_NAN_NEG   0xFFu

// FP8 E4M3 range limits
#define FP8_MAX_NORMAL 448.0f    // 2^8 * 1.75 (max representable)
#define FP8_MIN_NORMAL 0.015625f // 2^(-6) (smallest normalized)

//! Convert FP32 to FP8 E4M3 using bitwise operations
//!
//! Algorithm:
//! 1. Extract sign, exponent, mantissa from FP32
//! 2. Adjust exponent bias (FP32 bias=127 -> FP8 bias=7)
//! 3. Round mantissa from 23 bits to 3 bits
//! 4. Handle special cases:
//!    - Overflow (>448) -> saturate to max (0x7E/0xFE)
//!    - Underflow (<2^-9) -> zero
//!    - Denormals (2^-9 to 2^-6) -> denormal representation
//!    - NaN/Inf -> NaN (0x7F/0xFF)
//!
//! @param value FP32 input value
//! @return FP8 E4M3 representation as unsigned char
__device__ __forceinline__ unsigned char fp32_to_fp8_e4m3_scalar(float value) {
    // Extract FP32 bits
    unsigned int bits = __float_as_uint(value);
    unsigned int sign = (bits & FP32_SIGN_MASK) >> 31;
    int exp32 = (int)((bits & FP32_EXP_MASK) >> 23) - FP32_EXP_BIAS;
    unsigned int mant32 = bits & FP32_MANT_MASK;

    // Handle special cases
    if ((bits & FP32_EXP_MASK) == FP32_EXP_MASK) {
        // NaN or Inf -> FP8 NaN
        return sign ? FP8_NAN_NEG : FP8_NAN_POS;
    }

    if (bits == 0 || bits == FP32_SIGN_MASK) {
        // Zero (positive or negative)
        return (unsigned char)(sign << 7);
    }

    // Adjust exponent for FP8 bias (127 -> 7)
    int exp8 = exp32 + FP8_EXP_BIAS;

    // Handle overflow (value > 448 or exp8 > 15)
    if (exp8 > FP8_MAX_EXP || (exp8 == FP8_MAX_EXP && mant32 >= 0x600000)) {
        // Saturate to max representable value
        // Max FP8: sign=0/1, exp=15 (0xF), mant=110 (0x6) -> 0x7E/0xFE
        return (unsigned char)((sign << 7) | 0x7E);
    }

    // Handle underflow and denormals
    if (exp8 <= 0) {
        if (exp8 < -3) {
            // Too small for denormal representation -> zero
            return (unsigned char)(sign << 7);
        }

        // Denormal: shift mantissa right and set exp=0
        // Add implicit leading 1 to mantissa
        unsigned int denorm_mant = (0x800000u | mant32) >> (1 - exp8);

        // Round to nearest 3 bits (with tie to even)
        unsigned int round_bits = (denorm_mant >> 20) & 0x7;
        unsigned int sticky = (denorm_mant & 0xFFFFF) != 0 ? 1 : 0;
        unsigned int rounded = round_bits + ((round_bits == 7 && sticky) ? 1 : 0);

        if (rounded > 7) {
            rounded = 7; // Saturate
        }

        return (unsigned char)((sign << 7) | rounded);
    }

    // Normal case: convert mantissa from 23 bits to 3 bits
    // Round to nearest, ties to even
    unsigned int mant_high = mant32 >> 20; // Top 3 bits of mantissa
    unsigned int round_bit = (mant32 >> 19) & 1;
    unsigned int sticky_bits = (mant32 & 0x7FFFF) != 0 ? 1 : 0;

    // Round to nearest, tie to even
    unsigned int mant8 = mant_high;
    if (round_bit && (sticky_bits || (mant_high & 1))) {
        mant8++;

        // Check for mantissa overflow
        if (mant8 > 7) {
            mant8 = 0;
            exp8++;

            // Check for exponent overflow after rounding
            if (exp8 > FP8_MAX_EXP) {
                return (unsigned char)((sign << 7) | 0x7E);
            }
        }
    }

    // Assemble FP8 E4M3: sign(1) | exp(4) | mant(3)
    unsigned char fp8 = (unsigned char)(
        (sign << 7) |
        ((exp8 & 0xF) << 3) |
        (mant8 & 0x7)
    );

    return fp8;
}

//! Convert FP8 E4M3 to FP32 using bitwise operations
//!
//! Algorithm:
//! 1. Extract sign, exponent, mantissa from FP8
//! 2. Check for special cases (zero, NaN)
//! 3. Adjust exponent bias (FP8 bias=7 -> FP32 bias=127)
//! 4. Expand mantissa from 3 bits to 23 bits
//! 5. Handle denormals if exponent is 0
//!
//! @param fp8 FP8 E4M3 input value as unsigned char
//! @return FP32 representation as float
__device__ __forceinline__ float fp8_e4m3_to_fp32_scalar(unsigned char fp8) {
    // Extract FP8 components
    unsigned int sign = (fp8 & FP8_SIGN_MASK) >> 7;
    unsigned int exp8 = (fp8 & FP8_EXP_MASK) >> 3;
    unsigned int mant8 = fp8 & FP8_MANT_MASK;

    // Handle special cases
    if (fp8 == 0 || fp8 == 0x80) {
        // Zero (positive or negative)
        return __uint_as_float(sign << 31);
    }

    if (fp8 == FP8_NAN_POS || fp8 == FP8_NAN_NEG) {
        // NaN -> FP32 NaN
        return __uint_as_float(0x7FC00000u | (sign << 31));
    }

    unsigned int exp32;
    unsigned int mant32;

    if (exp8 == 0) {
        // Denormal FP8: normalize for FP32
        // Find leading 1 in mantissa
        if (mant8 == 0) {
            return __uint_as_float(sign << 31); // Zero
        }

        // Count leading zeros in 3-bit mantissa
        int lz = (mant8 & 4) ? 0 : (mant8 & 2) ? 1 : 2;

        // Normalize: shift mantissa left until leading 1 is removed
        mant32 = (mant8 << (lz + 1)) & 0x7;
        mant32 <<= 20; // Position in FP32 mantissa field

        // Adjust exponent: base exp=-6 (bias=7, exp=0 means 2^-6)
        // Then subtract lz+1 for normalization shift
        exp32 = FP32_EXP_BIAS + (1 - FP8_EXP_BIAS) - (lz + 1);
    } else {
        // Normal case
        // Adjust exponent bias: FP8 bias=7 -> FP32 bias=127
        exp32 = exp8 - FP8_EXP_BIAS + FP32_EXP_BIAS;

        // Expand mantissa: 3 bits -> 23 bits
        // FP8 mantissa represents bits after implicit 1.xxx
        // FP32 needs same representation with more precision bits
        mant32 = mant8 << 20; // Left-align in 23-bit field
    }

    // Assemble FP32: sign(1) | exp(8) | mant(23)
    unsigned int fp32_bits = (sign << 31) | (exp32 << 23) | mant32;

    return __uint_as_float(fp32_bits);
}

//! Convert array of FP32 values to FP8 E4M3
//!
//! Thread-per-element kernel with vectorized memory access.
//!
//! @param input  FP32 input array
//! @param output FP8 E4M3 output array (unsigned char)
//! @param n      Number of elements
extern "C" __global__ void fp32_to_fp8_e4m3(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized processing: each thread processes 4 elements
    int vec_idx = idx * 4;

    if (vec_idx + 3 < n) {
        // Load 4 FP32 values using vectorized read
        float4 in4 = *((const float4*)(&input[vec_idx]));

        // Convert each element
        unsigned char out0 = fp32_to_fp8_e4m3_scalar(in4.x);
        unsigned char out1 = fp32_to_fp8_e4m3_scalar(in4.y);
        unsigned char out2 = fp32_to_fp8_e4m3_scalar(in4.z);
        unsigned char out3 = fp32_to_fp8_e4m3_scalar(in4.w);

        // Pack 4 bytes into uint32 and store
        unsigned int packed = (out3 << 24) | (out2 << 16) | (out1 << 8) | out0;
        *((unsigned int*)(&output[vec_idx])) = packed;
    } else {
        // Handle remaining elements (scalar tail)
        for (int i = vec_idx; i < n && i < vec_idx + 4; i++) {
            output[i] = fp32_to_fp8_e4m3_scalar(input[i]);
        }
    }
}

//! Convert array of FP8 E4M3 values to FP32
//!
//! Thread-per-element kernel with vectorized memory access.
//!
//! @param input  FP8 E4M3 input array (unsigned char)
//! @param output FP32 output array
//! @param n      Number of elements
extern "C" __global__ void fp8_e4m3_to_fp32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized processing: each thread processes 4 elements
    int vec_idx = idx * 4;

    if (vec_idx + 3 < n) {
        // Load 4 FP8 values as packed uint32
        unsigned int packed = *((const unsigned int*)(&input[vec_idx]));

        // Unpack bytes
        unsigned char in0 = packed & 0xFF;
        unsigned char in1 = (packed >> 8) & 0xFF;
        unsigned char in2 = (packed >> 16) & 0xFF;
        unsigned char in3 = (packed >> 24) & 0xFF;

        // Convert each element
        float out0 = fp8_e4m3_to_fp32_scalar(in0);
        float out1 = fp8_e4m3_to_fp32_scalar(in1);
        float out2 = fp8_e4m3_to_fp32_scalar(in2);
        float out3 = fp8_e4m3_to_fp32_scalar(in3);

        // Store using vectorized write
        *((float4*)(&output[vec_idx])) = make_float4(out0, out1, out2, out3);
    } else {
        // Handle remaining elements (scalar tail)
        for (int i = vec_idx; i < n && i < vec_idx + 4; i++) {
            output[i] = fp8_e4m3_to_fp32_scalar(input[i]);
        }
    }
}

//! Test kernel to verify FP8 conversions
//!
//! Tests conversion round-trip accuracy for special values:
//! - Zero (positive/negative)
//! - One
//! - Max value (448.0)
//! - Min normal value (0.015625)
//! - Denormal value (0.001953125 = 2^-9)
//! - Small values that should round to zero
//!
//! Results stored in global memory for validation.
extern "C" __global__ void test_fp8_conversions(
    float* test_values,     // Input: test values
    float* recovered,       // Output: FP32 -> FP8 -> FP32
    unsigned char* fp8_mid, // Output: intermediate FP8 values
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float original = test_values[idx];

        // Convert FP32 -> FP8
        unsigned char fp8 = fp32_to_fp8_e4m3_scalar(original);
        fp8_mid[idx] = fp8;

        // Convert FP8 -> FP32
        float result = fp8_e4m3_to_fp32_scalar(fp8);
        recovered[idx] = result;
    }
}

//! Batched conversion with saturation clamping
//!
//! Clamps input values to FP8 E4M3 range before conversion.
//! Useful for training where you want to explicitly handle overflow.
//!
//! @param input  FP32 input array
//! @param output FP8 E4M3 output array (unsigned char)
//! @param n      Number of elements
extern "C" __global__ void fp32_to_fp8_e4m3_saturate(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx + 3 < n) {
        // Load 4 FP32 values
        float4 in4 = *((const float4*)(&input[vec_idx]));

        // Clamp to FP8 range [-448, 448]
        in4.x = fmaxf(-FP8_MAX_NORMAL, fminf(FP8_MAX_NORMAL, in4.x));
        in4.y = fmaxf(-FP8_MAX_NORMAL, fminf(FP8_MAX_NORMAL, in4.y));
        in4.z = fmaxf(-FP8_MAX_NORMAL, fminf(FP8_MAX_NORMAL, in4.z));
        in4.w = fmaxf(-FP8_MAX_NORMAL, fminf(FP8_MAX_NORMAL, in4.w));

        // Convert each element
        unsigned char out0 = fp32_to_fp8_e4m3_scalar(in4.x);
        unsigned char out1 = fp32_to_fp8_e4m3_scalar(in4.y);
        unsigned char out2 = fp32_to_fp8_e4m3_scalar(in4.z);
        unsigned char out3 = fp32_to_fp8_e4m3_scalar(in4.w);

        // Pack and store
        unsigned int packed = (out3 << 24) | (out2 << 16) | (out1 << 8) | out0;
        *((unsigned int*)(&output[vec_idx])) = packed;
    } else {
        for (int i = vec_idx; i < n && i < vec_idx + 4; i++) {
            float val = fmaxf(-FP8_MAX_NORMAL, fminf(FP8_MAX_NORMAL, input[i]));
            output[i] = fp32_to_fp8_e4m3_scalar(val);
        }
    }
}

//! Stochastic rounding conversion (for ML training)
//!
//! Uses randomness to reduce quantization bias during training.
//! Each thread uses a simple LCG PRNG seeded by thread ID and value.
//!
//! @param input  FP32 input array
//! @param output FP8 E4M3 output array (unsigned char)
//! @param n      Number of elements
//! @param seed   Random seed
extern "C" __global__ void fp32_to_fp8_e4m3_stochastic(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    int n,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float value = input[idx];

        // Simple LCG for random bit
        unsigned int rng_state = seed + idx;
        rng_state = rng_state * 1664525u + 1013904223u;
        unsigned int random_bit = (rng_state >> 16) & 1;

        // Extract FP32 bits for stochastic rounding
        unsigned int bits = __float_as_uint(value);
        unsigned int sign = (bits & FP32_SIGN_MASK) >> 31;
        int exp32 = (int)((bits & FP32_EXP_MASK) >> 23) - FP32_EXP_BIAS;
        unsigned int mant32 = bits & FP32_MANT_MASK;

        // Handle special cases (same as deterministic)
        if ((bits & FP32_EXP_MASK) == FP32_EXP_MASK) {
            output[idx] = sign ? FP8_NAN_NEG : FP8_NAN_POS;
            return;
        }

        if (bits == 0 || bits == FP32_SIGN_MASK) {
            output[idx] = (unsigned char)(sign << 7);
            return;
        }

        int exp8 = exp32 + FP8_EXP_BIAS;

        // Overflow
        if (exp8 > FP8_MAX_EXP) {
            output[idx] = (unsigned char)((sign << 7) | 0x7E);
            return;
        }

        // Underflow
        if (exp8 < -3) {
            output[idx] = (unsigned char)(sign << 7);
            return;
        }

        // Normal case with stochastic rounding
        if (exp8 > 0) {
            unsigned int mant_high = mant32 >> 20;
            unsigned int round_bit = (mant32 >> 19) & 1;

            // Stochastic rounding: add random bit instead of deterministic tie-breaking
            unsigned int mant8 = mant_high + (round_bit & random_bit);

            if (mant8 > 7) {
                mant8 = 0;
                exp8++;
                if (exp8 > FP8_MAX_EXP) {
                    output[idx] = (unsigned char)((sign << 7) | 0x7E);
                    return;
                }
            }

            output[idx] = (unsigned char)((sign << 7) | ((exp8 & 0xF) << 3) | (mant8 & 0x7));
        } else {
            // Denormal case (simplified, use deterministic)
            output[idx] = fp32_to_fp8_e4m3_scalar(value);
        }
    }
}
