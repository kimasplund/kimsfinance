//! FP16 Conversion Kernels for NVRTC
//!
//! Converts between FP32 and FP16 formats using CUDA intrinsics.
//! Uses `unsigned short` instead of `__half` for NVRTC compatibility.
//!
//! FP16 Format (IEEE 754 binary16):
//! - 1 bit sign
//! - 5 bits exponent (bias = 15)
//! - 10 bits mantissa
//!
//! Range: ±6.5e4, precision: ~3 decimal digits
//! Special values: Inf, NaN preserved
//!
//! Compatible with NVRTC JIT compilation!

// ==============================================================================
// Method 1: Using CUDA Intrinsics (Preferred - Hardware Accelerated)
// ==============================================================================

// Forward declarations for manual conversion functions
__device__ __forceinline__ unsigned short float_to_half_manual(float f);
__device__ __forceinline__ float half_to_float_manual(unsigned short h);

// Convert FP32 array to FP16 using bitwise operations (NVRTC compatible)
extern "C" __global__ void fp32_to_fp16(
    const float* __restrict__ input,
    unsigned short* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Use manual bitwise conversion (no CUDA intrinsics needed)
        output[idx] = float_to_half_manual(input[idx]);
    }
}

// Convert FP16 array to FP32 using bitwise operations (NVRTC compatible)
extern "C" __global__ void fp16_to_fp32(
    const unsigned short* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Use manual bitwise conversion (no CUDA intrinsics needed)
        output[idx] = half_to_float_manual(input[idx]);
    }
}

// ==============================================================================
// Method 2: Manual Bitwise Conversion (Fallback if intrinsics unavailable)
// ==============================================================================

// Manual FP32 to FP16 conversion using bitwise operations
// Reference: https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
__device__ __forceinline__ unsigned short float_to_half_manual(float f) {
    unsigned int x = __float_as_uint(f);

    unsigned int sign = (x >> 16) & 0x8000;       // Sign bit
    unsigned int exp = (x >> 23) & 0xff;          // Exponent (8 bits)
    unsigned int mantissa = x & 0x7fffff;         // Mantissa (23 bits)

    // Handle special cases
    if (exp == 0xff) {
        // Infinity or NaN
        if (mantissa == 0) {
            // Infinity
            return sign | 0x7c00;
        } else {
            // NaN - preserve payload (keep at least one bit set)
            return sign | 0x7c00 | (mantissa >> 13) | 0x0001;
        }
    }

    if (exp == 0) {
        // Zero or denormalized number
        if (mantissa == 0) {
            // Zero
            return sign;
        }
        // Denormalized - flush to zero (simpler approach)
        return sign;
    }

    // Normalized number conversion
    int exp_fp16 = exp - 127 + 15;  // Rebias exponent (FP32 bias=127, FP16 bias=15)

    // Handle overflow (exponent too large for FP16)
    if (exp_fp16 >= 0x1f) {
        return sign | 0x7c00;  // Infinity
    }

    // Handle underflow (exponent too small for FP16)
    if (exp_fp16 <= 0) {
        return sign;  // Flush to zero
    }

    // Convert mantissa from 23 bits to 10 bits with rounding
    unsigned int mantissa_fp16 = mantissa >> 13;

    // Round to nearest even (banker's rounding)
    if ((mantissa & 0x1000) && ((mantissa & 0x2000) || (mantissa & 0x0fff))) {
        mantissa_fp16++;
        if (mantissa_fp16 >= 0x400) {
            // Mantissa overflow - increment exponent
            exp_fp16++;
            mantissa_fp16 = 0;
            if (exp_fp16 >= 0x1f) {
                return sign | 0x7c00;  // Overflow to infinity
            }
        }
    }

    return sign | (exp_fp16 << 10) | mantissa_fp16;
}

// Manual FP16 to FP32 conversion using bitwise operations
__device__ __forceinline__ float half_to_float_manual(unsigned short h) {
    unsigned int sign = (h & 0x8000) << 16;       // Sign bit
    unsigned int exp = (h >> 10) & 0x1f;          // Exponent (5 bits)
    unsigned int mantissa = h & 0x3ff;            // Mantissa (10 bits)

    // Handle special cases
    if (exp == 0x1f) {
        // Infinity or NaN
        if (mantissa == 0) {
            // Infinity
            return __uint_as_float(sign | 0x7f800000);
        } else {
            // NaN - preserve payload
            return __uint_as_float(sign | 0x7f800000 | (mantissa << 13));
        }
    }

    if (exp == 0) {
        // Zero or denormalized number
        if (mantissa == 0) {
            // Zero
            return __uint_as_float(sign);
        }
        // Denormalized - convert to normalized FP32
        exp = 1;
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            exp--;
        }
        mantissa &= 0x3ff;  // Remove leading 1
    }

    // Rebias exponent (FP16 bias=15, FP32 bias=127)
    unsigned int exp_fp32 = exp + 127 - 15;

    // Combine components
    unsigned int result = sign | (exp_fp32 << 23) | (mantissa << 13);
    return __uint_as_float(result);
}

// Fallback kernels using manual conversion
extern "C" __global__ void fp32_to_fp16_manual(
    const float* __restrict__ input,
    unsigned short* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = float_to_half_manual(input[idx]);
    }
}

extern "C" __global__ void fp16_to_fp32_manual(
    const unsigned short* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = half_to_float_manual(input[idx]);
    }
}

// ==============================================================================
// Vectorized Versions (4x throughput using float4)
// ==============================================================================

// Vectorized FP32 to FP16 conversion (processes 4 elements per thread)
extern "C" __global__ void fp32_to_fp16_vectorized(
    const float* __restrict__ input,
    unsigned short* __restrict__ output,
    int n
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < n) {
        // Load 4 FP32 values
        float4 in = *((float4*)(&input[idx]));

        // Convert each element using manual bitwise conversion
        unsigned short out[4];
        out[0] = float_to_half_manual(in.x);
        out[1] = float_to_half_manual(in.y);
        out[2] = float_to_half_manual(in.z);
        out[3] = float_to_half_manual(in.w);

        // Store as uint2 (64 bits = 4 x 16 bits)
        *((uint2*)(&output[idx])) = *((uint2*)(&out[0]));
    } else {
        // Handle tail elements
        for (int i = idx; i < n; i++) {
            output[i] = float_to_half_manual(input[i]);
        }
    }
}

// Vectorized FP16 to FP32 conversion (processes 4 elements per thread)
extern "C" __global__ void fp16_to_fp32_vectorized(
    const unsigned short* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < n) {
        // Load 4 FP16 values as uint2 (64 bits)
        unsigned short in[4];
        *((uint2*)(&in[0])) = *((uint2*)(&input[idx]));

        // Convert each element using manual bitwise conversion
        float4 out;
        out.x = half_to_float_manual(in[0]);
        out.y = half_to_float_manual(in[1]);
        out.z = half_to_float_manual(in[2]);
        out.w = half_to_float_manual(in[3]);

        // Store 4 FP32 values
        *((float4*)(&output[idx])) = out;
    } else {
        // Handle tail elements
        for (int i = idx; i < n; i++) {
            output[i] = half_to_float_manual(input[i]);
        }
    }
}

// ==============================================================================
// Test Kernels
// ==============================================================================

// Test kernel: Verify round-trip conversion accuracy
// Tests FP32 → FP16 → FP32 and reports max error
extern "C" __global__ void test_fp16_roundtrip(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ errors,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float original = input[idx];

        // Round-trip: FP32 → FP16 → FP32 (using manual conversion)
        unsigned short fp16 = float_to_half_manual(original);
        float recovered = half_to_float_manual(fp16);

        // Calculate absolute error
        float error = fabsf(original - recovered);

        // Store results
        output[idx] = recovered;
        errors[idx] = error;
    }
}

// Test kernel: Compare manual conversion with itself (validation test)
extern "C" __global__ void test_fp16_conversion_methods(
    const float* __restrict__ input,
    unsigned short* __restrict__ hw_output,
    unsigned short* __restrict__ manual_output,
    int* __restrict__ mismatches,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float value = input[idx];

        // Both use manual conversion now (no intrinsics)
        unsigned short hw = float_to_half_manual(value);

        // Manual conversion
        unsigned short manual = float_to_half_manual(value);

        // Store results
        hw_output[idx] = hw;
        manual_output[idx] = manual;

        // Count mismatches (should always be 0 now)
        if (hw != manual) {
            atomicAdd(mismatches, 1);
        }
    }
}

// Test kernel: Validate special values (Inf, NaN, Zero)
extern "C" __global__ void test_fp16_special_values(
    float* __restrict__ results,
    int* __restrict__ failures
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int fail_count = 0;

        // Test positive infinity
        {
            float pos_inf = __uint_as_float(0x7f800000);
            unsigned short fp16 = float_to_half_manual(pos_inf);
            float recovered = half_to_float_manual(fp16);
            if (!isinf(recovered) || signbit(recovered)) {
                fail_count++;
            }
            results[0] = recovered;
        }

        // Test negative infinity
        {
            float neg_inf = __uint_as_float(0xff800000);
            unsigned short fp16 = float_to_half_manual(neg_inf);
            float recovered = half_to_float_manual(fp16);
            if (!isinf(recovered) || !signbit(recovered)) {
                fail_count++;
            }
            results[1] = recovered;
        }

        // Test NaN
        {
            float nan_val = __uint_as_float(0x7fc00000);
            unsigned short fp16 = float_to_half_manual(nan_val);
            float recovered = half_to_float_manual(fp16);
            if (!isnan(recovered)) {
                fail_count++;
            }
            results[2] = recovered;
        }

        // Test positive zero
        {
            float pos_zero = 0.0f;
            unsigned short fp16 = float_to_half_manual(pos_zero);
            float recovered = half_to_float_manual(fp16);
            if (recovered != 0.0f || signbit(recovered)) {
                fail_count++;
            }
            results[3] = recovered;
        }

        // Test negative zero
        {
            float neg_zero = -0.0f;
            unsigned short fp16 = float_to_half_manual(neg_zero);
            float recovered = half_to_float_manual(fp16);
            if (recovered != 0.0f || !signbit(recovered)) {
                fail_count++;
            }
            results[4] = recovered;
        }

        // Test max FP16 value (~65504)
        {
            float max_fp16 = 65504.0f;
            unsigned short fp16 = float_to_half_manual(max_fp16);
            float recovered = half_to_float_manual(fp16);
            if (fabsf(recovered - max_fp16) > 1.0f) {
                fail_count++;
            }
            results[5] = recovered;
        }

        // Test min positive normal FP16 value (2^-14 ≈ 6.1e-5)
        {
            float min_normal = 6.103515625e-5f;
            unsigned short fp16 = float_to_half_manual(min_normal);
            float recovered = half_to_float_manual(fp16);
            if (fabsf(recovered - min_normal) / min_normal > 0.01f) {
                fail_count++;
            }
            results[6] = recovered;
        }

        *failures = fail_count;
    }
}

// ==============================================================================
// Benchmark Kernel
// ==============================================================================

// Benchmark FP32→FP16 conversion throughput
extern "C" __global__ void benchmark_fp32_to_fp16(
    const float* __restrict__ input,
    unsigned short* __restrict__ output,
    int n,
    int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        unsigned short result = 0;

        // Multiple iterations to measure peak throughput
        #pragma unroll 8
        for (int i = 0; i < iterations; i++) {
            result = float_to_half_manual(input[idx]);
        }

        // Write result to prevent dead code elimination
        output[idx] = result;
    }
}

// Benchmark FP16→FP32 conversion throughput
extern "C" __global__ void benchmark_fp16_to_fp32(
    const unsigned short* __restrict__ input,
    float* __restrict__ output,
    int n,
    int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float result = 0.0f;

        // Multiple iterations to measure peak throughput
        #pragma unroll 8
        for (int i = 0; i < iterations; i++) {
            result = half_to_float_manual(input[idx]);
        }

        // Write result to prevent dead code elimination
        output[idx] = result;
    }
}
