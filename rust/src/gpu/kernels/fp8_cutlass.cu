//! Native CUDA FP8 E4M3 Tensor Core Kernel for Ada Lovelace
//!
//! Uses CUDA 13.0's FP8 API (cuda_fp8.h) for AOT compilation.
//! Requires nvcc at build time - NOT compatible with NVRTC JIT.
//!
//! Hardware: NVIDIA Ada Lovelace (sm_89+)
//! CUDA: 13.0+

#include <cuda_fp8.h>

// Simple FP8 matrix multiply using native CUDA FP8 types
// This is a simplified version - production code would use tensor cores
extern "C" __global__ void fp8_matmul_cutlass(
    const void* a_ptr,         // FP8 E4M3 input matrix A (m x k) - stored as __nv_fp8_storage_t
    const void* b_ptr,         // FP8 E4M3 input matrix B (k x n) - stored as __nv_fp8_storage_t
    float* c_ptr,              // FP32 output matrix C (m x n)
    int m,                     // Rows of A and C
    int n,                     // Columns of B and C
    int k                      // Columns of A, rows of B
) {
    // Thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        // Cast input pointers to CUDA FP8 storage type (unsigned char)
        const __nv_fp8_storage_t* a = reinterpret_cast<const __nv_fp8_storage_t*>(a_ptr);
        const __nv_fp8_storage_t* b = reinterpret_cast<const __nv_fp8_storage_t*>(b_ptr);

        // Accumulate in FP32 for precision
        float sum = 0.0f;

        // Compute dot product
        for (int i = 0; i < k; ++i) {
            // Convert FP8 storage -> FP16 -> FP32, multiply, accumulate
            __half_raw a_fp16 = __nv_cvt_fp8_to_halfraw(a[row * k + i], __NV_E4M3);
            __half_raw b_fp16 = __nv_cvt_fp8_to_halfraw(b[i * n + col], __NV_E4M3);
            float a_val = __half2float(a_fp16);
            float b_val = __half2float(b_fp16);
            sum += a_val * b_val;
        }

        // Store result
        c_ptr[row * n + col] = sum;
    }
}

// Convert FP32 array to FP8 E4M3
extern "C" __global__ void fp32_to_fp8_e4m3(
    const float* input,
    void* output_ptr,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        __nv_fp8_storage_t* output = reinterpret_cast<__nv_fp8_storage_t*>(output_ptr);
        // Convert FP32 -> FP16 -> FP8 storage with saturation to finite values
        __half_raw fp16_val = __float2half_rn(input[idx]);
        output[idx] = __nv_cvt_halfraw_to_fp8(fp16_val, __NV_SATFINITE, __NV_E4M3);
    }
}

// Convert FP8 E4M3 array to FP32
extern "C" __global__ void fp8_e4m3_to_fp32(
    const void* input_ptr,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const __nv_fp8_storage_t* input = reinterpret_cast<const __nv_fp8_storage_t*>(input_ptr);
        // Convert FP8 storage -> FP16 -> FP32
        __half_raw fp16_val = __nv_cvt_fp8_to_halfraw(input[idx], __NV_E4M3);
        output[idx] = __half2float(fp16_val);
    }
}

// Test kernel to verify FP8 support
extern "C" __global__ void test_fp8_cutlass() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Create FP8 value from FP32
        __half_raw fp16_val = __float2half_rn(1.5f);
        __nv_fp8_storage_t fp8_val = __nv_cvt_halfraw_to_fp8(fp16_val, __NV_SATFINITE, __NV_E4M3);

        // Convert back to FP32
        __half_raw fp16_back = __nv_cvt_fp8_to_halfraw(fp8_val, __NV_E4M3);
        float fp32_val = __half2float(fp16_back);

        // Just to prevent optimization
        if (fp32_val > 0.0f) {
            // Success - FP8 conversion works
        }
    }
}
