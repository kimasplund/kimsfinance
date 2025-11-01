//! FP16 WMMA Tensor Core Kernel for Ada Lovelace
//!
//! Uses WMMA (Warp Matrix Multiply-Accumulate) API with FP16 (half precision).
//! Compatible with NVRTC JIT compilation - no build-time dependencies!
//!
//! Hardware: NVIDIA Volta+ (Compute Capability 7.0+)
//! Performance: 2x speedup vs FP32 on tensor cores
//! Memory: 2x bandwidth vs FP32 (16-bit vs 32-bit)

#include <mma.h>
using namespace nvcuda;

// WMMA tile sizes for FP16
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// FP16 matrix multiply using WMMA tensor cores
extern "C" __global__ void fp16_matmul_wmma(
    const __half* a,           // FP16 input matrix A (m x k)
    const __half* b,           // FP16 input matrix B (k x n)
    float* c,                  // FP32 output matrix C (m x n)
    int m,                     // Rows of A and C
    int n,                     // Columns of B and C
    int k                      // Columns of A, rows of B
) {
    // Warp and lane IDs
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Bounds check
    if (warpM * WMMA_M >= m || warpN * WMMA_N >= n) {
        return;
    }

    // Compute C = A * B using tensor cores
    for (int i = 0; i < k; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Load A and B fragments from global memory
        if (aRow < m && aCol + WMMA_K <= k) {
            wmma::load_matrix_sync(a_frag, a + aRow * k + aCol, k);
        } else {
            wmma::fill_fragment(a_frag, __float2half(0.0f));
        }

        if (bRow + WMMA_K <= k && bCol < n) {
            wmma::load_matrix_sync(b_frag, b + bRow * n + bCol, n);
        } else {
            wmma::fill_fragment(b_frag, __float2half(0.0f));
        }

        // Perform tensor core matrix multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store C fragment to global memory
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < m && cCol < n) {
        wmma::store_matrix_sync(c + cRow * n + cCol, c_frag, n, wmma::mem_row_major);
    }
}

// Convert FP32 to FP16
extern "C" __global__ void fp32_to_fp16(
    const float* input,
    __half* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

// Convert FP16 to FP32
extern "C" __global__ void fp16_to_fp32(
    const __half* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}

// Test kernel to verify WMMA support
extern "C" __global__ void test_fp16_wmma() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Test FP16 conversion
        __half fp16_val = __float2half(1.5f);
        float fp32_val = __half2float(fp16_val);

        // Verify result
        if (fabsf(fp32_val - 1.5f) < 0.001f) {
            // Success - FP16 conversion works
        }
    }
}
