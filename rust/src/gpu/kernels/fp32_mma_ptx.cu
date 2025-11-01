//! FP32 TensorFloat-32 (TF32) Tensor Core Kernel Using Raw PTX Inline Assembly
//!
//! Based on NVIDIA PTX ISA and successful FP16/FP8 raw PTX implementations
//!
//! Uses raw PTX inline assembly to access TF32 tensor cores WITHOUT requiring:
//! - mma.h header
//! - cuda::ptx namespace
//! - Any CUDA SDK C++ templates
//!
//! Compatible with NVRTC JIT compilation!
//!
//! Hardware: NVIDIA Ampere (sm_80+), Ada Lovelace (sm_89), Hopper (sm_90+)
//! Performance: ~8x throughput vs FP32 CUDA cores (with automatic TF32 conversion)
//!
//! TensorFloat-32 (TF32) Format:
//! - 19 bits total: 1 sign, 8 exponent, 10 mantissa
//! - Same range as FP32, reduced precision (10-bit mantissa vs 23-bit)
//! - Hardware automatically converts FP32 → TF32 on Ampere+ tensor cores
//! - Instruction: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32

#define MMA_M 16
#define MMA_N 8
#define MMA_K 8  // TF32 uses K=8 (smaller K because TF32 is 19-bit, not 16-bit)

#define WARP_SIZE 32

// Macro for LDMATRIX.X4 (load 4 matrix fragments from shared memory)
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile( \
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) \
        : "r"(addr) \
    )

// Macro for LDMATRIX.X2 (load 2 matrix fragments from shared memory)
#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile( \
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" \
        : "=r"(R0), "=r"(R1) \
        : "r"(addr) \
    )

// Macro for TF32 MMA.1688.TF32 (TensorFloat-32 tensor core matrix multiply-accumulate)
// Computes D = A*B + C using 16x8x8 tiles with TF32 inputs, FP32 accumulation
// Note: Hardware automatically converts FP32 input → TF32 (19-bit) on Ampere+ GPUs
// Input format: FP32 (stored as float, hardware converts to TF32 internally)
// Output format: FP32 accumulator (full precision)
#define HMMA1688_TF32(D0, D1, D2, D3, A0, A1, A2, A3, B0, B1, C0, C1, C2, C3) \
    asm volatile( \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 " \
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" \
        : "=f"(D0), "=f"(D1), "=f"(D2), "=f"(D3) \
        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), \
          "r"(B0), "r"(B1), \
          "f"(C0), "f"(C1), "f"(C2), "f"(C3) \
    )

// Convert shared memory pointer to shared memory address for PTX
__device__ __forceinline__ unsigned int cvta_to_shared(const void* ptr) {
    unsigned int addr;
    asm volatile(
        "{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(addr)
        : "l"(ptr)
    );
    return addr;
}

// FP32 TF32 matrix multiply using MMA PTX tensor cores
// C = A * B where A (m x k), B (k x n), C (m x n)
// Note: Input is FP32 (float), hardware converts to TF32 automatically
// TF32 provides ~8x throughput vs FP32 CUDA cores with minimal accuracy loss
extern "C" __global__ void fp32_matmul_mma_ptx(
    const float* __restrict__ A,  // FP32 input (auto-converted to TF32 by hardware)
    const float* __restrict__ B,  // FP32 input (auto-converted to TF32 by hardware)
    float* __restrict__ C,         // FP32 output (full precision accumulator)
    int m, int n, int k
) {
    // Warp-level coordinates
    const int warp_row = blockIdx.y * MMA_M;
    const int warp_col = blockIdx.x * MMA_N;

    if (warp_row >= m || warp_col >= n) {
        return;
    }

    // Shared memory for tiles (stored as FP32)
    __shared__ float A_shmem[MMA_M][MMA_K];
    __shared__ float B_shmem[MMA_N][MMA_K];
    __shared__ float C_shmem[MMA_M][MMA_N];

    const int lane_id = threadIdx.x % WARP_SIZE;

    // Accumulator registers (FP32 for full precision) - 4 registers for m16n8k8
    float RC[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Iterate over K dimension in steps of MMA_K
    const int K_tiles = (k + MMA_K - 1) / MMA_K;

    #pragma unroll
    for (int tile = 0; tile < K_tiles; ++tile) {
        // Load A tile from global to shared memory
        // FP32 is 32-bit, K=8 so 8 floats = 32 bytes per row
        if (lane_id < 16) {  // MMA_M threads
            const int row = lane_id;
            const int col_base = tile * MMA_K;

            if (warp_row + row < m && col_base + 7 < k) {
                // Load 8 floats (32 bytes) per thread using 2x float4
                *((float4*)(&A_shmem[row][0])) = *((float4*)(&A[(warp_row + row) * k + col_base]));
                *((float4*)(&A_shmem[row][4])) = *((float4*)(&A[(warp_row + row) * k + col_base + 4]));
            } else {
                // Zero-pad if out of bounds
                #pragma unroll
                for (int i = 0; i < MMA_K; ++i) {
                    A_shmem[row][i] = 0.0f;
                }
            }
        }

        // Load B tile from global to shared memory
        // 16 threads (MMA_N * 2) for 8 rows × 8 cols
        if (lane_id < MMA_N * 2) {  // 16 threads for 8 rows
            const int row = lane_id / 2;
            const int col_offset = (lane_id % 2) * 4;
            const int col_base = tile * MMA_K + col_offset;

            if (warp_col + row < n && col_base + 3 < k) {
                // Load 4 floats (16 bytes) per access
                *((float4*)(&B_shmem[row][col_offset])) =
                    *((float4*)(&B[(warp_col + row) * k + col_base]));
            } else {
                // Zero-pad if out of bounds
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    B_shmem[row][col_offset + i] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Load fragments from shared memory using LDMATRIX
        // Note: LDMATRIX works with 16-bit (.b16) chunks, so we load FP32 pairs as 2x16-bit
        unsigned int RA[4];  // Matrix A fragments (FP32 stored as unsigned int pairs)
        unsigned int RB[2];  // Matrix B fragments (FP32 stored as unsigned int pairs)

        // LDMATRIX for A (16x8 tile, load 4 fragments as .b16)
        // For K=8, we load 16 bytes (4 floats) per fragment
        unsigned int A_addr = cvta_to_shared(&A_shmem[lane_id % 16][(lane_id / 16) * 4]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_addr);

        // LDMATRIX for B (8x8 tile, load 2 fragments as .b16)
        unsigned int B_addr = cvta_to_shared(&B_shmem[lane_id % 8][((lane_id / 8) % 2) * 4]);
        LDMATRIX_X2(RB[0], RB[1], B_addr);

        // TF32 tensor core matrix multiply-accumulate (m16n8k8)
        // Input: FP32 (hardware converts to TF32 19-bit internally)
        // Accumulator: FP32 for full precision output
        // Note: 4 accumulator registers for FP32 output
        HMMA1688_TF32(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1], RC[2], RC[3]);

        __syncthreads();
    }

    // Store result to shared memory (FP32) - 4 float registers
    // Each warp outputs 16x8 tile
    *((float*)(&C_shmem[lane_id / 4][0]) + (lane_id % 4) * 2 + 0) = RC[0];
    *((float*)(&C_shmem[lane_id / 4][0]) + (lane_id % 4) * 2 + 1) = RC[1];
    *((float*)(&C_shmem[lane_id / 4 + 8][0]) + (lane_id % 4) * 2 + 0) = RC[2];
    *((float*)(&C_shmem[lane_id / 4 + 8][0]) + (lane_id % 4) * 2 + 1) = RC[3];

    __syncthreads();

    // Write result back to global memory (FP32)
    if (lane_id < MMA_M) {
        const int row = warp_row + lane_id;
        if (row < m) {
            // Write 8 floats (32 bytes) per thread using 2x float4
            *((float4*)(&C[row * n + warp_col])) = *((float4*)(&C_shmem[lane_id][0]));
            *((float4*)(&C[row * n + warp_col + 4])) = *((float4*)(&C_shmem[lane_id][4]));
        }
    }
}

// Test kernel to verify TF32 tensor core support (m16n8k8)
extern "C" __global__ void test_fp32_mma_ptx() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Simple test: FP32 identity-like operation
        // TF32 has same range as FP32, reduced precision (10-bit mantissa)
        unsigned int RA[4] = {0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000};  // FP32(1.0)
        unsigned int RB[2] = {0x3f800000, 0x3f800000};
        float RC[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // FP32 accumulator

        // This will fail at compile time if TF32 tensor cores (m16n8k8) not supported
        // Requires sm_80+ (Ampere/Ada/Hopper)
        HMMA1688_TF32(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1], RC[2], RC[3]);
    }
}
