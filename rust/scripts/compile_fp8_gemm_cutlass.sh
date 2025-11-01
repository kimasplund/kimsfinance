#!/bin/bash
#
# Compile FP8 GEMM CUTLASS kernel for Ada Lovelace (sm_89)
#
# Requirements:
# - CUDA 13.0+ (for FP8 support)
# - CUTLASS 3.5.0 (located at /tmp/cutlass)
# - NVIDIA GPU with compute capability 8.9 (Ada Lovelace)
#
# Usage:
#   ./scripts/compile_fp8_gemm_cutlass.sh
#
# Output:
#   - fp8_gemm_cutlass.cubin (GPU binary)
#   - fp8_gemm_cutlass.ptx (PTX intermediate representation)
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

KERNEL_SOURCE="src/gpu/kernels/fp8_gemm_cutlass.cu"
OUTPUT_CUBIN="fp8_gemm_cutlass.cubin"
OUTPUT_PTX="fp8_gemm_cutlass.ptx"

CUDA_VERSION="13.0"
CUTLASS_PATH="/tmp/cutlass"
CUDA_INCLUDE="/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/include/cccl"

# Compute capability (8.9 = Ada Lovelace)
SM_ARCH="89"

# ============================================================================
# Validation
# ============================================================================

echo "===================================="
echo "FP8 GEMM CUTLASS Kernel Compilation"
echo "===================================="
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit ${CUDA_VERSION}."
    exit 1
fi

# Check CUDA version
NVCC_VERSION=$(nvcc --version | grep "release" | sed -E 's/.*release ([0-9]+\.[0-9]+).*/\1/')
echo "CUDA Version: ${NVCC_VERSION}"

if [[ "${NVCC_VERSION}" < "13.0" ]]; then
    echo "WARNING: CUDA 13.0+ recommended for FP8 support. Found: ${NVCC_VERSION}"
fi

# Check if CUTLASS exists
if [ ! -d "${CUTLASS_PATH}" ]; then
    echo "ERROR: CUTLASS not found at ${CUTLASS_PATH}"
    echo "Download from: https://github.com/NVIDIA/cutlass"
    exit 1
fi

echo "CUTLASS Path: ${CUTLASS_PATH}"

# Check if source file exists
if [ ! -f "${KERNEL_SOURCE}" ]; then
    echo "ERROR: Kernel source not found: ${KERNEL_SOURCE}"
    exit 1
fi

echo "Kernel Source: ${KERNEL_SOURCE}"
echo ""

# ============================================================================
# Compilation
# ============================================================================

echo "Compiling FP8 GEMM kernel..."
echo ""

# Compile to CUBIN (GPU binary)
nvcc -o "${OUTPUT_CUBIN}" \
     -arch=sm_${SM_ARCH} \
     -std=c++17 \
     -I"${CUTLASS_PATH}/include" \
     -I"${CUDA_INCLUDE}" \
     --cubin \
     -O3 \
     -use_fast_math \
     -DNDEBUG \
     "${KERNEL_SOURCE}"

if [ $? -eq 0 ]; then
    echo "✓ CUBIN compiled successfully: ${OUTPUT_CUBIN}"
else
    echo "✗ CUBIN compilation failed"
    exit 1
fi

echo ""

# Compile to PTX (portable intermediate representation)
nvcc -o "${OUTPUT_PTX}" \
     -arch=sm_${SM_ARCH} \
     -std=c++17 \
     -I"${CUTLASS_PATH}/include" \
     -I"${CUDA_INCLUDE}" \
     --ptx \
     -O3 \
     -use_fast_math \
     -DNDEBUG \
     "${KERNEL_SOURCE}"

if [ $? -eq 0 ]; then
    echo "✓ PTX compiled successfully: ${OUTPUT_PTX}"
else
    echo "✗ PTX compilation failed"
    exit 1
fi

echo ""

# ============================================================================
# Validation
# ============================================================================

echo "Validating compiled binaries..."
echo ""

# Check file sizes
CUBIN_SIZE=$(stat -c%s "${OUTPUT_CUBIN}" 2>/dev/null || stat -f%z "${OUTPUT_CUBIN}" 2>/dev/null)
PTX_SIZE=$(stat -c%s "${OUTPUT_PTX}" 2>/dev/null || stat -f%z "${OUTPUT_PTX}" 2>/dev/null)

echo "CUBIN size: ${CUBIN_SIZE} bytes"
echo "PTX size:   ${PTX_SIZE} bytes"

if [ "${CUBIN_SIZE}" -lt 1000 ]; then
    echo "WARNING: CUBIN size suspiciously small (< 1 KB). Compilation may have failed."
fi

if [ "${PTX_SIZE}" -lt 1000 ]; then
    echo "WARNING: PTX size suspiciously small (< 1 KB). Compilation may have failed."
fi

echo ""

# Use cuobjdump to inspect CUBIN (if available)
if command -v cuobjdump &> /dev/null; then
    echo "CUBIN Analysis (cuobjdump):"
    echo "----------------------------"
    cuobjdump -sass "${OUTPUT_CUBIN}" 2>&1 | head -n 20
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================

echo "===================================="
echo "Compilation Complete"
echo "===================================="
echo ""
echo "Output files:"
echo "  - ${OUTPUT_CUBIN} (GPU binary)"
echo "  - ${OUTPUT_PTX} (PTX intermediate)"
echo ""
echo "Next steps:"
echo "  1. Test with: cargo test --features gpu fp8_gemm"
echo "  2. Benchmark with: cargo bench --features gpu fp8_gemm"
echo "  3. Profile with: ncu --set full ./target/release/benchmark"
echo ""
echo "Expected performance (RTX 3500 Ada):"
echo "  - 2-4x speedup over FP32 GEMM"
echo "  - 4x memory bandwidth reduction"
echo "  - <1% numerical error (FP8 E4M3 precision)"
echo ""
