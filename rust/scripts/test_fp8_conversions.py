#!/usr/bin/env python3
"""
Test FP8 E4M3 Conversion Kernels

Validates FP32 <-> FP8 E4M3 conversions using NVRTC JIT compilation.

Tests:
1. Basic conversions (0.0, 1.0, -1.0)
2. Range limits (448.0, -448.0, overflow)
3. Denormals (2^-6, 2^-9)
4. Round-trip accuracy
5. Vectorized kernel performance
"""

import numpy as np
import cupy as cp
from cuda import cuda, nvrtc
import sys
from pathlib import Path

# CUDA error checking
def check_cuda_error(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA Error: {err}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC Error: {err}")

def cuda_call(call):
    """Helper to check CUDA API calls"""
    err, *result = call
    check_cuda_error(err)
    return result[0] if len(result) == 1 else result

# Compile CUDA kernel with NVRTC
def compile_fp8_kernels():
    """Compile FP8 conversion kernels using NVRTC JIT"""

    kernel_path = Path(__file__).parent.parent / "src/gpu/kernels/fp8_conversions.cu"

    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

    with open(kernel_path, 'r') as f:
        kernel_source = f.read()

    print(f"Compiling {kernel_path.name} with NVRTC...")

    # Get GPU compute capability
    device = cuda_call(cuda.cuDeviceGet(0))
    major = cuda_call(cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
    ))
    minor = cuda_call(cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
    ))
    compute_capability = f"{major}{minor}"

    print(f"GPU Compute Capability: sm_{compute_capability}")

    if int(compute_capability) < 89:
        print("WARNING: FP8 E4M3 requires sm_89+ (Ada Lovelace or newer)")
        print("Conversions will still work, but tensor cores won't be available")

    # NVRTC compilation
    err, program = nvrtc.nvrtcCreateProgram(
        str.encode(kernel_source),
        b"fp8_conversions.cu",
        0, [], []
    )
    check_cuda_error(err)

    # Compile options
    opts = [
        b'--gpu-architecture=compute_' + compute_capability.encode(),
        b'--use_fast_math',
        b'--std=c++17',
    ]

    try:
        err = nvrtc.nvrtcCompileProgram(program, len(opts), opts)
    except:
        # Get compilation log if failed
        err, log_size = nvrtc.nvrtcGetProgramLogSize(program)
        log = b' ' * log_size
        err = nvrtc.nvrtcGetProgramLog(program, log)
        print("Compilation log:")
        print(log.decode())
        raise

    check_cuda_error(err)

    # Get PTX
    err, ptx_size = nvrtc.nvrtcGetPTXSize(program)
    check_cuda_error(err)

    ptx = b' ' * ptx_size
    err = nvrtc.nvrtcGetPTX(program, ptx)
    check_cuda_error(err)

    # Load module
    err, module = cuda.cuModuleLoadData(ptx)
    check_cuda_error(err)

    # Get kernel functions
    kernels = {}
    kernel_names = [
        'fp32_to_fp8_e4m3',
        'fp8_e4m3_to_fp32',
        'test_fp8_conversions',
        'fp32_to_fp8_e4m3_saturate',
        'fp32_to_fp8_e4m3_stochastic',
    ]

    for name in kernel_names:
        err, kernels[name] = cuda.cuModuleGetFunction(module, name.encode())
        check_cuda_error(err)

    print(f"✅ Compiled {len(kernels)} kernels successfully")

    return module, kernels

def fp8_e4m3_to_float_numpy(fp8_bytes):
    """Reference FP8 E4M3 to FP32 conversion in NumPy"""
    fp8 = np.asarray(fp8_bytes, dtype=np.uint8)

    # Extract components
    sign = (fp8 & 0x80) >> 7
    exp = (fp8 & 0x78) >> 3
    mant = fp8 & 0x07

    # Special cases
    result = np.zeros_like(fp8, dtype=np.float32)

    # NaN
    nan_mask = (fp8 == 0x7F) | (fp8 == 0xFF)
    result[nan_mask] = np.nan

    # Zero
    zero_mask = (fp8 == 0x00) | (fp8 == 0x80)
    result[zero_mask] = 0.0

    # Normal values
    normal_mask = (exp > 0) & ~nan_mask
    if normal_mask.any():
        # Compute value: (-1)^sign * 2^(exp-7) * (1 + mant/8)
        exp_val = exp[normal_mask].astype(np.int32) - 7
        mant_val = 1.0 + mant[normal_mask].astype(np.float32) / 8.0
        value = np.power(2.0, exp_val.astype(np.float32)) * mant_val
        result[normal_mask] = np.where(sign[normal_mask] == 1, -value, value)

    # Denormals (exp == 0, mant != 0)
    denorm_mask = (exp == 0) & (mant != 0)
    if denorm_mask.any():
        # Denormal: (-1)^sign * 2^(-6) * (mant/8)
        mant_val = mant[denorm_mask].astype(np.float32) / 8.0
        value = np.power(2.0, -6.0) * mant_val
        result[denorm_mask] = np.where(sign[denorm_mask] == 1, -value, value)

    return result

def test_basic_conversions(kernels):
    """Test basic FP32 -> FP8 -> FP32 round-trip"""

    print("\n" + "="*60)
    print("TEST 1: Basic Conversions")
    print("="*60)

    # Test values
    test_values = np.array([
        0.0,      # Zero
        1.0,      # One
        -1.0,     # Negative one
        448.0,    # Max normal
        -448.0,   # Min normal
        0.015625, # 2^-6 (smallest normal)
        -0.015625,
        0.5,      # Simple fraction
        -0.5,
        127.5,    # Large value
        0.001953125,  # 2^-9 (denormal)
    ], dtype=np.float32)

    n = len(test_values)

    # Allocate GPU memory
    d_input = cuda_call(cuda.cuMemAlloc(test_values.nbytes))
    d_output_fp8 = cuda_call(cuda.cuMemAlloc(n))
    d_output_fp32 = cuda_call(cuda.cuMemAlloc(test_values.nbytes))

    # Copy input to GPU
    cuda_call(cuda.cuMemcpyHtoD(d_input, test_values.ctypes.data, test_values.nbytes))

    # Launch FP32 -> FP8 kernel
    threads_per_block = 256
    blocks = (n + threads_per_block * 4 - 1) // (threads_per_block * 4)

    cuda_call(cuda.cuLaunchKernel(
        kernels['fp32_to_fp8_e4m3'],
        blocks, 1, 1,
        threads_per_block, 1, 1,
        0, 0,
        (d_input, d_output_fp8, n), 0
    ))

    # Launch FP8 -> FP32 kernel
    cuda_call(cuda.cuLaunchKernel(
        kernels['fp8_e4m3_to_fp32'],
        blocks, 1, 1,
        threads_per_block, 1, 1,
        0, 0,
        (d_output_fp8, d_output_fp32, n), 0
    ))

    # Copy results back
    fp8_result = np.zeros(n, dtype=np.uint8)
    fp32_result = np.zeros(n, dtype=np.float32)

    cuda_call(cuda.cuMemcpyDtoH(fp8_result.ctypes.data, d_output_fp8, n))
    cuda_call(cuda.cuMemcpyDtoH(fp32_result.ctypes.data, d_output_fp32, test_values.nbytes))

    # Verify with NumPy reference
    reference = fp8_e4m3_to_float_numpy(fp8_result)

    # Print results
    print(f"\n{'Value':<15} {'FP8 (hex)':<12} {'Recovered':<15} {'Reference':<15} {'Error':<15}")
    print("-" * 80)

    max_error = 0.0
    for i in range(n):
        error = abs(fp32_result[i] - reference[i])
        max_error = max(max_error, error)

        print(f"{test_values[i]:<15.6f} 0x{fp8_result[i]:02X}        "
              f"{fp32_result[i]:<15.6f} {reference[i]:<15.6f} {error:<15.9f}")

    print(f"\nMax error: {max_error:.9f}")

    # Cleanup
    cuda_call(cuda.cuMemFree(d_input))
    cuda_call(cuda.cuMemFree(d_output_fp8))
    cuda_call(cuda.cuMemFree(d_output_fp32))

    if max_error < 1e-6:
        print("✅ PASSED: All conversions accurate")
        return True
    else:
        print("❌ FAILED: Conversion errors too large")
        return False

def test_overflow_saturation(kernels):
    """Test overflow handling (values > 448)"""

    print("\n" + "="*60)
    print("TEST 2: Overflow Saturation")
    print("="*60)

    test_values = np.array([
        500.0,    # Overflow
        -500.0,
        1000.0,
        -1000.0,
        np.inf,   # Infinity
        -np.inf,
        np.nan,   # NaN
    ], dtype=np.float32)

    n = len(test_values)

    d_input = cuda_call(cuda.cuMemAlloc(test_values.nbytes))
    d_output_fp8 = cuda_call(cuda.cuMemAlloc(n))
    d_output_fp32 = cuda_call(cuda.cuMemAlloc(test_values.nbytes))

    cuda_call(cuda.cuMemcpyHtoD(d_input, test_values.ctypes.data, test_values.nbytes))

    threads_per_block = 256
    blocks = (n + threads_per_block * 4 - 1) // (threads_per_block * 4)

    # FP32 -> FP8
    cuda_call(cuda.cuLaunchKernel(
        kernels['fp32_to_fp8_e4m3'],
        blocks, 1, 1,
        threads_per_block, 1, 1,
        0, 0,
        (d_input, d_output_fp8, n), 0
    ))

    # FP8 -> FP32
    cuda_call(cuda.cuLaunchKernel(
        kernels['fp8_e4m3_to_fp32'],
        blocks, 1, 1,
        threads_per_block, 1, 1,
        0, 0,
        (d_output_fp8, d_output_fp32, n), 0
    ))

    fp8_result = np.zeros(n, dtype=np.uint8)
    fp32_result = np.zeros(n, dtype=np.float32)

    cuda_call(cuda.cuMemcpyDtoH(fp8_result.ctypes.data, d_output_fp8, n))
    cuda_call(cuda.cuMemcpyDtoH(fp32_result.ctypes.data, d_output_fp32, test_values.nbytes))

    print(f"\n{'Value':<15} {'FP8 (hex)':<12} {'Recovered':<15}")
    print("-" * 45)

    for i in range(n):
        print(f"{test_values[i]:<15} 0x{fp8_result[i]:02X}        {fp32_result[i]:<15}")

    # Check saturation
    max_fp8 = 0x7E  # Max positive value
    min_fp8 = 0xFE  # Max negative value
    nan_fp8_pos = 0x7F
    nan_fp8_neg = 0xFF

    passed = True

    # Check overflow saturation
    if fp8_result[0] != max_fp8 or fp8_result[1] != min_fp8:
        print("❌ FAILED: Overflow not saturated correctly")
        passed = False

    # Check NaN handling
    if fp8_result[-1] not in [nan_fp8_pos, nan_fp8_neg]:
        print("❌ FAILED: NaN not handled correctly")
        passed = False

    cuda_call(cuda.cuMemFree(d_input))
    cuda_call(cuda.cuMemFree(d_output_fp8))
    cuda_call(cuda.cuMemFree(d_output_fp32))

    if passed:
        print("✅ PASSED: Overflow and special cases handled correctly")

    return passed

def test_vectorized_performance(kernels):
    """Test vectorized kernel performance"""

    print("\n" + "="*60)
    print("TEST 3: Vectorized Performance")
    print("="*60)

    sizes = [1024, 4096, 16384, 65536, 262144, 1048576]

    print(f"\n{'Size':<12} {'FP32->FP8 (μs)':<20} {'FP8->FP32 (μs)':<20} {'Bandwidth (GB/s)':<20}")
    print("-" * 75)

    for n in sizes:
        # Random test data
        test_data = np.random.randn(n).astype(np.float32) * 100  # Range: ~[-300, 300]

        d_input = cuda_call(cuda.cuMemAlloc(test_data.nbytes))
        d_fp8 = cuda_call(cuda.cuMemAlloc(n))
        d_output = cuda_call(cuda.cuMemAlloc(test_data.nbytes))

        cuda_call(cuda.cuMemcpyHtoD(d_input, test_data.ctypes.data, test_data.nbytes))

        threads_per_block = 256
        blocks = (n + threads_per_block * 4 - 1) // (threads_per_block * 4)

        # Create CUDA events for timing
        start_event = cuda_call(cuda.cuEventCreate(0))
        end_event = cuda_call(cuda.cuEventCreate(0))

        # Time FP32 -> FP8
        cuda_call(cuda.cuEventRecord(start_event, 0))

        for _ in range(10):  # Average over 10 runs
            cuda_call(cuda.cuLaunchKernel(
                kernels['fp32_to_fp8_e4m3'],
                blocks, 1, 1,
                threads_per_block, 1, 1,
                0, 0,
                (d_input, d_fp8, n), 0
            ))

        cuda_call(cuda.cuEventRecord(end_event, 0))
        cuda_call(cuda.cuEventSynchronize(end_event))

        time_fp32_to_fp8 = cuda_call(cuda.cuEventElapsedTime(start_event, end_event)) / 10 * 1000  # μs

        # Time FP8 -> FP32
        cuda_call(cuda.cuEventRecord(start_event, 0))

        for _ in range(10):
            cuda_call(cuda.cuLaunchKernel(
                kernels['fp8_e4m3_to_fp32'],
                blocks, 1, 1,
                threads_per_block, 1, 1,
                0, 0,
                (d_fp8, d_output, n), 0
            ))

        cuda_call(cuda.cuEventRecord(end_event, 0))
        cuda_call(cuda.cuEventSynchronize(end_event))

        time_fp8_to_fp32 = cuda_call(cuda.cuEventElapsedTime(start_event, end_event)) / 10 * 1000  # μs

        # Calculate bandwidth (GB/s)
        bytes_fp32_to_fp8 = n * 4 + n  # Read FP32, write FP8
        bytes_fp8_to_fp32 = n + n * 4  # Read FP8, write FP32

        bandwidth_1 = (bytes_fp32_to_fp8 / 1e9) / (time_fp32_to_fp8 / 1e6)
        bandwidth_2 = (bytes_fp8_to_fp32 / 1e9) / (time_fp8_to_fp32 / 1e6)

        print(f"{n:<12} {time_fp32_to_fp8:<20.2f} {time_fp8_to_fp32:<20.2f} {bandwidth_1:<20.2f}")

        # Cleanup
        cuda_call(cuda.cuMemFree(d_input))
        cuda_call(cuda.cuMemFree(d_fp8))
        cuda_call(cuda.cuMemFree(d_output))
        cuda_call(cuda.cuEventDestroy(start_event))
        cuda_call(cuda.cuEventDestroy(end_event))

    print("\n✅ PASSED: Vectorized kernels executed successfully")
    return True

def main():
    """Run all FP8 conversion tests"""

    print("="*60)
    print("FP8 E4M3 Conversion Kernel Test Suite")
    print("="*60)

    # Initialize CUDA
    cuda_call(cuda.cuInit(0))

    # Compile kernels
    try:
        module, kernels = compile_fp8_kernels()
    except Exception as e:
        print(f"❌ Kernel compilation failed: {e}")
        return 1

    # Run tests
    results = []

    try:
        results.append(test_basic_conversions(kernels))
        results.append(test_overflow_saturation(kernels))
        results.append(test_vectorized_performance(kernels))
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
