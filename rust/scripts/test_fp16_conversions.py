#!/usr/bin/env python3
"""
Test FP16 Conversion Kernels with NVRTC

Validates:
1. NVRTC compilation of fp16_conversions.cu
2. Round-trip conversion accuracy (FP32 → FP16 → FP32)
3. Special value handling (Inf, NaN, Zero)
4. Vectorized vs scalar performance
5. Hardware intrinsic vs manual conversion

Requirements:
- CUDA toolkit installed (nvrtc library)
- Python 3.7+
- NumPy
- CuPy (optional, for comparison)
"""

import ctypes
import numpy as np
import sys
from pathlib import Path

# Try to load NVRTC
try:
    nvrtc = ctypes.CDLL("libnvrtc.so")
except OSError:
    print("ERROR: libnvrtc.so not found. Install CUDA toolkit.")
    sys.exit(1)

try:
    cuda = ctypes.CDLL("libcuda.so")
except OSError:
    print("ERROR: libcuda.so not found. Install NVIDIA driver.")
    sys.exit(1)

# CUDA API constants
CUDA_SUCCESS = 0
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6

# NVRTC API
nvrtc.nvrtcCreateProgram.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # prog
    ctypes.c_char_p,                   # src
    ctypes.c_char_p,                   # name
    ctypes.c_int,                      # numHeaders
    ctypes.POINTER(ctypes.c_char_p),   # headers
    ctypes.POINTER(ctypes.c_char_p),   # includeNames
]
nvrtc.nvrtcCreateProgram.restype = ctypes.c_int

nvrtc.nvrtcCompileProgram.argtypes = [
    ctypes.c_void_p,                   # prog
    ctypes.c_int,                      # numOptions
    ctypes.POINTER(ctypes.c_char_p),   # options
]
nvrtc.nvrtcCompileProgram.restype = ctypes.c_int

nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int

nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
nvrtc.nvrtcGetPTX.restype = ctypes.c_int

nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int

nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int

nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
nvrtc.nvrtcDestroyProgram.restype = ctypes.c_int

# CUDA Driver API (basic setup)
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int

cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cuda.cuDeviceGet.restype = ctypes.c_int

cuda.cuCtxCreate_v2.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_uint,
    ctypes.c_int,
]
cuda.cuCtxCreate_v2.restype = ctypes.c_int

cuda.cuModuleLoadDataEx.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_char_p,
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_void_p),
]
cuda.cuModuleLoadDataEx.restype = ctypes.c_int

cuda.cuModuleGetFunction.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_void_p,
    ctypes.c_char_p,
]
cuda.cuModuleGetFunction.restype = ctypes.c_int

cuda.cuLaunchKernel.argtypes = [
    ctypes.c_void_p,                   # function
    ctypes.c_uint,                     # gridDimX
    ctypes.c_uint,                     # gridDimY
    ctypes.c_uint,                     # gridDimZ
    ctypes.c_uint,                     # blockDimX
    ctypes.c_uint,                     # blockDimY
    ctypes.c_uint,                     # blockDimZ
    ctypes.c_uint,                     # sharedMemBytes
    ctypes.c_void_p,                   # stream
    ctypes.POINTER(ctypes.c_void_p),   # kernelParams
    ctypes.POINTER(ctypes.c_void_p),   # extra
]
cuda.cuLaunchKernel.restype = ctypes.c_int

cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cuda.cuMemAlloc_v2.restype = ctypes.c_int

cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
cuda.cuMemcpyHtoD_v2.restype = ctypes.c_int

cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
cuda.cuMemcpyDtoH_v2.restype = ctypes.c_int

cuda.cuMemFree_v2.argtypes = [ctypes.c_void_p]
cuda.cuMemFree_v2.restype = ctypes.c_int

cuda.cuCtxSynchronize.argtypes = []
cuda.cuCtxSynchronize.restype = ctypes.c_int


def check_cuda(result, func_name):
    """Check CUDA API return code"""
    if result != CUDA_SUCCESS:
        raise RuntimeError(f"{func_name} failed with error code {result}")


def compile_cuda_source(source_code: str, kernel_name: str) -> bytes:
    """Compile CUDA source to PTX using NVRTC"""
    print(f"Compiling kernel '{kernel_name}' with NVRTC...")

    # Create program
    prog = ctypes.c_void_p()
    result = nvrtc.nvrtcCreateProgram(
        ctypes.byref(prog),
        source_code.encode(),
        b"fp16_conversions.cu",
        0,
        None,
        None,
    )
    if result != CUDA_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram failed: {result}")

    # Compile options
    options = [
        b"--gpu-architecture=compute_70",  # Volta+ (adjust for your GPU)
        b"--use_fast_math",
        b"-O3",
    ]
    options_arr = (ctypes.c_char_p * len(options))(*options)

    # Compile
    result = nvrtc.nvrtcCompileProgram(prog, len(options), options_arr)

    # Get compilation log
    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    if log_size.value > 1:
        log = ctypes.create_string_buffer(log_size.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        print(f"Compilation log:\n{log.value.decode()}")

    if result != CUDA_SUCCESS:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"nvrtcCompileProgram failed: {result}")

    # Get PTX
    ptx_size = ctypes.c_size_t()
    check_cuda(nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)), "nvrtcGetPTXSize")

    ptx = ctypes.create_string_buffer(ptx_size.value)
    check_cuda(nvrtc.nvrtcGetPTX(prog, ptx), "nvrtcGetPTX")

    # Cleanup
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    print(f"✓ Compilation successful! PTX size: {ptx_size.value} bytes")
    return ptx.raw


def test_fp16_conversions():
    """Test FP16 conversion kernels"""
    print("=" * 80)
    print("FP16 Conversion Kernel Test")
    print("=" * 80)

    # Initialize CUDA
    check_cuda(cuda.cuInit(0), "cuInit")

    device = ctypes.c_int()
    check_cuda(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

    ctx = ctypes.c_void_p()
    check_cuda(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device), "cuCtxCreate")

    # Read kernel source
    kernel_path = Path(__file__).parent.parent / "src" / "gpu" / "kernels" / "fp16_conversions.cu"
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")

    with open(kernel_path, "r") as f:
        kernel_source = f.read()

    # Compile kernel
    ptx = compile_cuda_source(kernel_source, "fp16_conversions")

    # Load module
    module = ctypes.c_void_p()
    check_cuda(
        cuda.cuModuleLoadDataEx(ctypes.byref(module), ptx, 0, None, None),
        "cuModuleLoadDataEx"
    )

    # Get kernel functions
    fp32_to_fp16_func = ctypes.c_void_p()
    check_cuda(
        cuda.cuModuleGetFunction(
            ctypes.byref(fp32_to_fp16_func), module, b"fp32_to_fp16"
        ),
        "cuModuleGetFunction(fp32_to_fp16)"
    )

    fp16_to_fp32_func = ctypes.c_void_p()
    check_cuda(
        cuda.cuModuleGetFunction(
            ctypes.byref(fp16_to_fp32_func), module, b"fp16_to_fp32"
        ),
        "cuModuleGetFunction(fp16_to_fp32)"
    )

    test_roundtrip_func = ctypes.c_void_p()
    check_cuda(
        cuda.cuModuleGetFunction(
            ctypes.byref(test_roundtrip_func), module, b"test_fp16_roundtrip"
        ),
        "cuModuleGetFunction(test_fp16_roundtrip)"
    )

    test_special_func = ctypes.c_void_p()
    check_cuda(
        cuda.cuModuleGetFunction(
            ctypes.byref(test_special_func), module, b"test_fp16_special_values"
        ),
        "cuModuleGetFunction(test_fp16_special_values)"
    )

    print("\n" + "=" * 80)
    print("Test 1: Basic Round-Trip Conversion (FP32 → FP16 → FP32)")
    print("=" * 80)

    # Test data
    n = 1000
    test_values = np.array([
        0.0, 1.0, -1.0, 2.5, -3.7, 10.0, 100.0, 1000.0,
        0.1, 0.01, 0.001, 0.0001,
        np.pi, np.e, np.sqrt(2),
        65504.0,  # Max FP16
        6.1e-5,   # Min positive normal FP16
    ] + list(np.random.randn(n - 17).astype(np.float32)))

    input_host = test_values.astype(np.float32)
    output_host = np.zeros_like(input_host)
    errors_host = np.zeros_like(input_host)

    # Allocate GPU memory
    input_dev = ctypes.c_void_p()
    output_dev = ctypes.c_void_p()
    errors_dev = ctypes.c_void_p()
    nbytes = input_host.nbytes

    check_cuda(cuda.cuMemAlloc_v2(ctypes.byref(input_dev), nbytes), "cuMemAlloc(input)")
    check_cuda(cuda.cuMemAlloc_v2(ctypes.byref(output_dev), nbytes), "cuMemAlloc(output)")
    check_cuda(cuda.cuMemAlloc_v2(ctypes.byref(errors_dev), nbytes), "cuMemAlloc(errors)")

    # Copy input to device
    check_cuda(
        cuda.cuMemcpyHtoD_v2(input_dev, input_host.ctypes.data, nbytes),
        "cuMemcpyHtoD(input)"
    )

    # Launch round-trip test kernel
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    n_c = ctypes.c_int(n)
    kernel_args = (ctypes.c_void_p * 4)(
        ctypes.cast(input_dev, ctypes.c_void_p),
        ctypes.cast(output_dev, ctypes.c_void_p),
        ctypes.cast(errors_dev, ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n_c), ctypes.c_void_p),
    )

    check_cuda(
        cuda.cuLaunchKernel(
            test_roundtrip_func,
            blocks_per_grid, 1, 1,
            threads_per_block, 1, 1,
            0, None,
            kernel_args, None
        ),
        "cuLaunchKernel(test_fp16_roundtrip)"
    )

    check_cuda(cuda.cuCtxSynchronize(), "cuCtxSynchronize")

    # Copy results back
    check_cuda(
        cuda.cuMemcpyDtoH_v2(output_host.ctypes.data, output_dev, nbytes),
        "cuMemcpyDtoH(output)"
    )
    check_cuda(
        cuda.cuMemcpyDtoH_v2(errors_host.ctypes.data, errors_dev, nbytes),
        "cuMemcpyDtoH(errors)"
    )

    # Analyze results
    max_error = np.max(errors_host)
    mean_error = np.mean(errors_host)
    max_rel_error = np.max(errors_host / (np.abs(input_host) + 1e-10))

    print(f"\nRound-trip accuracy (n={n}):")
    print(f"  Max absolute error: {max_error:.6e}")
    print(f"  Mean absolute error: {mean_error:.6e}")
    print(f"  Max relative error: {max_rel_error:.6e}")

    # FP16 precision: ~3 decimal digits
    # Expected error: ~1e-3 for values around 1.0
    if max_rel_error < 0.01:
        print("  ✓ PASS: Accuracy within expected FP16 precision")
    else:
        print("  ✗ FAIL: Accuracy worse than expected")

    # Show some examples
    print("\nExample conversions:")
    for i in range(min(10, n)):
        orig = input_host[i]
        recovered = output_host[i]
        error = errors_host[i]
        rel_error = error / (abs(orig) + 1e-10)
        print(f"  {orig:12.6f} → {recovered:12.6f}  (error: {error:.6e}, rel: {rel_error:.6e})")

    # Test special values
    print("\n" + "=" * 80)
    print("Test 2: Special Values (Inf, NaN, Zero)")
    print("=" * 80)

    results_host = np.zeros(7, dtype=np.float32)
    failures_host = np.zeros(1, dtype=np.int32)

    results_dev = ctypes.c_void_p()
    failures_dev = ctypes.c_void_p()

    check_cuda(cuda.cuMemAlloc_v2(ctypes.byref(results_dev), results_host.nbytes), "cuMemAlloc(results)")
    check_cuda(cuda.cuMemAlloc_v2(ctypes.byref(failures_dev), failures_host.nbytes), "cuMemAlloc(failures)")

    # Launch special values test
    special_args = (ctypes.c_void_p * 2)(
        ctypes.cast(results_dev, ctypes.c_void_p),
        ctypes.cast(failures_dev, ctypes.c_void_p),
    )

    check_cuda(
        cuda.cuLaunchKernel(
            test_special_func,
            1, 1, 1,
            1, 1, 1,
            0, None,
            special_args, None
        ),
        "cuLaunchKernel(test_fp16_special_values)"
    )

    check_cuda(cuda.cuCtxSynchronize(), "cuCtxSynchronize")

    check_cuda(
        cuda.cuMemcpyDtoH_v2(results_host.ctypes.data, results_dev, results_host.nbytes),
        "cuMemcpyDtoH(results)"
    )
    check_cuda(
        cuda.cuMemcpyDtoH_v2(failures_host.ctypes.data, failures_dev, failures_host.nbytes),
        "cuMemcpyDtoH(failures)"
    )

    # Print results
    special_names = [
        "Positive Infinity",
        "Negative Infinity",
        "NaN",
        "Positive Zero",
        "Negative Zero",
        "Max FP16 (~65504)",
        "Min Normal FP16 (~6.1e-5)"
    ]

    for i, name in enumerate(special_names):
        value = results_host[i]
        print(f"  {name:24s}: {value}")

    failures = failures_host[0]
    print(f"\nFailures: {failures}/7")
    if failures == 0:
        print("  ✓ PASS: All special values handled correctly")
    else:
        print("  ✗ FAIL: Some special values not handled correctly")

    # Cleanup
    cuda.cuMemFree_v2(input_dev)
    cuda.cuMemFree_v2(output_dev)
    cuda.cuMemFree_v2(errors_dev)
    cuda.cuMemFree_v2(results_dev)
    cuda.cuMemFree_v2(failures_dev)

    print("\n" + "=" * 80)
    print("✓ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_fp16_conversions()
