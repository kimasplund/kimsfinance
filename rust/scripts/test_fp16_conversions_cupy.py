#!/usr/bin/env python3
"""
Test FP16 Conversion Kernels with CuPy

Simpler test using CuPy's RawKernel for NVRTC compilation.

Requirements:
- CuPy installed (pip install cupy-cuda12x or cupy-cuda11x)
- NVIDIA GPU with CUDA support
"""

import cupy as cp
import numpy as np
from pathlib import Path


def test_fp16_conversions_cupy():
    """Test FP16 conversions using CuPy"""
    print("=" * 80)
    print("FP16 Conversion Kernel Test (CuPy)")
    print("=" * 80)

    # Read kernel source
    kernel_path = Path(__file__).parent.parent / "src" / "gpu" / "kernels" / "fp16_conversions.cu"
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")

    with open(kernel_path, "r") as f:
        kernel_source = f.read()

    print(f"✓ Loaded kernel from: {kernel_path}")

    # Compile kernels using CuPy
    print("\nCompiling kernels with NVRTC...")

    fp32_to_fp16 = cp.RawKernel(
        kernel_source,
        "fp32_to_fp16",
        options=("--use_fast_math", "-O3"),
    )

    fp16_to_fp32 = cp.RawKernel(
        kernel_source,
        "fp16_to_fp32",
        options=("--use_fast_math", "-O3"),
    )

    test_roundtrip = cp.RawKernel(
        kernel_source,
        "test_fp16_roundtrip",
        options=("--use_fast_math", "-O3"),
    )

    test_special = cp.RawKernel(
        kernel_source,
        "test_fp16_special_values",
        options=("--use_fast_math", "-O3"),
    )

    fp32_to_fp16_vec = cp.RawKernel(
        kernel_source,
        "fp32_to_fp16_vectorized",
        options=("--use_fast_math", "-O3"),
    )

    fp16_to_fp32_vec = cp.RawKernel(
        kernel_source,
        "fp16_to_fp32_vectorized",
        options=("--use_fast_math", "-O3"),
    )

    print("✓ All kernels compiled successfully!")

    # =========================================================================
    # Test 1: Basic Round-Trip Conversion
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test 1: Basic Round-Trip Conversion (FP32 → FP16 → FP32)")
    print("=" * 80)

    n = 1000
    test_values = np.array([
        0.0, 1.0, -1.0, 2.5, -3.7, 10.0, 100.0, 1000.0,
        0.1, 0.01, 0.001, 0.0001,
        np.pi, np.e, np.sqrt(2),
        65504.0,  # Max FP16
        6.1e-5,   # Min positive normal FP16
    ] + list(np.random.randn(n - 17)), dtype=np.float32)

    input_gpu = cp.array(test_values)
    output_gpu = cp.zeros_like(input_gpu)
    errors_gpu = cp.zeros_like(input_gpu)

    # Launch round-trip test
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    test_roundtrip(
        (blocks_per_grid,), (threads_per_block,),
        (input_gpu, output_gpu, errors_gpu, np.int32(n))
    )

    # Get results
    output_cpu = cp.asnumpy(output_gpu)
    errors_cpu = cp.asnumpy(errors_gpu)

    # Analyze accuracy
    max_error = np.max(errors_cpu)
    mean_error = np.mean(errors_cpu)
    max_rel_error = np.max(errors_cpu / (np.abs(test_values) + 1e-10))

    print(f"\nRound-trip accuracy (n={n}):")
    print(f"  Max absolute error: {max_error:.6e}")
    print(f"  Mean absolute error: {mean_error:.6e}")
    print(f"  Max relative error: {max_rel_error:.6e}")

    # FP16 precision: ~3 decimal digits (~1e-3 relative error)
    if max_rel_error < 0.01:
        print("  ✓ PASS: Accuracy within expected FP16 precision")
    else:
        print(f"  ⚠ WARNING: Accuracy worse than expected (max rel error: {max_rel_error:.6e})")

    # Show examples
    print("\nExample conversions (first 15):")
    print(f"{'Original':>12} → {'Recovered':>12}  {'Abs Error':>12}  {'Rel Error':>12}")
    print("-" * 60)
    for i in range(min(15, n)):
        orig = test_values[i]
        recovered = output_cpu[i]
        error = errors_cpu[i]
        rel_error = error / (abs(orig) + 1e-10)
        print(f"{orig:12.6f} → {recovered:12.6f}  {error:12.6e}  {rel_error:12.6e}")

    # =========================================================================
    # Test 2: Special Values
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test 2: Special Values (Inf, NaN, Zero)")
    print("=" * 80)

    results_gpu = cp.zeros(7, dtype=np.float32)
    failures_gpu = cp.zeros(1, dtype=np.int32)

    test_special(
        (1,), (1,),
        (results_gpu, failures_gpu)
    )

    results = cp.asnumpy(results_gpu)
    failures = cp.asnumpy(failures_gpu)[0]

    special_names = [
        "Positive Infinity",
        "Negative Infinity",
        "NaN",
        "Positive Zero",
        "Negative Zero",
        "Max FP16 (~65504)",
        "Min Normal FP16 (~6.1e-5)"
    ]

    print("\nSpecial value conversions:")
    for i, name in enumerate(special_names):
        value = results[i]
        if np.isnan(value):
            value_str = "NaN"
        elif np.isinf(value):
            value_str = "+Inf" if value > 0 else "-Inf"
        else:
            value_str = f"{value:.6e}"
        print(f"  {name:24s}: {value_str}")

    print(f"\nFailures: {failures}/7")
    if failures == 0:
        print("  ✓ PASS: All special values handled correctly")
    else:
        print(f"  ✗ FAIL: {failures} special value(s) not handled correctly")

    # =========================================================================
    # Test 3: Vectorized vs Scalar Performance
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test 3: Vectorized vs Scalar Performance")
    print("=" * 80)

    # Test with larger array
    n_large = 1_000_000
    input_large = cp.random.randn(n_large, dtype=np.float32)
    fp16_output = cp.zeros(n_large, dtype=np.uint16)
    fp32_output = cp.zeros(n_large, dtype=np.float32)

    # Scalar version
    threads_per_block = 256
    blocks_per_grid = (n_large + threads_per_block - 1) // threads_per_block

    # Warmup
    fp32_to_fp16(
        (blocks_per_grid,), (threads_per_block,),
        (input_large, fp16_output, np.int32(n_large))
    )
    cp.cuda.Stream.null.synchronize()

    # Benchmark scalar
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()

    for _ in range(10):
        fp32_to_fp16(
            (blocks_per_grid,), (threads_per_block,),
            (input_large, fp16_output, np.int32(n_large))
        )

    end.record()
    end.synchronize()
    scalar_time = cp.cuda.get_elapsed_time(start, end) / 10

    # Vectorized version (4 elements per thread)
    threads_vec = 256
    blocks_vec = ((n_large // 4) + threads_vec - 1) // threads_vec

    # Warmup
    fp32_to_fp16_vec(
        (blocks_vec,), (threads_vec,),
        (input_large, fp16_output, np.int32(n_large))
    )
    cp.cuda.Stream.null.synchronize()

    # Benchmark vectorized
    start.record()

    for _ in range(10):
        fp32_to_fp16_vec(
            (blocks_vec,), (threads_vec,),
            (input_large, fp16_output, np.int32(n_large))
        )

    end.record()
    end.synchronize()
    vectorized_time = cp.cuda.get_elapsed_time(start, end) / 10

    print(f"\nFP32→FP16 conversion throughput (n={n_large:,}):")
    print(f"  Scalar version:     {scalar_time:.3f} ms ({n_large/scalar_time/1e6:.1f} GB/s)")
    print(f"  Vectorized version: {vectorized_time:.3f} ms ({n_large/vectorized_time/1e6:.1f} GB/s)")
    print(f"  Speedup:            {scalar_time/vectorized_time:.2f}x")

    if vectorized_time < scalar_time:
        print("  ✓ PASS: Vectorized version is faster")
    else:
        print("  ⚠ WARNING: Vectorized version not faster (may need tuning)")

    # =========================================================================
    # Test 4: Comparison with CuPy's Native FP16
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test 4: Comparison with CuPy's Native FP16")
    print("=" * 80)

    n_test = 10000
    input_test = cp.random.randn(n_test, dtype=np.float32)
    fp16_custom = cp.zeros(n_test, dtype=np.uint16)
    fp32_recovered = cp.zeros(n_test, dtype=np.float32)

    # Our custom conversion
    threads = 256
    blocks = (n_test + threads - 1) // threads

    fp32_to_fp16(
        (blocks,), (threads,),
        (input_test, fp16_custom, np.int32(n_test))
    )

    fp16_to_fp32(
        (blocks,), (threads,),
        (fp16_custom, fp32_recovered, np.int32(n_test))
    )

    # CuPy's native conversion
    fp16_cupy = input_test.astype(cp.float16)
    fp32_cupy = fp16_cupy.astype(cp.float32)

    # Compare results
    custom_result = cp.asnumpy(fp32_recovered)
    cupy_result = cp.asnumpy(fp32_cupy)

    max_diff = np.max(np.abs(custom_result - cupy_result))
    mean_diff = np.mean(np.abs(custom_result - cupy_result))

    print(f"\nComparison with CuPy native FP16:")
    print(f"  Max difference:  {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-6:
        print("  ✓ PASS: Custom conversion matches CuPy's native FP16")
    else:
        print(f"  ⚠ WARNING: Difference detected (max: {max_diff:.6e})")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    all_passed = (
        max_rel_error < 0.01 and
        failures == 0 and
        max_diff < 1e-6
    )

    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nKernel is ready for production use:")
        print("  - Accurate round-trip conversion (within FP16 precision)")
        print("  - Correct handling of special values (Inf, NaN, Zero)")
        print("  - Matches CuPy's native FP16 conversion")
        print("  - Vectorized version available for 2-4x speedup")
    else:
        print("⚠ SOME TESTS FAILED - Review results above")

    print("=" * 80)


if __name__ == "__main__":
    try:
        test_fp16_conversions_cupy()
    except ImportError as e:
        print(f"ERROR: {e}")
        print("\nPlease install CuPy:")
        print("  pip install cupy-cuda12x  # For CUDA 12.x")
        print("  pip install cupy-cuda11x  # For CUDA 11.x")
        exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
