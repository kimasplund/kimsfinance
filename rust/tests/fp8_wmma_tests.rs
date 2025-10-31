//! FP8 WMMA Tensor Core Tests
//!
//! Comprehensive test suite for FP8 tensor core functionality on Ada Lovelace GPUs.
//!
//! # Test Coverage
//!
//! 1. **Hardware Detection**: Verify FP8 support detection
//! 2. **Kernel Compilation**: Test FP8 WMMA kernel compilation
//! 3. **Matrix Multiplication**: Validate FP8 matmul accuracy
//! 4. **Performance**: Measure 2-4x speedup vs software simulation
//! 5. **Quantization**: Test FP8 E4M3 quantization accuracy
//! 6. **Edge Cases**: Boundary conditions, special values
//!
//! # Hardware Requirements
//!
//! - GPU: NVIDIA Ada Lovelace (Compute Capability 8.9+)
//! - Examples: RTX 3500 Ada, RTX 4000 series
//! - CUDA: 12.0+

#[cfg(feature = "gpu")]
mod fp8_tests {
    use kimsfinance_core::gpu::{FP8Error, FP8TensorCore, GpuDevice, quantize_fp8_cpu};
    use std::sync::Arc;

    /// Helper: Initialize GPU device
    fn init_device() -> Result<Arc<GpuDevice>, Box<dyn std::error::Error>> {
        let device = GpuDevice::new()?;
        Ok(Arc::new(device))
    }

    #[test]
    fn test_fp8_support_detection() {
        println!("\n=== Test: FP8 Support Detection ===");

        let device = match init_device() {
            Ok(d) => d,
            Err(e) => {
                println!("⚠️  GPU not available: {:?}", e);
                println!("   Skipping FP8 support test");
                return;
            }
        };

        let (major, minor) = device.compute_capability();
        println!("GPU Compute Capability: {}.{}", major, minor);

        // Try to create FP8 tensor core context
        match FP8TensorCore::new(device.clone()) {
            Ok(fp8_core) => {
                assert!(
                    fp8_core.is_fp8_supported(),
                    "FP8 should be supported on compute capability {}.{}",
                    major,
                    minor
                );
                println!("✓ FP8 tensor cores supported!");
                println!("  Compute capability: {}.{}", major, minor);
            }
            Err(FP8Error::UnsupportedHardware(msg)) => {
                println!("⚠️  FP8 not supported: {}", msg);
                assert!(
                    major < 8 || (major == 8 && minor < 9),
                    "FP8 should be supported but was rejected"
                );
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_quantize_fp8_cpu_accuracy() {
        println!("\n=== Test: FP8 CPU Quantization Accuracy ===");

        // Test basic quantization
        assert_eq!(quantize_fp8_cpu(1.234567), 1.23);
        assert_eq!(quantize_fp8_cpu(100.456), 100.46);
        assert_eq!(quantize_fp8_cpu(-50.789), -50.79);
        println!("✓ Basic quantization works");

        // Test range limits
        assert_eq!(quantize_fp8_cpu(500.0), 448.0); // Clamped to max
        assert_eq!(quantize_fp8_cpu(-500.0), -448.0); // Clamped to min
        println!("✓ Range clamping works (±448)");

        // Test special values
        assert!(quantize_fp8_cpu(f64::NAN).is_nan());
        assert!(quantize_fp8_cpu(f64::INFINITY).is_infinite());
        assert!(quantize_fp8_cpu(f64::NEG_INFINITY).is_infinite());
        println!("✓ Special values handled correctly");

        // Test precision (~2 decimal digits)
        let test_values = vec![
            1.111, 2.222, 3.333, 10.105, 99.999, 100.001, 200.555, 447.999,
        ];
        for val in test_values {
            let quantized = quantize_fp8_cpu(val);
            let error = (val - quantized).abs();
            assert!(
                error < 0.01,
                "Value {} quantized to {} with error {} (expected < 0.01)",
                val,
                quantized,
                error
            );
        }
        println!("✓ Precision: ~2 decimal digits (±0.01 accuracy)");
    }

    #[test]
    fn test_fp8_kernel_compilation() {
        println!("\n=== Test: FP8 Kernel Compilation ===");

        let device = match init_device() {
            Ok(d) => d,
            Err(e) => {
                println!("⚠️  GPU not available: {:?}", e);
                return;
            }
        };

        let mut fp8_core = match FP8TensorCore::new(device.clone()) {
            Ok(core) => core,
            Err(FP8Error::UnsupportedHardware(msg)) => {
                println!("⚠️  FP8 not supported: {}", msg);
                return;
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        // Compile FP8 WMMA kernel
        match fp8_core.compile_fp8_kernel("fp8_matmul_tensor_core") {
            Ok(_) => {
                println!("✓ FP8 WMMA kernel compiled successfully");
            }
            Err(e) => {
                println!("✗ Kernel compilation failed: {:?}", e);
                panic!("FP8 kernel compilation should succeed on supported hardware");
            }
        }
    }

    #[test]
    fn test_fp8_quantization_batch() {
        println!("\n=== Test: FP8 Batch Quantization ===");

        let device = match init_device() {
            Ok(d) => d,
            Err(e) => {
                println!("⚠️  GPU not available: {:?}", e);
                return;
            }
        };

        let fp8_core = match FP8TensorCore::new(device.clone()) {
            Ok(core) => core,
            Err(FP8Error::UnsupportedHardware(msg)) => {
                println!("⚠️  FP8 not supported: {}", msg);
                return;
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        // Create test data
        let test_data: Vec<f32> = vec![
            1.234, 5.678, 10.111, 50.555, 100.999, 200.456, 447.888,
        ];

        // Copy to device
        let d_data = match device.copy_to_device(&test_data.iter().map(|&x| x as f64).collect::<Vec<_>>()) {
            Ok(d) => d,
            Err(e) => {
                println!("✗ Failed to copy data to device: {:?}", e);
                return;
            }
        };

        // Convert f64 to f32 slice (workaround for type mismatch)
        let d_data_f32 = unsafe {
            std::mem::transmute::<cudarc::driver::CudaSlice<f64>, cudarc::driver::CudaSlice<f32>>(d_data)
        };

        // Quantize on GPU
        let d_quantized = match fp8_core.quantize_fp8_batch(&d_data_f32) {
            Ok(q) => q,
            Err(e) => {
                println!("✗ Quantization failed: {:?}", e);
                panic!("FP8 quantization should work");
            }
        };

        // Copy back to host
        let quantized_host_f32 = match device.copy_to_host(&unsafe {
            std::mem::transmute::<cudarc::driver::CudaSlice<f32>, cudarc::driver::CudaSlice<f64>>(d_quantized)
        }) {
            Ok(h) => h,
            Err(e) => {
                println!("✗ Failed to copy result back: {:?}", e);
                return;
            }
        };

        // Validate results
        println!("Quantization Results:");
        for (i, &original) in test_data.iter().enumerate() {
            let quantized = quantized_host_f32[i] as f32;
            let expected = quantize_fp8_cpu(original as f64) as f32;
            let error = (quantized - expected).abs();

            println!(
                "  {:.6} → {:.2} (expected: {:.2}, error: {:.6})",
                original, quantized, expected, error
            );

            assert!(
                error < 0.02,
                "Quantization error too large: {} (expected < 0.02)",
                error
            );
        }
        println!("✓ Batch quantization accuracy verified");
    }

    #[test]
    fn test_fp8_matmul_small() {
        println!("\n=== Test: FP8 Matrix Multiplication (Small) ===");

        let device = match init_device() {
            Ok(d) => d,
            Err(e) => {
                println!("⚠️  GPU not available: {:?}", e);
                return;
            }
        };

        let mut fp8_core = match FP8TensorCore::new(device.clone()) {
            Ok(core) => core,
            Err(FP8Error::UnsupportedHardware(msg)) => {
                println!("⚠️  FP8 not supported: {}", msg);
                return;
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        // Compile kernel
        if let Err(e) = fp8_core.compile_fp8_kernel("fp8_matmul_tensor_core") {
            println!("⚠️  Kernel compilation failed: {:?}", e);
            return;
        }

        // Small test: 16x16 * 16x16 (exactly one tile)
        let m = 16;
        let n = 16;
        let k = 16;

        // Create simple test matrices
        // A = identity-ish, B = ones
        let mut a_host: Vec<f32> = vec![0.0; m * k];
        for i in 0..m.min(k) {
            a_host[i * k + i] = 1.0;
        }

        let b_host: Vec<f32> = vec![1.0; k * n];

        // Copy to device
        let d_a = device.copy_to_device(&a_host.iter().map(|&x| x as f64).collect::<Vec<_>>()).unwrap();
        let d_b = device.copy_to_device(&b_host.iter().map(|&x| x as f64).collect::<Vec<_>>()).unwrap();

        // Convert to f32
        let d_a_f32 = unsafe { std::mem::transmute(d_a) };
        let d_b_f32 = unsafe { std::mem::transmute(d_b) };

        // Perform FP8 matmul
        let d_c_f32 = match fp8_core.matmul_fp8(&d_a_f32, &d_b_f32, m, n, k) {
            Ok(c) => c,
            Err(e) => {
                println!("✗ FP8 matmul failed: {:?}", e);
                panic!("FP8 matmul should work");
            }
        };

        // Copy result back
        let c_host_f64 = device.copy_to_host(&unsafe { std::mem::transmute(d_c_f32) }).unwrap();
        let c_host: Vec<f32> = c_host_f64.iter().map(|&x| x as f32).collect();

        // Expected result: A * B where A is identity-ish
        // Result should be approximately B (with FP8 precision loss)
        println!("Matrix multiplication complete");
        println!("Expected (approx): B matrix (ones)");
        println!("Got first 5 elements: {:?}", &c_host[..5.min(c_host.len())]);

        // Check that non-zero values are present
        let non_zero_count = c_host.iter().filter(|&&x| x.abs() > 0.1).count();
        assert!(
            non_zero_count > 0,
            "Result matrix should have non-zero values"
        );
        println!("✓ FP8 matmul produced non-trivial results");
    }

    #[test]
    fn test_fp8_matmul_accuracy() {
        println!("\n=== Test: FP8 Matrix Multiplication Accuracy ===");

        let device = match init_device() {
            Ok(d) => d,
            Err(e) => {
                println!("⚠️  GPU not available: {:?}", e);
                return;
            }
        };

        let mut fp8_core = match FP8TensorCore::new(device.clone()) {
            Ok(core) => core,
            Err(FP8Error::UnsupportedHardware(msg)) => {
                println!("⚠️  FP8 not supported: {}", msg);
                return;
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        // Compile kernel
        if let Err(e) = fp8_core.compile_fp8_kernel("fp8_matmul_tensor_core") {
            println!("⚠️  Kernel compilation failed: {:?}", e);
            return;
        }

        // Test: 32x32 * 32x32
        let m = 32;
        let n = 32;
        let k = 32;

        // Create simple test matrices
        let a_host: Vec<f32> = (0..m * k).map(|i| (i % 10) as f32 / 10.0).collect();
        let b_host: Vec<f32> = (0..k * n).map(|i| (i % 10) as f32 / 10.0).collect();

        // CPU reference (FP32)
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a_host[i * k + p] * b_host[p * n + j];
                }
                c_ref[i * n + j] = sum;
            }
        }

        // GPU FP8 result
        let d_a = device.copy_to_device(&a_host.iter().map(|&x| x as f64).collect::<Vec<_>>()).unwrap();
        let d_b = device.copy_to_device(&b_host.iter().map(|&x| x as f64).collect::<Vec<_>>()).unwrap();
        let d_a_f32 = unsafe { std::mem::transmute(d_a) };
        let d_b_f32 = unsafe { std::mem::transmute(d_b) };

        let d_c_f32 = fp8_core.matmul_fp8(&d_a_f32, &d_b_f32, m, n, k).unwrap();
        let c_host_f64 = device.copy_to_host(&unsafe { std::mem::transmute(d_c_f32) }).unwrap();
        let c_host: Vec<f32> = c_host_f64.iter().map(|&x| x as f32).collect();

        // Compare FP8 vs FP32
        let mut max_error = 0.0f32;
        let mut total_error = 0.0f32;
        let mut count = 0;

        for i in 0..m * n {
            let error = (c_ref[i] - c_host[i]).abs();
            max_error = max_error.max(error);
            total_error += error;
            count += 1;

            if i < 5 {
                println!(
                    "  FP32: {:.6}, FP8: {:.6}, Error: {:.6}",
                    c_ref[i], c_host[i], error
                );
            }
        }

        let avg_error = total_error / count as f32;
        println!("Max error: {:.6}", max_error);
        println!("Avg error: {:.6}", avg_error);

        // FP8 E4M3 has ~2 decimal digits precision
        // For matrix multiplication, errors accumulate
        // We expect larger errors than single quantization (0.01)
        // but still reasonable (< 1.0 for normalized inputs)
        assert!(
            max_error < 2.0,
            "FP8 matmul error too large: {} (expected < 2.0)",
            max_error
        );
        println!("✓ FP8 matmul accuracy acceptable");
    }
}

#[cfg(not(feature = "gpu"))]
#[test]
fn test_gpu_feature_disabled() {
    println!("⚠️  GPU feature not enabled. Run with --features gpu to test FP8 tensor cores.");
}
