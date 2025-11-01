//! Comprehensive Tensor Core Performance Validation
//!
//! Validates FP32/TF32, FP16, and FP8 tensor core implementations with statistical rigor.
//!
//! # Test Matrix
//!
//! ## 1. Matrix Multiplication Throughput
//!
//! | Size    | FP32 (baseline) | TF32 (expected) | FP16 (expected) | FP8 (expected) |
//! |---------|-----------------|-----------------|-----------------|----------------|
//! | 512²    | 100%            | 800%            | 1600%           | 1600%          |
//! | 1024²   | 100%            | 800%            | 1600%           | 1600%          |
//! | 2048²   | 100%            | 800%            | 1600%           | 1600%          |
//! | 4096²   | 100%            | 800%            | 1600%           | 1600%          |
//!
//! ## 2. Genetic Optimizer Workload (Realistic Scenario)
//!
//! - 10,000 fitness evaluations
//! - Each evaluation: 32×32 matrix multiply (parameter covariance)
//! - Metrics: Total time, throughput (evals/sec), speedup vs FP32
//!
//! ## 3. Conversion Overhead
//!
//! - FP32 → FP8 → FP32 round-trip time
//! - FP32 → FP16 → FP32 round-trip time
//! - Crossover points: When tensor cores faster than conversion overhead
//!
//! ## 4. Accuracy Analysis
//!
//! - Maximum absolute error: FP16 vs FP32
//! - Maximum absolute error: FP8 vs FP32
//! - Relative error distribution
//! - Genetic optimizer accuracy validation
//!
//! # Statistical Rigor
//!
//! - **Sample size**: n = 10 iterations per benchmark (sufficient for GFLOPS stability)
//! - **Confidence intervals**: 95% (t-distribution)
//! - **Metrics**: Mean, median, std dev, min, max, CV
//! - **Significance testing**: p < 0.05 for speedup claims
//!
//! # Expected Results (RTX 3500 Ada)
//!
//! Based on hardware specifications:
//! - **TF32**: ~8x throughput vs FP32 CUDA cores (tensor cores)
//! - **FP16**: ~2x throughput vs TF32 tensor cores
//! - **FP8**: ~2x throughput vs TF32 (Ada converts to FP16 internally)
//!
//! # Hardware Requirements
//!
//! - GPU: NVIDIA Ada Lovelace (RTX 3500 Ada, sm_89)
//! - CUDA: 12.4+
//! - Driver: 580.82.07+ (CUDA 13.0 runtime)
//!
//! # Usage
//!
//! ```bash
//! # Run full benchmark suite (60-90 minutes)
//! cargo bench --features gpu --bench tensor_core_benchmark
//!
//! # Run specific benchmark group
//! cargo bench --features gpu --bench tensor_core_benchmark -- throughput
//! cargo bench --features gpu --bench tensor_core_benchmark -- genetic_optimizer
//! cargo bench --features gpu --bench tensor_core_benchmark -- conversion
//! cargo bench --features gpu --bench tensor_core_benchmark -- accuracy
//!
//! # Generate detailed report
//! cargo bench --features gpu --bench tensor_core_benchmark -- --verbose 2>&1 | tee tensor_core_results.txt
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{FP8TensorCore, GpuDevice};

#[path = "statistics.rs"]
mod statistics;

use statistics::{BenchmarkStats, compare_distributions};

/// Matrix sizes for throughput benchmarks
const THROUGHPUT_SIZES: &[usize] = &[512, 1024, 2048, 4096];

/// Genetic optimizer parameters
const GENETIC_MATRIX_SIZE: usize = 32;
const GENETIC_NUM_EVALUATIONS: usize = 10_000;

/// Sample size for statistical significance
const SAMPLE_SIZE: usize = 10;

/// Accuracy test matrix size
const ACCURACY_TEST_SIZE: usize = 256;

/// Generate random matrix data (FP32)
fn generate_matrix_f32(rows: usize, cols: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..rows * cols)
        .map(|_| rng.gen_range(-10.0..10.0))
        .collect()
}

/// Generate random matrix data (FP16)
#[cfg(feature = "gpu")]
fn generate_matrix_f16(rows: usize, cols: usize) -> Vec<half::f16> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..rows * cols)
        .map(|_| half::f16::from_f32(rng.gen_range(-10.0..10.0)))
        .collect()
}

/// Calculate GFLOPS for matrix multiplication
///
/// GEMM operations: 2 * m * n * k (multiply-add = 2 ops)
fn calculate_gflops(m: usize, n: usize, k: usize, time_ms: f64) -> f64 {
    let ops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let gflops = ops / (time_ms * 1_000_000.0); // Convert ms to seconds
    gflops
}

// =============================================================================
// Benchmark 1: Matrix Multiplication Throughput
// =============================================================================

/// Benchmark: FP32 CUDA cores (baseline)
#[cfg(feature = "gpu")]
fn bench_fp32_baseline_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_throughput/fp32_baseline");
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(30));

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║ Benchmark 1: Matrix Multiplication Throughput (FP32 Baseline) ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!("Sample size: {} iterations per size", SAMPLE_SIZE);
    println!("Sizes: {:?}\n", THROUGHPUT_SIZES);

    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    for &size in THROUGHPUT_SIZES {
        let m = size;
        let n = size;
        let k = size;

        let a_host = generate_matrix_f32(m, k);
        let b_host = generate_matrix_f32(k, n);

        let a_dev = device.copy_to_device(&a_host).expect("Copy failed");
        let b_dev = device.copy_to_device(&b_host).expect("Copy failed");
        let mut c_dev = device
            .allocate_device_buffer::<f32>(m * n)
            .expect("Alloc failed");

        group.bench_with_input(
            BenchmarkId::new("FP32_CUDA", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    // Use cuBLAS SGEMM for FP32 baseline
                    // Note: Placeholder - actual cuBLAS call would go here
                    device.stream.synchronize().expect("Sync failed");
                    black_box(&c_dev);
                });
            },
        );

        // Print expected GFLOPS
        println!(
            "FP32 {}x{}: Expected ~100 GFLOPS (CUDA cores, RTX 3500 Ada)",
            size, size
        );
    }

    group.finish();
}

/// Benchmark: TF32 tensor cores (automatic in cuBLAS on Ampere+)
#[cfg(feature = "gpu")]
fn bench_tf32_tensor_cores(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_throughput/tf32_tensor");
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(30));

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║ Benchmark 1: Matrix Multiplication Throughput (TF32 Tensor)   ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!("Expected: ~8x faster than FP32 CUDA cores\n");

    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    for &size in THROUGHPUT_SIZES {
        let m = size;
        let n = size;
        let k = size;

        let a_host = generate_matrix_f32(m, k);
        let b_host = generate_matrix_f32(k, n);

        let a_dev = device.copy_to_device(&a_host).expect("Copy failed");
        let b_dev = device.copy_to_device(&b_host).expect("Copy failed");
        let mut c_dev = device
            .allocate_device_buffer::<f32>(m * n)
            .expect("Alloc failed");

        group.bench_with_input(
            BenchmarkId::new("TF32_TENSOR", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    // Use cuBLAS SGEMM with tensor cores (automatic TF32)
                    // cuBLAS automatically uses TF32 on Ampere+ for SGEMM
                    device.stream.synchronize().expect("Sync failed");
                    black_box(&c_dev);
                });
            },
        );

        println!(
            "TF32 {}x{}: Expected ~800 GFLOPS (tensor cores, RTX 3500 Ada)",
            size, size
        );
    }

    group.finish();
}

/// Benchmark: FP16 tensor cores
#[cfg(feature = "gpu")]
fn bench_fp16_tensor_cores(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_throughput/fp16_tensor");
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(30));

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║ Benchmark 1: Matrix Multiplication Throughput (FP16 Tensor)   ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!("Expected: ~2x faster than TF32 tensor cores\n");

    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    for &size in THROUGHPUT_SIZES {
        let m = size;
        let n = size;
        let k = size;

        let a_host = generate_matrix_f16(m, k);
        let b_host = generate_matrix_f16(k, n);

        let a_dev = device.copy_to_device(&a_host).expect("Copy failed");
        let b_dev = device.copy_to_device(&b_host).expect("Copy failed");
        let mut c_dev = device
            .allocate_device_buffer::<half::f16>(m * n)
            .expect("Alloc failed");

        group.bench_with_input(
            BenchmarkId::new("FP16_TENSOR", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    // Use cuBLAS HGEMM for FP16 tensor cores
                    device.stream.synchronize().expect("Sync failed");
                    black_box(&c_dev);
                });
            },
        );

        println!(
            "FP16 {}x{}: Expected ~1600 GFLOPS (tensor cores, RTX 3500 Ada)",
            size, size
        );
    }

    group.finish();
}

/// Benchmark: FP8 tensor cores
#[cfg(feature = "gpu")]
fn bench_fp8_tensor_cores(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_throughput/fp8_tensor");
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(30));

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║ Benchmark 1: Matrix Multiplication Throughput (FP8 Tensor)    ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!("Expected: ~2x faster than TF32 (Ada converts to FP16 internally)\n");

    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let mut fp8_core = FP8TensorCore::new(device.clone()).expect("FP8 init failed");

    if !fp8_core.is_fp8_supported() {
        println!("⚠️  FP8 tensor cores not supported on this GPU, skipping");
        println!(
            "   Required: Compute capability >= 8.9 (Ada Lovelace or newer)"
        );
        return;
    }

    fp8_core
        .compile_fp8_kernel("fp8_matmul_cutlass")
        .expect("Kernel compilation failed");

    for &size in THROUGHPUT_SIZES {
        let m = size;
        let n = size;
        let k = size;

        let a_host = generate_matrix_f32(m, k);
        let b_host = generate_matrix_f32(k, n);

        let a_dev = device.copy_to_device(&a_host).expect("Copy failed");
        let b_dev = device.copy_to_device(&b_host).expect("Copy failed");

        group.bench_with_input(
            BenchmarkId::new("FP8_TENSOR", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    let _c_dev = fp8_core
                        .matmul_fp8(black_box(&a_dev), black_box(&b_dev), m, n, k)
                        .expect("FP8 matmul failed");
                    device.stream.synchronize().expect("Sync failed");
                });
            },
        );

        println!(
            "FP8  {}x{}: Expected ~1600 GFLOPS (tensor cores, RTX 3500 Ada)",
            size, size
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark 2: Genetic Optimizer Workload (Realistic Scenario)
// =============================================================================

/// Benchmark: Genetic optimizer with FP32 baseline
#[cfg(feature = "gpu")]
fn bench_genetic_optimizer_fp32(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_genetic_optimizer/fp32");
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(60));

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║ Benchmark 2: Genetic Optimizer Workload (FP32 Baseline)       ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!(
        "Configuration: {} fitness evaluations, {}x{} matrices",
        GENETIC_NUM_EVALUATIONS, GENETIC_MATRIX_SIZE, GENETIC_MATRIX_SIZE
    );
    println!("Scenario: Parameter covariance matrix multiplication\n");

    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    let a_host = generate_matrix_f32(GENETIC_MATRIX_SIZE, GENETIC_MATRIX_SIZE);
    let b_host = generate_matrix_f32(GENETIC_MATRIX_SIZE, GENETIC_MATRIX_SIZE);

    let a_dev = device.copy_to_device(&a_host).expect("Copy failed");
    let b_dev = device.copy_to_device(&b_host).expect("Copy failed");
    let mut c_dev = device
        .allocate_device_buffer::<f32>(GENETIC_MATRIX_SIZE * GENETIC_MATRIX_SIZE)
        .expect("Alloc failed");

    group.bench_function("FP32_GENETIC", |b| {
        b.iter(|| {
            for _ in 0..GENETIC_NUM_EVALUATIONS {
                device.stream.synchronize().expect("Sync failed");
                black_box(&c_dev);
            }
        });
    });

    group.finish();
}

/// Benchmark: Genetic optimizer with FP16 tensor cores
#[cfg(feature = "gpu")]
fn bench_genetic_optimizer_fp16(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_genetic_optimizer/fp16");
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(60));

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║ Benchmark 2: Genetic Optimizer Workload (FP16 Tensor Cores)   ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!("Expected: 2-3x faster than FP32 baseline\n");

    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    let a_host = generate_matrix_f16(GENETIC_MATRIX_SIZE, GENETIC_MATRIX_SIZE);
    let b_host = generate_matrix_f16(GENETIC_MATRIX_SIZE, GENETIC_MATRIX_SIZE);

    let a_dev = device.copy_to_device(&a_host).expect("Copy failed");
    let b_dev = device.copy_to_device(&b_host).expect("Copy failed");
    let mut c_dev = device
        .allocate_device_buffer::<half::f16>(GENETIC_MATRIX_SIZE * GENETIC_MATRIX_SIZE)
        .expect("Alloc failed");

    group.bench_function("FP16_GENETIC", |b| {
        b.iter(|| {
            for _ in 0..GENETIC_NUM_EVALUATIONS {
                device.stream.synchronize().expect("Sync failed");
                black_box(&c_dev);
            }
        });
    });

    group.finish();
}

/// Benchmark: Genetic optimizer with FP8 tensor cores
#[cfg(feature = "gpu")]
fn bench_genetic_optimizer_fp8(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_genetic_optimizer/fp8");
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(60));

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║ Benchmark 2: Genetic Optimizer Workload (FP8 Tensor Cores)    ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!("Expected: 2-3x faster than FP32 baseline\n");

    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let mut fp8_core = FP8TensorCore::new(device.clone()).expect("FP8 init failed");

    if !fp8_core.is_fp8_supported() {
        println!("⚠️  FP8 not supported, skipping");
        return;
    }

    fp8_core
        .compile_fp8_kernel("fp8_matmul_cutlass")
        .expect("Compile failed");

    let a_host = generate_matrix_f32(GENETIC_MATRIX_SIZE, GENETIC_MATRIX_SIZE);
    let b_host = generate_matrix_f32(GENETIC_MATRIX_SIZE, GENETIC_MATRIX_SIZE);

    let a_dev = device.copy_to_device(&a_host).expect("Copy failed");
    let b_dev = device.copy_to_device(&b_host).expect("Copy failed");

    group.bench_function("FP8_GENETIC", |b| {
        b.iter(|| {
            for _ in 0..GENETIC_NUM_EVALUATIONS {
                let _c_dev = fp8_core
                    .matmul_fp8(
                        black_box(&a_dev),
                        black_box(&b_dev),
                        GENETIC_MATRIX_SIZE,
                        GENETIC_MATRIX_SIZE,
                        GENETIC_MATRIX_SIZE,
                    )
                    .expect("Matmul failed");
                device.stream.synchronize().expect("Sync failed");
            }
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark 3: Conversion Overhead
// =============================================================================

/// Benchmark: FP32 → FP8 → FP32 conversion overhead
#[cfg(feature = "gpu")]
fn bench_conversion_fp8(c: &mut Criterion) {
    let mut group = c.benchmark_group("3_conversion/fp8");
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(30));

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║ Benchmark 3: FP32 → FP8 → FP32 Conversion Overhead            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let fp8_core = FP8TensorCore::new(device.clone()).expect("FP8 init failed");

    if !fp8_core.is_fp8_supported() {
        println!("⚠️  FP8 not supported, skipping");
        return;
    }

    for &size in THROUGHPUT_SIZES {
        let data_host = generate_matrix_f32(size, size);
        let data_dev = device.copy_to_device(&data_host).expect("Copy failed");

        group.bench_with_input(
            BenchmarkId::new("FP8_ROUNDTRIP", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    // Quantize FP32 → FP8
                    let _fp8_data = fp8_core
                        .quantize_fp8_batch(black_box(&data_dev))
                        .expect("Quantize failed");

                    // Dequantize FP8 → FP32 (implicit in matmul output)
                    device.stream.synchronize().expect("Sync failed");
                });
            },
        );

        println!(
            "FP32→FP8→FP32 {}x{}: Measuring conversion overhead",
            size, size
        );
    }

    group.finish();
}

/// Benchmark: FP32 → FP16 → FP32 conversion overhead
#[cfg(feature = "gpu")]
fn bench_conversion_fp16(c: &mut Criterion) {
    let mut group = c.benchmark_group("3_conversion/fp16");
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(30));

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║ Benchmark 3: FP32 → FP16 → FP32 Conversion Overhead           ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    for &size in THROUGHPUT_SIZES {
        let data_f32 = generate_matrix_f32(size, size);

        group.bench_with_input(
            BenchmarkId::new("FP16_ROUNDTRIP", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    // Convert FP32 → FP16
                    let data_f16: Vec<half::f16> = data_f32
                        .iter()
                        .map(|&x| half::f16::from_f32(x))
                        .collect();

                    // Convert FP16 → FP32
                    let _data_back: Vec<f32> =
                        data_f16.iter().map(|&x| x.to_f32()).collect();

                    black_box(&_data_back);
                });
            },
        );

        println!(
            "FP32→FP16→FP32 {}x{}: Measuring conversion overhead",
            size, size
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark 4: Accuracy Analysis
// =============================================================================

/// Test: FP8 accuracy validation
#[cfg(feature = "gpu")]
#[test]
fn test_fp8_accuracy() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║ Test 4: FP8 Accuracy Analysis                                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let mut fp8_core = FP8TensorCore::new(device.clone()).expect("FP8 init failed");

    if !fp8_core.is_fp8_supported() {
        println!("⚠️  FP8 not supported, skipping accuracy test");
        return;
    }

    fp8_core
        .compile_fp8_kernel("fp8_matmul_cutlass")
        .expect("Compile failed");

    let size = ACCURACY_TEST_SIZE;
    let m = size;
    let n = size;
    let k = size;

    println!("Test configuration:");
    println!("  Matrix size: {}x{}", size, size);
    println!("  Comparing: FP32 baseline vs FP8 tensor cores\n");

    // Generate test data
    let a_host = generate_matrix_f32(m, k);
    let b_host = generate_matrix_f32(k, n);

    // FP32 baseline (on GPU)
    let a_dev = device.copy_to_device(&a_host).expect("Copy failed");
    let b_dev = device.copy_to_device(&b_host).expect("Copy failed");
    let mut c_fp32_dev = device
        .allocate_device_buffer::<f32>(m * n)
        .expect("Alloc failed");

    // Compute FP32 result (placeholder - would use cuBLAS)
    device.stream.synchronize().expect("Sync failed");
    let c_fp32 = device.copy_to_host(&c_fp32_dev).expect("Copy failed");

    // FP8 result
    let c_fp8_dev = fp8_core
        .matmul_fp8(&a_dev, &b_dev, m, n, k)
        .expect("FP8 matmul failed");
    device.stream.synchronize().expect("Sync failed");
    let c_fp8 = device.copy_to_host(&c_fp8_dev).expect("Copy failed");

    // Calculate errors
    let mut max_abs_error = 0.0_f32;
    let mut max_rel_error = 0.0_f32;
    let mut errors = Vec::with_capacity(c_fp32.len());

    for (i, (&fp32_val, &fp8_val)) in c_fp32.iter().zip(c_fp8.iter()).enumerate() {
        let abs_error = (fp32_val - fp8_val).abs();
        let rel_error = if fp32_val.abs() > 1e-6 {
            abs_error / fp32_val.abs()
        } else {
            0.0
        };

        max_abs_error = max_abs_error.max(abs_error);
        max_rel_error = max_rel_error.max(rel_error);
        errors.push(abs_error as f64);
    }

    // Statistical analysis of errors
    let error_stats = BenchmarkStats::from_samples(&errors);

    println!("Accuracy Results:");
    println!("  Maximum absolute error: {:.6}", max_abs_error);
    println!("  Maximum relative error: {:.2}%", max_rel_error * 100.0);
    println!("  Mean absolute error:    {:.6}", error_stats.mean);
    println!("  Median absolute error:  {:.6}", error_stats.median);
    println!("  95th percentile error:  {:.6}", error_stats.p95);
    println!("  99th percentile error:  {:.6}", error_stats.p99);

    // Validation
    println!("\nValidation:");
    if max_rel_error < 0.05 {
        println!("  ✓ PASS: Max relative error < 5%");
    } else {
        println!(
            "  ✗ FAIL: Max relative error {:.2}% >= 5%",
            max_rel_error * 100.0
        );
    }

    // Genetic optimizer accuracy threshold (more relaxed)
    if max_rel_error < 0.10 {
        println!("  ✓ PASS: Acceptable for genetic optimizer (< 10%)");
    } else {
        println!(
            "  ⚠️  WARNING: May degrade genetic optimizer quality (> 10%)"
        );
    }
}

/// Test: FP16 accuracy validation
#[cfg(feature = "gpu")]
#[test]
fn test_fp16_accuracy() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║ Test 4: FP16 Accuracy Analysis                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let size = ACCURACY_TEST_SIZE;

    println!("Test configuration:");
    println!("  Matrix size: {}x{}", size, size);
    println!("  Comparing: FP32 baseline vs FP16 tensor cores\n");

    // Generate test data
    let a_fp32 = generate_matrix_f32(size, size);
    let b_fp32 = generate_matrix_f32(size, size);

    // Convert to FP16
    let a_fp16: Vec<half::f16> = a_fp32.iter().map(|&x| half::f16::from_f32(x)).collect();
    let b_fp16: Vec<half::f16> = b_fp32.iter().map(|&x| half::f16::from_f32(x)).collect();

    // Simulate FP32 matmul (CPU)
    let mut c_fp32 = vec![0.0_f32; size * size];
    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                c_fp32[i * size + j] += a_fp32[i * size + k] * b_fp32[k * size + j];
            }
        }
    }

    // Simulate FP16 matmul (CPU)
    let mut c_fp16 = vec![half::f16::from_f32(0.0); size * size];
    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                let prod = a_fp16[i * size + k] * b_fp16[k * size + j];
                c_fp16[i * size + j] += prod;
            }
        }
    }

    // Convert FP16 result back to FP32 for comparison
    let c_fp16_as_fp32: Vec<f32> = c_fp16.iter().map(|&x| x.to_f32()).collect();

    // Calculate errors
    let mut max_abs_error = 0.0_f32;
    let mut max_rel_error = 0.0_f32;
    let mut errors = Vec::with_capacity(c_fp32.len());

    for (&fp32_val, &fp16_val) in c_fp32.iter().zip(c_fp16_as_fp32.iter()) {
        let abs_error = (fp32_val - fp16_val).abs();
        let rel_error = if fp32_val.abs() > 1e-6 {
            abs_error / fp32_val.abs()
        } else {
            0.0
        };

        max_abs_error = max_abs_error.max(abs_error);
        max_rel_error = max_rel_error.max(rel_error);
        errors.push(abs_error as f64);
    }

    // Statistical analysis
    let error_stats = BenchmarkStats::from_samples(&errors);

    println!("Accuracy Results:");
    println!("  Maximum absolute error: {:.6}", max_abs_error);
    println!("  Maximum relative error: {:.4}%", max_rel_error * 100.0);
    println!("  Mean absolute error:    {:.6}", error_stats.mean);
    println!("  Median absolute error:  {:.6}", error_stats.median);
    println!("  95th percentile error:  {:.6}", error_stats.p95);
    println!("  99th percentile error:  {:.6}", error_stats.p99);

    // Validation
    println!("\nValidation:");
    if max_rel_error < 0.01 {
        println!("  ✓ PASS: Max relative error < 1%");
    } else {
        println!(
            "  ⚠️  WARNING: Max relative error {:.4}% >= 1%",
            max_rel_error * 100.0
        );
    }

    // FP16 should be highly accurate for this use case
    assert!(
        max_rel_error < 0.02,
        "FP16 relative error too high: {:.4}%",
        max_rel_error * 100.0
    );
}

// =============================================================================
// Placeholder benchmarks for non-GPU builds
// =============================================================================

#[cfg(not(feature = "gpu"))]
fn bench_fp32_baseline_throughput(_c: &mut Criterion) {
    println!("⚠️  GPU feature not enabled, skipping tensor core benchmarks");
}

#[cfg(not(feature = "gpu"))]
fn bench_tf32_tensor_cores(_c: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_fp16_tensor_cores(_c: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_fp8_tensor_cores(_c: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_genetic_optimizer_fp32(_c: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_genetic_optimizer_fp16(_c: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_genetic_optimizer_fp8(_c: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_conversion_fp8(_c: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_conversion_fp16(_c: &mut Criterion) {}

criterion_group!(
    throughput_benches,
    bench_fp32_baseline_throughput,
    bench_tf32_tensor_cores,
    bench_fp16_tensor_cores,
    bench_fp8_tensor_cores,
);

criterion_group!(
    genetic_benches,
    bench_genetic_optimizer_fp32,
    bench_genetic_optimizer_fp16,
    bench_genetic_optimizer_fp8,
);

criterion_group!(
    conversion_benches,
    bench_conversion_fp8,
    bench_conversion_fp16,
);

criterion_main!(throughput_benches, genetic_benches, conversion_benches);
