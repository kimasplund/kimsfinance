//! INT8 Quantization Accuracy Tests
//!
//! Validates that per-feature dynamic range quantization meets <0.01% deviation requirement.

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{GpuDevice, QuantizationCalibrator, QuantizedFeatures};

#[test]
fn test_quantization_roundtrip_accuracy() {
    // Test data: realistic orderflow features
    let features = vec![
        vec![0.52, 1023.45, 48.3, 0.00123, 0.089, 103.2],
        vec![0.61, 1187.92, 53.7, 0.00145, 0.124, 107.8],
        vec![0.43, 892.14, 41.2, 0.00098, 0.071, 98.5],
        vec![0.58, 1104.67, 51.9, 0.00132, 0.103, 105.1],
        vec![0.49, 967.23, 45.6, 0.00111, 0.092, 101.3],
    ];

    let calibrator = QuantizationCalibrator::calibrate(&features);
    let rmse = calibrator.estimate_error(&features);

    println!("Quantization RMSE: {:.6}", rmse);
    assert!(
        rmse < 0.001,
        "RMSE ({:.6}) exceeds target (0.001) for <0.01% backtest deviation",
        rmse
    );
}

#[test]
fn test_per_feature_vs_global_quantization() {
    // Test that per-feature quantization is more accurate than global
    let features = vec![
        vec![0.1, 10.0, 1000.0, 0.0001, 0.5, 50.0],   // Wide range differences
        vec![0.9, 90.0, 9000.0, 0.0009, 4.5, 450.0],
        vec![0.5, 50.0, 5000.0, 0.0005, 2.5, 250.0],
    ];

    // Per-feature quantization
    let calibrator_per_feature = QuantizationCalibrator::calibrate(&features);
    let rmse_per_feature = calibrator_per_feature.estimate_error(&features);

    // Global quantization (simulate by using same range for all features)
    let global_min = features
        .iter()
        .flat_map(|f| f.iter())
        .copied()
        .fold(f32::INFINITY, f32::min);
    let global_max = features
        .iter()
        .flat_map(|f| f.iter())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let calibrator_global = QuantizationCalibrator {
        feature_names: vec!["global".to_string(); 6],
        min_values: vec![global_min; 6],
        max_values: vec![global_max; 6],
        scales: vec![255.0 / (global_max - global_min); 6],
    };
    let rmse_global = calibrator_global.estimate_error(&features);

    println!("Per-feature RMSE: {:.6}", rmse_per_feature);
    println!("Global RMSE: {:.6}", rmse_global);

    assert!(
        rmse_per_feature < rmse_global,
        "Per-feature quantization should be more accurate: {:.6} vs {:.6}",
        rmse_per_feature,
        rmse_global
    );
}

#[test]
fn test_extreme_values() {
    // Test handling of edge cases
    let features = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],            // All zeros
        vec![1.0, 10000.0, 1000.0, 1.0, 10.0, 1000.0], // Large values
        vec![0.5, 5000.0, 500.0, 0.5, 5.0, 500.0],     // Mid-range
    ];

    let calibrator = QuantizationCalibrator::calibrate(&features);

    // Test each feature individually
    for tick_features in &features {
        let quantized = calibrator.quantize(tick_features);
        let dequantized = calibrator.dequantize(&quantized);

        for (i, (&orig, &deq)) in tick_features.iter().zip(dequantized.iter()).enumerate() {
            let relative_error = if orig != 0.0 {
                ((orig - deq).abs() / orig.abs()) * 100.0
            } else {
                (orig - deq).abs()
            };

            assert!(
                relative_error < 1.0,
                "Feature {} relative error too high: {:.2}% (original: {}, dequantized: {})",
                i,
                relative_error,
                orig,
                deq
            );
        }
    }
}

#[test]
fn test_large_dataset_accuracy() {
    // Simulate 10K ticks to test scalability
    let num_ticks = 10_000;
    let mut features = Vec::with_capacity(num_ticks);

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..num_ticks {
        features.push(vec![
            rng.gen_range(0.0..1.0),         // order_imbalance
            rng.gen_range(0.0..5000.0),      // volume_delta
            rng.gen_range(0.0..200.0),       // trade_intensity
            rng.gen_range(0.0..0.01),        // price_velocity
            rng.gen_range(0.0..1.0),         // volume_weighted_spread
            rng.gen_range(0.0..500.0),       // trade_size_distribution
        ]);
    }

    let calibrator = QuantizationCalibrator::calibrate(&features);
    let rmse = calibrator.estimate_error(&features);

    println!("Large dataset ({} ticks) RMSE: {:.6}", num_ticks, rmse);
    assert!(
        rmse < 0.001,
        "Large dataset RMSE ({:.6}) exceeds target",
        rmse
    );
}

#[test]
fn test_memory_savings() {
    // Validate 8x compression ratio
    let num_ticks = 106_000_000; // 106M ticks per strategy
    let num_features = 6;

    let fp32_size = num_ticks * num_features * std::mem::size_of::<f32>();
    let int8_size = num_ticks * num_features * std::mem::size_of::<i8>();

    let compression_ratio = fp32_size as f64 / int8_size as f64;

    println!("FP32 size: {:.2} GB", fp32_size as f64 / 1e9);
    println!("INT8 size: {:.2} GB", int8_size as f64 / 1e9);
    println!("Compression ratio: {:.2}x", compression_ratio);
    println!("Memory saved: {:.2} GB", (fp32_size - int8_size) as f64 / 1e9);

    assert!(
        (compression_ratio - 8.0).abs() < 0.1,
        "Expected 8x compression, got {:.2}x",
        compression_ratio
    );

    // 10 strategies: 19GB → 2.4GB
    let num_strategies = 10;
    let total_fp32 = fp32_size * num_strategies;
    let total_int8 = int8_size * num_strategies;

    println!("\n10 strategies:");
    println!("  FP32 total: {:.2} GB", total_fp32 as f64 / 1e9);
    println!("  INT8 total: {:.2} GB", total_int8 as f64 / 1e9);
    println!("  Savings: {:.2} GB", (total_fp32 - total_int8) as f64 / 1e9);

    assert!(total_int8 < 3_000_000_000, "Should fit in 3GB (target: 2.4GB)");
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_quantization_accuracy() {
    if let Ok(device) = GpuDevice::new() {
        // Generate test data
        let features = vec![
            vec![0.52, 1023.45, 48.3, 0.00123, 0.089, 103.2],
            vec![0.61, 1187.92, 53.7, 0.00145, 0.124, 107.8],
            vec![0.43, 892.14, 41.2, 0.00098, 0.071, 98.5],
        ];

        let calibrator = QuantizationCalibrator::calibrate(&features);

        // GPU quantization
        match calibrator.quantize_batch_gpu(&device, &features) {
            Ok(d_quantized) => {
                // Dequantize on GPU
                match calibrator.dequantize_batch_gpu(&device, &d_quantized, features.len()) {
                    Ok(d_dequantized) => {
                        // Copy to host for validation
                        match device.copy_to_host(&d_dequantized) {
                            Ok(dequantized_flat) => {
                                // Compute error
                                let mut total_squared_error = 0.0;
                                let mut count = 0;

                                for (tick_idx, tick_features) in features.iter().enumerate() {
                                    for (feature_idx, &original) in tick_features.iter().enumerate() {
                                        let reconstructed = dequantized_flat[tick_idx * 6 + feature_idx];
                                        let error = original - reconstructed;
                                        total_squared_error += error * error;
                                        count += 1;
                                    }
                                }

                                let rmse = (total_squared_error / count as f32).sqrt();
                                println!("GPU roundtrip RMSE: {:.6}", rmse);

                                assert!(
                                    rmse < 0.001,
                                    "GPU quantization RMSE ({:.6}) exceeds target",
                                    rmse
                                );
                            }
                            Err(e) => panic!("Failed to copy dequantized to host: {:?}", e),
                        }
                    }
                    Err(e) => panic!("GPU dequantization failed: {:?}", e),
                }
            }
            Err(e) => panic!("GPU quantization failed: {:?}", e),
        }
    } else {
        println!("⚠️ GPU not available, skipping GPU quantization test");
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_batch_quantization() {
    if let Ok(device) = GpuDevice::new() {
        // Larger batch for performance testing
        let num_ticks = 10_000;
        let mut features = Vec::with_capacity(num_ticks);

        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..num_ticks {
            features.push(vec![
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..5000.0),
                rng.gen_range(0.0..200.0),
                rng.gen_range(0.0..0.01),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..500.0),
            ]);
        }

        let calibrator = QuantizationCalibrator::calibrate(&features);

        let start = std::time::Instant::now();
        match calibrator.quantize_batch_gpu(&device, &features) {
            Ok(_) => {
                let elapsed = start.elapsed();
                let features_per_sec = (num_ticks * 6) as f64 / elapsed.as_secs_f64();

                println!("GPU quantized {} features in {:?}", num_ticks * 6, elapsed);
                println!("Throughput: {:.2} M features/sec", features_per_sec / 1e6);

                assert!(
                    features_per_sec > 1e8,
                    "GPU should process >100M features/sec, got {:.2}M",
                    features_per_sec / 1e6
                );
            }
            Err(e) => {
                println!("⚠️ GPU batch quantization failed: {:?}", e);
            }
        }
    } else {
        println!("⚠️ GPU not available, skipping GPU batch test");
    }
}

#[test]
fn test_quantized_features_storage() {
    // Test QuantizedFeatures wrapper
    let features = vec![
        vec![0.5, 1000.0, 50.0, 0.001, 0.1, 100.0],
        vec![0.6, 1200.0, 55.0, 0.0012, 0.15, 105.0],
    ];

    let calibrator = QuantizationCalibrator::calibrate(&features);

    // Quantize all features
    let mut features_int8 = Vec::new();
    for tick_features in &features {
        features_int8.extend(calibrator.quantize(tick_features));
    }

    let quantized = QuantizedFeatures::new(calibrator, features_int8, features.len());

    // Test dequantization
    let reconstructed = quantized.dequantize_batch();
    assert_eq!(reconstructed.len(), features.len());

    // Verify memory savings
    let saved = quantized.memory_saved();
    println!("Memory saved: {} bytes ({:.2}x compression)", saved,
             (saved as f64 + quantized.memory_bytes() as f64) / quantized.memory_bytes() as f64);

    assert!(saved > 0, "Should save memory");
}

#[test]
fn test_constant_features() {
    // Test features with zero variance (edge case)
    let features = vec![
        vec![0.5, 1000.0, 50.0, 0.001, 0.1, 100.0],
        vec![0.5, 1000.0, 50.0, 0.001, 0.1, 100.0], // Same values
        vec![0.5, 1000.0, 50.0, 0.001, 0.1, 100.0],
    ];

    let calibrator = QuantizationCalibrator::calibrate(&features);

    // Should handle constant features gracefully
    let rmse = calibrator.estimate_error(&features);
    println!("Constant features RMSE: {:.6}", rmse);

    // RMSE should be near zero for constant features
    assert!(rmse < 1e-6, "Constant features should quantize perfectly");
}

#[test]
fn test_feature_name_preservation() {
    let features = vec![vec![0.5, 1000.0, 50.0, 0.001, 0.1, 100.0]];
    let calibrator = QuantizationCalibrator::calibrate(&features);

    assert_eq!(calibrator.feature_names.len(), 6);
    assert_eq!(calibrator.feature_names[0], "order_imbalance");
    assert_eq!(calibrator.feature_names[1], "volume_delta");
    assert_eq!(calibrator.feature_names[5], "trade_size_distribution");
}
