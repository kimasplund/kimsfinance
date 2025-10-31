#!/usr/bin/env -S cargo bench --bench
//! Performance Regression Test Suite
//!
//! Validates that indicator performance remains within acceptable bounds.
//! Compares actual performance against baseline thresholds from baselines.json.
//!
//! Exit codes:
//! - 0: All tests pass (within 10% tolerance)
//! - 1: One or more tests fail (>10% regression)
//! - 2: Configuration error

use kimsfinance_core::gpu::device::GpuDevice;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

// ============================================================================
// Baseline Configuration Structures
// ============================================================================

#[derive(Debug, Deserialize)]
struct BaselineConfig {
    version: String,
    hardware: HardwareConfig,
    test_config: TestConfig,
    baselines: Baselines,
    metadata: Metadata,
}

#[derive(Debug, Deserialize)]
struct HardwareConfig {
    gpu: String,
    cpu: String,
    cuda: String,
    compute_capability: String,
}

#[derive(Debug, Deserialize)]
struct TestConfig {
    candles: usize,
    warmup_runs: usize,
    measurement_runs: usize,
    build: String,
}

#[derive(Debug, Deserialize)]
struct Baselines {
    simple_indicators: IndicatorGroup,
    medium_indicators: IndicatorGroup,
    complex_indicators: IndicatorGroup,
    known_issues: IndicatorGroup,
}

#[derive(Debug, Deserialize)]
struct IndicatorGroup {
    description: String,
    indicators: HashMap<String, IndicatorBaseline>,
}

#[derive(Debug, Deserialize)]
struct IndicatorBaseline {
    baseline_us: u64,
    tolerance_percent: u8,
    warn_percent: u8,
    notes: String,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    created_by: String,
    date: String,
    after_optimizations: Vec<String>,
}

// ============================================================================
// Test Results
// ============================================================================

#[derive(Debug)]
struct TestResult {
    name: String,
    baseline_us: u64,
    measured_us: u64,
    diff_percent: f64,
    status: TestStatus,
}

#[derive(Debug, PartialEq)]
enum TestStatus {
    Pass,
    Warn,
    Fail,
    Improvement,
}

impl TestResult {
    fn new(
        name: &str,
        baseline: &IndicatorBaseline,
        measured_us: u64,
    ) -> Self {
        let baseline_us = baseline.baseline_us;
        let diff_percent = ((measured_us as f64 - baseline_us as f64) / baseline_us as f64) * 100.0;

        let status = if measured_us < baseline_us {
            TestStatus::Improvement
        } else if diff_percent > baseline.tolerance_percent as f64 {
            TestStatus::Fail
        } else if diff_percent > baseline.warn_percent as f64 {
            TestStatus::Warn
        } else {
            TestStatus::Pass
        };

        TestResult {
            name: name.to_string(),
            baseline_us,
            measured_us,
            diff_percent,
            status,
        }
    }
}

// ============================================================================
// Indicator Benchmark Functions
// ============================================================================

fn benchmark_indicator<F>(
    name: &str,
    warmup_runs: usize,
    measurement_runs: usize,
    mut indicator_fn: F,
) -> u64
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    // Warmup
    for _ in 0..warmup_runs {
        indicator_fn().unwrap_or_else(|e| eprintln!("Warmup error for {}: {}", name, e));
    }

    // Measurement
    let start = Instant::now();
    for _ in 0..measurement_runs {
        indicator_fn().unwrap_or_else(|e| eprintln!("Measurement error for {}: {}", name, e));
    }
    let total_us = start.elapsed().as_micros() as u64;

    total_us / measurement_runs as u64
}

// ============================================================================
// Main Test Runner
// ============================================================================

fn main() {
    println!("\n{:=^100}", " PERFORMANCE REGRESSION TEST SUITE ");

    // Load baseline configuration
    let config_path = "benches/baselines.json";
    let config_str = fs::read_to_string(config_path)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: Failed to read {}: {}", config_path, e);
            std::process::exit(2);
        });

    let config: BaselineConfig = serde_json::from_str(&config_str)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: Failed to parse {}: {}", config_path, e);
            std::process::exit(2);
        });

    println!("\nConfiguration:");
    println!("  Version: {}", config.version);
    println!("  Hardware: {} ({})", config.hardware.gpu, config.hardware.cpu);
    println!("  CUDA: {} (Compute {})", config.hardware.cuda, config.hardware.compute_capability);
    println!("  Test: {} candles, {} warmup, {} measurement runs",
             config.test_config.candles,
             config.test_config.warmup_runs,
             config.test_config.measurement_runs);

    // Initialize GPU
    let device = GpuDevice::new().unwrap_or_else(|e| {
        eprintln!("ERROR: Failed to initialize GPU: {}", e);
        std::process::exit(2);
    });

    // Generate test data
    let n = config.test_config.candles;
    let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64 * 0.01)).collect());
    let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64 * 0.01)).collect());
    let close = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64 * 0.01)).collect());
    let volume = Array1::from_vec((0..n).map(|i| 1000000.0 + (i as f64 * 100.0)).collect());

    let warmup = config.test_config.warmup_runs;
    let measurement = config.test_config.measurement_runs;

    let mut results = Vec::new();

    println!("\n{:-^100}", " RUNNING TESTS ");
    println!("{:<25} {:>12} {:>12} {:>12} {:>15}",
             "Indicator", "Baseline", "Measured", "Diff %", "Status");
    println!("{:-^100}", "");

    // ========================================================================
    // GROUP 1: SIMPLE INDICATORS
    // ========================================================================

    println!("\n{}", "GROUP 1: SIMPLE INDICATORS");

    for (name, baseline) in &config.baselines.simple_indicators.indicators {
        let measured_us = match name.as_str() {
            "ema_hybrid" => {
                use kimsfinance_core::gpu::ema::ema_hybrid;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = ema_hybrid(&device, &close, 14, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            "roc" => {
                use kimsfinance_core::gpu::roc::roc_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = roc_gpu(&device, &close, 12, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            "sma" => {
                use kimsfinance_core::gpu::sma::sma_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = sma_gpu(&device, &close, 14, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            "wma" => {
                use kimsfinance_core::gpu::wma::wma_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = wma_gpu(&device, &close, 14, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            "vwma" => {
                use kimsfinance_core::gpu::vwma::vwma_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = vwma_gpu(&device, &close, &volume, 14, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            _ => {
                eprintln!("WARNING: Unknown indicator: {}", name);
                continue;
            }
        };

        let result = TestResult::new(name, baseline, measured_us);
        print_result(&result);
        results.push(result);
    }

    // ========================================================================
    // GROUP 2: MEDIUM INDICATORS
    // ========================================================================

    println!("\n{}", "GROUP 2: MEDIUM INDICATORS");

    for (name, baseline) in &config.baselines.medium_indicators.indicators {
        let measured_us = match name.as_str() {
            "williams_r" => {
                use kimsfinance_core::gpu::williams_r::williams_r_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = williams_r_gpu(&device, &high, &low, &close, 14, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            "cci" => {
                use kimsfinance_core::gpu::cci::cci_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = cci_gpu(&device, &high, &low, &close, 20, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            "donchian" => {
                use kimsfinance_core::gpu::donchian::donchian_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = donchian_gpu(&device, &high, &low, 20, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            "stochastic" => {
                use kimsfinance_core::gpu::stochastic::stochastic_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = stochastic_gpu(&device, &high, &low, &close, 14, 3, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            "elder_ray" => {
                use kimsfinance_core::gpu::elder_ray::elder_ray_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = elder_ray_gpu(&device, &high, &low, &close, 13, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            "cmf" => {
                use kimsfinance_core::gpu::cmf::cmf_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = cmf_gpu(&device, &high, &low, &close, &volume, 20, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            _ => {
                eprintln!("WARNING: Unknown indicator: {}", name);
                continue;
            }
        };

        let result = TestResult::new(name, baseline, measured_us);
        print_result(&result);
        results.push(result);
    }

    // ========================================================================
    // GROUP 3: COMPLEX INDICATORS
    // ========================================================================

    println!("\n{}", "GROUP 3: COMPLEX INDICATORS");

    for (name, baseline) in &config.baselines.complex_indicators.indicators {
        let measured_us = match name.as_str() {
            "atr" => {
                use kimsfinance_core::gpu::atr::atr_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = atr_gpu(&device, &high, &low, &close, 14, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            "rsi" => {
                use kimsfinance_core::gpu::rsi::rsi_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = rsi_gpu(&device, &close, 14, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            "rsi_sync" => {
                use kimsfinance_core::gpu::rsi_sync::rsi_gpu_sync;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = rsi_gpu_sync(&device, &close, 14, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            _ => {
                eprintln!("WARNING: Unknown indicator: {}", name);
                continue;
            }
        };

        let result = TestResult::new(name, baseline, measured_us);
        print_result(&result);
        results.push(result);
    }

    // ========================================================================
    // GROUP 4: KNOWN ISSUES
    // ========================================================================

    println!("\n{}", "GROUP 4: KNOWN ISSUES (tracked but not failed)");

    for (name, baseline) in &config.baselines.known_issues.indicators {
        let measured_us = match name.as_str() {
            "macd_cpu" => {
                use kimsfinance_core::cpu::macd_cpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = macd_cpu(&close, 12, 26, 9)?;
                    Ok(())
                })
            },
            "obv" => {
                use kimsfinance_core::gpu::obv::obv_gpu;
                benchmark_indicator(name, warmup, measurement, || {
                    let _ = obv_gpu(&device, &close, &volume, None)?;
                    device.synchronize()?;
                    Ok(())
                })
            },
            _ => {
                eprintln!("WARNING: Unknown indicator: {}", name);
                continue;
            }
        };

        let mut result = TestResult::new(name, baseline, measured_us);
        // Don't fail on known issues
        if result.status == TestStatus::Fail {
            result.status = TestStatus::Warn;
        }
        print_result(&result);
        results.push(result);
    }

    // ========================================================================
    // SUMMARY
    // ========================================================================

    println!("\n{:=^100}", " SUMMARY ");

    let pass_count = results.iter().filter(|r| r.status == TestStatus::Pass).count();
    let warn_count = results.iter().filter(|r| r.status == TestStatus::Warn).count();
    let fail_count = results.iter().filter(|r| r.status == TestStatus::Fail).count();
    let improve_count = results.iter().filter(|r| r.status == TestStatus::Improvement).count();

    println!("\nTest Results:");
    println!("  ✅ Pass: {} / {}", pass_count, results.len());
    println!("  ⚠️  Warn: {}", warn_count);
    println!("  ❌ Fail: {}", fail_count);
    println!("  ⚡ Improvements: {}", improve_count);

    if fail_count > 0 {
        println!("\n❌ PERFORMANCE REGRESSION DETECTED");
        println!("\nFailed Tests:");
        for result in results.iter().filter(|r| r.status == TestStatus::Fail) {
            println!("  - {}: {:.1}% slower than baseline ({} μs -> {} μs)",
                     result.name, result.diff_percent, result.baseline_us, result.measured_us);
        }
        std::process::exit(1);
    }

    if warn_count > 0 {
        println!("\n⚠️  WARNINGS DETECTED");
        println!("\nWarning Tests:");
        for result in results.iter().filter(|r| r.status == TestStatus::Warn) {
            println!("  - {}: {:.1}% slower than baseline ({} μs -> {} μs)",
                     result.name, result.diff_percent, result.baseline_us, result.measured_us);
        }
    }

    if improve_count > 0 {
        println!("\n⚡ PERFORMANCE IMPROVEMENTS DETECTED");
        println!("\nImproved Tests:");
        for result in results.iter().filter(|r| r.status == TestStatus::Improvement) {
            println!("  - {}: {:.1}% faster than baseline ({} μs -> {} μs)",
                     result.name, -result.diff_percent, result.baseline_us, result.measured_us);
        }
    }

    println!("\n✅ ALL TESTS PASSED");
    println!("{:=^100}\n", "");
}

fn print_result(result: &TestResult) {
    let status_str = match result.status {
        TestStatus::Pass => "✅ PASS",
        TestStatus::Warn => "⚠️  WARN",
        TestStatus::Fail => "❌ FAIL",
        TestStatus::Improvement => "⚡ FASTER",
    };

    println!("{:<25} {:>12} {:>12} {:>11.1}% {:>15}",
             result.name,
             result.baseline_us,
             result.measured_us,
             result.diff_percent,
             status_str);
}
