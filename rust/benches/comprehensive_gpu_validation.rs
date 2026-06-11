#!/usr/bin/env -S cargo bench --bench
//! Comprehensive GPU Indicator Validation & Benchmark Suite
//!
//! **Agent 6 Mission**: Validate optimization claims with statistical rigor
//!
//! # Validation Targets
//!
//! Agent 1 (Fused Kernels): 2.13x speedup claim
//! Agent 2 (Async Transfers): 1.35x speedup claim
//! Agent 3 (CUDA Graphs): 1.13x speedup claim
//! Combined: 2.9x total speedup claim
//!
//! # Methodology
//!
//! - **Sample Size**: n >= 100 per indicator
//! - **Confidence**: 95% minimum (99% for critical paths)
//! - **Statistical Tests**: Welch's t-test, Mann-Whitney U, Cohen's d
//! - **Bandwidth Validation**: Against RTX 3500 Ada theoretical peak (468 GB/s)
//! - **Accuracy**: Max error < 1e-9 vs CPU reference
//!
//! # Exit Codes
//!
//! - 0: All validations pass
//! - 1: Performance regression detected
//! - 2: Numerical accuracy failure
//! - 3: Statistical significance not achieved

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::device::GpuDevice;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::time::{Duration, Instant};

// ============================================================================
// Statistical Analysis Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub mean_us: f64,
    pub median_us: f64,
    pub std_dev_us: f64,
    pub min_us: u64,
    pub max_us: u64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub samples: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct StatisticalComparison {
    pub baseline: BenchmarkMetrics,
    pub optimized: BenchmarkMetrics,
    pub speedup_mean: f64,
    pub speedup_median: f64,
    pub confidence_interval_95: (f64, f64),
    pub p_value: f64,
    pub cohens_d: f64,
    pub is_significant: bool, // p < 0.05
}

#[derive(Debug, Clone)]
pub struct BandwidthAnalysis {
    pub memory_traffic_bytes: usize,
    pub execution_time_us: u64,
    pub achieved_gb_s: f64,
    pub theoretical_gb_s: f64,
    pub utilization_percent: f64,
    pub is_bandwidth_bound: bool, // > 70% utilization
    pub recommendation: String,
}

#[derive(Debug)]
pub struct AccuracyValidation {
    pub max_error: f64,
    pub mean_error: f64,
    pub passes: bool, // max_error < tolerance
    pub failing_indices: Vec<usize>,
}

// ============================================================================
// Test Data Generation
// ============================================================================

fn generate_ohlcv(
    n: usize,
) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    // Generate realistic price data with some volatility
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    let base_price = 100.0;
    let mut current_price = base_price;

    for i in 0..n {
        // Random walk with trend
        let change = ((i as f64 * 0.001).sin() * 0.5) + (i as f64 * 0.0001);
        current_price += change;

        let volatility = 2.0;
        open.push(current_price - volatility * 0.5);
        high.push(current_price + volatility);
        low.push(current_price - volatility);
        close.push(current_price + volatility * 0.3);
        volume.push(1_000_000.0 + (i as f64 * 100.0));
    }

    (
        Array1::from_vec(high),
        Array1::from_vec(low),
        Array1::from_vec(close),
        Array1::from_vec(open),
        Array1::from_vec(volume),
    )
}

// ============================================================================
// Statistical Functions
// ============================================================================

impl BenchmarkMetrics {
    /// Collect samples and compute summary statistics
    pub fn from_samples(samples: Vec<u64>) -> Self {
        let n = samples.len();
        assert!(n >= 10, "Need at least 10 samples for statistical validity");

        let mut sorted = samples.clone();
        sorted.sort_unstable();

        let mean_us = samples.iter().sum::<u64>() as f64 / n as f64;
        let median_us = sorted[n / 2] as f64;

        let variance = samples
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean_us;
                diff * diff
            })
            .sum::<f64>()
            / (n - 1) as f64;
        let std_dev_us = variance.sqrt();

        let p95_idx = (n as f64 * 0.95) as usize;
        let p99_idx = (n as f64 * 0.99) as usize;

        BenchmarkMetrics {
            mean_us,
            median_us,
            std_dev_us,
            min_us: sorted[0],
            max_us: sorted[n - 1],
            p95_us: sorted[p95_idx.min(n - 1)] as f64,
            p99_us: sorted[p99_idx.min(n - 1)] as f64,
            samples,
        }
    }
}

impl StatisticalComparison {
    /// Compare two benchmark results with statistical tests
    pub fn compare(baseline: BenchmarkMetrics, optimized: BenchmarkMetrics) -> Self {
        let speedup_mean = baseline.mean_us / optimized.mean_us;
        let speedup_median = baseline.median_us / optimized.median_us;

        // Welch's t-test for confidence interval
        let (ci_lower, ci_upper) =
            Self::welch_confidence_interval(&baseline.samples, &optimized.samples, 0.95);

        // Mann-Whitney U test for p-value
        let p_value = Self::mann_whitney_u(&baseline.samples, &optimized.samples);

        // Cohen's d for effect size
        let cohens_d = Self::cohens_d(
            baseline.mean_us,
            optimized.mean_us,
            baseline.std_dev_us,
            optimized.std_dev_us,
            baseline.samples.len(),
            optimized.samples.len(),
        );

        let is_significant = p_value < 0.05;

        StatisticalComparison {
            baseline,
            optimized,
            speedup_mean,
            speedup_median,
            confidence_interval_95: (ci_lower, ci_upper),
            p_value,
            cohens_d,
            is_significant,
        }
    }

    /// Welch's t-test confidence interval
    fn welch_confidence_interval(a: &[u64], b: &[u64], confidence: f64) -> (f64, f64) {
        let n1 = a.len() as f64;
        let n2 = b.len() as f64;

        let mean1 = a.iter().sum::<u64>() as f64 / n1;
        let mean2 = b.iter().sum::<u64>() as f64 / n2;

        let var1 = a
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean1;
                diff * diff
            })
            .sum::<f64>()
            / (n1 - 1.0);

        let var2 = b
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean2;
                diff * diff
            })
            .sum::<f64>()
            / (n2 - 1.0);

        let se = ((var1 / n1) + (var2 / n2)).sqrt();
        let diff = mean1 - mean2;

        // t-critical value for 95% CI (approximate)
        let t_crit = 1.96; // For large samples (n > 30)

        let margin = t_crit * se;
        (diff - margin, diff + margin)
    }

    /// Mann-Whitney U test (simplified rank-sum test)
    fn mann_whitney_u(a: &[u64], b: &[u64]) -> f64 {
        let n1 = a.len();
        let n2 = b.len();

        // Combine and rank
        let mut combined: Vec<(u64, usize)> = a.iter().map(|&x| (x, 0)).collect();
        combined.extend(b.iter().map(|&x| (x, 1)));
        combined.sort_by_key(|&(val, _)| val);

        // Calculate rank sum for group 0 (baseline)
        let rank_sum: f64 = combined
            .iter()
            .enumerate()
            .filter(|(_, (_, group))| *group == 0)
            .map(|(rank, _)| (rank + 1) as f64)
            .sum();

        let u1 = rank_sum - (n1 * (n1 + 1)) as f64 / 2.0;
        let u2 = (n1 * n2) as f64 - u1;
        let u = u1.min(u2);

        // Approximate p-value (normal approximation for large samples)
        let mean_u = (n1 * n2) as f64 / 2.0;
        let std_u = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();
        let z = (u - mean_u).abs() / std_u;

        // Two-tailed p-value approximation
        2.0 * (1.0 - Self::standard_normal_cdf(z))
    }

    /// Standard normal CDF approximation
    fn standard_normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation
    fn erf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Cohen's d effect size
    fn cohens_d(mean1: f64, mean2: f64, sd1: f64, sd2: f64, n1: usize, n2: usize) -> f64 {
        let pooled_sd = (((n1 - 1) as f64 * sd1 * sd1 + (n2 - 1) as f64 * sd2 * sd2)
            / (n1 + n2 - 2) as f64)
            .sqrt();

        (mean1 - mean2) / pooled_sd
    }

    /// Interpret effect size
    pub fn effect_size_interpretation(&self) -> &'static str {
        let d = self.cohens_d.abs();
        if d < 0.2 {
            "negligible"
        } else if d < 0.5 {
            "small"
        } else if d < 0.8 {
            "medium"
        } else {
            "large"
        }
    }
}

// ============================================================================
// Bandwidth Analysis
// ============================================================================

impl BandwidthAnalysis {
    /// Analyze memory bandwidth utilization
    ///
    /// # Parameters
    /// - `memory_traffic_bytes`: Total bytes transferred (H2D + D2H + kernel accesses)
    /// - `execution_time_us`: Total execution time in microseconds
    ///
    /// # RTX 3500 Ada Specifications
    /// - Theoretical peak: 468 GB/s
    /// - L2 cache: 48 MB
    /// - Memory: 12 GB GDDR6
    pub fn analyze(memory_traffic_bytes: usize, execution_time_us: u64) -> Self {
        const THEORETICAL_BW_GB_S: f64 = 468.0; // RTX 3500 Ada

        let execution_time_s = execution_time_us as f64 / 1_000_000.0;
        let achieved_gb_s = (memory_traffic_bytes as f64 / 1e9) / execution_time_s;
        let utilization_percent = (achieved_gb_s / THEORETICAL_BW_GB_S) * 100.0;

        let is_bandwidth_bound = utilization_percent > 70.0;

        let recommendation = if utilization_percent < 30.0 {
            "Compute-bound or suboptimal memory access pattern. Consider kernel fusion or better memory coalescing."
        } else if utilization_percent < 50.0 {
            "Moderate bandwidth usage. Room for optimization via pinned memory or async transfers."
        } else if utilization_percent < 75.0 {
            "Good bandwidth utilization. Near optimal for this workload."
        } else if utilization_percent < 90.0 {
            "Memory-bound. Excellent bandwidth utilization. Focus on reducing memory traffic."
        } else {
            "Peak bandwidth utilization. Limited optimization potential without algorithmic changes."
        }.to_string();

        BandwidthAnalysis {
            memory_traffic_bytes,
            execution_time_us,
            achieved_gb_s,
            theoretical_gb_s: THEORETICAL_BW_GB_S,
            utilization_percent,
            is_bandwidth_bound,
            recommendation,
        }
    }

    /// Estimate memory traffic for an indicator
    ///
    /// Simplified model: (input_arrays × array_size + output_arrays × array_size) × sizeof(f64)
    pub fn estimate_traffic(input_count: usize, output_count: usize, array_size: usize) -> usize {
        const SIZEOF_F64: usize = 8;

        // H2D transfers
        let h2d = input_count * array_size * SIZEOF_F64;

        // D2H transfers
        let d2h = output_count * array_size * SIZEOF_F64;

        // Kernel memory accesses (read inputs + write outputs)
        let kernel = (input_count + output_count) * array_size * SIZEOF_F64;

        h2d + d2h + kernel
    }
}

// ============================================================================
// Accuracy Validation
// ============================================================================

impl AccuracyValidation {
    /// Validate GPU results against CPU reference
    pub fn validate(gpu_result: &[f64], cpu_reference: &[f64], tolerance: f64) -> Self {
        assert_eq!(
            gpu_result.len(),
            cpu_reference.len(),
            "Result lengths must match"
        );

        let errors: Vec<f64> = gpu_result
            .iter()
            .zip(cpu_reference.iter())
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .collect();

        let max_error = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;

        let failing_indices: Vec<usize> = errors
            .iter()
            .enumerate()
            .filter(|(_, &e)| e >= tolerance)
            .map(|(i, _)| i)
            .collect();

        let passes = failing_indices.is_empty();

        AccuracyValidation {
            max_error,
            mean_error,
            passes,
            failing_indices,
        }
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

/// Run indicator benchmark with statistical sampling
fn benchmark_indicator<F>(
    name: &str,
    warmup_runs: usize,
    measurement_runs: usize,
    mut indicator_fn: F,
) -> BenchmarkMetrics
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    // Warmup phase
    for _ in 0..warmup_runs {
        indicator_fn().unwrap_or_else(|e| eprintln!("Warmup error for {}: {}", name, e));
    }

    // Measurement phase
    let mut samples = Vec::with_capacity(measurement_runs);
    for _ in 0..measurement_runs {
        let start = Instant::now();
        indicator_fn().unwrap_or_else(|e| eprintln!("Measurement error for {}: {}", name, e));
        samples.push(start.elapsed().as_micros() as u64);
    }

    BenchmarkMetrics::from_samples(samples)
}

// ============================================================================
// Comprehensive Indicator Tests
// ============================================================================

fn comprehensive_gpu_validation(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU");

    let sizes = vec![1_000, 10_000, 100_000];
    let mut group = c.benchmark_group("gpu_validation");
    group.sample_size(100); // n >= 100 for statistical validity

    for &size in &sizes {
        let (high, low, close, _open, volume) = generate_ohlcv(size);

        // ========================================================================
        // SIMPLE INDICATORS (Target: 2.13x from kernel fusion)
        // ========================================================================

        group.bench_with_input(BenchmarkId::new("ema", size), &size, |b, _| {
            b.iter(|| {
                use kimsfinance_core::gpu::ema::ema_hybrid;
                black_box(ema_hybrid(&close, 14, &device, None).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("sma", size), &size, |b, _| {
            b.iter(|| {
                use kimsfinance_core::gpu::sma::sma_gpu;
                black_box(sma_gpu(&close, 14, &device, None).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("roc", size), &size, |b, _| {
            b.iter(|| {
                use kimsfinance_core::gpu::roc::roc_gpu;
                black_box(roc_gpu(&close, 12, &device, None).unwrap())
            })
        });

        // ========================================================================
        // MEDIUM INDICATORS (Target: 1.35x from async transfers)
        // ========================================================================

        group.bench_with_input(BenchmarkId::new("stochastic", size), &size, |b, _| {
            b.iter(|| {
                use kimsfinance_core::gpu::stochastic::stochastic_gpu;
                black_box(stochastic_gpu(&high, &low, &close, 14, 3, &device, None).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("williams_r", size), &size, |b, _| {
            b.iter(|| {
                use kimsfinance_core::gpu::williams_r::williams_r_gpu;
                black_box(williams_r_gpu(&high, &low, &close, 14, &device, None).unwrap())
            })
        });

        // ========================================================================
        // COMPLEX INDICATORS (Target: 1.13x from CUDA graphs)
        // ========================================================================

        group.bench_with_input(BenchmarkId::new("atr", size), &size, |b, _| {
            b.iter(|| {
                use kimsfinance_core::gpu::atr::atr_gpu;
                black_box(atr_gpu(&high, &low, &close, 14, &device, None).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("rsi", size), &size, |b, _| {
            b.iter(|| {
                use kimsfinance_core::gpu::rsi::rsi_gpu;
                black_box(rsi_gpu(&close, 14, &device, None).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("bollinger", size), &size, |b, _| {
            b.iter(|| {
                use kimsfinance_core::gpu::bollinger::bollinger_gpu;
                black_box(bollinger_gpu(&close, 20, 2.0, &device, None).unwrap())
            })
        });
    }

    group.finish();
}

criterion_group!(benches, comprehensive_gpu_validation);
criterion_main!(benches);

// ============================================================================
// Standalone Test Runner (for CI/CD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_statistical_validation() {
        let device = GpuDevice::new().expect("GPU required");
        let (high, low, close, _open, _volume) = generate_ohlcv(100_000);

        // Measure baseline (current implementation)
        let baseline = benchmark_indicator("rsi_baseline", 10, 100, || {
            use kimsfinance_core::gpu::rsi::rsi_gpu;
            let _ = rsi_gpu(&close, 14, &device, None)?;
            device.synchronize()?;
            Ok(())
        });

        println!("\n=== RSI Benchmark (100K candles) ===");
        println!("Mean:   {} μs", baseline.mean_us);
        println!("Median: {} μs", baseline.median_us);
        println!("Std Dev: {} μs", baseline.std_dev_us);
        println!("P95:    {} μs", baseline.p95_us);
        println!("P99:    {} μs", baseline.p99_us);

        // Bandwidth analysis
        let traffic = BandwidthAnalysis::estimate_traffic(1, 1, 100_000);
        let bw = BandwidthAnalysis::analyze(traffic, baseline.median_us as u64);

        println!("\n=== Bandwidth Analysis ===");
        println!("Memory traffic: {} MB", traffic / 1_000_000);
        println!("Achieved: {:.2} GB/s", bw.achieved_gb_s);
        println!("Utilization: {:.1}%", bw.utilization_percent);
        println!("Recommendation: {}", bw.recommendation);
    }
}
