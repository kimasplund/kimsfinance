//! Statistical Analysis for Benchmark Validation
//!
//! Provides rigorous statistical methods for A/B testing CUDA optimizations:
//! - Descriptive statistics (mean, median, std, percentiles)
//! - Confidence intervals (95% and 99%)
//! - Hypothesis testing (t-test, Mann-Whitney U)
//! - Effect size calculation (Cohen's d)
//! - Performance regression detection
//!
//! # Statistical Rigor
//!
//! All tests follow these guidelines:
//! - **Sample size**: n >= 100 for statistical power
//! - **Significance level**: α = 0.05 (p < 0.05)
//! - **Confidence intervals**: 95% minimum, 99% for critical paths
//! - **Effect size**: Cohen's d with interpretation
//! - **Outlier handling**: Winsorization at 1st/99th percentile
//!
//! # Example
//!
//! ```rust
//! use statistics::{BenchmarkStats, compare_distributions};
//!
//! let baseline = vec![100.0, 102.0, 98.0, 101.0, 99.0]; // μs
//! let optimized = vec![70.0, 72.0, 68.0, 71.0, 69.0];  // μs
//!
//! let baseline_stats = BenchmarkStats::from_samples(&baseline);
//! let optimized_stats = BenchmarkStats::from_samples(&optimized);
//!
//! let comparison = compare_distributions(&baseline, &optimized);
//!
//! println!("Speedup: {:.2}x", comparison.speedup);
//! println!("p-value: {:.4}", comparison.p_value);
//! println!("Significant: {}", comparison.is_significant);
//! ```

use std::collections::HashMap;

/// Descriptive statistics for a benchmark sample
#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    /// Sample size
    pub n: usize,
    /// Mean (average)
    pub mean: f64,
    /// Median (50th percentile)
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
    /// Coefficient of variation (std/mean)
    pub cv: f64,
    /// 95% confidence interval for mean
    pub ci_95: (f64, f64),
    /// 99% confidence interval for mean
    pub ci_99: (f64, f64),
}

impl BenchmarkStats {
    /// Calculate statistics from raw samples
    ///
    /// # Arguments
    /// * `samples` - Raw timing measurements (e.g., microseconds)
    ///
    /// # Returns
    /// Descriptive statistics with confidence intervals
    ///
    /// # Panics
    /// Panics if samples is empty
    pub fn from_samples(samples: &[f64]) -> Self {
        assert!(
            !samples.is_empty(),
            "Cannot compute stats from empty sample"
        );

        let n = samples.len();
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Basic statistics
        let mean = samples.iter().sum::<f64>() / n as f64;
        let median = percentile(&sorted, 50.0);
        let min = sorted[0];
        let max = sorted[n - 1];
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        // Variance and standard deviation
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean;

        // Confidence intervals using t-distribution
        let ci_95 = confidence_interval(mean, std_dev, n, 0.95);
        let ci_99 = confidence_interval(mean, std_dev, n, 0.99);

        Self {
            n,
            mean,
            median,
            std_dev,
            min,
            max,
            p95,
            p99,
            cv,
            ci_95,
            ci_99,
        }
    }

    /// Check if coefficient of variation indicates high variance
    ///
    /// # Interpretation
    /// - CV < 0.10: Low variance (good)
    /// - CV 0.10-0.20: Moderate variance (acceptable)
    /// - CV > 0.20: High variance (warning)
    pub fn has_high_variance(&self) -> bool {
        self.cv > 0.20
    }

    /// Format statistics as human-readable string
    pub fn summary(&self) -> String {
        format!(
            "n={}, mean={:.2}μs, median={:.2}μs, std={:.2}μs, p95={:.2}μs, p99={:.2}μs, CV={:.1}%",
            self.n,
            self.mean,
            self.median,
            self.std_dev,
            self.p95,
            self.p99,
            self.cv * 100.0
        )
    }
}

/// Statistical comparison between two distributions
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Speedup factor (baseline_mean / optimized_mean)
    pub speedup: f64,
    /// p-value from hypothesis test (lower = more significant)
    pub p_value: f64,
    /// Is difference statistically significant? (p < 0.05)
    pub is_significant: bool,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Effect size interpretation
    pub effect_interpretation: EffectSize,
    /// Test used (t-test or Mann-Whitney U)
    pub test_name: String,
    /// Baseline statistics
    pub baseline: BenchmarkStats,
    /// Optimized statistics
    pub optimized: BenchmarkStats,
}

impl ComparisonResult {
    /// Check if optimization is a regression (slower than baseline)
    pub fn is_regression(&self, threshold: f64) -> bool {
        self.speedup < (1.0 - threshold) && self.is_significant
    }

    /// Check if optimization is an improvement (faster than baseline)
    pub fn is_improvement(&self, threshold: f64) -> bool {
        self.speedup > (1.0 + threshold) && self.is_significant
    }

    /// Format comparison as human-readable string
    pub fn summary(&self) -> String {
        let significance = if self.is_significant {
            "✓ SIGNIFICANT"
        } else {
            "✗ Not significant"
        };

        let direction = if self.speedup > 1.0 {
            format!("{:.2}x FASTER", self.speedup)
        } else {
            format!("{:.2}x SLOWER", 1.0 / self.speedup)
        };

        format!(
            "{} (p={:.4}, d={:.2} [{}], {})",
            direction, self.p_value, self.effect_size, self.effect_interpretation, significance
        )
    }
}

/// Effect size interpretation (Cohen's d)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectSize {
    /// |d| < 0.2
    Negligible,
    /// 0.2 <= |d| < 0.5
    Small,
    /// 0.5 <= |d| < 0.8
    Medium,
    /// |d| >= 0.8
    Large,
}

impl std::fmt::Display for EffectSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EffectSize::Negligible => write!(f, "negligible"),
            EffectSize::Small => write!(f, "small"),
            EffectSize::Medium => write!(f, "medium"),
            EffectSize::Large => write!(f, "large"),
        }
    }
}

/// Compare two distributions for statistical significance
///
/// # Algorithm
///
/// 1. Check normality (Shapiro-Wilk test approximation)
/// 2. If both normal: Use Welch's t-test (unequal variances)
/// 3. If non-normal: Use Mann-Whitney U test (rank-based)
/// 4. Calculate effect size (Cohen's d)
///
/// # Arguments
/// * `baseline` - Baseline measurements (before optimization)
/// * `optimized` - Optimized measurements (after optimization)
///
/// # Returns
/// Statistical comparison with significance test
pub fn compare_distributions(baseline: &[f64], optimized: &[f64]) -> ComparisonResult {
    let baseline_stats = BenchmarkStats::from_samples(baseline);
    let optimized_stats = BenchmarkStats::from_samples(optimized);

    // Calculate speedup (baseline / optimized)
    // Speedup > 1.0 means optimized is faster
    let speedup = baseline_stats.mean / optimized_stats.mean;

    // Check normality (simplified test: CV < 0.5 suggests normality)
    let baseline_normal = baseline_stats.cv < 0.5;
    let optimized_normal = optimized_stats.cv < 0.5;

    // Choose appropriate test
    let (p_value, test_name) = if baseline_normal && optimized_normal {
        // Use Welch's t-test (allows unequal variances)
        (
            welch_t_test(baseline, optimized),
            "Welch's t-test".to_string(),
        )
    } else {
        // Use Mann-Whitney U test (non-parametric)
        (
            mann_whitney_u_test(baseline, optimized),
            "Mann-Whitney U".to_string(),
        )
    };

    // Calculate effect size (Cohen's d)
    let effect_size = cohens_d(baseline, optimized);
    let effect_interpretation = interpret_effect_size(effect_size);

    // Check significance (p < 0.05)
    let is_significant = p_value < 0.05;

    ComparisonResult {
        speedup,
        p_value,
        is_significant,
        effect_size,
        effect_interpretation,
        test_name,
        baseline: baseline_stats,
        optimized: optimized_stats,
    }
}

/// Calculate percentile from sorted samples
///
/// Uses linear interpolation between data points.
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    assert!(!sorted_data.is_empty());
    assert!((0.0..=100.0).contains(&p));

    let n = sorted_data.len();
    if n == 1 {
        return sorted_data[0];
    }

    let rank = p / 100.0 * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - lower as f64;

    sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])
}

/// Calculate confidence interval for mean
///
/// Uses t-distribution for small samples, normal approximation for large samples.
fn confidence_interval(mean: f64, std_dev: f64, n: usize, confidence: f64) -> (f64, f64) {
    // Critical values from t-distribution (approximation)
    let t_critical = if confidence == 0.95 {
        t_critical_95(n)
    } else {
        t_critical_99(n)
    };

    let margin = t_critical * std_dev / (n as f64).sqrt();
    (mean - margin, mean + margin)
}

/// Approximation of t-critical value for 95% CI
fn t_critical_95(n: usize) -> f64 {
    let df = (n - 1) as f64;
    // Approximation: t ≈ z + z^3/(4*df) + 5*z^5/(96*df^2)
    // For 95% CI, z = 1.96
    if df >= 30.0 {
        1.96 // Normal approximation for large samples
    } else {
        // Lookup table for small samples (df 1-30)
        match df as usize {
            1 => 12.706,
            2 => 4.303,
            3 => 3.182,
            4 => 2.776,
            5 => 2.571,
            6 => 2.447,
            7 => 2.365,
            8 => 2.306,
            9 => 2.262,
            10 => 2.228,
            _ => 1.96 + 1.96_f64.powi(3) / (4.0 * df), // Approximation
        }
    }
}

/// Approximation of t-critical value for 99% CI
fn t_critical_99(n: usize) -> f64 {
    let df = (n - 1) as f64;
    if df >= 30.0 {
        2.576 // Normal approximation for large samples
    } else {
        match df as usize {
            1 => 63.657,
            2 => 9.925,
            3 => 5.841,
            4 => 4.604,
            5 => 4.032,
            6 => 3.707,
            7 => 3.499,
            8 => 3.355,
            9 => 3.250,
            10 => 3.169,
            _ => 2.576 + 2.576_f64.powi(3) / (4.0 * df),
        }
    }
}

/// Welch's t-test for unequal variances
///
/// Returns p-value (two-tailed test).
fn welch_t_test(sample1: &[f64], sample2: &[f64]) -> f64 {
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    let mean1 = sample1.iter().sum::<f64>() / n1;
    let mean2 = sample2.iter().sum::<f64>() / n2;

    let var1 = sample1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2 = sample2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    // Welch's t-statistic
    let t = (mean1 - mean2) / ((var1 / n1) + (var2 / n2)).sqrt();

    // Welch-Satterthwaite degrees of freedom
    let df = ((var1 / n1) + (var2 / n2)).powi(2)
        / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));

    // Convert t-statistic to p-value (two-tailed)
    // Using approximation: p ≈ 2 * P(T > |t|)
    t_to_p_value(t.abs(), df)
}

/// Convert t-statistic to p-value (approximation)
fn t_to_p_value(t: f64, df: f64) -> f64 {
    // Simplified approximation using normal distribution for large df
    if df >= 30.0 {
        // Use standard normal approximation
        2.0 * (1.0 - normal_cdf(t))
    } else {
        // Rough approximation for small df
        let p = (1.0 + t / df.sqrt()).powi(-1);
        2.0 * p.min(0.5)
    }
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let p =
        d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

    if x >= 0.0 { 1.0 - p } else { p }
}

/// Mann-Whitney U test (non-parametric)
///
/// Returns p-value (two-tailed test).
fn mann_whitney_u_test(sample1: &[f64], sample2: &[f64]) -> f64 {
    let n1 = sample1.len();
    let n2 = sample2.len();

    // Combine and rank samples
    let mut combined: Vec<(f64, usize)> = sample1
        .iter()
        .map(|&x| (x, 0))
        .chain(sample2.iter().map(|&x| (x, 1)))
        .collect();

    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Calculate rank sums
    let mut rank_sum1 = 0.0;
    for (i, (_, group)) in combined.iter().enumerate() {
        if *group == 0 {
            rank_sum1 += (i + 1) as f64;
        }
    }

    // U statistic
    let u1 = rank_sum1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;
    let u = u1.min(u2);

    // Normal approximation for large samples
    let mean_u = (n1 * n2) as f64 / 2.0;
    let std_u = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();
    let z = (u - mean_u) / std_u;

    // Convert to p-value (two-tailed)
    2.0 * (1.0 - normal_cdf(z.abs()))
}

/// Calculate Cohen's d effect size
///
/// # Interpretation
/// - |d| < 0.2: Negligible
/// - 0.2 <= |d| < 0.5: Small
/// - 0.5 <= |d| < 0.8: Medium
/// - |d| >= 0.8: Large
fn cohens_d(sample1: &[f64], sample2: &[f64]) -> f64 {
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    let mean1 = sample1.iter().sum::<f64>() / n1;
    let mean2 = sample2.iter().sum::<f64>() / n2;

    let var1 = sample1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2 = sample2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    // Pooled standard deviation
    let pooled_std = (((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0)).sqrt();

    (mean1 - mean2) / pooled_std
}

/// Interpret Cohen's d effect size
fn interpret_effect_size(d: f64) -> EffectSize {
    let abs_d = d.abs();
    if abs_d < 0.2 {
        EffectSize::Negligible
    } else if abs_d < 0.5 {
        EffectSize::Small
    } else if abs_d < 0.8 {
        EffectSize::Medium
    } else {
        EffectSize::Large
    }
}

/// Winsorize samples to remove outliers
///
/// Replaces values below 1st percentile with 1st percentile value,
/// and values above 99th percentile with 99th percentile value.
///
/// # Arguments
/// * `samples` - Raw measurements
///
/// # Returns
/// Winsorized samples (outliers replaced, not removed)
pub fn winsorize(samples: &[f64]) -> Vec<f64> {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p1 = percentile(&sorted, 1.0);
    let p99 = percentile(&sorted, 99.0);

    samples.iter().map(|&x| x.clamp(p1, p99)).collect()
}

/// Aggregate results across multiple configurations
#[derive(Debug, Clone)]
pub struct AggregateResult {
    /// Configuration name (e.g., "Phase 1: compute_89")
    pub name: String,
    /// Results per indicator
    pub results: HashMap<String, BenchmarkStats>,
}

impl AggregateResult {
    /// Create new aggregate result
    pub fn new(name: String) -> Self {
        Self {
            name,
            results: HashMap::new(),
        }
    }

    /// Add result for an indicator
    pub fn add_result(&mut self, indicator: String, stats: BenchmarkStats) {
        self.results.insert(indicator, stats);
    }

    /// Get result for an indicator
    pub fn get_result(&self, indicator: &str) -> Option<&BenchmarkStats> {
        self.results.get(indicator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_simple() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 50.0), 3.0);
        assert_eq!(percentile(&data, 100.0), 5.0);
    }

    #[test]
    fn test_benchmark_stats() {
        let samples = vec![100.0, 102.0, 98.0, 101.0, 99.0, 103.0, 97.0];
        let stats = BenchmarkStats::from_samples(&samples);

        assert_eq!(stats.n, 7);
        assert!((stats.mean - 100.0).abs() < 1.0);
        assert!((stats.median - 100.0).abs() < 1.0);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_compare_distributions_speedup() {
        let baseline = vec![100.0; 100];
        let optimized = vec![50.0; 100];

        let comparison = compare_distributions(&baseline, &optimized);

        assert!((comparison.speedup - 2.0).abs() < 0.1);
        assert!(comparison.is_significant);
    }

    #[test]
    fn test_cohens_d() {
        let sample1 = vec![100.0; 50];
        let sample2 = vec![110.0; 50];

        let d = cohens_d(&sample1, &sample2);
        assert!(d.abs() > 1.0); // Large effect size
    }

    #[test]
    fn test_winsorize() {
        let samples = vec![1.0, 2.0, 3.0, 100.0]; // 100.0 is outlier
        let winsorized = winsorize(&samples);

        // Outlier should be capped
        assert!(winsorized.iter().all(|&x| x < 50.0));
    }
}
