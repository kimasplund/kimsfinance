#!/usr/bin/env python3
"""
Statistical Analysis Script for Optimizer Comparison Benchmarks

Analyzes criterion benchmark results and generates:
- Summary statistics (mean, median, std, percentiles)
- 95% confidence intervals
- Statistical significance tests (t-test, Mann-Whitney U)
- Effect size calculations (Cohen's d)
- Comparison tables
- Convergence plots

Usage:
    python analyze_optimizer_benchmarks.py

Requirements:
    pip install pandas numpy scipy matplotlib seaborn
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Configuration
CRITERION_DIR = Path("../target/criterion/optimizer_comparison")
OUTPUT_DIR = Path("../docs/benchmark_results")
ALPHA = 0.05  # Significance level (95% confidence)

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Data Loading
# ============================================================================


def load_criterion_results(benchmark_name: str) -> pd.DataFrame:
    """
    Load criterion benchmark results from JSON files.

    Args:
        benchmark_name: Name of benchmark (e.g., "grid_search/2d")

    Returns:
        DataFrame with columns: [time_ms, optimizer, scenario]
    """
    estimates_file = CRITERION_DIR / benchmark_name / "base" / "estimates.json"

    if not estimates_file.exists():
        print(f"Warning: {estimates_file} not found, skipping")
        return pd.DataFrame()

    with open(estimates_file, "r") as f:
        data = json.load(f)

    # Extract mean time in milliseconds
    mean_time_ns = data.get("mean", {}).get("point_estimate", 0)
    mean_time_ms = mean_time_ns / 1e6

    # Extract standard deviation
    std_time_ns = data.get("std_dev", {}).get("point_estimate", 0)
    std_time_ms = std_time_ns / 1e6

    return pd.DataFrame(
        {
            "time_ms": [mean_time_ms],
            "std_ms": [std_time_ms],
            "optimizer": [benchmark_name.split("/")[0]],
            "scenario": [benchmark_name.split("/")[1]],
        }
    )


def load_all_benchmarks() -> pd.DataFrame:
    """Load all optimizer comparison benchmarks."""
    benchmarks = [
        "grid_search/2d",
        "grid_search/3d",
        "euler_search/2d",
        "euler_search/3d",
        "euler_search/5d",
        "genetic/2d",
        "genetic/3d",
        "genetic/5d",
    ]

    dfs = []
    for benchmark in benchmarks:
        df = load_criterion_results(benchmark)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("Error: No benchmark data found!")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# ============================================================================
# Statistical Analysis
# ============================================================================


def calculate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for each optimizer/scenario.

    Returns:
        DataFrame with columns: [optimizer, scenario, mean, std, median, p95, p99]
    """
    summary = (
        df.groupby(["optimizer", "scenario"])
        .agg(
            {
                "time_ms": [
                    "mean",
                    "std",
                    "median",
                    lambda x: np.percentile(x, 95),
                    lambda x: np.percentile(x, 99),
                ]
            }
        )
        .reset_index()
    )

    summary.columns = ["optimizer", "scenario", "mean", "std", "median", "p95", "p99"]
    return summary


def calculate_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval using t-distribution.

    Args:
        data: Array of sample values
        confidence: Confidence level (default 0.95 for 95%)

    Returns:
        (lower_bound, upper_bound)
    """
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of mean
    margin = sem * stats.t.ppf((1 + confidence) / 2, n - 1)

    return (mean - margin, mean + margin)


def test_normality(data: np.ndarray) -> Tuple[float, bool]:
    """
    Test if data is normally distributed using Shapiro-Wilk test.

    Args:
        data: Array of sample values

    Returns:
        (p_value, is_normal)
    """
    if len(data) < 3:
        return 1.0, False  # Not enough data

    statistic, p_value = stats.shapiro(data)
    is_normal = p_value > ALPHA

    return p_value, is_normal


def test_significance(
    group1: np.ndarray, group2: np.ndarray
) -> Tuple[str, float, float]:
    """
    Test statistical significance between two groups.

    Uses t-test if both groups are normal, otherwise Mann-Whitney U.

    Args:
        group1: First group samples
        group2: Second group samples

    Returns:
        (test_name, statistic, p_value)
    """
    # Check normality
    _, normal1 = test_normality(group1)
    _, normal2 = test_normality(group2)

    if normal1 and normal2:
        # Both normal - use t-test
        statistic, p_value = stats.ttest_ind(group1, group2)
        return "t-test", statistic, p_value
    else:
        # Non-normal - use Mann-Whitney U
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
        return "Mann-Whitney U", statistic, p_value


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        group1: First group samples
        group2: Second group samples

    Returns:
        Cohen's d value
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    # Cohen's d
    d = (mean1 - mean2) / pooled_std

    return d


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)

    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


# ============================================================================
# Comparison Tables
# ============================================================================


def generate_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comparison table across all optimizers and scenarios.

    Returns:
        DataFrame with columns: [scenario, grid_time, euler_time, genetic_time,
                                  euler_speedup, genetic_speedup]
    """
    # Pivot table
    pivot = df.pivot_table(
        index="scenario", columns="optimizer", values="time_ms", aggfunc="mean"
    )

    # Calculate speedups vs Grid Search
    if "grid_search" in pivot.columns:
        pivot["euler_speedup"] = pivot["grid_search"] / pivot["euler_search"]
        pivot["genetic_speedup"] = pivot["grid_search"] / pivot["genetic"]
    else:
        print("Warning: Grid Search baseline not found, speedup not calculated")

    return pivot


def generate_significance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate pairwise significance test matrix.

    Returns:
        DataFrame with p-values for each optimizer pair in each scenario
    """
    scenarios = df["scenario"].unique()
    optimizers = df["optimizer"].unique()

    results = []

    for scenario in scenarios:
        scenario_data = df[df["scenario"] == scenario]

        for i, opt1 in enumerate(optimizers):
            for opt2 in optimizers[i + 1 :]:
                data1 = scenario_data[scenario_data["optimizer"] == opt1]["time_ms"]
                data2 = scenario_data[scenario_data["optimizer"] == opt2]["time_ms"]

                if len(data1) < 2 or len(data2) < 2:
                    continue

                test_name, statistic, p_value = test_significance(data1, data2)
                d = calculate_cohens_d(data1, data2)

                results.append(
                    {
                        "scenario": scenario,
                        "comparison": f"{opt1} vs {opt2}",
                        "test": test_name,
                        "p_value": p_value,
                        "significant": p_value < ALPHA,
                        "cohens_d": d,
                        "effect_size": interpret_cohens_d(d),
                    }
                )

    return pd.DataFrame(results)


# ============================================================================
# Visualization
# ============================================================================


def plot_execution_time_comparison(df: pd.DataFrame, output_file: Path):
    """Generate bar plot comparing execution times."""
    plt.figure(figsize=(12, 6))

    # Prepare data
    pivot = df.pivot_table(
        index="scenario", columns="optimizer", values="time_ms", aggfunc="mean"
    )

    pivot.plot(kind="bar", rot=0)
    plt.xlabel("Scenario (Parameter Dimensions)", fontsize=12)
    plt.ylabel("Execution Time (ms)", fontsize=12)
    plt.title("Optimizer Execution Time Comparison", fontsize=14, fontweight="bold")
    plt.legend(title="Optimizer", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_speedup_comparison(df: pd.DataFrame, output_file: Path):
    """Generate speedup plot vs Grid Search baseline."""
    plt.figure(figsize=(10, 6))

    # Calculate speedups
    grid_times = df[df["optimizer"] == "grid_search"].set_index("scenario")["time_ms"]

    for optimizer in ["euler_search", "genetic"]:
        opt_data = df[df["optimizer"] == optimizer].set_index("scenario")
        speedups = grid_times / opt_data["time_ms"]

        plt.plot(
            speedups.index, speedups.values, marker="o", linewidth=2, label=optimizer
        )

    plt.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="Grid (baseline)")
    plt.xlabel("Scenario (Parameter Dimensions)", fontsize=12)
    plt.ylabel("Speedup vs Grid Search", fontsize=12)
    plt.title("Optimizer Speedup Comparison", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_confidence_intervals(df: pd.DataFrame, output_file: Path):
    """Generate plot with 95% confidence intervals."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    scenarios = sorted(df["scenario"].unique())

    for i, scenario in enumerate(scenarios):
        if i >= 3:
            break  # Only plot first 3 scenarios

        ax = axes[i]
        scenario_data = df[df["scenario"] == scenario]

        optimizers = sorted(scenario_data["optimizer"].unique())
        x_pos = np.arange(len(optimizers))

        means = []
        cis = []

        for optimizer in optimizers:
            data = scenario_data[scenario_data["optimizer"] == optimizer]["time_ms"]

            if len(data) < 2:
                means.append(data.iloc[0] if len(data) == 1 else 0)
                cis.append((0, 0))
            else:
                means.append(np.mean(data))
                cis.append(calculate_confidence_interval(data))

        # Convert CIs to error bars
        lower = [ci[0] for ci in cis]
        upper = [ci[1] for ci in cis]
        yerr = [
            [means[j] - lower[j] for j in range(len(means))],
            [upper[j] - means[j] for j in range(len(means))],
        ]

        ax.bar(x_pos, means, yerr=yerr, capsize=5, alpha=0.7, edgecolor="black")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(optimizers, rotation=15, ha="right")
        ax.set_title(f"{scenario.upper()} Scenario", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        if i == 0:
            ax.set_ylabel("Execution Time (ms)", fontsize=12)

    plt.suptitle(
        "Optimizer Performance with 95% Confidence Intervals",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_file}")
    plt.close()


# ============================================================================
# Report Generation
# ============================================================================


def generate_markdown_report(
    df: pd.DataFrame, summary: pd.DataFrame, significance: pd.DataFrame
) -> str:
    """Generate markdown report with all results."""
    report = []

    report.append("# Optimizer Comparison Benchmark Results\n")
    report.append(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(
        f"**Sample Size**: {len(df)} benchmarks, {ALPHA*100:.0f}% confidence level\n"
    )
    report.append("\n---\n\n")

    # Summary Statistics
    report.append("## Summary Statistics\n\n")
    report.append(summary.to_markdown(index=False))
    report.append("\n\n")

    # Comparison Table
    report.append("## Execution Time Comparison\n\n")
    comparison = generate_comparison_table(df)
    report.append(comparison.to_markdown())
    report.append("\n\n")

    # Statistical Significance
    report.append("## Statistical Significance Tests\n\n")
    report.append(significance.to_markdown(index=False))
    report.append("\n\n")

    # Interpretation
    report.append("## Interpretation\n\n")

    for _, row in significance.iterrows():
        scenario = row["scenario"]
        comparison = row["comparison"]
        significant = "✅ Significant" if row["significant"] else "❌ Not significant"
        effect = row["effect_size"]

        report.append(
            f"- **{scenario} - {comparison}**: {significant} (p={row['p_value']:.4f}), "
            f"Effect size: {effect} (d={row['cohens_d']:.2f})\n"
        )

    report.append("\n")

    return "".join(report)


# ============================================================================
# Main Analysis Pipeline
# ============================================================================


def main():
    """Run complete benchmark analysis pipeline."""
    print("=" * 80)
    print("Optimizer Comparison Benchmark Analysis")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading benchmark results...")
    df = load_all_benchmarks()

    if df.empty:
        print("Error: No benchmark data found. Run benchmarks first:")
        print("  cargo bench --bench optimizer_comparison")
        return

    print(f"Loaded {len(df)} benchmark results")

    # Calculate summary statistics
    print("\n[2/6] Calculating summary statistics...")
    summary = calculate_summary_statistics(df)
    print(summary)

    # Generate comparison table
    print("\n[3/6] Generating comparison table...")
    comparison = generate_comparison_table(df)
    print(comparison)

    # Statistical significance tests
    print("\n[4/6] Running statistical significance tests...")
    significance = generate_significance_matrix(df)
    print(significance)

    # Generate plots
    print("\n[5/6] Generating plots...")
    plot_execution_time_comparison(df, OUTPUT_DIR / "execution_time_comparison.png")
    plot_speedup_comparison(df, OUTPUT_DIR / "speedup_comparison.png")
    plot_confidence_intervals(df, OUTPUT_DIR / "confidence_intervals.png")

    # Generate markdown report
    print("\n[6/6] Generating markdown report...")
    report = generate_markdown_report(df, summary, significance)

    report_file = OUTPUT_DIR / "analysis_report.md"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\nReport saved: {report_file}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
