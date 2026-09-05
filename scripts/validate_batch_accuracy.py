#!/usr/bin/env python3
"""
GPU Batch Backtesting Accuracy Validation Script

**Purpose**: Statistically validate that GPU batch results match CPU sequential
            within acceptable tolerance (<0.01% difference).

**Methodology**:
1. Generate 100 random strategy configurations
2. Run each strategy on both CPU (sequential) and GPU (batch)
3. Compare results using statistical tests:
   - Mean difference (target: <0.01%)
   - Max difference (target: <0.1%)
   - Pearson correlation (target: >0.9999)
   - Paired t-test (p-value > 0.05 = means equal)

**Usage**:
    python scripts/validate_batch_accuracy.py

**Requirements**:
    pip install numpy scipy pandas matplotlib

**Output**:
    - Console: Statistical test results
    - File: validation_report.txt
    - Plots: accuracy_validation_*.png
"""

import sys
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# TODO: Import after Rust/Python bindings are ready
# from kimsfinance_core import batch_backtest, BacktestEngine


class BacktestResult:
    """Container for backtest metrics"""

    def __init__(
        self,
        sharpe_ratio: float,
        max_drawdown: float,
        total_return: float,
        win_rate: float,
        num_trades: int,
    ):
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.total_return = total_return
        self.win_rate = win_rate
        self.num_trades = num_trades

    def to_dict(self) -> Dict[str, float]:
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "win_rate": self.win_rate,
            "num_trades": float(self.num_trades),
        }


def generate_test_data(n_candles: int = 10000, seed: int = 12345) -> Tuple:
    """Generate synthetic OHLCV data for testing"""
    np.random.seed(seed)

    timestamps = np.arange(n_candles) * 60  # 1-minute bars
    close = np.zeros(n_candles)
    close[0] = 100.0

    # Random walk with drift
    for i in range(1, n_candles):
        change = np.random.randn() * 0.02  # 2% volatility
        close[i] = close[i - 1] * (1 + change)

    # Generate OHLV from close
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    high = np.maximum(open_, close) * (1 + np.abs(np.random.randn(n_candles)) * 0.01)
    low = np.minimum(open_, close) * (1 - np.abs(np.random.randn(n_candles)) * 0.01)
    volume = np.random.uniform(1000, 10000, n_candles)

    return timestamps, open_, high, low, close, volume


def generate_random_strategies(n_strategies: int, seed: int = 67890) -> List[Dict]:
    """Generate random RSI strategy parameters"""
    np.random.seed(seed)

    strategies = []
    for _ in range(n_strategies):
        strategies.append(
            {
                "type": "rsi_crossover",
                "rsi_period": np.random.randint(10, 21),
                "buy_threshold": np.random.uniform(20, 40),
                "sell_threshold": np.random.uniform(60, 80),
            }
        )

    return strategies


def run_cpu_sequential_backtests(data: Tuple, strategies: List[Dict]) -> List[BacktestResult]:
    """Run backtests sequentially on CPU (baseline)"""
    print(f"Running {len(strategies)} CPU sequential backtests...")

    # TODO: Implement after BacktestEngine is ready
    # timestamps, open_, high, low, close, volume = data
    # engine = BacktestEngine(use_gpu=False)
    # results = []
    # for strategy in strategies:
    #     result = engine.run(strategy, timestamps, open_, high, low, close, volume)
    #     results.append(result)
    # return results

    # PLACEHOLDER: Generate fake results for now
    results = []
    for _ in strategies:
        results.append(
            BacktestResult(
                sharpe_ratio=np.random.uniform(0.5, 2.0),
                max_drawdown=np.random.uniform(0.1, 0.3),
                total_return=np.random.uniform(-0.1, 0.5),
                win_rate=np.random.uniform(0.4, 0.6),
                num_trades=np.random.randint(50, 200),
            )
        )
    return results


def run_gpu_batch_backtests(data: Tuple, strategies: List[Dict]) -> List[BacktestResult]:
    """Run backtests in batch on GPU"""
    print(f"Running {len(strategies)} GPU batch backtests...")

    # TODO: Implement after GPU batch backtest is ready
    # timestamps, open_, high, low, close, volume = data
    # results = batch_backtest(
    #     strategy_type="rsi_crossover",
    #     parameters=[s for s in strategies],
    #     timestamps=timestamps,
    #     open=open_,
    #     high=high,
    #     low=low,
    #     close=close,
    #     volume=volume,
    # )
    # return results

    # PLACEHOLDER: Generate fake results (slightly different from CPU)
    results = []
    for _ in strategies:
        results.append(
            BacktestResult(
                sharpe_ratio=np.random.uniform(0.5, 2.0),
                max_drawdown=np.random.uniform(0.1, 0.3),
                total_return=np.random.uniform(-0.1, 0.5),
                win_rate=np.random.uniform(0.4, 0.6),
                num_trades=np.random.randint(50, 200),
            )
        )
    return results


def calculate_differences(
    cpu_results: List[BacktestResult], gpu_results: List[BacktestResult], metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate absolute and relative differences for a metric"""
    cpu_values = np.array([getattr(r, metric) for r in cpu_results])
    gpu_values = np.array([getattr(r, metric) for r in gpu_results])

    abs_diff = np.abs(cpu_values - gpu_values)
    rel_diff = abs_diff / (np.abs(cpu_values) + 1e-10)

    return cpu_values, gpu_values, rel_diff


def validate_accuracy(
    cpu_results: List[BacktestResult], gpu_results: List[BacktestResult], tolerance: float = 0.0001
) -> Dict[str, Dict]:
    """
    Validate GPU batch results match CPU sequential within tolerance.

    Args:
        cpu_results: List of CPU backtest results
        gpu_results: List of GPU batch backtest results
        tolerance: Maximum allowed relative difference (default: 0.01%)

    Returns:
        Dictionary with validation results per metric
    """
    assert len(cpu_results) == len(gpu_results), "Result count mismatch"

    metrics = ["sharpe_ratio", "max_drawdown", "total_return", "win_rate", "num_trades"]
    validation_results = {}

    for metric in metrics:
        cpu_values, gpu_values, rel_diff = calculate_differences(cpu_results, gpu_results, metric)

        # Statistical tests
        correlation = np.corrcoef(cpu_values, gpu_values)[0, 1]
        t_stat, p_value = stats.ttest_rel(cpu_values, gpu_values)

        # Summary statistics
        mean_diff = rel_diff.mean()
        max_diff = rel_diff.max()
        std_diff = rel_diff.std()

        # Pass/fail criteria
        pass_mean = mean_diff < tolerance
        pass_max = max_diff < tolerance * 10  # Allow 10x tolerance for max
        pass_corr = correlation > 0.9999
        pass_pvalue = p_value > 0.05  # Means are equal

        validation_results[metric] = {
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "std_diff": std_diff,
            "correlation": correlation,
            "t_stat": t_stat,
            "p_value": p_value,
            "pass_mean": pass_mean,
            "pass_max": pass_max,
            "pass_corr": pass_corr,
            "pass_pvalue": pass_pvalue,
            "overall_pass": all([pass_mean, pass_max, pass_corr, pass_pvalue]),
        }

    return validation_results


def print_validation_report(validation_results: Dict[str, Dict], tolerance: float):
    """Print formatted validation report to console"""
    print("\n" + "=" * 80)
    print("GPU BATCH BACKTESTING ACCURACY VALIDATION REPORT")
    print("=" * 80)

    for metric, results in validation_results.items():
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  Mean difference:     {results['mean_diff']:.6f} (target: <{tolerance})")
        print(f"  Max difference:      {results['max_diff']:.6f} (target: <{tolerance*10})")
        print(f"  Std deviation:       {results['std_diff']:.6f}")
        print(f"  Correlation:         {results['correlation']:.6f} (target: >0.9999)")
        print("  Paired t-test:")
        print(f"    t-statistic:       {results['t_stat']:.4f}")
        print(f"    p-value:           {results['p_value']:.6f} (target: >0.05)")
        print(f"  Status:              {'✅ PASS' if results['overall_pass'] else '❌ FAIL'}")

    # Overall validation
    all_pass = all(r["overall_pass"] for r in validation_results.values())
    print("\n" + "=" * 80)
    print(f"OVERALL VALIDATION: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
    print("=" * 80 + "\n")

    return all_pass


def plot_accuracy_validation(
    cpu_results: List[BacktestResult],
    gpu_results: List[BacktestResult],
    output_dir: Path = Path("benchmarks/figures"),
):
    """Generate accuracy validation plots"""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["sharpe_ratio", "max_drawdown", "total_return", "win_rate"]

    for metric in metrics:
        cpu_values, gpu_values, _ = calculate_differences(cpu_results, gpu_results, metric)

        # Scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: CPU vs GPU scatter
        ax1.scatter(cpu_values, gpu_values, alpha=0.5, s=20)
        ax1.plot(
            [cpu_values.min(), cpu_values.max()],
            [cpu_values.min(), cpu_values.max()],
            "r--",
            label="Perfect agreement",
        )
        ax1.set_xlabel(f'CPU {metric.replace("_", " ").title()}')
        ax1.set_ylabel(f'GPU {metric.replace("_", " ").title()}')
        ax1.set_title(f'{metric.replace("_", " ").title()}: CPU vs GPU')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Difference distribution
        diff = gpu_values - cpu_values
        ax2.hist(diff, bins=30, alpha=0.7, edgecolor="black")
        ax2.axvline(0, color="r", linestyle="--", label="Zero difference")
        ax2.set_xlabel("Difference (GPU - CPU)")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f'{metric.replace("_", " ").title()}: Difference Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f"accuracy_validation_{metric}.png"
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot: {output_path}")
        plt.close()


def save_validation_report(
    validation_results: Dict[str, Dict],
    tolerance: float,
    output_path: Path = Path("benchmarks/validation_report.txt"),
):
    """Save validation report to file"""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GPU BATCH BACKTESTING ACCURACY VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Tolerance: {tolerance:.6f} ({tolerance*100:.4f}%)\n")
        f.write("Target correlation: >0.9999\n")
        f.write("Target p-value: >0.05\n\n")

        for metric, results in validation_results.items():
            f.write(f"{metric.upper().replace('_', ' ')}:\n")
            f.write(f"  Mean difference:     {results['mean_diff']:.6f}\n")
            f.write(f"  Max difference:      {results['max_diff']:.6f}\n")
            f.write(f"  Std deviation:       {results['std_diff']:.6f}\n")
            f.write(f"  Correlation:         {results['correlation']:.6f}\n")
            f.write(f"  t-statistic:         {results['t_stat']:.4f}\n")
            f.write(f"  p-value:             {results['p_value']:.6f}\n")
            f.write(f"  Status:              {'PASS' if results['overall_pass'] else 'FAIL'}\n")
            f.write("\n")

        all_pass = all(r["overall_pass"] for r in validation_results.values())
        f.write("=" * 80 + "\n")
        f.write(f"OVERALL VALIDATION: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}\n")
        f.write("=" * 80 + "\n")

    print(f"Saved report: {output_path}")


def main():
    """Main validation workflow"""
    print("GPU Batch Backtesting Accuracy Validation")
    print("=" * 80)

    # Configuration
    n_strategies = 100  # Number of strategies to test
    n_candles = 10000  # Dataset size
    tolerance = 0.0001  # 0.01% relative difference

    print("Configuration:")
    print(f"  Strategies: {n_strategies}")
    print(f"  Candles: {n_candles}")
    print(f"  Tolerance: {tolerance:.6f} ({tolerance*100:.4f}%)")
    print()

    # Step 1: Generate test data
    print("Step 1: Generating test data...")
    data = generate_test_data(n_candles=n_candles)
    strategies = generate_random_strategies(n_strategies=n_strategies)
    print(f"  Generated {n_strategies} random strategies")
    print()

    # Step 2: Run CPU sequential backtests
    print("Step 2: Running CPU sequential backtests...")
    cpu_results = run_cpu_sequential_backtests(data, strategies)
    print(f"  Completed {len(cpu_results)} CPU backtests")
    print()

    # Step 3: Run GPU batch backtests
    print("Step 3: Running GPU batch backtests...")
    gpu_results = run_gpu_batch_backtests(data, strategies)
    print(f"  Completed {len(gpu_results)} GPU backtests")
    print()

    # Step 4: Validate accuracy
    print("Step 4: Validating accuracy...")
    validation_results = validate_accuracy(cpu_results, gpu_results, tolerance)

    # Step 5: Print results
    all_pass = print_validation_report(validation_results, tolerance)

    # Step 6: Generate plots
    print("Step 5: Generating validation plots...")
    plot_accuracy_validation(cpu_results, gpu_results)

    # Step 7: Save report
    print("Step 6: Saving validation report...")
    save_validation_report(validation_results, tolerance)

    # Exit code
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
