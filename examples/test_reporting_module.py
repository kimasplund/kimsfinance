"""
Test script to verify reporting module functionality.

This script tests all major components of the reporting module:
- Metric calculations
- Chart generation
- PDF report generation
- HTML report generation
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test all imports work correctly."""
    print("Testing imports...")
    try:
        from kimsfinance.reporting import (
            BacktestReport,
            HTMLReport,
            ReportConfig,
            calculate_performance_metrics,
            calculate_trade_statistics,
            calculate_risk_metrics,
            calculate_monthly_returns,
            create_equity_curve,
            create_drawdown_chart,
            create_returns_distribution,
            create_monthly_heatmap,
            create_rolling_sharpe,
        )

        imported = (
            BacktestReport,
            HTMLReport,
            ReportConfig,
            calculate_performance_metrics,
            calculate_trade_statistics,
            calculate_risk_metrics,
            calculate_monthly_returns,
            create_equity_curve,
            create_drawdown_chart,
            create_returns_distribution,
            create_monthly_heatmap,
            create_rolling_sharpe,
        )
        print(f"  ✓ All imports successful ({len(imported)} symbols)")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_metrics():
    """Test metric calculations."""
    print("\nTesting metric calculations...")

    # Generate sample data
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    equity = [100000 * (1 + np.random.normal(0.0005, 0.015)) ** i for i in range(len(dates))]
    equity_curve = pd.Series(equity, index=dates)

    # Test performance metrics
    from kimsfinance.reporting import calculate_performance_metrics

    perf = calculate_performance_metrics(equity_curve)
    print(
        f"  ✓ Performance metrics: Sharpe={perf.sharpe_ratio:.2f}, MaxDD={perf.max_drawdown*100:.2f}%"
    )

    # Test trade statistics
    from kimsfinance.reporting import calculate_trade_statistics

    trades = pd.DataFrame(
        {
            "entry_time": dates[:10],
            "exit_time": dates[1:11],
            "pnl": np.random.normal(100, 500, 10),
            "direction": ["LONG"] * 10,
        }
    )
    trade_stats = calculate_trade_statistics(trades)
    print(
        f"  ✓ Trade statistics: WinRate={trade_stats.win_rate*100:.1f}%, PF={trade_stats.profit_factor:.2f}"
    )

    # Test risk metrics
    from kimsfinance.reporting import calculate_risk_metrics

    returns = equity_curve.pct_change().dropna()
    risk = calculate_risk_metrics(returns, equity_curve)
    print(
        f"  ✓ Risk metrics: VaR95={risk.value_at_risk_95*100:.2f}%, CVaR95={risk.cvar_95*100:.2f}%"
    )

    # Test monthly returns
    from kimsfinance.reporting import calculate_monthly_returns

    monthly = calculate_monthly_returns(equity_curve)
    print(f"  ✓ Monthly returns: {monthly.shape[0]} years, {monthly.shape[1]} months")

    return True


def test_charts():
    """Test chart generation."""
    print("\nTesting chart generation...")

    # Generate sample data
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    equity = [100000 * (1 + np.random.normal(0.0005, 0.015)) ** i for i in range(len(dates))]
    equity_curve = pd.Series(equity, index=dates)
    returns = equity_curve.pct_change().dropna()

    from kimsfinance.reporting import calculate_monthly_returns
    from kimsfinance.reporting.charts import (
        create_equity_curve,
        create_drawdown_chart,
        create_returns_distribution,
        create_monthly_heatmap,
        create_rolling_sharpe,
    )

    # Test all charts
    try:
        img = create_equity_curve(equity_curve, width=400, height=200, dpi=50)
        print(f"  ✓ Equity curve: {img.size}")

        img = create_drawdown_chart(equity_curve, width=400, height=200, dpi=50)
        print(f"  ✓ Drawdown chart: {img.size}")

        img = create_returns_distribution(returns, width=400, height=200, dpi=50)
        print(f"  ✓ Returns distribution: {img.size}")

        monthly = calculate_monthly_returns(equity_curve)
        img = create_monthly_heatmap(monthly, width=400, height=200, dpi=50)
        print(f"  ✓ Monthly heatmap: {img.size}")

        img = create_rolling_sharpe(returns, window=30, width=400, height=200, dpi=50)
        print(f"  ✓ Rolling Sharpe: {img.size}")

        return True
    except Exception as e:
        print(f"  ✗ Chart generation failed: {e}")
        return False


def test_reports():
    """Test report generation."""
    print("\nTesting report generation...")

    # Generate sample data
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="D")
    equity = [100000 * (1 + np.random.normal(0.0005, 0.015)) ** i for i in range(len(dates))]
    equity_curve = pd.Series(equity, index=dates)

    trades = pd.DataFrame(
        {
            "entry_time": dates[:50:5],
            "exit_time": dates[2:52:5],
            "pnl": np.random.normal(100, 500, 10),
            "direction": ["LONG"] * 10,
        }
    )

    backtest_data = {
        "equity_curve": equity_curve,
        "trades": trades,
        "strategy_params": {"test": True},
    }

    from kimsfinance.reporting import BacktestReport, HTMLReport, ReportConfig

    # Test PDF report
    try:
        config = ReportConfig(title="Test Report", chart_dpi=50)
        report = BacktestReport(backtest_data, config)
        time = report.generate("test_report.pdf")
        print(f"  ✓ PDF report generated in {time:.2f}s")
    except Exception as e:
        print(f"  ✗ PDF generation failed: {e}")
        return False

    # Test HTML report
    try:
        report = HTMLReport(backtest_data, config)
        time = report.generate("test_report.html")
        print(f"  ✓ HTML report generated in {time:.2f}s")
    except Exception as e:
        print(f"  ✗ HTML generation failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("kimsfinance Reporting Module - Test Suite")
    print("=" * 80)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Metrics", test_metrics()))
    results.append(("Charts", test_charts()))
    results.append(("Reports", test_reports()))

    print("\n" + "=" * 80)
    print("Test Results:")
    print("-" * 80)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:20s} {status}")
        all_passed = all_passed and passed

    print("=" * 80)

    if all_passed:
        print("\n✓ All tests passed!")
        print("\nGenerated files:")
        print("  - test_report.pdf")
        print("  - test_report.html")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
