"""
Example: Generate comprehensive backtest reports (PDF and HTML)

This example demonstrates how to create professional backtest reports
using kimsfinance's reporting module.

Features demonstrated:
- Generate synthetic backtest data
- Calculate performance metrics
- Create PDF report
- Create HTML report
- Customize report appearance

Usage:
    python examples/generate_backtest_report.py
"""

import importlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kimsfinance.reporting import BacktestReport, HTMLReport, ReportConfig


def generate_sample_backtest_data():
    """
    Generate realistic synthetic backtest data.

    Returns:
        Dictionary with equity curve, trades, and parameters
    """
    # Generate date range (1 year of daily data)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq="D")

    # Generate synthetic equity curve with realistic characteristics
    initial_capital = 100000.0
    daily_returns = np.random.normal(0.0005, 0.015, len(dates))  # 0.05% mean, 1.5% std

    # Add some trends and volatility regimes
    trend = np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 0.002
    daily_returns += trend

    # Calculate equity curve
    equity = [initial_capital]
    for ret in daily_returns[1:]:
        equity.append(equity[-1] * (1 + ret))

    equity_curve = pd.Series(equity, index=dates)

    # Generate synthetic trades
    num_trades = 150
    trade_indices = np.random.choice(len(dates) - 10, num_trades, replace=False)
    trade_dates = [dates[i] for i in sorted(trade_indices)]

    trades = []
    for i, entry_date in enumerate(trade_dates):
        # Random exit date (1-10 days later)
        exit_offset = timedelta(days=int(np.random.randint(1, 11)))
        exit_date = entry_date + exit_offset
        if exit_date > end_date:
            exit_date = end_date

        # Generate realistic P&L
        # 55% win rate with 1.5:1 reward:risk ratio
        is_winner = np.random.random() < 0.55
        if is_winner:
            pnl = np.random.exponential(1500)  # Winners
        else:
            pnl = -np.random.exponential(1000)  # Losers

        # Direction
        direction = np.random.choice(["LONG", "SHORT"])

        trades.append(
            {
                "entry_time": entry_date,
                "exit_time": exit_date,
                "direction": direction,
                "pnl": pnl,
            }
        )

    trades_df = pd.DataFrame(trades)

    # Strategy parameters
    strategy_params = {
        "strategy": "Mean Reversion",
        "entry_lookback": 20,
        "exit_threshold": 2.0,
        "stop_loss": 0.02,
        "take_profit": 0.03,
        "position_size": 0.1,  # 10% of capital
    }

    # Generate benchmark (SPY-like returns)
    benchmark_returns = np.random.normal(0.0003, 0.012, len(dates))
    benchmark = [initial_capital]
    for ret in benchmark_returns[1:]:
        benchmark.append(benchmark[-1] * (1 + ret))
    benchmark_series = pd.Series(benchmark, index=dates)

    return {
        "equity_curve": equity_curve,
        "trades": trades_df,
        "strategy_params": strategy_params,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
    }, benchmark_series


def main():
    """Generate sample backtest reports."""
    print("=" * 80)
    print("kimsfinance Backtest Report Generator")
    print("=" * 80)
    print()

    # Generate sample data
    print("Generating synthetic backtest data...")
    backtest_data, benchmark = generate_sample_backtest_data()

    print(f"  Equity curve: {len(backtest_data['equity_curve']):,} days")
    print(f"  Total trades: {len(backtest_data['trades']):,}")
    print()

    # Create custom report configuration
    config = ReportConfig(
        title="Mean Reversion Strategy Backtest",
        strategy_name="Momentum + Mean Reversion Hybrid",
        company_name="kimsfinance",
        chart_dpi=150,
        include_monthly_heatmap=True,
        include_rolling_sharpe=True,
        include_trade_list=True,
    )

    # Generate PDF report
    print("Generating PDF report...")
    pdf_report = BacktestReport(backtest_data, config, benchmark)
    pdf_time = pdf_report.generate("backtest_report.pdf")
    print(f"  PDF generated in {pdf_time:.2f}s: backtest_report.pdf")
    print()

    # Generate HTML report
    print("Generating HTML report...")
    html_report = HTMLReport(backtest_data, config, benchmark)
    html_time = html_report.generate("backtest_report.html")
    print(f"  HTML generated in {html_time:.2f}s: backtest_report.html")
    print()

    # Display summary metrics
    print("Report Summary:")
    print("-" * 80)
    perf = pdf_report.perf_metrics
    trade = pdf_report.trade_stats

    print(f"Total Return:        {perf.total_return * 100:>8.2f}%")
    print(f"Annualized Return:   {perf.annualized_return * 100:>8.2f}%")
    print(f"Sharpe Ratio:        {perf.sharpe_ratio:>8.2f}")
    print(f"Max Drawdown:        {perf.max_drawdown * 100:>8.2f}%")
    print(f"Win Rate:            {trade.win_rate * 100:>8.1f}%")
    print(f"Profit Factor:       {trade.profit_factor:>8.2f}")
    print()

    print("=" * 80)
    print("Reports generated successfully!")
    print("  PDF:  backtest_report.pdf")
    print("  HTML: backtest_report.html")
    print("=" * 80)


if __name__ == "__main__":
    # Check dependencies
    try:
        importlib.import_module("reportlab")
        importlib.import_module("matplotlib")

        main()
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print()
        print("Install required packages:")
        print("  pip install reportlab matplotlib")
        sys.exit(1)
