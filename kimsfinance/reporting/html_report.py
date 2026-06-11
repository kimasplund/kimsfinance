"""
HTML backtest report generation.

Interactive HTML reports with:
- Embedded charts (base64 PNG)
- Responsive design
- Printable layout
- Copy-to-clipboard for metrics
"""

from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .metrics import (
    calculate_performance_metrics,
    calculate_trade_statistics,
    calculate_risk_metrics,
    calculate_monthly_returns,
)
from .charts import (
    create_equity_curve,
    create_drawdown_chart,
    create_returns_distribution,
    create_monthly_heatmap,
    create_rolling_sharpe,
)
from .pdf_report import ReportConfig


class HTMLReport:
    """
    Interactive HTML backtest report generator.

    Example:
        >>> from kimsfinance.reporting import HTMLReport
        >>>
        >>> # Generate HTML report
        >>> report = HTMLReport(backtest_data)
        >>> report.generate('my_backtest.html')
        Report generated in 2.1 seconds: my_backtest.html
    """

    def __init__(
        self,
        backtest_data: Dict[str, Any],
        config: Optional[ReportConfig] = None,
        benchmark: Optional[pd.Series] = None,
    ):
        """
        Initialize HTML report.

        Args:
            backtest_data: Dictionary with:
                - equity_curve: pd.Series of portfolio equity
                - trades: pd.DataFrame with trade history
                - strategy_params: Dict of strategy parameters
            config: Optional ReportConfig for customization
            benchmark: Optional benchmark equity series
        """
        self.data = backtest_data
        self.config = config or ReportConfig()
        self.benchmark = benchmark

        # Extract data
        self.equity_curve = backtest_data["equity_curve"]
        self.trades = backtest_data.get("trades", pd.DataFrame())
        self.strategy_params = backtest_data.get("strategy_params", {})
        self.start_date = backtest_data.get("start_date", self.equity_curve.index[0])
        self.end_date = backtest_data.get("end_date", self.equity_curve.index[-1])
        self.initial_capital = backtest_data.get("initial_capital", self.equity_curve.iloc[0])

        # Calculate metrics
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Pre-calculate all metrics for the report."""
        self.perf_metrics = calculate_performance_metrics(
            self.equity_curve, self.benchmark, self.config.risk_free_rate
        )
        self.trade_stats = calculate_trade_statistics(self.trades)

        returns = self.equity_curve.pct_change().dropna()
        benchmark_returns = self.benchmark.pct_change().dropna() if self.benchmark is not None else None
        self.risk_metrics = calculate_risk_metrics(returns, self.equity_curve, benchmark_returns)

        self.monthly_returns = calculate_monthly_returns(self.equity_curve)

    def generate(self, output_path: str | Path) -> float:
        """
        Generate HTML report.

        Args:
            output_path: Path to output HTML file

        Returns:
            Generation time in seconds
        """
        start_time = time.time()

        # Build HTML content
        html = self._build_html()

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        elapsed = time.time() - start_time
        print(f"HTML report generated in {elapsed:.1f} seconds: {output_path}")
        return elapsed

    def _build_html(self) -> str:
        """Build complete HTML document."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title} - {self.config.strategy_name}</title>
    {self._get_css()}
</head>
<body>
    <div class="container">
        {self._build_header()}
        {self._build_summary()}
        {self._build_charts_section()}
        {self._build_performance_section()}
        {self._build_trade_analysis()}
        {self._build_risk_analysis()}
        {self._build_trade_table()}
    </div>
    {self._get_javascript()}
</body>
</html>
"""
        return html

    def _get_css(self) -> str:
        """Get CSS styles for the report."""
        return """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }

        .header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 3px solid #2E86AB;
        }

        .header h1 {
            color: #2E86AB;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header h2 {
            color: #666;
            font-size: 1.5em;
            font-weight: 300;
        }

        .header .date-range {
            color: #999;
            margin-top: 10px;
        }

        .section {
            margin: 40px 0;
            padding: 30px;
            background: #fafafa;
            border-radius: 8px;
        }

        .section h2 {
            color: #2E86AB;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #2E86AB;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .metric-card .label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .metric-card .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #2E86AB;
        }

        .metric-card .value.positive {
            color: #2A9D8F;
        }

        .metric-card .value.negative {
            color: #D62828;
        }

        .chart-container {
            margin: 30px 0;
            text-align: center;
        }

        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        th {
            background: #2E86AB;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }

        td {
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }

        tr:hover {
            background: #f9f9f9;
        }

        .positive {
            color: #2A9D8F;
        }

        .negative {
            color: #D62828;
        }

        @media print {
            .container {
                box-shadow: none;
            }

            .section {
                page-break-inside: avoid;
            }
        }

        @media (max-width: 768px) {
            .metrics-grid {
                grid-template-columns: 1fr;
            }

            .header h1 {
                font-size: 1.8em;
            }
        }
    </style>
"""

    def _get_javascript(self) -> str:
        """Get JavaScript for interactivity."""
        return """
    <script>
        // Add copy-to-clipboard functionality
        document.querySelectorAll('.metric-card').forEach(card => {
            card.style.cursor = 'pointer';
            card.addEventListener('click', () => {
                const label = card.querySelector('.label').textContent;
                const value = card.querySelector('.value').textContent;
                navigator.clipboard.writeText(`${label}: ${value}`);

                // Visual feedback
                const original = card.style.background;
                card.style.background = '#e8f4f8';
                setTimeout(() => {
                    card.style.background = original;
                }, 200);
            });
        });

        // Print button
        const printBtn = document.createElement('button');
        printBtn.textContent = 'Print Report';
        printBtn.style.cssText = 'position: fixed; bottom: 20px; right: 20px; padding: 15px 30px; background: #2E86AB; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);';
        printBtn.onclick = () => window.print();
        document.body.appendChild(printBtn);
    </script>
"""

    def _build_header(self) -> str:
        """Build report header."""
        return f"""
    <div class="header">
        <h1>{self.config.title}</h1>
        <h2>{self.config.strategy_name}</h2>
        <div class="date-range">
            {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}
        </div>
    </div>
"""

    def _build_summary(self) -> str:
        """Build summary section with key metrics."""
        return_class = "positive" if self.perf_metrics.total_return > 0 else "negative"

        return f"""
    <div class="section">
        <h2>Performance Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Total Return</div>
                <div class="value {return_class}">{self.perf_metrics.total_return * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="label">Annualized Return</div>
                <div class="value {return_class}">{self.perf_metrics.annualized_return * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="label">Sharpe Ratio</div>
                <div class="value">{self.perf_metrics.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Max Drawdown</div>
                <div class="value negative">{self.perf_metrics.max_drawdown * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="label">Win Rate</div>
                <div class="value">{self.trade_stats.win_rate * 100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Trades</div>
                <div class="value">{self.trade_stats.total_trades:,}</div>
            </div>
        </div>
    </div>
"""

    def _build_charts_section(self) -> str:
        """Build charts section."""
        # Generate charts
        equity_img = create_equity_curve(self.equity_curve, self.benchmark, width=800, height=400, dpi=100)
        drawdown_img = create_drawdown_chart(self.equity_curve, width=800, height=300, dpi=100)

        # Convert to base64
        equity_b64 = self._img_to_base64(equity_img)
        drawdown_b64 = self._img_to_base64(drawdown_img)

        return f"""
    <div class="section">
        <h2>Equity Curve</h2>
        <div class="chart-container">
            <img src="data:image/png;base64,{equity_b64}" alt="Equity Curve">
        </div>
    </div>

    <div class="section">
        <h2>Drawdown Analysis</h2>
        <div class="chart-container">
            <img src="data:image/png;base64,{drawdown_b64}" alt="Drawdown">
        </div>
    </div>
"""

    def _build_performance_section(self) -> str:
        """Build performance metrics section."""
        returns = self.equity_curve.pct_change().dropna()
        dist_img = create_returns_distribution(returns, width=600, height=400, dpi=100)
        dist_b64 = self._img_to_base64(dist_img)

        return f"""
    <div class="section">
        <h2>Performance Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Annualized Volatility</td>
                <td>{self.perf_metrics.volatility_annual * 100:.2f}%</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{self.perf_metrics.sharpe_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Sortino Ratio</td>
                <td>{self.perf_metrics.sortino_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Calmar Ratio</td>
                <td>{self.perf_metrics.calmar_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Best Day</td>
                <td class="positive">{self.perf_metrics.best_day * 100:.2f}%</td>
            </tr>
            <tr>
                <td>Worst Day</td>
                <td class="negative">{self.perf_metrics.worst_day * 100:.2f}%</td>
            </tr>
            <tr>
                <td>Positive Days</td>
                <td>{self.perf_metrics.positive_days_pct:.1f}%</td>
            </tr>
        </table>

        <div class="chart-container">
            <img src="data:image/png;base64,{dist_b64}" alt="Returns Distribution">
        </div>
    </div>
"""

    def _build_trade_analysis(self) -> str:
        """Build trade analysis section."""
        return f"""
    <div class="section">
        <h2>Trade Analysis</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Trades</td>
                <td>{self.trade_stats.total_trades:,}</td>
            </tr>
            <tr>
                <td>Winning Trades</td>
                <td class="positive">{self.trade_stats.winning_trades:,}</td>
            </tr>
            <tr>
                <td>Losing Trades</td>
                <td class="negative">{self.trade_stats.losing_trades:,}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{self.trade_stats.win_rate * 100:.1f}%</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td>{self.trade_stats.profit_factor:.2f}</td>
            </tr>
            <tr>
                <td>Average Trade</td>
                <td>${self.trade_stats.avg_trade:,.2f}</td>
            </tr>
            <tr>
                <td>Average Winner</td>
                <td class="positive">${self.trade_stats.avg_winning_trade:,.2f}</td>
            </tr>
            <tr>
                <td>Average Loser</td>
                <td class="negative">${self.trade_stats.avg_losing_trade:,.2f}</td>
            </tr>
            <tr>
                <td>Largest Win</td>
                <td class="positive">${self.trade_stats.largest_win:,.2f}</td>
            </tr>
            <tr>
                <td>Largest Loss</td>
                <td class="negative">${self.trade_stats.largest_loss:,.2f}</td>
            </tr>
        </table>
    </div>
"""

    def _build_risk_analysis(self) -> str:
        """Build risk analysis section."""
        return f"""
    <div class="section">
        <h2>Risk Analysis</h2>
        <table>
            <tr>
                <th>Risk Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Value at Risk (95%)</td>
                <td>{self.risk_metrics.value_at_risk_95 * 100:.2f}%</td>
            </tr>
            <tr>
                <td>Value at Risk (99%)</td>
                <td>{self.risk_metrics.value_at_risk_99 * 100:.2f}%</td>
            </tr>
            <tr>
                <td>Conditional VaR (95%)</td>
                <td>{self.risk_metrics.cvar_95 * 100:.2f}%</td>
            </tr>
            <tr>
                <td>Conditional VaR (99%)</td>
                <td>{self.risk_metrics.cvar_99 * 100:.2f}%</td>
            </tr>
            <tr>
                <td>Downside Deviation</td>
                <td>{self.risk_metrics.downside_deviation * 100:.2f}%</td>
            </tr>
            <tr>
                <td>Ulcer Index</td>
                <td>{self.risk_metrics.ulcer_index:.2f}</td>
            </tr>
        </table>
    </div>
"""

    def _build_trade_table(self) -> str:
        """Build trade list table."""
        if len(self.trades) == 0:
            return ""

        # Limit to first 100 trades
        trades_to_show = self.trades.head(100)

        rows = []
        for i, (idx, trade) in enumerate(trades_to_show.iterrows(), 1):
            pnl = trade.get("pnl", 0)
            pnl_class = "positive" if pnl > 0 else "negative"

            entry = trade.get("entry_time", "N/A")
            exit = trade.get("exit_time", "N/A")

            entry_str = entry.strftime("%Y-%m-%d %H:%M") if hasattr(entry, "strftime") else str(entry)
            exit_str = exit.strftime("%Y-%m-%d %H:%M") if hasattr(exit, "strftime") else str(exit)

            rows.append(
                f"""
            <tr>
                <td>{i}</td>
                <td>{entry_str}</td>
                <td>{exit_str}</td>
                <td>{trade.get('direction', 'N/A')}</td>
                <td class="{pnl_class}">${pnl:,.2f}</td>
            </tr>
"""
            )

        return f"""
    <div class="section">
        <h2>Trade List</h2>
        <table>
            <tr>
                <th>#</th>
                <th>Entry Time</th>
                <th>Exit Time</th>
                <th>Direction</th>
                <th>P&L</th>
            </tr>
            {"".join(rows)}
        </table>
        {f'<p><i>Showing first 100 of {len(self.trades):,} total trades.</i></p>' if len(self.trades) > 100 else ''}
    </div>
"""

    def _img_to_base64(self, img) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()
