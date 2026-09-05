"""
PDF backtest report generation using ReportLab.

Professional multi-page PDF reports with:
- Cover page with key metrics
- Executive summary
- Performance charts and statistics
- Trade analysis
- Risk metrics
- Full appendix

Target: <5 seconds for standard report generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
        PageBreak,
        Image as RLImage,
    )
    from reportlab.lib.enums import TA_CENTER

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    # Fallbacks so module-level dataclass defaults (e.g. ``ReportConfig.page_size``)
    # don't raise NameError when reportlab is absent. Values match
    # ``reportlab.lib.pagesizes`` (in points). Features that actually render a PDF
    # still require reportlab and are guarded by ``REPORTLAB_AVAILABLE``.
    letter = (612.0, 792.0)
    A4 = (595.2755905511812, 841.8897637795277)

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


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    # Branding
    title: str = "Backtest Report"
    strategy_name: str = "Trading Strategy"
    logo_path: Optional[str] = None
    company_name: str = "kimsfinance"

    # Style
    page_size: tuple = letter  # or A4
    primary_color: tuple = (46, 134, 171)  # RGB
    secondary_color: tuple = (162, 59, 114)  # RGB

    # Performance
    chart_dpi: int = 150  # DPI for embedded charts
    include_appendix: bool = True
    risk_free_rate: float = 0.02  # 2% annual

    # Optional sections
    include_monthly_heatmap: bool = True
    include_rolling_sharpe: bool = True
    include_trade_list: bool = True


class BacktestReport:
    """
    Professional PDF backtest report generator.

    Example:
        >>> from kimsfinance.reporting import BacktestReport
        >>>
        >>> # Prepare data
        >>> backtest_data = {
        ...     'equity_curve': equity_series,
        ...     'trades': trades_df,
        ...     'strategy_params': {...}
        ... }
        >>>
        >>> # Generate report
        >>> report = BacktestReport(backtest_data)
        >>> report.generate('my_backtest.pdf')
        Report generated in 3.2 seconds: my_backtest.pdf
    """

    def __init__(
        self,
        backtest_data: Dict[str, Any],
        config: Optional[ReportConfig] = None,
        benchmark: Optional[pd.Series] = None,
    ):
        """
        Initialize backtest report.

        Args:
            backtest_data: Dictionary with:
                - equity_curve: pd.Series of portfolio equity
                - trades: pd.DataFrame with trade history
                - strategy_params: Dict of strategy parameters
                - start_date: datetime
                - end_date: datetime
                - initial_capital: float
            config: Optional ReportConfig for customization
            benchmark: Optional benchmark equity series
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for PDF generation. " "Install with: pip install reportlab"
            )

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
        # Performance metrics
        self.perf_metrics = calculate_performance_metrics(
            self.equity_curve, self.benchmark, self.config.risk_free_rate
        )

        # Trade statistics
        self.trade_stats = calculate_trade_statistics(self.trades)

        # Risk metrics
        returns = self.equity_curve.pct_change().dropna()
        benchmark_returns = (
            self.benchmark.pct_change().dropna() if self.benchmark is not None else None
        )
        self.risk_metrics = calculate_risk_metrics(returns, self.equity_curve, benchmark_returns)

        # Monthly returns
        self.monthly_returns = calculate_monthly_returns(self.equity_curve)

    def generate(self, output_path: str | Path) -> float:
        """
        Generate PDF report.

        Args:
            output_path: Path to output PDF file

        Returns:
            Generation time in seconds
        """
        start_time = time.time()

        # Create document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=self.config.page_size,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        # Build story (content)
        story = []
        story.extend(self._create_cover_page())
        story.append(PageBreak())
        story.extend(self._create_executive_summary())
        story.append(PageBreak())
        story.extend(self._create_equity_curve_page())
        story.append(PageBreak())
        story.extend(self._create_performance_analysis())
        story.append(PageBreak())
        story.extend(self._create_trade_analysis())
        story.append(PageBreak())
        story.extend(self._create_risk_analysis())

        if self.config.include_appendix and len(self.trades) > 0:
            story.append(PageBreak())
            story.extend(self._create_appendix())

        # Build PDF
        doc.build(story)

        elapsed = time.time() - start_time
        print(f"Report generated in {elapsed:.1f} seconds: {output_path}")
        return elapsed

    def _create_cover_page(self) -> list:
        """Create cover page with strategy summary."""
        styles = getSampleStyleSheet()
        story = []

        # Title style
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#%02x%02x%02x" % self.config.primary_color),
            spaceAfter=30,
            alignment=TA_CENTER,
        )

        # Add logo if provided
        if self.config.logo_path:
            try:
                logo = RLImage(self.config.logo_path, width=2 * inch, height=1 * inch)
                story.append(logo)
                story.append(Spacer(1, 0.3 * inch))
            except Exception:
                pass

        # Title
        story.append(Paragraph(self.config.title, title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Strategy name
        subtitle_style = ParagraphStyle(
            "Subtitle", parent=styles["Normal"], fontSize=16, alignment=TA_CENTER
        )
        story.append(Paragraph(f"<b>{self.config.strategy_name}</b>", subtitle_style))
        story.append(Spacer(1, 0.5 * inch))

        # Date range
        date_range = (
            f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}"
        )
        story.append(Paragraph(f"<b>Period:</b> {date_range}", styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))

        # Key metrics summary table
        data = [
            ["Metric", "Value"],
            ["Total Return", f"{self.perf_metrics.total_return * 100:.2f}%"],
            ["Annualized Return", f"{self.perf_metrics.annualized_return * 100:.2f}%"],
            ["Sharpe Ratio", f"{self.perf_metrics.sharpe_ratio:.2f}"],
            ["Max Drawdown", f"{self.perf_metrics.max_drawdown * 100:.2f}%"],
            ["Total Trades", f"{self.trade_stats.total_trades:,}"],
            ["Win Rate", f"{self.trade_stats.win_rate * 100:.1f}%"],
        ]

        table = Table(data, colWidths=[3 * inch, 2 * inch])
        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#%02x%02x%02x" % self.config.primary_color),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        story.append(table)
        story.append(Spacer(1, 0.5 * inch))

        # Footer
        footer_style = ParagraphStyle(
            "Footer", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER
        )
        story.append(
            Paragraph(
                f"Generated by {self.config.company_name} on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                footer_style,
            )
        )

        return story

    def _create_executive_summary(self) -> list:
        """Create executive summary page."""
        styles = getSampleStyleSheet()
        story = []

        # Section title
        story.append(Paragraph("<b>Executive Summary</b>", styles["Heading1"]))
        story.append(Spacer(1, 0.2 * inch))

        # Performance overview
        story.append(Paragraph("<b>Performance Overview</b>", styles["Heading2"]))
        overview_text = f"""
        The strategy generated a total return of <b>{self.perf_metrics.total_return * 100:.2f}%</b>
        over the backtest period, with an annualized return of <b>{self.perf_metrics.annualized_return * 100:.2f}%</b>.
        The maximum drawdown was <b>{self.perf_metrics.max_drawdown * 100:.2f}%</b>, lasting
        {self.perf_metrics.max_drawdown_duration} days.
        """
        story.append(Paragraph(overview_text, styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

        # Statistics table
        data = [
            ["Performance Metric", "Value"],
            ["Total Return", f"{self.perf_metrics.total_return * 100:.2f}%"],
            ["Annualized Return", f"{self.perf_metrics.annualized_return * 100:.2f}%"],
            ["Annualized Volatility", f"{self.perf_metrics.volatility_annual * 100:.2f}%"],
            ["Sharpe Ratio", f"{self.perf_metrics.sharpe_ratio:.2f}"],
            ["Sortino Ratio", f"{self.perf_metrics.sortino_ratio:.2f}"],
            ["Calmar Ratio", f"{self.perf_metrics.calmar_ratio:.2f}"],
            ["Max Drawdown", f"{self.perf_metrics.max_drawdown * 100:.2f}%"],
            ["Best Day", f"{self.perf_metrics.best_day * 100:.2f}%"],
            ["Worst Day", f"{self.perf_metrics.worst_day * 100:.2f}%"],
            ["Positive Days", f"{self.perf_metrics.positive_days_pct:.1f}%"],
        ]

        table = Table(data, colWidths=[3.5 * inch, 2 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ]
            )
        )

        story.append(table)

        # Monthly heatmap if enabled
        if self.config.include_monthly_heatmap:
            story.append(Spacer(1, 0.3 * inch))
            story.append(Paragraph("<b>Monthly Returns</b>", styles["Heading2"]))
            story.append(Spacer(1, 0.1 * inch))

            # Generate chart
            heatmap_img = create_monthly_heatmap(
                self.monthly_returns, width=600, height=300, dpi=self.config.chart_dpi
            )

            # Save to temp file and embed
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                heatmap_img.save(tmp.name)
                img = RLImage(tmp.name, width=5.5 * inch, height=2.75 * inch)
                story.append(img)

        return story

    def _create_equity_curve_page(self) -> list:
        """Create equity curve visualization page."""
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>Equity Curve</b>", styles["Heading1"]))
        story.append(Spacer(1, 0.2 * inch))

        # Generate equity curve chart
        equity_img = create_equity_curve(
            self.equity_curve, self.benchmark, width=700, height=350, dpi=self.config.chart_dpi
        )

        # Embed chart
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            equity_img.save(tmp.name)
            img = RLImage(tmp.name, width=6 * inch, height=3 * inch)
            story.append(img)

        story.append(Spacer(1, 0.3 * inch))

        # Drawdown chart
        story.append(Paragraph("<b>Drawdown Analysis</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))

        drawdown_img = create_drawdown_chart(
            self.equity_curve, width=700, height=250, dpi=self.config.chart_dpi
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            drawdown_img.save(tmp.name)
            img = RLImage(tmp.name, width=6 * inch, height=2.14 * inch)
            story.append(img)

        # Rolling Sharpe if enabled
        if self.config.include_rolling_sharpe:
            story.append(Spacer(1, 0.3 * inch))
            story.append(Paragraph("<b>Rolling Sharpe Ratio</b>", styles["Heading2"]))
            story.append(Spacer(1, 0.1 * inch))

            returns = self.equity_curve.pct_change().dropna()
            sharpe_img = create_rolling_sharpe(
                returns, width=700, height=250, dpi=self.config.chart_dpi
            )

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                sharpe_img.save(tmp.name)
                img = RLImage(tmp.name, width=6 * inch, height=2.14 * inch)
                story.append(img)

        return story

    def _create_performance_analysis(self) -> list:
        """Create performance analysis page."""
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>Performance Analysis</b>", styles["Heading1"]))
        story.append(Spacer(1, 0.2 * inch))

        # Returns distribution
        story.append(Paragraph("<b>Returns Distribution</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))

        returns = self.equity_curve.pct_change().dropna()
        dist_img = create_returns_distribution(
            returns, width=600, height=350, dpi=self.config.chart_dpi
        )

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            dist_img.save(tmp.name)
            img = RLImage(tmp.name, width=5 * inch, height=2.92 * inch)
            story.append(img)

        story.append(Spacer(1, 0.3 * inch))

        # Risk-adjusted metrics
        story.append(Paragraph("<b>Risk-Adjusted Returns</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))

        text = f"""
        The strategy achieved a Sharpe ratio of <b>{self.perf_metrics.sharpe_ratio:.2f}</b> and
        a Sortino ratio of <b>{self.perf_metrics.sortino_ratio:.2f}</b>. The Calmar ratio, which
        measures return relative to maximum drawdown, is <b>{self.perf_metrics.calmar_ratio:.2f}</b>.
        """
        story.append(Paragraph(text, styles["Normal"]))

        return story

    def _create_trade_analysis(self) -> list:
        """Create trade analysis page."""
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>Trade Analysis</b>", styles["Heading1"]))
        story.append(Spacer(1, 0.2 * inch))

        # Trade statistics table
        data = [
            ["Trade Metric", "Value"],
            ["Total Trades", f"{self.trade_stats.total_trades:,}"],
            ["Winning Trades", f"{self.trade_stats.winning_trades:,}"],
            ["Losing Trades", f"{self.trade_stats.losing_trades:,}"],
            ["Win Rate", f"{self.trade_stats.win_rate * 100:.1f}%"],
            ["Profit Factor", f"{self.trade_stats.profit_factor:.2f}"],
            ["Average Trade", f"${self.trade_stats.avg_trade:,.2f}"],
            ["Average Winner", f"${self.trade_stats.avg_winning_trade:,.2f}"],
            ["Average Loser", f"${self.trade_stats.avg_losing_trade:,.2f}"],
            ["Largest Win", f"${self.trade_stats.largest_win:,.2f}"],
            ["Largest Loss", f"${self.trade_stats.largest_loss:,.2f}"],
            ["Avg Trade Duration", f"{self.trade_stats.avg_trade_duration:.1f} hours"],
            ["Max Consecutive Wins", f"{self.trade_stats.max_consecutive_wins}"],
            ["Max Consecutive Losses", f"{self.trade_stats.max_consecutive_losses}"],
        ]

        table = Table(data, colWidths=[3.5 * inch, 2 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ]
            )
        )

        story.append(table)

        return story

    def _create_risk_analysis(self) -> list:
        """Create risk analysis page."""
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>Risk Analysis</b>", styles["Heading1"]))
        story.append(Spacer(1, 0.2 * inch))

        # Risk metrics table
        data = [
            ["Risk Metric", "Value"],
            ["Value at Risk (95%)", f"{self.risk_metrics.value_at_risk_95 * 100:.2f}%"],
            ["Value at Risk (99%)", f"{self.risk_metrics.value_at_risk_99 * 100:.2f}%"],
            ["Conditional VaR (95%)", f"{self.risk_metrics.cvar_95 * 100:.2f}%"],
            ["Conditional VaR (99%)", f"{self.risk_metrics.cvar_99 * 100:.2f}%"],
            ["Downside Deviation", f"{self.risk_metrics.downside_deviation * 100:.2f}%"],
            ["Ulcer Index", f"{self.risk_metrics.ulcer_index:.2f}"],
        ]

        if self.risk_metrics.beta is not None:
            data.append(["Beta (vs Benchmark)", f"{self.risk_metrics.beta:.2f}"])
        if self.risk_metrics.alpha is not None:
            data.append(["Alpha (vs Benchmark)", f"{self.risk_metrics.alpha * 100:.2f}%"])

        table = Table(data, colWidths=[3.5 * inch, 2 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ]
            )
        )

        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

        # Interpretation
        text = """
        <b>Value at Risk (VaR)</b> represents the maximum expected loss at a given confidence level.
        <b>Conditional VaR (CVaR)</b>, also known as Expected Shortfall, represents the average loss
        in the worst-case scenarios beyond the VaR threshold.
        """
        story.append(Paragraph(text, styles["Normal"]))

        return story

    def _create_appendix(self) -> list:
        """Create appendix with full trade list."""
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>Appendix: Trade List</b>", styles["Heading1"]))
        story.append(Spacer(1, 0.2 * inch))

        if not self.config.include_trade_list or len(self.trades) == 0:
            story.append(Paragraph("No trades to display.", styles["Normal"]))
            return story

        # Limit to first 100 trades for PDF size
        trades_to_show = self.trades.head(100)

        # Create table data
        headers = ["#", "Entry", "Exit", "Direction", "P&L"]
        data = [headers]

        for i, (idx, trade) in enumerate(trades_to_show.iterrows(), 1):
            row = [
                str(i),
                (
                    trade.get("entry_time", "N/A").strftime("%Y-%m-%d %H:%M")
                    if hasattr(trade.get("entry_time", ""), "strftime")
                    else str(trade.get("entry_time", "N/A"))
                ),
                (
                    trade.get("exit_time", "N/A").strftime("%Y-%m-%d %H:%M")
                    if hasattr(trade.get("exit_time", ""), "strftime")
                    else str(trade.get("exit_time", "N/A"))
                ),
                trade.get("direction", "N/A"),
                f"${trade.get('pnl', 0):,.2f}",
            ]
            data.append(row)

        # Create table
        table = Table(data, colWidths=[0.5 * inch, 1.5 * inch, 1.5 * inch, 1 * inch, 1.5 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ]
            )
        )

        story.append(table)

        if len(self.trades) > 100:
            story.append(Spacer(1, 0.2 * inch))
            story.append(
                Paragraph(
                    f"<i>Showing first 100 of {len(self.trades):,} total trades.</i>",
                    styles["Normal"],
                )
            )

        return story
