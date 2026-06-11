"""
kimsfinance.reporting: Professional Backtest Report Generation
================================================================

Generate comprehensive PDF and HTML backtest reports with charts, metrics,
and analysis, similar to QuantConnect/LEAN reports.

Features:
    - Professional PDF reports with multi-page layouts
    - Interactive HTML reports with charts
    - Comprehensive performance metrics
    - Risk analysis and trade statistics
    - Customizable templates and branding

Quick Start:
    >>> from kimsfinance.reporting import BacktestReport
    >>>
    >>> # Generate PDF report
    >>> report = BacktestReport(backtest_results, strategy_params)
    >>> report.generate('backtest_report.pdf')
    >>>
    >>> # Generate HTML report
    >>> report.generate_html('backtest_report.html')

Report Sections:
    1. Cover Page - Strategy summary and key metrics
    2. Executive Summary - Performance overview and statistics
    3. Equity Curve - Returns and drawdown visualization
    4. Performance Analysis - Risk-adjusted metrics and distributions
    5. Trade Analysis - Win/loss statistics and patterns
    6. Risk Analysis - VaR, CVaR, and risk metrics
    7. Parameter Details - Strategy configuration
    8. Appendix - Full trade list and monthly statistics
"""

from __future__ import annotations

from .pdf_report import BacktestReport, ReportConfig
from .html_report import HTMLReport
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

__all__ = [
    # Main report classes
    "BacktestReport",
    "HTMLReport",
    "ReportConfig",
    # Metrics calculations
    "calculate_performance_metrics",
    "calculate_trade_statistics",
    "calculate_risk_metrics",
    "calculate_monthly_returns",
    # Chart generation
    "create_equity_curve",
    "create_drawdown_chart",
    "create_returns_distribution",
    "create_monthly_heatmap",
    "create_rolling_sharpe",
]
