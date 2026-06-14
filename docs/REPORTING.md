# Backtest Report Generation

Professional PDF and HTML backtest reports for kimsfinance, similar to QuantConnect/LEAN reports.

## Overview

The `kimsfinance.reporting` module provides comprehensive backtest report generation with:

- **PDF Reports**: Multi-page professional reports with embedded charts
- **HTML Reports**: Interactive web-based reports with responsive design
- **Performance Metrics**: Comprehensive statistics (returns, Sharpe, drawdown, etc.)
- **Trade Analysis**: Win/loss statistics, profit factor, trade duration
- **Risk Metrics**: VaR, CVaR, downside deviation, Ulcer Index
- **Visualizations**: Equity curves, drawdowns, distributions, heatmaps

**Performance**: <5 seconds for standard reports (target met: 0.6s PDF, 0.3s HTML)

## Installation

Install the reporting dependencies:

```bash
pip install kimsfinance[reporting]
```

Or manually:

```bash
pip install reportlab matplotlib
```

## Quick Start

### Basic PDF Report

```python
from kimsfinance.reporting import BacktestReport
import pandas as pd

# Prepare backtest data
backtest_data = {
    'equity_curve': equity_series,  # pd.Series with datetime index
    'trades': trades_df,            # pd.DataFrame with trade history
    'strategy_params': {...}        # Dict of strategy parameters
}

# Generate report
report = BacktestReport(backtest_data)
report.generate('my_backtest.pdf')
```

### HTML Report

```python
from kimsfinance.reporting import HTMLReport

# Generate interactive HTML report
report = HTMLReport(backtest_data)
report.generate('my_backtest.html')
```

## Data Format

### Required Data Structure

```python
backtest_data = {
    'equity_curve': pd.Series,  # Portfolio equity over time
    'trades': pd.DataFrame,     # Trade history (optional)
    'strategy_params': dict,    # Strategy configuration (optional)
    'start_date': datetime,     # Backtest start (optional)
    'end_date': datetime,       # Backtest end (optional)
    'initial_capital': float,   # Starting capital (optional)
}
```

### Equity Curve Format

```python
# pd.Series with DatetimeIndex
equity_curve = pd.Series(
    data=[100000, 101500, 103200, ...],  # Portfolio values
    index=pd.date_range('2023-01-01', '2023-12-31', freq='D')
)
```

### Trades DataFrame Format

```python
# pd.DataFrame with required columns
trades = pd.DataFrame({
    'entry_time': [...],   # pd.Timestamp
    'exit_time': [...],    # pd.Timestamp
    'pnl': [...],          # float (profit/loss in $)
    'direction': [...],    # str ('LONG', 'SHORT', etc.)
})
```

## Report Sections

### Cover Page
- Strategy name and date range
- Key metrics summary table
- Branding (logo, company name)

### Executive Summary
- Performance overview text
- Comprehensive statistics table
- Monthly returns heatmap

### Equity Curve
- Equity curve chart (with optional benchmark)
- Drawdown analysis chart
- Rolling Sharpe ratio chart

### Performance Analysis
- Returns distribution histogram
- Risk-adjusted metrics table
- Best/worst day statistics

### Trade Analysis
- Trade statistics table
- Win/loss breakdown
- Average trade metrics
- Consecutive wins/losses

### Risk Analysis
- Value at Risk (VaR) 95% and 99%
- Conditional VaR (CVaR)
- Downside deviation
- Ulcer Index
- Beta/Alpha (if benchmark provided)

### Appendix
- Full trade list (up to 100 trades shown)
- Monthly statistics
- Strategy parameters

## Customization

### Report Configuration

```python
from kimsfinance.reporting import ReportConfig

config = ReportConfig(
    title="My Backtest Report",
    strategy_name="Momentum + Mean Reversion",
    company_name="Your Company",
    logo_path="/path/to/logo.png",  # Optional
    chart_dpi=150,                   # Chart resolution
    include_monthly_heatmap=True,
    include_rolling_sharpe=True,
    include_trade_list=True,
    risk_free_rate=0.02,            # 2% annual
)

report = BacktestReport(backtest_data, config)
report.generate('custom_report.pdf')
```

### Benchmark Comparison

```python
# Compare strategy to benchmark (e.g., SPY)
benchmark_equity = pd.Series(...)  # Same format as equity_curve

report = BacktestReport(backtest_data, benchmark=benchmark_equity)
report.generate('report_with_benchmark.pdf')
```

This will:
- Add benchmark to equity curve chart
- Calculate beta and alpha
- Show relative performance

### Custom Colors

```python
config = ReportConfig(
    primary_color=(46, 134, 171),    # RGB for headers
    secondary_color=(162, 59, 114),  # RGB for accents
)
```

## Performance Metrics

### Calculated Metrics

The reporting module automatically calculates:

**Returns**:
- Total return
- Annualized return
- Daily return mean/std
- Monthly returns

**Risk-Adjusted**:
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Information ratio (if benchmark)

**Risk Metrics**:
- Maximum drawdown
- Drawdown duration
- Annualized volatility
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Downside deviation
- Ulcer Index

**Trade Statistics**:
- Total trades
- Win rate
- Profit factor
- Average trade P&L
- Average winner/loser
- Largest win/loss
- Max consecutive wins/losses

### Manual Metric Calculation

You can also calculate metrics independently:

```python
from kimsfinance.reporting import (
    calculate_performance_metrics,
    calculate_trade_statistics,
    calculate_risk_metrics,
    calculate_monthly_returns,
)

# Performance metrics
perf = calculate_performance_metrics(equity_curve, benchmark, risk_free_rate=0.02)
print(f"Sharpe Ratio: {perf.sharpe_ratio:.2f}")
print(f"Max Drawdown: {perf.max_drawdown * 100:.2f}%")

# Trade statistics
trade_stats = calculate_trade_statistics(trades_df)
print(f"Win Rate: {trade_stats.win_rate * 100:.1f}%")
print(f"Profit Factor: {trade_stats.profit_factor:.2f}")

# Risk metrics
returns = equity_curve.pct_change().dropna()
risk = calculate_risk_metrics(returns, equity_curve, benchmark_returns)
print(f"VaR (95%): {risk.value_at_risk_95 * 100:.2f}%")

# Monthly returns heatmap data
monthly = calculate_monthly_returns(equity_curve)
print(monthly)  # DataFrame with year x month
```

## Chart Generation

### Individual Charts

You can generate individual charts without creating a full report:

```python
from kimsfinance.reporting.charts import (
    create_equity_curve,
    create_drawdown_chart,
    create_returns_distribution,
    create_monthly_heatmap,
    create_rolling_sharpe,
)

# Equity curve
img = create_equity_curve(equity_curve, benchmark, width=800, height=400, dpi=100)
img.save('equity_curve.png')

# Drawdown chart
img = create_drawdown_chart(equity_curve, width=800, height=300, dpi=100)
img.save('drawdown.png')

# Returns distribution
returns = equity_curve.pct_change().dropna()
img = create_returns_distribution(returns, width=600, height=400, dpi=100)
img.save('returns_dist.png')

# Monthly heatmap
monthly = calculate_monthly_returns(equity_curve)
img = create_monthly_heatmap(monthly, width=700, height=400, dpi=100)
img.save('monthly_heatmap.png')

# Rolling Sharpe
img = create_rolling_sharpe(returns, window=63, width=800, height=300, dpi=100)
img.save('rolling_sharpe.png')
```

All charts return PIL Image objects and can be saved or embedded.

## Examples

### Complete Example with Real Data

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from kimsfinance.reporting import BacktestReport, HTMLReport, ReportConfig

# Load your backtest results
equity_curve = pd.read_csv('equity.csv', index_col=0, parse_dates=True)['equity']
trades = pd.read_csv('trades.csv', parse_dates=['entry_time', 'exit_time'])

# Strategy parameters
strategy_params = {
    'strategy': 'Mean Reversion',
    'entry_lookback': 20,
    'exit_threshold': 2.0,
    'stop_loss': 0.02,
}

# Prepare data
backtest_data = {
    'equity_curve': equity_curve,
    'trades': trades,
    'strategy_params': strategy_params,
}

# Custom configuration
config = ReportConfig(
    title="Mean Reversion Strategy Backtest",
    strategy_name="20-Period Mean Reversion",
    company_name="Quant Capital",
    chart_dpi=150,
)

# Generate both reports
pdf_report = BacktestReport(backtest_data, config)
pdf_report.generate('backtest_report.pdf')

html_report = HTMLReport(backtest_data, config)
html_report.generate('backtest_report.html')

print("Reports generated successfully!")
```

### Minimal Example (No Trades)

```python
# Even without trades, you can generate a report with performance metrics
backtest_data = {
    'equity_curve': equity_curve,
}

report = BacktestReport(backtest_data)
report.generate('performance_only.pdf')
```

## HTML Report Features

The HTML report includes:

- **Responsive Design**: Works on desktop and mobile
- **Print Support**: Optimized for printing
- **Interactive**: Click metrics to copy to clipboard
- **Embedded Charts**: Base64-encoded PNG charts (no external files)
- **Print Button**: Floating button for easy printing

Open in browser and use "Print to PDF" for PDF export from HTML.

## Performance

**Benchmarked on kimsfinance hardware** (Intel i9-13980HX, RTX 3500 Ada):

| Report Type | Time | Size |
|-------------|------|------|
| PDF (10 pages) | 0.6s | 184KB |
| HTML | 0.3s | 226KB |

**Optimization features**:
- Lazy chart generation (only when needed)
- Efficient PIL rendering (kimsfinance's optimized renderer)
- Compressed images (PNG with optimization)
- Single-pass report building

**Target met**: <5 seconds for standard report (actual: <1 second)

## Troubleshooting

### Missing Dependencies

```
ImportError: ReportLab is required for PDF generation
```

**Solution**: Install reporting dependencies:
```bash
pip install kimsfinance[reporting]
```

### Matplotlib Backend Issues

```
ImportError: Cannot load backend: TkAgg
```

**Solution**: The reporting module uses the 'Agg' backend automatically (non-interactive). No action needed.

### Chart Generation Fails

If chart generation fails, placeholder charts are shown with error message. Enable matplotlib:
```bash
pip install matplotlib
```

### Font Issues (PDF)

ReportLab uses standard fonts by default. For custom fonts:

```python
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont('CustomFont', '/path/to/font.ttf'))
```

## Advanced Usage

### Custom Report Sections

While the reporting module provides comprehensive default sections, you can extend it:

```python
from kimsfinance.reporting.pdf_report import BacktestReport

class CustomBacktestReport(BacktestReport):
    def _create_custom_section(self):
        """Add your custom section."""
        from reportlab.platypus import Paragraph
        from reportlab.lib.styles import getSampleStyleSheet

        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>My Custom Section</b>", styles['Heading1']))
        # Add your custom content

        return story

    def generate(self, output_path):
        """Override to include custom section."""
        # Add custom section to story before building
        # (Requires modifying the _build_story method)
        return super().generate(output_path)
```

### Export Metrics to JSON

```python
import json

# Calculate metrics
from kimsfinance.reporting import calculate_performance_metrics, calculate_trade_statistics

perf = calculate_performance_metrics(equity_curve)
trades = calculate_trade_statistics(trades_df)

# Export to JSON
metrics = {
    'performance': {
        'total_return': perf.total_return,
        'sharpe_ratio': perf.sharpe_ratio,
        'max_drawdown': perf.max_drawdown,
    },
    'trades': {
        'total_trades': trades.total_trades,
        'win_rate': trades.win_rate,
        'profit_factor': trades.profit_factor,
    }
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

## Integration with Backtesting

### QuantConnect/LEAN Integration

For QuantConnect users, convert results to kimsfinance format:

```python
# Convert QuantConnect results
def qc_to_kimsfinance(qc_results):
    # Extract equity curve
    equity_curve = pd.Series(
        data=qc_results['TotalPortfolioValue'],
        index=pd.to_datetime(qc_results.index)
    )

    # Extract trades
    trades = pd.DataFrame({
        'entry_time': qc_results['Trade']['EntryTime'],
        'exit_time': qc_results['Trade']['ExitTime'],
        'pnl': qc_results['Trade']['ProfitLoss'],
        'direction': qc_results['Trade']['Direction'],
    })

    return {
        'equity_curve': equity_curve,
        'trades': trades,
    }

# Generate report
backtest_data = qc_to_kimsfinance(qc_results)
report = BacktestReport(backtest_data)
report.generate('qc_backtest.pdf')
```

### Backtrader Integration

```python
import backtrader as bt

class BacktestReporter(bt.Analyzer):
    def __init__(self):
        self.equity = []
        self.dates = []
        self.trades = []

    def next(self):
        self.dates.append(self.datas[0].datetime.date(0))
        self.equity.append(self.strategy.broker.getvalue())

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                'entry_time': bt.num2date(trade.dtopen),
                'exit_time': bt.num2date(trade.dtclose),
                'pnl': trade.pnl,
                'direction': 'LONG' if trade.size > 0 else 'SHORT',
            })

# Run backtest
cerebro = bt.Cerebro()
cerebro.addanalyzer(BacktestReporter)
results = cerebro.run()

# Generate report
reporter = results[0].analyzers.backtestReporter
backtest_data = {
    'equity_curve': pd.Series(reporter.equity, index=reporter.dates),
    'trades': pd.DataFrame(reporter.trades),
}
report = BacktestReport(backtest_data)
report.generate('bt_backtest.pdf')
```

## Best Practices

1. **Always include a benchmark**: Provides context for performance
2. **Document strategy parameters**: Include in `strategy_params`
3. **Use descriptive titles**: Help identify reports later
4. **Archive reports**: Keep PDF copies of all backtests
5. **Compare multiple strategies**: Generate reports for each variant
6. **Review risk metrics**: Don't just focus on returns

## API Reference

See inline documentation in:
- `kimsfinance/reporting/__init__.py` - Module overview
- `kimsfinance/reporting/pdf_report.py` - PDF report generation
- `kimsfinance/reporting/html_report.py` - HTML report generation
- `kimsfinance/reporting/metrics.py` - Metric calculations
- `kimsfinance/reporting/charts.py` - Chart generation

## Contributing

To improve the reporting module:

1. Add new metrics in `metrics.py`
2. Add new charts in `charts.py`
3. Extend report sections in `pdf_report.py` and `html_report.py`
4. Update this documentation

## License

Part of kimsfinance - AGPL-3.0-or-later

---

**Last Updated**: 2025-11-03
**Version**: 0.2.0
