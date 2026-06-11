# kimsfinance.reporting

Professional backtest report generation for quantitative trading.

## Quick Start

```python
from kimsfinance.reporting import BacktestReport

# Prepare data
backtest_data = {
    'equity_curve': equity_series,  # pd.Series
    'trades': trades_df,            # pd.DataFrame
}

# Generate PDF
report = BacktestReport(backtest_data)
report.generate('backtest.pdf')
```

## Installation

```bash
pip install kimsfinance[reporting]
```

## Features

- PDF and HTML report generation
- Comprehensive performance metrics
- Trade analysis and statistics
- Risk metrics (VaR, CVaR, etc.)
- Professional charts and visualizations
- Customizable templates and branding
- <5 second generation time

## Documentation

Full documentation: `/docs/REPORTING.md`

## Example

```bash
python examples/generate_backtest_report.py
```

Generates:
- `backtest_report.pdf` - Professional multi-page PDF
- `backtest_report.html` - Interactive HTML report

## Report Sections

1. **Cover Page** - Strategy summary and key metrics
2. **Executive Summary** - Performance overview and statistics
3. **Equity Curve** - Returns and drawdown visualization
4. **Performance Analysis** - Risk-adjusted metrics
5. **Trade Analysis** - Win/loss statistics
6. **Risk Analysis** - VaR, CVaR, risk metrics
7. **Appendix** - Full trade list

## Metrics Calculated

- Total return, annualized return
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown, drawdown duration
- Win rate, profit factor
- VaR, CVaR, downside deviation
- Beta, alpha (vs benchmark)

## Charts Generated

- Equity curve (with optional benchmark)
- Drawdown analysis
- Returns distribution
- Monthly returns heatmap
- Rolling Sharpe ratio

## Performance

- PDF: 0.6 seconds (184KB)
- HTML: 0.3 seconds (226KB)

Tested on: Intel i9-13980HX, RTX 3500 Ada

## License

AGPL-3.0-or-later
