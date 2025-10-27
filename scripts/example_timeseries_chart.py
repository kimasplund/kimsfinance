#!/usr/bin/env python3
"""
Quick test for render_timeseries_chart function
"""

import numpy as np
from kimsfinance.plotting import render_timeseries_chart

# Generate sample equity curve data
np.random.seed(42)
n_points = 100

# Equity curve with some trend and volatility
equity = np.zeros(n_points)
equity[0] = 10000
for i in range(1, n_points):
    returns = np.random.randn() * 0.02 + 0.001  # 0.1% drift, 2% volatility
    equity[i] = equity[i-1] * (1 + returns)

# Benchmark (more stable)
benchmark = np.zeros(n_points)
benchmark[0] = 10000
for i in range(1, n_points):
    returns = np.random.randn() * 0.01 + 0.0005  # 0.05% drift, 1% volatility
    benchmark[i] = benchmark[i-1] * (1 + returns)

# Test 1: Single equity curve
print("Test 1: Single equity curve...")
img1 = render_timeseries_chart(
    {'Equity': equity},
    title='Portfolio Equity Curve',
    x_label='Time',
    y_label='Equity ($)',
    theme='modern',
    width=1200,
    height=800,
)
img1.save('/tmp/test_equity_single.webp', 'WEBP', quality=95)
print(f"  ✓ Saved to /tmp/test_equity_single.webp ({img1.size})")

# Test 2: Multi-line comparison
print("\nTest 2: Multi-line comparison...")
img2 = render_timeseries_chart(
    {
        'Strategy': equity,
        'Benchmark': benchmark,
    },
    title='Strategy vs Benchmark',
    x_label='Trading Days',
    y_label='Portfolio Value ($)',
    theme='classic',
    line_colors=['#2E86AB', '#F18F01'],
    width=1200,
    height=800,
)
img2.save('/tmp/test_equity_multi.webp', 'WEBP', quality=95)
print(f"  ✓ Saved to /tmp/test_equity_multi.webp ({img2.size})")

# Test 3: Drawdown chart with area fill
print("\nTest 3: Drawdown chart with area fill...")
running_max = np.maximum.accumulate(equity)
drawdown = (equity - running_max) / running_max * 100

img3 = render_timeseries_chart(
    {'Drawdown': drawdown},
    title='Portfolio Drawdown',
    x_label='Time',
    y_label='Drawdown (%)',
    theme='modern',
    fill_area=True,
    line_colors=['#A23B72'],
    width=1200,
    height=800,
)
img3.save('/tmp/test_drawdown.webp', 'WEBP', quality=95)
print(f"  ✓ Saved to /tmp/test_drawdown.webp ({img3.size})")

# Test 4: Multiple strategies comparison
print("\nTest 4: Multiple strategies...")
strategy2 = equity * 0.95 + np.random.randn(n_points) * 50
strategy3 = equity * 1.05 - np.random.randn(n_points) * 30

img4 = render_timeseries_chart(
    {
        'RSI Strategy': equity,
        'MACD Strategy': strategy2,
        'Combined': strategy3,
        'Benchmark': benchmark,
    },
    title='Multi-Strategy Performance',
    x_label='Days',
    y_label='Equity ($)',
    theme='tradingview',
    line_colors=['#2E86AB', '#A23B72', '#F18F01', '#5A67D8'],
    width=1600,
    height=900,
)
img4.save('/tmp/test_multi_strategy.webp', 'WEBP', quality=95)
print(f"  ✓ Saved to /tmp/test_multi_strategy.webp ({img4.size})")

print("\n✅ All tests passed! Check /tmp/test_*.webp for outputs.")
print("\nUsage example for backtest visualization:")
print("""
from kimsfinance.plotting import render_timeseries_chart

# From your backtest result
result = backtest_engine.run(...)
equity_curve = result['equity_curve']

img = render_timeseries_chart(
    {'Equity': equity_curve},
    title='Backtest Results',
    y_label='Portfolio Value ($)',
    x_label='Time',
    fill_area=True
)
img.save('backtest_equity.webp', 'WEBP')
""")
