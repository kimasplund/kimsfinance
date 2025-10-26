"""
Visualization tools for backtest results

Provides plotting functions for equity curves, drawdowns, trade analysis,
and performance metrics dashboards.
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.dates import DateFormatter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    import warnings
    warnings.warn("matplotlib not available. Install with: pip install matplotlib")


def plot_equity_curve(result, timestamps=None, title="Equity Curve", figsize=(12, 6)):
    """
    Plot equity curve from backtest results

    Parameters:
    - result: Backtest result dictionary
    - timestamps: Optional array of timestamps for x-axis
    - title: Plot title
    - figsize: Figure size tuple (width, height)

    Returns:
    - matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for visualization")

    fig, ax = plt.subplots(figsize=figsize)

    equity = result['equity_curve']
    x = timestamps if timestamps is not None else np.arange(len(equity))

    ax.plot(x, equity, linewidth=2, color='#2E86AB', label='Equity')
    ax.axhline(y=result.get('final_equity', equity[-1]), color='gray',
               linestyle='--', alpha=0.5, label=f"Final: ${result.get('final_equity', equity[-1]):,.2f}")

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time' if timestamps is not None else 'Bars', fontsize=12)
    ax.set_ylabel('Equity ($)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_drawdown(result, timestamps=None, title="Drawdown", figsize=(12, 4)):
    """
    Plot drawdown from equity curve

    Parameters:
    - result: Backtest result dictionary
    - timestamps: Optional array of timestamps for x-axis
    - title: Plot title
    - figsize: Figure size tuple

    Returns:
    - matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for visualization")

    fig, ax = plt.subplots(figsize=figsize)

    equity = result['equity_curve']
    x = timestamps if timestamps is not None else np.arange(len(equity))

    # Calculate drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100

    ax.fill_between(x, drawdown, 0, color='#A23B72', alpha=0.3)
    ax.plot(x, drawdown, color='#A23B72', linewidth=2)
    ax.axhline(y=result.get('max_drawdown', drawdown.min()), color='red',
               linestyle='--', alpha=0.7, label=f"Max DD: {result.get('max_drawdown', drawdown.min()):.2f}%")

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time' if timestamps is not None else 'Bars', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_trade_distribution(result, title="Trade P&L Distribution", figsize=(10, 6)):
    """
    Plot trade profit/loss distribution

    Parameters:
    - result: Backtest result dictionary
    - title: Plot title
    - figsize: Figure size tuple

    Returns:
    - matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for visualization")

    trades = result.get('trades', [])
    if not trades:
        print("No trades to plot")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Extract P&L
    pnl = [t['pnl'] for t in trades]
    pnl_pct = [t['pnl_percent'] for t in trades]

    # Histogram of P&L
    ax1.hist(pnl, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_title('Trade P&L Distribution ($)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('P&L ($)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Histogram of P&L percentage
    ax2.hist(pnl_pct, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_title('Trade P&L Distribution (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('P&L (%)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_performance_dashboard(result, timestamps=None, figsize=(16, 10)):
    """
    Comprehensive performance dashboard with multiple subplots

    Parameters:
    - result: Backtest result dictionary
    - timestamps: Optional array of timestamps for x-axis
    - figsize: Figure size tuple

    Returns:
    - matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for visualization")

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    equity = result['equity_curve']
    x = timestamps if timestamps is not None else np.arange(len(equity))
    trades = result.get('trades', [])

    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, equity, linewidth=2, color='#2E86AB')
    ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    ax2.fill_between(x, drawdown, 0, color='#A23B72', alpha=0.3)
    ax2.plot(x, drawdown, color='#A23B72', linewidth=2)
    ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Trade P&L Distribution
    ax3 = fig.add_subplot(gs[2, 0])
    if trades:
        pnl = [t['pnl'] for t in trades]
        ax3.hist(pnl, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('P&L ($)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Performance Metrics
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    metrics_text = f"""
    PERFORMANCE METRICS

    Total Return:      {result.get('total_return', 0.0):>10.2f}%
    Sharpe Ratio:      {result.get('sharpe_ratio', 0.0):>10.2f}
    Max Drawdown:      {result.get('max_drawdown', 0.0):>10.2f}%

    Win Rate:          {result.get('win_rate', 0.0):>10.1f}%
    Profit Factor:     {result.get('profit_factor', 0.0):>10.2f}

    Number of Trades:  {result.get('num_trades', 0):>10d}
    Final Equity:      ${result.get('final_equity', 0.0):>10,.2f}
    """

    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle('Backtest Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)

    return fig


def print_performance_summary(result):
    """
    Print formatted performance summary to console

    Parameters:
    - result: Backtest result dictionary
    """
    print("\n" + "="*60)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Final Equity:      ${result.get('final_equity', 0.0):,.2f}")
    print(f"Total Return:      {result.get('total_return', 0.0):+.2f}%")
    print(f"Sharpe Ratio:      {result.get('sharpe_ratio', 0.0):.3f}")
    print(f"Max Drawdown:      {result.get('max_drawdown', 0.0):.2f}%")
    print(f"Win Rate:          {result.get('win_rate', 0.0):.1f}%")
    print(f"Profit Factor:     {result.get('profit_factor', 0.0):.2f}")
    print(f"Number of Trades:  {result.get('num_trades', 0)}")
    print("="*60)

    # Print sample trades
    trades = result.get('trades', [])
    if trades:
        print("\nSample Trades (first 5):")
        for i, trade in enumerate(trades[:5]):
            print(f"\n  Trade #{i+1}")
            print(f"    Entry: {trade.get('entry_time', 'N/A')} @ ${trade.get('entry_price', 0.0):.2f}")
            print(f"    Exit:  {trade.get('exit_time', 'N/A')} @ ${trade.get('exit_price', 0.0):.2f}")
            print(f"    Type:  {trade.get('direction', 'N/A')}")
            print(f"    P&L:   ${trade.get('pnl', 0.0):+.2f} ({trade.get('pnl_percent', 0.0):+.2f}%)")

        if len(trades) > 5:
            print(f"\n  ... and {len(trades) - 5} more trades")

    print()
