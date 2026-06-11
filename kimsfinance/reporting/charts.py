"""
Chart generation utilities for backtest reports.

Uses kimsfinance's PIL renderer for fast, high-quality chart generation.
All charts are optimized for PDF embedding at 300 DPI.
"""

from __future__ import annotations

from io import BytesIO
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Try to import matplotlib for some specialized charts
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def create_equity_curve(
    equity_curve: pd.Series,
    benchmark: pd.Series | None = None,
    width: int = 800,
    height: int = 400,
    dpi: int = 100,
) -> Image.Image:
    """
    Create equity curve chart.

    Args:
        equity_curve: Time series of portfolio equity
        benchmark: Optional benchmark equity for comparison
        width: Chart width in pixels
        height: Chart height in pixels
        dpi: DPI for rendering

    Returns:
        PIL Image of equity curve
    """
    if not MATPLOTLIB_AVAILABLE:
        return _create_placeholder_chart(width, height, "Equity Curve\n(matplotlib required)")

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve.values, linewidth=2, label="Strategy", color="#2E86AB")

    # Plot benchmark if provided
    if benchmark is not None:
        # Normalize to same starting value
        normalized_benchmark = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
        ax.plot(
            normalized_benchmark.index,
            normalized_benchmark.values,
            linewidth=1.5,
            label="Benchmark",
            color="#A23B72",
            alpha=0.7,
        )

    # Formatting
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Equity ($)", fontsize=10)
    ax.set_title("Equity Curve", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Tight layout
    plt.tight_layout()

    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def create_drawdown_chart(
    equity_curve: pd.Series, width: int = 800, height: int = 300, dpi: int = 100
) -> Image.Image:
    """
    Create drawdown chart showing portfolio drawdowns over time.

    Args:
        equity_curve: Time series of portfolio equity
        width: Chart width in pixels
        height: Chart height in pixels
        dpi: DPI for rendering

    Returns:
        PIL Image of drawdown chart
    """
    if not MATPLOTLIB_AVAILABLE:
        return _create_placeholder_chart(width, height, "Drawdown Chart\n(matplotlib required)")

    # Calculate drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Plot drawdown as filled area
    ax.fill_between(drawdown.index, 0, drawdown.values, color="#D62828", alpha=0.6, label="Drawdown")
    ax.plot(drawdown.index, drawdown.values, linewidth=1.5, color="#9B2226")

    # Formatting
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Drawdown (%)", fontsize=10)
    ax.set_title("Portfolio Drawdown", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Tight layout
    plt.tight_layout()

    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def create_returns_distribution(
    returns: pd.Series, width: int = 600, height: int = 400, dpi: int = 100
) -> Image.Image:
    """
    Create returns distribution histogram.

    Args:
        returns: Daily returns series
        width: Chart width in pixels
        height: Chart height in pixels
        dpi: DPI for rendering

    Returns:
        PIL Image of distribution histogram
    """
    if not MATPLOTLIB_AVAILABLE:
        return _create_placeholder_chart(
            width, height, "Returns Distribution\n(matplotlib required)"
        )

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Create histogram
    n, bins, patches = ax.hist(
        returns.values * 100, bins=50, color="#2E86AB", alpha=0.7, edgecolor="black", linewidth=0.5
    )

    # Color negative returns differently
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor("#D62828")

    # Add normal distribution overlay
    mu = returns.mean() * 100
    sigma = returns.std() * 100
    x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
    normal_dist = (
        1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2) * len(returns)
    )
    # Scale to match histogram
    bin_width = bins[1] - bins[0]
    normal_dist = normal_dist * bin_width

    ax.plot(x, normal_dist, "k--", linewidth=2, label="Normal Distribution")

    # Formatting
    ax.set_xlabel("Daily Return (%)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Returns Distribution", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
    ax.legend(fontsize=9)

    # Tight layout
    plt.tight_layout()

    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def create_monthly_heatmap(
    monthly_returns: pd.DataFrame, width: int = 700, height: int = 400, dpi: int = 100
) -> Image.Image:
    """
    Create monthly returns heatmap.

    Args:
        monthly_returns: DataFrame from calculate_monthly_returns()
        width: Chart width in pixels
        height: Chart height in pixels
        dpi: DPI for rendering

    Returns:
        PIL Image of heatmap
    """
    if not MATPLOTLIB_AVAILABLE:
        return _create_placeholder_chart(width, height, "Monthly Returns\n(matplotlib required)")

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Create heatmap
    data = monthly_returns.values * 100  # Convert to percentage

    # Custom colormap: red for negative, green for positive
    import matplotlib.colors as mcolors

    colors = ["#D62828", "#FFFFFF", "#2A9D8F"]
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # Plot heatmap
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-10, vmax=10)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(monthly_returns.columns)))
    ax.set_yticks(np.arange(len(monthly_returns.index)))
    ax.set_xticklabels(monthly_returns.columns, fontsize=9)
    ax.set_yticklabels(monthly_returns.index, fontsize=9)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add text annotations
    for i in range(len(monthly_returns.index)):
        for j in range(len(monthly_returns.columns)):
            value = data[i, j]
            if not np.isnan(value):
                text_color = "white" if abs(value) > 5 else "black"
                ax.text(
                    j, i, f"{value:.1f}%", ha="center", va="center", color=text_color, fontsize=7
                )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Return (%)", rotation=270, labelpad=15, fontsize=9)

    # Formatting
    ax.set_title("Monthly Returns (%)", fontsize=12, fontweight="bold")

    # Tight layout
    plt.tight_layout()

    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def create_rolling_sharpe(
    returns: pd.Series,
    window: int = 63,  # ~3 months
    width: int = 800,
    height: int = 300,
    dpi: int = 100,
) -> Image.Image:
    """
    Create rolling Sharpe ratio chart.

    Args:
        returns: Daily returns series
        window: Rolling window size in days (default 63 = ~3 months)
        width: Chart width in pixels
        height: Chart height in pixels
        dpi: DPI for rendering

    Returns:
        PIL Image of rolling Sharpe chart
    """
    if not MATPLOTLIB_AVAILABLE:
        return _create_placeholder_chart(width, height, "Rolling Sharpe\n(matplotlib required)")

    # Calculate rolling Sharpe
    rolling_return = returns.rolling(window).mean() * 252  # Annualized
    rolling_std = returns.rolling(window).std() * np.sqrt(252)  # Annualized
    rolling_sharpe = rolling_return / rolling_std

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Plot rolling Sharpe
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color="#2E86AB")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.axhline(y=1, color="green", linewidth=0.8, linestyle="--", alpha=0.5, label="Sharpe = 1")
    ax.axhline(y=2, color="darkgreen", linewidth=0.8, linestyle="--", alpha=0.5, label="Sharpe = 2")

    # Formatting
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Sharpe Ratio", fontsize=10)
    ax.set_title(f"Rolling Sharpe Ratio ({window}-day)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Tight layout
    plt.tight_layout()

    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def _create_placeholder_chart(width: int, height: int, text: str) -> Image.Image:
    """Create a placeholder chart when matplotlib is not available."""
    img = Image.new("RGB", (width, height), color="#F0F0F0")
    draw = ImageDraw.Draw(img)

    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Draw text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(position, text, fill="#333333", font=font)

    return img
