"""
Performance metrics calculation for backtest reports.

Provides comprehensive metrics including:
- Returns (total, annualized, monthly, daily)
- Risk metrics (volatility, Sharpe, Sortino, max drawdown)
- Trade statistics (win rate, profit factor, average trade)
- Risk analytics (VaR, CVaR, correlation)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Performance metrics for a backtest."""

    # Returns
    total_return: float
    annualized_return: float
    daily_return_mean: float
    daily_return_std: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    calmar_ratio: float

    # Additional metrics
    volatility_annual: float
    best_day: float
    worst_day: float
    positive_days_pct: float


@dataclass
class TradeStatistics:
    """Trade-level statistics."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float

    avg_trade: float
    avg_winning_trade: float
    avg_losing_trade: float

    largest_win: float
    largest_loss: float

    avg_trade_duration: float  # hours
    max_consecutive_wins: int
    max_consecutive_losses: int


@dataclass
class RiskMetrics:
    """Risk analysis metrics."""

    value_at_risk_95: float  # 95% VaR
    value_at_risk_99: float  # 99% VaR
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    cvar_99: float

    downside_deviation: float
    ulcer_index: float
    beta: float | None  # vs benchmark
    alpha: float | None  # vs benchmark


def calculate_performance_metrics(
    equity_curve: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.02,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from equity curve.

    Args:
        equity_curve: Time series of portfolio equity values
        benchmark: Optional benchmark returns for comparison
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        PerformanceMetrics with all calculated metrics
    """
    # Calculate returns
    returns = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Annualized metrics
    days = len(equity_curve)
    years = days / 252  # Trading days
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Daily statistics
    daily_return_mean = returns.mean()
    daily_return_std = returns.std()

    # Annualized volatility
    volatility_annual = daily_return_std * np.sqrt(252)

    # Sharpe ratio
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility_annual if volatility_annual > 0 else 0

    # Sortino ratio (using downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

    # Drawdown calculation
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    # Drawdown duration (days in drawdown)
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        # Find longest consecutive True sequence
        groups = (in_drawdown != in_drawdown.shift()).cumsum()[in_drawdown]
        max_drawdown_duration = groups.value_counts().max() if len(groups) > 0 else 0
    else:
        max_drawdown_duration = 0

    # Calmar ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

    # Best/worst days
    best_day = returns.max()
    worst_day = returns.min()
    positive_days_pct = (returns > 0).sum() / len(returns) * 100

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        daily_return_mean=daily_return_mean,
        daily_return_std=daily_return_std,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        calmar_ratio=calmar_ratio,
        volatility_annual=volatility_annual,
        best_day=best_day,
        worst_day=worst_day,
        positive_days_pct=positive_days_pct,
    )


def calculate_trade_statistics(trades: pd.DataFrame) -> TradeStatistics:
    """
    Calculate trade-level statistics.

    Args:
        trades: DataFrame with columns: entry_time, exit_time, pnl, direction

    Returns:
        TradeStatistics with all calculated metrics
    """
    if len(trades) == 0:
        # Return empty stats
        return TradeStatistics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            profit_factor=0,
            avg_trade=0,
            avg_winning_trade=0,
            avg_losing_trade=0,
            largest_win=0,
            largest_loss=0,
            avg_trade_duration=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
        )

    total_trades = len(trades)
    pnl = trades["pnl"]

    # Win/loss statistics
    winning_trades = (pnl > 0).sum()
    losing_trades = (pnl < 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Profit factor
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average metrics
    avg_trade = pnl.mean()
    avg_winning_trade = pnl[pnl > 0].mean() if winning_trades > 0 else 0
    avg_losing_trade = pnl[pnl < 0].mean() if losing_trades > 0 else 0

    # Largest win/loss
    largest_win = pnl.max()
    largest_loss = pnl.min()

    # Trade duration
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        durations = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600
        avg_trade_duration = durations.mean()
    else:
        avg_trade_duration = 0

    # Consecutive wins/losses
    is_winner = pnl > 0
    max_consecutive_wins = _max_consecutive(is_winner)
    max_consecutive_losses = _max_consecutive(~is_winner)

    return TradeStatistics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade=avg_trade,
        avg_winning_trade=avg_winning_trade,
        avg_losing_trade=avg_losing_trade,
        largest_win=largest_win,
        largest_loss=largest_loss,
        avg_trade_duration=avg_trade_duration,
        max_consecutive_wins=max_consecutive_wins,
        max_consecutive_losses=max_consecutive_losses,
    )


def calculate_risk_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> RiskMetrics:
    """
    Calculate risk analysis metrics.

    Args:
        returns: Daily returns series
        equity_curve: Equity curve for drawdown calculations
        benchmark_returns: Optional benchmark returns for beta/alpha

    Returns:
        RiskMetrics with all calculated metrics
    """
    # Value at Risk (historical method)
    var_95 = abs(returns.quantile(0.05))
    var_99 = abs(returns.quantile(0.01))

    # Conditional VaR (Expected Shortfall)
    cvar_95 = abs(returns[returns <= -var_95].mean()) if (returns <= -var_95).any() else var_95
    cvar_99 = abs(returns[returns <= -var_99].mean()) if (returns <= -var_99).any() else var_99

    # Downside deviation (semi-deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0

    # Ulcer Index (drawdown-based risk measure)
    running_max = equity_curve.expanding().max()
    drawdown_pct = ((equity_curve - running_max) / running_max) * 100
    ulcer_index = np.sqrt((drawdown_pct**2).mean())

    # Beta and Alpha (if benchmark provided)
    beta = None
    alpha = None
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        # Align indices
        aligned = pd.DataFrame({"portfolio": returns, "benchmark": benchmark_returns}).dropna()
        if len(aligned) > 0:
            covariance = aligned.cov().loc["portfolio", "benchmark"]
            benchmark_variance = aligned["benchmark"].var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

            # Alpha = portfolio return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
            portfolio_return = (1 + aligned["portfolio"]).prod() - 1
            benchmark_return = (1 + aligned["benchmark"]).prod() - 1
            risk_free_rate = 0.02  # 2% annual
            alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))

    return RiskMetrics(
        value_at_risk_95=var_95,
        value_at_risk_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        downside_deviation=downside_deviation,
        ulcer_index=ulcer_index,
        beta=beta,
        alpha=alpha,
    )


def calculate_monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns from equity curve.

    Args:
        equity_curve: Time series of portfolio equity

    Returns:
        DataFrame with monthly returns, indexed by year-month
    """
    # Ensure datetime index
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        equity_curve.index = pd.to_datetime(equity_curve.index)

    # Resample to month-end
    monthly_equity = equity_curve.resample("ME").last()

    # Calculate monthly returns
    monthly_returns = monthly_equity.pct_change().dropna()

    # Create DataFrame with year and month columns
    df = pd.DataFrame(
        {
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
            "return": monthly_returns.values,
        }
    )

    # Pivot to heatmap format
    heatmap = df.pivot(index="year", columns="month", values="return")

    # Rename columns to month names
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    heatmap.columns = [month_names[m - 1] for m in heatmap.columns]

    return heatmap


def _max_consecutive(boolean_series: pd.Series) -> int:
    """Helper to find maximum consecutive True values."""
    if not boolean_series.any():
        return 0

    groups = (boolean_series != boolean_series.shift()).cumsum()[boolean_series]
    return groups.value_counts().max() if len(groups) > 0 else 0
