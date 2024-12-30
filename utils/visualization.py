#!/usr/bin/env python3
"""
Visualization utilities for trading performance analysis
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

# Set styling for plots
# NOTE: seaborn-v0_8-darkgrid was removed in matplotlib 3.6+
# Using a modern equivalent that works across versions
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    # Fallback for newer matplotlib versions
    plt.style.use("seaborn-darkgrid" if "seaborn-darkgrid" in plt.style.available else "default")
sns.set_palette("deep")


def plot_equity_curve(
    equity_curve: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None
) -> Figure:
    """
    Plot equity curve with optional benchmark comparison

    Args:
        equity_curve: DataFrame with 'equity' column and datetime index
        benchmark_data: Optional benchmark equity curve for comparison

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot equity curve
    equity_curve["equity"].plot(ax=ax, linewidth=2)

    # Plot benchmark if provided
    if benchmark_data is not None and "equity" in benchmark_data.columns:
        # Normalize to same starting value for fair comparison
        norm_factor = equity_curve["equity"].iloc[0] / benchmark_data["equity"].iloc[0]
        benchmark_norm = benchmark_data["equity"] * norm_factor
        benchmark_norm.plot(ax=ax, linewidth=2, linestyle="--", label="Benchmark")

    # Add drawdown subplot
    if "drawdown" in equity_curve.columns:
        ax2 = ax.twinx()
        equity_curve["drawdown"].plot(ax=ax2, color="red", alpha=0.3, linewidth=1)
        ax2.fill_between(equity_curve.index, 0, equity_curve["drawdown"], color="red", alpha=0.1)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_ylim(equity_curve["drawdown"].min() * 1.1, 0)  # Give 10% more space below min

    # Format plot
    ax.set_title("Strategy Equity Curve", fontsize=16)
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig


def plot_returns_distribution(returns: pd.Series) -> Figure:
    """
    Plot returns distribution with normal curve overlay

    Args:
        returns: Series of returns data

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot histogram of returns
    sns.histplot(returns, kde=True, ax=ax, stat="density", bins=50)

    # Add normal distribution curve for comparison
    x = np.linspace(returns.min(), returns.max(), 100)
    y = np.exp(-((x - returns.mean()) ** 2) / (2 * returns.std() ** 2)) / (
        returns.std() * np.sqrt(2 * np.pi)
    )
    ax.plot(x, y, "r--", linewidth=2, label="Normal Distribution")

    # Calculate skew and kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # Add statistics to plot
    stats_text = f"Mean: {returns.mean():.4f}\nStd Dev: {returns.std():.4f}\nSkewness: {skewness:.4f}\nKurtosis: {kurtosis:.4f}"
    ax.text(
        0.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    # Format plot
    ax.set_title("Returns Distribution", fontsize=16)
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    return fig


def plot_drawdown_periods(equity_curve: pd.DataFrame, top_n: int = 5) -> Figure:
    """
    Plot top drawdown periods

    Args:
        equity_curve: DataFrame with 'equity' and 'drawdown' columns
        top_n: Number of top drawdown periods to highlight

    Returns:
        Matplotlib figure object
    """
    if "drawdown" not in equity_curve.columns:
        # Calculate drawdown if not in dataframe
        equity_curve = equity_curve.copy()
        equity_curve["drawdown"] = calculate_drawdown(equity_curve["equity"])

    # Find drawdown periods
    drawdown_periods = find_drawdown_periods(equity_curve["drawdown"])

    # Sort by maximum drawdown
    drawdown_periods.sort_values("max_drawdown", ascending=True, inplace=True)
    top_periods = drawdown_periods.head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot full equity curve
    equity_curve["equity"].plot(ax=ax, color="blue", alpha=0.3, linewidth=1)

    # Highlight top drawdown periods
    cmap = plt.get_cmap("rainbow")
    colors = cmap(np.linspace(0, 1, len(top_periods)))

    for i, (_, period) in enumerate(top_periods.iterrows()):
        start_idx = period["start_date"]
        end_idx = period["end_date"]
        recovery_idx = period["recovery_date"]

        # Get slice of equity curve for this period
        if recovery_idx:
            mask = (equity_curve.index >= start_idx) & (equity_curve.index <= recovery_idx)
        else:
            mask = (equity_curve.index >= start_idx) & (equity_curve.index <= end_idx)

        period_equity = equity_curve.loc[mask, "equity"]

        # Plot this drawdown period
        ax.plot(
            period_equity.index,
            period_equity.values,
            linewidth=2,
            color=colors[i],
            label=f"DD #{i+1}: {period['max_drawdown']:.1%}, {period['duration']} days",
        )

        # Mark drawdown start, bottom and recovery points
        ax.scatter(start_idx, period_equity.iloc[0], color=colors[i], marker="^", s=100)
        ax.scatter(end_idx, equity_curve.loc[end_idx, "equity"], color=colors[i], marker="v", s=100)

        if recovery_idx:
            ax.scatter(
                recovery_idx,
                equity_curve.loc[recovery_idx, "equity"],
                color=colors[i],
                marker="o",
                s=100,
            )

    # Format plot
    ax.set_title("Top Drawdown Periods", fontsize=16)
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig


def plot_monthly_returns(returns: pd.Series) -> Figure:
    """
    Plot monthly returns heatmap

    Args:
        returns: Series of returns with datetime index

    Returns:
        Matplotlib figure object
    """
    # Resample returns to daily if higher frequency
    if returns.index.inferred_freq not in ["D", "B", None]:
        daily_returns = returns.resample("D").sum()
    else:
        daily_returns = returns

    # Group returns by year and month
    monthly_returns = daily_returns.groupby(
        [daily_returns.index.year, daily_returns.index.month]
    ).sum()

    # Reshape to matrix for heatmap (year x month)
    returns_matrix = monthly_returns.unstack()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        returns_matrix,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn",
        center=0,
        linewidths=1,
        ax=ax,
        cbar_kws={"label": "Return"},
    )

    # Format plot
    ax.set_title("Monthly Returns (%)", fontsize=16)
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    fig.tight_layout()

    return fig


def plot_rolling_performance(returns: pd.Series, window: int = 60) -> Figure:
    """
    Plot rolling Sharpe ratio and volatility

    Args:
        returns: Series of returns with datetime index
        window: Rolling window in days

    Returns:
        Matplotlib figure object
    """
    # Calculate rolling metrics
    rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot rolling Sharpe ratio
    rolling_sharpe.plot(ax=axes[0], linewidth=2)
    axes[0].axhline(y=1, color="r", linestyle="--", alpha=0.5)
    axes[0].axhline(y=2, color="g", linestyle="--", alpha=0.5)
    axes[0].set_title(f"Rolling {window}-day Sharpe Ratio", fontsize=14)
    axes[0].set_ylabel("Sharpe Ratio")
    axes[0].grid(True)

    # Plot rolling volatility
    rolling_vol.plot(ax=axes[1], linewidth=2, color="orange")
    axes[1].set_title(f"Rolling {window}-day Annualized Volatility", fontsize=14)
    axes[1].set_ylabel("Volatility")
    axes[1].set_xlabel("Date")
    axes[1].grid(True)

    fig.tight_layout()
    return fig


def plot_trade_analysis(trades: List[Dict]) -> Tuple[Figure, Figure]:
    """
    Plot trade analysis charts

    Args:
        trades: List of trade dictionaries

    Returns:
        Tuple of Matplotlib figure objects (trade_results, trade_durations)
    """
    # Convert trades to DataFrame
    if not trades:
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        ax1.axis("off")
        ax1.text(0.5, 0.5, "No trades to analyze", ha="center", va="center", fontsize=14)

        fig2, ax2 = plt.subplots(figsize=(12, 7))
        ax2.axis("off")
        ax2.text(0.5, 0.5, "No trades to analyze", ha="center", va="center", fontsize=14)

        return fig1, fig2

    trade_df = pd.DataFrame(trades)

    # Calculate necessary fields if not present
    if (
        "profit_loss_pct" not in trade_df.columns
        and "entry_price" in trade_df.columns
        and "exit_price" in trade_df.columns
    ):
        trade_df["profit_loss_pct"] = np.where(
            trade_df["side"] == "long",
            (trade_df["exit_price"] - trade_df["entry_price"]) / trade_df["entry_price"],
            (trade_df["entry_price"] - trade_df["exit_price"]) / trade_df["entry_price"],
        )

    if (
        "duration" not in trade_df.columns
        and "entry_time" in trade_df.columns
        and "exit_time" in trade_df.columns
    ):
        trade_df["duration"] = (
            pd.to_datetime(trade_df["exit_time"]) - pd.to_datetime(trade_df["entry_time"])
        ).dt.days

    # Figure 1: Trade Results
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Winners vs Losers pie chart
    winners = (trade_df["profit_loss_pct"] > 0).sum()
    losers = (trade_df["profit_loss_pct"] <= 0).sum()
    axes1[0, 0].pie(
        [winners, losers], labels=["Winners", "Losers"], autopct="%1.1f%%", colors=["green", "red"]
    )
    axes1[0, 0].set_title("Trade Outcomes")

    # Plot 2: Profit/Loss distribution
    sns.histplot(data=trade_df, x="profit_loss_pct", ax=axes1[0, 1], kde=True)
    axes1[0, 1].axvline(x=0, color="r", linestyle="--")
    axes1[0, 1].set_title("Profit/Loss Distribution")
    axes1[0, 1].set_xlabel("Profit/Loss %")

    # Plot 3: Cumulative returns
    trade_df["cumulative_return"] = (1 + trade_df["profit_loss_pct"]).cumprod() - 1
    trade_df["cumulative_return"].plot(ax=axes1[1, 0])
    axes1[1, 0].set_title("Cumulative Returns from Trades")
    axes1[1, 0].set_ylabel("Return")
    axes1[1, 0].set_xlabel("Trade #")

    # Plot 4: Trade P&L by symbol
    if "symbol" in trade_df.columns:
        symbol_profit = trade_df.groupby("symbol")["profit_loss_pct"].mean().sort_values()
        symbol_profit.plot(kind="barh", ax=axes1[1, 1])
        axes1[1, 1].set_title("Average P&L by Symbol")
        axes1[1, 1].set_xlabel("Average Profit/Loss %")

    fig1.tight_layout()

    # Figure 2: Trade Durations
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Trade duration histogram
    sns.histplot(data=trade_df, x="duration", ax=axes2[0])
    axes2[0].set_title("Trade Duration Distribution")
    axes2[0].set_xlabel("Duration (days)")

    # Plot 2: Return vs Duration scatter
    sns.scatterplot(
        data=trade_df,
        x="duration",
        y="profit_loss_pct",
        hue="side" if "side" in trade_df.columns else None,
        ax=axes2[1],
    )
    axes2[1].axhline(y=0, color="r", linestyle="--")
    axes2[1].set_title("Return vs Duration")
    axes2[1].set_xlabel("Duration (days)")
    axes2[1].set_ylabel("Profit/Loss %")

    fig2.tight_layout()

    return fig1, fig2


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from equity curve

    Args:
        equity_curve: Series of equity values

    Returns:
        Series of drawdown values (negative percentages)
    """
    # Calculate running maximum
    running_max = equity_curve.cummax()

    # Calculate drawdown percentage
    drawdown = (equity_curve - running_max) / running_max

    return drawdown


def find_drawdown_periods(drawdown_series: pd.Series) -> pd.DataFrame:
    """
    Find significant drawdown periods

    Args:
        drawdown_series: Series of drawdown values

    Returns:
        DataFrame with drawdown periods information
    """
    # Initialize variables
    periods = []
    in_drawdown = False
    start_idx = None
    current_dd = 0

    # Iterate through drawdown series
    for idx, dd in drawdown_series.items():
        if not in_drawdown and dd < 0:
            # Start of a new drawdown period
            in_drawdown = True
            start_idx = idx
            current_dd = dd
        elif in_drawdown:
            if dd < current_dd:
                # Drawdown getting worse
                current_dd = dd
            elif dd == 0:
                # Drawdown recovery
                periods.append(
                    {
                        "start_date": start_idx,
                        "end_date": drawdown_series[current_dd == drawdown_series].index[0],
                        "recovery_date": idx,
                        "max_drawdown": current_dd,
                        "duration": (idx - start_idx).days,
                    }
                )
                in_drawdown = False
                start_idx = None
                current_dd = 0

    # Handle case where we're still in a drawdown at the end of the series
    if in_drawdown:
        periods.append(
            {
                "start_date": start_idx,
                "end_date": drawdown_series[current_dd == drawdown_series].index[0],
                "recovery_date": None,
                "max_drawdown": current_dd,
                "duration": (drawdown_series.index[-1] - start_idx).days,
            }
        )

    return pd.DataFrame(periods)


def create_performance_report(
    backtest_results: Dict,
    benchmark_data: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Create and save comprehensive performance report

    Args:
        backtest_results: Dictionary containing backtest results
        benchmark_data: Optional benchmark data for comparison
        output_path: Path to save report figures (None to display)
    """
    # Extract data from backtest results
    equity_curve = backtest_results["equity_curve"]
    trades = backtest_results["trades"]

    # Calculate returns if not present
    if "returns" not in equity_curve.columns:
        equity_curve["returns"] = equity_curve["equity"].pct_change()

    # Calculate drawdown if not present
    if "drawdown" not in equity_curve.columns:
        equity_curve["drawdown"] = calculate_drawdown(equity_curve["equity"])

    # Generate all plots
    figures = {}

    # Equity curve plot
    figures["equity_curve"] = plot_equity_curve(equity_curve, benchmark_data)

    # Returns distribution
    figures["returns_dist"] = plot_returns_distribution(equity_curve["returns"].dropna())

    # Drawdown periods
    figures["drawdowns"] = plot_drawdown_periods(equity_curve)

    # Monthly returns heatmap
    figures["monthly_returns"] = plot_monthly_returns(equity_curve["returns"].dropna())

    # Rolling performance
    figures["rolling_performance"] = plot_rolling_performance(equity_curve["returns"].dropna())

    # Trade analysis
    if trades:
        figures["trade_results"], figures["trade_durations"] = plot_trade_analysis(trades)

    # Save or display figures
    if output_path:
        import os

        os.makedirs(output_path, exist_ok=True)

        for name, fig in figures.items():
            if fig is not None:
                fig.savefig(os.path.join(output_path, f"{name}.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)
    else:
        # Display figures
        for fig in figures.values():
            if fig is not None:
                plt.show()
