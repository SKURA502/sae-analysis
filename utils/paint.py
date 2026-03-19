import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ArrayLike1D = Union[np.ndarray, Sequence[float]]

def plot_1d_distribution_hist(
    data: ArrayLike1D,
    save_path: str,
    mask: Optional[np.ndarray] = None,
    title: str = "Distribution",
    xlabel: str = "Value",
    ylabel: str = "Count",
    bins: int = 120,
    figsize: Tuple[float, float] = (7.0, 4.5),
    dpi: int = 200,
    show_stats: bool = True,
    stat_precision: int = 4,
    clip_percentile: Optional[Tuple[float, float]] = None,
    x_ticks: int = 6,
    y_ticks: int = 6,
    grid: bool = True,
) -> dict:
    """
    Plot a histogram for 1D data distribution with academic-ish styling.

    Args:
        data: 1D array-like data.
        save_path: Path to save figure (e.g., ".../distribution_mean.png").
        title/xlabel/ylabel: Figure labels.
        bins: Histogram bins.
        figsize: Figure size.
        dpi: Save dpi.
        show_stats: Draw vertical lines for min/mean/max (computed on finite data).
        stat_precision: Decimal places for stats in legend.
        clip_percentile: If set like (1, 99), clip data to percentile range for visualization.
        x_ticks/y_ticks: Target number of ticks (uses MaxNLocator for "even" distribution).
        grid: Whether to draw subtle grid.

    Returns:
        stats dict: {"n":..., "min":..., "mean":..., "max":...}
    """
    x = np.asarray(data, dtype=np.float64).reshape(-1)
    if mask is not None:
        m = np.asarray(mask).astype(bool).reshape(-1)
        if m.shape[0] != x.shape[0]:
            raise ValueError("plot_binned_proportion_bar: mask length must match data length.")
        x = x[m]

    # keep only finite values (drop nan/inf)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("plot_1d_distribution_hist: `data` has no finite values to plot.")

    # optional clipping for better visualization (keeps stats consistent with clipped data if enabled)
    if clip_percentile is not None:
        lo_p, hi_p = clip_percentile
        lo = np.percentile(x, lo_p)
        hi = np.percentile(x, hi_p)
        x = np.clip(x, lo, hi)

    stats = {
        "n": int(x.size),
        "min": float(np.min(x)),
        "mean": float(np.mean(x)),
        "max": float(np.max(x)),
    }

    # ---- Style defaults (matplotlib-only, clean academic look) ----
    # Histogram (neutral) + stat lines (distinct, harmonious)
    hist_color = "#9AA0A6"   # neutral gray
    edge_color = "#FFFFFF"   # white edges to look crisp
    mean_color = "#1F77B4"   # blue
    min_color  = "#2CA02C"   # green
    max_color  = "#D62728"   # red

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # Histogram
    ax.hist(
        x,
        bins=bins,
        color=hist_color,
        alpha=0.85,
        edgecolor=edge_color,
        linewidth=0.6,
    )

    # Stats lines
    if show_stats:
        ax.axvline(
            stats["mean"],
            linewidth=2.2,
            color=mean_color,
            label=f"mean={stats['mean']:.{stat_precision}f}",
            zorder=3,
        )
        ax.axvline(
            stats["min"],
            linewidth=2.0,
            color=min_color,
            label=f"min={stats['min']:.{stat_precision}f}",
            zorder=3,
        )
        ax.axvline(
            stats["max"],
            linewidth=2.0,
            color=max_color,
            label=f"max={stats['max']:.{stat_precision}f}",
            zorder=3,
        )

    # Labels & title
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    # Even-ish ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=x_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))

    # Grid (subtle)
    if grid:
        ax.grid(True, which="major", axis="both", linewidth=0.6, alpha=0.25)
        ax.set_axisbelow(True)

    # Cleaner spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    if show_stats:
        ax.legend(frameon=False, fontsize=10, loc="best")

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)

    return stats

def plot_binned_proportion_bar(
    data: ArrayLike1D,
    save_path: str,
    bins: Union[np.ndarray, Sequence[float]] = np.arange(0.0, 1.1, 0.1),
    mask: Optional[np.ndarray] = None,
    title: str = "Binned Distribution",
    xlabel: str = "Interval",
    ylabel: str = "Proportion",
    figsize: Tuple[float, float] = (10.0, 6.0),
    dpi: int = 200,
    rotate_xticks: int = 45,
    descending: bool = True,
    show_values: bool = False,
    y_ticks: int = 6,
    grid: bool = True,
) -> dict:
    """
    Plot binned proportions as a bar chart, useful for ratio distributions.

    Args:
        data: 1D array-like.
        save_path: output file path.
        bins: bin edges.
        mask: optional boolean mask (same length as data). True means include.
        title/xlabel/ylabel: labels.
        figsize/dpi: figure settings.
        rotate_xticks: rotation degrees for x tick labels.
        descending: if True, reverse bins so larger interval on the left/top.
        show_values: if True, annotate each bar with proportion value.
        y_ticks: target number of y ticks.
        grid: subtle grid.

    Returns:
        dict: {"total":..., "counts":..., "proportions":..., "bin_edges":...}
    """
    x = np.asarray(data, dtype=np.float64).reshape(-1)
    if mask is not None:
        m = np.asarray(mask).astype(bool).reshape(-1)
        if m.shape[0] != x.shape[0]:
            raise ValueError("plot_binned_proportion_bar: mask length must match data length.")
        x = x[m]

    # keep finite
    x = x[np.isfinite(x)]
    total = int(x.size)
    total_safe = max(total, 1)

    bin_edges = np.asarray(bins, dtype=np.float64)
    counts, _ = np.histogram(x, bins=bin_edges)
    proportions = counts / total_safe

    # labels like [a, b) ... last [a, b]
    labels = []
    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i < len(bin_edges) - 2:
            labels.append(f"[{left:.2g}, {right:.2g})")
        else:
            labels.append(f"[{left:.2g}, {right:.2g}]")

    if descending:
        labels = labels[::-1]
        proportions = proportions[::-1]
        counts = counts[::-1]

    # ---- style ----
    bar_color = "#4C78A8"   # muted blue
    edge_color = "#FFFFFF"

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.bar(labels, proportions, color=bar_color, alpha=0.9, edgecolor=edge_color, linewidth=0.6)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))
    if grid:
        ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25)
        ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()

    if show_values:
        # annotate after tight_layout so positions are final
        for i, p in enumerate(proportions):
            ax.text(i, p, f"{p:.3f}", ha="center", va="bottom", fontsize=9, rotation=0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)

    return {
        "total": total,
        "counts": counts,
        "proportions": proportions,
        "bin_edges": bin_edges,
    }