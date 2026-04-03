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

    # optional clipping for better visualization
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
    hist_color = "#1a1a1a"   # near-black bars
    edge_color = "#444444"   # dark edges
    mean_color = "#1F77B4"   # blue
    min_color  = "#2CA02C"   # green
    max_color  = "#D62728"   # red

    p90 = float(np.percentile(x, 90))
    p95 = float(np.percentile(x, 95))
    pct_lines = [(90, p90, "#F58518"), (95, p95, "#72B7B2")]

    fig, (ax, ax_log) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.8))

    def _draw_hist(ax, log_scale=False):
        counts_arr, bin_edges_arr, _ = ax.hist(
            x,
            bins=bins,
            color=hist_color,
            alpha=0.85,
            edgecolor=edge_color,
            linewidth=0.6,
            zorder=2,
            log=log_scale,
        )

        if show_stats:
            import matplotlib.patches as mpatches
            stat_entries = [
                (stats["mean"], mean_color, "mean"),
                (stats["min"],  min_color,  "min"),
                (stats["max"],  max_color,  "max"),
            ]
            legend_handles = []
            for val, color, key in stat_entries:
                ax.axvline(val, color=color, alpha=0.45, linewidth=3, zorder=1)
                legend_handles.append(
                    mpatches.Patch(
                        color=color, alpha=0.7,
                        label=f"{key} = {val:.{stat_precision}f}",
                    )
                )
            ax.legend(handles=legend_handles, fontsize=8, loc="center right",
                      framealpha=0.7, edgecolor="#cccccc")

        suffix = " (log scale)" if log_scale else ""
        ax.set_title(title + suffix, fontsize=12, pad=10)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=x_ticks))
        if not log_scale:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))

        if grid:
            ax.grid(True, which="major", axis="both", linewidth=0.6, alpha=0.25)
            ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)

        # ---- CDF on twin right axis ----
        cum = np.cumsum(counts_arr) / counts_arr.sum()
        cdf_x = np.concatenate([[bin_edges_arr[0]], bin_edges_arr[1:]])
        cdf_y = np.concatenate([[0.0], cum])
        ax2 = ax.twinx()
        ax2.plot(cdf_x, cdf_y, color="#E45756", linewidth=1.5, zorder=3)
        ax2.set_ylabel("CDF", fontsize=10, color="#E45756")
        ax2.tick_params(axis="y", labelcolor="#E45756", labelsize=8)
        ax2.set_ylim(0.7, 1.02)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax2.spines["top"].set_visible(False)

        # ---- percentile lines + annotations ----
        xform = ax.get_xaxis_transform()  # x: data coords, y: axes fraction
        y2_min, y2_max = ax2.get_ylim()
        for pct, val, color in pct_lines:
            ax.axvline(val, color=color, linestyle="--", linewidth=1.2, alpha=0.85, zorder=4)
            ax2.axhline(pct / 100, color=color, linestyle=":", linewidth=0.8, alpha=0.5)
            # convert ax2 data coord to axes fraction so text sits on the horizontal line
            y_frac = (pct / 100 - y2_min) / (y2_max - y2_min)
            ax.text(
                val, y_frac, f"p{pct}={val:.4g}",
                transform=xform, color=color,
                fontsize=8, va="bottom", ha="left", rotation=0,
            )

    _draw_hist(ax, log_scale=False)
    _draw_hist(ax_log, log_scale=True)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
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