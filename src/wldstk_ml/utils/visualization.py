"""This module provides utility functions for visualizing results."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


# Plotting options are keyword-only parameters by design, so the counts are expected.
def scatter_plot_2d(  # pylint: disable=too-many-arguments,too-many-locals
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    title: str = "2D Scatter Plot",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    legend_title: str = "Classes",
    features: tuple[int, int] = (0, 1),
    cmap: str | Colormap = "viridis",
    alpha: float = 0.5,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create a 2D scatter plot of the data points, colored by their labels.

    Args:
        X: Data of shape (n_samples, n_features), with at least two features.
        y: Optional labels of shape (n_samples,) used to color the points.
        title: Title of the plot.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        legend_title: Title of the legend (e.g. "Classes" or "Clusters").
        features: Indices of the two columns of X to plot.
        cmap: Matplotlib colormap (name or object) used for the labels.
        alpha: Transparency of the points.
        save_path: If given, the figure is saved to this file.
        show: If True, the figure is displayed. If no GUI backend is available,
            the figure is saved to "<title>.png" instead (unless save_path is set).

    Returns:
        The matplotlib figure and axes, so callers can customize them further.
    """
    X = _check_features(X)
    i, j = features
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, i], X[:, j], c=y, cmap=cmap if y is not None else None, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y is not None:
        ax.legend(*scatter.legend_elements(), title=legend_title)

    _finalize(fig, title, save_path, show)
    return fig, ax


def scatter_plot_2d_overlay(  # pylint: disable=too-many-arguments,too-many-locals
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Prediction vs. Ground Truth",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    legend_title: str = "Classes",
    features: tuple[int, int] = (0, 1),
    cmap: str | Colormap = "viridis",
    align_labels: bool = False,
    highlight_errors: bool = True,
    error_color: str = "red",
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Overlay predicted labels on top of the ground truth in one 2D scatter plot.

    Each point is drawn as a ring whose color is its ground-truth label, filled with
    a dot whose color is its predicted label. Wherever they disagree, the point is
    additionally marked with a red cross.

    Args:
        X: Data of shape (n_samples, n_features), with at least two features.
        y_true: Ground-truth labels of shape (n_samples,).
        y_pred: Predicted labels (or cluster indices) of shape (n_samples,).
        title: Title of the plot. The accuracy is appended to it.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        legend_title: Title of the color legend.
        features: Indices of the two columns of X to plot.
        cmap: Matplotlib colormap (name or object) used for the labels.
        align_labels: Set to True for clustering results. Cluster indices are
            arbitrary, so each cluster is first mapped to the ground-truth label
            that is most common within it (see `align_cluster_labels`).
        highlight_errors: If True, mismatching points are marked with a red cross.
        error_color: Color of the mismatch crosses.
        save_path: If given, the figure is saved to this file.
        show: If True, the figure is displayed (see `scatter_plot_2d`).

    Returns:
        The matplotlib figure and axes, so callers can customize them further.
    """
    X = _check_features(X)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not len(X) == len(y_true) == len(y_pred):
        raise ValueError("X, y_true and y_pred must have the same number of samples")
    if align_labels:
        y_pred = align_cluster_labels(y_true, y_pred)

    # One shared discrete color per label, so ring and dot colors are comparable.
    labels = np.unique(np.concatenate([y_true, y_pred]))
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(labels)))
    color_map = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(labels) + 1) - 0.5, len(labels))
    true_idx = np.searchsorted(labels, y_true)
    pred_idx = np.searchsorted(labels, y_pred)

    i, j = features
    fig, ax = plt.subplots(figsize=(8, 6))
    # Ground truth: colored rings.
    ax.scatter(
        X[:, i],
        X[:, j],
        s=120,
        facecolors="none",
        edgecolors=[tuple(c) for c in colors[true_idx]],
        linewidths=2,
        alpha=0.8,
    )
    # Prediction: colored dots inside the rings.
    ax.scatter(X[:, i], X[:, j], s=25, c=pred_idx, cmap=color_map, norm=norm, alpha=0.9)

    errors = y_true != y_pred
    if highlight_errors and errors.any():
        ax.scatter(X[errors, i], X[errors, j], s=160, marker="x", c=error_color, linewidths=1.5)

    accuracy = 1.0 - errors.mean()
    ax.set_title(f"{title} (accuracy: {accuracy:.1%})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    color_handles = [
        Line2D([], [], marker="o", linestyle="", color=colors[k], label=str(label))
        for k, label in enumerate(labels)
    ]
    color_legend = ax.legend(handles=color_handles, title=legend_title, loc="upper left")
    ax.add_artist(color_legend)
    style_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor="gray",
            markeredgewidth=2,
            markersize=11,
            label="ground truth",
        ),
        Line2D([], [], marker="o", linestyle="", color="gray", markersize=5, label="prediction"),
    ]
    if highlight_errors:
        style_handles.append(
            Line2D(
                [],
                [],
                marker="x",
                linestyle="",
                color=error_color,
                markersize=10,
                label=f"mismatch ({errors.sum()})",
            )
        )
    ax.legend(handles=style_handles, loc="upper right")

    _finalize(fig, title, save_path, show)
    return fig, ax


def align_cluster_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Map each predicted cluster to the ground-truth label most common within it.

    Cluster indices from unsupervised methods such as k-means are arbitrary
    (cluster 0 may correspond to class 2), so they must be aligned before they
    can be compared with ground-truth labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    aligned = np.empty_like(y_true)
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        values, counts = np.unique(y_true[mask], return_counts=True)
        aligned[mask] = values[np.argmax(counts)]
    return aligned


def _check_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(f"X must have shape (n_samples, >=2), got {X.shape}")
    return X


def _finalize(fig: Figure, title: str, save_path: str | Path | None, show: bool) -> None:
    """Save and/or show the figure, falling back to a PNG when there is no GUI."""
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        if _has_gui_backend():
            plt.show()
        elif save_path is None:
            fallback = Path(f"{title.replace(' ', '_')}.png")
            fig.savefig(fallback, bbox_inches="tight")
            print(f"No GUI backend available; plot saved to {fallback.resolve()}")


def _has_gui_backend() -> bool:
    """Return True if the active matplotlib backend can open a window."""
    backend = matplotlib.get_backend().lower()
    return backend not in {"agg", "pdf", "ps", "svg", "cairo", "template"}
