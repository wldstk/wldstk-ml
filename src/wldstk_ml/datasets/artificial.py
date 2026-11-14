"""Artificial Gaussian clusters with a controlled overlap, for developing clustering algorithms.

How `overlap` works
-------------------
Two Gaussian clusters with standard deviation sigma and centers at distance d overlap by

    overlap = Phi(-d / (2 * sigma))

where Phi is the standard normal CDF. This is the fraction of a cluster's samples that lie
on the other cluster's side of the midpoint between them, so even an ideal classifier gets
them wrong. `generate_clusters` inverts it to get the distance between the centers:

    d = -2 * sigma * Phi^-1(overlap)

overlap = 0.10 therefore means "about 10 % of each cluster lie in the neighboring cluster".
"""

import dataclasses
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from wldstk_ml.datasets._base import Dataset

NOISE_LABEL = -1
WELL_SEPARATED = 10.0  # distance between centers in standard deviations when overlap == 0
NOISE_PADDING = 0.1  # background noise extends the clusters' bounding box by 10 % per side


def generate_clusters(  # pylint: disable=too-many-arguments,too-many-locals
    n_samples: int,
    n_clusters: int = 2,
    n_features: int = 2,
    *,
    overlap: float = 0.0,
    weights: Sequence[float] | None = None,
    cluster_std: float = 1.0,
    anisotropy: float = 1.0,
    noise: float = 0.0,
    shuffle: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> Dataset:
    """Generate Gaussian clusters with `n_samples` samples in total.

    Args:
        n_samples: Total number of samples, including the noise samples.
        n_clusters: Number of clusters.
        n_features: Number of dimensions.
        overlap: Fraction of a cluster's samples that fall on the side of a neighboring
            cluster, in [0, 0.5). 0 gives well-separated clusters, 0.5 would put the
            centers on top of each other. See the module docstring.
        weights: Relative size of each cluster, e.g. (3, 1) for a 75 % / 25 % split.
            Equal sizes if None.
        cluster_std: Standard deviation of every cluster.
        anisotropy: Ratio of the largest to the smallest standard deviation of a cluster.
            Values > 1 stretch each cluster into a randomly oriented ellipse. The overlap
            is exact only for 1 (round clusters); stretched clusters can overlap more or
            less, depending on how they are oriented.
        noise: Fraction of `n_samples` drawn uniformly in the bounding box of the clusters
            instead of from a cluster. They get the label -1, like DBSCAN outliers.
        shuffle: Shuffle the samples. If False, they are ordered by cluster, noise last.
        random_state: Seed or generator for reproducible data.

    Returns:
        A `Dataset` with features x1, x2, ... and the cluster index as target "cluster".
        `X, y = generate_clusters(...).to_numpy()` gives numpy arrays.

    The cluster centers are placed so that every center has its nearest neighbor at the
    distance computed from `overlap`. With n_features >= n_clusters - 1, all centers are
    equally far apart (corners of a regular simplex: a line, triangle, tetrahedron, ...),
    so every pair of clusters overlaps by `overlap`. With fewer dimensions they sit on a
    regular polygon, where only neighbors on the polygon overlap by `overlap`.
    """
    _check_arguments(n_samples, n_clusters, n_features, overlap, cluster_std, anisotropy, noise)
    rng = np.random.default_rng(random_state)

    n_noise = round(noise * n_samples)
    counts = _cluster_sizes(n_samples - n_noise, n_clusters, weights)

    distance = cluster_std * _separation(overlap)
    centers = _random_rotation(n_features, rng) @ _center_layout(n_clusters, n_features, distance)

    blocks = [
        _sample_cluster(center, count, cluster_std, anisotropy, rng)
        for center, count in zip(centers.T, counts)
    ]
    labels = [np.full(count, cluster) for cluster, count in enumerate(counts)]

    clustered = np.vstack(blocks)
    if n_noise:
        low, high = clustered.min(axis=0), clustered.max(axis=0)
        pad = NOISE_PADDING * (high - low)
        blocks.append(rng.uniform(low - pad, high + pad, size=(n_noise, n_features)))
        labels.append(np.full(n_noise, NOISE_LABEL))

    X, y = np.vstack(blocks), np.concatenate(labels)
    if shuffle:
        order = rng.permutation(n_samples)
        X, y = X[order], y[order]

    return Dataset(
        name=f"clusters_k{n_clusters}_d{n_features}_overlap{overlap:g}",
        X=pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(n_features)]),
        y=pd.Series(y, name="cluster"),
        target_names=[f"cluster {i}" for i in range(n_clusters)],
    )


# Named settings for `load_clusters`. Every entry is passed to `generate_clusters`.
CLUSTER_PRESETS: dict[str, dict[str, Any]] = {
    # 2 clusters
    "default": {"n_clusters": 2},
    "two_well_separated": {"n_clusters": 2, "overlap": 0.0},
    "two_touching": {"n_clusters": 2, "overlap": 0.01},
    "two_overlapping": {"n_clusters": 2, "overlap": 0.1},
    "two_heavily_overlapping": {"n_clusters": 2, "overlap": 0.3},
    "two_imbalanced": {"n_clusters": 2, "overlap": 0.01, "weights": (9, 1)},
    # 3 or more clusters
    "three_well_separated": {"n_clusters": 3, "overlap": 0.0},
    "three_overlapping": {"n_clusters": 3, "overlap": 0.1},
    "three_imbalanced": {"n_clusters": 3, "overlap": 0.01, "weights": (6, 3, 1)},
    "five_on_a_ring": {"n_clusters": 5, "overlap": 0.02},
    # Hard cases
    "elongated": {"n_clusters": 3, "overlap": 0.01, "anisotropy": 5.0},
    "noisy": {"n_clusters": 3, "overlap": 0.01, "noise": 0.15},
    "high_dimensional": {"n_clusters": 4, "n_features": 10, "overlap": 0.05},
}


def load_clusters(
    name: str = "default",
    n_samples: int = 300,
    random_state: int | np.random.Generator | None = 0,
    **overrides: Any,
) -> Dataset:
    """Generate one of the named cluster datasets in `CLUSTER_PRESETS`.

    Args:
        name: Key of `CLUSTER_PRESETS`, e.g. "two_well_separated" or "noisy".
        n_samples: Total number of samples.
        random_state: Seed; fixed to 0 by default so a preset always gives the same data.
        **overrides: Arguments of `generate_clusters` that replace the preset's values,
            e.g. `load_clusters("noisy", noise=0.3)`.

    Example:
        X, y = load_clusters("two_overlapping").to_numpy()
    """
    if name not in CLUSTER_PRESETS:
        raise KeyError(f"Unknown preset {name!r}; choose from {', '.join(CLUSTER_PRESETS)}")
    settings = {**CLUSTER_PRESETS[name], **overrides}
    dataset = generate_clusters(n_samples, random_state=random_state, **settings)
    return dataclasses.replace(dataset, name=name)


def _check_arguments(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    n_samples: int,
    n_clusters: int,
    n_features: int,
    overlap: float,
    cluster_std: float,
    anisotropy: float,
    noise: float,
) -> None:
    if n_clusters < 1 or n_features < 1:
        raise ValueError("n_clusters and n_features must be at least 1")
    if not 0 <= overlap < 0.5:
        raise ValueError(f"overlap must be in [0, 0.5), got {overlap}")
    if cluster_std <= 0:
        raise ValueError(f"cluster_std must be positive, got {cluster_std}")
    if anisotropy < 1:
        raise ValueError(f"anisotropy must be at least 1, got {anisotropy}")
    if not 0 <= noise < 1:
        raise ValueError(f"noise must be in [0, 1), got {noise}")
    if n_samples - round(noise * n_samples) < n_clusters:
        raise ValueError("n_samples is too small to give every cluster at least one sample")


def _separation(overlap: float) -> float:
    """Distance between neighboring centers in standard deviations: d = -2 Phi^-1(overlap)."""
    if overlap == 0:
        return WELL_SEPARATED
    return -2.0 * NormalDist().inv_cdf(overlap)


def _cluster_sizes(n: int, n_clusters: int, weights: Sequence[float] | None) -> np.ndarray:
    """Split n samples by the weights; rounding leftovers go to the largest remainders."""
    w = np.ones(n_clusters) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n_clusters,) or np.any(w <= 0):
        raise ValueError(f"weights must be {n_clusters} positive numbers, got {weights}")

    exact = n * w / w.sum()
    counts = np.floor(exact).astype(int)
    counts[np.argsort(counts - exact)[: n - counts.sum()]] += 1  # largest remainders first
    if np.any(counts == 0):
        raise ValueError("n_samples is too small for these weights: a cluster would be empty")
    return counts


def _center_layout(n_clusters: int, n_features: int, distance: float) -> np.ndarray:
    """Centers as columns of a (n_features, n_clusters) array, centered on the origin.

    Every center has its nearest neighbor at `distance`.
    """
    layout = np.zeros((n_features, n_clusters))
    if n_clusters == 1:
        return layout

    if n_features >= n_clusters - 1:
        # Regular simplex: the unit vectors e_1..e_k are all sqrt(2) apart. Centering them
        # puts them into a (k-1)-dimensional subspace; the SVD gives coordinates in it.
        corners = np.eye(n_clusters) - 1.0 / n_clusters
        _, _, vt = np.linalg.svd(corners)
        coords = corners @ vt[: n_clusters - 1].T * distance / np.sqrt(2)
        layout[: n_clusters - 1] = coords.T
    elif n_features == 1:
        layout[0] = (np.arange(n_clusters) - (n_clusters - 1) / 2) * distance
    else:
        # Regular polygon: neighboring corners are 2 R sin(pi / k) apart.
        angles = 2 * np.pi * np.arange(n_clusters) / n_clusters
        radius = distance / (2 * np.sin(np.pi / n_clusters))
        layout[0], layout[1] = radius * np.cos(angles), radius * np.sin(angles)
    return layout


def _random_rotation(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random orthogonal (n, n) matrix (QR of a Gaussian matrix, signs fixed)."""
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def _sample_cluster(
    center: np.ndarray,
    count: int,
    cluster_std: float,
    anisotropy: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw `count` Gaussian samples around `center`.

    The standard deviations per axis are spread geometrically between
    cluster_std / sqrt(anisotropy) and cluster_std * sqrt(anisotropy), then rotated randomly.
    """
    n_features = len(center)
    stds = cluster_std * anisotropy ** np.linspace(-0.5, 0.5, n_features)
    samples = rng.standard_normal((count, n_features)) * stds
    if anisotropy != 1:
        samples = samples @ _random_rotation(n_features, rng).T
    return samples + center
