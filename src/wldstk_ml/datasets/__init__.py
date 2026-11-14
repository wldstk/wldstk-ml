"""Bundled datasets. Each loader returns a `Dataset` that unpacks as `X, y`."""

from wldstk_ml.datasets._base import Dataset, load_csv_dataset
from wldstk_ml.datasets.artificial import (
    CLUSTER_PRESETS,
    NOISE_LABEL,
    generate_clusters,
    load_clusters,
)
from wldstk_ml.datasets.loader import load_iris

__all__ = [
    "CLUSTER_PRESETS",
    "NOISE_LABEL",
    "Dataset",
    "generate_clusters",
    "load_clusters",
    "load_csv_dataset",
    "load_iris",
]
