"""Implementation of the

Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

clustering algorithm.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from wldstk_ml.utils.trees import KdTree

NOISE = -1


class PointType(StrEnum):
    """Role of a sample in the density-based clustering."""

    CORE = "core"
    BORDER = "border"
    OUTLIER = "outlier"


@dataclass
class Point:
    """Clustering result of one sample."""

    type: PointType = PointType.OUTLIER
    cluster: int = NOISE


class DBSCAN:
    """Clusters samples by density and marks samples in sparse regions as outliers.

    Attributes:
        eps: Radius of the neighborhood of a sample.
        min_samples: Neighbors needed for a core point, counting the sample itself.
        points: Type and cluster of every sample, set by `predict`.
    """

    def __init__(self, eps: float = 1.0, min_samples: int = 5):
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be at least 1, got {min_samples}")

        self.eps = eps
        self.min_samples = min_samples
        self.points: list[Point] = []

    def _expand_cluster(
        self, start: int, cluster: int, neighbors: list[list[int]], is_core: np.ndarray
    ) -> None:
        """Adds all samples that are density-reachable from the core point `start`."""

        self.points[start] = Point(PointType.CORE, cluster)
        stack = [start]

        while stack:
            for neighbor in neighbors[stack.pop()]:
                if self.points[neighbor].cluster != NOISE:
                    continue  # a border point stays in the first cluster that reached it

                if is_core[neighbor]:
                    self.points[neighbor] = Point(PointType.CORE, cluster)
                    stack.append(neighbor)
                else:
                    self.points[neighbor] = Point(PointType.BORDER, cluster)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Run DBSCAN and return the cluster index of every sample, or -1 for outliers."""

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must have shape (n_samples, n_features), got {X.shape}")

        samples = X.tolist()
        tree = KdTree(samples, k=X.shape[1])
        neighbors = [
            tree.query_radius(sample, self.eps)
            for sample in tqdm(samples, desc="DBSCAN Neighborhoods")
        ]
        is_core = np.array([len(found) >= self.min_samples for found in neighbors], dtype=bool)

        self.points = [Point() for _ in range(len(X))]
        cluster = 0
        for index in np.flatnonzero(is_core):
            if self.points[index].cluster == NOISE:
                self._expand_cluster(int(index), cluster, neighbors, is_core)
                cluster += 1

        return np.array([point.cluster for point in self.points], dtype=int)
