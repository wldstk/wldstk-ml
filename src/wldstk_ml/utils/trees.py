"""This module provides basic tree algorithms such as the kd-tree."""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

Point = Sequence[float]


class Leaf(NamedTuple):
    """End of a KdTree branch: a small group of points that is searched directly."""

    indices: tuple[int, ...]


class Node(NamedTuple):
    """Inner node of a KdTree, holding one point that splits the others.

    Points in the left subtree have values <= `split` on `axis`, and points in the
    right subtree have values >= it.
    """

    point_index: int
    axis: int
    split: float
    left: "Node | Leaf | None"
    right: "Node | Leaf | None"


class KdTree:
    """k-d tree over points with k coordinates, for nearest-neighbor and radius search.

    Like a binary search tree, each node splits its points into a left and a right
    subtree, here at the median of one coordinate. The coordinate (axis) cycles with
    the depth. Subtrees of at most `leaf_size` points are stored as a Leaf and searched
    point by point, which is faster in Python than splitting them further.
    """

    _distance = staticmethod(math.dist)

    def __init__(
        self,
        points: Sequence[Point] | np.ndarray,
        k: int | None = None,
        leaf_size: int = 16,
        max_tree_dims: int = 6,
    ):
        """Builds the tree in constructor."""
        self._k = k if k is not None else (len(points[0]) if len(points) > 0 else 0)

        if self._k < 1 and (k is not None or len(points) > 0):
            raise ValueError(f"k must be at least 1, got {self._k}")
        if any(len(point) != self._k for point in points):
            raise ValueError(f"all points must have k={self._k} coordinates")
        if leaf_size < 1:
            raise ValueError(f"leaf_size must be at least 1, got {leaf_size}")

        self._points = list(points)
        self._leaf_size = leaf_size
        self._coordinates = np.asarray(self._points, dtype=float).reshape(-1, max(self._k, 1))
        self._use_tree = self._k <= max_tree_dims
        self._root = (
            self._build_tree(np.arange(len(self._points)), depth=0) if self._use_tree else None
        )

    @property
    def k(self) -> int:
        """Number of coordinates per point."""
        return self._k

    def _build_tree(self, indices: np.ndarray, depth: int) -> Node | Leaf | None:
        """Build the subtree of the points with the given indices and return its root."""
        n = indices.size

        if n == 0:
            return None
        if n <= self._leaf_size:
            return Leaf(tuple(indices.tolist()))

        axis = depth % self._k
        median = n // 2

        # argpartition finds the median without sorting all points: O(n) per level.
        order = np.argpartition(self._coordinates[indices, axis], median)
        index = int(indices[order[median]])

        return Node(
            point_index=index,
            axis=axis,
            split=float(self._coordinates[index, axis]),
            left=self._build_tree(
                indices[order[:median]], depth + 1
            ),  # Left side of the median -> smaller
            right=self._build_tree(
                indices[order[median + 1 :]], depth + 1
            ),  # Right side of the median -> greater
        )

    @staticmethod
    def distance_squared(point1: Point, point2: Point) -> float:
        """Return the squared Euclidean distance between two points of any dimension.

        numpy has no built-in distance for two points (np.linalg.norm is the closest);
        math.dist is written in C and much faster for single pairs.
        """
        return math.dist(point1, point2) ** 2

    def _check_query(self, point: Point) -> None:
        """Raise if the query does not have k coordinates."""
        if len(point) != self._k and self._points:
            raise ValueError(f"point must have k={self._k} coordinates, got {len(point)}")

    # Values are unpacked into locals on purpose: local lookups keep the search loop fast.
    def closest_point(self, point: Point) -> Point | None:  # pylint: disable=too-many-locals
        """Return the stored point closest to `point`, or None if the tree is empty."""
        self._check_query(point)
        if not self._points:
            return None
        if not self._use_tree:
            distances = np.linalg.norm(self._coordinates - np.asarray(point, dtype=float), axis=1)
            return self._points[int(np.argmin(distances))]

        points, distance = self._points, self._distance
        best_index, best_distance = -1, math.inf
        # Each entry holds a subtree and the distance from the query to its region along
        # one axis. The subtree can only hold a closer point if that is below the best.
        stack: list[tuple[Node | Leaf | None, float]] = [(self._root, 0.0)]

        while stack:
            node, plane_distance = stack.pop()
            if node is None or plane_distance >= best_distance:
                continue

            if isinstance(node, Leaf):
                for index in node.indices:
                    d = distance(point, points[index])
                    if d < best_distance:
                        best_index, best_distance = index, d
                continue

            index, axis, split, left, right = node
            d = distance(point, points[index])
            if d < best_distance:
                best_index, best_distance = index, d

            # Search the side of the splitting plane the point lies on first (pushed last).
            offset = point[axis] - split
            near, far = (left, right) if offset < 0 else (right, left)
            stack.append((far, abs(offset)))
            stack.append((near, 0.0))

        return points[best_index]

    def query_radius(self, point: Point, radius: float) -> list[int]:
        """Return the indices of all stored points within radius of point."""
        self._check_query(point)
        if not self._points:
            return []
        if not self._use_tree:
            distances = np.linalg.norm(self._coordinates - np.asarray(point, dtype=float), axis=1)
            return np.flatnonzero(distances <= radius).tolist()

        points, distance = self._points, self._distance
        matches: list[int] = []
        stack: list[Node | Leaf | None] = [self._root]

        while stack:
            node = stack.pop()
            if node is None:
                continue

            if isinstance(node, Leaf):
                matches.extend(i for i in node.indices if distance(point, points[i]) <= radius)
                continue

            index, axis, split, left, right = node
            if distance(point, points[index]) <= radius:
                matches.append(index)

            value = point[axis]
            if value - radius <= split:
                stack.append(left)
            if value + radius >= split:
                stack.append(right)

        return matches
