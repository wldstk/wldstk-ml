"""This script demonstrates the kd-tree on random 3D points.

Three panels:
1. How the tree splits the space: every node splits its box with a plane at its point,
   cycling between the x, y and z axes. Only the top levels are drawn to keep it readable.
2. Nearest-neighbor search with `closest_point`.
3. Radius search with `query_radius`, as used by DBSCAN.

The panel titles show how many points the search had to compare, out of all points.
"""

# matplotlib leaves `Axes3D.scatter(zs=0)` unannotated, so pyright infers `zs: int` and
# rejects float arrays. The calls are correct; only this check is disabled.
# pyright: reportArgumentType=false

import math
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from wldstk_ml.utils.logger import get_logger
from wldstk_ml.utils.trees import Leaf, Node, Point, KdTree

logger = get_logger(__name__)

N_POINTS = 125
QUERY = (0.55, 0.45, 0.6)
RADIUS = 0.2
DRAWN_LEVELS = 3  # splitting planes of deeper levels are not drawn
BOUNDS = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))  # (min, max) per axis
AXIS_NAMES = ("x", "y", "z")


class CountingKdTree(KdTree):
    """KdTree that records the points it compares, to show how much the tree prunes."""

    compared: list[Point] = []

    @staticmethod
    def _distance(point1: Point, point2: Point) -> float:
        CountingKdTree.compared.append(point2)
        return math.dist(point1, point2)


def plot_compared(ax: Axes3D) -> int:
    """Highlight the points the last search compared and return how many there were."""
    compared = np.array(CountingKdTree.compared).reshape(-1, 3)
    ax.scatter(
        compared[:, 0], compared[:, 1], compared[:, 2], s=14, color="tab:green", label="compared"
    )
    return len(compared)


def tree_point(tree: KdTree, node: Node) -> Point:
    """Return the point that a node splits at."""
    return tree._points[node.point_index]  # pylint: disable=protected-access


def tree_depth(node: Node | Leaf | None) -> int:
    """Return the number of splitting levels below and including `node`."""
    if not isinstance(node, Node):
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def split_plane(axis: int, value: float, bounds: tuple) -> list[tuple[float, float, float]]:
    """Return the 4 corners of the plane `axis = value`, clipped to the given box."""
    (a_min, a_max), (b_min, b_max) = [bounds[i] for i in range(3) if i != axis]
    corners = []
    for a, b in ((a_min, b_min), (a_max, b_min), (a_max, b_max), (a_min, b_max)):
        corner = [a, b]
        corner.insert(axis, value)
        corners.append((corner[0], corner[1], corner[2]))
    return corners


def draw_splits(
    ax: Axes3D, tree: KdTree, node: Node | Leaf | None, depth: int, bounds: tuple
) -> None:
    """Draw the splitting plane of every node in the top levels, clipped to its box."""
    if not isinstance(node, Node) or depth >= DRAWN_LEVELS:
        return

    color = plt.get_cmap("viridis")(depth / DRAWN_LEVELS)
    plane = Poly3DCollection(
        [split_plane(node.axis, node.split, bounds)],
        facecolor=color,
        edgecolor=color,
        # Fade from 0.25 at the root to 0.05 at the deepest drawn level, for any DRAWN_LEVELS.
        alpha=0.25 - 0.2 * depth / max(DRAWN_LEVELS - 1, 1),
    )
    ax.add_collection3d(plane)

    lower, upper = list(bounds), list(bounds)
    lower[node.axis] = (bounds[node.axis][0], node.split)
    upper[node.axis] = (node.split, bounds[node.axis][1])
    draw_splits(ax, tree, node.left, depth + 1, tuple(lower))
    draw_splits(ax, tree, node.right, depth + 1, tuple(upper))


def draw_sphere(ax: Axes3D, center: Point, radius: float, color: str) -> None:
    """Draw a wireframe sphere."""
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 15j]
    ax.plot_wireframe(
        center[0] + radius * np.cos(u) * np.sin(v),
        center[1] + radius * np.sin(u) * np.sin(v),
        center[2] + radius * np.cos(v),
        color=color,
        linewidth=0.5,
        alpha=0.6,
    )


def setup_axes(ax: Axes3D, title: str) -> None:
    """Apply the common look of all panels."""
    ax.set_title(title)
    for axis, (low, high) in enumerate(BOUNDS):
        getattr(ax, f"set_{AXIS_NAMES[axis]}lim")(low, high)
        getattr(ax, f"set_{AXIS_NAMES[axis]}label")(AXIS_NAMES[axis])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-60)


def main():
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, size=(N_POINTS, 3))
    points = [tuple(p) for p in X.tolist()]
    tree = CountingKdTree(points, leaf_size=1)  # one point per leaf, like the 2D demo
    root = tree._root  # pylint: disable=protected-access
    assert isinstance(root, Node)
    depth = tree_depth(root)
    logger.info("Built 3D kd-tree with %d points and %d levels", N_POINTS, depth)

    fig = plt.figure(figsize=(19, 7))
    axes = [cast(Axes3D, fig.add_subplot(1, 3, i, projection="3d")) for i in (1, 2, 3)]

    # 1. Partition of the space
    ax = axes[0]
    draw_splits(ax, tree, root, 0, BOUNDS)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=6, color="black", depthshade=True)
    ax.scatter(*tree_point(tree, root), s=120, color="red", label="root (splits on x)")
    ax.legend(loc="upper left")
    setup_axes(
        ax,
        f"Tree structure: {N_POINTS} points, {depth} levels\n"
        f"(top {DRAWN_LEVELS} levels: x, then y, then z planes)",
    )

    # 2. Nearest neighbor
    ax = axes[1]
    CountingKdTree.compared = []
    nearest = tree.closest_point(QUERY)
    assert nearest is not None
    nn_distance = math.dist(QUERY, nearest)

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=6, color="lightgray")
    nn_checks = plot_compared(ax)
    logger.info("Nearest neighbor of %s: %s, %d distance checks", QUERY, nearest, nn_checks)
    draw_sphere(ax, QUERY, nn_distance, "tab:orange")
    ax.plot(*zip(QUERY, nearest), color="tab:orange", linewidth=2)
    ax.scatter(*QUERY, marker="*", s=250, color="tab:red", label="query")
    ax.scatter(*nearest, s=80, color="tab:orange", label="nearest point")
    ax.legend(loc="upper left")
    setup_axes(
        ax,
        f"closest_point: compared {nn_checks} of {N_POINTS} points\n(distance {nn_distance:.3f})",
    )

    # 3. Radius search
    ax = axes[2]
    CountingKdTree.compared = []
    found = tree.query_radius(QUERY, RADIUS)

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=6, color="lightgray")
    radius_checks = plot_compared(ax)
    logger.info("Points within %.2f of %s: %d, %d checks", RADIUS, QUERY, len(found), radius_checks)
    draw_sphere(ax, QUERY, RADIUS, "tab:blue")
    ax.scatter(X[found, 0], X[found, 1], X[found, 2], s=40, color="tab:blue", label="within radius")
    ax.scatter(*QUERY, marker="*", s=250, color="tab:red", label="query")
    ax.legend(loc="upper left")
    setup_axes(
        ax,
        f"query_radius(r={RADIUS}): {len(found)} found,\ncompared {radius_checks} of {N_POINTS} points",
    )

    fig.suptitle("kd-tree demo in 3D (green = points the search compared)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    if matplotlib.get_backend().lower() in {"agg", "pdf", "ps", "svg", "cairo", "template"}:
        path = Path("kdtree_demo_3d.png").resolve()
        fig.savefig(path, dpi=110, bbox_inches="tight")
        logger.info("No GUI backend available; demo saved to %s", path)
    else:
        plt.show()  # 3D panels can be rotated with the mouse


if __name__ == "__main__":
    main()
