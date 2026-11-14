"""This script demonstrates the kd-tree on random 2D points.

Three panels:
1. How the tree splits the plane: every node splits its region at its point, alternating
   between vertical (x) and horizontal (y) lines.
2. Nearest-neighbor search with `closest_point`.
3. Radius search with `query_radius`, as used by DBSCAN.

The panel titles show how many points the search had to compare, out of all points.
"""

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle

from wldstk_ml.utils.logger import get_logger
from wldstk_ml.utils.trees import Leaf, Node, Point, KdTree

logger = get_logger(__name__)

N_POINTS = 5
QUERY = (0.42, 0.41)
RADIUS = 0.5
BOUNDS = (0.0, 1.0, 0.0, 1.0)  # x_min, x_max, y_min, y_max


class CountingKdTree(KdTree):
    """KdTree that counts its distance calculations, to show how much the tree prunes."""

    checks = 0

    @staticmethod
    def _distance(point1: Point, point2: Point) -> float:
        CountingKdTree.checks += 1
        return math.dist(point1, point2)


def draw_splits(
    ax: Axes, tree: KdTree, node: Node | Leaf | None, depth: int, bounds: tuple, max_depth: int
) -> None:
    """Draw the splitting line of every node, clipped to the region the node covers."""
    if not isinstance(node, Node) or depth > max_depth:
        return

    x_min, x_max, y_min, y_max = bounds
    x, y = tree_point(tree, node)
    color = plt.get_cmap("viridis")(depth / max_depth)
    width = 2.5 - 2 * depth / max_depth

    if depth % 2 == 0:  # split on x: vertical line, left region / right region
        ax.plot([x, x], [y_min, y_max], color=color, linewidth=width)
        left, right = (x_min, x, y_min, y_max), (x, x_max, y_min, y_max)
    else:  # split on y: horizontal line, lower region / upper region
        ax.plot([x_min, x_max], [y, y], color=color, linewidth=width)
        left, right = (x_min, x_max, y_min, y), (x_min, x_max, y, y_max)

    draw_splits(ax, tree, node.left, depth + 1, left, max_depth)
    draw_splits(ax, tree, node.right, depth + 1, right, max_depth)


def tree_point(tree: KdTree, node: Node) -> Point:
    """Return the point that a node splits at."""
    return tree._points[node.point_index]  # pylint: disable=protected-access


def tree_depth(node: Node | Leaf | None) -> int:
    """Return the number of splitting levels below and including `node`."""
    if not isinstance(node, Node):
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def setup_axes(ax: Axes, title: str) -> None:
    """Apply the common look of all panels."""
    ax.set_title(title)
    ax.set_xlim(BOUNDS[0], BOUNDS[1])
    ax.set_ylim(BOUNDS[2], BOUNDS[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def main():
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, size=(N_POINTS, 2))
    points = [tuple(p) for p in X.tolist()]
    tree = CountingKdTree(points, leaf_size=1)  # one point per leaf shows every split
    root = tree._root  # pylint: disable=protected-access
    depth = tree_depth(root)
    logger.info("Built kd-tree with %d points and %d levels", N_POINTS, depth)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    # 1. Partition of the plane
    ax = axes[0]
    draw_splits(ax, tree, root, 0, BOUNDS, max_depth=max(depth - 1, 1))
    ax.scatter(X[:, 0], X[:, 1], s=12, color="black", zorder=3)
    assert isinstance(root, Node)
    ax.scatter(*tree_point(tree, root), s=120, color="red", zorder=4, label="root (splits on x)")
    ax.legend(loc="upper right")
    setup_axes(ax, f"Tree structure: {N_POINTS} points, {depth} levels\n(darker = closer to root)")

    # 2. Nearest neighbor
    ax = axes[1]
    CountingKdTree.checks = 0
    nearest = tree.closest_point(QUERY)
    assert nearest is not None
    nn_checks = CountingKdTree.checks
    nn_distance = KdTree.distance_squared(QUERY, nearest) ** 0.5
    logger.info("Nearest neighbor of %s: %s, %d distance checks", QUERY, nearest, nn_checks)

    draw_splits(ax, tree, root, 0, BOUNDS, max_depth=max(depth - 1, 1))
    ax.scatter(X[:, 0], X[:, 1], s=12, color="lightgray", zorder=3)
    ax.add_patch(Circle(QUERY, nn_distance, fill=False, linestyle="--", color="tab:orange"))
    ax.plot(*zip(QUERY, nearest), color="tab:orange", zorder=4)
    ax.scatter(*QUERY, marker="*", s=250, color="tab:red", zorder=5, label="query")
    ax.scatter(*nearest, s=80, color="tab:orange", zorder=5, label="nearest point")
    ax.legend(loc="upper right")
    setup_axes(
        ax,
        f"closest_point: compared {nn_checks} of {N_POINTS} points\n(distance {nn_distance:.3f})",
    )

    # 3. Radius search
    ax = axes[2]
    CountingKdTree.checks = 0
    found = tree.query_radius(QUERY, RADIUS)
    radius_checks = CountingKdTree.checks
    logger.info("Points within %.2f of %s: %d, %d checks", RADIUS, QUERY, len(found), radius_checks)

    draw_splits(ax, tree, root, 0, BOUNDS, max_depth=max(depth - 1, 1))
    ax.scatter(X[:, 0], X[:, 1], s=12, color="lightgray", zorder=3)
    ax.add_patch(Circle(QUERY, RADIUS, alpha=0.15, color="tab:blue"))
    ax.add_patch(Circle(QUERY, RADIUS, fill=False, color="tab:blue"))
    ax.scatter(X[found, 0], X[found, 1], s=40, color="tab:blue", zorder=4, label="within radius")
    ax.scatter(*QUERY, marker="*", s=250, color="tab:red", zorder=5, label="query")
    ax.legend(loc="upper right")
    setup_axes(
        ax,
        f"query_radius(r={RADIUS}): {len(found)} found,\ncompared {radius_checks} of {N_POINTS} points",
    )

    fig.suptitle("kd-tree demo", fontsize=16)
    fig.tight_layout()

    if matplotlib.get_backend().lower() in {"agg", "pdf", "ps", "svg", "cairo", "template"}:
        path = Path("kdtree_demo.png").resolve()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        logger.info("No GUI backend available; demo saved to %s", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
