"""This script renders the three 3D kd-tree panels as separate blog figures.

Run:  python scripts/make_kdtree_3d_figures.py [output folder]   (default: blog_figures)

Each panel of demo_kdtree_3d.py becomes its own SVG and PNG, in the wildstack.de light and
dark styles: the tree's splitting planes, the nearest-neighbor search and the radius
search. Data and settings (N_POINTS, QUERY, RADIUS, DRAWN_LEVELS) come from
demo_kdtree_3d.py, so both scripts always show the same tree.
"""

# matplotlib leaves `Axes3D.scatter(zs=0)` unannotated, so pyright infers `zs: int` and
# rejects float arrays. The calls are correct; only this check is disabled.
# pyright: reportArgumentType=false

import math
import sys
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

# demo_kdtree_3d.py sits next to this script, so Python finds it when the script runs.
from demo_kdtree_3d import (  # pylint: disable=import-error
    AXIS_NAMES,
    BOUNDS,
    DRAWN_LEVELS,
    N_POINTS,
    QUERY,
    RADIUS,
    CountingKdTree,
    split_plane,
    tree_depth,
    tree_point,
)
from matplotlib.lines import Line2D
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axis3d import Axis as Axis3D_

from wldstk_ml.utils.blog import BLOG_FIGURE_CSS, glow_gradient, save_blog_figure
from wldstk_ml.utils.logger import get_logger
from wldstk_ml.utils.trees import Leaf, Node

logger = get_logger(__name__)

# Inches. 3D axes leave a wide empty margin that saving crops away; this size ends up
# about as wide as a post column, so text is not scaled up in the browser.
SIZE = (8.6, 8.0)


def sparkle_path() -> Path:
    """Four-pointed star with curved, pinched sides (a sparkle), as a marker path."""
    pinch = 0.16
    tips = [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]
    controls = [(pinch, pinch), (pinch, -pinch), (-pinch, -pinch), (-pinch, pinch)]
    vertices, codes = [tips[0]], [Path.MOVETO]
    for control, tip in zip(controls, tips[1:] + tips[:1]):
        vertices += [control, tip]
        codes += [Path.CURVE3, Path.CURVE3]
    vertices.append(tips[0])
    codes.append(Path.CLOSEPOLY)
    return Path(vertices, codes)


SPARKLE = sparkle_path()


def style_3d(ax: Axes3D, p: Any) -> None:
    """Minimal 3D frame: no pane fills, hairline grid and edges, muted mono ticks."""
    axes_3d = [cast(Axis3D_, a) for a in (ax.xaxis, ax.yaxis, ax.zaxis)]  # 3D axes have panes
    for axis, (low, high) in zip(axes_3d, BOUNDS):
        axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        axis.pane.set_edgecolor(p.ui["rule"])
        axis.line.set_color(p.ui["rule"])
        axis.set_ticks(np.linspace(low, high, 3))
        axis.set_ticklabels([])  # grid lines only: corner labels would collide
    for name, (low, high) in zip(AXIS_NAMES, BOUNDS):
        getattr(ax, f"set_{name}lim")(low, high)
        getattr(ax, f"set_{name}label")(name, color=p.ui["muted"], labelpad=-12)
    ax.set_box_aspect((1, 1, 1), zoom=0.92)
    ax.view_init(elev=22, azim=-60)


def new_axes() -> Axes3D:
    """Create a figure with one 3D axes, using the size from the blog style."""
    fig = plt.figure()
    return cast(Axes3D, fig.add_subplot(projection="3d"))


def draw_points(ax: Axes3D, X: np.ndarray, p: Any) -> None:
    """All samples, quiet: small faint dots without rings."""
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=7, color=p.ui["faint"], alpha=0.45, linewidths=0)


def draw_query(ax: Axes3D, p: Any) -> Line2D:
    """The query as an accent sparkle with a soft glow; returns its legend handle."""
    glow = glow_gradient(p.ui["accent"], steps=6)
    if p.mode == "dark":  # phosphor glow: darker haze around a bright core
        halo = ((glow[0], 1400, 0.12), (glow[1], 750, 0.2), (glow[2], 360, 0.3))
    else:  # on a light page a darker haze looks muddy; use a tint of the accent instead
        accent = p.ui["accent"]
        halo = ((accent, 1400, 0.05), (accent, 750, 0.09), (accent, 360, 0.14))
    for color, size, alpha in halo:
        ax.scatter(
            *QUERY, s=size, color=color, alpha=alpha, linewidths=0, depthshade=False, zorder=8
        )
    ax.scatter(
        *QUERY,
        marker=SPARKLE,
        s=520,
        color=p.ui["accent"],
        edgecolors=p.ui["bg"],
        linewidths=1.2,
        depthshade=False,
        zorder=9,
    )
    ax.scatter(
        *QUERY, s=10, color=glow[-1], linewidths=0, depthshade=False, zorder=10
    )  # bright core
    return Line2D(
        [], [], marker=SPARKLE, markersize=13, linestyle="", color=p.ui["accent"], label="query"
    )


def draw_compared(ax: Axes3D, p: Any) -> tuple[int, Line2D]:
    """Points the last search compared, as neutral rings; returns count and legend handle."""
    compared = np.array(CountingKdTree.compared).reshape(-1, 3)
    ax.scatter(
        compared[:, 0],
        compared[:, 1],
        compared[:, 2],
        s=26,
        facecolors="none",
        edgecolors=p.ui["muted"],
        linewidths=0.8,
        zorder=5,
    )
    handle = Line2D(
        [],
        [],
        marker="o",
        markersize=6,
        linestyle="",
        markerfacecolor="none",
        markeredgecolor=p.ui["muted"],
        markeredgewidth=0.8,
        label=f"compared ({len(compared)})",
    )
    return len(compared), handle


def draw_sphere(ax: Axes3D, radius: float, color: str) -> None:
    """Thin wireframe sphere around the query."""
    u, v = np.mgrid[0 : 2 * np.pi : 19j, 0 : np.pi : 10j]
    ax.plot_wireframe(
        QUERY[0] + radius * np.cos(u) * np.sin(v),
        QUERY[1] + radius * np.sin(u) * np.sin(v),
        QUERY[2] + radius * np.cos(v),
        color=color,
        linewidth=0.4,
        alpha=0.3,
        zorder=4,
    )


def legend(ax: Axes3D, handles: list[Line2D]) -> None:
    """Small frameless legend in the top-left corner."""
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.4)


def draw_planes(
    ax: Axes3D, tree: CountingKdTree, node: Node | Leaf | None, depth: int, bounds: tuple, p: Any
) -> None:
    """Splitting planes of the top levels in the accent colour, fading with depth."""
    if not isinstance(node, Node) or depth >= DRAWN_LEVELS:
        return
    fade = depth / max(DRAWN_LEVELS - 1, 1)
    ax.add_collection3d(
        Poly3DCollection(
            [split_plane(node.axis, node.split, bounds)],
            facecolor=p.ui["accent"],
            edgecolor=p.ui["accent"],
            linewidths=0.6,
            alpha=0.16 - 0.10 * fade,
        )
    )
    lower, upper = list(bounds), list(bounds)
    lower[node.axis] = (bounds[node.axis][0], node.split)
    upper[node.axis] = (node.split, bounds[node.axis][1])
    draw_planes(ax, tree, node.left, depth + 1, tuple(lower), p)
    draw_planes(ax, tree, node.right, depth + 1, tuple(upper), p)


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "blog_figures"

    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, size=(N_POINTS, 3))
    points = [tuple(row) for row in X.tolist()]
    tree = CountingKdTree(points, leaf_size=1)  # one point per leaf, like the demo
    root = tree._root  # pylint: disable=protected-access
    assert isinstance(root, Node)
    depth = tree_depth(root)

    # Run both searches once to get the numbers for alt texts and captions.
    CountingKdTree.compared = []
    nearest = tree.closest_point(QUERY)
    assert nearest is not None
    nn_checks = len(CountingKdTree.compared)
    CountingKdTree.compared = []
    found = tree.query_radius(QUERY, RADIUS)
    radius_checks = len(CountingKdTree.compared)

    def draw_structure(p: Any):
        ax = new_axes()
        draw_planes(ax, tree, root, 0, BOUNDS, p)
        draw_points(ax, X, p)
        ax.scatter(*tree_point(tree, root), s=60, color=p.ui["accent"], zorder=6)
        legend(
            ax,
            [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=7,
                    color=p.ui["accent"],
                    markeredgecolor=p.ui["bg"],
                    label="root (splits on x)",
                ),
            ],
        )
        style_3d(ax, p)
        return ax.figure

    def draw_nearest(p: Any):
        ax = new_axes()
        ax.computed_zorder = False  # keep the query on top instead of depth-sorting it
        draw_points(ax, X, p)
        CountingKdTree.compared = []
        tree.closest_point(QUERY)
        _, compared_handle = draw_compared(ax, p)
        result = p.series[1]
        draw_sphere(ax, math.dist(QUERY, nearest), result)
        ax.plot(*zip(QUERY, nearest), color=result, linewidth=1.2, zorder=7)
        ax.scatter(*nearest, s=48, color=result, depthshade=False, zorder=7)
        query_handle = draw_query(ax, p)
        nearest_handle = Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=7,
            color=result,
            markeredgecolor=p.ui["bg"],
            label="nearest point",
        )
        legend(ax, [query_handle, nearest_handle, compared_handle])
        style_3d(ax, p)
        return ax.figure

    def draw_radius(p: Any):
        ax = new_axes()
        ax.computed_zorder = False
        draw_points(ax, X, p)
        CountingKdTree.compared = []
        tree.query_radius(QUERY, RADIUS)
        _, compared_handle = draw_compared(ax, p)
        result = p.series[1]
        draw_sphere(ax, RADIUS, result)
        ax.scatter(
            X[found, 0], X[found, 1], X[found, 2], s=40, color=result, depthshade=False, zorder=7
        )
        query_handle = draw_query(ax, p)
        found_handle = Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=7,
            color=result,
            markeredgecolor=p.ui["bg"],
            label=f"within r={RADIUS} ({len(found)})",
        )
        legend(ax, [query_handle, found_handle, compared_handle])
        style_3d(ax, p)
        return ax.figure

    figures = [
        save_blog_figure(
            draw_structure,
            "kdtree-3d-structure",
            out_dir,
            alt=f"3D scatter of {N_POINTS} random points in a unit cube, cut by the kd-tree's "
            f"top {DRAWN_LEVELS} levels of splitting planes: one plane on x through the root "
            "point, then planes on y, then on z.",
            label="Fig. 1",
            caption=f"kd-tree over {N_POINTS} points: {depth} levels, top {DRAWN_LEVELS} drawn",
            size=SIZE,
        ),
        save_blog_figure(
            draw_nearest,
            "kdtree-3d-nearest",
            out_dir,
            alt=f"3D scatter with a highlighted query point and its nearest neighbor at "
            f"distance {math.dist(QUERY, nearest):.3f}. Only {nn_checks} of {N_POINTS} points, "
            "drawn as rings, were compared.",
            label="Fig. 2",
            caption=f"closest_point compares {nn_checks} of {N_POINTS} points",
            size=SIZE,
        ),
        save_blog_figure(
            draw_radius,
            "kdtree-3d-radius",
            out_dir,
            alt=f"3D scatter with a sphere of radius {RADIUS} around a query point; "
            f"{len(found)} points inside are highlighted. {radius_checks} of {N_POINTS} "
            "points, drawn as rings, were compared.",
            label="Fig. 3",
            caption=f"query_radius(r={RADIUS}) finds {len(found)} points, comparing "
            f"{radius_checks} of {N_POINTS}",
            size=SIZE,
        ),
    ]

    for figure in figures:
        files = ", ".join(str(path) for paths in figure.paths.values() for path in paths)
        logger.info("Saved %s: %s", figure.name, files)
        print(f"\n<!-- {figure.name} -->\n{figure.html}")
    print(f"\n/* CSS, add once */\n{BLOG_FIGURE_CSS}")


if __name__ == "__main__":
    main()
