# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
"""Unit tests for wldstk_ml.utils.visualization."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")  # headless: never open windows during tests

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.legend import Legend

from wldstk_ml.utils import visualization
from wldstk_ml.utils.visualization import (
    align_cluster_labels,
    scatter_plot_2d,
    scatter_plot_2d_overlay,
)

X = np.array(
    [
        [0.0, 0.0, 10.0],
        [1.0, 1.0, 11.0],
        [2.0, 2.0, 12.0],
        [3.0, 3.0, 13.0],
        [4.0, 4.0, 14.0],
        [5.0, 5.0, 15.0],
    ]
)
Y_TRUE = np.array([0, 0, 1, 1, 2, 2])


def legend_of(ax: Axes) -> Legend:
    """Return the axes' (last added) legend, failing if there is none."""
    legend = ax.get_legend()
    assert legend is not None, "axes has no legend"
    return legend


def legend_texts(legend: Legend) -> list[str]:
    return [t.get_text() for t in legend.get_texts()]


def scatter(ax: Axes, index: int) -> PathCollection:
    """Return the index-th scatter collection of the axes."""
    collection = ax.collections[index]
    assert isinstance(collection, PathCollection)
    return collection


class PlotTestCase(unittest.TestCase):
    """Closes all figures after each test so they don't leak between tests."""

    def tearDown(self) -> None:
        plt.close("all")


class TestScatterPlot2d(PlotTestCase):
    def test_returns_figure_and_axes_with_labels(self) -> None:
        fig, ax = scatter_plot_2d(X, Y_TRUE, title="T", xlabel="A", ylabel="B", show=False)
        self.assertIs(ax.figure, fig)
        self.assertEqual(ax.get_title(), "T")
        self.assertEqual(ax.get_xlabel(), "A")
        self.assertEqual(ax.get_ylabel(), "B")

    def test_plots_all_points_of_first_two_features(self) -> None:
        _, ax = scatter_plot_2d(X, Y_TRUE, show=False)
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), X[:, :2])

    def test_features_selects_columns(self) -> None:
        _, ax = scatter_plot_2d(X, Y_TRUE, features=(2, 0), show=False)
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), X[:, [2, 0]])

    def test_points_are_colored_by_label(self) -> None:
        _, ax = scatter_plot_2d(X, Y_TRUE, show=False)
        np.testing.assert_array_equal(ax.collections[0].get_array(), Y_TRUE)

    def test_legend_has_one_entry_per_class(self) -> None:
        _, ax = scatter_plot_2d(X, Y_TRUE, legend_title="Clusters", show=False)
        legend = legend_of(ax)
        self.assertEqual(legend.get_title().get_text(), "Clusters")
        self.assertEqual(len(legend_texts(legend)), 3)

    def test_without_labels_has_no_legend(self) -> None:
        _, ax = scatter_plot_2d(X, show=False)
        self.assertIsNone(ax.get_legend())

    def test_invalid_X_raises(self) -> None:
        for bad in (np.zeros(5), np.zeros((5, 1)), np.zeros((2, 2, 2))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                scatter_plot_2d(bad, show=False)


class TestScatterPlot2dOverlay(PlotTestCase):
    def test_title_contains_accuracy(self) -> None:
        y_pred = np.array([0, 0, 1, 2, 2, 2])  # 5 of 6 correct
        _, ax = scatter_plot_2d_overlay(X, Y_TRUE, y_pred, title="T", show=False)
        self.assertEqual(ax.get_title(), "T (accuracy: 83.3%)")

    def test_axis_labels(self) -> None:
        _, ax = scatter_plot_2d_overlay(X, Y_TRUE, Y_TRUE, xlabel="A", ylabel="B", show=False)
        self.assertEqual(ax.get_xlabel(), "A")
        self.assertEqual(ax.get_ylabel(), "B")

    def test_rings_and_dots_cover_all_points(self) -> None:
        _, ax = scatter_plot_2d_overlay(X, Y_TRUE, Y_TRUE, show=False)
        rings, dots = ax.collections[:2]
        np.testing.assert_array_equal(rings.get_offsets(), X[:, :2])
        np.testing.assert_array_equal(dots.get_offsets(), X[:, :2])

    def test_ring_and_dot_colors_use_the_same_palette(self) -> None:
        y_pred = np.array([1, 0, 1, 2, 0, 2])
        _, ax = scatter_plot_2d_overlay(X, Y_TRUE, y_pred, show=False)
        rings, dots = scatter(ax, 0), scatter(ax, 1)
        ring_colors = np.asarray(rings.get_edgecolor())
        dot_array = dots.get_array()
        assert dot_array is not None
        dot_colors = dots.to_rgba(np.asarray(dot_array))
        # The color of a label is the same whether it is used for truth or prediction.
        for label in np.unique(Y_TRUE):
            ring = ring_colors[Y_TRUE == label]
            dot = dot_colors[y_pred == label]
            np.testing.assert_allclose(ring[:, :3], ring[0:1, :3].repeat(len(ring), 0))
            np.testing.assert_allclose(dot[:, :3], ring[0:1, :3].repeat(len(dot), 0))
        # Different labels get different colors.
        self.assertEqual(len({tuple(c) for c in ring_colors[:, :3]}), 3)

    def test_mismatches_are_marked(self) -> None:
        y_pred = np.array([0, 1, 1, 1, 0, 2])  # points 1 and 4 are wrong
        _, ax = scatter_plot_2d_overlay(X, Y_TRUE, y_pred, show=False)
        self.assertEqual(len(ax.collections), 3)
        np.testing.assert_array_equal(ax.collections[2].get_offsets(), X[[1, 4], :2])
        self.assertIn("mismatch (2)", legend_texts(legend_of(ax)))

    def test_no_mismatch_markers_for_perfect_prediction(self) -> None:
        _, ax = scatter_plot_2d_overlay(X, Y_TRUE, Y_TRUE, show=False)
        self.assertEqual(len(ax.collections), 2)
        self.assertIn("accuracy: 100.0%", ax.get_title())

    def test_highlight_errors_false_hides_markers(self) -> None:
        y_pred = np.array([1, 1, 1, 1, 1, 1])
        _, ax = scatter_plot_2d_overlay(X, Y_TRUE, y_pred, highlight_errors=False, show=False)
        self.assertEqual(len(ax.collections), 2)
        texts = legend_texts(legend_of(ax))
        self.assertFalse(any(t.startswith("mismatch") for t in texts))

    def test_legends(self) -> None:
        _, ax = scatter_plot_2d_overlay(X, Y_TRUE, Y_TRUE, legend_title="Classes", show=False)
        legends = [a for a in ax.artists if isinstance(a, Legend)]
        self.assertEqual(len(legends), 1)
        class_legend = legends[0]
        assert isinstance(class_legend, Legend)
        self.assertEqual(class_legend.get_title().get_text(), "Classes")
        self.assertEqual(legend_texts(class_legend), ["0", "1", "2"])
        style_texts = legend_texts(legend_of(ax))
        self.assertEqual(style_texts[:2], ["ground truth", "prediction"])

    def test_align_labels_fixes_permuted_cluster_ids(self) -> None:
        clusters = np.array([2, 2, 0, 0, 1, 1])  # same grouping, different ids
        _, ax = scatter_plot_2d_overlay(X, Y_TRUE, clusters, show=False)
        self.assertIn("accuracy: 0.0%", ax.get_title())
        plt.close("all")
        _, ax = scatter_plot_2d_overlay(X, Y_TRUE, clusters, align_labels=True, show=False)
        self.assertIn("accuracy: 100.0%", ax.get_title())

    def test_string_labels(self) -> None:
        y_true = np.array(["a", "a", "b", "b", "c", "c"])
        y_pred = np.array(["a", "b", "b", "b", "c", "c"])
        _, ax = scatter_plot_2d_overlay(X, y_true, y_pred, show=False)
        self.assertIn("accuracy: 83.3%", ax.get_title())

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            scatter_plot_2d_overlay(X, Y_TRUE, Y_TRUE[:-1], show=False)

    def test_invalid_X_raises(self) -> None:
        with self.assertRaises(ValueError):
            scatter_plot_2d_overlay(np.zeros((6, 1)), Y_TRUE, Y_TRUE, show=False)


class TestAlignClusterLabels(unittest.TestCase):
    def test_permutation_is_undone(self) -> None:
        clusters = np.array([2, 2, 0, 0, 1, 1])
        np.testing.assert_array_equal(align_cluster_labels(Y_TRUE, clusters), Y_TRUE)

    def test_cluster_takes_majority_label(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1, 1])
        clusters = np.array([5, 5, 5, 7, 7, 7])  # cluster 5 is mostly 0, cluster 7 all 1
        np.testing.assert_array_equal(align_cluster_labels(y_true, clusters), [0, 0, 0, 1, 1, 1])

    def test_several_clusters_can_map_to_same_label(self) -> None:
        y_true = np.array([0, 0, 0, 0])
        clusters = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(align_cluster_labels(y_true, clusters), [0, 0, 0, 0])

    def test_string_labels_and_lists(self) -> None:
        result = align_cluster_labels(["x", "x", "y"], [1, 1, 0])  # type: ignore[arg-type]
        np.testing.assert_array_equal(result, ["x", "x", "y"])


class TestSaveAndShow(PlotTestCase):
    def test_save_path_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for func, args in (
                (scatter_plot_2d, (X, Y_TRUE)),
                (scatter_plot_2d_overlay, (X, Y_TRUE, Y_TRUE)),
            ):
                with self.subTest(func=func.__name__):
                    path = Path(tmp) / f"{func.__name__}.png"
                    func(*args, save_path=path, show=False)
                    self.assertTrue(path.is_file())
                    self.assertGreater(path.stat().st_size, 0)

    def test_show_false_does_not_call_show(self) -> None:
        with (
            mock.patch.object(visualization, "_has_gui_backend", return_value=True),
            mock.patch.object(plt, "show") as show,
        ):
            scatter_plot_2d(X, Y_TRUE, show=False)
        show.assert_not_called()

    def test_show_with_gui_calls_plt_show(self) -> None:
        with (
            mock.patch.object(visualization, "_has_gui_backend", return_value=True),
            mock.patch.object(plt, "show") as show,
        ):
            scatter_plot_2d_overlay(X, Y_TRUE, Y_TRUE)
        show.assert_called_once()

    def test_show_without_gui_saves_fallback_png(self) -> None:
        cwd = os.getcwd()
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("builtins.print"),
            mock.patch.object(plt, "show") as show,
        ):
            os.chdir(tmp)
            try:
                scatter_plot_2d(X, Y_TRUE, title="My Plot")
                self.assertTrue(Path(tmp, "My_Plot.png").is_file())
            finally:
                os.chdir(cwd)
        show.assert_not_called()

    def test_show_without_gui_and_save_path_writes_only_save_path(self) -> None:
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                scatter_plot_2d(X, Y_TRUE, title="My Plot", save_path="out.png")
                self.assertEqual(sorted(os.listdir(tmp)), ["out.png"])
            finally:
                os.chdir(cwd)

    def test_has_gui_backend(self) -> None:
        for backend, expected in (
            ("agg", False),
            ("Agg", False),
            ("pdf", False),
            ("TkAgg", True),
            ("QtAgg", True),
        ):
            with (
                self.subTest(backend=backend),
                mock.patch.object(matplotlib, "get_backend", return_value=backend),
            ):
                self.assertEqual(visualization._has_gui_backend(), expected)


if __name__ == "__main__":
    unittest.main()
