# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
"""Unit tests for wldstk_ml.datasets.artificial."""

import unittest

import numpy as np

from wldstk_ml.datasets import (
    CLUSTER_PRESETS,
    NOISE_LABEL,
    Dataset,
    generate_clusters,
    load_clusters,
)
from wldstk_ml.datasets.artificial import _center_layout, _cluster_sizes


def nearest_center_error(X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of samples closer to another cluster's mean than to their own."""
    centers = np.array([X[y == i].mean(axis=0) for i in np.unique(y)])
    nearest = np.linalg.norm(X[:, None] - centers[None], axis=2).argmin(axis=1)
    return float(np.mean(nearest != y))


class TestOutput(unittest.TestCase):
    def test_returns_dataset_with_n_samples_in_total(self) -> None:
        ds = generate_clusters(101, n_clusters=3, n_features=4, noise=0.1, random_state=0)
        self.assertIsInstance(ds, Dataset)
        X, y = ds.to_numpy()
        self.assertEqual(X.shape, (101, 4))
        self.assertEqual(y.shape, (101,))
        self.assertEqual(ds.feature_names, ["x1", "x2", "x3", "x4"])
        self.assertEqual(ds.target_names, ["cluster 0", "cluster 1", "cluster 2"])

    def test_labels(self) -> None:
        _, y = generate_clusters(60, n_clusters=3, random_state=0).to_numpy()
        np.testing.assert_array_equal(np.unique(y), [0, 1, 2])

    def test_equal_sizes_by_default(self) -> None:
        _, y = generate_clusters(90, n_clusters=3, random_state=0).to_numpy()
        np.testing.assert_array_equal(np.bincount(y), [30, 30, 30])

    def test_weights_set_the_ratio(self) -> None:
        _, y = generate_clusters(100, n_clusters=2, weights=(3, 1), random_state=0).to_numpy()
        np.testing.assert_array_equal(np.bincount(y), [75, 25])

    def test_noise_fraction_and_label(self) -> None:
        _, y = generate_clusters(200, noise=0.15, random_state=0).to_numpy()
        self.assertEqual(np.sum(y == NOISE_LABEL), 30)
        self.assertEqual(len(y), 200)

    def test_noise_lies_around_the_clusters(self) -> None:
        X, y = generate_clusters(500, noise=0.2, random_state=0).to_numpy()
        clustered, noise = X[y != NOISE_LABEL], X[y == NOISE_LABEL]
        span = clustered.max(axis=0) - clustered.min(axis=0)
        self.assertTrue(np.all(noise >= clustered.min(axis=0) - 0.1 * span - 1e-9))
        self.assertTrue(np.all(noise <= clustered.max(axis=0) + 0.1 * span + 1e-9))

    def test_random_state_is_reproducible(self) -> None:
        a = generate_clusters(50, overlap=0.1, noise=0.1, anisotropy=3, random_state=7)
        b = generate_clusters(50, overlap=0.1, noise=0.1, anisotropy=3, random_state=7)
        np.testing.assert_array_equal(a.to_numpy()[0], b.to_numpy()[0])
        np.testing.assert_array_equal(a.to_numpy()[1], b.to_numpy()[1])

    def test_different_seeds_differ(self) -> None:
        a, _ = generate_clusters(50, random_state=1).to_numpy()
        b, _ = generate_clusters(50, random_state=2).to_numpy()
        self.assertFalse(np.allclose(a, b))

    def test_no_shuffle_orders_by_cluster_with_noise_last(self) -> None:
        _, y = generate_clusters(
            40, n_clusters=2, noise=0.25, shuffle=False, random_state=0
        ).to_numpy()
        np.testing.assert_array_equal(y, [0] * 15 + [1] * 15 + [NOISE_LABEL] * 10)

    def test_cluster_std(self) -> None:
        X, y = generate_clusters(20_000, n_clusters=1, cluster_std=2.5, random_state=0).to_numpy()
        np.testing.assert_allclose(X[y == 0].std(axis=0), [2.5, 2.5], rtol=0.03)

    def test_anisotropy_is_the_ratio_of_the_principal_stds(self) -> None:
        X, _ = generate_clusters(20_000, n_clusters=1, anisotropy=4, random_state=0).to_numpy()
        stds = np.sqrt(np.linalg.eigvalsh(np.cov(X.T)))
        self.assertAlmostEqual(stds.max() / stds.min(), 4.0, delta=0.15)
        self.assertAlmostEqual(np.sqrt(stds.max() * stds.min()), 1.0, delta=0.03)


class TestOverlap(unittest.TestCase):
    def test_measured_overlap_matches_requested(self) -> None:
        for overlap in (0.01, 0.1, 0.25, 0.4):
            for n_features in (1, 2, 5):
                X, y = generate_clusters(
                    50_000, n_features=n_features, overlap=overlap, random_state=0
                ).to_numpy()
                with self.subTest(overlap=overlap, n_features=n_features):
                    self.assertAlmostEqual(nearest_center_error(X, y), overlap, delta=0.006)

    def test_overlap_zero_is_well_separated(self) -> None:
        X, y = generate_clusters(10_000, n_clusters=4, overlap=0.0, random_state=0).to_numpy()
        self.assertEqual(nearest_center_error(X, y), 0.0)

    def test_overlap_scales_with_cluster_std(self) -> None:
        X, y = generate_clusters(50_000, overlap=0.1, cluster_std=7.0, random_state=0).to_numpy()
        self.assertAlmostEqual(nearest_center_error(X, y), 0.1, delta=0.006)


class TestCenterLayout(unittest.TestCase):
    def nearest_distances(self, centers: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(centers[:, :, None] - centers[:, None, :], axis=0)
        np.fill_diagonal(distances, np.inf)
        return distances.min(axis=1)

    def test_every_center_has_its_nearest_neighbor_at_distance(self) -> None:
        for n_clusters, n_features in [(2, 1), (3, 1), (2, 2), (3, 2), (6, 2), (4, 3), (5, 10)]:
            centers = _center_layout(n_clusters, n_features, 2.0)
            with self.subTest(n_clusters=n_clusters, n_features=n_features):
                self.assertEqual(centers.shape, (n_features, n_clusters))
                np.testing.assert_allclose(self.nearest_distances(centers), 2.0)
                np.testing.assert_allclose(centers.mean(axis=1), 0.0, atol=1e-12)

    def test_simplex_is_equidistant(self) -> None:
        centers = _center_layout(4, 3, 1.0)  # tetrahedron
        distances = np.linalg.norm(centers[:, :, None] - centers[:, None, :], axis=0)
        np.testing.assert_allclose(distances[~np.eye(4, dtype=bool)], 1.0)

    def test_single_cluster_is_at_origin(self) -> None:
        np.testing.assert_array_equal(_center_layout(1, 3, 5.0), np.zeros((3, 1)))


class TestClusterSizes(unittest.TestCase):
    def test_sizes_add_up(self) -> None:
        sizes = _cluster_sizes(10, 3, None)
        self.assertEqual(sizes.sum(), 10)
        self.assertLessEqual(sizes.max() - sizes.min(), 1)

    def test_largest_remainder_gets_the_leftover(self) -> None:
        np.testing.assert_array_equal(_cluster_sizes(10, 3, (0.5, 0.3, 0.2)), [5, 3, 2])
        np.testing.assert_array_equal(_cluster_sizes(7, 2, (2, 1)), [5, 2])


class TestValidation(unittest.TestCase):
    def test_invalid_arguments_raise(self) -> None:
        cases = {
            "overlap negative": {"overlap": -0.1},
            "overlap 0.5": {"overlap": 0.5},
            "cluster_std 0": {"cluster_std": 0.0},
            "anisotropy < 1": {"anisotropy": 0.5},
            "noise 1": {"noise": 1.0},
            "wrong number of weights": {"weights": (1, 2, 3)},
            "zero weight": {"weights": (1, 0)},
            "no clusters": {"n_clusters": 0},
            "no features": {"n_features": 0},
        }
        for name, kwargs in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                generate_clusters(100, **kwargs)

    def test_too_few_samples_raise(self) -> None:
        with self.assertRaises(ValueError):
            generate_clusters(2, n_clusters=3)
        with self.assertRaises(ValueError):
            generate_clusters(10, n_clusters=2, weights=(100, 1))


class TestPresets(unittest.TestCase):
    def test_every_preset_generates(self) -> None:
        for name, settings in CLUSTER_PRESETS.items():
            with self.subTest(name):
                ds = load_clusters(name, n_samples=120)
                X, y = ds.to_numpy()
                self.assertEqual(ds.name, name)
                self.assertEqual(X.shape, (120, settings.get("n_features", 2)))
                clusters = np.unique(y[y != NOISE_LABEL])
                self.assertEqual(len(clusters), settings.get("n_clusters", 2))

    def test_default_is_two_well_separated_clusters(self) -> None:
        X, y = load_clusters().to_numpy()
        self.assertEqual((len(X), len(np.unique(y))), (300, 2))
        self.assertEqual(nearest_center_error(X, y), 0.0)

    def test_presets_are_reproducible_by_default(self) -> None:
        a, _ = load_clusters("noisy").to_numpy()
        b, _ = load_clusters("noisy").to_numpy()
        np.testing.assert_array_equal(a, b)

    def test_random_state_changes_the_data(self) -> None:
        a, _ = load_clusters("default", random_state=1).to_numpy()
        b, _ = load_clusters("default", random_state=2).to_numpy()
        self.assertFalse(np.allclose(a, b))

    def test_overrides_replace_preset_values(self) -> None:
        _, y = load_clusters("noisy", n_samples=200, noise=0.5).to_numpy()
        self.assertEqual(np.sum(y == NOISE_LABEL), 100)

    def test_imbalanced_sizes(self) -> None:
        _, y = load_clusters("two_imbalanced", n_samples=200).to_numpy()
        np.testing.assert_array_equal(np.bincount(y), [180, 20])

    def test_unknown_name_lists_the_presets(self) -> None:
        with self.assertRaisesRegex(KeyError, "two_well_separated"):
            load_clusters("does_not_exist")


if __name__ == "__main__":
    unittest.main()
