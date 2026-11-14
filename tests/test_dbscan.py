# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
"""Unit tests for wldstk_ml.unsupervised_learning.dbscan."""

import unittest
from unittest import mock

import numpy as np

from wldstk_ml.datasets import NOISE_LABEL, load_clusters
from wldstk_ml.unsupervised_learning import dbscan
from wldstk_ml.unsupervised_learning.dbscan import DBSCAN, NOISE, Point, PointType
from wldstk_ml.utils.visualization import align_cluster_labels

# Two tight groups on the x-axis (spacing 1) and one isolated point.
#   cluster A: x = 0, 1, 2      cluster B: x = 10, 11, 12      noise: x = 30
LINE = np.array([[0.0, 0], [1, 0], [2, 0], [10, 0], [11, 0], [12, 0], [30, 0]])


def reference_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Brute-force DBSCAN with the same conventions as the implementation under test.

    Core points: at least `min_samples` samples within `eps` (inclusive, counting itself).
    Clusters: connected groups of core points, numbered in order of their lowest index.
    Border points: join the lowest-numbered cluster among their core neighbors.
    """
    distances = np.linalg.norm(X[:, None] - X[None], axis=2)
    adjacent = distances <= eps
    is_core = adjacent.sum(axis=1) >= min_samples

    labels = np.full(len(X), NOISE)
    cluster = 0
    for start in np.flatnonzero(is_core):
        if labels[start] != NOISE:
            continue
        labels[start] = cluster
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for neighbor in np.flatnonzero(adjacent[current] & is_core & (labels == NOISE)):
                labels[neighbor] = cluster
                frontier.append(neighbor)
        cluster += 1

    for border in np.flatnonzero(~is_core):
        core_labels = labels[adjacent[border] & is_core]
        if core_labels.size:
            labels[border] = core_labels.min()
    return labels


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two labelings group the samples identically (up to renaming the labels)."""
    return bool(np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :]))


class DBSCANTestCase(unittest.TestCase):
    """Silences the tqdm progress bar."""

    def setUp(self) -> None:
        patcher = mock.patch.object(dbscan, "tqdm", side_effect=lambda it, **_: it)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestInit(unittest.TestCase):
    def test_defaults(self) -> None:
        model = DBSCAN()
        self.assertEqual((model.eps, model.min_samples), (1.0, 5))
        self.assertEqual(model.points, [])

    def test_invalid_parameters_raise(self) -> None:
        for kwargs in [{"eps": 0}, {"eps": -1.0}, {"min_samples": 0}]:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                DBSCAN(**kwargs)

    def test_noise_label_matches_the_datasets(self) -> None:
        self.assertEqual(NOISE, NOISE_LABEL)


class TestPredictBasics(DBSCANTestCase):
    def test_two_groups_and_an_outlier(self) -> None:
        labels = DBSCAN(eps=1.0, min_samples=2).predict(LINE)
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1, NOISE])

    def test_output_shape_and_dtype(self) -> None:
        labels = DBSCAN(eps=1.0, min_samples=2).predict(LINE)
        self.assertEqual(labels.shape, (len(LINE),))
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))

    def test_clusters_are_numbered_from_zero_in_order_of_appearance(self) -> None:
        X = LINE[[3, 4, 5, 0, 1, 2]]  # cluster B first
        labels = DBSCAN(eps=1.0, min_samples=2).predict(X)
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])

    def test_eps_is_inclusive(self) -> None:
        X = np.array([[0.0], [1.0]])
        np.testing.assert_array_equal(DBSCAN(eps=1.0, min_samples=2).predict(X), [0, 0])
        np.testing.assert_array_equal(DBSCAN(eps=0.999, min_samples=2).predict(X), [NOISE, NOISE])

    def test_min_samples_counts_the_point_itself(self) -> None:
        X = np.array([[0.0], [1.0]])  # each point has itself + 1 neighbor
        np.testing.assert_array_equal(DBSCAN(eps=1.0, min_samples=2).predict(X), [0, 0])
        np.testing.assert_array_equal(DBSCAN(eps=1.0, min_samples=3).predict(X), [NOISE, NOISE])

    def test_min_samples_one_makes_every_point_a_cluster_member(self) -> None:
        labels = DBSCAN(eps=0.1, min_samples=1).predict(LINE)
        np.testing.assert_array_equal(labels, np.arange(len(LINE)))  # all isolated

    def test_tiny_eps_gives_only_noise(self) -> None:
        labels = DBSCAN(eps=0.01, min_samples=2).predict(LINE)
        self.assertTrue(np.all(labels == NOISE))

    def test_huge_eps_gives_one_cluster(self) -> None:
        labels = DBSCAN(eps=100.0, min_samples=2).predict(LINE)
        np.testing.assert_array_equal(labels, np.zeros(len(LINE)))

    def test_density_reachable_chain_is_one_cluster(self) -> None:
        # Neighbors are 1 apart, the ends are 49 apart: still one cluster.
        X = np.arange(50.0)[:, None]
        np.testing.assert_array_equal(DBSCAN(eps=1.0, min_samples=3).predict(X), np.zeros(50))

    def test_duplicate_points(self) -> None:
        X = np.array([[1.0, 1.0]] * 4 + [[9.0, 9.0]])
        labels = DBSCAN(eps=0.5, min_samples=4).predict(X)
        np.testing.assert_array_equal(labels, [0, 0, 0, 0, NOISE])

    def test_single_sample(self) -> None:
        np.testing.assert_array_equal(DBSCAN(min_samples=1).predict(np.array([[3.0]])), [0])
        np.testing.assert_array_equal(DBSCAN(min_samples=2).predict(np.array([[3.0]])), [NOISE])

    def test_accepts_lists_and_integers(self) -> None:
        labels = DBSCAN(eps=1.0, min_samples=2).predict([[0, 0], [1, 0], [5, 5]])
        np.testing.assert_array_equal(labels, [0, 0, NOISE])

    def test_rejects_non_2d_input(self) -> None:
        for X in [np.array([1.0, 2.0, 3.0]), np.zeros((2, 2, 2))]:
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                DBSCAN().predict(X)

    def test_repeated_predict_resets_state(self) -> None:
        model = DBSCAN(eps=1.0, min_samples=2)
        model.predict(LINE)
        labels = model.predict(LINE[:3])
        np.testing.assert_array_equal(labels, [0, 0, 0])
        self.assertEqual(len(model.points), 3)


class TestPointTypes(DBSCANTestCase):
    def test_core_border_and_outlier(self) -> None:
        # 0..2 are core (3 samples within eps), 3 is only next to a core point: border.
        X = np.array([[0.0], [0.5], [1.0], [1.9], [5.0]])
        model = DBSCAN(eps=1.0, min_samples=3)
        labels = model.predict(X)

        types = [point.type for point in model.points]
        self.assertEqual(
            types,
            [PointType.CORE, PointType.CORE, PointType.CORE, PointType.BORDER, PointType.OUTLIER],
        )
        np.testing.assert_array_equal(labels, [0, 0, 0, 0, NOISE])

    def test_points_match_the_returned_labels(self) -> None:
        X, _ = load_clusters("noisy", n_samples=200).to_numpy()
        model = DBSCAN(eps=0.6, min_samples=5)
        labels = model.predict(X)
        self.assertEqual([point.cluster for point in model.points], labels.tolist())
        for point in model.points:
            self.assertIsInstance(point, Point)
            self.assertEqual(point.type == PointType.OUTLIER, point.cluster == NOISE)

    def test_border_points_do_not_expand_the_cluster(self) -> None:
        # 0..0.9 are core points. 1.8 is a border point (only 3 samples within eps), and
        # 2.7 is within eps of 1.8 only. A border point does not pull it into the cluster.
        X = np.array([[0.0], [0.3], [0.6], [0.9], [1.8], [2.7]])
        model = DBSCAN(eps=1.0, min_samples=4)
        labels = model.predict(X)
        np.testing.assert_array_equal(labels, [0, 0, 0, 0, 0, NOISE])
        self.assertEqual(model.points[4].type, PointType.BORDER)
        self.assertEqual(model.points[5].type, PointType.OUTLIER)

    def test_shared_border_point_joins_the_first_cluster(self) -> None:
        # x=5 is within eps of a core point of both clusters but is no core point itself,
        # so it must not merge them. It joins the cluster that is found first.
        X = np.array([[3.0], [3.3], [3.6], [4.0], [5.0], [6.0], [6.4], [6.7], [7.0]])
        model = DBSCAN(eps=1.0, min_samples=4)
        labels = model.predict(X)
        np.testing.assert_array_equal(labels, [0, 0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(model.points[4].type, PointType.BORDER)


class TestAgainstReference(DBSCANTestCase):
    def test_matches_brute_force_on_random_data(self) -> None:
        rng = np.random.default_rng(0)
        for n_features in (1, 2, 3, 8):
            for eps, min_samples in [(0.3, 3), (0.5, 5), (1.0, 10)]:
                X = rng.normal(size=(150, n_features))
                with self.subTest(n_features=n_features, eps=eps, min_samples=min_samples):
                    np.testing.assert_array_equal(
                        DBSCAN(eps, min_samples).predict(X),
                        reference_dbscan(X, eps, min_samples),
                    )

    def test_matches_brute_force_on_noisy_clusters(self) -> None:
        X, _ = load_clusters("noisy", n_samples=400).to_numpy()
        np.testing.assert_array_equal(DBSCAN(0.5, 5).predict(X), reference_dbscan(X, 0.5, 5))

    def test_core_partition_is_invariant_to_sample_order(self) -> None:
        X, _ = load_clusters("three_overlapping", n_samples=200).to_numpy()
        perm = np.random.default_rng(1).permutation(len(X))
        model = DBSCAN(eps=0.4, min_samples=5)
        labels = model.predict(X)
        core = np.array([point.type == PointType.CORE for point in model.points])
        labels_perm = DBSCAN(eps=0.4, min_samples=5).predict(X[perm])
        # Border points may switch between touching clusters; core points may not.
        self.assertTrue(same_partition(labels[perm][core[perm]], labels_perm[core[perm]]))


class TestOnGeneratedClusters(DBSCANTestCase):
    def test_finds_well_separated_clusters_exactly(self) -> None:
        X, y = load_clusters("three_well_separated", n_samples=300).to_numpy()
        labels = DBSCAN(eps=1.0, min_samples=5).predict(X)
        self.assertNotIn(NOISE, labels[labels != NOISE].tolist())
        self.assertEqual(len(set(labels.tolist()) - {NOISE}), 3)
        clustered = labels != NOISE
        self.assertGreater(clustered.mean(), 0.95)
        self.assertTrue(same_partition(labels[clustered], y[clustered]))

    def test_finds_most_background_noise(self) -> None:
        # Uniform noise also lands inside the clusters, where it cannot be told apart.
        X, y = load_clusters("noisy", n_samples=600).to_numpy()
        labels = DBSCAN(eps=0.6, min_samples=12).predict(X)
        is_noise = y == NOISE_LABEL
        self.assertEqual(len(set(labels.tolist()) - {NOISE}), 3)
        self.assertGreater(np.mean(labels[is_noise] == NOISE), 0.65)  # noise is found ...
        self.assertGreater(np.mean(labels[~is_noise] != NOISE), 0.85)  # ... clusters are kept

    def test_high_dimensional_data(self) -> None:
        X, y = load_clusters("two_well_separated", n_samples=200, n_features=10).to_numpy()
        labels = DBSCAN(eps=6.0, min_samples=5).predict(X)
        clustered = labels != NOISE
        aligned = align_cluster_labels(y[clustered], labels[clustered])
        self.assertGreater(clustered.mean(), 0.9)
        np.testing.assert_array_equal(aligned, y[clustered])


if __name__ == "__main__":
    unittest.main()
