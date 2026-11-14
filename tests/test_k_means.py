# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
"""Unit tests for wldstk_ml.unsupervised_learning.k_means."""

import unittest
import warnings
from unittest import mock

import numpy as np

from wldstk_ml.datasets import load_iris
from wldstk_ml.unsupervised_learning import k_means
from wldstk_ml.unsupervised_learning.k_means import KMeans
from wldstk_ml.utils.visualization import align_cluster_labels

# Three well-separated blobs of three points each.
BLOBS = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 11.0],
        [-11.0, 10.0],
    ]
)
BLOB_LABELS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
# One sample from each blob, used to make the initialization deterministic.
ONE_PER_BLOB = BLOBS[[0, 3, 6]]


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two labelings group the samples identically (up to renaming the labels)."""
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :]))


class KMeansTestCase(unittest.TestCase):
    """Silences the tqdm progress bar and seeds numpy for reproducible tests."""

    def setUp(self) -> None:
        patcher = mock.patch.object(k_means, "tqdm", side_effect=lambda it, **_: it)
        patcher.start()
        self.addCleanup(patcher.stop)
        np.random.seed(0)


class TestInit(unittest.TestCase):
    def test_defaults(self) -> None:
        model = KMeans()
        self.assertEqual(model.k, 3)
        self.assertEqual(model.max_iter, 300)
        self.assertIsNone(model.centroids)

    def test_custom_parameters(self) -> None:
        model = KMeans(k=5, max_iter=10)
        self.assertEqual((model.k, model.max_iter), (5, 10))


class TestHelpers(KMeansTestCase):
    def test_init_random_centroids_are_samples_of_X(self) -> None:
        centroids = KMeans(k=3)._init_random_centroids(BLOBS)
        self.assertEqual(centroids.shape, (3, 2))
        for centroid in centroids:
            self.assertTrue(any(np.array_equal(centroid, x) for x in BLOBS))

    def test_init_random_centroids_are_distinct(self) -> None:
        X = BLOBS[:3]
        for seed in range(20):
            np.random.seed(seed)
            centroids = KMeans(k=3)._init_random_centroids(X)
            self.assertEqual(len(np.unique(centroids, axis=0)), 3, f"seed={seed}")

    def test_init_random_centroids_raises_if_k_exceeds_samples(self) -> None:
        with self.assertRaises(ValueError):
            KMeans(k=10)._init_random_centroids(BLOBS)

    def test_assign_labels(self) -> None:
        samples = np.array([[0.4, 0.2], [9.0, 9.0], [-9.0, 12.0]])
        labels = KMeans(k=3)._assign_labels(samples, ONE_PER_BLOB)
        np.testing.assert_array_equal(labels, [0, 1, 2])
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))

    def test_assign_labels_of_blobs(self) -> None:
        labels = KMeans(k=3)._assign_labels(BLOBS, ONE_PER_BLOB)
        np.testing.assert_array_equal(labels, BLOB_LABELS)

    def test_assign_labels_tie_picks_first(self) -> None:
        centroids = np.array([[-1.0, 0.0], [1.0, 0.0]])
        labels = KMeans(k=2)._assign_labels(np.array([[0.0, 0.0]]), centroids)
        np.testing.assert_array_equal(labels, [0])

    def test_assign_labels_can_leave_a_cluster_empty(self) -> None:
        centroids = np.vstack([ONE_PER_BLOB, [[100.0, 100.0]]])
        labels = KMeans(k=4)._assign_labels(BLOBS, centroids)
        self.assertNotIn(3, labels)

    def test_calculate_centroids_are_cluster_means(self) -> None:
        centroids = KMeans(k=3)._calculate_centroids(BLOB_LABELS, BLOBS, ONE_PER_BLOB)
        expected = [[1 / 3, 1 / 3], [31 / 3, 31 / 3], [-31 / 3, 31 / 3]]
        np.testing.assert_allclose(centroids, expected)

    def test_calculate_centroids_keeps_empty_cluster_centroid(self) -> None:
        prev = np.vstack([ONE_PER_BLOB, [[100.0, 100.0]]])
        centroids = KMeans(k=4)._calculate_centroids(BLOB_LABELS, BLOBS, prev)
        self.assertFalse(np.isnan(centroids).any())
        np.testing.assert_array_equal(centroids[3], [100.0, 100.0])


class TestPredict(KMeansTestCase):
    def test_output_shape_dtype_and_range(self) -> None:
        labels = KMeans(k=3).predict(BLOBS)
        self.assertEqual(labels.shape, (len(BLOBS),))
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))
        self.assertTrue(np.all((labels >= 0) & (labels < 3)))

    def test_recovers_separated_blobs(self) -> None:
        model = KMeans(k=3)
        with mock.patch.object(model, "_init_random_centroids", return_value=ONE_PER_BLOB):
            labels = model.predict(BLOBS)
        np.testing.assert_array_equal(labels, BLOB_LABELS)

    def test_recovers_blobs_from_bad_initialization(self) -> None:
        # Two centroids start in the same blob; the iterations must move them apart.
        X = np.vstack([BLOBS[:6], [[5.0, 5.0]]])  # two blobs plus a point in between
        model = KMeans(k=2)
        with mock.patch.object(model, "_init_random_centroids", return_value=BLOBS[[0, 1]]):
            labels = model.predict(X)
        self.assertTrue(same_partition(labels[:6], BLOB_LABELS[:6]))

    def test_result_is_invariant_to_sample_order(self) -> None:
        perm = np.random.permutation(len(BLOBS))
        model = KMeans(k=3)
        with mock.patch.object(model, "_init_random_centroids", return_value=ONE_PER_BLOB):
            labels = model.predict(BLOBS[perm])
        self.assertTrue(same_partition(labels, BLOB_LABELS[perm]))

    def test_k_equals_one_puts_everything_in_one_cluster(self) -> None:
        np.testing.assert_array_equal(KMeans(k=1).predict(BLOBS), np.zeros(len(BLOBS)))

    def test_stops_early_after_convergence(self) -> None:
        model = KMeans(k=3, max_iter=100)
        with (
            mock.patch.object(model, "_init_random_centroids", return_value=ONE_PER_BLOB),
            mock.patch.object(
                model, "_calculate_centroids", wraps=model._calculate_centroids
            ) as calc,
        ):
            model.predict(BLOBS)
        # 1st iteration moves the centroids to the blob means, 2nd confirms convergence.
        self.assertEqual(calc.call_count, 2)

    def test_respects_max_iter(self) -> None:
        model = KMeans(k=3, max_iter=1)
        with mock.patch.object(
            model, "_calculate_centroids", wraps=model._calculate_centroids
        ) as calc:
            model.predict(BLOBS)
        self.assertEqual(calc.call_count, 1)

    def test_max_iter_zero(self) -> None:
        labels = KMeans(k=3, max_iter=0).predict(BLOBS)
        self.assertEqual(labels.shape, (len(BLOBS),))

    def test_no_empty_cluster_warnings(self) -> None:
        X = np.array([[0.0, 0.0], [0.0, 0.1], [10.0, 10.0], [10.0, 10.1]])
        for seed in range(20):
            np.random.seed(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # "Mean of empty slice" -> exception
                KMeans(k=4, max_iter=10).predict(X)

    def test_iris_accuracy(self) -> None:
        X, y = load_iris().to_numpy()
        labels = KMeans(k=3, max_iter=100).predict(X)  # seeded in setUp
        accuracy = np.mean(align_cluster_labels(y, labels) == y)
        self.assertGreaterEqual(accuracy, 0.85)

    def test_stores_final_centroids(self) -> None:
        model = KMeans(k=3)
        with mock.patch.object(model, "_init_random_centroids", return_value=ONE_PER_BLOB):
            model.predict(BLOBS)
        assert model.centroids is not None
        expected = [[1 / 3, 1 / 3], [31 / 3, 31 / 3], [-31 / 3, 31 / 3]]
        np.testing.assert_allclose(model.centroids, expected)

    def test_accepts_integer_input(self) -> None:
        labels = KMeans(k=3).predict(BLOBS.astype(int))
        self.assertEqual(labels.shape, (len(BLOBS),))


if __name__ == "__main__":
    unittest.main()
