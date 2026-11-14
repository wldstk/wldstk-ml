"""Implementation of the k-Means clustering algorithm."""

import numpy as np
from tqdm import tqdm


class KMeans:
    """Clusters samples into k groups around the mean of each group (Lloyd's algorithm)."""

    def __init__(self, k: int = 3, max_iter: int = 300):
        self.k = k
        self.max_iter = max_iter
        self.centroids: np.ndarray | None = None

    def _init_random_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initializes the centroids as k distinct random samples of X."""

        indices = np.random.choice(len(X), self.k, replace=False)
        return X[indices].astype(float)

    def _assign_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Returns the index of the closest centroid for every sample."""

        # (n_samples, 1, n_features) - (1, k, n_features) -> distances of shape (n_samples, k)
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)

        return np.argmin(distances, axis=1)

    def _calculate_centroids(
        self, labels: np.ndarray, X: np.ndarray, prev_centroids: np.ndarray
    ) -> np.ndarray:
        """Returns the mean of the samples of each cluster."""

        one_hot = labels[:, None] == np.arange(self.k)  # (n_samples, k)
        counts = one_hot.sum(axis=0)
        sums = one_hot.T @ X  # (k, n_features)

        return np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], prev_centroids)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run k-Means and return the cluster index of every sample."""

        X = np.asarray(X, dtype=float)
        centroids = self._init_random_centroids(X)
        labels = self._assign_labels(X, centroids)

        for _ in tqdm(range(self.max_iter), desc="k-Means Iterations"):
            new_centroids = self._calculate_centroids(labels, X, centroids)

            if np.array_equal(new_centroids, centroids):
                break

            centroids = new_centroids
            labels = self._assign_labels(X, centroids)

        self.centroids = centroids

        return labels
