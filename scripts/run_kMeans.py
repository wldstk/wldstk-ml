"""This script demonstrates kMeans clustering on the iris dataset."""

from wldstk_ml.datasets import load_iris
from wldstk_ml.unsupervised_learning.k_means import KMeans
from wldstk_ml.utils.logger import get_logger
from wldstk_ml.utils.time_profiling import log_time
from wldstk_ml.utils.visualization import scatter_plot_2d_overlay

logger = get_logger(__name__)

""" k-Means clustering:
1. Initial guess of cluster centroids
2. Assign each sample to the nearest centroid
3. Recompute centroids as the mean of assigned samples
4. Repeat steps 2-3 until convergence (no change in assignments or centroids)
"""


def main():
    X, y = load_iris().to_numpy()
    kmeans = KMeans(k=3, max_iter=100)
    logger.info("Running k-Means with k=%d, max_iter=%d", kmeans.k, kmeans.max_iter)

    with log_time("k-Means clustering", logger):
        y_predicted = kmeans.predict(X)

    scatter_plot_2d_overlay(
        X,
        y,
        y_predicted,
        title="K-Means vs. Ground Truth on Iris Dataset",
        xlabel="Sepal Length",
        ylabel="Sepal Width",
        legend_title="Classes",
        align_labels=True,  # cluster ids are arbitrary -> map to majority class
    )


if __name__ == "__main__":
    main()
