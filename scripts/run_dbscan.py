"""This script demonstrates DBSCAN clustering on the iris dataset."""

from wldstk_ml.datasets import load_iris
from wldstk_ml.unsupervised_learning.dbscan import DBSCAN
from wldstk_ml.utils.logger import get_logger
from wldstk_ml.utils.scaler import StandardScaler
from wldstk_ml.utils.time_profiling import log_time
from wldstk_ml.utils.visualization import scatter_plot_2d_overlay

logger = get_logger(__name__)


def main():
    X, y = load_iris().to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    dbscan = DBSCAN(eps=0.7, min_samples=5)
    logger.info("Running DBSCAN with eps=%.2f, min_samples=%d", dbscan.eps, dbscan.min_samples)

    with log_time("DBSCAN clustering", logger):
        y_predicted = dbscan.predict(X_scaled)

    scatter_plot_2d_overlay(
        X,
        y,
        y_predicted,
        title="DBSCAN vs. Ground Truth on Iris Dataset",
        xlabel="Sepal Length",
        ylabel="Sepal Width",
        legend_title="Classes",
        align_labels=True,  # cluster ids are arbitrary -> map to majority class
    )


if __name__ == "__main__":
    main()
