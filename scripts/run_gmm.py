"""This runs Gaussian Mixture Models."""

from wldstk_ml.datasets import load_clusters
from wldstk_ml.utils.logger import get_logger
from wldstk_ml.utils.scaler import StandardScaler
from wldstk_ml.utils.time_profiling import log_time
from wldstk_ml.utils.visualization import scatter_plot_2d_overlay

logger = get_logger(__name__)


def main():
    X, y = load_clusters().to_numpy()

    scatter_plot_2d_overlay(
        X,
        y,
        y,
        title="DBSCAN vs. Ground Truth on Iris Dataset",
        xlabel="Sepal Length",
        ylabel="Sepal Width",
        legend_title="Classes",
        align_labels=True,  # cluster ids are arbitrary -> map to majority class
    )


if __name__ == "__main__":
    main()
