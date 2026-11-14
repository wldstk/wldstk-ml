"""This script renders example blog figures in the wildstack.de light and dark styles.

Run:  python scripts/make_blog_figures.py [output folder]   (default: blog_figures)

It writes an SVG and a PNG per figure and mode, and prints the HTML to paste into the
post plus the CSS the site needs once.
"""

import sys
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np

from wldstk_ml.datasets import load_iris
from wldstk_ml.unsupervised_learning import dbscan as dbscan_module
from wldstk_ml.unsupervised_learning.dbscan import DBSCAN, NOISE
from wldstk_ml.unsupervised_learning.k_means import KMeans
from wldstk_ml.utils.blog import BLOG_FIGURE_CSS, save_blog_figure, series_cmap
from wldstk_ml.utils.logger import get_logger
from wldstk_ml.utils.scaler import StandardScaler
from wldstk_ml.utils.visualization import align_cluster_labels, scatter_plot_2d_overlay

logger = get_logger(__name__)


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "blog_figures"
    iris = load_iris()
    X, y = iris.to_numpy()
    species = np.array(iris.target_names)[y]  # names instead of 0, 1, 2 in the legend
    X_scaled = StandardScaler().fit_transform(X)

    np.random.seed(0)
    y_kmeans = KMeans(k=3, max_iter=100).predict(X)

    # Outliers per eps for a few min_samples values (one series each).
    eps_values = np.round(np.arange(0.3, 1.31, 0.05), 2)
    min_samples_values = (3, 5, 10)
    with mock.patch.object(dbscan_module, "tqdm", side_effect=lambda it, **_: it):
        outliers = {
            m: [int(np.sum(DBSCAN(eps, m).predict(X_scaled) == NOISE)) for eps in eps_values]
            for m in min_samples_values
        }

    mismatches = int(np.sum(align_cluster_labels(species, y_kmeans) != species))

    figures = []

    # 1. An existing plotting function, coloured with the site's palette.
    figures.append(
        save_blog_figure(
            lambda p: scatter_plot_2d_overlay(
                X,
                species,
                y_kmeans,
                title="k-Means vs. ground truth",
                xlabel="Sepal length (cm)",
                ylabel="Sepal width (cm)",
                legend_title="Species",
                cmap=series_cmap(p.mode, 3),
                error_color=p.ui["warn"],
                align_labels=True,
                show=False,
            ),
            "kmeans-iris",
            out_dir,
            alt="Scatter plot of iris sepal length against sepal width. Rings show the true "
            f"species, dots the k-Means cluster; {mismatches} of 150 points are marked as "
            "mismatches.",
            label="Fig. 1",
            caption="k-Means on all four iris features, shown on the first two",
        )
    )

    # 2. A plot drawn directly: colours follow the series palette automatically.
    def draw_outliers(_):
        fig, ax = plt.subplots()
        # The lines converge at the right edge, so the legend identifies them instead of
        # end labels, which would overlap.
        for min_samples in min_samples_values:
            ax.plot(eps_values, outliers[min_samples], label=f"min_samples={min_samples}")
        ax.set_title("Outliers per eps")
        ax.set_xlabel("eps (standard deviations)")
        ax.set_ylabel("Outliers (of 150)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        return fig

    figures.append(
        save_blog_figure(
            draw_outliers,
            "dbscan-eps",
            out_dir,
            alt="Line chart of DBSCAN outliers on the scaled iris data against eps for "
            "min_samples 3, 5 and 10. All three fall towards zero as eps grows; larger "
            "min_samples need a larger eps.",
            label="Fig. 2",
            caption="DBSCAN on standardized iris data",
        )
    )

    for figure in figures:
        files = ", ".join(str(path) for paths in figure.paths.values() for path in paths)
        logger.info("Saved %s: %s", figure.name, files)
        print(f"\n<!-- {figure.name} -->\n{figure.html}")
    print(f"\n/* CSS, add once */\n{BLOG_FIGURE_CSS}")


if __name__ == "__main__":
    main()
