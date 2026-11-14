from wldstk_ml.datasets import load_clusters, CLUSTER_PRESETS

X, y = load_clusters().to_numpy()  # "default": 2 well-separated clusters
X, y = load_clusters("two_overlapping").to_numpy()
X, y = load_clusters("noisy", n_samples=1000).to_numpy()
X, y = load_clusters("noisy", noise=0.3).to_numpy()  # change any setting of a preset


print(list(CLUSTER_PRESETS))  # all names
