"""Loaders for the datasets shipped with the package."""

from wldstk_ml.datasets._base import Dataset, load_csv_dataset


def load_iris() -> Dataset:
    """Loads the iris dataset."""
    return load_csv_dataset(
        "iris.csv",
        target="species",
        name="iris",
        dtype={
            "sepal_length": float,
            "sepal_width": float,
            "petal_length": float,
            "petal_width": float,
            "species": str,
        },
        categorical_target=True,
    )


if __name__ == "__main__":
    ds = load_iris()
    print(ds.frame.head())
