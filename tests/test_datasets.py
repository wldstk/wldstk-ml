# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
"""Unit tests for wldstk_ml.datasets."""

import dataclasses
import unittest

import numpy as np
import pandas as pd

from wldstk_ml.datasets import Dataset, load_csv_dataset, load_iris


def make_categorical_dataset() -> Dataset:
    """Small in-memory dataset with a categorical target."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10, 20, 30, 40]})
    y = pd.Series(["cat", "dog", "cat", "bird"], name="label", dtype="category")
    return Dataset(name="toy", X=X, y=y, target_names=[str(c) for c in y.cat.categories])


def make_numeric_dataset() -> Dataset:
    """Small in-memory dataset with a numeric target and no target names."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([0.5, 1.5, 2.5], name="value")
    return Dataset(name="num", X=X, y=y)


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = make_categorical_dataset()

    def test_unpacking_yields_X_and_y(self) -> None:
        X, y = self.ds
        self.assertIs(X, self.ds.X)
        self.assertIs(y, self.ds.y)

    def test_len(self) -> None:
        self.assertEqual(len(self.ds), 4)

    def test_feature_names(self) -> None:
        self.assertEqual(self.ds.feature_names, ["a", "b"])

    def test_frame_contains_features_and_target(self) -> None:
        frame = self.ds.frame
        self.assertEqual(list(frame.columns), ["a", "b", "label"])
        self.assertEqual(len(frame), 4)

    def test_is_frozen(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.ds.name = "other"  # type: ignore[misc]

    def test_to_numpy_categorical_target_becomes_codes(self) -> None:
        X, y = self.ds.to_numpy()
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.dtype, np.float64)
        self.assertEqual(X.shape, (4, 2))
        # categories are sorted: bird=0, cat=1, dog=2
        np.testing.assert_array_equal(y, [1, 2, 1, 0])

    def test_to_numpy_numeric_target_unchanged(self) -> None:
        _, y = make_numeric_dataset().to_numpy()
        np.testing.assert_array_equal(y, [0.5, 1.5, 2.5])

    def test_decode_roundtrip(self) -> None:
        _, y = self.ds.to_numpy()
        np.testing.assert_array_equal(self.ds.decode(y), self.ds.y.to_numpy())

    def test_decode_accepts_list(self) -> None:
        np.testing.assert_array_equal(self.ds.decode([0, 2]), ["bird", "dog"])

    def test_decode_minus_one_is_none(self) -> None:
        result = self.ds.decode([-1, 0])
        self.assertIsNone(result[0])
        self.assertEqual(result[1], "bird")

    def test_decode_empty(self) -> None:
        self.assertEqual(self.ds.decode([]).size, 0)

    def test_decode_out_of_range_raises(self) -> None:
        for bad in ([3], [-2]):
            with self.subTest(codes=bad), self.assertRaises(ValueError):
                self.ds.decode(bad)

    def test_decode_without_target_names_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_numeric_dataset().decode([0])


class TestLoadCsvDataset(unittest.TestCase):
    def test_name_defaults_to_filename_stem(self) -> None:
        ds = load_csv_dataset("iris.csv", target="species")
        self.assertEqual(ds.name, "iris")

    def test_target_column_is_removed_from_features(self) -> None:
        ds = load_csv_dataset("iris.csv", target="species")
        self.assertNotIn("species", ds.feature_names)
        self.assertEqual(ds.y.name, "species")

    def test_without_categorical_target(self) -> None:
        ds = load_csv_dataset("iris.csv", target="species")
        self.assertNotIsInstance(ds.y.dtype, pd.CategoricalDtype)
        self.assertEqual(ds.target_names, [])

    def test_categorical_target_sets_target_names(self) -> None:
        ds = load_csv_dataset("iris.csv", target="species", categorical_target=True)
        self.assertIsInstance(ds.y.dtype, pd.CategoricalDtype)
        self.assertEqual(ds.target_names, list(ds.y.cat.categories))

    def test_dtype_is_applied(self) -> None:
        ds = load_csv_dataset("iris.csv", target="species", dtype={"sepal_length": "float32"})
        self.assertEqual(ds.X["sepal_length"].dtype, np.float32)

    def test_missing_target_column_raises(self) -> None:
        with self.assertRaises(KeyError):
            load_csv_dataset("iris.csv", target="does_not_exist")

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_csv_dataset("does_not_exist.csv", target="x")


class TestLoadIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ds = load_iris()

    def test_shape(self) -> None:
        self.assertEqual(self.ds.X.shape, (150, 4))
        self.assertEqual(len(self.ds.y), 150)

    def test_feature_names(self) -> None:
        self.assertEqual(
            self.ds.feature_names,
            ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )

    def test_target_names(self) -> None:
        self.assertEqual(self.ds.target_names, ["setosa", "versicolor", "virginica"])

    def test_features_are_float(self) -> None:
        self.assertTrue(all(dt == np.float64 for dt in self.ds.X.dtypes))

    def test_class_balance(self) -> None:
        _, y = self.ds.to_numpy()
        np.testing.assert_array_equal(np.bincount(y), [50, 50, 50])

    def test_no_missing_values(self) -> None:
        self.assertFalse(self.ds.frame.isna().any().any())


if __name__ == "__main__":
    unittest.main()
