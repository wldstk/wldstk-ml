"""Unit tests for wldstk_ml.utils.scaler."""

import unittest
from typing import assert_type

import numpy as np

from wldstk_ml.datasets import load_iris
from wldstk_ml.utils.scaler import MinMaxScaler, RobustScaler, Scaler, StandardScaler

ALL_SCALERS = (StandardScaler, MinMaxScaler, RobustScaler)

# Two features with very different means and spreads.
X_SMALL = np.array([[1.0, 10.0], [3.0, 30.0]])


def random_data(seed: int = 0) -> np.ndarray:
    """Return 500 samples of 4 features with very different scales and offsets."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(500, 4)) * [0.01, 1.0, 50.0, 1000.0] + [5.0, -3.0, 0.0, 1e4]


class TestFit(unittest.TestCase):
    """Tests the statistics learned by fit."""

    def test_returns_self(self) -> None:
        """fit returns the scaler, so calls can be chained."""
        scaler = StandardScaler()
        self.assertIs(scaler.fit(X_SMALL), scaler)

    def test_known_mean_and_scale(self) -> None:
        """Mean and population standard deviation are computed per feature."""
        scaler = StandardScaler().fit(X_SMALL)
        np.testing.assert_allclose(scaler.mean_, [2.0, 20.0])
        np.testing.assert_allclose(scaler.scale_, [1.0, 10.0])

    def test_matches_numpy_per_feature(self) -> None:
        """The statistics equal numpy's column-wise mean and std."""
        X = random_data()
        scaler = StandardScaler().fit(X)
        np.testing.assert_allclose(scaler.mean_, X.mean(axis=0))
        np.testing.assert_allclose(scaler.scale_, X.std(axis=0))

    def test_constant_feature_gets_scale_one(self) -> None:
        """A feature without spread is not divided by zero."""
        scaler = StandardScaler().fit(np.array([[7.0, 1.0], [7.0, 2.0], [7.0, 3.0]]))
        self.assertEqual(scaler.scale_[0], 1.0)

    def test_refit_replaces_statistics(self) -> None:
        """Fitting again forgets the earlier data."""
        scaler = StandardScaler().fit(X_SMALL)
        scaler.fit(X_SMALL * 2)
        np.testing.assert_allclose(scaler.mean_, [4.0, 40.0])
        np.testing.assert_allclose(scaler.scale_, [2.0, 20.0])

    def test_input_is_not_modified(self) -> None:
        """fit and transform leave the caller's array unchanged."""
        X = random_data()
        original = X.copy()
        StandardScaler().fit_transform(X)
        np.testing.assert_array_equal(X, original)


class TestTransform(unittest.TestCase):
    """Tests the scaled output."""

    def test_known_values(self) -> None:
        """Each value becomes its distance from the mean in standard deviations."""
        np.testing.assert_allclose(StandardScaler().fit_transform(X_SMALL), [[-1, -1], [1, 1]])

    def test_every_feature_has_mean_zero_and_std_one(self) -> None:
        """Features are scaled separately, regardless of their original scale."""
        Z = StandardScaler().fit_transform(random_data())
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(Z.std(axis=0), 1.0)

    def test_shape_is_preserved(self) -> None:
        """The output has the same shape as the input."""
        X = random_data()
        self.assertEqual(StandardScaler().fit_transform(X).shape, X.shape)

    def test_constant_feature_becomes_zero(self) -> None:
        """A constant feature is centered to 0 instead of becoming NaN."""
        Z = StandardScaler().fit_transform(np.array([[7.0, 1.0], [7.0, 2.0], [7.0, 3.0]]))
        self.assertFalse(np.isnan(Z).any())
        np.testing.assert_array_equal(Z[:, 0], [0.0, 0.0, 0.0])

    def test_uses_statistics_of_fitted_data(self) -> None:
        """New data is scaled with the training statistics, not its own."""
        scaler = StandardScaler().fit(X_SMALL)
        np.testing.assert_allclose(
            scaler.transform(np.array([[2.0, 20.0], [4.0, 0.0]])), [[0, 0], [2, -2]]
        )

    def test_fit_transform_equals_fit_then_transform(self) -> None:
        """The shortcut gives the same result as the two steps."""
        X = random_data()
        np.testing.assert_allclose(
            StandardScaler().fit_transform(X), StandardScaler().fit(X).transform(X)
        )

    def test_original_values_can_be_recovered(self) -> None:
        """Scaling can be undone with the stored mean and scale."""
        X = random_data()
        scaler = StandardScaler()
        Z = scaler.fit_transform(X)
        np.testing.assert_allclose(Z * scaler.scale_ + scaler.mean_, X)

    def test_order_of_samples_is_kept(self) -> None:
        """Scaling is monotonic per feature, so the ranking of samples does not change."""
        X = random_data()
        Z = StandardScaler().fit_transform(X)
        np.testing.assert_array_equal(np.argsort(Z, axis=0), np.argsort(X, axis=0))

    def test_integer_input_gives_float_output(self) -> None:
        """Integer data is scaled to floats instead of being truncated."""
        Z = StandardScaler().fit_transform(np.array([[1, 10], [3, 30]]))
        self.assertTrue(np.issubdtype(Z.dtype, np.floating))
        np.testing.assert_allclose(Z, [[-1, -1], [1, 1]])

    def test_single_feature_vector(self) -> None:
        """A 1-D array is treated as one feature."""
        np.testing.assert_allclose(
            StandardScaler().fit_transform(np.array([1.0, 2.0, 3.0])),
            np.array([-1, 0, 1]) * 1.5**0.5,
        )

    def test_iris(self) -> None:
        """All four iris features end up with mean 0 and std 1."""
        X, _ = load_iris().to_numpy()
        Z = StandardScaler().fit_transform(X)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(Z.std(axis=0), 1.0)


class TestAllScalers(unittest.TestCase):
    """Behavior every scaler shares through the Scaler base class."""

    def test_fit_returns_the_subclass(self) -> None:
        """fit is typed as Self, so the chained call keeps the concrete type."""
        standard = StandardScaler().fit(X_SMALL)
        minmax = MinMaxScaler().fit(X_SMALL)
        robust = RobustScaler().fit(X_SMALL)
        assert_type(standard, StandardScaler)  # checked by pyright
        assert_type(minmax, MinMaxScaler)
        assert_type(robust, RobustScaler)
        self.assertIsInstance(minmax, MinMaxScaler)

    def test_base_class_is_abstract(self) -> None:
        """Scaler cannot be used directly."""
        with self.assertRaises(TypeError):
            Scaler()  # type: ignore[abstract]  # pylint: disable=abstract-class-instantiated

    def test_unfitted_scaler_raises_with_its_name(self) -> None:
        """transform before fit names the scaler in the error."""
        for cls in ALL_SCALERS:
            with self.subTest(cls.__name__):
                with self.assertRaisesRegex(ValueError, f"{cls.__name__} is not fitted"):
                    cls().transform(X_SMALL)

    def test_inverse_transform_recovers_the_data(self) -> None:
        """inverse_transform(transform(X)) gives X back."""
        X = random_data()
        for cls in ALL_SCALERS:
            with self.subTest(cls.__name__):
                scaler = cls().fit(X)
                np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(X)), X)

    def test_constant_feature_does_not_divide_by_zero(self) -> None:
        """A feature without spread gives finite values."""
        X = np.array([[7.0, 1.0], [7.0, 2.0], [7.0, 3.0], [7.0, 9.0]])
        for cls in ALL_SCALERS:
            with self.subTest(cls.__name__):
                self.assertTrue(np.all(np.isfinite(cls().fit_transform(X))))

    def test_input_is_not_modified(self) -> None:
        """No scaler changes the caller's array."""
        X = random_data()
        original = X.copy()
        for cls in ALL_SCALERS:
            cls().fit_transform(X)
        np.testing.assert_array_equal(X, original)

    def test_integer_input_gives_float_output(self) -> None:
        """Integer data is not truncated."""
        for cls in ALL_SCALERS:
            with self.subTest(cls.__name__):
                Z = cls().fit_transform(np.array([[1, 10], [2, 20], [4, 30]]))
                self.assertTrue(np.issubdtype(Z.dtype, np.floating))


class TestMinMaxScaler(unittest.TestCase):
    """Tests the scaling to a fixed range."""

    def test_default_range_is_zero_to_one(self) -> None:
        """Each feature's min becomes 0 and its max becomes 1."""
        Z = MinMaxScaler().fit_transform(random_data())
        np.testing.assert_allclose(Z.min(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(Z.max(axis=0), 1.0)

    def test_known_values(self) -> None:
        """Values in between are mapped linearly."""
        Z = MinMaxScaler().fit_transform(np.array([[0.0], [5.0], [20.0]]))
        np.testing.assert_allclose(Z, [[0.0], [0.25], [1.0]])

    def test_custom_range(self) -> None:
        """feature_range sets the target interval."""
        Z = MinMaxScaler(feature_range=(-1, 1)).fit_transform(random_data())
        np.testing.assert_allclose(Z.min(axis=0), -1.0)
        np.testing.assert_allclose(Z.max(axis=0), 1.0)

    def test_data_min_and_max(self) -> None:
        """The learned min and max of the data are available."""
        scaler = MinMaxScaler(feature_range=(2, 5)).fit(X_SMALL)
        np.testing.assert_allclose(scaler.data_min_, [1.0, 10.0])
        np.testing.assert_allclose(scaler.data_max_, [3.0, 30.0])

    def test_new_data_can_leave_the_range(self) -> None:
        """Values outside the fitted min and max are not clipped."""
        scaler = MinMaxScaler().fit(np.array([[0.0], [10.0]]))
        np.testing.assert_allclose(scaler.transform(np.array([[-10.0], [20.0]])), [[-1.0], [2.0]])

    def test_constant_feature_maps_to_range_start(self) -> None:
        """A constant feature becomes the lower end of the range."""
        Z = MinMaxScaler(feature_range=(3, 4)).fit_transform(np.array([[7.0], [7.0]]))
        np.testing.assert_allclose(Z, [[3.0], [3.0]])

    def test_invalid_range_raises(self) -> None:
        """The range must be increasing."""
        for feature_range in [(1, 0), (1, 1)]:
            with self.subTest(feature_range), self.assertRaises(ValueError):
                MinMaxScaler(feature_range=feature_range)


class TestRobustScaler(unittest.TestCase):
    """Tests the median / IQR scaling."""

    def test_median_zero_and_iqr_one(self) -> None:
        """Every feature ends up with median 0 and interquartile range 1."""
        Z = RobustScaler().fit_transform(random_data())
        q1, median, q3 = np.percentile(Z, [25, 50, 75], axis=0)
        np.testing.assert_allclose(median, 0.0, atol=1e-12)
        np.testing.assert_allclose(q3 - q1, 1.0)

    def test_known_values(self) -> None:
        """1..5 has median 3 and quartiles 2 and 4."""
        scaler = RobustScaler().fit(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]))
        np.testing.assert_allclose(scaler.center_, [3.0])
        np.testing.assert_allclose(scaler.scale_, [2.0])

    def test_outlier_does_not_change_the_scaling(self) -> None:
        """Replacing the largest value by a huge outlier keeps median and IQR."""
        X = np.arange(1.0, 102.0)[:, None]
        X_outlier = X.copy()
        X_outlier[-1] = 1e9
        a, b = RobustScaler().fit(X), RobustScaler().fit(X_outlier)
        np.testing.assert_allclose(a.center_, b.center_)
        np.testing.assert_allclose(a.scale_, b.scale_)

    def test_custom_quantile_range(self) -> None:
        """quantile_range selects which quantiles define the scale."""
        X = np.arange(0.0, 101.0)[:, None]
        np.testing.assert_allclose(RobustScaler((10, 90)).fit(X).scale_, [80.0])

    def test_invalid_quantile_range_raises(self) -> None:
        """Quantiles must be ordered and within 0..100."""
        for quantile_range in [(75, 25), (-1, 50), (50, 101), (30, 30)]:
            with self.subTest(quantile_range), self.assertRaises(ValueError):
                RobustScaler(quantile_range=quantile_range)


if __name__ == "__main__":
    unittest.main()
