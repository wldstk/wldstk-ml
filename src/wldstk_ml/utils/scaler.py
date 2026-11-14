"""Feature scalers. Every scaler maps each feature with z = (x - offset) / scale.

| Scaler         | offset         | scale                   | Result                         |
|----------------|----------------|-------------------------|--------------------------------|
| StandardScaler | mean           | standard deviation      | mean 0, std 1                  |
| MinMaxScaler   | min (shifted)  | (max - min) / (b - a)   | values in [a, b]               |
| RobustScaler   | median         | interquartile range     | median 0, IQR 1; robust to outliers |
"""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

__all__ = ["MinMaxScaler", "RobustScaler", "Scaler", "StandardScaler"]


class Scaler(ABC):
    """Base class: learns an offset and a scale per feature and applies (x - offset) / scale.

    Subclasses only implement `_statistics`. Features without spread (scale 0) get the
    scale 1 instead, so they are shifted but never divided by zero.
    """

    def __init__(self) -> None:
        self._offset: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    @abstractmethod
    def _statistics(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (offset, scale) per feature of the float array X."""

    @property
    def offset_(self) -> np.ndarray:
        """Value subtracted from each feature, learned by `fit`."""
        if self._offset is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet; call fit first")
        return self._offset

    @property
    def scale_(self) -> np.ndarray:
        """Value each feature is divided by, learned by `fit`."""
        if self._scale is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet; call fit first")
        return self._scale

    def fit(self, X: np.ndarray) -> Self:
        """Learn offset and scale of each feature and return the scaler itself."""
        offset, scale = self._statistics(np.asarray(X, dtype=float))
        self._offset = offset
        self._scale = np.where(scale == 0, 1.0, scale)  # constant features would divide by 0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale X with the statistics learned by `fit`."""
        return (np.asarray(X, dtype=float) - self.offset_) / self.scale_

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Undo `transform`: map scaled data back to the original units."""
        return np.asarray(Z, dtype=float) * self.scale_ + self.offset_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to X and return X scaled."""
        return self.fit(X).transform(X)


class StandardScaler(Scaler):
    """Scales each feature to mean 0 and standard deviation 1: z = (x - mu) / sigma."""

    @property
    def mean_(self) -> np.ndarray:
        """Mean of each feature, learned by `fit`."""
        return self.offset_

    def _statistics(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X.mean(axis=0), X.std(axis=0)


class MinMaxScaler(Scaler):
    """Scales each feature linearly to `feature_range`, (0, 1) by default.

    z = a + (x - min) * (b - a) / (max - min). Outliers squeeze all other values together,
    because they become the new min or max. Constant features are mapped to a.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        super().__init__()
        low, high = feature_range
        if low >= high:
            raise ValueError(
                f"feature_range must be (min, max) with min < max, got {feature_range}"
            )
        self.feature_range = (float(low), float(high))

    @property
    def data_min_(self) -> np.ndarray:
        """Minimum of each feature in the fitted data."""
        return self.offset_ + self.feature_range[0] * self.scale_

    @property
    def data_max_(self) -> np.ndarray:
        """Maximum of each feature in the fitted data."""
        return self.offset_ + self.feature_range[1] * self.scale_

    def _statistics(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.feature_range
        data_min = X.min(axis=0)
        scale = (X.max(axis=0) - data_min) / (high - low)
        scale = np.where(scale == 0, 1.0, scale)  # also needed for the offset below
        # a + (x - min) / s  ==  (x - (min - a * s)) / s
        return data_min - low * scale, scale


class RobustScaler(Scaler):
    """Centers each feature on its median and divides by its interquartile range (IQR).

    z = (x - median) / (Q3 - Q1). Median and quartiles ignore extreme values, so a few
    outliers do not change how the rest of the data is scaled.
    """

    def __init__(self, quantile_range: tuple[float, float] = (25.0, 75.0)) -> None:
        super().__init__()
        low, high = quantile_range
        if not 0 <= low < high <= 100:
            raise ValueError(
                f"quantile_range must satisfy 0 <= q1 < q3 <= 100, got {quantile_range}"
            )
        self.quantile_range = (float(low), float(high))

    @property
    def center_(self) -> np.ndarray:
        """Median of each feature, learned by `fit`."""
        return self.offset_

    def _statistics(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q1, median, q3 = np.percentile(
            X, [self.quantile_range[0], 50, self.quantile_range[1]], axis=0
        )
        return median, q3 - q1
