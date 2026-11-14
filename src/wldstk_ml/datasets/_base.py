"""Shared building blocks for all datasets."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

# CSV files live in the package: src/wldstk_ml/datasets/data/
_DATA_PACKAGE = "wldstk_ml.datasets"
_DATA_SUBDIR = "data"


@dataclass(frozen=True)
class Dataset:
    """Dataset representation. Provides features + target of a dataset."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    target_names: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        """Returns the names of the feature columns."""
        return list(self.X.columns)

    @property
    def frame(self) -> pd.DataFrame:
        """Returns features and target in one DataFrame."""
        return pd.concat([self.X, self.y], axis=1)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) as numpy arrays. Categorical targets become integer codes."""
        y = self.y.cat.codes if isinstance(self.y.dtype, pd.CategoricalDtype) else self.y
        return self.X.to_numpy(dtype=float), y.to_numpy()

    def decode(self, codes: np.ndarray | list[int]) -> np.ndarray:
        """Translate integer codes (e.g. y from `to_numpy()` or predictions) back to names."""
        if not self.target_names:
            raise ValueError("Dataset has no target names")
        codes = np.asarray(codes, dtype=int)
        n = len(self.target_names)
        if codes.size and (codes.min() < -1 or codes.max() >= n):
            raise ValueError(f"Codes must be in range [-1, {n - 1}]")
        names = np.array([*self.target_names, None], dtype=object)
        return names[codes]  # -1 indexes the trailing None

    def __iter__(self) -> Iterator[Any]:
        yield self.X
        yield self.y

    def __len__(self) -> int:
        return len(self.X)


def load_csv_dataset(
    filename: str,
    target: str,
    *,
    name: str | None = None,
    dtype: Mapping[str, Any] | None = None,
    categorical_target: bool = False,
) -> Dataset:
    """Load a CSV from the package data folder and split it into features and target."""
    source = resources.files(_DATA_PACKAGE) / _DATA_SUBDIR / filename
    with resources.as_file(source) as path:
        df = pd.read_csv(path)
    if dtype:
        df = df.astype(dict(dtype))

    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not found in {filename}")

    y: pd.Series = df.pop(target)
    target_names: list[str] = []
    if categorical_target:
        y = y.astype("category")
        target_names = [str(c) for c in y.cat.categories]

    return Dataset(
        name=name or filename.rsplit(".", 1)[0],
        X=df,
        y=y,
        target_names=target_names,
    )
