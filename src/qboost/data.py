"""Airline delay data (OpenML 1169, built from US BTS on-time records)."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.datasets import fetch_openml

AIRLINES_OPENML_ID = 1169
DEFAULT_CACHE = Path.home() / ".cache" / "qboost"


def load_airlines(
    n_samples: int | None = 20_000,
    seed: int = 0,
    cache_dir: Path = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, npt.NDArray[np.int8]]:
    """Returns (X, y) with y = +1 for delayed, -1 otherwise.

    Cached in cache_dir after the first download. n_samples takes a random
    subset (None = all ~539k rows).
    """
    bunch = fetch_openml(data_id=AIRLINES_OPENML_ID, as_frame=True, data_home=str(cache_dir))
    # flight number is just an ID
    X: pd.DataFrame = bunch.data.drop(columns=["Flight"])
    y = np.where(bunch.target.astype(str) == "1", 1, -1).astype(np.int8)

    if n_samples is not None and n_samples < len(X):
        idx = np.random.default_rng(seed).choice(len(X), size=n_samples, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    return X, y
