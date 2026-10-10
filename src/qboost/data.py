"""Dataset loading.

Airline on-time data: OpenML dataset 1169 ("airlines"), derived from US Bureau of
Transportation Statistics records. Target: 1 if the flight was delayed, else 0.
"""

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
    """Return features and labels in {-1, +1} (+1 = delayed).

    Downloads once and caches under ``cache_dir``. ``n_samples`` draws a
    reproducible random subset; pass ``None`` for all ~539k flights.
    """
    bunch = fetch_openml(data_id=AIRLINES_OPENML_ID, as_frame=True, data_home=str(cache_dir))
    # Flight number is an identifier, not a predictive feature.
    X: pd.DataFrame = bunch.data.drop(columns=["Flight"])
    y = np.where(bunch.target.astype(str) == "1", 1, -1).astype(np.int8)

    if n_samples is not None and n_samples < len(X):
        idx = np.random.default_rng(seed).choice(len(X), size=n_samples, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    return X, y
