"""Decision stumps (the weak learners). Each candidate stump is one variable / qubit."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class Stump:
    """Votes +polarity when the condition is true, -polarity otherwise.

    numeric feature: x > threshold, categorical: x == category
    """

    feature: str
    threshold: float | None = None
    category: str | None = None
    polarity: int = 1

    def __post_init__(self) -> None:
        if (self.threshold is None) == (self.category is None):
            raise ValueError("Set exactly one of threshold or category.")
        if self.polarity not in (-1, 1):
            raise ValueError("polarity must be +1 or -1.")

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.int8]:
        col = X[self.feature]
        if self.category is None:
            hit = col.to_numpy(dtype=float) > self.threshold
        else:
            hit = col.astype(str).to_numpy() == self.category
        return np.where(hit, self.polarity, -self.polarity).astype(np.int8)

    def __str__(self) -> str:
        if self.category is None:
            cond = f"{self.feature} > {self.threshold:g}"
        else:
            cond = f"{self.feature} == {self.category}"
        return cond if self.polarity == 1 else f"NOT({cond})"


def candidate_stumps(
    X: pd.DataFrame, n_thresholds: int = 8, top_categories: int = 10
) -> list[Stump]:
    """Quantile thresholds for numeric columns, top categories for the rest."""
    stumps: list[Stump] = []
    for name in X.columns:
        col = X[name]
        if pd.api.types.is_numeric_dtype(col):
            qs = np.linspace(0, 1, n_thresholds + 2)[1:-1]
            for t in np.unique(np.quantile(col.to_numpy(dtype=float), qs)):
                stumps.append(Stump(feature=str(name), threshold=float(t)))
        else:
            for cat in col.astype(str).value_counts().index[:top_categories]:
                stumps.append(Stump(feature=str(name), category=str(cat)))
    return stumps


def predict_matrix(stumps: list[Stump], X: pd.DataFrame) -> npt.NDArray[np.int8]:
    """H[s, j] = vote of stump j on sample s."""
    return np.column_stack([s.predict(X) for s in stumps]).astype(np.int8)


def fit_stumps(
    X: pd.DataFrame,
    y: npt.NDArray[np.integer],
    n_learners: int,
    n_thresholds: int = 8,
    top_categories: int = 10,
) -> list[Stump]:
    """Flip polarity where needed and keep the n_learners most accurate stumps.

    Several of these end up correlated (Time > 900, Time > 1000, ...). The
    off-diagonal terms in Q penalise picking those together.
    """
    scored: list[tuple[float, Stump]] = []
    for s in candidate_stumps(X, n_thresholds, top_categories):
        acc = float(np.mean(s.predict(X) == y))
        if acc < 0.5:
            s = Stump(s.feature, s.threshold, s.category, polarity=-1)
            acc = 1.0 - acc
        scored.append((acc, s))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [s for _, s in scored[:n_learners]]
