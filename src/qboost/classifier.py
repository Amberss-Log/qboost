"""QBoostClassifier: fit candidate stumps, build the QUBO, let a solver pick the ensemble."""

import numpy as np
import numpy.typing as npt
import pandas as pd

from qboost.qubo import QUBO, qboost_qubo
from qboost.solvers.base import Solver, SolveResult
from qboost.weak import Stump, fit_stumps, predict_matrix


class QBoostClassifier:
    def __init__(self, solver: Solver, n_learners: int = 12, lam: float = 0.0) -> None:
        self.solver = solver
        self.n_learners = n_learners
        self.lam = lam

    def fit(self, X: pd.DataFrame, y: npt.NDArray[np.integer]) -> "QBoostClassifier":
        self.candidates_: list[Stump] = fit_stumps(X, y, self.n_learners)
        H = predict_matrix(self.candidates_, X)
        self.qubo_: QUBO = qboost_qubo(H, y, self.lam)
        self.result_: SolveResult = self.solver.solve(self.qubo_)
        self.selected_ = [s for s, b in zip(self.candidates_, self.result_.bits, strict=True) if b]
        # Ties (including an empty ensemble) go to the training majority class.
        self.tie_label_ = 1 if np.mean(y == 1) >= 0.5 else -1
        return self

    def decision_function(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        H = predict_matrix(self.candidates_, X).astype(float)
        score: npt.NDArray[np.float64] = H @ self.result_.bits.astype(float)
        return score

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.int8]:
        score = self.decision_function(X)
        return np.where(score > 0, 1, np.where(score < 0, -1, self.tie_label_)).astype(np.int8)
