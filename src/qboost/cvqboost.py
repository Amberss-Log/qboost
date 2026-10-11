"""CVQBoost: QBoost loss with continuous weights on the simplex (w_i >= 0, sum w_i = 1).
This is the version QCi runs on Dirac-3.

    min  w^T J w + C^T w,   J = H^T H / S + lam*I,   C = -2 H^T y / S

Solved here with scipy (SLSQP). J is PSD so it's convex and this is the global optimum.
w_j^2 != w_j for continuous w, so C is kept separate (unlike qubo.py).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

from qboost.weak import Stump, fit_stumps, predict_matrix

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SimplexQP:
    J: FloatArray
    C: FloatArray

    def objective(self, w: npt.ArrayLike) -> float:
        v = np.asarray(w, dtype=float)
        return float(v @ self.J @ v + self.C @ v)


def cvqboost_problem(
    H: npt.NDArray[np.integer], y: npt.NDArray[np.integer], lam: float
) -> SimplexQP:
    S, n = H.shape
    Hf, yf = H.astype(float), y.astype(float)
    return SimplexQP(J=Hf.T @ Hf / S + lam * np.eye(n), C=-2.0 * (Hf.T @ yf) / S)


def solve_simplex_qp(qp: SimplexQP, tol: float = 1e-10) -> FloatArray:
    """Solve the simplex-constrained QP with SLSQP."""
    n = qp.C.shape[0]
    res = minimize(
        qp.objective,
        np.full(n, 1.0 / n),
        jac=lambda w: 2.0 * qp.J @ w + qp.C,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        options={"ftol": tol, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"Simplex QP did not converge: {res.message}")
    w: FloatArray = np.clip(res.x, 0.0, None)
    return w / w.sum()


class CVQBoostClassifier:
    def __init__(self, n_learners: int = 12, lam: float = 0.0, prune_below: float = 1e-4) -> None:
        self.n_learners = n_learners
        self.lam = lam
        self.prune_below = prune_below

    def fit(self, X: pd.DataFrame, y: npt.NDArray[np.integer]) -> "CVQBoostClassifier":
        self.candidates_: list[Stump] = fit_stumps(X, y, self.n_learners)
        self.problem_ = cvqboost_problem(predict_matrix(self.candidates_, X), y, self.lam)
        self.weights_ = solve_simplex_qp(self.problem_)
        self.selected_ = [
            s for s, w in zip(self.candidates_, self.weights_, strict=True) if w > self.prune_below
        ]
        self.tie_label_ = 1 if np.mean(y == 1) >= 0.5 else -1
        return self

    def decision_function(self, X: pd.DataFrame) -> FloatArray:
        score: FloatArray = predict_matrix(self.candidates_, X).astype(float) @ self.weights_
        return score

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.int8]:
        score = self.decision_function(X)
        return np.where(score > 0, 1, np.where(score < 0, -1, self.tie_label_)).astype(np.int8)
