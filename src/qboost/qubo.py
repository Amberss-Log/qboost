"""QUBO container and the QBoost formulation (Neven, Denchev, Rose & Macready, 2008).

QBoost loss for binary weights w in {0,1}^n, weak-learner votes H (S x n), labels y in {-1,+1}:

    L(w) = (1/S) * sum_s ( (1/n) * sum_j w_j H[s,j] - y_s )^2  +  lam * sum_j w_j

Expanding the square and using w_j^2 = w_j gives L(w) = w^T Q w + offset with

    Q        = H^T H / (n^2 S)
    Q_jj    += -2 (H^T y)_j / (n S) + lam
    offset   = 1                         (since y_s^2 = 1)
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class QUBO:
    """Minimise E(w) = w^T Q w + offset over w in {0,1}^n. Q is symmetric."""

    Q: FloatArray
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError(f"Q must be square, got shape {self.Q.shape}.")
        if not np.allclose(self.Q, self.Q.T):
            raise ValueError("Q must be symmetric.")

    @property
    def n(self) -> int:
        return int(self.Q.shape[0])

    def energy(self, w: npt.ArrayLike) -> float:
        v = np.asarray(w, dtype=float)
        return float(v @ self.Q @ v + self.offset)

    def energies(self, W: npt.ArrayLike) -> FloatArray:
        """Energies for a batch of bitstrings, shape (batch, n)."""
        M = np.asarray(W, dtype=float)
        E: FloatArray = np.einsum("bi,ij,bj->b", M, self.Q, M) + self.offset
        return E


def qboost_qubo(H: npt.NDArray[np.integer], y: npt.NDArray[np.integer], lam: float) -> QUBO:
    """Build the QBoost QUBO. ``lam`` > 0 rewards sparser ensembles."""
    S, n = H.shape
    if y.shape != (S,):
        raise ValueError(f"y has shape {y.shape}, expected ({S},).")
    Hf, yf = H.astype(float), y.astype(float)
    Q = Hf.T @ Hf / (n * n * S)
    Q[np.diag_indices(n)] += -2.0 * (Hf.T @ yf) / (n * S) + lam
    return QUBO(Q=Q, offset=1.0)


def qboost_loss(
    H: npt.NDArray[np.integer], y: npt.NDArray[np.integer], w: npt.ArrayLike, lam: float
) -> float:
    """The QBoost loss computed directly (used to test the QUBO expansion)."""
    v = np.asarray(w, dtype=float)
    n = H.shape[1]
    residual = H @ v / n - y
    return float(np.mean(residual**2) + lam * v.sum())
