"""QUBO class and the QBoost loss (Neven et al. 2008).

For w in {0,1}^n, stump votes H (S x n) and labels y in {-1,+1}:

    L(w) = mean_s ((1/n) sum_j w_j H[s,j] - y_s)^2 + lam * sum_j w_j

Expanding the square and using w_j^2 = w_j:

    Q      = H^T H / (n^2 S)
    Q_jj  += -2 (H^T y)_j / (n S) + lam
    offset = 1   (because y_s^2 = 1)
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class QUBO:
    """E(w) = w^T Q w + offset for w in {0,1}^n. Q has to be symmetric."""

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
        """Energy for each row of a (batch, n) array of bitstrings."""
        M = np.asarray(W, dtype=float)
        E: FloatArray = np.einsum("bi,ij,bj->b", M, self.Q, M) + self.offset
        return E


def qboost_qubo(H: npt.NDArray[np.integer], y: npt.NDArray[np.integer], lam: float) -> QUBO:
    """QBoost QUBO. Bigger lam -> fewer stumps get picked."""
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
    """Same loss computed directly, only used to check the QUBO in tests."""
    v = np.asarray(w, dtype=float)
    n = H.shape[1]
    residual = H @ v / n - y
    return float(np.mean(residual**2) + lam * v.sum())
