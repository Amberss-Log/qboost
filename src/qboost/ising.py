"""QUBO -> Ising.

With w = (1 - z)/2 and z in {+1, -1}:
    E(z) = const + sum_j h_j z_j + sum_{j<k} J_jk z_j z_k

z is the Pauli-Z eigenvalue: |0> -> z=+1 -> w=0, |1> -> z=-1 -> w=1. So a measured
bitstring can be read directly as the QUBO solution.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qboost.qubo import QUBO

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Ising:
    h: FloatArray  # shape (n,), local fields
    J: FloatArray  # shape (n, n), strictly upper triangular couplings
    const: float

    @property
    def n(self) -> int:
        return int(self.h.shape[0])

    def energy(self, z: npt.ArrayLike) -> float:
        s = np.asarray(z, dtype=float)
        return float(self.const + self.h @ s + s @ self.J @ s)


def qubo_to_ising(qubo: QUBO) -> Ising:
    """Convert a QUBO to Ising form. E_ising(1 - 2w) == E_qubo(w)."""
    Q = qubo.Q
    diag = np.diag(Q)
    off = Q - np.diag(diag)  # symmetric, zero diagonal
    # E = sum_j Q_jj w_j + 2 sum_{j<k} Q_jk w_j w_k + offset, then put w = (1 - z)/2
    h = -diag / 2.0 - off.sum(axis=1) / 2.0
    J = np.triu(off, k=1) / 2.0
    const = qubo.offset + diag.sum() / 2.0 + np.triu(off, k=1).sum() / 2.0
    return Ising(h=h, J=J, const=float(const))
